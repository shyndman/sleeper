"""Conversation ownership, turn state machine, and the assistant turn loop."""

import asyncio
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.server import ServerConnection

from sleeper.messages import TURN_TRANSCRIPT_ADAPTER, TurnTranscript
from sleeper.playback import PlaybackTracker
from sleeper.tts import EndOfTurn, SpeakWord, SpeechQueueItem

# Words are flushed to TTS the moment they're whitespace-terminated; the
# partial tail stays buffered until the next delta completes it.
WORD_BREAK = re.compile(r"\s+")

# A turn owns its destination connection: the socket the assistant reply is
# spoken to plus the recognized user prompt that triggered it.
type Turn = tuple[ServerConnection, str]
type TurnQueueItem = Turn | None


@dataclass(slots=True)
class ConversationSession:
    """Single-owner conversation state: connection, phase, and interruption.

    Exactly one `/conversation` socket owns the session at a time; ownership is
    held by `_connection_guard` and only released by `disconnect()` after a
    successful `try_connect()`. Phase (`user`/`assistant`) and the interruption
    signal are the turn-taking state machine: user speech, barge-in, and
    assistant completion each move phase and toggle interruption as one
    state-locked transition so the TTS and turn loops observe a consistent view.
    """

    _connection_guard: threading.Lock = field(default_factory=threading.Lock)
    _state_guard: threading.Lock = field(default_factory=threading.Lock)
    _connection: ServerConnection | None = None
    _phase: Literal["user", "assistant"] = "user"
    _interrupted: threading.Event = field(default_factory=threading.Event)

    def try_connect(self, ws: ServerConnection) -> bool:
        """Acquire ownership non-blockingly, then publish the connection."""
        if not self._connection_guard.acquire(blocking=False):
            return False
        with self._state_guard:
            self._connection = ws
            self._phase = "user"
        return True

    def disconnect(self) -> None:
        """Interrupt, clear the connection, and release ownership.

        Only valid after a successful `try_connect()`; never used for shutdown.
        """
        with self._state_guard:
            self._interrupted.set()
            self._connection = None
        self._connection_guard.release()

    def active_connection(self) -> ServerConnection | None:
        with self._state_guard:
            return self._connection

    def is_assistant(self) -> bool:
        with self._state_guard:
            return self._phase == "assistant"

    def user_turn_finished(self) -> None:
        """User turn recognized: hand off to the assistant, keep interruption."""
        with self._state_guard:
            self._phase = "assistant"

    def assistant_turn_started(self) -> threading.Event:
        """Begin an assistant turn and return its cleared interruption event."""
        with self._state_guard:
            self._phase = "assistant"
            self._interrupted.clear()
            return self._interrupted

    def assistant_turn_finished(self, completed: bool) -> None:
        """End an assistant turn: return to user only if it completed."""
        with self._state_guard:
            if completed:
                self._phase = "user"
            self._interrupted.clear()

    def barge_in(self) -> None:
        """User interrupted the assistant: interrupt and reclaim the mic."""
        with self._state_guard:
            self._interrupted.set()
            self._phase = "user"

    def return_to_user(self) -> None:
        """Fall back to the user phase without touching interruption."""
        with self._state_guard:
            self._phase = "user"

    def interrupt(self) -> None:
        """Set interruption only; used for process shutdown."""
        with self._state_guard:
            self._interrupted.set()

    @property
    def interrupted(self) -> threading.Event:
        """Stable interruption event for TTS and turn-loop checks."""
        return self._interrupted


def send_transcript(ws: ServerConnection, transcript: TurnTranscript) -> None:
    ws.send(TURN_TRANSCRIPT_ADAPTER.dump_json(transcript).decode())


async def _run_turn(
    agent: Agent,
    turn: Turn,
    history: list[ModelMessage],
    speech_jobs: queue.Queue[SpeechQueueItem],
    session: ConversationSession,
    playback: PlaybackTracker,
    stopping: threading.Event,
    default_voice: str,
) -> None:
    ws, prompt = turn

    interrupted = session.assistant_turn_started()
    playback.reset_turn()

    turn_done = threading.Event()
    buffer = ""
    started_at = time.perf_counter()
    first_response_at: float | None = None
    received_chars = 0
    queued_words = 0
    stream_succeeded = False

    print(f"[llm] request prompt={prompt!r}", flush=True)
    try:
        async with agent.run_stream(prompt, message_history=history) as result:
            async for delta in result.stream_text(delta=True):
                received_chars += len(delta)

                if delta and first_response_at is None:
                    first_response_at = time.perf_counter()
                    print(
                        f"[llm] first response "
                        f"{first_response_at - started_at:.2f}s",
                        flush=True,
                    )

                if interrupted.is_set():
                    break

                buffer += delta
                *words, buffer = WORD_BREAK.split(buffer)
                for word in words:
                    if word:
                        speech_jobs.put(SpeakWord(ws, default_voice, word))
                        queued_words += 1
        stream_succeeded = True
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        print(
            f"[llm error] {elapsed:.2f}s {type(exc).__name__}: {exc}",
            flush=True,
        )

    tail = buffer.strip()
    if tail and not interrupted.is_set():
        speech_jobs.put(SpeakWord(ws, default_voice, tail))
        queued_words += 1

    if stream_succeeded:
        elapsed = time.perf_counter() - started_at
        outcome = "interrupted" if interrupted.is_set() else "complete"
        print(
            f"[llm] {outcome} {elapsed:.2f}s "
            f"chars={received_chars} words={queued_words}",
            flush=True,
        )

    # EndOfTurn drains the generator's delay tail (or aborts it after a
    # barge-in) and fires `turn_done` once the worker is finished here.
    speech_jobs.put(EndOfTurn(turn_done))
    completed = await asyncio.to_thread(
        playback.wait_until_complete, turn_done, interrupted, stopping
    )
    spoken = playback.heard_text()
    ended_by = "completed" if completed else "interrupted"

    try:
        send_transcript(ws, TurnTranscript("assistant", spoken, ended_by))
    except ConnectionClosedOK:
        print(
            "[conversation] client disconnected before transcript",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[conversation] transcript send failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    if interrupted.is_set():
        # The worker may still be skipping queued words; don't start the
        # next turn until it has processed this turn's EndOfTurn.
        await asyncio.to_thread(turn_done.wait)

    history.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))
    history.append(
        ModelResponse(
            parts=[TextPart(content=spoken or "(cut off before speaking)")]
        )
    )

    session.assistant_turn_finished(completed)


async def turn_loop(
    agent: Agent,
    turns: queue.Queue[TurnQueueItem],
    speech_jobs: queue.Queue[SpeechQueueItem],
    session: ConversationSession,
    playback: PlaybackTracker,
    stopping: threading.Event,
    default_voice: str,
) -> None:
    history: list[ModelMessage] = []

    while not stopping.is_set():
        turn = await asyncio.to_thread(turns.get)
        if turn is None:
            return
        await _run_turn(
            agent,
            turn,
            history,
            speech_jobs,
            session,
            playback,
            stopping,
            default_voice,
        )
