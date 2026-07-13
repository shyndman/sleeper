"""Conversation ownership, turn state machine, and the assistant turn loop."""

import asyncio
import queue
import re
import threading
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
from websockets.sync.server import ServerConnection

from sleeper.messages import TURN_TRANSCRIPT_ADAPTER, TurnTranscript
from sleeper.playback import PlaybackTracker
from sleeper.tts import SpeechQueueItem

# Flush a sentence to TTS as soon as it's complete; the tail stays buffered.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

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

    def assistant_turn_started(self) -> None:
        """Begin an assistant turn: assistant phase, clear interruption."""
        with self._state_guard:
            self._phase = "assistant"
            self._interrupted.clear()

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


def _wait_synth_or_interrupt(done: threading.Event, interrupted: threading.Event) -> None:
    while not done.wait(0.02) and not interrupted.is_set():
        pass


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
    interrupted = session.interrupted
    while not stopping.is_set():
        item = await asyncio.to_thread(turns.get)
        if item is None:
            return
        ws, prompt = item
        session.assistant_turn_started()
        playback.reset_turn()
        last_synth_done = threading.Event()
        last_synth_done.set()
        buffer = ""
        try:
            async with agent.run_stream(prompt, message_history=history) as result:
                async for delta in result.stream_text(delta=True):
                    if interrupted.is_set():
                        break
                    buffer += delta
                    *complete, buffer = SENTENCE_END.split(buffer)
                    for sentence in complete:
                        sentence = sentence.strip()
                        if sentence:
                            last_synth_done = threading.Event()
                            speech_jobs.put(
                                (
                                    "conversation",
                                    ws,
                                    default_voice,
                                    sentence,
                                    last_synth_done,
                                )
                            )
                            await asyncio.to_thread(
                                _wait_synth_or_interrupt, last_synth_done, interrupted
                            )
                            if interrupted.is_set():
                                break
        except Exception as exc:
            print(f"[llm error] {exc}")
        if buffer.strip() and not interrupted.is_set():
            last_synth_done = threading.Event()
            speech_jobs.put(
                ("conversation", ws, default_voice, buffer.strip(), last_synth_done)
            )
            await asyncio.to_thread(
                _wait_synth_or_interrupt, last_synth_done, interrupted
            )
        completed = await asyncio.to_thread(
            playback.wait_until_complete, last_synth_done, interrupted, stopping
        )
        spoken = playback.heard_text()
        ended_by = "completed" if completed else "interrupted"
        try:
            send_transcript(ws, TurnTranscript("assistant", spoken, ended_by))
        except Exception:
            pass
        if interrupted.is_set():
            await asyncio.to_thread(last_synth_done.wait)
        history.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))
        history.append(
            ModelResponse(
                parts=[TextPart(content=spoken or "(cut off before speaking)")]
            )
        )
        session.assistant_turn_finished(completed)
