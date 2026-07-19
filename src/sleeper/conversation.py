"""Conversation ownership, turn state machine, and the assistant turn loop."""

import asyncio
import queue
import threading
from collections.abc import AsyncGenerator
from concurrent.futures import Future
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from libsh import get_logger
from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError
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
from sleeper.text import iter_words
from sleeper.tts import TTS

_logger = get_logger("conv")

if TYPE_CHECKING:
  from moshi.models.tts import TTSModel


@dataclass(slots=True, frozen=True)
class QueuedTurn:
  """A recognized prompt bound to the connection that spoke it.

  The owning connection travels with the prompt so the turn loop can drop a
  prompt whose speaker has since disconnected, instead of answering it aloud to
  whoever owns the session at dequeue time.
  """

  ws: ServerConnection
  prompt: str


type TurnQueueItem = QueuedTurn | None


@dataclass(slots=True)
class AssistantTurn:
  """The speech, interruption, and playback lifecycle of one assistant reply."""

  _tts: TTS
  _playback: PlaybackTracker
  interrupted: threading.Event
  target: ServerConnection
  _done: threading.Event = field(default_factory=threading.Event)

  def speak(self, word: str) -> None:
    self._tts.speak_word(self.target, word)

  def end(self) -> None:
    """Close generated speech; TTS signals `_done` after flushing or aborting."""
    self._tts.end_turn(self._done)

  def wait_for_playback(self, stopping: threading.Event) -> tuple[bool, str]:
    """Wait until playback drains, is interrupted, or shutdown begins."""
    completed = self._playback.wait_until_complete(self._done, self.interrupted, stopping)
    return completed, self._playback.heard_text()

  def wait_for_cleanup(self) -> None:
    """Wait until the TTS worker has closed or aborted this turn."""
    self._done.wait()


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

  playback: PlaybackTracker = field(default_factory=PlaybackTracker)
  _tts: TTS = field(default_factory=TTS)
  _connection_guard: threading.Lock = field(default_factory=threading.Lock)
  _state_guard: threading.Lock = field(default_factory=threading.Lock)
  _connection: ServerConnection | None = None
  _phase: Literal["user", "assistant"] = "user"
  _interrupted: threading.Event = field(default_factory=threading.Event)
  # The in-flight assistant turn, retained so a failed model stream can be
  # abandoned from turn_loop, which never sees the turn object directly.
  _turn: AssistantTurn | None = None

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
    self.playback.wake_waiters()
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

  def assistant_turn_started(self, target: ServerConnection) -> AssistantTurn:
    """Begin and return the complete speech lifecycle for one assistant reply."""
    with self._state_guard:
      self._phase = "assistant"
      self._interrupted.clear()
      self.playback.reset_turn()
      self._turn = AssistantTurn(self._tts, self.playback, self._interrupted, target)
      return self._turn

  def assistant_turn_finished(self, completed: bool) -> None:
    """End an assistant turn: return to user only if it completed."""
    with self._state_guard:
      if completed:
        self._phase = "user"
      self._interrupted.clear()
      self._turn = None

  def abandon_assistant_turn(self, stopping: threading.Event) -> None:
    """Give up on a turn whose model stream failed, then reclaim the mic.

    Drains the turn's queued speech so the TTS worker closes it -- an unclosed
    turn would bleed into the next one -- then returns to the user phase
    unconditionally: a failed stream has no completion to honor and its words
    are never recorded to history. The drain blocks on TTS/playback, so callers
    run this off the event loop.
    """
    with self._state_guard:
      turn = self._turn
      self._turn = None
    if turn is not None:
      turn.end()
      turn.wait_for_playback(stopping)
      if turn.interrupted.is_set():
        turn.wait_for_cleanup()
    with self._state_guard:
      self._phase = "user"
      self._interrupted.clear()

  def barge_in(self) -> None:
    """User interrupted the assistant: interrupt and reclaim the mic."""
    with self._state_guard:
      self._interrupted.set()
      self._phase = "user"
    self.playback.wake_waiters()

  def return_to_user(self) -> None:
    """Fall back to the user phase without touching interruption."""
    with self._state_guard:
      self._phase = "user"

  def interrupt(self) -> None:
    """Set interruption and wake playback waiters."""
    with self._state_guard:
      self._interrupted.set()
    self.playback.wake_waiters()

  def say(self, ws: ServerConnection, voice: str | None, text: str) -> None:
    """Speak one isolated `/say` request outside the conversation lifecycle."""
    self._tts.say(ws, voice, text)

  def set_conversation_voice(self, voice: str) -> None:
    """Change the voice used for subsequent conversational replies."""
    self._tts.set_conversation_voice(voice)

  def run_tts(
    self,
    tts_model: "TTSModel",
    startup: Future[None],
    ready_message: str,
  ) -> None:
    self._tts.run(tts_model, self, startup, ready_message)

  def stop(self) -> None:
    """Wake every session worker and stop accepting queued speech."""
    self.interrupt()
    self._tts.stop()

  @property
  def interrupted(self) -> threading.Event:
    """Stable interruption event for TTS and turn-loop checks."""
    return self._interrupted


def send_transcript(ws: ServerConnection, transcript: TurnTranscript) -> None:
  ws.send(TURN_TRANSCRIPT_ADAPTER.dump_json(transcript).decode())


async def _prompt_agent(
  agent: Agent[None, str],
  prompt: str,
  history: list[ModelMessage],
) -> AsyncGenerator[str, None]:
  """Stream an assistant response into word-sized speech jobs.

  Complete words are queued as soon as they arrive so playback can begin while
  the model is still responding. The final partial word is queued unless the
  user interrupted the turn.
  """
  _logger.info("llm request", prompt=prompt)
  # `async for` abandons its iterator on GeneratorExit rather than closing it, so
  # an interrupted turn would leak the nested stream_text/iter_words generators to
  # the loop's shutdown_asyncgens finalizer, which then trips `aclose():
  # asynchronous generator is already running` against the half-torn-down
  # pydantic-ai stream. Explicitly close the whole chain when the turn unwinds.
  async with (
    agent.run_stream(prompt, message_history=history) as result,
    # pydantic-ai types stream_text() as AsyncIterator, but it is a real async
    # generator at runtime; narrow it so aclosing() can drive its finalizer.
    aclosing(cast("AsyncGenerator[str]", result.stream_text(delta=True))) as stream,
    aclosing(iter_words(stream)) as words,
  ):
    async for word in words:
      yield word


async def _run_turn(
  agent: Agent[None, str],
  item: QueuedTurn,
  history: list[ModelMessage],
  session: ConversationSession,
  stopping: threading.Event,
) -> None:
  ws = session.active_connection()
  if ws is None or ws is not item.ws:
    # The prompt's speaker is gone, or a different client now owns the session;
    # dropping it keeps one client's question from being answered to another.
    session.return_to_user()
    return

  turn = session.assistant_turn_started(ws)

  async with aclosing(_prompt_agent(agent, item.prompt, history)) as words:
    async for word in words:
      if turn.interrupted.is_set():
        break
      turn.speak(word)
  turn.end()

  completed, spoken = await asyncio.to_thread(turn.wait_for_playback, stopping)
  history.extend(
    [
      ModelRequest(parts=[UserPromptPart(content=item.prompt)]),
      ModelResponse(parts=[TextPart(content=spoken)]),
    ]
  )
  if not completed:
    # Barge-in cut the reply short: the recorded response is only what was
    # actually spoken. Follow it with an application-authored notice using the
    # user role because some chat templates require every system-role message
    # to precede the conversation. The XML sentinel distinguishes this context
    # from the user's own words.
    history.append(
      ModelRequest(
        parts=[
          UserPromptPart(
            content="<system-reminder>User interrupted your previous reply.</system-reminder>"
          )
        ]
      )
    )

  transcript = TurnTranscript("assistant", spoken, "completed" if completed else "interrupted")
  _logger.info(
    "assistant transcript",
    text=transcript.text,
    ended_by=transcript.ended_by,
  )
  try:
    send_transcript(ws, transcript)
  except ConnectionClosedOK:
    _logger.info("client disconnected before transcript")
  except Exception:
    _logger.exception("transcript send failed")

  if turn.interrupted.is_set():
    # The interrupted transcript is the client's playback-flush signal. Send
    # it before waiting, while cleanup still gates reuse of the TTS generator.
    await asyncio.to_thread(turn.wait_for_cleanup)

  session.assistant_turn_finished(completed)


async def turn_loop(
  agent: Agent[None, str],
  turns: queue.Queue[TurnQueueItem],
  session: ConversationSession,
  stopping: threading.Event,
) -> None:
  history: list[ModelMessage] = []

  while not stopping.is_set():
    item = await asyncio.to_thread(turns.get)
    if item is None:
      return
    try:
      await _run_turn(agent, item, history, session, stopping)
    except AgentRunError:
      # Resilience boundary: a turn's model call can fail (host unreachable) or
      # its orchestration can give up (tool-call retries exhausted). Abandon the
      # turn and hand the mic back so the next prompt runs the machinery again,
      # instead of letting the error kill this daemon thread and strand the
      # session in the assistant phase forever, deaf to everything but barge-in.
      # Terminal tool failures (once tools land) join this net via a shared
      # tool-error base added to the except.
      _logger.exception("turn error")
      await asyncio.to_thread(session.abandon_assistant_turn, stopping)
