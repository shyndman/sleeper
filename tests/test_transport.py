"""Headless contract tests for Sleeper's websocket transports."""

import asyncio
import functools
import http.server
import json
import queue
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelAPIError
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets.frames import Close
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from sleeper import client, llm, server, tts
from sleeper.conversation import ConversationSession, send_transcript, turn_loop
from sleeper.messages import SAY_ADAPTER, TURN_TRANSCRIPT_ADAPTER, Say, TurnTranscript
from sleeper.playback import PlaybackTracker
from sleeper.tts import EndOfTurn, SayJob, SpeakWord


@contextmanager
def running_server(handler: Callable[[ServerConnection], None]):
  with serve(handler, "127.0.0.1", 0) as srv:
    thread = threading.Thread(target=srv.serve_forever)
    thread.start()
    try:
      port = srv.socket.getsockname()[1]
      yield f"ws://127.0.0.1:{port}"
    finally:
      srv.shutdown()
      thread.join(timeout=2)
      assert not thread.is_alive()


@dataclass(slots=True)
class TransportHarness:
  """Fresh, isolated transport dependencies plus a bound one-arg handler."""

  session: ConversationSession
  playback: PlaybackTracker
  mic_frames: queue.Queue[np.ndarray]
  handler: Callable[[ServerConnection], None]


@pytest.fixture
def harness() -> TransportHarness:
  session = ConversationSession()
  playback = session.playback
  mic_frames: queue.Queue[np.ndarray] = queue.Queue()
  handler = functools.partial(
    server.handler,
    session=session,
    mic_frames=mic_frames,
  )
  return TransportHarness(
    session=session,
    playback=playback,
    mic_frames=mic_frames,
    handler=handler,
  )


def test_agent_disables_thinking_for_every_run() -> None:
  requests: list[dict[str, object]] = []

  class ChatHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
      content_length = self.headers.get("Content-Length")
      assert content_length is not None
      requests.append(json.loads(self.rfile.read(int(content_length))))
      response = json.dumps(
        {
          "id": "chatcmpl-test",
          "object": "chat.completion",
          "created": 1,
          "model": llm.LLM_MODEL,
          "choices": [
            {
              "index": 0,
              "message": {"role": "assistant", "content": "ok"},
              "finish_reason": "stop",
            }
          ],
          "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
          },
        }
      ).encode()
      self.send_response(200)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(response)))
      self.end_headers()
      self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
      pass

  llm_url = llm.LLM_URL
  httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
  thread = threading.Thread(target=httpd.serve_forever)
  thread.start()
  llm.LLM_URL = f"http://127.0.0.1:{httpd.server_port}/v1"
  try:
    agent = llm.create_llm_agent()

    async def run_calls() -> tuple[str, str]:
      first = await agent.run("first")
      second = await agent.run("second")
      return first.output, second.output

    assert asyncio.run(run_calls()) == ("ok", "ok")
  finally:
    llm.LLM_URL = llm_url
    httpd.shutdown()
    thread.join()

  assert [request["reasoning_effort"] for request in requests] == ["none", "none"]


def test_startup_warms_ollama_with_model_and_keep_alive() -> None:
  received: dict[str, object] = {}

  class WarmupHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
      content_length = self.headers.get("Content-Length")
      assert content_length is not None
      received["path"] = self.path
      received["body"] = json.loads(self.rfile.read(int(content_length)))
      self.send_response(200)
      self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
      pass

  ollama_url = llm.OLLAMA_URL
  httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), WarmupHandler)
  thread = threading.Thread(target=httpd.serve_forever)
  thread.start()
  llm.OLLAMA_URL = f"http://127.0.0.1:{httpd.server_port}"
  try:
    llm.warm_llm()
  finally:
    llm.OLLAMA_URL = ollama_url
    httpd.shutdown()
    thread.join()

  assert received == {
    "path": "/api/chat",
    "body": {
      "model": llm.LLM_MODEL,
      "messages": [{"role": "user", "content": "hi"}],
      "stream": False,
      "keep_alive": "1440m",
      "think": False,
    },
  }


@pytest.mark.parametrize(
  ("adapter", "payload"),
  [
    (SAY_ADAPTER, '{"text":"hello","extra":true}'),
    (SAY_ADAPTER, '{"text":1}'),
    (
      TURN_TRANSCRIPT_ADAPTER,
      '{"role":"system","text":"hello","ended_by":"completed"}',
    ),
    (
      TURN_TRANSCRIPT_ADAPTER,
      '{"role":"assistant","text":"hello","ended_by":"cancelled"}',
    ),
    (
      TURN_TRANSCRIPT_ADAPTER,
      '{"role":"assistant","text":"hello","ended_by":"completed","extra":true}',
    ),
  ],
)
def test_json_messages_reject_wrong_types_literals_and_extra_fields(adapter, payload):
  with pytest.raises(ValidationError):
    adapter.validate_json(payload)


def test_json_messages_round_trip_valid_contracts():
  say = SAY_ADAPTER.validate_json('{"text":"hello","voice":null}')
  transcript = TURN_TRANSCRIPT_ADAPTER.validate_json(
    '{"role":"assistant","text":"hello","ended_by":"interrupted"}'
  )
  assert say == Say(text="hello", voice=None)
  assert transcript == TurnTranscript(role="assistant", text="hello", ended_by="interrupted")


def test_conversation_decodes_pcm_and_routes_audio_and_transcript(harness):
  samples = np.array([-32768, -1, 0, 1, 32767] + [1234] * 507, dtype="<i2")
  audio = bytes(3840)
  transcript = TurnTranscript(role="user", text="testing", ended_by="turn_detected")

  with (
    running_server(harness.handler) as base_url,
    connect(f"{base_url}/conversation", compression=None) as websocket,
  ):
    websocket.send(samples.tobytes())
    decoded = harness.mic_frames.get(timeout=1)
    np.testing.assert_array_equal(decoded, samples.astype(np.float32) / 32768.0)

    connection = harness.session.active_connection()
    assert connection is not None
    connection.send(audio)
    send_transcript(connection, transcript)
    assert websocket.recv() == audio
    assert TURN_TRANSCRIPT_ADAPTER.validate_json(websocket.recv()) == transcript


def test_conversation_rejects_wrong_sized_pcm_frame(harness):
  with (
    running_server(harness.handler) as base_url,
    connect(f"{base_url}/conversation", compression=None) as websocket,
  ):
    websocket.send(bytes(1023))
    with pytest.raises(ConnectionClosed) as closed:
      websocket.recv()
  assert closed.value.rcvd is not None
  assert closed.value.rcvd.code == 1003
  assert harness.mic_frames.empty()


def test_conversation_ownership_is_exclusive_and_reacquirable(harness):
  frame = np.zeros(512, dtype="<i2").tobytes()  # exactly 1024 bytes

  with running_server(harness.handler) as base_url:
    with connect(f"{base_url}/conversation", compression=None) as first:
      first.send(frame)
      assert len(harness.mic_frames.get(timeout=1)) == 512  # first now owns

      with (
        connect(f"{base_url}/conversation", compression=None) as second,
        pytest.raises(ConnectionClosed) as closed,
      ):
        second.recv()
      assert closed.value.rcvd is not None
      assert closed.value.rcvd.code == 1013

      first.send(frame)  # owner keeps streaming despite the refusal
      assert len(harness.mic_frames.get(timeout=1)) == 512

    # first closed on context exit; ownership must release
    deadline = time.monotonic() + 2
    while harness.session.active_connection() is not None and time.monotonic() < deadline:
      time.sleep(0.01)
    assert harness.session.active_connection() is None

    with connect(f"{base_url}/conversation", compression=None) as third:
      third.send(frame)
      assert len(harness.mic_frames.get(timeout=1)) == 512  # reacquired


def test_say_routes_request_to_isolated_consumer_and_closes(harness):
  audio = bytes([1, 2]) * 1920
  consumed = queue.Queue()

  def consume_one():
    item = harness.session._tts._jobs.get(timeout=2)
    assert isinstance(item, SayJob)
    consumed.put((item.voice, item.text))
    item.ws.send(audio)
    item.done.set()

  consumer = threading.Thread(target=consume_one)
  consumer.start()
  with (
    running_server(harness.handler) as base_url,
    connect(f"{base_url}/say", compression=None) as websocket,
  ):
    websocket.send(SAY_ADAPTER.dump_json(Say(text="hello", voice="test-voice")).decode())
    assert websocket.recv() == audio
    with pytest.raises(ConnectionClosed) as closed:
      websocket.recv()
  consumer.join(timeout=2)
  assert not consumer.is_alive()
  assert consumed.get_nowait() == ("test-voice", "hello")
  assert closed.value.rcvd is not None
  assert closed.value.rcvd.code == 1000


def test_client_flushes_queued_playback_on_interrupted_assistant_transcript(capsys):
  audio = bytes(3840)
  interrupted = TURN_TRANSCRIPT_ADAPTER.dump_json(
    TurnTranscript(role="assistant", text="cut off", ended_by="interrupted")
  ).decode()

  class Messages:
    def __aiter__(self):
      async def messages():
        yield audio
        yield audio
        yield interrupted

      return messages()

  playback = queue.SimpleQueue()
  asyncio.run(client._receive(Messages(), playback))
  assert playback.empty()
  assert "assistant: cut off [interrupted]" in capsys.readouterr().out


def test_turn_loop_streams_response_to_speech(capsys):
  class FakeConnection:
    def __init__(self) -> None:
      self.sent: list[str | bytes] = []

    def send(self, message: str | bytes) -> None:
      self.sent.append(message)

  class FakeStreamResult:
    async def stream_text(self, *, delta: bool):
      assert delta
      yield "Hello "
      yield "there"

  class FakeRunStream:
    async def __aenter__(self) -> FakeStreamResult:
      return FakeStreamResult()

    async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc: BaseException | None,
      traceback: TracebackType | None,
    ) -> bool:
      return False

  class FakeAgent:
    def run_stream(self, prompt: str, *, message_history: list[object]) -> FakeRunStream:
      assert prompt == "How are you?"
      assert message_history == []
      return FakeRunStream()

  class FakePlayback:
    def reset_turn(self) -> None:
      pass

    def wait_until_complete(
      self,
      turn_done: threading.Event,
      interrupted: threading.Event,
      stopping: threading.Event,
    ) -> bool:
      return True

    def heard_text(self) -> str:
      return "Hello there"

  connection = FakeConnection()
  session = ConversationSession(playback=FakePlayback())
  assert session.try_connect(connection)
  turns: queue.Queue = queue.Queue()
  turns.put("How are you?")
  turns.put(None)
  asyncio.run(
    turn_loop(
      FakeAgent(),
      turns,
      session,
      threading.Event(),
    )
  )

  output = capsys.readouterr().out
  assert "[llm] request prompt='How are you?'" in output
  assert isinstance(session._tts._jobs.get_nowait(), SpeakWord)
  assert isinstance(session._tts._jobs.get_nowait(), SpeakWord)
  assert isinstance(session._tts._jobs.get_nowait(), EndOfTurn)
  assert len(connection.sent) == 1


def test_turn_loop_survives_failed_stream_and_reclaims_mic(capsys):
  """A model-stream failure abandons the turn without killing the loop.

  Reproduces issue #2: an unhandled `agent.run_stream` error used to escape
  `turn_loop`, kill the daemon thread, and strand the session in the assistant
  phase forever. The turn is now abandoned and the mic handed back so the next
  prompt runs the machinery again.
  """

  class FakeConnection:
    def send(self, message: str | bytes) -> None:
      pass

  class FailingRunStream:
    async def __aenter__(self):
      # Mirrors "Ollama briefly unreachable when the prompt is dequeued":
      # pydantic_ai wraps the transport failure as ModelAPIError (an
      # AgentRunError subclass) at stream entry.
      raise ModelAPIError(model_name="fake", message="Connection error.")

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
      return False

  class FailingAgent:
    def run_stream(self, prompt: str, *, message_history: list[object]) -> FailingRunStream:
      return FailingRunStream()

  class FakePlayback:
    def reset_turn(self) -> None:
      pass

    def wait_until_complete(
      self,
      turn_done: threading.Event,
      interrupted: threading.Event,
      stopping: threading.Event,
    ) -> bool:
      return True

    def heard_text(self) -> str:
      return ""

  session = ConversationSession(playback=FakePlayback())
  assert session.try_connect(FakeConnection())
  session.user_turn_finished()
  assert session.is_assistant()

  turns: queue.Queue = queue.Queue()
  turns.put("How are you?")
  turns.put(None)

  # turn_loop must return normally: the failure is caught, not propagated.
  asyncio.run(turn_loop(FailingAgent(), turns, session, threading.Event()))

  output = capsys.readouterr().out
  assert "[turn error] ModelAPIError: Connection error." in output
  # Mic reclaimed: the next user prompt can run the turn machinery again.
  assert not session.is_assistant()
  # The abandoned turn is drained so the TTS worker closes it.
  assert isinstance(session._tts._jobs.get_nowait(), EndOfTurn)
  assert session._tts._jobs.empty()


def test_tts_clean_close_abandons_turn_once(monkeypatch, capsys):
  class FakeSynth:
    def __init__(self, tts_model: object, session: ConversationSession) -> None:
      self.session = session
      self.target: object | None = None
      self.conversation_audio = False
      self.turn_open = False
      self.turn_failed = False
      self.turn_started_at: float | None = None
      self.conversation_speak_calls = 0
      self.abort_calls = 0
      created.append(self)

    def set_voice(self, voice: str) -> None:
      pass

    def speak(self, text: str) -> None:
      if not self.conversation_audio:
        return
      self.conversation_speak_calls += 1
      raise ConnectionClosedOK(Close(1000, ""), Close(1000, ""), rcvd_then_sent=True)

    def end_turn(self) -> None:
      pass

    def abort_turn(self) -> None:
      self.abort_calls += 1

  created: list[FakeSynth] = []

  class FakeMimi:
    def streaming(self, batch_size: int):
      assert batch_size == 1
      return nullcontext()

  class FakeTtsModel:
    mimi = FakeMimi()

  monkeypatch.setattr(tts, "Synth", FakeSynth)
  done = threading.Event()
  connection = object()
  session = ConversationSession()
  assert session.try_connect(connection)
  session._tts.speak_word("one")
  session._tts.speak_word("two")
  session._tts.speak_word("three")
  session._tts.end_turn(done)
  session._tts.stop()

  session._tts.run(
    FakeTtsModel(),
    session,
    threading.Event(),
    "[ready]",
  )

  output = capsys.readouterr().out
  assert output.count("[tts] client disconnected; turn abandoned") == 1
  assert "[tts error]" not in output
  assert created[0].conversation_speak_calls == 1
  assert created[0].abort_calls == 1
  assert done.is_set()
