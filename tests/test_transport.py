"""Headless contract tests for Sleeper's websocket transports."""

import asyncio
import functools
import http.server
import json
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from sleeper import client
from sleeper import server
from sleeper import sleeper
from sleeper.conversation import ConversationSession, send_transcript
from sleeper.messages import SAY_ADAPTER, TURN_TRANSCRIPT_ADAPTER, Say, TurnTranscript
from sleeper.playback import PlaybackTracker
from sleeper.tts import SayJob, SpeechQueueItem


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
    speech_jobs: queue.Queue[SpeechQueueItem]
    handler: Callable[[ServerConnection], None]


@pytest.fixture
def harness() -> TransportHarness:
    session = ConversationSession()
    playback = PlaybackTracker()
    mic_frames: queue.Queue[np.ndarray] = queue.Queue()
    speech_jobs: queue.Queue[SpeechQueueItem] = queue.Queue()
    handler = functools.partial(
        server.handler,
        session=session,
        playback=playback,
        mic_frames=mic_frames,
        speech_jobs=speech_jobs,
        default_voice=sleeper.VOICE,
    )
    return TransportHarness(
        session=session,
        playback=playback,
        mic_frames=mic_frames,
        speech_jobs=speech_jobs,
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
                    "model": sleeper.LLM_MODEL,
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

    llm_url = sleeper.LLM_URL
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    sleeper.LLM_URL = f"http://127.0.0.1:{httpd.server_port}/v1"
    try:
        agent = sleeper.create_llm_agent()

        async def run_calls() -> tuple[str, str]:
            first = await agent.run("first")
            second = await agent.run("second")
            return first.output, second.output

        assert asyncio.run(run_calls()) == ("ok", "ok")
    finally:
        sleeper.LLM_URL = llm_url
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

    ollama_url = sleeper.OLLAMA_URL
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), WarmupHandler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    sleeper.OLLAMA_URL = f"http://127.0.0.1:{httpd.server_port}"
    try:
        sleeper.warm_llm()
    finally:
        sleeper.OLLAMA_URL = ollama_url
        httpd.shutdown()
        thread.join()

    assert received == {
        "path": "/api/chat",
        "body": {
            "model": sleeper.LLM_MODEL,
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
    assert transcript == TurnTranscript(
        role="assistant", text="hello", ended_by="interrupted"
    )


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
    assert closed.value.rcvd.code == 1003
    assert harness.mic_frames.empty()


def test_conversation_ownership_is_exclusive_and_reacquirable(harness):
    frame = np.zeros(512, dtype="<i2").tobytes()  # exactly 1024 bytes

    with running_server(harness.handler) as base_url:
        with connect(f"{base_url}/conversation", compression=None) as first:
            first.send(frame)
            assert len(harness.mic_frames.get(timeout=1)) == 512  # first now owns

            with connect(f"{base_url}/conversation", compression=None) as second:
                with pytest.raises(ConnectionClosed) as closed:
                    second.recv()
            assert closed.value.rcvd.code == 1013

            first.send(frame)  # owner keeps streaming despite the refusal
            assert len(harness.mic_frames.get(timeout=1)) == 512

        # first closed on context exit; ownership must release
        deadline = time.monotonic() + 2
        while (
            harness.session.active_connection() is not None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert harness.session.active_connection() is None

        with connect(f"{base_url}/conversation", compression=None) as third:
            third.send(frame)
            assert len(harness.mic_frames.get(timeout=1)) == 512  # reacquired


def test_say_routes_request_to_isolated_consumer_and_closes(harness):
    audio = bytes([1, 2]) * 1920
    consumed = queue.Queue()

    def consume_one():
        item = harness.speech_jobs.get(timeout=2)
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
        websocket.send(
            SAY_ADAPTER.dump_json(Say(text="hello", voice="test-voice")).decode()
        )
        assert websocket.recv() == audio
        with pytest.raises(ConnectionClosed) as closed:
            websocket.recv()
    consumer.join(timeout=2)
    assert not consumer.is_alive()
    assert consumed.get_nowait() == ("test-voice", "hello")
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
