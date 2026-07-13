"""Headless contract tests for Sleeper's websocket transports."""

import asyncio
import queue
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import serve

from sleeper import client
from sleeper import sleeper as transport
from sleeper.messages import SAY_ADAPTER, TURN_TRANSCRIPT_ADAPTER, Say, TurnTranscript


@contextmanager
def running_server(handler=transport.handler):
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.socket.getsockname()[1]
            yield f"ws://127.0.0.1:{port}"
        finally:
            server.shutdown()
            thread.join(timeout=2)
            assert not thread.is_alive()


def drain(items: queue.Queue[object]) -> None:
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return


@pytest.fixture(autouse=True)
def reset_transport_globals():
    drain(transport.mic_q)
    drain(transport.sentences)
    drain(transport.turns)
    transport.interrupted.clear()
    transport.playback_changed.clear()
    transport.stopping.clear()
    transport.mode["v"] = "user"
    transport.conversation["ws"] = None
    yield
    deadline = time.monotonic() + 2
    while transport.conversation_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not transport.conversation_lock.locked()
    drain(transport.mic_q)
    drain(transport.sentences)
    drain(transport.turns)
    transport.interrupted.clear()
    transport.playback_changed.clear()
    transport.stopping.clear()
    transport.conversation["ws"] = None


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


def test_conversation_decodes_pcm_and_routes_audio_and_transcript():
    samples = np.array([-32768, -1, 0, 1, 32767] + [1234] * 507, dtype="<i2")
    audio = bytes(3840)
    transcript = TurnTranscript(role="user", text="testing", ended_by="turn_detected")

    with (
        running_server() as base_url,
        connect(f"{base_url}/conversation", compression=None) as websocket,
    ):
        websocket.send(samples.tobytes())
        decoded = transport.mic_q.get(timeout=1)
        np.testing.assert_array_equal(decoded, samples.astype(np.float32) / 32768.0)

        connection = transport.conversation["ws"]
        assert connection is not None
        connection.send(audio)
        transport._send_transcript(connection, transcript)
        assert websocket.recv() == audio
        assert TURN_TRANSCRIPT_ADAPTER.validate_json(websocket.recv()) == transcript


def test_conversation_rejects_wrong_sized_pcm_frame():
    with (
        running_server() as base_url,
        connect(f"{base_url}/conversation", compression=None) as websocket,
    ):
        websocket.send(bytes(1023))
        with pytest.raises(ConnectionClosed) as closed:
            websocket.recv()
    assert closed.value.rcvd.code == 1003
    assert transport.mic_q.empty()


def test_say_routes_request_to_isolated_consumer_and_closes():
    audio = bytes([1, 2]) * 1920
    consumed = queue.Queue()

    def consume_one():
        item = transport.sentences.get(timeout=2)
        assert item is not None
        kind, websocket, voice, text, done = item
        consumed.put((kind, voice, text))
        websocket.send(audio)
        done.set()

    consumer = threading.Thread(target=consume_one)
    consumer.start()
    with (
        running_server() as base_url,
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
    assert consumed.get_nowait() == ("say", "test-voice", "hello")
    assert closed.value.rcvd.code == 1000
    assert transport.turns.empty()


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
