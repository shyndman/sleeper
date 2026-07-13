"""WebSocket transport: route dispatch and per-route handlers."""

import queue
import threading

import numpy as np
from websockets.sync.server import ServerConnection

from sleeper.conversation import ConversationSession
from sleeper.messages import SAY_ADAPTER
from sleeper.playback import PlaybackTracker
from sleeper.tts import SpeechQueueItem


def conversation_handler(
    ws: ServerConnection,
    *,
    session: ConversationSession,
    playback: PlaybackTracker,
    mic_frames: queue.Queue[np.ndarray],
) -> None:
    if not session.try_connect(ws):
        ws.close(1013, "conversation already active")
        return
    try:
        for message in ws:
            if not isinstance(message, bytes) or len(message) != 1024:
                ws.close(1003, "expected 1024-byte PCM frames")
                return
            mic_frames.put(
                np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0
            )
    finally:
        session.disconnect()
        playback.wake_waiters()


def say_handler(
    ws: ServerConnection,
    *,
    speech_jobs: queue.Queue[SpeechQueueItem],
    default_voice: str,
) -> None:
    try:
        raw = ws.recv()
        if not isinstance(raw, str):
            ws.close(1003, "expected Say JSON")
            return
        request = SAY_ADAPTER.validate_json(raw)
        done = threading.Event()
        speech_jobs.put(
            ("say", ws, request.voice or default_voice, request.text, done)
        )
        done.wait()
    except Exception as exc:
        ws.close(1007, str(exc)[:120])


def handler(
    ws: ServerConnection,
    *,
    session: ConversationSession,
    playback: PlaybackTracker,
    mic_frames: queue.Queue[np.ndarray],
    speech_jobs: queue.Queue[SpeechQueueItem],
    default_voice: str,
) -> None:
    path = ws.request.path
    if path == "/conversation":
        conversation_handler(
            ws, session=session, playback=playback, mic_frames=mic_frames
        )
    elif path == "/say":
        say_handler(ws, speech_jobs=speech_jobs, default_voice=default_voice)
    else:
        ws.close(1008, "unknown route")
