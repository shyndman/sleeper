"""WebSocket transport: route dispatch and per-route handlers."""

import queue

import numpy as np
from websockets.sync.server import ServerConnection

from sleeper.conversation import ConversationSession
from sleeper.messages import SAY_ADAPTER, SET_VOICE_ADAPTER


def conversation_handler(
  ws: ServerConnection,
  *,
  session: ConversationSession,
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
      mic_frames.put(np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0)
  finally:
    session.disconnect()


def say_handler(ws: ServerConnection, *, session: ConversationSession) -> None:
  try:
    raw = ws.recv()
    if not isinstance(raw, str):
      ws.close(1003, "expected Say JSON")
      return

    request = SAY_ADAPTER.validate_json(raw)
    session.say(ws, request.voice, request.text)
  except Exception as exc:
    ws.close(1007, str(exc)[:120])


def voice_handler(ws: ServerConnection, *, session: ConversationSession) -> None:
  try:
    raw = ws.recv()
    if not isinstance(raw, str):
      ws.close(1003, "expected SetVoice JSON")
      return

    request = SET_VOICE_ADAPTER.validate_json(raw)
    # Blocks until the TTS worker accepts and validates the voice, so the echo
    # only ever confirms a selection that actually took effect.
    session.set_conversation_voice(request.voice)
    ws.send(SET_VOICE_ADAPTER.dump_json(request).decode())
  except Exception as exc:
    ws.close(1007, str(exc)[:120])


def handler(
  ws: ServerConnection,
  *,
  session: ConversationSession,
  mic_frames: queue.Queue[np.ndarray],
) -> None:
  assert ws.request is not None

  match ws.request.path:
    case "/conversation":
      conversation_handler(ws, session=session, mic_frames=mic_frames)
    case "/say":
      say_handler(ws, session=session)
    case "/voice":
      voice_handler(ws, session=session)

    case _:
      ws.close(1008, "unknown route")
