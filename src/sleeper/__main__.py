"""Remote full-duplex voice chat over separate WebSocket routes."""

import asyncio
import functools
import queue
import threading
from contextlib import suppress

import numpy as np
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from websockets.sync.server import serve

from sleeper import server
from sleeper.conversation import ConversationSession, TurnQueueItem, turn_loop
from sleeper.llm import create_llm_agent, warm_llm
from sleeper.voice_input import VAD_FRAME_SAMPLES, listen_worker

PORT = 17393


def main() -> None:
  mic_frames: queue.Queue[np.ndarray] = queue.Queue()
  turns: queue.Queue[TurnQueueItem] = queue.Queue()

  session = ConversationSession()
  stopping = threading.Event()

  print("Loading Kyutai TTS...")
  ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
  tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device="cuda")
  agent = create_llm_agent()
  warm_llm()
  ready_message = f"[ready] ws://0.0.0.0:{PORT}/conversation and /say"

  threading.Thread(
    target=session.run_tts,
    args=(tts_model, stopping, ready_message),
    daemon=True,
  ).start()
  threading.Thread(
    target=lambda: asyncio.run(
      turn_loop(agent, turns, session, stopping)
    ),
    daemon=True,
  ).start()
  listener = threading.Thread(
    target=listen_worker,
    args=(mic_frames, turns, session, stopping),
    daemon=True,
  )
  listener.start()

  bound = functools.partial(
    server.handler,
    session=session,
    mic_frames=mic_frames,
  )

  try:
    with serve(bound, "0.0.0.0", PORT, compression=None) as srv:
      srv.serve_forever()
  except KeyboardInterrupt:
    print("\n[shutdown] closing server...")
  finally:
    stopping.set()
    session.stop()
    mic_frames.put(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32))
    turns.put(None)
    listener.join(timeout=2)


if __name__ == "__main__":
  with suppress(KeyboardInterrupt):
    main()
