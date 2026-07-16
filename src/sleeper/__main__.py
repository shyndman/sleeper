"""Remote full-duplex voice chat over separate WebSocket routes."""

import asyncio
import functools
import queue
import threading
from concurrent.futures import Future
from contextlib import suppress

import numpy as np
from langfuse import get_client
from libsh import get_logger, setup_logging_from_env
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from pydantic_ai import Agent
from websockets.sync.server import serve

from sleeper import server
from sleeper.conversation import ConversationSession, TurnQueueItem, turn_loop
from sleeper.llm import create_llm_agent, warm_llm
from sleeper.voice_input import VAD_FRAME_SAMPLES, listen_worker

PORT = 17393

_logger = get_logger("main")


def main() -> None:
  setup_logging_from_env()
  get_client()
  Agent.instrument_all()

  mic_frames: queue.Queue[np.ndarray] = queue.Queue()
  turns: queue.Queue[TurnQueueItem] = queue.Queue()

  session = ConversationSession()
  stopping = threading.Event()

  # Model load, LLM warmup, and TTS/CUDA warmup all reach over the network or to
  # the GPU and can fail fatally before any service is up. Route the failure
  # through libsh so the traceback renders in the structured format, then exit
  # non-zero so the container reports the crash instead of dumping a raw trace.
  try:
    _logger.info("loading TTS model")
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device="cuda")
    agent = create_llm_agent()
    warm_llm()
    ready_message = f"ws://0.0.0.0:{PORT}/conversation and /say"

    # The TTS worker warms up CUDA/model state before publishing readiness.
    # Block here so a warmup failure unwinds main() before any service starts.
    startup: Future[None] = Future()
    threading.Thread(
      target=session.run_tts,
      args=(tts_model, startup, ready_message),
      daemon=True,
    ).start()
    startup.result()
  except Exception:
    _logger.exception("startup failed")
    raise SystemExit(1) from None

  threading.Thread(
    target=lambda: asyncio.run(turn_loop(agent, turns, session, stopping)),
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
    _logger.info("shutting down")
  finally:
    stopping.set()
    session.stop()
    mic_frames.put(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32))
    turns.put(None)
    listener.join(timeout=2)


if __name__ == "__main__":
  with suppress(KeyboardInterrupt):
    main()
