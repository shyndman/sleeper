"""Remote full-duplex voice chat over separate WebSocket routes."""

import asyncio
import functools
import queue
import threading

import numpy as np
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from websockets.sync.server import serve

from sleeper import server
from sleeper.conversation import ConversationSession, TurnQueueItem, turn_loop
from sleeper.playback import PlaybackTracker
from sleeper.tts import SpeechQueueItem, tts_worker
from sleeper.voice_input import VAD_FRAME_SAMPLES, listen_worker

PORT = 17393
LLM_URL = "http://ollama-nvidia:11434/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# Everything the LLM emits goes straight to TTS, so the instructions steer it toward
# speakable prose: no markup for the synthesizer to read aloud, numbers written out
# the way they're pronounced, and no long-form structure that only works on a screen.
SPOKEN_INSTRUCTIONS = """\
You are a voice assistant. Everything you write is spoken aloud by a text-to-speech
engine, so write for the ear, not the page:
- Plain prose only, direct and to the point. No filler, no pleasantries, no markdown,
  lists, headings, code, emojis, or URLs.
- Write everything as it should be pronounced: "twenty three degrees", "three thirty
  PM", "kilometres per hour" -- never digits, symbols, or abbreviations.
- No parentheticals or asides; if something is secondary, drop it.
- Be brief. Lead with the answer. If a question genuinely needs a long answer, give
  the short version and offer to go deeper.
"""


def main() -> None:
    mic_frames: queue.Queue[np.ndarray] = queue.Queue()
    turns: queue.Queue[TurnQueueItem] = queue.Queue()
    speech_jobs: queue.Queue[SpeechQueueItem] = queue.Queue()
    session = ConversationSession()
    playback = PlaybackTracker()
    stopping = threading.Event()

    print("Loading Kyutai TTS...")
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device="cuda")
    agent = Agent(
        OpenAIChatModel(LLM_MODEL, provider=OllamaProvider(base_url=LLM_URL)),
        instructions=SPOKEN_INSTRUCTIONS,
    )
    ready_message = f"[ready] ws://0.0.0.0:{PORT}/conversation and /say"

    threading.Thread(
        target=tts_worker,
        args=(
            tts_model,
            speech_jobs,
            playback,
            session.interrupted,
            stopping,
            VOICE,
            ready_message,
        ),
        daemon=True,
    ).start()
    threading.Thread(
        target=lambda: asyncio.run(
            turn_loop(agent, turns, speech_jobs, session, playback, stopping, VOICE)
        ),
        daemon=True,
    ).start()
    listener = threading.Thread(
        target=listen_worker,
        args=(mic_frames, turns, session, playback, stopping),
        daemon=True,
    )
    listener.start()

    bound = functools.partial(
        server.handler,
        session=session,
        playback=playback,
        mic_frames=mic_frames,
        speech_jobs=speech_jobs,
        default_voice=VOICE,
    )
    try:
        with serve(bound, "0.0.0.0", PORT, compression=None) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] closing server...")
    finally:
        stopping.set()
        session.interrupt()
        playback.wake_waiters()
        mic_frames.put(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32))
        turns.put(None)
        speech_jobs.put(None)
        listener.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
