"""Play one remote Sleeper text-to-speech response."""

import argparse
import asyncio
import queue

import sounddevice as sd
from websockets.asyncio.client import connect

from .messages import SAY_ADAPTER, Say

OUTPUT_SAMPLES = 1920
OUTPUT_BYTES = OUTPUT_SAMPLES * 2
SILENCE = bytes(OUTPUT_BYTES)
DEFAULT_URL = "ws://127.0.0.1:17393/say"


async def _run(url: str, text: str, voice: str | None) -> None:
  playback: queue.SimpleQueue[bytes] = queue.SimpleQueue()

  def play(
    outdata: memoryview,
    frames: int,
    _time: object,
    _status: sd.CallbackFlags,
  ) -> None:
    if frames != OUTPUT_SAMPLES:
      outdata[:] = bytes(len(outdata))
      return
    try:
      outdata[:] = playback.get_nowait()
    except queue.Empty:
      outdata[:] = SILENCE

  with sd.RawOutputStream(
    samplerate=24_000,
    blocksize=OUTPUT_SAMPLES,
    channels=1,
    dtype="int16",
    callback=play,
  ):
    async with connect(url, compression=None) as websocket:
      await websocket.send(SAY_ADAPTER.dump_json(Say(text=text, voice=voice)).decode())
      async for message in websocket:
        if not isinstance(message, bytes):
          raise ValueError("/say returned a non-audio message")
        if len(message) != OUTPUT_BYTES:
          raise ValueError(f"expected {OUTPUT_BYTES} PCM bytes, got {len(message)}")
        playback.put(message)

    while not playback.empty():
      await asyncio.sleep(0.005)
    await asyncio.sleep(OUTPUT_SAMPLES / 24_000)


def main() -> None:
  parser = argparse.ArgumentParser(description="Speak text through remote Sleeper TTS")
  parser.add_argument("text", nargs="+", help="text to speak")
  parser.add_argument("--url", default=DEFAULT_URL, help="say WebSocket URL")
  parser.add_argument("--voice", help="TTS voice")
  args: argparse.Namespace = parser.parse_args()
  asyncio.run(_run(args.url, " ".join(args.text), args.voice))


if __name__ == "__main__":
  main()
