"""Remote microphone and speaker client for Sleeper conversations."""

import argparse
import asyncio
import contextlib
import os
import queue
import sys
import termios
import tty
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import sounddevice as sd
from libsh import get_logger, setup_logging_from_env
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .messages import TURN_TRANSCRIPT_ADAPTER

INPUT_SAMPLES = 512
INPUT_BYTES = INPUT_SAMPLES * 2
OUTPUT_SAMPLES = 1920
OUTPUT_BYTES = OUTPUT_SAMPLES * 2
SILENCE = bytes(OUTPUT_BYTES)
DEFAULT_URL = "ws://127.0.0.1:17393/conversation"

# Gain resets to 0 dB (unity) on each launch and steps in 2 dB increments,
# clamped to the inclusive range below. It scales only Sleeper playback
# samples; it never touches the operating system's output volume.
DEFAULT_GAIN_DECIBELS = 0
GAIN_STEP_DECIBELS = 2
MIN_GAIN_DECIBELS = -12
MAX_GAIN_DECIBELS = 24
_UP_SEQUENCE = b"\x1b[A"
_DOWN_SEQUENCE = b"\x1b[B"

_logger = get_logger("client")


class PlaybackGain:
  """Client-local playback gain stored as one immutable snapshot.

  The level lives in a single ``(decibels, multiplier)`` tuple that ``adjust``
  replaces atomically. The real-time audio callback reads that tuple via
  ``snapshot`` and therefore always sees either the complete old or the
  complete new value without taking a lock.
  """

  def __init__(self) -> None:
    self._snapshot: tuple[int, float] = (
      DEFAULT_GAIN_DECIBELS,
      10.0 ** (DEFAULT_GAIN_DECIBELS / 20.0),
    )

  def snapshot(self) -> tuple[int, float]:
    return self._snapshot

  def adjust(self, delta_decibels: int) -> tuple[int, float]:
    decibels = max(MIN_GAIN_DECIBELS, min(MAX_GAIN_DECIBELS, self._snapshot[0] + delta_decibels))
    snapshot = (decibels, 10.0 ** (decibels / 20.0))
    self._snapshot = snapshot
    return snapshot


def _write_playback_frame(
  outdata: memoryview,
  frame: bytes,
  multiplier: float,
  scratch: npt.NDArray[np.float32],
) -> None:
  """Scale one PCM frame by ``multiplier`` into ``outdata`` without wrapping."""
  if multiplier == 1.0:
    # Unity gain is bit-for-bit identical to the source and allocation-free.
    outdata[:] = frame
    return
  samples = np.frombuffer(frame, dtype="<i2")
  np.multiply(samples, multiplier, out=scratch)
  np.clip(scratch, np.iinfo(np.int16).min, np.iinfo(np.int16).max, out=scratch)
  outdata[:] = scratch.astype("<i2").tobytes()


def _consume_gain_keys(buffer: bytearray) -> tuple[int, ...]:
  """Drain complete Kitty Up/Down escape sequences into dB adjustments.

  Complete Up (``ESC [ A``) and Down (``ESC [ B``) sequences are removed and
  mapped to +/- ``GAIN_STEP_DECIBELS``. An incomplete trailing escape is left
  in ``buffer`` for the next read; any other byte is discarded.
  """
  adjustments: list[int] = []
  while buffer:
    if buffer[0] != 0x1B:
      del buffer[0]
      continue
    if len(buffer) < 3:
      break
    prefix = bytes(buffer[:3])
    if prefix == _UP_SEQUENCE:
      adjustments.append(GAIN_STEP_DECIBELS)
      del buffer[:3]
    elif prefix == _DOWN_SEQUENCE:
      adjustments.append(-GAIN_STEP_DECIBELS)
      del buffer[:3]
    else:
      del buffer[0]
  return tuple(adjustments)


@contextlib.contextmanager
def _gain_controls(gain: PlaybackGain) -> Iterator[None]:
  # HACK: This single-user Kitty debug client parses arrow-key escape
  # sequences directly off stdin rather than adding a keyboard dependency.
  # Buffering across reads is required because one escape sequence
  # (ESC '[' 'A') may be split across separate os.read() calls, so an
  # incomplete tail is retained until the remaining bytes arrive.
  fd = sys.stdin.fileno()
  saved = termios.tcgetattr(fd)
  buffer = bytearray()
  loop = asyncio.get_running_loop()

  def on_readable() -> None:
    chunk = os.read(fd, 1024)
    if not chunk:
      return
    buffer.extend(chunk)
    for delta in _consume_gain_keys(buffer):
      decibels, multiplier = gain.adjust(delta)
      _logger.info("playback gain set", decibels=decibels, multiplier=multiplier)

  # cbreak leaves ISIG enabled so Ctrl-C keeps generating SIGINT normally.
  tty.setcbreak(fd)
  loop.add_reader(fd, on_readable)
  try:
    yield
  finally:
    loop.remove_reader(fd)
    termios.tcsetattr(fd, termios.TCSADRAIN, saved)


async def _send_mic(websocket: ClientConnection, microphone: queue.Queue[bytes]) -> None:
  while True:
    try:
      frame = microphone.get_nowait()
    except queue.Empty:
      await asyncio.sleep(0.005)
      continue

    await websocket.send(frame)


def _flush(playback: queue.SimpleQueue[bytes] | queue.Queue[bytes]) -> None:
  while True:
    try:
      playback.get_nowait()
    except queue.Empty:
      return


async def _receive(websocket: ClientConnection, playback: queue.SimpleQueue[bytes]) -> None:
  async for message in websocket:
    if isinstance(message, bytes):
      if len(message) != OUTPUT_BYTES:
        raise ValueError(f"expected {OUTPUT_BYTES} PCM bytes, got {len(message)}")
      playback.put(message)
      continue

    transcript = TURN_TRANSCRIPT_ADAPTER.validate_json(message)
    _logger.info(
      "transcript",
      role=transcript.role,
      text=transcript.text,
      ended_by=transcript.ended_by,
    )
    if transcript.role == "assistant" and transcript.ended_by == "interrupted":
      _flush(playback)


async def _run(url: str) -> None:
  microphone: queue.Queue[bytes] = queue.Queue(maxsize=32)
  playback: queue.SimpleQueue[bytes] = queue.SimpleQueue()
  gain = PlaybackGain()
  scratch = np.empty(OUTPUT_SAMPLES, dtype=np.float32)

  def capture(
    indata: memoryview,
    frames: int,
    _time: object,
    _status: sd.CallbackFlags,
  ) -> None:
    if frames != INPUT_SAMPLES:
      return
    with contextlib.suppress(queue.Full):
      microphone.put_nowait(bytes(indata))

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
      frame = playback.get_nowait()
    except queue.Empty:
      outdata[:] = SILENCE
      return
    _, multiplier = gain.snapshot()
    _write_playback_frame(outdata, frame, multiplier, scratch)

  with contextlib.ExitStack() as streams:
    streams.enter_context(
      sd.RawInputStream(
        samplerate=16_000,
        blocksize=INPUT_SAMPLES,
        channels=1,
        dtype="int16",
        callback=capture,
      )
    )
    _logger.info("microphone acquired")
    streams.enter_context(
      sd.RawOutputStream(
        samplerate=24_000,
        blocksize=OUTPUT_SAMPLES,
        channels=1,
        dtype="int16",
        callback=play,
      )
    )
    streams.enter_context(_gain_controls(gain))
    while True:
      try:
        _logger.info("connecting", url=url)
        async with connect(url, compression=None) as websocket:
          _logger.info("connected", url=url)
          sender = asyncio.create_task(_send_mic(websocket, microphone))
          receiver = asyncio.create_task(_receive(websocket, playback))
          done, pending = await asyncio.wait(
            (sender, receiver), return_when=asyncio.FIRST_COMPLETED
          )
          for task in pending:
            task.cancel()
          await asyncio.gather(*pending, return_exceptions=True)
          for task in done:
            task.result()

      except (OSError, ConnectionClosed, TimeoutError):
        _logger.exception("connection lost; retrying")

      _flush(microphone)
      _flush(playback)
      await asyncio.sleep(1)


def main() -> None:
  setup_logging_from_env()
  parser = argparse.ArgumentParser(description="Run a remote Sleeper conversation")
  parser.add_argument("--url", default=DEFAULT_URL, help="conversation WebSocket URL")
  args: argparse.Namespace = parser.parse_args()
  asyncio.run(_run(args.url))


if __name__ == "__main__":
  main()
