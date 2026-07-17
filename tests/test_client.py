"""Deterministic tests for client-local playback gain and key parsing."""

import numpy as np

from sleeper.client import (
  MAX_GAIN_DECIBELS,
  MIN_GAIN_DECIBELS,
  OUTPUT_BYTES,
  OUTPUT_SAMPLES,
  PlaybackGain,
  _consume_gain_keys,
  _write_playback_frame,
)


def test_gain_starts_at_unity() -> None:
  assert PlaybackGain().snapshot() == (0, 1.0)


def test_gain_steps_by_supplied_deltas() -> None:
  gain = PlaybackGain()
  assert gain.adjust(2)[0] == 2
  assert gain.adjust(2)[0] == 4
  assert gain.adjust(-2)[0] == 2


def test_gain_clamps_at_bounds() -> None:
  gain = PlaybackGain()
  for _ in range(20):
    gain.adjust(2)
  assert gain.snapshot()[0] == MAX_GAIN_DECIBELS
  for _ in range(40):
    gain.adjust(-2)
  assert gain.snapshot()[0] == MIN_GAIN_DECIBELS


def test_fragmented_and_concatenated_sequences_parse_in_order() -> None:
  # An escape split across reads yields nothing until it completes.
  buffer = bytearray(b"\x1b")
  assert _consume_gain_keys(buffer) == ()
  buffer.extend(b"[A")
  assert _consume_gain_keys(buffer) == (2,)
  # Two sequences concatenated in one read parse to ordered adjustments.
  assert _consume_gain_keys(bytearray(b"\x1b[A\x1b[B")) == (2, -2)


def test_unrelated_bytes_are_ignored() -> None:
  assert _consume_gain_keys(bytearray(b"q\x1b[A\x1b[C")) == (2,)


def test_unity_gain_writes_bytes_unchanged() -> None:
  frame = np.arange(OUTPUT_SAMPLES, dtype="<i2").tobytes()
  outdata = bytearray(OUTPUT_BYTES)
  scratch = np.empty(OUTPUT_SAMPLES, dtype=np.float32)
  _write_playback_frame(memoryview(outdata), frame, 1.0, scratch)
  assert bytes(outdata) == frame


def test_positive_gain_increases_magnitude() -> None:
  frame = np.full(OUTPUT_SAMPLES, 100, dtype="<i2").tobytes()
  outdata = bytearray(OUTPUT_BYTES)
  scratch = np.empty(OUTPUT_SAMPLES, dtype=np.float32)
  _write_playback_frame(memoryview(outdata), frame, 2.0, scratch)
  result = np.frombuffer(bytes(outdata), dtype="<i2")
  assert np.all(result == 200)


def test_gain_saturates_without_wrapping() -> None:
  frame = np.array([30000, -30000], dtype="<i2").tobytes()
  outdata = bytearray(4)
  scratch = np.empty(2, dtype=np.float32)
  _write_playback_frame(memoryview(outdata), frame, 4.0, scratch)
  result = np.frombuffer(bytes(outdata), dtype="<i2")
  assert result[0] == np.iinfo(np.int16).max
  assert result[1] == np.iinfo(np.int16).min
