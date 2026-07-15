"""Headless smoke test for the voice-input stack in voice_input.py.

Streams bria.mp3 through StreamingASR in mic-sized blocks (512 samples),
verifies the running transcript grows and reset() clears state, and checks
smart-turn scores a finished sentence higher than a mid-utterance cut.
"""

from pathlib import Path

import numpy as np
import pytest
import sphn
import torch

from sleeper.voice_input import MIC_SR, SMART_TURN_ONNX, StreamingASR, TurnDetector

AUDIO_FILE = Path(__file__).parent / "data" / "bria.mp3"
BLOCK = 512  # same mic blocksize the live loop uses


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
  data, _ = sphn.read(str(AUDIO_FILE), sample_rate=MIC_SR)
  return (data.mean(axis=0) if data.ndim == 2 else data).astype(np.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="StreamingASR requires CUDA")
def test_asr_transcribes_and_resets(audio: np.ndarray) -> None:
  asr = StreamingASR()
  for start in range(0, len(audio), BLOCK):
    asr.feed(audio[start : start + BLOCK])
  full_text = asr.text.strip()
  assert len(full_text.split()) > 5, "transcript suspiciously short"

  # reset() must give a genuinely fresh utterance, not a continuation.
  asr.reset()
  assert asr.text == ""
  for start in range(0, 3 * MIC_SR, BLOCK):
    asr.feed(audio[start : start + BLOCK])
  partial = asr.text.strip()
  assert partial and full_text.lower().startswith(partial.split()[0].lower())


def test_smart_turn_separates_done_from_mid_sentence(audio: np.ndarray) -> None:
  turn = TurnDetector(SMART_TURN_ONNX)
  p_done = turn.complete_probability(audio)  # ends at a natural stop
  p_cut = turn.complete_probability(audio[: int(20 * MIC_SR)])  # mid-sentence
  assert p_done > 0.5 > p_cut, "smart-turn failed to separate done vs mid-sentence"
