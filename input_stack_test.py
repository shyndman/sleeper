#!/usr/bin/env python
"""Headless smoke test for the voice-input stack in serve_chat.py.

Streams an audio file through StreamingASR in mic-sized blocks (512 samples),
verifies the running transcript grows and reset() clears state, and checks
smart-turn scores a finished sentence higher than a mid-utterance cut.

Usage:
    uv run python input_stack_test.py [--file audio.mp3]
"""
import argparse
from pathlib import Path

import numpy as np
import sphn

from serve_chat import MIC_SR, SMART_TURN_ONNX, StreamingASR, TurnDetector

BLOCK = 512  # same mic blocksize the live loop uses

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(__file__).parent / "delayed-streams-modeling" / "audio" / "bria.mp3",
    )
    args = parser.parse_args()

    data, _ = sphn.read(str(args.file), sample_rate=MIC_SR)
    audio = (data.mean(axis=0) if data.ndim == 2 else data).astype(np.float32)
    print(f"{args.file.name}: {len(audio) / MIC_SR:.1f}s")

    asr = StreamingASR()
    for start in range(0, len(audio), BLOCK):
        asr.feed(audio[start : start + BLOCK])
    full_text = asr.text.strip()
    print(f"[asr] {full_text!r}")
    assert len(full_text.split()) > 5, "transcript suspiciously short"

    # reset() must give a genuinely fresh utterance, not a continuation.
    asr.reset()
    assert asr.text == ""
    for start in range(0, 3 * MIC_SR, BLOCK):
        asr.feed(audio[start : start + BLOCK])
    partial = asr.text.strip()
    print(f"[asr after reset, first 3s] {partial!r}")
    assert partial and full_text.lower().startswith(partial.split()[0].lower())

    turn = TurnDetector(SMART_TURN_ONNX)
    p_done = turn.complete_probability(audio)  # ends at a natural stop
    p_cut = turn.complete_probability(audio[: int(20 * MIC_SR)])  # mid-sentence
    print(f"[smart-turn] complete={p_done:.3f}  mid-sentence={p_cut:.3f}")
    assert p_done > 0.5 > p_cut, "smart-turn failed to separate done vs mid-sentence"

    print("all checks passed")

if __name__ == "__main__":
    main()
