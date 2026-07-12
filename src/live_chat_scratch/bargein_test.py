#!/usr/bin/env python
"""Standalone smoke test for the bnovikov/bargein-classifier ONNX model.

Streams audio through the barge-in classifier and prints turn-state changes as
they happen: BACKCHANNEL (user is just acknowledging / silent) <-> BARGE-IN
(user is genuinely interrupting). Feature extraction mirrors the model's
`torch_cnn` training backend: a 64-band natural-log power mel-spectrogram over a
2-second window, per-bin standardized with the mean/std shipped in the model's
`.meta.npz`, then compared against the model's recall-tuned threshold.

Usage:
    uv run python bargein_test.py                 # live microphone
    uv run python bargein_test.py --file a.mp3    # stream an audio file
    uv run python bargein_test.py --file a.mp3 --fast   # no realtime pacing
"""

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

# --- Fixed feature/streaming geometry (from bargein.onnx.meta feature_config) ---
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000  # 2s classification window the model was trained on
HOP_SAMPLES = 1600  # 100ms sliding-window hop, per the model card
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
TARGET_FRAMES = 200  # 2s / 160-hop, the model's fixed time dimension
LOG_EPS = 1e-6  # natural-log floor; feature_mean bottoms out near log(eps)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "bargein.onnx"
META_PATH = MODEL_DIR / "bargein.onnx.meta.npz"

BARGE_IN = "BARGE-IN"
BACKCHANNEL = "BACKCHANNEL"
IDLE = "IDLE"

VAD_FRAME_SAMPLES = 512  # Silero requires exactly 512-sample frames at 16 kHz
VAD_THRESHOLD = 0.5
VAD_HANGOVER_MS = 250  # keep the gate open this long after the last speech frame


class BargeInDetector:
    """Wraps the ONNX classifier plus its exact feature pipeline."""

    def __init__(self, model_path: Path, meta_path: Path):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        meta = np.load(str(meta_path), allow_pickle=True)
        self.feature_mean = meta["feature_mean"]  # (1, 64, 200)
        self.feature_std = meta["feature_std"]  # (1, 64, 200)
        self.threshold = float(meta["threshold"][0])

        # torchaudio's MelSpectrogram (power=2.0, htk mel scale, no norm) matches
        # the torch_cnn training backend; the natural log is applied separately so
        # the LOG_EPS floor lines up with the shipped normalization stats.
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )

    def probability(self, window: np.ndarray) -> float:
        """Barge-in probability for one 2s window of float32 mono PCM."""
        x = torch.from_numpy(window).float()
        logmel = torch.log(self.melspec(x[None]) + LOG_EPS)[0].numpy()  # (64, T)
        logmel = logmel[:, :TARGET_FRAMES]
        feat = ((logmel[None, None] - self.feature_mean) / self.feature_std).astype(
            np.float32
        )
        logit = self.session.run(None, {self.input_name: feat})[0].ravel()[0]
        return float(1.0 / (1.0 + np.exp(-logit)))


class StateTracker:
    """Prints only when the reported state flips (IDLE / BACKCHANNEL / BARGE-IN)."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.state: str | None = None

    def _emit(self, audio_time_s: float, new_state: str, prob: float | None) -> None:
        if new_state != self.state:
            arrow = "--" if self.state is None else "->"
            prev = "start" if self.state is None else self.state
            tail = f" (p={prob:.3f})" if prob is not None else ""
            print(
                f"[{audio_time_s:7.2f}s] {prev:>11} {arrow} {new_state:<11}{tail}",
                flush=True,
            )
            self.state = new_state

    def classify(self, audio_time_s: float, prob: float) -> None:
        self._emit(audio_time_s, BARGE_IN if prob >= self.threshold else BACKCHANNEL, prob)

    def idle(self, audio_time_s: float) -> None:
        self._emit(audio_time_s, IDLE, None)

class SpeechGate:
    """Silero-VAD gate reporting whether the user is currently speaking.

    The barge-in classifier is only meaningful on actual speech (it assumes
    speech is present and asks interrupt-vs-backchannel), so we run it only while
    this gate is open. This mirrors the model card's "place downstream of VAD".
    """

    def __init__(self, model, threshold: float = VAD_THRESHOLD, hangover_ms: int = VAD_HANGOVER_MS):
        self.model = model
        self.threshold = threshold
        frame_ms = VAD_FRAME_SAMPLES / SAMPLE_RATE * 1000
        self.hangover_frames = max(1, round(hangover_ms / frame_ms))
        self._buf = np.empty(0, dtype=np.float32)
        self._since_speech = self.hangover_frames  # start closed

    def update(self, chunk: np.ndarray) -> bool:
        """Feed a chunk; return True while speech is active (with hangover)."""
        self._buf = np.concatenate([self._buf, chunk])
        while len(self._buf) >= VAD_FRAME_SAMPLES:
            frame = self._buf[:VAD_FRAME_SAMPLES]
            self._buf = self._buf[VAD_FRAME_SAMPLES:]
            prob = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()
            self._since_speech = 0 if prob >= self.threshold else self._since_speech + 1
        return self._since_speech < self.hangover_frames


def run_stream(detector: BargeInDetector, chunks, tracker: StateTracker, gate=None) -> None:
    """Feed 100ms chunks through a rolling 2s buffer; classify only while the gate is open."""
    buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    processed = 0
    for chunk in chunks:
        n = len(chunk)
        buffer = np.concatenate([buffer[n:], chunk])[-WINDOW_SAMPLES:]
        processed += n
        audio_time_s = processed / SAMPLE_RATE
        if gate is not None and not gate.update(chunk):
            tracker.idle(audio_time_s)
            continue
        tracker.classify(audio_time_s, detector.probability(buffer))


def file_chunks(path: Path, fast: bool):
    """Yield 100ms mono chunks from an audio file, paced to realtime by default."""
    import sphn

    data, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    audio = data.mean(axis=0) if data.ndim == 2 else data
    audio = audio.astype(np.float32)
    hop_seconds = HOP_SAMPLES / SAMPLE_RATE
    for start in range(0, len(audio), HOP_SAMPLES):
        yield audio[start : start + HOP_SAMPLES]
        if not fast:
            time.sleep(hop_seconds)


def mic_chunks(device=None):
    """Yield 100ms mono chunks from an input device (default when device is None)."""
    import sounddevice as sd

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    name = sd.query_devices(device)["name"] if device is not None else "default"
    print(f"Listening on: {name}. Ctrl-C to stop.", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=HOP_SAMPLES,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        while True:
            yield q.get()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", type=Path, help="Audio file to stream instead of the microphone."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Process a --file as fast as possible (no realtime pacing).",
    )
    parser.add_argument(
        "--device",
        help="Input device index or name (e.g. easyeffects_source for AEC).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the barge-in probability threshold (default from model meta).",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable the Silero VAD gate and classify every window.",
    )
    args = parser.parse_args()

    detector = BargeInDetector(MODEL_PATH, META_PATH)
    detector.probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32))  # warm up ORT
    if args.threshold is not None:
        detector.threshold = args.threshold
    tracker = StateTracker(detector.threshold)
    print(f"threshold = {detector.threshold:.3f}", flush=True)

    device = None
    if args.device is not None:
        device = int(args.device) if args.device.isdigit() else args.device
    chunks = file_chunks(args.file, args.fast) if args.file else mic_chunks(device)
    gate = None
    if not args.no_vad:
        from silero_vad import load_silero_vad

        vad = load_silero_vad()
        vad.reset_states()
        gate = SpeechGate(vad)
    try:
        run_stream(detector, chunks, tracker, gate)
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)


if __name__ == "__main__":
    main()
