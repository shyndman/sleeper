"""Speech gating and barge-in classification for assistant interruption."""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
import torchaudio

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
TARGET_FRAMES = 200
LOG_EPS = 1e-6

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "bargein.onnx"
META_PATH = MODEL_DIR / "bargein.onnx.meta.npz"

VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = VAD_FRAME_SAMPLES / SAMPLE_RATE * 1000
VAD_THRESHOLD = 0.5
VAD_HANGOVER_MS = 250


class VoiceActivityModel(Protocol):
  def __call__(self, x: torch.Tensor, sr: int) -> torch.Tensor: ...


class BargeInDetector:
  """Classify speech as a genuine attempt to interrupt the assistant."""

  def __init__(self, model_path: Path, meta_path: Path) -> None:
    self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    self.input_name = self.session.get_inputs()[0].name

    meta = np.load(str(meta_path), allow_pickle=True)
    self.feature_mean = meta["feature_mean"]
    self.feature_std = meta["feature_std"]
    self.threshold = float(meta["threshold"][0])

    # This exactly matches the classifier's torch_cnn training backend. The
    # natural-log floor must align with the normalization statistics shipped
    # beside the model or its probabilities become meaningless.
    self.melspec = torchaudio.transforms.MelSpectrogram(
      sample_rate=SAMPLE_RATE,
      n_fft=N_FFT,
      hop_length=HOP_LENGTH,
      n_mels=N_MELS,
      power=2.0,
    )

  def probability(self, window: npt.NDArray[np.float32]) -> float:
    """Return interruption probability for two seconds of mono float32 PCM."""
    samples = torch.from_numpy(window).float()
    logmel = torch.log(self.melspec(samples[None]) + LOG_EPS)[0].numpy()
    logmel = logmel[:, :TARGET_FRAMES]
    features = ((logmel[None, None] - self.feature_mean) / self.feature_std).astype(np.float32)
    output = np.asarray(self.session.run(None, {self.input_name: features})[0])
    logit = output.ravel()[0]
    return float(1.0 / (1.0 + np.exp(-logit)))


class SpeechGate:
  """Report active speech with a short hangover between VAD-positive frames."""

  def __init__(
    self,
    model: VoiceActivityModel,
    threshold: float = VAD_THRESHOLD,
    hangover_ms: int = VAD_HANGOVER_MS,
  ) -> None:
    self.model = model
    self.threshold = threshold
    self.hangover_frames = max(1, round(hangover_ms / VAD_FRAME_MS))
    self._buffer = np.empty(0, dtype=np.float32)
    self._since_speech = self.hangover_frames

  def update(self, chunk: npt.NDArray[np.float32]) -> bool:
    """Consume PCM and report whether speech is active after hangover."""
    self._buffer = np.concatenate([self._buffer, chunk])

    while len(self._buffer) >= VAD_FRAME_SAMPLES:
      frame = self._buffer[:VAD_FRAME_SAMPLES]
      self._buffer = self._buffer[VAD_FRAME_SAMPLES:]
      probability = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()
      self._since_speech = 0 if probability >= self.threshold else self._since_speech + 1
    return self._since_speech < self.hangover_frames
