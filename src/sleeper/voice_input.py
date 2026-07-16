"""Voice input: mic -> VAD -> {ASR + smart-turn | barge-in} -> turn queue."""

import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from libsh import get_logger
from omegaconf import open_dict

from sleeper.conversation import ConversationSession, QueuedTurn, TurnQueueItem, send_transcript
from sleeper.interruption import (
  META_PATH as BARGEIN_META,
)
from sleeper.interruption import (
  MODEL_PATH as BARGEIN_ONNX,
)
from sleeper.interruption import (
  SAMPLE_RATE as MIC_SR,
)
from sleeper.interruption import (
  VAD_FRAME_SAMPLES,
  WINDOW_SAMPLES,
  BargeInDetector,
  SpeechGate,
)
from sleeper.messages import TurnTranscript

_logger = get_logger("voice")

# ---- Voice input (mic -> VAD -> {ASR + smart-turn | barge-in}) ----
ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
# Left/right context in 80ms frames: 160ms chunks. NeMo takes a list, so the
# frozen tuple is expanded with list(...) at the call site.
ASR_ATT_CONTEXT: tuple[int, int] = (70, 1)
SMART_TURN_ONNX = Path(__file__).parent / "models" / "smart-turn-v3.2-cpu.onnx"
TURN_COMPLETE_THRESHOLD = 0.5  # smart-turn sigmoid; >= means the user is done
TURN_CHECK_EVERY_BLOCKS = 16  # re-run smart-turn every ~512ms of silence
FORCE_TURN_END_MS = 2500  # stop waiting on smart-turn after this much silence
PREROLL_SECONDS = 2.0  # mic history replayed into ASR at capture start


class StreamingASR:
  """Nemotron cache-aware streaming ASR: feed 16kHz float32, read .text.

  Follows pipecat's production integration (nemotron-january-2026 server.py)
  rather than NVIDIA's mic notebook: every step re-runs the model's own mel
  preprocessor over ALL audio accumulated this utterance, then slices out
  only the new frames for conformer_stream_step. Computing mel per-chunk
  corrupts frames at every chunk seam (the center-padded STFT reflects
  across boundaries) and measurably drops words; whole-buffer mel keeps
  every frame exact while the encoder still runs incrementally on caches.
  The last mel frame is always excluded because its STFT window extends
  past the audio we have so far. .text is the full running transcript,
  re-decoded from the whole RNNT token sequence every step.
  """

  HOP = 160  # preprocessor hop: one mel frame per 10ms at 16kHz

  def __init__(self) -> None:
    import nemo.collections.asr as nemo_asr  # slow import; keep it local

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL)
    if isinstance(model, str):
      raise TypeError(f"Expected an ASR model, got artifact path: {model}")
    model.eval()
    self.model = model.cuda()

    # Runtime latency knob; rebuilds encoder.streaming_cfg for the sizes below.
    self.model.encoder.set_default_att_context_size(list(ASR_ATT_CONTEXT))
    decoding = self.model.cfg.decoding
    with open_dict(decoding):
      decoding.strategy = "greedy"
      decoding.preserve_alignments = False
      decoding.greedy.max_symbols = 10
      decoding.fused_batch_size = -1
    self.model.change_decoding_strategy(decoding)
    self.model.preprocessor.featurizer.dither = 0.0  # deterministic mel

    scfg = self.model.encoder.streaming_cfg
    self.shift_frames: int = scfg.shift_size[1]  # 16 mel frames = 160ms
    self.pre_cache_frames: int = scfg.pre_encode_cache_size[1]  # 9 frames
    self.drop_extra: int = scfg.drop_extra_pre_encoded
    self.text = ""

    self.reset()

  def reset(self) -> None:
    """Fresh utterance: new encoder caches, no hypotheses, empty transcript."""
    enc = self.model.encoder
    self._cache_ch, self._cache_t, self._cache_len = enc.get_initial_cache_state(batch_size=1)
    self._hyps = None
    self._pred = None

    # ponytail: per-utterance audio buffer grows unbounded; mel over it is
    # O(n) per 160ms step but trivial on GPU for conversation-length turns.
    self._audio = np.empty(0, dtype=np.float32)
    self._emitted = 0  # mel frames already consumed by the encoder
    self.text = ""

  def feed(self, chunk: np.ndarray) -> str:
    """Buffer arbitrary-sized audio; step the encoder per 160ms of new mel."""
    self._audio = np.concatenate([self._audio, chunk])

    # +1: the trailing edge frame is never consumed (incomplete STFT window).
    while len(self._audio) >= (self._emitted + self.shift_frames + 1) * self.HOP:
      self._step()
    return self.text

  @torch.inference_mode()
  def _step(self) -> None:
    sig = torch.from_numpy(self._audio).unsqueeze(0).to(self.model.device)
    sig_len = torch.tensor([len(self._audio)], device=self.model.device)
    mel, _ = self.model.preprocessor(input_signal=sig, length=sig_len)

    if self._emitted == 0:
      start, end, drop = 0, self.shift_frames, 0
    else:
      start = self._emitted - self.pre_cache_frames
      end = self._emitted + self.shift_frames
      drop = self.drop_extra

    chunk_mel = mel[:, :, start:end]
    chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.model.device)
    (
      self._pred,
      texts,
      self._cache_ch,
      self._cache_t,
      self._cache_len,
      self._hyps,
    ) = self.model.conformer_stream_step(
      processed_signal=chunk_mel,
      processed_signal_length=chunk_len,
      cache_last_channel=self._cache_ch,
      cache_last_time=self._cache_t,
      cache_last_channel_len=self._cache_len,
      keep_all_outputs=False,  # a mic stream never "ends"
      previous_hypotheses=self._hyps,
      previous_pred_out=self._pred,
      drop_extra_pre_encoded=drop,
      return_transcription=True,
    )
    self._emitted += self.shift_frames
    self.text = texts[0].text


class TurnDetector:
  """pipecat smart-turn v3: P(user finished their turn) from raw audio."""

  def __init__(self, path: Path) -> None:
    from transformers import WhisperFeatureExtractor

    self.extractor = WhisperFeatureExtractor(chunk_length=8)
    self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

  def complete_probability(self, audio: np.ndarray) -> float:
    """Probability the utterance (last <=8s kept, per model window) is complete."""
    inputs = self.extractor(
      audio[-8 * MIC_SR :],
      sampling_rate=MIC_SR,
      return_tensors="np",
      padding="max_length",
      max_length=8 * MIC_SR,
      truncation=True,
      do_normalize=True,
    )
    feats = inputs.input_features.astype(np.float32)
    output = np.asarray(self.session.run(None, {"input_features": feats})[0])
    return float(output.ravel()[0])


def listen_worker(
  mic_frames: queue.Queue[np.ndarray],
  turns: queue.Queue[TurnQueueItem],
  session: ConversationSession,
  stopping: threading.Event,
) -> None:
  """Consume 512-sample remote mic frames; own all voice-input models."""
  from silero_vad import load_silero_vad

  _logger.info("loading voice input models")

  vad = load_silero_vad()
  vad.reset_states()
  gate = SpeechGate(vad)
  bargein = BargeInDetector(BARGEIN_ONNX, BARGEIN_META)
  bargein.probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
  turn_detector = TurnDetector(SMART_TURN_ONNX)
  turn_detector.complete_probability(np.zeros(MIC_SR, dtype=np.float32))
  asr = StreamingASR()
  asr.feed(np.zeros(MIC_SR, dtype=np.float32))
  asr.reset()

  preroll: deque[np.ndarray] = deque(maxlen=int(PREROLL_SECONDS * MIC_SR / VAD_FRAME_SAMPLES))
  utterance: deque[np.ndarray] = deque(maxlen=int(8 * MIC_SR / VAD_FRAME_SAMPLES))
  window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
  capturing = False
  silence_blocks = 0
  force_blocks = int(FORCE_TURN_END_MS / 1000 * MIC_SR / VAD_FRAME_SAMPLES)

  def start_capture() -> None:
    nonlocal capturing, silence_blocks
    asr.reset()
    utterance.clear()
    utterance.extend(preroll)
    for block in preroll:
      asr.feed(block)
    capturing = True
    silence_blocks = 0

  while not stopping.is_set():
    try:
      chunk = mic_frames.get(timeout=0.25)
    except queue.Empty:
      continue
    preroll.append(chunk)
    window = np.concatenate([window[len(chunk) :], chunk])
    speaking = gate.update(chunk)

    if session.is_assistant():
      capturing = False
      if speaking and bargein.probability(window) >= bargein.threshold:
        _logger.info("barge-in")
        session.barge_in()
        start_capture()
      continue

    if not capturing:
      if speaking:
        start_capture()
      continue
    utterance.append(chunk)
    asr.feed(chunk)

    if speaking:
      silence_blocks = 0
      continue
    silence_blocks += 1
    if silence_blocks % TURN_CHECK_EVERY_BLOCKS == 1:
      ended = (
        turn_detector.complete_probability(np.concatenate(utterance)) >= TURN_COMPLETE_THRESHOLD
      )
    else:
      ended = silence_blocks >= force_blocks

    if ended:
      capturing = False
      text = asr.text.strip()
      ws = session.active_connection()
      if text and ws is not None:
        _logger.info("user transcript", text=text)
        try:
          send_transcript(ws, TurnTranscript("user", text, "turn_detected"))
        except Exception:
          capturing = False
          session.return_to_user()
          continue
        session.user_turn_finished()
        turns.put(QueuedTurn(ws, text))
