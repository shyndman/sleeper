"""Voice input: mic -> VAD -> {ASR + smart-turn | barge-in} -> turn queue."""

import queue
import threading
from collections import deque
from enum import Enum, auto
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

_logger = get_logger("asr")

# ---- Voice input (mic -> VAD -> {ASR + smart-turn | barge-in}) ----
ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
# Left/right context in 80ms frames: 160ms chunks. NeMo takes a list, so the
# frozen tuple is expanded with list(...) at the call site.
ASR_ATT_CONTEXT: tuple[int, int] = (70, 1)
SMART_TURN_ONNX = Path(__file__).parent / "models" / "smart-turn-v3.2-cpu.onnx"
SMART_TURN_WINDOW_SECONDS = 8
TURN_COMPLETE_THRESHOLD = 0.5  # smart-turn sigmoid; >= means the user is done
PREROLL_SECONDS = 0.3  # mic history replayed into ASR at capture start


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

  # One chunk supplies encoder lookahead; the second lets RNNT emit its tail.
  FINAL_PADDING_MULTIPLIER = ASR_ATT_CONTEXT[1] + 1

  def __init__(self) -> None:
    # Silence their extremely noisy logs
    from nemo.utils.nemo_logging import Logger

    nemo_logger = Logger()
    nemo_logger.remove_stream_handlers()

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
      self._step(final=False)
    return self.text

  def finish(self) -> str:
    """Close the utterance and emit text still waiting on trailing context."""
    padding_frames = self.FINAL_PADDING_MULTIPLIER * self.shift_frames
    self._audio = np.concatenate(
      [self._audio, np.zeros(padding_frames * self.HOP, dtype=np.float32)]
    )
    self._step(final=True)
    return self.text

  @torch.inference_mode()
  def _step(self, *, final: bool) -> None:
    sig = torch.from_numpy(self._audio).unsqueeze(0).to(self.model.device)
    sig_len = torch.tensor([len(self._audio)], device=self.model.device)
    mel, _ = self.model.preprocessor(input_signal=sig, length=sig_len)

    if self._emitted == 0:
      start, drop = 0, 0
    else:
      start = self._emitted - self.pre_cache_frames
      drop = self.drop_extra
    end = mel.shape[-1] if final else self._emitted + self.shift_frames

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
      keep_all_outputs=final,
      previous_hypotheses=self._hyps,
      previous_pred_out=self._pred,
      drop_extra_pre_encoded=drop,
      return_transcription=True,
    )
    self._emitted = end if final else self._emitted + self.shift_frames
    self.text = texts[0].text


class TurnState(Enum):
  """Whether a user turn is absent, recording speech, or waiting for continuation."""

  IDLE = auto()
  RECORDING = auto()
  AWAITING_SPEECH = auto()


class TurnDetector:
  """pipecat smart-turn v3: P(user finished their turn) from raw audio."""

  def __init__(self, path: Path) -> None:
    from transformers import WhisperFeatureExtractor

    self.extractor = WhisperFeatureExtractor(chunk_length=8)
    self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

  @staticmethod
  def _model_window(audio: np.ndarray) -> np.ndarray:
    """Place recent audio at the end of the model's fixed-length input."""
    window_samples = SMART_TURN_WINDOW_SECONDS * MIC_SR
    audio = audio[-window_samples:]
    return np.pad(audio, (window_samples - len(audio), 0))

  def complete_probability(self, audio: np.ndarray) -> float:
    """Probability the utterance is complete from its last eight seconds."""
    window_samples = SMART_TURN_WINDOW_SECONDS * MIC_SR
    inputs = self.extractor(
      self._model_window(audio),
      sampling_rate=MIC_SR,
      return_tensors="np",
      padding="max_length",
      max_length=window_samples,
      truncation=True,
      do_normalize=True,
    )
    feats = inputs.input_features.astype(np.float32)
    output = np.asarray(self.session.run(None, {"input_features": feats})[0])
    return float(output.ravel()[0])


class VoiceInputProcessor:
  """Route microphone blocks through barge-in or one persistent user turn."""

  def __init__(
    self,
    turns: queue.Queue[TurnQueueItem],
    session: ConversationSession,
  ) -> None:
    from silero_vad import load_silero_vad

    _logger.info("loading voice input models")
    vad = load_silero_vad()
    vad.reset_states()

    self.turns = turns
    self.session = session
    self.gate = SpeechGate(vad)
    self.bargein = BargeInDetector(BARGEIN_ONNX, BARGEIN_META)
    self.bargein.probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    self.turn_detector = TurnDetector(SMART_TURN_ONNX)
    self.turn_detector.complete_probability(np.zeros(MIC_SR, dtype=np.float32))
    self.asr = StreamingASR()
    self.asr.feed(np.zeros(MIC_SR, dtype=np.float32))
    self.asr.reset()

    self.preroll: deque[np.ndarray] = deque(
      maxlen=int(PREROLL_SECONDS * MIC_SR / VAD_FRAME_SAMPLES)
    )
    self.utterance: deque[np.ndarray] = deque(
      maxlen=int(SMART_TURN_WINDOW_SECONDS * MIC_SR / VAD_FRAME_SAMPLES)
    )
    self.bargein_window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    self.turn_state = TurnState.IDLE

  def process(self, chunk: np.ndarray) -> None:
    """Consume one microphone block according to the active speaker."""
    self.preroll.append(chunk)
    self.bargein_window = np.concatenate([self.bargein_window[len(chunk) :], chunk])
    speaking = self.gate.update(chunk)

    if self.session.is_assistant():
      self._process_assistant_audio(speaking)
    else:
      self._process_user_audio(chunk, speaking)

  def _process_assistant_audio(self, speaking: bool) -> None:
    self.turn_state = TurnState.IDLE
    if not speaking or self.bargein.probability(self.bargein_window) < self.bargein.threshold:
      return

    _logger.info("barge-in")
    self.session.barge_in()
    self._start_turn()

  def _process_user_audio(self, chunk: np.ndarray, speaking: bool) -> None:
    if self.turn_state is TurnState.IDLE:
      if speaking:
        self._start_turn()
      return

    if speaking:
      self.turn_state = TurnState.RECORDING
      self._record(chunk)
      return

    if self.turn_state is TurnState.AWAITING_SPEECH:
      return

    # Keep one VAD-negative block after the gate's hangover so Smart Turn sees
    # a natural acoustic endpoint, but do not keep recording a rejected pause.
    self._record(chunk)
    p_turn_end = self.turn_detector.complete_probability(np.concatenate(self.utterance))
    if p_turn_end < TURN_COMPLETE_THRESHOLD:
      # An incomplete semantic turn deliberately remains open indefinitely.
      # Only more speech followed by another VAD endpoint can reconsider it;
      # there is no timeout because observed use has not justified one.
      self.turn_state = TurnState.AWAITING_SPEECH
      return

    self._finish_turn()

  def _start_turn(self) -> None:
    self.asr.reset()
    self.utterance.clear()
    self.utterance.extend(self.preroll)
    for block in self.preroll:
      self.asr.feed(block)
    self.turn_state = TurnState.RECORDING

  def _record(self, chunk: np.ndarray) -> None:
    self.utterance.append(chunk)
    self.asr.feed(chunk)

  def _finish_turn(self) -> None:
    self.turn_state = TurnState.IDLE
    text = self.asr.finish().strip()
    ws = self.session.active_connection()
    if not text or ws is None:
      return

    _logger.info("user transcript", text=text)
    try:
      send_transcript(ws, TurnTranscript("user", text, "turn_detected"))
    except Exception:
      _logger.exception("user transcript send failed")
      self.session.return_to_user()
      return
    self.session.user_turn_finished()
    self.turns.put(QueuedTurn(ws, text))


def listen_worker(
  mic_frames: queue.Queue[np.ndarray],
  turns: queue.Queue[TurnQueueItem],
  session: ConversationSession,
  stopping: threading.Event,
) -> None:
  """Forward queued microphone blocks into the stateful voice-input processor."""
  processor = VoiceInputProcessor(turns, session)
  while not stopping.is_set():
    try:
      chunk = mic_frames.get(timeout=0.25)
    except queue.Empty:
      continue
    processor.process(chunk)
