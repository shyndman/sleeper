"""Streaming Kyutai TTS synthesis and the single FIFO speech worker."""

import queue
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from libsh import get_logger
from moshi.conditioners import ConditionAttributes, dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.tts import TTSModel, script_to_entries
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.server import ServerConnection

if TYPE_CHECKING:
  from sleeper.conversation import ConversationSession

DEFAULT_VOICE = "cml-tts/fr/9834_9697_000150-0003_enhanced.wav"

_logger = get_logger("tts")

# One resident synth serves two producers. A conversation turn streams in as
# individual SpeakWord jobs -- words are fed to the generator as the LLM emits
# them -- and is closed by a single EndOfTurn. A say job is a self-contained
# utterance outside the conversation state machine; its audio is not tracked
# for playback/interruption.


@dataclass(slots=True, frozen=True)
class SpeakWord:
  """One whitespace-delimited word of the open assistant turn."""

  target: ServerConnection
  text: str


@dataclass(slots=True, frozen=True)
class EndOfTurn:
  """Close the open assistant turn; `done` fires after flush or abort."""

  done: threading.Event


@dataclass(slots=True, frozen=True)
class SayJob:
  """Self-contained /say utterance: spoken and flushed as one unit."""

  ws: ServerConnection
  voice: str | None
  text: str
  done: threading.Event


@dataclass(slots=True, frozen=True)
class SetConversationVoice:
  """Change the voice used when the next assistant turn opens."""

  voice: str
  result: Future[None]


type SpeechQueueItem = SpeakWord | EndOfTurn | SayJob | SetConversationVoice | None


class TTS:
  """Serialize conversation speech and isolated `/say` requests through one synth."""

  def __init__(self) -> None:
    self._jobs: queue.Queue[SpeechQueueItem] = queue.Queue()
    # Admission and termination share one lock so a job either lands ahead of
    # the sentinel (and is guaranteed to run and signal its waiter) or is
    # rejected outright. Without this a producer could enqueue behind the
    # sentinel and block forever on a worker that has already exited.
    self._lock = threading.Lock()
    self._stopped = False

  def _try_enqueue(self, job: SpeakWord | EndOfTurn | SayJob | SetConversationVoice) -> bool:
    """Admit a job to the FIFO, or refuse it once shutdown has begun."""
    with self._lock:
      if self._stopped:
        return False
      self._jobs.put(job)
      return True

  def speak_word(self, target: ServerConnection, text: str) -> None:
    """Append one complete word to the current streaming assistant turn."""
    # Speech dropped after shutdown is silent: there is no waiter to release.
    self._try_enqueue(SpeakWord(target, text))

  def end_turn(self, done: threading.Event) -> None:
    """Flush or abort the assistant turn, then signal its waiter."""
    # A rejected close still must release wait_for_cleanup(), so signal here.
    if not self._try_enqueue(EndOfTurn(done)):
      done.set()

  def say(self, ws: ServerConnection, voice: str | None, text: str) -> None:
    """Speak one isolated `/say` request and block until synthesis finishes."""
    done = threading.Event()
    # Fail fast instead of blocking on a worker that will never dequeue this.
    if not self._try_enqueue(SayJob(ws, voice, text, done)):
      raise RuntimeError("TTS worker has stopped")
    done.wait()

  def set_conversation_voice(self, voice: str) -> None:
    """Change the conversation voice and block until the worker validates it."""
    result: Future[None] = Future()
    # Fail fast instead of blocking on a worker that will never dequeue this.
    if not self._try_enqueue(SetConversationVoice(voice, result)):
      raise RuntimeError("TTS worker has stopped")
    # Block so the endpoint cannot acknowledge an unvalidated selection; a
    # failed voice lookup/conditioning build re-raises here.
    result.result()

  def stop(self) -> None:
    with self._lock:
      if self._stopped:
        return
      self._stopped = True
      self._jobs.put(None)

  def run(
    self,
    tts_model: TTSModel,
    session: "ConversationSession",
    startup: Future[None],
    ready_message: str,
  ) -> None:
    try:
      synth = Synth(tts_model, session)
      with torch.no_grad(), tts_model.mimi.streaming(1):
        synth.set_voice(DEFAULT_VOICE)
        synth.speak("Warming up.")
        synth.end_turn()
        _logger.info("ready", endpoints=ready_message)
        # Readiness is published only after warmup succeeds; main() blocks on
        # this before starting any service that could enqueue work.
        startup.set_result(None)

        # Unconditional dequeue: every job admitted before the sentinel runs
        # and signals its waiter before the worker exits on None.
        while True:
          job = self._jobs.get()
          if job is None:
            return
          _process_job(synth, job)
    except Exception as exc:
      if not startup.done():
        # Warmup failed before readiness: hand the fault to main() and stop.
        startup.set_exception(exc)
        return
      # A fault after readiness is unexpected; stay fail-loud.
      raise


class Synth:
  """Owns the streaming TTS generator and its per-voice/per-turn lifecycle.

  The LMGen wiring — sampling hooks, delay-masked audio tokens, and the
  pump/flush loops — is adapted from kyutai's delayed-streams-modeling
  scripts/tts_pytorch_streaming.py (MIT license).
  """

  def __init__(self, tts_model: TTSModel, session: "ConversationSession") -> None:
    self.tts_model = tts_model
    self.session = session

    self.lm_gen: LMGen | None = None
    self.voice: str | None = None
    self.first_turn = True
    # The voice a new assistant turn opens with; changed out-of-band by
    # /voice and applied only when the next turn opens (see _speak_word).
    self.conversation_voice = DEFAULT_VOICE
    self._attrs_cache: dict[str, ConditionAttributes] = {}

    # Script state machine: text entries queued but not yet consumed by the LM.
    self.script_state = tts_model.machine.new_state([])
    self.offset = 0  # LM steps since the last reset; drives the delay masks
    self._streaming: ExitStack | None = None  # owns LMGen streaming teardown
    self.target: ServerConnection | None = None
    self.conversation_audio = False
    self.turn_open = False
    self.turn_failed = False
    self.turn_started_at: float | None = None

  def _attrs(self, voice: str) -> ConditionAttributes:
    if voice not in self._attrs_cache:
      self._attrs_cache[voice] = self.tts_model.make_condition_attributes(
        [self.tts_model.get_voice_path(voice)], cfg_coef=2.0
      )
    return self._attrs_cache[voice]

  def _condition_tensors(self, attrs: ConditionAttributes) -> dict:
    tts_model = self.tts_model
    attributes = [attrs]
    if tts_model.cfg_coef != 1.0:
      if tts_model.valid_cfg_conditionings:
        raise ValueError(
          "This model does not support direct CFG, but was trained "
          "with CFG distillation. Pass `cfg_coef` to "
          "`make_condition_attributes` instead."
        )
      # Direct CFG doubles the batch: real conditions + nulled conditions.
      attributes = attributes + dropout_all_conditions(attributes)

    assert tts_model.lm.condition_provider is not None
    prepared = tts_model.lm.condition_provider.prepare(attributes)
    return tts_model.lm.condition_provider(prepared)

  # ---- LMGen sampling hooks (see class docstring for provenance) ----

  def _on_text_logits_hook(self, text_logits: torch.Tensor) -> None:
    if self.tts_model.padding_bonus:
      text_logits[..., self.tts_model.machine.token_ids.pad] += self.tts_model.padding_bonus

  def _on_audio_hook(self, audio_tokens: torch.Tensor) -> None:
    # Zero the audio codebooks until each one's delay has elapsed.
    audio_offset = self.tts_model.lm.audio_offset
    delays = self.tts_model.lm.delays
    for q in range(audio_tokens.shape[1]):
      delay = delays[q + audio_offset]

      if self.offset < delay + self.tts_model.delay_steps:
        audio_tokens[:, q] = self.tts_model.machine.token_ids.zero

  def _on_text_hook(self, text_tokens: torch.Tensor) -> None:
    # The script state machine substitutes queued script tokens for the
    # model's sampled text tokens, consuming entries as it goes.
    out_tokens = [
      self.tts_model.machine.process(self.offset, self.script_state, token)[0]
      for token in text_tokens.tolist()
    ]
    text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

  def _on_frame(self, frame: torch.Tensor) -> None:
    if (frame != -1).all():
      pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()

      if self.target is None:
        return
      if self.conversation_audio and self.session.interrupted.is_set():
        return
      samples = np.clip(pcm[0, 0], -1, 1)
      wire = (samples * 32767.0).astype("<i2").tobytes()
      self.target.send(wire)

      if self.conversation_audio:
        self.session.playback.record_emission(len(samples))

  def _step(self) -> None:
    assert self.lm_gen is not None
    tts_model = self.tts_model
    missing = tts_model.lm.n_q - tts_model.lm.dep_q
    input_tokens = torch.full(
      (1, missing, 1),
      tts_model.machine.token_ids.zero,
      dtype=torch.long,
      device=tts_model.lm.device,
    )
    frame = self.lm_gen.step(input_tokens)
    self.offset += 1

    if frame is not None:
      self._on_frame(frame)

  def set_voice(self, voice: str) -> None:
    if voice == self.voice:
      return
    # Fetch the voice embedding BEFORE tearing down the old gen: a bad
    # voice name (HF 404) must leave the current voice fully usable.
    attrs = self._attrs(voice)

    if self._streaming is not None:
      self.lm_gen, self.voice = None, None
      self._streaming.close()  # exits every submodule streaming state
      self._streaming = None

    # ponytail: full LMGen rebuild per voice switch (re-captures CUDA
    # graphs, ~1s). Fine while playback buffer covers it; swap condition
    # tensors in-place if switches ever need to be instant.
    tts_model = self.tts_model
    tts_model.lm.dep_q = tts_model.n_q
    self.script_state = tts_model.machine.new_state([])
    self.offset = 0
    self.lm_gen = LMGen(
      tts_model.lm,
      temp=tts_model.temp,
      temp_text=tts_model.temp,
      cfg_coef=tts_model.cfg_coef,
      condition_tensors=self._condition_tensors(attrs),
      on_text_logits_hook=self._on_text_logits_hook,
      on_text_hook=self._on_text_hook,
      on_audio_hook=self._on_audio_hook,
      cfg_is_masked_until=None,
      cfg_is_no_text=True,
    )
    # streaming() enters immediately and hands back the ExitStack that
    # undoes it -- held on self so the next set_voice can close it.
    self._streaming = self.lm_gen.streaming(1)
    self.voice = voice
    self.first_turn = True

    _logger.info("voice set", voice=voice)

  def select_conversation_voice(self, voice: str) -> None:
    """Set the voice for the next assistant turn without disturbing this one.

    Validating the voice eagerly (`_attrs`) means a bad id or a failed
    Hugging Face lookup raises here and leaves the prior selection intact.
    Deliberately does not call `set_voice`/reset/close: an open turn keeps
    streaming on its current voice and the change lands when the next opens.
    """
    self._attrs(voice)
    self.conversation_voice = voice
    _logger.info("conversation voice set", voice=voice)

  def speak(self, text: str) -> None:
    assert self.lm_gen is not None, "set_voice must run before speak"
    entries = script_to_entries(
      self.tts_model.tokenizer,
      self.tts_model.machine.token_ids,
      self.tts_model.mimi.frame_rate,
      [text],
      multi_speaker=self.first_turn and self.tts_model.multi_speaker,
      padding_between=1,
    )

    for entry in entries:
      self.script_state.entries.append(entry)
      # Pump until only the machine's lookahead buffer remains queued.
      while len(self.script_state.entries) > self.tts_model.machine.second_stream_ahead:
        self._step()

    self.first_turn = False

  def end_turn(self) -> None:
    assert self.lm_gen is not None, "set_voice must run before end_turn"
    # Drain queued entries, then run out the model's delay tail so the
    # last words actually reach the speakers.
    while len(self.script_state.entries) > 0 or self.script_state.end_step is not None:
      self._step()

    additional = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
    for _ in range(additional):
      self._step()

    self._reset_sequence()

  def abort_turn(self) -> None:
    """Reset without draining: for barge-in and error recovery.

    `_on_frame` drops conversation audio once `interrupted` is set, so
    draining the queued entries and the delay tail would only burn GPU
    steps and delay the next turn's first word.
    """
    self._reset_sequence()

  def _reset_sequence(self) -> None:
    assert self.lm_gen is not None, "set_voice must run before a reset"
    # Streaming stays open across turns; the supported way to start a
    # fresh sequence is resetting the still-open streaming state.
    self.lm_gen.reset_streaming()
    self.tts_model.mimi.reset_streaming()
    self.offset = 0
    self.script_state = self.tts_model.machine.new_state([])
    self.first_turn = True


def _abandon_turn(synth: Synth, exc: Exception) -> None:
  """Drop a broken conversation destination until its EndOfTurn arrives."""
  elapsed = (
    time.perf_counter() - synth.turn_started_at if synth.turn_started_at is not None else 0.0
  )
  if isinstance(exc, ConnectionClosedOK):
    _logger.info("client disconnected; turn abandoned", elapsed=elapsed)
  else:
    _logger.exception("turn abandoned", exc_info=exc, elapsed=elapsed)
  try:
    synth.abort_turn()
  except Exception:
    _logger.exception("turn reset failed")
  synth.target = None
  synth.turn_open = False
  synth.turn_failed = True
  synth.turn_started_at = None


def _close_turn(synth: Synth) -> None:
  if not synth.turn_open:
    synth.turn_failed = False
    return

  interrupted = synth.session.interrupted.is_set()
  try:
    synth.abort_turn() if interrupted else synth.end_turn()
    assert synth.turn_started_at is not None
    elapsed = time.perf_counter() - synth.turn_started_at
    outcome = "interrupted" if interrupted else "complete"
    _logger.info("turn finished", outcome=outcome, elapsed=elapsed)
  except Exception as exc:
    # A failed drain leaves the machine mid-sequence; abort to a clean state.
    _abandon_turn(synth, exc)
  finally:
    synth.target = None
    synth.turn_open = False
    synth.turn_failed = False
    synth.turn_started_at = None


def _speak_word(synth: Synth, job: SpeakWord) -> None:
  # Words queued behind a barge-in or failed socket wait for EndOfTurn.
  if synth.session.interrupted.is_set() or synth.turn_failed:
    return
  try:
    if not synth.turn_open:
      synth.set_voice(synth.conversation_voice)
      synth.target = job.target
      synth.conversation_audio = True
      synth.turn_open = True
      synth.turn_started_at = time.perf_counter()
      _logger.info("turn started", voice=synth.conversation_voice)
    synth.speak(job.text)
    synth.session.playback.mark_spoken(job.text)
  except Exception as exc:
    _abandon_turn(synth, exc)


def _say(synth: Synth, job: SayJob) -> None:
  # The single generator must finish conversation audio before an isolated say.
  _close_turn(synth)
  synth.target = job.ws
  synth.conversation_audio = False
  voice = job.voice or DEFAULT_VOICE
  started_at = time.perf_counter()
  _logger.info("say started", voice=voice, chars=len(job.text))

  try:
    synth.set_voice(voice)
    synth.speak(job.text)
    synth.end_turn()
    _logger.info("say complete", elapsed=time.perf_counter() - started_at)
  except ConnectionClosedOK:
    _logger.info("say client disconnected")
    synth.abort_turn()
  except Exception:
    _logger.exception("say failed")
    synth.abort_turn()
  finally:
    synth.target = None
    job.done.set()


def _set_conversation_voice(synth: Synth, job: SetConversationVoice) -> None:
  try:
    synth.select_conversation_voice(job.voice)
    job.result.set_result(None)
  except Exception as exc:
    # A bad voice must not kill the worker; report it to the waiting endpoint.
    _logger.exception("conversation voice selection failed", exc_info=exc, voice=job.voice)
    job.result.set_exception(exc)


def _process_job(synth: Synth, job: SpeakWord | EndOfTurn | SayJob | SetConversationVoice) -> None:
  match job:
    case EndOfTurn(done):
      _close_turn(synth)
      done.set()
    case SpeakWord():
      _speak_word(synth, job)
    case SayJob():
      _say(synth, job)
    case SetConversationVoice():
      _set_conversation_voice(synth, job)
