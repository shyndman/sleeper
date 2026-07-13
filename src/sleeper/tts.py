"""Streaming Kyutai TTS synthesis and the single FIFO speech worker."""

import queue
import threading
import time
from contextlib import ExitStack
from typing import Literal

import numpy as np
import torch
from moshi.conditioners import ConditionAttributes, dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.tts import TTSModel, script_to_entries
from websockets.sync.server import ServerConnection

from sleeper.playback import PlaybackTracker

# A say job shares the resident synth but never enters the conversation state
# machine; a conversation job's audio is tracked for playback/interruption.
type SpeechKind = Literal["conversation", "say"]
type SpeechJob = tuple[SpeechKind, ServerConnection, str, str, threading.Event]
type SpeechQueueItem = SpeechJob | None


class Synth:
    """Owns the streaming TTS generator and its per-voice/per-turn lifecycle.

    The LMGen wiring — sampling hooks, delay-masked audio tokens, and the
    pump/flush loops — is adapted from kyutai's delayed-streams-modeling
    scripts/tts_pytorch_streaming.py (MIT license).
    """

    def __init__(
        self,
        tts_model: TTSModel,
        playback: PlaybackTracker,
        interrupted: threading.Event,
    ) -> None:
        self.tts_model = tts_model
        self.playback = playback
        self.interrupted = interrupted
        self.lm_gen: LMGen | None = None
        self.voice: str | None = None
        self.first_turn = True
        self._attrs_cache: dict[str, ConditionAttributes] = {}
        # Script state machine: text entries queued but not yet consumed by the LM.
        self.script_state = tts_model.machine.new_state([])
        self.offset = 0  # LM steps since the last reset; drives the delay masks
        self._streaming: ExitStack | None = None  # owns LMGen streaming teardown
        self.target: ServerConnection | None = None
        self.conversation_audio = False

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

    def _on_text_logits_hook(self, text_logits: torch.Tensor) -> torch.Tensor:
        if self.tts_model.padding_bonus:
            text_logits[
                ..., self.tts_model.machine.token_ids.pad
            ] += self.tts_model.padding_bonus
        return text_logits

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
        text_tokens[:] = torch.tensor(
            out_tokens, dtype=torch.long, device=text_tokens.device
        )

    def _on_frame(self, frame: torch.Tensor) -> None:
        if (frame != -1).all():
            pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            if self.target is None:
                return
            if self.conversation_audio and self.interrupted.is_set():
                return
            samples = np.clip(pcm[0, 0], -1, 1)
            wire = (samples * 32767.0).astype("<i2").tobytes()
            self.target.send(wire)
            if self.conversation_audio:
                self.playback.record_emission(len(samples))

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
        print(f"[voice] {voice}")

    def speak(self, text: str) -> None:
        assert self.lm_gen is not None, "set_voice must run before speak"
        t = time.perf_counter()
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
            while (
                len(self.script_state.entries)
                > self.tts_model.machine.second_stream_ahead
            ):
                self._step()
        self.first_turn = False
        print(f"[tts] {time.perf_counter() - t:.2f}s  {text!r}")

    def end_turn(self) -> None:
        assert self.lm_gen is not None, "set_voice must run before end_turn"
        # Drain queued entries, then run out the model's delay tail so the
        # last words actually reach the speakers.
        while (
            len(self.script_state.entries) > 0 or self.script_state.end_step is not None
        ):
            self._step()
        additional = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        for _ in range(additional):
            self._step()
        # Streaming stays open across turns; the supported way to start a
        # fresh sequence is resetting the still-open streaming state.
        self.lm_gen.reset_streaming()
        self.tts_model.mimi.reset_streaming()
        self.offset = 0
        self.script_state = self.tts_model.machine.new_state([])
        self.first_turn = True


def tts_worker(
    tts_model: TTSModel,
    jobs: queue.Queue[SpeechQueueItem],
    playback: PlaybackTracker,
    interrupted: threading.Event,
    stopping: threading.Event,
    default_voice: str,
    ready_message: str,
) -> None:
    synth = Synth(tts_model, playback, interrupted)
    with torch.no_grad(), tts_model.mimi.streaming(1):
        synth.set_voice(default_voice)
        synth.speak("Warming up.")
        synth.end_turn()
        print(ready_message)
        while not stopping.is_set():
            job = jobs.get()
            if job is None:
                return
            kind, ws, voice, text, done = job
            synth.target = ws
            synth.conversation_audio = kind == "conversation"
            try:
                synth.set_voice(voice)
                if kind == "conversation":
                    if not interrupted.is_set():
                        synth.speak(text)
                        playback.begin_sentence(text)
                else:
                    synth.speak(text)
                synth.end_turn()
                if kind == "conversation":
                    playback.finish_sentence()
            except Exception as exc:
                print(f"[tts error] {exc}")
            finally:
                synth.target = None
                done.set()
