"""Streaming Kyutai TTS synthesis and the single FIFO speech worker."""

import queue
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import torch
from moshi.conditioners import ConditionAttributes, dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.tts import TTSModel, script_to_entries
from websockets.sync.server import ServerConnection
from websockets.exceptions import ConnectionClosedOK

from sleeper.playback import PlaybackTracker

# One resident synth serves two producers. A conversation turn streams in as
# individual SpeakWord jobs -- words are fed to the generator as the LLM emits
# them -- and is closed by a single EndOfTurn. A say job is a self-contained
# utterance outside the conversation state machine; its audio is not tracked
# for playback/interruption.


@dataclass(slots=True, frozen=True)
class SpeakWord:
    """One whitespace-delimited word of the open assistant turn."""

    ws: ServerConnection
    voice: str
    text: str


@dataclass(slots=True, frozen=True)
class EndOfTurn:
    """Close the open assistant turn; `done` fires after flush or abort."""

    done: threading.Event


@dataclass(slots=True, frozen=True)
class SayJob:
    """Self-contained /say utterance: spoken and flushed as one unit."""

    ws: ServerConnection
    voice: str
    text: str
    done: threading.Event


type SpeechQueueItem = SpeakWord | EndOfTurn | SayJob | None


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
        print(ready_message, flush=True)
        # A conversation turn stays open across SpeakWord jobs so the generator
        # synthesizes one continuous stream; EndOfTurn (or an interleaved say
        # job) closes it.
        turn_open = False
        turn_failed = False
        turn_started_at: float | None = None

        def abandon_turn(exc: Exception) -> None:
            """Drop a broken destination once; queued words wait for EndOfTurn."""
            nonlocal turn_open, turn_failed, turn_started_at
            elapsed = (
                time.perf_counter() - turn_started_at
                if turn_started_at is not None
                else 0.0
            )
            if isinstance(exc, ConnectionClosedOK):
                print(
                    f"[tts] client disconnected; turn abandoned {elapsed:.2f}s",
                    flush=True,
                )
            else:
                print(
                    f"[tts error] {elapsed:.2f}s {type(exc).__name__}: {exc}",
                    flush=True,
                )
            try:
                synth.abort_turn()
            except Exception as reset_exc:
                print(
                    f"[tts reset error] {type(reset_exc).__name__}: {reset_exc}",
                    flush=True,
                )
            synth.target = None
            turn_open = False
            turn_failed = True
            turn_started_at = None

        def close_turn() -> None:
            nonlocal turn_open, turn_failed, turn_started_at
            if not turn_open:
                turn_failed = False
                return
            interrupted_turn = interrupted.is_set()
            try:
                if interrupted_turn:
                    synth.abort_turn()
                else:
                    synth.end_turn()
                assert turn_started_at is not None
                elapsed = time.perf_counter() - turn_started_at
                outcome = "interrupted" if interrupted_turn else "complete"
                print(f"[tts] turn {outcome} {elapsed:.2f}s", flush=True)
            except Exception as exc:
                # A failed drain (e.g. the client socket died mid-flush) leaves
                # the machine mid-sequence; abort back to a clean state.
                abandon_turn(exc)
            finally:
                synth.target = None
                turn_open = False
                turn_failed = False
                turn_started_at = None

        while not stopping.is_set():
            job = jobs.get()
            if job is None:
                return
            if isinstance(job, EndOfTurn):
                close_turn()
                job.done.set()
            elif isinstance(job, SpeakWord):
                # Words queued behind a barge-in or failed socket are skipped;
                # EndOfTurn resets the generator and opens the next turn cleanly.
                if interrupted.is_set() or turn_failed:
                    continue
                try:
                    if not turn_open:
                        synth.set_voice(job.voice)
                        synth.target = job.ws
                        synth.conversation_audio = True
                        turn_open = True
                        turn_started_at = time.perf_counter()
                        print(f"[tts] turn started voice={job.voice!r}", flush=True)
                    synth.speak(job.text)
                    playback.mark_spoken(job.text)
                except Exception as exc:
                    abandon_turn(exc)
            else:
                # A say job may land mid-conversation-turn; the single
                # generator must finish that turn's audio first.
                close_turn()
                synth.target = job.ws
                synth.conversation_audio = False
                started_at = time.perf_counter()
                print(
                    f"[tts] say started voice={job.voice!r} chars={len(job.text)}",
                    flush=True,
                )
                try:
                    synth.set_voice(job.voice)
                    synth.speak(job.text)
                    synth.end_turn()
                    print(
                        f"[tts] say complete "
                        f"{time.perf_counter() - started_at:.2f}s",
                        flush=True,
                    )
                except ConnectionClosedOK:
                    print("[tts] say client disconnected", flush=True)
                    synth.abort_turn()
                except Exception as exc:
                    print(
                        f"[tts error] say {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    synth.abort_turn()
                finally:
                    synth.target = None
                    job.done.set()
