"""Remote full-duplex voice chat over separate WebSocket routes."""

import asyncio
import queue
import re
import threading
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Literal


import numpy as np
import onnxruntime as ort
import torch

from moshi.conditioners import ConditionAttributes, dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel, script_to_entries
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from sleeper.messages import (
    SAY_ADAPTER,
    TURN_TRANSCRIPT_ADAPTER,
    TurnTranscript,
)
from sleeper.bargein_test import (
    META_PATH as BARGEIN_META,
    MODEL_PATH as BARGEIN_ONNX,
    SAMPLE_RATE as MIC_SR,
    VAD_FRAME_SAMPLES,
    WINDOW_SAMPLES,
    BargeInDetector,
    SpeechGate,
)
from omegaconf import open_dict
from websockets.sync.server import ServerConnection, serve
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

PORT = 17393
LLM_URL = "http://ollama-nvidia:11434/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# Everything the LLM emits goes straight to TTS, so the instructions steer it toward
# speakable prose: no markup for the synthesizer to read aloud, numbers written out
# the way they're pronounced, and no long-form structure that only works on a screen.
SPOKEN_INSTRUCTIONS = """\
You are a voice assistant. Everything you write is spoken aloud by a text-to-speech
engine, so write for the ear, not the page:
- Plain prose only, direct and to the point. No filler, no pleasantries, no markdown,
  lists, headings, code, emojis, or URLs.
- Write everything as it should be pronounced: "twenty three degrees", "three thirty
  PM", "kilometres per hour" -- never digits, symbols, or abbreviations.
- No parentheticals or asides; if something is secondary, drop it.
- Be brief. Lead with the answer. If a question genuinely needs a long answer, give
  the short version and offer to go deeper.
"""
# Flush a sentence to TTS as soon as it's complete; the tail stays buffered.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# ---- Voice input (mic -> VAD -> {ASR + smart-turn | barge-in}) ----
ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
ASR_ATT_CONTEXT = [70, 1]  # left/right context in 80ms frames: 160ms chunks
SMART_TURN_ONNX = Path(__file__).parent / "models" / "smart-turn-v3.2-cpu.onnx"
TURN_COMPLETE_THRESHOLD = 0.5  # smart-turn sigmoid; >= means the user is done
TURN_CHECK_EVERY_BLOCKS = 16  # re-run smart-turn every ~512ms of silence
FORCE_TURN_END_MS = 2500  # stop waiting on smart-turn after this much silence
PREROLL_SECONDS = 2.0  # mic history replayed into ASR at capture start

# A turn owns its destination connection. Say jobs share the resident synth but
# never enter the conversation state machine.
type TurnItem = tuple[ServerConnection, str] | None
type SentenceItem = (
    tuple[Literal["conversation", "say"], ServerConnection, str, str, threading.Event]
    | None
)
turns: queue.Queue[TurnItem] = queue.Queue()
sentences: queue.Queue[SentenceItem] = queue.Queue()
mute = {"on": True}
mic_q: queue.Queue[np.ndarray] = queue.Queue()
mode: dict[str, str] = {"v": "user"}
interrupted = threading.Event()
playback_changed = threading.Event()
stopping = threading.Event()
conversation_lock = threading.Lock()
conversation: dict[str, ServerConnection | None] = {"ws": None}
# Samples emitted since the current assistant turn began. Marks identify the
# sample immediately after each complete synthesized sentence.
playback: dict[str, float | int | None] = {"first": None, "samples": 0}
marks: list[tuple[int, str]] = []


class Synth:
    """Owns the streaming TTS generator and its per-voice/per-turn lifecycle.

    The LMGen wiring — sampling hooks, delay-masked audio tokens, and the
    pump/flush loops — is adapted from kyutai's delayed-streams-modeling
    scripts/tts_pytorch_streaming.py (MIT license).
    """

    def __init__(self, tts_model: TTSModel) -> None:
        self.tts_model = tts_model
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
            if mute["on"] or self.target is None:
                return
            if self.conversation_audio and interrupted.is_set():
                return
            samples = np.clip(pcm[0, 0], -1, 1)
            wire = (samples * 32767.0).astype("<i2").tobytes()
            self.target.send(wire)
            if self.conversation_audio:
                if playback["first"] is None:
                    playback["first"] = time.monotonic()
                playback["samples"] = int(playback["samples"] or 0) + len(samples)
                playback_changed.set()

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


def _send_transcript(ws: ServerConnection, transcript: TurnTranscript) -> None:
    ws.send(TURN_TRANSCRIPT_ADAPTER.dump_json(transcript).decode())


def _played_samples() -> int:
    first = playback["first"]
    emitted = int(playback["samples"] or 0)
    if first is None:
        return 0
    return min(emitted, int(max(0.0, time.monotonic() - float(first)) * 24_000))


def _heard_text() -> str:
    played = _played_samples()
    return " ".join(text for watermark, text in marks if watermark <= played)


def tts_worker(tts_model: TTSModel) -> None:
    synth = Synth(tts_model)
    with torch.no_grad(), tts_model.mimi.streaming(1):
        synth.set_voice(VOICE)
        synth.speak("Warming up.")
        synth.end_turn()
        mute["on"] = False
        print(f"[ready] ws://0.0.0.0:{PORT}/conversation and /say")
        while not stopping.is_set():
            job = sentences.get()
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
                        marks.append((int(playback["samples"] or 0), text))
                else:
                    synth.speak(text)
                synth.end_turn()
                if kind == "conversation" and marks:
                    marks[-1] = (int(playback["samples"] or 0), marks[-1][1])
            except Exception as exc:
                print(f"[tts error] {exc}")
            finally:
                synth.target = None
                done.set()


def _wait_synth_or_interrupt(done: threading.Event) -> None:
    while not done.wait(0.02) and not interrupted.is_set():
        pass


def _wait_for_playback(done_synth: threading.Event) -> bool:
    while not stopping.is_set():
        if interrupted.is_set():
            return False
        emitted = int(playback["samples"] or 0)
        if done_synth.is_set() and _played_samples() >= emitted:
            return True
        playback_changed.wait(0.05)
        playback_changed.clear()
    return False


async def turn_loop(agent: Agent) -> None:
    history: list[ModelMessage] = []
    while not stopping.is_set():
        item = await asyncio.to_thread(turns.get)
        if item is None:
            return
        ws, prompt = item
        mode["v"] = "assistant"
        interrupted.clear()
        playback.update(first=None, samples=0)
        marks.clear()
        last_synth_done = threading.Event()
        last_synth_done.set()
        buffer = ""
        try:
            async with agent.run_stream(prompt, message_history=history) as result:
                async for delta in result.stream_text(delta=True):
                    if interrupted.is_set():
                        break
                    buffer += delta
                    *complete, buffer = SENTENCE_END.split(buffer)
                    for sentence in complete:
                        sentence = sentence.strip()
                        if sentence:
                            last_synth_done = threading.Event()
                            sentences.put(
                                ("conversation", ws, VOICE, sentence, last_synth_done)
                            )
                            await asyncio.to_thread(
                                _wait_synth_or_interrupt, last_synth_done
                            )
                            if interrupted.is_set():
                                break
        except Exception as exc:
            print(f"[llm error] {exc}")
        if buffer.strip() and not interrupted.is_set():
            last_synth_done = threading.Event()
            sentences.put(("conversation", ws, VOICE, buffer.strip(), last_synth_done))
            await asyncio.to_thread(_wait_synth_or_interrupt, last_synth_done)
        completed = await asyncio.to_thread(_wait_for_playback, last_synth_done)
        spoken = _heard_text()
        ended_by = "completed" if completed else "interrupted"
        try:
            _send_transcript(ws, TurnTranscript("assistant", spoken, ended_by))
        except Exception:
            pass
        if interrupted.is_set():
            await asyncio.to_thread(last_synth_done.wait)
        history.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))
        history.append(
            ModelResponse(
                parts=[TextPart(content=spoken or "(cut off before speaking)")]
            )
        )
        if completed:
            mode["v"] = "user"
        interrupted.clear()


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
        model.eval()
        self.model = model.cuda()
        # Runtime latency knob; rebuilds encoder.streaming_cfg for the sizes below.
        self.model.encoder.set_default_att_context_size(ASR_ATT_CONTEXT)
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
        self._cache_ch, self._cache_t, self._cache_len = enc.get_initial_cache_state(
            batch_size=1
        )
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
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )

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
        return float(self.session.run(None, {"input_features": feats})[0].ravel()[0])


def listen_worker() -> None:
    """Consume 512-sample remote mic frames; own all voice-input models."""
    from silero_vad import load_silero_vad

    print("Loading voice input models (VAD / barge-in / smart-turn / ASR)...")
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
    preroll: deque[np.ndarray] = deque(
        maxlen=int(PREROLL_SECONDS * MIC_SR / VAD_FRAME_SAMPLES)
    )
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
            chunk = mic_q.get(timeout=0.25)
        except queue.Empty:
            continue
        preroll.append(chunk)
        window = np.concatenate([window[len(chunk) :], chunk])
        speaking = gate.update(chunk)
        if mode["v"] == "assistant":
            capturing = False
            if speaking and bargein.probability(window) >= bargein.threshold:
                print("[barge-in]")
                interrupted.set()
                playback_changed.set()
                mode["v"] = "user"
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
                turn_detector.complete_probability(np.concatenate(utterance))
                >= TURN_COMPLETE_THRESHOLD
            )
        else:
            ended = silence_blocks >= force_blocks
        if ended:
            capturing = False
            text = asr.text.strip()
            ws = conversation["ws"]
            if text and ws is not None:
                print(f"[user] {text}")
                try:
                    _send_transcript(ws, TurnTranscript("user", text, "turn_detected"))
                except Exception:
                    capturing = False
                    mode["v"] = "user"
                    continue
                mode["v"] = "assistant"
                turns.put((ws, text))


def conversation_handler(ws: ServerConnection) -> None:
    if not conversation_lock.acquire(blocking=False):
        ws.close(1013, "conversation already active")
        return
    conversation["ws"] = ws
    mode["v"] = "user"
    try:
        for message in ws:
            if not isinstance(message, bytes) or len(message) != 1024:
                ws.close(1003, "expected 1024-byte PCM frames")
                return
            mic_q.put(np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0)
    finally:
        interrupted.set()
        playback_changed.set()
        conversation["ws"] = None
        conversation_lock.release()


def say_handler(ws: ServerConnection) -> None:
    try:
        raw = ws.recv()
        if not isinstance(raw, str):
            ws.close(1003, "expected Say JSON")
            return
        request = SAY_ADAPTER.validate_json(raw)
        done = threading.Event()
        sentences.put(("say", ws, request.voice or VOICE, request.text, done))
        done.wait()
    except Exception as exc:
        ws.close(1007, str(exc)[:120])


def handler(ws: ServerConnection) -> None:
    path = ws.request.path
    if path == "/conversation":
        conversation_handler(ws)
    elif path == "/say":
        say_handler(ws)
    else:
        ws.close(1008, "unknown route")


def main() -> None:
    print("Loading Kyutai TTS...")
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device="cuda")
    agent = Agent(
        OpenAIChatModel(LLM_MODEL, provider=OllamaProvider(base_url=LLM_URL)),
        instructions=SPOKEN_INSTRUCTIONS,
    )
    threading.Thread(target=tts_worker, args=(tts_model,), daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(turn_loop(agent)), daemon=True).start()
    listener = threading.Thread(target=listen_worker, daemon=True)
    listener.start()
    try:
        with serve(handler, "0.0.0.0", PORT, compression=None) as server:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown] closing server...")
    finally:
        stopping.set()
        interrupted.set()
        playback_changed.set()
        mic_q.put(np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32))
        turns.put(None)
        sentences.put(None)
        listener.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
