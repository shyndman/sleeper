"""Full-duplex voice chat: mic -> VAD -> ASR/turn-taking -> LLM -> TTS -> speakers.

Output side (also reachable over HTTP, see ask.py / audition.py):

  POST /ask {"prompt": "..."}            speak the LLM's streamed answer
  POST /say {"text": "...", "voice"?}    speak text directly, optional voice
                                         (kyutai/tts-voices name, see GET /voices)

  mic or HTTP -> turns queue -> turn thread (LLM streaming, full chat history)
       -> sentences queue -> TTS thread (persistent TTSGen)
       -> bounded pcm queue -> sounddevice callback -> speakers

Input side (listen_worker): Silero VAD fronts everything. While the user holds
the floor their speech streams through Nemotron cache-aware ASR and smart-turn
v3 decides end-of-turn (the transcript becomes the next LLM prompt). While the
assistant holds the floor the barge-in classifier separates real interruptions
from backchannels; a barge-in cancels the LLM stream, flushes all queued audio,
and hands the floor back to the user. Chat history keeps only the sentences
that actually reached the speakers.

Assumes echo-cancelled input (EasyEffects AEC) so the mic never hears the TTS.
"""
import asyncio
import json
import os
import queue
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Capture through the EasyEffects echo-cancelled source. Without AEC the mic
# hears the TTS: the barge-in model interrupts on the bot's own voice and the
# ASR transcribes it back into the conversation (self-talk loop). Must be set
# before PortAudio connects to PulseAudio/PipeWire, i.e. before any stream opens.
os.environ.setdefault("PULSE_SOURCE", "easyeffects_source")

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import torch

from moshi.conditioners import ConditionAttributes
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from tts_pytorch_streaming import TTSGen, prepare_script
from bargein_test import (
    META_PATH as BARGEIN_META,
    MODEL_PATH as BARGEIN_ONNX,
    SAMPLE_RATE as MIC_SR,
    VAD_FRAME_SAMPLES,
    WINDOW_SAMPLES,
    BargeInDetector,
    SpeechGate,
)
from omegaconf import open_dict
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

PORT = 17393
LLM_URL = "http://localhost:19922/v1"
LLM_MODEL = "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
# Flush a sentence to TTS as soon as it's complete; the tail stays buffered.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
FRAME_SAMPLES = 1920  # one mimi frame: 80ms @ 24kHz, matches output blocksize
PCM_BUFFER_FRAMES = 64  # ~5s of audio; synthesis blocks when this far ahead

# ---- Voice input (mic -> VAD -> {ASR + smart-turn | barge-in}) ----
ASR_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
ASR_ATT_CONTEXT = [70, 1]  # left/right context in 80ms frames: 160ms chunks
SMART_TURN_ONNX = Path(__file__).parent / "models" / "smart-turn-v3.2-cpu.onnx"
TURN_COMPLETE_THRESHOLD = 0.5  # smart-turn sigmoid; >= means the user is done
TURN_CHECK_EVERY_BLOCKS = 16  # re-run smart-turn every ~512ms of silence
FORCE_TURN_END_MS = 2500  # stop waiting on smart-turn after this much silence
PREROLL_SECONDS = 2.0  # mic history replayed into ASR at capture start
MIC_DEVICE: str | int | None = None  # default + PULSE_SOURCE = easyeffects_source

# Turns: ("ask", prompt) | ("say", voice, text, done_event). Single consumer
# (turn thread) so a /say can never interleave into a streaming /ask turn.
turns: queue.Queue[tuple] = queue.Queue()
# Sentence stream: ("voice", name) | ("text", sentence) | ("end", done_event?)
sentences: queue.Queue[tuple[str, str | threading.Event | None]] = queue.Queue()
pcms: queue.Queue[np.ndarray] = queue.Queue(maxsize=PCM_BUFFER_FRAMES)
mute = {"on": True}  # True during warmup so the JIT turn stays silent

# ---- Voice input state ----
mic_q: queue.Queue[np.ndarray] = queue.Queue()
# Who holds the floor. Flipped to "assistant" whenever a turn starts, back to
# "user" on barge-in (listener) or when playback drains (TTS thread).
mode: dict[str, str] = {"v": "user"}
# Set on barge-in: stops LLM streaming, makes the synth drop frames, and makes
# the audio callback flush queued pcm. Cleared by the TTS thread at end-of-turn.
interrupted = threading.Event()
# Monotonic pcm-frame counters (1 frame = 80ms). "enqueued" bumped by the synth,
# "played" by the audio callback; marks[i] = (enqueued watermark when sentence i
# finished synthesizing, its text) -> reconstructs what the user actually heard.
counters: dict[str, int] = {"enqueued": 0, "played": 0}
marks: list[tuple[int, str]] = []
# Ctrl+C: main sets this so the listener exits its loop and closes the mic
# stream before the process tears down PortAudio underneath it.
stopping = threading.Event()


class Synth:
    """Owns the TTSGen and its per-voice/per-turn lifecycle."""

    def __init__(self, tts_model: TTSModel) -> None:
        self.tts_model = tts_model
        self.gen: TTSGen | None = None
        self.voice: str | None = None
        self.first_turn = True
        self._attrs_cache: dict[str, ConditionAttributes] = {}

    def _attrs(self, voice: str) -> ConditionAttributes:
        if voice not in self._attrs_cache:
            self._attrs_cache[voice] = self.tts_model.make_condition_attributes(
                [self.tts_model.get_voice_path(voice)], cfg_coef=2.0
            )
        return self._attrs_cache[voice]

    def _on_frame(self, frame: torch.Tensor) -> None:
        if (frame != -1).all():
            pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            if not mute["on"] and not interrupted.is_set():
                pcms.put(np.clip(pcm[0, 0], -1, 1))  # blocks = backpressure
                counters["enqueued"] += 1

    def set_voice(self, voice: str) -> None:
        if voice == self.voice:
            return
        # Fetch the voice embedding BEFORE tearing down the old gen: a bad
        # voice name (HF 404) must leave the current voice fully usable.
        attrs = self._attrs(voice)
        if self.gen is not None:
            # TTSGen.__post_init__ uses streaming_forever, which discards the
            # ExitStack that would undo streaming. Replicate its close by hand:
            # clear LMGen-tree state, then state.__exit__ closes the state's
            # own exit_stack, which exits lm_model.streaming(...).
            gen, self.gen, self.voice = self.gen, None, None
            state = gen.lm_gen._streaming_state
            assert state is not None
            gen.lm_gen._stop_streaming()
            state.__exit__(None, None, None)
        # ponytail: full TTSGen rebuild per voice switch (re-captures CUDA
        # graphs, ~1s). Fine while playback buffer covers it; swap condition
        # tensors in-place if switches ever need to be instant.
        self.gen = TTSGen(self.tts_model, [attrs], on_frame=self._on_frame)
        self.voice = voice
        self.first_turn = True
        print(f"[voice] {voice}")

    def speak(self, text: str) -> None:
        assert self.gen is not None, "set_voice must run before speak"
        t = time.perf_counter()
        for entry in prepare_script(self.tts_model, text, first_turn=self.first_turn):
            self.gen.append_entry(entry)
            self.gen.process()
        self.first_turn = False
        print(f"[tts] {time.perf_counter() - t:.2f}s  {text!r}")

    def end_turn(self) -> None:
        assert self.gen is not None, "set_voice must run before end_turn"
        self.gen.process_last()
        # streaming_forever can't be stopped/restarted; the supported way to
        # start a fresh sequence is resetting the still-open streaming state.
        self.gen.lm_gen.reset_streaming()
        self.tts_model.mimi.reset_streaming()
        self.gen.offset = 0
        self.gen.state = self.tts_model.machine.new_state([])
        self.first_turn = True


def tts_worker(tts_model: TTSModel) -> None:
    synth = Synth(tts_model)
    with torch.no_grad(), tts_model.mimi.streaming(1):
        # Muted warmup: pays triton JIT + CUDA warmth (~10s) at startup.
        synth.set_voice(VOICE)
        synth.speak("Warming up.")
        synth.end_turn()
        mute["on"] = False
        print(f"[ready] listening on http://127.0.0.1:{PORT}/ask and /say")

        while True:
            kind, value = sentences.get()
            try:
                if kind == "voice":
                    assert isinstance(value, str)
                    synth.set_voice(value)
                elif kind == "end":
                    synth.end_turn()
                    if marks:
                        # process_last emits the tail of the final sentence.
                        marks[-1] = (counters["enqueued"], marks[-1][1])
                else:
                    assert isinstance(value, str)
                    if not interrupted.is_set():  # barge-in: drop queued sentences
                        synth.speak(value)
                        marks.append((counters["enqueued"], value))
            except Exception as exc:  # bad voice name etc.: skip, stay alive
                print(f"[error] {kind}={value!r}: {exc}")
            finally:
                if kind == "end":
                    # Wait for playback to drain (audio_callback task_done's each
                    # frame; on barge-in it flushes), so end-of-turn == silence.
                    pcms.join()
                    was_interrupted = interrupted.is_set()
                    interrupted.clear()
                    if not was_interrupted:
                        mode["v"] = "user"  # floor back to the user
                    if isinstance(value, threading.Event):
                        value.set()


async def turn_loop() -> None:
    """Serialize turns: stream the LLM into sentences, then reconcile history
    with what actually reached the speakers (barge-in truncates at the last
    fully-played sentence, so the model knows exactly what the user heard)."""
    agent = Agent(OpenAIChatModel(LLM_MODEL, provider=OllamaProvider(base_url=LLM_URL)))
    # Breaking out of run_stream on barge-in abandons the httpx response
    # mid-read; its teardown lands in an unawaited task as a ReadError
    # ("Task exception was never retrieved"). Expected — drop just those.
    def quiet_aborted_stream(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        import httpcore
        import httpx

        if isinstance(context.get("exception"), (httpx.ReadError, httpcore.ReadError)):
            return
        loop.default_exception_handler(context)

    asyncio.get_running_loop().set_exception_handler(quiet_aborted_stream)
    history: list[ModelMessage] = []
    while True:
        turn = await asyncio.to_thread(turns.get)
        mode["v"] = "assistant"
        if turn[0] == "say":
            _, voice, text, done = turn
            sentences.put(("voice", voice))
            sentences.put(("text", text))
            sentences.put(("end", done))
            continue
        _, prompt = turn
        marks.clear()
        t0 = time.perf_counter()
        sentences.put(("voice", VOICE))
        buffer = ""
        try:
            async with agent.run_stream(prompt, message_history=history) as result:
                async for delta in result.stream_text(delta=True):
                    if interrupted.is_set():
                        break  # barge-in: stop generating, drop the tail
                    buffer += delta
                    *complete, buffer = SENTENCE_END.split(buffer)
                    for sentence in complete:
                        if sentence.strip():
                            sentences.put(("text", sentence.strip()))
        except Exception as exc:  # LLM down etc.: keep the loop alive
            print(f"[llm error] {exc}")
        if buffer.strip() and not interrupted.is_set():
            sentences.put(("text", buffer.strip()))
        done = threading.Event()
        sentences.put(("end", done))
        await asyncio.to_thread(done.wait)  # audio finished playing or was flushed
        spoken = " ".join(t for watermark, t in marks if counters["played"] >= watermark)
        history.append(ModelRequest(parts=[UserPromptPart(content=prompt)]))
        history.append(
            ModelResponse(parts=[TextPart(content=spoken or "(cut off before speaking)")])
        )
        print(f"[turn done] {time.perf_counter() - t0:.2f}s for prompt {prompt!r}")


def audio_callback(outdata: np.ndarray, *_args) -> None:
    if interrupted.is_set():
        # Barge-in: dump whatever synthesis already queued; play silence.
        outdata[:] = 0
        try:
            while True:
                pcms.get(block=False)
                pcms.task_done()
        except queue.Empty:
            pass
        return
    try:
        outdata[:, 0] = pcms.get(block=False)
        pcms.task_done()
        counters["played"] += 1
    except queue.Empty:
        outdata[:] = 0


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
        return float(self.session.run(None, {"input_features": feats})[0].ravel()[0])


def listen_worker() -> None:
    """Full-duplex mic loop; owns every voice-input model.

    User turn:      VAD-gated capture -> streaming ASR; on silence, smart-turn
                    decides end-of-turn and the transcript is queued as the
                    next LLM prompt. A 2s preroll is replayed into the ASR at
                    capture start so VAD attack never clips the first word.
    Assistant turn: VAD-gated barge-in classifier over a rolling 2s window;
                    a barge-in flushes playback and seeds the next user
                    capture with the interruption itself.
    """
    from silero_vad import load_silero_vad

    print("Loading voice input models (VAD / barge-in / smart-turn / ASR)...")
    vad = load_silero_vad()
    vad.reset_states()
    gate = SpeechGate(vad)
    bargein = BargeInDetector(BARGEIN_ONNX, BARGEIN_META)
    bargein.probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32))  # ORT warmup
    turn_detector = TurnDetector(SMART_TURN_ONNX)
    turn_detector.complete_probability(np.zeros(MIC_SR, dtype=np.float32))
    asr = StreamingASR()
    asr.feed(np.zeros(MIC_SR, dtype=np.float32))  # pay cudnn/autotune cost now
    asr.reset()

    preroll: deque[np.ndarray] = deque(
        maxlen=int(PREROLL_SECONDS * MIC_SR / VAD_FRAME_SAMPLES)
    )
    # Utterance audio for smart-turn, capped at its 8s analysis window.
    utterance: deque[np.ndarray] = deque(maxlen=int(8 * MIC_SR / VAD_FRAME_SAMPLES))
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)  # rolling 2s for barge-in
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

    def mic_callback(indata: np.ndarray, *_args) -> None:
        mic_q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=MIC_SR,
        blocksize=VAD_FRAME_SAMPLES,  # 512 samples = 32ms, Silero's frame size
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
        callback=mic_callback,
    ):
        print("[ready] voice loop live")
        while not stopping.is_set():
            try:
                chunk = mic_q.get(timeout=0.25)
            except queue.Empty:
                continue
            preroll.append(chunk)
            window = np.concatenate([window[len(chunk):], chunk])
            speaking = gate.update(chunk)

            if mode["v"] == "assistant":
                capturing = False
                if speaking and bargein.probability(window) >= bargein.threshold:
                    print("[barge-in]")
                    interrupted.set()
                    mode["v"] = "user"
                    start_capture()  # the interruption is the next utterance
                continue

            if not capturing:
                if speaking:
                    start_capture()
                continue
            utterance.append(chunk)
            asr.feed(chunk)
            if speaking:  # gate holds 250ms past the last speech frame
                silence_blocks = 0
                continue
            silence_blocks += 1
            if silence_blocks % TURN_CHECK_EVERY_BLOCKS == 1:
                prob = turn_detector.complete_probability(np.concatenate(utterance))
                ended = prob >= TURN_COMPLETE_THRESHOLD
            else:
                # ponytail: fixed silence cap as the smart-turn fallback;
                # adaptive endpointing only if 2.5s ever feels wrong.
                ended = silence_blocks >= force_blocks
            if ended:
                capturing = False
                text = asr.text.strip()
                if text:
                    print(f"[user] {text}")
                    mode["v"] = "assistant"
                    turns.put(("ask", text))


class Handler(BaseHTTPRequestHandler):
    tts_model: TTSModel | None = None
    _voices: list[str] | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        done: threading.Event | None = None
        try:
            body = json.loads(self.rfile.read(length))
            if self.path == "/ask":
                turns.put(("ask", str(body["prompt"])))
            elif self.path == "/say":
                done = threading.Event()
                turns.put(("say", str(body.get("voice", VOICE)), str(body["text"]), done))
            else:
                self.send_error(404)
                return
        except (json.JSONDecodeError, KeyError):
            self.send_error(400, "expected JSON body with 'prompt' or 'text'")
            return
        if done is not None:
            done.wait()  # /say responds only once the audio finished playing
            self.send_response(200)
        else:
            self.send_response(202)
        self.end_headers()

    def do_GET(self) -> None:
        """GET /voices: correctly-suffixed voice names from the voice repo."""
        if self.path != "/voices":
            self.send_error(404)
            return
        if Handler._voices is None:
            from huggingface_hub import list_repo_files

            tts_model = Handler.tts_model
            assert tts_model is not None
            Handler._voices = sorted(
                f.removesuffix(tts_model.voice_suffix)
                for f in list_repo_files(tts_model.voice_repo)
                if f.endswith(tts_model.voice_suffix)
            )
        body = json.dumps(Handler._voices).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass  # keep stdout for pipeline timing lines


def main() -> None:
    print("Loading Kyutai TTS...")
    ckpt = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    tts_model = TTSModel.from_checkpoint_info(ckpt, n_q=32, temp=0.6, device="cuda")
    Handler.tts_model = tts_model

    threading.Thread(target=tts_worker, args=(tts_model,), daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(turn_loop()), daemon=True).start()
    listener = threading.Thread(target=listen_worker, daemon=True)
    listener.start()

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    with sd.OutputStream(
        samplerate=tts_model.mimi.sample_rate,
        blocksize=FRAME_SAMPLES,
        channels=1,
        callback=audio_callback,
    ):
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[shutdown] closing mic, audio, and server...")
        finally:
            server.server_close()
            interrupted.set()  # unstick any tts pcms.put backpressure block
            stopping.set()
            listener.join(timeout=2)  # let the mic InputStream close cleanly
    #! HACK: never let interpreter teardown run (see __main__). Both audio
    #! streams and the HTTP socket are already closed above.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C during model load, or a second Ctrl+C that lands inside
        # main's shutdown path and unwinds past its os._exit.
        pass
    finally:
        #! HACK: exit without interpreter teardown. turn_loop's
        #! `await asyncio.to_thread(turns.get)` parks a non-daemon
        #! concurrent.futures worker on the turns queue forever, and Python's
        #! threading._shutdown joins those workers at exit, so a normal exit
        #! hangs the process after Ctrl+C (verified with py-spy).
        os._exit(0)
