# Third-party assets and code

| What | Origin | License |
|---|---|---|
| `src/sleeper/models/smart-turn-v3.2-cpu.onnx` | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | BSD-2-Clause |
| `src/sleeper/models/bargein.onnx` + `.meta.npz` | [bnovikov/bargein-classifier](https://huggingface.co/bnovikov/bargein-classifier) | CC BY 4.0 |
| `tests/data/bria.mp3` | [kyutai-labs/delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) (`audio/bria.mp3`) | MIT |
| `Synth` LMGen wiring in `src/sleeper/sleeper.py` | Adapted from the same repo's `scripts/tts_pytorch_streaming.py` | MIT |

Model weights fetched at runtime (Kyutai TTS, Nemotron ASR, Silero VAD) are
downloaded from their upstream repos and not redistributed here.
