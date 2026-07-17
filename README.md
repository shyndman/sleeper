# Sleeper

Sleeper runs speech and language models on a server while a remote client owns
the physical microphone and speaker. The server itself does not open audio
devices.

## Architecture

`sleeper` listens on port `17393` and owns one resident instance of each model
(TTS, VAD, ASR, barge-in, and turn detection) plus the language-model agent.
Models are loaded once at startup and reused; they are not created per
connection or request. `sleeper-client` captures and plays live conversation
audio. `say` only plays a requested utterance.

The server accepts at most one `/conversation` connection at a time. `/say`
jobs share the resident synthesizer but do not enter or modify conversation
history, turn detection, interruption state, or transcript flow.

## Wire protocol

WebSocket compression is disabled. All PCM is mono signed 16-bit little-endian.
Frame sizes are fixed:

| Direction | Format | Frame |
| --- | --- | --- |
| `/conversation` client to server | 16 kHz PCM | 512 samples / 1024 bytes |
| `/conversation` server to client | 24 kHz PCM | 1920 samples / 3840 bytes |
| `/say` server to client | 24 kHz PCM | 1920 samples / 3840 bytes |

### `/conversation`

This is a long-lived, full-duplex connection. The client sends only 1024-byte
binary microphone frames. The server sends 3840-byte binary playback frames
and strict `TurnTranscript` JSON text messages:

```json
{"role":"user","text":"What time is it?","ended_by":"turn_detected"}
```

```json
{"role":"assistant","text":"It is noon.","ended_by":"completed"}
```

`role` is `user` or `assistant`. A user turn ends with `turn_detected`.
An assistant turn ends with `completed`, or `interrupted` when barge-in stops
playback. Assistant transcript text contains what was heard through completed
synthesis marks, not necessarily the full generated response. On an
`assistant`/`interrupted` transcript, the client immediately flushes queued
playback audio.

### `/say`

The client sends exactly one strict `Say` JSON text message:

```json
{"text":"Dinner is ready.","voice":null}
```

`text` is required. `voice` is optional and selects the server TTS voice; omit
it or use `null` for the default. The server replies only with 3840-byte binary
PCM frames, then closes the connection. `/say` never emits transcripts and is
isolated from conversation state.

Both JSON message types reject unknown fields and invalid enum values.

## Run

Start the model server:

```console
uv run sleeper
```

On the machine with the microphone and speaker, start a conversation:

```console
uv run sleeper-client --url ws://SERVER:17393/conversation
```

While that Kitty terminal is focused, Up/Down change only Sleeper playback gain
in 2 dB steps, starting at 0 dB and clamping from -12 through +24 dB.

On any machine with the playback speaker, speak one message:

```console
uv run say --url ws://SERVER:17393/say "Dinner is ready."
```

Select a voice for a one-shot request:

```console
uv run say --url ws://SERVER:17393/say --voice expresso/ex03-ex01_happy_001_channel1_334s.wav "Hello."
```

For a server on the same machine, replace `SERVER` with `127.0.0.1`; both
clients already default to their corresponding loopback URL when `--url` is
omitted.