"""Remote microphone and speaker client for Sleeper conversations."""

import argparse
import asyncio
import queue

import sounddevice as sd
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .messages import TURN_TRANSCRIPT_ADAPTER

INPUT_SAMPLES = 512
INPUT_BYTES = INPUT_SAMPLES * 2
OUTPUT_SAMPLES = 1920
OUTPUT_BYTES = OUTPUT_SAMPLES * 2
SILENCE = bytes(OUTPUT_BYTES)
DEFAULT_URL = "ws://127.0.0.1:17393/conversation"


async def _send_mic(
    websocket: ClientConnection, microphone: queue.Queue[bytes]
) -> None:
    while True:
        try:
            frame = microphone.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue
        await websocket.send(frame)


def _flush(playback: queue.SimpleQueue[bytes] | queue.Queue[bytes]) -> None:
    while True:
        try:
            playback.get_nowait()
        except queue.Empty:
            return


async def _receive(
    websocket: ClientConnection, playback: queue.SimpleQueue[bytes]
) -> None:
    async for message in websocket:
        if isinstance(message, bytes):
            if len(message) != OUTPUT_BYTES:
                raise ValueError(
                    f"expected {OUTPUT_BYTES} PCM bytes, got {len(message)}"
                )
            playback.put(message)
            continue

        transcript = TURN_TRANSCRIPT_ADAPTER.validate_json(message)
        print(
            f"{transcript.role}: {transcript.text} [{transcript.ended_by}]", flush=True
        )
        if transcript.role == "assistant" and transcript.ended_by == "interrupted":
            _flush(playback)


async def _run(url: str) -> None:
    microphone: queue.Queue[bytes] = queue.Queue(maxsize=32)
    playback: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def capture(
        indata: memoryview,
        frames: int,
        _time: object,
        _status: sd.CallbackFlags,
    ) -> None:
        if frames != INPUT_SAMPLES:
            return
        try:
            microphone.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def play(
        outdata: memoryview,
        frames: int,
        _time: object,
        _status: sd.CallbackFlags,
    ) -> None:
        if frames != OUTPUT_SAMPLES:
            outdata[:] = bytes(len(outdata))
            return
        try:
            outdata[:] = playback.get_nowait()
        except queue.Empty:
            outdata[:] = SILENCE

    with (
        sd.RawInputStream(
            samplerate=16_000,
            blocksize=INPUT_SAMPLES,
            channels=1,
            dtype="int16",
            callback=capture,
        ),
        sd.RawOutputStream(
            samplerate=24_000,
            blocksize=OUTPUT_SAMPLES,
            channels=1,
            dtype="int16",
            callback=play,
        ),
    ):
        while True:
            try:
                async with connect(url, compression=None) as websocket:
                    sender = asyncio.create_task(_send_mic(websocket, microphone))
                    receiver = asyncio.create_task(_receive(websocket, playback))
                    done, pending = await asyncio.wait(
                        (sender, receiver), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        task.result()
            except (OSError, ConnectionClosed, TimeoutError) as error:
                print(f"connection lost ({error}); retrying", flush=True)
            _flush(microphone)
            _flush(playback)
            await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a remote Sleeper conversation")
    parser.add_argument("--url", default=DEFAULT_URL, help="conversation WebSocket URL")
    args: argparse.Namespace = parser.parse_args()
    asyncio.run(_run(args.url))


if __name__ == "__main__":
    main()
