#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "websockets==16.1",
# ]
# ///
"""Change the voice Sleeper uses for subsequent conversational replies.

Talks to the /voice WebSocket route exposed by Sleeper's single websockets
handler. The route validates the voice through the TTS worker before echoing
the request, so a clean exit here means the selection actually took effect.
The change is process-local on the server: a Sleeper restart reverts to its
built-in default voice.
"""

import argparse
import json

from websockets.sync.client import connect

DEFAULT_URL = "ws://127.0.0.1:17393/voice"


def set_voice(url: str, voice: str) -> None:
  """Send one SetVoice request and require the matching acknowledgement."""
  with connect(url, compression=None) as ws:
    ws.send(json.dumps({"voice": voice}))
    reply = ws.recv()
    # A binary frame, non-JSON text, or any mismatch means the server did not
    # confirm this exact selection; refuse to report success.
    if not isinstance(reply, str):
      raise ValueError("expected a text acknowledgement")
    if json.loads(reply) != {"voice": voice}:
      raise ValueError(f"unexpected acknowledgement: {reply}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Set the voice for Sleeper's conversational replies")
  parser.add_argument("voice", metavar="VOICE", help="repo-relative voice id")
  parser.add_argument(
    "--url", default=DEFAULT_URL, help=f"/voice endpoint (default: {DEFAULT_URL})"
  )
  args = parser.parse_args()

  try:
    set_voice(args.url, args.voice)
  except Exception as exc:
    # One concise stderr line instead of a traceback: refusal, error-close, or
    # a mismatched acknowledgement all surface as a non-zero exit.
    parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
  main()
