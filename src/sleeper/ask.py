"""Client: send a prompt to the sleeper.py pipeline.

Usage: uv run python ask.py what is the tallest lighthouse
"""
import json
import sys
import urllib.request

PORT = 17393


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        sys.exit("usage: ask.py <prompt...>")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/ask",
        data=json.dumps({"prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        print(resp.status)


if __name__ == "__main__":
    main()
