"""Interactive text-only chat entry point for debugging the LLM in-container."""

from sleeper.llm import create_llm_agent


# Runs an interactive terminal chat against the container-resident llama-server,
# using the exact same agent the voice pipeline builds: same model, loopback
# OpenAI-compatible URL, speakable system prompt, model settings, and the
# get_local_datetime tool. This lets us debug LLM behaviour as plain text via
# `docker exec` without touching audio, WebSocket, or network routing. Pydantic AI
# owns terminal/session lifecycle and model-error handling, so no extra handling
# is added here.
def main() -> None:
  create_llm_agent().to_cli_sync(prog_name="sleeper-chat")
