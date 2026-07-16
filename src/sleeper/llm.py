"""Voice-assistant LLM configuration, creation, and startup warmup."""

import json
import urllib.request

from libsh import get_logger
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.ollama import OllamaProvider

_logger = get_logger("llm")

OLLAMA_URL = "http://ollama-nvidia:11434"
LLM_URL = f"{OLLAMA_URL}/v1"
LLM_MODEL = "hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q5_K_M"
LLM_KEEP_ALIVE = "1440m"
LLM_REASONING_EFFORT = "none"
LLM_ENABLE_THINKING = False
LLM_WARMUP_PROMPT = "hi"

# Everything the LLM emits goes straight to TTS, so the instructions steer it toward
# speakable prose: no markup for the synthesizer to read aloud, numbers written out
# the way they're pronounced, and no long-form structure that only works on a screen.
SPEAKABLE_SYSTEM_PROMPT = """\
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


def create_llm_agent() -> Agent[None, str]:
  """Create the voice assistant agent over Ollama's OpenAI-compatible API."""
  return Agent(
    OpenAIChatModel(LLM_MODEL, provider=OllamaProvider(base_url=LLM_URL)),
    instructions=SPEAKABLE_SYSTEM_PROMPT,
    model_settings=OpenAIChatModelSettings(
      openai_reasoning_effort=LLM_REASONING_EFFORT,
    ),
  )


def warm_llm() -> None:
  """Load the LLM at startup and retain it for one day; discard its reply."""
  # HACK: Ollama's OpenAI-compatible endpoint silently drops keep_alive.
  # Sending the warmup through its native chat endpoint sets the loaded runner's
  # retention period. Later OpenAI-compatible requests reuse that runner without
  # replacing the period, so each completed request resets the full one-day timer.
  payload = json.dumps(
    {
      "model": LLM_MODEL,
      "messages": [{"role": "user", "content": LLM_WARMUP_PROMPT}],
      "stream": False,
      "keep_alive": LLM_KEEP_ALIVE,
      "think": LLM_ENABLE_THINKING,
    }
  ).encode("utf-8")

  request = urllib.request.Request(
    f"{OLLAMA_URL}/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
  )
  _logger.info("warming LLM")
  with urllib.request.urlopen(request) as response:
    response.read()
