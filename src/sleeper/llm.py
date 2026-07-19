"""Voice-assistant LLM configuration and creation."""

from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Sleeper talks to the in-container llama-server over its local OpenAI-compatible
# endpoint. The model is loaded and health-gated by the entrypoint before this
# process starts, so no client-side warmup is needed. LLM_MODEL is any non-empty
# string -- llama-server serves the single model it loaded and ignores selection.
LLM_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "ternary-bonsai-27b"

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

_LOCAL_DATETIME_FORMAT = "%A, %B %d, %Y at %I:%M %p %Z"


# Gives the assistant the machine's current wall-clock time, including the weekday
# and local timezone, so time-sensitive answers do not depend on model knowledge.
def get_local_datetime() -> str:
  return datetime.now().astimezone().strftime(_LOCAL_DATETIME_FORMAT)


def create_llm_agent() -> Agent[None, str]:
  """Create the voice assistant agent over the local OpenAI-compatible server."""
  # api_key is a non-empty placeholder; the local llama-server does not check it.
  return Agent(
    OpenAIChatModel(LLM_MODEL, provider=OpenAIProvider(base_url=LLM_URL, api_key="local")),
    instructions=SPEAKABLE_SYSTEM_PROMPT,
    tools=[get_local_datetime],
  )
