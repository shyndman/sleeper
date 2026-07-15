"""Validated JSON messages shared by Sleeper's WebSocket peers."""

from typing import Literal

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class Say:
  text: str
  voice: str | None = None


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class TurnTranscript:
  role: Literal["user", "assistant"]
  text: str
  ended_by: Literal["turn_detected", "completed", "interrupted"]


SAY_ADAPTER = TypeAdapter(Say)
TURN_TRANSCRIPT_ADAPTER = TypeAdapter(TurnTranscript)
