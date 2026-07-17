#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==1.23.0",
#   "pyyaml==6.0.3",
#   "sounddevice==0.5.5",
#   "textual==8.2.8",
#   "websockets==16.1",
# ]
# ///
"""Audition Kyutai TTS voices through Sleeper's /say WebSocket backend.

A tiny Textual TUI: a spinner shows one voice at a time, left/right change the
voice and immediately synthesize a short random phrase, switching mid-playback
kills the current audio, space replays with a fresh phrase, `f` toggles a
favorite persisted to favorite-voices.yaml, and `a` auto-plays one phrase per
voice across the active set. Radio buttons (or 1/2/3) switch which voice set the
spinner cycles. This is a standalone client -- it never imports `sleeper`, which
would drag the server's torch/moshi dependency graph into a tiny UI, so the
two-field /say JSON is sent by hand.
"""

import argparse
import asyncio
import contextlib
import json
import queue
import random
import re
from pathlib import Path
from typing import Literal, cast

import sounddevice as sd
import yaml
from huggingface_hub import list_repo_files
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, RadioButton, RadioSet, Static
from textual.worker import Worker, WorkerState
from websockets.asyncio.client import connect

# The /say voice field takes a repo-relative id (e.g. expresso/ex03...wav).
# kyutai/tts-voices stores model-ready files as <voice_id>.<hex signature>@<epoch>.safetensors.
VOICE_REPO = "kyutai/tts-voices"
VOICE_SUFFIX_RE = re.compile(r"\.[0-9a-f]+@\d+\.safetensors$")
ENGLISH_DIRS = frozenset({"alba-mackenna", "ears", "expresso", "vctk"})
DEFAULT_URL = "ws://127.0.0.1:17393/say"
FAVORITES_PATH = Path("favorite-voices.yaml")

# Playback constants copied from src/sleeper/say.py: one 24 kHz mono int16 frame
# is 1920 samples / 3840 bytes, and the callback emits silence between frames.
OUTPUT_SAMPLES = 1920
OUTPUT_BYTES = OUTPUT_SAMPLES * 2
SAMPLE_RATE = 24_000
SILENCE = bytes(OUTPUT_BYTES)

VoiceSet = Literal["all", "english", "expresso"]
_SET_BY_ID: dict[str, VoiceSet] = {
  "all": "all",
  "english": "english",
  "expresso": "expresso",
}

PHRASES: tuple[str, ...] = (
  "I could get used to sounding like this.",
  "Is this thing on? Testing, testing.",
  "Somewhere, a kettle is about to boil.",
  "I'd narrate your grocery list if you asked.",
  "Warning: excessive charm detected.",
  "The forecast today is mildly dramatic.",
  "Pick me, I promise I'm the fun one.",
  "One small phrase for a voice, one giant leap for vibes.",
  "I sound exactly like this all the time, honestly.",
  "Left, right, left, right — keep them coming.",
  "This is the voice you never knew you needed.",
  "Beep boop, just kidding, I have feelings.",
  "Careful, I might grow on you.",
  "Roses are red, my latency is low.",
  "I'd read the terms and conditions and mean it.",
  "Consider this my audition tape.",
  "Ten out of ten, would speak again.",
  "Somebody order a smooth baritone?",
  "I contain multitudes, and also opinions.",
  "Fresh phrase, who dis?",
  "Marinating in reverb, one moment.",
  "You had me at spacebar.",
  "I'm contractually obligated to sound delightful.",
  "Plot twist: I'm your new favorite.",
  "Say the word and I'll say the words.",
  "This voice brought to you by pure enthusiasm.",
  "I practiced this in the mirror. The mirror approved.",
  "Adjusting my imaginary tie.",
  "Buffering personality... done.",
  "Encore already? You flatter me.",
  "If voices had resumes, mine would be suspiciously well written.",
  "I've been told I could read a spreadsheet and make it sound heroic.",
  "Somewhere out there is a perfect sentence, and I intend to find it.",
  "Give me a phone book and forty seconds, and I'll break your heart.",
  "I rehearse in the shower, which explains the reverb you're hearing.",
  "They said pick a lane, so I picked the one with the best acoustics.",
  "I promise to enunciate even the words that don't deserve it.",
  "Every syllable I say has been focus-grouped by exactly one person: me.",
  "I can do warm and reassuring, or I can do dramatic; your call, really.",
  "Think of me as the voice your smart speaker wishes it could grow into.",
  "I've got range, I've got timing, and I've got absolutely nothing to prove.",
  "Somewhere a director is yelling cut, but I'm just getting started here.",
  "I'll take your longest paragraph and return it with feeling, no charge.",
  "You can skip past me if you like, but you'll be thinking about me later.",
  "I contain the confidence of a weather anchor and the warmth of soup.",
  "Pick me and I'll pronounce your name correctly on the very first try.",
  "I've spent years training for a moment exactly this length and no longer.",
  "There are hundreds of us in here, and yet I feel uniquely worth your time.",
  "I sound like a Sunday morning that decided to get its act together.",
  "If you're auditioning voices, I'd like to formally submit myself as evidence.",
  "I can narrate your commute so well you'll miss your stop on purpose.",
  "Consider the possibility that I am, in fact, the one you've been scrolling for.",
  "I don't do filler words, except that one, and I regret it already.",
  "My whole personality fits in one breath, and I'm about to use it wisely.",
  "I've been practicing this exact cadence since before you pressed the key.",
  "Somebody has to be the voice of reason, and frankly it should probably be me.",
  "I'll read the fine print like it's poetry and the poetry like it's fine print.",
  "Between you and me, the other voices are lovely, but they can't do this part.",
  "I have a face for radio and a radio for a face; it works out beautifully.",
  "Give me a sentence and I'll give you a small, dignified performance.",
  "I peak at exactly the length of a to-do list read aloud with conviction.",
  "I was going to keep this short, but you seem like you appreciate a full thought.",
  "The forecast for this recording is clear skies with a chance of charisma.",
  "I can whisper a secret or announce a train; the versatility is genuinely absurd.",
  "You'll know within one sentence, and I think we both already know.",
  "I treat every phrase like it's the last one before the credits roll.",
  "My agent said keep them wanting more, so I'll stop right after this bit.",
  "I've got a voice built for late nights and long drives and longer stories.",
  "If you loop me for an hour, I promise to still sound like I mean it.",
  "I would like to gently insist that you have excellent taste for landing here.",
  "There's a version of this sentence in every voice, and this one is the good one.",
  "I do my best work in the space between two ordinary words.",
  "I'll say your grocery list with the gravity of a nightly news broadcast.",
  "You could keep browsing, but let's be honest about how this ends.",
  "I sound like the friend who always knows a quiet place to eat.",
  "I've been described as reassuring, which is a polite way of saying unforgettable.",
  "Give me the boring announcement and watch me make it strangely moving.",
  "I promise the second half of this sentence is even better than the first.",
  "I'm the voice you'd trust to read the instructions and mean every step.",
  "Somewhere a metronome is jealous of how well I keep a room's attention.",
  "I can be the calm before the storm or just the calm; dealer's choice.",
  "If enthusiasm were a currency, this recording would be embarrassingly rich.",
  "I'll take the pause before the punchline as seriously as the punchline itself.",
  "You've heard a lot of voices today, but you'll remember exactly one of them.",
  "I sound like a warm room with a good lamp and absolutely no small talk.",
  "Let the record show I arrived on time and left them wanting a second listen.",
  "I can carry a sentence the way a good waiter carries a full tray.",
  "My range spans bedtime story to breaking news, with a detour through comfort.",
  "I'd narrate your inner monologue, but honestly it's already in good hands.",
  "That's my closing argument, delivered with the confidence of a closing argument.",
)


class VoiceSpinner(Static):
  """A focusable one-voice-at-a-time selector line."""

  can_focus = True


class FavoriteStar(Static):
  """Clickable favorite indicator: filled star when the current voice is a favorite."""

  def on_click(self) -> None:
    cast("VoicePickerApp", self.app).action_toggle_favorite()


class VoicePickerApp(App[None]):
  """Textual TUI that auditions TTS voices via the standalone /say protocol."""

  CSS = """
  Screen {
    align: center middle;
  }
  #panel {
    width: 90;
    max-width: 100%;
    height: auto;
    border: round $accent;
    padding: 1;
  }
  #voice-set {
    layout: horizontal;
    height: auto;
  }
  #spinner {
    height: 3;
    content-align: center middle;
    text-style: bold;
  }
  #star {
    height: 3;
    content-align: center middle;
    color: $text-muted;
  }
  #star.-fav {
    background: $warning;
    color: $background;
    text-style: bold;
  }
  #phrase {
    height: 1;
    content-align: center middle;
  }
  #status {
    height: 1;
    content-align: center middle;
  }
  """

  # Priority bindings reserve the audition controls even when a radio button
  # holds focus; radio sets stay selectable by mouse and by 1/2/3.
  BINDINGS = [
    Binding("left", "previous_voice", "Previous", priority=True),
    Binding("right", "next_voice", "Next", priority=True),
    Binding("space", "replay", "New phrase", priority=True),
    Binding("f", "toggle_favorite", "Favorite", priority=True),
    Binding("a", "toggle_auto", "Auto", priority=True),
    Binding("1", "select_all", "All", show=False, priority=True),
    Binding("2", "select_english", "English", show=False, priority=True),
    Binding("3", "select_expresso", "Expresso", show=False, priority=True),
    Binding("q", "quit", "Quit", priority=True),
    Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
  ]

  def __init__(self, url: str) -> None:
    super().__init__()
    self.url = url
    self.active_set: VoiceSet = "all"
    self.voice_sets: dict[VoiceSet, list[str]] = {}
    self.index = 0
    self.phrase = ""
    self.auto = False
    self._auto_remaining = 0
    self._playback_worker: Worker[None] | None = None
    self._audio_queue: queue.SimpleQueue[bytes] = queue.SimpleQueue()
    self._audio_stream: sd.RawOutputStream | None = None
    self.favorites: set[str] = set()
    self._favorite_error: str | None = None
    self._load_favorites()

  # -- Typed views over the active voice set ------------------------------
  @property
  def voices(self) -> list[str]:
    return self.voice_sets.get(self.active_set, [])

  @property
  def current_voice(self) -> str:
    return self.voices[self.index]

  # -- Layout -------------------------------------------------------------
  def compose(self) -> ComposeResult:
    with Vertical(id="panel"):
      yield RadioSet(
        RadioButton("All", value=True, id="all"),
        RadioButton("English", id="english"),
        RadioButton("Expresso", id="expresso"),
        id="voice-set",
      )
      yield VoiceSpinner("Loading voices…", id="spinner")
      yield FavoriteStar("", id="star")
      yield Static("", id="phrase")
      yield Static("Idle", id="status")
    yield Footer()

  def on_mount(self) -> None:
    self.query_one("#spinner", VoiceSpinner).focus()
    self.run_worker(self._load_voices(), group="voices", exit_on_error=False)

  # -- Rendering helpers --------------------------------------------------
  def _render_voice(self) -> None:
    self._update_spinner()
    self._update_star()

  def _update_spinner(self) -> None:
    spinner = self.query_one("#spinner", VoiceSpinner)
    if not self.voices:
      spinner.update("Loading voices…")
      return
    # Build a plain Text so repository filenames cannot be read as Rich markup.
    spinner.update(Text(f"◀   {self.index + 1}/{len(self.voices)}   {self.current_voice}   ▶"))

  def _update_star(self) -> None:
    # The favorite control is a full-width band: it fills solid gold when the
    # current voice is a favorite and stays a faint hollow outline otherwise, so
    # the filled/outlined state reads at a glance rather than as a tiny glyph.
    star = self.query_one("#star", FavoriteStar)
    if not self.voices:
      star.update("")
      star.remove_class("-fav")
      return
    favorited = self.current_voice in self.favorites
    star.set_class(favorited, "-fav")
    star.update(Text(f"{'★' if favorited else '☆'}  Favorite"))

  def _update_phrase(self) -> None:
    self.query_one("#phrase", Static).update(Text(f"“{self.phrase}”" if self.phrase else ""))

  def _set_status(self, message: str) -> None:
    self.query_one("#status", Static).update(Text(message))

  # -- Voice enumeration --------------------------------------------------
  async def _load_voices(self) -> None:
    try:
      files = await asyncio.to_thread(list_repo_files, VOICE_REPO)
    except Exception as exc:
      self._set_status(f"Error loading voices: {exc}")
      return
    ids = sorted({VOICE_SUFFIX_RE.sub("", f) for f in files if f.endswith(".safetensors")})
    if not ids:
      self._set_status("Error loading voices: no voices found")
      return
    english = [v for v in ids if v.split("/", 1)[0] in ENGLISH_DIRS]
    expresso = [v for v in ids if v.split("/", 1)[0] == "expresso"]
    self.voice_sets = {"all": ids, "english": english, "expresso": expresso}
    self.active_set = "all"
    self.index = 0
    self._render_voice()
    # Surface a deferred favorites-load error once, without auto-playing.
    self._set_status(self._favorite_error if self._favorite_error is not None else "Idle")

  # -- Audio stream -------------------------------------------------------
  def _play_audio(
    self,
    outdata: memoryview,
    frames: int,
    _time: object,
    _status: sd.CallbackFlags,
  ) -> None:
    if frames != OUTPUT_SAMPLES:
      outdata[:] = bytes(len(outdata))
      return
    try:
      outdata[:] = self._audio_queue.get_nowait()
    except queue.Empty:
      outdata[:] = SILENCE

  def _ensure_audio(self) -> bool:
    if self._audio_stream is not None:
      return True
    stream: sd.RawOutputStream | None = None
    try:
      stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=OUTPUT_SAMPLES,
        channels=1,
        dtype="int16",
        callback=self._play_audio,
      )
      stream.start()
    except Exception as exc:
      if stream is not None:
        with contextlib.suppress(Exception):
          stream.close()
      self._set_status(f"Audio error: {exc}")
      return False
    self._audio_stream = stream
    return True

  def close_audio(self) -> None:
    stream = self._audio_stream
    if stream is None:
      return
    self._audio_stream = None
    stream.stop()
    stream.close()

  def _flush_audio(self) -> None:
    # Swapping in a fresh empty queue makes the PortAudio callback fall to
    # silence on its next 80 ms tick, cutting playback before Textual has even
    # delivered cancellation to the superseded worker.
    self._audio_queue = queue.SimpleQueue()

  # -- The single exclusive /say worker -----------------------------------
  async def _play(self, voice: str, text: str, job_queue: queue.SimpleQueue[bytes]) -> None:
    if self._audio_queue is job_queue:
      self._set_status("Synthesizing…")
    try:
      async with connect(self.url, compression=None) as websocket:
        await websocket.send(json.dumps({"text": text, "voice": voice}))
        first = True
        async for message in websocket:
          if not isinstance(message, bytes):
            raise ValueError("/say returned a non-audio message")
          if len(message) != OUTPUT_BYTES:
            raise ValueError(f"expected {OUTPUT_BYTES} PCM bytes, got {len(message)}")
          job_queue.put(message)
          if first:
            first = False
            if self._audio_queue is job_queue:
              self._set_status("Playing")
      # Drain the callback: wait for queued frames, then one frame's tail.
      while not job_queue.empty():
        await asyncio.sleep(0.005)
      await asyncio.sleep(OUTPUT_SAMPLES / SAMPLE_RATE)
    except asyncio.CancelledError:
      # Only flush when we still own the active queue; a newer job may already
      # have installed its own queue that must not be cleared.
      if self._audio_queue is job_queue:
        self._flush_audio()
      raise

  def _start_playback(self) -> None:
    if not self.voices or not self._ensure_audio():
      return
    text = random.choice(PHRASES)
    self.phrase = text
    self._update_phrase()
    self._flush_audio()
    job_queue = self._audio_queue
    self._playback_worker = self.run_worker(
      self._play(self.current_voice, text, job_queue),
      group="playback",
      exclusive=True,
      exit_on_error=False,
    )

  def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
    # Group names do not distinguish stale workers, so filter on identity: only
    # the live playback worker's terminal events drive status and sequencing.
    if event.worker is not self._playback_worker:
      return
    if event.state == WorkerState.ERROR:
      self.auto = False
      self._auto_remaining = 0
      self._flush_audio()
      self._set_status(f"Error: {event.worker.error}")
      return
    if event.state != WorkerState.SUCCESS:
      return
    if not self.auto:
      self._set_status("Idle")
      return
    self._auto_remaining -= 1
    if self._auto_remaining <= 0:
      self.auto = False
      self._set_status("Idle — auto complete")
      return
    # Advance without _move so auto stays enabled.
    self.index = (self.index + 1) % len(self.voices)
    self._render_voice()
    self._start_playback()

  # -- Navigation ---------------------------------------------------------
  def _move(self, delta: int) -> None:
    if not self.voices:
      return
    self.auto = False
    self._auto_remaining = 0
    self.index = (self.index + delta) % len(self.voices)
    self._render_voice()
    self._start_playback()

  def action_previous_voice(self) -> None:
    self._move(-1)

  def action_next_voice(self) -> None:
    self._move(1)

  def action_replay(self) -> None:
    if not self.voices:
      return
    self.auto = False
    self._auto_remaining = 0
    self._start_playback()

  def action_toggle_auto(self) -> None:
    if not self.voices:
      return
    if self.auto:
      # Leave the current phrase playing; completion just won't advance.
      self.auto = False
      self._auto_remaining = 0
      return
    self.auto = True
    self._auto_remaining = len(self.voices)
    self._start_playback()

  def action_toggle_favorite(self) -> None:
    if not self.voices:
      return
    voice = self.current_voice
    updated = set(self.favorites)
    if voice in updated:
      updated.discard(voice)
    else:
      updated.add(voice)
    try:
      FAVORITES_PATH.write_text(
        yaml.safe_dump({"favorites": sorted(updated)}, sort_keys=False),
        encoding="utf-8",
      )
    except Exception as exc:
      self._set_status(f"Error saving favorites: {exc}")
      return
    self.favorites = updated
    self._update_star()

  def action_select_all(self) -> None:
    self.query_one("#all", RadioButton).value = True

  def action_select_english(self) -> None:
    self.query_one("#english", RadioButton).value = True

  def action_select_expresso(self) -> None:
    self.query_one("#expresso", RadioButton).value = True

  def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
    if not self.voice_sets or event.pressed.id is None:
      return
    new_set = _SET_BY_ID.get(event.pressed.id)
    if new_set is None:
      return
    self.auto = False
    self._auto_remaining = 0
    previous_voice = self.current_voice
    self.active_set = new_set
    self.index = self.voices.index(previous_voice) if previous_voice in self.voices else 0
    self._render_voice()
    self.query_one("#spinner", VoiceSpinner).focus()
    if self.current_voice != previous_voice:
      self._start_playback()

  # -- Favorites persistence ----------------------------------------------
  def _load_favorites(self) -> None:
    if not FAVORITES_PATH.exists():
      return
    try:
      raw = yaml.safe_load(FAVORITES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
      self._favorite_error = f"Error loading favorites: {exc}"
      return
    if raw is None:
      return
    favorites = raw.get("favorites") if isinstance(raw, dict) else None
    if not isinstance(favorites, list) or not all(isinstance(v, str) for v in favorites):
      self._favorite_error = "Error loading favorites: malformed favorite-voices.yaml"
      return
    self.favorites = set(favorites)


def main() -> None:
  parser = argparse.ArgumentParser(description="Audition Kyutai TTS voices")
  parser.add_argument("--url", default=DEFAULT_URL, help="say WebSocket URL")
  args: argparse.Namespace = parser.parse_args()
  app = VoicePickerApp(url=args.url)
  try:
    app.run()
  finally:
    app.close_audio()


if __name__ == "__main__":
  main()
