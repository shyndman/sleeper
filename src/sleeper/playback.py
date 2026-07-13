"""Playback progress tracking for the assistant's spoken turn.

The tracker owns the timeline of a single assistant turn: samples emitted to the
client and, per finished sentence, the emitted-sample watermark at which that
sentence has fully played. "Heard" audio is derived, not asserted -- the client
plays at a fixed rate, so a sentence counts as heard only once wall-clock time
since the first frame implies its watermark of samples has left the speaker.
This lets the transcript report exactly what the user actually heard when a turn
is interrupted mid-sentence.
"""

import threading
import time
from dataclasses import dataclass, field

# Client plays back conversation PCM at this fixed rate; the heard-sample
# estimate multiplies elapsed wall-clock time by it.
PLAYBACK_SAMPLE_RATE = 24_000


@dataclass(slots=True)
class PlaybackTracker:
    """Tracks emitted vs. heard playback for one assistant turn.

    All fields are private; callers mutate only through the public methods so
    every read/write of the shared timeline stays under `_lock`.
    """

    _first_emitted_at: float | None = None
    _emitted_samples: int = 0
    _marks: list[tuple[int, str]] = field(default_factory=list)
    _changed: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_turn(self) -> None:
        """Clear timing, emitted count, and sentence marks for a new turn."""
        with self._lock:
            self._first_emitted_at = None
            self._emitted_samples = 0
            self._marks.clear()

    def record_emission(self, sample_count: int) -> None:
        """Account for `sample_count` samples sent to the client this frame."""
        with self._lock:
            if self._first_emitted_at is None:
                self._first_emitted_at = time.monotonic()
            self._emitted_samples += sample_count
        self._changed.set()

    def begin_sentence(self, text: str) -> None:
        """Mark a sentence as starting at the current emitted-sample count."""
        with self._lock:
            self._marks.append((self._emitted_samples, text))

    def finish_sentence(self) -> None:
        """Move the last sentence's watermark to the now-final emitted count.

        A no-op when no sentence mark exists, matching the guarded update the
        TTS worker performed before this refactor.
        """
        with self._lock:
            if self._marks:
                _, text = self._marks[-1]
                self._marks[-1] = (self._emitted_samples, text)

    def _played_locked(self) -> int:
        """Heard-sample estimate; caller must hold `_lock`."""
        if self._first_emitted_at is None:
            return 0
        elapsed = max(0.0, time.monotonic() - self._first_emitted_at)
        return min(self._emitted_samples, int(elapsed * PLAYBACK_SAMPLE_RATE))

    def played_samples(self) -> int:
        """Samples estimated to have actually played so far."""
        with self._lock:
            return self._played_locked()

    def heard_text(self) -> str:
        """Concatenated text of every sentence whose watermark has played."""
        with self._lock:
            played = self._played_locked()
            return " ".join(
                text for watermark, text in self._marks if watermark <= played
            )

    def wake_waiters(self) -> None:
        """Nudge any thread blocked in `wait_until_complete`."""
        self._changed.set()

    def wait_until_complete(
        self,
        synthesis_done: threading.Event,
        interrupted: threading.Event,
        stopping: threading.Event,
    ) -> bool:
        """Block until playback drains, is interrupted, or shutdown begins.

        Returns True only when synthesis has finished AND every emitted sample
        has played; False on interruption or shutdown. Reads state under
        `_lock`, releases it, then waits on `_changed` so emission never blocks
        behind the waiter.
        """
        while not stopping.is_set():
            if interrupted.is_set():
                return False
            with self._lock:
                emitted = self._emitted_samples
                played = self._played_locked()
            if synthesis_done.is_set() and played >= emitted:
                return True
            self._changed.wait(0.05)
            self._changed.clear()
        return False
