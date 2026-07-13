"""Deterministic tests for PlaybackTracker timing and per-turn isolation."""

import threading

import pytest

from sleeper.playback import PlaybackTracker


class Clock:
    """Controllable stand-in for time.monotonic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr("sleeper.playback.time.monotonic", c)
    return c


def test_no_emission_means_zero_played(clock: Clock) -> None:
    tracker = PlaybackTracker()
    assert tracker.played_samples() == 0
    assert tracker.heard_text() == ""


def test_played_samples_track_elapsed_wall_clock(clock: Clock) -> None:
    tracker = PlaybackTracker()
    clock.t = 0.0
    tracker.record_emission(4800)  # first frame stamps the clock at 0.0
    clock.t = 0.1  # 0.1s * 24_000 = 2_400 samples heard so far
    assert tracker.played_samples() == 2400


def test_word_heard_only_after_its_watermark_plays(clock: Clock) -> None:
    tracker = PlaybackTracker()
    clock.t = 0.0
    tracker.record_emission(4800)  # audio pumped while the word was consumed
    tracker.mark_spoken("hello")
    clock.t = 0.1  # only 2_400 played, watermark not yet reached
    assert tracker.heard_text() == ""
    clock.t = 0.3  # 7_200 clamps to 4_800 emitted; watermark reached
    assert tracker.heard_text() == "hello"


def test_reset_turn_clears_timing_counts_and_marks(clock: Clock) -> None:
    tracker = PlaybackTracker()
    clock.t = 0.0
    tracker.record_emission(4800)
    tracker.mark_spoken("hi")
    clock.t = 1.0
    assert tracker.played_samples() > 0
    assert tracker.heard_text() == "hi"
    tracker.reset_turn()
    assert tracker.played_samples() == 0
    assert tracker.heard_text() == ""


def test_wait_until_complete_returns_false_on_interruption(clock: Clock) -> None:
    tracker = PlaybackTracker()
    done = threading.Event()
    done.set()
    interrupted = threading.Event()
    interrupted.set()
    stopping = threading.Event()
    assert tracker.wait_until_complete(done, interrupted, stopping) is False


def test_wait_until_complete_returns_true_when_drained(clock: Clock) -> None:
    tracker = PlaybackTracker()
    done = threading.Event()
    done.set()
    interrupted = threading.Event()
    stopping = threading.Event()
    clock.t = 0.0
    tracker.record_emission(2400)
    clock.t = 10.0  # far past playback; heard clamps to the emitted count
    assert tracker.wait_until_complete(done, interrupted, stopping) is True


def test_two_trackers_are_isolated(clock: Clock) -> None:
    a = PlaybackTracker()
    b = PlaybackTracker()
    clock.t = 0.0
    a.record_emission(4800)
    a.mark_spoken("only a")
    clock.t = 1.0
    assert a.played_samples() == 4800
    assert b.played_samples() == 0
    assert b.heard_text() == ""
