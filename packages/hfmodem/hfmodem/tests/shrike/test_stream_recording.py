# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The session keeps one unbroken recording, on the windows' own sample clock.

A shrike session writes one WAV per receive window, and every one of them starts
at our own data end: `_LiveInput.flush_to(tx_end)` walks the capture floor past
each sample taken while our carrier was up, so that span reaches no file at all.
Two measurements are unavailable as a result, and both were wanted:

  * A PACKET CANNOT BE FOLLOWED ACROSS A WINDOW BOUNDARY. Concatenating the
    windows splices out every stretch we were keyed, so a drift measured across
    two of them can be observed and not attributed -- the peer re-timing itself
    and our own grid walking produce the same column of numbers.
  * A BURST ALREADY RUNNING AT A WINDOW'S FIRST SAMPLE HAS NO ONSET.
    `p1_burst_onsets` will not call one an onset (rxfront.py): no rising edge was
    observed. With only the window, that is indistinguishable from a burst that
    began inside the 55 ms our receiver is still muted for after the carrier
    drops (`onair.TR_SWITCH_S`, measured at -50 dB over 46 listening windows).
    The muted span is in the continuous file, so the profile runs across the
    window's start and the two cases separate.

Both are answered by the same file and only if it shares the windows' timebase:
the capture-stream sample index, which is what `end_stream_sample` in every
window sidecar is counted in. So the claim under test is positional, not merely
that a recording exists -- sample k of the session stream is capture-stream
sample k, and a window that says it ends at stream sample N is the N-len(w)..N
slice of it, byte for byte.

That is the claim with the teeth in it, so it is worth saying what these have
power against: arming the recorder one block late -- attaching it after the
stream rather than before -- leaves a file that is complete-looking, correctly
levelled, and 128 samples out of register, and it fails three of the six checks
below.

Run: pytest hfmodem/tests/shrike/test_stream_recording.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from hfmodem.core import levels, wav
from hfmodem.shrike import onair

FS = onair.FS
BLOCK = 128
BLOCKS = 60


class _Times:
    def __init__(self, t: float) -> None:
        self.currentTime = t
        self.inputBufferAdcTime = t
        self.outputBufferDacTime = t


class _Status:
    input_overflow = input_underflow = output_underflow = False
    priming_output = False


class _Card:
    """A sound card that delivers a known array to `_LiveInput`, a block at a time.

    `start` delivers one block because the constructor refuses to hand back a
    stream that has delivered nothing; the rest is driven by the test.
    """

    blocksize = BLOCK
    latency = 0.0

    def __init__(self, audio: np.ndarray, callback) -> None:
        self.audio, self.callback, self.n = audio, callback, 0

    def start(self) -> None:
        self.deliver(1)

    def deliver(self, blocks: int = 1) -> None:
        for _ in range(blocks):
            blk = self.audio[self.n:self.n + BLOCK]
            self.callback(blk.reshape(-1, 1), blk.size,
                          _Times(self.n / FS), _Status())
            self.n += blk.size

    def starve(self, blocks: int = 1) -> None:
        """Blocks the converter timestamped and the interpreter never accepted.

        No callback runs and no status flag is raised -- PortAudio saw nothing
        wrong, because from its side nothing was. The only trace is the step in
        `inputBufferAdcTime - samples/fs` the next callback carries.
        """
        self.n += blocks * BLOCK

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def _plant(n: int, seed: int) -> np.ndarray:
    """Samples that survive int16 on disk exactly, so equality is equality.

    k/32768 for |k| < 2**15 is representable in float32 and in the WAV, which
    makes a position check a comparison rather than a tolerance.
    """
    rng = np.random.default_rng(seed)
    return (rng.integers(-30000, 30001, n) / 32768.0).astype(np.float32)


def _open(monkeypatch, audio: np.ndarray, path):
    class _FakeSd:
        @staticmethod
        def InputStream(callback=None, **_):
            return _Card(audio, callback)

    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd)
    return onair._LiveInput("fake", None, record=path)


def test_the_stream_holds_the_span_the_windows_discard(tmp_path, monkeypatch):
    """Our own keyed span, which no window can contain, is on the disk.

    `flush_to` is the session's own discard: everything below it was captured
    while our carrier was up. It is the pre-roll a run that is already going at a
    window's first sample has to be read against, and no window has it.
    """
    planted = _plant(BLOCKS * BLOCK, 5)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    live._stream.deliver(BLOCKS - 1)
    live.flush_to(20 * BLOCK)                    # where our carrier dropped
    live.take_until(50 * BLOCK)
    live.close()

    got = wav.read(path)
    assert got.size == BLOCKS * BLOCK, f"{got.size} of {BLOCKS * BLOCK} recorded"
    assert np.array_equal(got[:20 * BLOCK], planted[:20 * BLOCK])


def test_a_window_sits_where_its_own_sidecar_says_it_does(tmp_path, monkeypatch):
    """`end_stream_sample` indexes the session stream directly.

    This is what following a packet across a window boundary rests on: two
    windows and the gap between them are one contiguous slice of this file.
    """
    planted = _plant(BLOCKS * BLOCK, 11)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    live._stream.deliver(BLOCKS - 1)
    windows = []
    # Two cycles: our carrier drops, we listen, we key again, we listen again.
    for keyed_to, listen_to in ((20 * BLOCK, 30 * BLOCK),
                                (45 * BLOCK, 55 * BLOCK)):
        live.flush_to(keyed_to)                  # where our carrier dropped
        seg = live.take_until(listen_to)
        seg_start = live.pos - seg.size          # the session loop's own two lines
        onair._save_capture(tmp_path / f"hold_{len(windows):02d}.wav", seg,
                            live.xruns, end=seg_start + seg.size)
        windows.append(seg)
    live.close()

    got = wav.read(path)
    ends = [json.loads((tmp_path / f"hold_{i:02d}.json").read_text())
            ["end_stream_sample"] for i in range(len(windows))]
    for end, seg in zip(ends, windows):
        assert np.array_equal(got[end - seg.size:end], seg), end
    # The stretch between the two windows -- our second transmission and the T/R
    # recovery around it -- which concatenating the windows splices out and which
    # is where a drift is either attributed or is not.
    gap = slice(ends[0], ends[1] - windows[1].size)
    assert gap.stop - gap.start == 15 * BLOCK, (gap.start, gap.stop)
    assert np.array_equal(got[gap], planted[gap])


def test_the_sidecar_pins_the_file_to_the_session_clock(tmp_path, monkeypatch):
    """A file nobody can place on the stream clock answers nothing months later,
    and neither does one whose grid is invalid without saying so."""
    planted = _plant(BLOCKS * BLOCK, 3)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    live._stream.deliver(BLOCKS - 1)
    live.close()

    side = json.loads(path.with_suffix(".json").read_text())
    assert side["first_stream_sample"] == 0
    assert side["samples"] == BLOCKS * BLOCK
    assert side["samplerate"] == FS
    assert side["normalised_on_write"] is False
    assert side["grid_loss_seen"] is False
    assert side["xruns"] == 0
    assert side["lost_samples"] == 0
    # The count and the file are two independent statements of the same length,
    # so a recording that lost its tail cannot read as a whole one.
    assert wav.read(path).size == side["samples"]
    assert f"{BLOCKS * BLOCK / FS:.1f} s" in live.stream_report()


def test_an_xrun_invalidates_the_stream_grid_as_it_does_a_window_s(
        tmp_path, monkeypatch):
    """An overflow drops samples the driver never handed over, so every index
    after it is offset from the air by an unknown amount. The windows carry
    `grid_loss_seen` for that; the continuous file is cut on the same clock
    and is wrong in the same way."""
    planted = _plant(BLOCKS * BLOCK, 9)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    card = live._stream
    card.deliver(10)
    over = _Status()
    over.input_overflow = True
    blk = card.audio[card.n:card.n + BLOCK]
    card.callback(blk.reshape(-1, 1), blk.size, _Times(card.n / FS), over)
    card.n += blk.size
    card.deliver(BLOCKS - 12)
    live.close()

    side = json.loads(path.with_suffix(".json").read_text())
    assert side["xruns"] == 1
    assert side["grid_loss_seen"] is True


def test_blocks_the_driver_never_flagged_reach_the_sidecars(tmp_path, monkeypatch):
    """The loss with no status word behind it, which is the loss this station has.

    A starved interpreter is not handed blocks the converter already timestamped.
    PortAudio raises nothing -- from its side nothing went wrong -- so `xruns`
    stays 0 and the old `grid_valid` read True over a spliced file. It did, on all
    68 capture-clock readings in this station's record, including the arm missing
    47.7 s of its own 86.8 s.

    The third window is the point of the cumulative reading: nothing is lost
    inside it, and it is still not on the air's clock, because everything ahead
    of it moved.
    """
    planted = _plant(BLOCKS * BLOCK, 17)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    sides = []

    def window(k: int, until: int) -> None:
        seg = live.take_until(until)
        onair._save_capture(tmp_path / f"rx_{k:02d}.wav", seg, live.xruns,
                            end=live.pos, lost=live.lost)
        sides.append(json.loads((tmp_path / f"rx_{k:02d}.json").read_text()))

    live._stream.deliver(19)
    window(0, 20 * BLOCK)                        # before the hole
    live._stream.starve(4)                       # 512 samples, 10.7 ms, unflagged
    live._stream.deliver(20)
    window(1, 40 * BLOCK)                        # across it
    live._stream.deliver(10)
    window(2, 50 * BLOCK)                        # after it, and still short
    live.close()

    assert live.xruns == 0, "the driver flagged what this test needs it to miss"
    assert [s["lost_samples"] for s in sides] == [0, 4 * BLOCK, 4 * BLOCK]
    assert [s["grid_loss_seen"] for s in sides] == [False, True, True]
    # The field that overclaimed is gone rather than shadowed: a reader that
    # still asks for it gets a KeyError and not a stale True.
    assert all("grid_valid" not in s for s in sides)

    side = json.loads(path.with_suffix(".json").read_text())
    assert side["lost_samples"] == 4 * BLOCK
    assert side["grid_loss_seen"] is True
    # And the file really is short by what it says: 50 blocks on the disk where
    # the converter timestamped 54.
    assert side["samples"] == 50 * BLOCK
    assert live._stream.n == 54 * BLOCK


def test_a_recording_that_lost_its_tail_says_so_and_does_not_stop_the_session(
        tmp_path, monkeypatch):
    """A full disk is the one way this file loses samples, and it loses them in
    the shape the file exists to avoid: a shorter recording that reads as a
    complete one. The tally names it, the sidecar carries it, and neither the
    write nor the close is allowed to raise through the teardown that unkeys."""
    planted = _plant(BLOCKS * BLOCK, 13)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    live._stream.deliver(BLOCKS - 1)

    def _full() -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(live._rec, "drain", _full)
    live._record()                               # the writing thread's own body
    live.close()

    assert "INCOMPLETE" in live.stream_report(), live.stream_report()
    assert "No space left on device" in live.stream_report()
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["truncated_by"] == "[Errno 28] No space left on device"


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))


def test_no_report_of_a_lossy_capture_can_read_as_a_clean_one(
        tmp_path, monkeypatch, capsys):
    """Every surface that grades this capture, against a stream the driver cleared.

    The loss is the one PortAudio does not raise: `input_overflow` stays false,
    `xruns` stays 0, and a grader that asks the flag says clean over a spliced
    stream. It did, over all 68 capture-clock readings on this record and over
    every one of the 450 windows that lost audio.

    So the flag staying zero is an assertion here rather than an accident, and
    every reading beside it has to disagree with it: the accessor the session,
    the preflight gate and the duplex bench all now ask, the line the session
    prints while it is running, the window sidecar, the continuous recording's
    sidecar, and the capture-clock line. A capture that lost samples has no way
    left to describe itself as whole.
    """
    monkeypatch.setattr(onair, "_peer_bursts", lambda seg, start: [])
    monkeypatch.setattr(onair, "_report_collision", lambda *a: None)
    # Written here rather than queued to the writer thread, so the sidecar is on
    # the disk by the time this reads it. Same body, same arguments.
    monkeypatch.setattr(onair, "_save_capture_async", onair._save_capture)
    planted = _plant(BLOCKS * BLOCK, 23)
    path = tmp_path / "stream.wav"
    live = _open(monkeypatch, planted, path)
    evidence = onair._CycleEvidence(live, None, None, tmp_path)

    live._stream.deliver(19)
    evidence.record("rx_00", live.take_until(20 * BLOCK), 0)
    assert onair._loss_seen(live) is None
    assert "CAPTURE LOSS" not in capsys.readouterr().out

    live._stream.starve(4)                       # 512 samples, 10.7 ms, unflagged
    live._stream.deliver(20)
    evidence.record("rx_01", live.take_until(40 * BLOCK), 20 * BLOCK)
    said = capsys.readouterr().out
    live.close()

    assert live.xruns == 0, "the driver flagged what this test needs it to miss"
    seen = onair._loss_seen(live)
    assert seen is not None and "512 samples" in seen and "11 ms" in seen, seen
    assert "CAPTURE LOSS: 512 samples" in said, said
    assert "SESSION INVALID for timing" in said, said
    assert "THE STREAM IS SPLICED" in live.clock_report()

    window = json.loads((tmp_path / "rx_01.json").read_text())
    stream = json.loads(path.with_suffix(".json").read_text())
    for side in (window, stream):
        assert side["lost_samples"] == 4 * BLOCK
        assert side["grid_loss_seen"] is True
        # And it is the count a reader meets first. `xruns` led this record
        # while it was the number that never moved, which is how a table of
        # per-session loss came to be read as a rate of nothing.
        assert next(k for k in side if "lost" in k or k == "xruns") \
            == "lost_samples", list(side)


def test_a_deaf_window_names_the_control_that_is_per_band(
        tmp_path, monkeypatch, capsys):
    """The other half of `_CycleEvidence`: what the window says about our own
    receiver. It said "raise the codec input gain", which is shared across bands
    -- following it to lift a deaf 20 m would have overdriven the 40 m that was
    already right. One sentence, in `core.levels`, for this and for `rx_verdict`.
    """
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **k: None)
    monkeypatch.setattr(onair, "_peer_bursts", lambda seg, start: [])
    monkeypatch.setattr(onair, "_report_collision", lambda *a: None)

    class _Live:
        fs, xruns, lost = FS, 0, 0

    onair._CycleEvidence(_Live(), None, None, tmp_path).record(
        "rx_00", np.zeros(FS, np.float32), 0)
    said = capsys.readouterr().out
    assert "DEAF" in said and levels.QUIET_CONTROLS in said, said
