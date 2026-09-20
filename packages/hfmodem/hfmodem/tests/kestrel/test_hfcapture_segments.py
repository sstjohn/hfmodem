# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A segment holds the length it was asked for, or the shortfall is on the record.

This file exists because of what the 2026-08-14 monitoring slots measured across
nine windows: `hfcapture.py record` asked ffmpeg for `-segment_time 600` and got
back 536-550 s, in every multi-segment recording on disk, on three bands. Nothing
was logged — ffmpeg ran at `-loglevel error` and had nothing to say, because from
its own point of view it did as it was told. Its
segment muxer cuts on *input timestamps*, avfoundation stamps the buffers that
arrive with true host time, and the buffers that never arrived left no timestamp
to notice. So the audio went missing from the middle of each ten-minute stretch,
every published wav offset was degraded by up to 10% of elapsed time, and a burst
straddling a gap was destroyed rather than truncated.

The measurement that pinned it, 2026-08-14: ffmpeg reads 2507 avfoundation packets
across a 29.80 s span, which is 26.74 s of audio; PortAudio on the same card over
the same seconds returns 0.9979 of real time with zero overflows. The loss is in
that one input — not the codec, the rig, the load, or the segmenter.

So `Capture` cuts on the sample count, which is the only clock a wav has, and the
claims tested here are the ones that were false before: a completed segment holds
exactly the frames it was asked for, audio that does go missing is reported with
its offset and its length, and a run that falls behind the wall clock with nothing
to show for it stops calling itself continuous.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import types
import wave

import pytest

from hfmodem.tests.kestrel import corpora

hfcapture = corpora.harness("hfcapture")

RATE = hfcapture.RATE
BLOCK = 512                                    # what this codec hands a callback
BLOCK_S = BLOCK / RATE


def blocks(count: int) -> bytes:
    return bytes(2 * BLOCK * count)


def wav_seconds(path) -> float:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
        return w.getnframes() / w.getframerate()


def device_run(seconds: float, deliver: float = 1.0, adc0: float = 1000.0):
    """``(pcm, adc_time)`` for a device running `seconds` and handing over
    `deliver` of them.

    The shortfall is spread across the run the way a dropping input spreads it,
    rather than taken out in one lump: a single hole is the easy case, and is not
    what the nine windows on disk contain.
    """
    adc, sent, due = adc0, 0, 0.0
    while adc - adc0 < seconds:
        due += deliver
        if int(due) > sent:
            yield blocks(int(due) - sent), adc
            sent = int(due)
        adc += BLOCK_S


def test_a_completed_segment_holds_the_frames_it_asked_for(tmp_path):
    cap = hfcapture.Capture(tmp_path, segment_frames=5 * RATE)
    for pcm, adc in device_run(12.0):
        cap.feed(pcm, adc)
    cap.close()

    assert len(cap.parts) == 3
    assert [wav_seconds(p) for p in cap.parts[:-1]] == [5.0, 5.0]
    assert cap.gaps == []


def test_an_underdelivering_device_lengthens_a_segment_never_shortens_it(tmp_path):
    """The 2026-08-14 failure replayed against the writer that replaced it.

    A device handing over 0.904 of real time while stamping true host time is what
    produced 545.65 s inside a segment cut at 600. The segment still holds its
    5 s here; what changes is that it takes 5.5 s of clock, and says where the
    missing audio was.
    """
    cap = hfcapture.Capture(tmp_path, segment_frames=5 * RATE)
    span = 0.0
    for pcm, adc in device_run(12.0, deliver=0.904):
        cap.feed(pcm, adc)
        span = adc + BLOCK_S - 1000.0
    cap.close()

    assert [wav_seconds(p) for p in cap.parts[:-1]] == [5.0, 5.0]
    assert cap.seconds == pytest.approx(0.904 * 12.0, rel=0.01)
    assert cap.lost == pytest.approx(span - cap.seconds, abs=0.05)
    # Every gap carries the offset it happened at in the wav's own seconds, in
    # order, so an analysis reads between them instead of across them.
    assert cap.gaps and all(0 <= at <= cap.seconds for at, _ in cap.gaps)
    assert [at for at, _ in cap.gaps] == sorted(at for at, _ in cap.gaps)


def test_one_hole_is_reported_at_its_offset_with_its_length(tmp_path):
    cap = hfcapture.Capture(tmp_path, segment_frames=5 * RATE)
    cap.feed(blocks(150), 500.0)                          # 150 blocks is 1.6 s
    cap.feed(blocks(75), 500.0 + 1.6 + 0.75)              # 75 is 0.8 s
    cap.close()

    assert cap.gaps == [(pytest.approx(1.6), pytest.approx(0.75))]
    assert wav_seconds(cap.parts[0]) == pytest.approx(2.4)


def test_clock_jitter_is_not_a_gap(tmp_path):
    """PortAudio's ADC stamps tracked the sample count to 0.1 ms over 20 s on this
    station's codec, with no single-buffer step past 3 us. A threshold that fired
    on that would report a gap in every run and would mean nothing."""
    cap = hfcapture.Capture(tmp_path, segment_frames=5 * RATE)
    for i in range(400):
        cap.feed(blocks(1), 900.0 + i * BLOCK_S + (3e-6 if i % 2 else -3e-6))
    cap.close()

    assert cap.gaps == []


def test_a_run_that_falls_behind_the_clock_is_not_continuous(tmp_path):
    """The backstop, and the one the old path needed most. A gap list is only as
    honest as the clock reporting it, so the wall clock is asked as well: 545.65 s
    of audio inside a 600 s stretch is the recorded failure, and it has to fail
    this even though the device admitted nothing."""
    cap = hfcapture.Capture(tmp_path, segment_frames=600 * RATE)
    cap.feed(blocks(int(545.65 / BLOCK_S)))              # no ADC clock at all
    cap.close()

    assert cap.gaps == []
    assert not cap.continuous(600.0)
    assert cap.continuous(cap.seconds + hfcapture.SLIP_S / 2)


# --- the whole verb, against a device that drops ---------------------------

class FakeStream:
    """A capture device that under-delivers, and then ends the run the way the
    queue runner does — `working/listenqueue.sh` sends SIGTERM to the recorder
    itself, having no ffmpeg child left to kill."""

    def __init__(self, callback, run):
        self.callback = callback
        self.run = run
        self.closed = threading.Event()

    def __enter__(self):
        threading.Thread(target=self._pump, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.closed.set()
        return False

    def _pump(self):
        for pcm, adc in self.run:
            self.callback(pcm, len(pcm) // 2,
                          types.SimpleNamespace(inputBufferAdcTime=adc), None)
        # Only while the recorder is the one holding the handler. If it left the
        # stream early — an assertion, a raise — this signal would land on the
        # test runner instead, and a suite that can kill pytest is worse than no
        # suite at all.
        if not self.closed.wait(0.2):
            os.kill(os.getpid(), signal.SIGTERM)


@pytest.fixture
def restore_signals():
    kept = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in kept.items():
        signal.signal(sig, handler)


def test_record_reports_the_loss_in_session_json(tmp_path, monkeypatch,
                                                 restore_signals):
    """What a reader meets two days later is `session.json`, so the loss has to be
    in it: every gap with its offset and length, and one field that says the
    recording is not a single timeline."""
    run = device_run(4.0, deliver=0.904)
    fake = types.SimpleNamespace(
        query_devices=lambda: [{"name": "USB Audio Device", "max_input_channels": 1}],
        RawInputStream=lambda **kw: FakeStream(kw["callback"], run))
    monkeypatch.setattr(hfcapture, "portaudio", lambda: fake)

    hfcapture.cmd_record(types.SimpleNamespace(
        device="USB Audio", outdir=str(tmp_path / "cap"), call="W9SSJ",
        peer="KB5LZK", band="40m", freq="7102.0", bandwidth="500", rig="FT-891",
        notes="", rig_model=None, rigctld=None, rig_poll=5.0))

    meta = json.loads((tmp_path / "cap" / "session.json").read_text())
    assert meta["continuous"] is False
    assert meta["gap_seconds"] > 0
    assert meta["gaps"] and all({"at_seconds", "seconds"} == set(g)
                                for g in meta["gaps"])
    assert meta["device"] == "USB Audio Device"
    # The audio that did arrive is all of it, and it is readable.
    assert meta["duration_seconds"] == pytest.approx(0.904 * 4.0, rel=0.05)
    assert wav_seconds(tmp_path / "cap" / "rx-000.wav") == \
        pytest.approx(meta["duration_seconds"], abs=0.05)
