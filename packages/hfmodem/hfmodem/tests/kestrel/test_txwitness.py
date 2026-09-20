# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The witness's audio detector, against every waveform this station transmits.

It used to look for shrike's 1400/1600 pair over the in-band median and it was wrong
twice over. It could not see VARA's OFDM or ARDOP's at all, so a 1.77 s connect
request came back as a scatter of 40 ms fragments and a VARA conclusion was published
off that and withdrawn. And it ANDed the two tone bins, which FSK never satisfies for
long — mark or space, one at a time — so it shredded shrike's own transmissions, the
one case everybody believed it had right.

Both defects die to the same test: energy in the passband, whatever is making it.
"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import time
import warnings
import wave

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora

T = corpora.harness("txwitness")

#: This station's own witness recordings. They live beside the checkout rather than
#: in it -- `working/` is written by a run and is in no clone -- so a run without
#: them measures the detector against synthesis alone, which is exactly where a
#: threshold change hides. Named here and warned about rather than skipped in
#: silence: a skip reason is invisible without `-rs`.
WITNESS = sorted((evidence.WORKING / "txwitness").glob("*.wav"))
if not WITNESS:
    warnings.warn(f"NO WITNESS RECORDINGS under {evidence.WORKING / 'txwitness'} -- "
                  "the mute detector is measured against synthesis only in this run",
                  stacklevel=1)
requires_witness = pytest.mark.skipif(
    not WITNESS, reason=f"no recordings under {evidence.WORKING / 'txwitness'}")

#: The shortest recording this station has made, and the one an end-to-end record
#: is measured against below. Shortest because a tap is recorded in something like
#: real time and the anatomy is what is being compared, not the length.
SHORTEST = [min(WITNESS, key=lambda p: p.stat().st_size)] if WITNESS else []

FS = 48000
#: The occupied bandwidth of each modem, and the one shape of emission each makes.
BANDS = {
    "shrike PACTOR-1": (1400.0, 1600.0),
    "ARDOP 500": (1250.0, 1750.0),
    "ARDOP 2000": (500.0, 2500.0),
    "VARA BW2300": (680.0, 2297.0),
}


def _carriers(secs: float, lo: float, hi: float, rms: float, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(secs * FS))) / FS
    freqs = np.linspace(lo, hi, max(2, int((hi - lo) // 50)))
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs)
    return x / np.sqrt(np.mean(x ** 2)) * rms


def _fsk(symbols: str, rms: float, baud: float = 100.0) -> np.ndarray:
    """PACTOR-1's pair at PACTOR-1's rate. ``symbols`` is a string of 0s and 1s."""
    n = int(round(FS / baud))
    hz = np.repeat([1400.0 if s == "0" else 1600.0 for s in symbols], n)
    return np.sin(2 * np.pi * np.cumsum(hz) / FS) * rms * np.sqrt(2)


def _noise(secs: float, rms: float, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(round(secs * FS))) * rms


#: The levels `20260815-205842-cal033.wav` measured with MONITOR_GAIN at its
#: calibration: band audio through the live receiver, the codec's own floor under
#: the mute, and the transmit audio over it.
LIVE, FLOOR, MONITOR = 0.05, 0.0001, 0.059


def _keying(emission: np.ndarray, lead=0.30, tail=1.10, before=3.0, after=3.0) -> np.ndarray:
    return np.concatenate([_noise(before, LIVE), _noise(lead, FLOOR),
                           emission + _noise(len(emission) / FS, FLOOR),
                           _noise(tail, FLOOR), _noise(after, LIVE, seed=9)])


def _wav(path, samples: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _measured(path) -> str:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.measure(str(path))
    return buf.getvalue()


def _spans(emission: np.ndarray, lead=0.30, tail=1.10, before=3.0):
    """``(audio over the emission, audio over the lead and the tail)``.

    A step either side of each edge straddles it and is left out of both.
    """
    audio = T.transmit_audio(_keying(emission, lead=lead, tail=tail, before=before))
    n = int(round(before / T.STEP_S))
    i0, i1 = n + int(round(lead / T.STEP_S)), n + int(round((lead + len(emission) / FS) / T.STEP_S))
    return audio[i0 + 1:i1 - 1], np.concatenate([audio[n + 1:i0 - 1],
                                                 audio[i1 + 1:i1 + int(round(tail / T.STEP_S)) - 1]])


@pytest.mark.parametrize("modem", list(BANDS))
def test_a_burst_from_any_modem_is_heard_for_its_whole_length(modem):
    """1.76 s of emission has to read as 1.76 s of transmit audio — for 200 Hz of FSK
    and for 2300 Hz of OFDM alike, and for the dead carrier at each end as silence."""
    lo, hi = BANDS[modem]
    burst, quiet = _spans(_carriers(1.76, lo, hi, MONITOR))
    assert burst.all(), f"{modem} came back in pieces: {100 * burst.mean():.0f}% heard"
    assert not quiet.any(), f"{modem}'s dead carrier read as audio"


def test_runs_of_one_fsk_symbol_do_not_break_the_burst():
    """The defect that made the old detector wrong about shrike, the modem everyone
    believed it had right: it wanted mark AND space inside one 20 ms step, and FSK
    sends one at a time, so a run of either read as a dead carrier. Half of this
    burst is one symbol repeated."""
    symbols = "01" * 40 + "0" * 40 + "01" * 20 + "1" * 40
    burst, quiet = _spans(_fsk(symbols, MONITOR))
    assert burst.all(), f"a run of identical symbols read as silence: {100 * burst.mean():.0f}% heard"
    assert not quiet.any()


def test_a_keyed_carrier_with_nothing_on_it_is_not_audio():
    burst, quiet = _spans(np.zeros(int(1.76 * FS)))
    assert not burst.any() and not quiet.any()


def test_the_anatomy_of_a_keying_comes_back_off_the_recording(tmp_path):
    """End to end: one keying, and the lead and tail read off its edges."""
    path = _wav(tmp_path / "ardop.wav", _keying(_carriers(1.76, 1250.0, 1750.0, MONITOR)))
    assert T.measure(str(path)) == 0
    key = [ln for ln in _measured(path).splitlines() if "key 1" in ln][0]
    audio, lead, tail = (float(v) for v in
                         re.search(r"audio.+\(([\d.]+) s\).+LEAD +(-?\d+) ms +TAIL +(-?\d+) ms",
                                   key).groups())
    assert audio == pytest.approx(1.76, abs=0.02)
    assert lead == pytest.approx(300, abs=20)
    assert tail == pytest.approx(1100, abs=20)


@pytest.mark.parametrize("rms,ok", [
    (MONITOR, True),
    (FLOOR * 2, False),                     # MONITOR_GAIN 0.2: down among the tap's own noise
    (0.5, False),                           # what the 2026-08-15 slot recorded at: railed
])
def test_a_monitor_without_headroom_over_its_floor_says_so_and_fails(tmp_path, rms, ok):
    """A measurement instrument that cannot tell you its input was out of range is
    how a retracted conclusion gets published. Neither half of the range is a level:
    both are read off the recording being measured, so no calibration of the day
    can go stale and none has to be believed."""
    path = _wav(tmp_path / f"{rms:.4f}.wav", _keying(_carriers(1.76, 1250.0, 1750.0, rms)))
    assert ("MONITOR UNUSABLE" not in _measured(path)) is ok
    assert (T.measure(str(path)) == 0) is ok


def test_a_burst_reaching_into_the_old_noise_band_is_one_keying():
    """The mute used to be read off 2000-2800 Hz alone, which is inside what this
    station transmits — so a wide burst RAISED the band the mute was read from
    instead of collapsing it, and the keying disappeared into its own edges. This
    emission reaches 300 Hz into that band; measured against one band it reads as no
    keying at all and two 40 ms fragments where the carrier came up and went down.

    The 2026-08-15 corpus is where the size of it shows: on
    `20260815-0201-vara-n0lcr1-force.wav` the keyed steps came out 10.5 dB ABOVE the
    listening ones, and 15 VARA keyings of 2.00 to 2.20 s read as 190 events.
    """
    keying = _keying(_carriers(2.14, 1000.0, 2300.0, MONITOR), lead=0.04, tail=0.02)
    spans = [(b - a) * T.STEP_S for a, b in T._runs(T.muted(keying))]
    assert len(spans) == 1, f"the keying came apart into {len(spans)}: {spans}"
    assert spans[0] == pytest.approx(2.20, abs=0.06)


def test_a_step_of_splatter_inside_a_keying_does_not_end_it():
    """What actually broke the keyings apart, and it is not the emission stopping —
    that only mutes the tap harder. It is the emission getting momentarily WIDER
    than itself: a monitor at the rail splatters across the bands the mute is read
    from, the level comes back up for a step or two, and a detector holding one
    threshold calls that a key-down and a key-up. The fragment gaps on the
    2026-08-15 captures are one and two steps almost to the exclusion of anything
    else, which is that and nothing else. Nothing this station transmits unkeys for
    20 ms and keys back up.
    """
    burst = _carriers(1.76, 1250.0, 1750.0, MONITOR)
    for at in (0.3, 0.5, 0.7):
        step = slice(int(at * FS), int((at + T.STEP_S) * FS))
        burst[step] += _noise(T.STEP_S, LIVE, seed=int(at * 10))
    assert len(T._runs(T.muted(_keying(burst)))) == 1


# -- the listening level the mute is read against ----------------------------
#
# One figure for a whole recording is a claim that the receiver read the same at
# minute 55 as at minute 2, and the taps this is pointed at are whole sessions.


def _across(monkeypatch):
    """Make the tool read its listening level the old way: one figure per file."""
    monkeypatch.setattr(T, "receiver_live_track",
                        lambda hi, step_s=T.STEP_S, window_s=T.LIVE_TRACK_S:
                        np.full(len(hi), T.receiver_live_level(hi)))


def test_the_listening_level_follows_a_receiver_that_gets_louder():
    """A session of overs and the listening between them, with eighteen seconds
    where the receiver reads 9.5 dB louder -- a station arriving on frequency, or a
    hand on the AF gain. Taken across the file the listening level is the quiet one,
    because that is where most of the listening is, and the keyings in the loud
    stretch are then judged against a level 9.5 dB under what the receiver was
    actually doing. That is the direction where a real emission stops reading as
    one.
    """
    listen, over, loud = np.full(500, 0.03), np.full(400, 0.0004), np.full(900, 0.09)
    session = np.tile(np.r_[listen, over], 5)
    level = np.concatenate([session, loud, session])
    assert T.receiver_live_level(level) == pytest.approx(0.03, rel=0.1), (
        "the level taken across this recording is meant to come out the quiet one")
    tracked = T.receiver_live_track(level)
    assert tracked[4700:5200].min() == pytest.approx(0.09, rel=0.2), (
        "the tracked level did not follow the receiver up")
    assert tracked[:4000] == pytest.approx(0.03, rel=0.1), (
        "and it followed something in the quiet stretch that was not the receiver")


@pytest.mark.parametrize("shape", ["a densely keyed stretch", "one long over", "flat"])
def test_the_tracked_level_never_reads_below_the_level_taken_across_the_file(shape):
    """The floor, and the reason for it. A window's estimate goes wrong by being
    built from too few listening steps -- and those are the steps nearest the mute,
    so the error is always downward. Unfloored, this cost real emission on real
    recordings: the 2026-08-15 calibration capture lost a 2.44 s keying whose every
    step carried transmit audio.
    """
    listen, mute = np.full(50, 0.03), np.full(100, 0.0004)
    level = {"a densely keyed stretch": np.tile(np.r_[listen, mute], 40),
             "one long over": np.r_[listen, np.full(4000, 0.0004), listen],
             "flat": np.full(4000, 0.03)}[shape]
    assert (T.receiver_live_track(level) >= T.receiver_live_level(level)).all()


@requires_witness
@pytest.mark.parametrize("path", WITNESS, ids=lambda p: p.name)
def test_no_keying_this_station_recorded_is_lost_to_tracking(path, monkeypatch):
    """Every keying the level taken across the file finds is still found, and none
    of them is shorter. Asserted against this station's own 2.65 hours of tap and
    the 794 keyings the old baseline finds in it, because that is a population no
    synthetic stands in for
    -- and because the failure it guards is silent: an emission that stops reading
    as one leaves a shorter number in a report and nothing else.
    """
    a = T._read(str(path))
    after = T.muted(a)
    with monkeypatch.context() as m:
        _across(m)
        before = T.muted(a)
    lost = [(u * T.STEP_S, (d - u) * T.STEP_S) for u, d in T._runs(before & ~after)]
    assert not lost, f"{path.name}: keyed time the tracked level gave up: {lost}"


# -- the anatomy at one millisecond ------------------------------------------
#
# `STEP_S` resolves an edge to two PACTOR-1 bits, and where bit 0 of a control
# signal leaves is a question about the first two.


def _edged(path) -> str:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T.edges(str(path))
    return buf.getvalue()


def _anatomy(line: str) -> tuple[float, float, float]:
    """``(body from, body to, receiver back)``, all in ms and the first two against
    key-up. `edges` prints no emission length and this returns none: see its
    docstring for why their difference is not one."""
    return tuple(float(v) for v in re.search(  # type: ignore[return-value]
        r"body from +([-+\d.]+) ms +to +([-+\d.]+) ms +receiver back +([-+\d.]+)",
        line).groups())


@pytest.mark.parametrize("lead,tail", [(0.30, 1.10), (0.05, 0.03), (0.028, 0.033)])
def test_both_instants_and_the_recovery_come_back_to_a_millisecond(tmp_path, lead, tail):
    """The three instants a 20 ms grain cannot separate. The last pair is this
    station's own: 28 ms of PTT lead and the 33 ms the FT-891 takes to let the
    receiver back, both of which round to one or two steps of `measure`.

    Synthesised, so there is no muted receiver and no flush and the far instant IS
    the lead plus the burst. That is what makes it a test of the detectors and not
    of the rig; what a recording through the rig does to the two is the subject of
    :func:`test_the_far_instant_is_the_one_a_self_capture_holds_still`.
    """
    burst = _fsk("01" * 48, MONITOR)
    path = _wav(tmp_path / f"{lead:.3f}.wav", _keying(burst, lead=lead, tail=tail))
    at, to, back = _anatomy([ln for ln in _edged(path).splitlines() if "key  1" in ln][0])
    assert at == pytest.approx(lead * 1e3, abs=2)
    assert to == pytest.approx((lead + len(burst) / FS) * 1e3, abs=3)
    assert back == pytest.approx(tail * 1e3, abs=3)


def test_the_span_the_detector_finds_does_not_move_with_the_key_around_it():
    """:func:`emission` reads both ends off the body level of the keying it is given
    and neither off the mute, so a rig that holds the key longer cannot move them.

    That is a property of the detector and it is all this asserts. It is not a
    licence to report the span as an emission length: on a recording made through
    our own receiver the near end lands on a transient of the mute's and the far is
    short by the flush, which is why `edges` prints the two instants and refuses
    their difference.
    """
    burst = _fsk("01" * 48, MONITOR)
    spans = set()
    for lead, tail in ((0.30, 1.10), (0.04, 0.02), (0.10, 0.50)):
        a = _keying(burst, lead=lead, tail=tail)
        up = T.key_edge(a, int(3.0 * FS))
        em = T.emission(a, up, T.key_edge(a, len(a) - int(3.0 * FS), down=True))
        spans.add(em[1] - em[0])
    assert len(spans) == 1, f"the same emission read {sorted(spans)} samples"
    assert spans.pop() / FS == pytest.approx(len(burst) / FS, abs=2 * T.EDGE_HOP_S)


def test_a_recording_that_ends_keyed_still_reports_what_came_before_it(tmp_path):
    """A session's own `stream.wav` stops when the session does, which is inside or
    just after its last transmission as a matter of course. `measure` returned on
    that and printed nothing, so the whole file was unreadable for the sake of its
    last burst -- and it is the one shape of recording this station has most of."""
    one = _keying(_fsk("01" * 48, MONITOR), before=3.0, after=3.0)
    a = np.concatenate([one, _noise(0.30, FLOOR), _fsk("01" * 48, MONITOR)])
    out = _measured(_wav(tmp_path / "cut.wav", a))
    assert "ends still keyed" in out
    assert "key 1:" in out, "the complete keying in front of it went unreported"


#: Taps recorded too quiet to hold the evidence. Named rather than measured, because
#: no measurement separates them from the 69 that read.
#:
#: `stream.wav` carries the mute only as the receiver's own audio going away, and our
#: transmission comes back under it through the codec's DAC-to-ADC crosstalk -- a level
#: that electrical path fixes rather than the operator sets, and it reads -34.4 dBFS
#: in-band on every 2026-08-22 tap. On these two the receive side was 16 dB down, at
#: -51.0 dBFS against -35.2 to -36.2 on `onair-0822-0929` through `-0938`, one rig and
#: minutes apart. The receiver therefore sat UNDER our own emission, and the deepest
#: 200 Hz slice anywhere in the spectrum falls 22.0 dB where :data:`MUTED` asks for 26.
#: The same slice falls 37.4 dB on `0929`, where our emission is within 1 dB of what it
#: is here and only the receiver moved. Read with a listening level taken off
#: hand-labelled listening steps, the median band ratio through a keying here is 0.489
#: against 0.839 while listening -- so the estimator is not what costs these keyings,
#: and there is no collapse for any estimator to find.
#:
#: Three thresholds were measured across all 71 taps and each puts a reading tap on the
#: wrong side. By receiver level, `onair-0820-1042` reads 21 keyings at -59.3 dBFS. By
#: receiver level against the loudest the tap sustains, `onair-0821-2054` reads 50 at
#: -21.4 dB where these two are at -18.2 -- the tap cannot tell our own emission from a
#: peer's, which is the whole reason the general precondition does not exist. The
#: deepest collapse held for `MIN_KEYING_S` overlaps outright. The 5th percentile of
#: the mute statistic does separate, by 0.50 dB, with eight reading taps inside 3 dB of
#: it: a line through two points, and the next wobble of the receive gain moves a tap
#: across it and excuses it in silence.
TOO_QUIET_TO_READ = {"onair-0822-0922", "onair-0822-0924"}
if TOO_QUIET_TO_READ:
    warnings.warn("recorded too quiet to hold a mute, so no keying is asserted off "
                  "them: " + ", ".join(sorted(TOO_QUIET_TO_READ)), stacklevel=1)

#: Sessions whose own transcript under `working/<name>/` says the tape holds no
#: self-witness, named with that verdict for the reason :data:`TOO_QUIET_TO_READ`
#: is: no statistic of the tape separates them from the taps that read. Six lost
#: the PTT line to the USB serial adapter mid-arm, three flew a 40 m receiver under
#: `levels.RX_FLOOR_DBFS`, where the mute never lifts between cycles and `muted`
#: merges them, and one is a 20 m tape on which mark and space come back 15 dB
#: apart, so no 10 ms of the body holds over :data:`EDGE_AT_BODY` of the loud tone.
#: The tapes that pass are no cleaner: on 30 m there is no self-emission on the
#: tape at all, and the test passes only because `emission`'s 80th-percentile
#: read treats an all-floor keying as fully on, while 20 m and 40 m read back at
#: -37 to -47 dBFS, bracketing the -34.4 dBFS this file generalises above.
NOT_A_SELF_WITNESS = {
    "onair-0906-1547": "PTT OFF NOT CONFIRMED after 86 rigctl failures: the serial "
                       "adapter left the bus and the key stood",
    "onair-0906-1553": "stuck PTT: RX level 0.0001 RMS through all 27 hush cycles, "
                       "digital silence under every keying",
    "onair-0906-1558": "KEYING FAILED at TX[10]: PTT port gone",
    "onair-0909-1236": "KEYING FAILED at TX[1]: PTT port gone, 1.0 s of tape",
    "onair-0909-1238": "20 m at 25 W: the emission is on the tape at -40 dBFS but "
                       "swings 18 dB bit by bit, and `emission` holds no 10 ms of it",
    "onair-0909-1931": "KEYING FAILED at TX[12]: PTT port gone",
    "onair-0909-1934": "KEYING FAILED at TX[5]: PTT port gone, 6.0 s of tape",
    "onair-0910-0930": "receiver -34.0 dBFS at the gate and under the floor by the "
                       "end: `muted` reads 2 keyings of the 16 keyed",
    "onair-0910-0932": "receiver under the floor by the end: 1 keying of 20 holds "
                       "both edges",
    "onair-0910-0959": "refused as TOO QUIET at -37.4 dBFS by its own gate and flown "
                       "anyway: cycles 1-20 merge into one 26 s keying",
}


@pytest.mark.parametrize("path", sorted(evidence.CAPTURES.glob("onair-*/stream.wav")),
                         ids=lambda p: p.parent.name)
def test_a_session_hears_its_own_transmissions_through_its_own_codec(path):
    """`_LiveInput` writes `stream.wav` from the capture callback, in front of the
    discard floor, so what the half-duplex mute takes off the decoder is still in
    the file -- including our own emission, back through the rig at about -42 dBFS
    against a -88 dBFS mute. Nothing had ever read it: the window captures start at
    `tx_end` by construction and `tools/rehear --ours` reports nothing for shrike
    because of it.

    Asserted loosely and on the shape rather than on this station's numbers -- what
    would break it is the recorder moving behind the discard, and that shows up as
    no emission at all rather than as a different figure.

    A held link answers most of its own cycles with a 115-135 ms control signal,
    not a 960 ms packet (`pactor1.control_signal`) -- a session this station spent
    mostly holding has more of those on the air than it has full packets, and the
    MEDIAN keying then reads short by protocol, correctly. So the span is asserted
    off the longest keying instead: a session that transmitted a real packet at all
    read at least one of them back whole.

    A tap on :data:`TOO_QUIET_TO_READ` is asserted only to be still unreadable. What
    the mute is read as is the receiver's own audio going away, and a tap recorded with
    the receive side under our own emission has none to lose.

    THE NEAR INSTANT IS WHERE THE BURST REACHES ITS BODY, not where its first
    sample left. PACTOR-1 snaps on and the two are one instant; a PACTOR-3 packet
    opens 30 to 38 dB under its body and takes 20 to 35 ms of its own to climb
    there, which no threshold recovers. `onair-0822-0934` upgraded mid-session and
    holds both: thirteen PACTOR-1 packets reach body at 44-52 ms and thirty-five
    PACTOR-3 packets at 78-90, one rig, one minute apart — and the key stood open
    1019 ms for 960 ms of PACTOR-1 and 919 ms for 862 ms of PACTOR-3, which is the
    same envelope around each. So the bound below covers the lead AND a
    multi-carrier opening ramp, and it is not a reading of the lead on its own.
    """
    if why := NOT_A_SELF_WITNESS.get(path.parent.name):
        pytest.skip(why)
    out = _edged(path)
    keys = [_anatomy(ln) for ln in out.splitlines() if re.search(r"key +\d+: key-up", ln)]
    if path.parent.name in TOO_QUIET_TO_READ:
        assert len(keys) < 3, (
            f"{path.parent.name} reads {len(keys)} keyings and is still on "
            "TOO_QUIET_TO_READ, where nothing is asserted off it -- take it off")
        return
    assert len(keys) >= 3, f"no keying read back out of {path.parent.name}:\n{out}"
    at, back = (float(np.median([k[i] for k in keys])) for i in (0, 2))
    ends = max(k[1] for k in keys)
    assert 20 <= at <= 90, (
        f"the body arrives {at:.0f} ms after key-up, which does not fit a PTT "
        f"lead, the transmit chain's latency and the burst's own opening ramp")
    assert ends > 0.5 * 960, (
        f"the longest keying ran to {ends:.0f} ms after key-up, where a 960 ms "
        "burst puts the end of the audio near 988")
    assert back > 0, "the receiver came back before the audio ended"
    assert "no emission length" in out, (
        "`edges` is offering an emission length off a recording made through our "
        "own receiver again")


#: Every arm of 2026-08-19 and -20 that holds a session recording: three bands,
#: two days, one burst length. All twenty rendered the same 960 ms connect, so a
#: figure that is the transmission reads the same on all twenty and a figure that
#: is this rig's receiver does not.
CONNECT_ARMS = [
    "onair-0819-1747", "onair-0819-1808", "onair-0819-1810", "onair-0819-1813",
    "onair-0819-1816", "onair-0819-2020", "onair-0819-2041", "onair-0819-2048",
    "onair-0819-2053", "onair-0819-2108", "onair-0819-2116", "onair-0819-2200",
    "onair-0819-2203", "onair-0819-2207", "onair-0819-2210", "onair-0820-1037",
    "onair-0820-1041", "onair-0820-1042", "onair-0820-1046", "onair-0820-1048",
]


def test_the_far_instant_is_the_one_a_self_capture_holds_still() -> None:
    """WHY `edges` PRINTS TWO INSTANTS AND NOT AN EMISSION LENGTH, on the corpus.

    All twenty arms sent the same 960 ms connect. Key-up to the END of the audio
    comes back at 986.5-990.0 ms on every one of them -- 3.5 ms across three bands
    and two days -- so it is a figure about the transmission. Key-up to the START
    swings 36.0-51.0 ms over the same twenty, in two clusters 13 ms apart, and it
    is not: `test_the_recording_loses_the_tail_the_air_kept` measures these same
    bursts by correlating each against the render that made it, which needs no
    level threshold at all, and finds the head a flat 20.8-23.9 ms behind the DAC
    on all twenty and exactly two bits gone off every tail. The near instant moves
    where the burst does not.

    What moves it is not the burst. Correlating the render across a whole keying
    finds it ONCE, at 0.97, 46-51 ms after key-up on all twenty -- including the
    five this reads at 36 -- so there is no early copy of the burst to have been
    found. What is there instead is low frequency: at +30 ms those five carry
    0.39-0.40 of full scale between 300 and 1000 Hz against 0.43-0.74 in the
    1250-1750 Hz tone pair, where the burst body carries 0.03-0.07 against 1.25,
    and PACTOR-1 puts nothing below 1400 Hz. `in_band_level` takes its RMS across
    the whole 300-2800 Hz passband, which is what it must do to see OFDM, so that
    transient alone clears `EDGE_AT_BODY` of the body and the leading edge lands
    on it.

    Asserted as the CONTRAST rather than as either figure, because the contrast is
    the argument: subtract the two and 17 ms of emission length appears that no
    recording here contains.
    """
    ends, starts = {}, {}
    for arm in CONNECT_ARMS:
        path = evidence.CAPTURES / arm / "stream.wav"
        if not path.exists():
            pytest.skip(f"{arm} is not in this checkout's evidence")
        keys = [_anatomy(ln) for ln in _edged(path).splitlines()
                if re.search(r"key +\d+: key-up", ln)]
        assert len(keys) >= 15, f"{arm} read back {len(keys)} keyings"
        starts[arm] = float(np.median([k[0] for k in keys]))
        ends[arm] = float(np.median([k[1] for k in keys]))

    said = "\n".join(f"  {a}  body {starts[a]:6.1f} -> {ends[a]:7.1f} ms"
                     for a in CONNECT_ARMS)
    spread = max(ends.values()) - min(ends.values())
    assert spread < 6.0, (
        f"key-up to the end of the audio moved {spread:.1f} ms across arms that "
        f"all sent the same 960 ms burst, so it is no longer the invariant "
        f"`edges` reports:\n{said}")
    assert max(starts.values()) - min(starts.values()) > 12.0, (
        "the leading edge no longer moves across these arms. If the transient "
        "has stopped reaching `EDGE_AT_BODY`, the emission length may be readable "
        f"after all -- check it against an off-air witness before believing it:\n{said}")


#: The seven arms a 40 m ingress was read off, and the bands they were called on.
#: 20m-arm3a is the deaf one -- the RF gain had not been raised yet -- and it is
#: kept, because a receiver 15 dB down is what makes the spread below worth
#: asserting.
SETTLE_ARMS = {
    "onair-0819-2048": "80m", "onair-0819-1808": "40m", "onair-0820-1042": "20m",
    "onair-0820-1046": "20m", "onair-0820-1037": "40m", "onair-0820-1041": "40m",
    "onair-0820-1048": "40m",
}


def _keyings(path):
    return [(float(up), *_anatomy(ln)) for ln in _edged(path).splitlines()
            for up in re.findall(r"key-up +([\d.]+) s", ln)]


def _band_dbfs(x: np.ndarray, lo: float, hi: float) -> float:
    f = np.fft.rfftfreq(len(x), 1 / FS)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / len(x) ** 2 * 8 / 3 + 1e-30)


def test_what_a_session_hears_inside_its_own_keying_is_a_pedestal_not_an_emission():
    """The level inside one of our own overs is set by the codec, not by the band.

    On 2026-08-20 the span between PTT and our first sample was found carrying as
    much 1250-1750 Hz energy on 40 m as the burst body that follows, against 16 to
    22 dB less on 20 m and 80 m, and read as in-band ingress specific to 40 m. The
    reference is what fails: the body is inside the mute too, and what comes back
    through it is our own transmission via the DAC-to-ADC crosstalk the note on
    :data:`TOO_QUIET_TO_READ` measures -- an electrical path, at a level no band
    and no operator moves.

    Seven arms, three bands, two days, and the receive chain 22 dB apart end to
    end: the body moves 0.5 dB. A settle read against it is a receive-gain meter,
    and the 16 dB it showed is the RF gain that had not been raised on 20m-arm3a.

    Asserted on the SPREAD rather than on the figure, because the figure is this
    station's codec and its transmit drive; what would break it is the pedestal
    starting to follow the band, which is the reading that has to stay refuted.
    """
    body, oob, rms = {}, {}, {}
    for cap in SETTLE_ARMS:
        path = evidence.CAPTURES / cap / "stream.wav"
        if not path.exists():
            pytest.skip(f"{cap} is not in this checkout's evidence")
        with wave.open(str(path)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768.0
        rms[cap] = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-30)
        mid = [int((up + (frm + to) / 2e3) * FS)
               for up, frm, to, _back in _keyings(path) if to > 500]
        assert len(mid) >= 3, f"{cap} read back {len(mid)} full keyings"
        n = int(0.2 * FS)
        body[cap] = float(np.median([_band_dbfs(x[i:i + n], 1250, 1750) for i in mid]))
        oob[cap] = float(np.median([_band_dbfs(x[i:i + n], 3200, 3800) for i in mid]))

    said = "\n".join(f"  {c} {SETTLE_ARMS[c]}  rms {rms[c]:7.2f}  body {body[c]:7.2f}"
                     f"  out of band {oob[c]:7.2f}" for c in SETTLE_ARMS)
    assert max(rms.values()) - min(rms.values()) > 15, (
        f"the seven arms no longer span the receive chain that makes this a "
        f"measurement:\n{said}")
    assert max(body.values()) - min(body.values()) < 2.0, (
        f"the level inside our own keyings has started to follow the band or the "
        f"receiver, and a settle read against it would mean something again:\n{said}")
    for cap in SETTLE_ARMS:
        assert body[cap] - oob[cap] > 25, (
            f"{cap}: the in-band pedestal is no longer above the muted receiver "
            f"beside it, so it is not our own emission coming back:\n{said}")


# -- the recording that is on disk while it is being made --------------------
#
# `record` held the whole capture in memory and wrote it when the duration
# elapsed, so a crash at minute 4 of a 5-minute tap yielded nothing at all -- and
# on 2026-08-26 a SIGSEGV from a USB interface going away mid-arm did exactly
# that, to the only instrument here that can measure a leading edge.

#: A sound card that plays a WAV instead of a room, faster than real time and
#: silent once it runs out -- which is what an interface going away looks like
#: from Python, in the case where it does not take the process with it.
_FAKE_CARD = '''
import os
import threading
import wave

import numpy as np

BLOCK = 4800
_WAV, _SPEED = os.environ["FAKE_CARD_WAV"], float(os.environ["FAKE_CARD_SPEED"])


def query_devices(device=None):
    card = {"name": "fake card", "max_input_channels": 1,
            "max_output_channels": 0, "default_samplerate": 48000.0}
    return card if device is not None else [card]


class _NothingLost:
    input_overflow = False


class InputStream:
    def __init__(self, samplerate=48000, channels=1, device=None, callback=None, **kw):
        with wave.open(_WAV) as w:
            self._a = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768.0
        self._callback, self._done = callback, threading.Event()
        self._thread = threading.Thread(target=self._play, daemon=True)

    def _play(self):
        for at in range(0, len(self._a), BLOCK):
            if self._done.wait(BLOCK / 48000.0 / _SPEED):
                return
            block = self._a[at:at + BLOCK].astype(np.float32).reshape(-1, 1)
            self._callback(block, len(block), None, _NothingLost())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join(timeout=2)
'''

TOOL = corpora.TOOLS / "txwitness.py"


def _tap(tmp_path, source: np.ndarray, seconds: float = 600, speed: float = 8):
    """`record` running against :data:`_FAKE_CARD`, playing `source`."""
    (tmp_path / "sounddevice.py").write_text(_FAKE_CARD)
    src = _wav(tmp_path / "source.wav", source)
    out = tmp_path / "tap.wav"
    env = corpora.child_env(FAKE_CARD_WAV=str(src), FAKE_CARD_SPEED=str(speed))
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env["PYTHONPATH"]])
    proc = subprocess.Popen(
        [sys.executable, str(TOOL), "record", str(seconds), str(out), "--device", "0"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, out


def _recorded_seconds(path) -> float:
    if not path.exists():
        return 0.0
    with contextlib.suppress(EOFError, wave.Error), wave.open(str(path)) as w:
        return w.getnframes() / FS
    return 0.0


def _wait_for(path, seconds: float, timeout: float = 30.0) -> float:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if (got := _recorded_seconds(path)) >= seconds:
            return got
        time.sleep(0.05)
    raise AssertionError(f"{path} held {_recorded_seconds(path):.2f} s after {timeout:g} s, "
                         f"not the {seconds:g} s the tap should have written through")


@pytest.mark.parametrize("sig", [
    pytest.param(signal.SIGKILL, id="SIGKILL"),     # the SIGSEGV's exit, near enough
    pytest.param(signal.SIGTERM, id="SIGTERM"),     # timeout(1), a supervisor
    pytest.param(signal.SIGINT, id="SIGINT"),       # the operator's keyboard
    pytest.param(signal.SIGHUP, id="SIGHUP"),       # a dropped ssh session
])
def test_a_tap_that_dies_mid_recording_keeps_everything_it_had(tmp_path, sig):
    """The whole defect. A crash, an unplug or a kill has to cost the tail and not
    the recording: what is on disk when the process stops must be a WAV that opens,
    holding the samples the card had delivered, and it must still be measurable.

    SIGKILL stands in for the SIGSEGV, which is the case nothing in this process
    gets to answer -- so it is the one that says whether writing through is enough
    on its own. The sidecar is what separates the two: a signal this tool handles
    closes the recorder and leaves one, and a file without a sidecar says the
    recording stopped rather than ended.
    """
    source = np.tile(_keying(_carriers(1.76, 1250.0, 1750.0, MONITOR)), 3)
    proc, out = _tap(tmp_path, source)
    try:
        _wait_for(out, 5.0)
        proc.send_signal(sig)
        proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()

    kept = T._read(str(out))
    assert len(kept) >= 5 * FS, f"only {len(kept) / FS:.2f} s survived {sig.name}"
    assert np.allclose(kept, source[:len(kept)], atol=2 / 32768), (
        f"what {sig.name} left is not what the card delivered")
    assert T._runs(T.muted(kept)), "the survivor holds no measurable keying"
    assert out.with_suffix(".json").exists() is (sig is not signal.SIGKILL), (
        "a sidecar is written when the tap is closed and not when it is killed")



@pytest.mark.parametrize("capture", ["synthetic", *SHORTEST],
                         ids=lambda c: getattr(c, "name", c))
def test_what_the_tap_writes_reads_back_with_the_same_anatomy(tmp_path, capture):
    """The emission figures may not move for the sake of the fix above.

    A recording made through `record` is compared against the same samples written
    in one go, instant by instant, `edges` against `edges` -- so a block boundary
    that ate a sample, or a scale that moved a level, shows up as a leading edge
    that has moved rather than as anything subtler.

    The card stopping is the other half. It runs out of audio here, which is an
    interface going away in the case where PortAudio does not take the process
    down with it, and the tap has to close on that rather than hold the device
    open for whatever wanted it next.
    """
    if capture == "synthetic":
        source = np.tile(_keying(_fsk("01" * 48, MONITOR), lead=0.05, tail=0.03), 2)
    else:
        source = T._read(str(capture))
    proc, out = _tap(tmp_path, source)
    said = proc.communicate(timeout=120)[0]

    assert proc.returncode == 1, f"the tap did not report a card that stopped:\n{said}"
    assert "went away" in said, said
    assert _recorded_seconds(out) == pytest.approx(len(source) / FS, abs=0.01), (
        f"the tap wrote {_recorded_seconds(out):.2f} s of the {len(source) / FS:.2f} s "
        f"the card delivered:\n{said}")
    whole = _wav(tmp_path / "whole.wav", source)
    assert _edged(out).splitlines()[1:] == _edged(whole).splitlines()[1:], (
        "the anatomy read off the tap's own recording is not the one read off the "
        "same samples written in one go")
