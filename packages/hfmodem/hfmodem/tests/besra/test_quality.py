# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The decode quality we report back, and what a peer does with it.

Every DATAACK/DATANAK carries a five-bit quality code, and the far end climbs its
FSK→PSK→QAM ladder on a running average of exactly that number — shifting up when the
average clears a per-mode threshold. ``GetShiftUpThresholds`` (ARQ.c) sets those per
bandwidth *and* per rung, from 60 to 85, so there is no one bar to clear; the robust
rungs of the 500 and 2000 Hz ladders happen to sit at 80. So the number has to fall
when the channel does, or the peer is being told to go faster than the path allows.
Measured on air (2026-08-06 15:37Z, KE8LVA, BW2000): every ACK went out claiming
100 and the gateway climbed, from 4PSK.500.100 to 4PSK.1000.100 — one rung, and
nothing at that rung has ever been read here. The `16QAM.500` and `8PSK.2000` this
file used to name were never on the air; see
`test_the_gateway_we_told_100_climbed_one_rung_of_4psk`.

These tests hold the two halves of that: the demodulator's measurement tracks the
channel, and the session hands it to exactly the frames that grade a decode.
"""

from __future__ import annotations

import logging
import re
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.besra.arq import session as S
from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.frame import callsign
from hfmodem.besra.frame import frame as F
from hfmodem.besra.host.modem_core import ModemObserver
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.phy import quality as Q
from hfmodem.besra.phy.demodulator import (SAMPLE_RATE, DecodedFrame, Demodulator,
                                            _HEADER_LEN, _apply_rs_floor, _body_span)
from hfmodem.besra.sim.air import StepAir
from hfmodem.besra.sim.channel import add_awgn

from . import groundtruth as gt
from .test_arq_session import Pair

_FIXTURES = Path(__file__).parent / "fixtures"

#: The bar the robust rungs of ``byt500`` and ``byt2000`` set (ARQ.c), tested there
#: with a strict ``>``: a peer whose average sits at or under it holds *that* mode.
#: Not the lowest bar in the tables — six of their twenty-three sit below it, which
#: is the point `test_the_rs_floor_and_its_bars_are_read_off_the_reference` makes.
SHIFT_UP_FLOOR = 80

#: The unconnected ID frame, one of the three types that ride the forced 0xFF
#: session id and so reach the state machine without being addressed to it.
_IDFRAME = S._type_of("IDFrame")


def _reported(frame_type: int) -> int:
    """The quality a DATAACK/DATANAK of this type states (``DecodeACKNAK``)."""
    return 38 + 2 * (frame_type & 0x1F)


# -- the formulas, against their own definitions -----------------------------

def test_perfect_4fsk_symbols_score_100():
    """All the power on the decided tone: nothing landed off-constellation."""
    mags = np.zeros((4, 40))
    mags[np.arange(40) % 4, np.arange(40)] = 1.0
    assert Q.fsk_quality(mags) == 100


def test_indistinguishable_4fsk_symbols_score_0():
    """Four equal tones — the decision is a coin toss, and the score says so."""
    assert Q.fsk_quality(np.ones((4, 40))) == 0


def _mags(share_off_tone: float, nsym: int) -> np.ndarray:
    """``nsym`` 4FSK symbols each leaking ``share_off_tone`` of their power evenly
    onto the three tones that lost."""
    m = np.full((4, nsym), share_off_tone / 3.0)
    m[0] = 1.0 - share_off_tone
    return m


def test_the_4fsk_distance_is_clamped_at_37():
    """``min(37, ...)`` is the reference's ideal-radius arithmetic, and it is load
    bearing: a symbol that leaks three quarters of its power scores 60 before the
    clamp. Half such symbols and half perfect ones average 18.5 clamped and 30 not,
    which is 50 against 19 — and neither run is at an end of the scale where the
    0-100 clamp would hide the difference."""
    mags = np.concatenate([_mags(0.0, 20), _mags(0.75, 20)], axis=1)
    assert Q.fsk_quality(mags) == 50


def test_the_4fsk_distance_is_a_floor_not_a_round():
    """Integer arithmetic throughout, ``⌊80·off⌋``.

    The off-tone share has to be picked so that flooring, rounding and doing
    neither all part company. This one used 0.30625, where 80·off is 24.5 exactly:
    numpy rounds a half to even, so ``np.round`` gave the same 24 as ``np.floor``
    and the check was blind to the substitution it names. At 0.30875, 80·off is
    24.7 — floor 24 and 100 − 2.7·24 = 35.2 reports 35, round 25 and 32.5 reports
    32, no truncation at all and 33.31 reports 33.
    """
    assert Q.fsk_quality(_mags(0.30875, 40)) == 35


@pytest.mark.parametrize("raw,want", [(150.0, 100), (100.0, 100), (0.0, 0),
                                      (-62.0, 0), (float("nan"), 0)])
def test_the_reported_quality_stays_inside_the_wire_code(raw, want):
    """0-100 is all a five-bit DATAACK code can carry, so the clamp is part of the
    metric rather than tidiness. NaN floors rather than clamping: ``min(100.0, nan)``
    is ``100.0``, so the plain two-sided form answered *flawless* for a measurement
    that was not a number."""
    assert Q._clip(raw) == want


def test_a_qam_ring_that_carried_nothing_falls_back_to_the_phases():
    """Where that NaN came from, and what it cost. A carrier that dropped out leaves
    the inner ring at zero, and the scatter term divides that ring's error by its own
    mean: the whole score went NaN and came back as 100, the highest number this
    receiver can tell a peer, off a frame whose amplitudes were gone. An unmeasurable
    ring means the radius error is unmeasurable, which is already what this metric
    does when every symbol lands in one ring — the phase score stands alone."""
    step = 785.4
    dphase = np.array([step * (k + 0.25) for k in range(64)])   # scuffed, not perfect
    mag = np.concatenate([[1.0], np.zeros(32), np.ones(32)])
    assert Q.qam_quality(dphase, mag, step) == Q.psk_quality(dphase, step) == 50


def test_the_qam_ring_split_sits_between_the_two_rings():
    """Where the 75% of the peak that separates the rings may and may not sit.

    The fallback test above uses magnitudes of exactly 0 and 1, and against two
    exact rings *every* split strictly between them partitions the symbols the
    same way — 0.30, 0.60 and 0.99 all report the same number, so it pins nothing.
    Rings arrive off a channel with jitter on them, and then the split has room to
    be wrong in both directions.

    These are the rings ARDOP actually sends: 16QAM's half-magnitude flag is a
    literal ``>> 1`` (`modulator._psk_data`), so the inner ring is half the outer,
    here with 2% scatter and phases a tenth of a step off the grid. Anything from
    0.52 to 0.93 of the peak reads the same 76 — which is why the pin here is a
    band and one number stands for it. Outside the band the reading moves, and it
    moves the way that matters: at 0.30 every symbol counts as one ring, the
    radius error goes unmeasured and the score falls back to the phases' own 80,
    reporting a better path than was measured. At 0.99 most of the outer ring is
    misfiled as inner, the scatter term reads that misfiling as channel and the
    score collapses to 42, holding a peer well below the rung the path carries.
    """
    step = 785.4
    dphase = np.array([step * (k + 0.1) for k in range(64)])
    rings = np.empty(64)
    rings[0::2], rings[1::2] = 0.5, 1.0
    mag = rings + np.random.default_rng(0).normal(0.0, 0.02, 64)

    assert Q.qam_quality(dphase, mag, step) == 76
    assert Q.psk_quality(dphase, step) == 80, "the phases alone, ungraded by radius"


def test_psk_phases_on_the_grid_score_100():
    step = 785.4
    assert Q.psk_quality(np.array([step * k for k in range(64)]), step) == 100


def test_psk_phases_between_grid_points_score_0():
    """Half a step from every decision is as wrong as a differential phase can be.

    Not exactly zero: the reference sums the error over every phase but the first and
    still divides by the full count, so one symbol's worth of error goes missing. That
    off-by-one is reproduced rather than tidied — it is the number peers were tuned
    against — and at 64 symbols it is worth exactly 1.5 points, which the integer cast
    reports as 1. A tidied metric, dividing by the 63 it summed, would report 0.
    """
    step = 785.4
    assert Q.psk_quality(np.array([step * (k + 0.5) for k in range(64)]), step) == 1


def test_the_first_differential_phase_is_not_measured():
    """The other half of the same off-by-one: the reference skips its first stored
    phase, so wrecking it changes nothing. Both halves have to be pinned — dropping
    the skip alone leaves the divisor wrong in the opposite direction."""
    step = 785.4
    on_grid = np.array([step * k for k in range(64)], dtype=float)
    assert Q.psk_quality(on_grid, step) == 100

    wrecked = on_grid.copy()
    wrecked[0] += step / 2
    assert Q.psk_quality(wrecked, step) == 100

    wrecked = on_grid.copy()
    wrecked[1] += step / 2
    assert Q.psk_quality(wrecked, step) < 100


# -- the measurement, against a channel --------------------------------------

def _decode_one(samples):
    frames = Demodulator().decode(np.concatenate(
        [np.zeros(1200, dtype="<i2"), samples, np.zeros(2400, dtype="<i2")]))
    assert frames, "frame did not decode"
    return frames[0]


@pytest.mark.parametrize("ftype", [0x4C, 0x4A])
def test_clean_data_frame_reports_near_perfect_quality(ftype):
    frame = _decode_one(M.render_frame(ftype, payload=b"QUALITY" * 4, session_id=0x5E))
    assert frame.ok
    assert frame.quality > SHIFT_UP_FLOOR, "a clean path must be allowed to speed up"


@pytest.mark.parametrize("ftype,snr", [(0x4C, -4), (0x4A, -4), (0x40, -4)])
def test_marginal_data_frame_reports_a_quality_that_holds_the_mode(ftype, snr):
    """A frame that only just decodes must not be reported as flawless.

    This is the channel that matters: noisy enough to scuff the constellation, clean
    enough that the payload still arrives — exactly where the on-air session sat
    while it kept answering 100. Measured, 8 noise realisations per point: every one
    of these decodes 8/8 and reports 80, the RS floor, so a peer on a rung whose bar
    is 80 holds the mode. It does *not* hold every rung — 80 clears six of the
    reference's twenty-three bars, and reproducing that is deliberate
    (`besra.phy.quality.RS_CLEAN_FLOOR`). What the floor is doing here rather than
    the constellation is `test_the_rs_floor_is_what_lifts_a_marginal_decode`; without
    it this assertion compares one constant against another.
    """
    payload = b"QUALITY" * 4
    clean = M.render_frame(ftype, payload=payload, session_id=0x5E)
    scored = [f.quality for f in (_decode_one(add_awgn(clean, snr, seed=s))
                                  for s in range(8))
              if f.ok and f.payload == payload]
    assert len(scored) >= 6, f"only {len(scored)}/8 decoded — pick a kinder channel"
    assert max(scored) <= SHIFT_UP_FLOOR, (
        f"marginal decodes reported {scored}; a peer reading that shifts up")


def test_undecodable_frame_reports_well_below_the_floor():
    """The NAK path. A frame the RS could not rescue has nothing propping its score
    up, so the number the peer averages in is a long way under any shift-up trip."""
    clean = M.render_frame(0x44, payload=b"QUALITY" * 4, session_id=0x5E)
    frames = [_decode_one(add_awgn(clean, -4, seed=s)) for s in range(6)]
    assert not any(f.ok for f in frames), "channel was kinder than intended"
    assert max(f.quality for f in frames) < 70


def test_quality_falls_as_the_channel_does():
    clean = M.render_frame(0x4C, payload=b"QUALITY" * 4, session_id=0x5E)
    by_snr = {snr: float(np.median([_decode_one(add_awgn(clean, snr, seed=s)).quality
                                    for s in range(6)]))
              for snr in (30, 8, 2)}
    assert by_snr[30] > by_snr[8] > by_snr[2], by_snr


def test_a_real_gateway_frame_off_air_is_not_reported_perfect():
    """The arbiter that owes nothing to a simulated channel.

    ``offair_ke8lva_greeting.wav`` is a genuine 4PSK.500.100.O from KE8LVA on
    7103.5 kHz — the frame that carried the Winlink greeting, RS-corrected, payload
    byte-exact. It measures 66: a decode that worked, over a path that was working
    hard for it. besra answered that frame 100 and every other frame 100, and the
    next session with this gateway climbed a rung on the strength of it and was not
    read again. 66 sits under every shift-up threshold in the table.
    """
    path = _FIXTURES / "offair_ke8lva_greeting.wav"
    with wave.open(str(path)) as w:
        au = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    frame = next(f for f in Demodulator().decode(au) if f.ok and f.payload)
    assert frame.payload.startswith(b"RMS Trimode")
    assert frame.quality == 66


#: The eight tones a PSK/QAM mode rides (`dsp.templates.PSK_CARRIERS_HZ` less the
#: 1500 Hz the multi-carrier modes skip), and the frequencies halfway between them.
#: How many of the first a burst lights is its width, and nothing about it is ours:
#: two carriers is a 500 Hz mode, four a 1000, eight a 2000.
_CARRIER_GRID = (800, 1000, 1200, 1400, 1600, 1800, 2000, 2200)
_BETWEEN_CARRIERS = (900, 1100, 1300, 1500, 1700, 1900, 2100)


def _carrier_comb(au: np.ndarray, lo: int, hi: int) -> dict[int, float]:
    """Power at each grid frequency over the median power between them, across one
    frame's body. Welch over 32k windows at quarter overlap; 50 Hz around each tone,
    which is the 100-baud symbol's own null-to-null width."""
    seg = np.asarray(au[lo:hi], dtype=np.float64)
    n = 1 << 15
    acc = np.zeros(n // 2 + 1)
    for i in range(0, seg.size - n + 1, n // 4):
        acc += np.abs(np.fft.rfft(seg[i:i + n] * np.hanning(n))) ** 2
    hz = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)

    def power(f: int) -> float:
        return float(acc[(hz >= f - 25.0) & (hz <= f + 25.0)].sum())

    floor = float(np.median([power(f) for f in _BETWEEN_CARRIERS]))
    return {f: power(f) / floor for f in _CARRIER_GRID}


@pytest.mark.parametrize("fixture, mode, carriers", [
    ("offair_ke8lva_greeting.wav", "4PSK.500.100.O", (1400, 1600)),
    ("offair_ke8lva_climbed.wav", "4PSK.1000.100.O", (1200, 1400, 1600, 1800)),
])
def test_the_gateway_we_told_100_climbed_one_rung_of_4psk(fixture, mode, carriers):
    """What KE8LVA did with a hardcoded 100, read off its own carriers.

    This file said for three weeks that the gateway climbed to `16QAM.500` and
    `8PSK.2000`, while a 2026-08-26 sweep of the corpus found that no station has
    ever sent this receiver a 16QAM frame at all. Both cannot hold, and nothing had
    re-read the session. Replayed through `RollingDecoder` with the session's own id,
    `20260806T153709Z-besra-7102000.wav` holds one gearshift: twelve
    `4PSK.500.100` arrivals carrying the greeting and the `;PQ:` challenge, our
    ACKs, then thirteen `4PSK.1000.100.O` from 161.30 s that never decoded — the
    session that ended on `*** Unknown client types are not allowed`. There is no
    16QAM sighting anywhere in it.

    The two fixtures are one burst each, 6.25 s of the same gateway on either side
    of that shift, and the comb settles it without asking the header anything: a
    500 Hz mode rides two carriers and a 1000 Hz mode rides four, so the claim
    under test is a count of lit tones. That rules out both names this file carried
    by geometry alone — `16QAM.500.100` rides the same two carriers any 500 Hz mode
    does and would leave 1200 and 1800 dark, and `8PSK.2000.100` rides all eight
    and would light 800 through 2200. Within the 1000 Hz family the header picks
    4PSK over 8PSK and 16QAM, corroborated by the session id beside it, which a tone
    grid cannot fake.

    Every burst the recording holds whole was measured, not just these two. Over
    the twelve at the 500 Hz rung, 1400 and 1600 read 2.1 to 21.6 while 1200 and
    1800 read 0.7 to 1.2 — bare band on this recording reads 0.7 to 1.4 right
    across the grid. Over the twelve at the rung above, all four of 1200 to 1800
    read 2.0 to 13.0 while 800, 1000, 2000 and 2200 read 0.3 to 1.3. Not one burst
    on either side of the shift is ambiguous. The fixtures are the clearest of
    each, and the bars below are theirs.
    """
    with wave.open(str(_FIXTURES / fixture)) as w:
        assert w.getframerate() == SAMPLE_RATE
        au = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)

    got = Demodulator(expect_session=lambda: 0x51).decode(au)
    assert [(f.name, f.session_id) for f in got] == [(mode, 0x51)]

    body = got[0].offset + _HEADER_LEN
    comb = _carrier_comb(au, body, body + _body_span(got[0].type))
    lit = {f: v for f, v in comb.items() if f in carriers}
    dark = {f: v for f, v in comb.items() if f not in carriers}
    assert min(lit.values()) > 3.0, comb
    assert max(dark.values()) < 2.5, comb


def test_bare_control_frame_has_nothing_to_measure():
    """No body, no grade — the last reading stands, as it does in the reference."""
    assert _decode_one(M.render_frame(S.IDLE, session_id=0x5E)).quality is None


@pytest.mark.parametrize("ftype", [0x4C, 0x50, 0x70, 0x7A, 0x7B])
def test_a_body_the_capture_cut_short_reports_nothing_arrived(ftype):
    """A body that ran off the end of the window used to report *no* measurement,
    and `intLastRcvdFrameQuality` is a standing value here as it is in the
    reference — so the NAK went out carrying the last good frame's score. Measured:
    a 96, then a DATANAK 0x1D, which reads back as 96.

    It reports 0. Grading the prefix instead is no better: past a capture's edge
    there is only padding, the padding is phase-perfect, and half a frame plus
    silence scored 100. Truncation is the characteristic failure at the top of the
    gearshift ladder, where an honest number matters most.

    The 600-baud pair are here because the first repair inverted itself on exactly
    them. They are the only frames read in three sub-blocks, the measurement was
    kept as soon as *any* sub-block fitted, and a half-capture graded its surviving
    prefix at 95 — where the same frame complete but weak reads 23-29 at −3 dB and
    13 at −9 dB (five noise realisations each). A truncated frame outscoring every
    real channel is the defect this test was written for, arriving by the other door.

    Where this stops. A cut shallower than the 0.4 s of zeros `RollingDecoder`
    appends as the demodulator's flush pad still fits inside the capture, and
    zero-power symbols are both phase- and tone-perfect — a 4FSK.500.100S cut by
    200 ms reports 99 on either side of this repair. So what moves here is the
    *width* of the exposure: measured by sweeping the cut in 100 ms steps, a
    4FSK.2000.600 was graded at cuts down to 3.7 s of its 5.525 s and now only to
    0.4 s, which is where the single-block frames always sat.

    Neither width reaches the live path. `RollingDecoder` withholds an un-ok frame
    until its own extent has arrived — the header names it, `phy.demodulator
    .frame_span` reads it — so a body the newest window edge cut is never handed up
    to be graded at all. That guard is the frame's own length and this repair does
    not move it.
    """
    payload = b"Q" * F.FRAMES[ftype].net_payload
    full = M.render_frame(ftype, payload=payload, session_id=0x5E)
    assert _decode_one(full).quality > SHIFT_UP_FLOOR

    cut = np.concatenate([np.zeros(1200, dtype="<i2"), full[:len(full) // 2],
                          np.zeros(2400, dtype="<i2")])
    frame = next(f for f in Demodulator().decode(cut) if f.type == ftype)
    # A sub-block that fitted still hands up its bytes, as a carrier that decoded
    # does on the PSK path; what it must not do is grade the frame.
    assert not frame.ok and frame.payload != payload
    assert frame.quality == 0
    # 0 is also a legal grade, and on 2026-08-26 five sightings reading q=0 were
    # carried for days as the demodulator scoring a body it could not follow when
    # nothing had arrived to score. The sentinel says which it is.
    assert frame.truncated and not _decode_one(full).truncated


def test_a_frame_with_no_measurement_leaves_the_last_one_standing():
    """Why the demodulator has to produce a number rather than None: the session's
    reading is deliberately a standing one, so an unmeasured frame reports whatever
    the frame before it did."""
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"mail body")
    ft, payload, sid = _outstanding_data(pair)

    pair.responder.on_receive(ft, payload, sid, True, quality=96)
    assert _reported(_last_ack(pair)) == 96

    # the odd twin, so this is a fresh frame rather than one already delivered
    pair.responder.on_receive(ft ^ 1, payload, sid, False, quality=None)
    assert _last_ack(pair) == 0x1D and _reported(_last_ack(pair)) == 96

    pair.responder.on_receive(ft ^ 1, payload, sid, False, quality=0)
    assert _reported(_last_ack(pair)) == 38, "the floor of the wire code"


# -- the RS floor: the reference's rule, at the reference's numbers -------------

@pytest.mark.parametrize("errors,blocks,r,ok,before,after", [
    # (errors // blocks) < (r // 4), exactly as the C computes it.
    (15, 2, 32, True, 40, 80),      # 7 < 8: the frame was read well, RS barely worked
    (16, 2, 32, True, 40, 40),      # 8 < 8 is false: RS worked hard, the score stands
    (12, 1, 50, True, 40, 40),      # 12 < 12 is false; a true quotient says 12 < 12.5
    (0, 1, 32, False, 40, 40),      # a frame that did not decode is never lifted
    (0, 1, 32, True, 90, 90),       # and the floor never pulls a good decode down
])
def test_the_rs_floor_applies_where_the_reference_applies_it(
        errors, blocks, r, ok, before, after):
    """``SoundInput.c``, immediately before ``returnframe``::

        if (blnDecodeOK && (totalRSErrors / intNumCar) < (intRSLen / 4)
            && intLastRcvdFrameQuality < 80)
                intLastRcvdFrameQuality = 80;

    Both quotients are integer there, and the ``r`` = 50 row is where that shows.
    """
    frame = DecodedFrame(type=0x4C, session_id=0x5E, ok=ok, quality=before)
    _apply_rs_floor(frame, errors, blocks, r)
    assert frame.quality == after


@pytest.mark.parametrize("ftype,blocks,r,bar", [
    (0x7C, 1, 50, 11),     # single block; this one always matched
    (0x7A, 1, 150, 36),    # three sub-blocks, but one carrier and one RS length
    (0x7B, 1, 150, 36),
])
def test_the_fsk_floor_reads_the_whole_frame_not_a_sub_block(
        ftype, blocks, r, bar, monkeypatch):
    """`intNumCar` and `intRSLen`, not the sub-block count and its share of the
    parity.

    The three parts of a 600-baud frame are the reference's own construction
    (SoundInput.c case 0x7A: ``intPartRSLen = intRSLen / 3``), but they are a
    detail of the RS decode and nothing else. ``totalRSErrors`` accumulates across
    all three, and the floor compares it against the *frame's* ``intNumCar`` = 1
    and ``intRSLen`` = 150 — a bar of 36 errors. Passing the part count and the
    part's parity computed 35 instead, so a frame RS-corrected at exactly the
    reference's limit reported its scruffy constellation where the reference
    reports 80. 0x7C is single-block and so was right by coincidence.
    """
    seen = []
    monkeypatch.setattr("hfmodem.besra.phy.demodulator._apply_rs_floor",
                        lambda base, errors, b, rr: seen.append((b, rr)))
    _decode_one(M.render_frame(ftype, payload=b"Q" * F.FRAMES[ftype].net_payload,
                               session_id=0x5E))
    assert seen == [(blocks, r)]

    frame = DecodedFrame(type=ftype, session_id=0x5E, ok=True, quality=40)
    _apply_rs_floor(frame, bar, blocks, r)
    assert frame.quality == Q.RS_CLEAN_FLOOR, "the reference lifts this one"
    frame.quality = 40
    _apply_rs_floor(frame, bar + 1, blocks, r)
    assert frame.quality == 40, "and not the one past its bar"


@gt.requires_reference
def test_the_fsk_floor_operands_are_read_off_the_reference():
    """``ARDOPC.c``'s own ``intNumCar`` and ``intRSLen`` for the 600-baud frames,
    read in place — the arbiter for the parametrisation above."""
    src = gt.ardopcf_dir() / "src" / "common"
    if not src.is_dir():
        pytest.skip("ardopcf source tree absent")

    text = (src / "ARDOPC.c").read_text()
    for label, carriers, rslen in (("0x7a", 1, 150), ("0x7c", 1, 50)):
        arm = re.search(rf"case {label}:.*?break;", text, re.S | re.I)
        assert arm, f"case {label} is no longer where this test reads it"
        assert int(re.search(r"\*intNumCar = (\d+)", arm.group()).group(1)) == carriers
        assert int(re.search(r"\*intRSLen = (\d+)", arm.group()).group(1)) == rslen
        fd = F.FRAMES[int(label, 16)]
        assert (fd.carriers, fd.r) == (carriers, rslen)


def test_the_rs_floor_is_what_lifts_a_marginal_decode(monkeypatch):
    """And it does its work on real audio, not only in the arithmetic above: the
    same noisy frame reports 80 with the floor and its own scruffier constellation
    without it. Without this the headline marginal-decode test asserts one constant
    against another — both were 80, both written in the same commit."""
    clean = M.render_frame(0x4C, payload=b"QUALITY" * 4, session_id=0x5E)
    noisy = add_awgn(clean, -4, seed=0)

    floored = _decode_one(noisy)
    monkeypatch.setattr(Q, "RS_CLEAN_FLOOR", 0)
    raw = _decode_one(noisy)

    assert floored.ok and raw.ok and floored.payload == raw.payload
    assert floored.quality == Q.RS_CLEAN_FLOOR + 80    # the patched constant is 0
    assert raw.quality < 80, f"the constellation already read {raw.quality}"


@gt.requires_reference
def test_the_rs_floor_and_its_bars_are_read_off_the_reference():
    """The arbiter that owes nothing to anything written here. ``RS_CLEAN_FLOOR``'s
    justification used to be that it was "the lowest shift-up bar"; it is not the
    lowest of anything. ``GetShiftUpThresholds`` holds five tables and 80 clears six
    of their twenty-three bars outright, so an RS-clean but scruffy decode reports 80
    and invites a shift up wherever the bar is 60, 75, 76 or 79. That is the
    reference's behaviour, kept on purpose — the number exists so a gateway's ladder
    meets the peer it was tuned against — but it is not what the note claimed."""
    src = gt.ardopcf_dir() / "src" / "common"
    if not src.is_dir():
        pytest.skip("ardopcf source tree absent")

    floor = re.search(r"totalRSErrors / intNumCar\) < \(intRSLen / (\d+)\)"
                      r"\s*&&\s*intLastRcvdFrameQuality < (\d+)",
                      (src / "SoundInput.c").read_text())
    assert floor, "the RS-floor rule is no longer where this test reads it"
    assert int(floor.group(1)) == Q.RS_CLEAN_DIVISOR
    assert int(floor.group(2)) == Q.RS_CLEAN_FLOOR

    bars = [int(n)
            for table in re.findall(r"static UCHAR byt\w+\[\] = \{([^}]*)\}",
                                    (src / "ARQ.c").read_text())
            for n in table.split(",") if int(n)]
    assert len(bars) == 23, f"the threshold tables have changed: {bars}"
    assert sum(b < Q.RS_CLEAN_FLOOR for b in bars) == 6
    assert Q.RS_CLEAN_FLOOR != min(bars)


# -- the session hands it to the right frames --------------------------------

def _outstanding_data(pair: Pair):
    """The data frame the caller has queued but not yet delivered, taken off the
    relay so the test can deliver it with a decode quality of its choosing."""
    for _dest, ft, payload, sid in list(pair.relay.q):
        if S._is_data(ft):
            pair.relay.q.clear()
            pair.relay.log.clear()
            return ft, payload, sid
    raise AssertionError("no data frame queued")


def _last_ack(pair: Pair) -> int:
    return [ft for who, ft in pair.relay.log
            if who == "K7ABC" and (S._is_dataack(ft) or S._is_datanak(ft))][-1]


def test_data_ack_carries_the_measured_quality():
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"mail body")
    ft, payload, sid = _outstanding_data(pair)

    pair.responder.on_receive(ft, payload, sid, True, quality=62)

    ack = _last_ack(pair)
    assert S._is_dataack(ack)
    assert _reported(ack) == 62, f"reported {_reported(ack)} for a decode measured 62"


def test_data_nak_carries_the_measured_quality():
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"mail body")
    ft, payload, sid = _outstanding_data(pair)

    pair.responder.on_receive(ft, payload, sid, False, quality=44)

    nak = _last_ack(pair)
    assert S._is_datanak(nak)
    assert _reported(nak) == 44


def test_a_strangers_unconnected_frame_does_not_set_what_we_report():
    """The one door the session filter leaves open, and what walks through it.

    ConReq/Ping/ID travel with the forced 0xFF wire session and are addressed by
    callsign, so they pass the session filter by design — a listening station has
    to hear a call meant for it. What they are not is a grade of this session's
    path: they are unconnected traffic, from a stranger as readily as from the
    peer, and a 12-byte 4FSK ID is a different frame from the data whatever it
    reads. Measured on ``logs/onair/20260805T235717Z-besra-7102000.wav``: the ID
    frames the demodulator reads off it score 76 (t = 46.9) and 90 (t = 169.3),
    while the session's own frames read 81 at the ConAck and 19-54 across the data
    frames it was failing to decode. Answering 90 for a path whose data is arriving at 19 is
    the gearshift-into-the-noise this whole file exists to stop.
    """
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"mail body")
    ft, payload, sid = _outstanding_data(pair)
    pair.responder.on_receive(ft, payload, sid, True, quality=66)
    assert _reported(_last_ack(pair)) == 66

    stranger = (callsign.pack_callsign("K7XYZ") + callsign.pack_callsign("W1AW"))
    pair.responder.on_receive(S.CONREQ_MAX[500], stranger, 0xFF, True, quality=100)
    pair.responder.on_receive(_IDFRAME, stranger, 0xFF, True, quality=40)

    # The peer repeats the over it never heard acknowledged; the re-ACK reports
    # the last measurement of the peer's OWN path, which is still 66.
    pair.responder.on_receive(ft, payload, sid, True)
    assert _reported(_last_ack(pair)) == 66, (
        f"a stranger set the quality we report our peer: "
        f"{_reported(_last_ack(pair))} for a path measured 66")


def test_turnover_ack_is_not_a_decode_grade():
    """A BREAK carries no body, so acknowledging it grades nothing. The reference
    passes a literal 100 at this site, and the peer's averager never sees it — it
    averages only ACKs of data frames (``blnLastFrameSentData``).

    Twice, because a control frame reaching an IRS is answered on its repeat."""
    pair = Pair()
    pair.connect()
    pair.relay.q.clear()
    pair.relay.log.clear()

    for _ in range(2):
        pair.responder.on_receive(S.BREAK, b"", pair.responder._session, True, quality=41)

    assert _reported(_last_ack(pair)) == _reported(S._dataack_for(100))


# -- the seam that carries it, end to end ------------------------------------

class _Recorder(ModemObserver):
    """The host events a modem pair emits; only the delivered payload matters here."""

    def __init__(self) -> None:
        self.rx = bytearray()

    def modem_newstate(self, state): pass
    def modem_connected(self, remote, bw): pass
    def modem_disconnected(self): pass
    def modem_ptt(self, on): pass
    def modem_buffer(self, n): pass
    def modem_data_received(self, kind, blob): self.rx += blob
    def modem_status(self, text): pass


def test_the_ack_on_the_wire_carries_what_the_demodulator_measured():
    """Demodulator → modem → session → the audio that goes back out.

    The two halves above are each tested against their own definition, and both
    stay green with the seam between them cut: the demodulator measures, the
    session forwards what a *test* hands it, and nothing joins them. Dropping
    ``frame.quality`` from ``BesraModem.receive_frames``'s call to
    ``on_receive`` — one argument — restores the on-air defect with every other
    besra test passing, because the session then reports the unmeasured constant
    and no test drives that call.

    So this one drives it: two real modems over a virtual air at -4 dB SNR, and
    the ACK is read back off the responder's own transmitted audio rather than
    off its call log. The channel is chosen to scuff the constellation without
    costing the payload — the frame decodes, its bytes arrive, and it measures 87
    rather than the 100 an unmeasured session claims. One frame's worth of payload,
    so there is one grade and one ACK to hold against each other.
    """
    air = StepAir(channel=lambda s: add_awgn(s, snr_db=-4, seed=3))
    caller = BesraModem(bandwidth=500)
    caller.set_mycall("W9SSJ")
    responder = BesraModem(bandwidth=500)
    responder.set_mycall("K7ABC")
    responder.set_listen(True)
    host = _Recorder()
    caller.start(_Recorder())
    responder.start(host)
    air.join(caller)
    air.join(responder)

    measured: list[int] = []
    deliver = responder.receive_frames

    def watch(frames):
        measured.extend(f.quality for f in frames if S._is_data(f.type) and f.ok)
        deliver(frames)

    responder.receive_frames = watch

    on_air: list[np.ndarray] = []
    transmit = responder.audio_out

    def record(samples):
        on_air.append(np.asarray(samples))
        transmit(samples)

    responder.audio_out = record

    caller.connect("K7ABC")
    air.run(max_time=90)
    assert caller.connected and responder.connected, "the pair never connected"
    on_air.clear()
    caller.transmit(b"ardop, measured")
    air.run(max_time=90)

    assert bytes(host.rx) == b"ardop, measured", (
        f"the channel cost the payload: {bytes(host.rx)!r}")
    assert measured == [87], f"the data frame measured {measured}, expected [87]"
    acks = [f.type for burst in on_air for f in Demodulator().decode(
        np.concatenate([np.zeros(2400, dtype="<i2"), burst,
                        np.zeros(4800, dtype="<i2")]))
        if S._is_dataack(f.type)]
    assert acks, "the responder put no DATAACK on the air"
    assert [_reported(a) for a in acks] == [_reported(S._dataack_for(measured[-1]))], (
        f"acked {[_reported(a) for a in acks]} for a decode measured {measured[-1]}")


def test_the_peers_grade_of_our_transmission_reaches_the_receive_log(caplog):
    """The same number in the other direction, and the one nobody could see. A
    DATAACK's own type is the peer's grade of the frame it answers — the whole of
    ARDOP's transmit-rate feedback — and the receive log named all 32 codes
    `DATAACK`. On 2026-08-23 WW2MI graded 32 of our frames at a median 96 and not
    one line said so."""
    modem = BesraModem(bandwidth=200)
    modem.set_mycall("W9SSJ")
    modem.start(_Recorder())
    with caplog.at_level(logging.INFO, logger="hfmodem.besra.arq.modem"):
        modem.receive_frames([DecodedFrame(S._dataack_for(96), 0xF3, True,
                                           name="DATAACK"),
                              DecodedFrame(S.IDLE, 0xF3, True, name="IDLE")])

    lines = [ln for ln in caplog.text.splitlines() if "RX " in ln]
    assert "theirq=96" in lines[0], lines[0]
    assert "theirq" not in lines[1], "a frame that grades nothing must not claim to"


def test_a_zero_that_is_a_sentinel_says_so_in_the_receive_log(caplog):
    """`q=0` reads as a grade, and the log had no way to say when it is not one —
    which is how the five `16QAM.500.100.O ... q=0` sightings of 2026-08-26 were
    read as a 16QAM body decoding to nothing."""
    modem = BesraModem(bandwidth=200)
    modem.set_mycall("W9SSJ")
    modem.start(_Recorder())
    with caplog.at_level(logging.INFO, logger="hfmodem.besra.arq.modem"):
        modem.receive_frames([
            DecodedFrame(0x40, 0xF3, False, quality=0, truncated=True, name="4PSK.200.100.E"),
            DecodedFrame(0x40, 0xF3, False, quality=0, name="4PSK.200.100.E")])

    lines = [ln for ln in caplog.text.splitlines() if "RX " in ln]
    assert lines[0].endswith("q=0 TRUNCATED"), lines[0]
    assert lines[1].endswith("q=0"), lines[1]



# -- the wire codes stay legal ------------------------------------------------

@gt.requires_reference
def test_the_reference_names_the_same_code_for_the_qualities_it_ships():
    """Our quality→type map against the reference's own artifacts, not against our
    reading of its formula.

    Asked on 2026-08-15 after W4RJG (7101.9, 21:31): three DATANAKs carrying
    q=55/60/49 and then the gateway sending 16QAM where a NAK is meant to gear it
    down — so whether besra's NAK could be arriving as an ACK, or as a quality high
    enough to justify going faster. It cannot. The reference ships one ground-truth
    render per (type, quality) it names, and those two names are the whole
    question: DataNAK at q60 *is* 0x0B and DataACK at q80 *is* 0xF5, which is what
    `_datanak_for`/`_dataack_for` compute. `test_modulator` then holds the renders
    themselves to ≤1 LSB of those files, so the code and the waveform carrying it
    are both the reference's.

    The three that went out were 0x08/0x0B/0x05, read back by ardopcf as Q=54/60/48
    (`DecodeACKNAK`, SoundInput.c) — under every rung of every `GetShiftUpThresholds`
    table, and the second consecutive one trips `Gearshift_9`'s shift *down*.
    """
    manifest = gt.txframe_manifest()
    assert manifest["txframe_DataNAK-q60.wav"]["type"] == S._datanak_for(60)
    assert manifest["txframe_DataACK-q80.wav"]["type"] == S._dataack_for(80)


@pytest.mark.parametrize("q", [0, 38, 39, 44, 62, 66, 69, 80, 99, 100, 255])
def test_every_quality_maps_into_a_real_frame_type(q):
    for frame_type in (S._dataack_for(q), S._datanak_for(q)):
        assert frame_type in F.FRAMES
    assert S.DATAACK_MIN <= S._dataack_for(q) <= S.DATAACK_MAX
    assert S.DATANAK_MIN <= S._datanak_for(q) <= S.DATANAK_MAX
