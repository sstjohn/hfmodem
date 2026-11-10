# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The handshake recogniser against real off-air gateway answers.

Ground truth: recordings of sessions where VARA itself connected to a named
gateway through the rig, so the called callsign is known. Measured with
``lock_preamble`` locating each burst:

    KC9GHZ  connect-request   29/31 = 0.935
    NS0A    connect-response  15/15 = 1.000
    wrong callsigns                 <= 0.10

The off-air wrong-callsign figure is five hand-picked calls, which is too small a
sample to place the acceptance cut against: swept over every pair of a fixed
270-callsign grid and every burst kind, the generator's own worst wrong score is
5/12 = 0.417, on the shortest burst. That is the number the cut has to clear, and
:func:`test_the_threshold_separates_the_measured_populations` clears it by the
width of the gap rather than by a whisker — a recogniser cut low enough to admit
wrong-callsign bursts passed every other test in this suite.

An earlier version of this file hardcoded 11/15 for the NS0A response and
asserted the acceptance threshold lay in [0.10, 0.73]. Both came from its own
hand-rolled timing search, which kept the *earliest* offset attaining maximal
preamble match rather than the best — and a real gateway answer loses no tones at
all once it is correctly located. That version therefore failed when the locator
was fixed, i.e. it defended the defect. It is replaced by this one, which uses the
library locator so that a regression in either shows up here.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

# (recording, gateway actually called, burst kind, search window, expected match)
_CASES = [
    ("NS0A_2300", "NS0A", VF.CONNECT_RESPONSE, (9.0, 11.5), 15, 15),
    ("KC9GHZ_2300", "KC9GHZ", VF.CR, (4.5, 7.0), 29, 31),
]
_IDS = [c[0] for c in _CASES]


def _window(session: str, span: tuple[float, float]) -> tuple[int, np.ndarray]:
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    from scipy.io import wavfile
    fs, x = wavfile.read(str(path))
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    x = x / (np.abs(x).max() or 1.0)
    return fs, x[int(span[0] * fs):int(span[1] * fs)]


def _located(session, kind, span):
    fs, audio = _window(session, span)
    at = MK.lock_preamble(audio, kind)
    assert at is not None, "lock_preamble found no burst in a window known to hold one"
    n = len(kind.preamble) + kind.n_payload
    return MK.demod_tones(audio[at:], n)


@pytest.mark.parametrize("session,call,kind,span,exp_m,exp_n", _CASES, ids=_IDS)
def test_real_gateway_burst_matches_its_callsign(session, call, kind, span, exp_m, exp_n):
    tones = _located(session, kind, span)
    m, n = VF.payload_match(tones[len(kind.preamble):], call, kind)
    assert n == exp_n
    assert m == exp_m, (
        f"payload match is {m}/{n}, expected {exp_m}/{exp_n}. A drop here usually "
        "means the locator regressed to an off-centre alignment, not that the "
        "recording changed.")
    assert VF.recognize(tones, call, kind), f"genuine {call} burst rejected at {m}/{n}"


@pytest.mark.parametrize("session,call,kind,span,exp_m,exp_n", _CASES, ids=_IDS)
def test_wrong_callsigns_are_rejected(session, call, kind, span, exp_m, exp_n):
    tones = _located(session, kind, span)
    for wrong in ("W9SSJ", "K7ABC", "W2XYZ", "N0CALL", "KO2F"):
        if wrong == call:
            continue
        m, n = VF.payload_match(tones[len(kind.preamble):], wrong, kind)
        assert not VF.recognize(tones, wrong, kind), (
            f"accepted {wrong} for a burst addressed to {call} at {m}/{n}")


_WORST_GENUINE = 29 / 31        # KC9GHZ connect-request, located off air
# Every pair of these, over every burst kind: 217,890 wrong-callsign comparisons,
# and a third of a second, because payload_bins is memoised. The connected-ack is
# not among the kinds: it carries no callsign at all [spec 04 §4.2C], so there is no
# wrong-callsign comparison to make against it.
_WRONG_POP = [f"{p}{d}{a}{b}" for p in "KNW" for d in "0123456789"
              for a in "ABC" for b in "XYZ"]


#: Worst wrong-callsign score over every pair and every burst, by payload
#: alphabet. The two are pinned apart so neither can hide behind the other.
#:
#: BW500 sits higher and has to: its alphabet is 14 carriers where BW2300's is 70,
#: so a wrong callsign collides on 1 tone in 14 rather than 1 in 35 — parity splits
#: BW2300's alphabet into disjoint halves and does not exist at BW500. Measured
#: 2026-08-15 over the same 36,315 pairs, mean wrong matches run 1.17/15 at BW500
#: against 0.45/15 at BW2300. The cut is unchanged and still clears the rule below
#: by a wide margin; what would not clear it is a 15-tone BW500 burst at an accept
#: fraction much under 0.75.
_WORST_WRONG = {"BW2300": 5 / 15, "BW500": 7 / 15}


def test_the_threshold_separates_the_measured_populations():
    """The cut must sit between the two populations, and nearer the genuine one.

    Being inside the gap is not enough: a cut a hair above the wrong population
    accepts the first wrong burst that fades favourably, and every other test in
    this suite still passes. So the cut is required to leave at least as much room
    below the worst genuine answer as above the worst wrong one — no free
    parameter, just the midpoint of the two measured populations.
    """
    measured = {}
    for label, alphabet in (("BW2300", VF.BW2300_TONES), ("BW500", VF.BW500_TONES)):
        measured[label] = max(
            (VF.payload_match(VF.payload_bins(a, k), b, k)[0] / k.n_payload, k.name)
            for k in VF.BURSTS.values() if k.tones is alphabet
            for a, b in itertools.combinations(_WRONG_POP, 2))
    for label, (score, kind) in measured.items():
        assert score == pytest.approx(_WORST_WRONG[label]), (
            f"the {label} wrong-callsign population moved to {score:.4f} ({kind}); "
            "the acceptance cut is placed against it and has to be re-measured")
    worst_wrong = max(s for s, _ in measured.values())
    assert VF._ACCEPT_FRAC <= _WORST_GENUINE, (
        f"_ACCEPT_FRAC {VF._ACCEPT_FRAC} rejects the worst genuine off-air answer "
        f"measured ({_WORST_GENUINE:.3f})")
    assert VF._ACCEPT_FRAC - worst_wrong >= _WORST_GENUINE - VF._ACCEPT_FRAC, (
        f"_ACCEPT_FRAC {VF._ACCEPT_FRAC} sits nearer the wrong-callsign population "
        f"({worst_wrong:.3f}) than the genuine one ({_WORST_GENUINE:.3f}); the cut "
        f"belongs at or above {(worst_wrong + _WORST_GENUINE) / 2:.3f}")
