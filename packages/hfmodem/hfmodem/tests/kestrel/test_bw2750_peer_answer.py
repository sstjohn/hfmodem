# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bandwidth-scoped recognition of lower-speed connect offers and negatives."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _norm, band_limit

MYCALL = "W9SSJ"

#: Positions a BW2750 answer might sit at, minus the offer's own. Every test that
#: searches for one searches for all of them, so a candidate that fires on band
#: noise is caught by whichever row is quietest rather than only by its own.
CANDIDATES = (0, 1, 2)


class IO:
    def __init__(self):
        self.logs: list[str] = []
        self.keyed: list[bool] = []

    def log(self, message):
        self.logs.append(message)

    def key(self, on):
        self.keyed.append(on)

    def tx(self, samples): ...
    def pending(self): ...
    def connected(self, *a): ...


def awaiting(call: str, bw: str = "2750") -> A.VaraStationHandshake:
    hs = A.VaraStationHandshake([MYCALL], IO(), bw=bw)
    hs.role, hs.called, hs.caller = "initiator", call, MYCALL
    hs.state, hs.step = A.VaraState.CONNECTING, A._I_CR_SENT
    return hs


def searching(monkeypatch, *positions: int) -> None:
    """Give the route a candidate set at BW2750 for the length of one test."""
    monkeypatch.setitem(A._ANSWER_LATTICE_BY_BW, "2750", tuple(positions))


def tones_at(call: str, position: int, bw: str = "2750") -> list[int]:
    kind = VF.connect_response(bw)
    return VF.payload_bins(call, replace(
        kind, preadv=1 + VF.lattice_step(kind) * position))


def observe(hs: A.VaraStationHandshake, payload) -> bool:
    """One synthesised burst, every tone delivered clear, straight at the route."""
    n = int(A._PAY_OFF[-1]) + 1
    track = np.zeros((n, 3), dtype=np.int32)
    track[A._PAY_OFF, 0] = payload
    return hs._peer_answer(track, np.full((n, 2), 20.0),
                           np.ones(n, dtype=bool), 0)


def drive(call: str, name: str, window: tuple[float, float] | None = None,
          hs: A.VaraStationHandshake | None = None) -> A.VaraStationHandshake:
    """A slice of a recording through the live stream search, as the radio drives it."""
    hs = hs or awaiting(call)
    x = band_limit(_norm(corpora.c_samples(name, external=True)))
    if window is not None:
        t0, t1 = (int(t * MK.FS) for t in window)
        x = x[t0:t1]
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])
    return hs


def offer_position(bw: str = "2750") -> int:
    kind = VF.connect_response(bw)
    return (kind.preadv - 1) // VF.lattice_step(kind)


# --------------------------------------------------------------------------- #
# The canonical level-4 descriptor has its original full-response fast path.
def test_the_offer_is_position_three_and_never_a_candidate():
    assert offer_position() == 3
    assert offer_position() not in A._ANSWER_LATTICE_BY_BW.get("2750", ())


def test_the_offer_payload_is_left_to_the_accept_route(monkeypatch):
    searching(monkeypatch, *CANDIDATES)
    hs = awaiting("KB5LZK")
    assert not observe(hs, VF.payload_bins("KB5LZK", VF.connect_response("2750")))
    assert not hs.answers


# --------------------------------------------------------------------------- #
# The instrument: a candidate goes in, a report comes out. Which candidate is a
# parameter, because the measurement that would fix one is not settled.
@pytest.mark.parametrize("position", CANDIDATES)
def test_lower_speed_offer_sends_setup_at_selected_level(monkeypatch, position):
    searching(monkeypatch, position)
    hs = awaiting("KB5LZK")
    assert observe(hs, tones_at("KB5LZK", position))
    assert [a.position for a in hs.answers] == [position]
    assert hs.answers[0].tones == 15 and hs.answers[0].shift == 0
    assert hs.state == A.VaraState.CONNECTING and hs.step == A._I_LINKSETUP_SENT
    assert not hs.answer_retry          # no retry cadence is established at BW2750
    assert hs.io.keyed == [True, False]
    assert hs._setup_level == position + 1
    line = hs.io.logs[0]
    assert "KB5LZK" in line and "BW2750" in line and str(position) in line
    assert f"level {position + 1}" in line


@pytest.mark.parametrize("position", CANDIDATES)
@pytest.mark.parametrize("call,bw", [("W1AW", "2750"), ("KB5LZK", "2300"),
                                     ("KB5LZK", "500")])
def test_a_candidate_matches_neither_another_call_nor_another_bandwidth(
        monkeypatch, position, call, bw):
    searching(monkeypatch, position)
    hs = awaiting(call, bw)
    assert not observe(hs, tones_at("KB5LZK", position))
    assert not hs.answers


@pytest.mark.parametrize("position", CANDIDATES)
def test_a_candidate_needs_the_tones_it_asks_for(monkeypatch, position):
    """Half a payload is not an answer, at any candidate."""
    searching(monkeypatch, position)
    hs = awaiting("KB5LZK")
    assert not observe(hs, tones_at("KB5LZK", position)[:7] + [0] * 8)
    assert not hs.answers


# --------------------------------------------------------------------------- #
# Whole 0930/0920 KC9GHZ taps: no below-offer candidates (0, 1, 2).
# The 0930 tape does hold a position-3 offer, checked separately below.
RECORDED_NULL = (
    ("kc9ghz-0930-whole-no-candidates.flac", "KC9GHZ"),
    ("kc9ghz-0920-whole-unanswered.flac", "KC9GHZ"),
)


@pytest.mark.parametrize("name,call", RECORDED_NULL,
                         ids=[r[0] for r in RECORDED_NULL])
def test_no_candidate_fires_on_a_recorded_window(monkeypatch, name, call):
    searching(monkeypatch, *CANDIDATES)
    hs = drive(call, name)
    assert not hs.answers, [(a.at, a.position, a.tones) for a in hs.answers]
    assert not hs.unattributed


def test_a_recorded_offer_still_reaches_the_accept_route(monkeypatch):
    """The 0930 control: KC9GHZ offered the link on
    position 3 and the accept route takes it, candidates in the table or not."""
    searching(monkeypatch, *CANDIDATES)
    hs = drive("KC9GHZ", "kc9ghz-0930-whole-no-candidates.flac", (30.0, 35.0))
    assert hs.step == A._I_LINKSETUP_SENT
    assert not hs.answers
    assert any("connect-response for KC9GHZ" in line for line in hs.io.logs), hs.io.logs
