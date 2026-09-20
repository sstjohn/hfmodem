# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 response positions 14..17 select setup speed levels 1..4."""
from __future__ import annotations

import collections
import itertools

import numpy as np
import pytest

from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara.vara_arq import (
    _ANSWER_LATTICE,
    _I_CONNECTED,
    _I_CR_SENT,
    _I_LINKSETUP_SENT,
    VaraState,
    VaraStationHandshake,
)
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _IO, _load_48k

RESP = VF.CONNECT_RESPONSE
MYCALL = "W9SSJ"


def _awaiting(called: str) -> VaraStationHandshake:
    hs = VaraStationHandshake([MYCALL], _IO())
    hs.role, hs.called, hs.caller = "initiator", called, MYCALL
    hs.state, hs.step = VaraState.CONNECTING, _I_CR_SENT
    return hs


def _drive(called: str, path) -> VaraStationHandshake:
    hs = _awaiting(called)
    x = _load_48k(path)
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])
    return hs


# --------------------------------------------------------------------------- #
# The lattice.
@pytest.mark.parametrize("kind", [RESP, VF.CONNECT_RESPONSE_500, VF.SESSION_CONFIRM,
                                  VF.SESSION_TURN_RELEASE, VF.CR])
def test_every_pre_advance_is_a_position_on_its_familys_lattice(kind):
    """One payload is ``2 * N`` draws, and every descriptor sits a whole number of
    them past the start. A pre-advance off the lattice would be a frame nothing
    else in the family could be counted from."""
    assert (kind.preadv - 1) % VF.lattice_step(kind) == 0


@pytest.mark.parametrize("position", range(0, 20))
@pytest.mark.parametrize("call", ["W1AW", "KB3AC-10", "N5WAJ", "NS0A", "VE7QRP"])
def test_the_lattice_reads_back_the_position_a_payload_was_drawn_at(call, position):
    from dataclasses import replace

    kind = replace(RESP, preadv=1 + VF.lattice_step(RESP) * position)
    assert VF.payload_position(VF.payload_bins(call, kind), call, RESP) == position


def test_a_payload_of_another_callsigns_stream_has_no_position_here():
    """The rejection the reading rests on: the position exists because the
    callsign's own seed puts it there, so a burst keyed to somebody else has
    none."""
    tones = VF.payload_bins("KB3AC-10", RESP)
    assert VF.payload_position(tones, "KB3AC-10", RESP) == 17
    for other in ("N5WAJ", "NS0A", "KC9GHZ", "W9SSJ"):
        assert VF.payload_position(tones, other, RESP) is None


def test_the_connect_response_is_frame_seventeen_and_this_route_leaves_it_alone():
    """What the accept path regenerates, said as a frame — so the route that
    reports an answer and the route that brings a link up cannot claim the same
    burst."""
    assert VF.payload_position(VF.payload_bins("NS0A", RESP), "NS0A", RESP) == 17
    assert 17 not in _ANSWER_LATTICE
    assert max(_ANSWER_LATTICE) == 16


# --------------------------------------------------------------------------- #
# The four attempts.
@pytest.mark.parametrize("path,called,count,positions", corpora.ONAIR_PEER_ANSWERS,
                         ids=lambda v: getattr(v, "stem", ""))
@corpora.requires_onair_peer_answers
def test_an_attempt_reported_unanswered_was_answered(path, called, count, positions):
    """Driven over the live path exactly as the radio drives it."""
    hs = _drive(called, path)
    assert len(hs.answers) == count, [(a.at, a.position) for a in hs.answers]
    assert collections.Counter(a.position for a in hs.answers) == positions
    assert all(a.shift == 0 for a in hs.answers)
    assert all(a.tones >= 10 for a in hs.answers)


@corpora.requires_onair_peer_answers
def test_every_answer_is_the_dialled_stations_own_payload():
    """The tones the receiver read, put back through the generator."""
    for path, called, _count, _positions in corpora.ONAIR_PEER_ANSWERS:
        for a in _drive(called, path).answers:
            assert VF.payload_position(a.payload, called, RESP) == a.position, (
                f"{path.name} at {a.at:.2f}s no longer belongs to {called}")


def _reached(state: int, span: int = 1200) -> list[tuple[int, int]]:
    """Every ``(seed, advance)`` a payload state can be reached from, over the whole
    callsign space: 2**15 seeds by every pre-advance the family can carry."""
    cur = np.array([VF._start(s) for s in range(1 << 15)], dtype=np.int64)
    out: list[tuple[int, int]] = []
    for n in range(span):
        out += [(int(i), n) for i in np.flatnonzero(cur == state)]
        cur = (cur * VF._LCG_MULT + VF._LCG_ADD) & VF._LCG_MASK
    return out


@corpora.requires_onair_peer_answers
def test_the_only_callsign_these_answers_can_belong_to_is_the_one_dialled():
    """The reading against the whole callsign space rather than a panel, which is
    what a panel search cannot do and what said "nobody" for two years of tapes.

    ``G`` is nine bits and its value one below ``mult`` is unreachable for each of
    these, so the carry is nought and the CRC is one exact value: the dialled
    station's own.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    reachable_g = {VF._g_hash("".join(t))
                   for t in itertools.product(alphabet, repeat=3)}
    for path, called, _count, _positions in corpora.ONAIR_PEER_ANSWERS:
        a = _drive(called, path).answers[0]
        states = VF.payload_states(a.payload, RESP)
        assert len(states) == 1, states
        crc = crc16_genibus(called.encode())
        live = [(seed, n - 1 - VF.lattice_step(RESP) * a.position)
                for seed, n in _reached(states[0])]
        live = [(seed, mult) for seed, mult in live if mult in reachable_g
                or mult - 1 in reachable_g or mult - 2 in reachable_g]
        assert live == [((crc + RESP.seed_off) & 0x7FFF,
                         VF._g_hash(called) + ((crc + 50) >> 15))], (
            f"{path.name}: {live}, and {called} is "
            f"{((crc + RESP.seed_off) & 0x7FFF, VF._g_hash(called))}")


@corpora.requires_onair_peer_answers
def test_the_two_kb3ac_bursts_are_that_gateways_own_frames():
    """What used to say the two arms were "one generator stream 60 draws apart"
    without being able to say whose. They are frames 16 and 14 of KB3AC-10's, and
    60 draws is the two frames between them."""
    a, b = (state for *_, state in corpora.ONAIR_UNATTRIBUTED_ANSWERS)
    assert VF._lcg_advance(b, 60) == a
    for state, position in ((a, 16), (b, 14)):
        found = next(p for p in range(64)
                     if VF._lcg_advance(
                         VF._start((crc16_genibus(b"KB3AC-10") + RESP.seed_off) & 0x7FFF),
                         VF._g_hash("KB3AC-10")
                         + ((crc16_genibus(b"KB3AC-10") + 50) >> 15)
                         + 1 + VF.lattice_step(RESP) * p) == state)
        assert found == position


@corpora.requires_onair_peer_answers
def test_frame_seventeen_still_brings_the_link_up():
    """The 2026-08-26 01:17z arm answers three times off the connect-response's
    frame and then keys the connect-response itself. Reading the three must not
    cost the fourth."""
    path, called, count, _positions = corpora.ONAIR_PEER_ANSWERS[0]
    hs = _drive(called, path)
    assert any("tx link-setup" in line for line in hs.io.log_lines)
    # KB3AC-10 keys its turn-request 5.6 s behind that link-setup, six times over
    # the arm, and the stream route takes the first of them  [_stream_connect_ask].
    assert hs.step in (_I_LINKSETUP_SENT, _I_CONNECTED)
    assert hs.answers


@corpora.requires_onair_peer_answers
def test_lower_speed_offer_starts_setup_but_needs_confirmation():
    """Lower-speed offers send setup; they alone cannot confirm a connection."""
    for path, called, _count, _positions in corpora.ONAIR_PEER_ANSWERS[1:]:
        hs = _drive(called, path)
        assert hs.answers
        assert hs.step == _I_LINKSETUP_SENT and hs.state is VaraState.CONNECTING
        assert any("link-setup" in line for line in hs.io.log_lines)


@corpora.requires_onair_peer_answers
def test_lower_offers_are_not_reported_as_rejections():
    path, called, _count, _positions = corpora.ONAIR_PEER_ANSWERS[1]
    hs = _drive(called, path)
    assert hs.answers and not hs.answer_retry
    assert any("connect-response" in line and "level" in line for line in hs.io.log_lines)
    assert not any("could not read" in line for line in hs.io.log_lines)


@corpora.requires_onair_peer_answers
def test_the_arms_of_that_slot_nobody_answered_report_nothing():
    """Two more calls from the same slot on the same instrument, one on each band.
    An absence has to read as an absence or the count above says nothing."""
    for path, called in corpora.ONAIR_UNANSWERED_ARMS:
        hs = _drive(called, path)
        assert hs.answers == [], [(a.at, a.position) for a in hs.answers]


@corpora.requires_regress_fixtures
@corpora.requires_clear_channel
@corpora.requires_onair_silent_calls
def test_the_negative_corpus_holds_no_answer_at_any_position():
    """The floor, over the population the rest of this package is measured on: the
    31 shared regression fixtures, the verified clear channel, and the two
    2026-08-06 calls nobody answered.

    Four positions and five shifts is twenty hypotheses an alignment where the
    accept path has one, and the callsign is what holds them all: every tone is
    regenerated for the station dialled, so :data:`_RESP_MIN_HEARD` comparable
    tones swept clean is 35**-8 per hypothesis.
    """
    named = {}
    for path in _negatives():
        hs = _drive("KC9GHZ", path)
        if hs.answers:
            named[path.name] = [(a.at, a.position, a.tones) for a in hs.answers]
    assert not named, named


def _negatives():
    out = [corpora.CLEAR_CHANNEL, *corpora.ONAIR_SILENT_CALLS]
    if corpora.REGRESS_FIXTURES.is_dir():
        out += sorted(corpora.REGRESS_FIXTURES.glob("*.wav"))
    return [p for p in out if p.exists()]
