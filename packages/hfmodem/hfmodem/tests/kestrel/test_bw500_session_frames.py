# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The post-connect session frames, against the carriers a real VARA emitted at BW500.

Ground truth: four stock VARA HF 4.9.0 BW500 sessions with two callsign pairs —
the two 2026-07-13 loopbacks (`AAAA1` → `BBBB2`), the answered pair of 2026-08-30
and a 2026-09-02 bench run against a stock responder (both `W9SSJ` → `W1AW`), every
keying attributed by the PTT ledger of the one-way cable that holds it. The
carriers below are what ``vara_mfsk.demod_tones`` reads off those recordings over
an offset sweep — the modem's own symbols, not a round trip through kestrel's
generator.

**A session frame's ``(SEED_OFF, PREADV)`` pair is the state and its alphabet is
the bandwidth's.** Every frame here regenerates from the descriptor the module
already holds with :data:`vara_frames.BW500_TONES` substituted and nothing else
changed — same preamble, same ``keyed_by``, same symbol grid. Twelve kinds, 41
keyings, 1233 of 1233 carriers whole-burst. Repeated keyings of one kind read
identically, so one row stands for all of them.

The BW2300 alphabet is the control on the same audio and is exercised below: it
takes a burst's shared fixed preamble and then nothing that is not chance.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from hfmodem.kestrel.vara import vara_frames as VF

#: (caller, called) of each link the rows come from.
_W = ("W9SSJ", "W1AW")
_A = ("AAAA1", "BBBB2")

#: kind -> (link, the payload carriers that link's station emitted). The callsign
#: the row is scored against is the link member ``kind.keyed_by`` names, which is
#: the caller for the two turn-request frames and the called station for the rest.
_BENCH = {
    VF.CR500: (_W, [60, 72, 58, 58, 52, 68, 58, 72, 66, 52, 52, 52, 68, 68, 52, 60,
                    64, 50, 72, 62, 52, 64, 70, 68, 72, 54, 58, 76, 72, 58, 76]),
    VF.CONNECT_RESPONSE_500: (_W, [54, 66, 74, 52, 68, 54, 76, 68, 62, 58, 54, 64,
                                   58, 56, 68]),
    VF.SESSION_CONFIRM: (_A, [74, 66, 52, 66, 70, 76, 66, 58, 62, 72, 50, 66, 66,
                              70, 64]),
    VF.SESSION_KEEPALIVE_A: (_A, [76, 60, 52, 66, 74, 76, 58, 58, 68, 76, 70, 54,
                                  56, 50, 60, 58, 76, 66, 64, 58, 64, 58, 58, 56,
                                  56, 54, 58, 60, 58, 58, 56]),
    VF.SESSION_KEEPALIVE_B: (_A, [62, 50, 52, 60, 50, 50, 68, 56, 66, 52, 54, 76,
                                  74, 56, 50, 60, 70, 70, 60, 66, 52, 50, 58, 54,
                                  54, 56, 64, 68, 68, 54, 62]),
    VF.SESSION_TURN_REQUEST: (_A, [56, 50, 76, 64, 72, 58, 68, 62, 72, 74, 76, 60,
                                   76, 72, 50, 66, 72, 70, 56, 68, 68, 58, 68, 54,
                                   74, 74, 52, 66, 76, 58, 68]),
    VF.SESSION_TURN_REQUEST_RESPONDER: (_W, [68, 52, 72, 76, 62, 58, 66, 58, 74,
                                             60, 64, 62, 60, 72, 74, 52, 66, 66,
                                             64, 68, 74, 52, 56, 52, 56, 56, 56,
                                             60, 56, 50, 52]),
    VF.SESSION_DRAINED: (_A, [64, 68, 70, 50, 72, 52, 70, 56, 54, 62, 68, 76, 66,
                              70, 62, 52, 70, 70, 74, 64, 70, 66, 56, 70, 64, 62,
                              68, 64, 58, 58, 62]),
    VF.SESSION_DRAINED_RESPONDER: (_W, [56, 76, 52, 64, 50, 68, 74, 56, 70, 54, 74,
                                        50, 52, 52, 58, 58, 56, 74, 72, 60, 76, 72,
                                        62, 58, 52, 76, 68, 56, 60, 58, 60]),
    VF.SESSION_IDLE_RESPONSE: (_W, [56, 68, 54, 52, 62, 76, 62, 76, 56, 56, 60, 72,
                                    68, 74, 50, 64, 64, 50, 76, 76, 68, 64, 58, 62,
                                    68, 56, 68, 66, 62, 74, 52]),
    VF.SESSION_TURN_RELEASE: (_A, [66, 70, 76, 72, 58, 68, 52, 66, 58, 50, 54, 66,
                                   58, 50, 62]),
    VF.SESSION_DISCONNECT_FINAL: (_A, [72, 74, 76, 60, 60, 50, 60, 58, 56, 76, 52,
                                       68, 54, 58, 52]),
}

_IDS = {k: k.name for k in _BENCH}


def _at_bw500(kind: VF.BurstKind) -> VF.BurstKind:
    """``kind`` as a station running at BW500 keys and reads it."""
    return kind if kind.tones is VF.BW500_TONES else replace(
        kind, tones=VF.BW500_TONES)


def _callsign(kind, link):
    caller, called = link
    return caller if kind.keyed_by == "caller" else called


@pytest.mark.parametrize("kind", _BENCH, ids=_IDS.get)
def test_the_descriptor_reproduces_the_modems_own_carriers(kind):
    link, emitted = _BENCH[kind]
    assert VF.payload_bins(_callsign(kind, link), _at_bw500(kind)) == emitted


@pytest.mark.parametrize("kind", _BENCH, ids=_IDS.get)
def test_the_wide_alphabet_reads_the_burst_as_chance(kind):
    """The control. A station reading these frames on the BW2300 alphabet is deaf
    to every one of them: no BW500 burst can emit an odd carrier, and the wide
    descriptor takes at most three of the fifteen to thirty-one that are there."""
    link, emitted = _BENCH[kind]
    wide = replace(kind, tones=VF.BW2300_TONES)
    got = VF.payload_bins(_callsign(kind, link), wide)
    assert sum(1 for a, b in zip(got, emitted) if a == b) <= 3


@pytest.mark.parametrize("kind", _BENCH, ids=_IDS.get)
def test_every_carrier_is_on_the_fourteen_even_bins(kind):
    assert set(_BENCH[kind][1]) <= VF.BW500_TONES.carriers


@pytest.mark.parametrize("kind", _BENCH, ids=_IDS.get)
def test_only_the_alphabet_changes(kind):
    """The whole hypothesis, stated as an invariant: same preamble, same payload
    count, same state, same key. Only the two handshake bursts carry a genuinely
    different descriptor at BW500, and the module already holds both."""
    at500 = _at_bw500(kind)
    assert (at500.preamble, at500.n_payload, at500.seed_off, at500.preadv,
            at500.par0, at500.keyed_by) == (
        kind.preamble, kind.n_payload, kind.seed_off, kind.preadv,
        kind.par0, kind.keyed_by)
    assert at500.tones is VF.BW500_TONES


def test_the_turn_requests_are_the_two_keyed_to_the_caller():
    """Which end a frame speaks for is not a bandwidth question, and getting it
    wrong is what leaves a station scoring a peer's ask against its own call."""
    caller = {k for k in _BENCH if k.keyed_by == "caller"}
    assert caller == {VF.SESSION_TURN_REQUEST,
                      VF.SESSION_TURN_REQUEST_RESPONDER}


def test_the_bench_agrees_with_the_handshake_corpus():
    """`test_bw500_handshake` pins the same two bursts for W1AW off a different
    day's recordings; these are read off the 2026-08-30 and 2026-09-02 sessions
    and are the same carriers."""
    assert VF.payload_bins("W1AW", VF.CR500) == _BENCH[VF.CR500][1]
    assert (VF.payload_bins("W1AW", VF.CONNECT_RESPONSE_500)
            == _BENCH[VF.CONNECT_RESPONSE_500][1])
