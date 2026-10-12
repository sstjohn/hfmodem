# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW2300 NAK: a receiver that failed an over asks for it again.

Measured 2026-09-07 off two stock VARA HF 4.9.0 instances on the virtual cables,
noise injected into a receiver's input until it registered a burst and failed the
CRC. The station that failed the over keyed an 8-symbol two-tone burst in the
turnaround, four copies each side, identical across overs, and the sender
answered by dropping a speed level and re-sending — both a forward and a reversed
delivery closed byte-exact. It refutes the note this file's subject used to carry,
that BW2300 keys nothing in that turnaround.

The two NAKs are role-keyed: caller and responder share only the lead pair, the
way a link's two control tails do. The receiver keys its OWN role's burst — the
caller in RESPONDER, the responder in CALLER.

Nothing here opens a device or keys a radio.
"""
from __future__ import annotations

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

from .test_vara_giveup import _connected


def _last_pairs(io) -> list:
    return [tuple(sorted(p)) for p in MK.demod_tone_pairs(io.sent[-1], 8)]


def test_the_caller_keys_its_own_nak_when_an_over_will_not_decode():
    hs, io = _connected()               # initiator == the caller
    assert hs._undecoded_over()
    assert _last_pairs(io) == list(VF.nak(hs.called, "2300")[0])
    assert any("tx NAK" in m for m in io.msgs), io.msgs


def test_the_responder_keys_the_other_nak_of_the_link():
    hs, io = _connected()
    hs.role = "responder"
    assert hs._undecoded_over()
    assert _last_pairs(io) == list(VF.nak(hs.called, "2300")[1])


def test_the_two_naks_share_only_the_lead_pair():
    caller, responder = VF.nak("KC9GHZ", "2300")
    assert caller[0] == responder[0], "the family opens on one lead pair"
    shared = set(caller[1:]) & set(responder[1:])
    assert not shared, f"role-keyed frames share a tail symbol: {shared}"


def test_the_nak_is_keyed_to_the_called_station_whoever_called():
    hs, io = _connected()
    hs.caller = "N0XYZ"
    assert hs._undecoded_over()
    assert _last_pairs(io) == list(VF.nak(hs.called, "2300")[0])


def test_the_link_closes_once_the_overs_will_not_stop_failing():
    """The ask is bounded: after `_OVER_NAK_MAX` turnarounds still failing, the
    link closes rather than key a repeat request into a peer that cannot reach us."""
    hs, io = _connected()
    for _ in range(VA._OVER_NAK_MAX):
        assert hs._undecoded_over()
    assert hs.state is VA.VaraState.CONNECTED
    hs._undecoded_over()
    assert hs.state is VA.VaraState.DISCONNECTED
    assert any("closing the link" in m for m in io.msgs), io.msgs


def _short_kinds():
    return [k for k in vars(VF).values() if isinstance(k, VF.BurstKind)
            and len(k.preamble) + k.n_payload == 16]


def test_kc9ghz_keyed_its_short_over_idle_behind_our_nak():
    """The 16 tones read off logs/onair/20260926T201057Z-W9SSJ-KC9GHZ at
    38.01 s, 3.2 s behind our NAK of its level-6 over: the responder's short
    stream at the over-idle's lattice position, and no other frame's."""
    tape = [62, 36, 93, 88, 95, 52, 87, 52, 41, 76, 83, 60, 29, 44, 87, 34]
    kind = VF.SESSION_RESPONDER_OVER_IDLE_SHORT
    assert VF.handshake_tones("KC9GHZ", kind) == tape
    assert (VF.payload_position(tape[1:], "KC9GHZ", kind)
            == VF.payload_position(VF.handshake_tones("KC9GHZ", VF.SESSION_RESPONDER_OVER_IDLE)[1:],
                                   "KC9GHZ", VF.SESSION_RESPONDER_OVER_IDLE))
    assert not any(VF.recognize(tape, cs, k) for k in _short_kinds()
                   if k is not kind for cs in ("KC9GHZ", "W9SSJ"))


def test_k5fit_keyed_its_short_responder_idle_behind_unread_overs():
    """The 16 tones read twice off logs/onair/20260927T041306Z-W9SSJ-K5FIT, at
    29.43 s and 49.07 s, behind a level-6 and a level-5 over we did not read:
    the responder's short stream at 288/683's lattice position, and no other
    frame's."""
    tape = [62, 68, 76, 76, 76, 60, 50, 62, 66, 56, 58, 64, 58, 52, 70, 70]
    kind = VF.for_bw(VF.SESSION_RESPONDER_IDLE_SHORT, "500")
    long = VF.for_bw(VF.SESSION_RESPONDER_IDLE, "500")
    assert VF.handshake_tones("K5FIT", kind) == tape
    assert (VF.payload_position(tape[1:], "K5FIT", kind)
            == VF.payload_position(VF.handshake_tones("K5FIT", long)[1:], "K5FIT", long))
    assert not any(VF.recognize(tape, cs, VF.for_bw(k, "500")) for k in _short_kinds()
                   if k is not VF.SESSION_RESPONDER_IDLE_SHORT for cs in ("K5FIT", "W9SSJ"))


def test_the_callers_short_answer_queries_are_the_long_ones_positions():
    """61/331 and 61/361 sit where 60/683 and 60/745 do, one per query phase."""
    for short, long in zip(VF.SESSION_ANSWER_QUERIES_SHORT,
                           VF.SESSION_INTERMEDIATE_ANSWER_QUERIES):
        assert (VF.payload_position(VF.handshake_tones("K5FIT", short)[1:], "K5FIT", short)
                == VF.payload_position(VF.handshake_tones("K5FIT", long)[1:], "K5FIT", long))


def test_the_callers_data_nak_short_is_position_one_of_its_own_stream():
    """61/31 stands to 289/31 as every caller short frame stands to its
    responder twin: same lattice position, keyed to the called station, and
    no other 16-symbol frame of either station reads it."""
    tones = VF.handshake_tones("WI9YZA", VF.SESSION_DATA_NAK_CALLER_SHORT)
    assert (VF.payload_position(tones[1:], "WI9YZA", VF.SESSION_DATA_NAK_CALLER_SHORT)
            == VF.payload_position(VF.handshake_tones("WI9YZA", VF.SESSION_DATA_NAK_SHORT)[1:],
                                   "WI9YZA", VF.SESSION_DATA_NAK_SHORT) == 1)
    assert VF.recognize(tones, "WI9YZA", VF.SESSION_DATA_NAK_CALLER_SHORT)
    assert not any(VF.recognize(tones, cs, k) for k in _short_kinds()
                   if k is not VF.SESSION_DATA_NAK_CALLER_SHORT for cs in ("WI9YZA", "W9SSJ"))
