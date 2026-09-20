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
    assert _last_pairs(io) == list(VF.NAK_CALLER_2300)
    assert any("tx NAK" in m for m in io.msgs), io.msgs


def test_the_responder_keys_the_other_nak_of_the_link():
    hs, io = _connected()
    hs.role = "responder"
    assert hs._undecoded_over()
    assert _last_pairs(io) == list(VF.NAK_RESPONDER_2300)


def test_the_two_naks_share_only_the_lead_pair():
    caller, responder = VF.nak("W9SSJ", "2300")
    assert caller[0] == responder[0], "the family opens on one lead pair"
    shared = set(caller[1:]) & set(responder[1:])
    assert not shared, f"role-keyed frames share a tail symbol: {shared}"


def test_an_unmeasured_link_leaves_the_turnaround_empty():
    """The NAK is measured for a W9SSJ-called link only. Answering under another
    callsign keys no stranger's tail — it falls back to the silence the peer's own
    repeat timer covers, and says so."""
    hs, io = _connected()
    hs.caller = "N0XYZ"
    assert hs._undecoded_over()
    assert not io.sent, "keyed a NAK for a link with none measured"
    assert any("no NAK is measured" in m for m in io.msgs), io.msgs


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
