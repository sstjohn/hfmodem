# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Regression states from the September10 K7UNI/N5TW idle loops.

These test recovery obligations and termination, not an independent peer model.
The NAK/control vocabulary is supported separately by stock-VARA recordings.
"""
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.tests.kestrel.test_vara_giveup import _connected


def waiting(bw="2300", transmits=True):
    hs, io = _connected(transmits=transmits)
    hs.bw = bw
    hs.turn = VA._TURN_PEER
    return hs, io


def partial_window(hs):
    hs._held_answer = (False, False)
    hs._release_held_answer(VA._ANSWER_HOLD_MAX)


def test_lost_nak_is_repeated_without_acknowledging_the_unread_data():
    hs, io = waiting()
    hs._undecoded_over()
    first = io.sent[-1]
    for _ in range(2):
        assert hs._reack()
        assert (io.sent[-1] == first).all()
        assert hs._owed_block
    assert hs._reacks == 3


def test_a_partial_window_timeout_never_positively_acknowledges_it():
    hs, io = waiting()
    partial_window(hs)
    assert not io.sent
    assert hs._held_answer is None
    assert hs._owed_block
    assert hs._reack()
    assert "tx NAK" in io.msgs[-1]
    assert hs._owed_block


def test_recovery_retains_the_delivered_prefix_until_missing_data_arrives():
    hs, io = waiting()
    a, b = (phy.vara_body(x * 89, "W9SSJ") for x in (b"A", b"B"))
    hs._deliver([a], hold=True)
    partial_window(hs)
    hs._reack()
    hs._deliver([a], hold=True)
    hs._deliver([b], hold=False)
    hs._key_over_answer(last=False, owes_release=False)
    assert io.delivered == [b"A" * 89, b"B" * 89]
    assert not hs._owed_block


def test_controls_cannot_settle_or_reset_missing_data_recovery():
    hs, _ = waiting()
    hs._undecoded_over()
    hs._since_progress = 2
    before = hs.progress
    hs._took_stall_answer()
    assert hs._owed_block
    assert hs._answer_owed == VA._OWED_OVER
    assert hs._since_progress == 2
    assert hs.progress == before


def test_unsupported_nak_closes_instead_of_spinning_forever():
    hs, io = waiting("2750")
    hs.caller = "N0XYZ"  # W9SSJ now has measured BW2750 idle recovery.
    partial_window(hs)
    hs._reack()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert any("no NAK is measured" in msg for msg in io.msgs)
    assert hs._reacks == 0  # no fictitious NAK transmission


def test_missing_data_retries_end_without_a_generic_ack_ladder():
    hs, io = waiting()
    partial_window(hs)
    for _ in range(VA._REACK_MAX):
        assert hs._reack()
    hs._reack()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert sum("tx NAK" in msg for msg in io.msgs) == VA._REACK_MAX


def test_transport_refusal_preserves_the_request_and_its_budget():
    hs, io = waiting(transmits=False)
    partial_window(hs)
    for _ in range(12):
        assert not hs._reack()
    assert hs.state == VA.VaraState.CONNECTED
    assert hs._reacks == 0
    assert hs._owed_block
    assert not io.sent


def test_nak_transmissions_are_charged_once_on_either_clock():
    hs, _ = waiting()
    hs._undecoded_over()
    assert hs._since_progress == hs.idle_keyed == 1
    hs.idle_keepalive()  # the peer-gap NAK already served this tick
    assert hs._since_progress == hs.idle_keyed == 1
    hs.idle_keepalive()  # no later peer gap: cadence fallback now owes the NAK
    assert hs._since_progress == hs.idle_keyed == 2


def test_initial_nak_refusal_is_not_reported_as_an_unmeasured_frame():
    hs, io = waiting(transmits=False)
    hs._undecoded_over()
    assert hs._reacks == hs._since_progress == hs.idle_keyed == 0
    assert any("transmission declined" in msg for msg in io.msgs)
    assert not any("no NAK is measured" in msg for msg in io.msgs)


def test_cadence_keys_nothing_inside_a_deferred_window():
    hs, io = waiting()
    hs._held_answer = (False, False)
    hs.idle_keepalive()
    assert not io.sent
    assert hs._since_progress == 0


@pytest.mark.parametrize("payload", [b"", b"truncated greeting"])
def test_final_ack_is_owed_even_before_the_host_can_answer(payload):
    hs, io = waiting()
    hs._deliver([phy.vara_body(payload, "W9SSJ")])
    hs._key_over_answer(last=True, owes_release=False)
    original = io.sent[-1]
    assert hs._answer_owed == VA._OWED_RELEASE
    for _ in range(VA._REACK_MAX):
        assert hs._reack()
        assert (io.sent[-1] == original).all()
    hs._reack()
    assert hs._answer_owed is None
    assert hs._asked == 0
