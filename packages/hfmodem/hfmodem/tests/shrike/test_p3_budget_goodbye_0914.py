# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A spent P3 retry budget must not buy a second train of repeat requests."""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_refused_breakin_ack import RefusingTurn


def receiving(protocol=spec.Protocol.PACTOR3):
    io = RefusingTurn()
    io.protocol = protocol
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    return a, io


@pytest.mark.parametrize('refuse_controls', [False, True])
def test_exhausted_budget_does_not_key_fallback_after_refused_qrt(refuse_controls):
    a, io = receiving()
    io.refuse_ack = refuse_controls
    for _ in range(a.cfg.max_retries + 1):
        a.on_cycle()
    assert a._qrt_pending and a._budget_goodbye
    io.events.clear()
    a.on_cycle()
    assert len(io.events) == 1 and io.events[0][0] == 'packet'
    assert io.events[0][3]  # A QRT break-in was offered to the placement guard.
    assert a.state == arq.State.DISCONNECTED
    assert a.goodbye_unplaceable and not a.said_goodbye
    assert not a._budget_goodbye
    for _ in range(arq.GOODBYE_PLACE_TICKS + 1):
        a.on_cycle()
    assert len(io.events) == 1


def test_operator_close_still_has_its_normal_repeat_fallback():
    a, io = receiving()
    a.on_host_disconnect()
    a.on_cycle()
    assert not a._budget_goodbye
    assert [e[0] for e in io.events] == ['packet', 'cs']
    assert io.events[-1] == ('cs', arq.CS_REQUEST)
    assert a.state == arq.State.CONNECTED and a._qrt_pending


def test_a_placeable_budget_goodbye_still_reaches_the_peer():
    a, io = receiving()
    a._give_up('offline exhausted retry budget')
    io.refuse_packet = False
    a.on_cycle()
    assert a.said_goodbye and not a.goodbye_unplaceable
    assert len(io.events) == 1 and io.events[0][0] == 'packet'
    assert io.events[0][3]


def test_skipped_slots_still_expire_goodbye_without_a_late_attempt():
    a, io = receiving()
    a._give_up('offline exhausted retry budget')
    a.on_cycle(elapsed_ticks=arq.GOODBYE_PLACE_TICKS)
    assert a.state == arq.State.DISCONNECTED
    assert a.goodbye_unplaceable and io.events == []


def test_p1_budget_goodbye_keeps_existing_behavior():
    a, io = receiving(spec.Protocol.PACTOR1)
    a._give_up('offline exhausted retry budget')
    a.on_cycle()
    assert io.events[-1] == ('cs', arq.CS_REQUEST)
    assert a._qrt_pending and not a._budget_goodbye


def test_crc_repeats_do_not_spend_the_silence_budget():
    a, io = receiving()
    for _ in range(3 * a.cfg.max_retries):
        a.on_rx_packet(4, b'', 0, True, protocol=spec.Protocol.PACTOR3)
        a.on_cycle()
    assert a.state == arq.State.CONNECTED
    assert not a._qrt_pending and not a._budget_goodbye
