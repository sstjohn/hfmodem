# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS MAXDown is a receiver error run, separate from sender MAXTry."""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_v27_arq_emission_opportunity import linked, packet


@pytest.mark.parametrize('hold', ['held', 'once'])
@pytest.mark.parametrize('threshold', [2, 6, 30])
@pytest.mark.parametrize('deferred', [False, True])
def test_maxdown_requests_lower_speed_on_threshold_error(threshold, deferred, hold):
    a, io = linked(deferred=deferred)
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = threshold
    a.cfg.max_retries = 64
    for n in range(threshold):
        packet(a, crc=False)
        io.emit()
        assert io.emissions[-1] == (arq.CS_NAK if n == threshold-1 else arq.CS_REQUEST)
    assert a._p3_rx_errors == (threshold if hold == 'held' else 0)
    assert a.state == arq.State.CONNECTED
    for _ in range(3):
        packet(a, crc=False)
        io.emit()
    assert io.emissions[threshold:] == (
        [arq.CS_NAK] * 3 if hold == 'held' else
        [arq.CS_NAK if n % threshold == 0 else arq.CS_REQUEST for n in range(1, 4)])


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_even_a_valid_duplicate_resets_consecutive_error_run(hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    packet(a)
    io.emit()
    for _ in range(7):
        packet(a, crc=False)
        io.emit()
    assert io.emissions[-2:] == [arq.CS_NAK,
                                arq.CS_NAK if hold == 'held' else arq.CS_REQUEST]
    packet(a)
    io.emit()
    assert a._p3_rx_errors == 0
    packet(a, crc=False)
    io.emit()
    assert a._p3_rx_errors == 1 and io.emissions[-1] != arq.CS_NAK
    assert io.delivered == b'repeat'


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_refused_cs5_keeps_request_due_until_actual_emission(hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = 2
    packet(a, crc=False)
    io.emit()
    io.refuse = True
    packet(a, crc=False)
    assert a._p3_rx_errors == 2 and io.pending is None
    assert arq.CS_NAK not in io.emissions
    io.refuse = False
    packet(a, crc=False)
    assert a._p3_rx_errors == 3 and io.pending == arq.CS_NAK
    io.emit()
    assert a._p3_rx_errors == (3 if hold == 'held' else 0)
    assert io.emissions[-1] == arq.CS_NAK


@pytest.mark.parametrize('hold', ['held', 'once'])
@pytest.mark.parametrize('sl', [1, 3])
def test_multiple_failed_candidates_in_one_window_count_once(sl, hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    token = a.begin_receive_opportunity()
    for _ in range(12):
        a.on_rx_packet(sl, b'bad', 1, False, protocol=spec.Protocol.PACTOR3)
    a.note_burst(0, at_anchor=True)
    a.finish_receive_opportunity(token)
    assert a._p3_rx_errors == (1 if sl > 1 or hold == 'held' else 0)
    assert a._unrepaired == 1
    assert not a._qrt_pending
    io.emit()
    assert io.emissions == [arq.CS_REQUEST]


def test_late_valid_decode_cancels_window_error():
    a, io = linked()
    a.cfg.p3_max_down = 2
    packet(a, crc=False)
    io.emit()
    token = a.begin_receive_opportunity()
    packet(a, crc=False)  # Queues CS5, but the final reader can still repair it.
    assert io.pending == arq.CS_NAK
    packet(a, seq=2)
    a.finish_receive_opportunity(token)
    io.emit()
    assert a._p3_rx_errors == 0 and io.emissions[-1] == arq.CS_ACK
    assert arq.CS_NAK not in io.emissions


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_live_occupied_windows_count_once_after_all_readers(hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = 2
    packet(a)
    io.emit()
    a.on_cycle()  # Retire the successful packet's cycle.
    for _ in range(2):
        token = a.begin_receive_opportunity()
        a.on_cycle(elapsed_ticks=7)  # Missed historical slots add no errors.
        io.emit()
        a.note_burst(0, at_anchor=True)
        a.finish_receive_opportunity(token)
    assert a._p3_rx_errors == 2
    token = a.begin_receive_opportunity()
    a.on_cycle()
    assert io.pending == arq.CS_NAK
    io.emit()
    a.finish_receive_opportunity(token)
    assert a._p3_rx_errors == (2 if hold == 'held' else 0)


def test_silence_and_out_of_slot_energy_are_not_received_error_packets():
    a, io = linked()
    packet(a)
    io.emit()
    a.on_cycle()
    for _ in range(3):
        token = a.begin_receive_opportunity()
        a.on_cycle()
        a.note_burst(200, at_anchor=False)
        a.finish_receive_opportunity(token)
    assert a._p3_rx_errors == 0


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_sl1_maxdown_response(hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    for _ in range(7):
        a.on_rx_packet(1, b'bad', 0, False, protocol=spec.Protocol.PACTOR3)
        io.emit()
    assert io.emissions == ([arq.CS_REQUEST] * 5 + [arq.CS_NAK] * 2
                            if hold == 'held' else [arq.CS_REQUEST] * 7)


@pytest.mark.parametrize('hold', ['held', 'once'])
def test_maxdown_remains_held_through_downshifts_to_sl1(hold):
    a, io = linked()
    if hold == 'once':
        a.cfg.p3_maxdown_hold = hold
    a.cfg.p3_max_down = 2
    for _ in range(2):
        packet(a, crc=False)
        io.emit()
    assert io.emissions == [arq.CS_REQUEST, arq.CS_NAK]
    for sl in (2, 1, 1, 1):
        a.on_rx_packet(sl, b'bad', 1, False, protocol=spec.Protocol.PACTOR3)
        io.emit()
    assert io.emissions[2:] == [arq.CS_NAK if hold == 'held' else arq.CS_REQUEST] * 4
    a.on_rx_packet(1, b'read', 1, True, protocol=spec.Protocol.PACTOR3)
    io.emit()
    assert io.emissions[-1] == arq.CS_REQUEST
    assert io.delivered == b'read'
    assert a._p3_rx_errors == 0
    a.on_rx_packet(1, b'bad', 2, False, protocol=spec.Protocol.PACTOR3)
    io.emit()
    assert io.emissions[-1] == arq.CS_REQUEST


@pytest.mark.parametrize('transition', ['_give_link', '_take_link', '_enter_connected'])
def test_error_run_does_not_cross_roles_or_connections(transition):
    a, io = linked()
    packet(a, crc=False)
    getattr(a, transition)()
    assert a._p3_rx_errors == 0 and a._p3_rx_sl is None


def test_p2_retains_legacy_error_response():
    a, io = linked()
    io.protocol = spec.Protocol.PACTOR2
    a.on_rx_packet(3, b'bad', 0, False, protocol=spec.Protocol.PACTOR2)
    io.emit()
    assert io.emissions == [arq.CS_NAK]


@pytest.mark.parametrize('kwargs', [dict(p3_max_try=0), dict(p3_max_try=10),
                                  dict(p3_max_down=1), dict(p3_max_down=31)])
def test_config_rejects_values_outside_manual_ranges(kwargs):
    with pytest.raises(ValueError):
        arq.ArqConfig(**kwargs)
