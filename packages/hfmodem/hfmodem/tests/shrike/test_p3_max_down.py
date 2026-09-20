# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS MAXDown is a receiver error run, separate from sender MAXTry."""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_v27_arq_emission_opportunity import linked, packet


@pytest.mark.parametrize('threshold', [2, 6, 30])
@pytest.mark.parametrize('deferred', [False, True])
def test_maxdown_requests_lower_speed_on_threshold_error(threshold, deferred):
    a, io = linked(deferred=deferred)
    a.cfg.p3_max_down = threshold
    a.cfg.max_retries = 64
    for n in range(threshold):
        packet(a, crc=False)
        io.emit()
        assert io.emissions[-1] == (arq.CS_NAK if n == threshold-1 else arq.CS_REQUEST)
    assert a._p3_rx_errors == 0 and a.state == arq.State.CONNECTED
    packet(a, crc=False)
    io.emit()
    assert io.emissions[-1] == arq.CS_REQUEST


def test_even_a_valid_duplicate_resets_consecutive_error_run():
    a, io = linked()
    packet(a)
    io.emit()
    for _ in range(5):
        packet(a, crc=False)
        io.emit()
    packet(a)  # Same packet counter and bytes, but correctly received.
    io.emit()
    assert a._p3_rx_errors == 0
    packet(a, crc=False)
    io.emit()
    assert a._p3_rx_errors == 1 and io.emissions[-1] != arq.CS_NAK
    assert io.delivered == b'repeat'


def test_refused_cs5_keeps_request_due_until_actual_emission():
    a, io = linked()
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
    assert a._p3_rx_errors == 0 and io.emissions[-1] == arq.CS_NAK


@pytest.mark.parametrize('sl', [1, 3])
def test_multiple_failed_candidates_in_one_window_count_once(sl):
    a, io = linked()
    token = a.begin_receive_opportunity()
    for _ in range(12):
        a.on_rx_packet(sl, b'bad', 1, False, protocol=spec.Protocol.PACTOR3)
    a.note_burst(0, at_anchor=True)
    a.finish_receive_opportunity(token)
    assert a._p3_rx_errors == (1 if sl > 1 else 0) and a._unrepaired == 1
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


def test_live_occupied_windows_count_once_after_all_readers():
    a, io = linked()
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
    assert a._p3_rx_errors == 0


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


def test_sl1_cannot_request_a_lower_p3_speed():
    a, io = linked()
    for _ in range(6):
        a.on_rx_packet(1, b'bad', 0, False, protocol=spec.Protocol.PACTOR3)
        io.emit()
    assert arq.CS_NAK not in io.emissions


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
