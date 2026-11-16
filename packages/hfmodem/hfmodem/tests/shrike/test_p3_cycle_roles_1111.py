# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.p3_arq_support import linked
from hfmodem.tests.shrike.p3_cycle_support import emit, packet, scene


def test_long_changeover_retries_keep_cadence_then_next_irs_stint_is_short(
        tmp_path, monkeypatch):
    s, tx, grid, live = scene(tmp_path, monkeypatch, old_long=True)
    a = s.host.arq
    packet(s, tx, grid, live, end_slot=0, long=True, status=0x20,
           decision=grid.boundary(0) - 10000)
    assert emit(s, tx, grid, live) == (0, 0)
    grid.p3_interval(True, 0)
    a._take_link()
    assert a.role == arq.ISS and a.cycle_long
    onair._grid_reversal(grid, s.host)
    assert grid.sending and grid.ticks == 3 and grid._p3_interval is None
    sent, pulses = [], []
    slot = grid.next_slot(tx.slot)
    for attempt in range(3):
        tx.aim(grid, slot)
        live.spend(max(0, tx.boundary - 10000 - live.now))
        if attempt == 0:
            a._start_next_packet(breakin=True)
        else:
            a.on_cycle(cycle_ticks=3, elapsed_ticks=3)
        assert not tx.refused and a.cycle_long and grid.ticks == 3
        p = a._inflight
        sent.append((tx.slot, p.sl, p.status, p.payload, p.breakin))
        pulses.append(tx.tx_audio_start + min(tx.tx_pulse_offsets))
        slot = grid.next_slot(tx.slot)
    assert sent == [(3, 1, 0x19, b'', True),
                    (6, 1, 0x19, b'', True),
                    (9, 1, 0x19, b'', True)]
    assert pulses == [780852, 960852, 1140852]
    assert a._inflight.retries == 2
    assert len(live.submitted) == 4
    a._on_ack()
    assert a.role == arq.ISS and not a.cycle_long and a._inflight is None
    grid.regear(a.cycle_long)
    assert grid.ticks == 1
    a._give_link()
    onair._grid_reversal(grid, s.host)
    assert a.role == arq.IRS and not a.cycle_long
    assert not grid.sending and grid.ticks == 1 and grid._p3_interval is None
    assert [emit(s, tx, grid, live, slot) for slot in (10, 11, 12)] == [
        (10, 1), (11, 1), (12, 1)]


@pytest.mark.parametrize('old_long', [False, True])
def test_p2_cs6_latches_at_command_time_without_p3_interval(old_long):
    a, io = linked(deferred=False)
    a.ladder = arq.P2_LADDER
    a.cycle_long = old_long
    grid = onair._MasterGrid(600000, 60000, 8880, packet_n=38880,
                             cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR2, False
    grid.regear(old_long)
    status = spec.status_byte(1, data_type=spec.DataType.ASCII_8BIT,
                              long_cycle_request=not old_long)
    a.on_rx_packet(1, b'abc', status, True, protocol=spec.Protocol.PACTOR2)
    assert io.controls == [5]
    assert a.cycle_long is not old_long
    assert a.cycle_request is None and not a.cycle_command_emitted
    grid.p3_interval(old_long, 7, pending=True, old_long=not old_long)
    assert grid._p3_interval is None and grid.cycle_long is old_long
    grid.regear(a.cycle_long)
    assert grid.ticks == (1 if old_long else 3)
    assert grid.next_slot(7) == (8 if old_long else 10)
    a.on_rx_packet(1, b'abc', status, True, protocol=spec.Protocol.PACTOR2)
    assert io.controls == [5, 0]
    assert bytes(io.delivered) == b'abc'
    assert a.cycle_long is not old_long and grid._p3_interval is None


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1,
                                      spec.Protocol.PACTOR2])
def test_other_protocols_ignore_p3_interval_and_keep_shared_regear(
        tmp_path, monkeypatch, protocol):
    s, tx, grid, live = scene(tmp_path, monkeypatch)
    s.host.protocol = grid.protocol = protocol
    before = grid.anchor, tx.slot, tx.boundary
    grid.p3_interval(True, 7, pending=True, old_long=False)
    tx.reconcile_p3_cycle(436320, True)
    tx.arm_p3_reply()
    assert grid._p3_interval is None
    assert (grid.anchor, tx.slot, tx.boundary) == before
    assert live.armed_at is None and not live.submitted
    assert grid.regear(True) is not None
    assert grid.ticks == 3 and grid.next_slot(7) == 10
    assert grid.regear(False) is not None
    assert grid.ticks == 1 and grid.next_slot(7) == 8
    assert grid.anchor == before[0]


def test_link_end_clears_pending_interval(tmp_path, monkeypatch):
    s, tx, grid, live = scene(tmp_path, monkeypatch)
    grid.p3_interval(True, 0, pending=True, old_long=False)
    s.host.arq.on_host_abort()
    onair._grid_reversal(grid, s.host)
    assert grid._p3_interval is None
    assert not live.submitted


def test_p4_owned_probe_does_not_open_p3_interval(tmp_path, monkeypatch):
    s, tx, grid, live = scene(tmp_path, monkeypatch)
    tx.p4_probe = SimpleNamespace(requested=True, emitting=False)
    tx.reconcile_p3_cycle(436320, True)
    tx.arm_p3_reply()
    assert grid._p3_interval is None and grid.ticks == 1
    assert live.armed_at is None and not live.submitted
