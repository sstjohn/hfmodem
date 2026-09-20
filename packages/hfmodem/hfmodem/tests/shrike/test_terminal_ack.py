# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A received P3 QRT closes only after its scheduled ACK has reached the seam.

Events here specify a synthetic closing peer on the measured M06 geometry.
Actual emitted intervals pass through RadioTx and the sample-clock bench.
"""

import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.test_p3_morning_timing import harness

PHASE = 2808732


def _packet(status, *, breakin=True, phase=PHASE, payload=b"bye"):
    return rxfront.Event(phase / onair.FS, "packet", "specified peer close",
                         protocol=spec.Protocol.PACTOR3,
                         packet=(1, status, payload, True), breakin=breakin,
                         cycle_long=False)


@pytest.mark.parametrize("closing", [False, True])
@pytest.mark.parametrize("seq", [0, 1])
def test_peer_qrt_keeps_protocol_and_role_until_safe_ack_emission(tmp_path, closing, seq):
    host, rx, tx, grid, bench = harness(tmp_path)
    if closing:
        host.arq.state = arq.State.DISCONNECTING
        host.arq._qrt_pending = host.arq.said_goodbye = True
    rx._on(_packet(spec.STATUS_QRT | seq))
    assert host.protocol == spec.Protocol.PACTOR3
    assert host.arq.state == arq.State.DISCONNECTING
    assert host.arq.role == arq.IRS
    assert host.arq._rx_close_pending
    assert tx._pending_p3_cs == seq  # CS1/even, CS2/odd, specified wire phase.
    assert not bench.emissions

    onair._reverse_before_key(grid, host, tx, 48)
    assert grid.anchor == 31432
    host.tick()
    assert not bench.emissions, "the pending peer close must not key a local QRT"
    tx.emit_pending_cs()
    assert len(bench.emissions) == 1 and not tx.refused
    start, end = bench.emissions[0]
    peer = PHASE + ((start - PHASE) // 60000) * 60000
    assert start - 1920 > peer + 38880
    assert end < peer + 60000
    assert host.arq.state == arq.State.DISCONNECTED
    assert host.arq.role is None
    assert host.protocol == spec.Protocol.PACTOR1
    assert not host.arq._rx_close_pending
    assert tx._pending_p3_cs is None
    assert bytes(host.channel(host.ptchn).rx) == b"bye"
    assert not any("DISCONNECTING -> CONNECTED" in line for line in host.log_lines)
    tx.emit_pending_cs()
    assert len(bench.emissions) == 1


def test_refused_terminal_ack_retries_without_pretending_close(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path)
    rx._on(_packet(spec.STATUS_QRT))
    onair._reverse_before_key(grid, host, tx, 48)
    # Put the proposed ACK inside the next peer burst. The comb is held on the
    # answer slot now, so what covers that slot is the burst this station read:
    # the real CRC guard must refuse it, and that refusal cannot be reported as
    # an emitted ACK.
    at, safe_width, period, seen = grid._p3_peer
    grid._p3_peer = (at, round(.95 * onair.FS), period, seen)
    tx.emit_pending_cs()
    assert tx.refused and not bench.emissions
    assert host.arq.state == arq.State.DISCONNECTING
    assert tx._pending_p3_cs == arq.CS_ACK
    host.tick()
    # The peer repeats its close on its own raster, and that fresh CRC frame is
    # what re-validates the reply phase the guard invalidated.
    rx._on(_packet(spec.STATUS_QRT, phase=at + period))
    assert grid._p3_peer[1] == safe_width and not grid._p3_reply_phase_invalid
    tx.emit_pending_cs()
    assert len(bench.emissions) == 1 and not tx.refused
    assert host.arq.state == arq.State.DISCONNECTED


@pytest.mark.parametrize("long_cycle", [False, True])
def test_terminal_ack_wait_is_bounded_even_when_peer_repeats_qrt(tmp_path, long_cycle):
    host, rx, tx, grid, bench = harness(tmp_path)
    host.arq.cycle_long = long_cycle
    ticks = arq.LONG_TICKS if long_cycle else 1
    event = _packet(spec.STATUS_QRT)
    host.on_rx_event(event)
    for attempt in range(arq.GOODBYE_CYCLES + 1):
        host.on_rx_event(event)
        for _ in range(ticks):
            host.tick()
        if attempt < arq.GOODBYE_CYCLES:
            assert host.arq.state == arq.State.DISCONNECTING
    assert host.arq.state == arq.State.DISCONNECTED
    assert tx._pending_p3_cs is None
    assert not bench.emissions
    assert bytes(host.channel(host.ptchn).rx) == b"bye"


def test_new_breakin_request_cancels_older_ack_even_if_breakin_is_refused(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path)
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role = arq.IRS
    host.on_rx_event(_packet(0, breakin=False, payload=b"one"))
    assert tx._pending_p3_cs == arq.CS_ACK
    host.on_rx_event(_packet(1 | spec.STATUS_CHANGEOVER,
                             breakin=False, payload=b"two"))
    assert tx._pending_p3_cs is None
    assert host.arq._breakin_pending and host.arq._breakin_armed
    tx.listening = True  # The real render seam refuses this cycle's break-in.
    host.tick()
    assert host.arq.role == arq.IRS
    assert not bench.emissions
    tx.emit_pending_cs()
    assert not bench.emissions
    assert bytes(host.channel(host.ptchn).rx) == b"onetwo"


def test_closed_link_cancels_queue_and_cannot_leak_ack_into_next_link(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path)
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role = arq.IRS
    host.on_rx_event(_packet(0, breakin=False))
    assert tx._pending_p3_cs == arq.CS_ACK
    host.arq.on_host_abort()
    assert tx._pending_p3_cs is None
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    host.protocol = spec.Protocol.PACTOR3
    tx.emit_pending_cs()
    assert not bench.emissions


def test_nonclosing_traffic_cannot_replace_pending_final_ack(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path)
    host.on_rx_event(_packet(spec.STATUS_QRT | 1))
    host.on_rx_event(_packet(2, payload=b"new", breakin=False))
    assert tx._pending_p3_cs == 1
    assert host.arq._rx_close_pending
    assert bytes(host.channel(host.ptchn).rx) == b"bye"


@pytest.mark.parametrize("cs", [arq.CS_BREAKIN, arq.CS_ACK])
def test_only_late_breakin_head_follows_p3_during_p1_teardown(tmp_path, cs):
    host, rx, tx, grid, bench = harness(tmp_path)
    host.arq.state = arq.State.DISCONNECTING
    host.arq._qrt_pending = host.arq.said_goodbye = True
    host.on_rx_event(rxfront.Event(0, "cs", "specified late control",
                                  protocol=spec.Protocol.PACTOR3, cs=cs))
    assert host.protocol == (spec.Protocol.PACTOR3 if cs == arq.CS_BREAKIN
                             else spec.Protocol.PACTOR1)
    if cs == arq.CS_BREAKIN:
        assert host.arq.role == arq.IRS
        assert host.arq.state == arq.State.DISCONNECTING
