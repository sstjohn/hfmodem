# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keep negotiated long SL1 waveform, byte accounting, and driver clock aligned."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3rx, spec
from hfmodem.shrike.ptc import PtcHost

class MemoryTx(onair.RadioTx):
    def __init__(self):
        super().__init__(None, transmit=False, out_dev=None, outdir=Path('.'))
        self.frames = []
    def _tx(self, audio, what, **kwargs):
        self.refused = False
        self.frames.append((what, np.asarray(audio)))


def linked(role=arq.ISS, sl=1, long=False):
    tx = MemoryTx()
    host = PtcHost(tx)
    host.protocol = spec.Protocol.PACTOR3
    host.arq._enter_connected()
    host.arq.role = role
    host.arq.speed_level = sl
    host.arq.cycle_long = long
    return host, tx


def row(name, host, tx, expected):
    what, pcm = tx.frames[-1]
    decoded = p3rx.decode_p3_packets(np.pad(pcm, (14400, 14400))).packets
    assert len(decoded) == 1, (name, what, len(decoded))
    pkt = decoded[0]
    assert pkt.long_cycle == host.arq.cycle_long
    held = (host.arq._inflight.payload if host.arq._inflight else b'') + bytes(host.arq._outbuf)
    assert held == expected, (name, held, expected)
    print(json.dumps(dict(case=name, what=what, state=str(host.arq.state),
        role=host.arq.role, logical_long=host.arq.cycle_long,
        physical_long=pkt.long_cycle, sl=pkt.sl, status=hex(pkt.status),
        frame_payload=len(pkt.payload), inflight=len(host.arq._inflight.payload),
        queued=len(host.arq._outbuf), bytes_preserved=True)))

PAYLOAD = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 5


def test_unsolicited_cs6_at_sl1_renders_long_waveform_and_keeps_clock():
    host, tx = linked()
    host.arq.on_host_data(PAYLOAD)
    host.tick()
    acked = len(host.arq._inflight.payload)
    host.arq.on_rx_cs(arq.CS_CYCLE_TOG)
    row('CS6', host, tx, PAYLOAD[acked:])
    assert host.arq.cycle_long


def test_long_sl2_nak_to_sl1_preserves_bytes_and_long_clock():
    host, tx = linked(sl=2, long=True)
    host.arq.on_host_data(PAYLOAD)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    host.arq.on_rx_cs(arq.CS_NAK)
    row('CS5', host, tx, PAYLOAD)
    assert host.arq.speed_level == 1 and host.arq.cycle_long


def test_long_peer_clock_survives_breakin_through_ack_and_ordinary_long_sl1():
    host, tx = linked(role=arq.IRS, long=True)
    host.arq.on_host_data(PAYLOAD)
    host.arq.on_host_breakin()
    host.arq.on_rx_packet(3, b'abc', spec.status_byte(1, long_cycle_request=True), True,
                          protocol=spec.Protocol.PACTOR3, cycle_long=True)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    assert host.arq.role == arq.ISS and host.arq._inflight.breakin
    assert host.arq.cycle_long
    host.arq.on_rx_cs(arq.CS_REQUEST)
    assert host.arq._inflight.breakin and host.arq.cycle_long
    acked = len(host.arq._inflight.payload)
    host.arq.on_rx_cs(arq.CS_ACK)
    row('breakin-ACK', host, tx, PAYLOAD[acked:])
    assert host.arq.cycle_long


def test_sl1_ordinary_qrt_keeps_long_clock_and_can_close():
    host, tx = linked(long=True)
    host.arq.on_host_disconnect()
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    row('QRT', host, tx, b'')
    assert host.arq.said_goodbye and host.arq.cycle_long
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq.goodbye_acked and host.arq.state == arq.State.DISCONNECTED


def test_first_ordinary_qrt_after_long_breakin_ack_uses_long_clock():
    host, tx = linked(role=arq.IRS, long=True)
    host.arq.on_host_data(b'abc')
    host.arq.on_host_breakin()
    host.arq.on_rx_packet(3, b'xyz', spec.status_byte(1, long_cycle_request=True), True,
                          protocol=spec.Protocol.PACTOR3, cycle_long=True)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    assert host.arq._inflight.breakin and host.arq.cycle_long
    host.arq.on_host_disconnect()
    assert host.arq._inflight.breakin and host.arq.cycle_long
    host.arq.on_rx_cs(arq.CS_ACK)
    row('QRT-after-breakin-ACK', host, tx, b'')
    assert host.arq.said_goodbye and not host.arq._inflight.breakin
    assert host.arq.cycle_long
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq.goodbye_acked and host.arq.state == arq.State.DISCONNECTED


@pytest.mark.parametrize('sl', [1, 2, 3, 4, 5, 6])
def test_supported_long_levels_still_follow_cs6(sl):
    host, tx = linked(sl=sl)
    host.arq.on_host_data(b'abc')
    host.tick()
    host.arq.on_rx_cs(arq.CS_CYCLE_TOG)
    assert host.arq.cycle_long
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    what, pcm = tx.frames[-1]
    decoded = p3rx.decode_p3_packets(np.pad(pcm, (14400, 14400))).packets
    assert len(decoded) == 1 and decoded[0].long_cycle and decoded[0].sl == sl
    assert host.arq.cycle_long and 'LONG' in what


def test_downshift_then_all_long_acks_delivers_each_byte_once():
    host, tx = linked(sl=2, long=True)
    host.arq.on_host_data(PAYLOAD)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    host.arq.on_rx_cs(arq.CS_NAK)
    delivered = bytearray()
    for _ in range(40):
        assert host.arq.cycle_long
        _, pcm = tx.frames[-1]
        packets = p3rx.decode_p3_packets(np.pad(pcm, (14400, 14400))).packets
        assert len(packets) == 1 and packets[0].long_cycle
        delivered.extend(packets[0].payload)
        host.arq.on_rx_cs(arq.CS_ACK)
        if host.arq._inflight is None:
            break
    assert bytes(delivered) == PAYLOAD
    assert not host.arq._outbuf and host.arq._buffer_raw == 0


def test_cs6_ack_of_last_sl1_payload_leaves_next_idle_slot_long():
    host, tx = linked()
    host.arq.on_host_data(b'abc')
    host.tick()
    host.arq.on_rx_cs(arq.CS_CYCLE_TOG)
    assert host.arq._inflight is None and host.arq.cycle_long
    before = len(tx.frames)
    host.tick(elapsed_ticks=3, cycle_ticks=3)
    assert len(tx.frames) == before + 1
    row('first-idle-after-CS6', host, tx, b'')


@pytest.mark.parametrize('trigger', ['CS5', 'breakin ACK'])
def test_driver_keeps_next_slot_and_answer_window_long_at_sl1(trigger):
    if trigger == 'CS5':
        host, tx = linked(sl=2, long=True)
        host.arq.on_host_data(PAYLOAD)
        host.tick(elapsed_ticks=3, cycle_ticks=3)
        answer = arq.CS_NAK
    else:
        host, tx = linked(role=arq.IRS, long=True)
        host.arq.on_host_data(PAYLOAD)
        host.arq.on_host_breakin()
        host.arq.on_rx_packet(3, b'abc', spec.status_byte(1, long_cycle_request=True), True,
                              protocol=spec.Protocol.PACTOR3, cycle_long=True)
        host.tick(elapsed_ticks=3, cycle_ticks=3)
        assert host.arq._inflight.breakin and host.arq.cycle_long
        answer = arq.CS_ACK

    # The driver is now sending on the old long raster. Preserve its acquired
    # 80ms turnaround at SL1, just as the live regear seam does.
    grid = onair._MasterGrid(0, 60000, 4320, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.d_n, grid.d_ref_n = round(.080 * 48000), round(.810 * 48000)
    grid.regear(True)
    tx.aim(grid, 3)
    assert grid.rx_due(3) - grid.boundary(3) == round(3.390 * 48000)
    host.arq.on_rx_cs(answer)  # Production PtcHost/RadioTx renders long SL1.
    assert host.arq.cycle_long and tx.frames[-1][0].startswith('SL1 LONG pkt')
    next_slot, _ = onair._regear_next_slot(
        grid, grid.next_slot(tx.slot), host.arq.cycle_long)
    assert next_slot == tx.slot + 3 == 6 and grid.ticks == 3
    assert grid.rx_due(next_slot) - grid.boundary(next_slot) == round(3.390 * 48000)
    assert tx.key_instant(grid, next_slot) == grid.boundary(next_slot)
