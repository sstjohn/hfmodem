# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded K0NTS controls through the real host and outbound accounting.

These tests prove the trial's behavior and byte conservation, not the remote
modem's hidden state. An on-air trial must adjudicate the held-command reading.
"""
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, p3acquire, placement, rxfront, spec, traffic
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_repeat_gear_stall import Seam

FIXTURE = Path(__file__).with_name('fixtures')/'gear-0920'
BODY = bytes(range(32, 127))*4


class Sender(Seam):
    def __init__(self):
        super().__init__()
        self.frames = []
        self.refuse = False

    def send_packet(self, sl, payload, status, breakin=False):
        if self.refuse:
            return arq.REFUSED
        n = (placement.CHANGEOVER if breakin else placement.SPEED_PATHS[sl]).crc_bytes-3
        self.frames.append((sl, status, payload[:n]))
        return n


def sender(enabled=True):
    seam = Sender()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    host.arq.role, host.arq.dxcall = arq.ISS, 'K0NTS'
    host.arq._enter_connected()
    host.protocol = spec.Protocol.PACTOR3
    host.p3_tx_gear_hold = enabled
    a = host.arq
    a.cfg.long_cycle = False
    a._inflight = arq._Packet(0,b'\x01/W',1,breakin=True)
    a._next_seq = 1
    a._outbuf = bytearray(BODY)
    a._buffer_raw = len(BODY)+3
    return host,seam


def control(host, cs):
    host.on_rx_event(rxfront.Event(0,'cs','test',protocol=spec.Protocol.PACTOR3,cs=cs-1))


def recorded_controls():
    if not (FIXTURE / 'k0nts-outbound-controls.json').is_file():
        pytest.skip(f'optional recording metadata is not included: {FIXTURE}')
    meta = json.loads((FIXTURE/'k0nts-outbound-controls.json').read_text())
    path = FIXTURE/'k0nts-outbound-controls.wav'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
    audio = rxfront.load_wav(str(path))
    n = meta['window_samples']
    for i,row in enumerate(meta['controls']):
        cut = p3acquire.compensate(audio[i*n:(i+1)*n],meta['offset_hz'])
        ev = rxfront.SyncedRx().control_signal_at(cut,row['local_phase'])
        assert ev is not None and ev.cs+1 == row['cs']
        yield ev


@pytest.mark.parametrize('enabled', [False,True])
def test_recorded_climb_does_not_spend_two_extra_fields_in_hold_trial(enabled):
    host,seam = sender(enabled)
    for ev in recorded_controls():
        host.on_rx_event(ev)
    if enabled:
        # The hold trial still preserves bytes/counter; the MAXTry
        # limit now bounds the time it can keep an unread packet at SL2.
        assert [sl for sl,_,_ in seam.frames] == [2,2,1,1,1,1,1,1]
        assert {status for _,status,_ in seam.frames} == {1}
        assert host.arq._buffer_raw == len(BODY)
        assert host.arq._inflight.payload + bytes(host.arq._outbuf) == BODY
    else:
        assert [sl for sl,_,_ in seam.frames] == [2,3,4,4,3,2,1,1]
        assert [status for _,status,_ in seam.frames] == [1,2,3,3,3,3,3,3]
        assert host.arq._buffer_raw == len(BODY)-23-59
        assert host.arq._inflight.payload + bytes(host.arq._outbuf) == BODY[23+59:]


def test_ack_releases_hold_and_the_next_speedup_can_advance():
    host,seam = sender()
    control(host,4)
    lines = traffic.TrafficLog().control('RX',3,host.arq)
    assert 'HELD SPEED-UP / REPEAT seq=1 (trial)' in lines[0]
    assert 'no additional bytes acknowledged' in lines[1]
    control(host,4)
    first = host.arq._inflight.payload
    assert len(first) == 23
    control(host,2)  # Actual matching ACK for counter 1.
    assert host.arq.tx_seq == 2
    assert host.arq._p3_tx_gear_command is None
    control(host,4)
    assert host.arq.tx_seq == 3 and host.arq.speed_level == 3
    assert host.arq._buffer_raw == len(BODY)-46


def test_refused_retransmission_retains_bytes_and_held_command():
    host,seam = sender()
    control(host,4)
    pending = host.arq._inflight
    seam.refuse = True
    for _ in range(3):
        control(host,4)
    assert host.arq._inflight is pending and len(seam.frames) == 1
    assert host.arq._buffer_raw == len(BODY)
    seam.refuse = False
    control(host,4)
    assert seam.frames[-1] == seam.frames[0]


@pytest.mark.parametrize('transition', ['_give_link','_take_link','_enter_connected'])
def test_gear_hold_cannot_leak_across_turn_or_connection(transition):
    host,_ = sender()
    control(host,4)
    assert host.arq._p3_tx_gear_command == arq.CS_SPEED_UP
    getattr(host.arq,transition)()
    assert host.arq._p3_tx_gear_command is None


def test_pactor2_is_not_changed_by_the_p3_trial():
    host,_ = sender()
    host.protocol = spec.Protocol.PACTOR2
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    assert host.arq.speed_level == 3 and host.arq.tx_seq == 2
