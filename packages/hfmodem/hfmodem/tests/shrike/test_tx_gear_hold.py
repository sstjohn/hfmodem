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

from hfmodem.shrike import arq, p3acquire, rxfront, spec, traffic
from hfmodem.tests.shrike.p3_arq_support import (
    BODY,
    Seam as Seam,
    Sender as Sender,
    control,
    sender,
)

FIXTURE = Path(__file__).with_name('fixtures')/'gear-0920'


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
    # CS4 acknowledges under the trial too (VE3KPG 0924-0033); only the
    # repeated CS5 is held, so the trial keeps SL3 where the ladder walks down.
    if enabled:
        assert [sl for sl,_,_ in seam.frames] == [2,3,4,4,3,3,3,3]
        assert [status for _,status,_ in seam.frames] == [1,2,3,3,3,3,3,3]
        assert host.arq._buffer_raw == len(BODY)-23-59
    else:
        assert [sl for sl,_,_ in seam.frames] == [2,3,4,4,3,2,1,1]
        assert [status for _,status,_ in seam.frames] == [1,2,3,3,3,3,3,3]
        assert host.arq._buffer_raw == len(BODY)-23-59
        assert host.arq._inflight.payload + bytes(host.arq._outbuf) == BODY[23+59:]


def test_a_repeated_cs4_acknowledges_under_the_trial():
    host,seam = sender()
    control(host,4)
    assert host.arq.tx_seq == 1 and host.arq._p3_tx_gear_command is None
    control(host,4)
    assert host.arq.tx_seq == 2 and host.arq._buffer_raw == len(BODY)-23


def test_ack_releases_a_held_cs5():
    host,seam = sender()
    control(host,4)
    control(host,5)
    lines = traffic.TrafficLog().control('RX',4,host.arq)
    assert 'HELD NAK / REPEAT seq=1 (trial)' in lines[0]
    control(host,5)
    assert host.arq.speed_level == 1 and host.arq.tx_seq == 1
    control(host,2)  # Actual matching ACK for counter 1.
    assert host.arq.tx_seq == 2 and host.arq._p3_tx_gear_command is None


def test_refused_retransmission_retains_bytes_and_held_command():
    host,seam = sender()
    control(host,4)
    control(host,5)
    pending = host.arq._inflight
    frames = len(seam.frames)
    seam.refuse = True
    for _ in range(3):
        control(host,5)
    assert host.arq._inflight is pending and len(seam.frames) == frames
    seam.refuse = False
    control(host,5)
    assert seam.frames[-1] == seam.frames[frames-1]
    assert host.arq._p3_tx_gear_command == arq.CS_NAK


@pytest.mark.parametrize('transition', ['_give_link','_take_link','_enter_connected'])
def test_gear_hold_cannot_leak_across_turn_or_connection(transition):
    host,_ = sender()
    control(host,4)
    control(host,5)
    assert host.arq._p3_tx_gear_command == arq.CS_NAK
    getattr(host.arq,transition)()
    assert host.arq._p3_tx_gear_command is None


def test_pactor2_is_not_changed_by_the_p3_trial():
    host,_ = sender()
    host.protocol = spec.Protocol.PACTOR2
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    host.arq.on_rx_cs(arq.CS_SPEED_UP)
    assert host.arq.speed_level == 3 and host.arq.tx_seq == 2
