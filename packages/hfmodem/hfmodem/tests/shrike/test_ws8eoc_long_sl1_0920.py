"""WS8EOC granted long SL1 while our sender silently reverted to short."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, p3rx, placement, rxfront, spec
from hfmodem.tests.shrike.test_long_sl1 import linked

FIX = Path(__file__).with_name('fixtures')/'ws8eoc-long-sl1-0920'


def recorded():
    if not (FIX/'controls.json').is_file():
        pytest.skip(f'optional recording metadata is not included: {FIX / "controls.json"}')
    meta = json.loads((FIX/'controls.json').read_text())
    path = FIX/'controls.wav'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
    audio = rxfront.load_wav(str(path))
    n = meta['window_samples']
    events = []
    for i, row in enumerate(meta['controls']):
        cut = p3acquire.compensate(audio[i*n:(i+1)*n],meta['offset_hz'])
        ev = rxfront.SyncedRx().control_signal_at(cut,meta['local_phase'])
        assert ev is not None and ev.cs+1 == row['cs'] == 6
        events.append(ev)
    return meta, events


def test_recording_contains_twelve_grants_on_the_long_cycle():
    meta, events = recorded()
    assert len(events) == 12
    intervals = np.diff([r['phase'] for r in meta['controls']])/48000
    assert np.all(np.abs(intervals-3.75)<.004)


def test_captured_grant_drives_host_to_real_long_sl1_and_long_reply_grid():
    _, events = recorded()
    host, tx = linked()
    host.arq._next_seq = 1  # TX20 was counter 1, status 0x21.
    body = b': W9SSJ\r[Pat-1.0.0-B2FHM$]\r' + b'B'*150
    host.arq.on_host_data(body)
    host.tick()
    assert host.arq._inflight.status == 0x21
    assert host.arq._inflight.payload == body[:5]
    grid = onair._MasterGrid(0,60000,4320,packet_n=46080,cs_n=5760,d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.d_n, grid.d_ref_n = 3840, round(.810*48000)
    tx.aim(grid,23)
    host.on_rx_event(events[0])
    assert host.arq.cycle_long and host.arq.speed_level == 1
    assert host.arq.tx_seq == 2
    pending = host.arq._inflight.payload
    assert pending == body[5:41] and len(pending) == 36
    assert pending + bytes(host.arq._outbuf) == body[5:]
    packets = p3rx.decode_p3_packets(np.pad(tx.frames[-1][1],(14400,14400))).packets
    assert len(packets) == 1
    pkt = packets[0]
    assert pkt.long_cycle and pkt.sl == 1 and pkt.status & 3 == 2
    assert pkt.payload == pending
    slot, _ = onair._regear_next_slot(grid,grid.next_slot(tx.slot),host.arq.cycle_long)
    assert slot == 26 and grid.ticks == 3
    assert grid.rx_due(slot)-grid.boundary(slot) == round(3.390*48000)
    # A normal ACK must progress by the full transmitted field, not five bytes.
    host.arq.on_rx_cs(arq.CS_ACK)
    assert host.arq.tx_seq == 3 and host.arq.cycle_long
    assert host.arq._inflight.payload + bytes(host.arq._outbuf) == body[41:]


@pytest.mark.parametrize('swapped',[False,True])
def test_long_transmit_keeps_the_retained_ordinary_pulse(swapped):
    host, tx = linked(long=True)
    grid = onair._MasterGrid(0,60000,4320,packet_n=46080,cs_n=5760,d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.regear(True)
    tx.reply_clock = onair.ReplyClock()
    tx.aim(grid,6)
    tx.invert = swapped
    tx.reply_clock.pulse_epoch = grid.boundary(6)+852
    emitted=[]
    def collect(audio, label, **kw):
        tx.refused=False
        emitted.append((audio,kw))
    tx._tx=collect
    tx.send_long_packet(1,b'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',2)
    pcm,kw=emitted[0]
    assert tx.boundary == grid.boundary(6)+852
    assert kw['lead_n'] == placement.pulse_lead(pcm)
    assert min(kw['pulse_offsets']) == kw['lead_n']
    assert abs(kw['pulse_offsets'][0]-kw['pulse_offsets'][1]) == 240
