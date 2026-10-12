"""Long receive frames must reach their ACK without the general search ladder."""
import numpy as np
import pytest
from hfmodem.shrike import arq, onair, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session


@pytest.mark.parametrize('long', [False, True])
@pytest.mark.parametrize('swapped', [False, True])
def test_bounded_sl1_decodes_header_selected_cycle_and_refuses_missing_body(long, swapped):
    payload = b'long receive deadline' if long else b'short'
    packet = placement.link_packet(1, payload, 0x21, swapped=swapped, long_cycle=long)
    pcm = np.pad(packet, (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1)//2 + 4320
    paths = placement.LONG_PATHS if long else placement.SPEED_PATHS
    span = rxfront._packet_span(rxfront._frame_span(paths[1]))
    sync = rxfront.SyncedRx()
    assert sync.sl1_packet_at(pcm[:row+span-1], row) is None
    assert sync.packet_level is None
    ev = sync.sl1_packet_at(pcm[:row+span+240], row)
    assert ev is not None and ev.packet == (1, 0x21, payload, True)
    assert ev.cycle_long == long


def test_long_current_frame_uses_bounded_reader_and_reaches_arq(monkeypatch):
    s = _Session(role=arq.IRS)
    r=s.rx
    s.host.arq.cycle_long=True
    payload=b'current long frame'
    pcm=np.pad(placement.link_packet(1,payload,0x21,long_cycle=True),(4800,4800))
    row=4800+(placement.protocol_config().pulse().size-1)//2+4320
    r._p3_row0=row-180000
    r._p3_cycle_n=180000
    r._p3_span=rxfront._frame_span(placement.LONG_PATHS[1])
    r.sync.packet_level=1
    def forbidden(*args,**kwargs):
        pytest.fail('long pre-key frame opened the general decoder')
    monkeypatch.setattr(r,'_read_p3_packet',forbidden)
    deadline=row+r._p3_span+2000
    ready=onair._p3_frame_ready(r,deadline,660)
    assert ready==row+rxfront._packet_span(r._p3_span)+660+240
    onair._scan_frame(r,pcm[:ready-660],0,tracked_only=True)
    assert s.packets[-1].packet[2]==payload
    assert s.host.arq.rx_seq==1
    assert s.host.peer.sent[-1]==('cs',arq.CS_REQUEST)


@pytest.mark.parametrize('phase', [0, 96, 192, 288])
def test_long_ack_leaves_exact_dac_deadline_across_callback_batches(phase):
    from hfmodem.tests.shrike.test_p3_callback_budget import BatchedStream
    from hfmodem.tests.shrike.test_tx_latency_correction import _key

    s = _Session(role=arq.IRS)
    r = s.rx
    s.host.arq.cycle_long = True
    r._p3_row0 = 1_000_000
    r._p3_cycle_n = 180000
    r._p3_span = rxfront._frame_span(placement.LONG_PATHS[1])
    r.sync.packet_level = 1
    end = r._p3_row0 + rxfront._packet_span(r._p3_span)
    # VE3KPG's selected reply epoch, not a claim about its RF acceptance.
    key_at = end + round(.1108 * onair.FS)
    live = BatchedStream(phase)
    ready = onair._p3_frame_ready(r, onair._p3_decode_deadline(live, key_at, 1920), 660)
    live.pos = r._p3_row0
    live.take_until(ready - live.holdback)
    # Reserve 8 ms for decoding and handling, above the observed 3.4 ms p95.
    live._advance(live.now + round(.008 * onair.FS))
    assert live.clamp_late(key_at) == 0
    lead = onair._prekey_lead(live, 1920)
    live.take_until(key_at - lead)
    live.wait_until(key_at - lead)
    live._advance(live.now + 72)
    assert live.clamp_late(key_at) == 0
    start, ptt = _key(live, key_at)
    assert start == key_at - live._lat - live.tx_latency_n
    assert live._dac_time(start) - ptt == pytest.approx(.04)


@pytest.mark.parametrize('budget', [False, True])
def test_long_prekey_miss_does_not_fall_through_to_expensive_reader(monkeypatch, budget):
    s = _Session(role=arq.IRS)
    r = s.rx
    s.host.arq.cycle_long = True
    r._p3_row0 = 9120
    r._p3_cycle_n = 180000
    r.sync.packet_level = 1
    r._scan_origin, r._tracked_only = 0, True
    monkeypatch.setattr(r, '_p3_acquisition_fits', lambda _: budget)
    monkeypatch.setattr(r.sync, 'sl1_packet_at', lambda *args: None)
    def forbidden(*args, **kwargs):
        pytest.fail('bounded miss started a second decoder before the key')
    monkeypatch.setattr(r, '_read_p3_packet', forbidden)
    assert r._p3_packet(np.zeros(170000)) is None
    assert not s.packets
