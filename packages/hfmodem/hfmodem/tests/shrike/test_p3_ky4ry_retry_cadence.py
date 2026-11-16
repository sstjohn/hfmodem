# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.onair_support import _Session
from hfmodem.tests.shrike.p3_reply_support import duplex


def ky4ry_retry(tmp_path):
    grid = onair._MasterGrid(1112003, 60000, 8880, packet_n=38880,
                            cs_n=10080, d_max_n=6240)
    grid.protocol, grid.sending = spec.Protocol.PACTOR3, True
    grid.acquired = grid.corroborated = True
    grid._p3_peer = (4969283, 39120, 60000, 64)
    grid._p3_peer_confirmed = True
    grid.cycles = 113
    grid.note_peer_codeword(8162836, 10080, 'REQ', 'KY4RY',
                            protocol=spec.Protocol.PACTOR3)
    grid.cycles = 115
    contact = _Session(role=arq.ISS)
    tx = duplex(grid, tmp_path)
    tx.attach(contact.host)
    contact.host.peer = tx
    tx.sessrx = contact.rx
    contact.host.arq._inflight = arq._Packet(
        status=0, payload=b';FW', sl=1, breakin=True)
    contact.host.arq._next_seq = 1
    contact.host.arq._buffer_raw = 3
    tx.reply_clock = ReplyClock(pulse_epoch=8312003)
    tx.breakin_due = True
    tx.aim(grid, 121)
    tx.live.holdback, tx.live.lat_in = 916, 788
    tx.live.LAT = tx.live._lat = 1268
    tx.live.tx_latency_n = tx.live.geometry_n = 960
    tx.live.now = tx.live.pos = 8351332
    return tx, grid


def test_ky4ry_unplaced_retry_reserves_its_held_pulse(tmp_path):
    tx, grid = ky4ry_retry(tmp_path)
    assert grid.breakin_refusal(grid.peer_packet_end(121)) is not None
    assert tx._p3_retry_phase(grid, 121) is None
    planned = tx.key_instant(grid, 121)
    assert tx._place_breakin() == ' on the held grid'
    assert tx.boundary == 8372003
    assert 0 <= tx.boundary - planned < 1000


def test_ky4ry_unplaced_retry_keeps_next_owed_slot(tmp_path):
    tx, grid = ky4ry_retry(tmp_path)
    early = grid.boundary(121) - tx.key_instant(grid, 121)
    assert onair._keyable_slot(tx.live, grid, 121, 1920 + early) == 121
    tx.send_packet(1, b';FW', 0, breakin=True)
    assert not tx.refused
    assert tx.slots_used == [121]
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == 8372003
    assert tx.host.arq._inflight.payload == b';FW'


def test_ky4ry_post_tick_repeat_spends_advanced_slot_only(monkeypatch):
    contact = _Session(role=arq.ISS)
    a, peer = contact.host.arq, contact.host.peer
    a.cfg.traffic_sl = 1
    a.cfg.long_cycle = False
    a._inflight = arq._Packet(status=0, payload=b';FW', sl=1,
                              breakin=True, sent_sl=1, sent_at_level=1)
    a._next_seq = 1
    a._buffer_raw = 3
    peer.slot, peer.slots_used = 88, [88]
    emitted = []

    def packet(sl, payload, status, breakin=False):
        if peer.slot in peer.slots_used:
            peer.slot += 1
        peer.slots_used.append(peer.slot)
        emitted.append((peer.slot, sl, payload, status, breakin))

    monkeypatch.setattr(peer, 'send_packet', packet)
    token = a.begin_receive_opportunity()
    contact.host.on_rx_event(rxfront.Event(
        6380285 / onair.FS, 'cs', 'REQ', protocol=spec.Protocol.PACTOR3,
        cs=arq.CS_REQUEST))
    assert emitted == [(89, 1, b';FW', 0, True)]
    assert a._rx_this_cycle and a.unconfirmed_breakin
    a.finish_receive_opportunity(token)
    peer.slot = 90
    token = a.begin_receive_opportunity()
    contact.host.tick()
    a.finish_receive_opportunity(token)
    assert emitted == [(89, 1, b';FW', 0, True), (90, 1, b';FW', 0, True)]
    assert peer.slots_used == [88, 89, 90]
    assert a._inflight.payload == b';FW' and a._next_seq == 1
