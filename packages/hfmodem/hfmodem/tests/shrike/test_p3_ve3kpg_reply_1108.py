# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _CallbackBench, _Session


CAPTURE = (evidence.OPERATOR_DATA / 'captures'
           / 'onair-20261108T193952651697Z-622cfc95-4b75-43ac-999d-2e339a321c45'
           / 'stream.wav')
ENTRY_STATUS = 0x1a
DATA_STATUS = 0x5b
ACK_SAMPLE = 1614576
BREAKIN_SAMPLE = 1669868
DATA_START, DATA_END = 1623051, 1663514
FINAL_DEADLINE = 1680801


@pytest.mark.parametrize('after_tick,response_slot', [(False, 4), (True, 4), (True, 5)])
def test_entry_ack_suppression_belongs_to_its_emitted_slot(
        monkeypatch, after_tick, response_slot):
    s = _Session(entry_pending=True)
    a = s.host.arq
    a.cfg.traffic_sl = 1
    a.cfg.long_cycle = False
    a._next_seq = 3
    a._over_pending = True
    a._inflight = arq._Packet(status=ENTRY_STATUS, payload=b'', sl=1,
                              entry=True, sent_sl=1, sent_at_level=1)
    slot = 4
    emitted = []
    s.host.peer.slot = slot
    s.host.peer.slots_used = []

    def packet(sl, payload, status, breakin=False):
        if status == ENTRY_STATUS:
            return arq.REFUSED
        emitted.append((slot, sl, payload, status, breakin))
        s.host.peer.slots_used.append(slot)

    monkeypatch.setattr(s.host.peer, 'send_packet', packet)
    token = a.begin_receive_opportunity()
    if after_tick:
        s.host.tick()
    slot = response_slot
    s.host.peer.slot = slot
    s.host.on_rx_event(rxfront.Event(
        ACK_SAMPLE / onair.FS, 'cs', 'ACK', protocol=spec.Protocol.PACTOR3,
        cs=arq.CS_ACK))
    assert emitted == [(response_slot, 1, b'', DATA_STATUS, False)]
    if not after_tick:
        s.host.tick()
        assert len(emitted) == 1
    a.finish_receive_opportunity(token)
    slot = 5
    s.host.peer.slot = slot
    token = a.begin_receive_opportunity()
    s.host.tick()
    a.finish_receive_opportunity(token)
    expected = [(response_slot, 1, b'', DATA_STATUS, False)]
    if response_slot == 4:
        expected.append((5, 1, b'', DATA_STATUS, False))
    assert emitted == expected


@pytest.mark.parametrize('body_received', [False, True])
def test_peer_turn_does_not_inherit_the_previous_data_emission(monkeypatch, body_received):
    s = _Session(entry_pending=True)
    a = s.host.arq
    a.cfg.traffic_sl = 1
    a.cfg.long_cycle = False
    a._next_seq = 3
    a._over_pending = True
    a._inflight = arq._Packet(status=ENTRY_STATUS, payload=b'', sl=1,
                              entry=True, sent_sl=1, sent_at_level=1)
    s.host.peer.slot = 4
    s.host.peer.slots_used = []

    def packet(sl, payload, status, breakin=False):
        if status == ENTRY_STATUS:
            return arq.REFUSED
        s.host.peer.slots_used.append(s.host.peer.slot)

    monkeypatch.setattr(s.host.peer, 'send_packet', packet)
    token = a.begin_receive_opportunity()
    s.host.tick()
    s.host.on_rx_event(rxfront.Event(
        ACK_SAMPLE / onair.FS, 'cs', 'ACK', protocol=spec.Protocol.PACTOR3,
        cs=arq.CS_ACK))
    assert s.host.peer.slots_used == [4]
    a.finish_receive_opportunity(token)
    s.host.peer.slot = 5
    token = a.begin_receive_opportunity()
    s.host.on_rx_event(rxfront.Event(
        BREAKIN_SAMPLE / onair.FS, 'cs', 'BREAK-IN', protocol=spec.Protocol.PACTOR3,
        cs=arq.CS_BREAKIN))
    assert a.role == arq.IRS
    if body_received:
        s.host.on_rx_event(rxfront.Event(
            1669874 / onair.FS, 'packet', 'CHANGEOVER', protocol=spec.Protocol.PACTOR3,
            cs=arq.CS_BREAKIN, packet=(1, 0, b'RMS', True), breakin=True,
            carrier_swapped=True, cycle_long=False))
        assert s.host.rcvd_total == 3
    s.host.tick()
    a.finish_receive_opportunity(token)
    assert s.host.peer.sent == [('cs', arq.CS_ACK if body_received else arq.CS_REQUEST)]


def reply_window(status=DATA_STATUS, sent_at_level=1):
    s = _Session()
    a = s.host.arq
    a._inflight = arq._Packet(status=status, payload=b'', sl=1,
                              sent_sl=1, sent_at_level=sent_at_level)
    g = onair._MasterGrid(1382817, 60000, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.keyed_slot = 4
    g._p3_keyed_reply = (4, DATA_START, DATA_END, 60000)
    g.d_ref_n, g.d_n = 46080, 5038
    s.host.peer.raster = g
    s.rx.new_cycle()
    return s, g


@pytest.mark.parametrize('status,sent_at_level', [(0x1b, 1), (DATA_STATUS, 0)])
def test_cold_data_reply_without_a_keyed_invitation_retains_reserve(status, sent_at_level):
    s, g = reply_window(status, sent_at_level)
    until = s.rx.control_collect_until(FINAL_DEADLINE, DATA_END, g)
    assert until == FINAL_DEADLINE - round(s.rx.P3_ANSWER_ACQUIRE_RESERVE_S * onair.FS)


def test_first_data_reply_keeps_the_recorded_breakin_head(monkeypatch):
    evidence.require_input(CAPTURE.with_name(
        'stream-l7278-20261108T1940Z-s00000000000000000000.flac'))
    s, g = reply_window()
    until = s.rx.control_collect_until(FINAL_DEADLINE, DATA_END, g)
    live = _CallbackBench(seconds=0, holdback=916, lat_in=788)
    live.LAT = live._lat = 1268
    live.now = until
    s.host.peer.live, s.host.peer.slot = live, 5
    s.host.peer.key_instant = lambda raster, slot: raster.boundary(slot)
    delivered_end = live.samples
    audio = wav.read(CAPTURE, start=DATA_END, stop=delivered_end).astype(np.float32)
    acquired = []
    acquire = s.rx._acquire_answer

    def acquiring(*args):
        acquired.append(True)
        return acquire(*args)

    monkeypatch.setattr(s.rx, '_acquire_answer', acquiring)
    s.rx.control_signal_in(audio, DATA_END, g)
    assert [(event.kind, event.cs, round(event.t * onair.FS))
            for event in s.events] == [('cs', arq.CS_BREAKIN, BREAKIN_SAMPLE)]
    assert not acquired
