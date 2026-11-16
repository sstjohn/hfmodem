# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A pending CS6 bounds both geometries in front of the owed reply's key.

0923-1948 WS8EOC-80: CS6 at slot 21, the peer went long on the next row, and
its long replies were due at slots 24, 27, 30 and 33. Each walk read the long
frame (9.66 ms) and then a short read of the whole grown window (37.58 ms) in
front of the endpoint's key, against 23.0 ms of room: SLOT 24 IS GONE +24.2 ms,
then 27, 28, 29, 33. Here the grid's slot 0 is the arm's slot 21.
"""
import json
from types import SimpleNamespace

import numpy as np

from hfmodem.core import wav

from hfmodem.shrike import arq, onair, p3frame, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _CallbackBench, _Rig, _Session
from hfmodem.tests.shrike.p3_cycle_support import (
    LONG_MS, SHORT_MS, ROOM_MS, _walks)

FS = 48000


def test_each_long_reply_slot_is_reached_with_bounded_reads_in_front_of_it(
        tmp_path, monkeypatch):
    s, tx, g, walks = _walks(tmp_path, monkeypatch)
    assert [end for _, end, _ in walks] == [3, 6, 9, 12]  # the arm's 24..33
    assert s.host.arq.cycle_command_emitted and not s.host.arq.cycle_long
    for start, end, reads in walks:
        gate = [r for r in reads if r[0] > tx.key_instant(g, end - 1)]
        assert [r[3] for r in gate] == [7200 + (end - 3) * 60000,
                                      127200 + (end - 3) * 60000]
        spent = sum(LONG_MS if r[3] is not None else SHORT_MS for r in gate)
        assert spent <= ROOM_MS


def test_probe_reads_take_only_the_frame_they_can_decode(tmp_path, monkeypatch):
    s, _, _, walks = _walks(tmp_path, monkeypatch)
    span, cycle_n = onair._p3_receive_geometry(s.rx)
    lead = p3frame.DATA_OFFSET * rxfront.SPS + 2400
    for _, _, reads in walks:
        for _, origin, size, row in reads:
            if row is None:
                assert size <= cycle_n + rxfront._packet_span(span) + lead
            else:
                assert origin == row - lead


def test_the_transmitter_keys_every_long_reply_slot(tmp_path, monkeypatch):
    tx = _walks(tmp_path, monkeypatch)[1]
    assert tx.slots_used == [3, 6, 9, 12] and not tx.refused


def test_recorded_long_repeat_remains_audible_until_its_control_slot(
        tmp_path, monkeypatch):
    cap = evidence.CAPTURES / 'onair-0924-0455'
    evidence.require_input(cap / 'stream.wav')
    opening = json.loads(evidence.require_input(cap / 'tx_10.json').read_text())
    audio = wav.read(cap / 'stream.wav', stop=2200000).astype(np.float32)
    masks = []
    for path in cap.glob('tx_*.json'):
        burst = json.loads(path.read_text())
        start, end = burst['ptt_up'], burst['audio_end']
        masks.append((start, end))
        audio[max(0, start):end] = 0
    live = _CallbackBench(seconds=0, holdback=916, lat_in=788)
    live.LAT = live._lat = 1268
    live.tx_latency_n = live.geometry_n = 960
    live.audio, live.limit = audio, len(audio)
    live.now = opening['audio_end'] + 788
    live.flush_to(opening['audio_end'])
    s = _Session(role=arq.ISS, entry_pending=True)
    a = s.host.arq
    a.dxcall = 'WS8EOC'
    a.cfg.speed_up, a.cfg.long_cycle = 'auto', True
    a.cfg.p3_max_down, a.cfg.p3_max_try = 6, 2
    a.cfg.traffic_sl, a.cfg.repeat_gear = 1, 0
    s.host.p3_mail_active = s.host.no_p3_fallback = s.host.p3_qrt_confirm = True
    g = onair._MasterGrid(543676, 60000, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.acquired = g.corroborated = True
    g.d_ref_n, g.d_n = 46080, 3674
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04,
                       record_audio=False)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.sessrx, tx.p4_probe = live, s.rx, None
    tx.defer_p3_cs, tx.breakin_due = True, s.host.breakin_due
    tx.p3_control_instant, tx.p3_control_waveform = 'peer', 'current'
    tx.p3_control_stagger, tx.p3_follow_offset = 'lead-12', 'all'
    tx.aim(g, 0)
    tx.tx_end = opening['audio_end']
    tx.last_dur = (opening['audio_end'] - opening['audio_start']) / FS
    tx.reply_clock = ReplyClock(entry_phase=opening['entry_phase'])
    a._next_seq = 3
    a._inflight = arq._Packet(status=0x1a, payload=b'', sl=1, entry=True,
                              sent_sl=1, sent_at_level=1)
    tx.seq_sent = [2]
    tx._p3_packet_swap_bias = g.shift(0)
    g.keyed_slot = 0
    g.keying(spec.Protocol.PACTOR3, extent_n=opening['samples'],
             entry_pending=True, entry_variant='template')
    g._p3_keyed_reply = (0, opening['audio_start'], opening['audio_end'], 60000)
    s.rx._p3_presence_gaps = masks
    s.rx.skip(opening['audio_end'] / FS)
    emissions, delivered = [], []
    transmit = live.transmit

    def keyed(*args, **kwargs):
        delivered.append((tx.slot, s.rx._p3_delivered_at))
        result = transmit(*args, **kwargs)
        live.audio[live.keyed_at:result[1]] = 0
        emissions.append(tx.slot)
        return result

    controls = []
    render = tx._render_p3_control

    def control(word, **kwargs):
        before = len(emissions)
        result = render(word, **kwargs)
        if len(emissions) > before:
            controls.append((tx.slot, word))
        return result

    cycle = s.rx.new_cycle
    closed = []

    def next_cycle():
        if emissions and emissions[-1] >= 22:
            closed.append(not a.cycle_command_emitted)
            a.on_host_abort()
        return cycle()

    monkeypatch.setattr(live, 'transmit', keyed)
    monkeypatch.setattr(s.rx, 'new_cycle', next_cycle)
    monkeypatch.setattr(tx, '_render_p3_control', control)
    monkeypatch.setattr(a, 'on_host_connect', lambda *args: None)
    monkeypatch.setattr(onair, '_MasterGrid', lambda *args, **kwargs: g)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *args, **kwargs: None)
    monkeypatch.setattr(onair.time, 'monotonic', lambda: live.sample_now() / FS)
    monkeypatch.setattr(onair.time, 'perf_counter', lambda: live.sample_now() / FS)
    monkeypatch.setattr(onair, '_call_clock', lambda: live.sample_now() / FS)
    args = onair.build_parser().parse_args(['--mycall', 'W9SSJ', '--hold', '100'])
    args.over = True
    identity = SimpleNamespace(cw_call='W9SSJ',
                               closing_required=lambda *args, **kwargs: False)
    onair._drive_session(args, tx=tx, host=s.host, sessrx=s.rx, live=live, rig=None,
                         dx='WS8EOC', reply=None, outdir=tmp_path, identity=identity)
    assert [slot for slot in emissions if 19 <= slot <= 22] == [19, 22]
    assert dict(controls)[19] == arq.CS_CYCLE_TOG
    assert dict(controls)[22] in (arq.CS_REQUEST, arq.CS_CYCLE_TOG)
    repeats = [ev for ev in s.packets if ev.cycle_long and ev.packet ==
               (1, 1, b'S via WS8EOC >\r', True)]
    assert len(repeats) == 2
    assert closed == [True]
    assert abs(dict(delivered)[22] - 1733165) <= rxfront.SPS
