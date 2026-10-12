# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""0058 proposal hand-back and k0003 empty acceptance / FS prefix credit."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4frame, p4shorttx
from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Emission, Role
from hfmodem.shrike.p4turn import P4Turn, State
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_data_turn_0930 import CYCLE, T, receive, sending
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.test_p4_normal_short_0928 import normal
from hfmodem.tests.shrike.test_p4_breakin_v27_0929 import GRID
from hfmodem.tests.shrike import p4_stock_peer as stock


@pytest.mark.parametrize('queued', [False, True])
def test_k0003_empty_reply_commits_f_gt_once_then_yields_and_receives_fs(queued):
    m, a = sending()
    settled, delivered = [], []
    original = a.settle_bytes
    def settle(data):
        settled.append(data)
        original(data)
    a.settle_bytes = settle
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    if queued:
        a.on_host_data(b'next field')
    grant = receive(m, T, responds_to=0)
    assert grant.state == State.ISS_GRANT and grant.status == 0x40 and not grant.payload
    assert settled == [b'F> 3B\r'] and m.counter == 1 and not m.outbox.inflight
    m.keyed()
    fs = receive(m, grant.stream_sample+T, b'FS YY\r', responds_to=grant.stream_sample)
    assert fs.kind == Kind.WORD
    m.keyed()
    # k0003: FS in a c0 body, then the same source in a c2 long.
    counted = replace(normal(fs.stream_sample+CYCLE, 0x42, b'FS YY\r'), kind=Kind.LONG)
    counted = replace(counted, frame=p4frame.decode_field(
        p4frame.encode_field(b'FS YY\r'.ljust(69, b'\x1e'), p4frame.Status.from_byte(0x42)),
        data_bytes=69))
    m.observe(RegisteredObservation(counted, Registered.NEW))
    assert delivered == [b'FS YY\r'] and settled == [b'F> 3B\r']


@pytest.mark.parametrize('suffix', [b'', b'additional source\r'])
def test_k0003_fs_prefix_credit_delivers_suffix_and_expires_on_next_counted_frame(suffix):
    m, _ = machine(role='irs', data='body', ladder='none')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    receive(m, FS, b'FS YY\r')
    m.keyed()
    counted = normal(3*FS, 0x42, b'FS YY\r'+suffix)
    m.observe(RegisteredObservation(counted, Registered.NEW))
    assert delivered == [b'FS YY\r'] + ([suffix] if suffix else [])
    m.observe(RegisteredObservation(replace(counted, stream_sample=4*FS), Registered.NEW))
    assert delivered == [b'FS YY\r'] + ([suffix] if suffix else [])
    # A later counter carrying equal application text must remain new data.
    later = normal(5*FS, 0x43, b'FS YY\r')
    m.observe(RegisteredObservation(later, Registered.NEW))
    assert delivered[-1] == b'FS YY\r' and delivered.count(b'FS YY\r') == 2


def test_k0003_prefix_credit_does_not_drop_a_mismatching_next_packet():
    m, _ = machine(role='irs', data='body', ladder='none')
    received = []
    m.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    receive(m, FS, b'FS YY\r')
    m.keyed()
    for k, data in enumerate((b'FF\r', b'FS YY\r')):
        m.observe(RegisteredObservation(normal((3+k)*FS, 2+k, data), Registered.NEW))
    assert received == [b'FS YY\r', b'FF\r', b'FS YY\r']


def test_normal_reply_to_last_body_commits_before_delivery():
    m, a = sending()
    received = []
    def deliver(data):
        assert a._buffer_raw == 0 and not m.outbox.inflight
        received.append(data)
    m.adapter.host = SimpleNamespace(deliver=deliver, supervisor=lambda *a: None)
    obs = normal(T, 0, b'FS Y\r')
    m.observe(RegisteredObservation(obs, Registered.NEW, responds_to=0), now=T+round(.81*FS))
    p = m.decide(2*T-m.lead)
    assert p.kind == Kind.WORD and received == [b'FS Y\r'] and m.counter == 1


def test_normal_idle_ack_is_counted_once_and_partial_ack_waits():
    m, _ = sending()
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    idle = m.keyed()
    assert idle.kind == Kind.NORMAL and idle.status == 1
    at = idle.stream_sample+T
    early = word('V27', at, partial=True, responds_to=idle.stream_sample)
    m.observe(early)
    assert m.decide(idle.stream_sample+CYCLE-m.lead) is None
    full = word('V27', at, outcome=Registered.CONFIRMED, replaced=early.observation,
                responds_to=idle.stream_sample)
    m.observe(full, now=at+12000)
    m.observe(replace(full, outcome=Registered.DUPLICATE), now=at+12001)
    p = m.decide(idle.stream_sample+CYCLE-m.lead)
    assert p.status == 2 and m.counter == 2 and p.stream_sample == idle.stream_sample+CYCLE


@pytest.mark.parametrize('reply', ['normal', 'breakin-normal', 'idle-ack-normal', 'last-body-normal'])
def test_0058_prompt_then_scripted_parity_delivery_idle_and_peer_reply(monkeypatch, reply):
    root = Path(__file__).resolve().parents[5]
    capture = root/'captures/onair-0930-0058'
    if not (capture/'stream.flac').exists():
        pytest.skip('local WS8EOC 0058 recording absent')
    probe = json.loads((capture/'p4-probe.json').read_text())
    recorded = onair.session.load_wav(str(capture/'stream.flac'), FS)
    fields = [e for e in probe['emissions'] if e.get('payload') and e['audio_start'] < 43*FS]
    assert [e['status'] for e in fields] == [0, 1, 2, 3]
    assert bytes.fromhex(fields[-1]['payload']).endswith(b'F> 4E\r')
    cutoff = fields[0]['audio_start']  # Only prompt and yield seed are recorded below.
    audio = np.zeros(65*FS)
    audio[:cutoff] = recorded[:cutoff]
    for e in probe['emissions']:
        audio[e['audio_start']:min(e['audio_end'],len(audio))] = 0
    stream = Stream(audio)
    stream.pos = 1483639  # after our V27, before the recorded c2/c3 prompt
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = stream.pos-FS, audio[stream.pos-FS:stream.pos].copy()
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_hold_cycles = 16
    tx.p4_link_options.p4_link_timeout = 23
    rx = tx.sessrx.p4
    rx.cfo_hz = 4.7
    rx.peer_onset = 1427858
    rx._peer_heads = [1367805, 1427858]
    rx.note_emission(Emission(Kind.WORD, Role.IRS, 1472018, .242))
    delivered, emitted, waves = [], [], []
    def deliver(data):
        delivered.append(data)
        if b'CMS via WS8EOC >' in data:
            tx.p4_link_host.arq.on_host_data(b''.join(bytes.fromhex(e['payload']) for e in fields))
    tx.p4_link_host.deliver = deliver
    init = P4Turn.__init__
    def begin(turn, *a, **kw):
        init(turn, *a, **{**kw, 'role': 'irs'})
        turn.first_body = False
        turn.acked_counter = 1
    monkeypatch.setattr(P4Turn, '__init__', begin)
    transmit = tx._tx
    def transmit_audio(wave, label):
        waves.append(wave.copy())
        transmit(wave, label)
        stream.pos = tx.tx_end
    tx._tx = transmit_audio
    keyed = P4Turn.keyed
    def place(at, wave):
        audio[at:at+len(wave)] = wave
    def peer(turn, at=None):
        p = keyed(turn, at)
        emitted.append(p)
        answer = p.stream_sample+T
        if p.payload:
            rx.cfo_hz = 0.
            if reply == 'last-body-normal' and p.payload.endswith(b'F> 4E\r'):
                place(answer, p4shorttx.short_packet(b'FS Y\r', 0))
            else:
                place(answer, stock.word('V27' if p.status & 1 else 'X27'))
        elif p.stream_sample >= cutoff:
            rx.cfo_hz = 0.
            if p.kind == Kind.NORMAL:
                np.testing.assert_array_equal(waves[-1], p4shorttx.short_packet(b'', p.status))
                idles = [e for e in emitted if e.kind == Kind.NORMAL and not e.payload]
                if reply == 'breakin-normal':
                    place(answer, stock.body(b'', 0x40))
                elif reply == 'idle-ack-normal' and len(idles) == 1:
                    place(answer, stock.word('V27' if p.status & 1 else 'X27'))
                else:
                    place(answer, p4shorttx.short_packet(b'FS Y\r', 0))
            elif p.state == State.ISS_HOLD and p.kind == Kind.BODY:
                # The old offer repeats the 0058 IRS request: the peer yields V27.
                place(answer, stock.word('V27'))
            elif p.state == State.ISS_GRANT:
                place(answer, p4shorttx.short_packet(b'FS Y\r', 0))
        return p
    monkeypatch.setattr(P4Turn, 'keyed', peer)
    onair._run_p4_link(tx, stream, GRID)
    outgoing = [e for e in emitted if e.payload]
    assert b''.join(p.payload for p in outgoing) == b''.join(bytes.fromhex(e['payload']) for e in fields)
    assert [p.status for p in outgoing] == [0x40, 1, 2]
    assert [p.kind for p in outgoing] == [Kind.BODY, Kind.NORMAL, Kind.NORMAL]
    assert abs(outgoing[0].stream_sample-fields[0]['audio_start']) <= FS//1000
    assert [p.stream_sample-outgoing[0].stream_sample for p in outgoing] == [k*CYCLE for k in range(3)]
    assert delivered.count(b'FS Y\r') == 1
    assert tx.p4_link_host.arq._buffer_raw == 0
    fs = next(r for r in tx.p4_probe.heard if r.get('kind') == Kind.NORMAL
              and bytes.fromhex(r.get('data','')).startswith(b'FS Y\r'))
    assert any(e.kind == Kind.WORD and abs(e.stream_sample-fs['onset_sample']-T) <= 1 for e in emitted)
    assert not any(fs['onset_sample'] < e.stream_sample < fs['onset_sample']+round(.805*FS) for e in emitted)
    if reply != 'last-body-normal':
        idles = [e for e in emitted if e.kind == Kind.NORMAL and not e.payload]
        assert idles[0].status == 3
        assert idles[0].stream_sample == outgoing[-1].stream_sample+CYCLE
        if reply == 'idle-ack-normal':
            assert idles[1].status == 0 and idles[1].stream_sample == idles[0].stream_sample+CYCLE
        if reply == 'breakin-normal':
            assert any(e.state == State.ISS_GRANT and not e.payload and e.status == 0x40 for e in emitted)
