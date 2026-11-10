# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""ISS-ACK-RULE: 0117/0127 repeated X27, 0058 seed, and counted packet form."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, p4shorttx, p4words
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4receiver import Emission, Role
from hfmodem.shrike.p4turn import P4Turn, State
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_data_turn_0930 import CYCLE, T, receive, sending
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
from hfmodem.tests.shrike.test_p4_breakin_v27_0929 import GRID
from hfmodem.tests.shrike import p4_stock_peer as stock


def test_iss_word_parity_property():
    for counter in range(4):
        for name in ('X27', 'V27', 'T27', 'U27', 'Z27', 'Y27'):
            m, a = machine(data='body', ladder='none')
            a.on_host_data(b'packet')
            m.counter = counter
            m._data(0)
            m.keyed()
            m.observe(word(name, T, responds_to=0), now=T+12000)
            accepted = name == ('V27' if counter & 1 else 'X27')
            assert m.counter == (counter+accepted) % 4, (counter, name)
            assert a._buffer_raw == (0 if accepted else 6), (counter, name)
            assert m.outbox.inflight == (b'' if accepted else b'packet')


def test_0058_v27_seed_never_delivers_counter_zero():
    m, a = sending(b'proposal'*15)
    original = a._buffer_raw
    for _ in range(12):
        last = m.last
        m.observe(word('V27', last.stream_sample+T, responds_to=last.stream_sample),
                  now=last.stream_sample+T+12000)
        assert m.counter == 0 and a._buffer_raw == original
        assert m.state != State.ISS_HOLD
        p = m.decide(last.stream_sample+CYCLE-m.lead)
        if m.state in (State.DOWN, State.ISS_CLOSE) or p and p.state == State.ISS_CLOSE:
            break
        assert p and p.payload == last.payload and p.status == 0
        m.keyed()
    else:
        pytest.fail('V27 resend loop exceeded data budget')


def test_two_z27_refusals_close_without_committing():
    m, a = sending()
    for i in range(2):
        last = m.last
        m.observe(word('Z27', last.stream_sample+T, responds_to=last.stream_sample),
                  now=last.stream_sample+T+12000)
        # A duplicate registration cannot spend another refusal.
        m.observe(word('Z27', last.stream_sample+T, responds_to=last.stream_sample,
                       outcome=Registered.DUPLICATE))
        p = m.decide(last.stream_sample+CYCLE-m.lead)
        assert m.counter == 0 and a._buffer_raw == 6
        if i == 0:
            assert p.payload == b'F> 3B\r' and p.status == 0
            m.keyed()
        else:
            assert p.state == State.ISS_CLOSE and p.status == 0x20 and not p.payload
            assert not m.outbox.inflight and bytes(a._outbuf) == b'F> 3B\r'


@pytest.mark.parametrize('answered', [False, True])
def test_late_wrong_parity_never_commits(answered):
    m, a = sending()
    early = word('V27', T, partial=True, responds_to=0)
    m.observe(early)
    if answered:
        m.silence(CYCLE-m.lead)
        if m._next is not None:
            m.keyed()
    m.observe(word('V27', T, outcome=Registered.CONFIRMED, replaced=early.observation,
                   answered=CYCLE if answered else None, responds_to=0), now=CYCLE+100)
    assert m.counter == 0 and a._buffer_raw == 6 and m.outbox.inflight == b'F> 3B\r'


def test_body_then_normals_wrap_counter_and_restart_next_turn():
    m, a = sending(bytes(range(32, 122))*3)
    sent = []
    for i in range(6):
        p = m.last
        sent.append(p)
        assert p.kind == (Kind.BODY if i == 0 else Kind.NORMAL)
        assert p.status == i % 4
        assert len(p.payload) == (31 if i == 0 else 40)
        m.observe(word('V27' if i & 1 else 'X27', p.stream_sample+T,
                       responds_to=p.stream_sample), now=p.stream_sample+T+12000)
        m.keyed()
    # A checked empty reply takes the peer's turn; later grant starts ours at 0.
    p = receive(m, m.last.stream_sample+T, responds_to=m.last.stream_sample)
    assert p.state == State.ISS_GRANT
    m.keyed()
    receive(m, p.stream_sample+T, b'FS Y\r')
    m.keyed()
    a.on_host_data(b'next turn')
    m._body(m.last.stream_sample+CYCLE, 0x40, State.IRS_BREAKIN)
    request = m.keyed()
    p = receive(m, request.stream_sample+T, responds_to=request.stream_sample)
    assert (p.kind, p.status) == (Kind.BODY, 0)


@pytest.mark.parametrize('arm', ['0117', '0127'])
def test_recorded_x27_repeat_through_real_iss_data_driver(monkeypatch, arm):
    root = Path(__file__).resolve().parents[5]
    capture = root/f'captures/onair-0930-{arm}'
    if not (capture/'stream.flac').exists():
        pytest.skip(f'local {arm} recording absent')
    probe = json.loads((capture/'p4-probe.json').read_text())
    recorded = onair.session.load_wav(str(capture/'stream.flac'), FS)
    bodies = [e for e in probe['emissions'] if e.get('payload')]
    first = bodies[0]['audio_start']
    request = next(e['audio_start'] for e in probe['emissions'] if e.get('status') == 0x40)
    # Keep the grant, both X27s and first Z27; answer the c1 retry at +3 cycles
    # with a scripted V27, so the continuation must finally send counter 2.
    cutoff = first+3*CYCLE
    audio = np.zeros(first+7*CYCLE)
    audio[:cutoff] = recorded[:cutoff]
    for e in probe['emissions']:
        audio[e['audio_start']:min(e['audio_end'], len(audio))] = 0
    stream = Stream(audio)
    stream.pos = request+38286
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = stream.pos-FS, audio[stream.pos-FS:stream.pos].copy()
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_normal_ack = 'parity'
    tx.p4_link_options.p4_link_timeout = 9
    tx.p4_link_host.arq.on_host_data(b''.join(bytes.fromhex(e['payload']) for e in bodies[:3])+b'end'*30)
    rx = tx.sessrx.p4
    # The old seed was the recorded X27's absolute word reference.
    rx.cfo_hz = 4.-p4words.CARRIER_HZ['X27']
    init = P4Turn.__init__
    def begin(turn, *args, **kw):
        init(turn, *args, **{**kw, 'role': 'irs'})
        turn._body(request, 0x40, State.IRS_BREAKIN)
        turn.keyed()
        rx.note_emission(Emission(Kind.BODY, Role.ISS, request, 38286/FS))
    monkeypatch.setattr(P4Turn, '__init__', begin)
    transmit = tx._tx
    waves, sent = [], []
    def transmit_audio(wave, label):
        waves.append(wave.copy())
        transmit(wave, label)
        stream.pos = tx.tx_end
    tx._tx = transmit_audio
    keyed = P4Turn.keyed
    def peer(turn, at=None):
        p = keyed(turn, at)
        if p.payload:
            sent.append(p)
            if p.kind == Kind.NORMAL:
                np.testing.assert_array_equal(waves[-1], p4shorttx.short_packet(p.payload, p.status))
            if abs(p.stream_sample-(first+3*CYCLE)) < FS//100:
                rx.cfo_hz = 0.
                wave = stock.word('V27')
                answer = p.stream_sample+T
                audio[answer:answer+len(wave)] = wave
        return p
    monkeypatch.setattr(P4Turn, 'keyed', peer)
    onair._run_p4_link(tx, stream, GRID)
    assert len(sent) >= 5
    assert [p.status for p in sent[:5]] == [0, 1, 1, 1, 2]
    assert sent[1].payload == sent[2].payload == sent[3].payload
    assert abs(sent[3].stream_sample-(first+3*CYCLE)) < FS//1000
    assert sent[0].kind == Kind.BODY and all(p.kind == Kind.NORMAL for p in sent[1:])
    assert [p.stream_sample-sent[0].stream_sample for p in sent[:5]] == [i*CYCLE for i in range(5)]


@pytest.mark.parametrize('arm,counter,index,reply', [('0058', 0, 0, 'V27'), ('0117', 2, 2, 'Z27')])
def test_recorded_refusals_end_data_loop_without_settlement(monkeypatch, arm, counter, index, reply):
    capture = Path(__file__).resolve().parents[5]/f'captures/onair-0930-{arm}'
    if not (capture/'stream.flac').exists():
        pytest.skip(f'local {arm} recording absent')
    probe = json.loads((capture/'p4-probe.json').read_text())
    audio = onair.session.load_wav(str(capture/'stream.flac'), FS)
    field = [e for e in probe['emissions'] if e.get('payload')][index]
    start = field['audio_start']
    payload = bytes.fromhex(field['payload'])
    for e in probe['emissions']:
        audio[e['audio_start']:e['audio_end']] = 0
    stream = Stream(audio)
    stream.pos = start+38286
    tx = Tx(stream)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = stream.pos-FS, audio[stream.pos-FS:stream.pos].copy()
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_link_timeout = 15
    tx.p4_link_host.arq.on_host_data(payload)
    rx = tx.sessrx.p4
    rx.cfo_hz = 4.
    init, keyed = P4Turn.__init__, P4Turn.keyed
    turns, sent = [], []
    def begin(turn, *args, **kw):
        init(turn, *args, **kw)
        turns.append(turn)
        turn.counter = counter
        turn._data(start)
        turn.keyed()
        rx.note_emission(Emission(Kind.BODY, Role.ISS, start, 38286/FS))
    monkeypatch.setattr(P4Turn, '__init__', begin)
    def track(turn, at=None):
        p = keyed(turn, at)
        sent.append(p)
        return p
    monkeypatch.setattr(P4Turn, 'keyed', track)
    transmit = tx._tx
    def transmit_audio(wave, label):
        transmit(wave, label)
        stream.pos = tx.tx_end
    tx._tx = transmit_audio
    onair._run_p4_link(tx, stream, GRID)
    m = turns[0]
    assert m.counter == counter and tx.p4_link_host.arq._buffer_raw == len(payload)
    assert all(p.status == counter and p.payload == payload for p in sent if p.payload)
    assert not any(p.state == State.ISS_HOLD for p in sent)
    assert sum(r.get('word') == reply and not r['provisional'] for r in tx.p4_probe.heard) >= 2
    if reply == 'Z27':
        assert len([p for p in sent if p.payload]) == 2  # original + one resend
        close = next(p for p in sent if p.state == State.ISS_CLOSE)
        assert close.status == 0x20 and close.stream_sample == start+2*CYCLE
    else:
        assert m.state == State.DOWN
        close = next(p for p in sent if p.state == State.ISS_CLOSE)
        assert close.kind == Kind.BODY and close.status == 0x20 and not close.payload
        assert close.stream_sample == start+8*CYCLE


@pytest.mark.parametrize('speculative', [False, True])
def test_deferred_iss_data_ack_resolution_requires_parity(speculative):
    m, a = machine(data='body', ladder='long')
    a.on_host_data(b'packet')
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    early = word('X27' if speculative else 'V27', T, partial=True, responds_to=0)
    m.observe(early)
    if speculative:
        m.keyed()
    m.observe(word('V27', T, outcome=Registered.CONTRADICTED,
                   replaced=early.observation, answered=CYCLE, responds_to=0), now=CYCLE+100)
    assert m.counter == 0 and a._buffer_raw == 6 and m.outbox.inflight == b'packet'


def test_hold_seed_and_wrong_parity_idle_do_not_advance():
    m, _ = sending()
    m.observe(word('X27', T, responds_to=0), now=T+12000)
    idle = m.keyed()
    m.observe(word('V27', idle.stream_sample+T, responds_to=None), now=idle.stream_sample+T+12000)
    assert m.counter == 1
    m.observe(word('X27', idle.stream_sample+T+1, responds_to=idle.stream_sample),
              now=idle.stream_sample+T+12001)
    assert m.counter == 1
