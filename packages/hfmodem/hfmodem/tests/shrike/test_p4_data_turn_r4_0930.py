# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Round-4 pins: unaccepted opening lifetime and the 0210 word-yield turn."""
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, p4breakintx
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4turn import P4Turn, State
from hfmodem.tests.shrike.test_p4turn import FS, machine, word

CYCLE, T = 60000, 44160


@pytest.mark.parametrize('retry', ['silence', 'deferred'])
def test_unaccepted_c0_stays_identical_after_long_ladder_restart(retry):
    m, a = machine(data='body', ladder='long')
    a.on_host_data(b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;' + b'proposal'*10)
    m._begin_data_turn(0, State.ISS_DATA_ACK)
    original = m.keyed()
    for _ in range(8):
        due = m.last.stream_sample+CYCLE
        if retry == 'silence':
            m.silence(due-m.lead)
        else:
            m._retry_data(due)
        p = m.keyed()
        if p.kind == Kind.SHORT:
            break
        assert (p.kind, p.status, p.payload) == (original.kind, original.status, original.payload)
    assert p.value == 13 and m.counter == 0
    for ack in ('X27', 'U27'):
        m.observe(word(ack, p.stream_sample+T, responds_to=p.stream_sample))
        p = m.keyed()
    assert (p.kind, p.status, p.payload) == (Kind.BODY, 0, original.payload)
    m.observe(word('X27', p.stream_sample+T, responds_to=p.stream_sample))
    p = m.keyed()
    assert (p.kind, p.status, len(p.payload)) == (Kind.NORMAL, 1, 40)



def test_late_weak_counter_zero_ack_retries_on_future_raster():
    m, a = machine(data='body', ladder='none', prekey_lead=2131/FS)
    a.on_host_data(b'packet')
    m._data(0, State.ISS_DATA_ACK)
    m.keyed()
    reg = word('X27', round(.9426*FS), responds_to=0)
    reg = replace(reg, observation=replace(reg.observation, weak=True))
    m.observe(reg, now=CYCLE-100)
    p = m.decide(CYCLE-100)
    assert (p.kind, p.payload, p.status, p.stream_sample) == (Kind.BODY, b'packet', 0, 2*CYCLE)
    assert a._buffer_raw == 6 and m.counter == 0
    m.keyed()
    early = word('X27', 2*CYCLE+round(.9426*FS), partial=True, responds_to=2*CYCLE)
    early = replace(early, observation=replace(early.observation, weak=True))
    m.observe(early)
    m.silence(3*CYCLE-m.lead)
    m.observe(replace(early, observation=replace(early.observation, provisional=False),
                      outcome=Registered.CONFIRMED, replaced=early.observation), now=3*CYCLE-100)
    assert a._buffer_raw == 0 and m.counter == 1

def word_yield():
    m, a = machine(data='body', ladder='none', role='irs')
    a.on_host_data(b';FW: W9SSJ\r' + b'proposal'*10)
    m._body(0, 0x40, State.IRS_BREAKIN)
    m.keyed()
    m.observe(word('V27', T, responds_to=0))
    return m, a, m.keyed()


def test_word_yield_sends_changeover_data_then_parity_accepted_normal():
    m, a, p = word_yield()
    assert (p.kind, p.status, p.stream_sample) == (Kind.BODY, 0x40, CYCLE)
    assert p.payload.startswith(b';FW:') and m.counter == 0
    original = a._buffer_raw
    m.observe(word('X27', p.stream_sample+T, responds_to=p.stream_sample))
    q = m.keyed()
    assert (q.kind, q.status) == (Kind.NORMAL, 1)
    assert a._buffer_raw == original-len(p.payload)


@pytest.mark.parametrize('name', ['V27', 'T27', 'U27', 'Z27'])
def test_word_yield_uses_refusal_or_ordinary_retry_budget(name):
    m, a, p = word_yield()
    original = a._buffer_raw
    first = p
    copies = 2 if name == 'Z27' else 8
    for i in range(copies):
        m.observe(word(name, p.stream_sample+T, responds_to=p.stream_sample))
        m.observe(word(name, p.stream_sample+T+1, responds_to=p.stream_sample,
                       outcome=Registered.DUPLICATE))
        q = m.keyed()
        assert m.counter == 0 and a._buffer_raw == original
        if i < copies-1:
            assert (q.kind, q.status, q.payload) == (first.kind, first.status, first.payload)
            assert m.data_repeats == i+1
        else:
            assert (q.kind, q.status, q.payload, q.state) == (Kind.BODY, 0x20, b'', State.ISS_CLOSE)
            assert not m.outbox.inflight and len(a._outbuf) == original
        p = q


@pytest.mark.parametrize('invalid', ['provisional', 'weak', 'unexpected', 'expired', 'unassociated'])
def test_word_yield_refusal_budget_excludes_unconfirmed_answers(invalid):
    m, _, p = word_yield()
    reg = word('V27', p.stream_sample+T, responds_to=p.stream_sample)
    if invalid in ('provisional', 'weak', 'unexpected'):
        reg = replace(reg, observation=replace(reg.observation, **{invalid: True}))
    elif invalid == 'expired':
        reg = replace(reg, outcome=Registered.EXPIRED)
    else:
        reg = replace(reg, responds_to=None)
    m.observe(reg)
    assert not m._data_refusals


def test_0210_full_stream_driver_stops_at_word_yield_retry_divergence(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    capture = root/'captures/onair-0930-0210'
    # Required evidence: missing audio must fail this pin, never silently skip it.
    require_input(capture/'stream.flac')
    require_input(capture/'p4-probe.json')
    require_input(root/'working/p4-plan-0930/stall-bisect/drive.py')
    probe = json.loads((capture/'p4-probe.json').read_text())
    opening = probe['emissions'][0]
    request = next(e for e in probe['emissions'] if e.get('status') == 0x40)
    proposal = b''.join(bytes.fromhex(e['payload']) for e in probe['emissions'] if e.get('payload'))
    run = onair._run_p4_link
    results, sent, waves = [], [], []
    class FirstDivergentKey(Exception):
        pass
    keyed = P4Turn.keyed
    def track(turn, at=None):
        p = keyed(turn, at)
        sent.append(p)
        if p.payload:
            np.testing.assert_array_equal(waves[-1], p4breakintx.changeover_packet(p.payload, p.status))
        if p.stream_sample >= request['audio_start'] + 3*CYCLE:
            raise FirstDivergentKey
        return p
    monkeypatch.setattr(P4Turn, 'keyed', track)
    def drive(tx, stream, raster):
        take = stream.take_until
        queued = False
        def feed(end):
            nonlocal queued
            if not queued and end >= request['audio_start']-round(1.3*FS):
                tx.p4_link_host.arq.on_host_data(proposal)
                queued = True
            return take(end)
        stream.take_until = feed
        transmit = tx._tx
        def tx_audio(audio, what):
            waves.append(audio.copy())
            transmit(audio, what)
        tx._tx = tx_audio
        with pytest.raises(FirstDivergentKey):
            run(tx, stream, raster)
        results.append(tx)
    monkeypatch.setattr(onair, '_run_p4_link', drive)
    monkeypatch.setattr(sys, 'argv', ['drive.py', str(capture/'stream.flac'),
        str(opening['audio_start'] % CYCLE), str(opening['audio_start']), '1',
        str(opening['audio_end']/FS), '60'])
    runpy.run_path(str(root/'working/p4-plan-0930/stall-bisect/drive.py'), run_name='__main__')
    tx, = results
    print('REPLAY SUMMARY '+json.dumps(dict(
        statuses=[p.status for p in sent if p.payload],
        data_times=[round(p.stream_sample/FS, 3) for p in sent if p.payload],
        buffered=tx.p4_link_host.arq._buffer_raw, reason=tx.p4_probe.reason)))
    before = [p for p in sent if p.stream_sample < request['audio_start']
              and p.current_slot_accepted]
    recorded = [e for e in probe['emissions'][1:] if e['audio_start'] < request['audio_start']]
    expected = sorted([(e['word'], e['audio_start']) for e in recorded]
                      + [('X27', 751861), ('V27', 1111868)], key=lambda e: e[1])
    assert [p.word for p in before] == [word for word, _ in expected]
    assert [p.stream_sample for p in before] == [at for _, at in expected]
    bodies = [p for p in sent if p.payload]
    assert len(bodies) == 3
    assert all((p.kind, p.status, p.payload) == (Kind.BODY, 0x40, proposal[:31]) for p in bodies)
    assert [p.stream_sample for p in bodies] == [request['audio_start']+i*CYCLE for i in (1, 2, 3)]
    assert not any(p.state in (State.ISS_HOLD, State.ISS_CLOSE) for p in sent)
    assert tx.p4_link_host.arq._buffer_raw == len(proposal) == 106
    assert bytes(tx.p4_link_host.arq._outbuf) == proposal
    # The third data key replaces the historical close. No fixed peer audio
    # after that divergence is used as evidence for a counterfactual recovery.
    assert bodies[-1].stream_sample == request['audio_start']+3*CYCLE
