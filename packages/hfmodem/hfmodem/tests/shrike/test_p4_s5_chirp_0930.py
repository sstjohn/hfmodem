# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""S5(e) pins from p4-probe heard rows; no recording or hardware required."""

import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, p4frame, p4turn
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4turn import machine

FIXTURES = Path(__file__).parent / 'fixtures/s5-chirp'


def recording(arm):
    return json.loads(require_input(FIXTURES / f'{arm}.json').read_text())


def observation(row):
    at = row['onset_sample']
    frame = None
    if 'status' in row:
        data = bytes.fromhex(row['data'])
        frame = p4frame.decode_field(p4frame.encode_field(
            data, p4frame.Status.from_byte(row['status'])), data_bytes=len(data))
    return Observation(Kind(row['kind']), FS, 0, at, at,
                       3.28 if frame else .758, 0., .8, frame=frame,
                       value=row.get('value'), header=row.get('header'),
                       walsh_peak=64 if frame is None else None, placed=frame is None)


@pytest.mark.parametrize('arm,count', [('0930-0012', 40), ('0929-1845', 92)])
def test_recorded_chirps_get_x27_in_each_available_answer_slot(arm, count):
    tape = recording(arm)
    m, _ = machine(role='irs')
    chirps = 0
    for row in tape['rows']:
        obs = observation(row)
        due = obs.stream_sample + round((3.4 if obs.frame else .92)*FS)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=due-m.lead)
        proposal = m.decide(due-m.lead)
        assert proposal is not None
        assert proposal.word == ('U27' if obs.value == 53 else 'X27')
        assert proposal.stream_sample == due
        m.keyed()
        chirps += obs.kind == Kind.CHIRP
    assert chirps == count


def test_condensed_0012_three_v53_then_forty_chirps():
    rows = recording('0930-0012')['rows']
    controls = [r for r in rows if r.get('value') == 53][:3]
    chirp = next(r for r in rows if r['kind'] == 'chirp')
    m, _ = machine(role='irs')
    for row in controls + [dict(chirp, onset_sample=round((48.508+k*3.75)*FS)) for k in range(40)]:
        obs = observation(row)
        due = obs.stream_sample + round((3.4 if obs.frame else .92)*FS)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=due-m.lead)
        p = m.decide(due-m.lead)
        assert p.word == ('X27' if obs.frame else 'U27')
        assert p.stream_sample == due
        m.keyed()


def test_0012_seven_late_chirps_defer_x27_one_cycle():
    tape = recording('0930-0012')
    rows = {r['onset_sample']: r for r in tape['rows']}
    assert len(tape['deferred_chirps']) == 7
    for at in tape['deferred_chirps']:
        m, _ = machine(role='irs')
        obs = observation(rows[at])
        due = at+round(3.4*FS)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=due-m.lead+1)
        # B: still before its due, the chirp is answered in its own slot; the
        # one-cycle deferral now follows the driver's deadline refusal.
        current = m.decide(due-m.lead+1)
        assert (current.word, current.stream_sample) == ('X27', due)
        m.refused('deadline')
        p = m.decide(due+round(3.75*FS)-m.lead)
        assert p.word == 'X27'
        assert p.stream_sample == due+round(3.75*FS)


@pytest.mark.parametrize('arm', ['0929-1903', '0929-2052', '0929-2352'])
def test_recorded_eighth_v53_is_answered_without_dropping_link(arm):
    m, _ = machine(role='irs')
    rows = [r for r in recording(arm)['rows'] if r.get('value') == 53]
    assert len(rows) == 8
    for row in rows:
        obs = observation(row)
        due = obs.stream_sample+round(.92*FS)
        m.observe(RegisteredObservation(obs, Registered.NEW), now=due-m.lead)
        assert m.state != State.DOWN
        p = m.decide(due-m.lead)
        assert (p.word, p.stream_sample) == ('U27', due)
        m.keyed()
    assert m.reason is None


@pytest.mark.parametrize('termination', ['silence', 'timeout'])
def test_driver_ends_control_train_only_on_silence_or_timeout(monkeypatch, termination):
    rows = [r for r in recording('0929-1903')['rows'] if r.get('value') == 53]
    start = rows[0]['onset_sample']
    end = rows[-1]['onset_sample']
    live = Stream(np.zeros(end+round(95*FS)))
    live.pos = start-round(.1*FS)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 12. if termination == 'timeout' else 90.
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, start, .330, 1.25, 0.)
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **dict(kw, role='irs')))
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)

    def packet(audio, origin, partial=False, long_fallback=False):
        row = next((r for r in rows if abs(r['onset_sample']-rx.clock.next_due) < FS//10), None)
        if row is None and termination == 'timeout':
            row = dict(rows[-1], onset_sample=rx.clock.next_due)
        if row is None or origin+len(audio) < row['onset_sample']+round(.8*FS):
            return None
        return observation(row)

    monkeypatch.setattr(rx, '_packet', packet)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert tx.p4_probe.reason == ('peer silent' if termination == 'silence' else 'session ceiling')
    sent = tx.p4_probe.emissions
    assert len(sent) >= 8
    requests = {e['at'] for e in tx.p4_probe.link_events
                if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
    assert all(e['word'] == 'U27' for e in sent if e['audio_start'] not in requests)
    assert sent[7]['audio_start'] == end+round(.92*FS)
