# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""S5(g): K9KDJ 0146 weak normals, with the recorded decision allowance."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p4decode, p4frame, p4short, p4words, session
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver, RegisteredObservation, Registered
from hfmodem.tests.shrike.test_p4turn import machine

ROOT = Path(__file__).resolve().parents[5]
FS = 48000
SINGLES = (2508144, 2628141, 2748141, 2868138, 3168138, 3288138)


@pytest.fixture(scope='module')
def recording():
    path = ROOT/'captures/onair-0930-0146'
    if not (path/'stream.flac').exists():
        pytest.skip('local K9KDJ 0146 recording absent')
    return session.load_wav(str(path/'stream.flac'), FS), json.loads((path/'p4-probe.json').read_text())


def receiver(recording, at):
    audio, probe = recording
    row = next(r for r in probe['heard'] if r['onset_sample'] == at and 'status' in r)
    cfo = row['cfo_hz']
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    rx.peer_onset = at-60000
    rx.acked_counter = 0
    origin = at-round(.05*FS)
    return rx, audio[origin:at+round(.810*FS)], origin, row


@pytest.mark.parametrize('at', SINGLES)
def test_recorded_weak_copy_answers_in_its_own_slot(recording, monkeypatch, at):
    rx, audio, origin, expected = receiver(recording, at)
    elapsed = [0.]
    decode = p4decode.component_decode

    def costed(*args, **kwargs):
        # Conservative component cost, independent of host load. Both trees
        # execute real trellises; the old duplicate polarity pays twice.
        elapsed[0] += .0015
        return decode(*args, **kwargs)

    monkeypatch.setattr(p4decode, 'component_decode', costed)
    rx.hold.now = lambda: elapsed[0]
    rx.normal_deadline = lambda at: .029
    obs = rx._normal(audio, origin)
    assert obs is not None and obs.frame is not None
    assert obs.frame.status.byte == expected['status']
    assert obs.frame.data.hex() == expected['data']
    events = [e for e in rx.hold.events if e['event'] == 'normal-decode']
    assert [(e['iterations'], e['reason']) for e in events] == [(1, 'CRC'), (8, 'accepted')]
    assert elapsed[0] <= .029
    turn, _ = machine(role='irs', normal_ack='parity')
    turn.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.837*FS))
    proposal = turn.decide(at+round(.846*FS))
    assert proposal is not None and proposal.word == 'V27'
    assert proposal.stream_sample == obs.stream_sample+round(.920*FS)
    assert not any(e['event'] == 'answer-deferred' for e in turn.link_events)


@pytest.mark.parametrize('polarity', (1, -1))
def test_recorded_polarities_share_one_trellis(recording, monkeypatch, polarity):
    rx, audio, origin, expected = receiver(recording, SINGLES[0])
    rx._normal(audio, origin)
    scores = rx.hold.members[-1].llr*polarity
    calls = []
    decode = p4decode.turbo_decode

    def counted(*args, **kwargs):
        calls.append(kwargs['iterations'])
        return decode(*args, **kwargs)

    monkeypatch.setattr(p4decode, 'turbo_decode', counted)
    frame, sign = p4short.decode_scores(scores)
    assert frame.data.hex() == expected['data'] and sign == polarity
    assert calls == [8]


def test_polarity_ties_remain_bit_zero(monkeypatch):
    wires = []
    monkeypatch.setattr(p4frame, 'decode_field', lambda wire, **kw: wires.append(wire))
    assert p4short.decode_scores(np.zeros(1056)) == (None, None)
    assert wires == [bytes(43), bytes(43)]


def test_recorded_unread_decisions_are_resolved_after_the_link(recording):
    from hfmodem.shrike.p4budget import decision_summary
    _, probe = recording
    summary = decision_summary(probe['link_events'], probe['heard'], FS)
    assert summary['late_crc_valid'] == 10
    assert summary['other_late_admission'] == 0
    assert summary['never_decoded'] == 16
    assert summary['available_at_decision'] == 0
    assert sum(summary[k] for k in ('late_crc_valid', 'other_late_admission',
                                   'never_decoded', 'available_at_decision')) == summary['missed_decisions']
    late = [e for e in probe['link_events'] if e.get('decode_resolution') == 'late_crc_valid']
    deferred = [e for e in probe['link_events'] if e['event'] == 'answer-deferred']
    assert len(late) == len(deferred)
    assert all(any(abs(d['at']-e['at']) < .05*FS for d in deferred) for e in late)


def test_late_word_and_expired_head_have_different_accounting(recording):
    from copy import deepcopy
    from hfmodem.shrike.p4budget import decision_summary
    _, probe = recording
    decision = deepcopy(next(e for e in probe['link_events']
                             if e.get('reason') == 'body unread at decision'))
    # Recorded decision and word, with arrival shifted to the decision's slot.
    word = deepcopy(next(r for r in probe['heard'] if r.get('word') == 'X27'
                         and not r.get('provisional')))
    word.update(onset_sample=decision['at'], received_at_sample=decision['decision_at']+1)
    summary = decision_summary([decision], [word], FS)
    assert summary['other_late_admission'] == 1
    word['outcome'] = 'expired'
    assert decision_summary([decision], [word], FS)['never_decoded'] == 1
    word.update(outcome='confirmed', received_at_sample=decision['decision_at'])
    assert decision_summary([decision], [word], FS)['available_at_decision'] == 1


@pytest.mark.parametrize('summary_error', (None, KeyError('received_at_sample')),
                         ids=('summary', 'summary_error'))
def test_recorded_driver_emits_budget_summary(recording, monkeypatch, capsys, summary_error):
    from hfmodem.shrike import onair, p4budget
    from hfmodem.shrike.arq import State
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    audio, probe = recording
    stream = Stream(audio)
    stream.pos = round(38.8*FS)
    tx = Tx(stream)
    tx.sessrx.p4.cfo_hz = next(r['cfo_hz'] for r in probe['heard']
                              if r['onset_sample'] == 1848145 and not r['provisional']
                              ) - p4words.CARRIER_HZ['X27']
    tx.p4_probe.emissions = [dict(audio_start=1322877+k*60000,
                                 audio_end=1322877+k*60000+36367) for k in range(10)]
    tx.p4_link_options.p4_link_timeout = 2.
    if summary_error is not None:
        def fail_summary(*args):
            raise summary_error
        monkeypatch.setattr(p4budget, 'decision_summary', fail_summary)
    reason = onair._run_p4_link(tx, stream, onair._MasterGrid(
        2877, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert tx.p4_probe.link_events[-1]['event'] == 'Stopped'
    assert reason == tx.p4_probe.reason == 'session ceiling'
    assert tx.p4_link_host.arq.state == State.DISCONNECTING
    output = capsys.readouterr().out
    assert 'LINK DOWN: session ceiling' in output
    if summary_error is not None:
        assert "summary failed: 'received_at_sample'" in output
        assert not any(e['event'] == 'DecodeBudgetSummary' for e in tx.p4_probe.link_events)
        return
    summary = next(e for e in tx.p4_probe.link_events if e['event'] == 'DecodeBudgetSummary')
    assert summary['decisions'] == summary['never_decoded'] == 1
    decision = next(e for e in tx.p4_probe.link_events if e['event'] == 'Decision')
    assert decision['decode_resolution'] == 'never_decoded'
    assert decision['capture_lag_before_read_ms'] == 0.
