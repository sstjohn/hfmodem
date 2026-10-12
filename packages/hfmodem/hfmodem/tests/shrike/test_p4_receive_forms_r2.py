# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Unit 1: strict evidence, original-input replay, and disjoint CPU charging.

Missing state is an explicit fixture failure; it is not a passed full-session
claim. Decoder-only evidence never becomes a receiver registration.
"""
from copy import deepcopy
import importlib.util
import json

import pytest
from hfmodem.tests.evidence import CAPTURES
from hfmodem.tests.shrike.r2_replay import ROOT, BoundaryError, FS, digest

PATH = ROOT / 'working/p4-plan-0930/designs-1001/receive_forms_replay.py'
if not (PATH).exists():
    pytest.skip('working/p4-plan-0930/designs-1001/receive_forms_replay.py is not in this tree', allow_module_level=True)
spec = importlib.util.spec_from_file_location('receive_forms_replay', PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_manifest_and_full_inputs():
    m = replay.manifest()
    assert m['sets'] == replay.SETS
    assert len(m['fixtures']) == 20
    hashes = {
        '1001-1009': '3cd86b9d99ed28520598fc0cf889e04612ab3d6e940e89519c97f1d87f9a7d69',
        '1001-1027': 'e69a4adcbf8586d0ec10ce6911a07f145ed46582d736645ec5a0c4bbd394bc48',
        '1001-1045': '4106893e8e2486946d9efb7f00b13144182c6a697f5661ed5856c1deaf5d1f2c',
        '1001-1059': '531e5f314fd321d09adda22de30d60205c0e079700a96ff5d72d68cae1692ce5',
        '1001-1039': '2e51f278d339af33ac0fbb8074619d77f38b86ce30bdca53fcb256c4e9df68fc',
    }
    for arm, expected in hashes.items():
        c = m['fixtures'][arm]
        assert c['files'][c['stream']] == expected
    for arm in m['fixtures']:
        tape = replay.Tape(arm)
        assert tape.contract['options']
        assert tape.contract['opening']['values'] == [7,38]
        assert tape.contract['opening']['headers'] == ['A','A']
        assert tape.contract['source']['artifacts'] or arm.startswith('0927-')
        assert tape.contract['metadata']['first_stream_sample'] == 0
    primitive = 'packages/hfmodem/hfmodem/tests/shrike/fixtures/p4-long/audio.npz'
    assert digest(ROOT/primitive) == m['primitives'][primitive] == 'c641415075f315e839691230e186136601c06c74b64d95539cabc2a243858fd7'
    assert m['fixtures']['1001-0336']['options']['p4-answer-offset'] == '0.908'
    assert m['fixtures']['1001-0336']['options']['tx-latency-ms'] == '0.0'
    assert m['fixtures']['1001-0336']['lead_samples'] == 2132
    # Required missing queued suffix must fail, without skip or invented bytes.
    with pytest.raises(BoundaryError, match='missing queued authentication bytes'):
        replay.replay('1001-0336')
    evidence = replay.decoder_evidence()
    assert sum(x is not None for x in evidence['robust']) == 12
    assert evidence['robust'][12] is None
    assert evidence['shifted_status'] == 0x61
    assert sum(x is not None for x in evidence['robust'])+1 == 13
    for arm, count in [('1001-1027',5),('1001-1059',12)]:
        rows = evidence[arm]
        assert len(rows) == count and all(r['status']==0x61 for r in rows)
        assert [r['marker'] for r in rows] == [8]+[9]*(count-1)
    assert evidence['1001-1027'][0]['start'] == 2148525
    assert evidence['1001-1059'][0]['start'] == 5328548
    assert evidence['negatives'] == [[189,0,4],[161,1,1],[392,5,2],[352,8,11]]
    rows = evidence['1039']
    assert len(rows)==16 and rows[0]['open']
    assert all(not r['header'] and .08 < r['spread'] < .12 for r in rows)
    assert [r['onset'] for r in rows] == [3107734+60000*k for k in range(16)]


    normals,crcs = replay.normal_long_evidence()
    assert crcs == [0x5a2c,0x1bf8]
    assert [r['score'] for r in normals] == pytest.approx([.191,.252,.482,.475,.302],abs=.001)
    assert [r['status'] for r in normals] == [None,None,3,3,None]
    assert normals[-1]['score']>.30  # work gate cannot imply CRC acceptance
    payload=bytes.fromhex(normals[2]['payload'])
    assert len(payload)==203 and payload.startswith(b'aining with K4MSU (EM56UO)\r[WL2K-5.0-B2F')
    assert normals[2]['payload']==normals[3]['payload']
    first_chirp = replay.Tape('1001-1059')
    lo=round(110.501*FS)
    for end in (111.811,111.851):
        audio=first_chirp.audio[lo:round(end*FS)]
        assert not replay.open_prefix(audio,FS//2,len(audio))


def test_required_input_mutations_fail(tmp_path):
    m = replay.manifest()
    c = deepcopy(m['fixtures']['0930-0117'])
    for key, value in [('options', {}), ('source', {}), ('own_tx_masks', []), ('lead_samples', 1)]:
        bad = deepcopy(c)
        bad[key] = value
        with pytest.raises(BoundaryError, match='contract differs'):
            replay.Tape('0930-0117',contract=bad)
    # Missing recording is a failure, never pytest.skip.
    with pytest.raises(BoundaryError):
        replay.Tape('0930-0117',root=tmp_path)


def long_decisions():
    c = replay.manifest()['fixtures']['1001-1059']
    return [r for r in c['decisions'] if 54.7*FS < r['at'] < 107.3*FS]


def test_charge_advances_wall_not_audio():
    rows = long_decisions()
    for row in rows[:2]:
        c = replay.ChargedClock(sample=row['capture_end_sample'], wall=row['now_sample'],
                                lead=row['scheduled_at']-row['decision_at'])
        before = c.sample,c.wall
        c.charge('full-reader-failed-or-successful',480)
        assert c.sample==before[0] and c.wall==before[1]+480
        assert c.ledger[-1]['support']==before[0]
    c = replay.ChargedClock(sample=100,wall=100,lead=20)
    assert c.eligible(120)
    c.charge('postdecision',1)
    assert not c.eligible(120) and c.sample==100
    c.deliver(110,115)
    assert c.sample==110 and c.wall==115
    with c.stage('full-read',480):
        with c.stage('nested-CRC',240):
            pass
    assert c.wall==595 and len([r for r in c.ledger if r['stage']=='nested-CRC'])==0
    for bad in [-1, .5]:
        with pytest.raises(BoundaryError):c.charge('invalid',bad)


def test_baseline_timestamps():
    rows = long_decisions()
    assert len(rows)==15
    assert sum(r['reason']=='proposal' for r in rows)==11
    assert sum(r['reason']=='answer deferred' for r in rows)==2
    first = rows[0]
    assert [first[k] for k in ('at','scheduled_at','decision_at','capture_end_sample','now_sample')] == [2628029,2791229,2788953,2786796,2788848]
    assert replay.audit_decision(first)==dict(lead=2276,margin=105,lag=2052,stage_availability=None)
    deferred = [r for r in rows if r['reason']=='answer deferred']
    assert [replay.audit_decision(r)['margin'] for r in deferred] == [-427,-25]
    # Poison aggregate CPU data: the ledger audit must still use timestamps.
    bad = dict(first,work_ms={'read_window':1000000})
    assert replay.audit_decision(bad)==replay.audit_decision(first)
    p=json.loads((CAPTURES/'onair-1001-1059/p4-probe.json').read_text())
    assert len([h for h in p['heard'] if h['kind']=='long' and h.get('status') is not None])==13


def _check_saved_result(actual, expected):
    # Retain the original final-rounding tolerance and every field comparison.
    if isinstance(expected, dict):
        assert actual.keys()==expected.keys()
        for key in expected:
            if key in ('score', 'coherence', 'lattice') and isinstance(expected[key], float):
                assert actual[key]==pytest.approx(expected[key], rel=0, abs=1e-12)
            else:
                _check_saved_result(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual)==len(expected)
        for got, want in zip(actual, expected):
            _check_saved_result(got, want)
    else:
        assert actual==expected


def _through_0210_close(actual, expected):
    """Both 0210 results up to the golden's close key, which is now a third data copy.

    Word-yield DESIGN-V2 Unit A (working/p4-plan-0930/p4-word-yield-recovery-1004)
    retries held V27 on the ordinary budget. The fixed tape after that key answers
    a close we no longer send, so nothing after it is compared.
    """
    close = next(e for e in expected['emissions'] if e[0]=='body' and e[3]==0x20)
    copies = [e for e in actual['emissions'] if e[0]=='body' and e[4]]
    cycle = round(1.25*FS)
    assert [e[2] for e in copies[:3]] == [close[2]-2*cycle, close[2]-cycle, close[2]]
    assert all(e[3:5] == copies[0][3:5] for e in copies[:3])
    def cut(result):
        kept = {k: v for k, v in result.items() if k not in ('end_sample', 'wall', 'reason')}
        kept['emissions'] = [e for e in result['emissions'] if e[2] < close[2]]
        kept['registrations'] = [r for r in result['registrations'] if r['onset_sample'] < close[2]]
        kept['ledger'] = [r for r in result['ledger'] if r['support'] < close[2]]
        return kept
    return cut(actual), cut(expected)


@pytest.mark.parametrize('arm',replay.SETS['G'])
def test_saved_legacy_baseline(arm):
    from hfmodem.tests.shrike.p4_nak_qam_support import neutral_keys
    baseline=json.loads((PATH.parent/'receive_forms_baseline.json').read_text())[arm]
    if 'fixture_failure' in baseline:
        with pytest.raises(BoundaryError,match='missing queued authentication bytes'):
            replay.replay(arm)
        return
    with neutral_keys() as neutral:
        actual=replay.baseline_result(replay.replay(arm))
    expected=deepcopy(baseline['result'])
    checked=actual
    if arm=='0930-0210':
        # The original golden is immutable. Exactly two approved neutral keys
        # extend it; every ordinary waveform/byte identity before its close remains required.
        pins=json.loads((PATH.parent/'receive_forms_nak_0210.json').read_text())
        # The held-answer rule (830721f49) keys them as repeats of the held ACK, unlabelled.
        assert neutral==[dict({k:row[k] for k in ('at','word','kind','state')},purpose=None)
                         for row in pins['neutral_keys']]
        expected['emissions']=sorted(expected['emissions']+pins['neutral_emissions'],
                                     key=lambda row:row[2])
        expected['ledger']=pins['ledger']
        checked,expected=_through_0210_close(actual,expected)
    _check_saved_result(checked, expected)
    with neutral_keys() as traced_neutral:
        traced = replay.replay(arm, timing_trace=True)
    assert traced_neutral==neutral
    assert traced.pop('timing_trace')
    assert replay.baseline_result(traced) == actual
    assert actual['emissions'] and actual['delivered']
    assert actual['end_sample']>replay.manifest()['fixtures'][arm]['handoff_window'][1]


def test_0210_without_neutral_requests_preserves_original_full_golden(monkeypatch):
    from hfmodem.shrike import onair,p4turn
    silence=p4turn.P4Turn.silence
    def no_request(self,*args,**kwargs):
        silence(self,*args,**kwargs)
        if self._next is not None and not self._next.current_slot_accepted:
            self._next=None
    monkeypatch.setattr(p4turn.P4Turn,'silence',no_request)
    monkeypatch.setattr(p4turn.P4Turn,'_answer_unresolved',lambda *a:None)
    monkeypatch.setattr(onair,'_p4_early_empty_answer',lambda *a:False)
    original=json.loads((PATH.parent/'receive_forms_baseline.json').read_text())['0930-0210']['result']
    _check_saved_result(*_through_0210_close(replay.baseline_result(replay.replay('0930-0210')),original))


def test_reader_log_1045_word(capsys, monkeypatch):
    from hfmodem.shrike import onair
    from hfmodem.shrike.p4receiver import Expect
    # The host suffix is unavailable: keep the full-session failure explicit.
    with pytest.raises(BoundaryError, match='missing queued authentication bytes'):
        replay.replay('1001-1045')
    tape, rx, covered = replay.recorded_reader_prefix('1001-1045', round(118.6*FS))
    decisions = tape.contract['decisions']
    assert covered[:-1] == sorted(d['capture_end_sample'] for d in decisions
                                 if d['capture_end_sample'] <= round(118.6*FS))
    assert any(o.word == 'Y27' and o.stream_sample == 5656523 for o in rx.observations)
    assert rx.clock.expects == Expect.WORD
    rx.finish_cycle(rx.clock.next_due)  # account for liveness consumed by the prefix
    rx.new_cycle(round(118.6*FS))
    onset = rx.clock.next_due
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    from types import SimpleNamespace
    end = onset+round(1.25*FS)
    audio = tape.audio[:end].copy()
    for a, b in tape.contract['own_tx_masks'] + tape.contract['capture_gaps']:
        audio[a:b] = 0
    stream = Stream(audio)
    stream.pos = round(118.6*FS)
    tx = Tx(stream)
    tx.sessrx = SimpleNamespace(p4=rx)
    # The prefix has already consumed these registrations; retain its clock,
    # observations and emission associations without feeding historical work.
    rx.registrations.clear()
    tx.p4_probe.emissions = []
    origin = onset-FS//2
    tx.p4_probe.handoff_audio = (origin, audio[origin:stream.pos].copy())
    tx.p4_link_options.p4_link_timeout = (end-stream.pos)/FS
    finish = rx.finish_cycle
    def finish_cycle(at):
        silent = finish(at)
        rx.normal_coherence = .987  # expose any stale metric in the driver log
        return silent
    monkeypatch.setattr(rx, 'finish_cycle', finish_cycle)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        tape.contract['opening']['emissions'][0]['audio_start'] % 60000,
        60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines()
             if f'no live packet at {onset/FS:.3f};' in line]
    assert lines, output
    log = lines[0]
    assert 'expect=WORD' in log and 'read_partial' in log and 'read_window' in log
    assert "'registrations': 0" in log
    assert 'normal coherence' not in log


def test_reader_log_1009_packet(capsys):
    result = replay.replay('1001-1009')
    log = capsys.readouterr().out
    line = next(s for s in log.splitlines() if 'no live packet at 48.494;' in s)
    assert 'expect=PACKET' in line and 'read_window' in line
    assert "'expectation': 'packet'" in line
    assert 'normal coherence 0.194' in line
    assert 'timing_trace' not in result


@pytest.mark.parametrize('arm', replay.SETS['P'])
def test_timing_1059_ledger(arm, capsys):
    plain = replay.replay(arm)
    traced = replay.replay(arm, timing_trace=True)
    rows = traced.pop('timing_trace')
    assert rows and rows[-1]['stage'] == 'stop'
    assert replay.baseline_result(traced) == replay.baseline_result(plain)
    assert all(a['end'] == b['start'] for a,b in zip(rows,rows[1:]))
    assert all(r['end'] >= r['start'] for r in rows)
    assert rows[-1]['end'] == traced['wall']
    assert sum(r['end']-r['start'] for r in rows) == traced['wall']-rows[0]['start']
    assert any(r['stage'] == 'read_window' and r['phase'] == 'postdecision' for r in rows)
    assert {'waveform_cache','prepare','feed','read_partial','read_window','decide','capture','decision','emit'} <= {r['stage'] for r in rows}
    for row in rows:
        if row['stage'] == 'capture':
            assert row['requested_end'] >= row['support_start']
            assert row['available_at'] == row['end']
        if row['stage'] == 'decision':
            assert row['deadline'] == row['due']-row['lead']
    if arm != '1001-1059':
        return
    # Stress ONLY the first full-support long read; earlier prefix work is unchanged.
    def full_read_cost(audio, origin, **kw):
        return 5760 if origin+len(audio) == 2786796 else 0
    late = replay.replay(arm, timing_trace=True, charges={'read_window': full_read_cost})
    decision = next(r for r in late['timing_trace'] if r['stage']=='decision'
                    and r['due']==2791229)
    assert decision['reason'] == 'answer deferred' and decision['proposed_at'] is None
    assert decision['start'] > decision['deadline']
    assert not any(e[2] == 2791229 for e in late['emissions'])
    assert any(e[2] == 2791229+180000 for e in late['emissions'])
    charged = [r for r in late['ledger'] if r['stage']=='read_window' and r['end']-r['start']==5760]
    assert len(charged) == 1
    matching = next(r for r in late['timing_trace'] if r['stage']=='read_window'
                    and r['start']==charged[0]['start'] and r['end']==charged[0]['end'])
    assert matching['support_start'] == matching['support_end'] == charged[0]['support']
    assert all(a['end']==b['start'] for a,b in zip(late['timing_trace'],late['timing_trace'][1:]))


def test_reader_diagnostic_expansion_only_at_miss_log(monkeypatch, capsys):
    from hfmodem.shrike import onair
    from hfmodem.shrike.p4receiver import P4Receiver
    from hfmodem.shrike.p4turn import P4Turn
    expand = onair._p4_reader_results
    finish = P4Receiver.finish_cycle
    decide = P4Turn.decide
    allowed = False
    expansions = misses = decisions = 0

    def decided(self, *a, **kw):
        nonlocal allowed, decisions
        allowed = False
        decisions += 1
        return decide(self, *a, **kw)

    def finished(self, *a, **kw):
        nonlocal allowed, misses
        silent = finish(self, *a, **kw)
        allowed = bool(silent)
        misses += bool(silent)
        return silent

    def expanded(calls):
        nonlocal allowed, expansions
        # No dict/kind-list expansion in read, feed, or read_cost, with
        # tracing either off or on. Successful cycles never expand at all.
        assert allowed and decisions
        allowed = False
        expansions += 1
        return expand(calls)

    monkeypatch.setattr(P4Turn, 'decide', decided)
    monkeypatch.setattr(P4Receiver, 'finish_cycle', finished)
    monkeypatch.setattr(onair, '_p4_reader_results', expanded)
    for traced in (False, True):
        expansions = misses = decisions = 0
        replay.replay('1001-1059', timing_trace=traced)
        log = capsys.readouterr().out
        assert expansions == misses == log.count('[p4 link] no live packet at ')
        assert expansions > 0
        assert decisions > expansions  # successful cycles added zero expansion calls


@pytest.mark.parametrize('arm,begin,end,cfo', [
    ('1001-1009',44,49,.9), ('1001-1027',44,49,.32),
    ('1001-1059',110,115,.93)])
def test_scan_duration_and_reuse(arm, begin, end, cfo, monkeypatch):
    from hfmodem.shrike import p4oracle
    calls = []
    scan = p4oracle.word_scan
    def counted(audio, fs, *args, **kw):
        calls.append((len(audio), kw['origin'], kw['gaps']))
        return scan(audio, fs, *args, **kw)
    monkeypatch.setattr(p4oracle, 'word_scan', counted)
    rx, clocks = replay.acquisition_replay(arm, begin*FS, end*FS, cfo=cfo, cycle_s=3.75,
        charges={'read_window': round(.120*FS), 'scan_idle':round(.080*FS)})
    assert len(calls)>=3 and all(n == 60000 for n, _, _ in calls)
    assert all(b[1]-a[1] == 19200 for a,b in zip(calls,calls[1:]))
    assert 60000-19200 == 40800
    assert all(e['next_event']-e['wall_start'] >= round(.080*FS)
               for e in rx.live_events if e['event']=='word scan')
    assert not [e for e in rx.live_events if e['event']=='acquisition coverage lost']
    before = len(calls)
    tape = replay.Tape(arm)
    for _ in range(3):
        rx.read_window(tape.audio[begin*FS:end*FS],begin*FS,whole=True)
        rx.scan_idle(now=clocks.wall,next_event=clocks.wall+FS)
    assert len(calls)==before
    assert clocks.sample==end*FS and clocks.wall>=clocks.sample


@pytest.mark.parametrize('candidate',[False,True])
@pytest.mark.parametrize('arm,first', [('0929-1857',26.594),
    ('0929-2034',40.345),('0929-2253',24.100)])
def test_scan_actual_off_raster_words(arm, first, candidate):
    from hfmodem.shrike.p4receiver import Registered
    begin,end = round((first-.9)*FS),round((first+13.6)*FS)
    legacy,_ = replay.acquisition_replay(arm,begin,end,scan='legacy')
    baseline = [(o.word,o.stream_sample) for o in legacy.observations
                if o.word=='V27' and not o.provisional and not o.weak]
    assert len(baseline)>=10
    prefix=None
    if candidate:
        qrm=replay.Tape('0927-1553').audio
        size=round(1.35*FS)
        prefix=next(qrm[a:a+size] for a in range(0,len(qrm)-size,FS//4)
                    if replay.open_prefix(qrm[a:a+size],FS//2,size))
    rx,_ = replay.acquisition_replay(arm,begin,end,pending_prefix=prefix)
    assert not [r for r in rx.registrations if r.observation.provisional
                and r.observation.kind.value=='normal' and r.outcome != Registered.EXPIRED]
    words=[(o.word,o.stream_sample) for o in rx.observations
           if o.word=='V27' and not o.provisional and not o.weak]
    assert all(any(w==v and abs(at-s)<=round(.005*FS) for v,s in words) for w,at in baseline)
    assert not [e for e in rx.live_events if e['event']=='acquisition coverage lost']


def test_scan_edge_horizon_generations_and_masks(monkeypatch):
    import numpy as np
    from hfmodem.shrike import p4oracle
    from hfmodem.shrike.p4receiver import P4Receiver
    from hfmodem.shrike.p4observation import Kind, Observation
    rx=P4Receiver(FS,admitted=True,scan='incremental')
    at=round(.9*FS)
    calls=[]
    def scan(audio,fs,cfo,**kw):
        calls.append((kw['origin'],len(audio),cfo,kw['gaps']))
        return [Observation(Kind.WORD,fs,kw['origin'],at-kw['origin'],at,
                            .240,0.,.9,word='Y27',score=.99)]
    monkeypatch.setattr(p4oracle,'word_scan',scan)
    monkeypatch.setattr(p4oracle,'_extent',lambda *a,**kw:.244)
    rx.read_window(np.zeros(60000),0)
    assert not rx.scan_idle(now=60000,next_event=63839)
    rx._scan_upper_s=.080001
    assert not rx.scan_idle(now=60000,next_event=63840)
    assert not calls  # measured allowance cannot round down to available slack
    rx._scan_upper_s=.080
    assert not rx.scan_idle(now=60000,next_event=64800,critical=True)
    assert not rx.scan_idle(now=60000,next_event=64800)
    assert len(calls)==1 and rx._scan_pending and not rx.observations
    rx.read_window(np.zeros(90000),0)
    regs=rx.scan_idle(now=90000,next_event=94800)
    assert len(regs)==1 and regs[0].observation.extent_s==.244
    assert len(rx.observations)==1
    # Edge resolution reacquired the clock. Re-arm fallback explicitly for
    # the independent carrier-generation and same-support checks below.
    rx.clock.misses=3
    rx.cfo_hz=3.
    rx.read_window(np.zeros(90000),0)
    rx.scan_idle(now=90000,next_event=94800)
    assert calls[-1][0]==19200 and calls[-1][2]==3.
    before=len(calls)
    rx.read_window(np.zeros(90000),0)
    rx.scan_idle(now=90000,next_event=94800)
    assert len(calls)==before  # unchanged support cannot repeat a complete block
    rx.new_call()
    assert rx._scan_cursor is None and not rx._scan_pending
    rx.note_gap(at,at+100)
    rx.read_window(np.zeros(90000),0)
    assert not rx.scan_idle(now=90000,next_event=94800)
    assert calls[-1][3] and not rx.observations


def test_scan_overrun_reports_exact_interval(monkeypatch):
    import numpy as np
    from hfmodem.shrike import p4oracle
    from hfmodem.shrike.p4receiver import P4Receiver
    monkeypatch.setattr(p4oracle,'word_scan',lambda *a,**kw:[])
    rx=P4Receiver(FS,admitted=True,scan='incremental')
    rx.read_window(np.zeros(FS),0)
    rx.read_window(np.zeros(7*FS),0)
    loss=[e for e in rx.live_events if e['event']=='acquisition coverage lost']
    assert loss==[dict(event='acquisition coverage lost',begin=0,end=FS)]
    assert len(rx._scan_audio)==6*FS
    rx.new_call()
    rx.read_window(np.zeros(FS),0)
    rx.read_window(np.zeros(FS),600*FS)
    assert len(rx._scan_audio)==6*FS and rx._scan_origin==595*FS
    loss=[e for e in rx.live_events if e['event']=='acquisition coverage lost']
    assert loss==[dict(event='acquisition coverage lost',begin=0,end=595*FS)]
    rx.new_call()
    rx.read_window(np.zeros(FS),0)
    rx.read_window(np.zeros(FS),2*FS)
    rx.scan_idle(now=3*FS,next_event=4*FS)
    assert rx._scan_cursor==19200  # a narrower new crop cannot discard queued audio
    assert rx.scan=='incremental'
    with pytest.raises(ValueError):
        P4Receiver(scan='surprise')


@pytest.mark.parametrize('onset', [.390,.400,.410,.790,.800,.810,.990])
def test_scan_boundary_words_are_physical_hits(onset):
    import numpy as np
    from hfmodem.shrike.p4receiver import P4Receiver
    # In-memory placement of the recorded 2253 V27, crossing the .400 grid
    # and the first crop edge. This tests acquisition, not injected results.
    tape=replay.Tape('0929-2253')
    source_start=round(24.020*FS)
    piece=tape.audio[source_start:round(24.420*FS)]
    audio=np.zeros(3*FS,np.float32)
    start=round((onset-.080)*FS)
    audio[start:start+len(piece)]=piece
    rx=P4Receiver(FS,admitted=True,cfo_hz=1.,scan='incremental')
    clocks=replay.ChargedClock()
    for stop in range(FS//4,len(audio)+1,FS//4):
        clocks.deliver(stop,stop)
        rx.read_window(audio[:stop],0)
        with clocks.stage('scan_idle',3840):
            rx.scan_idle(now=clocks.wall,next_event=clocks.wall+FS//4)
    words=[o for o in rx.observations if o.word=='V27' and not o.weak]
    assert len(words)==1
    assert abs(words[0].stream_sample-round(onset*FS))<round(.005*FS)
    assert not [e for e in rx.live_events if e['event']=='acquisition coverage lost']


def test_scan_placed_alias_outcomes_are_unchanged(monkeypatch):
    from hfmodem.tests.shrike import test_p4_cfo_alias_0930 as alias
    from hfmodem.shrike.p4receiver import P4Receiver
    audio=[replay.Tape(arm).audio for arm in ('0930-2153','0930-2221')]
    monkeypatch.setattr(alias,'P4Receiver',lambda *a,**kw:P4Receiver(*a,**kw,scan='incremental'))
    alias.test_2153_carrier_step_registers_u27(audio)
    alias.test_2153_step_from_a_calibrated_body_carrier(audio)
    alias.test_2221_existing_alias_recovers_without_extra_trellis(audio,monkeypatch)


def test_scan_periodic_energy_does_not_suppress_acquisition(monkeypatch):
    import numpy as np
    from hfmodem.shrike import p4oracle
    from hfmodem.shrike.p4observation import Observation,Kind
    from hfmodem.shrike.p4receiver import P4Receiver,Clock,Expect
    # State-only pending classification, nominated by an actual 1553 prefix.
    # Candidate ownership/CRC transitions belong to unit 4, not this test.
    tape=replay.Tape('0927-1553')
    prefixes=[tape.audio[a:a+round(1.35*FS)]
              for a in range(0,len(tape.audio)-round(1.35*FS),FS//4)]
    prefix=next(p for p in prefixes if replay.open_prefix(p,FS//2,len(p)))
    rx=P4Receiver(FS,admitted=True,cfo_hz=1.,scan='incremental')
    rx.clock=Clock(Expect.PACKET,0,.330,1.25,1.)
    monkeypatch.setattr(rx,'_word',lambda *a,**kw:None)
    monkeypatch.setattr(rx,'_packet',lambda *a,**kw:None)
    monkeypatch.setattr(rx,'_classification',lambda at:Observation(
        Kind.NORMAL,FS,0,at,at,.85,0.,0.,provisional=True))
    calls=[]
    monkeypatch.setattr(p4oracle,'word_scan',lambda *a,**kw:calls.append(kw['origin']) or [])
    audio=np.tile(prefix,5)
    for k in range(3):
        end=round((k*1.25+.85)*FS)
        rx.read_window(audio[:end],0,whole=True)
        rx.new_cycle()
    assert rx.clock.misses==0 and rx._scan_eligible
    rx.scan_idle(now=end,next_event=end+FS//4)
    assert calls==[0]
    assert not rx.registrations


@pytest.mark.parametrize('packet', [False, True])
@pytest.mark.parametrize('pending', [False, True])
def test_scan_disarms_after_successful_placement(monkeypatch, packet, pending):
    import numpy as np
    from hfmodem.shrike import p4oracle
    from hfmodem.shrike.p4observation import Observation, Kind
    from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect
    rx = P4Receiver(FS, admitted=True, cfo_hz=0., scan='incremental')
    rx.clock = Clock(Expect.PACKET if packet else Expect.WORD, 0, .935, 1.25, 0., misses=3)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
    edge = Observation(Kind.WORD, FS, 0, round(.9*FS), round(.9*FS),
                       .240, 0., .9, word='Y27', score=.99)
    calls = []
    def scan(*args, **kw):
        calls.append(kw['origin'])
        return [edge] if pending else []
    monkeypatch.setattr(p4oracle, 'word_scan', scan)
    monkeypatch.setattr(p4oracle, '_extent', lambda *a, **kw: .244)
    rx.read_window(np.zeros(60000), 0, whole=True)
    rx.scan_idle(now=60000, next_event=72000)
    assert calls == [0] and rx._scan_eligible
    assert bool(rx._scan_pending) == pending

    # Placement must clear both the fallback misses and the armed flag in
    # this same read, even when a carrier refinement requests a recent rescan.
    rx.clock.next_due = 60000
    placed = Observation(Kind.SHORT if packet else Kind.WORD, FS, 0, 60000,
                         60000, .240, 3., .9, placed=True,
                         word=None if packet else 'V27', score=.99)
    monkeypatch.setattr(rx, '_packet' if packet else '_word', lambda *a, **kw: placed)
    rx.read_window(np.zeros(120000), 0, whole=True)
    assert rx.clock.misses == rx._scan_misses == 0
    assert not rx._scan_eligible
    rx.read_window(np.zeros(120000), 0, whole=True)
    assert rx._scan_rescan == 0
    cursor = rx._scan_cursor
    result = rx.scan_idle(now=120000, next_event=132000)
    assert calls == [0] and rx._scan_cursor == cursor
    assert not rx._scan_pending
    assert [r.observation.word for r in result] == (['Y27'] if pending else [])
    # While placed reception continues, unscanned older support is no longer
    # acquisition work. Aging it out must not report a coverage overrun.
    for end in (8*FS, 12*FS):
        rx.read_window(np.zeros(6*FS), end-6*FS, whole=True)
        rx.scan_idle(now=end, next_event=end+FS//4)
    assert calls == [0]
    assert not [e for e in rx.live_events if e['event'] == 'acquisition coverage lost']


def test_scan_0121_free_carrier_alias_matches_legacy():
    begin,end=round(13.851*FS),round(20.251*FS)
    old,_=replay.acquisition_replay('0929-0121',begin,end,scan='legacy',cfo=115.6)
    rx,_=replay.acquisition_replay('0929-0121',begin,end,cfo=115.6)
    baseline=[o for o in old.observations if o.word=='T27' and not o.weak]
    words=[o for o in rx.observations if o.word=='T27' and not o.weak]
    assert len(baseline)>=4 and len(words)>=4
    assert all(any(abs(o.stream_sample-b.stream_sample)<=48 for o in words) for b in baseline)
    assert all(3 < o.cfo_hz < 4 for o in words)
    assert not [e for e in rx.live_events if e['event']=='acquisition coverage lost']


@pytest.mark.parametrize('arm',['0930-0117','0930-0127','0930-0210','0930-0352'])
def test_scan_driver_working_arms(arm):
    baseline=replay.baseline_result(replay.replay(arm))
    result=replay.replay(arm,scan='incremental',timing_trace=True,
                         charges={'scan_idle':3840})
    current=replay.baseline_result(result)
    # Diagnostic rows change with acquisition; protocol outputs remain exact.
    for key in ('emissions','delivered','reason','unread','deferred'):
        assert current[key]==baseline[key]
    assert not [e for e in result['scan_coverage'] if e['event']=='acquisition coverage lost']
    trace=result['timing_trace']
    assert all(b['start']>=a['end'] for a,b in zip(trace,trace[1:]))
    assert all(e['wall_start']+3840<=e['next_event'] for e in result['scan_coverage']
               if e['event']=='word scan')


def test_scan_unknown_driver_option_fails_before_session():
    import numpy as np
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream,Tx
    live=Stream(np.zeros(FS,np.float32))
    tx=Tx(live)
    session_rx=tx.sessrx
    tx.p4_link_options.p4_rx_scan='unknown'
    with pytest.raises(ValueError,match='scan mode'):
        onair._run_p4_link(tx,live,onair._MasterGrid(0,60000,0,
            packet_n=36000,cs_n=12000,d_max_n=4800))
    assert tx.sessrx is session_rx and live.pos==0


@pytest.mark.parametrize('delayed_work', ['normal_decode', 'long_read'])
def test_declared_replay_ignores_host_delay_with_exact_ledger(delayed_work, monkeypatch, capsys):
    """Host scheduling cannot spend undeclared samples in a virtual replay."""
    import time
    from hfmodem.shrike.p4hold import BreakinHold
    from hfmodem.shrike.p4receiver import P4Receiver
    original_clock = time.monotonic
    plain = replay.replay('1001-1009')
    delays = []
    if delayed_work == 'normal_decode':
        original = BreakinHold._decode_normal
        def decode(self, scores, *, iterations):
            result = original(self, scores, iterations=iterations)
            time.sleep(.025)
            delays.append(.025)
            return result
        monkeypatch.setattr(BreakinHold, '_decode_normal', decode)
    else:
        original = P4Receiver.read_window
        def read(self, audio, origin, **kw):
            result = original(self, audio, origin, **kw)
            if origin+len(audio) == 2486621:
                # Before the repair this host delay removed the optional
                # capture/read at 2487341 while all 16 other fields stayed equal.
                time.sleep(.040)
                delays.append(.040)
            return result
        monkeypatch.setattr(P4Receiver, 'read_window', read)
    traced = replay.replay('1001-1009', timing_trace=True)
    rows = traced.pop('timing_trace')
    assert delays and rows
    assert replay.baseline_result(traced) == replay.baseline_result(plain)
    assert traced['ledger'] == plain['ledger']  # retain every call, including zero cost
    assert any(r.get('support') == 2487341 and r['stage'] == 'capture'
               for r in plain['ledger'])
    assert time.monotonic is original_clock
    capsys.readouterr()


def test_declared_long_read_cost_controls_reread_and_exact_trace(capsys):
    """The same work becomes authoritative when declared in sample ticks."""
    def cost(audio, origin, **kw):
        return 1920 if origin+len(audio) == 2486621 else 0
    plain = replay.replay('1001-1009', charges={'read_window': cost})
    traced = replay.replay('1001-1009', charges={'read_window': cost}, timing_trace=True)
    rows = traced.pop('timing_trace')
    assert replay.baseline_result(traced) == replay.baseline_result(plain)
    charged = [r for r in plain['ledger'] if r['stage'] == 'read_window'
               and r['end']-r['start'] == 1920]
    assert len(charged) == 1
    assert charged[0]['support'] == 2486621
    assert charged[0]['start'] == 2486621 and charged[0]['end'] == 2488541
    assert not any(r.get('support') == 2487341 and r['stage'] == 'capture'
                   for r in plain['ledger'])
    assert all(a['end'] == b['start'] for a,b in zip(rows,rows[1:]))
    assert rows[-1]['end'] == traced['wall']
    capsys.readouterr()


def test_declared_clock_keeps_deadlines_and_restores_all_bindings_on_failure():
    import time
    from hfmodem.shrike import p4hold, p4receiver, p4shortlevels, p4turn
    clock = replay.ChargedClock(wall=48000)
    original = (time.monotonic, time.perf_counter, p4hold.monotonic,
                p4receiver.monotonic, p4shortlevels.ShortDecode.__init__,
                p4turn.P4Turn.__init__)
    with pytest.raises(RuntimeError, match='replay failed'):
        with replay.declared_time(clock):
            hold = p4hold.BreakinHold(FS)
            decoder = p4shortlevels.ShortDecode([], costs={}, events=[], at=0,
                                               status_ok=lambda frame: True)
            assert hold.now() == decoder.now() == time.monotonic() == 1.
            deadline = time.monotonic()+.020
            clock.charge('declared decode', 1440)
            assert hold.now() == decoder.now() == 1.030
            assert time.monotonic() > deadline
            raise RuntimeError('replay failed')
    assert (time.monotonic, time.perf_counter, p4hold.monotonic,
            p4receiver.monotonic, p4shortlevels.ShortDecode.__init__,
            p4turn.P4Turn.__init__) == original


@pytest.mark.parametrize('arm,expected_end,keyed_at', [
    ('0930-0117', 1195784, 1112024),
    ('0930-0127', 1255702, 1171942),
    ('0930-0210', 1287068, 1231868),
    ('0930-0352', 1435765, 1352005),
])
def test_legacy_breakin_keeps_its_original_read_schedule(monkeypatch, arm, expected_end, keyed_at):
    from hfmodem.shrike import p4turn
    keyed, take = p4turn.P4Turn.keyed, replay.Capture.take_until
    owner, captures = [], []
    def key(self, *args, **kwargs):
        result = keyed(self, *args, **kwargs)
        owner[:] = [self]
        return result
    def capture(self, end):
        if end == expected_end and owner:
            m = owner[0]
            captures.append((m.state, m.last.state, m.last.stream_sample))
        return take(self, end)
    monkeypatch.setattr(p4turn.P4Turn, 'keyed', key)
    monkeypatch.setattr(replay.Capture, 'take_until', capture)
    replay.replay(arm)
    assert (p4turn.State.IRS_BREAKIN, p4turn.State.IRS_BREAKIN, keyed_at) in captures
