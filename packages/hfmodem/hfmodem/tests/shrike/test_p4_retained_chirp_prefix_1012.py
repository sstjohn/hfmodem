# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Retained PCM boundaries, genuine acquisitions, and physical owner accounting.

Every clock assignment below is initial history setup. Reader/CRC/registration
results are never replaced. Counts wrap and delegate production operations.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4chirp, p4chirprx, p4robusttx, p4turn
from hfmodem.shrike.p4observation import Kind, Registered
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, P4Receiver, Role
from hfmodem.tests.shrike.test_p4_part_a_driver_1011 import AbsoluteStream
from hfmodem.tests.shrike.test_p4_link_0927 import Tx
from hfmodem.tests.shrike.test_p4turn import machine

FS = 48000
AT = FS
DATA = b'retained source\r'

def evidence(label, **values):
    # Ordinary root gate needs no special plugin; audited runs retain the same
    # observations in JSON in addition to pytest's raw captured output.
    import sys
    audit = sys.modules.get('retained_audit_1012')
    if audit is not None:
        audit.record(label, **values)
    print(label, values)

@pytest.fixture(scope='module')
def pcm():
    wave = p4chirp.entry_packet(DATA, 0x01)
    # Genuine generated signal plus measured deterministic noise after its end.
    # The late-buffer controls therefore do not substitute all-zero proof.
    tail = np.random.default_rng(1012).normal(0, .0001, FS)
    return np.r_[np.zeros(AT), wave, tail]

def receiver(nominal=AT, cycle=1.25, cfo=0.):
    rx = P4Receiver(FS, admitted=True, chirp=True, cfo_hz=cfo)
    rx.expected = True
    rx.clock = Clock(Expect.PACKET, nominal, .330, cycle, cfo)
    return rx

@pytest.fixture
def calls(monkeypatch):
    counts = {'acquire': [], 'decode': []}
    for name, key in (('acquire_header', 'acquire'), ('decode_packet', 'decode')):
        original = getattr(p4chirprx, name)
        def counted(audio, _original=original, _key=key, **kw):
            counts[_key].append(dict(length=len(audio), **kw))
            return _original(audio, **kw)
        monkeypatch.setattr(p4chirprx, name, counted)
    return counts

def acquire(rx, audio):
    n = rx.clock.next_due
    # A late acquired onset needs its own >.800 s provisional support; using
    # only N+.810 can end before that precondition when H is later than N.
    end = max(n, AT)+round(.812*FS)
    rows = rx.read_partial(audio[:end], 0, deadline=end)
    chirps = [r for r in rows if r.observation.kind == Kind.CHIRP]
    assert len(chirps) == 1 and chirps[0].observation.provisional
    key = (n, rx.clock.cfo_hz)
    header = rx._chirp_headers[key]
    assert header is not None
    h = header[0]
    assert abs(h-n) <= round(.035*FS)
    assert chirps[0].observation.stream_sample == h
    assert rx.has_chirp_header(h) and rx.clock.cycle_s == 3.75
    evidence('actual_acquisition', nominal=n, onset=h, header=header,
             prior_cycle=rx._chirp_prior_cycles[key][2])
    return h, key, chirps[0]

@pytest.mark.parametrize('lag', [1, round(.010*FS), round(.235*FS)+1,
                               round(.810*FS)+1, round(3.355*FS)+1])
@pytest.mark.parametrize('first', ['partial', 'whole'])
def test_uncached_prediction_before_actual_pcm_never_calls_strict_api(pcm, calls, lag, first):
    rx = receiver(cycle=3.75)
    owner = rx.clock
    origin = AT+lag
    audio = pcm[origin:]
    assert origin > owner.next_due and audio.size and np.any(audio)
    operations = ('partial', 'whole') if first == 'partial' else ('whole', 'partial')
    for operation in operations*2:
        rows = (rx.read_partial(audio, origin, deadline=AT+round(.810*FS))
                if operation == 'partial' else rx.read_window(audio, origin, whole=True))
        assert not any(r.observation.kind == Kind.CHIRP for r in rows)
        assert calls == {'acquire': [], 'decode': []}
        assert rx._chirp_headers == {} and rx._chirp_prior_cycles == {}
        assert rx.clock is owner and rx.clock.cycle_s == 3.75
        assert rx.live_onsets == [] and rx.peer_watch is None
    evidence('uncached_unavailable', nominal=AT, origin=origin,
             end=origin+len(audio), partial_length=max(0, AT+round(.810*FS)-origin),
             calls=calls, registrations=rx.registrations)

@pytest.mark.parametrize('prediction', ['early', 'late'])
@pytest.mark.parametrize('cut', [1, round(.020*FS), round(.200*FS)+1, round(3.355*FS)+1])
@pytest.mark.parametrize('order', [('partial', 'whole', 'fast'), ('fast', 'whole', 'partial')])
def test_genuine_cached_prefix_loss_keeps_prior_owner_without_new_authority(pcm, calls, prediction, cut, order):
    n = AT + (round(.030*FS) if prediction == 'late' else -round(.020*FS))
    rx = receiver(n)
    owner = rx.clock
    h, key, reg = acquire(rx, pcm)
    assert rx._chirp_prior_cycles[key] == (owner, rx._train_generation, 1.25)
    assert h < n if prediction == 'late' else h > n
    prior = dict(rx._chirp_prior_cycles)
    headers = dict(rx._chirp_headers)
    old_observations = tuple(rx.observations)
    old_live = tuple(rx.live_onsets)
    original_registration_count = len(rx.registrations)
    origin = h+cut
    audio = pcm[origin:]
    assert origin > h and origin+len(audio) > h+round(3.35*FS)
    # The genuine early acquired H covers H<O<=N too, not just negative N-O.
    if prediction == 'late' and cut <= round(.020*FS):
        assert n-origin >= 0
    if prediction == 'early':
        assert n-origin < 0
    for operation in order*2:
        if operation == 'fast':
            rows = rx.read_chirp(audio, origin, onset=h)
            assert rows is None
            assert not (rx.packet_read and rx.packet_read.get('decoder') == 'chirp'
                        and rx.packet_read.get('reason') == 'CRC/header unread')
        else:
            rows = (rx.read_partial(audio, origin, deadline=origin+len(audio))
                    if operation == 'partial' else rx.read_window(audio, origin, whole=True))
            assert not any(r.observation.kind == Kind.CHIRP or r.outcome == Registered.EXPIRED for r in rows)
        assert len(calls['acquire']) == 1 and calls['decode'] == []
        assert rx._chirp_headers == headers and rx._chirp_prior_cycles == prior
        assert rx.clock is owner and owner.cycle_s == 3.75
        assert tuple(rx.observations) == old_observations
        assert tuple(rx.live_onsets) == old_live
        assert len(rx.registrations) == original_registration_count
        assert not any(e['event'] == 'chirp header unread' for e in rx.live_events)
        assert rx.peer_watch is None and rx._chirp_cache == {}
    evidence('cached_lost_prefix', nominal=n, onset=h, origin=origin,
             nominal_sample=n-origin, order=order, cut=cut, calls=calls,
             prior_cycle=prior[key][2], observations=rx.observations)

@pytest.mark.parametrize('fast_first', [False, True])
@pytest.mark.parametrize('origin_mode', ['ordinary', 'cached_negative_nominal', 'at_header'])
def test_retained_complete_crc_refines_once_and_shares_source_registry(pcm, calls, fast_first, origin_mode):
    n = AT-round(.020*FS) if origin_mode == 'cached_negative_nominal' else AT
    rx = receiver(n)
    h, key, provisional = acquire(rx, pcm)
    origin = {'ordinary': 0, 'cached_negative_nominal': n+1, 'at_header': h}[origin_mode]
    assert origin <= h
    if origin_mode == 'cached_negative_nominal':
        assert n < origin <= h
    incomplete_end = h+round(2.*FS)
    assert rx.read_chirp(pcm[origin:incomplete_end], origin, onset=h) is None
    assert rx.read_window(pcm[origin:incomplete_end], origin, whole=True) == []
    assert calls['decode'] == [] and rx._chirp_headers[key] is not None
    fast = lambda: rx.read_chirp(pcm[origin:], origin, onset=h)
    whole = lambda: rx.read_window(pcm[origin:], origin, whole=True)
    first, second = (fast, whole) if fast_first else (whole, fast)
    rows = first()
    assert len(rows) == 1 and rows[0].observation.frame is not None
    assert rows[0].observation.kind == Kind.CHIRP
    assert rows[0].replaced is provisional.observation
    assert second() == [] and first() == []
    # The completed registration refines the prediction from N to actual H.
    # Its new cache key can acquire once on a later ordinary reread.
    assert len(calls['decode']) == 1
    assert len(calls['acquire']) == len(rx._chirp_headers)
    assert 1 <= len(calls['acquire']) <= 2
    assert len(rx.observations) == 1 and not rx.observations[0].provisional
    delivered = []
    m, arq = machine(role='irs', data='body', ladder='none')
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    m.observe(rows[0], now=h+round(3.31*FS))
    m.observe(rows[0], now=h+round(3.31*FS))
    assert delivered == [DATA] and arq.tx_acked_bytes == 0
    assert m.breakins == m.grants == 0
    evidence('retained_positive', nominal=n, onset=h, origin=origin, calls=calls,
             outcome=rows[0].outcome, delivered=delivered)

def test_nominal_zero_is_a_real_legal_acquisition(pcm, calls):
    audio = pcm[AT:]
    rx = receiver(0)
    rows = rx.read_partial(audio[:round(.810*FS)], 0, deadline=round(.810*FS))
    h = rx._chirp_headers[(0, 0.)][0]
    assert 0 <= h <= round(.035*FS)
    assert len(rows) == 1 and rows[0].observation.stream_sample == h
    assert calls['acquire'][0]['nominal_sample'] == 0
    result = rx.read_window(audio, 0, whole=True)
    assert len(result) == 1 and result[0].observation.frame is not None
    assert len(calls['decode']) == 1

@pytest.mark.parametrize('error', [-round(.035*FS), round(.035*FS),
                                   -round(.060*FS), round(.060*FS)])
def test_real_header_bracket_edges_do_not_borrow_another_onset(pcm, error):
    n = AT+error
    h = p4chirprx.acquire_header(pcm, nominal_sample=n)
    if abs(error) <= round(.035*FS):
        assert h is not None and abs(h.start_sample-n) <= round(.035*FS)
        result = p4chirprx.decode_packet(pcm, start_sample=h.start_sample)
        assert result is not None and result.frame.data.startswith(DATA)
    else:
        assert h is None
    evidence('bracket', nominal=n, header=h)

@pytest.mark.parametrize('function,kw', [
    ('acquire_header', {'nominal_sample': -1}),
    ('acquire_header', {'nominal_sample': 0, 'fs': 7999}),
    ('acquire_header', {'nominal_sample': 0, 'frequency_offset': float('nan')}),
    ('acquire_header', {'nominal_sample': 0, 'slack_s': .036}),
    ('decode_packet', {'start_sample': -1}),
    ('decode_packet', {'frequency_offset': float('inf')})])
def test_direct_strict_configuration_remains_strict(pcm, function, kw):
    with pytest.raises(ValueError):
        getattr(p4chirprx, function)(pcm, **kw)

@pytest.mark.parametrize('function', ['acquire_header', 'decode_packet'])
@pytest.mark.parametrize('audio', [np.array([np.nan]), np.array([np.inf]),
                                  np.array([1j]), np.zeros((2, 2))])
def test_direct_api_finite_real_vector_guards(function, audio):
    kw = {'nominal_sample': 0} if function == 'acquire_header' else {}
    with pytest.raises(ValueError):
        getattr(p4chirprx, function)(audio, **kw)

@pytest.mark.parametrize('fast_first', [False, True])
def test_retained_bad_crc_still_rejects_and_expires(pcm, calls, fast_first):
    bad = pcm.copy()
    # Retain the genuine marker and corrupt the body with nonzero real noise.
    start = AT+round(.24*FS)
    stop = AT+round(p4chirp.PACKET_S*FS)
    bad[start:stop] = np.random.default_rng(1201).normal(0, .1, stop-start)
    rx = receiver()
    h, key, old = acquire(rx, bad)
    assert calls['decode'] == []
    if fast_first:
        assert rx.read_chirp(bad, 0, onset=h) is None
    rows = rx.read_window(bad, 0, whole=True)
    assert any(r.outcome == Registered.EXPIRED and r.observation is old.observation for r in rows)
    assert rx._chirp_headers[key] is None
    assert rx.clock.cycle_s == 1.25
    assert len(calls['decode']) == 1
    assert any(e['event'] == 'chirp header unread' for e in rx.live_events)

@pytest.mark.parametrize('transition', ['cfo', 'emission', 'phase', 'watch_revoke', 'new_call'])
def test_actual_context_changes_do_not_revive_old_chirp_authority(pcm, calls, transition):
    rx = receiver()
    h, key, _ = acquire(rx, pcm)
    if transition == 'cfo':
        rx.clock.cfo_hz += 1
        assert not rx.has_chirp_header(h)
    elif transition == 'emission':
        rx.note_emission(Emission(Kind.WORD, Role.ISS, h+round(3.4*FS), .244))
        assert rx.clock.expects == Expect.WORD
        assert rx.clock.next_due == h+round(3.4*FS)+44880
    elif transition == 'phase':
        rx.set_receive_phase(1)
        rx.set_receive_phase(2)
        assert rx._long_phase == 2
        assert rx._chirp_headers[key] is not None  # long invalidation is not a chirp purge
    elif transition == 'watch_revoke':
        watch = SimpleNamespace(generation=7, onset=h-FS*15//4,
                                cycle=FS*15//4, cfo_hz=0., attempts=0, last_attempt=None)
        rx.sync_peer_watch(watch)
        rx.sync_peer_watch(None)
        assert rx.peer_watch is None
    else:
        rx.new_call()
        assert rx._chirp_headers == {} and rx.clock is None and not rx.expected
    owner = rx.clock
    before = len(rx.registrations)
    assert rx.read_chirp(pcm[h+1:], h+1, onset=h) is None
    assert rx.clock is owner and len(rx.registrations) == before
    assert calls['decode'] == []

@pytest.mark.parametrize('signal', ['robust', 'word', 'grant'])
def test_independent_real_short_readers_keep_their_branches(signal):
    from hfmodem.shrike import p4breakintx
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.note_emission(Emission(Kind.WORD if signal == 'robust' else Kind.SHORT,
                              Role.IRS if signal == 'robust' else Role.ISS, 0, .244))
    n = rx.clock.next_due
    wave = (p4robusttx.robust_short(13, 'A') if signal == 'robust'
            else p4robusttx.robust_word('X27') if signal == 'word'
            else p4breakintx.changeover_packet(b'', 0x40))
    audio = np.r_[np.zeros(n), wave, np.zeros(FS//10)]
    rows = rx.read_window(audio, 0, whole=True)
    assert rows
    if signal == 'robust':
        assert any(r.observation.kind == Kind.SHORT and r.observation.value == 13 for r in rows)
    elif signal == 'word':
        assert any(r.observation.kind == Kind.WORD and r.observation.word == 'X27' for r in rows)
    else:
        assert any(r.observation.kind == Kind.BODY and r.observation.frame.status.byte == 0x40 for r in rows)

@pytest.mark.parametrize('fixture,kind,end_s', [('n9vc-normal.wav', Kind.NORMAL, 2.2),
                                               ('n9vc-long-ack.wav', Kind.LONG, 4.40)])
def test_independent_real_normal_long_pcm_still_completes(fixture, kind, end_s):
    from scipy.io import wavfile
    path = Path(__file__).parent/'fixtures/p4-breakin'/fixture
    import sys
    if 'retained_audit_1012' in sys.modules:
        sys.modules['retained_audit_1012'].remember(path)
    fs, pcm = wavfile.read(path)
    audio = np.r_[np.tile(pcm[:600], 20), pcm].astype(float)/32768
    rx = P4Receiver(fs, admitted=True, chirp=True)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, round(.75*fs), .244))
    early = round((1.89 if kind == Kind.NORMAL else 1.926)*fs)
    rows = rx.read_partial(audio[:early], 0, deadline=early)
    assert any(r.observation.kind == kind and r.observation.provisional for r in rows)
    rows = rx.read_window(audio[:round(end_s*fs)], 0, whole=True)
    assert any(r.observation.kind == kind and not r.observation.provisional for r in rows)

@pytest.mark.parametrize('evidence_kind', ['unread', 'live', 'mask'])
def test_lost_chirp_watch_finishes_actual_nominal_once_without_invented_hearing(pcm, calls, evidence_kind):
    if evidence_kind == 'mask':
        # Leave room for a genuinely decoded, earlier peer WORD and its
        # completed accounting. Initial history precedes the retained read.
        pcm = np.r_[np.zeros(4*FS), pcm]
    rx = receiver(AT+4*FS if evidence_kind == 'mask' else AT)
    packet_clock = rx.clock
    rx.enable_hold()
    if evidence_kind == 'mask':
        rx.note_emission(Emission(Kind.SHORT, Role.ISS,
                                  packet_clock.next_due-60000-44880, .758))
        prior = rx.clock.next_due
        word_pcm = np.r_[np.zeros(prior), p4robusttx.robust_word('Z27'), np.zeros(FS//10)]
        records = rx.read_window(word_pcm, 0, whole=True)
        assert records and records[0].observation.word == 'Z27'
        assert rx.peer_onset == records[0].observation.stream_sample
        assert rx.finish_cycle(rx.peer_onset) == 0
    # The supplied initial PACKET owner scopes acquisition without repairing
    # the primary WORD clock or changing its actual emission association.
    with rx.scoped_clock(packet_clock):
        h, key, _ = acquire(rx, pcm)
    # Initial semantic watch history, with a real existing primary WORD owner.
    watch = SimpleNamespace(generation=9, onset=h-FS*15//4,
                            cycle=FS*15//4, cfo_hz=0., attempts=2, last_attempt=0)
    rx.sync_peer_watch(watch)
    rx.note_emission(Emission(Kind.WORD, Role.ISS, h-44880+12000, .244))
    primary = rx.clock
    saved = rx.peer_watch
    if evidence_kind == 'mask':
        from hfmodem.shrike import p4breakintx
        masked_wave = p4breakintx.changeover_packet(b'local history', 0)
        rx.note_emission(Emission(Kind.BODY, Role.ISS, h-48, len(masked_wave)/FS))
        primary = rx.clock
        assert rx._masked(h)
    elif evidence_kind == 'live':
        # An actual associated WORD decode supplies qualifying liveness.
        due = primary.next_due
        wave = p4robusttx.robust_word('X27')
        rows = rx.read_window(np.r_[np.zeros(due), wave, np.zeros(FS//10)], 0, whole=True)
        assert rows and rows[0].observation.word == 'X27'
        assert rows[0].responds_to == rx.last_emission.stream_sample
        assert rx.live_onsets
        primary = rx.clock
    live_before = tuple(rx.live_onsets)
    with rx.scoped_clock(saved.clock, watch_generation=saved.generation):
        rows = rx.read_window(pcm[h+1:], h+1, whole=True)
        assert not any(r.observation.kind == Kind.CHIRP for r in rows)
    assert rx.clock is primary and primary.expects == Expect.WORD
    assert tuple(rx.live_onsets) == live_before
    due = saved.clock.next_due
    assert due == h
    expected = 1 if evidence_kind == 'unread' else 0
    assert rx.finish_peer_watch(saved, due) == expected
    assert saved.clock.next_due == due+180000
    assert rx.finish_cycle(due) == expected
    assert saved.clock.next_due == due+180000
    assert calls['decode'] == []
    evidence('watch_accounting', kind=evidence_kind, nominal=due, origin=h+1,
             end=len(pcm), owner_due=primary.next_due, silent=rx.silent_cycles)

@pytest.mark.parametrize('lag', [1, round(.020*FS), round(.900*FS), round(3.400*FS)])
def test_driver_initial_history_late_watch_records_trimmed_support_and_charge(pcm, lag, monkeypatch):
    # Genuine complete PCM establishes initial machine history before launch;
    # current input starts after the next watched prefix, and a later new packet
    # remains available. This is explicitly a constructed late-start capture.
    history = receiver()
    history_pcm = np.r_[np.zeros(AT), p4chirp.entry_packet(DATA, 0x41), np.zeros(FS//10)]
    records = history.read_window(history_pcm, 0, whole=True)
    assert len(records) == 1 and records[0].observation.frame is not None
    h = records[0].observation.stream_sample
    n = h+180000
    origin = n+lag
    next_at = n+180000
    stop = next_at+round(3.65*FS)
    wave = p4chirp.entry_packet(b'following source\r', 0x02)
    full = np.random.default_rng(101201).normal(0, .0001, stop)
    first_wave = p4chirp.entry_packet(DATA, 0x01)
    full[n:n+len(first_wave)] += first_wave
    full[next_at:next_at+len(wave)] += wave
    stream = AbsoluteStream(full[origin:], origin)
    tx = Tx(stream)
    tx.settle = .025
    tx.p4_probe.emissions = []
    tx.p4_probe.peer_breakin = True
    opts = tx.p4_link_options
    opts.p4_ladder, opts.p4_data = 'none', 'body'
    opts.p4_link_timeout = (stop-origin)/FS
    rx = tx.sessrx.p4
    rx.chirp = rx.expected = True
    rx.cfo_hz = 0.
    machines, reads, finishes, peers, advances, owners = [], [], [], [], [], []
    delivered = []
    tx.p4_link_host.deliver = delivered.append
    tx.p4_link_host.arq.on_host_data(b'our queued source\r')
    cls = p4turn.P4Turn
    def factory(*args, **kw):
        kw['role'] = 'irs'
        m = cls(*args, **kw)
        m.observe(records[0], now=h+round(3.31*FS))
        answer = m.decide(h+round(3.4*FS)-m.lead)
        assert answer is not None and answer.kind == Kind.WORD and answer.word == 'Y27'
        assert answer.state == p4turn.State.IRS_BREAKIN
        m.keyed(answer.stream_sample)
        # Initial actual emission history gives associated WORD; the saved
        # independent watch is behind this late capture's support.
        rx.sync_peer_watch(m.peer_watch)
        rx.note_emission(Emission(Kind.WORD, Role.ISS, answer.stream_sample,
                                 len(p4robusttx.robust_word('Y27'))/FS))
        assert rx.clock.next_due == answer.stream_sample+44880
        machines.append(m)
        original = m.peer_cycle
        def peer(*a, **k):
            peers.append((a, stream.pos))
            return original(*a, **k)
        m.peer_cycle = peer
        return m
    from contextlib import contextmanager
    scoped = rx.scoped_clock
    @contextmanager
    def owned(clock, **kw):
        primary = rx.clock
        before = (primary.expects, primary.next_due) if primary else None
        due = clock.next_due
        with scoped(clock, **kw):
            yield
        owners.append((due, before, rx.clock is primary,
                       (rx.clock.expects, rx.clock.next_due) if rx.clock else None))
        if due == n:
            assert rx.clock is primary and primary.expects == Expect.WORD
    monkeypatch.setattr(rx, 'scoped_clock', owned)
    finishing = rx.finish_peer_watch
    def advance(watch, onset):
        before = watch.clock.next_due
        silent = finishing(watch, onset)
        advances.append((onset, before, watch.clock.next_due, silent))
        if onset == n:
            assert before == n and watch.clock.next_due == n+180000
        return silent
    monkeypatch.setattr(rx, 'finish_peer_watch', advance)
    for name in ('read_partial', 'read_window'):
        original = getattr(rx, name)
        def read(audio, begin, _original=original, _name=name, **kw):
            clock = rx.clock
            owner = rx._watch_read_generation
            rows = _original(audio, begin, **kw)
            reads.append(dict(name=_name, origin=begin, end=begin+len(audio),
                              clipped_end=(min(begin+len(audio), max(begin, kw['deadline']))
                                           if _name == 'read_partial' else begin+len(audio)),
                              due=clock.next_due, expects=clock.expects, owner=owner,
                              rows=[(r.observation.kind, r.observation.stream_sample,
                                     r.observation.frame is not None) for r in rows]))
            return rows
        monkeypatch.setattr(rx, name, read)
    original = rx.finish_cycle
    def finish(onset):
        before = rx.silent_cycles
        previous = rx._counted_cycle
        result = original(onset)
        finishes.append((onset, previous, before, result, stream.pos,
                         tuple(rx.live_onsets), rx._masked(onset)))
        return result
    monkeypatch.setattr(rx, 'finish_cycle', finish)
    monkeypatch.setattr(p4turn, 'P4Turn', factory)
    onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    watched = [r for r in reads if r['owner'] is not None and r['due'] == n]
    assert {r['name'] for r in watched} == {'read_partial', 'read_window'}
    assert all(r['origin'] >= origin > n for r in watched)
    assert all(not any(kind == Kind.CHIRP for kind, at, valid in r['rows']) for r in watched)
    charges = [f for f in finishes if f[0] == n]
    assert charges and charges[0][2:4] == (0, 1)
    assert not charges[0][5] and not charges[0][6]
    assert any(args == (1, n) for args, at in peers)
    assert all(after == before for at, previous, before, after, end, live, mask in finishes
               if previous is not None and at <= previous+FS//10)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    assert tx.p4_link_host.arq._inflight is None
    assert bytes(tx.p4_link_host.arq._outbuf) == b'our queued source\r'
    assert any(r['kind'] == Kind.CHIRP and not r['provisional']
               and abs(r['onset_sample']-next_at) <= 1680 for r in tx.p4_probe.heard)
    # The first delivery belongs to explicit initial decoded history.
    assert delivered == [DATA, b'following source\r']
    accepted = [r['onset_sample'] for r in tx.p4_probe.heard
                if r['kind'] == Kind.CHIRP and not r['provisional']
                and abs(r['onset_sample']-next_at) <= 1680]
    assert any(e['event'] == 'Keyed' and e.get('accepted_target') in accepted
               for e in tx.p4_probe.link_events)
    evidence('driver_late_watch', lag=lag, origin=origin, reads=reads,
             finishes=finishes, peers=peers, advances=advances, owners=owners,
             keys=tx.p4_probe.emissions,
             delivered=delivered, reason=machines[0].reason)
