# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""S10: recorded IRS role flips, 0058 seed words, and k0003 hand-back."""

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4oracle
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import Proposal, State
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_data_turn_0930 import CYCLE, T, receive
from hfmodem.tests.shrike.test_p4_bare_y27_0929 import k0003

ROOT = Path(__file__).resolve().parents[5]


@lru_cache
def tape(arm):
    path = ROOT / f'captures/onair-{arm}'
    if not (path / 'stream.flac').exists():
        pytest.skip(f'local {arm} recording absent')
    return (onair.session.load_wav(str(path / 'stream.flac'), FS),
            json.loads((path / 'p4-probe.json').read_text()))


def recorded_word(arm, seconds, expected="V27"):
    audio, _ = tape(arm)
    at = round(seconds * FS)
    origin = at - round(.08 * FS)
    obs = p4oracle.word_at(audio[origin:at + round(.32 * FS)], FS,
                           origin, (at-origin)/FS, cfo_hz=3., whole=True)
    assert obs is not None and obs.word == expected and not obs.provisional
    return obs


def empty_hold(obs, counter=0):
    # Counterfactual drained turn: 0058 itself never settled its proposal.
    m, a = machine(data='body', ladder='none', hold_cycles=16)
    m.counter = counter
    m._first_data = False
    m._data(obs.stream_sample - round(.930 * FS))
    m.keyed()
    return m, a


def offer_from_0058(counter=0):
    obs = recorded_word('0930-0058', 44.096)
    m, a = empty_hold(obs, counter)
    deadline = m._hold_until
    m.observe(RegisteredObservation(obs, Registered.NEW, responds_to=m.last.stream_sample),
              now=obs.stream_sample + round(.30 * FS))
    p = m.decide(obs.stream_sample + round(.30 * FS))
    assert p is not None and (p.kind, p.status, p.payload) == (Kind.BODY, 0x40, b'')
    assert str(p.state) == 'ISS_HANDBACK'
    assert (p.stream_sample-m.last.stream_sample) % CYCLE == 0
    assert p.stream_sample >= obs.stream_sample + round(.30 * FS) + m.lead
    assert m.counter == counter and a._buffer_raw == 0
    assert m._hold_until == p.stream_sample + 4*CYCLE
    assert m._hold_until <= deadline
    m.keyed()
    return m, a


@pytest.mark.parametrize('arm,onset', [('0929-1857', 27.843), ('0929-2253', 24.100)])
def test_recorded_irs_v27_flip_offers_empty_body_on_our_raster(arm, onset):
    obs = recorded_word(arm, onset)
    _, meta = tape(arm)
    prior = next(e for e in reversed(meta['emissions'])
                 if e.get('word') and e['audio_start'] < obs.stream_sample)
    m, a = machine(role='irs', data='body', ladder='none')
    m._next = Proposal(Kind.WORD, prior['audio_start'], word=prior['word'],
                       state=State.IRS_ANSWER)
    m.keyed()
    m.observe(RegisteredObservation(obs, Registered.NEW),
              now=obs.stream_sample + round(.30 * FS))
    p = m.decide(obs.stream_sample + round(.30 * FS))
    assert p is not None and (p.kind, p.status, p.payload) == (Kind.BODY, 0x40, b'')
    assert (p.stream_sample-prior['audio_start']) % CYCLE == 0
    assert p.stream_sample >= obs.stream_sample + round(.30 * FS) + m.lead
    assert m.counter == 0 and a._buffer_raw == 0


@pytest.mark.parametrize('idle_ack', [False, True])
def test_recorded_irs_flip_starts_a_fresh_counter_only_when_offer_keys(idle_ack):
    obs = recorded_word('0929-1857', 27.843)
    _, meta = tape('0929-1857')
    prior = meta['emissions'][-1]
    m, a = machine(role='irs', data='body', ladder='none')
    m.counter = 3
    m._first_data = False
    m._next = Proposal(Kind.WORD, prior['audio_start'], word=prior['word'], state=State.IRS_ANSWER)
    m.keyed()
    m.observe(RegisteredObservation(obs, Registered.NEW), now=obs.stream_sample+round(.3*FS))
    assert m._next.kind == Kind.BODY and m.counter == 3
    m.keyed()
    assert m.counter == 0
    at = m.last.stream_sample + T
    m.observe(word('V27', at, responds_to=m.last.stream_sample), now=at+round(.3*FS))
    m.keyed()
    if idle_ack:
        a.on_host_data(b'new bytes')
        at = m.last.stream_sample+T
        m.observe(word('X27', at, responds_to=m.last.stream_sample), now=at+round(.3*FS))
        p = m.decide(at+round(.3*FS))
    else:
        a.on_host_data(b'new bytes')
        p = m.decide(m.last.stream_sample+CYCLE-m.lead)
    assert (p.kind, p.status, p.payload) == (
        Kind.NORMAL if idle_ack else Kind.BODY, 1 if idle_ack else 0x40, b'new bytes')
    assert a._buffer_raw == 9


@pytest.mark.parametrize('progress', ['our-data', 'peer-data'])
def test_recorded_seed_can_recover_again_only_after_data_progress(progress):
    m, a = offer_from_0058()
    at = m.last.stream_sample+T
    if progress == 'our-data':
        m.observe(word('V27', at, responds_to=m.last.stream_sample), now=at+round(.3*FS))
        m.keyed()
        a.on_host_data(b'accepted')
        m.decide(m.last.stream_sample+CYCLE-m.lead)
        packet = m.keyed()
        # Explicitly scripted parity acceptance; 0058 accepted no such data.
        at = packet.stream_sample+T
        m.observe(word('X27', at, responds_to=packet.stream_sample), now=at+round(.3*FS))
        idle = m.keyed()
        at = idle.stream_sample+T
        m.observe(word('V27', at, responds_to=idle.stream_sample), now=at+round(.3*FS))
        m.keyed()
    else:
        receive(m, at, b'FS Y\r', 0x40, responds_to=m.last.stream_sample)
        m.keyed()
    obs = recorded_word('0930-0058', 44.096)
    obs = replace(obs, stream_sample=m.last.stream_sample+round(.930*FS))
    m.observe(RegisteredObservation(obs, Registered.NEW, responds_to=m.last.stream_sample),
              now=obs.stream_sample+round(.3*FS))
    assert m._next.kind == Kind.BODY and m._next.status == 0x40
    assert str(m._next.state) == 'ISS_HANDBACK' and a._buffer_raw == 0


@pytest.mark.parametrize('counter', [0, 2])
def test_0058_empty_hold_offers_once_and_seed_never_confirms_handback(counter):
    m, a = offer_from_0058(counter)
    deadline = m._hold_until
    close = None
    for seconds in (46.596, 47.846, 49.096):
        obs = recorded_word('0930-0058', seconds)
        # Translate this real seed to the answer slot of our proposed packet.
        obs = replace(obs, stream_sample=m.last.stream_sample + round(.930 * FS))
        m.observe(RegisteredObservation(obs, Registered.NEW, responds_to=m.last.stream_sample),
                  now=obs.stream_sample + round(.30 * FS))
        p = m.decide(obs.stream_sample + round(.30 * FS))
        if p.state == State.ISS_CLOSE:
            close = p
            break
        assert (p.kind, p.status, p.payload, p.state) == (Kind.NORMAL, 0, b'', State.ISS_HOLD)
        m.keyed()
        assert m.counter == 0 and a._buffer_raw == 0 and m._hold_until == deadline
    close = close or m.decide(deadline)
    assert (close.kind, close.status, close.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert close.stream_sample <= deadline
    assert m.counter == 0 and a._buffer_raw == 0


def test_k0003_checked_empty_reply_yields_and_peer_resumes():
    m, a = offer_from_0058()
    audio = k0003(1481.712, 1482.812)
    heads = p4oracle.word_scan(audio, FS)
    frames = [p4oracle.body_after(audio, FS, 0, h) for h in heads if h.word == 'Y27']
    frame = next(f for f in frames if f is not None)
    assert frame.status.byte == 0x40 and frame.data == b'\x1e' * 31
    at = m.last.stream_sample + round(.928 * FS)
    p = receive(m, at, frame.data, frame.status.byte, responds_to=m.last.stream_sample)
    assert (p.kind, p.word, p.state) == (Kind.WORD, 'V27', State.IRS_ANSWER)
    assert p.stream_sample == at + T and m.counter == 0 and a._buffer_raw == 0
    m.keyed()
    assert m._hold_until is None
    # The next packet is the recorded v13, translated with its +.330 spacing.
    audio = k0003(1482.960, 1483.960)
    result = p4oracle.robust_at(audio, FS, .1, 4.8)
    assert result['p4_value'] == 13 and result['p4_hdr'] == 'A'
    at = m.last.stream_sample + round(.330 * FS)
    obs = Observation(Kind.SHORT, FS, at, 0, at, .758, 4.8,
                      result['coherence'], value=result['p4_value'], header=result['p4_hdr'])
    m.observe(RegisteredObservation(obs, Registered.NEW), now=at+round(.8*FS))
    p = m.decide(at+T-m.lead)
    assert p.kind == Kind.WORD and p.word == 'X27' and p.state == State.IRS_ANSWER


@pytest.mark.parametrize('case', ['partial', 'weak', 'unexpected', 'expired', 'duplicate',
                                 'unrelated', 'queued', 'inflight', 'odd', 'long-extent'])
def test_0058_seed_does_not_override_ack_or_unsettled_bytes(case):
    obs = recorded_word('0930-0058', 44.096)
    m, a = empty_hold(obs, 1 if case == 'odd' else 0)
    if case in ('queued', 'inflight'):
        a.on_host_data(b'not accepted')
        if case == 'inflight':
            m.outbox.transfer()
    obs = replace(obs, provisional=case == 'partial', weak=case == 'weak',
                  unexpected=case == 'unexpected', extent_s=.8 if case == 'long-extent' else obs.extent_s)
    outcome = {'expired': Registered.EXPIRED, 'duplicate': Registered.DUPLICATE}.get(case, Registered.NEW)
    m.observe(RegisteredObservation(obs, outcome,
              responds_to=-1 if case == 'unrelated' else m.last.stream_sample),
              now=obs.stream_sample + round(.30 * FS))
    assert m._next is None or str(m._next.state) != 'ISS_HANDBACK'
    assert a._buffer_raw == (12 if case in ('queued', 'inflight') else 0)


def test_0058_handback_silence_has_four_copy_budget_and_no_ladder_restart():
    m, _ = offer_from_0058()
    for _ in range(3):
        now = m.last.stream_sample + CYCLE - m.lead
        m.silence(now)
        p = m.decide(now)
        assert p.kind == Kind.BODY and p.status == 0x40
        m.keyed()
    now = m.last.stream_sample + CYCLE - m.lead
    m.silence(now)
    p = m.decide(now)
    assert p.kind == Kind.BODY and p.status == 0x20 and p.state == State.ISS_CLOSE


@pytest.mark.parametrize('keyed', [False, True])
@pytest.mark.parametrize('resolution', ['confirm', 'expire', 'contradict'])
def test_provisional_seed_reserves_first_raster_and_retracts_on_expiry(keyed, resolution):
    m, a = offer_from_0058()
    offer = m.last
    at = offer.stream_sample + T
    early = word('V27', at, partial=True, responds_to=offer.stream_sample)
    m.observe(early)
    now = offer.stream_sample + CYCLE - m.lead
    m.silence(now)
    p = m.decide(now)
    assert (p.kind, p.status, p.stream_sample) == (Kind.NORMAL, 0, offer.stream_sample+CYCLE)
    if keyed:
        m.keyed()
    outcome = dict(confirm=Registered.CONFIRMED, expire=Registered.EXPIRED,
                   contradict=Registered.CONTRADICTED)[resolution]
    m.observe(word('V27' if resolution != 'contradict' else 'T27', at, outcome=outcome,
                   replaced=early.observation, responds_to=offer.stream_sample,
                   answered=p.stream_sample if keyed else None), now=now+round(.05*FS))
    if resolution == 'confirm':
        assert (m.last if keyed else m._next).kind == Kind.NORMAL
        assert len([e for e in m.link_events if e['event'] == 'handback-seed']) == 1
    else:
        assert m._next.kind == Kind.BODY and m._next.status == 0x40
        assert m._next.stream_sample >= now+round(.05*FS)+m.lead
        assert not any(e['event'] == 'handback-seed' for e in m.link_events)
    assert m.counter == 0 and a._buffer_raw == 0


@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('counter', [0, 2])
def test_recorded_0117_x27_accepts_offer_and_continues_normal_c1(queued, counter):
    m, a = offer_from_0058(counter)
    deadline = m._hold_until
    obs = recorded_word('0930-0117', 25.947, 'X27')
    obs = replace(obs, stream_sample=m.last.stream_sample+T)
    if queued:
        a.on_host_data(b'new bytes')
    m.observe(RegisteredObservation(obs, Registered.NEW, responds_to=m.last.stream_sample),
              now=obs.stream_sample+round(.3*FS))
    p = m.decide(obs.stream_sample+round(.3*FS))
    assert (p.kind, p.status, p.payload) == (Kind.NORMAL, 1, b'new bytes' if queued else b'')
    assert m.counter == 1 and not m._first_data and not m._word_yield
    assert m._hold_until == deadline and a._buffer_raw == (9 if queued else 0)
    assert any(e['event'] == 'handback-accepted' for e in m.link_events)
    assert p.status != 0x20


@pytest.mark.parametrize('case', ['provisional', 'weak', 'unexpected', 'unrelated', 'expired', 'duplicate'])
def test_recorded_0117_x27_offer_acceptance_requires_full_associated_word(case):
    m, _ = offer_from_0058()
    obs = recorded_word('0930-0117', 25.947, 'X27')
    obs = replace(obs, stream_sample=m.last.stream_sample+T,
                  provisional=case == 'provisional', weak=case == 'weak', unexpected=case == 'unexpected')
    outcome = {'expired': Registered.EXPIRED, 'duplicate': Registered.DUPLICATE}.get(case, Registered.NEW)
    m.observe(RegisteredObservation(obs, outcome, responds_to=-1 if case == 'unrelated' else m.last.stream_sample))
    assert m.counter == 0 and not any(e['event'] == 'handback-accepted' for e in m.link_events)


@pytest.mark.parametrize('provisional', [False, True])
def test_two_recorded_z27_refusals_close_offer_without_four_copies(provisional):
    m, _ = offer_from_0058()
    for copy in range(2):
        obs = recorded_word('0930-0117', 28.447, 'Z27')
        obs = replace(obs, stream_sample=m.last.stream_sample+T)
        now = m.last.stream_sample+CYCLE-m.lead
        early = replace(obs, provisional=True) if provisional else None
        if early is not None:
            m.observe(RegisteredObservation(early, Registered.NEW, responds_to=m.last.stream_sample))
            m.silence(now)
            assert m.decide(now) is None
        m.observe(RegisteredObservation(obs, Registered.CONFIRMED if provisional else Registered.NEW,
                                       early, responds_to=m.last.stream_sample), now=now)
        m.silence(now)
        p = m.decide(now)
        assert p.status == (0x40 if copy == 0 else 0x20)
        m.keyed()
    assert m.counter == 0


def test_recorded_0058_peer_body_after_repeated_grant_starts_with_x27():
    m, a = offer_from_0058()
    for _ in range(2):
        p = receive(m, m.last.stream_sample + T, responds_to=m.last.stream_sample)
        assert p.kind == Kind.WORD and p.word == 'V27'
        m.keyed()
    audio, _ = tape('0930-0058')
    at = round(68.446*FS)
    origin = at-round(.1*FS)
    segment = audio[origin:at+FS]
    heads = p4oracle.word_scan(segment, FS)
    frames = [p4oracle.body_after(segment, FS, 0, h) for h in heads if h.word == 'Y27']
    frame = next(f for f in frames if f is not None)
    assert frame.status.byte == 0 and frame.data == b'\x1e'*31
    p = receive(m, m.last.stream_sample+round(.330*FS), frame.data, frame.status.byte)
    assert p.kind == Kind.WORD and p.word == 'X27' and p.state == State.IRS_ANSWER
    assert a._buffer_raw == 0


@pytest.mark.parametrize('case', ['off-phase', 'queued', 'queued-gated', 'inflight',
                                  'hold-policy', 'chirp-policy', 'no-key'])
def test_recorded_irs_seed_requires_empty_data_mode_and_our_key_phase(case):
    obs = recorded_word('0929-1857', 27.843)
    _, meta = tape('0929-1857')
    prior = meta['emissions'][-1]
    policy = ({} if case == 'hold-policy' else dict(data='chirp', ladder='chirp')
              if case == 'chirp-policy' else dict(data='body', ladder='none'))
    m, a = machine(role='irs', **policy)
    if case != 'no-key':
        m._next = Proposal(Kind.WORD, prior['audio_start'], word=prior['word'], state=State.IRS_ANSWER)
        m.keyed()
    if case == 'off-phase':
        obs = replace(obs, stream_sample=obs.stream_sample+round(.1*FS))
    if case in ('queued', 'queued-gated', 'inflight'):
        a.on_host_data(b'waiting')
        if case == 'inflight':
            m.outbox.transfer()
        if case == 'queued-gated':
            m.allow_breakin = False
    m.observe(RegisteredObservation(obs, Registered.NEW), now=obs.stream_sample+round(.3*FS))
    p = m.decide(obs.stream_sample+round(.3*FS))
    if case in ('queued', 'queued-gated', 'chirp-policy'):
        assert (p.kind, p.status, p.payload, p.state) == (
            Kind.BODY, 0x40, b'', State.IRS_BREAKIN if case == 'queued' else State.ISS_HANDBACK)
    else:
        assert p is None
    assert a._buffer_raw == (7 if case in ('queued', 'queued-gated', 'inflight') else 0)


@pytest.mark.parametrize('arm,reply', [('0929-2253', 'seed'), ('0929-2253', 'grant')])
def test_real_driver_keys_handback_and_reads_reply(monkeypatch, arm, reply):
    from hfmodem.shrike import p4turn, p4breakintx
    from hfmodem.shrike.p4receiver import Emission, Role
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    import numpy as np

    audio, meta = tape(arm)
    audio = audio.copy()
    last = meta['emissions'][-1]
    live = Stream(audio)
    live.pos = last['audio_end']
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_link_timeout = 12.
    rx = tx.sessrx.p4
    rx.cfo_hz = 1.
    rx.note_emission(Emission(Kind.WORD, Role.IRS, last['audio_start'],
                              (last['audio_end']-last['audio_start'])/FS))
    tx.p4_probe.handoff_audio = live.pos, audio[live.pos:live.pos].copy()
    original = p4turn.P4Turn
    def start(*a, **kw):
        m = original(*a, **{**kw, 'role': 'irs'})
        m._next = Proposal(Kind.WORD, last['audio_start'], word=last['word'], state=State.IRS_ANSWER)
        m.keyed()
        return m
    monkeypatch.setattr(p4turn, 'P4Turn', start)
    waves = []
    transmit = tx._tx
    def keyed(wave, label):
        waves.append(wave.copy())
        transmit(wave, label)
        live.pos = tx.tx_end
        if reply == 'grant' and len(waves) == 1:
            # Script the response from recorded k0003 audio; 2253 never granted.
            answer = tx.tx_audio_start + round(.928 * FS)
            audio[tx.tx_end:] = 0
            grant = k0003(1481.712, 1482.712)
            origin = answer-round(.1*FS)
            audio[origin:origin+len(grant)] = grant
            short = k0003(1482.960, 1483.960)
            origin = answer+T+round(.330*FS)-round(.1*FS)
            audio[origin:origin+len(short)] = short
    tx._tx = keyed
    onair._run_p4_link(tx, live, onair._MasterGrid(0, CYCLE, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    offers = [e for e in tx.p4_probe.emissions if e.get('status') == 0x40]
    assert len(offers) == 1
    assert (offers[0]['audio_start']-last['audio_start']) % CYCLE == 0
    np.testing.assert_array_equal(waves[0], p4breakintx.changeover_packet(b'', 0x40))
    if reply == 'seed':
        assert any(e['event'] == 'handback-seed' for e in tx.p4_probe.link_events)
        idles = [e for e in rx.emissions if e.kind == Kind.NORMAL]
        assert idles and idles[0].stream_sample == offers[0]['audio_start']+CYCLE
        assert not any(e['event'] == 'Refused' and e.get('reason') == 'deadline'
                       and e['at'] > offers[0]['audio_start'] for e in tx.p4_probe.link_events)
        assert len([e for e in tx.p4_probe.link_events if e['event'] == 'handback-offer']) == 1
    else:
        requests = {e['at'] for e in tx.p4_probe.link_events
                    if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
        words = [e for e in tx.p4_probe.emissions
                 if e.get('word') and e['audio_start'] not in requests]
        assert [e['word'] for e in words] == ['V27', 'X27']
        assert any(o.kind == Kind.BODY and o.frame is not None and o.frame.status.byte == 0x40
                   for o in rx.observations)
        assert any(o.kind == Kind.SHORT and o.value == 13 for o in rx.observations)
    assert tx.p4_link_host.arq._buffer_raw == 0


def test_recorded_2118_full_driver_keys_seed_fallback_without_deadline_refusal():
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx

    audio, meta = tape('0929-2118')
    pre = []
    for e in meta['emissions']:
        if e.get('value') is None:
            break
        pre.append(e)
    live = Stream(audio[:round(70*FS)])
    live.pos = pre[-1]['audio_end']+2000
    tx = Tx(live)
    tx.p4_probe.emissions = [dict(audio_start=e['audio_start'], audio_end=e['audio_end']) for e in pre]
    o = tx.p4_link_options
    o.p4_ladder, o.p4_data, o.p4_hold_cycles = 'none', 'body', 16
    o.p4_normal_ack, o.p4_breakin = 'parity', 'body'
    o.p4_link_timeout = 70-live.pos/FS
    onair._run_p4_link(tx, live, onair._MasterGrid(pre[0]['audio_start'], CYCLE, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    events = tx.p4_probe.link_events
    offers = [e for e in events if e['event'] == 'Keyed'
              and e['kind'] == Kind.BODY and e.get('state') == State.ISS_HANDBACK]
    closes = [e for e in events if e['event'] == 'Keyed'
              and e['kind'] == Kind.BODY and e.get('status') == 0x20]
    assert offers and closes and closes[0]['at'] <= offers[0]['at']+4*CYCLE
    assert not any(e['event'] == 'Refused' and e.get('reason') == 'deadline' for e in events)
    assert len([e for e in events if e['event'] == 'handback-offer']) == 1
    assert len([e for e in events if e['event'] == 'handback-seed']) == 1
    assert tx.p4_link_host.arq._buffer_raw == 0


def test_provisional_x27_waits_for_confirmation_before_accepting_offer():
    m, a = offer_from_0058()
    offer = m.last
    obs = recorded_word('0930-0117', 25.947, 'X27')
    obs = replace(obs, stream_sample=offer.stream_sample+T)
    early = replace(obs, provisional=True)
    m.observe(RegisteredObservation(early, Registered.NEW, responds_to=offer.stream_sample))
    now = offer.stream_sample+CYCLE-m.lead
    m.silence(now)
    assert m.decide(now) is None and m.counter == 0
    m.observe(RegisteredObservation(obs, Registered.CONFIRMED, early,
                                   responds_to=offer.stream_sample), now=now+round(.05*FS))
    assert (m._next.kind, m._next.status) == (Kind.NORMAL, 1)
    assert a._buffer_raw == 0


def test_expired_speculative_seeds_keep_four_offer_copy_budget():
    m, _ = offer_from_0058()
    deadline = m._hold_until
    for copy in range(4):
        offer = m.last
        early = word('V27', offer.stream_sample+T, partial=True, responds_to=offer.stream_sample)
        m.observe(early)
        idle = m.keyed()
        m.observe(RegisteredObservation(early.observation, Registered.EXPIRED,
                                       early.observation, idle.stream_sample,
                                       responds_to=offer.stream_sample), now=idle.stream_sample)
        assert m._next.status == (0x40 if copy < 3 else 0x20)
        assert m._hold_until == deadline
        m.keyed()
    assert len([e for e in m.link_events if e['event'] == 'handback-offer']) == 1


@pytest.mark.parametrize('name', ['V27', 'X27', 'Z27'])
def test_reply_to_retracted_speculative_idle_cannot_resolve_the_offer(name):
    m, _ = offer_from_0058()
    offer = m.last
    early = word('V27', offer.stream_sample+T, partial=True, responds_to=offer.stream_sample)
    m.observe(early)
    idle = m.keyed()
    m.observe(RegisteredObservation(early.observation, Registered.EXPIRED,
                                   early.observation, idle.stream_sample,
                                   responds_to=offer.stream_sample), now=idle.stream_sample)
    retry = m._next
    at = idle.stream_sample+T
    m.observe(word(name, at, responds_to=idle.stream_sample), now=at)
    assert m._next is retry and m.counter == 0
    assert not any(e['event'] in ('handback-seed', 'handback-accepted') for e in m.link_events)
