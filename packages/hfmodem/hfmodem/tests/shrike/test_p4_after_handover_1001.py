# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""2028/2011 word yields and 2031/0103 IRS holds, from the recorded probe."""
from dataclasses import replace

import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import p4turn
from hfmodem.shrike.p4observation import Registered
from hfmodem.shrike.p4receiver import Kind
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.p4_after_handover_helpers import probe, replay
from hfmodem.tests.shrike.test_p4turn import FS, body, machine, word
from hfmodem.tests.shrike.test_p4_close_refused_0930 import answered, BLOCK


@pytest.mark.parametrize('arm,at', [('2028', 1651906), ('2011', 1471853)])
def test_probe_replay_starts_v13_after_v27(arm, at):
    tx = replay(arm, end=at/FS+2)
    sent = next(e for e in tx.p4_probe.emissions if abs(e['audio_start']-at) < FS//100)
    assert abs(sent['audio_start']-at) <= 4
    assert sent['value'] == 13 and not sent['payload']
    assert tx.p4_link_host.arq._buffer_raw > 0


def yielded(arm='2028'):
    recorded = probe(arm)
    h = next(h for h in recorded['heard'] if h.get('word') == 'V27'
             and not h['provisional'] and h['outcome'] == 'confirmed')
    m, a = machine(role='irs', ladder='chirp', data='chirp', normal_ack='parity', prekey_lead=.044)
    a.on_host_data(BLOCK)
    m._body(h['responds_to'], 0x40, State.IRS_BREAKIN)
    m.keyed()
    m.observe(word('V27', h['onset_sample'], responds_to=h['responds_to']),
              now=h['received_at_sample'])
    return m, a


def test_word_yield_ladder_then_chirp_counter_zero_and_parity():
    m, a = yielded()
    first = m.keyed()
    assert (first.kind, first.value, first.stream_sample) == (Kind.SHORT, 13, 1651906)
    assert not m.outbox.inflight and bytes(a._outbuf) == BLOCK
    assert answered(m, 'X27').value == 49
    assert answered(m, 'X27').value == 53
    chirp = answered(m, 'X27')
    assert (chirp.kind, chirp.status, chirp.payload) == (Kind.CHIRP, 0x60, BLOCK[:22])
    second = answered(m, 'X27')
    assert (second.kind, second.status, second.payload) == (Kind.CHIRP, 0x61, BLOCK[22:44])


def test_recorded_yield_with_scripted_ladder_acks_keys_prepared_0x60_chirp(monkeypatch):
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike import p4_stock_peer as stock

    load = onair.session.load_wav

    def continuation(*args, **kwargs):
        audio = load(*args, **kwargs).copy()
        audio[round(34.40*FS):] = 0
        # The recording's body carrier is about -5 Hz; X27 adds its own offset.
        wave = stock.word('X27', cfo=-5.)
        for at in (35.345, 36.595, 37.845):
            start = round(at*FS)
            audio[start:start+len(wave)] = wave
        return audio

    monkeypatch.setattr(onair.session, 'load_wav', continuation)
    tx = replay('2028', end=40.3)
    chirps = [e for e in tx.p4_probe.emissions if e.get('payload')]
    assert chirps and chirps[0]['status'] == 0x60
    assert len(bytes.fromhex(chirps[0]['payload'])) == 22
    assert abs(chirps[0]['audio_start']/FS-38.165) < .01
    assert chirps[0]['audio_end']-chirps[0]['audio_start'] > 3*FS
    assert not any(e.get('reason') == 'waveform not prepared' for e in tx.p4_probe.link_events)


def test_held_seed_repeats_v13_with_header_b_without_spending_data_refusals():
    m, a = yielded()
    m.keyed()
    repeat = answered(m, 'V27')
    assert (repeat.kind, repeat.value, repeat.header) == (Kind.SHORT, 13, 'B')
    assert not m._data_refusals and bytes(a._outbuf) == BLOCK


@pytest.mark.parametrize('arm', ['2028', '2011'])
@pytest.mark.parametrize('continuation', ['held', 'changed', 'silent'])
def test_exhausted_word_yield_ladder_sends_close(arm, continuation):
    """Extend the recorded yield with a held V27 through the walk and eight v53s."""
    m, a = yielded(arm)
    first = m.keyed()
    assert first.kind == Kind.SHORT and first.value == 13
    walk = [(13, 'B'), (49, 'A'), (49, 'B'), (53, 'A')] + [(53, 'B')]*7
    for n in range(1, len(walk)+2):
        at = m.last.stream_sample+round(.93*FS)
        reg = word('V27', at, responds_to=m.last.stream_sample)
        # The receiver marks V27 unexpected while v13 requires X27.
        m.observe(replace(reg, observation=replace(reg.observation, unexpected=True)),
                  now=at+round(.24*FS))
        m.silence(m.last.stream_sample+60000-m.lead)
        proposal = m.decide(m.last.stream_sample+60000-m.lead)
        assert proposal is not None
        short = m.keyed()
        if n <= len(walk):
            assert (short.kind, short.value, short.header, short.stream_sample) == (
                Kind.SHORT, *walk[n-1], first.stream_sample+n*60000)
        assert not m._data_refusals
    close = short
    assert (close.kind, close.status, close.payload, close.stream_sample) == (
        Kind.BODY, 0x20, b'', first.stream_sample+(len(walk)+1)*60000)
    assert m.state == State.ISS_CLOSE
    assert bytes(a._outbuf) == BLOCK and not m.outbox.inflight

    if continuation == 'silent':
        for n in range(1, 4):
            m.silence(m.last.stream_sample+60000-m.lead)
            short = m.keyed()
            assert (short.kind, short.value, short.stream_sample) == (
                Kind.SHORT, 37, close.stream_sample+n*60000)
        m.silence(m.last.stream_sample+60000-m.lead)
        assert m.reason == 'close silence'
    else:
        short = answered(m, 'V27' if continuation == 'held' else 'X27')
        assert (short.kind, short.value, short.stream_sample) == (
            Kind.SHORT, 37, close.stream_sample+60000)
        if continuation == 'held':
            for _ in range(7):
                short = answered(m, 'V27')
                assert short.kind == Kind.SHORT and short.value == 37
            assert answered(m, 'V27') is None
            assert m.reason == 'close refused'
        else:
            assert answered(m, 'V27') is None
            assert m.reason == 'close acknowledged'
    assert m.state == State.DOWN
    assert bytes(a._outbuf) == BLOCK and not m.outbox.inflight


@pytest.mark.parametrize('names', [('Z27', 'Z27'), ('V27', 'Z27'),
                                 ('Z27', 'V27'), ('T27', 'T27'), ('U27', 'U27')])
def test_word_yield_returns_to_ladder_once_and_only_for_two_z27(names):
    m, a = yielded()
    m.keyed()
    answered(m, 'X27')
    answered(m, 'X27')
    chirp = answered(m, 'X27')
    assert chirp.kind == Kind.CHIRP and chirp.status == 0x60
    assert answered(m, names[0]).payload == chirp.payload
    result = answered(m, names[1])
    if names == ('Z27', 'Z27'):
        assert result.kind == Kind.SHORT and result.value == 13
        assert answered(m, 'X27').value == 49
        assert answered(m, 'X27').value == 53
        retry = answered(m, 'X27')
        assert (retry.kind, retry.status, retry.payload) == (chirp.kind, chirp.status, chirp.payload)
        answered(m, 'Z27')
        result = answered(m, 'Z27')
        assert result.status == 0x20 and result.state == State.ISS_CLOSE
        assert bytes(a._outbuf) == BLOCK
    else:
        # Only Z27 refuses a chirp; any other word repeats it.
        assert (result.kind, result.payload, result.repeat) == (Kind.CHIRP, chirp.payload, True)
        assert result.state == State.ISS_DATA and m.outbox.inflight == chirp.payload
    assert m.counter == 0


@pytest.mark.parametrize('arm,last_word,request_at', [('2031', 3392326, 3452326),
                                                ('0103', 2552115, 2612115)])
def test_hold_driver_requests_the_next_slot_after_last_u27(arm, last_word, request_at):
    tx = replay(arm, end=request_at/FS+4)
    sent = next(e for e in tx.p4_probe.emissions if e['audio_start'] > last_word)
    assert abs(sent['audio_start']-request_at) < FS//100
    assert sent['status'] == 0x40 and not sent['payload'] and sent['value'] is None
    assert any(e.get('value') == 13 for e in tx.p4_probe.emissions if e['audio_start'] > request_at)


def test_2031_short_cadence_registers_at_least_thirty_held_v27(monkeypatch):
    # Receive-only continuation isolates cadence from the new request/ladder keys.
    monkeypatch.setattr(p4turn.P4Turn, '_handback_seed', lambda self, reg, now: False)
    tx = replay('2031', end=116, queue_at=1e9)
    heard = {h['onset_sample'] for h in tx.p4_probe.heard
             if h.get('word') == 'V27' and not h['provisional']
             and h['outcome'] not in ('duplicate', 'expired')
             and 71.5*FS < h['onset_sample'] < 114.5*FS}
    assert len(heard) >= 30
    assert tx.sessrx.p4.chirp and tx.sessrx.p4.clock.cycle_s == 1.25


def held_irs():
    m, a = machine(role='irs', ladder='chirp', data='chirp', normal_ack='parity', prekey_lead=.044)
    a.on_host_data(BLOCK)
    # 2031's final v53/U27 pair; the next peer word begins at 71.605 s.
    m._next = p4turn.Proposal(Kind.WORD, 3392326, word='U27', state=State.IRS_ANSWER,
                             peer_control=53, responds_to=3348166)
    m.keyed()
    return m, a, word('V27', round(71.605*FS), responds_to=None)


@pytest.mark.parametrize('reply', ['V27', 'grant'])
def test_irs_hold_request_continues_on_existing_yield_and_grant_paths(reply):
    m, _, reg = held_irs()
    m.observe(reg, now=reg.observation.stream_sample+round(.245*FS))
    request = m.keyed()
    assert (request.kind, request.status, request.payload, request.stream_sample) == (
        Kind.BODY, 0x40, b'', 3452326)
    assert m.state == State.IRS_BREAKIN
    if reply == 'V27':
        assert answered(m, reply).value == 13
    else:
        at = request.stream_sample+round(.93*FS)
        m.observe(word('Y27', at, extent=.8, responds_to=request.stream_sample))
        m.observe(body(at, b'', 0x40))
        first = m.decide(at+round(.920*FS)-m.lead)
        assert (first.kind, first.status, first.payload) == (Kind.BODY, 0, BLOCK[:31])


@pytest.mark.parametrize('invalid', ['provisional', 'weak', 'unexpected', 'expired',
                                   'duplicate', 'off-slot', 'body-target', 'stale-target',
                                   'long-extent', 'no-bytes', 'disabled', 'X27', 'T27', 'Z27'])
def test_irs_hold_requires_confirmed_bare_v27_on_raster_with_queued_bytes(invalid):
    m, a, reg = held_irs()
    if invalid in ('provisional', 'weak', 'unexpected'):
        reg = replace(reg, observation=replace(reg.observation, **{invalid: True}))
    elif invalid in ('expired', 'duplicate'):
        reg = replace(reg, outcome=Registered(invalid))
    elif invalid == 'off-slot':
        reg = replace(reg, observation=replace(reg.observation, stream_sample=reg.observation.stream_sample+FS//5))
    elif invalid == 'long-extent':
        reg = replace(reg, observation=replace(reg.observation, extent_s=.8))
    elif invalid == 'no-bytes':
        a._outbuf.clear()
    elif invalid == 'disabled':
        m.allow_breakin = False
    elif invalid in ('body-target', 'stale-target'):
        target = replace(m.last, kind=Kind.BODY if invalid == 'body-target' else Kind.WORD,
                         stream_sample=m.last.stream_sample-60000)
        m._emissions[target.stream_sample] = target
        reg = replace(reg, responds_to=target.stream_sample)
    else:
        reg = replace(reg, observation=replace(reg.observation, word=invalid))
    m.observe(reg)
    proposal = m.decide(reg.observation.stream_sample)
    if invalid in ('no-bytes', 'disabled'):
        # These are now eligible for the bounded empty hand-back.
        assert proposal is not None and (proposal.kind, proposal.status, proposal.payload) == (
            Kind.BODY, 0x40, b'')
        assert proposal.state == State.ISS_HANDBACK
    else:
        assert proposal is None


def test_irs_hold_accepts_association_with_last_word_and_keeps_request_budget():
    m, _, reg = held_irs()
    reg = replace(reg, responds_to=m.last.stream_sample)
    m.observe(reg)
    request = m.keyed()
    assert request.kind == Kind.BODY and request.status == 0x40
    for _ in range(3):
        m.silence(m.last.stream_sample+60000-m.lead)
        assert m.keyed().state == State.IRS_BREAKIN
    m.silence(m.last.stream_sample+60000-m.lead)
    # R2 §6 bounds this by four actual request keys; silence ends the attempt.
    assert m.reason == 'break-in unanswered' and m.breakins == 4


def test_v53_ack_is_still_u27():
    assert p4turn.PEER_ACK[53] == 'U27'


@pytest.mark.parametrize('case', ['full', 'truncated', 'floor'])
def test_2031_placed_turn_word_requires_a_full_word_at_the_irs_answer_slot(case):
    from hfmodem.shrike import onair
    from hfmodem.shrike.p4receiver import Emission, Expect, P4Receiver, Role
    from hfmodem.tests.shrike.p4_after_handover_helpers import ROOT

    path = require_input(ROOT/'captures/onair-0930-2031/stream.flac')
    audio = onair.session.load_wav(str(path), FS)
    rx = P4Receiver(FS, admitted=True, cfo_hz=2.633705096044118)
    rx.enable_hold()
    at = 3392326 + (30000 if case == 'floor' else 0)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, at, .242))
    end = round((71.70 if case == 'truncated' else 72.48 if case == 'floor' else 71.87)*FS)
    origin = round(70.9*FS)
    heard = rx.read_turn_word(audio[origin:end], origin)
    if case == 'full':
        obs, = [r.observation for r in heard]
        assert obs.word == 'V27' and not obs.provisional and obs.score > .98
        assert abs(obs.stream_sample/FS-71.605) < .003
        assert rx.clock.expects == Expect.WORD
    else:
        assert not heard and rx.clock.expects == Expect.PACKET


def test_stock_chirp_header_opens_long_read_from_short_cadence():
    import numpy as np
    from scipy.signal import hilbert
    from hfmodem.tests.shrike.corpora import n9vc_0802
    from hfmodem.tests.shrike.test_p4_peer_ladder_0928 import receiver

    fs, pcm = n9vc_0802()
    audio = pcm/32768
    audio = (hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/fs)).real
    at = round(6.660*fs)
    rx = receiver(at, cfo=0.)
    rx.chirp = True
    assert rx.clock.cycle_s == 1.25
    origin, end = at-2400, at+round(.845*fs)
    heads = rx.read_partial(audio[origin:end], origin, deadline=end)
    assert any(r.observation.kind == Kind.CHIRP for r in heads)
    assert rx.clock.cycle_s == 3.75
    full = rx.read_window(audio[origin:at+round(3.33*fs)], origin, whole=True)
    packet = next(r.observation for r in full if r.observation.frame is not None)
    assert packet.kind == Kind.CHIRP and packet.frame.status.byte == 0x61
    assert packet.frame.data == b'[BPQ-6.0.25.1-B2FWIHJM'
