# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The K4MSU 1512 and WS8EOC 1509 turns of 2026-09-30: a heard refusal is not silence.

Both arms sent the first data body after the grant, drew X27, sent counter 1
as a normal frame, drew X27 then Z27 twice, closed with 0x20 and short v37,
drew Z27 to every copy, and ended "close silence" while the gateway was still
keying Z27 on its raster (captures/onair-0930-1512/stream.flac 41.591-52.191 s,
captures/onair-0930-1510/witness.wav 41.54-62.16 s).
"""

import json
from dataclasses import replace
import numpy as np
import pytest

from hfmodem.tests.evidence import RECORD as REPO, require_input
from hfmodem.shrike.p4receiver import Kind
from hfmodem.shrike.p4observation import Registered
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, body, machine, word

GRANT = 1996368          # K4MSU's Y27 + idle 0x40 body at 41.591 s
BLOCK = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 16377127\r; K4MSU DE W9SSJ (EN63bc)\rFC EM IFBGN8NOCJZA 619 464 0\rF> 67\r'


def granted(ladder):
    m, a = machine(role="irs", ladder=ladder,
                   data="chirp" if ladder == "chirp" else "body", normal_ack="parity")
    a.on_host_data(BLOCK)
    m._body(GRANT - round(.93 * FS), 0x40, State.IRS_BREAKIN)
    request = m.keyed()
    m.observe(word("Y27", GRANT, extent=0.8, responds_to=request.stream_sample))
    m.observe(body(GRANT, b"", 0x40))
    c0 = m.decide(GRANT + round(.846 * FS))
    assert c0.kind == Kind.BODY and c0.status == 0 and c0.payload.startswith(b';FW: W9SSJ')
    return m, m.keyed()


def answered(m, name):
    at = m.last.stream_sample + round((3.4 if m.last.kind == Kind.CHIRP else .93) * FS)
    m.observe(word(name, at, responds_to=m.last.stream_sample), now=at + round(.24 * FS))
    decision = at + round(.256 * FS)
    if m.decide(decision) is None:
        m.silence(decision)
        if m.decide(decision) is None:
            return None
    return m.keyed()


def test_k4msu_1512_refused_close_is_not_silence():
    m, c0 = granted("none")
    c1 = answered(m, "X27")
    assert c1.kind == Kind.NORMAL and c1.status == 1 and c1.payload.startswith(b'PR: 16377127')
    assert answered(m, "X27").payload == c1.payload
    assert answered(m, "Z27").payload == c1.payload
    close = answered(m, "Z27")
    assert close.kind == Kind.BODY and close.status == 0x20 and m.state == State.ISS_CLOSE
    shorts = 0
    while (p := answered(m, "Z27")) is not None:
        assert p.kind == Kind.SHORT and p.value == 37
        shorts += 1
    assert shorts == 8 and m.reason == "close refused"
    assert m.outbox.queued and bytes(m.outbox.arq._outbuf).startswith(b'PR: 16377127')


def test_unanswered_close_is_still_silence_after_three_shorts():
    m, a = machine(app_done=True)
    m._data(0)
    m.keyed()
    for _ in range(3):
        m.silence(FS)
        assert m.keyed().value == 37
    m.silence(FS)
    assert m.state == State.DOWN and m.reason == "close silence"


def test_v27_change_to_closing_body_allows_v27_to_v37():
    m, _ = granted('none')
    answered(m, 'X27')
    answered(m, 'Z27')
    close = answered(m, 'Z27')
    assert close.status == 0x20
    assert answered(m, 'V27').value == 37
    assert answered(m, 'V27') is None
    assert m.reason == 'close acknowledged'


@pytest.mark.parametrize('body_answer', [None, 'V27'])
def test_odd_counter_ack_does_not_hold_v27_through_close(body_answer):
    """k0003: 0x20 at 1539.207, v37 at 1540.455, V27 at 1541.383.

    Precede that close with an acknowledged odd-counter packet; also cover
    a peer that answers 0x20 itself with V27.
    """
    m, a = machine(ladder='none', data='body', app_done=True, normal_ack='parity')
    a.on_host_data(b'A'*32)
    m._begin_data_turn(round(1536.707*FS))
    m.keyed()
    c1 = answered(m, 'X27')
    assert c1.kind == Kind.NORMAL and c1.status == 1
    close = answered(m, 'V27')
    assert close.status == 0x20 and close.stream_sample == round(1539.207*FS)
    assert not m.outbox.queued and not m.outbox.inflight
    if body_answer:
        m.observe(word(body_answer, close.stream_sample+round(.928*FS),
                       responds_to=close.stream_sample))
    m.silence(round(1540.455*FS)-m.lead)
    short = m.keyed(at=round(1540.455*FS))
    assert short.kind == Kind.SHORT and short.value == 37
    m.observe(word('V27', round(1541.383*FS), responds_to=short.stream_sample))
    assert m.state == State.DOWN and m.reason == 'close acknowledged'


@pytest.mark.parametrize('name', ['X27', 'Z27', 'U27', 'T27'])
def test_unassociated_non_v27_close_words_are_refusals(name):
    """The 1512/1509 refusal stream still counts if association is missing."""
    m, _ = machine(app_done=True)
    m._data(0)
    m.keyed()
    shorts = 0
    for _ in range(10):
        at = m.last.stream_sample+round(.935*FS)
        m.observe(word(name, at, responds_to=None))
        m.silence(at+round(.256*FS))
        if m.state != State.DOWN:
            assert m.keyed().value == 37
            shorts += 1
        else:
            break
    assert m.state == State.DOWN and shorts == 8 and m.reason == 'close refused'


def test_changed_stream_can_confirm_the_previous_v37_after_a_repeat_keyed():
    m, h, _ = held_close_2028()
    m.observe(word('X27', h['onset_sample'], responds_to=m.last.stream_sample))
    m.silence(h['onset_sample']+round(.256*FS))
    first = m.keyed()
    m.silence(first.stream_sample+60000-m.lead)
    assert m.keyed().value == 37
    m.observe(word('V27', first.stream_sample+round(.935*FS), responds_to=first.stream_sample))
    assert m.reason == 'close acknowledged'


def test_two_refusals_return_to_the_ladder_only_once_per_data_turn():
    m, c0 = granted("chirp")
    first = answered(m, "X27")
    assert first.kind == Kind.SHORT and first.value == 13
    assert answered(m, "X27").value == 49
    assert answered(m, "X27").value == 53
    answered(m, "X27")
    chirp = m.last
    assert chirp.kind == Kind.CHIRP and chirp.status == 0x61 and len(chirp.payload) == 22
    assert answered(m, "Z27").kind == Kind.CHIRP
    again = answered(m, "Z27")
    assert again.kind == Kind.SHORT and again.value == 13 and m.state == State.ISS_LADDER
    assert m.outbox.queued and bytes(m.outbox.arq._outbuf).startswith(b'PR: 16377127')
    assert answered(m, "X27").value == 49
    assert answered(m, "X27").value == 53
    retry = answered(m, "X27")
    assert (retry.kind, retry.status, retry.payload) == (Kind.CHIRP, 0x61, chirp.payload)
    assert answered(m, "Z27").kind == Kind.CHIRP
    close = answered(m, "Z27")
    assert (close.kind, close.status, m.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert m.reason == "data refused twice by Z27"
    assert m.counter == 1 and not m.outbox.inflight
    assert bytes(m.outbox.arq._outbuf) == BLOCK[len(c0.payload):]


@pytest.mark.parametrize("name", ["V27", "T27", "U27"])
def test_word_yield_exhaustion_closes_after_the_peer_changes_word(name):
    """Ordinary exhaustion retains the existing changed-word close proof."""
    m, a = machine(role="irs", ladder="none", data="body", normal_ack="parity")
    a.on_host_data(BLOCK)
    m._body(0, 0x40, State.IRS_BREAKIN)
    m.keyed()
    c0 = answered(m, "V27")
    assert (c0.kind, c0.status) == (Kind.BODY, 0x40)
    assert answered(m, name).payload == c0.payload
    for _ in range(6):
        retry = answered(m, name)
        assert (retry.kind, retry.status, retry.payload) == (c0.kind, c0.status, c0.payload)
    close = answered(m, name)
    assert (close.kind, close.status, m.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert m.reason == "retry budget exhausted"
    short = answered(m, "X27")
    assert short.kind == Kind.SHORT and short.value == 37
    assert answered(m, "V27") is None
    assert m.state == State.DOWN and m.reason == "close acknowledged"
    assert m.counter == 0 and bytes(a._outbuf) == BLOCK

def test_ws8eoc_1509_witness_names_the_word_that_answered_our_first_body():
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from hfmodem.shrike import p4oracle
    cap = REPO / 'captures/onair-0930-1510'
    require_input(cap / 'witness.wav')
    rate = json.loads((cap / 'witness.json').read_text())['sample_rate']
    t0, t1 = 43.0, 50.2
    _, pcm = wavfile.read(cap / 'witness.wav', mmap=True)
    x = np.asarray(pcm[int(t0 * rate):int(t1 * rate)], np.float64) / 32768
    x, fs = resample_poly(x, 4, 1), round(4 * rate)
    heard = [p4oracle.read_burst(x, t - t0, 0.28, fs) for t in (43.38, 44.64, 45.90, 47.14, 48.40, 49.64)]
    assert [r['word'] for r in heard] == ['W27'] * 6
    assert min(r['word_score'] for r in heard) > 0.95


def held_close_2028():
    """Keep the recorded body transcript under the no-ladder policy.

    The peer held V27 on every key. The tape's third V27 drew our close; under
    word-yield DESIGN-V2 Unit A (working/p4-plan-0930/p4-word-yield-recovery-1004)
    it draws a third data copy instead. From there the same held V27 is modeled
    at the recorded lag on each key, through ordinary exhaustion to close and v37.
    """
    probe = json.loads(require_input(REPO / 'captures/onair-0930-2028/p4-probe.json').read_text())
    heard = [h for h in probe['heard'] if h['word'] == 'V27'
             and h['outcome'] == 'confirmed' and not h['provisional']]
    assert len(heard) == 5
    m, a = machine(role='irs', ladder='none', data='body', normal_ack='parity')
    payload = bytes.fromhex(next(e['payload'] for e in probe['emissions'] if e.get('payload')))
    a.on_host_data(payload)
    m._body(heard[0]['responds_to'], 0x40, State.IRS_BREAKIN)
    m.keyed()
    recorded = iter(heard[:3])
    lag = heard[-1]['onset_sample'] - heard[-1]['responds_to']
    wait = heard[-1]['received_at_sample'] - heard[-1]['onset_sample']

    def held():
        if (h := next(recorded, None)) is not None:
            assert m.last.stream_sample == h['responds_to']
            return h
        key = m.last.stream_sample
        return dict(onset_sample=key+lag, responds_to=key, received_at_sample=key+lag+wait)

    copies = 0
    while not (m.state == State.ISS_CLOSE and m.last.value == 37):
        h = held()
        at = h['onset_sample']
        m.observe(word('V27', at, responds_to=h['responds_to']), now=at+round(.24*FS))
        now = at+round(.256*FS)
        if m.decide(now) is None:
            m.silence(now)
        p = m.keyed()
        copies += p.payload == payload
        assert copies <= 8
    assert copies == 8
    assert bytes(a._outbuf) == payload
    return m, held(), probe


@pytest.mark.parametrize('late', [False, True])
def test_2028_held_v27_does_not_acknowledge_close(late):
    m, h, probe = held_close_2028()
    if late:
        # The flight had already keyed its second v37, one cycle on, before confirmation.
        cycle = probe['emissions'][-1]['audio_start'] - probe['emissions'][-2]['audio_start']
        m.silence(m.last.stream_sample+cycle-m.lead)
        assert m.keyed().value == 37
    m.observe(word('V27', h['onset_sample'], responds_to=h['responds_to']),
              now=h['received_at_sample'])
    assert m.state == State.ISS_CLOSE and m.reason != 'close acknowledged'
    assert m._close_refusals


@pytest.mark.parametrize('invalid', [None, 'provisional', 'weak', 'unexpected',
                                   'expired', 'duplicate', 'unassociated', 'off-slot', 'stale'])
def test_close_needs_a_demonstrable_change_in_the_held_word(invalid):
    m, h, _ = held_close_2028()
    at = h['onset_sample']
    change = word('X27', at, responds_to=m.last.stream_sample)
    if invalid in ('provisional', 'weak', 'unexpected'):
        change = replace(change, observation=replace(change.observation, **{invalid: True}))
    elif invalid in ('expired', 'duplicate'):
        change = replace(change, outcome=Registered(invalid))
    elif invalid == 'unassociated':
        change = replace(change, responds_to=None)
    elif invalid == 'off-slot':
        change = replace(change, observation=replace(change.observation, stream_sample=at+FS//5))
    elif invalid == 'stale':
        change = word('X27', at-3*60000, responds_to=m.last.stream_sample-3*60000)
    m.observe(change)
    m.silence(at+round(.256*FS))
    p = m.keyed()
    assert p.value == 37
    m.observe(word('V27', p.stream_sample+round(.935*FS), responds_to=p.stream_sample))
    if invalid is None:
        assert m.state == State.DOWN and m.reason == 'close acknowledged'
    else:
        assert m.state == State.ISS_CLOSE


@pytest.mark.parametrize('invalid', ['provisional', 'weak', 'unexpected', 'expired',
                                   'duplicate', 'unassociated', 'off-slot', 'wrong-short'])
def test_changed_word_still_needs_a_valid_answer_to_v37(invalid):
    m, h, _ = held_close_2028()
    m.observe(word('X27', h['onset_sample'], responds_to=m.last.stream_sample))
    m.silence(h['onset_sample']+round(.256*FS))
    p = m.keyed()
    reg = word('V27', p.stream_sample+round(.935*FS), responds_to=p.stream_sample)
    if invalid in ('provisional', 'weak', 'unexpected'):
        reg = replace(reg, observation=replace(reg.observation, **{invalid: True}))
    elif invalid in ('expired', 'duplicate'):
        reg = replace(reg, outcome=Registered(invalid))
    elif invalid == 'unassociated':
        reg = replace(reg, responds_to=None)
    elif invalid == 'off-slot':
        reg = replace(reg, observation=replace(reg.observation, stream_sample=p.stream_sample+FS//2))
    else:
        m._emissions[p.stream_sample] = replace(p, value=13)
    m.observe(reg)
    assert m.state == State.ISS_CLOSE


def test_2028_witness_keeps_v27_then_retakes_the_turn_after_our_last_key():
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from hfmodem.shrike import p4oracle
    cap = REPO / 'captures/onair-0930-2028'
    require_input(cap/'witness.wav')
    rate = json.loads(require_input(cap/'witness.json').read_text())['sample_rate']
    _, pcm = wavfile.read(cap/'witness.wav', mmap=True)
    origin = 37
    x = resample_poly(np.asarray(pcm[int(origin*rate):], np.float64)/32768, 4, 1)
    times = (37.975, 39.225, 40.475, 41.725, 42.975, 45.48, 46.73, 47.98)
    heard = [p4oracle.read_burst(x, t-origin, .28, round(4*rate)) for t in times]
    assert [r['word'] for r in heard] == ['V27']*5 + ['Y27']*3
    assert min(r['word_score'] for r in heard) > .95
    probe = json.loads((cap/'p4-probe.json').read_text())
    # Greeting and our X27/V27 keys independently place witness = stream + 3.88 s.
    last_key = probe['emissions'][-1]['audio_start']/FS + 3.88
    assert 2 < times[-3]-last_key < 2.3
    assert 4.5 < times[-1]-last_key < 4.8
