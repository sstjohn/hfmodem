# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An unread counted packet slot keys the repeat-request word on grid, never silence.

WS8EOC 40 m, 2026-11-07: the gateway's greeting broke into our opening ladder
as Y27 + body at 17.242 s. Three copies were unread at their decisions and
three reply slots (18.18, 23.17, 24.42 s) went out with no key.
"""
from dataclasses import replace

import pytest

from hfmodem.shrike.p4observation import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, body, machine, word
from hfmodem.tests.shrike.test_p4_irs_silent_1001 import short

GREETING = b'RMS Trimode 1.4.4.0\r\nW9SSJ has '
V38 = 60000
CYCLE = 60000


def opening():
    m, _ = machine(data='body', managed_application=True)
    m.seed_opening([dict(audio_start=0, value=7), dict(audio_start=V38, value=38)], [], values=(7, 38))
    return m


def head(m, at, first=False):
    m.observe(word('Y27', at, extent=.8, responds_to=V38 if first else 'none'), now=at+round(.30*FS))


def decide(m, at):
    decision = at + round(.92*FS) - m.lead
    p = m.decide(decision)
    if p is None:
        m.silence(decision, answer_due=at+round(.92*FS))
        p = m.decide(decision)
    return p


def test_unread_greeting_copies_each_key_v27_at_their_answer_instant():
    m = opening()
    first = V38 + round(.92*FS)
    for k in range(3):
        at = first + k*CYCLE
        head(m, at, first=k == 0)
        p = decide(m, at)
        assert p is not None, f'copy {k} left its reply slot silent'
        assert (p.word, p.stream_sample, p.purpose) == ('V27', at+round(.92*FS), 'repeat-request')
        assert p.ack_identity is None and not p.current_slot_accepted
        m.keyed()
        # The unread body decides the role once read; the opening stands.
        assert m.state == State.ISS_LADDER and m._opening
    assert m.unread == 3 and m.first_body


def test_a_copy_read_after_its_v27_is_acked_x27_on_the_repeat():
    m = opening()
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    assert decide(m, at).word == 'V27'
    m.keyed()
    m.observe(body(at, GREETING), now=at+round(1.0*FS))
    repeat = at + CYCLE
    head(m, repeat)
    m.observe(body(repeat, GREETING), now=repeat+round(.5*FS))
    p = decide(m, repeat)
    assert (p.word, p.ack_reason, p.stream_sample) == ('X27', 'opening body', repeat+round(.92*FS))
    m.keyed()
    assert m.state == State.IRS_ANSWER and not m._opening


def test_a_changeover_request_read_after_its_v27_draws_our_grant():
    # 1002-1018: the opening's Y27 + body was an empty 0x40 request, read a cycle late.
    m = opening()
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    assert decide(m, at).word == 'V27'
    m.keyed()
    repeat = at + CYCLE
    head(m, repeat)
    m.observe(body(repeat, b'', 0x40), now=repeat+round(.5*FS))
    p = decide(m, repeat)
    assert (p.kind, p.status, p.state, p.stream_sample) == (
        Kind.BODY, 0x40, State.ISS_GRANT, repeat+round(.92*FS))


def test_after_a_ladder_answer_with_no_counter_no_word_could_ack_the_unread_body():
    # Our X27 to v13 cleared first_body, so the body's own ack would be V27.
    m, _ = machine(role='irs')
    m.observe(short(FS, 13, 'A'), now=FS+round(.81*FS))
    assert m.decide(FS+round(.92*FS)-m.lead).word == 'X27'
    m.keyed()
    at = FS + CYCLE
    m.observe(word('Y27', at, extent=.8, responds_to='none'), now=at+round(.30*FS))
    p = decide(m, at)
    assert p is None or p.word != 'V27'


@pytest.mark.parametrize('unread_body', [False, True])
@pytest.mark.parametrize('status', [0, 0x61])
def test_an_unread_y27_in_a_long_peer_cycle_keys_nothing_at_the_short_offset(status, unread_body):
    m, _ = machine(role='irs')
    m.require_long_crc = True
    m.observe(ack_frame(FS, b'prior', status, Kind.LONG if status == 0 else Kind.CHIRP), now=206400)
    m.decide(207648)
    m.keyed()
    m.observe(word('Y27', 228000, extent=.8, responds_to='none'), now=242400)
    if unread_body:
        m.observe(RegisteredObservation(replace(body(228000).observation, frame=None), Registered.ENRICHED),
                  now=266880)
    p = m.decide(268608)
    assert p is None or p.stream_sample != 228000 + round(.92*FS)


def test_a_held_v53_answer_is_never_replayed_as_an_unread_chirps_ack():
    # Review 1107: under rotation the held v53 answer was X27, the 0x61 chirp's ack.
    from hfmodem.tests.shrike.test_p4_v53_0929 import control
    m, _ = machine(role='irs', ack_53_rotate=1)
    m.state = State.IRS_ANSWER
    for at in (48000, 108000):
        m.observe(RegisteredObservation(control(at), Registered.NEW), now=at+round(.81*FS))
        m.decide(at+44160-m.lead)
        m.keyed()
    assert m.last.word == 'X27'
    chirp = replace(body(168000).observation, kind=Kind.CHIRP, frame=None)
    m.observe(RegisteredObservation(chirp, Registered.NEW), now=168000+round(3.3*FS))
    p = m.decide(331200-m.lead)
    assert p is None or p.word not in ('X27', 'V27', 'U27')


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
def test_a_later_long_registration_withdraws_the_opening_repeat_request(kind):
    m = opening()
    m.observe(word('Y27', 104880, extent=.8, responds_to=V38), now=104880+round(.30*FS))
    p = m.decide(145488)
    assert (p.word, p.stream_sample) == ('V27', 149040)
    long = replace(body(96000).observation, kind=kind, frame=None, provisional=True)
    m.observe(RegisteredObservation(long, Registered.NEW), now=145489)
    again = m.decide(145489)
    assert again is None or again.stream_sample != 149040
    assert not m.can_key(p)


def test_zero_lead_keys_the_repeat_request_at_its_answer_instant():
    m, _ = machine(data='body', managed_application=True, prekey_lead=0)
    m.seed_opening([dict(audio_start=0, value=7), dict(audio_start=V38, value=38)], [], values=(7, 38))
    at = V38 + round(.92*FS)
    head(m, at, first=True)
    p = m.decide(at+round(.92*FS))
    assert p is not None and (p.word, p.stream_sample) == ('V27', at+round(.92*FS))


def v53_held(**options):
    from hfmodem.tests.shrike.test_p4_v53_0929 import control
    m, _ = machine(role='irs', **options)
    m.state = State.IRS_ANSWER
    for at in ((48000, 108000) if options.get('ack_53_rotate') else (48000,)):
        m.observe(RegisteredObservation(control(at), Registered.NEW), now=at+round(.81*FS))
        m.decide(at+44160-m.lead)
        m.keyed()
    return m


def test_an_unread_chirp_registered_after_the_decision_withdraws_a_held_x27():
    m = v53_held(ack_53_rotate=1)
    m.silence(327648, answer_due=331200)
    p = m.decide(327648)
    assert p is not None and p.word == 'X27'
    chirp = replace(body(168000).observation, kind=Kind.CHIRP, frame=None)
    m.observe(RegisteredObservation(chirp, Registered.NEW), now=327649)
    assert m._next is None and not m.can_key(p)


def test_an_expired_chirp_hypothesis_keeps_the_opening_repeat_request():
    m = opening()
    m.observe(word('Y27', 104880, extent=.8, responds_to=V38), now=119280)
    p = m.decide(145488)
    assert (p.word, p.stream_sample) == ('V27', 149040)
    obs = replace(body(96000).observation, kind=Kind.CHIRP, frame=None, provisional=True)
    m.observe(RegisteredObservation(obs, Registered.EXPIRED, replaced=obs), now=145489)
    assert m._next is p and m.can_key(p)


def test_a_held_u27_still_answers_an_unread_chirp():
    # Chirp acks are X27/V27; U27 after v53 is the 1002 repeat request.
    m = v53_held(ack_53='U27')
    chirp = replace(body(108000).observation, kind=Kind.CHIRP, frame=None)
    m.observe(RegisteredObservation(chirp, Registered.NEW), now=108000+round(3.3*FS))
    p = m.decide(267648)
    if p is None:
        m.silence(267648, answer_due=271200)
        p = m.decide(267648)
    assert p is not None and (p.word, p.stream_sample) == ('U27', 271200)
