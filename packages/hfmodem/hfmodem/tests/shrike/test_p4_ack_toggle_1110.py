# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Replay CRC-valid probe registrations through packet admission and physical keys."""

from collections import Counter
from dataclasses import replace
import json

import pytest

from hfmodem.shrike import p4frame
from hfmodem.shrike.p4receiver import Kind, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.evidence import OPERATOR_DATA, RECORD, require_input
from hfmodem.tests.shrike.test_p4turn import FS, ack_frame, body, machine, propose_ack, word


ARMS = [
    ('captures/onair-1002-2150', None,
     {('long', 0x61): 'X27', ('long', 0x62): 'V27'}),
    ('operator-data/captures/onair-20261010T233501182327Z-54a54140-3f95-46d2-856b-daed7e6f24f7', None,
     {('long', 0x62): 'X27', ('long', 0x03): 'V27'}),
    ('captures/onair-0930-2206', None,
     {('chirp', 0x61): 'X27', ('chirp', 0x62): 'V27'}),
    ('captures/onair-0929-2243', None,
     {('long', 0x61): 'X27', ('long', 0x62): 'V27'}),
    ('captures/onair-0929-0939', None,
     {('body', 0): 'X27', ('normal', 1): 'V27', ('normal', 2): 'X27'}),
    ('operator-data/captures/onair-20261027T064359603967Z-732431ae-9d5f-4908-b964-efb24ed92a98', 289*FS,
     {('body', 0): 'X27', ('normal', 0x61): 'V27', ('long', 0x63): 'X27',
      ('long', 0x60): 'V27', ('chirp', 0x60): 'V27', ('chirp', 0x61): 'X27',
      ('chirp', 0xe2): 'V27', ('chirp', 0x63): 'X27',
      ('chirp', 0x00): 'V27', ('chirp', 0x41): 'X27', ('chirp', 0x42): 'V27'}),
]


def registrations(path, end):
    stem = path.removeprefix('operator-data/')
    probe = OPERATOR_DATA / stem / 'p4-probe.json'
    if stem == path or not probe.is_file():
        probe = RECORD / stem / 'p4-probe.json'
    require_input(probe)
    copies = {}
    for row in json.loads(probe.read_text())['heard']:
        counted = (row['kind'] in ('body', 'normal', 'long', 'chirp')
                   and row.get('status') is not None and row.get('data') is not None)
        control = row['kind'] == 'short' and row.get('value') is not None
        if (counted or control) and (end is None or row['onset_sample'] < end):
            copies[round(row['onset_sample']/FS, 1)] = row
    return sorted(copies.values(), key=lambda row: row['onset_sample'])


def registered(row):
    kind, at = Kind(row['kind']), row['onset_sample']
    obs = replace(body(at).observation, kind=kind, frame=None,
                  header=row.get('header') or 'A', value=row.get('value'),
                  speedlevel=row.get('speedlevel'), sf8=row.get('sf8', False))
    if row.get('status') is not None:
        data = bytes.fromhex(row['data'])
        frame = p4frame.decode_field(p4frame.encode_field(
            data, p4frame.Status.from_byte(row['status'])), data_bytes=len(data))
        assert frame is not None and frame.data == data
        obs = replace(obs, frame=frame)
    return RegisteredObservation(obs, Registered.NEW)


@pytest.mark.parametrize('path,end,expected', ARMS, ids=[r[0].split('/')[-1] for r in ARMS])
def test_recorded_packets(path, end, expected):
    turn, _ = machine(role='irs')
    found = Counter()
    for row in registrations(path, end):
        reply = propose_ack(turn, registered(row))
        assert reply is not None, row
        key = row['kind'], row.get('status')
        if key in expected:
            assert reply.word == expected[key], row
            found[key] += 1
        turn.keyed()
    assert set(found) == set(expected), f'Listed packets absent from probe: {set(expected)-set(found)}'
    if '233501' in path:
        from hfmodem.tests.shrike.test_p4_late_long_ack_1010 import DATA_62, DATA_03
        fields = {r['status']: bytes.fromhex(r['data']) for r in registrations(path, end)
                  if r['kind'] == 'long'}
        assert fields == {0x62: DATA_62, 0x03: DATA_03}


def short(at, value=13):
    obs = replace(body(at).observation, kind=Kind.SHORT, frame=None, value=value, header='A')
    return RegisteredObservation(obs, Registered.NEW)


def key_packet(turn, at, status, kind=Kind.NORMAL, **tags):
    reg = ack_frame(at, b'packet', status, kind)
    reg = replace(reg, observation=replace(reg.observation, **tags))
    answer = propose_ack(turn, reg)
    assert answer is not None
    turn.keyed()
    return answer


@pytest.mark.parametrize('first,control,expected', [('x27', 'X27', 'V27'), ('v27', 'V27', 'X27')])
def test_unread_after_control_requests_repeat(first, control, expected):
    turn, _ = machine(role='irs', first_ack=first, peer_ack={13: control})
    assert propose_ack(turn, short(FS)).word == control
    turn.keyed()
    at = 300000
    unread = replace(body(at).observation, kind=Kind.LONG, frame=None)
    turn.observe(RegisteredObservation(unread, Registered.NEW))
    reply = turn.decide(at+round(3.4*FS)-turn.lead)
    assert reply.word == expected and reply.ack_identity is None
    turn.keyed()
    assert turn._ack_chain is None


def test_unread_mid_chain_repeats_word():
    turn, _ = machine(role='irs')
    key_packet(turn, FS, 0)
    key_packet(turn, 300000, 1)
    at = 480000
    unread = replace(body(at).observation, kind=Kind.NORMAL, frame=None)
    turn.observe(RegisteredObservation(unread, Registered.NEW))
    reply = turn.decide(at+round(.92*FS)-turn.lead)
    assert reply.word == 'V27'
    turn.keyed()
    assert turn._ack_chain == (1, 'V27')


@pytest.mark.parametrize('kind,tags,repeat', [
    (Kind.BODY, {}, 'V27'),
    (Kind.LONG, {}, 'V27'),
    (Kind.NORMAL, {}, 'V27'),
    (Kind.CHIRP, {}, 'V27'),
    (Kind.LONG, {'speedlevel': 'SL5-long'}, 'T27'),
])
@pytest.mark.parametrize('control,control_answer', [
    (7, 'X27'), (13, 'X27'), (38, 'X27'), (49, 'X27'), (59, 'X27'), (59, 'W27'),
])
def test_unread_after_mid_chain_control_keeps_repeat_word(kind, tags, repeat, control, control_answer):
    turn, _ = machine(role='irs', peer_ack={59: control_answer})
    assert key_packet(turn, FS, 1, kind, **tags).word == 'X27'
    assert key_packet(turn, 200000, 2, kind, **tags).word == repeat
    reg = short(turn.last.stream_sample + round(.330*FS), control)
    reg = replace(reg, observation=replace(reg.observation, placed=True),
                  responds_to=turn.last.stream_sample)
    assert propose_ack(turn, reg).word == control_answer
    turn.keyed()
    at = 480000
    unread = replace(body(at).observation, kind=kind, frame=None, **tags)
    due = at + round(turn.t[kind]*FS)
    turn.observe(RegisteredObservation(unread, Registered.NEW), now=due-turn.lead)
    answer = turn.decide(due-turn.lead)
    assert answer.word == repeat and not answer.current_slot_accepted
    assert answer.purpose == 'repeat-request' and answer.peer_control is None
    turn.keyed()
    assert turn._ack_chain == (2, repeat)
    assert key_packet(turn, 700000, 3, kind, **tags).word == 'X27'
    assert turn._ack_chain == (3, 'X27')


@pytest.mark.parametrize('control', [7, 13, 38, 49, 59])
def test_unread_control_repeat_keeps_held_control_answer(control):
    turn, _ = machine(role='irs')
    assert key_packet(turn, FS, 1).word == 'X27'
    assert key_packet(turn, 200000, 2).word == 'V27'
    accepted = propose_ack(turn, short(350000, control))
    assert accepted.word == 'X27' and accepted.peer_control == control
    turn.keyed()
    held = turn._held_answer
    due = turn.last.stream_sample + round(1.25*FS)
    turn.silence(due-turn.lead, answer_due=due)
    answer = turn.decide(due-turn.lead)
    assert answer.word == accepted.word and answer.peer_control == control
    assert answer.purpose is None and answer.ack_identity is None
    assert not answer.current_slot_accepted
    turn.keyed()
    assert turn._ack_chain == (2, 'V27') and turn._held_answer == held


@pytest.mark.parametrize('held_state', ['none', 'epoch', 'generation'])
@pytest.mark.parametrize('tags,partner', [
    ({'speedlevel': 'SL5-long'}, 'T27'),
    ({'form': 'normal-long/SL7'}, 'T27'),
    ({}, 'V27'),
])
@pytest.mark.parametrize('first', ['x27', 'v27'])
def test_unread_chain_start_uses_form_partner(held_state, tags, partner, first):
    turn, _ = machine(role='irs', first_ack=first)
    if held_state != 'none':
        assert propose_ack(turn, short(FS)).word == 'X27'
        turn.keyed()
        if held_state == 'epoch':
            turn._receive_epoch += 1
        else:
            turn._peer_generation += 1
    at = 480000
    unread = replace(body(at).observation, kind=Kind.LONG, frame=None, **tags)
    due = at + round(turn.t[Kind.LONG]*FS)
    turn.observe(RegisteredObservation(unread, Registered.NEW), now=due-turn.lead)
    answer = turn.decide(due-turn.lead)
    assert answer.word == (partner if first == 'x27' else 'X27')
    assert answer.ack_identity is None and not answer.current_slot_accepted
    turn.keyed()
    assert turn._ack_chain is None


@pytest.mark.parametrize('policy,form,partner', [
    ('toggle-t27', 'robust', 'T27'),
    ('toggle-t27', 'normal long', 'T27'),
    ('toggle', 'robust', 'V27'),
    ('toggle', 'normal long', 'T27'),
])
@pytest.mark.parametrize('first', ['x27', 'v27'])
@pytest.mark.parametrize('held_control', [False, True])
def test_unread_silence_chain_start_uses_long_policy(policy, form, partner, first, held_control):
    turn, _ = machine(role='irs', long_ack=policy, first_ack=first,
                      peer_ack={13: first.upper()})
    if held_control:
        assert propose_ack(turn, short(FS)).word == first.upper()
        turn.keyed()
    due = 480000
    turn._ack_form = form
    turn._unread_data_due = due, Kind.LONG, form, partner if first == 'x27' else 'X27'
    turn.silence(due-turn.lead, answer_due=due)
    answer = turn.decide(due-turn.lead)
    assert answer.word == (partner if first == 'x27' else 'X27')
    assert answer.purpose == 'repeat-request' and answer.ack_identity is None
    assert turn.can_key(answer)
    assert turn.keyed().word == answer.word
    assert turn._ack_chain is None


def test_refused_key_records_nothing():
    turn, _ = machine(role='irs')
    assert propose_ack(turn, ack_frame(FS, b'unkeyed', 1)).word == 'X27'
    turn.refused('deadline')
    assert turn._ack_chain is None
    assert key_packet(turn, 300000, 2).word == 'X27'


CHAIN_RESETS = ('turn', 'form', 'ladder', 'turn,form', 'turn,ladder',
                'form,ladder', 'turn,form,ladder')


@pytest.mark.parametrize('policy', CHAIN_RESETS)
@pytest.mark.parametrize('boundary,kind,tags', [
    ('turn', Kind.NORMAL, {}), ('ladder', Kind.NORMAL, {}),
    ('form', Kind.LONG, {'speedlevel': '3'}), ('form', Kind.LONG, {'sf8': True}),
    ('form', Kind.CHIRP, {}), ('form', Kind.LONG, {'speedlevel': 'SL5-long'}),
])
@pytest.mark.parametrize('counter', [0, 1], ids=['reack', 'distinct'])
def test_chain_reset_boundaries(policy, boundary, kind, tags, counter):
    turn, _ = machine(role='irs', chain_reset=policy)
    assert key_packet(turn, FS, 0, Kind.BODY).word == 'X27'
    if boundary == 'turn':
        turn._close_receive_epoch('peer turn')
        turn._open_receive_epoch(250000)
    elif boundary == 'ladder':
        assert propose_ack(turn, short(250000)).word == 'X27'
        turn.keyed()
    partner = 'T27' if tags.get('speedlevel') == 'SL5-long' else 'V27'
    expected = 'X27' if (counter == 0 and boundary != 'turn'
                         or boundary in policy.split(',')) else partner
    assert key_packet(turn, 480000, counter, kind, **tags).word == expected


@pytest.mark.parametrize('policy', CHAIN_RESETS)
@pytest.mark.parametrize('kind', [Kind.CHIRP, Kind.LONG])
def test_same_form_carries_across_robust_shorts(policy, kind):
    turn, _ = machine(role='irs', chain_reset=policy)
    assert key_packet(turn, FS, 3, kind).word == 'X27'
    for i, value in enumerate((36, 7, 38)):
        answer = propose_ack(turn, short(300000+i*60000, value))
        assert answer is not None
        turn.keyed()
    expected = 'X27' if 'ladder' in policy.split(',') else 'V27'
    assert key_packet(turn, 660000, 0, kind).word == expected


@pytest.mark.parametrize('policy', CHAIN_RESETS)
def test_same_counter_step_down_keeps_word(policy):
    turn, _ = machine(role='irs', chain_reset=policy)
    assert key_packet(turn, FS, 3, Kind.LONG, speedlevel='4').word == 'X27'
    assert key_packet(turn, 300000, 0, Kind.LONG, speedlevel='3').word == 'V27'
    assert key_packet(turn, 480000, 0, Kind.LONG, speedlevel='2').word == 'V27'
    assert key_packet(turn, 660000, 0, Kind.CHIRP).word == 'V27'
    # Same bytes with a changed counter are a new packet.
    assert key_packet(turn, 840000, 1, Kind.CHIRP).word == 'X27'


@pytest.mark.parametrize('policy,tag,partner', [
    ('toggle', {}, 'V27'), ('toggle', {'speedlevel': '2'}, 'V27'),
    ('toggle', {'speedlevel': '3'}, 'V27'), ('toggle', {'sf8': True}, 'V27'),
    ('toggle', {'speedlevel': 'SL5-long'}, 'T27'),
    ('toggle', {'speedlevel': 'SL8-long'}, 'T27'), ('toggle', {'speedlevel': 'SL10-long'}, 'T27'),
    ('toggle', {'speedlevel': 'SL4-long'}, 'V27'), ('toggle', {'form': 'normal-long'}, 'T27'),
    ('toggle-t27', {'speedlevel': '3'}, 'T27'),
])
def test_long_partners_and_cross_level_reack(policy, tag, partner):
    turn, _ = machine(role='irs', long_ack=policy)
    assert key_packet(turn, FS, 1, Kind.LONG, **tag).word == 'X27'
    assert key_packet(turn, 300000, 2, Kind.LONG, **tag).word == partner
    assert key_packet(turn, 480000, 2, Kind.CHIRP).word == partner
    assert key_packet(turn, 660000, 3).word == 'X27'


@pytest.mark.parametrize('kind', [Kind.BODY, Kind.NORMAL, Kind.CHIRP])
def test_iss_commits_and_data_turn_reset(kind):
    turn, arq = machine(data='body')
    arq.on_host_data(b'ours'*40)
    turn._begin_data_turn(0)
    for i, expected in enumerate(('X27', 'V27', 'X27')):
        turn._data(i*180000, State.ISS_DATA_ACK, kind)
        sent = turn.keyed()
        assert turn._expected_data_ack() == expected
        wrong = 'V27' if expected == 'X27' else 'X27'
        assert not turn._data_ack(word(wrong, sent.stream_sample+44160).observation)
        turn.observe(word(expected, sent.stream_sample+44160, responds_to=sent.stream_sample))
        assert turn._last_data_ack == expected
    arq.on_host_data(b'next turn')
    turn._begin_data_turn(900000)
    turn.keyed()
    assert turn._expected_data_ack() == 'X27'
    turn._last_data_ack = 'X27'
    turn._last_data_form = 'normal short'
    turn._after_ladder(1000000)
    turn.keyed()
    assert turn._expected_data_ack() == 'V27'


def test_first_v27_is_irs_only():
    turn, _ = machine(role='irs', first_ack='v27')
    assert key_packet(turn, FS, 0, Kind.BODY).word == 'V27'
    assert key_packet(turn, 300000, 1).word == 'X27'
    assert key_packet(turn, 480000, 2).word == 'V27'
    iss, _ = machine(first_ack='v27')
    iss._idle(0)
    iss.keyed()
    assert iss._expected_data_ack(idle=True) == 'X27'
    iss._last_data_ack = 'X27'
    iss._last_data_form = 'normal short'
    assert iss._expected_data_ack(idle=True) == 'V27'
    iss._last_data_ack = 'V27'
    assert iss._expected_data_ack(idle=True) == 'X27'


@pytest.mark.parametrize('first', ['x27', 'v27'])
def test_iss_form_change_starts_x27(first):
    turn, arq = machine(data='body', first_ack=first)
    arq.on_host_data(b'ours'*100)
    turn._begin_data_turn(0)
    for i, kind in enumerate((Kind.BODY, Kind.NORMAL, Kind.CHIRP, Kind.NORMAL)):
        turn._data(i*180000, State.ISS_DATA_ACK, kind)
        sent = turn.keyed()
        expected = 'V27' if i == 1 else 'X27'
        assert turn._expected_data_ack() == expected
        for rejected in ({'X27', 'V27', 'T27', 'U27'} - {expected}):
            assert not turn._data_ack(word(rejected, sent.stream_sample+44160).observation)
        turn.observe(word(expected, sent.stream_sample+44160, responds_to=sent.stream_sample))
        assert turn._last_data_ack == expected


@pytest.mark.parametrize('last,expected', [(None, 'X27'), ('X27', 'V27'), ('V27', 'X27')])
def test_iss_hold_counted_idle_and_handback_seed(last, expected):
    turn, _ = machine(data='body', first_ack='v27')
    turn._last_data_ack = last
    turn._last_data_form = 'normal short'
    turn._idle(FS)
    idle = turn.keyed()
    at = idle.stream_sample+round(.935*FS)
    # Unassociated and weak words do not commit the idle or seed a handback.
    turn.observe(word(expected, at-400, responds_to=None))
    assert turn._acked_idle is None
    weak = word(expected, at-200, responds_to=idle.stream_sample)
    turn.observe(replace(weak, observation=replace(weak.observation, weak=True)))
    assert turn._acked_idle is None
    if expected == 'X27':
        turn.observe(word('V27', at, responds_to=idle.stream_sample))
        offer = turn.decide(at+round(.30*FS))
        assert (offer.kind, offer.state) == (Kind.BODY, State.ISS_HANDBACK)
        assert turn._acked_idle is None and turn._last_data_ack == last
        turn.keyed()
        turn.observe(word('X27', offer.stream_sample+round(.935*FS),
                          responds_to=offer.stream_sample))
        assert turn._last_data_ack == 'X27' and turn.counter == 1
    else:
        turn.observe(word('V27', at, responds_to=idle.stream_sample))
        assert turn._acked_idle == idle.stream_sample
        assert turn._last_data_ack == 'V27' and turn.counter == 1
        assert not any(e['event'] == 'handback-offer' for e in turn.link_events)


LEGACY_WORDS = [
    ('normal_ack', 'parity', Kind.NORMAL, ('X27', 'V27', 'X27', 'V27')),
    ('long_ack', 'counter-even-t27', Kind.LONG, ('T27', 'X27', 'T27', 'X27')),
    ('chirp_ack', 'parity', Kind.CHIRP, ('V27', 'X27', 'V27', 'X27')),
]


@pytest.mark.parametrize('name,policy,kind,expected', LEGACY_WORDS)
@pytest.mark.parametrize('first', ['x27', 'v27'])
@pytest.mark.parametrize('reset', CHAIN_RESETS)
def test_off_default_head_word_equivalence(name, policy, kind, expected, first, reset):
    # HEAD pins these words in p4turn, long_ack_1002 and v27_handback_1002.
    turn, _ = machine(role='irs', first_ack=first, chain_reset=reset, **{name: policy})
    for i, want in enumerate(expected):
        at = FS+i*600000
        assert key_packet(turn, at, i, kind).word == want
        assert key_packet(turn, at+180000, i, kind).word == want
        turn._close_receive_epoch('peer turn')
        turn._open_receive_epoch(at+400000)
        assert propose_ack(turn, short(at+420000)).word == 'X27'
        turn.keyed()


@pytest.mark.parametrize('policy,expected', [('turn,form', ('V27', 'X27', 'V27')),
                                            ('turn,form,ladder', ('X27', 'V27', 'X27'))])
def test_k4spb_late_chirp_carries_after_probes(policy, expected):
    path = ARMS[-1][0]
    rows = registrations(path, 289*FS)
    turn, _ = machine(role='irs', chain_reset=policy)
    late = {}
    controls = set()
    for row in rows:
        answer = propose_ack(turn, registered(row))
        assert answer is not None
        if 187*FS < row['onset_sample'] < 238*FS and row['kind'] == 'short':
            controls.add(row['value'])
        for target, status, want in zip((272.26, 283.52, 287.27), (0x00, 0x41, 0x42), expected):
            if row['kind'] == 'chirp' and abs(row['onset_sample']/FS-target) < .02:
                assert row['status'] == status and answer.word == want
                late[status] = answer.word
        turn.keyed()
    assert controls >= {36, 7, 38} and len(late) == 3


def test_decoded_ws8eoc_normal_long_tag_and_partner():
    from hfmodem.shrike import session
    from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
    path = 'captures/onair-20261010T231907712980Z-4e7a3dcc-d17a-4cb9-b588-1a49d3996714'
    audio_path = RECORD / path / 'stream.flac'
    require_input(audio_path)
    row = next(r for r in registrations(path, None) if r['kind'] == 'long')
    at, cfo = row['onset_sample'], row['cfo_hz']
    origin = at-24000
    audio = session.load_wav(str(audio_path), FS)[origin:at+round(3.7*FS)]
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, cfo)
    # The real normal-long lattice selects the tag, even before the CRC read.
    observed, coherence = rx._normal_long(audio, origin)
    assert coherence > .6 and observed.frame is not None
    assert observed.speedlevel == 'SL5-long' and not observed.sf8
    assert observed.frame.status.byte == row['status'] == 2
    assert observed.frame.data == bytes.fromhex(row['data']) and len(observed.frame.data) == 203
    turn, _ = machine(role='irs')
    assert key_packet(turn, FS, 1, Kind.LONG, speedlevel='SL5-long').word == 'X27'
    answer = propose_ack(turn, RegisteredObservation(observed, Registered.NEW))
    assert answer.word == 'T27'
    turn.keyed()
    assert key_packet(turn, at+240000, 3, Kind.LONG, speedlevel='3').word == 'X27'


@pytest.mark.parametrize('rejection', ['unread', 'provisional', 'weak', 'unexpected', 'refused'])
def test_k4spb_late_chirp_rejected_copy_does_not_advance(rejection):
    rows = registrations(ARMS[-1][0], 289*FS)
    turn, _ = machine(role='irs')
    for row in rows:
        if row['onset_sample'] >= 238*FS:
            break
        assert propose_ack(turn, registered(row)) is not None
        turn.keyed()
    assert turn._ack_chain == (3, 'X27')
    row = next(r for r in rows if r['kind'] == 'chirp'
               and abs(r['onset_sample']/FS-272.26) < .02)
    reg = registered(row)
    if rejection != 'refused':
        changes = {'frame': None} if rejection == 'unread' else {rejection: True}
        reg = replace(reg, observation=replace(reg.observation, **changes))
    answer = propose_ack(turn, reg)
    assert turn._ack_chain == (3, 'X27')
    if rejection == 'refused':
        assert answer.word == 'V27'
        turn.refused('deadline')
    elif answer is not None:
        assert not answer.current_slot_accepted
        turn.keyed()
    assert turn._ack_chain == (3, 'X27')


@pytest.mark.parametrize('policy', ['parity'])
@pytest.mark.parametrize('protocol,expected', [(0, ('X27', 'V27')), (3, ('U27', 'U27'))])
def test_off_default_head_body_words(policy, protocol, expected):
    turn, _ = machine(role='irs', normal_ack=policy, first_ack='v27', chain_reset='turn,form,ladder')
    for i, want in enumerate(expected):
        assert key_packet(turn, FS+i*180000, protocol*32+i, Kind.BODY).word == want


@pytest.mark.parametrize('policy,kind,expected', [
    ('parity', Kind.BODY, ('X27', 'V27', 'X27', 'V27')),
    ('parity', Kind.NORMAL, ('X27', 'V27', 'X27', 'V27')),
    ('parity', Kind.CHIRP, ('X27', 'V27', 'X27', 'V27')),
])
def test_off_default_head_iss_words(policy, kind, expected):
    name = 'chirp_ack' if kind == Kind.CHIRP else 'normal_ack'
    turn, arq = machine(data='body', first_ack='v27', **{name: policy})
    arq.on_host_data(b'ours'*100)
    turn._begin_data_turn(0)
    for i, want in enumerate(expected):
        turn._data(i*180000, State.ISS_DATA_ACK, kind)
        turn._next = replace(turn._next, status=0x60+i if kind == Kind.NORMAL else i)
        sent = turn.keyed()
        assert turn._expected_data_ack() == want
        turn.observe(word(want, sent.stream_sample+44160, responds_to=sent.stream_sample))


@pytest.mark.parametrize('policy', CHAIN_RESETS)
@pytest.mark.parametrize('boundary,kind,tags', [
    ('turn', Kind.NORMAL, {}), ('ladder', Kind.NORMAL, {}),
    ('form', Kind.LONG, {'speedlevel': '3'}), ('form', Kind.CHIRP, {}),
])
def test_boundary_reack_preserves_partner_word(policy, boundary, kind, tags):
    turn, _ = machine(role='irs', chain_reset=policy)
    assert key_packet(turn, FS, 3, Kind.BODY).word == 'X27'
    assert key_packet(turn, 180000, 0, Kind.NORMAL).word == 'V27'
    if boundary == 'turn':
        turn._close_receive_epoch('peer turn')
        turn._open_receive_epoch(350000)
    elif boundary == 'ladder':
        assert propose_ack(turn, short(350000)).word == 'X27'
        turn.keyed()
    expected = 'X27' if (boundary == 'turn'
                         or boundary == 'ladder' and boundary in policy.split(',')) else 'V27'
    assert key_packet(turn, 660000, 0, kind, **tags).word == expected


@pytest.mark.parametrize('policy', CHAIN_RESETS)
@pytest.mark.parametrize('tags,partner', [({}, 'V27'), ({'speedlevel': 'SL5-long'}, 'T27')])
def test_unread_at_ladder_boundary_uses_packet_partner(policy, tags, partner):
    turn, _ = machine(role='irs', chain_reset=policy)
    assert key_packet(turn, FS, 3, Kind.LONG, **tags).word == 'X27'
    assert propose_ack(turn, short(300000)).word == 'X27'
    turn.keyed()
    at = 480000
    unread = replace(body(at).observation, kind=Kind.LONG, frame=None, **tags)
    turn.observe(RegisteredObservation(unread, Registered.NEW))
    answer = turn.decide(at+163200-turn.lead)
    assert answer.word == (partner if 'ladder' in policy.split(',') else 'X27')
    assert not answer.current_slot_accepted and answer.ack_identity is None
    turn.keyed()
    assert turn._ack_chain == (3, 'X27')


@pytest.mark.parametrize('boundary', ['turn', 'ladder'])
@pytest.mark.parametrize('reset', ['turn,form', 'turn,form,ladder', 'form'])
def test_recorded_matching_counter_after_boundary(boundary, reset):
    rows = registrations(ARMS[-1][0], 289*FS)
    turn, _ = machine(role='irs', chain_reset=reset)
    for row in rows:
        if row['onset_sample'] >= 187*FS:
            break
        assert propose_ack(turn, registered(row)) is not None
        turn.keyed()
    assert turn._ack_chain == (3, 'X27')
    # Carry a real late c0 to the partner before opening the next turn/ladder.
    c0 = next(r for r in rows if r['kind'] == 'chirp' and r['status'] == 0x00
              and abs(r['onset_sample']/FS-272.26) < .02)
    assert propose_ack(turn, registered(c0)).word == 'V27'
    turn.keyed()
    at = c0['onset_sample'] + 180000
    if boundary == 'turn':
        turn._close_receive_epoch('peer turn')
        turn._open_receive_epoch(at)
    else:
        control = next(r for r in rows if r['kind'] == 'short' and r['value'] == 36)
        assert propose_ack(turn, registered(dict(control, onset_sample=at))) is not None
        turn.keyed()
    copy = registered(dict(c0, onset_sample=at+180000))
    expected = 'X27' if boundary == 'turn' or boundary in reset.split(',') else 'V27'
    proposal = propose_ack(turn, copy)
    assert proposal.word == expected
    turn.refused('deadline')
    assert turn._ack_chain == (0, 'V27')
    assert propose_ack(turn, replace(copy, observation=replace(
        copy.observation, stream_sample=at+360000))).word == expected


def test_recorded_empty_slot_after_k4spb_probes_keeps_robust_form():
    turn, _ = machine(role='irs')
    rows = registrations(ARMS[-1][0], 289*FS)
    for row in rows:
        if row['onset_sample'] >= 238*FS:
            break
        assert propose_ack(turn, registered(row)) is not None
        turn.keyed()
    assert turn._ack_chain == (3, 'X27') and turn._ack_form == 'robust'
    due = 250*FS
    turn.silence(due-turn.lead, answer_due=due)
    answer = turn.decide(due-turn.lead)
    assert answer is not None and answer.word == 'X27'
    assert answer.ack_identity is None and not answer.current_slot_accepted
    turn.keyed()
    assert turn._ack_chain == (3, 'X27') and turn._ack_form == 'robust'
    c0 = next(r for r in rows if r['kind'] == 'chirp' and r['status'] == 0x00
              and abs(r['onset_sample']/FS-272.26) < .02)
    assert propose_ack(turn, registered(c0)).word == 'V27'


@pytest.mark.parametrize('first,expected', [('x27', 'V27'), ('v27', 'X27')])
def test_empty_chain_without_held_answer_uses_opposite_first_ack(first, expected):
    turn, _ = machine(role='irs', first_ack=first)
    rows = registrations('operator-data/captures/onair-20261107T152917427998Z-64c4fbe2-3ac9-4306-899e-2b68aca47cf6', None)
    greeting = registered(next(r for r in rows if r['kind'] == 'body'))
    at = greeting.observation.stream_sample
    due = at + 44160
    turn.observe(replace(greeting, observation=replace(greeting.observation, frame=None)),
                 now=due-turn.lead)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and proposal.word == expected
    assert proposal.ack_identity is None and not proposal.current_slot_accepted
    turn.keyed()
    assert turn._ack_chain is None
    assert key_packet(turn, at+180000, 0, Kind.BODY).word == first.upper()


@pytest.mark.parametrize('first,expected', [('x27', 'V27'), ('v27', 'X27')])
def test_swapped_control_survives_recorded_unread_refinement(first, expected):
    rows = registrations('operator-data/captures/onair-20261107T152917427998Z-64c4fbe2-3ac9-4306-899e-2b68aca47cf6', None)
    control = next(r for r in rows if r['kind'] == 'short' and r['value'] == 53)
    turn, _ = machine(role='irs', peer_ack={53: first.upper()}, first_ack=first)
    assert propose_ack(turn, registered(control)).word == first.upper()
    turn.keyed()
    at = control['onset_sample'] + 60000
    due = at + 163200
    unread = replace(body(at).observation, kind=Kind.CHIRP, frame=None)
    turn.observe(RegisteredObservation(unread, Registered.NEW), now=due-turn.lead)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and proposal.word == expected
    assert proposal.purpose == 'repeat-request' and proposal.peer_control is None
    turn.observe(RegisteredObservation(unread, Registered.ENRICHED), now=due-turn.lead+1)
    assert turn.can_key(proposal) and turn.decide(due-turn.lead+1) is not None
    turn.keyed()
    assert turn._ack_chain is None



def test_recorded_body_bytes_keep_iss_phase_without_ladder():
    rows = registrations(ARMS[-1][0], 289*FS)
    payload = bytes.fromhex(next(r['data'] for r in rows if r['kind'] == 'body'))
    turn, arq = machine(data='body')
    arq.on_host_data(payload + b'next field')
    turn._begin_data_turn(0)
    sent = turn.keyed()
    assert sent.kind == Kind.BODY and sent.status == 0 and sent.payload == payload
    at = sent.stream_sample + 44160
    turn.observe(word('X27', at, responds_to=sent.stream_sample))
    assert turn._last_data_ack == 'X27' and turn._last_data_form == 'normal short'
    turn._after_ladder(300000)
    normal = turn.keyed()
    assert normal.kind == Kind.NORMAL and normal.status & 3 == 1
    assert turn._expected_data_ack() == 'V27'
    for rejected in ('X27', 'T27', 'U27'):
        assert not turn._data_ack(word(rejected, 344160).observation)
    turn.observe(word('V27', 344160, responds_to=normal.stream_sample))
    assert arq.tx_acked_bytes == len(payload) + len(b'next field')


@pytest.mark.parametrize('name,policy,kind,_', LEGACY_WORDS + [
    (name, policy, Kind.BODY, expected) for name, policy, _, expected in LEGACY_WORDS
    if name == 'normal_ack'])
def test_off_default_unread_keys_repeat_request(name, policy, kind, _):
    rows = registrations('operator-data/captures/onair-20261107T152917427998Z-64c4fbe2-3ac9-4306-899e-2b68aca47cf6', None)
    control = next(r for r in rows if r['kind'] == 'short' and r['value'] == 53)
    for held, expected in ((None, 'U27'), ('X27', 'U27'), ('U27', 'U27')):
        turn, _ = machine(role='irs', data='body', first_ack='v27',
                          peer_ack={53: held or 'X27'}, **{name: policy})
        if held:
            assert propose_ack(turn, registered(control)).word == held
            turn.keyed()
        at = control['onset_sample'] + 60000
        unread = replace(body(at).observation, kind=kind, frame=None)
        due = at + round(turn.t[kind]*FS)
        turn.observe(RegisteredObservation(unread, Registered.NEW), now=due-turn.lead)
        answer = turn.decide(due-turn.lead)
        assert answer is not None and answer.word == expected
        assert answer.ack_identity is None and not answer.current_slot_accepted
        if held == 'X27':
            assert answer.purpose == 'repeat-request' and answer.peer_control is None
        turn.keyed()
        assert turn._ack_chain is None


@pytest.mark.parametrize('reset,unread,checked', [('turn,form', 'V27', 'X27'),
                                                ('form', 'X27', 'V27')])
def test_recorded_unregistered_turn_uses_reset_or_carried_phase(reset, unread, checked):
    rows = registrations(ARMS[-1][0], 289*FS)
    turn, _ = machine(role='irs', chain_reset=reset)
    for row in rows:
        if row['onset_sample'] >= 187*FS:
            break
        assert propose_ack(turn, registered(row)) is not None
        turn.keyed()
    assert turn._ack_chain == (3, 'X27')
    turn._close_receive_epoch('peer turn')
    turn._open_receive_epoch(240*FS)
    assert turn._held_answer is None
    due = 250*FS
    turn.silence(due-turn.lead, answer_due=due)
    proposal = turn.decide(due-turn.lead)
    assert proposal is not None and proposal.word == unread
    assert proposal.ack_identity is None and not proposal.current_slot_accepted
    turn.keyed()
    assert turn._ack_chain == (3, 'X27')
    c0 = next(r for r in rows if r['kind'] == 'chirp' and r['status'] == 0x00
              and abs(r['onset_sample']/FS-272.26) < .02)
    assert propose_ack(turn, registered(c0)).word == checked
