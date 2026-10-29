# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Host

P1 = json.loads((Path(__file__).parent / 'fixtures' / 'varac-p1.json').read_text())
MESSAGES = {row['message_index']: row for row in P1}


def wire(index):
    return b''.join(bytes.fromhex(part) for part in MESSAGES[index]['fragments_hex'])


def test_utf8_lengths_concatenation_and_every_split():
    messages = ['', 'café <NAME:Alex>', '<R-05>', 'first\r\nsecond', '&IT&']
    raw = b''.join(v.frame_message(text) for text in messages)
    assert v.frame_message('café') == b'5 caf\xc3\xa9'
    for split in range(len(raw) + 1):
        parser = v.MessageParser()
        got = parser.feed(raw[:split]) + parser.feed(raw[split:])
        assert got == [text.encode() for text in messages]
        assert not parser.pending


@pytest.mark.parametrize('row', P1, ids=lambda row: str(row['message_index']))
def test_sanitized_capture_frames_reassemble(row):
    host = Host()
    for part in row['fragments_hex']:
        host.data.incoming.append(bytes.fromhex(part))
        host.client.poll()
    # The VMail's closing '>' followed its counted bytes in the same frame (session 021).
    tail = '>' if row['content'].startswith('<SM>') else ''
    if tail:
        assert not host.events
        host.data.incoming.append(b'>')
        host.client.poll()
    assert [e.raw for e in host.events if e.kind == 'message'] == [row['content'] + tail]
    assert ''.join(e.raw for e in host.events if e.kind in ('text', 'tag')) == row['content'] + tail
    assert v.frame_message(row['content']) == wire(row['message_index'])
    host.client.connected = True
    host.client.send_text(row['content'])
    assert host.data.sent == [wire(row['message_index'])]


@pytest.mark.parametrize('name,value,raw', [
    ('NAME', 'Alex', '<NAME:Alex>'), ('QTH', 'Test Town, ZZ', '<QTH:Test Town, ZZ>'),
    ('LOC', 'AA00AA', '<LOC:AA00AA>'), ('EM', 'a@example.test', '<EM:a@example.test>'),
    ('FC', 'N1ZZZ/M', '<FC:N1ZZZ/M>'), ('SBJ', 'RE: Test', '<SBJ:RE: Test>'),
    ('Q', None, '<Q>'), ('Q', '-150', '<Q:-150>'), ('S', 'VE', '<S:VE>'),
    ('R', '-05', '<R-05>'), ('R', '+00', '<R+00>'), ('R', '+11', '<R+11>'),
    *((name, None, '<' + name + '>') for name in v.BARE_TAGS),
    ('&IT&', None, '&IT&'),
])
def test_measured_tag_forms(name, value, raw):
    assert v.emit_tag(name, value) == raw
    parser, parts = v.TextParser(), []
    for char in 'a ' + raw + ' z':
        parts.extend(parser.feed(char))
    parts.extend(parser.feed('', final=True))
    assert ''.join(part.raw for part in parts) == 'a ' + raw + ' z'
    assert [(part.tag, part.value) for part in parts if part.tag] == [(name, value)]


@pytest.mark.parametrize('name,value', [('NAME', None), ('QTH', ''), ('LOC', 'a>b'),
    ('R', '5'), ('R', 100), ('R', '-005'), ('R', '+5'), ('Q', '1.5'),
    ('QSYR', 'yes'), ('RIG', 'test'), ('ANT', None), ('SR', None), ('S:A', None)])
def test_unmeasured_emission_is_rejected(name, value):
    with pytest.raises(ValueError):
        v.emit_tag(name, value)


def test_unknown_markup_and_raw_token_prefixes_remain_lossless():
    text = '<R:05><R+5><NAME><RIG:test>&I&IT& &IT <NAME:'
    parser, parts = v.TextParser(), []
    for char in text:
        parts += parser.feed(char)
    parts += parser.feed('', final=True)
    assert ''.join(part.raw for part in parts) == text
    assert [part.tag for part in parts if part.tag] == ['NAME', '&IT&']


@pytest.mark.parametrize('raw', [b'x ', b'-1 a', b'1000001 ', b'1234567890'])
def test_bad_length_blocks_contact_and_keeps_host_close_path(raw):
    host = Host()
    host.data.incoming.append(raw)
    host.client.poll()
    assert not host.command.closed and not host.data.closed
    assert host.client._receive_quarantined
    assert host.events[-1].kind == 'peer_protocol_error'
    assert not host.client.messages


def test_partial_message_does_not_trigger_tags_and_disconnect_resets_it():
    host = Host()
    host.client._text(b'6 <DIS')
    assert not host.events
    host.client._line('DISCONNECTED')
    assert host.events[0] == v.Event('incomplete', '6 <DIS')
    host.client._text(b'4 &IT&')
    assert host.events[-1] == v.Event('tag', '&IT&')


# CQ -> setup: R06/R07 W5EI 5/-5; R07 AE5JO 9/0, KQ4QWH 13/+3,
# K7QB 2/-2; R09 WE3MAY 12/+2; R11 AA6EA 15/+5, WX2DX 4/-4.
@pytest.mark.parametrize('call,slot,offset', [
    ('W5EI-5', 5, -3750), ('AE5JO-9', None, 0), ('KQ4QWH-13', 13, 2250),
    ('K7QB-2', 2, -1500), ('WE3MAY-12', 12, 1500),
    ('AA6EA-15', 15, 3750), ('WX2DX-4', 4, -3000),
])
def test_cq_suffix_evidence(call, slot, offset):
    codec = v.SlotCodec()
    assert codec.decode(call) == (call.split('-')[0], slot)
    if slot is not None:
        assert codec.offset_hz(slot) == offset
        assert codec.cq_command(call.split('-')[0], slot) == f'CQFRAME {call} 500'


@pytest.mark.parametrize('slot,offset', [(n, -n * 750) for n in range(1, 6)] +
                         [(n + 10, n * 750) for n in range(1, 6)])
def test_production_mapping_for_all_ten_slots(slot, offset):
    assert v.slot_frequency(20, slot) == 14_105_000 + offset
    assert v.DEFAULT_CODEC.cq_command('w9ssj-7', slot) == f'CQFRAME W9SSJ-{slot} 500'
    for bw in v.BANDWIDTHS:
        assert v.DEFAULT_CODEC.cq_command('w9ssj-7', slot, bw) == f'CQFRAME W9SSJ-{slot} {bw}'


@pytest.mark.parametrize('bw', [0, 1000, 2400, '500'])
def test_cq_bandwidth_is_one_of_varacs_three(bw):
    with pytest.raises(ValueError):
        v.DEFAULT_CODEC.cq_command('W9SSJ', 1, bw)


@pytest.mark.parametrize('slot', [0, 6, 9, 10, 16, -1])
def test_non_slot_suffixes_cannot_be_selected(slot):
    with pytest.raises(ValueError):
        v.slot_frequency(20, slot)


def session(mycall='KQ4QWH', snr=-4, services=None):
    host = Host(replies={'DISCONNECT': ['OK', 'DISCONNECTED']})
    host.client.mycall = mycall
    host.client.connected, host.client.snr = True, snr
    tunes = []
    flow = v.Flow(host.client, tunes.append, services=services,
                  profile={'NAME': 'Alex', 'QTH': 'Test Town', 'LOC': 'AA00AA'})
    def retune(target, generation):
        flow.tuner(target)
        yield from ()
    flow.received_retune_steps = retune
    flow.dial = v.CALLING_HZ[20]
    return host, tunes, flow


def test_r08_caller_qsy_waits_for_delivery_then_tl_qsyr_and_opening():
    host, tunes, flow = session()
    host.client._text(v.frame_message('<AWQ>'))  # current-link readiness prerequisite
    sent_on = []
    def reply(data):
        sent_on.append((flow.dial, data))
        if data == v.frame_message('<QSYF>000014108750</QSYF>'):
            host.lines('BUFFER 16', 'PTT ON', 'BUFFER 0', 'PTT OFF')
        elif data == wire(136):
            host.data.incoming.append(wire(137))
    host.data.respond = reply
    flow.request_qsy(15, timeout=1)
    assert tunes == [14_108_750]
    # Keep the recorded destination and handshake, using the explicit QF form
    # for our caller instead of assuming the peer supports frequency offsets.
    assert sent_on[:2] == [(14_105_000, v.frame_message('<QSYF>000014108750</QSYF>')), (14_108_750, wire(136))]
    assert host.data.sent[2] == v.frame_message(MESSAGES[138]['content'].split('\r\n')[0])
    assert host.data.sent[3] == v.frame_message('<NAME:Alex>\r\n<QTH:Test Town>\r\n<LOC:AA00AA>')
    assert flow.state == v.State.CONNECTED
    events = [(e.kind, e.raw) for e in host.events]
    assert events.index(('host', 'PTT OFF')) < events.index(('tune', '14108750'))
    assert events.index(('message', '<QSYR>')) < events.index(('sent', 'de KQ4QWH <R-04>'))


def test_r08_answerer_follows_q_tl_then_sends_qsyr_and_profile():
    host, tunes, flow = session('N9NLP', 1, 'VB')
    def tune(dial):
        tunes.append(dial)
        host.data.incoming.append(wire(136))
    flow.tuner = tune
    def reply(data):
        assert flow.dial == 14_108_750
        if data == wire(137):
            host.data.incoming.append(wire(138))
    host.data.respond = reply
    host.client._text(wire(94))
    request, after, generation, origin = flow.receive()
    host.client.finish(flow.answer_qsy_steps(request, after=after, generation=generation, origin=origin))
    assert tunes == [14_108_750]
    assert host.data.sent[:2] == [wire(137), wire(139)]
    events = [(e.kind, e.raw) for e in host.events]
    assert events.index(('message', '<TL>')) < events.index(('sent', '<QSYR>'))
    assert events.index(('message', MESSAGES[138]['content'])) < events.index(('sent', MESSAGES[139]['content']))
    flow.receive()
    assert len(tunes) == 1


def test_session_010_opening_precedes_profile_on_the_cq_slot():
    host, tunes, flow = session('K7QB', 11, 'VE')
    flow.dial = v.slot_frequency(20, 2)
    flow.opening()
    assert host.data.sent[0] == wire(74)
    assert host.data.sent[1].endswith(b'<NAME:Alex>\r\n<QTH:Test Town>\r\n<LOC:AA00AA>')
    assert not tunes
    assert len(MESSAGES[76]['fragments_hex']) == 2


@pytest.mark.parametrize('slot', [None, 2])
def test_caller_opening_waits_for_first_received_over_sn_after_connected(slot):
    host = Host(replies={'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    tunes = []
    flow = v.Flow(host.client, tunes.append, services='VE', profile={'NAME': 'Alex'})
    host.client.finish(flow.call_steps('K7QB', slot=slot))
    assert host.client.snr is None
    connected_at = host.now
    def ready(sockets, timeout):
        assert not host.data.sent
        host.now += timeout
        if host.now - connected_at >= 35:
            host.lines('SN -4.4', 'BITRATE (1) 18 bps RX')
        return [stream for stream in sockets if stream.incoming]
    host.client.ready = ready
    flow.opening()
    assert 35 <= host.now - connected_at < 36
    assert host.data.sent == [v.frame_message('de W9SSJ <R-04> <S:VE>'),
                              v.frame_message('<NAME:Alex>')]
    assert tunes == ([] if slot is None else [14_103_500])
    assert host.sent == ['LISTEN OFF', 'CONNECT W9SSJ K7QB']


def test_opening_uses_latest_host_sn_without_waiting():
    host, tunes, flow = session(snr=None)
    host.lines('SN -12', 'SN 7.2', 'BITRATE (1) 18 bps RX')
    host.client.poll()
    now = host.now
    flow.opening()
    assert host.now == now
    assert host.data.sent[0] == v.frame_message('de KQ4QWH <R+07>')


def test_opening_without_sn_times_out_without_report_or_profile():
    host, tunes, flow = session(snr=None)
    with pytest.raises(TimeoutError, match='Opening signal report timed out'):
        flow.opening(timeout=.25)
    assert host.now == pytest.approx(.25)
    assert not host.data.sent and not tunes


@pytest.mark.parametrize('ending', [b'DISCONNECTED\r', b''])
def test_link_ending_during_opening_wait_sends_no_report_or_profile(ending):
    host, tunes, flow = session(snr=None)
    host.command.incoming.append(ending)
    with pytest.raises(v.HostError, match='Link ended|host closed'):
        flow.opening()
    assert host.now < 1
    assert not host.data.sent and not tunes


@pytest.mark.parametrize('index', [135, 142])
def test_recorded_farewell_is_delivered_before_host_disconnect(index):
    host, tunes, flow = session()
    host.data.respond = lambda data: host.lines('BUFFER 0')
    flow.farewell(MESSAGES[index]['content'].removesuffix('<DISC>'))
    assert host.data.sent == [wire(index)]
    assert host.sent == ['LISTEN OFF', 'DISCONNECT'] and not host.client.connected
    assert not tunes


@pytest.mark.parametrize('reply', [[], ['BUFFER 5'], ['BUFFER 0', 'PTT ON']])
def test_qsy_never_retunes_before_delivery_and_unkey(reply):
    host, tunes, flow = session()
    host.client._text(v.frame_message('<AWQ>'))  # current-link readiness prerequisite
    host.data.respond = lambda data: host.lines(*reply)
    with pytest.raises(TimeoutError):
        flow.request_qsy(15, timeout=.2)
    assert not tunes and flow.state == v.State.FAILED
    assert host.data.sent == [v.frame_message('<QSYF>000014108750</QSYF>')]


def test_missing_qsyr_sends_no_profile_and_does_not_retune_back():
    host, tunes, flow = session()
    host.client._text(v.frame_message('<AWQ>'))  # current-link readiness prerequisite
    host.data.respond = lambda data: host.lines('BUFFER 0')
    with pytest.raises(TimeoutError):
        flow.request_qsy(15, timeout=.2)
    assert tunes == [14_108_750]
    assert host.data.sent == [v.frame_message('<QSYF>000014108750</QSYF>'), wire(136)]


@pytest.mark.parametrize('invitation', ['<Q:1>', '<Q:999999>', '<Q:0>', '<Q:no>'])
def test_bad_qsy_request_cannot_tune_or_emit(invitation):
    host, tunes, flow = session()
    with pytest.raises(ValueError):
        flow.answer_qsy(invitation)
    assert not tunes and not host.data.sent


def test_disconnected_or_wrong_frequency_qsy_cannot_tune():
    host, tunes, flow = session()
    host.client.connected = False
    with pytest.raises(v.HostError):
        flow.answer_qsy('<Q:375>')
    host.client.connected = True
    flow.dial += 750
    with pytest.raises(v.HostError):
        flow.request_qsy(15)
    assert not tunes and not host.data.sent


def test_q_and_tl_coalesced_in_one_read_are_not_lost():
    host, tunes, flow = session('N9NLP', 1, 'VB')
    host.client._text(wire(94) + wire(136))
    host.data.respond = lambda data: host.data.incoming.append(wire(138)) if data == wire(137) else None
    request, after, generation, origin = flow.receive()
    host.client.finish(flow.answer_qsy_steps(request, after=after, generation=generation, origin=origin))
    assert tunes == [14_108_750] and host.data.sent[:2] == [wire(137), wire(139)]


def test_old_empty_buffer_followed_by_nonempty_is_not_delivery():
    host, tunes, flow = session()
    host.data.respond = lambda data: host.lines('BUFFER 0', 'BUFFER 9')
    with pytest.raises(TimeoutError):
        flow.request_qsy(15, timeout=.2)
    assert not tunes


def test_answerer_tuner_failure_emits_no_acknowledgement_or_profile():
    host, tunes, flow = session()
    def fail(dial):
        raise OSError('tuner unavailable')
    flow.tuner = fail
    with pytest.raises(OSError):
        flow.answer_qsy('<Q:375>')
    assert not host.data.sent and flow.state == v.State.FAILED


def test_production_cq_commands_and_direct_slot_call_use_real_codec():
    command = 'CQFRAME W9SSJ-2 500'
    host = Host(replies={command: ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING'],
                         'LISTEN CQ': ['OK', 'CONNECTED KK6SVV W9SSJ 500']})
    tunes = []
    flow = v.Flow(host.client, tunes.append, occupied=lambda: False)
    assert host.client.finish(flow.cq_steps(2, sniff_seconds=.2))
    assert host.sent.count(command) == 2
    assert tunes == [14_103_500, 14_105_000, 14_103_500]
    other = Host(replies={'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    other.client.finish(v.Flow(other.client, tunes.append).call_steps('K7QB', slot=2))
    assert tunes[-1] == 14_103_500


def test_answerer_default_window_covers_r08_measured_q_to_tl_delay():
    host, tunes, flow = session('N9NLP', 1, 'VB')
    q_end = MESSAGES[94]['sources'][-1]['stop_s']
    tl_delay = MESSAGES[136]['sources'][-1]['stop_s'] - q_end
    assert tl_delay > 30
    scheduled = [(tl_delay, wire(136))]
    def ready(sockets, timeout):
        host.now += timeout
        if scheduled and host.now >= scheduled[0][0]:
            host.data.incoming.append(scheduled.pop(0)[1])
        return [stream for stream in sockets if stream.incoming]
    host.client.ready = ready
    host.data.respond = lambda data: scheduled.append((host.now + 6, wire(138))) if data == wire(137) else None
    flow.answer_qsy('<Q:375>')
    assert host.now > tl_delay and host.data.sent[:2] == [wire(137), wire(139)]


def test_session_021_vmail_parses_and_re_emits_byte_for_byte():
    # The counted 196 bytes end at CR LF; the closing '>' followed in the same frame.
    sent = wire(126) + b'>'
    assert len(wire(126)) == 4 + 196 and not MESSAGES[126]['content'].endswith('>')
    for split in range(1, len(sent)):
        parser = v.MessageParser()
        messages = parser.feed(sent[:split]) + parser.feed(sent[split:])
        assert messages == [sent[4:]], split
    assert v.MessageParser().feed(sent[:-1]) == [] and v.MessageParser().feed(sent[:-1], final=True) == [sent[4:-1]]
    assert v.MessageParser().feed(sent[:-1] + b'5 <SMR>') == [sent[4:-1], b'<SMR>']
    text = sent[4:].decode()
    vmail = v.VMail.parse(text)
    assert vmail.subject == 'RE: Exploring New Paths - 2'
    assert vmail.body.startswith('KL7QR» Hello Ale,\n\nI was excited') and vmail.body.endswith('MUCH !\r\n')
    assert [part.tag for part in v.TextParser().feed(text, final=True)] == ['SM', 'SBJ', 'MSG']
    host = Host()
    host.client.connected = True
    host.client.send_vmail(vmail)
    assert host.data.sent == [sent]
    assert v.emit_tag('SMR') == MESSAGES[123]['content'] == '<SMR>'
    assert v.emit_tag('AWAY') == '<AWAY>'


@pytest.mark.parametrize('text', ['<SM>', '<SM><SBJ:x>', '<SM><MSG:open', 'hello <SM><SBJ:x><MSG:y>',
                                  '<SBJ:x><MSG:y>', '<SM><SBJ:x><MSG:y>\r\n'])
def test_only_a_whole_vmail_message_is_a_vmail(text):
    assert v.VMail.parse(text) is None


def test_a_subjectless_relay_envelope_remains_forwardable():
    assert v.VMail.parse('<SM><MSG:hi>') == v.VMail('', 'hi')
    assert v.VMail('', 'hi').wire() == '<SM><SBJ:><MSG:hi>'
    for vmail in (v.VMail('a>b', 'hi'), v.VMail('two\r\nlines', 'hi'), v.VMail('s', 'a<b')):
        with pytest.raises(ValueError):
            vmail.wire()
