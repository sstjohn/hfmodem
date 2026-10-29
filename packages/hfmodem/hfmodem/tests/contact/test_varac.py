# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import tempfile

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.evidence import RECORD, require_input

SESSIONS = RECORD / 'operator-data' / 'working' / 'vara-chat' / 'sessions'


def utc(record):
    return datetime.fromisoformat(record['utc']).timestamp()

P0 = json.loads((Path(__file__).parent / 'fixtures' / 'varac-p0.json').read_text())
P1 = {row['message_index']: row for row in
      json.loads((Path(__file__).parent / 'fixtures' / 'varac-p1.json').read_text())}
#: Session 021's VMail, R04 slot -2, 1551.868–1561.077 s, two frames: 196 counted bytes
#: and the closing '>' after them.
VMAIL_021 = P1[126]['content'] + '>'


def vmail_arrives(desk, content=VMAIL_021):
    desk.host.data.incoming.append(v.frame_message(content[:-1]) + b'>')
    desk.tick()


class Socket:
    def __init__(self, respond=lambda value: None):
        self.incoming = deque()
        self.sent = []
        self.closed = False
        self.respond = respond

    def sendall(self, data):
        assert not self.closed
        self.sent.append(data)
        self.respond(data)

    def recv(self, size):
        return self.incoming.popleft()

    def close(self):
        self.closed = True


class Host:
    def __init__(self, *, replies=None, kiss=False):
        self.now = 0
        self.events = []
        self.replies = replies or {}
        self.command = Socket(self.respond)
        self.data = Socket()
        # A KISS frame keys as a CQFRAME does, with no reply of its own.
        self.kiss = Socket(lambda data: self.lines(*CQ_DONE[1:])) if kiss else None
        self.client = v.Client(self.command, self.data, self.kiss, clock=lambda: self.now, ready=self.ready,
                               event=self.events.append)
        self.client.mycall = 'W9SSJ'
        self.client.bandwidth = 500

    def respond(self, data):
        command = data.decode().rstrip('\r')
        response = self.replies.get(command, ['CLEANTXBUFFER BUFFEREMPTY']
                                    if command == 'CLEANTXBUFFER' else ['OK'])
        if callable(response):
            response = response()
        self.lines(*response)

    def lines(self, *lines):
        if lines:
            self.command.incoming.append(('\r'.join(lines) + '\r').encode())

    def ready(self, sockets, timeout):
        self.now += timeout
        return [stream for stream in sockets if stream.incoming]

    @property
    def sent(self):
        return [line.decode().rstrip('\r') for line in self.command.sent]


class TestCodec:
    __test__ = False
    source = 'synthetic test codec; not a public on-air encoding'

    def offset_hz(self, slot):
        return {1: -750, 2: 1500}[slot]

    def cq_command(self, call, slot, bandwidth):
        return f'CQFRAME {call} test-slot-{slot}'


@pytest.mark.parametrize('name,tag', [(name, tag) for name, tag in v.TAGS.items() if not tag.emit])
def test_documented_literal_tags_round_trip(name, tag):
    parser = v.TextParser()
    parts = []
    for char in 'hello ' + tag.wire + ' world':
        parts += parser.feed(char)
    parts += parser.feed('', final=True)
    assert ''.join(part.raw for part in parts) == 'hello ' + tag.wire + ' world'
    assert [part.tag for part in parts if part.tag] == ['S' if name.startswith('S:') else name]
    assert tag.source.startswith('https://')
    if tag.emit:
        assert v.emit_tag(name) == tag.wire
    else:
        with pytest.raises(ValueError, match='evidenced'):
            v.emit_tag(name)


def test_unknown_and_incomplete_tags_are_never_dropped():
    raw = 'José <UNKNOWN:a>b</UNKNOWN> <NAME>Saul <incomplete'
    parser = v.TextParser()
    parts = parser.feed(raw) + parser.feed('', final=True)
    assert ''.join(part.raw for part in parts) == raw
    assert [part.tag for part in parts if part.tag] == ['NAME']
    with pytest.raises(ValueError):
        v.emit_tag('UNKNOWN')


def test_unavailable_sn_clears_the_previous_stations_report():
    desk = Desk(500)
    desk.hear('SN 9.4', 'CQFRAME W9SSJ 500')
    desk.hear('SN --', 'CQFRAME K5CG 500')
    assert desk.host.client.snr is None
    assert 'SN   -- ' in desk.legacy_said[-1]


def test_fragmented_utf8_and_tags_survive_both_ports():
    host = Host()
    raw = 'José <NAME>Saul <FUTURE:x> <unfinished'
    host.command.incoming.extend([b'SN -', b'9.3\rBITRATE (1)  18 bps RX\r'])
    host.data.incoming.extend(bytes([byte]) for byte in v.frame_message(raw))
    while host.data.incoming or host.command.incoming:
        host.client.poll()
    host.client.close()
    assert host.client.snr == -9.3
    assert host.client.bitrate == 'BITRATE (1)  18 bps RX'
    assert ''.join(e.raw for e in host.events if e.kind in ('text', 'tag')) == raw


@pytest.mark.parametrize('path', P0)
def test_p0_host_lines_replay_verbatim(path):
    host = Host()
    raw = [row['text'] for row in P0[path] if row['kind'] == 'cmd_rx']
    payload = ('\r'.join(raw) + '\r').encode()
    host.command.incoming.extend(payload[i:i + 37] for i in range(0, len(payload), 37))
    while host.command.incoming:
        host.client.poll()
    assert [e.raw for e in host.events if e.kind == 'host'] == raw


def p0_reply(path, command, who):
    rows = P0[path]
    at = next(i for i, r in enumerate(rows) if r['kind'] == 'cmd_tx'
              and r['who'] == who and r['text'] == command)
    result = []
    for row in rows[at + 1:]:
        if row['who'] != who:
            continue
        if row['kind'] == 'cmd_tx':
            break
        result.append(row['text'])
    return result


def test_cq_command_waits_for_stock_completion_and_keeps_notifications():
    command = 'CQFRAME W9SSJ 500'
    reply = p0_reply('item4/20261018-135808-chat.jsonl', command, 'A')
    assert 'PENDING' in reply and 'CANCELPENDING' in reply
    host = Host(replies={command: reply})
    host.client.finish(host.client.cq_steps(command))
    assert not host.client.pending
    assert not host.client.ptt
    assert [e.raw for e in host.events if e.kind == 'host'] == reply


@pytest.mark.parametrize('reply', ['CLEANTXBUFFER BUFFEREMPTY', 'CLEANTXBUFFER FAILED ', 'CLEANTXBUFFER OK'])
def test_clean_buffer_exact_stock_replies(reply):
    assert any(row['text'] == reply for rows in P0.values() for row in rows)
    host = Host(replies={'CLEANTXBUFFER': ['IAMALIVE', reply]})
    assert host.client.clean_buffer() == reply[len('CLEANTXBUFFER '):]


def test_optional_commands_work_on_older_kestrel():
    host = Host(replies={'LISTEN CQ': ['WRONG'], 'CLEANTXBUFFER': ['WRONG']})
    host.client.listen()
    assert host.sent == ['LISTEN CQ', 'LISTEN ON']
    with pytest.raises(v.Unsupported):
        host.client.clean_buffer()


@pytest.mark.parametrize('reply, reason', [(['WRONG'], 'answered WRONG'), (['OK'], 'never keyed'),
                                           (['OK', 'BUSY ON'], 'stayed busy')])
def test_cq_rejection_or_noop_is_not_transmission(reply, reason):
    host = Host(replies={'CQFRAME W9SSJ 500': reply})
    with pytest.raises(v.Unsupported, match=reason):
        host.client.finish(host.client.cq_steps('CQFRAME W9SSJ 500', timeout=.5))


def test_cq_dropped_on_a_busy_channel_is_sent_again_once_it_clears():
    # 20261023T032225838464Z: OK while BUSY ON, then BUSY OFF 1.8 s later and no key.
    keyed = ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING']
    replies = iter([['OK', 'BUSY OFF'], keyed])
    host = Host(replies={'CQFRAME W9SSJ-1 500': lambda: next(replies)})
    host.client.finish(host.client.cq_steps('CQFRAME W9SSJ-1 500'))
    assert host.sent == ['CQFRAME W9SSJ-1 500'] * 2
    assert host.now >= v.CLEAR_S
    assert not host.client.ptt
    assert [e for e in host.events if e.kind == 'held'] == [v.Event('held', 'CQ')]


def scripted(host, script):
    """Deliver each (seconds, line) of `script` once the fake clock reaches it."""
    ready = host.ready
    def deliver(sockets, timeout):
        while script and script[0][0] <= host.now:
            host.lines(script.pop(0)[1])
        return ready(sockets, timeout)
    host.client.ready = deliver


def test_a_cq_dropped_as_busy_is_not_sent_into_the_20_m_flicker():
    # 16:01:32-52Z on 20 m: BUSY toggled every 0.2-0.7 s and the CQ went to the host 12 times.
    rows = [json.loads(line) for line in require_input(
        SESSIONS / '20261023T155947666857Z' / 'host.jsonl').read_text().splitlines()]
    sent = next(i for i, r in enumerate(rows) if r['utc'] == '2026-10-23T16:01:32.425+00:00'
                and r['raw'] == 'CQFRAME W9SSJ-3 500')
    start = utc(rows[sent])
    script = [(utc(r) - start, r['raw']) for r in rows[sent:]
              if r['raw'] in ('BUSY ON', 'BUSY OFF') and utc(r) - start <= 20]
    assert sum(r['raw'] == 'CQFRAME W9SSJ-3 500' for r in rows[sent:sent + 80]) == 12
    host = Host()
    host.client.busy = True
    scripted(host, script)
    with pytest.raises(v.Unsupported, match='stayed busy'):
        host.client.finish(host.client.cq_steps('CQFRAME W9SSJ-3 500'))
    assert host.sent == ['CQFRAME W9SSJ-3 500']
    assert [e for e in host.events if e.kind == 'held'] == [v.Event('held', 'CQ')]


def test_a_cq_dropped_as_busy_goes_once_the_host_reports_clear_long_enough():
    host = Host(replies={'CQFRAME W9SSJ-3 500': lambda: ['OK'] if len(host.sent) == 1 else CQ_DONE})
    host.client.busy = True
    scripted(host, [(1, 'BUSY OFF'), (3, 'BUSY ON'), (4, 'BUSY OFF')])
    host.client.finish(host.client.cq_steps('CQFRAME W9SSJ-3 500'))
    assert host.sent == ['CQFRAME W9SSJ-3 500'] * 2
    assert 4 + v.CLEAR_S <= host.now < 4 + v.CLEAR_S + 1


def test_initialization_leaves_the_host_not_listening_and_reporting_cqs():
    host = Host(replies={'VERSION': ['VERSION test-host']})
    host.client.initialize('w9ssj')
    assert host.sent == ['LISTEN OFF', 'PUBLIC ON', 'CWID OFF', 'COMPRESSION TEXT',
                         'MYCALL W9SSJ', 'BW500', 'VERSION', 'CHAT ON', 'LISTEN OFF']
    host.lines('CQFRAME WI9YZA-15 500')
    host.client.poll()
    assert host.events[-1] == v.Event('host', 'CQFRAME WI9YZA-15 500')
    with pytest.raises(v.HostError):
        host.client.send_text('no')
    assert not host.data.sent


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_initialization_sets_the_session_bandwidth(bw):
    host = Host(replies={'VERSION': ['VERSION test-host']})
    host.client.initialize('w9ssj', bw)
    assert host.sent[5] == f'BW{bw}' and host.client.bandwidth == bw


def test_initialization_refuses_a_width_varac_does_not_use():
    host = Host()
    with pytest.raises(ValueError):
        host.client.initialize('W9SSJ', 1000)
    assert not host.sent


def test_connect_text_and_called_station_disconnect():
    reply = p0_reply('item7/20261018-140023-chat.jsonl', 'DISCONNECT', 'B')
    assert 'DISCONNECTED' in reply
    host = Host(replies={'CONNECT W9SSJ WI9YZA': ['OK', 'SN 20.5', 'CONNECTED W9SSJ WI9YZA 500'],
                         'DISCONNECT': reply})
    host.client.finish(host.client.connect_steps('wi9yza'))
    host.client.send_text('café\n')
    host.client.send_tag('DISC')
    assert host.data.sent == [b'6 caf\xc3\xa9\n', b'6 <DISC>']
    host.client.disconnect()
    assert not host.client.connected


def test_timeout_closes_streams_to_prevent_reply_desynchronization():
    host = Host(replies={'BW500': []})
    with pytest.raises(TimeoutError):
        host.client.request('BW500', timeout=.2)
    assert host.command.closed and host.data.closed


def test_second_port_open_failure_closes_first():
    command = Socket()
    def connect(address, **kwargs):
        if address[1] == 8300:
            return command
        raise OSError('data unavailable')
    with pytest.raises(OSError):
        v.Client.open(connect=connect)
    assert command.closed


def test_eof_flushes_pending_text_and_closes_both_ports():
    host = Host()
    host.data.incoming.extend([b'7 <future', b''])
    host.client.poll()
    with pytest.raises(v.HostError, match='closed'):
        host.client.poll()
    assert host.command.closed and host.data.closed
    assert host.events[-1].raw == '<future'


def flow_host(answer=False):
    host = Host(replies={'CQFRAME W9SSJ test-slot-1':
                        ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING'],
                        'LISTEN CQ': ['OK', 'CONNECTED WI9YZA W9SSJ 500'] if answer else ['OK']})
    tunes = []
    flow = v.Flow(host.client, tunes.append, codec=TestCodec())
    return host, tunes, flow


@pytest.mark.parametrize('answer', [True, False])
def test_slot_sniff_two_cqs_qsy_wait_and_return(answer):
    host, tunes, flow = flow_host(answer)
    assert host.client.finish(flow.cq_steps(1, sniff_seconds=.3, wait_seconds=.5)) is answer
    assert tunes == [14_104_250, 14_105_000, 14_104_250] + ([] if answer else [14_105_000])
    assert host.sent.count('CQFRAME W9SSJ test-slot-1') == 2
    assert flow.state == (v.State.CONNECTED if answer else v.State.RETURNED)
    order = [(e.kind, e.raw) for e in host.events if e.kind in ('tune', 'command')]
    assert order.index(('tune', '14104250')) < order.index(('command', 'CQFRAME W9SSJ test-slot-1'))
    assert order.index(('command', 'LISTEN CQ')) > order.index(('command', 'CQFRAME W9SSJ test-slot-1'))


def test_cqs_start_one_varac_period_apart_on_the_calling_frequency():
    host, tunes, flow = flow_host()
    starts = []
    def frame():
        starts.append(host.now)
        host.now += 5.62
        return ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING']
    host.replies['CQFRAME W9SSJ test-slot-1'] = frame
    assert not host.client.finish(flow.cq_steps(1, sniff_seconds=.3, wait_seconds=.5))
    assert len(starts) == v.CQ_FRAMES == 2
    assert v.CQ_PERIOD_S <= starts[1] - starts[0] < v.CQ_PERIOD_S + .2
    assert [e.raw for e in host.events if e.kind == 'cq'] == ['1/2', '2/2']
    said = [e.raw for e in host.events]
    assert not any(e.kind == 'tune' for e in host.events[said.index('1/2'):said.index('2/2')])


def test_busy_slot_does_not_call():
    host, tunes, flow = flow_host()
    flow.occupied = lambda: True
    assert not host.client.finish(flow.cq_steps(1, sniff_seconds=.2))
    assert tunes[-1] == 14_105_000
    assert not any(command.startswith('CQFRAME') for command in host.sent)


def test_missing_slot_evidence_refuses_before_tuning_or_host_commands():
    host, tunes, flow = flow_host()
    flow.codec = None
    with pytest.raises(v.EvidenceMissing):
        host.client.finish(flow.cq_steps(1))
    with pytest.raises(v.EvidenceMissing):
        host.client.finish(flow.call_steps('WI9YZA', slot=1))
    assert not tunes and not host.sent


def test_qsy_failure_never_calls():
    host, tunes, flow = flow_host()
    def fail(dial):
        raise OSError('tuner unavailable')
    flow.tuner = fail
    with pytest.raises(OSError):
        host.client.finish(flow.cq_steps(1))
    assert flow.state == v.State.FAILED
    assert not any(command.startswith('CQFRAME') for command in host.sent)


@pytest.mark.parametrize('flag', ['ptt', 'pending', 'connected'])
def test_no_retune_during_link_or_keying(flag):
    host, tunes, flow = flow_host()
    setattr(host.client, flag, True)
    with pytest.raises(v.HostError, match='Cannot QSY'):
        flow.tune(14_105_000)
    assert not tunes


def test_call_with_known_slot_tunes_before_connect():
    host, tunes, flow = flow_host()
    host.replies['CONNECT W9SSJ WI9YZA'] = ['OK', 'CONNECTED W9SSJ WI9YZA 500']
    host.client.finish(flow.call_steps('WI9YZA', slot=2))
    assert tunes == [14_106_500]
    assert flow.state == v.State.CONNECTED


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_cq_at_each_bandwidth_advertises_it_and_waits_on_the_same_slot(bw):
    command = f'CQFRAME W9SSJ-2 {bw}'
    host = Host(replies={'VERSION': ['VERSION test-host'],
                         command: ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING'],
                         'LISTEN CQ': ['OK', f'CONNECTED KK6SVV W9SSJ {bw}']})
    host.client.initialize('W9SSJ', bw)
    tunes = []
    assert host.client.finish(v.Flow(host.client, tunes.append).cq_steps(2, sniff_seconds=.2, wait_seconds=.5))
    assert tunes == [14_103_500, 14_105_000, 14_103_500]
    assert host.sent.count(command) == 2
    assert host.sent.index(f'BW{bw}') < host.sent.index(command) < host.sent.index('LISTEN CQ')
    assert not any(line.startswith('BW') for line in host.sent[host.sent.index(command):])


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_call_at_each_bandwidth_connects_on_the_slot(bw):
    host = Host(replies={'VERSION': ['VERSION test-host'],
                         'CONNECT W9SSJ K7QB': ['OK', f'CONNECTED W9SSJ K7QB {bw}']})
    host.client.initialize('W9SSJ', bw)
    tunes = []
    host.client.finish(v.Flow(host.client, tunes.append).call_steps('K7QB', slot=12))
    assert tunes == [14_106_500]
    assert host.sent.index(f'BW{bw}') < host.sent.index('CONNECT W9SSJ K7QB')


def test_wait_timeout_during_pending_link_does_not_qsy():
    host, tunes, flow = flow_host()
    host.replies['LISTEN CQ'] = ['OK', 'PENDING']
    with pytest.raises(v.HostError, match='staying on the slot'):
        host.client.finish(flow.cq_steps(1, sniff_seconds=.2, wait_seconds=.2))
    assert tunes[-1] == 14_104_250


def test_failed_cq_never_moves_to_slot():
    host, tunes, flow = flow_host()
    host.replies['CQFRAME W9SSJ test-slot-1'] = ['OK']
    with pytest.raises(v.Unsupported):
        host.client.finish(flow.cq_steps(1, sniff_seconds=.2))
    assert tunes == [14_104_250, 14_105_000]
    assert flow.state == v.State.FAILED


def test_sniff_uses_fresh_busy_state_and_keeps_transient_occupancy():
    host = Host()
    flow = v.Flow(host.client, lambda hz: host.lines('BUSY OFF'))
    assert host.client.finish(flow.sniff_steps(14_104_250, .3))
    flow.tuner = lambda hz: host.lines('BUSY ON', 'BUSY OFF')
    assert not host.client.finish(flow.sniff_steps(14_104_250, .3))
    flow.tuner = lambda hz: None
    assert host.client.finish(flow.sniff_steps(14_104_250, .3))


def test_quiet_sniff_waits_full_window_after_tuning():
    host = Host()
    tuned_at = []
    def tune(dial):
        host.now += 2
        tuned_at.append(host.now)
    flow = v.Flow(host.client, tune)
    def ready(sockets, timeout):
        if tuned_at:
            assert flow.state == v.State.SNIFFING
            assert host.client.busy is None
        return host.ready(sockets, timeout)
    host.client.ready = ready
    assert host.client.finish(flow.sniff_steps(14_104_250, 5))
    assert host.now == pytest.approx(tuned_at[0] + 5)


@pytest.mark.parametrize('reports', [('BUSY ON',), ('BUSY ON', 'BUSY OFF')])
@pytest.mark.parametrize('arrival', ['early', 'late', 'during-tune'])
def test_busy_report_in_sniff_window_prevents_cq(reports, arrival):
    host, tunes, flow = flow_host()
    def tune(dial):
        tunes.append(dial)
        if arrival == 'during-tune':
            for report in reports:
                host.client._line(report)
    flow.tuner = tune
    reported = False
    def ready(sockets, timeout):
        nonlocal reported
        if flow.state == v.State.SNIFFING and not reported:
            if arrival == 'early' or (arrival == 'late' and host.now + timeout >= .6):
                host.lines(*reports)
                reported = True
        return host.ready(sockets, timeout)
    host.client.ready = ready
    assert not host.client.finish(flow.cq_steps(1, sniff_seconds=.5))
    assert tunes == [14_104_250, 14_105_000]
    assert not any(command.startswith('CQFRAME') for command in host.sent)


@pytest.mark.parametrize('stale', ['processed', 'queued', 'after-listen-off'])
def test_pre_tune_busy_report_does_not_prevent_cq(stale):
    host, tunes, flow = flow_host(answer=True)
    if stale == 'processed':
        host.client._line('BUSY ON')
    elif stale == 'queued':
        host.lines('BUSY ON')
    else:
        def listen_off():
            host.lines('OK')
            host.lines('BUSY ON')
            return []
        host.replies['LISTEN OFF'] = listen_off
    assert host.client.finish(flow.cq_steps(1, sniff_seconds=.3, wait_seconds=.5))
    assert host.sent.count('CQFRAME W9SSJ test-slot-1') == 2
    assert tunes == [14_104_250, 14_105_000, 14_104_250]


def test_old_kestrel_called_station_disconnect_refusal_is_immediate():
    host = Host(replies={'DISCONNECT': ['OK', 'WRONG DISCONNECT: called-station close is not implemented']})
    host.client.connected = True
    with pytest.raises(v.Unsupported, match='called-station'):
        host.client.disconnect()
    assert host.now < 1


def test_cq_completion_without_pending_is_not_success():
    host = Host(replies={'CQFRAME W9SSJ 500': ['OK', 'CANCELPENDING']})
    with pytest.raises(v.Unsupported):
        host.client.finish(host.client.cq_steps('CQFRAME W9SSJ 500'))


# -- the attended session ---------------------------------------------------------

CQ_DONE = ['OK', 'PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING']


class Desk:
    """A session on the fake host: its screen, its events and its keying line."""

    def __init__(self, bw=500, mailbox=None, kiss=False, loc=None, **replies):
        self.host = Host(replies={'VERSION': ['VERSION test-host'],
                                  'DISCONNECT': ['OK', 'DISCONNECTED'], **replies}, kiss=kiss)
        self.host.data.respond = lambda data: self.host.lines('BUFFER 0')
        self.host.client.initialize('W9SSJ', bw)
        self.tunes, self.said, self.events, self.armed, self.ids = [], [], [], [], []
        self.line = False
        self.flow = v.Flow(self.host.client, self.tunes.append, profile={'LOC': loc} if loc else None)
        def retune(target, generation):
            self.flow.tuner(target)
            yield from ()
        self.flow.received_retune_steps = retune
        self.flow.dial = 14_105_000
        self.mailbox = v.Mailbox(mailbox or tempfile.mkdtemp())
        self.session = v.Session(self.host.client, self.flow, say=self.said.append, mailbox=self.mailbox,
                                 emit=lambda kind, **record: self.events.append(dict(kind=kind, **record)),
                                 arm=self.armed.append, line_state=lambda: self.line,
                                 bandwidth=bw, wait_seconds=.5, identify=self.identify)
        def event(item):
            self.host.events.append(item)
            self.session.handle(item)
        self.host.client.event = event
        self.session.monitor()

    def identify(self, bandwidth, after_peer):
        self.ids.append(dict(bw=bandwidth, after_peer=after_peer, dial=self.flow.dial,
                             ptt=self.host.client.ptt, sent=len(self.sent)))

    def hear(self, *lines):
        self.host.lines(*lines)
        self.tick()

    def receive(self, text):
        self.host.data.incoming.append(v.frame_message(text))
        self.tick()

    def tick(self):
        """A real poll reads every byte waiting; the fake hands one chunk per recv."""
        self.session.tick()
        while self.host.command.incoming or self.host.data.incoming:
            self.session.tick()

    def drive(self, until=lambda: False):
        """Tick while a flow runs, or until the condition holds."""
        while self.session.steps is not None and not until():
            self.session.tick()

    def do(self, line):
        """One operator line, then the flow it started runs to its end."""
        going = self.session.command(line)
        self.drive()
        return going

    @property
    def legacy_said(self):
        # Older transcript assertions retain their exact pre-status lines.
        # New state/data notices have dedicated actual-loop snapshot tests.
        return [line for line in self.said if not line.startswith(('Operation: ', 'Data: '))]

    @property
    def sent(self):
        return self.host.sent

    @property
    def texts(self):
        return [blob.decode() for blob in self.host.data.sent]

    def kinds(self, kind):
        return [record for record in self.events if record['kind'] == kind]


def test_answer_a_heard_cq_from_monitoring_to_monitoring(monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(**{'CONNECT W9SSJ N0XYZ': ['OK', 'CONNECTED W9SSJ N0XYZ 500']})
    assert desk.sent[-1] == 'LISTEN CQ' and desk.armed == [True]
    desk.hear('SN -4.1', 'CQFRAME K4ABC-3 500')
    desk.hear('SN 6.0', 'CQFRAME N0XYZ-12 500')
    assert desk.do('/answer 2')
    desk.receive('de N0XYZ <R+03> <S:VE>\r\nGood evening')
    assert desk.do('hello from Chicago')
    assert desk.do('/bye')
    desk.tick()
    assert [line for line in desk.said if not line.startswith(('Operation: ', 'Data: '))] == [
        'monitoring 14105.000 kHz at 500 Hz, auto-answer on',
        '  1  K4ABC    slot 3   14102.750 kHz   500 Hz  SN  -4.1  2023-11-14 22:13:20Z',
        '  2  N0XYZ    slot 12  14106.500 kHz   500 Hz  SN  +6.0  2023-11-14 22:13:20Z',
        'answering N0XYZ on slot 12 (14106.500 kHz) at 500 Hz',
        'connecting N0XYZ on 14106.500 kHz at 500 Hz',
        'connected N0XYZ; Enter sends, /bye ends',
        'N0XYZ: de N0XYZ <R+03> <S:VE>',
        '       Good evening',
        'N0XYZ advertises: VMail, email (availability not yet checked).',
        'N0XYZ disconnected; keying line reads LOW',
        'monitoring 14105.000 kHz at 500 Hz, auto-answer on',
    ]
    assert desk.tunes == [14_106_500, 14_105_000]
    assert desk.texts == ['15 de W9SSJ <R+06>', '19 hello from Chicago\n', '8 73<DISC>']
    assert desk.sent[desk.sent.index('CONNECT W9SSJ N0XYZ') - 1] == 'LISTEN OFF'
    assert desk.sent[-3:] == ['DISCONNECT', 'LISTEN OFF', 'LISTEN CQ']
    assert [line for line in desk.sent if line.startswith('BW')] == ['BW500']
    assert desk.armed == [True, True, True]
    opened, contact = desk.kinds('contact')
    assert opened['ended_utc'] is None and opened['started_utc'] == contact['started_utc']
    assert contact['peer'] == 'N0XYZ' and contact['how'] == 'answer' and contact['n'] == 1
    assert contact['dial_hz'] == 14_106_500 and contact['line'] == 'keying line reads LOW'
    assert contact['ended_utc'] is not None
    assert [record['kind'] for record in desk.events] == [
        'state', 'heard', 'heard', 'state', 'link', 'contact', 'state', 'tx', 'rx', 'tag', 'tag', 'tx', 'tx',
        'link', 'contact', 'state', 'state']
    assert desk.kinds('heard')[1] == dict(kind='heard', n=2, call='N0XYZ', slot=12, beacon=False, band_m=20,
                                          dial_hz=14_106_500, bandwidth_hz=500, sn=6.0,
                                          rf_source="cq", rx_frequency_hz=14_106_500,
                                          received_unix=1700000000)
    assert [record['state'] for record in desk.kinds('state')] == [
        'monitoring', 'connecting', 'connected', 'returned-to-calling-frequency', 'monitoring']


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_answer_follows_the_cqs_bandwidth_and_returns_to_the_sessions(bw, monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', f'CONNECTED W9SSJ K7QB {bw}']})
    desk.hear('SN 11.0', f'CQFRAME K7QB-2 {bw}')
    assert desk.legacy_said[-1] == f'  1  K7QB     slot 2   14103.500 kHz  {bw:>4} Hz  SN +11.0  2023-11-14 22:13:20Z'
    desk.do('/answer 1')
    assert f'answering K7QB on slot 2 (14103.500 kHz) at {bw} Hz' in desk.said
    widths = [line for line in desk.sent if line.startswith('BW')]
    assert widths == ['BW500'] + ([f'BW{bw}'] if bw != 500 else [])
    if bw != 500:
        assert desk.sent.index(f'BW{bw}') < desk.sent.index('CONNECT W9SSJ K7QB')
    assert desk.host.client.bandwidth == bw
    desk.do('/bye')
    desk.tick()
    assert desk.host.client.bandwidth == 500
    assert [line for line in desk.sent if line.startswith('BW')] == widths + (['BW500'] if bw != 500 else [])
    assert desk.sent[-1] == 'LISTEN CQ' and desk.tunes == [14_103_500, 14_105_000]


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_cq_at_the_sessions_bandwidth_then_chat_then_monitoring(bw):
    linked = []
    desk = Desk(bw, **{f'CQFRAME W9SSJ-1 {bw}': CQ_DONE,
                       'LISTEN CQ': lambda: ['OK', f'CONNECTED KK6SVV W9SSJ {bw}'] if linked else ['OK']})
    linked.append(True)
    desk.host.lines('SN 9.4')
    desk.do('/cq 1')
    assert desk.sent.count(f'CQFRAME W9SSJ-1 {bw}') == 2
    assert desk.tunes == [14_104_250, 14_105_000, 14_104_250]
    assert [line for line in desk.said[1:] if not line.startswith(('Operation: ', 'Data: '))] == [
        f'sniffing slot 1 (14104.250 kHz) for {v.SNIFF_S} s',
        f'calling CQ 1/2 on 14105.000 kHz, slot 1 at {bw} Hz',
        f'calling CQ 2/2 on 14105.000 kHz, slot 1 at {bw} Hz',
        f'waiting on slot 1 (14104.250 kHz) at {bw} Hz for 0.5 s',
        '\aconnected KK6SVV; Enter sends, /bye ends']
    assert desk.texts == ['15 de W9SSJ <R+09>']
    opened, = desk.kinds('contact')
    assert opened['ended_utc'] is None and desk.session.contact['how'] == 'cq'
    linked.clear()
    desk.hear('DISCONNECTED')
    assert desk.legacy_said[-2:] == ['KK6SVV disconnected; keying line reads LOW',
                              f'monitoring 14105.000 kHz at {bw} Hz, auto-answer on']
    assert desk.tunes[-1] == 14_105_000 and desk.kinds('contact')[-1]['how'] == 'cq'


def test_unanswered_cq_returns_to_monitoring_and_says_so():
    desk = Desk(**{'CQFRAME W9SSJ-14 500': CQ_DONE})
    desk.do('/cq 14')
    assert desk.legacy_said[-2:] == ['no answer on slot 14', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.tunes == [14_108_000, 14_105_000, 14_108_000, 14_105_000]
    assert desk.sent[-1] == 'LISTEN CQ' and desk.session.contact is None


def test_busy_slot_sends_no_cq_and_picks_the_next_free_slot_by_default():
    desk = Desk()
    desk.hear('SN 1.0', 'CQFRAME K7QB-1 500')
    desk.flow.occupied = lambda: True
    desk.do('/cq')
    assert 'slot 2: no CQ heard on it in the last 15 min' in desk.said
    assert desk.legacy_said[-2:] == ['slot 2 is busy; no CQ sent', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert not any(line.startswith('CQFRAME') for line in desk.sent)


@pytest.mark.parametrize('mine,peer', [('W9SSJ', 'W9SSJ-1'), ('W9SSJ-1', 'W9SSJ'),
                                       ('X-W9SSJ', 'W9SSJ')])
@pytest.mark.parametrize('incoming', [False, True])
def test_connected_peer_keeps_its_full_call(mine, peer, incoming):
    desk = Desk()
    desk.host.client.mycall = mine
    caller, called = (peer, mine) if incoming else (mine, peer)
    desk.hear(f'CONNECTED {caller} {called} 500')
    assert desk.session.peer == peer


def test_call_uses_the_peers_heard_bandwidth_or_the_sessions():
    desk = Desk(**{'CONNECT W9SSJ WE3MAY': ['OK', 'CONNECTED W9SSJ WE3MAY 2300'],
                   'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.hear('SN 2.0', 'CQFRAME WE3MAY-12 2300')
    desk.do('/call we3may 12')
    assert desk.sent[-3:] == ['BW2300', 'LISTEN OFF', 'CONNECT W9SSJ WE3MAY']
    assert desk.tunes == [14_106_500] and desk.texts == ['15 de W9SSJ <R+02>']
    desk.hear('DISCONNECTED')
    desk.do('/call K7QB cf')
    assert desk.sent[-2:] == ['LISTEN OFF', 'CONNECT W9SSJ K7QB'] and 'BW2300' not in desk.sent[-6:]
    assert desk.tunes == [14_106_500, 14_105_000]
    assert desk.texts == ['15 de W9SSJ <R+02>']
    assert desk.legacy_said[-1] == 'connected K7QB; Enter sends, /bye ends'


def test_an_incoming_connect_is_a_contact_on_the_calling_frequency():
    desk = Desk()
    desk.hear('SN 7.7', 'CONNECTED KQ4QWH W9SSJ 500')
    assert desk.legacy_said[-1] == '\aconnected KQ4QWH; Enter sends, /bye ends'
    assert desk.texts == ['3 <Q>'] and desk.tunes == []
    desk.line = True
    desk.hear('DISCONNECTED')
    assert desk.legacy_said[-1] == 'KQ4QWH disconnected; keying line reads HIGH; check the transmitter'
    assert desk.flow.state is v.State.CONNECTED, 'physical HIGH must defer return to monitoring'
    _, contact = desk.kinds('contact')
    assert contact['how'] == 'incoming' and contact['dial_hz'] == 14_105_000
    assert contact['line'] == 'keying line reads HIGH; check the transmitter'
    desk.line = False
    desk.tick()
    assert desk.flow.state is v.State.MONITORING


def test_auto_off_stops_answering_but_keeps_the_heard_list():
    desk = Desk()
    desk.do('/auto off')
    assert desk.sent[-1] == 'LISTEN OFF' and desk.armed == [True, False]
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer off (/auto on to answer)'
    desk.hear('SN 3.0', 'CQFRAME K7QB-2 500')
    assert len(desk.session.heard) == 1
    desk.do('/auto on')
    assert desk.sent[-1] == 'LISTEN CQ' and desk.armed == [True, False, True]


def test_beacons_and_suffixless_cqs_are_listed_and_answered_on_the_calling_frequency(monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(**{'CONNECT W9SSJ AE5JO': ['OK', 'CONNECTED W9SSJ AE5JO 2750']})
    desk.hear('SN 2.0', 'CQFRAME AE5JO-9 2750')
    desk.hear('SN 0.5', 'CQFRAME N8IK 2300')
    assert desk.legacy_said[-2:] == ['  1  AE5JO    beacon   14105.000 kHz  2750 Hz  SN  +2.0  2023-11-14 22:13:20Z',
                              '  2  N8IK     no slot  14105.000 kHz  2300 Hz  SN  +0.5  2023-11-14 22:13:20Z']
    desk.session.command('/answer 1')
    desk.host.data.incoming.append(v.frame_message('<AWQ>'))
    desk.drive(until=lambda: '25 <QSYF>000014104250</QSYF>' in desk.texts)
    assert 'answering AE5JO on the calling frequency at 2750 Hz' in desk.said
    assert desk.tunes == [] and desk.texts == ['25 <QSYF>000014104250</QSYF>']
    assert desk.sent[-3:] == ['BW2750', 'LISTEN OFF', 'CONNECT W9SSJ AE5JO']


def test_heard_list_keeps_numbers_and_ages_and_warns_on_a_stale_answer(monkeypatch):
    monkeypatch.setattr(v.time, 'time', lambda: 1792972799)
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.hear('SN 1.0', 'CQFRAME K7QB-2 500')
    assert desk.legacy_said[-1].endswith('2026-10-25 23:59:59Z')
    desk.host.now += 95
    monkeypatch.setattr(v.time, 'time', lambda: 1792972894)
    desk.hear('CQFRAME WX2DX-4 500')
    assert desk.legacy_said[-1].endswith('2026-10-26 00:01:34Z')
    assert desk.kinds('heard')[-1]['received_unix'] == 1792972894
    desk.do('/heard')
    assert desk.legacy_said[-2:] == ['  1  K7QB     slot 2   14103.500 kHz   500 Hz  SN  +1.0  2 min',
                              '  2  WX2DX    slot 4   14102.000 kHz   500 Hz  SN  +1.0  now']
    desk.host.now += v.HEARD_STALE_S
    desk.do('/answer 1')
    assert desk.legacy_said[-4] == 'K7QB called 17 min ago and has likely returned to the calling frequency'
    assert desk.legacy_said[-1] == 'connected K7QB; Enter sends, /bye ends'


def test_errors_are_said_and_a_failed_call_returns_to_monitoring():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'DISCONNECTED']})
    desk.do('/answer 1')
    desk.do('/bogus')
    desk.do('/heard')
    assert desk.legacy_said[1:] == ['error: nothing heard yet',
                             'commands: ' + ', '.join(v.Session.COMMANDS) + '; anything else is sent when linked',
                             'nothing heard yet']
    desk.do('/call K7QB')
    desk.tick()
    assert desk.legacy_said[-2:] == ['error: Connection attempt ended without a link',
                              'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.kinds('error') == [dict(kind='error', text='nothing heard yet'),
                                   dict(kind='error', text='Connection attempt ended without a link')]
    assert [record['state'] for record in desk.kinds('state')][-3:] == ['connecting', 'failed', 'monitoring']


def test_a_tune_failure_parks_the_session_until_auto_on():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.hear('SN 1.0', 'CQFRAME K7QB-2 500')
    desk.do('/answer 1')
    def refuse(hz):
        raise v.HostError('QSY did not take')
    desk.flow.tuner = refuse
    desk.hear('DISCONNECTED')
    assert desk.legacy_said[-1] == 'error: QSY did not take; not monitoring. /auto on tries again'
    desk.tick()
    assert desk.legacy_said[-1] == 'error: QSY did not take; not monitoring. /auto on tries again'
    desk.flow.tuner = desk.tunes.append
    desk.do('/auto on')
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on'


def test_a_link_up_refuses_a_second_call_and_quit_closes_without_more_text():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.do('/call K7QB 2')
    desk.do('/cq 1')
    assert desk.legacy_said[-1] == 'error: linked with K7QB; /bye first'
    assert desk.do('/tag NAME Saul') and desk.texts[-1] == '11 <NAME:Saul>'
    assert not desk.do('/quit')
    desk.session.quit()
    assert desk.texts[-1] == '11 <NAME:Saul>'
    assert not desk.host.client.connected


def test_free_slot_skips_slots_heard_recently():
    desk = Desk()
    for n, slot in enumerate((1, 2, 3, 4, 5, 11), 1):
        desk.hear(f'CQFRAME K{n}AB-{slot} 500')
    assert desk.session.free_slot() == 12
    desk.host.now += v.HEARD_STALE_S + 1
    assert desk.session.free_slot() == 1


@pytest.mark.parametrize('seconds,said', [(0, 'now'), (42, '42 s'), (150, '2 min'), (4000, '1 h 6 min')])
def test_ages_read_plainly(seconds, said):
    assert v.age(seconds) == said


# -- the terminal stays live through a flow ------------------------------------------

FRAME = 'CQFRAME W9SSJ-1 500'


def keyed_frame():
    """A CQ frame whose end the test delivers itself, with `desk.hear('PTT OFF', 'CANCELPENDING')`."""
    return ['OK', 'PENDING', 'PTT ON']


def test_enter_during_the_sniff_cancels_before_any_cq():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1')
    assert desk.flow.state == v.State.SNIFFING and desk.session.steps is not None
    assert desk.session.command('')
    desk.drive()
    assert desk.legacy_said[1:] == [f'sniffing slot 1 (14104.250 kHz) for {v.SNIFF_S} s', 'stopping',
                             'CQ on slot 1 stopped', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert FRAME not in desk.sent and desk.tunes == [14_104_250, 14_105_000]
    assert desk.sent[-1] == 'LISTEN CQ' and desk.session.steps is None


def test_stop_during_cq_2_of_2_lets_the_frame_end_then_returns():
    desk = Desk(**{FRAME: keyed_frame})
    desk.session.command('/cq 1')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 1)
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 2)
    assert desk.host.client.ptt and desk.legacy_said[-1] == 'calling CQ 2/2 on 14105.000 kHz, slot 1 at 500 Hz'
    desk.session.command('/stop')
    assert desk.legacy_said[-1] == 'stopping after this frame'
    for _ in range(20):
        desk.session.tick()
    assert desk.host.client.ptt and desk.session.steps is not None, 'a keyed frame is never cut'
    assert desk.legacy_said[-1] == 'stopping after this frame' and desk.tunes == [14_104_250, 14_105_000]
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive()
    assert desk.legacy_said[-2:] == ['CQ on slot 1 stopped', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.tunes == [14_104_250, 14_105_000, 14_105_000], 'never moved to the slot'
    assert desk.sent.count('LISTEN CQ') == 2 and desk.sent[-1] == 'LISTEN CQ'


def test_a_cq_heard_during_the_slot_wait_is_listed_and_stop_returns(monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1')
    desk.drive(until=lambda: desk.flow.state == v.State.WAITING)
    assert desk.legacy_said[-1] == 'waiting on slot 1 (14104.250 kHz) at 500 Hz for 0.5 s'
    desk.hear('SN 2.0', 'CQFRAME K7QB-2 500')
    assert desk.legacy_said[-1] == '  1  K7QB     slot 2   14103.500 kHz   500 Hz  SN  +2.0  2023-11-14 22:13:20Z'
    assert desk.flow.state == v.State.WAITING and desk.session.steps is not None
    desk.session.command('/stop')
    desk.drive()
    assert desk.legacy_said[-3:] == ['stopping', 'CQ on slot 1 stopped', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.tunes[-1] == 14_105_000 and desk.host.now < 1 + v.SNIFF_S + 2 * v.CQ_PERIOD_S


def test_an_incoming_connect_during_the_slot_wait_is_taken():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1')
    desk.drive(until=lambda: desk.flow.state == v.State.WAITING)
    desk.hear('SN 4.0', 'CONNECTED WX2DX W9SSJ 500')
    desk.drive()
    assert desk.legacy_said[-1] == '\aconnected WX2DX; Enter sends, /bye ends'
    assert desk.texts == ['15 de W9SSJ <R+04>'] and desk.session.contact['how'] == 'cq'
    assert desk.do('hello') and desk.texts[-1] == '6 hello\n'


def test_quit_during_a_cq_aborts_without_another_frame_or_retune():
    desk = Desk(**{FRAME: keyed_frame})
    desk.session.command('/cq 1')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 1)
    tunes = desk.tunes[:]
    assert not desk.session.command('/quit')
    desk.host.lines('PTT OFF', 'CANCELPENDING')
    desk.session.quit()
    assert desk.session.steps is None and desk.tunes[-1] == 14_105_000
    assert desk.sent.count(FRAME) == 1 and desk.tunes == tunes
    assert desk.sent[-5:] == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']


def test_stop_during_a_connect_attempt_aborts_and_returns():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'PTT ON']})
    desk.hear('SN 1.0', 'CQFRAME K7QB-2 500')
    desk.session.command('/answer 1')
    desk.drive(until=lambda: desk.flow.state == v.State.CONNECTING)
    desk.hear('SN 3.0', 'CQFRAME WX2DX-4 500')
    assert desk.legacy_said[-1].startswith('  2  WX2DX')
    desk.session.command('/stop')
    assert desk.legacy_said[-1] == 'stopping after this frame'
    desk.drive(until=lambda: 'ABORT' in desk.sent)
    assert desk.session.steps is not None and desk.host.client.ptt
    desk.hear('PTT OFF')
    desk.drive()
    assert desk.legacy_said[-2:] == ['call to K7QB stopped', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.tunes == [14_103_500, 14_105_000] and desk.session.contact is None


def test_a_second_flow_waits_for_stop():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1')
    desk.session.command('/call K7QB')
    assert desk.legacy_said[-1] == 'error: cq is in progress; /stop first'
    assert desk.flow.state == v.State.SNIFFING
    desk.drive()
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on'


def test_a_cq_cancelled_in_the_sniff_returns_without_calling():
    host, tunes, flow = flow_host()
    steps = flow.cq_steps(1, sniff_seconds=.3, wait_seconds=.5)
    next(steps)
    flow.cancel()
    assert host.client.finish(steps) is False and flow.state == v.State.RETURNED
    assert tunes == [14_104_250, 14_105_000] and not any(s.startswith('CQFRAME') for s in host.sent)


# -- a live contact survives what goes wrong around it -------------------------------

def answered(**kwargs):
    desk = Desk(**{'CONNECT W9SSJ N0XYZ': ['OK', 'CONNECTED W9SSJ N0XYZ 500']}, **kwargs)
    desk.hear('SN 6.0', 'CQFRAME N0XYZ-12 500')
    desk.do('/answer 1')
    return desk


@pytest.mark.parametrize('request_,said', [
    ('<Q:-99999>', 'error: QSY must select one of the ten 750 Hz slots'),
    ('<Q:5>', 'error: QSY must select one of the ten 750 Hz slots')])
def test_a_peer_qsy_that_cannot_run_is_reported_and_the_contact_stays(request_, said):
    desk = answered()
    desk.receive(request_)
    assert desk.legacy_said[-1] == said and desk.kinds('error')[-1]['text'] == said[len('error: '):]
    assert desk.session.contact is not None and desk.host.client.connected
    assert desk.do('still here') and desk.texts[-1] == '11 still here\n'
    desk.hear('DISCONNECTED')
    assert desk.kinds('contact')[-1]['ended_utc'] is not None
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on'


def test_a_peer_qsy_that_times_out_is_reported_and_the_contact_stays():
    desk = Desk()
    desk.hear('SN 7.7', 'CONNECTED KQ4QWH W9SSJ 500')
    desk.receive('<Q:75>')
    desk.drive()
    assert desk.legacy_said[-1] == 'error: peer application response timed out'
    assert desk.session.contact is not None and desk.tunes == [14_105_750]


def test_a_contact_is_recorded_when_it_opens_and_when_quit_ends_it():
    desk = answered()
    opened, = desk.kinds('contact')
    assert opened['peer'] == 'N0XYZ' and opened['ended_utc'] is None
    assert desk.session.command('/quit') is False
    desk.session.quit()
    assert 'DISCONNECT' in desk.sent and desk.session.contact is None
    _, closed = desk.kinds('contact')
    assert closed['n'] == opened['n'] and closed['ended_utc'] is not None
    assert desk.kinds('link')[-1] == dict(kind='link', event='disconnected', peer='N0XYZ')


def test_a_link_lost_with_the_key_down_ends_now_and_returns_when_the_key_lifts():
    desk = answered()
    desk.hear('PTT ON', 'DISCONNECTED')
    assert desk.session.contact is None and desk.kinds('contact')[-1]['ended_utc'] is not None
    assert desk.tunes == [14_106_500], 'no QSY with the key down'
    desk.hear('PTT OFF')
    assert desk.flow.state == v.State.MONITORING and desk.sent[-1] == 'LISTEN CQ'
    assert desk.tunes == [14_106_500, 14_105_000]
    assert not any('not monitoring' in line for line in desk.said)


def test_stop_during_connected_opening_closes_without_sending_greeting():
    desk = Desk(**{'CONNECT W9SSJ N0XYZ': ['OK', 'CONNECTED W9SSJ N0XYZ 500']})
    desk.hear('CQFRAME N0XYZ-12 500')
    desk.session.command('/answer 1')
    desk.drive(until=lambda: desk.host.client.connected)
    assert desk.session.steps is not None
    desk.session.command('/stop')
    assert any('Stopping link' in line for line in desk.said)
    desk.hear('SN 4.0')
    desk.drive()
    assert desk.texts == []
    assert not desk.host.client.connected


# -- /cq repeat ----------------------------------------------------------------------

GAP = 'no answer on slot 1; monitoring 14105.000 kHz for 0.5 s, then CQ'


def test_cq_repeat_calls_again_after_each_gap_and_lists_what_it_hears_there():
    desk = Desk(**{FRAME: CQ_DONE})
    assert desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.said[-1].startswith(f'{GAP} 2 ('))
    gap_starts = desk.host.now
    desk.hear('SN 2.0', 'CQFRAME K7QB-2 500')
    assert desk.legacy_said[-1].startswith('  1  K7QB     slot 2')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 6)
    assert desk.host.now - gap_starts >= desk.session.wait_seconds
    assert [line for line in desk.said if 'repeating' in line] == [
        f'CQ {n} on slot 1 (repeating; /stop ends)' for n in (1, 2, 3)]
    assert [sum(line.startswith(f'{GAP} {n} (') for line in desk.said) for n in (2, 3)] == [1, 1]
    assert desk.tunes[:4] == [14_104_250, 14_105_000, 14_104_250, 14_105_000]
    assert desk.sent.count('LISTEN CQ') >= 3 and desk.session.steps is not None


def test_cq_repeat_listens_on_the_calling_frequency_as_long_as_on_the_slot():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.wait_seconds = v.SLOT_WAIT_S
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: 'then CQ 2' in desk.said[-1])
    gap_starts = desk.host.now
    assert desk.legacy_said[-1] == ('no answer on slot 1; monitoring 14105.000 kHz for 120 s, then CQ 2 '
                             '(listening on the calling frequency 47% of each cycle)')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 3)
    assert desk.host.now - gap_starts >= v.SLOT_WAIT_S


def test_cq_repeat_stops_during_the_gap_without_another_cq():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.said[-1].startswith(f'{GAP} 2 ('))
    desk.session.command('')
    desk.drive()
    assert desk.legacy_said[-3:] == ['stopping', 'repeating CQ stopped before CQ 2',
                              'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.sent.count(FRAME) == 2 and desk.tunes[-1] == 14_105_000


def test_cq_repeat_stopped_mid_frame_lets_the_frame_end_and_calls_no_more():
    desk = Desk(**{FRAME: keyed_frame})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 1)
    desk.session.command('/stop')
    assert desk.legacy_said[-1] == 'stopping after this frame'
    for _ in range(20):
        desk.session.tick()
    assert desk.host.client.ptt and desk.session.steps is not None, 'a keyed frame is never cut'
    desk.hear('PTT OFF', 'CANCELPENDING')
    desk.drive()
    assert desk.legacy_said[-2:] == ['repeating CQ stopped after CQ 1', 'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert desk.sent.count(FRAME) == 1 and desk.tunes == [14_104_250, 14_105_000, 14_105_000]


def test_a_connect_during_the_gap_ends_cq_repeat_as_a_contact():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.said[-1].startswith(f'{GAP} 2 ('))
    desk.hear('SN 4.0', 'CONNECTED WX2DX W9SSJ 500')
    desk.drive()
    assert desk.legacy_said[-1] == '\aconnected WX2DX; Enter sends, /bye ends'
    assert desk.session.steps is None and desk.session.contact['how'] == 'cq'
    assert desk.texts == ['3 <Q>'] and desk.sent.count(FRAME) == 2


def test_a_connect_on_the_slot_ends_cq_repeat():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 4 and desk.flow.state == v.State.WAITING)
    desk.hear('SN 4.0', 'CONNECTED WX2DX W9SSJ 500')
    desk.drive()
    assert desk.texts == ['15 de W9SSJ <R+04>'] and desk.session.contact['how'] == 'cq'
    assert desk.sent.count(FRAME) == 4 and desk.flow.state == v.State.CONNECTED


def test_cq_repeat_moves_off_a_busy_slot_to_the_next_quiet_one():
    slot2 = 'CQFRAME W9SSJ-2 500'
    desk = Desk(**{FRAME: CQ_DONE, slot2: CQ_DONE})
    desk.flow.occupied = lambda: desk.flow.dial == 14_104_250
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.sent.count(slot2) == 2)
    assert desk.legacy_said[1:5] == ['CQ 1 on slot 1 (repeating; /stop ends)',
                              f'sniffing slot 1 (14104.250 kHz) for {v.SNIFF_S} s',
                              'slot 1 is busy; moving to slot 2',
                              'CQ 1 on slot 2 (repeating; /stop ends)']
    assert FRAME not in desk.sent and desk.session.slot == 2


def test_answer_during_cq_repeat_ends_the_loop_and_answers():
    desk = Desk(**{FRAME: CQ_DONE, 'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.said[-1].startswith(f'{GAP} 2 ('))
    desk.hear('SN 2.0', 'CQFRAME K7QB-2 500')
    desk.session.command('/answer 1')
    assert 'repeating CQ stopped before CQ 2' in desk.said and not desk.session.repeat
    desk.drive(until=lambda: desk.flow.state == v.State.CONNECTED)
    assert desk.session.contact['how'] == 'answer' and desk.sent.count(FRAME) == 2


def test_free_slot_searches_onward_from_a_busy_slot():
    desk = Desk()
    assert desk.session.free_slot(after=1) == 2
    assert desk.session.free_slot(after=15) == 1
    desk.hear('SN 1.0', 'CQFRAME K7QB-2 500')
    assert desk.session.free_slot(after=1) == 3


def test_a_hyphenated_call_initializes_and_calls_cq_with_its_slot_suffix():
    from hfmodem.contact.varac import DEFAULT_CODEC, base_call, callsign
    assert callsign('x-w9ssj') == 'X-W9SSJ'
    assert DEFAULT_CODEC.cq_command('X-W9SSJ', 1) == 'CQFRAME X-W9SSJ-1 500'
    assert DEFAULT_CODEC.decode('X-W9SSJ-3') == ('X-W9SSJ', 3)
    assert base_call('X-W9SSJ-12') == 'X-W9SSJ'


# -- incoming attempts -----------------------------------------------------------------

def test_an_answered_caller_whose_link_never_came_up_is_shown_and_logged():
    # 20261023T042123057106Z, 11:13Z on 40 m: two requests answered, both setups unread.
    rows = [json.loads(line) for line in require_input(
        SESSIONS / '20261023T042123057106Z' / 'host.jsonl').read_text().splitlines()]
    window = [r for r in rows if '2026-10-23T11:13:10' <= r['utc'] < '2026-10-23T11:14:28']
    desk = Desk()
    start = utc(window[0])
    for r in window:
        desk.host.now = utc(r) - start
        desk.hear(r['raw'])
    assert desk.legacy_said[-2:] == ['\aa station called you; link setup not read'] * 2
    assert [(e['caller'], e['called'], e['bandwidth_hz'], e['seconds'])
            for e in desk.kinds('attempt')] == [(None, 'W9SSJ', 500, 30.0), (None, 'W9SSJ', 500, 30.1)]


def test_an_attempt_names_the_caller_the_host_read():
    desk = Desk()
    desk.session.caller = lambda: 'KA8CPF'
    desk.hear('PENDING', 'PTT ON', 'PTT OFF', 'CANCELPENDING')
    assert desk.legacy_said[-1] == '\aKA8CPF called you; link setup not read'
    assert desk.kinds('attempt')[-1]['caller'] == 'KA8CPF'


def test_our_own_cq_and_a_completed_link_are_not_attempts():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.do('/cq 1')
    desk.hear('PENDING', 'PTT ON', 'PTT OFF', 'SN 4.0', 'CONNECTED WX2DX W9SSJ 500')
    assert not desk.kinds('attempt')
    assert not any('called you' in line for line in desk.said)


BEACON = 'CQFRAME W9SSJ-9 500'


def wait(desk, seconds):
    end = desk.host.now + seconds
    while desk.host.now < end:
        desk.tick()
    desk.drive()


def test_a_beacon_goes_out_once_clear_and_again_each_interval_after_the_last():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    assert desk.legacy_said[-1] == (f'beacon every 15 min on 14105.000 kHz, each once it has been clear '
                             f'{v.BEACON_CLEAR_S} s while monitoring')
    wait(desk, v.BEACON_CLEAR_S - .5)
    assert BEACON not in desk.sent
    wait(desk, 1)
    assert desk.sent[-3:] == ['LISTEN OFF', BEACON, 'LISTEN CQ']
    assert desk.legacy_said[-1] == 'beacon sent; next in 15 min' and desk.armed[-2:] == [True, True]
    first = desk.host.now
    wait(desk, 15 * 60 - 1)
    assert desk.sent.count(BEACON) == 1
    wait(desk, 2)
    assert desk.sent.count(BEACON) == 2 and desk.host.now - first >= 15 * 60
    assert len(desk.kinds('beacon')) == 2
    desk.do('/beacon 30')
    assert desk.sent.count(BEACON) == 2
    wait(desk, 30 * 60 + 1)
    assert desk.legacy_said[-1] == 'beacon sent; next in 30 min'
    desk.do('/beacon 5')
    assert desk.session.beacon == 5


def test_a_beacon_waits_for_a_clear_calling_frequency_and_resends_a_busy_drop():
    replies = iter([['OK', 'BUSY OFF'], CQ_DONE])
    desk = Desk(**{BEACON: lambda: next(replies)})
    desk.hear('BUSY ON')
    desk.do('/beacon')
    wait(desk, 60)
    assert BEACON not in desk.sent
    desk.hear('BUSY OFF')
    wait(desk, 3)
    desk.hear('BUSY ON', 'BUSY OFF')
    wait(desk, v.BEACON_CLEAR_S - 1)
    assert BEACON not in desk.sent
    wait(desk, v.BEACON_CLEAR_S + 2)
    assert desk.sent.count(BEACON) == 2 and desk.legacy_said[-1] == 'beacon sent; next in 15 min'
    assert desk.legacy_said.count('beacon held: channel busy') == 1


def test_no_beacon_while_linked_or_calling_and_it_resumes_after():
    desk = Desk(**{BEACON: CQ_DONE, 'CQFRAME W9SSJ-1 500': CQ_DONE})
    desk.do('/beacon')
    desk.hear('CONNECTED KQ4QWH W9SSJ 500')
    wait(desk, 60)
    assert BEACON not in desk.sent
    desk.hear('DISCONNECTED')
    desk.session.wait_seconds = 30
    desk.do('/cq 1')
    assert BEACON not in desk.sent
    wait(desk, v.BEACON_CLEAR_S + 1)
    assert desk.sent[-2:] == [BEACON, 'LISTEN CQ']


def test_beacon_off_sends_no_more():
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    wait(desk, v.BEACON_CLEAR_S + 1)
    desk.do('/beacon off')
    assert desk.legacy_said[-1] == 'beacon off'
    wait(desk, 60 * 60)
    assert desk.sent.count(BEACON) == 1
    desk.do('/auto off')
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer off (/auto on to answer)'


def test_a_connect_after_a_beacon_is_taken_on_the_calling_frequency():
    # AE5JO-9's beacon drew KG5WM's connect on the CF 55 s later (P1 R07).
    desk = Desk(**{BEACON: CQ_DONE})
    desk.do('/beacon')
    desk.do('/auto on')
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on, beacon every 15 min'
    wait(desk, v.BEACON_CLEAR_S + 55)
    assert desk.sent.count(BEACON) == 1 and desk.sent[-1] == 'LISTEN CQ'
    desk.hear('CONNECTED KG5WM W9SSJ 500')
    assert desk.legacy_said[-1] == '\aconnected KG5WM; Enter sends, /bye ends'
    assert desk.texts == ['3 <Q>'] and desk.tunes == []
    assert desk.session.contact['how'] == 'incoming'


# -- away and VMail ------------------------------------------------------------------

def test_away_greets_a_caller_keeps_receiving_until_absolute_deadline(tmp_path):
    desk = Desk(mailbox=tmp_path)
    assert desk.do('/away Gone to the shed')
    assert desk.legacy_said[-1] == ('away: a caller gets <AWAY> and "Gone to the shed", may leave VMails, and is '
                             'closed at 5 min from connection; /back returns')
    desk.hear('SN -5.0', 'CONNECTED KB6CIO W9SSJ 500')
    assert desk.texts == ['5 <AWQ>'] and desk.tunes == []
    desk.receive('de KB6CIO <R-05>')
    assert desk.texts[1:] == ['24 <AWAY>\r\nGone to the shed'] and desk.tunes == []
    assert desk.legacy_said[-1] == 'away greeting sent to KB6CIO'
    vmail_arrives(desk)
    desk.drive()
    body = VMAIL_021[VMAIL_021.index('<MSG:') + 5:-1]
    assert desk.texts[2:] == ['5 <SMR>']
    assert desk.host.client.connected
    desk.host.now = desk.session._linked_at + v.AWAY_LINK_S
    desk.tick()
    desk.drive()
    assert desk.texts[2:] == ['5 <SMR>', '8 73<DISC>']
    assert desk.sent[-2:] == ['DISCONNECT', 'LISTEN CQ'] and desk.tunes == []
    assert desk.legacy_said[-3:] == [
        'away link at 5 min; closing',
        'KB6CIO disconnected; keying line reads LOW',
        'monitoring 14105.000 kHz at 500 Hz, auto-answer on']
    assert not any(line.startswith('error') for line in desk.said)
    assert desk.kinds('vmail') == [dict(kind='vmail', direction='in', peer='KB6CIO',
                                        id=desk.mailbox.entries()[0]['id'], subject='RE: Exploring New Paths - 2')]
    assert desk.do('/back') and desk.legacy_said[-1] == 'back; 1 new VMail'
    assert desk.do('/vmail')
    assert desk.legacy_said[-1].startswith('  1  2') and desk.legacy_said[-1].endswith('Z  from KB6CIO    RE: Exploring New Paths - 2')
    assert desk.do('/vmail 1')
    shown = len(body.replace('\r\n', '\n').split('\n'))
    assert desk.legacy_said[-shown - 2].startswith('VMail 1 from KB6CIO to W9SSJ, 2')
    assert desk.legacy_said[-shown - 2].endswith(', 14105.000 kHz')
    assert desk.legacy_said[-shown - 1] == 'subject: RE: Exploring New Paths - 2'
    assert desk.legacy_said[-shown] == '  KL7QR» Hello Ale,' and desk.legacy_said[-1] == '  '
    index, = (tmp_path / 'index.jsonl').read_text().splitlines()
    record = json.loads((tmp_path / (json.loads(index)['id'] + '.json')).read_text())
    assert record == {'id': record['id'], 'utc': record['utc'], 'direction': 'in', 'from': 'KB6CIO', 'to': 'W9SSJ',
                      'via': 'KB6CIO', 'subject': 'RE: Exploring New Paths - 2', 'body': body,
                      'dial_hz': 14_105_000, 'snr': -5.0}
    assert record['id'].endswith('-KB6CIO') and record['utc'].endswith('+00:00')


def test_a_vmail_during_a_chat_is_stored_and_acknowledged_and_the_link_stays(tmp_path):
    desk = answered(mailbox=tmp_path)
    vmail_arrives(desk)
    assert desk.texts[-1] == '5 <SMR>' and desk.host.client.connected and desk.session.steps is None
    assert desk.legacy_said[-1].startswith('VMail 1 from N0XYZ: RE: Exploring New Paths - 2')
    assert not any('<SM>' in line for line in desk.said), 'the VMail is shown as mail, not as chat'
    assert desk.do('got it') and desk.texts[-1] == '7 got it\n'
    assert desk.session.new_vmails == 0 and len(desk.mailbox.entries()) == 1


def test_vmail_send_composes_to_a_line_of_dot_and_the_peers_smr_confirms(tmp_path):
    desk = answered(mailbox=tmp_path)
    assert desk.do('/vmail send RE: Exploring New Paths - 2')
    assert desk.legacy_said[-1] == ('VMail to N0XYZ, subject "RE: Exploring New Paths - 2": type the body; a line of only '
                             '. sends it, /status and /help preserve it; other /commands drop it')
    for line in ('Hello Alex,', '', 'See you on 20 m.', '.'):
        assert desk.do(line)
    assert desk.texts[-1] == v.frame_message(
        '<SM><SBJ:RE: Exploring New Paths - 2><MSG:Hello Alex,\r\n\r\nSee you on 20 m.').decode() + '>'
    assert desk.legacy_said[-1] == 'VMail queued for N0XYZ; waiting for peer acknowledgement <SMR>'
    desk.receive('<SMR>')
    assert desk.legacy_said[-2:] == ['N0XYZ: <SMR>', 'N0XYZ acknowledged the VMail']
    desk.receive('<SMR>')
    assert desk.legacy_said[-1] == 'N0XYZ: <SMR>'
    entry, = desk.mailbox.entries()
    assert (entry['direction'], entry['from'], entry['to']) == ('out', 'W9SSJ', 'N0XYZ')
    assert desk.do('/vmail') and desk.legacy_said[-1].endswith('Z  to N0XYZ       RE: Exploring New Paths - 2')
    assert desk.kinds('vmail')[0]['direction'] == 'out'


def test_a_command_while_composing_drops_the_vmail_and_a_vmail_needs_a_link(tmp_path):
    desk = answered(mailbox=tmp_path)
    assert desk.do('/vmail send Hi') and desk.do('a line') and desk.do('/bogus')
    assert desk.legacy_said[-2] == 'VMail dropped' and desk.legacy_said[-1].startswith('commands: ')
    assert desk.session.compose is None
    assert desk.do('.') and desk.texts[-1] == '2 .\n'
    assert desk.do('/vmail send a>b') and desk.legacy_said[-1].startswith('error: A VMail needs a one-line subject')
    assert desk.do('/vmail 1') and desk.legacy_said[-1] == 'error: no VMails yet'
    assert desk.do('/bye')
    assert desk.do('/vmail send Hi')
    assert desk.legacy_said[-1] == 'error: usage: /vmail park CALL SUBJECT (or send CALL SUBJECT while unlinked)'


def test_a_long_vmail_is_refused_before_custody_or_transmission(tmp_path):
    desk = answered(mailbox=tmp_path)
    desk.do('/vmail send Long')
    desk.do('x' * v.VMAIL_MAX)
    desk.do('.')
    assert '500-character admission limit' in desk.legacy_said[-1]
    assert not any('<SM>' in text for text in desk.texts)
    assert not desk.session.parking.store.records


def test_an_away_link_without_a_vmail_closes_at_five_minutes(tmp_path):
    desk = Desk(mailbox=tmp_path)
    desk.do('/away')
    desk.flow.dial = v.slot_frequency(20, 3)
    desk.hear('SN 1.0', 'CONNECTED KQ4QWH W9SSJ 500')
    assert desk.texts == [f'{len(v.DEFAULT_AWAY) + 8} <AWAY>\r\n{v.DEFAULT_AWAY}']
    desk.receive('hello?')
    assert desk.legacy_said[-1] == 'KQ4QWH: hello?'
    desk.host.now += v.AWAY_LINK_S - 1
    desk.tick()
    assert desk.host.client.connected and len(desk.texts) == 1
    desk.host.now += 1
    desk.tick()
    desk.drive()
    assert 'away link at 5 min; closing' in desk.said and desk.texts[-1] == '8 73<DISC>'
    assert 'DISCONNECT' in desk.sent and desk.sent[-1] == 'LISTEN CQ'
    assert desk.do('/back') and desk.legacy_said[-1] == 'back; 0 new VMails'


def test_away_turns_auto_answer_on_and_an_outgoing_call_ends_it(tmp_path):
    desk = Desk(mailbox=tmp_path, **{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.do('/auto off')
    desk.do('/away')
    assert desk.sent[-1] == 'LISTEN CQ' and desk.session.auto and desk.armed[-1] is True
    assert desk.kinds('away')[-1] == dict(kind='away', on=True, text=v.DEFAULT_AWAY)
    desk.hear('SN 1.0', 'CQFRAME K7QB-2 500')
    desk.do('/answer 1')
    assert desk.session.away is None and 'back; 0 new VMails' in desk.said
    assert desk.texts == ['15 de W9SSJ <R+01>'], 'an answered CQ opens as usual, with no <AWAY>'
    assert desk.do('/away a<b>') and desk.legacy_said[-1].startswith('error: an away message is one line')


@pytest.mark.parametrize('back_first', [False, True])
def test_returning_from_away_does_not_prompt_the_callee_to_invite(back_first):
    desk = Desk(**{'CONNECT W9SSJ KG9B': ['OK', 'CONNECTED W9SSJ KG9B 500']})
    desk.do('/away')
    if back_first:
        desk.do('/back')
    desk.do('/call KG9B cf')
    assert desk.session.away is None
    assert desk.texts == [], 'the caller must not prompt a reciprocal QSY invitation'
    assert not desk.tunes


def test_a_beacon_keeps_away_on(tmp_path):
    desk = Desk(mailbox=tmp_path, **{BEACON: CQ_DONE})
    desk.do('/away')
    desk.do('/beacon')
    wait(desk, v.BEACON_CLEAR_S + 1)
    assert desk.sent.count(BEACON) == 1 and desk.session.away == v.DEFAULT_AWAY


def test_a_caller_who_hangs_up_after_the_acknowledgement_leaves_no_error(tmp_path):
    desk = Desk(mailbox=tmp_path)
    desk.do('/away')
    desk.hear('SN -5.0', 'CONNECTED KB6CIO W9SSJ 500')
    desk.host.data.respond = lambda data: desk.host.lines('DISCONNECTED' if b'<SMR>' in data else 'BUFFER 0')
    vmail_arrives(desk)
    desk.drive()
    assert desk.texts[1:] == ['5 <SMR>'] and 'DISCONNECT' not in desk.sent
    assert not any(line.startswith('error') for line in desk.said)
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on'
    assert desk.session.new_vmails == 1


# -- the advanced beacon over the KISS port -------------------------------------------

KISS_BEACON = v.kiss.encode(b'20AEN63BCW9SSJ      ', v.BEACON_KISS)


def kiss_arrives(desk, text):
    desk.host.kiss.incoming.append(v.kiss.encode(text.encode(), v.BEACON_KISS))
    desk.tick()


def test_an_advanced_beacon_is_heard_with_its_locator_and_answered_on_the_calling_frequency(monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(kiss=True, **{'CONNECT W9SSJ KG9B': ['OK', 'CONNECTED W9SSJ KG9B 500']})
    desk.hear('SN 12.3')
    kiss_arrives(desk, '20OEN54FLKG9B       ')
    assert desk.legacy_said[-1] == '  1  KG9B     beacon   14105.000 kHz   500 Hz  SN +12.3  2023-11-14 22:13:20Z  EN54FL VMail email AI BBS'
    assert desk.kinds('heard')[-1] == dict(kind='heard', n=1, call='KG9B', slot=None, beacon=True, band_m=20, dial_hz=14_105_000,
                                           bandwidth_hz=500, sn=12.3, grid='EN54FL', status='20O',
                                           rf_source='beacon', rx_frequency_hz=14_106_500,
                                           received_unix=1700000000)
    kiss_arrives(desk, 'P,=`0F')
    assert len(desk.session.heard) == 1
    desk.session.command('/answer 1')
    desk.host.data.incoming.append(v.frame_message('<AWQ>'))
    desk.drive(until=lambda: '25 <QSYF>000014104250</QSYF>' in desk.texts)
    assert 'answering KG9B on the calling frequency at 500 Hz' in desk.said
    assert desk.texts == ['25 <QSYF>000014104250</QSYF>']
    assert desk.sent[-2:] == ['LISTEN OFF', 'CONNECT W9SSJ KG9B'] and desk.tunes == []


def test_an_advanced_beacon_with_a_lowercase_subsquare_is_heard(monkeypatch):
    monkeypatch.setattr(v.time, "time", lambda: 1700000000)
    desk = Desk(kiss=True)
    desk.hear('SN 4.0')
    kiss_arrives(desk, '20OFN35flVA2OPS      ')
    heard = desk.kinds('heard')[-1]
    assert (heard['call'], heard['grid'], heard['beacon']) == ('VA2OPS', 'FN35FL', True)


def test_a_beacon_goes_out_as_the_kiss_frame_when_the_host_and_profile_allow():
    desk = Desk(kiss=True, loc='EN63bc')
    desk.do('/beacon')
    wait(desk, v.BEACON_CLEAR_S + 1)
    assert desk.host.kiss.sent == [KISS_BEACON] and BEACON not in desk.sent
    assert desk.sent[-2:] == ['LISTEN OFF', 'LISTEN CQ']
    assert desk.legacy_said[-1] == 'beacon sent; next in 15 min'
    assert desk.kinds('beacon')[-1]['text'] == '20AEN63BCW9SSJ'
    desk.do('/away')
    wait(desk, 15 * 60 + 1)
    assert desk.host.kiss.sent[-1] == v.kiss.encode(b'20AEN63BCW9SSJ      ', v.BEACON_KISS)


@pytest.mark.parametrize('kiss,loc', [(False, 'EN63bc'), (True, None), (True, 'nowhere')])
def test_without_a_kiss_port_or_a_locator_the_beacon_is_the_standard_cqframe(kiss, loc):
    desk = Desk(kiss=kiss, loc=loc, **{BEACON: CQ_DONE})
    desk.do('/beacon')
    wait(desk, v.BEACON_CLEAR_S + 1)
    assert desk.sent.count(BEACON) == 1 and desk.kinds('beacon')[-1]['text'] == 'W9SSJ-9'
    if kiss:
        assert desk.host.kiss.sent == []


def test_a_kiss_beacon_dropped_as_busy_is_sent_again_once_clear():
    desk = Desk(kiss=True, loc='EN63BC')
    keyed = iter([lambda: desk.host.lines('BUSY OFF'), lambda: desk.host.lines(*CQ_DONE[1:])])
    desk.host.kiss.respond = lambda data: next(keyed)()
    desk.hear('BUSY ON')
    desk.do('/beacon')
    wait(desk, 60)
    assert desk.host.kiss.sent == []
    desk.hear('BUSY OFF')
    wait(desk, 3)
    desk.hear('BUSY ON', 'BUSY OFF')
    wait(desk, v.BEACON_CLEAR_S - 1)
    assert desk.host.kiss.sent == []
    wait(desk, v.BEACON_CLEAR_S + 2)
    assert len(desk.host.kiss.sent) == 2 and desk.legacy_said[-1] == 'beacon sent; next in 15 min'


def test_the_client_opens_the_kiss_port_only_when_given_one():
    opened = []

    def connect(address, timeout):
        opened.append(address[1])
        return Socket()
    v.Client.open(command_port=1, data_port=2, kiss_port=3, connect=connect).close()
    assert opened == [1, 2, 3]
    client = v.Client.open(command_port=1, data_port=2, connect=connect)
    assert client.kiss_socket is None and opened[-2:] == [1, 2]
    client.close()
    with pytest.raises(v.Unsupported, match='no KISS port'):
        client.finish(client.beacon_steps('20OEN63BCW9SSJ'))


def test_a_beacon_status_letter_reads_as_the_services_it_announces():
    assert [v.services(s) for s in ('20A', '20C', '20G', '20K', '20O')] == [
        'VMail', 'VMail email', 'VMail email AI', 'VMail email BBS', 'VMail email AI BBS']


def test_the_beacon_letter_follows_the_services_we_announce():
    assert [v.status_letter(s) for s in (None, 'V', 'VE', 'VEB', 'VEAB')] == ['A', 'A', 'C', 'K', 'O']


# -- the closing CW ID -----------------------------------------------------------------

def test_a_contact_we_close_identifies_once_on_its_slot_before_returning():
    desk = answered()
    desk.hear('PTT ON', 'PTT OFF')
    assert desk.ids == []
    desk.do('/bye')
    desk.tick()
    assert desk.ids == [dict(bw=500, after_peer=False, dial=14_106_500, ptt=False,
                             sent=desk.sent.index('DISCONNECT') + 1)]
    assert desk.tunes == [14_106_500, 14_105_000] and desk.legacy_said[-1].startswith('monitoring')
    desk.tick()
    assert len(desk.ids) == 1


@pytest.mark.parametrize('bw', v.BANDWIDTHS)
def test_a_contact_the_peer_closes_identifies_after_the_peers_ident_at_its_width(bw):
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', f'CONNECTED W9SSJ K7QB {bw}']})
    desk.hear('SN 11.0', f'CQFRAME K7QB-2 {bw}')
    desk.do('/answer 1')
    desk.hear('PTT ON', 'PTT OFF', 'DISCONNECTED')
    assert [(i['bw'], i['after_peer'], i['dial']) for i in desk.ids] == [(bw, True, 14_103_500)]
    assert 'DISCONNECT' not in desk.sent and desk.host.client.bandwidth == 500


def test_a_link_lost_with_the_key_down_identifies_when_the_key_lifts():
    desk = answered()
    desk.hear('PTT ON', 'DISCONNECTED')
    assert desk.ids == []
    desk.hear('PTT OFF')
    assert [(i['after_peer'], i['ptt']) for i in desk.ids] == [(True, False)]


def test_an_unanswered_cq_identifies_after_the_cycle_and_a_busy_slot_does_not():
    desk = Desk(**{'CQFRAME W9SSJ-14 500': CQ_DONE})
    desk.do('/cq 14')
    assert desk.legacy_said[-2] == 'no answer on slot 14'
    assert [(i['bw'], i['after_peer']) for i in desk.ids] == [(500, False)]
    assert desk.sent[desk.ids[0]['sent'] - 1] == 'LISTEN OFF'
    assert desk.ids[0]['sent'] > desk.sent.index('CQFRAME W9SSJ-14 500')
    desk.flow.occupied = lambda: True
    desk.do('/cq 1')
    assert desk.legacy_said[-2] == 'slot 1 is busy; no CQ sent' and len(desk.ids) == 1


def test_cq_repeat_identifies_after_every_unanswered_cycle_before_listening():
    desk = Desk(**{FRAME: CQ_DONE})
    desk.session.command('/cq 1 repeat')
    desk.drive(until=lambda: desk.sent.count(FRAME) == 6)
    assert len(desk.ids) == 2
    for at in (i['sent'] for i in desk.ids):
        assert desk.sent[at - 1] == 'LISTEN OFF' and desk.sent[at] == 'LISTEN CQ'
    desk.session.command('/stop')
    desk.drive()
    assert len(desk.ids) == 3 and desk.legacy_said[-1].startswith('monitoring')


@pytest.mark.parametrize('reply', [['OK', 'PTT ON', 'PTT OFF', 'DISCONNECTED'], ['OK', 'PTT ON']])
def test_a_failed_or_stopped_call_identifies(reply):
    desk = Desk(**{'CONNECT W9SSJ K7QB': reply})
    desk.session.command('/call K7QB 2')
    if 'DISCONNECTED' not in reply:
        desk.drive(until=lambda: desk.flow.state == v.State.CONNECTING)
        desk.session.command('/stop')
        desk.drive(until=lambda: 'ABORT' in desk.sent)
        desk.hear('PTT OFF')
    desk.drive()
    desk.tick()
    assert [(i['dial'], i['after_peer']) for i in desk.ids] == [
        (14_103_500 if 'DISCONNECTED' in reply else 14_105_000, False)]
    assert desk.legacy_said[-1].startswith('monitoring')


def test_quit_reports_unsent_ident_only_when_something_went_out_since_the_last_id():
    desk = Desk()
    desk.session.quit()
    assert desk.ids == []
    desk = answered()
    desk.hear('PTT ON', 'PTT OFF')
    assert not desk.do('/quit')
    desk.session.quit()
    assert desk.ids == [] and 'DISCONNECT' in desk.sent
    assert sum('send W9SSJ by hand' in line for line in desk.said) == 1


def test_quit_reports_unsent_ident_when_the_goodbye_fails():
    desk = answered()
    desk.host.replies['DISCONNECT'] = ['WRONG']
    desk.hear('PTT ON', 'PTT OFF')
    desk.session.quit()
    assert desk.ids == [] and 'ABORT' in desk.sent
    assert sum('send W9SSJ by hand' in line for line in desk.said) == 1


def test_operator_close_cancels_scheduled_beacons_before_identification():
    desk = answered()
    desk.session.set_beacon(1)
    desk.hear('PTT ON', 'PTT OFF')
    desk.do('/bye')
    desk.tick()
    assert len(desk.ids) == 1
    assert desk.session.beacon is None and not desk.session._beacon_now
    before = list(desk.sent)
    desk.host.now += 120
    desk.tick()
    assert desk.sent == before and desk.sent[-1] == 'LISTEN CQ'


def test_ident_unkey_failure_never_returns_to_monitoring():
    desk = answered()
    def identify(bw, after_peer):
        desk.line = True
    desk.session.keyer = identify
    desk.hear('PTT ON', 'PTT OFF')
    with pytest.raises(v.HostError, match='HIGH after identification'):
        desk.do('/bye')
    assert desk.sent[-1] == 'DISCONNECT'
    assert desk.session._closing and desk.session._close_failed is not None
