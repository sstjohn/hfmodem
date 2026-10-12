# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import pytest

from hfmodem.shrike import p4frame, spec
from hfmodem.shrike.p4credit import ReceiptContext
from hfmodem.shrike.p4host import P4StreamFailure
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.test_p4turn import FS, machine

BODY = b'RMS Trimode 1.4.4.0 Welcome to '
FULL = (b'K4MSU Winlink Gateway at Murray State University\r\n'
        b'W9SSJ has 120 daily minutes remaining with K4MSU (EM56UO)\r'
        b'{SFI = 126 On 2026-1')
A = FULL[:40]
B = b'\x01\x01\x01' + FULL[40:77]
C = FULL[77:117]


KB5 = [
    (Kind.BODY, 17.225, 0x00, bytes.fromhex('524d53205472696d6f646520312e342e342e30204b42354c5a4b2061742074')),
    (Kind.NORMAL, 20.975, 0x01, bytes.fromhex('68652041726b616e736173204469766973696f6e206f6620456d657267656e6379204d616e6167656d656e740d0a573953534a206861732031343338206461696c79206d696e757465732072656d61696e696e672077697468204b42354c5a4b2028454d33345554290d5b574c324b2d352e302d4232465749484a4d245d0d3b')),
    (Kind.NORMAL, 28.475, 0x01, bytes.fromhex('68652041726b616e736173204469766973696f6e206f6620456d657267656e6379204d616e616765')),
]

WS8 = [
    (Kind.BODY, 19.760, 0x00, bytes.fromhex('524d53205472696d6f646520312e342e342e300d0a573953534a2068617320')),
    (Kind.NORMAL, 39.760, 0x01, bytes.fromhex('3736206461696c79206d696e757465732072656d61696e696e67207769746820575338454f432028454e37325151290d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a2038333032313138310d434d532076696120575338454f43203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    (Kind.NORMAL, 43.510, 0x01, bytes.fromhex('3736206461696c79206d696e757465732072656d61696e696e67207769746820575338454f432028')),
    (Kind.NORMAL, 58.510, 0x02, bytes.fromhex('454e37325151290d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a20383330323131')),
    (Kind.NORMAL, 61.010, 0x43, bytes.fromhex('38310d434d532076696120575338454f43203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
]

GOOD = [
    (Kind.BODY, 27.243, 0x00, bytes.fromhex('524d53205472696d6f646520312e342e342e302057656c636f6d6520746f20')),
    (Kind.NORMAL, 28.493, 0x01, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e')),
    (Kind.NORMAL, 29.743, 0x82, bytes.fromhex('01010169766572736974790d0a573953534a2068617320313031206461696c79206d696e75746573')),
    (Kind.NORMAL, 30.993, 0x82, bytes.fromhex('01010169766572736974790d0a573953534a2068617320313031206461696c79206d696e75746573')),
    (Kind.NORMAL, 32.243, 0x03, bytes.fromhex('2072656d61696e696e672077697468204b344d53552028454d3536554f290d5b574c324b2d352e30')),
    (Kind.NORMAL, 35.993, 0x00, bytes.fromhex('2d4232465749484a4d245d0d3b50513a2030313933333431330d434d5320766961204b344d535520')),
    (Kind.NORMAL, 37.243, 0x00, bytes.fromhex('2d4232465749484a4d245d0d3b50513a2030313933333431330d434d5320766961204b344d535520')),
    (Kind.NORMAL, 38.493, 0x01, bytes.fromhex('3e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    (Kind.NORMAL, 39.743, 0x02, bytes.fromhex('1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
]


SPEED_RESTART = [
    (Kind.BODY, 18.508, 0x00, bytes.fromhex('524d53205472696d6f646520312e342e342e302057656c636f6d6520746f20')),
    (Kind.NORMAL, 19.759, 0x01, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e')),
    (Kind.NORMAL, 21.009, 0x01, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e')),
    (Kind.NORMAL, 22.258, 0x01, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e')),
    (Kind.LONG, 29.754, 0x62, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e69766572736974790d0a573953534a2068617320313037206461696c79')),
    (Kind.LONG, 33.504, 0x62, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e69766572736974790d0a573953534a2068617320313037206461696c79')),
    (Kind.LONG, 37.253, 0x03, bytes.fromhex('206d696e757465732072656d61696e696e672077697468204b344d53552028454d3536554f290d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    (Kind.LONG, 41.003, 0x03, bytes.fromhex('206d696e757465732072656d61696e696e672077697468204b344d53552028454d3536554f290d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
]

SPEED_CONTROLS = [
    [
        (Kind.NORMAL, 58.507, 0x01, bytes.fromhex('4b344d53552057696e6c696e6b2047617465776179206174204d757272617920537461746520556e')),
        (Kind.LONG, 59.757, 0x02, bytes.fromhex('69766572736974790d0a573953534a2068617320313131206461696c79206d696e757465732072656d61696e696e672077697468204b344d53552028454d3536554f290d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    ],
    [
        (Kind.NORMAL, 12.262, 0x01, bytes.fromhex('313032206461696c79206d696e757465732072656d61696e696e67207769746820575338454f4320')),
        (Kind.LONG, 13.512, 0x02, bytes.fromhex('28454e37325151290d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    ],
    [
        (Kind.NORMAL, 18.509, 0x01, bytes.fromhex('3937206461696c79206d696e757465732072656d61696e696e67207769746820575338454f432028')),
        (Kind.LONG, 19.759, 0x02, bytes.fromhex('454e37325151290d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    ],
    [
        (Kind.NORMAL, 36.018, 0x02, bytes.fromhex('28464e30345645290d5b574c324b2d352e302d4232465749484a4d245d0d3b50513a203638383236')),
        (Kind.LONG, 37.268, 0x03, bytes.fromhex('3630380d434d5320766961205645334b5047203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')),
    ],
]


def recording():
    m, arq = machine(role='irs', data='body', ladder='none', normal_ack='parity',
                     prekey_lead=.025)
    events = []
    m.adapter.host = SimpleNamespace(deliver=lambda data: events.append(('data', data)),
                                    supervisor=lambda *args: events.append(('block', *args)))
    assert not arq._outbuf
    return m, events


def stream(events):
    return b''.join(event[1] for event in events if event[0] == 'data')


def packet(kind, seconds, status, data, width=None):
    at = round(seconds*FS)
    assert width is None or len(data) <= width
    data = data.ljust(width or len(data), b'\x1e')
    assert len(data) in (31, 40, 69, 128, 203)
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                 data_bytes=len(data))
    assert frame is not None and frame.data == data
    obs = Observation(kind, FS, 0, at, at, .796, 0., .9, frame=frame)
    return RegisteredObservation(obs, Registered.NEW)


def receive(m, kind, seconds, status, data, width=None):
    reg = packet(kind, seconds, status, data, width)
    at = reg.observation.stream_sample
    m.observe(reg, now=at)
    proposal = m.decide(at+round(m.t[kind]*FS)-m.lead)
    if proposal is not None:
        m.keyed()
    return proposal


def test_k4msu_shrink_preserves_all_source_and_ack_stall():
    m, events = recording()
    for at in (36.010, 37.260):
        receive(m, Kind.BODY, at, 0, BODY)
    answers = []
    for at, status, data in [(38.510, 1, FULL), (39.760, 1, FULL),
                             (41.010, 1, A), (42.260, 1, A), (87.264, 1, A),
                             (88.514, 0x82, B), (91.012, 3, C),
                             (92.264, 3, C), (211.005, 3, C)]:
        answer = receive(m, Kind.NORMAL, at, status, data)
        answers.append(answer.word)
        assert stream(events) == BODY+FULL
        assert m.state == State.IRS_ANSWER
        assert len(m.adapter._covered) <= len(FULL)
        if at in (38.510, 41.010, 88.514, 91.012):
            assert (len(m.adapter._covered), m.adapter._source_width) == {
                38.510: (128, 128), 41.010: (128, 40), 88.514: (88, 37),
                91.012: (51, 40)}[at]
    assert answers == ['V27']*5+['X27']+['V27']*3
    assert stream(events).count(b'K4MSU Winlink Gateway') == 1
    assert stream(events).count(b'W9SSJ has') == 1
    assert b'\x01' not in stream(events)
    assert len(m.adapter._covered)-m.adapter._source_width == 11
    normal = [row for row in m.link_events if row['event'] == 'normal-answer'
              and row['at'] >= round(38.510*FS)]
    assert [row['repeat'] for row in normal] == [False, True, False, True, True,
                                                False, False, True, True]
    assert not any(row['event'] == 'turn' for row in m.link_events)
    m.link_down('peer silent')
    assert stream(events) == BODY+FULL

    m, events = recording()
    expected = b''
    prior = None
    for kind, at, status, data in GOOD:
        answer = receive(m, kind, at, status, data)
        identity = status, data
        if identity != prior:
            expected += spec.field_payload(data[3:] if status == 0x82 else data)
        prior = identity
        assert stream(events) == expected
        assert answer.word == ('X27' if status & 3 in (0, 2) else 'V27')
    assert expected.endswith(b'CMS via K4MSU >\r')
    assert b'101 daily minutes' in expected and expected.count(b'W9SSJ has') == 1
    assert b'\x01' not in expected and b'\x1e' not in expected


def test_other_resegmentations_and_protocol2_control():
    for rows in (KB5, WS8):
        m, events = recording()
        expected = spec.field_payload(rows[0][3])+spec.field_payload(rows[1][3])
        for i, row in enumerate(rows):
            answer = receive(m, *row)
            assert stream(events) == (spec.field_payload(rows[0][3]) if i == 0 else expected)
        if rows is WS8:
            assert answer.kind == Kind.BODY and answer.status == 0x40
            assert m.state == State.ISS_GRANT
            replay = [row for row in m.link_events if row['event'] == 'source-replay'][-1]
            assert replay['replayed_source_bytes'] == 20
            assert replay['new_source_bytes'] == 0
        m.link_down('peer silent')
        assert stream(events) == expected


@pytest.mark.parametrize('counted_kind', [Kind.NORMAL, Kind.LONG])
@pytest.mark.parametrize('case', ['denied', 'partial', 'mismatch', 'gap', 'body',
                                 'close', 'epoch', 'link-down', 'extension',
                                 'opening-credit', 'changeover-credit', 'kind'])
def test_resegment_transaction_and_epoch_boundaries(case, counted_kind):
    m, events = recording()
    kind = Kind.LONG if case == 'opening-credit' else counted_kind
    field = b'X'*38+b'\x1cB7 ' if case == 'partial' else FULL
    if case.endswith('credit'):
        field = BODY+FULL[:97]
        receive(m, Kind.BODY, 36.010, 0 if case == 'opening-credit' else 0x40, BODY)
        if case == 'opening-credit':
            at = round(37.260*FS)
            obs = Observation(Kind.SHORT, FS, 0, at, at, .758, 0., .9,
                              value=49, header='B', walsh_peak=64)
            m.observe(RegisteredObservation(obs, Registered.NEW), now=at)
            answer = m.decide(at+round(m.t[Kind.SHORT]*FS)-m.lead)
            assert answer.word == 'X27'
            m.keyed()
            assert m._opening_negotiated
        assert m._receipt is not None
    receive(m, kind, 38.510, 1, field, 128)
    expected = b'X'*38 if case == 'partial' else BODY+FULL[:97] if case.endswith('credit') else FULL
    assert stream(events) == expected
    credited = m.adapter.result.committed_credit
    receive(m, kind, 41.010, 1, field[:40], 40)
    assert stream(events) == expected
    assert m.adapter.result.committed_credit == 0
    if case.endswith('credit'):
        assert credited == 31 and m.adapter._active is None
    reg = packet(kind, 88.514, 0 if case == 'gap' else 0x82,
                 b'\x01\x01\x01'+field[40:77], 40)
    if case == 'partial':
        reg = packet(kind, 88.514, 2, field[40:], 40)
    elif case.endswith('credit'):
        reg = packet(kind, 88.514, 2, FULL[9:49], 40)
    if case == 'mismatch':
        reg = packet(kind, 88.514, 0x82, b'\x01\x01\x01!'+field[41:77], 40)
    if case in ('denied', 'partial'):
        allow = [case == 'partial']
        if case == 'partial':
            events.clear()
            m, events = recording()
            original = m.adapter.host.deliver
            def deliver(data):
                original(data)
                allow[0] = False
            m.adapter.host.deliver = deliver
            first = packet(kind, 38.510, 1, field, 128)
        else:
            first = reg
        with m.adapter.admission(lambda: allow[0]):
            m.observe(first, now=first.observation.stream_sample)
        assert stream(events) == expected
        if case == 'partial':
            assert m.adapter._delivery_journal is not None and m.adapter.last is None
            assert m.adapter._covered == b'' and m.adapter._source_counter is None
        else:
            assert (m.adapter._covered, m.adapter._source_width) == (FULL, 40)
        allow[0] = True
        m.adapter.accept(first.observation.frame, counted=True,
                         receipt_context=ReceiptContext(None, kind), admit=lambda: allow[0])
        if case == 'partial':
            receive(m, kind, 41.010, 1, field[:40], 40)
        m.observe(reg, now=reg.observation.stream_sample)
    else:
        m.observe(reg, now=reg.observation.stream_sample)
    assert stream(events) == expected
    if case in ('gap', 'mismatch'):
        assert m.adapter.failure and m.state == State.IRS_CLOSE
        with pytest.raises(P4StreamFailure):
            m.adapter.accept(packet(kind, 91.012, 3, C).observation.frame, counted=True)
        return
    if case == 'partial':
        assert events == [('data', b'X'*38), ('block', 'B', b'7')]
        assert m.adapter._delivery_journal is None
    if case == 'extension':
        m, events = recording()
        receive(m, kind, 38.510, 1, A)
        receive(m, kind, 41.010, 1, FULL)
        receive(m, kind, 42.260, 1, FULL)
        assert events == [('data', A), ('data', FULL[40:])]
        return
    if case == 'kind':
        receive(m, Kind.LONG if kind == Kind.NORMAL else Kind.NORMAL, 91.012, 3, C)
        assert stream(events) == expected
        return
    if case == 'body':
        receive(m, Kind.BODY, 95.000, 0x40, b'next\r', 31)
        expected += b'next\r'
    elif case == 'close':
        receive(m, kind, 95.000, 0x23, field[77:117], 40)
    elif case == 'link-down':
        m.link_down('peer silent')
    else:
        m._close_receive_epoch('synthetic boundary')
    assert stream(events) == expected
    m._open_receive_epoch(round(100*FS))
    m.state = State.IRS_ANSWER
    receive(m, kind, 100.000, 1, FULL)
    assert stream(events) == expected+FULL


def test_k4msu_restart_at_speed_change():
    m, events = recording()
    expected = b''
    answers = []
    results = []
    for i, row in enumerate(SPEED_RESTART):
        answer = receive(m, *row)
        answers.append(answer.word)
        assert answer.kind == Kind.WORD and answer.ack_identity == (row[2], row[3])
        assert answer.stream_sample == round(row[1]*FS)+round(m.t[row[0]]*FS)
        result = m.adapter.result
        results.append((result.replayed_source_bytes, result.new_source_bytes))
        if i in (0, 1, 6):
            expected += spec.field_payload(row[3])
        elif i == 4:
            expected += row[3][40:]
        assert stream(events) == expected
        assert m.state == State.IRS_ANSWER
        assert len(m.adapter._covered) <= 69
    assert answers == ['X27']+['V27']*3+['X27']*4
    assert results[1:] == [(0, 40), (40, 0), (40, 0), (40, 29),
                          (69, 0), (0, 39), (39, 0)]
    assert events == [('data', BODY), ('data', A),
                      ('data', SPEED_RESTART[4][3][40:]),
                      ('data', spec.field_payload(SPEED_RESTART[6][3]))]
    assert expected.endswith(b'(EM56UO)\r')
    assert expected.count(b'K4MSU Winlink Gateway') == 1
    assert expected.count(b'W9SSJ has 107 daily') == 1
    assert b'\x1e' not in expected
    repeats = [row for row in m.link_events if row['event'] == 'acked-repeat'
               and row['at'] >= round(29.754*FS)]
    assert [(row['at'], row['word']) for row in repeats] == [
        (round(33.504*FS), 'X27'), (round(41.003*FS), 'X27')]
    assert not any(row['event'] == 'turn' for row in m.link_events)


@pytest.mark.parametrize('rows', SPEED_CONTROLS)
def test_recorded_ordinary_speed_changes(rows):
    m, events = recording()
    expected = b''
    for row in rows:
        answer = receive(m, *row)
        expected += spec.field_payload(row[3])
        assert stream(events) == expected
        assert m.adapter.result.replayed_source_bytes == 0
        assert m.adapter.result.new_source_bytes == len(spec.field_payload(row[3]))
        assert not m.adapter.result.resegmented
        assert answer.word == ('X27' if row[0] == Kind.LONG or row[2] & 3 in (0, 2) else 'V27')
    assert b'\x1e' not in expected


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
def test_advancing_replay_exhaustion_delivers_only_novel_suffix(kind):
    m, events = recording()
    receive(m, kind, 38.510, 1, FULL)
    receive(m, kind, 41.010, 1, A)
    receive(m, kind, 88.514, 0x82, B)
    assert stream(events) == FULL
    receive(m, kind, 91.012, 3, FULL[77:]+b'!\r', 128)
    assert stream(events) == FULL+b'!\r'
    assert m.adapter.result.replayed_source_bytes == 51
    assert m.adapter.result.new_source_bytes == 2
    assert len(m.adapter._covered) == m.adapter._source_width == 53
    receive(m, kind, 94.762, 3, FULL[77:]+b'!\r', 128)
    assert m.adapter.result.duplicate and m.adapter.result.new_source_bytes == 0
    assert events == [('data', FULL), ('data', b'!\r')]


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG], ids=['speed-up', 'synthetic-speed-down'])
@pytest.mark.parametrize('case', ['same-shrink', 'same-extend', 'ordinary', 'restart',
                                 'short-prefix', 'mismatch', 'gap', 'exact-repeat',
                                 'same-kind-text', 'empty', 'short-prefix-debt',
                                 'same-mismatch', 'same-unclassified', 'restart-debt'])
def test_speed_change_source_alignment(kind, case):
    m, events = recording()
    other = Kind.LONG if kind == Kind.NORMAL else Kind.NORMAL
    if case in ('same-shrink', 'mismatch', 'gap', 'short-prefix-debt',
                'same-mismatch', 'restart-debt'):
        receive(m, kind, 38.510, 1, FULL)
        receive(m, other if case == 'same-shrink' else kind, 41.010, 1, A)
        assert stream(events) == FULL
        if case == 'same-shrink':
            receive(m, kind, 88.514, 0x82, B)
            receive(m, other, 91.012, 3, C)
            assert stream(events) == FULL
            assert (len(m.adapter._covered), m.adapter._source_width) == (51, 40)
        elif case == 'restart-debt':
            receive(m, other, 88.514, 2, FULL[:69], 128)
            assert stream(events) == FULL
            assert m.adapter.result.replayed_source_bytes == 69
            assert (m.adapter._covered, m.adapter._source_width) == (FULL, 69)
            receive(m, other, 92.264, 3, FULL[69:]+b'!\r', 128)
            assert stream(events) == FULL+b'!\r'
            assert m.adapter.result.replayed_source_bytes == 59
            assert m.adapter.result.new_source_bytes == 2
        else:
            field = FULL[:50]+b'!'+FULL[51:69]
            if case == 'short-prefix-debt':
                field = A[:20]
            receive(m, other, 88.514, 0 if case == 'gap' else
                    1 if case == 'same-mismatch' else 2, field, 128)
            assert stream(events) == FULL
            assert m.adapter.failure.endswith('counter gap' if case == 'gap' else 'overlap mismatch')
            assert m.state == State.IRS_CLOSE
        return
    receive(m, kind, 38.510, 1, b'' if case == 'empty' else A, 40)
    if case == 'same-extend':
        field, counter, expected = FULL, 1, FULL
    elif case in ('restart', 'exact-repeat'):
        field, counter, expected = FULL[:69], 2, FULL[:69]
        if case == 'exact-repeat':
            receive(m, other, 41.010, 1, A)
            assert m.adapter.result.duplicate
            assert m.adapter._source_kind == other
            other = kind
    elif case == 'short-prefix':
        field, counter, expected = A[:20], 2, A+A[:20]
    elif case == 'same-kind-text':
        field, counter, expected = A, 2, A+A
        other = kind
    elif case == 'empty':
        field, counter, expected = A, 2, A
    elif case == 'same-unclassified':
        field, counter, expected = b'fresh\r', 1, A+b'fresh\r'
    else:
        field, counter, expected = FULL[40:77], 2, FULL[:77]
    receive(m, other, 88.514, counter, field, 128 if len(field) > 40 else 40)
    assert stream(events) == expected
    replayed = 40 if case in ('same-extend', 'restart', 'exact-repeat') else 0
    assert m.adapter.result.replayed_source_bytes == replayed
    assert m.adapter.result.new_source_bytes == len(field)-replayed
    if case == 'same-unclassified':
        assert m.adapter.result.reason == 'unclassified same-counter source'
    if case in ('restart', 'exact-repeat'):
        assert (m.adapter._source_counter, m.adapter._source_kind,
                m.adapter._source_width, m.adapter._covered) == (2, other, 69, FULL[:69])
        receive(m, other, 92.264, 3, FULL[69:109])
        assert stream(events) == FULL[:109]
    elif case == 'same-extend':
        assert m.adapter._covered == FULL


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG], ids=['speed-up', 'synthetic-speed-down'])
@pytest.mark.parametrize('stage', ['before', 'partial'])
def test_speed_restart_admission_rollback(kind, stage):
    m, events = recording()
    other = Kind.LONG if kind == Kind.NORMAL else Kind.NORMAL
    receive(m, kind, 38.510, 1, A)
    field = A+b'new\r\x1cB7 '
    reg = packet(other, 88.514, 2, field, 128)
    allow = [stage == 'partial']
    if stage == 'partial':
        original = m.adapter.host.deliver
        def deliver(data):
            original(data)
            allow[0] = False
        m.adapter.host.deliver = deliver
    with m.adapter.admission(lambda: allow[0]):
        m.observe(reg, now=reg.observation.stream_sample)
    assert stream(events) == A+(b'new\r' if stage == 'partial' else b'')
    assert (m.adapter._source_kind, m.adapter._source_counter,
            m.adapter._source_width, m.adapter._covered) == (kind, 1, 40, A)
    if stage == 'partial':
        assert m.adapter._delivery_journal is not None
    allow[0] = True
    m.adapter.accept(reg.observation.frame, counted=True,
                     receipt_context=ReceiptContext(None, other), admit=lambda: allow[0])
    assert events == [('data', A), ('data', b'new\r'), ('block', 'B', b'7')]
    assert m.adapter.result.replayed_source_bytes == 40
    assert m.adapter.result.new_source_bytes == len(field)-40
    assert m.adapter._source_kind == other and m.adapter._source_counter == 2
    assert m.adapter._covered == field and m.adapter._source_width == len(field)
    assert m.adapter._delivery_journal is None
    receive(m, other, 92.264, 2, field, 128)
    assert m.adapter.result.duplicate
    assert events == [('data', A), ('data', b'new\r'), ('block', 'B', b'7')]


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
@pytest.mark.parametrize('stage', ['before', 'release'])
def test_denied_counted_close_preserves_credit_until_retry(kind, stage, monkeypatch):
    m, events = recording()
    receive(m, Kind.BODY, 36.010, 0, BODY)
    receipt = m._receipt
    assert receipt is not None
    m.adapter.arm(receipt)
    m.adapter.accept(packet(Kind.CHIRP, 37.260, 0x61, BODY[:4], 40).observation.frame,
                     receipt_context=ReceiptContext(receipt, Kind.CHIRP))
    credit = m.adapter._receipts[receipt]
    assert credit.state == 'matching' and credit.held == BODY[:4]
    cancelled, epochs = [], []
    retire = m.adapter._retire
    def retiring(receipt, state):
        retire(receipt, state)
        if state == 'cancelled':
            cancelled.append(receipt)
    monkeypatch.setattr(m.adapter, '_retire', retiring)
    end_epoch = m.adapter.end_epoch
    def ending(epoch, reason, **kwargs):
        active = m._receive_active
        result = end_epoch(epoch, reason, **kwargs)
        if active:
            epochs.append(reason)
        return result
    monkeypatch.setattr(m.adapter, 'end_epoch', ending)
    allow = [stage == 'release']
    expire = [stage == 'release']
    if stage == 'release':
        feed = type(m.adapter.supervisor).feed
        def feeding(supervisor, field):
            payload = feed(supervisor, field)
            if expire[0] and field == BODY[:4]:
                allow[0] = False
            return payload
        monkeypatch.setattr(type(m.adapter.supervisor), 'feed', feeding)
    reg = packet(kind, 88.514, 0x22, b'bye\r', 40)
    with m.adapter.admission(lambda: allow[0]):
        m.observe(reg, now=reg.observation.stream_sample)
    assert stream(events) == BODY
    credit = m.adapter._receipts[receipt]
    assert credit.state == 'matching' and credit.held == BODY[:4]
    assert m._receipt == receipt and m.adapter._active == receipt
    assert m.adapter._source_counter is None and m.adapter._covered == b''
    assert epochs == []
    cancelled.clear()
    expire[0] = False
    allow[0] = True
    m.adapter.accept(reg.observation.frame, counted=True,
                     receipt_context=ReceiptContext(receipt, kind), admit=lambda: allow[0])
    receive(m, kind, 92.264, 0x22, b'bye\r', 40)
    assert stream(events) == BODY+BODY[:4]+b'bye\r'
    assert cancelled == [receipt] and epochs == ['peer close']
    assert m._receipt is None and m.adapter._active is None
    assert not m._receive_active and m.state == State.IRS_CLOSE


def test_counted_source_diagnostic_labels_and_counts():
    m, events = recording()
    for kind, at, status, data, width in [
            (Kind.NORMAL, 38.510, 1, A, 40),
            (Kind.NORMAL, 39.760, 1, A, 40),
            (Kind.LONG, 41.010, 2, FULL[:69], 128),
            (Kind.LONG, 44.760, 3, FULL[69:109], 40),
            (Kind.LONG, 48.510, 3, b'fresh\r', 40),
            (Kind.NORMAL, 52.260, 1, b'gap\r', 40)]:
        receive(m, kind, at, status, data, width)
    rows = [row for row in m.link_events if row['event'].startswith('source-')]
    assert [(row['event'], row['replayed_source_bytes'], row['new_source_bytes'],
             row['duplicate'], row['resegmented']) for row in rows] == [
        ('source-accept', 0, 40, False, False),
        ('source-replay', 40, 0, True, False),
        ('source-replay', 40, 29, False, True),
        ('source-accept', 0, 40, False, False),
        ('source-unclassified', 0, 6, False, False),
        ('source-unclassified', 0, 4, False, False)]
    normal = [row for row in m.link_events if row['event'] == 'normal-answer']
    assert [row['repeat'] for row in normal] == [False, True, False]
    assert stream(events) == FULL[:109]+b'fresh\rgap\r'
