# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest

from hfmodem.ax25.connected import Connection, State
from hfmodem.brubru.host.tracker import TrackerHost
from hfmodem.core import hostmode as hm


def setup():
    tx, settings = [], []
    engine = Connection('NOCALL', lambda f: tx.append(f) or True)
    host = TrackerHost(engine, configure=lambda k, v: settings.append((k, v)))
    return host, engine, tx, settings


def request(host, command, channel=1):
    wire = hm.unwrap(hm.command(channel, command))
    return tuple(hm.Decoder('master', crc=False).feed(host.feed(wire)))[0]


def connected(host, engine):
    engine.connect('N0CALL')
    engine._state(State.CONNECTED)
    assert request(host, 'G').code == hm.LINK


def test_exact_trimode_initialization_and_timer_units():
    host, engine, _, settings = setup()
    commands = ['S0', 'IW9SSJ', 'A0', 'E0', 'F200', 'MN', 'O2', 'P32',
                'T15', '%XR200', 'W10', 'R0', 'Y1', 'Z0', '@D0', '@F0',
                '@I60', '@T2100', '@T36000', '@U0', '@V0', '%A0', '%BR300', '%F1500', 'S1']
    for cmd in commands:
        reply = host.feed(b'\x1b' + cmd.encode() + b'\r')
        assert b'INVALID' not in reply, (cmd, reply)
    assert b'CHANNEL 1' in reply
    assert engine.mycall == 'W9SSJ' and engine.listen
    assert engine.rtt == 2. and engine.t2 == 1. and engine.t3 == 60.
    assert engine.window == 2 and engine.ipoll == 60
    assert settings == [('P', 32), ('T', 15), ('%XR', 200), ('W', 10), ('%BR', 300)]
    host.feed(b'\x1bJHOST1\r')
    assert host.mode == 1


def test_plain_hostmode_does_not_cache_identical_host_writes_or_polls():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    wire = hm.unwrap(hm.data(1, b'repeat'))
    assert host.feed(wire) == b'\x01\x00'
    assert host.feed(wire) == b'\x01\x00'
    assert engine.accepted == 12
    host._receive(b'a' * 128 + b'b' * 128)
    assert request(host, 'G').data == b'a' * 128
    assert request(host, 'G').data == b'b' * 128


def test_split_and_coalesced_packets_including_all_binary_octets():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    wire = hm.unwrap(hm.data(1, bytes(range(256)))) + hm.unwrap(hm.command(1, 'L'))
    response = b''.join(host.feed(bytes([b])) for b in wire)
    messages = tuple(hm.Decoder('master', crc=False).feed(response))
    assert messages[0].code == hm.OK
    assert messages[1].data == b'0 0 2 0 0 4'
    assert b''.join(engine.pending) == bytes(range(256))


def test_trimode_and_express_l_fields_have_separate_meaning():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    engine.controls.clear()
    engine.write(b'x' * 129)
    engine.tick()
    wire = host.feed(hm.unwrap(hm.command(1, 'L')))
    assert len(wire) == 14
    assert wire[6] == ord('1')       # Express: unsent
    assert wire[8] == ord('1')       # Trimode: unacknowledged
    assert wire == b'\x01\x010 0 1 1 0 4\x00'


def test_terminal_disconnect_waits_for_final_binary_receive():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    host._receive(b'last\x00\xff')
    engine.abort('peer disconnected')
    assert request(host, 'G1').code == hm.OK
    assert request(host, 'G').data == b'last\x00\xff'
    event = request(host, 'G')
    assert event.code == hm.LINK and b'DISCONNECTED to N0CALL' in event.data


def test_stopped_reader_sets_rnr_and_resumes_when_drained():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    host._receive(b'x' * (6 * 128))
    assert engine.local_busy
    for _ in range(3):
        assert request(host, 'G0').code == hm.DATA
    assert not engine.local_busy


def test_disconnected_and_overflow_admission_fail():
    host, engine, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    data = hm.unwrap(hm.data(1, b'x'))
    assert tuple(hm.Decoder('master', crc=False).feed(host.feed(data)))[0].code == hm.FAIL
    connected(host, engine)
    engine.buffer_limit = 1
    assert host.feed(data) == b'\x01\x00'
    assert tuple(hm.Decoder('master', crc=False).feed(host.feed(data)))[0].code == hm.FAIL
    assert engine.accepted == 1


def test_channel_zero_ui_has_no_fake_connection_or_host_credit():
    host, engine, tx, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    assert request(host, 'C CQ N0VIA', 0).code == hm.OK
    assert host.feed(hm.unwrap(hm.data(0, b'identity'))) == b'\x00\x00'
    engine.tick()
    assert tx and engine.state is State.DISCONNECTED
    assert engine.accepted == engine.delivered == 0


def test_invalid_rf_control_does_not_commit_setting():
    host, engine, _, _ = setup()
    host.configure = lambda *_: (_ for _ in ()).throw(ValueError('station refused'))
    with pytest.raises(ValueError):
        host.command('%XR300')
    assert host.values['%XR'] == 200
    with pytest.raises(ValueError):
        host.command('R2')
    assert host.values['R'] == 0
    engine.connect('N0CALL')
    with pytest.raises(ValueError):
        host.command('F100')
    assert engine.rtt == 5.


def test_identity_disablement_attachment_loss_and_reentry():
    host, engine, _, _ = setup()
    host.command('IXXXXX')
    assert not engine.listen
    host.command('IW9SSJ')
    assert engine.listen
    host.feed(b'\x1bJHOST1\r')
    connected(host, engine)
    host._receive(b'old')
    host.attachment_lost()
    assert not host.mode and not host.rx and not host.events and not engine.listen
    assert engine.state is State.DISCONNECTED
    host.feed(b'\x1bIW9SSJ\r\x1bJHOST1\r')
    assert engine.listen and host.mode == 1


def test_non_crc_reset_requires_attachment_reentry_not_packet_counter():
    host, _, _, _ = setup()
    host.feed(b'\x1bJHOST1\r')
    host.feed(b'\x01\x01\xfftruncated')
    host.attachment_lost()
    assert b'CHANNEL 1' in host.feed(b'\x1bS1\r')
    host.feed(b'\x1bJHOST1\r')
    assert request(host, 'V').code == hm.MSG
