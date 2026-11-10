# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KISS admission and framing recovery without radio hardware."""
import pytest

from hfmodem.ax25 import kiss
from hfmodem.kestrel.broadcast import encode_ui, normalize_kiss_ui
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.host.kiss import KissPort
from hfmodem.kestrel.host.kiss_tx import KissTxQueue
from hfmodem.kestrel.host.server import HostSession, VaraServer
from .test_kiss_hostless import Client, Clock, Transport


def address(base, *, command=0, last=False, ssid=0):
    return bytes(ord(c) << 1 for c in base.ljust(6 + command)) + bytes([
        0x60 | ssid << 1 | int(last)])


def ui(source='K1ABC', destination='ALL', *, command=0, path=(), info=b'hello'):
    calls = [destination, source, *path]
    return b''.join(address(call, command=command, last=i == len(calls) - 1)
                    for i, call in enumerate(calls)) + b'\x03\xf0' + info


@pytest.fixture
def admission(monkeypatch):
    from hfmodem.kestrel.tx import broadcast
    monkeypatch.setattr(broadcast, 'synth_broadcast', lambda message: message)
    clock, transport, logs, diagnostics = Clock(), Transport(), [], []
    transport.diagnostic = lambda kind, **details: diagnostics.append((kind, details))
    runtime = AudioHostRuntime(transport, clock=clock)
    server = VaraServer(cmd_port=0, data_port=0, modem_factory=runtime.modem_factory,
                        log=lambda *args: logs.append(args))
    port = object.__new__(KissPort)
    port.on_text, port.on_frame, port._log = server._kiss_text, server._kiss_frame, server._log
    yield runtime, server, transport, clock, logs, diagnostics, port
    server.stop()


@pytest.mark.parametrize('command,raw,reason', [
    (0, ui(source=''), 'no source'),
    (0, ui(source='      '), 'no source'),
    (0, ui(source='W9/SJ'), 'A-Z or 0-9'),
    (0, ui(source='X-W9SJ'), 'A-Z or 0-9'),
    (0, ui(source='ABCDEFG'), 'shifted ASCII'),
    (1, ui(source='X-16', command=1), 'A-Z or 0-9'),
    (0, bytes([ord('A')]) + ui()[1:], 'shifted ASCII'),
    (0, ui(path=('WIDE1',) * 9), 'eight digipeaters'),
    (1, ui(command=1, source='ABCDEFG', info=b'\xff' * 750), '18 physical packets'),
    (0, ui(info=b'x' * 900), '18 physical packets'),
])
def test_refusal_is_logged_diagnosed_and_next_frame_is_admitted(
        admission, command, raw, reason):
    _, server, transport, _, logs, diagnostics, port = admission
    port._frame(0, command, raw)
    assert not server.kiss_tx_queue and not transport.keys
    assert reason in logs[-1][1]
    assert diagnostics[-1][0] == 'kiss_refused'
    assert reason in diagnostics[-1][1]['reason']
    port._frame(0, 0, ui())
    assert len(server.kiss_tx_queue) == 1


@pytest.mark.parametrize('field', ['source', 'destination', 'digipeater'])
def test_every_address_is_validated(admission, field):
    *_, port = admission
    server, diagnostics = admission[1], admission[5]
    kwargs = {'path': ('W9/SJ',)} if field == 'digipeater' else {field: 'W9/SJ'}
    port._frame(0, 0, ui(**kwargs))
    assert not server.kiss_tx_queue
    assert 'A-Z or 0-9' in diagnostics[-1][1]['reason']


def test_ssid_above_fifteen_cannot_be_put_in_the_host_envelope():
    with pytest.raises(ValueError, match='SSID'):
        encode_ui('K1ABC-16', 'ALL', b'hello')


def test_eight_digipeaters_pass_identifier_validation():
    raw = ui(path=('WIDE1',) * 8)
    assert normalize_kiss_ui(raw, command=0) == (raw, False)


@pytest.mark.parametrize('command,source', [(0, 'k1abc'), (1, 'abcdefg')])
def test_lowercase_is_warned_and_uppercased(admission, command, source):
    runtime, server, transport, clock, logs, diagnostics, port = admission
    port._frame(0, command, ui(source=source, destination='all', command=command))
    assert len(server.kiss_tx_queue) == 1 and not diagnostics
    assert any('warning' in text and 'uppercased' in text for _, text in logs)
    clock.now = 2
    runtime.step(0)
    assert transport.sent[0].source == source.upper()
    assert transport.sent[0].destination == 'ALL'


@pytest.mark.parametrize('source', ['K1ABC', 'K1ABC-15', 'AAR5ZZ', '123456', 'X'])
def test_source_has_no_callsign_or_mycall_policy(admission, source):
    runtime, server, transport, clock, logs, diagnostics, port = admission
    modem = runtime.modem_factory()
    session = HostSession(Client(), Client(), modem)
    modem.start(session)
    session._dispatch('MYCALL W9SSJ')
    port._frame(0, 0, encode_ui(source, 'ALL', b'hello', command=0))
    clock.now = 2
    runtime.step(0)
    assert transport.sent[0].source == source
    assert not diagnostics
    assert not any('warning' in text for _, text in logs)


@pytest.mark.parametrize('wire,logged', [
    (kiss.encode(b'\x32', 0), True),
    (kiss.encode(b'\x32', 1), True),
    (kiss.encode(b'\x3f', 2), True),
    (kiss.encode(b'x' * 19, 2), True),
    (kiss.encode(b'x' * 21, 2), True),
    (kiss.encode(b'\x00' * 20, 2), True),
    (kiss.encode(b'\xff' * 20, 2), True),
    *[(kiss.encode(b'x', command), False) for command in range(3, 16)],
    (kiss.encode(ui(), 3), False),
    (b'\xc0\xff\xc0', False),
    (kiss.encode(ui(), port=1), False),
])
def test_command_table_ignores_without_reply_or_keying(admission, wire, logged):
    _, server, transport, _, logs, _, port = admission
    replies = []
    port.deliver_frame = lambda *args: replies.append(args)
    parser = kiss.Parser()
    for frame in parser.feed(wire):
        port._frame(*frame)
    assert bool(logs) is logged
    assert not server.kiss_tx_queue and not transport.keys and not replies
    for frame in parser.feed(kiss.encode(ui())):
        port._frame(*frame)
    assert len(server.kiss_tx_queue) == 1


@pytest.mark.parametrize('attached', [False, True])
def test_valid_advanced_beacon_uses_attachment_admission_without_queue(admission, monkeypatch, attached):
    from hfmodem.kestrel.host import vara_modem
    monkeypatch.setattr(vara_modem, 'synth_beacon', lambda text: text)
    runtime, server, transport, _, logs, diagnostics, port = admission
    if attached:
        class Socket(Client):
            def shutdown(self, how): pass
            def close(self): pass
        modem = runtime.modem_factory()
        session = HostSession(Socket(), Socket(), modem)
        server._session = session
        modem.start(session)
        runtime.step(0)
    port._frame(0, 2, b'20OEN63BCW9SSJ'.ljust(20))
    assert not server.kiss_tx_queue and not diagnostics
    runtime.step(0)
    if attached:
        assert len(transport.sent) == 1 and not logs
    else:
        assert not transport.sent and 'no application attached' in logs[-1][1]


@pytest.mark.parametrize('garbage', [
    b'garbage', b'\xc0\x00garbage\xc0', b'\xc0\x00\xdb\x01bad\xc0',
    b'\xc0\x00\xdb\xc0',
])
def test_parser_recovers_after_garbage_and_bad_escape(admission, garbage):
    _, server, _, _, _, _, port = admission
    parser = kiss.Parser()
    for byte in garbage + kiss.encode(ui()):
        for frame in parser.feed(bytes([byte])):
            port._frame(*frame)
    assert len(server.kiss_tx_queue) == 1


def test_refusal_never_writes_to_attached_application(admission):
    runtime, server, _, _, logs, diagnostics, port = admission
    modem = runtime.modem_factory()
    cmd, data = Client(), Client()
    session = HostSession(cmd, data, modem)
    modem.start(session)
    session._dispatch('MYCALL W9SSJ')
    before = list(cmd.wire)
    port._frame(0, 0, ui(source=''))
    assert cmd.wire == before and not data.wire
    assert logs and diagnostics and not server.kiss_tx_queue


@pytest.mark.parametrize('diagnostic', [None, 'broken'])
def test_optional_diagnostic_cannot_interrupt_admission(diagnostic):
    def broken(*args, **kwargs):
        raise RuntimeError('observer failure')
    queue = KissTxQueue(diagnostic=broken if diagnostic else None)
    assert not queue.broadcast(0, ui(source=''))
    assert queue.broadcast(0, ui())
    assert len(queue) == 1


def test_opaque_information_is_admitted_with_a_note(admission):
    runtime, server, transport, clock, logs, diagnostics, port = admission
    port._frame(0, 0, ui(info=b'\xff\x00\x80'))
    assert len(server.kiss_tx_queue) == 1
    assert any('not 7-bit text' in str(entry) for entry in logs)
    assert not diagnostics


@pytest.mark.parametrize('frame,admitted', [
    (ui('W9SSJ', 'APRS', path=('WIDE1', 'WIDE2'), info=b'!4303.14N/08924.53W-Kestrel'), True),
    (ui(info=b'x' * 812), True),
    (ui(info=b'x' * 900), False),
])
def test_fit_check_admits_paths_and_up_to_eighteen_packets(admission, frame, admitted):
    runtime, server, transport, clock, logs, diagnostics, port = admission
    port._frame(0, 0, frame)
    assert len(server.kiss_tx_queue) == int(admitted)
    assert bool(diagnostics) is not admitted
