# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Real Client single-pump and Session over fake command/data sockets."""
import pytest
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk

LIST = '<BL:weather.txt|2026-10-24|100>'
OFFER = '<SF:weather.txt.VarAC_gzip|25|0123456789ABCDEF0123456789ABCDEF>'


def linked(tmp_path):
    d = Desk(mailbox=tmp_path)
    d.flow.dial = 14_106_500
    d.hear('CONNECTED W9SSJ K6HN 500')
    d.session.command('/bbs list')
    d.receive(LIST)
    return d


def offered(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs offer 1')
    d.receive(OFFER)
    return d


def test_offer_raw_roundtrip_has_no_readiness(tmp_path):
    d = offered(tmp_path)
    assert d.texts == ['5 <BLR>', '16 <BG:weather.txt>']
    assert d.session.files.offer.raw == OFFER
    files = list(tmp_path.glob('file-captures/*/transport.bin'))
    assert len(files) == 1
    assert files[0].read_bytes() == v.frame_message('<BG:weather.txt>') + v.frame_message(OFFER)


def test_capture_switches_before_immediate_binary_and_keeps_command_pump(tmp_path):
    d = offered(tmp_path)
    payload = b'\x00\xff<SFB>\x1f\x8b\x00unresolved'
    def response(data):
        d.host.lines('BUFFER 0')
        d.host.data.incoming.append(payload)
    d.host.data.respond = response
    d.session.command('/bbs capture')
    d.tick()
    assert d.session.files.opaque
    assert d.host.client.buffer == 0
    assert not d.host.client._receive_quarantined
    recorder = d.session.files.recorder
    assert (recorder.directory/'transport.bin').read_bytes() == v.frame_message('<SFRD>')+payload
    assert not any('<SFOK>' in text for text in d.texts)
    d.session.command('/stop')
    d.drive()
    assert not d.host.client.connected
    assert not d.session.files.opaque
    assert not any('<DISC>' in text for text in d.texts)


@pytest.mark.parametrize('command', ['/stop', '/bye', '/quit'])
def test_capture_cancellation_closes_only_contact(tmp_path, command):
    d = offered(tmp_path)
    d.session.command('/bbs capture')
    d.tick()
    if command == '/quit':
        assert not d.session.command(command)
        d.session.quit()
    else:
        d.session.command(command)
        d.drive()
    assert not d.host.client.connected
    assert not d.host.client.closed
    assert not d.session.files.active
    assert d.session.files.offer is None


def test_rejection_and_no_stale_offer_on_reconnect(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs offer weather.txt')
    d.receive('<BGJ>')
    before = d.texts[:]
    d.session.command('/bbs capture')
    assert d.texts == before
    d.session.command('/bbs offer 1')
    d.receive(OFFER)
    d.hear('DISCONNECTED')
    d.flow.dial = 14_106_500
    d.hear('PENDING', 'CONNECTED W9SSJ K6HN 500')
    before = d.texts[:]
    d.session.command('/bbs capture')
    d.session.command('/bbs offer 1')
    assert d.texts == before


@pytest.mark.parametrize('failure', ['deadline', 'cap', 'write'])
def test_capture_limits_close_without_parsing_binary(tmp_path, monkeypatch, failure):
    d = offered(tmp_path)
    d.session.command('/bbs capture')
    d.tick()
    recorder = d.session.files.recorder
    if failure == 'deadline':
        recorder.deadline = d.host.client.clock()
    elif failure == 'cap':
        recorder.byte_limit = sum(recorder.offsets.values()) + 3
        d.host.data.incoming.append(b'\x00\xffbinary overflow')
    else:
        monkeypatch.setattr(recorder, 'append', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
        d.host.data.incoming.append(b'\xffbinary')
    d.tick()
    d.drive()
    assert not d.host.client.connected
    assert not d.host.client._receive_quarantined
    assert not d.session.files.active
    assert not any('<SFOK>' in text for text in d.texts)


def test_offer_requires_current_listing_and_off_cf(tmp_path):
    d = Desk(mailbox=tmp_path)
    d.hear('CONNECTED W9SSJ K6HN 500')
    before = d.texts[:]
    d.session.command('/bbs offer 1')
    assert d.texts == before
    d.flow.dial = 14_106_500
    d.session.command('/bbs offer weather.txt')
    assert d.texts == before


def test_fragmented_offer_and_provenance(tmp_path):
    import json
    d = linked(tmp_path)
    d.session.command('/bbs offer 1')
    for byte in v.frame_message(OFFER):
        d.host.data.incoming.append(bytes([byte]))
    d.tick()
    assert d.session.files.offer.raw == OFFER
    d.session.command('/bbs capture')
    record = json.loads((d.session.files.recorder.directory/'offer.json').read_text())
    assert record['raw_offer'] == OFFER
    assert (tmp_path/'file-captures') in __import__('pathlib').Path(record['offer_capture']).parents
    d.session.command('/stop')
    d.drive()


@pytest.mark.parametrize('kind', ['invalid', 'wrong-name', 'timeout', 'stop', 'disconnect'])
def test_offer_failure_never_sends_readiness(tmp_path, kind):
    d = linked(tmp_path)
    d.session.command('/bbs offer 1')
    if kind == 'invalid':
        d.receive('<SF:bad>')
    elif kind == 'wrong-name':
        d.receive(OFFER.replace('weather.txt', 'other.txt'))
    elif kind == 'timeout':
        d.session.files.recorder.deadline = d.host.client.clock()
        d.tick()
    elif kind == 'stop':
        d.session.command('/stop')
    else:
        d.hear('DISCONNECTED')
    d.drive()
    assert not d.host.client.connected
    assert d.session.files.offer is None
    assert not d.session.files.active
    assert not any('SFRD' in text or 'SFOK' in text for text in d.texts)


@pytest.mark.parametrize('phase', ['offer-create', 'capture-create', 'capture-provenance', 'capture-send'])
def test_file_operation_io_failure_does_not_end_terminal(tmp_path, monkeypatch, phase):
    from pathlib import Path
    d = linked(tmp_path) if phase == 'offer-create' else offered(tmp_path)
    fail = lambda *a, **kw: (_ for _ in ()).throw(OSError('cannot write evidence'))
    if phase.endswith('create'):
        monkeypatch.setattr(d.session.files, '_new_capture', fail)
    elif phase == 'capture-provenance':
        monkeypatch.setattr(Path, 'write_text', fail)
    else:
        monkeypatch.setattr(d.host.data, 'sendall', fail)
    assert d.session.command('/bbs offer 1' if phase == 'offer-create' else '/bbs capture')
    d.drive()
    assert not d.host.client.connected
    assert not d.host.client.closed
    assert not d.session.files.active


def test_retired_capture_rejects_late_bytes_until_new_link(tmp_path):
    d = offered(tmp_path)
    d.session.command('/bbs capture')
    d.tick()
    received = len(d.kinds('rx'))
    # Command socket is consumed first in this poll; late data must remain opaque.
    d.host.lines('DISCONNECTED')
    d.host.data.incoming.append(v.frame_message('<SM><SBJ:late><MSG:not mail>'))
    d.tick()
    d.host.data.incoming.append(b'\x00\xfflate binary')
    d.tick()
    assert len(d.kinds('rx')) == received
    assert not d.mailbox.entries()
    assert not d.host.client._receive_quarantined
    d.flow.dial = 14_106_500
    d.hear('PENDING', 'CONNECTED W9SSJ K6HN 500')
    d.receive('new contact text')
    assert d.kinds('rx')[-1]['text'] == 'new contact text'
    assert not d.session.files.retired_opaque
