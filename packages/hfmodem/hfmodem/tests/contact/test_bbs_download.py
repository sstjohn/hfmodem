# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import base64
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.contact.bbs_download import decode_file, save_file
from hfmodem.contact.filetransfer import parse_offer
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/kq4qwh-bbs-1025.json').read_text())
OFFER = parse_offer(FIXTURE['raw_offer'])
NAME = OFFER.wire_name.removesuffix('.VarAC_gzip')
WIRE = ''.join(FIXTURE['chunks'])
MESSAGE = WIRE.split(' ', 1)[1]


def linked(tmp_path):
    d = Desk(mailbox=tmp_path)
    d.flow.dial = 14_105_750
    d.hear('CONNECTED W9SSJ KQ4QWH 500')
    d.session.command('/bbs list')
    d.receive(f'<BL:{NAME}|2026-05-31|2516>')
    return d


def test_real_capture_decodes_exact_file():
    assert len(MESSAGE) == int(WIRE.split(' ', 1)[0])
    data = decode_file(OFFER, MESSAGE)
    assert len(data) == FIXTURE['expected_size']
    assert hashlib.sha256(data).hexdigest() == FIXTURE['expected_sha256']


@pytest.mark.parametrize('fragmentation', ['observed', 'bytes', 'coalesced'])
def test_get_validates_saves_acknowledges_and_preserves_chat(tmp_path, fragmentation):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(OFFER.raw)
    assert d.texts[-1] == '6 <SFRD>'
    assert d.session.files.receiving
    chunks = [c.encode() for c in FIXTURE['chunks']]
    if fragmentation == 'bytes':
        chunks = [bytes([b]) for b in WIRE.encode()]
    elif fragmentation == 'coalesced':
        chunks = [WIRE.encode() + v.frame_message('after file')]
    d.host.data.incoming.extend(chunks)
    while d.host.data.incoming:
        d.session.tick(.001)
    saved = list(tmp_path.glob('downloads/*/*.txt'))
    assert len(saved) == 1
    assert saved[0].read_bytes() == decode_file(OFFER, MESSAGE)
    assert d.texts[-1] == '6 <SFOK>'
    assert not d.session.files.active
    assert d.host.client.connected
    if fragmentation == 'coalesced':
        assert d.kinds('rx')[-1]['text'] == 'after file'


@pytest.mark.parametrize('failure', ['checksum', 'base64', 'truncated', 'timeout', 'disconnect', 'cancel', 'disk'])
def test_failure_never_acknowledges_or_publishes(tmp_path, monkeypatch, failure):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(OFFER.raw)
    if failure == 'checksum':
        d.receive(MESSAGE[:-2] + 'AA')
    elif failure == 'base64':
        d.receive(MESSAGE[:5] + '?' + MESSAGE[6:])
    elif failure == 'truncated':
        d.receive(MESSAGE[:-4])
    elif failure == 'timeout':
        d.session.files.recorder.deadline = d.host.client.clock()
    elif failure == 'disconnect':
        d.hear('DISCONNECTED')
    elif failure == 'cancel':
        d.session.command('/stop')
    else:
        from hfmodem.contact import bbs_download
        def fail(*a):
            raise OSError('disk full')
        monkeypatch.setattr(bbs_download, 'save_file', fail)
        d.receive(MESSAGE)
    d.tick()
    d.drive()
    assert not any('SFOK' in t for t in d.texts)
    assert not list(tmp_path.glob('downloads/*/*.txt'))
    assert not d.session.files.active


@pytest.mark.parametrize('data', [
    b'not gzip',
    gzip.compress(b'a', mtime=0)[:-2],
    gzip.compress(b'a', mtime=0) + b'extra',
    gzip.compress(b'a' * 1048577, mtime=0),
], ids=['not-gzip', 'truncated-footer', 'trailing-bytes', 'expansion-limit'])
def test_gzip_validation_and_expansion_bound(data):
    offer = parse_offer(f'<SF:x.VarAC_gzip|{len(data)}|{hashlib.md5(data).hexdigest()}>')
    with pytest.raises(ValueError):
        decode_file(offer, '<SFB>' + base64.b64encode(data).decode())


def test_save_does_not_overwrite_and_rejects_traversal(tmp_path):
    a = save_file(tmp_path, 'x.txt', b'one')
    b = save_file(tmp_path, 'x.txt', b'two')
    assert a != b and a.read_bytes() == b'one' and b.read_bytes() == b'two'
    with pytest.raises(ValueError):
        save_file(tmp_path, '../bad', b'bad')


def test_request_alias_does_not_accept_automatically(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs request 1')
    d.receive(OFFER.raw)
    assert not any('SFRD' in t for t in d.texts)


@pytest.mark.parametrize('offer', [OFFER.raw.replace('|1232|', '|32769|'), OFFER.raw.replace('.VarAC_gzip', ''), '<BGJ>'])
def test_unsupported_or_refused_offer_never_sends_readiness(tmp_path, offer):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(offer)
    d.drive()
    assert not any('SFRD' in t or 'SFOK' in t for t in d.texts)


def test_disconnect_and_late_complete_payload_cannot_save(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(OFFER.raw)
    d.host.lines('DISCONNECTED')
    d.host.data.incoming.append(WIRE.encode())
    d.tick()
    assert not list(tmp_path.glob('downloads/*/*.txt'))
    assert not any('SFOK' in t for t in d.texts)


def test_empty_gzip_file():
    compressed = gzip.compress(b'')
    offer = parse_offer(f'<SF:empty.VarAC_gzip|{len(compressed)}|{hashlib.md5(compressed).hexdigest()}>')
    assert decode_file(offer, '<SFB>' + base64.b64encode(compressed).decode()) == b''


def test_download_progress_counts_payload_and_status_shows_it(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(OFFER.raw)
    assert any('0%' in s and 'waiting for file data' in s for s in d.said)
    # Partial application header and marker must not count as file bytes.
    header = WIRE.index(' ') + 1 + 5
    d.host.data.incoming.append(WIRE[:header + 400].encode())
    d.tick()
    d.session.command('/status')
    assert d.session.files.received_bytes == 300
    assert any(f'300 / {OFFER.declared_integer:,} compressed bytes' in s for s in d.said)
    assert not list(tmp_path.glob('downloads/*/*.txt'))
    assert not any('SFOK' in t for t in d.texts)
    d.host.now += 16
    d.tick()
    assert 'no new data for' in [s for s in d.said if s.startswith('File:')][-1]
    before = len([s for s in d.said if s.startswith('File:')])
    d.tick()
    assert len([s for s in d.said if s.startswith('File:')]) == before
    # Resumed reception updates progress; completion still follows validation.
    d.host.now += 5
    d.host.data.incoming.append(WIRE[header + 400:header + 800].encode())
    d.tick()
    assert 'receiving' in [s for s in d.said if s.startswith('File:')][-1]
    assert d.session.files.received_bytes == 600
    d.host.data.incoming.append(WIRE[header + 800:].encode())
    d.tick()
    assert any('checksum verified' in s for s in d.said)
    assert d.session.files.progress_text() is None
    before = len(d.kinds('bbs_progress'))
    d.host.now += 20
    d.tick()
    assert len(d.kinds('bbs_progress')) == before


def test_unrelated_messages_and_transmissions_do_not_count_as_file_progress(tmp_path):
    d = linked(tmp_path)
    d.session.command('/bbs get 1')
    d.receive(OFFER.raw)
    d.receive('unrelated chat')
    d.session.files.raw('tx', b'outbound data')
    assert d.session.files.received_bytes == 0
    # A split marker also is not part of the compressed payload count.
    marker_end = WIRE.index(' ') + 1 + 4
    d.host.data.incoming.append(WIRE[:marker_end].encode())
    d.tick()
    assert d.session.files.received_bytes == 0
    d.session.command('/stop')
    d.drive()
    assert d.session.files.progress_text() is None
