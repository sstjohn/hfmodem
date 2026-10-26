# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os

import pytest

from hfmodem.contact.bbs import HostCatalog
from hfmodem.contact.bbs_server import BBSServer, configured_directory
from hfmodem.contact.bbs_download import decode_file
from hfmodem.contact.filetransfer import parse_offer
from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_varac import Desk


@pytest.fixture
def host(tmp_path):
    root = tmp_path / 'public'
    root.mkdir()
    (root / 'hello.txt').write_bytes(b'Hello from the BBS!\r\n' * 20)
    catalog = HostCatalog()
    catalog.enable(root)
    d = Desk(mailbox=tmp_path / 'mail', kiss=True, loc='EN63bc')
    d.session.bbs_host = BBSServer(d.session, catalog)
    d.flow.dial = 14_105_750
    d.hear('CONNECTED W9SSJ KQ4QWH 500')
    yield d, catalog, root
    catalog.close()


def text(d):
    return d.texts[-1].split(' ', 1)[1]


def test_host_list_offer_payload_and_acknowledgement(host):
    d, catalog, root = host
    assert 'B' in d.flow.services
    assert d.session.beacon_text().startswith('20I')
    d.receive('<BLR>')
    assert v.bbs_files(text(d))[0].name == 'hello.txt'
    d.receive('<BG:hello.txt>')
    offer = parse_offer(text(d))
    assert d.session.bbs_host.phase == 'offer'
    assert not any('<SFB>' in t for t in d.texts)
    d.receive('<SFRD>')
    assert decode_file(offer, text(d)) == (root / 'hello.txt').read_bytes()
    count = len(d.texts)
    d.receive('<SFRD>')
    assert len(d.texts) == count
    assert not d.kinds('bbs_served')
    d.receive('<SFOK>')
    assert d.kinds('bbs_served')[-1]['filename'] == 'hello.txt'
    assert not d.session.bbs_host.active


def test_client_server_application_roundtrip(host, tmp_path):
    server, _, root = host
    client = Desk(mailbox=tmp_path / 'receiver')
    client.flow.dial = 14_105_750
    client.hear('CONNECTED W9SSJ KQ4QWH 500')
    client.session.command('/bbs list')
    server.receive(text(client))
    client.receive(text(server))
    client.session.command('/bbs get 1')
    server.receive(text(client))
    client.receive(text(server))
    server.receive(text(client))
    client.receive(text(server))
    server.receive(text(client))
    saved = list((tmp_path / 'receiver').glob('downloads/*/hello.txt'))
    assert len(saved) == 1 and saved[0].read_bytes() == (root / 'hello.txt').read_bytes()
    assert server.kinds('bbs_served')


@pytest.mark.parametrize('wire_request', ['<BG:../secret>', '<BG:/etc/passwd>', '<BG:missing>', '<BG:.hidden>'])
def test_invalid_requests_are_refused(host, wire_request):
    d, _, _ = host
    d.receive(wire_request)
    assert text(d) == '<BGJ>'
    assert not d.session.bbs_host.active


def test_no_host_or_calling_frequency_never_serves(host):
    d, catalog, _ = host
    d.flow.dial = 14_105_000
    d.receive('<BLR>')
    assert text(d) == '<BLJ>'
    d.receive('<BG:hello.txt>')
    assert text(d) == '<BGJ>'
    catalog.disable()
    assert d.session.beacon_text().startswith('20A')


@pytest.mark.parametrize('change', ['disconnect', 'qsy', 'disable', 'timeout', 'stop', 'bye'])
def test_pending_transfer_does_not_survive_change(host, change):
    d, catalog, _ = host
    d.receive('<BG:hello.txt>')
    if change == 'disconnect':
        d.hear('DISCONNECTED')
        d.hear('PENDING', 'CONNECTED W9SSJ N1ZZZ 500')
    elif change == 'qsy':
        d.flow.dial += 750
    elif change == 'disable':
        catalog.disable()
    elif change == 'timeout':
        d.host.now += 61
    else:
        d.session.command('/' + change)
    d.tick()
    d.drive()
    d.receive('<SFRD>')
    assert not any('<SFB>' in t for t in d.texts)
    assert not d.session.bbs_host.active


def test_file_snapshot_is_immutable_and_oversize_compression_refused(host):
    d, _, root = host
    d.receive('<BG:hello.txt>')
    offer = parse_offer(text(d))
    original = (root / 'hello.txt').read_bytes()
    (root / 'hello.txt').write_bytes(b'changed')
    d.receive('<SFRD>')
    assert decode_file(offer, text(d)) == original
    d.receive('<SFOK>')
    (root / 'large.bin').write_bytes(os.urandom(40000))
    d.receive('<BG:large.bin>')
    assert text(d) == '<BGJ>'


def test_config_paths_and_cli_precedence(tmp_path, monkeypatch):
    from hfmodem.tests.core.test_config import EXAMPLE
    path = tmp_path / 'station.toml'
    # The example already has a vara table; add the field there.
    content = EXAMPLE.read_text().replace('[protocols.vara]', '[protocols.vara]\nbbs_directory = "public"')
    path.write_text(content)
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    assert configured_directory() == str(tmp_path / 'public')
    assert configured_directory('') is None
    assert configured_directory(str(tmp_path / 'override')) == str(tmp_path / 'override')
