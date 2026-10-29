# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json

import pytest

from hfmodem.ax25 import kiss
from hfmodem.contact.broadcast import decode_ui
from hfmodem.tests.contact.test_broadcast_client import answer, desk, wire
from hfmodem.tests.contact.test_varac import Desk, Host


# 20261028T051045679088Z/host.jsonl, C1 lines 293/440/1964 and
# C2 lines 2818/4681/4946/5145. These include both UI addresses, not just tags.
CAPTURED = [
    ('06:07:26.051', '8c688a8e8e4040e0ac9e62868498406103f03c534d503e', 'VO1CBL', 'F4EGG', False),
    ('06:38:04.069', '966a868e404040e0ae8870a04040406103f03c534d503e', 'WD8P', 'K5CG', False),
    ('10:54:21.075', '966a868e404040e0ae8870a04040406103f03c534d503e', 'WD8P', 'K5CG', False),
    ('12:10:30.032', 'ae70a4888e4040e0ae6896889640406103f03c534d553e', 'W4KDK', 'W8RDG', True),
    ('14:31:53.631', 'ae70a4888e4040e0ae6896889640406103f03c534d553e', 'W4KDK', 'W8RDG', True),
    ('15:00:30.902', 'ae70a4888e4040e0ae6896889640406103f03c534d553e', 'W4KDK', 'W8RDG', True),
    ('15:30:20.837', 'ae70a4888e4040e0ae6896889640406103f03c534d553e', 'W4KDK', 'W8RDG', True),
]


@pytest.mark.parametrize('at,raw,source,destination,urgent', CAPTURED)
def test_c1_c2_complete_ui_replay_at_every_kiss_split(at, raw, source, destination, urgent):
    payload = bytes.fromhex(raw)
    decoded = decode_ui(payload, command=1)
    assert (decoded.source, decoded.destination, decoded.command, decoded.info) == (
        source, destination, 1, b'<SMU>' if urgent else b'<SMP>')
    framed = kiss.encode(payload, 1)
    for split in range(1, len(framed)):
        host = Host(kiss=True)
        host.kiss.incoming.extend((framed[:split], framed[split:]))
        host.client.poll()
        host.client.poll()
        broadcasts = [json.loads(event.raw) for event in host.events if event.kind == 'broadcast']
        assert broadcasts == [dict(command=1, hex=raw)], at
        assert not host.data.sent and not host.command.sent


@pytest.mark.parametrize('at,raw,source,destination,urgent', CAPTURED)
def test_captured_notice_preserves_urgency_without_transmission(at, raw, source, destination, urgent):
    d = Desk(kiss=True)
    d.host.client.mycall = destination
    before = list(d.sent), list(d.texts), list(d.tunes)
    d.host.kiss.incoming.append(kiss.encode(bytes.fromhex(raw), 1))
    d.tick()
    notice = d.session.mail_collection._load()[0]
    assert (notice['source'], notice['destination'], notice['urgent']) == (source, destination, urgent)
    assert (d.sent, d.texts, d.tunes) == before
    assert len(d.kinds('vmail_notice')) == 1


@pytest.mark.parametrize('at,raw,source,destination,urgent', CAPTURED[:3])
def test_outgoing_smp_exactly_matches_captured_ui_bytes(at, raw, source, destination, urgent):
    d = desk()
    d.host.client.mycall = source
    d.hear('BUSY OFF')
    d.session.broadcast.directed(destination)
    d.tick()
    assert wire(d)[0].split()[3] == raw
    answer(d, 'QUEUED', 'SENT')
    assert not d.session.broadcast.active


def test_relay_off_logs_raw_smu_but_preserves_saved_notice():
    d = Desk(kiss=True)
    _, raw, _, destination, _ = CAPTURED[3]
    d.host.client.mycall = destination
    d.host.kiss.incoming.append(kiss.encode(bytes.fromhex(raw), 1))
    d.tick()
    saved = d.session.mail_collection._load()
    d.do('/vmail relay off')
    d.host.kiss.incoming.append(kiss.encode(bytes.fromhex(raw), 1))
    d.tick()
    assert d.session.mail_collection._load() == saved
    assert len(d.kinds('broadcast_rx')) == 2
    assert len(d.kinds('vmail_notice')) == 1
    d.do('/vmail pending')
    assert any('urgent' in line for line in d.said)


@pytest.mark.parametrize('configured,canonical', [('W9SSJ-0', 'W9SSJ'), ('W9SSJ-01', 'W9SSJ-1')])
def test_notice_receive_listing_receipt_and_dismiss_use_canonical_local_identity(configured, canonical):
    from hfmodem.contact.broadcast import encode_ui
    d = Desk(kiss=True)
    d.host.client.mycall = configured
    d.host.kiss.incoming.append(kiss.encode(encode_ui('K1ABC-1', canonical, b'<SMU>'), 1))
    d.tick()
    assert d.session.mail_collection._load()[0]['destination'] == canonical
    d.do('/vmail pending')
    assert any('K1ABC-1: mail notice' in line for line in d.said)
    d.session.peer = 'K1ABC-01'
    d.session.mail_collection.received()
    assert d.session.mail_collection._load()[0]['received'] == 1
    d.do('/vmail dismiss K1ABC-01')
    assert not d.session.mail_collection._load()
    d.host.kiss.incoming.append(kiss.encode(encode_ui(canonical, canonical, b'<SMP>'), 1))
    d.tick()
    assert not d.session.mail_collection._load()
