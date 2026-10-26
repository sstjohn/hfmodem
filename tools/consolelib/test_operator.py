# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fresh operator storage and refusals, with no hardware or private session data."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("fake_claim")

from consolelib import arm, claim, mail, modes, operator, power, roster, station, vara_evidence


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))
    cfg = tmp_path / 'station.toml'
    cfg.write_text('''schema=1
[station]
mycall="N0CALL"
grid="EM12jt"
control="local"
regulatory="unregulated"
because="offline release test"
transmit=false
[rig]
model="ft891"
centre_hz=7101500
''')
    monkeypatch.setenv('HFMODEM_STATION', str(cfg))
    monkeypatch.delenv('HFMODEM_DELIVERY_TO', raising=False)
    return tmp_path


def test_first_use_empty_and_explicit_roster_import(fresh):
    assert operator.run(['init']) == 0
    for name in modes.KNOWN:
        mode = modes.resolve(name)
        assert roster.peers(mode) == []
        with pytest.raises(ValueError, match='No .* peers'):
            roster.peer_for(mode, 'N1TEST', 40)
        with claim.lease(mode) as held:
            assert claim.hardware.inherited(9).offset() == held.offset()
            assert claim.hardware.claim_path().is_file()
            assert not mode.lock.exists()  # One station claim replaces per-mode locks.
    peer = dict(id='N1TEST-40', call='N1TEST', band_m=40, centre_hz=7101500, dial_hz=7100000)
    path = fresh / 'roster.json'
    path.write_text(json.dumps({'peers': [peer]}))
    assert operator.run(['import-roster', 'pactor', str(path)]) == 0
    assert roster.peer_for(modes.resolve('pactor'), 'N1TEST', 40) == peer
    with pytest.raises(ValueError, match='already contains'):
        operator.run(['import-roster', 'pactor', str(path)])
    peer['bw_hz'] = 2300
    peer['dial_hz'] += 1
    path.write_text(json.dumps({'peers': [peer]}))
    with pytest.raises(ValueError, match='dial must'):
        operator.run(['import-roster', 'vara', str(path)])


def test_delivery_uses_configured_identity_and_frozen_record(fresh, monkeypatch):
    now = datetime(2026, 10, 10, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='delivery_to'):
        mail.delivery_message('vara', {'call': 'N1TEST'}, fresh, now)
    monkeypatch.setenv('HFMODEM_DELIVERY_TO', 'recipient@example.test')
    record = mail.delivery_message('vara', {'call': 'N1TEST'}, fresh, now)
    assert record['sender'] == 'N0CALL'
    assert record['recipient'] == 'recipient@example.test'
    monkeypatch.setenv('HFMODEM_DELIVERY_TO', 'changed@example.test')
    monkeypatch.setenv('HFMODEM_STATION', str(fresh / 'absent.toml'))
    mail.write(record)
    data = Path(record['path']).read_bytes()
    assert b'N0CALL' in data and b'recipient@example.test' in data
    assert b'changed@example.test' not in data


def test_source_identity_has_no_private_baseline_dependency(fresh, monkeypatch):
    root = fresh / 'checkout'
    runtime = root / 'packages/hfmodem/hfmodem'
    runtime.mkdir(parents=True)
    code = runtime / 'runtime.py'
    code.write_text('value=1')
    monkeypatch.setattr(station, 'ROOT', root)
    monkeypatch.setattr(arm, 'STAGING', None)
    first = arm.source_evidence()
    assert first['runtime_files'] == 1 and first['baseline_available'] is False
    code.write_text('value=2')
    assert arm.source_evidence()['working_tree_sha256'] != first['working_tree_sha256']


def test_optional_power_refuses_before_socket_and_checks_mac(fresh, monkeypatch):
    monkeypatch.setattr(socket, 'socket', lambda *a, **k: pytest.fail('opened socket'))
    with pytest.raises(ValueError, match='explicit power.mac'):
        power.find()
    assert not power.identified({'alias': 'radio', 'mac': '00:00:00:00:00:01'},
                                'radio', '00:00:00:00:00:02')
    assert power.identified({'alias': 'radio', 'mac': '00:00:00:00:00:02'},
                            'radio', '00:00:00:00:00:02')


def test_missing_witness_is_immediate_and_optional(fresh, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('started witness'))
    assert station.place_kiwi({}, fresh) is None


def test_shipped_vara_parser_distinguishes_payload_and_next_level():
    report = vara_evidence.analyze_lines([
        ('test', 1, 'BW2300 BITRATE (9) 100 bps TX'),
        ('test', 2, 'tx DATA over #1 at record 8 (30 payload bytes, ok)'),
        ('test', 3, 'solicited pending over #1 acknowledged'),
    ], bw=2300)
    assert report['by_bw']['2300']['tx_keyed'] == [9]
    assert report['by_bw']['2300']['peer_acknowledged'] == [9]
    assert report['mail_status'] == 'unknown'


def test_kiwi_adapter_uses_usb_and_decodes_big_endian(monkeypatch):
    from consolelib import kiwi_client
    calls = []
    stream = SimpleNamespace(sample_rate=12000, blocks=lambda: iter([b'\x40\x00\xc0\x00']), close=lambda: calls.append('close'))
    monkeypatch.setattr(kiwi_client.KiwiStream, 'open', lambda *a, **k: calls.append((a,k)) or stream)
    session = kiwi_client.KiwiSession('example.test', 8073, 7100000, passband=(0,4000)).start()
    assert calls[0][1]['hi_hz'] == 4000
    assert list(session.blocks())[0].samples.tolist() == [0.5, -0.5]
    assert calls[0][0] == ('example.test', 8073, 7100)
    assert calls[0][1]['mode'] == 'usb'
    session.close()
    assert calls[-1] == 'close'


def test_transmit_disabled_refuses_before_capture_or_devices(fresh, monkeypatch):
    monkeypatch.setattr(arm, 'capture_plan', lambda *a: pytest.fail('allocated a capture'))
    with pytest.raises(ValueError, match='transmit=false'):
        arm.fly(None, {'env': {}})


def test_public_offline_storage_needs_no_private_rosters(tmp_path, monkeypatch):
    from . import conftest

    monkeypatch.setattr(conftest, 'FLOWN', tmp_path / 'absent-private-records')
    root = conftest.seed_operator_data(tmp_path / 'public-data', private=False)
    for name in modes.KNOWN:
        directory = root / 'working' / name
        assert directory.is_dir()
        assert not (directory / 'peers.json').exists()
    assert list((root / 'working' / 'vara' / 'messages').iterdir()) == []
