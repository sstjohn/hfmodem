"""Receive-first setup and gateway selection without opening a radio."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hfmodem.core import config, rigs

ROOT = Path(__file__).resolve().parents[5]
pytestmark = pytest.mark.skipif(not (ROOT / 'tools/consolelib/onboarding.py').exists(), reason='operator tools tested in checkout')

@pytest.fixture
def onboarding(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from consolelib import onboarding
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('HFMODEM_STATION', str(tmp_path / 'station.toml'))
    return onboarding


def station(onboarding, rig='ft891', licence='general', transmit=True):
    path = Path(os.environ['HFMODEM_STATION'])
    onboarding.write_station(path, rig=rig, call='W1TEST', grid='FN31', cat='/dev/fixture0', ptt='/dev/fixture1', audio_in='USB Fixture', audio_out='USB Fixture', licence=licence, regulatory='part97', because='', drive=.25, power=20, transmit=transmit)
    return config.load(path)


def test_ft891_saved_connections_and_audio(onboarding):
    station(onboarding)
    args = argparse.Namespace(rig='ft891', serial=None, ptt_port=None, audio_in=None, audio_out=None)
    rigs.flight_defaults(args)
    assert (args.serial, args.ptt_port, args.audio_in, args.audio_out) == ('/dev/fixture0', '/dev/fixture1', 'USB Fixture', 'USB Fixture')
    assert Path(os.environ['HFMODEM_STATION']).stat().st_mode & 0o777 == 0o600


def test_no_laptop_audio_selected(onboarding):
    devices = [dict(name='Built-in Microphone', max_input_channels=1, max_output_channels=0)]
    with pytest.raises(ValueError, match='No USB'):
        onboarding.audio_choices('input', devices)
    devices.append(dict(name='USB Audio CODEC', max_input_channels=1, max_output_channels=2))
    assert onboarding.audio_choices('input', devices) == [('USB Audio CODEC', 'USB Audio CODEC')]


CSV = 'Callsign,Frequency,Mode,Hours,GridSquare,QTH\nW1AAA,7101.5 kHz,PACTOR 3,00-24,FN31,Nearby\nW1BBB,28151.5 kHz,PACTOR 3,00-24,FN32,Ten metres\n'


def test_technician_gateway_filter(onboarding):
    from consolelib.gateways import normalize
    peers = normalize(CSV, 'pactor')
    assert [p['call'] for p in onboarding.eligible(peers, station(onboarding, licence='technician'), 'pactor')] == ['W1BBB']
    assert {p['call'] for p in onboarding.eligible(peers, station(onboarding), 'pactor')} == {'W1AAA', 'W1BBB'}


def test_bad_refresh_preserves_saved_roster(onboarding, tmp_path):
    from consolelib import gateways, operator
    good = tmp_path / 'good.csv'; good.write_text(CSV)
    gateways.refresh('pactor', csv_path=good)
    assert (operator.data_root() / 'cache/winlink-pactor-gateways.csv').read_text() == CSV
    target = operator.mode_root('pactor') / 'peers.json'; before = target.read_bytes()
    bad = tmp_path / 'bad.csv'; bad.write_text('error page')
    with pytest.raises(ValueError):
        gateways.refresh('pactor', csv_path=bad)
    assert target.read_bytes() == before


def test_vara_hf_width_and_fm_filter(onboarding):
    from consolelib.gateways import normalize
    peers = normalize('Callsign,Frequency,Mode\nW1AAA,14101.5 kHz,VARA HF 2750\nW1AAA,145050 kHz,VARA FM 2300\n', 'vara')
    assert len(peers) == 1
    assert peers[0]['bw_hz'] == 2750 and peers[0]['dial_hz'] == 14100000


def test_password_restored_and_receive_first(onboarding, tmp_path, monkeypatch):
    station(onboarding)
    csv = tmp_path / 'peers.csv'; csv.write_text(CSV)
    monkeypatch.setenv('WINLINK_PASSWORD', 'fixture secret')
    monkeypatch.setattr(onboarding, 'dependencies', lambda *args: None)
    seen = []
    with patch('hfmodem.operator.bootstrap.console', side_effect=lambda words: seen.append(words) or 0):
        assert onboarding.winlink(['pactor', 'W1AAA', '--csv', str(csv), '--execute']) == 0
    assert '--p3-mail' in seen[0] and '--send' not in seen[0]
    assert os.environ['WINLINK_PASSWORD'] == 'fixture secret'


def test_pactor_env_password_does_not_require_development_file(onboarding, monkeypatch):
    from consolelib import pactor
    monkeypatch.setenv('WINLINK_PASSWORD', 'fixture secret')
    assert pactor._password_flags(argparse.Namespace(mail_password_file=None)) == []
    plan = {'command': ['--mail-fetch']}
    with pactor.ready(None, plan):
        assert os.environ['WINLINK_PASSWORD'] == 'fixture secret'


def test_empty_password_file_refused_before_open(onboarding, tmp_path):
    from consolelib.station import winlink_password
    path = tmp_path / 'password'; path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        with winlink_password(path):
            pytest.fail('credential context entered')


def test_native_text_message_round_trip(onboarding, capsys):
    from hfmodem.winlink.message import Message
    cfg = station(onboarding)
    path = onboarding.compose_mail(cfg, 'operator@example.net', 'Radio test', 'Hello from the radio.\nSecond line.')
    msg = Message.parse(path.read_bytes())
    assert msg.sender == 'W1TEST' and msg.recipients == ['SMTP:operator@example.net']
    assert msg.body == b'Hello from the radio.\r\nSecond line.'
    destination = Path(os.environ['HFMODEM_DATA_DIR']) / 'working/vara/mail' / path.name
    destination.parent.mkdir(parents=True); destination.write_bytes(path.read_bytes())
    assert onboarding.winlink(['--inbox']) == 0
    assert 'Hello from the radio.' in capsys.readouterr().out


def test_public_plain_vara_service_uses_standard_hf_width(onboarding):
    from consolelib.gateways import normalize
    peers = normalize('Callsign,Frequency,Mode\nW1AAA,"14,101.500 KHz",VARA\n', 'vara')
    assert peers[0]['bw_hz'] == 2750


def test_pactor3_wizard_skips_pactor1_only_gateway(onboarding):
    from consolelib.gateways import normalize
    cfg = station(onboarding)
    peers = normalize('Callsign,Frequency,Mode\nW1AAA,7101.5 kHz,Pactor 1\nW1BBB,7105.5 kHz,"Pactor 3,4"\n', 'pactor')
    assert [p['call'] for p in onboarding.eligible(peers, cfg, 'pactor')] == ['W1BBB']


def test_system_install_failure_has_actionable_message(onboarding, monkeypatch):
    import subprocess
    monkeypatch.setattr(onboarding.shutil, 'which', lambda name: '/fixture/brew' if name == 'brew' else None)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(onboarding.subprocess, 'run', fail)
    with pytest.raises(ValueError, match='did not finish'):
        onboarding.dependencies(True)


@pytest.mark.parametrize('rig', ['ftdx10', 'ft891'])
def test_data_preparation_is_verified_before_keying(onboarding, rig, monkeypatch):
    from consolelib import station as radio
    from types import SimpleNamespace
    station(onboarding, rig=rig)
    calls = []
    def execute(argv, **kwargs):
        calls.append((argv, kwargs['input']))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(radio.subprocess, 'run', execute)
    expected = radio._MAIL_DATA_SETTINGS[rig]
    monkeypatch.setattr(radio, 'rig_settings', lambda **kwargs: dict(menu=expected, levels={'RFPOWER': '.2'}, mode='PKTUSB 3000'))
    plan = {'env': {'HFMODEM_STATION': str(Path(os.environ['HFMODEM_STATION']))}}
    radio.prepare_radio(plan)
    assert len(calls) == 1
    assert 'w MD0C;' in calls[0][1] and 'L RFPOWER 0.2' in calls[0][1]
    assert not any(line.startswith('T ') for line in calls[0][1].splitlines())
    monkeypatch.setattr(radio, 'rig_settings', lambda **kwargs: dict(menu=expected, levels={'RFPOWER': '.5'}, mode='PKTUSB 3000'))
    with pytest.raises(ValueError, match='instead of'):
        radio.prepare_radio(plan)
