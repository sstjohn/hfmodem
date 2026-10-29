# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem import cli
from hfmodem.station import process, service
from hfmodem.tests.station.test_process import configuration


@pytest.mark.parametrize('flags', [[], ['--detach'], ['--preflight']])
def test_station_parser(flags, tmp_path):
    path = tmp_path / 'station.toml'
    args = cli.build_parser().parse_args(['station', str(path), *flags])
    assert args.file == path
    assert args.detach == ('--detach' in flags)
    assert args.preflight == ('--preflight' in flags)


def test_station_stop_parser():
    args = cli.build_parser().parse_args(['station', 'stop', 'station.toml'])
    assert args.file == Path('stop') and args.target == Path('station.toml')


def test_station_file_defaults_from_environment(tmp_path, monkeypatch):
    path = tmp_path / 'env.toml'
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    seen = []
    def refuse(file):
        seen.append(file)
        raise OSError('refused')
    monkeypatch.setattr(service, 'Service', refuse)
    assert cli.main(['station']) == 2
    assert seen == [path]


def test_version_imports_no_numerics():
    probe = "import sys; from hfmodem import cli; cli.main(['version']); print(sorted({'numpy', 'scipy'} & set(sys.modules)))"
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    out = subprocess.run([sys.executable, '-c', probe], env=env, capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[-1] == '[]'


def test_station_stop_needs_no_control(tmp_path, monkeypatch):
    from hfmodem.station import client
    calls = []
    class Client:
        def command(self, kind, **kwargs):
            calls.append((kind, kwargs))
            return {'result': {'control_id': 'fake-control', 'generation': 7}}
    path = tmp_path / 'station.toml'
    def connect(target):
        assert target == path
        return Client()
    monkeypatch.setattr(client, 'connect', connect)
    assert cli.main(['station', 'stop', str(path)]) == 0
    assert calls == [('shutdown', {'wait': True, 'timeout': 45})]


def test_listener_binds_before_config_load(tmp_path, monkeypatch):
    path = tmp_path / 'station.toml'
    path.write_text('[station]\n')
    calls = []
    monkeypatch.setattr(service.Service, 'bind', lambda self: calls.append('bind'))
    monkeypatch.setattr(service.Service, 'close', lambda self: calls.append('close') or self.closing.set())
    monkeypatch.setattr(cli, '_config', lambda path: calls.append('config') or configuration())
    monkeypatch.setattr(process.Station, 'run', lambda self, **kw: calls.append(kw) or 0)
    assert cli.main(['station', str(path), '--detach']) == 0
    assert calls == ['bind', 'config', {'detach': True, 'preflight': False}, 'close']


@pytest.mark.parametrize('code', [3, 4])
def test_bind_conflict_never_loads_devices(tmp_path, monkeypatch, code):
    path = tmp_path / 'station.toml'
    path.write_text('[station]\n')
    def bind(self):
        raise service.BindConflict('already held', code)
    monkeypatch.setattr(service.Service, 'bind', bind)
    monkeypatch.setattr(cli, '_config', lambda path: pytest.fail('loaded before bind'))
    assert cli.main(['station', str(path), '--detach']) == code


@pytest.mark.parametrize('kind', ['rig', 'audio'])
def test_foreground_device_errors_keep_base_message_and_exit(tmp_path, monkeypatch, capsys, kind):
    from hfmodem.core.rig import RigError
    from hfmodem.core.audio import AudioError
    path = tmp_path / 'station.toml'
    path.write_text('[station]\n')
    monkeypatch.setattr(service.Service, 'bind', lambda self: None)
    monkeypatch.setattr(cli, '_config', lambda path: configuration())
    error = RigError if kind == 'rig' else AudioError
    def fail(self, **kw):
        raise error('fake device failed')
    monkeypatch.setattr(process.Station, 'run', fail)
    assert cli.main(['station', str(path)]) == 1
    assert 'the radio is not usable: fake device failed' in capsys.readouterr().err


@pytest.mark.parametrize('contents', [None, '[station', '[station]\nport = "bad"\n'])
def test_bad_station_file_is_configuration_exit(tmp_path, capsys, contents):
    path = tmp_path / 'station.toml'
    if contents is not None:
        path.write_text(contents)
    assert cli.main(['station', str(path)]) == 2
    message = capsys.readouterr().err
    assert str(path) in message and 'Traceback' not in message
