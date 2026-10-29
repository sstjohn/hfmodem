# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.bbs.config import configured_path
from hfmodem.core import config


def station_file(tmp_path, value='"board.toml"'):
    example = Path(__file__).resolve().parents[5] / 'examples/station.toml'
    path = tmp_path / 'station.toml'
    path.write_text(example.read_text() + f'\n[bbs]\nconfig = {value}\n')
    return path


def test_station_reference_and_cli_precedence(tmp_path, monkeypatch):
    path = station_file(tmp_path)
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    assert config.load(path).bbs.config == 'board.toml'
    assert configured_path() == tmp_path / 'board.toml'
    monkeypatch.chdir(tmp_path)
    assert configured_path('override.toml') == tmp_path / 'override.toml'
    assert configured_path('') is None
    monkeypatch.setenv('HFMODEM_STATION', str(tmp_path / 'missing'))
    assert configured_path('override.toml') == tmp_path / 'override.toml'


@pytest.mark.parametrize('value', ['123', 'false', '"bad\\u0000path"'])
def test_invalid_station_reference(tmp_path, value):
    with pytest.raises(ValueError, match='config'):
        config.load(station_file(tmp_path, value))


def test_console_plan_uses_station_reference_and_disable(tmp_path, monkeypatch):
    from tools.consolelib import varac
    from hfmodem.contact import bbs_server
    path = station_file(tmp_path)
    (tmp_path / 'board.toml').write_text('[bbs]\nstate_root = "state"\n')
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    monkeypatch.setattr(bbs_server, 'configured_directory', lambda _: None)
    monkeypatch.setattr(varac.station, 'resolve_selection', lambda: SimpleNamespace(
        CAT_PORT='unused', PTT_PORT='unused', RIG_NAME='ft891'))
    monkeypatch.setattr(varac.station, 'warn_receive_only', lambda _: None)
    monkeypatch.setattr(varac.arm, 'git_head', lambda: 'test-head')
    args = varac.parser().parse_args(['--loc', 'EN50'])
    command = varac.plan(args, datetime.now(timezone.utc))['command']
    assert command[command.index('--bbs-config') + 1] == str(tmp_path / 'board.toml')
    (tmp_path / 'board.toml').unlink()
    args = varac.parser().parse_args(['--loc', 'EN50', '--no-bbs-service'])
    command = varac.plan(args, datetime.now(timezone.utc))['command']
    assert '--no-bbs-service' in command and '--bbs-config' not in command
    with pytest.raises(FileNotFoundError):
        varac.plan(varac.parser().parse_args(['--loc', 'EN50']), datetime.now(timezone.utc))


def test_local_uses_station_reference_without_radio(tmp_path, monkeypatch):
    from hfmodem.bbs.frontends import local
    path = station_file(tmp_path)
    (tmp_path / 'board.toml').write_text('[bbs]\nstate_root = "state"\n')
    monkeypatch.setenv('HFMODEM_STATION', str(path))
    seen = []
    monkeypatch.setattr(local, 'run', lambda core, peer, **kw: seen.append(core.config.state_root) or 0)
    assert local.main(['--peer', 'W9SSJ']) == 0
    assert seen == [tmp_path / 'state']
