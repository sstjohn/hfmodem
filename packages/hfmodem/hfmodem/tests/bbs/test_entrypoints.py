# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import sys
from types import ModuleType

import pytest

from hfmodem import cli
from hfmodem.bbs.config import BBSConfig
from hfmodem.bbs.service import Service, catalog_root, open_service


def test_bbs_local_dispatch_never_enters_station_claim(monkeypatch):
    local = ModuleType('hfmodem.bbs.frontends.local')
    calls = []
    local.main = lambda argv: calls.append(argv) or 17
    monkeypatch.setitem(sys.modules, local.__name__, local)
    assert cli.main(['bbs', 'local', '--config', 'board.toml', '--peer', 'K1ABC']) == 17
    assert calls == [['--config', 'board.toml', '--peer', 'K1ABC']]


def test_bbs_help_lists_available_offline_entry(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['bbs', '--help'])
    assert exc.value.code == 0
    assert 'local' in capsys.readouterr().out


def test_door_administration_dispatch(monkeypatch):
    install = ModuleType('hfmodem.bbs.install')
    calls = []
    install.main = lambda argv: calls.append(argv) or 0
    monkeypatch.setitem(sys.modules, install.__name__, install)
    assert cli.main(['bbs', 'doors', '--root', '/tmp/board', 'list']) == 0
    assert calls == [['--root', '/tmp/board', 'list']]


def test_generic_catalog_refuses_second_mode_specific_root(tmp_path):
    config = BBSConfig(state_root=tmp_path / 'state', export_root=tmp_path / 'shared')
    assert catalog_root(config, tmp_path / 'shared') == config.export_root
    assert catalog_root(config) == config.export_root
    with pytest.raises(ValueError, match='conflicts'):
        catalog_root(config, tmp_path / 'varac-only')
    with pytest.raises(ValueError, match='conflicts'):
        catalog_root(BBSConfig(state_root=tmp_path / 'empty'), tmp_path / 'legacy')


def test_service_owns_one_catalog_and_releases_state_owner(tmp_path):
    from hfmodem.bbs.store import Store, OwnershipError
    export = tmp_path / 'files'
    export.mkdir()
    (export / 'welcome.txt').write_text('Welcome to the BBS')
    config = BBSConfig(state_root=tmp_path / 'state', export_root=export)
    with Service(config, games={}) as service:
        assert service.core.catalog is service.catalog
        assert service.catalog.list_files('K1ABC')[0].name == 'welcome.txt'
        with pytest.raises(OwnershipError):
            Store(config.state_root)
    with Store(config.state_root):
        pass


def test_live_reload_changes_limits_without_replacing_identity_or_world(tmp_path):
    path = tmp_path / 'bbs.toml'
    prefix = '[bbs]\nstate_root = "state"\n'
    path.write_text(prefix + 'door_seconds_per_day = 100\n')
    with open_service(path) as service:
        path.write_text(prefix + 'door_seconds_per_day = 0\n')
        updated = service.reload().result(timeout=2)
        # File-reading worker cannot apply changes to owner-thread state.
        assert service.core.config.limits.door_seconds_per_day == 100
        service.core.reload_policy(updated)
        assert service.core.config.limits.door_seconds_per_day == 0
        path.write_text('[bbs]\nstate_root = "other-world"\n')
        with pytest.raises(ValueError, match='cannot change'):
            service.reload().result(timeout=2)
        assert service.store.root == tmp_path / 'state'


def test_console_forwards_interactive_flags_independently(tools, tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from consolelib import varac
    from hfmodem.contact import bbs_server
    path = tmp_path / 'board.toml'
    path.write_text('[bbs]\nstate_root = "state"\n')
    monkeypatch.setattr(bbs_server, 'configured_directory', lambda _: None)
    monkeypatch.setattr(varac.station, 'resolve_selection', lambda: SimpleNamespace(
        CAT_PORT='unused', PTT_PORT='unused', RIG_NAME='ft891'))
    monkeypatch.setattr(varac.station, 'warn_receive_only', lambda _: None)
    monkeypatch.setattr(varac.arm, 'git_head', lambda: 'test-head')
    args = varac.parser().parse_args(['--loc', 'EN50', '--bbs-config', str(path), '--no-bbs'])
    command = varac.plan(args, datetime.now(timezone.utc))['command']
    assert '--no-bbs' in command
    assert command[command.index('--bbs-config') + 1] == str(path)
    args = varac.parser().parse_args(['--loc', 'EN50', '--bbs-config', 'ignored', '--no-bbs-service'])
    command = varac.plan(args, datetime.now(timezone.utc))['command']
    assert '--no-bbs-service' in command and '--bbs-config' not in command
