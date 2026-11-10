# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import errno
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call
import pytest
from hfmodem.bbs.store import Store, OwnerLock, OwnershipError


@pytest.mark.parametrize('error', [None, errno.EACCES, errno.EAGAIN, errno.EDEADLK, errno.EIO])
def test_windows_owner_lock_is_nonblocking_on_byte_zero_and_closes(tmp_path, monkeypatch, error):
    from hfmodem.bbs import store
    calls = Mock()
    calls.open.return_value = 42
    if error is not None:
        calls.locking.side_effect = OSError(error, 'lock failed')
    monkeypatch.setattr(store, 'os', SimpleNamespace(name='nt', O_RDWR=os.O_RDWR,
        O_CREAT=os.O_CREAT, SEEK_SET=os.SEEK_SET, open=calls.open,
        lseek=calls.lseek, close=calls.close))
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(locking=calls.locking, LK_NBLCK=2))
    path = (tmp_path / 'owner.lock').resolve()
    if error is None:
        with OwnerLock(path) as owner:
            assert owner.fileno() == 42
        owner.close()
        with pytest.raises(ValueError, match='closed'):
            owner.fileno()
    else:
        exception = OSError if error == errno.EIO else OwnershipError
        with pytest.raises(exception, match='lock failed' if error == errno.EIO else 'already owned'):
            OwnerLock(path)
    assert calls.mock_calls == [call.open(path, os.O_RDWR | os.O_CREAT, 0o600),
        call.lseek(42, 0, os.SEEK_SET), call.locking(42, 2, 1), call.close(42)]


def test_owner_lock_contention_and_close_allow_reacquisition(tmp_path):
    path = tmp_path / 'owner.lock'
    with OwnerLock(path):
        with pytest.raises(OwnershipError, match='already owned'):
            OwnerLock(path)
    assert path.exists()
    with OwnerLock(path):
        pass


def test_player_preferences_across_modes_and_restart(tmp_path):
    with Store(tmp_path) as store:
        player = store.player("k1abc-1", "varac")
        assert store.player("K1ABC-2", "local").id == player.id
        store.set_preferences(player.id, expert=True, color="ansi16", pause=False, page_length=12)
        with pytest.raises(OwnershipError):
            Store(tmp_path)
    with Store(tmp_path) as store:
        assert store.player("K1ABC").id == player.id
        assert store.preferences(player.id) == dict(expert=True, color="ansi16", pause=False, page_length=12)
        assert store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 3


def test_alias_existing_player_requires_explicit_merge(tmp_path):
    with Store(tmp_path) as store:
        first = store.player("K1ABC")
        second = store.player("W9SSJ")
        with pytest.raises(ValueError, match="merge"):
            store.player("K1ABC", aliases={"K1ABC": "W9SSJ"})
        store.merge_players(second.id, first.id)
        assert store.player("K1ABC").id == second.id
        assert store.player("K1ABC", aliases={"K1ABC": "W9SSJ"}).id == second.id


def test_preference_validation(tmp_path):
    with Store(tmp_path) as store:
        player = store.player("K1ABC")
        for update in ({"page_length": 0}, {"expert": 1}, {"color": "osc"}, {"other": True}):
            with pytest.raises(ValueError):
                store.set_preferences(player.id, **update)


@pytest.mark.skipif(os.name == "nt", reason="Windows has no pass_fds; its byte lock belongs to the process that took it")
def test_inherited_world_lock_survives_parent_descriptor_close(tmp_path):
    path = tmp_path / "world.lock"
    owner = OwnerLock(path)
    child = subprocess.Popen([sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(owner.fileno(),), text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        owner.close()
        with pytest.raises(OwnershipError):
            OwnerLock(path)
    finally:
        child.communicate("")
    with OwnerLock(path):
        pass


def test_unknown_schema_refuses_and_releases_owner(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "players.sqlite")
    db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(ValueError, match="schema"):
        Store(tmp_path)
    with OwnerLock(tmp_path / "owner.lock"):
        pass
