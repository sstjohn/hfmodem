# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import subprocess
import sys
import pytest
from hfmodem.bbs.store import Store, OwnerLock, OwnershipError


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
