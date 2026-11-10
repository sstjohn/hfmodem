# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import importlib
import os
import signal

import pytest

from hfmodem.core import paths, posix
from hfmodem.core.posix import _Deferred


@pytest.mark.posix_only("asserts the real POSIX modules are what is bound")
def test_posix_binds_the_real_module():
    assert posix.module("fcntl") is importlib.import_module("fcntl")
    assert posix.SIGHUP is signal.SIGHUP


@pytest.mark.posix_only("SIGTERM already interrupts POSIX blocking calls")
def test_on_stop_is_inert_on_posix():
    callbacks, handler = posix._stop_callbacks, signal.getsignal(signal.SIGTERM)
    posix.on_stop(lambda: pytest.fail("POSIX stop callback"))
    assert posix._stop_callbacks is callbacks
    assert signal.getsignal(signal.SIGTERM) is handler


def test_a_missing_module_fails_where_it_is_used_and_not_where_it_is_bound():
    stand_in = _Deferred("hfmodem_no_such_module")
    with pytest.raises(ModuleNotFoundError, match="hfmodem_no_such_module"):
        stand_in.flock
    with pytest.raises(ModuleNotFoundError):
        stand_in.flock = None


def test_user_directories_follow_the_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    if os.name == "nt":
        root = tmp_path / "local" / "hfmodem"
        assert paths.installed_data() == root / "data"
        assert paths.config_dir() == root / "config"
        assert paths.cache_dir() == root / "cache"
    else:
        assert paths.installed_data() == tmp_path / "xdg-data" / "hfmodem"
        assert paths.config_dir() == tmp_path / "xdg-config" / "hfmodem"
        assert paths.cache_dir() == tmp_path / "xdg-cache" / "hfmodem"
