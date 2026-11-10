# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Per-user directories: XDG on POSIX, %LOCALAPPDATA%\\hfmodem on Windows."""
import os
from pathlib import Path


def windows_dir(kind):
    return Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local') / 'hfmodem' / kind


def _user_dir(xdg, fallback, kind):
    if os.name == 'nt':
        return windows_dir(kind)
    return Path(os.environ.get(xdg, Path.home() / fallback)) / 'hfmodem'


def installed_data():
    return _user_dir('XDG_DATA_HOME', '.local/share', 'data')


def config_dir():
    return _user_dir('XDG_CONFIG_HOME', '.config', 'config')


def cache_dir():
    return _user_dir('XDG_CACHE_HOME', '.cache', 'cache')
