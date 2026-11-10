# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fault injection at file and directory durability boundaries on either platform."""
from hfmodem.contact import mail_storage


def instrument_syncs(monkeypatch, observe=None, *, file=None, directory=None):
    fsync = mail_storage.os.fsync
    directory_sync = mail_storage.sync_directory
    in_directory = False

    def sync_file(fd):
        if in_directory:
            return fsync(fd)
        observer = file or observe
        return observer(fsync, fd) if observer else fsync(fd)

    def sync_directory(path):
        nonlocal in_directory
        in_directory = True
        try:
            observer = directory or observe
            return observer(directory_sync, path) if observer else directory_sync(path)
        finally:
            in_directory = False

    monkeypatch.setattr(mail_storage.os, 'fsync', sync_file)
    monkeypatch.setattr(mail_storage, 'sync_directory', sync_directory)
