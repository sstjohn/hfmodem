# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import os
from pathlib import Path


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    path = Path(path)
    missing = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    sync_directory(path)
    sync_directory(path.parent)
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        sync_directory(directory)
        # The child's fsync cannot persist the parent entry naming it.
        sync_directory(directory.parent)
