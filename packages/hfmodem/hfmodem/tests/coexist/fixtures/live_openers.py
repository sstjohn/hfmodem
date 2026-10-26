# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Refresh only independently captured Linux fixture descriptors for lsof."""
from pathlib import Path

from hfmodem.tests.coexist import observer


def linux_openers(participants, descriptors, proc=Path('/proc')):
    """Omit closed resources without turning unknown live discovery into absence."""
    result = {}
    budget = observer._ExitConfirmationBudget()
    for key, captured in descriptors.items():
        pid = int(key)
        identity = participants[key]
        start, uids = identity['start'], tuple(identity['uids'])
        directory = proc / str(pid)
        try:
            observer._linux_identity_unchanged(directory, start, uids)
            if observer._linux_terminal(directory, start, uids, budget=budget):
                continue
            try:
                current = observer.linux_descriptors(pid, proc)
            except (PermissionError, FileNotFoundError, ProcessLookupError) as exc:
                denied = exc if isinstance(exc, PermissionError) else None
                if observer._linux_terminal(directory, start, uids, denied=denied, budget=budget):
                    continue
                raise observer.ObservationError('live descriptor discovery unavailable') from exc
            observer._linux_identity_unchanged(directory, start, uids)
            if observer._linux_terminal(directory, start, uids, budget=budget):
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = ('fd', 'path', 'dev', 'ino')
        expected = {tuple(row[field] for field in fields) for row in captured}
        result[key] = [row for row in current
                       if tuple(row[field] for field in fields) in expected]
    return result
