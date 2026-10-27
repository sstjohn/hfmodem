#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit offline Xara interop: setup, selftest, phy, session, or control."""
import os
import sys

CAPS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        print('Use check.sh setup to fetch/build the pinned peer; all other commands require setup.')
        print('Use check.sh COMMAND --help for options. No command is part of default builds/tests.')
        return 0 if len(sys.argv) > 1 else 2
    for name in CAPS:
        os.environ[name] = '1'
    try:
        priority = os.getpriority(os.PRIO_PROCESS, 0)
        if priority < 19:
            os.nice(19 - priority)
    except (AttributeError, OSError) as error:
        print(f'warning: unable to set nice 19: {error}', file=sys.stderr)
    kind = sys.argv[1]
    if kind == 'setup':
        import prepare
        return prepare.main(sys.argv[2:])
    if kind == 'selftest':
        import paths
        import pytest
        return pytest.main([str(paths.HERE / 'checks.py'),
                            str(paths.HERE / 'checks_ipc.py'), *sys.argv[2:]])
    if kind not in ('phy', 'session', 'control'):
        print(f'unknown command: {kind}', file=sys.stderr)
        return 2
    try:
        import run
        return run.main()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    try:
        status = main()
    except ModuleNotFoundError as error:
        print(f'Missing Python dependency: {error.name}. Select a prepared environment with '
              'HFMODEM_INTEROP_PYTHON=/path/to/venv/bin/python and rerun check.sh.', file=sys.stderr)
        status = 2
    raise SystemExit(status)
