# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Require native builds unless a missing extension is explicitly allowed."""
import importlib
import os

import pytest


def _missing(extension, command):
    reason = f'{extension} not built or stale'
    if os.environ.get('HFMODEM_NATIVE') == '0':
        pytest.skip(f'{reason}; HFMODEM_NATIVE=0 declares a pure run')
    pytest.fail(
        f'{reason}; run packages/hfmodem/build-native.sh, '
        f'or build manually: {command}. HFMODEM_NATIVE=0 declares a pure run',
        pytrace=False,
    )


def dsp():
    command = 'cd packages/hfmodem && HFMODEM_NATIVE=1 python setup.py build_rust --inplace --release'
    try:
        module = importlib.import_module('hfmodem.core._dsp')
    except ImportError:
        _missing('hfmodem.core._dsp', command)
    if not hasattr(module, 'sliding_magnitudes'):
        _missing('hfmodem.core._dsp', command)
    return module


def p4long():
    from hfmodem.shrike import p4longnative

    if p4longnative.backend is None:
        # A present but broken library must fail, even in a pure run.
        assert not p4longnative._library_path().is_file(), (
            'Built P4 LONG library failed ABI validation or self-test')
        _missing('hfmodem.shrike._p4long_native',
                 'cargo build --offline --release --manifest-path packages/hfmodem/native/p4long/Cargo.toml')
    return p4longnative.backend
