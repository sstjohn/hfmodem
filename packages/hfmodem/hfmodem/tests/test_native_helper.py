# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Missing native builds require an explicit pure-run declaration."""
import sys
from types import SimpleNamespace

import pytest

from hfmodem.tests import native

DSP_COMMAND = 'cd packages/hfmodem && HFMODEM_NATIVE=1 python setup.py build_rust --inplace --release'
P4LONG_COMMAND = 'cargo build --offline --release --manifest-path packages/hfmodem/native/p4long/Cargo.toml'


@pytest.mark.parametrize('stale', [False, True])
def test_dsp_missing(monkeypatch, stale):
    monkeypatch.setitem(sys.modules, 'hfmodem.core._dsp', SimpleNamespace() if stale else None)
    monkeypatch.delenv('HFMODEM_NATIVE', raising=False)
    with pytest.raises(pytest.fail.Exception) as failure:
        native.dsp()
    assert 'hfmodem.core._dsp' in str(failure.value)
    assert 'packages/hfmodem/build-native.sh' in str(failure.value)
    assert DSP_COMMAND in str(failure.value)
    assert 'HFMODEM_NATIVE=0' in str(failure.value)

    monkeypatch.setenv('HFMODEM_NATIVE', '0')
    with pytest.raises(pytest.skip.Exception, match='HFMODEM_NATIVE=0'):
        native.dsp()


def test_dsp_present(monkeypatch):
    module = SimpleNamespace(sliding_magnitudes=object())
    monkeypatch.setitem(sys.modules, 'hfmodem.core._dsp', module)
    monkeypatch.delenv('HFMODEM_NATIVE', raising=False)
    assert native.dsp() is module


def test_p4long_missing(monkeypatch, tmp_path):
    from hfmodem.shrike import p4longnative

    monkeypatch.setattr(p4longnative, 'backend', None)
    monkeypatch.setattr(p4longnative, '_library_path', lambda: tmp_path / 'absent.so')
    monkeypatch.delenv('HFMODEM_NATIVE', raising=False)
    with pytest.raises(pytest.fail.Exception) as failure:
        native.p4long()
    assert 'hfmodem.shrike._p4long_native' in str(failure.value)
    assert 'packages/hfmodem/build-native.sh' in str(failure.value)
    assert P4LONG_COMMAND in str(failure.value)
    assert 'HFMODEM_NATIVE=0' in str(failure.value)

    monkeypatch.setenv('HFMODEM_NATIVE', '0')
    with pytest.raises(pytest.skip.Exception, match='HFMODEM_NATIVE=0'):
        native.p4long()


@pytest.mark.parametrize('pure', [False, True])
def test_p4long_broken(monkeypatch, tmp_path, pure):
    from hfmodem.shrike import p4longnative

    path = tmp_path / 'broken.so'
    path.touch()
    monkeypatch.setattr(p4longnative, 'backend', None)
    monkeypatch.setattr(p4longnative, '_library_path', lambda: path)
    monkeypatch.setenv('HFMODEM_NATIVE', '0' if pure else '1')
    with pytest.raises(AssertionError, match='failed ABI validation or self-test'):
        native.p4long()


def test_p4long_present(monkeypatch):
    from hfmodem.shrike import p4longnative

    backend = object()
    monkeypatch.setattr(p4longnative, 'backend', backend)
    monkeypatch.delenv('HFMODEM_NATIVE', raising=False)
    assert native.p4long() is backend
