# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Portable runs do not inspect a modem-line pair, even when one is present."""
from types import SimpleNamespace

import pytest

from hfmodem.tests.coexist import conftest as coexist


REASON = 'modem-line proof not requested (portable software validation)'


@pytest.mark.parametrize('opt_in', [None, '', '0', 'false', 'true', 'yes', '01', '1 ', ' 1', '2'])
def test_portable_fixture_skips_before_platform_pair_or_device_probe(monkeypatch, opt_in):
    class PortableEnvironment(dict):
        def get(self, key, default=None):
            assert key == 'HFMODEM_COEXIST_LINE_PROOF', 'portable fixture inspected pair configuration'
            return super().get(key, default)

    env = PortableEnvironment(HFMODEM_COEXIST_TTY0TTY='/dev/tnt0,/dev/tnt1')
    if opt_in is not None:
        env['HFMODEM_COEXIST_LINE_PROOF'] = opt_in

    class PortablePlatform:
        @property
        def platform(self):
            pytest.fail('portable fixture inspected the platform')

    class InstalledPair:
        def __init__(self, path):
            pytest.fail('portable fixture probed an installed pair')

        def exists(self):
            return True

    monkeypatch.setattr(coexist, 'os', SimpleNamespace(environ=env))
    monkeypatch.setattr(coexist, 'sys', PortablePlatform())
    monkeypatch.setattr(coexist, 'Path', InstalledPair)
    with pytest.raises(pytest.skip.Exception) as skipped:
        next(coexist.tty0tty_pair.__wrapped__())
    assert str(skipped.value) == REASON


def configure_requested(monkeypatch, *, platform='linux', pair='/dev/tnt0,/dev/tnt1', exists=True):
    env = {'HFMODEM_COEXIST_LINE_PROOF': '1', 'HFMODEM_COEXIST_TTY0TTY': pair}
    probes = []

    class PairPath:
        def __init__(self, path):
            probes.append(path)

        def exists(self):
            return exists

    monkeypatch.setattr(coexist, 'os', SimpleNamespace(environ=env))
    monkeypatch.setattr(coexist, 'sys', SimpleNamespace(platform=platform))
    monkeypatch.setattr(coexist, 'Path', PairPath)
    return probes


def test_explicit_request_checks_and_yields_software_pair(monkeypatch):
    probes = configure_requested(monkeypatch)
    fixture = coexist.tty0tty_pair.__wrapped__()
    assert next(fixture) == ('/dev/tnt0', '/dev/tnt1')
    assert probes == ['/dev/tnt0', '/dev/tnt1']
    with pytest.raises(StopIteration):
        next(fixture)


@pytest.mark.parametrize('pair', ['/dev/tnt0', '/dev/tnt0,/dev/tnt0',
                                  '/dev/ttyS0,/dev/tnt1', '/dev/tnt0extra,/dev/tnt1'])
def test_explicit_request_keeps_pair_validation_before_device_probe(monkeypatch, pair):
    probes = configure_requested(monkeypatch, pair=pair)
    with pytest.raises(pytest.fail.Exception, match='only a distinct tty0tty'):
        next(coexist.tty0tty_pair.__wrapped__())
    assert probes == []


def test_explicit_request_missing_pair_fails(monkeypatch):
    probes = configure_requested(monkeypatch, exists=False)
    with pytest.raises(pytest.fail.Exception, match='modem-line proof requested but tty0tty is not installed'):
        next(coexist.tty0tty_pair.__wrapped__())
    assert probes == ['/dev/tnt0']


def test_explicit_request_preserves_macos_deferral_before_pair_probe(monkeypatch):
    probes = configure_requested(monkeypatch, platform='darwin')
    with pytest.raises(pytest.skip.Exception, match='macOS proof needs attended hw'):
        next(coexist.tty0tty_pair.__wrapped__())
    assert probes == []
