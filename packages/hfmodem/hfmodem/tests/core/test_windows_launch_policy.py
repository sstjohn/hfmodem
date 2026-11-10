# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import os
import signal
import sys
from types import SimpleNamespace

import pytest

from hfmodem.core import winproc
from hfmodem.operator import bootstrap


@pytest.mark.parametrize('role,python,own_session,expected', [
    ('native', False, False, 0x08000200),
    ('native', True, False, 0x08000200),
    ('supervisor', True, False, 0x08000200),
    ('one-shot', False, True, 0x08000200),
    ('one-shot', True, True, 0x200),
    ('python', True, True, 0x200),
    ('one-shot', False, False, 0),
    ('python', True, False, 0),
    ('terminal', True, True, 0),
    ('terminal', False, False, 0),
])
def test_creation_flags_by_role(role, python, own_session, expected):
    assert winproc.creation_flags(role, python=python, own_session=own_session) == expected


def test_direct_interpreter_preserves_venv_without_mutating_environment(monkeypatch):
    monkeypatch.setattr(sys, '_base_executable', '/base/python')
    monkeypatch.setattr(sys, 'executable', '/venv/python')
    environment = {'PATH': 'selected', '__PYVENV_LAUNCHER__': 'stale'}
    command, child_env = winproc.python_command(['-m', 'hfmodem', 'console'], env=environment)
    assert command == ['/base/python', '-m', 'hfmodem', 'console']
    assert child_env == {'PATH': 'selected', '__PYVENV_LAUNCHER__': '/venv/python'}
    assert environment['__PYVENV_LAUNCHER__'] == 'stale'


def test_suspended_interpreter_adds_only_the_suspend_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(winproc.subprocess, 'Popen', lambda *a, **kw: seen.append((a, kw)))
    winproc.python_suspended(['-m', 'hfmodem'], creationflags=0x200, env={'test': 'kept'}, stdin=-3)
    assert seen == [(([sys._base_executable, '-m', 'hfmodem'],),
                     dict(env={'test': 'kept', '__PYVENV_LAUNCHER__': sys.executable},
                          close_fds=True, creationflags=0x204, stdin=-3))]


@pytest.mark.parametrize('verb', ['console', 'winlink'])
def test_console_relaunch_is_once_on_same_console_and_returns_status(monkeypatch, verb):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(signal, 'SIGBREAK', 21, raising=False)
    monkeypatch.delenv('HFMODEM_CONSOLE_DIRECT', raising=False)
    calls, handlers = [], {signal.SIGINT: 'int', signal.SIGBREAK: 'break'}

    def set_handler(sig, handler):
        previous, handlers[sig] = handlers[sig], handler
        return previous

    def wait():
        assert all(handler == signal.SIG_IGN for handler in handlers.values())
        return 37

    def popen(command, **options):
        assert all(handler == signal.SIG_IGN for handler in handlers.values())
        calls.append((command, options))
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(signal, 'signal', set_handler)
    monkeypatch.setattr(bootstrap.subprocess, 'Popen', popen)
    monkeypatch.setattr(bootstrap, 'tools_root', lambda: pytest.fail('parent loaded operator'))
    assert getattr(bootstrap, verb)(['--help', 'non-ASCII é']) == 37
    command, options = calls.pop()
    assert command == [sys._base_executable, '-m', 'hfmodem', verb, '--help', 'non-ASCII é']
    assert options == dict(close_fds=True, creationflags=0, stdin=0, stdout=1, stderr=2,
                          env=dict(os.environ, HFMODEM_CONSOLE_DIRECT='1',
                                   __PYVENV_LAUNCHER__=sys.executable))
    assert handlers == {signal.SIGINT: 'int', signal.SIGBREAK: 'break'}
    monkeypatch.setenv('HFMODEM_CONSOLE_DIRECT', '1')
    assert bootstrap._relaunch(verb, []) is None
    assert not calls


@pytest.mark.parametrize('failure', ['spawn', 'wait', 'second-handler'])
def test_relaunch_restores_handlers_after_failure(monkeypatch, failure):
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(signal, 'SIGBREAK', 21, raising=False)
    monkeypatch.delenv('HFMODEM_CONSOLE_DIRECT', raising=False)
    originals = {signal.SIGINT: 'int', signal.SIGBREAK: 'break'}
    handlers = dict(originals)

    def set_handler(sig, handler):
        if failure == 'second-handler' and sig == signal.SIGBREAK and handler == signal.SIG_IGN:
            raise OSError('second-handler')
        previous, handlers[sig] = handlers[sig], handler
        return previous

    def wait():
        assert all(handler == signal.SIG_IGN for handler in handlers.values())
        raise OSError('wait')

    def popen(*args, **kwargs):
        assert all(handler == signal.SIG_IGN for handler in handlers.values())
        if failure == 'spawn':
            raise OSError('spawn')
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(signal, 'signal', set_handler)
    monkeypatch.setattr(bootstrap.subprocess, 'Popen', popen)
    with pytest.raises(OSError, match=failure):
        bootstrap._relaunch('console', [])
    assert handlers == originals


def test_posix_console_does_not_relaunch(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(bootstrap.subprocess, 'Popen', lambda *a, **kw: pytest.fail('relaunch'))
    assert bootstrap._relaunch('console', []) is None
