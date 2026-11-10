# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A fixture process table may omit only independently confirmed departures."""
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hfmodem.tests.coexist import test_rows_ownership as rows


@pytest.fixture
def ps_source(tmp_path, monkeypatch):
    scripts = {}
    monkeypatch.setattr(rows, 'binary', lambda harness, name, source: scripts.update({name: source}))
    rows.shell_setup(SimpleNamespace(root=tmp_path, env={}))
    (tmp_path / 'inventory.json').write_text(json.dumps({
        '41': {'command': 'fixture-target'}, '44': {'command': 'other-live-target'}}))
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('ROW_SHELL_PID', '42')
    monkeypatch.delenv('ROW_FAKE_METADATA', raising=False)
    monkeypatch.setattr(os, 'getppid', lambda: 43)
    monkeypatch.setattr(sys, 'argv', ['ps', '-axo', 'pid=,ppid=,uid=,lstart=,command='])
    return compile(textwrap.dedent(scripts['ps']), '<generated fixture ps>', 'exec')


@pytest.mark.parametrize('failed_query', ['stat=', 'lstart='])
@pytest.mark.parametrize('resolution', ['gone', 'live', 'permission'])
def test_failed_metadata_query_omits_only_confirmed_gone_row(ps_source, monkeypatch, capsys,
                                                           failed_query, resolution):
    checks = []

    def kill(pid, sig):
        assert sig == 0
        if pid == 41:
            checks.append(pid)
            if len(checks) == 2:
                if resolution == 'gone':
                    raise ProcessLookupError('departed between ps queries')
                if resolution == 'permission':
                    raise PermissionError('identity cannot be checked')

    def query(args, **kwargs):
        if args[2] == '41' and args[-1] == failed_query:
            raise subprocess.CalledProcessError(1, args)
        return SimpleNamespace(stdout='S' if args[-1] == 'stat=' else 'Mon Oct 26 12:00:00 2026')

    monkeypatch.setattr(os, 'kill', kill)
    monkeypatch.setattr(subprocess, 'run', query)
    if resolution == 'gone':
        exec(ps_source, {})
        output = capsys.readouterr().out
        assert 'fixture-target' not in output
        assert f'44 1 {os.getuid()} Mon Oct 26 12:00:00 2026 other-live-target\n' in output
        assert 'bash fake-row' in output and 'python helper' in output
    else:
        expected = PermissionError if resolution == 'permission' else subprocess.CalledProcessError
        with pytest.raises(expected):
            exec(ps_source, {})
    assert checks == [41, 41]


def test_live_row_still_has_real_metadata(ps_source, monkeypatch, capsys):
    queries = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: None)

    def query(args, **kwargs):
        queries.append(args)
        return SimpleNamespace(stdout='S' if args[-1] == 'stat=' else 'Mon Oct 26 12:00:00 2026')

    monkeypatch.setattr(subprocess, 'run', query)
    exec(ps_source, {})
    output = capsys.readouterr().out
    assert f'41 1 {os.getuid()} Mon Oct 26 12:00:00 2026 fixture-target\n' in output
    assert [args[-1] for args in queries if args[2] == '41'] == ['stat=', 'lstart=']


def test_unexpected_metadata_error_is_not_a_departure(ps_source, monkeypatch):
    checks = []
    monkeypatch.setattr(os, 'kill', lambda pid, sig: checks.append(pid))

    def denied(*args, **kwargs):
        raise PermissionError('metadata unavailable')

    monkeypatch.setattr(subprocess, 'run', denied)
    with pytest.raises(PermissionError, match='metadata unavailable'):
        exec(ps_source, {})
    assert checks == [41]
