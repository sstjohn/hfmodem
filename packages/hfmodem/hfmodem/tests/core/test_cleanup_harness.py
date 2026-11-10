# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hfmodem.tests import hermetic
from hfmodem.tests.coexist.conftest import Harness
from hfmodem.tests.coexist.test_rows_cleanup import CleanupRun, HEADER, _console_source, _survivor

BASH = pytest.mark.posix_only('the launchers and their participants are bash scripts')


@pytest.fixture
def run(tmp_path):
    harness = Harness(tmp_path)
    try:
        run = CleanupRun(harness)
        run.env['PYTHONPATH'] = os.pathsep.join([str(hermetic.GUARD), run.env['PYTHONPATH']])
        yield run
    finally:
        harness.close()


@BASH
def test_every_launcher_starts_with_isolated_absent_ports(run):
    for key in ('CAT_PORT', 'PTT_PORT'):
        path = Path(run.env[key])
        assert path.parent == run.root and not path.exists()
    assert Path(run.env['HOME']).parent == run.root
    # The stand-in interpreter is a bash script; run it as one.
    argv = ['/bin/bash', run.env['HFMODEM_PYTHON'], '-m', 'hfmodem.core.rigs']
    hermetic.guard.admit(argv, run.env)
    result = subprocess.run(argv, env=run.env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[5:7] == [run.env['CAT_PORT'], run.env['PTT_PORT']]


def test_port_mapping_requires_explicit_test_standin_and_installs_hooks(run):
    standin = str(run.root / 'fake-standin')
    run.env['CLEANUP_PTT_DEVICE'] = standin
    result = subprocess.run([sys.executable, '-c', '''
import json, os
from hfmodem.core import ptt, rig
line = ptt.RtsPtt(os.environ['PTT_PORT'])
print(json.dumps([line.port, line._fd, hasattr(ptt.RtsPtt.assert_, '__wrapped__'),
                  rig.Rig.key.__wrapped__.__name__]))
'''], env=run.env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and not result.stderr, result.stderr
    assert json.loads(result.stdout) == [standin, None, True, 'key']
    assert not Path(standin).exists()


@pytest.mark.parametrize('point', ['preparation', 'spawn', 'recorder_start', 'recorder_wait',
                                 'settle', 'readback', 'evidence'])
def test_generated_console_program_compiles(run, point):
    compile(HEADER + _console_source(run, point), '<console-participant>', 'exec')


@pytest.mark.parametrize('kind', ['claim', 'cleanup', 'admission', 'tty_only'])
def test_generated_survivor_program_compiles(kind):
    compile(_survivor(kind), '<survivor-participant>', 'exec')


@pytest.mark.parametrize('fields', ['pid=,ppid=', 'pid=,ppid=,uid=,lstart=,command='])
def test_fake_ps_preserves_requested_columns(run, monkeypatch, capsys, fields):
    from types import SimpleNamespace as NS
    from hfmodem.tests.coexist import test_rows_cleanup as rows

    monkeypatch.setenv('HFMODEM_STATE_DIR', str(run.root))
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setattr(sys, 'argv', ['ps', '-axo', fields])
    monkeypatch.setattr(rows, '_participants', lambda: {41: {'argv': 'fake modem'}})
    calls = []
    def fake_ps(argv, **kwargs):
        calls.append(argv)
        return NS(stdout='41 20 501 Sun Oct 18 10:00:00 2026\n99 1 501 Sun Oct 18 10:00:00 2026\n')
    monkeypatch.setattr(rows.subprocess, 'run', fake_ps)
    assert rows._binary('ps') == 0
    output = capsys.readouterr().out.strip()
    assert output == ('41 20' if fields == 'pid=,ppid=' else
                      '41 20 501 Sun Oct 18 10:00:00 2026 fake modem')
    assert calls == [['/bin/ps', '-axo', 'pid=,ppid=,uid=,lstart=']]


def test_fake_ps_discovery_does_not_recurse_into_path(run, monkeypatch):
    from types import SimpleNamespace as NS
    from hfmodem.tests.coexist import test_rows_cleanup as rows

    monkeypatch.setattr(rows.sys, 'platform', 'darwin')
    monkeypatch.setenv('HFMODEM_COEXIST_RUN', 'test-run')
    calls = []
    def fake_ps(argv, **kwargs):
        calls.append(argv)
        return NS(returncode=0, stderr='', stdout=(
            '41 Sun Oct 18 10:00:00 2026 fake HFMODEM_COEXIST_RUN=test-run\n'))
    monkeypatch.setattr(rows.subprocess, 'run', fake_ps)
    assert set(rows._participants()) == {41}
    assert calls == [['/bin/ps', '-E', '-axo', 'pid=,lstart=,command=']]


@pytest.mark.parametrize('wrapper', [
    *(pytest.param(name, marks=BASH) for name in ('onair.sh', 'campaign.sh', 'autopilot.sh', 'rigtest.sh')),
    'console'])
@pytest.mark.parametrize('point', ['trap_install', 'acquisition', 'child_spawn', 'child_alive', 'cleanup'])
@pytest.mark.parametrize('ending', ['SIGTERM', 'SIGHUP', 0, 4])
def test_wrapper_participants_compile(run, wrapper, point, ending):
    from hfmodem.tests.coexist.test_rows_cleanup import (
        _wrapper_source, _console_wrapper_source)

    if wrapper == 'console':
        compile(HEADER + _console_wrapper_source(run, point, ending), '<console>', 'exec')
    else:
        source = _wrapper_source(run, wrapper, point, ending)
        assert 'wrapper_boundary' in source
        script = run.root / f'{wrapper}-participant'
        script.write_text(source)
        result = subprocess.run(['/bin/bash', '-n', str(script)], env=run.env,
                                text=True, capture_output=True, timeout=5)
        assert result.returncode == 0, result.stderr


@BASH
@pytest.mark.parametrize('wrapper', ['pounce', 'rigtest_check'])
def test_nested_launcher_compiles(run, wrapper):
    from hfmodem.tests.coexist.test_rows_cleanup import _nested_launcher

    command = _nested_launcher(run, wrapper)
    result = subprocess.run(['/bin/bash', '-n', command[0]], env=run.env,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr


@BASH
@pytest.mark.parametrize('case', ['successful_trap', 'returned_none', 'kill_due', 'nonzero_after_complete'])
def test_nested_cleanup_program_compiles(run, case):
    from hfmodem.tests.coexist.test_rows_cleanup import _nested_cleanup_source

    compile(HEADER + _nested_cleanup_source(run, case), '<nested-cleanup>', 'exec')
    result = subprocess.run(['/bin/bash', '-n', str(run.root / 'trap-launcher')], env=run.env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind,configured', [('tap', False), ('tap', True), ('kiwi', False)])
def test_recorder_program_compiles(run, kind, configured):
    from hfmodem.tests.coexist.test_rows_cleanup import _recorder_source

    source, endpoint = _recorder_source(run, kind, configured)
    compile(HEADER + source, '<recorder-console>', 'exec')
    compile((run.root / 'recorder.py').read_text(), '<recorder>', 'exec')
    assert Path(endpoint).parent == run.root


@pytest.mark.parametrize('wrapper', ['campaign.sh', 'autopilot.sh'])
def test_private_wrapper_participant_extracts_match_source(wrapper):
    """Public claim/trap scenarios stay bound to their private launcher source."""
    from hfmodem.tests import evidence

    if not evidence.private_source_tree(evidence.TREE):
        pytest.skip('whole private launchers are omitted; claim/trap participants still run')
    source = (evidence.TREE / 'tools' / wrapper).read_text()
    from hfmodem.tests.coexist.fixtures.wrapper_claims import WRAPPERS

    row = WRAPPERS[wrapper]
    assert row['claim'] == next(line for line in source.splitlines()
                                if line.startswith('claim_rig '))
    expected = ('stop() {' + source.split('\nstop() {', 1)[1].split('\nprintf ', 1)[0]
                if wrapper == 'autopilot.sh' else '')
    assert row['cleanup'] == expected


@pytest.mark.parametrize('operation,before,after', [
    ('inventory', [], ['after_cleanup_discovery']),
    ('line', ['before_cleanup_proof'], ['after_cleanup_proof']),
    ('cat', ['cleanup_cat'], ['after_cleanup_cat']),
    ('record', [], ['after_cleanup_evidence']),
])
@pytest.mark.parametrize('failed', [False, True])
def test_python_cleanup_checkpoints_require_completed_work(tmp_path, monkeypatch, operation,
                                                          before, after, failed):
    from types import SimpleNamespace as NS
    from hfmodem.core import cat, claim, coexist, down, down_posix, native, ptt
    from hfmodem.tests.coexist import test_rows_cleanup as rows

    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('CLEANUP_CAT_DEVICE', '')
    monkeypatch.setenv('CLEANUP_PTT_DEVICE', '')
    monkeypatch.delenv('CLEANUP_CAT_FAIL', raising=False)
    monkeypatch.delenv('CLEANUP_LINE_FAULT', raising=False)
    for owner, name in ((native.NativeChild, '_signal'), (claim, 'line_factory'),
                        (claim, 'serial_nodes'), (ptt.RtsPtt, '__init__')):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    checkpoints, work = [], []
    monkeypatch.setattr(coexist, 'checkpoint', checkpoints.append)
    monkeypatch.setattr(cat, 'YaesuCat', cat.YaesuCat)
    result = down.Inventory([]) if operation == 'inventory' else object()
    def perform(name):
        work.append(name)
        if failed:
            raise RuntimeError('injected failure')
        return result
    monkeypatch.setattr(down_posix.Posix, 'inventory', lambda self, deadline: perform('inventory'))
    monkeypatch.setattr(claim, 'line_proof', lambda opener: perform('line'))
    monkeypatch.setattr(down.Settings, 'cat_proof', lambda self: perform('cat'))
    monkeypatch.setattr(down, '_record', lambda line, cat: perform('record'))
    rows._map_test_ports()
    calls = dict(inventory=lambda: down_posix.Posix.inventory(None, 0),
                 line=lambda: claim.line_proof(None),
                 cat=lambda: down.Settings.cat_proof(NS(backend='serial')),
                 record=lambda: down._record('low', 'ok'))
    if failed:
        with pytest.raises(RuntimeError, match='injected failure'):
            calls[operation]()
    else:
        returned = calls[operation]()
        if operation != 'record':
            assert returned is result
    assert work == [operation]
    assert checkpoints == before + ([] if failed else after)
