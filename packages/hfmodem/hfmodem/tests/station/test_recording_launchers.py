# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Launchers resolve before sensing and pass a policy independent of paths."""
from contextlib import contextmanager, nullcontext
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS
import wave

import numpy as np
import pytest

from hfmodem.core.recording import OFF_WARNING
from hfmodem.tests import hermetic

ROOT = Path(__file__).resolve().parents[5]
SHELL = pytest.mark.posix_only(
    'drives tools/onair.sh under bash; Windows launches through tools/consolelib/launch.py')


@pytest.mark.parametrize('owner,lane,frames', [('onair_session', 'sense', 240000),
                                               ('kestrel_connect', 'rx', 528000)])
def test_source_launcher_records_with_an_older_installed_hfhost(tmp_path, owner, lane, frames):
    installed = tmp_path / 'installed'
    shutil.copytree(ROOT / 'packages/hfhost/hfhost', installed / 'hfhost',
                    ignore=shutil.ignore_patterns('recordings.py', '__pycache__'))
    directory = tmp_path / 'audio'
    script = '''
from contextlib import nullcontext
import importlib, sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

sys.modules['sounddevice'] = NS()
sys.path.insert(0, sys.argv[1])
owner = importlib.import_module(sys.argv[2])
from hfmodem.core import wav
from hfmodem.core.recording import record_window
wav._flac = lambda *a: (_ for _ in ()).throw(OSError('encoder unavailable'))
root = Path(sys.argv[3])
if sys.argv[2] == 'onair_session':
    owner.device_claim = lambda *a, **k: nullcontext()
    owner.device_opened = lambda: None
    owner.sd = NS(rec=lambda n, **k: np.full((n, 1), .125, np.float32),
                  wait=lambda: None, stop=lambda: None)
    owner.is_busy = lambda *a: (False, [], -3.)
    owner.level_range = lambda *a: (-30., -40.)
    owner.scores = lambda *a: {'live_s': 5.}
    owner.listen(5, 'synthetic', root / 'sense.wav', policy='everything')
else:
    from hfmodem.core.recording import ResolvedRecording
    rec = owner.Recorder(root / 'attempt.wav', resolved=ResolvedRecording('everything', 'inherited'))
    try:
        record_window(rec.rx, np.full(528000, .125, np.float32))
        record_window(rec.tx, np.full(4800, .25, np.float32))
    finally:
        rec.close()
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(hermetic.GUARD), str(ROOT / 'packages/hfmodem'), str(installed)]))
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'tools'), owner,
                             str(directory)], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    [path] = directory.rglob('recording.json')
    facts = json.loads(path.read_text())
    assert facts['lanes'][lane]['position'] == frames
    assert facts['lanes'][lane]['closed']
    segments = facts['segments']
    assert segments, result.stdout + result.stderr
    expected = {lane: frames}
    if owner == 'kestrel_connect':
        expected['submitted-tx'] = 4800
    for name, count in expected.items():
        retained = [row for row in segments if row['lane'] == name]
        assert retained and all(row['state'] == 'finalized' for row in retained)
        gaps = facts['lanes'][name]['gaps']
        spans = sorted([row['actual'] for row in retained] + gaps)
        assert spans[0][0] == 0 and spans[-1][1] == count
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        total = 0
        for row in retained:
            with wave.open(str(path.parent / row['source'])) as audio:
                assert audio.getframerate() == 48000
                raw = audio.readframes(audio.getnframes())
                total += len(raw) // 2
                assert set(np.frombuffer(raw, '<i2')) == {8192 if name == 'submitted-tx' else 4096}
        assert total + sum(b-a for a,b in gaps) == count


def station_file(tmp_path, policy):
    path = tmp_path / 'station.toml'
    path.write_text('schema=1\n[station]\nmycall="W0QXR"\ntransmit=false\n'
                    'control="local"\nregulatory="unregulated"\nbecause="synthetic PCM"\ngrid="AA00aa"\n'
                    '[rig]\nmodel="ft891"\ncat="none"\n'
                    + (f'[recording]\npolicy="{policy}"\n' if policy else ''))
    return path


@pytest.fixture
def session_launch(tools, monkeypatch, tmp_path):
    launch = importlib.import_module('onair_session')
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    monkeypatch.setattr(launch, 'RECORD_DIR', tmp_path / 'logs')
    monkeypatch.setattr(launch, 'pick_targets', lambda *a: [('K9ZZT', 7105000, 1, 'VARA')])
    snapshots = []

    class StopAtClaim(Exception):
        pass

    @contextmanager
    def claim(*a, **kw):
        snapshots.append({'HFMODEM_RECORDING': os.environ.get('HFMODEM_RECORDING')})
        raise StopAtClaim
        yield

    monkeypatch.setattr(launch, 'device_claim', claim)

    def run(station=None, *extra):
        extra = [str(tmp_path / 'legacy') if arg == 'legacy' else arg for arg in extra]
        monkeypatch.setenv('HFMODEM_STATION', str(station_file(tmp_path, station)))
        monkeypatch.setattr(sys, 'argv', ['onair_session', '--mycall', 'W0QXR', *extra])
        try:
            launch.session()
        except StopAtClaim:
            pass
        except SystemExit as exc:
            assert exc.code in (0, 2)
        return snapshots

    return run


@pytest.mark.parametrize('station,flags,expected', [
    (None, (), 'everything'), ('off', (), 'off'), ('contacts', (), 'contacts'),
    ('off', ('--recording', 'contacts'), 'contacts'),
    ('contacts', ('--recording', 'off'), 'off'),
    ('everything', ('--no-record',), 'off'),
])
def test_python_session_resolves_before_claim(session_launch, capsys, station, flags, expected):
    calls = session_launch(station, *flags)
    assert len(calls) == 1, 'valid recording policy must not refuse operation'
    assert calls[0].get('HFMODEM_RECORDING') == expected
    assert capsys.readouterr().out.count(OFF_WARNING) == (expected == 'off')


def test_python_session_legacy_record_is_explicit_but_generated_root_is_neutral(session_launch, tmp_path):
    calls = session_launch('contacts', '--record', str(tmp_path / 'legacy'))
    assert calls[0].get('HFMODEM_RECORDING') == 'everything'


@pytest.mark.parametrize('flags', [('--recording', 'contacts', '--no-record'),
                                  ('--recording', 'off', '--record', 'legacy'),
                                  ('--no-record', '--record', 'legacy')])
def test_python_session_rejects_contradictions_before_claim(session_launch, capsys, flags):
    try:
        calls = session_launch(None, *flags)
    except ValueError as exc:
        assert 'contradict' in str(exc).lower() or 'conflict' in str(exc).lower()
        return
    assert calls == [], 'contradictory legacy flags reached the operating claim'
    error = capsys.readouterr()
    assert 'contradict' in (error.out + error.err).lower() or 'conflict' in (error.out + error.err).lower()


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_python_session_child_destination_is_neutral_and_transcript_is_retained(tools, monkeypatch, tmp_path, policy):
    launch = importlib.import_module('onair_session')
    monkeypatch.setenv('HFMODEM_STATION', str(station_file(tmp_path, policy)))
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    monkeypatch.setattr(launch, 'RECORD_DIR', tmp_path / 'logs')
    monkeypatch.setattr(launch, 'pick_targets', lambda *a: [('K9ZZT', 7105000, 1, 'VARA')])
    monkeypatch.setattr(launch, 'device_claim', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(launch, 'device_opened', lambda: None)
    monkeypatch.setattr(launch, 'arming_refusal', lambda *a: None)
    monkeypatch.setattr(launch, 'stop_child', lambda: None)
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launch.atexit, 'register', lambda *a: None)
    cat = NS(model=lambda: 'synthetic', set_mode=lambda *a: ('PKTUSB', 2400),
             qsy=lambda *a: None, line_held=True, ptt_is_down=lambda: True,
             shutdown=lambda: None, unkey=lambda *a: None)
    monkeypatch.setattr(launch, 'Cat', lambda *a, **kw: cat)
    monkeypatch.setattr(launch, 'listen', lambda *a, **kw: NS(
        loud=1, quiet=1, judged=True, busy=False, margin=-10))
    monkeypatch.setattr(launch, 'receiver_fault', lambda *a, **kw: None)
    launched = []

    def attempt(command, window, cat, transcript, **kwargs):
        launched.append((command, transcript, os.environ.get('HFMODEM_RECORDING')))
        return 1, ''

    monkeypatch.setattr(launch, 'run_attempt', attempt)
    monkeypatch.setattr(sys, 'argv', ['onair_session', '--mycall', 'W0QXR', '--arm',
                                    '--ptt-device', 'synthetic', '--expect-model', 'synthetic'])
    launch.session()
    assert len(launched) == 1
    command, transcript, inherited = launched[0]
    assert '--record' not in command and '--no-record' not in command, 'generated paths selected policy'
    assert inherited == policy
    assert transcript is not None and transcript.parent.is_dir(), 'off must retain transcript directories'
    assert str(transcript.parent) in command, 'child must receive the neutral run destination'


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
@pytest.mark.parametrize('explicit', (False, True))
def test_varac_resolves_before_claim(tools, monkeypatch, tmp_path, policy, explicit):
    launch = importlib.import_module('varac_chat')
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    monkeypatch.setenv('HFMODEM_STATION', str(station_file(tmp_path, 'everything' if explicit else policy)))
    seen = []
    monkeypatch.setattr(launch, 'claimed_main', lambda *a: seen.append(os.environ.get('HFMODEM_RECORDING')) or 0)
    argv = ['--mycall', 'W0QXR', '--band', '40', '--session', str(tmp_path / 'session'),
            '--expect-model', 'synthetic', '--ptt-device', 'synthetic',
            '--tx-device', 'synthetic-out', '--rx-device', 'synthetic-in', '--no-bbs-service']
    if explicit:
        argv += ['--recording', policy]
    try:
        launch.main(argv)
    except SystemExit as exc:
        assert exc.code == 2
    assert seen == [policy], 'VarAC must resolve before its operating claim'


@pytest.mark.parametrize('module_name', ('kestrel_connect', 'hfmodem.shrike.onair',
                                         'hfmodem.besra.host.run_server'))
def test_standalone_launchers_expose_shared_policy_choices(tools, monkeypatch, capsys, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, 'argv', [module_name, '--help'])
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 0
    assert '--recording {off,contacts,everything}' in capsys.readouterr().out


@pytest.fixture
def shell_launch(tmp_path):
    trace = tmp_path / 'trace.jsonl'
    guard_log = tmp_path / 'guard.log'
    guard_log.write_text('')
    guard_bin = tmp_path / 'guard-bin'
    guard_bin.mkdir()
    for name in ('rigctl', 'rigctld', 'python', 'python3', 'sox', 'rec', 'play',
                 'arecord', 'aplay', 'ffmpeg'):
        fake = guard_bin / name
        fake.write_text('''#!/bin/sh
{
    printf '%s' "${0##*/}"
    printf ' %s' "$@"
    printf '\\n'
} >> "$POLICY_GUARD_LOG"
exit 97
''')
        fake.chmod(0o755)
    shim = tmp_path / 'python'
    shim.write_text(f'#!{Path(sys.executable).absolute()}\n' + '''
import json, os, shlex, shutil, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if len(args) > 1 and args[0].endswith('capture_identity.py'):
    if os.environ.get('POLICY_REAL_CAPTURE') == '1':
        os.execv(sys.executable, [sys.executable, *args])
    prepared = os.environ.get('POLICY_PREPARED') == '1'
    if args[1] == 'classify':
        sys.exit(0 if prepared else 1)
    if args[1] == 'launch':
        command = args[args.index('--') + 1:]
        script = args[args.index('--script') + 1]
        fd = os.open(os.environ['POLICY_TRANSCRIPT'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.set_inheritable(fd, True)
        os.environ['HFMODEM_CAPTURE_FD'] = str(fd)
        bash = shutil.which('bash')
        os.execv(bash, [bash, script, '--capture-prepared', *command])
    if args[1] == 'prepared':
        command = args[args.index('--') + 1:]
        destination = (command[command.index('--outdir') + 1] if '--outdir' in command
                       else os.environ['POLICY_SESSION'])
        for key, value in dict(SESSION=destination, ARM_LOG=os.environ['POLICY_TRANSCRIPT'],
                               CAPTURE_ID='synthetic-id', CAPTURE_ROOT=os.environ['POLICY_SESSION'],
                               CAPTURE_KIND='arm').items():
            print(key + '=' + shlex.quote(value))
    sys.exit(0)
if args[:2] == ['-m', 'hfmodem.core.identification']:
    os.execv(sys.executable, [sys.executable, *args])
probe = args and args[0] == '--probe'
child = ('hfmodem.shrike.onair' in args or 'hfmodem.besra.host.run_server' in args or 'hfmodem.sabir.onair' in args
         or any(a.endswith(('kestrel_connect.py', 'ardop_call.py')) for a in args))
if probe or child:
    from hfmodem.core import capture_identity as ci
    inherited = os.environ.get('HFMODEM_RECORDING')
    from hfmodem.core.recording import resolve_recording
    resolved = resolve_recording()
    session = args[2] if probe and args[1] == 'sense' else None
    registry = ((Path(session) / ci.MANIFEST).read_text()
                if session and os.environ.get('POLICY_REAL_CAPTURE') == '1' else None)
    prior = (Path(os.environ['POLICY_TRACE']).read_text().splitlines()
             if Path(os.environ['POLICY_TRACE']).exists() else [])
    busy = (session is not None and os.environ.get('POLICY_BUSY_ONCE') == '1'
            and not any(json.loads(line)['kind'] == 'sense' for line in prior))
    if child and os.environ.get('POLICY_REAL_CAPTURE') == '1':
        def value(flag):
            return args[len(args) - 1 - args[::-1].index(flag) + 1]
        ci.prepare_output(value('--outdir'), capture_id=value('--capture-id'),
                          root=value('--capture-root'), role=value('--capture-role'))
    contacts = os.environ.get('POLICY_CONTACTS')
    if contacts:
        from hfmodem.tests.station.recording_stage5_support import entry, save_selection, wave_file, write_json
        def index(root, source):
            selected = entry(root, protocol='pactor', source=source)
            save_selection(root, selected)
            write_json(root / 'recording.json', {'policy': 'contacts', 'lanes': {},
                'contacts': {'c2': selected['recording']}, 'segments': []})
            return selected
        if child:
            index(Path(value('--outdir')), 'working/run/stream.wav')
        elif args[1] == 'witness':
            root = Path(args[2])
            selected = entry(root, protocol='pactor', source='working/run/stream.wav')
            wave_file(root / 'witness.wav', count=32)
            write_json(root / 'witness.json', {'recording': selected['recording'],
                'source_id': 'source-witness', 'clock': 'kiwi', 'rate': 8, 'actual': [8016, 8048],
                'source_anchor': {'clock': 'card', 'sample': 8016, 'position': 8016}})
            print('WITNESS COVERAGE: contact-only; retained 32 samples')
        elif busy:
            index(Path(session), 'working/busy/stream.wav')
    with Path(os.environ['POLICY_TRACE']).open('a') as out:
        out.write(json.dumps({'kind': args[1] if probe else 'child', 'argv': args,
                              'inherited': inherited, 'policy': resolved.policy,
                              'session': session, 'registry': registry, 'busy': busy}) + '\\n')
    sys.exit(1 if busy else 0)
os.execv(sys.executable, [sys.executable, *args])
''')
    shim.chmod(0o755)
    ports = tmp_path / 'ports.sh'
    ports.write_text('\n'.join([
        'REPO=' + shlex.quote(str(ROOT)), 'VENV=/unused',
        'HFMODEM_PYTHON=' + shlex.quote(str(shim)), 'export HFMODEM_PYTHON',
        'MYCALL=W0QXR; IDCALL=W0QXR; MYGRID=AA00aa',
        'CAT_PORT=/unused; PTT_PORT=/unused; AUDIO=synthetic; AUDIO_IN=synthetic; AUDIO_OUT=synthetic',
        'RIG_NAME=ft891; RIG_EXPECT_MODEL=synthetic',
        'station_mycall() { echo W0QXR; }', 'die() { echo "$*" >&2; exit 2; }',
    ]) + '\n')
    rig = tmp_path / 'rig.sh'
    rig.write_text('')
    script = (ROOT / 'tools/onair.sh').read_text()
    script = script.replace('source "$(dirname "${BASH_SOURCE[0]}")/lib/ports.sh"',
                            'source ' + shlex.quote(str(ports)))
    script = script.replace('source "$(dirname "${BASH_SOURCE[0]}")/lib/rig.sh"',
                            'source ' + shlex.quote(str(rig)))
    stubs = '''
require_hardware() { :; }
rigctld_start() { :; }
rigctld_stop() { :; }
rig_power() { :; }
tune() { :; }
before_the_gate() { :; }
gate_rx_level() { :; }
sense_channel() { "$HFMODEM_PYTHON" --probe sense "$SESSION"; }
witness_start() {
  [ -n "$WITNESS" ] || return 0
  [ -n "${POLICY_CONTACTS:-}" ] || { "$HFMODEM_PYTHON" --probe witness; return; }
  WITNESS_WAV="$SESSION/witness.wav"
  witness_log_target="$SESSION/witness.log"
  [ "$CAPTURE_PREPARED" != 1 ] || witness_log_target="/dev/fd/$HFMODEM_CAPTURE_WITNESS_LOG_FD"
  "$HFMODEM_PYTHON" --probe witness "$SESSION" > "$witness_log_target" &
  WITNESS_PID=$!
}
if [ -n "${POLICY_CONTACTS:-}" ]; then
  timeout() { printf "WITNESS ALIGN"; printf " <%s>" "$@"; printf "\\n"; }
else
  witness_stop() { :; }
fi
cat_free_or_die() { :; }
beacon_grid() { GRID_ARGS=(); GRID_FALLBACK=(); }
claim_rig() { :; }
claim_run() { "$@"; }
claim_start() { "$@"; CLAIM_PID=0; }
claim_processes() { :; }
claim_pending() { :; }
claim_wait() { :; }
claim_exit() { exit "$1"; }
nc() { return 0; }
# A sub-second sleep paces a poll of a live child, such as the witness's
# second to exit, and keeps its real length; the long waits are skipped.
sleep() { case "$1" in .*|0.*) command sleep "$1" ;; esac; }
unkey_and_exit() { exit "$1"; }
'''
    script = '\n'.join('    :' if 'exec > >(' in line else line for line in script.splitlines()) + '\n'
    before, after = script.rsplit('case "${1:-}" in\n', 1)
    script = before + stubs + '\ncase "${1:-}" in\n' + after
    path = tmp_path / 'onair.sh'
    path.write_text(script)
    path.chmod(0o755)

    def run(verb='ardop', station=None, flags=(), inherited=None, leading=(), prepared=False,
            real_capture=False, busy_once=False, contacts=False):
        trace.unlink(missing_ok=True)
        env = dict(os.environ, HFMODEM_STATION=str(station_file(tmp_path, station)),
                   HFMODEM_DATA_DIR=str(tmp_path / 'data'), POLICY_TRACE=str(trace),
                   POLICY_GUARD_LOG=str(guard_log), POLICY_PREPARED=str(int(prepared)),
                   POLICY_REAL_CAPTURE=str(int(real_capture)),
                   POLICY_BUSY_ONCE=str(int(busy_once)), POLICY_CONTACTS='1' if contacts else '',
                   POLICY_SESSION=str(tmp_path / 'prepared'),
                   POLICY_TRANSCRIPT=str(tmp_path / 'prepared.log'),
                   PATH=str(guard_bin) + os.pathsep + os.environ.get('PATH', os.defpath),
                   ONAIR_TREE='synthetic-tree', POUNCE_WAIT='0')
        env.pop('HFMODEM_RECORDING', None)
        if inherited:
            env['HFMODEM_RECORDING'] = inherited
        args = ([verb, '7105000'] if verb == 'sabir' else
                ['pounce', 'pactor', 'K9ZZT', '7105000'] if verb == 'pounce' else
                [verb, 'K9ZZT', '7105000'] if verb == 'pactor' else
                [verb, 'K9ZZT', '7105000', '500'])
        for launcher in (['bash'], [shutil.which('bash', path=env['PATH'])], []):
            hermetic.guard.admit([*launcher, str(path)], env)
        try:
            # A pounce starts some twenty Python children, a quarter second each
            # on an idle workstation; the bound is for a hang, not for speed.
            result = subprocess.run(['bash', str(path), *leading, *args, *flags], env=env,
                                    text=True, capture_output=True, timeout=120)
        finally:
            assert guard_log.read_text() == '', 'unexpected command: ' + guard_log.read_text()
        events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, events

    return run


@pytest.mark.parametrize('verb', ('ardop', 'ardop-mail'))
@pytest.mark.parametrize('station,flags,expected', [
    (None, (), 'everything'), ('off', (), 'off'), ('contacts', (), 'contacts'),
    ('off', ('--recording', 'contacts'), 'contacts'),
    ('contacts', ('--recording', 'off'), 'off'),
    ('contacts', ('--no-record',), 'off'),
])
@SHELL
def test_shell_ardop_policy_precedes_gate_and_generated_destinations(shell_launch, verb, station, flags, expected):
    result, events = shell_launch(verb, station, (*flags, '--witness'))
    assert result.returncode == 0, result.stderr + result.stdout
    assert {event['kind'] for event in events} >= {'sense', 'witness', 'child'}
    assert all(event['inherited'] == expected for event in events), events
    assert all(event['policy'] == expected for event in events), events
    server = next(event['argv'] for event in events if 'hfmodem.besra.host.run_server' in event['argv'])
    assert '--record' not in server, 'generated destination must not force everything'
    assert '--no-record' not in server
    assert any('/data/captures/onair-' in arg or arg == str(ROOT / 'logs/onair') for arg in server)
    assert result.stdout.count(OFF_WARNING) + result.stderr.count(OFF_WARNING) == (expected == 'off')


@SHELL
@pytest.mark.parametrize('verb', ('ardop', 'ardop-mail', 'sabir'))
def test_shell_legacy_record_alias_remains_explicit(shell_launch, tmp_path, verb):
    target = str(tmp_path / 'chosen destination')
    result, events = shell_launch(verb, 'contacts', ('--record', target))
    assert result.returncode == 0, result.stderr + result.stdout
    assert events and all(event['inherited'] == 'everything' for event in events)
    assert any(target in event['argv'] for event in events if event['kind'] == 'child')


@pytest.mark.parametrize('flags', [('--recording', 'contacts', '--no-record'),
                                  ('--recording', 'off', '--record', 'legacy'),
                                  ('--no-record', '--record', 'legacy')])
@SHELL
def test_shell_contradictions_fail_before_sensing(shell_launch, flags):
    result, events = shell_launch('ardop', None, flags)
    assert result.returncode != 0
    assert events == [], 'contradictions reached a sensing or witness child'
    assert any(word in (result.stdout + result.stderr).lower() for word in ('contradict', 'conflict'))


@SHELL
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_shell_sabir_uses_neutral_run_destination(shell_launch, policy):
    result, events = shell_launch('sabir', policy)
    assert result.returncode == 0, result.stderr + result.stdout
    assert events and all(event['inherited'] == policy for event in events)
    child = next(event['argv'] for event in events if 'hfmodem.sabir.onair' in event['argv'])
    assert '--wav' not in child, 'operating beacon persistence must use the policy adapter'
    assert any('/data/captures/onair-' in arg for arg in child), 'beacon needs a neutral run destination'


@SHELL
def test_inherited_off_does_not_warn_again(shell_launch):
    result, events = shell_launch('ardop-mail', 'everything', inherited='off')
    assert result.returncode == 0, result.stderr + result.stdout
    assert events and all(event['inherited'] == 'off' for event in events)
    assert OFF_WARNING not in result.stdout + result.stderr


@SHELL
def test_leading_witness_still_resolves_before_children(shell_launch):
    result, events = shell_launch('ardop', 'off', leading=('--witness',))
    assert result.returncode == 0, result.stderr + result.stdout
    assert {event['kind'] for event in events} >= {'sense', 'witness', 'child'}
    assert all(event['inherited'] == 'off' for event in events)
    assert (result.stdout + result.stderr).count(OFF_WARNING) == 1


@SHELL
@pytest.mark.parametrize('verb', ('pactor', 'pounce'))
def test_prepared_reexec_preserves_witness_and_record_destination(shell_launch, tmp_path, verb):
    target = str(tmp_path / 'chosen destination')
    result, events = shell_launch(verb, 'contacts', ('--record', target, '--witness'), prepared=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(event['kind'] == 'witness' for event in events), events
    child = next(event['argv'] for event in events if event['kind'] == 'child')
    assert child[child.index('--outdir') + 1] == target
    assert all(event['inherited'] == 'everything' for event in events)


@SHELL
@pytest.mark.parametrize('verb', ('pactor', 'pounce'))
def test_record_destination_is_registered_through_real_capture_handoff(shell_launch, tmp_path, verb):
    from hfmodem.core import capture_identity as ci
    target = tmp_path / 'chosen destination'
    result, events = shell_launch(verb, 'contacts', ('--record', str(target), '--witness'),
                                  real_capture=True, busy_once=verb == 'pounce')
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(event['kind'] == 'witness' for event in events)
    children = [event['argv'] for event in events if event['kind'] == 'child']
    assert len(children) == 1
    child = children[0]
    senses = [event for event in events if event['kind'] == 'sense']
    assert [event['busy'] for event in senses] == ([True, False] if verb == 'pounce' else [False])
    roots = [Path(event['session']) for event in senses]
    assert len(set(roots)) == len(roots)
    if verb == 'pounce':
        assert all(root.parent == target for root in roots)
    else:
        assert roots == [target]
    captures = [ci.load(root) for root in roots]
    for event, capture in zip(senses, captures):
        assert (capture.root('capture') / ci.MANIFEST).read_text() == event['registry']
        transcript = capture.root('transcript') / capture.record['transcript']['path']
        assert transcript.is_file()
        exits = [json.loads(line) for line in
                 (capture.root('transcript') / 'launcher-events.jsonl').read_text().splitlines()]
        assert [row['returncode'] for row in exits] == [4 if event['busy'] else 0]
    capture = captures[-1]
    root = roots[-1]
    assert capture.kind == 'arm'
    assert capture.root('capture') == root
    assert child[child.index('--capture-id') + 1] == capture.capture_id
    assert child[child.index('--capture-root') + 1] == str(root)
    assert child[len(child) - 1 - child[::-1].index('--outdir') + 1] == str(root)
    assert all(event['inherited'] == 'everything' for event in events)
    if verb == 'pounce':
        groups = [ci.load(path.parent) for path in (tmp_path / 'data/captures').glob('*/' + ci.MANIFEST)]
        group = next(item for item in groups if item.kind == 'launch-group')
        assert all(item.record['parent_capture_id'] == group.capture_id for item in captures)
        rows = [json.loads(line) for line in (group.root('transcript') / 'children.jsonl').read_text().splitlines()]
        assert [row['child_capture_root'] for row in rows] == [str(root) for root in roots]
        assert [row['child_capture_id'] for row in rows] == [item.capture_id for item in captures]


@pytest.mark.parametrize('leading', [('--log', 'synthetic'),
    ('--capture-id', 'synthetic-id', '--capture-root', '/unused', '--log=synthetic')])
@SHELL
def test_leading_globals_resolve_before_children(shell_launch, leading):
    result, events = shell_launch('ardop', 'off', ('--recording', 'everything', '--witness'), leading=leading)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {event['kind'] for event in events} >= {'sense', 'witness', 'child'}
    assert all(event['inherited'] == 'everything' for event in events), events


@pytest.mark.parametrize('policy,write_failure', [('off', False), ('contacts', False), ('everything', True)])
def test_real_sense_policy_and_write_failure_preserve_verdict(tools, monkeypatch, tmp_path, policy, write_failure):
    import numpy as np
    launch = importlib.import_module('onair_session')
    monkeypatch.setenv('HFMODEM_RECORDING', policy)
    monkeypatch.setattr(launch, 'device_claim', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(launch, 'device_opened', lambda: None)
    audio = np.random.default_rng(12).normal(0, .01, (4 * launch.FS, 1)).astype(np.float32)
    monkeypatch.setattr(launch, 'sd', NS(rec=lambda *a, **kw: audio.copy(),
                                       wait=lambda: None, stop=lambda: None))
    monkeypatch.chdir(tmp_path)
    expected = launch.listen(4, 'synthetic')
    writes = []
    if write_failure:
        def fail(path, **kwargs):
            writes.append(path)
            raise OSError('sense storage unavailable')
        from hfmodem.core import wav
        monkeypatch.setattr(wav, 'SegmentedWriter', fail)
    target = tmp_path / 'sense.wav'
    measured = launch.listen(4, 'synthetic', target, policy=policy)
    assert measured == expected
    assert not target.exists(), 'idle sense audio persisted under the resolved policy'
    assert len(writes) == int(write_failure)


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_session_sense_keeps_resolved_policy_when_station_becomes_unreadable(
        tools, monkeypatch, tmp_path, capsys, policy):
    import numpy as np
    launch = importlib.import_module('onair_session')
    station = station_file(tmp_path, policy)
    monkeypatch.setenv('HFMODEM_STATION', str(station))
    monkeypatch.delenv('HFMODEM_RECORDING', raising=False)
    monkeypatch.setattr(launch, 'RECORD_DIR', tmp_path / 'logs')
    monkeypatch.setattr(launch, 'pick_targets', lambda *a: [('K9ZZT', 7105000, 1, 'VARA')])
    monkeypatch.setattr(launch, 'device_opened', lambda: None)
    monkeypatch.setattr(launch, 'stop_child', lambda: None)
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launch.atexit, 'register', lambda *a: None)
    monkeypatch.setattr(launch.cwid, 'audio', lambda *a, **kw: np.zeros(8, np.float32))
    monkeypatch.setattr(launch, 'receiver_fault', lambda *a, **kw: None)
    cat = NS(model=lambda: 'synthetic', set_mode=lambda *a: ('PKTUSB', 2400),
             qsy=lambda *a: None, line_held=True, ptt_is_down=lambda: True,
             shutdown=lambda: None)
    monkeypatch.setattr(launch, 'Cat', lambda *a, **kw: cat)
    audio = np.random.default_rng(12).normal(0, .01, (4 * launch.FS, 1)).astype(np.float32)
    monkeypatch.setattr(launch, 'sd', NS(rec=lambda *a, **kw: audio.copy(),
                                       wait=lambda: None, stop=lambda: None,
                                       PortAudioError=RuntimeError))
    unreadable = False
    reads, measured, writes = [], [], []
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if path == station and unreadable:
            reads.append(path)
            raise PermissionError('station file unavailable')
        return read_text(path, *args, **kwargs)

    @contextmanager
    def claim(owner, **kwargs):
        nonlocal unreadable
        if owner == 'onair_session':
            assert os.environ['HFMODEM_RECORDING'] == policy
            unreadable = True
        yield

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(launch, 'device_claim', claim)
    real_listen = launch.listen
    monkeypatch.chdir(tmp_path)
    expected = real_listen(4, 'synthetic', band=launch.occupied_hz('vara', '2300'))

    def listen(*args, **kwargs):
        result = real_listen(*args, **kwargs)
        measured.append(result)
        return result

    monkeypatch.setattr(launch, 'listen', listen)
    from hfmodem.core.recording import RecordingSession
    stream = RecordingSession.stream
    def tracked(session, *args, **kwargs):
        lane = stream(session, *args, **kwargs)
        original = lane._push
        def push(first, pcm, *, acquired):
            writes.append((session, first, pcm.copy()))
            original(first, pcm, acquired=acquired)
        lane._push = push
        return lane
    monkeypatch.setattr(RecordingSession, 'stream', tracked)
    monkeypatch.setattr(sys, 'argv', ['onair_session', '--mycall', 'W0QXR', '--grid', 'AA00aa',
                                    '--dry-run', '--force', '--listen', '4'])
    assert launch.session() == 1
    assert unreadable
    assert measured == [expected]
    assert reads == []
    assert writes and all(item[0].policy == policy for item in writes)
    np.testing.assert_array_equal(np.concatenate([item[2] for item in writes]), audio[:, 0])
    session = writes[0][0]
    assert session.directory.name == '01-K9ZZT-listen'
    assert bool(session.index['segments']) == (policy == 'everything')
    output = capsys.readouterr().out
    assert 'cannot listen' not in output
    assert '(would call here)' in output


@SHELL
def test_contact_only_pounce_witness_aligns_its_own_attempts_contact(shell_launch, tmp_path):
    from hfmodem.tests.station.recording_stage5_support import STOP, entry, save_selection, write_json
    target = tmp_path / 'chosen destination'
    decoy = entry(target, protocol='pactor', source='working/parent/stream.wav')
    save_selection(target, decoy)
    write_json(target / 'recording.json', {'policy': 'contacts', 'lanes': {},
        'contacts': {'c2': decoy['recording']}, 'segments': []})
    result, events = shell_launch('pounce', 'contacts', ('--run-dir', str(target), '--witness'),
                                  real_capture=True, busy_once=True, contacts=True)
    assert result.returncode == 0, result.stdout + result.stderr
    senses = [event for event in events if event['kind'] == 'sense']
    assert [event['busy'] for event in senses] == [True, False]
    first, second = (Path(event['session']) for event in senses)
    assert first.parent == target and second.parent == target and first != second
    assert first.name.endswith('-attempt-1') and second.name.endswith('-attempt-2')
    assert (first / 'recording.json').is_file(), 'the busy attempt left no index to be mistaken for'
    assert all(event['inherited'] == 'contacts' for event in events), events
    aligned = [line.split(' <')[1:] for line in result.stdout.splitlines() if 'WITNESS ALIGN' in line]
    assert aligned, result.stdout + result.stderr
    for words in aligned:
        words = [word.rstrip('>') for word in words]
        assert words[3:6] == [str(second / 'working/run/stream.wav'), str(second / 'witness.wav'), '--selection']
        assert json.loads(words[6])['local'] == [8016, STOP]
