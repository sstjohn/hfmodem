# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import textwrap

import pytest

from .conftest import TREE
from .observer import (MacOSObserver, ObservationError, modem_bits, probe_claim,
                       set_modem_bits)
from .oracle import assert_claim_possession, assert_released, validate


FAULTS = ('after_mutex', 'before_claim_lock', 'after_claim_flock', 'before_holder_line', 'after_holder_line',
          'after_device_check', 'during_line_proof', 'before_release', 'after_release',
          'after_cleanup_discovery', 'before_cleanup_proof', 'after_cleanup_proof',
          'cleanup_cat', 'after_cleanup_cat', 'after_cleanup_evidence',
          'before_cleanup_signal', 'after_cleanup_signal',
          *(f'{when}_RtsPtt_{method}' for method in ('open', 'assert_', '_sense_fd', 'release')
            for when in ('before', 'after')))
HEADER = '''
import os, signal, subprocess, sys
from pathlib import Path
from hfmodem.core import claim, coexist
claim.bootstrap()
'''


def _map_test_ports():
    from hfmodem.core import claim, coexist, native, ptt

    if os.environ.get('HFMODEM_COEXIST_HARNESS') != '1':
        return
    original_signal = native.NativeChild._signal
    def signalled(child, signum):
        coexist.checkpoint('before_cleanup_signal')
        result = original_signal(child, signum)
        coexist.checkpoint('after_cleanup_signal')
        return result
    native.NativeChild._signal = signalled
    root = Path(os.environ['HFMODEM_STATE_DIR'])
    aliases = {str(root / 'absent-cat'): os.environ['CLEANUP_CAT_DEVICE'],
               str(root / 'absent-ptt'): os.environ['CLEANUP_PTT_DEVICE']}
    factory, nodes = claim.line_factory, claim.serial_nodes
    claim.line_factory = lambda port: factory(aliases.get(port, port))
    claim.serial_nodes = lambda port, **kw: nodes(aliases.get(port, port), **kw) if aliases.get(port, port) else ()
    init = ptt.RtsPtt.__init__
    ptt.RtsPtt.__init__ = lambda self, port, **kw: init(self, aliases.get(port, port), **kw)
    if os.environ.get('CLEANUP_LINE_FAULT') == 'unreadable':
        ptt.RtsPtt.sense = lambda self: None
    elif os.environ.get('CLEANUP_LINE_FAULT') == 'high':
        ptt.RtsPtt._clear = staticmethod(lambda fd, bits: ptt.RtsPtt._set(fd, termios.TIOCM_RTS))


def _participants():
    from .observer import linux_participants, macos_participants

    if sys.platform == 'linux':
        return linux_participants(os.environ['HFMODEM_COEXIST_RUN'],
                                  started_after=int(os.environ['HFMODEM_COEXIST_START_TICKS']))

    def runner(argv, **kwargs):
        return subprocess.run(['/bin/ps', *argv[1:]], **kwargs)

    return macos_participants(os.environ['HFMODEM_COEXIST_RUN'], runner=runner)


def _binary(name):
    root = Path(os.environ['HFMODEM_STATE_DIR'])
    assert os.environ['HFMODEM_COEXIST_HARNESS'] == '1'
    fault = os.environ.get('CLEANUP_DISCOVERY_FAULT')
    if name == 'ps':
        if '-axo' in sys.argv and fault:
            if fault == 'malformed':
                print('not a process record')
                return 0
            print(f'{fault}: process discovery denied', file=sys.stderr)
            return 1
        args = sys.argv[1:]
        if '-axo' in args:
            participants = _participants()
            result = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,uid=,lstart='],
                                    capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                fields = line.split()
                if not fields or int(fields[0]) not in participants:
                    continue
                if args[args.index('-axo') + 1] == 'pid=,ppid=':
                    print(' '.join(fields[:2]))
                    continue
                command = participants[int(fields[0])]['argv']
                if ' -c ' in command:
                    command = command.split(' -c ', 1)[0] + ' -c coexist-participant'
                fields.append(' '.join(command.split()))
                metadata = root / f'metadata-{fields[0]}.json'
                if metadata.exists():
                    fake = json.loads(metadata.read_text())
                    fields[8] = fake.get('command', fields[8])
                print(' '.join(fields))
            return 0
        return subprocess.run(['/bin/ps', *args]).returncode
    if name == 'lsof':
        if '-t' in sys.argv:
            listener = root / 'listener.json'
            if listener.exists():
                data = json.loads(listener.read_text())
                if f'-iTCP:{data["port"]}' in sys.argv:
                    print(data['pid'])
                    return 0
            return 1
        if fault == 'permission':
            print('permission denied', file=sys.stderr)
            return 1
        participants = _participants()
        if not participants:
            return 1
        aliases = {str(root / 'absent-cat'): os.environ['CLEANUP_CAT_DEVICE'],
                   str(root / 'absent-ptt'): os.environ['CLEANUP_PTT_DEVICE']}
        return subprocess.run([os.environ['CLEANUP_LSOF'], '-a', '-p',
                               ','.join(map(str, participants)),
                               *(aliases.get(arg, arg) for arg in sys.argv[1:])]).returncode
    if name == 'timeout':
        args = sys.argv[1:]
        if args[:1] == ['-k']:
            args = args[2:]
        os.execvp(args[1], args[1:])
    if name == 'rigctl':
        from hfmodem.core import claim, coexist
        claim.inherited(9)
        claim.verify_cleanup(8)
        coexist.checkpoint('cleanup_cat')
        if os.environ.get('CLEANUP_CAT_FAIL'):
            return 1
        port = sys.argv[sys.argv.index('-r') + 1]
        assert port == os.environ['CAT_PORT']
        from hfmodem.core.cat import YaesuCat
        from hfmodem.tests.core.fakeyaesu import PtySerial
        cat = YaesuCat(os.environ['CLEANUP_CAT_DEVICE'], 'ft891', serial_factory=PtySerial)
        cat.open()
        try:
            print(cat.freq())
        finally:
            cat.close()
        return 0
    if name == 'nc':
        with (root / 'nc.used').open('a') as stream:
            stream.write(' '.join(sys.argv) + '\n')
        return 1
    raise AssertionError(f'unexpected external binary: {name} in {root}')


class CleanupRun:
    def __init__(self, harness):
        self.h = harness
        self.root = harness.root
        self.env = harness.env
        self.env.update(HFMODEM_PYTHON=sys.executable, HAMLIB_BIN=str(self.root / 'bin'),
                        CAT_PORT=str(self.root / 'absent-cat'), PTT_PORT=str(self.root / 'absent-ptt'),
                        CLEANUP_CAT_DEVICE='', CLEANUP_PTT_DEVICE='', TMPDIR=str(self.root),
                        HFMODEM_COEXIST_INSTRUMENT='1', HOME=str(self.root / 'home'),
                        HFMODEM_DATA_DIR=str(self.root / 'data'))
        (self.root / 'home').mkdir()
        self.env.pop('RIG_CLAIMED', None)
        self.env.pop('HFMODEM_RIG', None)
        self.env['PYTHONPATH'] += os.pathsep + str(TREE / 'tools')
        bootstrap = self.root / 'bootstrap/sitecustomize.py'
        with bootstrap.open('a') as stream:
            stream.write('from hfmodem.tests.coexist.test_rows_cleanup import _map_test_ports\n'
                         '_map_test_ports()\n')
        lsof = shutil.which('lsof')
        if lsof:
            self.env['CLEANUP_LSOF'] = lsof
        for name in ('ps', 'lsof', 'rigctl', 'rigctld', 'timeout', 'nc'):
            path = self.root / 'bin' / name
            path.write_text('#!' + sys.executable + '\n'
                'from hfmodem.tests.coexist.test_rows_cleanup import _binary\n'
                f'raise SystemExit(_binary({name!r}))\n')
            path.chmod(0o700)
        python = self.root / 'bin/python'
        python.write_text('#!/bin/bash\nif [ "$1 $2" = "-m hfmodem.core.rigs" ]; then\n'
            'printf "ft891\\n1036\\n38400\\nPKTUSB\\nfake audio\\n%s\\n%s\\nfake audio\\n" "$CAT_PORT" "$PTT_PORT"\n'
            'else\nexec ' + shlex.quote(sys.executable) + ' "$@"\nfi\n')
        python.chmod(0o700)
        self.env['HFMODEM_PYTHON'] = str(python)
        self.station = self.root / 'station.toml'
        self.station.write_text('[station]\nport = 18743\n[rig]\ncat = "none"\n')
        self.env['HFMODEM_STATION'] = str(self.station)
        self.outputs = {}
        self.count = 0

    def observe(self, paths=()):
        if sys.platform == 'linux' and not shutil.which('strace'):
            pytest.skip('strace is required in hfmodem-ci; no untraced Linux substitute')
        try:
            self.h.observe([self.h.audio, *paths])
        except (ObservationError, PermissionError) as exc:
            self.h.observer = None
            if sys.platform == 'darwin' and ('not permitted' in str(exc) or 'denied' in str(exc)):
                pytest.skip(f'sandbox blocks independent ps/lsof observer; conductor reruns outside: {exc}')
            raise

    def launch(self, argv, **kwargs):
        index = self.count
        self.count += 1
        if sys.platform == 'linux':
            process = self.h.launch(argv, **kwargs)
            trace = self.h.traced[-1]
            self.outputs[process] = (trace.path.parent / 'stdout', trace.path.parent / 'stderr')
        else:
            paths = (self.root / f'process-{index}.stdout', self.root / f'process-{index}.stderr')
            with paths[0].open('wb') as out, paths[1].open('wb') as err:
                process = self.h.launch(argv, stdout=out, stderr=err, **kwargs)
            self.outputs[process] = paths
        return process

    def python(self, body, **kwargs):
        return self.launch([sys.executable, '-c', HEADER + textwrap.dedent(body)], **kwargs)

    def shell(self, body='station_down', *, supervised_exit=None, **kwargs):
        source = ('set -euo pipefail\n'
                  f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
                  f'source {shlex.quote(str(TREE / "tools/lib/rig.sh"))}\n'
                  'claim_down() { CAT_PORT="$HFMODEM_STATE_DIR/absent-cat" '
                  'PTT_PORT="$HFMODEM_STATE_DIR/absent-ptt" "$REPO/tools/onair.sh" down 7>&- 8>&-; }\n'
                  'CAT_PORT="$CLEANUP_CAT_DEVICE"\nPTT_PORT="$CLEANUP_PTT_DEVICE"\n'
                  + _cleanup_boundaries() + body)
        if supervised_exit is not None:
            return self.python(f"child = subprocess.Popen(['/bin/bash', '-c', {source!r}])\n"
                               f"assert child.wait(timeout=45) == {supervised_exit!r}\n", **kwargs)
        return self.launch(['/bin/bash', '-c', source], **kwargs)

    def finish(self, process, expected=0):
        code = process.wait(timeout=20)
        output = ''.join(path.read_text() for path in self.outputs[process])
        assert code == expected, (code, output)
        return output

    def supervise(self, source, expected):
        return self.python(f'''
child = subprocess.Popen([sys.executable, '-c', {HEADER + source!r}])
assert child.wait(timeout=45) == {expected}
coexist.checkpoint('root_dead')
''')

    def snapshot(self, point):
        observer = self.h.observer
        if isinstance(observer, MacOSObserver):
            return observer.barrier(point, claim_path=self.root / 'rig.claim', quiescent=True)
        return observer.capture(point)

    def released(self, *, line=None, require_line=False):
        snapshot = self.snapshot('released')
        assert_released(snapshot, paths=self.h.paths, claim_path=self.root / 'rig.claim',
                        line=line, require_line=require_line)
        evidence = self.h.collect()
        validate(evidence)
        return evidence


@pytest.fixture
def cleanup_run(harness):
    run = CleanupRun(harness)
    try:
        yield run
    finally:
        for path in harness.barriers.iterdir():
            if '.' not in path.name:
                harness.release(path.name)
        if harness.observer is not None:
            from hfmodem.core import claim, native
            snapshot = harness.observer.capture('teardown')
            for pid in snapshot.participants.keys() - {os.getpid()}:
                try:
                    child = native.NativeChild(pid, claim.process_started(pid), None, None, None)
                    child.terminate()
                    if not child.wait_gone(2):
                        child.kill()
                        assert child.wait_gone(2), f'coexist descendant survived: {pid}'
                except claim.ClaimError:
                    if pid in harness.observer.capture('recheck').participants:
                        raise
    used = run.root / 'nc.used'
    assert not used.exists(), f'a row reached rigctld directly through nc:\n{used.read_text()}'


@pytest.fixture
def proof_run(tty0tty_pair, cleanup_run, fake_yaesu):
    near, far = tty0tty_pair
    fd = os.open(far, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        set_modem_bits(fd, 0)
        assert modem_bits(fd) & termios.TIOCM_RTS == 0
        cleanup_run.env.update(CLEANUP_PTT_DEVICE=near, CLEANUP_CAT_DEVICE=fake_yaesu.port)
        cleanup_run.observe((near, fake_yaesu.port))
        cleanup_run.sense = lambda: bool(modem_bits(fd) & termios.TIOCM_CTS)
        yield cleanup_run
    finally:
        os.close(fd)


@pytest.mark.parametrize('point', ['before_holder_line', 'after_holder_line', 'before_release',
                                 'after_release', 'sibling'])
def test_holder_line_published_under_lock(cleanup_run, point):
    run = cleanup_run
    run.observe()
    if point == 'sibling':
        run.h.pin('sibling_read')
        child = HEADER + '''
held = claim.inherited(9)
before = held.offset()
published = claim.holder(held)
assert before == held.offset() == published.nonce
coexist.checkpoint('sibling_read')
claim.close(held)
'''
        actor = run.python(f'''
held = claim.acquire('test', 'sibling-publication')
os.dup2(held.fd, 9)
if held.fd != 9:
    claim.close(held)
child = subprocess.Popen([sys.executable, '-c', {child!r}], pass_fds=(9,))
assert child.wait(timeout=15) == 0
claim.release(9)
''')
        reached = run.h.reached('sibling_read', quiescent=True)
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('sibling_read'), pid=reached['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
        run.h.release('sibling_read')
        run.finish(actor)
        run.released()
        return
    run.h.pin(point)
    actor = run.python('''
held = claim.acquire('test', 'publication')
claim.release(held)
''')
    reached = run.h.reached(point, quiescent=True)
    snapshot = run.snapshot(point)
    if sys.platform == 'darwin':
        assert snapshot.claim_free is (point == 'after_release')
    if point == 'after_release':
        assert probe_claim(run.root / 'rig.claim', quiescent=True)
    else:
        assert not probe_claim(run.root / 'rig.claim', quiescent=True)
    reading = run.python('''
fd = os.open(claim.claim_path(), os.O_RDONLY)
try:
    try:
        published = claim.holder(fd)
        print('holder', published.nonce, flush=True)
    except claim.ClaimError as exc:
        print(str(exc), flush=True)
    assert os.lseek(fd, 0, os.SEEK_CUR) == 0
finally:
    os.close(fd)
''')
    text = run.finish(reading)
    if point in ('before_holder_line', 'after_release'):
        assert 'holder unknown' in text
        assert sum(event['event'] == 'holder_retry' for event in run.h.events()) == 2
    else:
        assert 'holder ' in text
    assert reached['pid'] != reading.pid
    run.h.release(point)
    run.finish(actor)
    run.released()


@pytest.mark.parametrize('failure', ['discovery', 'permission', 'malformed'])
def test_claim_free_cleanup_faults(cleanup_run, failure):
    run = cleanup_run
    run.observe()
    run.env['CLEANUP_CAT_DEVICE'] = str(run.h.audio)
    run.station.write_text('[station]\nport = 18743\n[rig]\ncat = "yaesu"\n')
    actor = run.shell(env=dict(run.env, CLEANUP_DISCOVERY_FAULT=failure))
    output = run.finish(actor, 1)
    assert 'NOT CONFIRMED' in output
    assert 'CLEANUP COMPLETE' not in output and 'PTT LINE low' not in output
    evidence = run.released()
    assert not any(event['event'] == 'device_open' for event in evidence.events)


@pytest.mark.parametrize('kind', ['station', 'launcher'])
def test_rigtest_trap_after_refusal(cleanup_run, kind):
    run = cleanup_run
    run.observe()
    run.h.pin('holding')
    owner = run.python(f'''
held = claim.acquire({kind!r}, 'refuse-rigtest')
coexist.checkpoint('holding')
claim.release(held)
''')
    run.h.reached('holding', quiescent=True)
    refused = run.launch(['/bin/bash', str(TREE / 'tools/rigtest.sh'), 'preflight'])
    output = run.finish(refused, 1)
    assert 'rig claimed' in output
    assert 'CLEANUP COMPLETE' not in output and 'PTT LINE' not in output
    assert not probe_claim(run.root / 'rig.claim', quiescent=True)
    run.h.release('holding')
    run.finish(owner)
    evidence = run.released()
    assert not any(event['event'] == 'device_open' for event in evidence.events)


def test_competing_down(proof_run):
    run = proof_run
    run.h.pin('during_line_proof')
    first = run.shell()
    run.h.reached('during_line_proof', quiescent=True)
    second = run.shell()
    output = run.finish(second, 2)
    assert 'cleanup in progress' in output and 'CLEANUP COMPLETE' not in output
    run.h.release('during_line_proof')
    assert 'CLEANUP COMPLETE' in run.finish(first)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('failure', ['cat', 'evidence', 'cat_none', 'console_readback', 'console_evidence'])
def test_cleanup_verdict_publication(proof_run, failure):
    run = proof_run
    if failure.startswith('console_'):
        _console_finalization(run, failure.removeprefix('console_'))
        return
    if failure == 'cat':
        run.station.write_text('[rig]\ncat = "yaesu"\n')
        run.env['CLEANUP_CAT_FAIL'] = '1'
    elif failure == 'evidence':
        (run.root / 'cleanup.json').mkdir()
    actor = run.shell()
    output = run.finish(actor, 0 if failure == 'cat_none' else 1)
    assert 'PTT LINE low' in output
    assert ('CLEANUP COMPLETE' in output) == (failure == 'cat_none')
    if failure == 'cat_none':
        assert 'CAT none' in output
    else:
        assert 'NOT CONFIRMED' in output
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('point,action', [(point, action) for point in FAULTS
                                       for action in ('raise', 'kill', 'term')]
                         + [(point, 'failure') for point in ('high', 'unreadable', 'cat', 'evidence')])
def test_prove_down_exit_propagates(proof_run, point, action):
    run = proof_run
    if action == 'failure':
        env = dict(run.env)
        if point in ('high', 'unreadable'):
            env['CLEANUP_LINE_FAULT'] = point
        elif point == 'cat':
            run.station.write_text('[rig]\ncat = "yaesu"\n')
            env['CLEANUP_CAT_FAIL'] = '1'
        else:
            (run.root / 'cleanup.json').mkdir()
        output = run.finish(run.shell(env=env), 1)
        assert 'NOT CONFIRMED' in output and 'CLEANUP COMPLETE' not in output
        if point in ('high', 'unreadable'):
            assert 'PTT LINE ' + point in output
        if point == 'high':
            assert run.sense() is True
        assert_released(run.snapshot('failed_proof'), paths=run.h.paths,
                        claim_path=run.root / 'rig.claim')
        if point == 'evidence':
            (run.root / 'cleanup.json').rmdir()
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
        run.released(line=run.sense(), require_line=True)
        return
    survivor = supervisor = None
    if point in ('cleanup_cat', 'after_cleanup_cat'):
        run.station.write_text('[rig]\ncat = "yaesu"\n')
    if point in ('before_cleanup_signal', 'after_cleanup_signal'):
        supervisor, survivor = _dead_root(run)
    if action == 'term':
        run.h.pin(point)
        actor = run.shell()
        event = run.h.reached(point, quiescent=True)
        os.kill(event['pid'], signal.SIGTERM)
        run.h.release(point)
    else:
        actor = run.shell(env=dict(run.env, HFMODEM_COEXIST_FAULT=f'{point}:{action}'))
    actor.wait(timeout=20)
    output = ''.join(path.read_text() for path in run.outputs[actor])
    assert actor.returncode != 0, output
    assert any(e['event'] == ('barrier' if action == 'term' else 'fault') and e['point'] == point
               for e in run.h.events())
    if point not in ('before_release', 'after_release'):
        assert 'CLEANUP COMPLETE' not in output
    if supervisor is not None:
        if point == 'before_cleanup_signal':
            nonce = int((run.root / 'rig.claim').read_text().split()[4])
            assert_claim_possession(run.snapshot('signal_not_sent'), pid=survivor['pid'],
                                    claim_path=run.root / 'rig.claim', nonce=nonce)
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
        run.h.release('root_dead')
        run.finish(supervisor)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('kind', ['shrike.onair', 'kestrel_connect', 'besra.host.run_server'])
def test_modem_as_root_holder(proof_run, kind):
    run = proof_run
    run.h.pin('modem_ready')
    owner = run.python(f'''
held = claim.acquire('modem', {kind!r}, prove_line=True,
                     ptt_factory=claim.line_factory(os.environ['PTT_PORT']))
coexist.checkpoint('modem_ready')
claim.release(held)
''')
    event = run.h.reached('modem_ready', quiescent=True)
    output = run.finish(run.shell(), 7)
    assert f"pid {event['pid']}" in output and 'holds the claim' in output
    assert 'SIGTERM' not in output and 'CLEANUP COMPLETE' not in output
    run.h.release('modem_ready')
    run.finish(owner)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('latency', [0, 960])
@pytest.mark.parametrize('lead', [.04, .08])
def test_station_reserved_path_lead(cleanup_run, latency, lead):
    run = cleanup_run
    run.observe()
    actor = run.python(f'''
from hfmodem.tests.core.test_arbiter import Card, LineRig, req
from hfmodem.station.arbiter import TxArbiter
held = claim.acquire('test', 'reserved-path')
card = Card()
card.tx_latency_n = {latency}
card.loop_lat = card.lat + card.tx_latency_n
rig = LineRig(card)
arbiter = TxArbiter(card, rig)
try:
    result = arbiter.submit_result(req(settle_s=.04, lead_s={lead}))
    assert result.outcome == 'sent', result
    key_index = next(call[1] for call in rig.calls if call[0] == 'key')
    assert result.emission_start == key_index + round({lead} * 48000) + {latency}
    assert result.emission_end == result.emission_start + 4800
    assert not rig.keyed
finally:
    arbiter.close()
    claim.release(held)
''')
    run.finish(actor)
    run.released()


@pytest.mark.parametrize('cwd', ['same', 'different'])
def test_same_port_and_file_resolved(cleanup_run, cwd):
    import errno
    import socket

    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 0))
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EPERM):
                pytest.skip(f'sandbox blocks loopback binds; conductor reruns outside: {exc}')
            raise
    run = cleanup_run
    run.observe()
    run.h.pin('service_ready')
    server = run.python('''
import json
from hfmodem.station.process import Station
from hfmodem.station.service import Service
from hfmodem.tests.station.test_process import configuration
path = Path(os.environ['HFMODEM_STATION'])
station = Station(configuration(transmit=False), station_file=path)
service = Service(path, station=station, port=0)
station.service = service
service.bind()
try:
    path.write_text('schema = 1\\n[station]\\ncontrol = "local"\\nregulatory = "unregulated"\\n'
                    'because = "offline fixture"\\nport = ' + str(service.port) + '\\n[rig]\\ncat = "none"\\n')
    Path(os.environ['HFMODEM_STATE_DIR'], 'listener.json').write_text(
        json.dumps(dict(pid=os.getpid(), port=service.port, incarnation=service.incarnation)))
    coexist.checkpoint('service_ready')
finally:
    service.close()
''')
    run.h.reached('service_ready', quiescent=True)
    directory = run.root / 'elsewhere'
    directory.mkdir()
    output = run.finish(run.shell(f'cd {shlex.quote(str(directory))}\nstation_down'))
    assert 'PTT LINE none' in output and 'CLEANUP COMPLETE' in output
    console = run.python(f'''
import json
from consolelib import claim as console_claim
os.chdir({str(directory if cwd == 'same' else run.root)!r})
client = console_claim.release(os.environ['HFMODEM_STATION'])
assert client is not None
print(json.dumps(client.state()['service']))
''')
    output = run.finish(console)
    actual = json.loads(output.strip().splitlines()[-1])
    expected = json.loads((run.root / 'listener.json').read_text())
    assert actual['incarnation'] == expected['incarnation'] and actual['pid'] == expected['pid']
    run.h.release('service_ready')
    run.finish(server)
    run.released()


MODEM = HEADER + '''
import sounddevice as sd
admission = claim.inherited(9, launch=True)
stream = sd.InputStream()
stream.start()
admission.opened()
signal.signal(signal.SIGINT, signal.SIG_IGN)
coexist.checkpoint('modem_ready')
try:
    signal.pause()
finally:
    stream.close()
    claim.close(9)
'''


RECORDER = '''
import os, signal
from pathlib import Path
fd = os.open(Path(os.environ['HFMODEM_STATE_DIR']) / 'tap-audio', os.O_RDONLY)
signal.pause()
os.close(fd)
'''


def _console_source(run, point, *, modem=MODEM):
    (run.root / 'tap-audio').touch()
    down = run.root / 'fake-down'
    down.write_text('#!/bin/bash\nset -euo pipefail\n'
        f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
        f'source {shlex.quote(str(TREE / "tools/lib/rig.sh"))}\nstation_down\n')
    down.chmod(0o700)
    return f'''
import io, json
from contextlib import nullcontext
from types import SimpleNamespace as NS
from consolelib import arm, claim as console_claim
point = {point!r}
root = Path(os.environ['HFMODEM_STATE_DIR'])
selection = NS(CAT_PORT=os.environ['CAT_PORT'], PTT_PORT=os.environ['PTT_PORT'],
               AUDIO_IN='fake audio', AUDIO_OUT='fake audio', _STATION_CONFIG=NS(rig=NS(cat='none')))
console_claim.station.resolve_selection = lambda **kw: selection
console_claim.release = lambda *a, **kw: None
console_claim.spacing_note = lambda *a: None
arm.station.ONAIR = Path({str(down)!r})
arm.qualified = lambda mode, plan, qualify: (plan, dict(modules=[], refusal=None))
arm.ready = lambda *a: nullcontext()
arm.spelled = lambda *a: 'fake flight'
arm.passthrough_note = lambda *a: None
arm.station.spell_environment = lambda *a, **kw: []
arm.station.selection_for_plan = lambda *a: selection
arm.residency.label = lambda *a: 'fake'
arm.residency.evidence = lambda *a: dict(modules=[])
arm.verdict = lambda *a: dict(reached=[], line='fake result')
def preparation(plan):
    if point == 'preparation':
        coexist.checkpoint('console_point')
arm.station.prepare_radio = preparation
arm.station.check_rig = lambda *a: {{}}
def readback(**kw):
    if point == 'readback':
        coexist.checkpoint('console_point')
    return {{}}
arm.station.read_environment = readback
def tap(*a, **kw):
    if point not in ('recorder_start', 'recorder_wait'):
        return None
    process = subprocess.Popen([sys.executable, '-c', {RECORDER!r}])
    waited = process.wait
    def wait(*a, **kw):
        if point == 'recorder_wait':
            coexist.checkpoint('console_point')
        return waited(*a, **kw)
    process.wait = wait
    return process
arm.station.start_tap = tap
arm.station.place_kiwi = lambda *a, **kw: None
original_settle = arm.settle
def settle(*args):
    if point == 'nested':
        return original_settle(*args)
    coexist.checkpoint('console_settle')
    original_pump = arm.pump
    def expired(*args):
        raise TimeoutError('injected settle deadline')
    arm.pump = expired
    try:
        return original_settle(*args)
    finally:
        arm.pump = original_pump
arm.settle = settle
remember = console_claim.remember
def remembered(process):
    result = remember(process)
    if point == 'spawn' and process.args[-1] == {modem!r}:
        coexist.checkpoint('console_point')
    if point == 'recorder_start' and process.args[-1] == {RECORDER!r}:
        coexist.checkpoint('console_point')
    return result
console_claim.remember = remembered
write_text = Path.write_text
def write(path, text, *args, **kw):
    if point == 'evidence' and path.name == 'provenance.json' and 'ended_utc' in text:
        coexist.checkpoint('console_point')
    return write_text(path, text, *args, **kw)
Path.write_text = write
plan = dict(directory=str(root / 'flight'), call='N0CALL', band_m=40,
            codec_gain_working_point=0, delivery=None, arm='fake', env=dict(os.environ),
            adjustments=dict(prepare_radio=True, kiwi=False, tap_tail_s=0),
            command=[sys.executable, '-c', {modem!r}])
try:
    arm._fly(NS(name='test'), plan)
except KeyboardInterrupt:
    pass
coexist.checkpoint('console_released')
'''


@pytest.mark.parametrize('point', ['preparation', 'spawn', 'recorder_start', 'recorder_wait',
                                 'settle', 'readback', 'evidence'])
def test_console_ends_children_before_release(proof_run, point):
    run = proof_run
    run.h.pin('console_point')
    run.h.pin('modem_ready')
    run.h.pin('console_released')
    if point == 'settle':
        run.h.pin('console_settle')
    actor = run.python(_console_source(run, point))
    if point in ('preparation', 'spawn', 'recorder_start'):
        event = run.h.reached('console_point', quiescent=True)
        os.kill(event['pid'], signal.SIGINT)
        run.h.release('console_point')
    else:
        modem = run.h.reached('modem_ready', quiescent=True)
        root = int((run.root / 'rig.claim').read_text().split()[0])
        if point == 'settle':
            os.kill(root, signal.SIGINT)
            run.h.reached('console_settle', quiescent=True)
            os.kill(root, signal.SIGINT)
            run.h.release('console_settle')
            run.h.release('modem_ready')
        else:
            os.kill(modem['pid'], signal.SIGTERM)
            event = run.h.reached('console_point', quiescent=True)
            os.kill(event['pid'], signal.SIGINT)
            run.h.release('console_point')
    run.h.reached('console_released', quiescent=True)
    assert_released(run.snapshot('console_done'), paths=run.h.paths,
                    claim_path=run.root / 'rig.claim', line=run.sense(), require_line=True)
    run.h.release('console_released')
    run.finish(actor)
    evidence = run.released(line=run.sense(), require_line=True)
    releases = [e for e in evidence.events if e['event'] == 'claim_released']
    proofs = [e for e in evidence.events if e['event'] == 'line_readback' and e['verdict'] == 'low']
    assert len(releases) == 1 and proofs
    assert max(e['monotonic_ns'] for e in proofs) < releases[0]['monotonic_ns']


@pytest.mark.parametrize('target', ['console', 'group'])
def test_interrupt_storms(proof_run, target):
    run = proof_run
    run.h.pin('modem_ready')
    run.h.pin('during_line_proof')
    run.h.pin('console_released')
    source = _console_source(run, 'storm')
    if sys.platform == 'linux':
        source = 'os.setsid()\n' + source
        actor = run.python(source)
    else:
        actor = run.python(source, start_new_session=True)
    run.h.reached('modem_ready', quiescent=True)
    root = int((run.root / 'rig.claim').read_text().split()[0])
    send = (lambda: os.kill(root, signal.SIGINT)) if target == 'console' else (
        lambda: os.killpg(root, signal.SIGINT))
    send()
    run.h.release('modem_ready')
    run.h.reached('during_line_proof', quiescent=True)
    for _ in range(4):
        os.kill(root, signal.SIGINT)
    run.h.release('during_line_proof')
    run.h.reached('console_released', quiescent=True)
    run.h.release('console_released')
    run.finish(actor)
    evidence = run.released(line=run.sense(), require_line=True)
    assert sum(e['event'] == 'cleanup_acquired' for e in evidence.events) == 1
    assert sum(e['event'] == 'claim_released' for e in evidence.events) == 1


def test_console_interrupt_cleanup_inherits(proof_run):
    run = proof_run
    for point in ('modem_ready', 'during_line_proof', 'console_point', 'console_released'):
        run.h.pin(point)
    actor = run.python(_console_source(run, 'readback'))
    modem = run.h.reached('modem_ready', quiescent=True)
    os.kill(modem['pid'], signal.SIGTERM)
    run.h.reached('during_line_proof', quiescent=True)
    contender = '''
try:
    claim.acquire('station', 'contender')
except claim.ClaimBusy as exc:
    print(str(exc))
else:
    raise AssertionError('successor acquired before console released')
'''
    assert 'cleanup in progress' in run.finish(run.python(contender))
    run.h.release('during_line_proof')
    run.h.reached('console_point', quiescent=True)
    assert 'rig claimed' in run.finish(run.python(contender))
    run.h.release('console_point')
    run.h.reached('console_released', quiescent=True)
    run.h.release('console_released')
    run.finish(actor)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('owner', ['legacy', 'registry'])
def test_two_consoles(proof_run, owner):
    if owner == 'registry':
        _registry_session(proof_run, refuse_fly=True)
        return
    run = proof_run
    run.h.pin('console_point')
    run.h.pin('console_released')
    first = run.python(_console_source(run, 'preparation'))
    event = run.h.reached('console_point', quiescent=True)
    second = run.python(_console_source(run, 'preparation'))
    assert 'rig claimed' in run.finish(second, 1)
    os.kill(event['pid'], signal.SIGINT)
    run.h.release('console_point')
    run.h.reached('console_released', quiescent=True)
    run.h.release('console_released')
    run.finish(first)
    evidence = run.released(line=run.sense(), require_line=True)
    assert sum(e['event'] == 'claim_acquired' for e in evidence.events) == 1


def _station_owner(child):
    source = HEADER + '''
from types import SimpleNamespace as NS
import sounddevice as sd
from hfmodem.station.process import Station
from hfmodem.tests.station.test_process import configuration
station = Station(configuration(transmit=False), station_file=os.environ['HFMODEM_STATION'])
def audio():
    stream = sd.InputStream()
    stream.start()
    station.audio = NS(breathe=lambda: None, close=stream.close)
station.open_audio = audio
station.open_rig = lambda **kw: None
station.build_arbiter = lambda: None
station.build_lanes = lambda: None
station._tick = lambda: station._poll_stop.wait()
station.own(receive=True)
'''
    if child:
        body = HEADER + '''
held = claim.inherited(9)
'''
        if child == 'tty_only':
            body += '''
import fcntl, termios
claim.close(held)
fd = os.open(os.environ['CLEANUP_PTT_DEVICE'], os.O_RDWR | os.O_NOCTTY)
fcntl.ioctl(fd, termios.TIOCEXCL)
coexist.event('device_open', path=os.environ['CLEANUP_PTT_DEVICE'])
coexist.checkpoint('survivor_ready')
fcntl.ioctl(fd, termios.TIOCNXCL)
os.close(fd)
coexist.checkpoint('survivor_closed')
'''
        else:
            body += '''
coexist.checkpoint('survivor_ready')
claim.close(held)
'''
        source += f'''
os.dup2(station.claim.fd, 9)
child = subprocess.Popen([sys.executable, '-c', {body!r}], pass_fds=(9,))
'''
    source += "coexist.checkpoint('station_ready')\nstation.release()\n"
    return source


@pytest.mark.parametrize('child', [None, 'claim', 'tty_only'], ids=['no_child', 'claim_child', 'tty_only_child'])
def test_station_crash_while_owning(proof_run, child):
    from .oracle import OracleFailure

    run = proof_run
    for point in ('station_ready', 'survivor_ready', 'survivor_closed', 'root_dead'):
        run.h.pin(point)
    owner = _station_owner(child)
    supervisor = run.python(f'''
owner = subprocess.Popen([sys.executable, '-c', {owner!r}])
assert owner.wait(timeout=15) == -signal.SIGKILL
coexist.checkpoint('root_dead')
''')
    station = run.h.reached('station_ready', quiescent=True)
    survivor = run.h.reached('survivor_ready', quiescent=True) if child else None
    os.kill(station['pid'], signal.SIGKILL)
    run.h.reached('root_dead', quiescent=True)
    if child == 'claim':
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('orphan'), pid=survivor['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
    elif child is None:
        assert probe_claim(run.root / 'rig.claim', quiescent=True)
    else:
        with pytest.raises(OracleFailure, match='surviving hardware authority'):
            assert_released(run.snapshot('tty_survived'), paths=run.h.paths,
                            claim_path=run.root / 'rig.claim')
    output = run.finish(run.shell(), 1 if child == 'tty_only' else 0)
    if child == 'tty_only':
        assert 'unidentified' in output and 'CLEANUP COMPLETE' not in output
        run.h.release('survivor_ready')
        run.h.reached('survivor_closed', quiescent=True)
        run.h.release('survivor_closed')
    else:
        assert 'CLEANUP COMPLETE' in output
        if child:
            assert f"pid {survivor['pid']} -- SIGTERM" in output
    run.h.release('root_dead')
    run.finish(supervisor)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('holder', ['free', 'station', 'launcher'])
def test_rigtest_direct_down_proof(proof_run, holder):
    run = proof_run
    owner = None
    run.h.pin('live_owner')
    if holder == 'launcher':
        wait = 'from hfmodem.core import claim, coexist; claim.inherited(9); coexist.checkpoint("live_owner")'
        owner = run.shell('claim_rig launcher\nclaim_admit\n'
            + shlex.quote(sys.executable) + ' -c ' + shlex.quote(wait)
            + '\nclaim_close\ntrap - EXIT\n')
    elif holder == 'station':
        source = _station_owner(None)
        source = source.replace("coexist.checkpoint('station_ready')", '''
import json
from hfmodem.station.service import Service
path = Path(os.environ['HFMODEM_STATION'])
service = Service(path, station=station, port=0)
station.service = service
service.bind()
path.write_text('[station]\\nport = ' + str(service.port) + '\\n[rig]\\ncat = "none"\\n')
Path(os.environ['HFMODEM_STATE_DIR'], 'listener.json').write_text(
    json.dumps(dict(pid=os.getpid(), port=service.port)))
try:
    coexist.checkpoint('live_owner')
finally:
    service.close()
''')
        owner = run.python(source)
    if owner:
        run.h.reached('live_owner', quiescent=True)
    actor = run.launch(['/bin/bash', str(TREE / 'tools/rigtest.sh'), 'down'])
    output = run.finish(actor, 7 if holder == 'launcher' else 0)
    if holder == 'launcher':
        assert 'holds the claim' in output and 'CLEANUP COMPLETE' not in output
    else:
        assert 'PTT LINE low' in output and 'CLEANUP COMPLETE' in output
    if owner:
        run.h.release('live_owner')
        run.finish(owner)
    run.released(line=run.sense(), require_line=True)


def _metadata(command, *, started=None):
    return f'''
import json
Path(os.environ['HFMODEM_STATE_DIR'], 'metadata-' + str(os.getpid()) + '.json').write_text(
    json.dumps(dict(command={command!r}, started={started!r})))
'''


def _survivor(kind='claim', *, point='survivor_ready', command='coexist-child'):
    source = HEADER + _metadata(command)
    if kind == 'admission':
        source += 'admission = claim.inherited(9, launch=True)\n'
    elif kind != 'tty_only':
        source += 'claim.inherited(9)\n'
    else:
        source += 'claim.close(9)\n'
    if kind == 'cleanup':
        source += 'claim.verify_cleanup(8)\n'
    source += f'''
fd = os.open(os.environ['CLEANUP_PTT_DEVICE'], os.O_RDWR | os.O_NOCTTY)
coexist.checkpoint({point!r})
os.close(fd)
'''
    if kind == 'admission':
        source += 'admission.failed()\n'
    if kind != 'tty_only':
        source += 'claim.close(9)\n'
    return source


def _dead_root(run, kind='claim', *, command='coexist-child', depth=1):
    for point in ('survivor_ready', 'root_ready', 'root_dead'):
        run.h.pin(point)
    body = _survivor(kind, command=command)
    if depth == 2:
        body = HEADER + f'''
claim.inherited(9)
child = subprocess.Popen([sys.executable, '-c', {body!r}], pass_fds=(8, 9) if {kind!r} == 'cleanup' else (9,))
assert child.wait(timeout=40) in (0, -signal.SIGTERM)
claim.close(9)
'''
    root = '''
held = claim.acquire('launcher', 'crash')
os.dup2(held.fd, 9)
if held.fd != 9:
    claim.close(held)
'''
    if kind == 'cleanup':
        root += '''
mutex = claim.mutex()
os.dup2(mutex, 8)
if mutex != 8:
    os.close(mutex)
'''
    root += f'''
child = subprocess.Popen([sys.executable, '-c', {body!r}], pass_fds={((8, 9) if kind == 'cleanup' else (9,))!r})
coexist.checkpoint('root_ready')
assert child.wait(timeout=40) == 0
'''
    supervisor = run.supervise(root, -signal.SIGKILL)
    owner = run.h.reached('root_ready', quiescent=True)
    survivor = run.h.reached('survivor_ready', quiescent=True)
    os.kill(owner['pid'], signal.SIGKILL)
    run.h.reached('root_dead', quiescent=True)
    return supervisor, survivor


@pytest.mark.parametrize('kind,depth', [('cat', 1), ('proof', 1), ('claim', 2),
                                     ('admission', 1), ('cleanup', 1), ('cleanup', 2),
                                     ('tty_only', 1), ('live_root_orphan', 2)])
def test_root_killed_children_survive(proof_run, kind, depth):
    from .oracle import OracleFailure

    run = proof_run
    if kind == 'live_root_orphan':
        run.h.pin('survivor_ready')
        run.h.pin('root_ready')
        child = _survivor()
        middle = HEADER + f'''
claim.inherited(9)
subprocess.Popen([sys.executable, '-c', {child!r}], pass_fds=(9,))
os._exit(0)
'''
        actor = run.shell('claim_rig launcher\nclaim_admit\n'
            + shlex.quote(sys.executable) + ' -c ' + shlex.quote(middle)
            + '\n' + shlex.quote(sys.executable) + ' -c '
            + shlex.quote(HEADER + "coexist.checkpoint('root_ready')"))
        run.h.reached('survivor_ready', quiescent=True)
        run.h.reached('root_ready', quiescent=True)
        assert 'holds the claim' in run.finish(run.shell(), 7)
        run.h.release('root_ready')
        run.finish(actor)
        run.released(line=run.sense(), require_line=True)
        return
    actual = kind if kind in ('admission', 'cleanup', 'tty_only') else 'claim'
    actor, survivor = _dead_root(run, actual, depth=depth)
    if actual != 'tty_only':
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('orphan'), pid=survivor['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
    if actual == 'admission':
        output = run.finish(run.python('''
try:
    claim.mutex(timeout=0)
except claim.ClaimBusy as exc:
    print(exc)
else:
    raise AssertionError('admission lost after root death')
'''))
        assert 'admission in progress' in output and str(survivor['pid']) in output
        run.h.release('survivor_ready')
    elif actual == 'cleanup':
        nonce = int((run.root / 'rig.claim.cleanup').read_text().split()[2])
        snapshot = run.snapshot('cleanup_child')
        assert any(fd.get('pos') == nonce and fd.get('path') == str(run.root / 'rig.claim.cleanup')
                   for fd in snapshot.descriptors[survivor['pid']])
        output = run.finish(run.shell(), 2)
        assert 'cleanup in progress' in output and str(survivor['pid']) in output
        assert 'CLEANUP COMPLETE' not in output
        run.h.release('survivor_ready')
    elif actual == 'tty_only':
        with pytest.raises(OracleFailure, match='surviving hardware authority'):
            assert_released(run.snapshot('tty_only'), paths=run.h.paths,
                            claim_path=run.root / 'rig.claim')
        output = run.finish(run.shell(), 1)
        assert 'unidentified' in output and 'CLEANUP COMPLETE' not in output
        run.h.release('survivor_ready')
    else:
        output = run.finish(run.shell())
        assert f"pid {survivor['pid']} -- SIGTERM" in output
        assert 'CLEANUP COMPLETE' in output
    run.h.release('root_dead')
    run.finish(actor)
    if actual in ('admission', 'cleanup', 'tty_only'):
        for traced in run.h.traced:
            traced.process.wait(timeout=20)
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
    if actual == 'tty_only':
        assert_released(run.snapshot('injected_child_gone'), paths=run.h.paths,
                        claim_path=run.root / 'rig.claim', line=run.sense(), require_line=True)
        with pytest.raises(OracleFailure, match='unlogged (sampled holder|traced opener)'):
            validate(run.h.collect())
    else:
        run.released(line=run.sense(), require_line=True)


def test_launcher_crash_modem_alive(proof_run):
    run = proof_run
    actor, survivor = _dead_root(run, command='hfmodem.shrike.onair')
    nonce = int((run.root / 'rig.claim').read_text().split()[4])
    assert_claim_possession(run.snapshot('modem_owner'), pid=survivor['pid'],
                            claim_path=run.root / 'rig.claim', nonce=nonce)
    output = run.finish(run.shell())
    assert f"pid {survivor['pid']} -- SIGTERM" in output and 'CLEANUP COMPLETE' in output
    run.h.release('root_dead')
    run.finish(actor)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('form', ['external', 'inherited'])
@pytest.mark.parametrize('identity', ['unidentified', 'legacy'])
def test_down_discovers_before_proof(proof_run, form, identity):
    from .oracle import OracleFailure

    run = proof_run
    run.h.pin('device_ready')
    command = 'hfmodem.lanner.qso' if identity == 'legacy' else 'unrelated'
    child = HEADER + _metadata(command) + '''
fd = os.open(os.environ['CLEANUP_PTT_DEVICE'], os.O_RDWR | os.O_NOCTTY)
coexist.checkpoint('device_ready')
os.close(fd)
'''
    if form == 'external':
        actor = run.supervise(child, -signal.SIGTERM if identity == 'legacy' else 0)
        device = run.h.reached('device_ready', quiescent=True)
        output = run.finish(run.shell(), 0 if identity == 'legacy' else 1)
    else:
        run.h.pin('inherited_down')
        run.h.pin('inherited_result')
        body = f'''claim_rig launcher
claim_admit
{shlex.quote(sys.executable)} -c {shlex.quote(child)} &
child=$!
{shlex.quote(sys.executable)} -c {shlex.quote(HEADER + "coexist.checkpoint('inherited_down')")}
rc=0
( station_down ) || rc=$?
{shlex.quote(sys.executable)} -c {shlex.quote(HEADER + "coexist.checkpoint('inherited_result')")}
wait "$child" || true
CLAIM_FAILED=1
claim_close
trap - EXIT
exit "$rc"
'''
        actor = run.shell(body)
        device = run.h.reached('device_ready', quiescent=True)
        run.h.reached('inherited_down', quiescent=True)
        run.h.release('inherited_down')
        run.h.reached('inherited_result', quiescent=True)
        if identity == 'unidentified':
            run.h.release('device_ready')
        run.h.release('inherited_result')
        output = run.finish(actor, 0 if identity == 'legacy' else 1)
    assert ('CLEANUP COMPLETE' in output) == (identity == 'legacy')
    if identity == 'unidentified':
        assert 'unidentified' in output and 'PTT LINE low' not in output
        if form == 'external':
            assert device['pid'] in run.snapshot('not_signalled').participants
            run.h.release('device_ready')
    else:
        assert f"pid {device['pid']} -- SIGTERM" in output
    if form == 'external':
        run.finish(actor)
    if identity == 'unidentified':
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
    assert_released(run.snapshot('device_gone'), paths=run.h.paths,
                    claim_path=run.root / 'rig.claim', line=run.sense(), require_line=True)
    with pytest.raises(OracleFailure, match='unlogged (sampled holder|traced opener)'):
        validate(run.h.collect())


@pytest.mark.parametrize('flags', ['matching', 'different'])
def test_rigctld_leaked(proof_run, flags):
    from .oracle import OracleFailure

    run = proof_run
    run.h.pin('daemon_ready')
    source = _metadata('rigctld -m 1036' if flags == 'matching' else 'rigctld -m 9999') + '''
fd = os.open(os.environ['CLEANUP_PTT_DEVICE'], os.O_RDWR | os.O_NOCTTY)
coexist.checkpoint('daemon_ready')
os.close(fd)
'''
    actor = run.supervise(source, -signal.SIGTERM if flags == 'matching' else 0)
    daemon = run.h.reached('daemon_ready', quiescent=True)
    output = run.finish(run.shell(), 0 if flags == 'matching' else 1)
    if flags == 'matching':
        assert f"pid {daemon['pid']} -- SIGTERM" in output and 'CLEANUP COMPLETE' in output
    else:
        assert 'unidentified' in output and 'CLEANUP COMPLETE' not in output
        assert daemon['pid'] in run.snapshot('daemon_not_signalled').participants
        run.h.release('daemon_ready')
    run.finish(actor)
    assert_released(run.snapshot('daemon_gone'), paths=run.h.paths,
                    claim_path=run.root / 'rig.claim')
    with pytest.raises(OracleFailure, match='unlogged (sampled holder|traced opener)'):
        validate(run.h.collect())


@pytest.mark.parametrize('case', ['unrelated', 'reused_pid', 'metadata_unreadable',
                                'stale_dead_pid', 'helper_pid'])
def test_down_signals_only_identified(proof_run, case):
    run = proof_run
    if case == 'unrelated':
        test_down_discovers_before_proof(run, 'external', 'unidentified')
        return
    if case == 'stale_dead_pid':
        writer = run.python('''
held = claim.acquire('launcher', 'old-root')
line = claim.claim_path().read_text()
claim.release(held)
claim.claim_path().write_text(line)
''')
        run.finish(writer)
        assert probe_claim(run.root / 'rig.claim', quiescent=True)
        output = run.finish(run.shell())
        assert 'CLEANUP COMPLETE' in output and 'SIGTERM' not in output
        assert (run.root / 'rig.claim').read_text() == ''
    else:
        run.h.pin('holding')
        if case == 'helper_pid':
            actor = run.shell('claim_rig launcher\nclaim_admit\n'
                + shlex.quote(sys.executable) + ' -c '
                + shlex.quote(HEADER + "claim.inherited(9)\ncoexist.checkpoint('holding')")
                + '\nclaim_close\ntrap - EXIT')
        else:
            actor = run.python('''
held = claim.acquire('launcher', 'identity')
coexist.checkpoint('holding')
claim.release(held)
''')
        barrier = run.h.reached('holding', quiescent=True)
        original = (run.root / 'rig.claim').read_text()
        published = original.split()
        owner, nonce = int(published[0]), int(published[4])
        assert_claim_possession(run.snapshot('identity'), pid=owner,
                                claim_path=run.root / 'rig.claim', nonce=nonce)
        env = dict(run.env)
        if case == 'reused_pid':
            published[1] = (published[1].rsplit(':', 1)[0] + ':0'
                            if published[1].startswith('linux:') else 'Thu_Jan_1_00:00:00_1970')
            (run.root / 'rig.claim').write_text(' '.join(published) + '\n')
        elif case == 'metadata_unreadable':
            env['CLEANUP_DISCOVERY_FAULT'] = 'discovery'
        else:
            assert owner != barrier['pid']
        output = run.finish(run.shell(env=env), 7 if case == 'helper_pid' else 1)
        assert 'CLEANUP COMPLETE' not in output and 'SIGTERM' not in output
        assert ('publisher pid reused' in output if case == 'reused_pid' else
                'holds the claim' in output if case == 'helper_pid' else 'NOT CONFIRMED' in output)
        assert owner in run.snapshot('not_signalled').participants
        (run.root / 'rig.claim').write_text(original)
        run.h.release('holding')
        run.finish(actor)
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('form', ['direct', 'timeout'])
def test_autopilot_trap_down(proof_run, form):
    run = proof_run
    run.h.pin('modem_ready')
    run.h.pin('trap_ready')
    target = run.root / 'targets'
    target.touch()
    script = _wrapper_text('autopilot.sh')
    trap = 'stop() {' + script.split('\nstop() {', 1)[1].split('\nprintf ', 1)[0]
    down = ''
    if form == 'timeout':
        down = '''claim_down() {
CAT_PORT="$HFMODEM_STATE_DIR/absent-cat" PTT_PORT="$HFMODEM_STATE_DIR/absent-ptt" \
timeout -k 5 60 "$REPO/tools/onair.sh" down 7>&- 8>&-
}
'''
    body = f'''claim_rig autopilot
claim_admit
TARGETS={shlex.quote(str(target))}
{down}
{trap}
{shlex.quote(sys.executable)} -c {shlex.quote(MODEM)} 7>&- 8>&- &
claim_record "$!"
{shlex.quote(sys.executable)} -c {shlex.quote(HEADER + "coexist.checkpoint('trap_ready')")}
exit 0
'''
    actor = run.shell(body)
    child = run.h.reached('modem_ready', quiescent=True)
    run.h.reached('trap_ready', quiescent=True)
    run.h.release('trap_ready')
    output = run.finish(actor)
    assert f"pid {child['pid']} -- SIGTERM" in output
    assert output.count('CLEANUP COMPLETE') == 1 and output.count('PTT LINE low') == 1
    assert not target.exists()
    evidence = run.released(line=run.sense(), require_line=True)
    assert sum(event['event'] == 'cleanup_acquired' for event in evidence.events) == 1


@pytest.mark.parametrize('parent', ['console', 'pounce', 'shell_campaign'])
def test_inherited_launcher_normal_exit(proof_run, parent):
    run = proof_run
    run.h.pin('between_attempts')
    run.h.pin('after_attempts')
    launcher = run.root / 'launcher'
    launcher.write_text('#!/bin/bash\nset -euo pipefail\n'
        f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
        'claim_rig launcher\nclaim_admit\nclaim_close\ntrap - EXIT\n')
    launcher.chmod(0o700)
    check = HEADER + "claim.inherited(9)\ncoexist.checkpoint('between_attempts')"
    final = HEADER + "claim.inherited(9)\ncoexist.checkpoint('after_attempts')"
    if parent == 'console':
        source = _console_source(run, 'normal').split('plan = dict(directory=', 1)[0]
        actor = run.python(source + f'''
with console_claim.lease(NS(name='campaign'), env=dict(os.environ)):
    first = subprocess.Popen([{str(launcher)!r}], pass_fds=(9,))
    assert first.wait(timeout=15) == 0
    claim.inherited(9)
    coexist.checkpoint('between_attempts')
    second = subprocess.Popen([{str(launcher)!r}], pass_fds=(9,))
    assert second.wait(timeout=15) == 0
    claim.inherited(9)
    coexist.checkpoint('after_attempts')
    assert arm.bring_down(root / 'cleanup', env=dict(os.environ))['complete']
''')
    else:
        actor = run.shell(f'''claim_rig {parent}
claim_run {shlex.quote(str(launcher))}
{shlex.quote(sys.executable)} -c {shlex.quote(check)}
claim_run {shlex.quote(str(launcher))}
{shlex.quote(sys.executable)} -c {shlex.quote(final)}
''')
    run.h.reached('between_attempts', quiescent=True)
    line = (run.root / 'rig.claim').read_text()
    root, _, _, _, nonce, *_ = line.split()
    assert_claim_possession(run.snapshot('parent'), pid=int(root),
                            claim_path=run.root / 'rig.claim', nonce=int(nonce))
    run.h.release('between_attempts')
    run.h.reached('after_attempts', quiescent=True)
    assert (run.root / 'rig.claim').read_text() == line
    run.h.release('after_attempts')
    run.finish(actor)
    evidence = run.released(line=run.sense(), require_line=True)
    assert sum(event['event'] == 'claim_acquired' for event in evidence.events) == 1
    assert sum(event['event'] == 'claim_released' for event in evidence.events) == 1


def _service_owner():
    return _station_owner(None).replace("coexist.checkpoint('station_ready')", '''
import json
from hfmodem.station.service import Service
path = Path(os.environ['HFMODEM_STATION'])
service = Service(path, station=station, port=0)
station.service = service
service.bind()
path.write_text('[station]\\nport = ' + str(service.port) + '\\n[rig]\\ncat = "none"\\n')
Path(os.environ['HFMODEM_STATE_DIR'], 'listener.json').write_text(
    json.dumps(dict(pid=os.getpid(), port=service.port)))
try:
    coexist.checkpoint('station_ready')
finally:
    service.close()
''')


def test_wedged_station(proof_run):
    run = proof_run
    run.h.pin('station_ready')
    actor = run.python(_service_owner())
    station = run.h.reached('station_ready', quiescent=True)
    os.kill(station['pid'], signal.SIGSTOP)
    try:
        output = run.finish(run.shell(), 1)
        assert 'station not released' in output and 'CLEANUP COMPLETE' not in output
        assert 'SIGTERM' not in output and 'PTT LINE low' not in output
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('wedged'), pid=station['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
        assert not any(event['event'] == 'line_readback' for event in run.h.events())
    finally:
        os.kill(station['pid'], signal.SIGCONT)
        run.h.release('station_ready')
    run.finish(actor)
    assert 'CLEANUP COMPLETE' in run.finish(run.shell())
    run.released(line=run.sense(), require_line=True)


def _console_finalization(run, point):
    run.h.pin('modem_ready')
    run.h.pin('console_released')
    source = _console_source(run, point).replace(
        "coexist.checkpoint('console_point')", "raise OSError('injected finalization failure')")
    actor = run.python(source)
    modem = run.h.reached('modem_ready', quiescent=True)
    os.kill(modem['pid'], signal.SIGTERM)
    run.h.reached('console_released', quiescent=True)
    run.h.release('console_released')
    output = run.finish(actor)
    assert 'console finalization failed' in output
    record = json.loads((run.root / 'flight/console-finalization.json').read_text())
    assert record['cleanup']['complete'] is True
    assert record['cleanup']['line'] == 'PTT LINE low'
    assert record['console finalization failed']
    evidence = run.released(line=run.sense(), require_line=True)
    release = next(event for event in evidence.events if event['event'] == 'claim_released')
    assert any(event['event'] == 'line_readback' and event['verdict'] == 'low'
               and event['monotonic_ns'] < release['monotonic_ns'] for event in evidence.events)


@pytest.mark.parametrize('signal_name', ['SIGTERM', 'SIGHUP'])
def test_owner_teardown_racing_launch(proof_run, signal_name):
    run = proof_run
    for point in ('modem_ready', 'during_line_proof', 'console_released'):
        run.h.pin(point)
    source = _console_source(run, 'race')
    actor = run.python(source)
    modem = run.h.reached('modem_ready', quiescent=True)
    owner = int((run.root / 'rig.claim').read_text().split()[0])
    assert 'holds the claim' in run.finish(run.shell(), 7)
    os.kill(modem['pid'], signal.SIGTERM)
    run.h.reached('during_line_proof', quiescent=True)
    output = run.finish(run.shell(), 2)
    assert 'cleanup in progress' in output and 'CLEANUP COMPLETE' not in output
    os.kill(owner, getattr(signal, signal_name))
    assert owner in run.snapshot('console_during_down').participants
    run.h.release('during_line_proof')
    run.h.reached('console_released', quiescent=True)
    run.h.release('console_released')
    run.finish(actor)
    run.released(line=run.sense(), require_line=True)


@pytest.mark.parametrize('parent', ['shell', 'console'])
def test_campaign_parent_between_attempts(proof_run, parent):
    run = proof_run
    run.h.pin('between_attempts')
    run.h.pin('during_line_proof')
    run.h.pin('competing_launch')
    run.h.pin('launch_refused')
    if parent == 'shell':
        cleanup = run.root / 'inherited-down'
        cleanup.write_text('#!/bin/bash\nset -euo pipefail\n'
            f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
            f'source {shlex.quote(str(TREE / "tools/lib/rig.sh"))}\nstation_down\n')
        cleanup.chmod(0o700)
        marker = HEADER + "coexist.checkpoint('between_attempts')"
        launching = HEADER + "coexist.checkpoint('competing_launch')"
        refused = HEADER + "coexist.checkpoint('launch_refused')"
        actor = run.shell(f'''claim_rig campaign
claim_admit
{shlex.quote(sys.executable)} -c {shlex.quote(marker)}
CAT_PORT="$HFMODEM_STATE_DIR/absent-cat" PTT_PORT="$HFMODEM_STATE_DIR/absent-ptt" \
{shlex.quote(str(cleanup))} 7>&- 8>&- &
down=$!
{shlex.quote(sys.executable)} -c {shlex.quote(launching)} 9>&-
if claim_launch_hold; then exit 99; fi
{shlex.quote(sys.executable)} -c {shlex.quote(refused)} 9>&-
wait "$down"
claim_close
trap - EXIT
''')
        event = run.h.reached('between_attempts', quiescent=True)
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('campaign_holds'), pid=event['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
        run.h.release('between_attempts')
        run.h.reached('during_line_proof', quiescent=True)
        run.h.reached('competing_launch', quiescent=True)
        run.h.release('competing_launch')
        run.h.reached('launch_refused', quiescent=True)
        run.h.release('during_line_proof')
        run.h.release('launch_refused')
        output = run.finish(actor)
        assert 'cleanup in progress' in output and 'CLEANUP COMPLETE' in output
    else:
        source = _console_source(run, 'campaign').split('plan = dict(directory=', 1)[0]
        actor = run.python(source + '''
with console_claim.lease(NS(name='campaign'), env=dict(os.environ)):
    pass
coexist.checkpoint('between_attempts')
coexist.checkpoint('competing_launch')
try:
    with console_claim.lease(NS(name='campaign'), env=dict(os.environ)):
        raise AssertionError('campaign acquired during cleanup')
except claim.ClaimBusy as exc:
    print(exc)
''')
        run.h.reached('between_attempts', quiescent=True)
        assert probe_claim(run.root / 'rig.claim', quiescent=True)
        cleanup = run.shell()
        run.h.reached('during_line_proof', quiescent=True)
        run.h.release('between_attempts')
        run.h.reached('competing_launch', quiescent=True)
        run.h.release('competing_launch')
        assert 'cleanup in progress' in run.finish(actor)
        run.h.release('during_line_proof')
        run.finish(cleanup)
    run.released(line=run.sense(), require_line=True)


def _shell_point(point, *, inherited=False):
    source = HEADER + ('claim.inherited(9)\n' if inherited else '')
    source += f"coexist.checkpoint({point!r})"
    return (shlex.quote(sys.executable) + ' -c ' + shlex.quote(source)
            + ' 7>&- 8>&-' + ('' if inherited else ' 9>&-') + '\n')


def _shell_prelude():
    return ('#!/bin/bash\nset -euo pipefail\n'
            f'source {shlex.quote(str(TREE / "tools/lib/ports.sh"))}\n'
            f'source {shlex.quote(str(TREE / "tools/lib/rig.sh"))}\n')


def _nested_launcher(run, wrapper):
    launcher = run.root / 'nested-launcher'
    child = ('claim_rig launcher\nclaim_admit\n'
             + _shell_point('nested_ready', inherited=True)
             + 'claim_close\ntrap - EXIT\n')
    if wrapper == 'pounce':
        source = (TREE / 'tools/onair.sh').read_text()
        branch = source.split('\npounce)', 1)[1].split('\n  ;;', 1)[0]
        body = ('say() { :; }\nCHANNEL_BUSY=6\n'
                'if [ "$1" = pounce ]; then\n' + branch + '\nelse\n' + child + 'fi\n')
        args = ['pounce', 'pactor', 'N0CALL', '7101500']
    else:
        source = (TREE / 'tools/rigtest.sh').read_text()
        air = 'air() {' + source.split('\nair() {', 1)[1].split('\n}', 1)[0] + '\n}\n'
        body = air + ('GW=N0CALL\nCENTRE=7101500\n'
                      'if [ "$1" = wrapper ]; then\n'
                      'claim_run_original=$(declare -f claim_run)\n'
                      'eval "${claim_run_original/claim_run/run_child}"\n'
                      'claim_run() { shift; run_child "$0" "$@"; }\n'
                      'air check\nelse\n') + child + 'fi\n'
        args = ['wrapper']
    launcher.write_text(_shell_prelude() + body)
    launcher.chmod(0o700)
    return [str(launcher), *args]


@pytest.mark.parametrize('wrapper', ['pounce', 'rigtest_check'])
def test_nested_launcher_inherits(proof_run, wrapper):
    run = proof_run
    run.h.pin('nested_ready')
    actor = run.launch(_nested_launcher(run, wrapper))
    nested = run.h.reached('nested_ready', quiescent=True)
    published = (run.root / 'rig.claim').read_text()
    owner, nonce = int(published.split()[0]), int(published.split()[4])
    assert owner != nested['pid']
    snapshot = run.snapshot('nested_inheritance')
    for pid in (owner, nested['pid']):
        assert_claim_possession(snapshot, pid=pid, claim_path=run.root / 'rig.claim', nonce=nonce)
    assert sum(event['event'] == 'claim_acquired' for event in run.h.events()) == 1
    run.h.release('nested_ready')
    output = run.finish(actor)
    assert 'CLAIM INHERITED' in output and 'CLAIM CLOSED' in output
    assert 'CLEANUP COMPLETE' in output
    evidence = run.released(line=run.sense(), require_line=True)
    assert sum(event['event'] == 'claim_acquired' for event in evidence.events) == 1
    assert sum(event['event'] == 'claim_released' for event in evidence.events) == 1


@pytest.mark.parametrize('holder', ['own_modem', 'station', 'unrelated'])
def test_attempts_cleanup_paths(proof_run, holder):
    run = proof_run
    for point in ('modem_ready', 'attempt_ready', 'device_ready', 'station_ready'):
        run.h.pin(point)
    other = None
    if holder == 'station':
        other = run.python(_service_owner())
        run.h.reached('station_ready', quiescent=True)
    elif holder == 'unrelated':
        other = run.supervise(_metadata('unrelated') + '''
fd = os.open(os.environ['CLEANUP_PTT_DEVICE'], os.O_RDWR | os.O_NOCTTY)
coexist.checkpoint('device_ready')
os.close(fd)
''', 0)
        run.h.reached('device_ready', quiescent=True)
    body = f'source {shlex.quote(str(TREE / "tools/lib/attempts.sh"))}\n'
    body += 'claim_rig campaign\nclaim_admit\n'
    if holder == 'own_modem':
        body += (shlex.quote(sys.executable) + ' -c ' + shlex.quote(MODEM)
                 + ' 7>&- 8>&- &\nclaim_record "$!"\n')
    body += _shell_point('attempt_ready', inherited=True)
    body += ('CAT_PORT="$HFMODEM_STATE_DIR/absent-cat" '
             'PTT_PORT="$HFMODEM_STATE_DIR/absent-ptt" settle\n'
             'claim_close\ntrap - EXIT\n')
    actor = run.shell(body)
    if holder != 'own_modem':
        output = run.finish(actor, 1)
        assert 'CLEANUP COMPLETE' not in output
        assert 'SIGTERM' not in output and 'PTT LINE low' not in output
        assert 'rig claimed' in output if holder == 'station' else 'held by pid' in output
        run.h.release('station_ready' if holder == 'station' else 'device_ready')
        run.finish(other)
        assert 'CLEANUP COMPLETE' in run.finish(run.shell())
        if holder == 'unrelated':
            from .oracle import OracleFailure
            assert_released(run.snapshot('released'), paths=run.h.paths,
                            claim_path=run.root / 'rig.claim', line=run.sense(), require_line=True)
            with pytest.raises(OracleFailure, match='unlogged (sampled holder|traced opener)'):
                validate(run.h.collect())
            return
    else:
        child = run.h.reached('modem_ready', quiescent=True)
        run.h.reached('attempt_ready', quiescent=True)
        run.h.release('attempt_ready')
        output = run.finish(actor)
        assert f"pid {child['pid']} -- SIGTERM" in output
        assert 'CLEANUP COMPLETE' in output
    run.released(line=run.sense(), require_line=True)


def _wrapper_text(wrapper):
    """Private launchers contribute only their claim/trap participant snippets.

    The public proof exercises those participants, not whole launcher operation.
    A development-source parity check pins these extracts to the real launchers.
    """
    if wrapper in {'campaign.sh', 'autopilot.sh'}:
        from .fixtures.wrapper_claims import WRAPPERS

        row = WRAPPERS[wrapper]
        return row['claim'] + '\n' + row['cleanup'] + '\nprintf '

    return (TREE / 'tools' / wrapper).read_text()


def _wrapper_source(run, wrapper, point, ending):
    source = _wrapper_text(wrapper)
    operation = wrapper.removesuffix('.sh')
    if wrapper == 'onair.sh':
        assert 'claim_rig || exit $?' in source
        operation = 'launcher'
    else:
        assert f'claim_rig {operation}' in source
    checkpoint = HEADER + ("coexist.event('wrapper_boundary', shell=int(os.environ['CLEANUP_SHELL_PID']))\n"
                           "coexist.checkpoint('wrapper_point')")
    marker = (shlex.quote(sys.executable) + ' -c ' + shlex.quote(checkpoint)
              + ' 7>&- 8>&- 9>&-\n')
    at = marker + (f'exit {ending}\n' if isinstance(ending, int) else '')
    body = 'export CLEANUP_SHELL_PID=$$\n'
    if point == 'trap_install':
        body += at
    body += f'claim_rig {operation}\n'
    if wrapper == 'autopilot.sh':
        body += 'TARGETS="$HFMODEM_STATE_DIR/targets"\ntouch "$TARGETS"\n'
        body += 'stop() {' + source.split('\nstop() {', 1)[1].split('\nprintf ', 1)[0]
    if point == 'acquisition':
        body += at
    body += 'claim_admit\n'
    if point == 'child_spawn':
        body += at
    body += (shlex.quote(sys.executable) + ' -c ' + shlex.quote(MODEM)
             + ' 7>&- 8>&- &\nclaim_record "$!"\n')
    if point == 'child_alive':
        body += at
    if point == 'cleanup':
        body += ('original_down=$(declare -f claim_down)\n'
                 'eval "${original_down/claim_down/original_down}"\n'
                 'claim_down() {\n' + marker + 'original_down\n}\n')
    body += f'exit {ending if isinstance(ending, int) else 0}\n'
    return body


@pytest.mark.parametrize('wrapper', ['onair.sh', 'campaign.sh', 'autopilot.sh', 'rigtest.sh', 'console'])
@pytest.mark.parametrize('point', ['trap_install', 'acquisition', 'child_spawn', 'child_alive', 'cleanup'])
@pytest.mark.parametrize('ending', ['SIGTERM', 'SIGHUP', 0, 4], ids=['term', 'hup', 'normal', 'nonzero'])
def test_wrapper_signals(proof_run, wrapper, point, ending):
    if wrapper == 'console':
        _console_wrapper(proof_run, point, ending)
        return
    run = proof_run
    run.h.pin('wrapper_point')
    run.h.pin('modem_ready')
    killed = point == 'trap_install' and isinstance(ending, str)
    actor = run.shell(_wrapper_source(run, wrapper, point, ending),
                      supervised_exit=-getattr(signal, ending) if killed else None)
    run.h.reached('wrapper_point', quiescent=True)
    shell_pid = next(e['shell'] for e in run.h.events() if e['event'] == 'wrapper_boundary')
    if point == 'trap_install':
        assert probe_claim(run.root / 'rig.claim', quiescent=True)
    else:
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(run.snapshot('wrapper_holder'), pid=shell_pid,
                                claim_path=run.root / 'rig.claim', nonce=nonce)
    if point == 'child_alive':
        run.h.reached('modem_ready', quiescent=True)
    if isinstance(ending, str):
        os.kill(shell_pid, getattr(signal, ending))
    run.h.release('wrapper_point')
    run.h.release('modem_ready')
    if point == 'cleanup':
        expected = ending if isinstance(ending, int) else 0
    elif point == 'trap_install':
        expected = -getattr(signal, ending) if isinstance(ending, str) else ending
    elif point == 'acquisition':
        expected = 143 if ending == 'SIGTERM' else 129 if ending == 'SIGHUP' else ending or 1
    else:
        expected = 143 if ending == 'SIGTERM' else 129 if ending == 'SIGHUP' else ending
    if killed:
        expected = 0
    output = run.finish(actor, expected)
    proven = point not in ('trap_install', 'acquisition')
    evidence = run.released(line=run.sense(), require_line=proven)
    assert ('CLEANUP COMPLETE' in output) == proven
    assert sum(e['event'] == 'cleanup_acquired' for e in evidence.events) == int(proven)



def _console_wrapper_source(run, point, ending):
    stage = dict(trap_install='unstarted', acquisition='preparation', child_spawn='spawn',
                 child_alive='running', cleanup='readback')[point]
    source = _console_source(run, stage)
    if point == 'trap_install':
        source = source.replace('try:\n    arm._fly', "coexist.checkpoint('console_point')\ntry:\n    arm._fly")
    elif point == 'child_alive':
        source = source.replace('plan = dict(directory=', """
original_pump = arm.pump
def pump(*args):
    coexist.checkpoint('console_point')
    return original_pump(*args)
arm.pump = pump
plan = dict(directory=""")
    if isinstance(ending, int):
        source = f"""
original_checkpoint = coexist.checkpoint
def checkpoint(point):
    original_checkpoint(point)
    if point == 'console_point':
        raise SystemExit({ending})
coexist.checkpoint = checkpoint
""" + source
    return source


def _console_wrapper(run, point, ending):
    source = _console_wrapper_source(run, point, ending)
    for barrier in ('console_point', 'modem_ready'):
        run.h.pin(barrier)
    killed = point == 'trap_install' and isinstance(ending, str)
    actor = (run.supervise(source, -getattr(signal, ending)) if killed else run.python(source))
    if point == 'cleanup':
        child = run.h.reached('modem_ready', quiescent=True)
        os.kill(child['pid'], signal.SIGTERM)
    boundary = run.h.reached('console_point', quiescent=True)
    if point == 'child_alive':
        run.h.reached('modem_ready', quiescent=True)
    if isinstance(ending, str):
        os.kill(boundary['pid'], getattr(signal, ending))
    run.h.release('console_point')
    run.h.release('modem_ready')
    expected = ending if isinstance(ending, int) else 0
    if killed:
        expected = 0
    run.finish(actor, expected)
    evidence = run.released(line=run.sense(), require_line=point != 'trap_install')
    assert sum(e['event'] == 'cleanup_acquired' for e in evidence.events) == int(point != 'trap_install')


def _nested_cleanup_source(run, case):
    launcher = run.root / 'trap-launcher'
    launcher.write_text(_shell_prelude() + 'claim_rig launcher\nclaim_admit\n'
        + ('trap \'exit 4\' INT\n' if case == 'nonzero_after_complete' else "trap 'exit 0' INT\n")
        + _shell_point('launcher_live', inherited=True) + 'exit 0\n')
    launcher.chmod(0o700)
    modem = HEADER + f"os.execv('/bin/bash', ['/bin/bash', {str(launcher)!r}])\n"
    source = _console_source(run, 'nested', modem=modem)
    hooks = ''
    if case in ('returned_none', 'kill_due'):
        hooks += '''
original_wait_gone = console_claim.NativeChild.wait_gone
def wait_gone(child, timeout):
    if console_claim.cleanup_tree(child.pid):
        return original_wait_gone(child, 0)
    return original_wait_gone(child, timeout)
console_claim.NativeChild.wait_gone = wait_gone
def unsettled(*args):
    coexist.checkpoint('settle_entered')
    return None
arm.settle = unsettled
original_down_once = arm._down_once
def down_once(*args):
    result = original_down_once(*args)
    if result[0] == 2:
        coexist.checkpoint('overlap_refused')
    return result
arm._down_once = down_once
'''
    if case == 'kill_due':
        hooks += '''
original_kill = console_claim.NativeChild.kill
def kill(child):
    assert not console_claim.cleanup_tree(child.pid), 'SIGKILL of a cleanup holder'
    return original_kill(child)
console_claim.NativeChild.kill = kill
'''
    return source.replace('plan = dict(directory=', hooks + '\nplan = dict(directory=')


@pytest.mark.parametrize('case', ['successful_trap', 'returned_none', 'kill_due', 'nonzero_after_complete'])
def test_nested_cleanup_overlap(proof_run, case):
    run = proof_run
    for point in ('launcher_live', 'during_line_proof', 'console_released',
                  'settle_entered', 'overlap_refused'):
        run.h.pin(point)
    actor = run.python(_nested_cleanup_source(run, case))
    run.h.reached('launcher_live', quiescent=True)
    owner = int((run.root / 'rig.claim').read_text().split()[0])
    os.kill(owner, signal.SIGINT)
    if case in ('returned_none', 'kill_due'):
        run.h.reached('settle_entered', quiescent=True)
    run.h.release('launcher_live')
    proof = run.h.reached('during_line_proof', quiescent=True)
    if case in ('returned_none', 'kill_due'):
        run.h.release('settle_entered')
        run.h.reached('overlap_refused', timeout=25, quiescent=True)
        assert proof['pid'] in run.snapshot('proof_survives_escalation').participants
        run.h.release('during_line_proof')
        run.h.release('overlap_refused')
    else:
        run.h.release('during_line_proof')
    run.h.reached('console_released', timeout=25, quiescent=True)
    run.h.release('console_released')
    run.finish(actor)
    record = json.loads((run.root / 'flight/provenance.json').read_text())
    assert record['cleanup']['complete'] is True and record['cleanup']['returncode'] == 0
    evidence = run.released(line=run.sense(), require_line=True)
    successful_proofs = [e for e in evidence.events if e['event'] == 'cleanup_acquired']
    assert len(successful_proofs) == (1 if case == 'successful_trap' else 2)
    assert sum(e['event'] == 'claim_released' for e in evidence.events) == 1



def _registry_session(run, *, refuse_fly=False):
    run.h.pin('station_ready')
    source = _service_owner().replace("coexist.checkpoint('station_ready')", """
session = station.sessions.start(kind='monitor', lane='test', attended=False,
    directory=Path(os.environ['HFMODEM_STATE_DIR']) / 'session', client='coexist-holder')
station.sessions.running(session.id)
Path(os.environ['HFMODEM_STATE_DIR'], 'session.json').write_text(json.dumps(session.record()))
coexist.checkpoint('station_ready')
""")
    owner = run.python(source)
    station = run.h.reached('station_ready', quiescent=True)
    session = json.loads((run.root / 'session.json').read_text())
    nonce = int((run.root / 'rig.claim').read_text().split()[4])
    assert_claim_possession(run.snapshot('session_owns'), pid=station['pid'],
                            claim_path=run.root / 'rig.claim', nonce=nonce)
    if refuse_fly:
        contender = run.python("""
from hfmodem.station.client import Client, Claimed
client = Client(os.environ['HFMODEM_STATION'])
try:
    client.command('fly', {'directory': str(Path(os.environ['HFMODEM_STATE_DIR']) / 'refused')}, wait=True)
except Claimed as exc:
    print(exc)
else:
    raise AssertionError('fly was not refused by the session holder')
""")
        output = run.finish(contender)
        assert session['id'] in output and 'monitor' in output and 'coexist-holder' in output
        assert not (run.root / 'refused').exists()
        assert_claim_possession(run.snapshot('session_still_owns'), pid=station['pid'],
                                claim_path=run.root / 'rig.claim', nonce=nonce)
    assert 'CLEANUP COMPLETE' in run.finish(run.shell())
    state = run.finish(run.python("""
import json
from hfmodem.station.client import Client
print(json.dumps(Client(os.environ['HFMODEM_STATION']).state()))
"""))
    state = json.loads(state.strip().splitlines()[-1])
    assert state['owner'] is None
    ended = next(s for s in state['sessions'] if s['id'] == session['id'])
    assert ended['state'] == 'ended' and ended['hand_stopped'] is True
    assert ended['outcome'] == 'stopped' and ended['ended_utc']
    assert_released(run.snapshot('session_drained'), paths=run.h.paths,
                    claim_path=run.root / 'rig.claim', line=run.sense(), require_line=True)
    run.h.release('station_ready')
    run.finish(owner)
    run.released(line=run.sense(), require_line=True)


def _recorder_source(run, kind, configured):
    tap = run.root / 'tap-audio'
    tap.touch()
    recorder = run.root / 'recorder.py'
    endpoint = str(run.h.audio if configured else tap)
    if kind == 'tap':
        recorder.write_text(HEADER + f'''
from hfcapture import device_claim, device_opened
os.environ['HFMODEM_COEXIST_AUDIO'] = {endpoint!r}
import sounddevice as sd
with device_claim('txwitness', devices=claim.Devices(capture='fake audio')):
    with sd.InputStream() as stream:
        device_opened()
        coexist.checkpoint('recorder_ready')
        signal.pause()
''')
    else:
        recorder.write_text(HEADER + f'''
fd = os.open({endpoint!r}, os.O_RDONLY)
print('WITNESS PLACED', flush=True)
coexist.checkpoint('recorder_ready')
signal.pause()
os.close(fd)
''')
    (run.root / 'witness-candidates').touch()
    source = _console_source(run, 'recording')
    prefix = '''
from consolelib import station as recorder_station
real_start_tap, real_place_kiwi = recorder_station.start_tap, recorder_station.place_kiwi
'''
    setup = f'''
from consolelib import operator
arm.station.VENV = Path(sys.executable)
arm.station.TXWITNESS = Path({str(recorder)!r})
arm.station.KIWI = Path({str(recorder)!r})
operator.witness_inventory = lambda: root / 'witness-candidates'
selection.RIG_NAME = 'ft891'
os.environ['HFMODEM_TAP_DEVICE'] = 'fake audio'
arm.station.start_tap = real_start_tap if {kind!r} == 'tap' else lambda *a: None
arm.station.place_kiwi = real_place_kiwi
'''
    source = source.replace('plan = dict(directory=', setup + '\nplan = dict(directory=')
    source = source.replace('try:\n    arm._fly', f"""
plan['dial_hz'] = 7100000
plan['adjustments'].update(witness_seconds=60, kiwi={'fake-receiver' if kind == 'kiwi' else False!r})
try:
    arm._fly""")
    return prefix + source, endpoint


@pytest.mark.parametrize('kind,configured', [
    ('tap', False), ('tap', True), ('kiwi', False), ('station_session', False),
])
def test_tap_and_witness_recorders(proof_run, kind, configured):
    run = proof_run
    if kind == 'station_session':
        _registry_session(run)
        return
    for point in ('recorder_ready', 'modem_ready', 'console_released'):
        run.h.pin(point)
    source, endpoint = _recorder_source(run, kind, configured)
    if kind == 'tap':
        run.h.paths.add(endpoint)
    actor = run.python(source)
    recorder = run.h.reached('recorder_ready', quiescent=True)
    modem = run.h.reached('modem_ready', timeout=10, quiescent=True)
    snapshot = run.snapshot('recording')
    assert any(fd.get('path') == endpoint for fd in snapshot.descriptors[recorder['pid']])
    if kind == 'tap':
        nonce = int((run.root / 'rig.claim').read_text().split()[4])
        assert_claim_possession(snapshot, pid=recorder['pid'], claim_path=run.root / 'rig.claim', nonce=nonce)
    else:
        assert not any(fd.get('path') == str(run.root / 'rig.claim')
                       for fd in snapshot.descriptors[recorder['pid']])
    os.kill(modem['pid'], signal.SIGTERM)
    run.h.reached('console_released', timeout=20, quiescent=True)
    assert recorder['pid'] not in run.snapshot('recorder_ended').participants
    run.h.release('console_released')
    run.finish(actor)
    run.released(line=run.sense(), require_line=True)



def _cleanup_boundaries():
    checkpoint = shlex.quote(sys.executable) + ' -c ' + shlex.quote(
        HEADER + "claim.verify_cleanup(8)\ncoexist.checkpoint(sys.argv[1])")
    return f"""
original_processes=$(declare -f claim_processes)
eval "${{original_processes/claim_processes/cleanup_processes}}"
claim_processes() {{
  cleanup_processes "$@" || return
  case "$1" in
    down) {checkpoint} after_cleanup_discovery ;;
    evidence) {checkpoint} after_cleanup_evidence ;;
  esac
}}
original_proof=$(declare -f prove_down)
eval "${{original_proof/prove_down/cleanup_proof}}"
prove_down() {{
  {checkpoint} before_cleanup_proof || return
  cleanup_proof || return
  {checkpoint} after_cleanup_proof
}}
original_native=$(declare -f claim_native)
eval "${{original_native/claim_native/cleanup_native}}"
claim_native() {{
  cleanup_native "$@" || return
  if [ "$1" = rigctl ]; then {checkpoint} after_cleanup_cat; fi
}}
"""
