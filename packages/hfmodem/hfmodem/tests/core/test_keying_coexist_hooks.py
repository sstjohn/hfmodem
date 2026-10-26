# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
from contextlib import nullcontext
import hashlib
import json
import marshal
import os
from pathlib import Path
import struct
import sys
import termios
from types import ModuleType, SimpleNamespace as NS

import pytest

from hfmodem.core import coexist, ptt, rig
from hfmodem.core.ptt import PttError
from hfmodem.core.regulatory import Control, Unregulated, centred
from hfmodem.core.rig import RigError

#: sha256 of ptt.py and rig.py without the trailing harness guard, as reviewed at
#: station/stage0 8b8127e68. A change here is a change to the keying path: review
#: it, then record the new digest.
REVIEWED = {
    'ptt': 'be84ae93b3f72e7dcf71505f1bd844b106cc9f786681872d5e461ef5e2ba6982',
    'rig': '3704e5b0a7822df6aab55c81fe14ada0414381b9fde3a4ca0429ccd48ca32404',
}
GUARD = '''
if os.environ.get("HFMODEM_COEXIST_HARNESS") == "1" and os.environ.get("HFMODEM_COEXIST_INSTRUMENT") == "1":
    from .coexist import instrument_{0}
    instrument_{0}({1})
'''
HOOKED = {'ptt': ('RtsPtt', ('open', 'assert_', '_sense_fd', 'release')),
          'rig': ('Rig', ('key', 'unkey'))}
SWITCHES = [(None, '1'), ('0', '1'), ('1', None), ('1', '0'), ('true', '1'), ('1', 'true'), ('1', '1')]
STAMPS = ('pid', 'run', 'monotonic_ns', 'time_ns')


def load(monkeypatch, name, *, baseline=False):
    path = Path(coexist.__file__).with_name(name + '.py')
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    guard = tree.body[-1]
    assert ast.dump(guard) == ast.dump(ast.parse(GUARD.format(name, HOOKED[name][0])).body[0])
    if baseline:
        source = ''.join(source.splitlines(keepends=True)[:guard.lineno - 1]).rstrip('\n') + '\n'
        assert hashlib.sha256(source.encode()).hexdigest() == REVIEWED[name], (
            f'{name}.py differs from the reviewed keying source')
    module = ModuleType('hfmodem.core._hook_' + name)
    module.__package__ = 'hfmodem.core'
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def switch(monkeypatch, harness, instrument):
    for name, value in [('HARNESS', harness), ('INSTRUMENT', instrument)]:
        if value is None:
            monkeypatch.delenv('HFMODEM_COEXIST_' + name, raising=False)
        else:
            monkeypatch.setenv('HFMODEM_COEXIST_' + name, value)


def events(path):
    if not path.exists():
        return []
    return [{k: v for k, v in json.loads(line).items() if k not in STAMPS}
            for line in path.read_text().splitlines()]


@pytest.fixture
def logs(monkeypatch, tmp_path):
    path = tmp_path / 'events'
    monkeypatch.setenv('HFMODEM_COEXIST_LOG', str(path))
    monkeypatch.setenv('HFMODEM_COEXIST_RUN', 'keying-hooks')
    monkeypatch.delenv('HFMODEM_COEXIST_BARRIERS', raising=False)
    monkeypatch.delenv('HFMODEM_COEXIST_FAULT', raising=False)
    return path


@pytest.mark.parametrize('harness,instrument', SWITCHES[:-1])
@pytest.mark.parametrize('module', HOOKED)
def test_disabled_methods_are_the_reviewed_bytecode(monkeypatch, logs, harness, instrument, module):
    switch(monkeypatch, harness, instrument)
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', 'before_RtsPtt_assert_:kill')
    cls, names = HOOKED[module]
    baseline = getattr(load(monkeypatch, module, baseline=True), cls)
    actual = getattr(load(monkeypatch, module), cls)
    for name in names:
        before, after = getattr(baseline, name), getattr(actual, name)
        assert marshal.dumps(before.__code__) == marshal.dumps(after.__code__)
        if hasattr(before, '__wrapped__'):
            assert marshal.dumps(before.__wrapped__.__code__) == marshal.dumps(after.__wrapped__.__code__)
    assert not logs.exists()


@pytest.mark.parametrize('harness,instrument', SWITCHES)
@pytest.mark.parametrize('module', HOOKED)
def test_module_guard_alone(monkeypatch, logs, harness, instrument, module):
    switch(monkeypatch, harness, instrument)
    seen = []
    monkeypatch.setattr(coexist, 'instrument_' + module, seen.append)
    loaded = load(monkeypatch, module)
    assert seen == ([getattr(loaded, HOOKED[module][0])] if (harness, instrument) == ('1', '1') else [])


@pytest.mark.parametrize('harness,instrument', SWITCHES)
@pytest.mark.parametrize('module', HOOKED)
def test_instrument_guard_alone(monkeypatch, logs, harness, instrument, module):
    switch(monkeypatch, harness, instrument)
    cls, names = HOOKED[module]
    sub = type(cls, (getattr(sys.modules['hfmodem.core.' + module], cls),), {})
    getattr(coexist, 'instrument_' + module)(sub)
    assert {name for name in names if name in vars(sub)} == (
        set(names) if (harness, instrument) == ('1', '1') else set())


def line_exercise(module, failure):
    calls = []
    state = 0

    def record(name, value=None):
        calls.append((name, value))
        if failure == name:
            raise OSError('injected ' + name)

    def opened(*args):
        record('open')
        return 41

    def ioctl(fd, op, data):
        nonlocal state
        bits = struct.unpack('I', data)[0]
        name = {termios.TIOCMBIC: 'clear', termios.TIOCMBIS: 'set', termios.TIOCMGET: 'sense'}[op]
        record(name, bits)
        if name == 'clear':
            state &= ~bits
        elif name == 'set':
            state |= bits
        return struct.pack('I', state)

    module.os = NS(open=opened, close=lambda fd: record('close', fd),
                   **{key: getattr(os, key) for key in ('O_RDWR', 'O_NOCTTY', 'O_NONBLOCK', 'O_CLOEXEC')})
    module.fcntl = NS(ioctl=ioctl)
    module.require_char_device = lambda path: record('check', path)
    module.RtsPtt._configure = lambda self, fd: record('configure', fd)
    line = module.RtsPtt('fake-line')
    results = []
    for name, args in [('open', ()), ('sense', ()), ('assert_', (True,)), ('sense', ()),
                       ('assert_', (False,)), ('release', ()), ('sense', ()), ('assert_', (True,))]:
        try:
            results.append((name, getattr(line, name)(*args)))
        except Exception as exc:
            results.append((name, type(exc).__name__, str(exc)))
    return calls, results, line.released_low


def wire(event, verdict=None, **fields):
    fields = dict(fields, event=event, path='fake-line')
    if event in ('device_open', 'key', 'deassert'):
        fields['fd'] = 41
    if verdict is not None:
        fields['verdict'] = verdict
    return fields


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('failure', [None, 'open', 'configure', 'set', 'clear', 'sense', 'close'])
def test_ptt_preserves_fake_io_calls_results_and_errors(monkeypatch, logs, enabled, failure):
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_COEXIST_INSTRUMENT', str(int(enabled)))
    baseline = line_exercise(load(monkeypatch, 'ptt', baseline=True), failure)
    actual = line_exercise(load(monkeypatch, 'ptt'), failure)
    assert actual == baseline
    if not enabled:
        assert not logs.exists()
    elif failure is None:
        assert events(logs) == [
            wire('device_open'), wire('line_readback', 'low'), wire('key'),
            wire('line_readback', 'high'), wire('deassert'), wire('line_readback', 'low'),
            wire('device_close', released_low=True)]
    elif failure == 'sense':
        assert events(logs) == [
            wire('device_open'), wire('line_readback', 'unreadable'), wire('key'),
            wire('line_readback', 'unreadable'), wire('deassert'), wire('line_readback', 'unreadable'),
            wire('device_close', released_low=False)]


def rig_exercise(module, failure):
    calls = []

    def drive(on):
        calls.append(('assert', on))
        if failure == ('key' if on else 'unkey'):
            raise PttError('injected line failure')

    def sense():
        calls.append(('sense',))
        return None if failure == 'unreadable' else False

    cat = NS(hold=lambda deadline=None: nullcontext(True), clock=lambda: 100,
             protect=lambda *a: 'protection', unprotect=lambda token: calls.append(('unprotect', token)))
    rig = module.Rig(model='generic', cat=cat, ptt=NS(assert_=drive, sense=sense),
                     profile=Unregulated(because='fake line'), control=Control.LOCAL,
                     mycall='N0CALL', transmit=True)
    rig._armed = True
    rig.line_state = 'low'
    rig._cache_dial(7100000)
    rig._arm_watchdog = lambda *a: calls.append(('watchdog',))
    rig.panic_unkey = lambda why: calls.append(('panic', why))
    results = []
    for name in ('key', 'unkey', 'key'):
        try:
            value = (rig.key(centred(7100000, 500), why='test', duration_s=1, attended=True)
                     if name == 'key' else rig.unkey())
            results.append((name, value.seq if value else None))
        except Exception as exc:
            results.append((name, type(exc).__name__, str(exc)))
    return calls, results, rig.line_state


def rig_state(state, keyed):
    return {'event': 'rig_line_state', 'state': state, 'keyed': keyed}


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('failure,logged', [
    (None, [('high', True), ('low', False), ('high', True)]),
    ('key', [('low', False)]),
    ('unkey', [('high', True), ('high', True)]),
    ('unreadable', [('high', True), ('unconfirmed', False)]),
])
def test_rig_preserves_line_calls_results_and_errors(monkeypatch, logs, enabled, failure, logged):
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_COEXIST_INSTRUMENT', str(int(enabled)))
    assert rig_exercise(load(monkeypatch, 'rig'), failure) == rig_exercise(
        load(monkeypatch, 'rig', baseline=True), failure)
    assert events(logs) == ([rig_state(*row) for row in logged] if enabled else [])


@pytest.fixture
def station(monkeypatch, logs):
    """The real RtsPtt and Rig, instrumented, over a fake serial line."""
    monkeypatch.setenv('HFMODEM_COEXIST_HARNESS', '1')
    monkeypatch.setenv('HFMODEM_COEXIST_INSTRUMENT', '1')
    wire = NS(bits=0, open=False, calls=[])

    def ioctl(fd, op, data):
        bits = struct.unpack('I', data)[0]
        wire.calls.append(op)
        if op == termios.TIOCMBIS:
            wire.bits |= bits
        elif op == termios.TIOCMBIC:
            wire.bits &= ~bits
        return struct.pack('I', wire.bits)

    def opened(*args):
        wire.open = True
        return 41

    monkeypatch.setattr(ptt, 'os', NS(open=opened, close=lambda fd: setattr(wire, 'open', False),
                                      **{k: getattr(os, k) for k in ('O_RDWR', 'O_NOCTTY', 'O_NONBLOCK', 'O_CLOEXEC')}))
    monkeypatch.setattr(ptt, 'fcntl', NS(ioctl=ioctl))
    monkeypatch.setattr(ptt, 'require_char_device', lambda path: None)
    Line = type('RtsPtt', (ptt.RtsPtt,), {'_configure': lambda self, fd: None})
    Rig = type('Rig', (rig.Rig,), {})
    coexist.instrument_ptt(Line)
    coexist.instrument_rig(Rig)
    rigs = []

    def build(*, opened=True):
        line = Line('fake-line')
        if opened:
            line.open()
        cat = NS(hold=lambda deadline=None: nullcontext(True), clock=lambda: 100,
                 protect=lambda *a: 'protection', unprotect=lambda token: None,
                 emergency_unkey=lambda: False, reads_ptt=False, close=lambda: None)
        made = Rig(model='generic', cat=cat, ptt=line, profile=Unregulated(because='fake line'),
                   control=Control.LOCAL, mycall='N0CALL', transmit=True)
        made._armed = True
        made.line_state = 'low'
        made._cache_dial(7100000)
        rigs.append(made)
        return made

    wire.high = lambda: bool(wire.bits & termios.TIOCM_RTS)
    wire.line, wire.rig = Line, build
    yield wire
    for made in rigs:
        made._cancel_watchdog()


def key(made, open_key):
    if open_key:
        return made.key(centred(7100000, 500), why='test', open=True, attended=True)
    return made.key(centred(7100000, 500), why='test', duration_s=60, attended=True)


def faults(logs, point):
    return [e for e in events(logs) if e['event'] == 'fault' and e['point'] == point]


@pytest.mark.parametrize('when', ['before', 'after'])
def test_open_fault_is_a_refused_open(monkeypatch, logs, station, when):
    point = f'{when}_RtsPtt_open'
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    with pytest.raises(PttError, match=point):
        station.line('fake-line').open()
    assert not station.open and not station.high()
    assert faults(logs, point)


@pytest.mark.parametrize('when', ['before', 'after'])
@pytest.mark.parametrize('on', [True, False])
def test_assert_fault_is_a_refused_ioctl_and_never_holds_the_line_up(monkeypatch, logs, station, when, on):
    point = f'{when}_RtsPtt_assert_'
    line = station.line('fake-line')
    line.open()
    station.bits = termios.TIOCM_RTS if not on else 0
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    with pytest.raises(PttError, match=point):
        line.assert_(on)
    assert station.high() is (on and when == 'after')
    assert faults(logs, point)


@pytest.mark.parametrize('when', ['before', 'after'])
def test_readback_fault_is_a_ptt_error(monkeypatch, logs, station, when):
    point = f'{when}_RtsPtt__sense_fd'
    line = station.line('fake-line')
    line.open()
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    with pytest.raises(PttError, match=point):
        line.sense()


@pytest.mark.parametrize('point', ['before_RtsPtt_release', 'after_RtsPtt_release',
                                   'before_RtsPtt__sense_fd', 'after_RtsPtt__sense_fd'])
def test_release_fault_lowers_closes_and_confirms_nothing(monkeypatch, logs, station, point):
    line = station.line('fake-line')
    line.open()
    line.assert_(True)
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    assert line.release() is None
    assert not station.high() and not station.open
    assert line.released_low is False
    assert faults(logs, point)


@pytest.mark.parametrize('open_key', [False, True])
@pytest.mark.parametrize('when', ['before', 'after'])
def test_line_fault_in_key_is_handled_as_production_does(monkeypatch, logs, station, capfd,
                                                         open_key, when):
    made = station.rig()
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', f'{when}_RtsPtt_assert_')
    with pytest.raises(RigError, match='cannot key'):
        key(made, open_key)
    assert made.retired and '!! unkeying' in capfd.readouterr().err
    assert made.line_state == 'low' and not made.keyed and made._timer is None
    assert not station.high() and not station.open


@pytest.mark.parametrize('open_key', [False, True])
@pytest.mark.parametrize('point', ['before_RtsPtt_assert_', 'after_RtsPtt_assert_',
                                   'before_RtsPtt__sense_fd', 'after_RtsPtt__sense_fd'])
def test_line_fault_in_unkey_still_takes_the_line_down(monkeypatch, logs, station, capfd,
                                                       open_key, point):
    made = station.rig()
    key(made, open_key)
    assert station.high()
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    made.unkey()
    assert not station.high() and not made.keyed and made._timer is None
    assert made.retired and '!! unkeying' in capfd.readouterr().err
    assert faults(logs, point)


@pytest.mark.parametrize('open_key', [False, True])
def test_key_fault_before_keys_nothing(monkeypatch, logs, station, open_key):
    made = station.rig()
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', 'before_Rig_key')
    with pytest.raises(RigError, match='before_Rig_key'):
        key(made, open_key)
    assert not station.high() and not made.keyed and made._timer is None
    assert termios.TIOCMBIS not in station.calls


@pytest.mark.parametrize('open_key', [False, True])
def test_key_fault_after_leaves_the_key_tracked(monkeypatch, logs, station, open_key):
    made = station.rig()
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', 'after_Rig_key')
    with pytest.raises(RigError, match='after_Rig_key'):
        key(made, open_key)
    assert station.high() and made.keyed and made.line_state == 'high'
    assert (made._timer is None) is open_key
    assert events(logs)[-2:] == [rig_state('high', True), {'event': 'fault', 'point': 'after_Rig_key'}]
    monkeypatch.delenv('HFMODEM_COEXIST_FAULT')
    made.close()
    assert not station.high() and not station.open and not made.keyed and made._timer is None


@pytest.mark.parametrize('open_key', [False, True])
@pytest.mark.parametrize('when', ['before', 'after'])
def test_unkey_fault_fires_after_the_line_is_down(monkeypatch, logs, station, open_key, when):
    point = f'{when}_Rig_unkey'
    made = station.rig()
    key(made, open_key)
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    with pytest.raises(RigError, match=point):
        made.unkey()
    assert not station.high() and not made.keyed and made._timer is None
    assert made.line_state == 'low' and not made.retired
    assert [e['event'] for e in events(logs)][-4:] == ['deassert', 'line_readback', 'rig_line_state', 'fault']


@pytest.mark.parametrize('point', ['before_RtsPtt_release', 'after_RtsPtt_release',
                                   'before_RtsPtt__sense_fd', 'after_RtsPtt__sense_fd'])
def test_panic_under_a_fault_still_takes_the_line_down(monkeypatch, logs, station, capfd, point):
    made = station.rig()
    key(made, True)
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', point)
    made.panic_unkey('test')
    assert not station.high() and not station.open and not made.keyed
    assert 'confirmed down' in capfd.readouterr().err


@pytest.mark.parametrize('open_key', [False, True])
def test_unwritable_event_log_never_reaches_the_key_path(monkeypatch, logs, station, tmp_path, open_key):
    barriers = tmp_path / 'barriers'
    barriers.mkdir()
    monkeypatch.setenv('HFMODEM_COEXIST_LOG', str(tmp_path / 'missing' / 'events'))
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    (barriers / 'after_RtsPtt_assert_').touch()
    (barriers / 'after_RtsPtt_assert_.release').touch()
    made = station.rig()
    key(made, open_key)
    assert station.high() and made.keyed and made.line_state == 'high'
    assert (made._timer is None) is open_key
    assert (barriers / 'after_RtsPtt_assert_.reached').exists()
    made.unkey()
    assert not station.high() and not made.keyed and made.line_state == 'low'
    made.ptt.release()
    assert made.ptt.released_low is True and not station.open
    monkeypatch.setenv('HFMODEM_COEXIST_FAULT', 'before_RtsPtt_release')
    other = station.rig()
    key(other, open_key)
    other.ptt.release()
    assert other.ptt.released_low is False and not station.high()
    assert not (tmp_path / 'missing').exists()


@pytest.fixture
def stalled(monkeypatch, tmp_path):
    """Barriers that are pinned and never released, timing out at once."""
    barriers = tmp_path / 'stalled'
    barriers.mkdir()
    monkeypatch.setenv('HFMODEM_COEXIST_BARRIERS', str(barriers))
    monkeypatch.setitem(coexist.barrier.__kwdefaults__, 'timeout', 0.05)
    return lambda point: (barriers / point).touch()


@pytest.mark.parametrize('open_key', [False, True])
def test_barrier_timeout_in_key_is_handled_as_production_does(logs, station, stalled, capfd, open_key):
    stalled('after_RtsPtt_assert_')
    made = station.rig()
    with pytest.raises(RigError, match='cannot key: coexist barrier after_RtsPtt_assert_'):
        key(made, open_key)
    assert made.retired and '!! unkeying' in capfd.readouterr().err
    assert not station.high() and not made.keyed and made._timer is None


@pytest.mark.parametrize('point', ['before_RtsPtt_release', 'after_RtsPtt__sense_fd'])
def test_barrier_timeout_in_release_never_raises(logs, station, stalled, point):
    stalled(point)
    line = station.line('fake-line')
    line.open()
    line.assert_(True)
    assert line.release() is None
    assert not station.high() and not station.open and line.released_low is False


def test_lost_event_leaves_a_mark_and_no_keying_fault(monkeypatch, station, stalled, tmp_path):
    log = tmp_path / 'log-is-a-directory'
    log.mkdir()
    monkeypatch.setenv('HFMODEM_COEXIST_LOG', str(log))
    lost = tmp_path / 'log-is-a-directory.lost'
    made = station.rig()
    key(made, True)
    assert station.high() and made.line_state == 'high' and lost.exists()
    lost.unlink()
    stalled('after_RtsPtt_assert_')
    other = station.rig()
    with pytest.raises(RigError, match='cannot key'):
        key(other, False)
    assert lost.exists() and not station.high()
