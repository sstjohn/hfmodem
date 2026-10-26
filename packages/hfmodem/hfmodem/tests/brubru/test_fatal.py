# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import os
import selectors
import subprocess
import sys
import time

import pytest


SCRIPT = r'''
import os
import sys
import threading
import time
from types import SimpleNamespace
from hfmodem.station import process
from hfmodem.core.rig import StuckTransmitter
from hfmodem.core.operations import OperationGate, LeaseTimeout

mode, entry = sys.argv[1:]
st = process.Station(SimpleNamespace(protocols={}))
st._install_signals()
original = process.arm_deadman
def arm(*args, **kwargs):
    assert st._fatal_entered
    os.write(2, b'marked and armed\n')
    original(*args, **kwargs)
process.arm_deadman = arm

class Rig:
    _panic_confirmed = None
    def panic_unkey(self, why):
        os.write(2, b'panic attempted\n')
        if mode == 'blocked_panic':
            while True:
                time.sleep(0.1)
        if mode == 'unconfirmed':
            self._panic_confirmed = False
            raise StuckTransmitter('unconfirmed')
        if mode == 'overlap':
            return
        self._panic_confirmed = True
st.rig = Rig()
lock = threading.Lock()
lock.acquire()
class Journal:
    def flush_events(self, timeout):
        assert timeout <= 0.02
        return lock.acquire(timeout=timeout)
st.lanes['rpr'] = Journal()

def enter():
    if mode == 'lease':
        gate = OperationGate()
        with gate.lease():
            try:
                gate.close(0.01)
            except LeaseTimeout:
                st.fatal_shutdown('lease did not drain')
    st.fatal_shutdown('test')
os.write(2, b'ready\n')
assert sys.stdin.buffer.read(1) == b'x'
if entry == 'worker':
    threading.Thread(target=enter).start()
    while True:
        st._handle_signals()
        time.sleep(0.01)
else:
    enter()
'''


@pytest.mark.parametrize('entry', ['main', 'worker'])
@pytest.mark.parametrize('mode', ['confirmed', 'unconfirmed', 'overlap', 'lease', 'blocked_panic'])
def test_fatal_order_confirmation_nonzero_exit_and_deadman(mode, entry):
    with subprocess.Popen([sys.executable, '-c', SCRIPT, mode, entry],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=os.environ.copy()) as child:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stderr, selectors.EVENT_READ)
            if not ready.select(timeout=30):
                child.kill()
                child.communicate()
                pytest.fail('fatal test child did not become ready within 30 seconds')
        assert child.stderr.readline() == b'ready\n'
        start = time.monotonic()
        try:
            _, stderr = child.communicate(b'x', timeout=7)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise
        elapsed = time.monotonic() - start
    assert child.returncode != 0
    output = stderr.decode()
    assert output.index('marked and armed') < output.index('panic attempted')
    expected = 'true' if mode in ('confirmed', 'lease') else 'false'
    assert f'confirmed={expected}' in output
    assert elapsed < 6.5
    if mode == 'blocked_panic':
        assert elapsed >= 4.8 and 'budget exhausted' in output


@pytest.mark.parametrize('entry', ['main', 'worker'])
def test_deadman_budget_excludes_child_initialization(monkeypatch, entry):
    delayed = SCRIPT.replace('mode, entry = sys.argv[1:]',
                             'time.sleep(1.5)\nmode, entry = sys.argv[1:]')
    monkeypatch.setitem(globals(), 'SCRIPT', delayed)
    test_fatal_order_confirmation_nonzero_exit_and_deadman('blocked_panic', entry)


def test_worker_arms_kernel_timer_before_signal_handoff(monkeypatch):
    import signal
    import threading
    from types import SimpleNamespace
    from hfmodem.station import process

    calls = []
    class Handoff(BaseException):
        pass
    def kill(pid, signum):
        calls.append(('kill', signum))
        raise Handoff
    monkeypatch.setattr(process.signal, 'setitimer', lambda *args: calls.append(('timer', *args)))
    monkeypatch.setattr(process.os, 'kill', kill)
    st = process.Station(SimpleNamespace(protocols={}))
    def worker():
        try:
            st.fatal_shutdown('worker failure')
        except Handoff:
            pass
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(1)
    assert not thread.is_alive()
    assert calls == [('timer', signal.ITIMER_REAL, 5.0), ('kill', signal.SIGTERM)]
    assert st._fatal_reason == 'worker failure'


@pytest.mark.parametrize('blocking', [False, True])
@pytest.mark.parametrize('write_fails', [False, True])
def test_fatal_write_restores_shared_stderr_mode(monkeypatch, blocking, write_fails):
    from hfmodem.core import rig

    modes = []
    monkeypatch.setattr(rig.os, 'get_blocking', lambda fd: blocking)
    monkeypatch.setattr(rig.os, 'set_blocking', lambda fd, value: modes.append(value))
    def write(fd, message):
        assert fd == 2 and message == b'fatal\n'
        if write_fails:
            raise BlockingIOError
    monkeypatch.setattr(rig.os, 'write', write)
    rig.fatal_write(b'fatal\n')
    assert modes == [False, blocking]


def test_worker_deadman_exits_without_main_thread_handoff():
    import signal

    script = r'''
import signal
import threading
from types import SimpleNamespace
from hfmodem.station import process
st = process.Station(SimpleNamespace(protocols={}))
st._install_signals()
# Simulate a main-thread handoff that never takes place.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
original = signal.setitimer
def timer(which, seconds):
    assert threading.current_thread() is not threading.main_thread()
    assert seconds == 5.0
    return original(which, 0.1)
signal.setitimer = timer
threading.Thread(target=lambda: st.fatal_shutdown('worker')).start()
while True:
    signal.pause()
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, timeout=2)
    assert result.returncode == -signal.SIGALRM
