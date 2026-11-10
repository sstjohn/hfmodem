# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Opt-in com0com evidence for termination without interpreter execution."""
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from hfmodem.tests.core.test_ptt_com_pair import pair, peer  # noqa: F401 -- fixtures

pytestmark = [pytest.mark.hardware(opt_in='HFMODEM_COM_PAIR'),
              pytest.mark.skipif(os.name != 'nt', reason='com0com pair requires Windows'),
              pytest.mark.wall_clock]


_TICK = .016


_CHILD = r'''
import os, re, sys, time
from hfmodem.core import posix, rig
from hfmodem.core.ptt import RtsPtt
assert posix._windows_deadman.native
class Finalizer:
    def __init__(self, line):
        self.line = line
    def __del__(self, clock=time.perf_counter, write=os.write,
                finalizing=sys.is_finalizing):
        write(1, f'FINALIZING {finalizing()}\n'.encode())
        end = clock() + 5
        while clock() < end:
            pass
line = RtsPtt(sys.argv[1])
line.open()
line.assert_(True)
mode = sys.argv[2]
if mode == 'finalization':
    blocker = Finalizer(line)
armed = time.perf_counter_ns()
rig.arm_deadman(.05, message=b'PYTHON DEADMAN\n')
if mode == 'cancel':
    posix.cancel_itimer()
print('READY', armed, flush=True)
if mode == 'gil':
    re.match(r'(a+)+$', 'a' * 30 + 'b')
elif mode == 'finalization':
    sys.exit(0)
elif mode == 'cancel':
    time.sleep(.7)
    print('DORMANT', time.perf_counter_ns(), flush=True)
else:
    time.sleep(5)
'''


@pytest.mark.parametrize('mode', ['gil', 'finalization', 'sleep'])
def test_backstop_exit_and_handle_close(pair, peer, mode):  # noqa: F811
    proc = subprocess.Popen([sys.executable, '-c', _CHILD, pair[0], mode],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = queue.Queue()
    def read():
        for line in proc.stdout:
            output.put(line.strip())
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    lines = []
    try:
        ready = output.get(timeout=5).split()
        assert len(ready) == 2 and ready[0] == 'READY', ready
        start = int(ready[1])
        assert peer.cts, 'CTS was not high after READY'
        exited = lowered = None
        deadline = time.perf_counter() + 1.5
        while time.perf_counter() < deadline and (exited is None or lowered is None):
            if not peer.cts and lowered is None:
                lowered = (time.perf_counter_ns() - start) / 1e9
            if proc.poll() is not None and exited is None:
                exited = (time.perf_counter_ns() - start) / 1e9
            time.sleep(.001)
        assert exited is not None and lowered is not None, (exited, lowered)
        reader.join(1)
        assert not reader.is_alive()
        while not output.empty():
            lines.append(output.get_nowait())
        print(f'{mode}: exit={proc.returncode} READY->exit={exited:.6f} s '
              f'READY->CTS low={lowered:.6f} s; output={lines}')
        assert proc.returncode == 1
        if mode == 'sleep':
            assert 0.05 <= exited < .25
            assert 0.05 <= lowered < .25
            assert 'PYTHON DEADMAN' in lines
        else:
            # Windows timer expiry rounds to the clock tick and was seen 0.6 ms short of due.
            assert .30 - _TICK <= exited <= 1.02
            assert .30 - _TICK <= lowered <= 1.02
            assert 'PYTHON DEADMAN' not in lines
            if mode == 'finalization':
                assert 'FINALIZING True' in lines
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        reader.join(1)
        proc.stdout.close()


def test_cancelled_backstop_is_dormant(pair, peer):  # noqa: F811
    proc = subprocess.Popen([sys.executable, '-c', _CHILD, pair[0], 'cancel'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = proc.communicate(timeout=2)
        assert proc.returncode == 0, output
        stamps = dict(line.split() for line in output.splitlines())
        assert (int(stamps['DORMANT']) - int(stamps['READY'])) / 1e9 > .6
        assert 'PYTHON DEADMAN' not in output
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=.5)
