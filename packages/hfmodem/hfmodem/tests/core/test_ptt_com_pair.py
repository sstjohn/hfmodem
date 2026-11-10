# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""com0com evidence only; real adapter process-death behaviour remains unproven.

HFMODEM_COM_PAIR names product,far with RTS->CTS and DTR->DSR crossed. No rig
may be attached: these tests deliberately assert virtual output lines.
"""
import os
import queue
import subprocess
import sys
import threading
import time

import pytest
import serial

from hfmodem.core import serialline
from hfmodem.core.ptt import RtsPtt, drop_rts
from hfmodem.tests.core.test_com_pair import follows

pytestmark = [pytest.mark.hardware(opt_in='HFMODEM_COM_PAIR'),
              pytest.mark.skipif(os.name != 'nt', reason='com0com pair requires Windows')]


@pytest.fixture
def pair():
    ports = [port.strip() for port in os.environ['HFMODEM_COM_PAIR'].split(',')]
    assert len(ports) == 2 and all(ports), 'HFMODEM_COM_PAIR must be product,far (COM41,COM42)'
    assert not serialline.same_port(*ports)
    return ports


@pytest.fixture
def peer(pair):
    ser = serialline.open_port(serial.Serial, pair[1], baudrate=9600,
                              rts=False, dtr=False, write_timeout=.25)
    try:
        yield ser
    finally:
        ser.close()


def test_open_keeps_far_cts_and_dsr_low(pair, peer):
    """Low once `open()` has returned; a rise inside the open is the driver's.

    CreateFile precedes pyserial's first SetCommState. com0com raises the lines
    for tens of microseconds in that window on some opens, which is reported as
    an expected failure with its timing. A real adapter's open is the
    rig-attached measurement: this window is exactly what it must show is empty."""
    samples, errors = [], []
    stop, started = threading.Event(), threading.Event()
    def sample():
        try:
            while not stop.is_set():
                samples.append((time.perf_counter(), peer.cts, peer.dsr))
                started.set()
                time.sleep(0)
        except Exception as exc:
            errors.append(exc)
            started.set()
    sampler = threading.Thread(target=sample, daemon=True)
    line = RtsPtt(pair[0])
    sampler.start()
    try:
        assert started.wait(1) and not errors
        before = len(samples)
        opening = time.perf_counter()
        line.open()
        opened = time.perf_counter()
        time.sleep(.02)
        stop.set()
        sampler.join(1)
        assert not sampler.is_alive() and not errors
        assert len(samples) > before > 0
        high = [(round((at - opening) * 1e3, 3), cts, dsr)
                for at, cts, dsr in samples if cts or dsr]
        print(f'open: {len(samples)} CTS/DSR samples; returned after '
              f'{(opened - opening) * 1e3:.3f} ms; high (ms from open): {high}')
        assert all(at < (opened - opening) * 1e3 for at, _, _ in high), high
    finally:
        stop.set()
        sampler.join(1)
        line.release()
    if high:
        pytest.xfail(f'the driver raised the lines during the open (ms, CTS, DSR): {high}')


def test_assert_edges_reach_far_cts_within_50ms(pair, peer):
    line = RtsPtt(pair[0])
    try:
        line.open()
        follows(peer, 'cts', False)
        for on in [True, False] * 5:
            start = time.perf_counter()
            line.assert_(on)
            remaining = .05 - (time.perf_counter() - start)
            assert remaining > 0, 'RTS assignment exceeded 50 ms'
            follows(peer, 'cts', on, timeout=remaining)
            latency = time.perf_counter() - start
            print(f'RtsPtt RTS->CTS {int(on)}: {latency * 1000:.3f} ms')
            assert latency <= .05
    finally:
        line.release()


def test_output_readback_matches_far_inputs_on_every_edge(pair, peer):
    rts, dtr = RtsPtt(pair[0]), RtsPtt(pair[0], line='dtr')
    try:
        rts.open()
        dtr._serial = rts._serial
        dtr._commanded = False
        state = 0
        for line, bit, far in [(rts, 2, 'cts'), (dtr, 1, 'dsr')]:
            for on in (True, False, True):
                line.assert_(on)
                state = state | bit if on else state & ~bit
                follows(peer, far, on)
                assert serialline.read_outputs(rts._serial) == state
                assert rts.sense() is bool(state & 2) is peer.cts
                assert dtr.sense() is bool(state & 1) is peer.dsr
        for line, bit, far in [(rts, 2, 'cts'), (dtr, 1, 'dsr')]:
            line.assert_(False)
            state &= ~bit
            follows(peer, far, False)
            assert serialline.read_outputs(rts._serial) == state
            assert rts.sense() is bool(state & 2) is peer.cts
            assert dtr.sense() is bool(state & 1) is peer.dsr
    finally:
        dtr._serial = None
        rts.release()
    assert rts.released_low is True


def test_release_and_drop_leave_far_cts_low(pair, peer):
    line = RtsPtt(pair[0])
    try:
        line.open()
        line.assert_(True)
        follows(peer, 'cts', True)
    finally:
        line.release()
    follows(peer, 'cts', False)
    follows(peer, 'dsr', False)
    assert line.released_low is True
    logs = []
    assert drop_rts(pair[0], logs.append) is True
    assert len(logs) == 1 and 'taken down directly' in logs[0], logs
    follows(peer, 'cts', False)
    follows(peer, 'dsr', False)
    print('\n'.join(logs))


def ready(proc):
    output = queue.Queue()
    reader = threading.Thread(target=lambda: output.put(proc.stdout.readline()), daemon=True)
    reader.start()
    assert output.get(timeout=3).strip() == 'READY', 'child did not become ready'
    reader.join(1)


_KEY_CHILD = '''
import sys, time
from hfmodem.core.ptt import RtsPtt
line = RtsPtt(sys.argv[1])
line.open()
line.assert_(True)
print('READY', flush=True)
while True:
    time.sleep(60)
'''


def test_terminateprocess_handle_close_lowers_com0com_cts(pair, peer):
    proc = subprocess.Popen([sys.executable, '-c', _KEY_CHILD, pair[0]],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ready(proc)
        follows(peer, 'cts', True)
        proc.kill()
        proc.wait(timeout=3)
        start = time.perf_counter()
        samples = []
        while True:
            high = peer.cts
            elapsed = time.perf_counter() - start
            samples.append((elapsed, high))
            if not high or elapsed >= .5:
                break
            time.sleep(.001)
        print(f'com0com TerminateProcess: CTS={"HIGH" if high else "LOW"} '
              f'at {elapsed * 1000:.3f} ms; samples={samples}; '
              'real CP210x/FTDI acceptance still needs a second adapter')
        assert high is False, 'com0com kept CTS high after process handle close'
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        proc.stdout.close()


_DEADMAN_CHILD = '''
import time
from hfmodem.core.rig import arm_deadman
arm_deadman(.5, message=b'windows deadman fired\\n')
print('READY', flush=True)
while True:
    time.sleep(60)
'''


def test_watchdog_really_ends_a_sleeping_child():
    proc = subprocess.Popen([sys.executable, '-c', _DEADMAN_CHILD],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ready(proc)
        start = time.monotonic()
        assert proc.wait(timeout=3) == 1
        elapsed = time.monotonic() - start
        evidence = proc.stdout.read()
        print(f'Windows deadman exit={proc.returncode} after {elapsed:.3f} s: {evidence.strip()}')
        assert elapsed < 3
        assert 'windows deadman fired' in evidence
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=3)
        proc.stdout.close()


def test_readback_waits_for_an_inflight_drive_with_far_cts(pair, peer, monkeypatch):
    drive, read_outputs = serialline.drive, serialline.read_outputs
    pending = {}

    def inflight_drive(ser, output, on):
        before = read_outputs(ser)
        drive(ser, output, on)
        # Each output keeps its old state until its own drive completes.
        pending[output] = (before, time.monotonic() + .02)

    def inflight_read_outputs(ser, wait_s=.25):
        state = read_outputs(ser, wait_s)
        if state is None:
            return None
        for output, bit in [('rts', 2), ('dtr', 1)]:
            before, until = pending.get(output, (None, 0))
            if time.monotonic() < until and before is not None:
                state = state & ~bit | before & bit
        return state

    monkeypatch.setattr(serialline, 'drive', inflight_drive)
    monkeypatch.setattr(serialline, 'read_outputs', inflight_read_outputs)
    line = RtsPtt(pair[0])
    try:
        line.open()
        for on in (True, False, True):
            line.assert_(on)
            assert line.sense() is on
            follows(peer, 'cts', on)
    finally:
        line.release()
    assert line.released_low is True
    follows(peer, 'cts', False)
    follows(peer, 'dsr', False)
