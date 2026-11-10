# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""com0com's default wiring crosses RTS to CTS and DTR to DSR/DCD."""
import os
import socket
import subprocess
import threading
import time

import pytest

from hfmodem.core import hamlib, serialline
from hfmodem.core.cat import RigctldCat, YaesuCat
from hfmodem.core.rigs import RIGS
from hfmodem.tests.core.fakeyaesu import SerialFakeYaesu

pytestmark = [pytest.mark.hardware(opt_in='HFMODEM_COM_PAIR'),
              pytest.mark.skipif(os.name != 'nt', reason='com0com pair requires Windows')]


@pytest.fixture
def pair():
    ports = [port.strip() for port in os.environ['HFMODEM_COM_PAIR'].split(',')]
    assert len(ports) == 2 and all(ports), 'HFMODEM_COM_PAIR must be product,far (COM41,COM42)'
    assert not serialline.same_port(*ports), 'the pair must name two different ports'
    return ports


@pytest.fixture
def peer(pair):
    fake = SerialFakeYaesu(pair[1])
    try:
        yield fake
    finally:
        fake.close()
        assert fake.error is None, f'serial fake failed: {fake.error}'
        assert not {'TX1;', 'TX2;'}.intersection(fake.log), fake.log


def follows(ser, pin, expected, *, timeout=.05):
    started = time.perf_counter()
    deadline = started + timeout
    while True:
        value = getattr(ser, pin)
        elapsed = time.perf_counter() - started
        if value == expected:
            assert elapsed <= timeout, f'{pin} transition took {elapsed * 1000:.3f} ms'
            return elapsed
        assert time.perf_counter() < deadline, f'{pin} did not become {expected} within {timeout * 1000:g} ms'
        time.sleep(0)


def test_cat_exchange_reopen_and_exclusive(pair, peer):
    import serial
    driver = YaesuCat(pair[0])
    try:
        driver.open()
        assert driver.identify() == RIGS['ft891']['name']
        assert peer.replies['ID;'] == 'ID0650;' and 'ID;' in peer.log
        assert driver.freq() == 7100000
        assert driver.mode().mode == 'PKTUSB'
        driver.set_freq(7102500)
        assert driver.freq() == 7102500
        with pytest.raises(serial.SerialException, match='is open in another process'):
            other = serialline.open_port(serial.Serial, pair[0], baudrate=38400,
                rts=False, dtr=False, write_timeout=.25)
            other.close()
        driver.close()
        driver.open()
        assert driver.identify() == RIGS['ft891']['name']
        assert driver.freq() == 7102500
    finally:
        driver.close()


def test_cat_open_line_levels(pair, peer):
    """Low across the open, and low once it has returned.

    The second is ours to keep. The first is not wholly: CreateFile precedes
    pyserial's first SetCommState, and com0com raises both lines for some tens of
    microseconds in that window on some opens. That is reported as an expected
    failure with its timing, so it stays visible; a real adapter's open is the
    rig-attached measurement."""
    samples, errors = [], []
    stop, started = threading.Event(), threading.Event()

    def sample():
        try:
            while not stop.is_set():
                samples.append((time.perf_counter(), peer.serial.cts, peer.serial.dsr))
                started.set()
                time.sleep(0)
        except Exception as exc:
            errors.append(exc)
            started.set()

    profile = {**RIGS['ft891'], 'cat_rts': False, 'cat_dtr': False}
    driver = YaesuCat(pair[0], profile)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        assert started.wait(1), 'line sampler did not start'
        assert not errors
        before = len(samples)
        opening = time.perf_counter()
        driver.open()
        opened = time.perf_counter()
        time.sleep(.02)
        stop.set()
        sampler.join(1)
        assert not sampler.is_alive() and not errors
        assert len(samples) > before > 0
        high = [(round((at - opening) * 1e3, 3), cts, dsr)
                for at, cts, dsr in samples if cts or dsr]
        assert all(at < (opened - opening) * 1e3 for at, _, _ in high), high
    finally:
        stop.set()
        sampler.join(1)
        driver.close()
    driver = YaesuCat(pair[0], {**profile, 'cat_rts': True})
    try:
        driver.open()
        follows(peer.serial, 'cts', True)
        assert not peer.serial.dsr
    finally:
        driver.close()
    if high:
        pytest.xfail(f'the driver raised the lines during the open, ms from its start '
                     f'(CTS, DSR): {high}')


def test_serialline_modem_line_latency(pair, peer):
    import serial
    product = serialline.open_port(serial.Serial, pair[0], baudrate=38400,
        rts=False, dtr=False, write_timeout=.25)
    try:
        follows(peer.serial, 'cts', False)
        follows(peer.serial, 'dsr', False)
        for output, input_pin in [('rts', 'cts'), ('dtr', 'dsr')]:
            for value in [True, False] * 3:
                started = time.perf_counter()
                setattr(product, output, value)
                remaining = .05 - (time.perf_counter() - started)
                assert remaining > 0, f'{output} assignment exceeded 50 ms'
                follows(peer.serial, input_pin, value, timeout=remaining)
                latency = time.perf_counter() - started
                print(f'{output.upper()}->{input_pin.upper()} {int(value)}: {latency * 1000:.3f} ms')
                assert latency <= .05
    finally:
        product.close()


def test_rigctld_over_pair(pair, peer, tmp_path):
    try:
        hamlib.rigctld_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    with (tmp_path / 'rigctld.log').open('w+') as log:
        proc = hamlib.spawn_rigctld(['-m', '1036', '-r', serialline.device_name(pair[0]),
            '-s', '38400', '-t', str(port), '-T', '127.0.0.1', '--set-conf=ptt_type=None'],
            stdout=log, stderr=subprocess.STDOUT)
        driver = RigctldCat(port=port, profile='ft891')
        try:
            deadline = time.monotonic() + 10
            while True:
                log.flush()
                log.seek(0)
                evidence = log.read()
                assert proc.poll() is None, f'rigctld exited: {evidence}; CAT: {peer.log}'
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    assert time.monotonic() < deadline, f'rigctld did not listen: {evidence}; CAT: {peer.log}'
                    time.sleep(.02)
            driver.open()
            assert driver.freq() == int(peer.replies['FA;'][2:-1])
        finally:
            try:
                driver.close()
            finally:
                hamlib.stop_rigctld(proc, 2)
