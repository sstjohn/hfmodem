# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keying waits keep their monotonic bounds through wall-clock steps.

CPython built without sem_clockwait parks a timed wait on the wall clock: a
forward step ends it at once, a backward step stretches it. The fixtures model
both on every timed Event wait in the process.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core.waits import Timer
from hfmodem.shrike import onair, ota
from hfmodem.shrike.station_runtime import StationPtcRuntime
from hfmodem.tests.clock_steps import step_forward


@pytest.fixture
def forward_step(monkeypatch):
    step_forward(monkeypatch, joins=True)


def _live():
    live = onair._LiveInput.__new__(onair._LiveInput)
    live._duplex = True
    live.fs = onair.FS
    live.samples = 0
    live._blk = 128
    live.geometry_n = live._lat = live.tx_latency_n = 0
    live._dac_time = lambda index: time.monotonic()
    live._tx_done = threading.Event()
    live._armed = None
    return live


def _armed(keyed):
    live = _live()
    b = live._queue(np.zeros(8, np.float32), at=1024, settle=0., key=keyed.append, max_key=10)
    return live, b


def _settle_until(pred, seconds):
    end = time.monotonic() + seconds
    while not pred() and time.monotonic() < end:
        time.sleep(0.005)
    return pred()


def test_a_forward_step_does_not_fire_the_burst_watchdog(forward_step):
    keyed = []
    live, b = _armed(keyed)
    time.sleep(0.3)
    assert keyed == [] and live._watchdog_at is None
    b.done.set()


def test_the_burst_watchdog_fires_on_monotonic_time_through_a_backward_step(backward_step):
    keyed = []
    start = time.monotonic()
    live, b = _armed(keyed)
    assert _settle_until(lambda: keyed, 5), "a backward step held the transmitter keyed"
    assert keyed == [False]
    assert 2.0 <= live._watchdog_at - start < 3.0


@pytest.mark.parametrize('withdraw', ['abort', 'disarm'])
def test_a_forward_step_does_not_cut_short_the_wait_for_the_keyer(forward_step, withdraw):
    keyed = []
    live, b = _armed(keyed)
    Timer(0.1, b.done.set).start()
    getattr(live, withdraw)()
    assert b.done.is_set(), f"{withdraw} gave up on the keyer at once"
    assert keyed == []


def test_a_forward_step_does_not_fire_the_ota_watchdog(forward_step, monkeypatch):
    stopped = []
    rig = SimpleNamespace(ptt=lambda state: True, stop=lambda: stopped.append(True))
    monkeypatch.setattr(ota, '_warm_output', lambda device: None)
    monkeypatch.setattr(ota, 'play_drained', lambda *args: time.sleep(0.3))
    ota._play(np.zeros(8, np.float32), 'fake-output', 10, rig, settle=0)
    time.sleep(0.05)
    assert stopped == []


def test_a_forward_step_does_not_report_the_ptc_owner_unretired(forward_step):
    runtime = StationPtcRuntime.__new__(StationPtcRuntime)
    runtime._stop, runtime._serve = threading.Event(), threading.Event()
    runtime._port_lock, runtime._port = threading.Lock(), None
    runtime._thread = threading.Thread(target=time.sleep, args=(0.1,))
    runtime._thread.start()
    runtime.stop()
    assert not runtime._thread.is_alive()

