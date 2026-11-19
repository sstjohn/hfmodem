# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keying waits keep their monotonic bounds through wall-clock steps.

CPython built without sem_clockwait parks a timed wait on the wall clock: a
forward step ends it at once, a backward step stretches it.
"""
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import hfmodem.besra.radio as R
from hfmodem.besra.station_runtime import StationArdopModem

PARK = threading.Condition.wait


@pytest.fixture
def forward_step(monkeypatch):
    """Every timed Condition wait, and so every Event and Future wait, ends within 5 ms."""
    monkeypatch.setattr(threading.Condition, 'wait',
                        lambda self, timeout=None: PARK(self, None if timeout is None else min(timeout, 0.005)))


@pytest.fixture
def backward_step(monkeypatch):
    monkeypatch.setattr(threading.Condition, 'wait', lambda self, timeout=None: PARK(self))


class Rig:
    settle = 0.

    def __init__(self):
        self.unkeyed = []

    def ptt(self, state):
        return True

    def unkey(self, why):
        self.unkeyed.append((time.monotonic(), why))


def _link(rig, max_key_s):
    link = R.RadioLink.__new__(R.RadioLink)
    link.rig, link.max_key_s, link.drive, link.out_device = rig, max_key_s, 0.5, None
    link.recording, link.muted, link.bursts_sent, link.burst_level = None, False, 0, None
    link._rx = SimpleNamespace(key_down=lambda: None)
    return link


@pytest.fixture
def playing(monkeypatch):
    monkeypatch.setattr(R, 'warm_output', lambda device: None)
    monkeypatch.setattr(R, 'play_drained', lambda *args, **kwargs: time.sleep(0.3))


def test_a_forward_step_does_not_fire_the_tx_watchdog(forward_step, playing):
    rig = Rig()
    _link(rig, 10.)._transmit(np.zeros(1200, '<i2'))
    time.sleep(0.05)
    assert rig.unkeyed == []


def test_the_tx_watchdog_fires_on_monotonic_time_through_a_backward_step(backward_step, playing):
    rig = Rig()
    start = time.monotonic()
    _link(rig, 0.1)._transmit(np.zeros(1200, '<i2'))
    assert len(rig.unkeyed) == 1, "a backward step held the transmitter keyed"
    assert 0.1 <= rig.unkeyed[0][0] - start < 0.25


def test_a_forward_step_does_not_fire_the_tune_watchdog(forward_step, playing):
    rig = Rig()
    R.tune_atu(rig, seconds=0.01, max_key_s=8.)
    time.sleep(0.05)
    assert rig.unkeyed == []


def test_the_tune_watchdog_fires_on_monotonic_time_through_a_backward_step(backward_step, playing):
    rig = Rig()
    start = time.monotonic()
    R.tune_atu(rig, seconds=0.01, max_key_s=0.1)
    assert len(rig.unkeyed) == 1, "a backward step held the transmitter keyed"
    assert 0.1 <= rig.unkeyed[0][0] - start < 0.25


def test_a_forward_step_does_not_time_out_an_owner_command(forward_step):
    modem = StationArdopModem.__new__(StationArdopModem)
    modem._stop, modem._commands = threading.Event(), queue.Queue()
    modem._thread = threading.current_thread()

    def owner():
        future, name, args = modem._commands.get()
        time.sleep(0.1)
        future.set_result(name)
    threading.Thread(target=owner, daemon=True).start()
    assert modem._call('listen') == 'listen'
