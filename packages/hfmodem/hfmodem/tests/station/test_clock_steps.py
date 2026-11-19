# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Keying-path waits keep their monotonic deadlines through wall-clock steps."""
import threading
import time
from types import SimpleNamespace as NS

from hfmodem.station import link, preflight
from hfmodem.station.arbiter import TxArbiter
from hfmodem.tests.clock_steps import SteppedLock


class Flight(NS):
    def __getattr__(self, name):
        return None


def test_unkey_containment_outlasts_a_forward_step(forward_step):
    panics, records = [], []
    arbiter = object.__new__(TxArbiter)
    arbiter.rig = NS(panic_unkey=panics.append, fatal=panics.append, line_state='low')
    arbiter.audio = NS(finish_burst=lambda: time.sleep(0.05) or 1.0)
    arbiter._admit, arbiter._halt_locked = threading.Lock(), lambda flight, why: panics.append(why)
    arbiter.records, arbiter._release = NS(put=records.append), lambda flight: None
    flight = Flight(stop=threading.Event(), key_done=threading.Event(), done=threading.Event(),
                    drained=threading.Event(), request=NS(protocol='p', authority='a'))
    arbiter._contain(flight)
    assert panics == [] and flight.error is None and flight.end == 1.0


class Ptt:
    def __init__(self):
        self.keyed, self.edges = lambda: None, []

    def assert_(self, high):
        self.edges.append((high, time.monotonic()))
        if high:
            self.keyed()

    def sense(self):
        return False


def keyer():
    check = object.__new__(preflight.Preflight)
    check.cancelled = lambda: False
    check.cancel_event = threading.Event()
    rig = NS(line_state='low', panics=[])
    rig.panic_unkey = lambda why: rig.panics.append(time.monotonic())
    rig.ptt = Ptt()
    return check, rig


def hold_s(rig):
    (_, up), (_, down) = rig.ptt.edges
    return down - up


def test_preflight_hold_and_deadman_outlast_a_forward_step(forward_step):
    check, rig = keyer()
    check._key(rig)
    assert hold_s(rig) >= preflight.HOLD_S
    assert rig.panics == []


def test_preflight_hold_ends_on_time_through_a_backward_step(backward_step):
    check, rig = keyer()
    worker = threading.Thread(target=check._key, args=(rig,), daemon=True)
    worker.start()
    worker.join(preflight.HOLD_S + 1)
    check.cancel_event.set()
    assert len(rig.ptt.edges) == 2 and preflight.HOLD_S <= hold_s(rig) < preflight.HOLD_S + 0.5
    assert rig.panics == []


def test_preflight_deadman_fires_on_time_through_a_backward_step(backward_step):
    check, rig = keyer()
    bound = preflight.HOLD_S + preflight.DEADMAN_SLACK_S
    rig.ptt.keyed = lambda: time.sleep(bound + 0.5)
    worker = threading.Thread(target=check._key, args=(rig,), daemon=True)
    start = time.monotonic()
    worker.start()
    worker.join(bound + 2)
    assert len(rig.panics) == 1 and bound <= rig.panics[0] - start < bound + 0.3


def test_rpr_journal_admission_outlasts_a_forward_step():
    fatal = []
    rpr = object.__new__(link.RprLink)
    rpr._journal_lock = SteppedLock(threading.Lock())
    rpr.station = NS(fatal_shutdown=fatal.append)
    rpr.events = NS(record=lambda *a, **kw: 'event', lines=lambda: [])
    rpr.outcomes, rpr._journal_path = [], None
    request = NS(identifier='r1', attempts=1, deferrals=0, state=NS(value='done'), completion=None)
    rpr._journal_lock.acquire()
    threading.Timer(0.05, rpr._journal_lock.release).start()
    rpr._record(request, 'sent')
    assert fatal == [] and request.completion == ('sent', '')
