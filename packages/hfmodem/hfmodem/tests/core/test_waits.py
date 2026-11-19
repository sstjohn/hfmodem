# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Waits keep their monotonic deadlines through wall-clock steps."""
from concurrent import futures
import threading
import time

import pytest

from hfmodem.core import waits
from hfmodem.core.waits import Timer, sleep_until, wait_until
from hfmodem.tests.clock_steps import step_forward
from hfmodem.tests.timing import within

CONDITION_WAIT = threading.Condition.wait


@pytest.fixture
def stepped_conditions(monkeypatch):
    """Every timed Condition wait, and so every Future wait, ends within 5 ms; counts them."""
    parked = []

    def wait(self, timeout=None):
        parked.append(timeout)
        return CONDITION_WAIT(self, None if timeout is None else min(timeout, 0.005))
    monkeypatch.setattr(threading.Condition, 'wait', wait)
    return parked


def test_a_forward_step_does_not_end_the_wait(forward_step):
    event = threading.Event()
    start = time.monotonic()
    assert not wait_until(event.wait, start + 0.2)
    assert time.monotonic() - start >= 0.2


def test_the_wait_ends_when_its_event_is_set(forward_step):
    event = threading.Event()
    threading.Timer(0.05, event.set).start()
    start = time.monotonic()
    assert wait_until(event.wait, start + 5)
    assert time.monotonic() - start < 1


def test_an_attempt_that_returns_early_is_retried_until_the_deadline():
    calls = []
    start = time.monotonic()
    assert not wait_until(lambda remaining: calls.append(remaining) or False, start + 0.05)
    assert len(calls) > 1 and time.monotonic() - start >= 0.05


def test_lock_join_and_future_shapes():
    lock, done = threading.Lock(), threading.Event()
    lock.acquire()
    thread = threading.Thread(target=done.wait)
    thread.start()
    pool = futures.ThreadPoolExecutor(1)
    future = pool.submit(done.wait)
    deadline = time.monotonic() + 0.05
    assert not wait_until(lambda r: lock.acquire(True, r), deadline)
    assert not wait_until(lambda r: (thread.join(r), not thread.is_alive())[1], deadline)
    assert not wait_until(lambda r: bool(futures.wait([future], r).done), deadline)
    lock.release()
    done.set()
    deadline = time.monotonic() + 5
    assert wait_until(lambda r: lock.acquire(True, r), deadline)
    assert wait_until(lambda r: (thread.join(r), not thread.is_alive())[1], deadline)
    assert wait_until(lambda r: bool(futures.wait([future], r).done), deadline)
    pool.shutdown()


@pytest.mark.parametrize('step', ['forward_step', 'backward_step'])
def test_sleep_until_keeps_its_deadline_through_a_step_either_way(step, request):
    request.getfixturevalue(step)
    event = threading.Event()
    start = time.monotonic()
    assert not sleep_until(event.is_set, start + 0.2)
    assert 0.2 <= time.monotonic() - start < 1
    Timer(0.05, event.set).start()
    assert sleep_until(event.is_set, time.monotonic() + 5)


@pytest.mark.parametrize('step', ['forward_step', 'backward_step'])
def test_a_timer_fires_on_monotonic_time_through_a_step_either_way(step, request):
    request.getfixturevalue(step)
    fired = []
    start = time.monotonic()
    timer = Timer(0.2, lambda: fired.append(time.monotonic() - start))
    timer.start()
    timer.join(5)
    assert len(fired) == 1 and 0.2 <= fired[0] < 1


def test_a_cancelled_timer_never_fires():
    fired = []
    timer = Timer(0.1, fired.append, args=(1,))
    timer.start()
    timer.cancel()
    timer.join(5)
    assert not fired and not timer.is_alive()


def test_a_cancelled_future_raises_at_once():
    future = futures.Future()
    cancelled = []

    def cancel():
        time.sleep(0.05)
        cancelled.append(time.monotonic())
        future.cancel()
    threading.Thread(target=cancel).start()
    start = time.monotonic()
    with pytest.raises(futures.CancelledError):
        waits.result(future, start + 5)
    assert time.monotonic() - start < 1, 'the cancel surfaced only at the deadline'
    within(time.monotonic() - cancelled[0], 'CancelledError after the cancel', at_most=0.1)


def test_a_result_keeps_its_deadline_through_a_forward_step(stepped_conditions):
    future = futures.Future()
    threading.Thread(target=lambda: (time.sleep(0.1), future.set_result('done'))).start()
    start = time.monotonic()
    assert waits.result(future, start + 5) == 'done'
    assert 0.1 <= time.monotonic() - start < 1 and len(stepped_conditions) > 1
    pending = futures.Future()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        waits.result(pending, start + 0.1)
    assert time.monotonic() - start >= 0.1


def test_a_future_that_raised_timeout_error_raises_it_at_once():
    future, own = futures.Future(), TimeoutError('the future\'s own')
    future.set_exception(own)
    start = time.monotonic()
    with pytest.raises(TimeoutError) as raised:
        waits.result(future, start + 5)
    assert raised.value is own and time.monotonic() - start < 1


def test_join_keeps_its_deadline_through_a_forward_step(monkeypatch):
    step_forward(monkeypatch, joins=True)
    joins, join = [], threading.Thread.join
    monkeypatch.setattr(threading.Thread, 'join', lambda self, timeout=None: (joins.append(timeout), join(self, timeout))[1])
    thread = threading.Thread(target=time.sleep, args=(0.1,))
    thread.start()
    start = time.monotonic()
    assert waits.join(thread, start + 5)
    assert 0.1 <= time.monotonic() - start < 1 and len(joins) > 1
    done = threading.Event()
    thread = threading.Thread(target=done.wait)
    thread.start()
    start = time.monotonic()
    assert not waits.join(thread, start + 0.1)
    assert time.monotonic() - start >= 0.1
    done.set()
    assert waits.join(thread, time.monotonic() + 5)
