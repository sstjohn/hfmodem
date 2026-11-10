# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import socket
import threading
import time
import weakref

import pytest

from hfmodem.core.operations import LeaseTimeout
from hfmodem.station.observer import BoundedObserver


class FakeSocket:
    def __init__(self, *, blocked=False, error=False, stubborn=False, late_success=False):
        self.started = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.shutdown_called = threading.Event()
        self.blocked = blocked
        self.error = error
        self.stubborn = stubborn
        self.late_success = late_success
        self.payloads = []
        self.calls = []

    def sendall(self, payload):
        self.calls.append(("send", payload))
        self.started.set()
        try:
            if self.blocked:
                assert self.release.wait(5), "fake socket was not released"
            if isinstance(self.error, Exception):
                raise self.error
            if self.error or (self.shutdown_called.is_set() and not self.late_success):
                raise OSError("disconnected")
            self.payloads.append(payload)
        finally:
            self.finished.set()

    def shutdown(self, how):
        assert how == socket.SHUT_RDWR
        self.calls.append(("shutdown", how))
        self.shutdown_called.set()
        if not self.stubborn:
            self.release.set()

    def close(self):
        raise AssertionError("only the socket owner may close it")


class Disconnects:
    def __init__(self):
        self.condition = threading.Condition()
        self.generations = []
        self.threads = []

    def __call__(self, generation):
        with self.condition:
            self.generations.append(generation)
            self.threads.append(threading.current_thread().name)
            self.condition.notify_all()

    def wait(self, generation):
        with self.condition:
            assert self.condition.wait_for(lambda: generation in self.generations, 2)


@pytest.fixture
def observers():
    created = []

    def make(**kwargs):
        disconnects = Disconnects()
        observer = BoundedObserver(disconnects, **kwargs)
        created.append(observer)
        return observer, disconnects

    yield make
    for observer in created:
        observer.stop()


def test_captures_socket_and_immutable_payload_in_fifo_order(observers):
    observer, _ = observers()
    blocker, first, second = FakeSocket(blocked=True), FakeSocket(), FakeSocket()
    observer.install(1, blocker)
    observer.install(2, first)
    observer.install(3, second)
    assert observer.deliver(1, blocker, b"hold")
    assert blocker.started.wait(1)
    payload = bytearray(b"notice")
    assert observer.deliver(2, first, payload)
    payload[:] = b"mutate"
    assert observer.deliver(3, second, b"reply")
    assert not first.started.is_set()
    blocker.release.set()
    assert second.finished.wait(1)
    assert first.payloads == [b"notice"]
    assert second.payloads == [b"reply"]


@pytest.mark.parametrize("data_first", [False, True])
@pytest.mark.parametrize("stalled_side", ["command", "data"])
def test_stall_closes_both_members_in_either_connection_order(observers, data_first, stalled_side):
    observer, disconnects = observers(write_timeout=0.05)
    command = FakeSocket(blocked=stalled_side == "command")
    data = FakeSocket(blocked=stalled_side == "data")
    first, second = (data, command) if data_first else (command, data)
    observer.install("session", first)
    observer.install("session", second)
    stalled = command if stalled_side == "command" else data
    other = data if stalled_side == "command" else command
    assert observer.deliver("session", stalled, b"key-up")
    assert stalled.started.wait(1)
    assert observer.deliver("session", other, b"queued")
    disconnects.wait("session")
    assert command.shutdown_called.is_set() and data.shutdown_called.is_set()
    assert stalled.finished.wait(1)
    assert not other.started.is_set()
    assert not observer.deliver("session", other, b"stale")
    assert disconnects.threads == ["observer-watchdog"]


def test_data_only_provisional_generation_stalls_before_command_arrives(observers):
    observer, disconnects = observers(write_timeout=0.02)
    data = FakeSocket(blocked=True)
    observer.install(1, data)
    assert observer.deliver(1, data, b"early")
    disconnects.wait(1)
    with pytest.raises(ValueError, match="closed"):
        observer.install(1, FakeSocket())


@pytest.mark.parametrize("stubborn", [True, False])
@pytest.mark.parametrize("late_success", [False, True])
@pytest.mark.parametrize("data_first", [False, True])
def test_replace_stalled_member_preserves_generation_and_drops_displaced_queue(
        observers, late_success, data_first, stubborn):
    observer, disconnects = observers(write_timeout=0.05)
    displaced = FakeSocket(blocked=True, stubborn=stubborn, late_success=late_success)
    command, replacement = FakeSocket(), FakeSocket()
    first, second = (displaced, command) if data_first else (command, displaced)
    observer.install(1, first)
    observer.install(1, second)
    assert observer.deliver(1, displaced, b"inflight")
    assert displaced.started.wait(1)
    assert observer.deliver(1, displaced, b"discard")
    assert observer.deliver(1, command, b"keep")
    observer.install(1, replacement, replace=displaced)
    assert displaced.shutdown_called.is_set()
    assert not command.shutdown_called.is_set()
    assert not observer.deliver(1, displaced, b"stale")
    time.sleep(0.08)
    assert not command.shutdown_called.is_set()
    assert not disconnects.generations
    assert observer.deliver(1, replacement, b"new")
    if not stubborn:
        assert replacement.finished.wait(1)
    displaced.release.set()
    assert replacement.finished.wait(1)
    assert command.payloads == [b"keep"]
    assert replacement.payloads == [b"new"]
    assert [call for call in displaced.calls if call[0] == "send"] == [("send", b"inflight")]
    assert not disconnects.generations


def test_reconnect_while_old_sendall_unwinds_cannot_close_new_generation(observers):
    observer, disconnects = observers(write_timeout=0.02)
    old, new = FakeSocket(blocked=True, stubborn=True), FakeSocket()
    observer.install(1, old)
    assert observer.deliver(1, old, b"old")
    assert old.started.wait(1)
    assert observer.deliver(1, old, b"discard")
    observer.close_generation(1)
    assert old.shutdown_called.is_set()
    observer.install(2, new)
    assert observer.deliver(2, new, b"new")
    old.release.set()
    assert new.finished.wait(1)
    assert new.payloads == [b"new"]
    assert not new.shutdown_called.is_set()
    observer.stop()
    assert disconnects.generations == []


def test_full_queue_refuses_and_watchdog_disconnects_only_refused_generation(observers):
    observer, disconnects = observers(capacity=1)
    busy, command, data = FakeSocket(blocked=True), FakeSocket(), FakeSocket()
    observer.install(1, busy)
    observer.install(2, command)
    observer.install(2, data)
    assert observer.deliver(1, busy, b"busy")
    assert busy.started.wait(1)
    assert observer.deliver(2, command, b"queued")
    assert not observer.deliver(2, data, b"overflow")
    disconnects.wait(2)
    assert command.shutdown_called.is_set() and data.shutdown_called.is_set()
    assert not command.started.is_set()
    assert not busy.shutdown_called.is_set()
    assert not observer.deliver(2, data, b"later")
    busy.release.set()
    assert busy.finished.wait(1)
    assert disconnects.threads == ["observer-watchdog"]


def test_overflow_wakes_its_blocked_writer_without_waiting_for_write_timeout(observers):
    observer, disconnects = observers(capacity=1, write_timeout=30)
    blocked, peer, new = FakeSocket(blocked=True), FakeSocket(), FakeSocket()
    observer.install(1, blocked)
    observer.install(1, peer)
    assert observer.deliver(1, blocked, b"blocked")
    assert blocked.started.wait(1)
    assert observer.deliver(1, peer, b"queued")
    assert not observer.deliver(1, peer, b"overflow")
    disconnects.wait(1)
    assert blocked.finished.wait(1)
    assert peer.shutdown_called.is_set()
    assert not peer.started.is_set()
    observer.install(2, new)
    assert observer.deliver(2, new, b"reconnected")
    assert new.finished.wait(1)


def test_writer_serves_new_generation_after_watchdog_timeout(observers):
    observer, disconnects = observers(write_timeout=0.02)
    blocked, new = FakeSocket(blocked=True), FakeSocket()
    observer.install(1, blocked)
    assert observer.deliver(1, blocked, b"blocked")
    disconnects.wait(1)
    observer.install(2, new)
    assert observer.deliver(2, new, b"reconnected")
    assert new.finished.wait(1)
    assert new.payloads == [b"reconnected"]
    assert not new.shutdown_called.is_set()


def test_late_success_is_still_a_stall_when_watchdog_was_delayed():
    entered, release = threading.Event(), threading.Event()
    disconnects = Disconnects()

    def disconnect(generation):
        if generation == 1:
            entered.set()
            assert release.wait(5)
        disconnects(generation)

    observer = BoundedObserver(disconnect, write_timeout=0.02)
    slow = FakeSocket(blocked=True)
    try:
        trigger = FakeSocket(error=True)
        observer.install(1, trigger)
        assert observer.deliver(1, trigger, b"fail")
        assert entered.wait(1)
        observer.install(2, slow)
        assert observer.deliver(2, slow, b"late")
        assert slow.started.wait(1)
        time.sleep(0.04)
        slow.release.set()
        assert slow.finished.wait(1)
        marker = FakeSocket()
        observer.install(3, marker)
        assert observer.deliver(3, marker, b"after late send")
        assert marker.finished.wait(1)
        release.set()
        disconnects.wait(2)
        assert slow.shutdown_called.is_set()
    finally:
        slow.release.set()
        release.set()
        observer.stop()


@pytest.mark.parametrize("error", [OSError("disconnected"), RuntimeError("send broke")])
def test_send_failure_disconnects_pair_once_and_writer_serves_next_client(
        observers, caplog, error):
    observer, disconnects = observers()
    failed, peer, new = FakeSocket(error=error), FakeSocket(), FakeSocket()
    observer.install(1, failed)
    observer.install(1, peer)
    assert observer.deliver(1, failed, b"broken")
    disconnects.wait(1)
    observer.close_generation(1)
    observer.install(2, new)
    assert observer.deliver(2, new, b"ok")
    assert new.finished.wait(1)
    assert failed.shutdown_called.is_set() and peer.shutdown_called.is_set()
    assert disconnects.generations == [1]
    assert ("send broke" in caplog.text) == isinstance(error, RuntimeError)


def test_stop_wakes_blocked_writer_drops_queue_and_joins_both_threads(observers):
    observer, disconnects = observers()
    blocked, peer = FakeSocket(blocked=True), FakeSocket()
    observer.install(1, blocked)
    observer.install(1, peer)
    assert observer.deliver(1, blocked, b"blocked")
    assert blocked.started.wait(1)
    assert observer.deliver(1, peer, b"discard")
    observer.stop(timeout=1)
    assert blocked.finished.is_set()
    assert peer.shutdown_called.is_set()
    assert not peer.started.is_set()
    assert disconnects.generations == []
    assert not observer.deliver(1, blocked, b"stale")
    with pytest.raises(ValueError, match="closed"):
        observer.install(2, FakeSocket())
    observer.stop()


def test_stop_reports_writer_that_ignores_shutdown_and_can_be_retried(observers):
    observer, _ = observers()
    blocked = FakeSocket(blocked=True, stubborn=True)
    observer.install(1, blocked)
    assert observer.deliver(1, blocked, b"blocked")
    assert blocked.started.wait(1)
    try:
        started = time.monotonic()
        with pytest.raises(LeaseTimeout, match="observer-writer"):
            observer.stop(timeout=0.02)
        assert time.monotonic() - started < 0.5
        assert blocked.shutdown_called.is_set()
    finally:
        blocked.release.set()
    observer.stop()


def test_stop_reports_watchdog_stuck_in_server_disconnect():
    entered, release = threading.Event(), threading.Event()

    def disconnect(generation):
        entered.set()
        assert release.wait(5)

    observer = BoundedObserver(disconnect)
    trigger = FakeSocket(error=True)
    observer.install(1, trigger)
    assert observer.deliver(1, trigger, b"fail")
    assert entered.wait(1)
    try:
        with pytest.raises(LeaseTimeout, match="observer-watchdog"):
            observer.stop(timeout=0.02)
    finally:
        release.set()
        observer.stop()


def test_disconnect_callback_can_stop_observer():
    stopped = threading.Event()
    errors = []

    def disconnect(generation):
        try:
            observer.stop(timeout=1)
        except Exception as error:
            errors.append(error)
        finally:
            stopped.set()

    observer = BoundedObserver(disconnect)
    failed = FakeSocket(error=True)
    try:
        observer.install(1, failed)
        assert observer.deliver(1, failed, b"fail")
        assert stopped.wait(2)
        assert errors == []
        assert not observer.deliver(1, failed, b"stale")
    finally:
        observer.stop()


def test_retired_generation_history_is_bounded(observers):
    class Generation:
        pass

    observer, disconnects = observers()
    generation = Generation()
    oldest = weakref.ref(generation)
    observer.install(generation, FakeSocket())
    observer.close_generation(generation)
    for _ in range(256):
        generation = Generation()
        observer.install(generation, FakeSocket())
        observer.close_generation(generation)
    assert oldest() is None
    with pytest.raises(ValueError, match="closed"):
        observer.install(generation, FakeSocket())
    observer.stop()
    assert disconnects.generations == []


def test_disconnect_callback_runs_without_observer_lock_and_errors_are_reported(caplog):
    called = threading.Event()
    observer = None

    def disconnect(generation):
        def producer():
            assert not observer.deliver(generation, sock, b"stale")
            called.set()

        thread = threading.Thread(target=producer)
        thread.start()
        thread.join(1)
        assert not thread.is_alive()
        raise RuntimeError("callback broke")

    observer = BoundedObserver(disconnect)
    sock = FakeSocket(error=True)
    try:
        observer.install(1, sock)
        assert observer.deliver(1, sock, b"fail")
        assert called.wait(1)
    finally:
        observer.stop()
    assert "callback broke" in caplog.text


def test_socket_shutdown_errors_do_not_prevent_peer_shutdown_or_join(observers):
    class GoneSocket(FakeSocket):
        def shutdown(self, how):
            super().shutdown(how)
            raise OSError("already gone")

    observer, disconnects = observers()
    gone, peer = GoneSocket(blocked=True), FakeSocket()
    observer.install(1, gone)
    observer.install(1, peer)
    assert observer.deliver(1, gone, b"blocked")
    assert gone.started.wait(1)
    observer.stop()
    assert peer.shutdown_called.is_set()
    assert disconnects.generations == []


def test_rejects_unregistered_socket_and_invalid_replacements(observers):
    observer, _ = observers()
    sock, other = FakeSocket(), FakeSocket()
    assert not observer.deliver(1, sock, b"unknown")
    observer.install(1, sock)
    assert not observer.deliver(2, sock, b"wrong generation")
    assert not observer.deliver(1, other, b"wrong socket")
    with pytest.raises(ValueError, match="already installed"):
        observer.install(2, sock)
    with pytest.raises(ValueError, match="displaced"):
        observer.install(1, other, replace=FakeSocket())
    assert observer.deliver(1, sock, b"still here")
    assert sock.finished.wait(1)


@pytest.mark.parametrize("kwargs", [
    {"capacity": 0}, {"capacity": -1}, {"capacity": 1.5},
    {"write_timeout": 0}, {"write_timeout": -1},
    {"write_timeout": float("nan")}, {"write_timeout": float("inf")},
])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        BoundedObserver(lambda generation: None, **kwargs)


def test_default_write_bound_is_two_seconds(observers):
    observer, disconnects = observers()
    sock = FakeSocket(blocked=True)
    observer.install(1, sock)
    started = time.monotonic()
    assert observer.deliver(1, sock, b"blocked")
    assert sock.started.wait(1)
    assert sock.shutdown_called.wait(3)
    disconnects.wait(1)
    assert time.monotonic() - started >= 2.0
