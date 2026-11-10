# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
import signal
import threading

import pytest

from hfmodem.core import claim, coexist, winclaim


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("HARNESS", "FAULT", "LOG", "BARRIERS", "RUN"):
        monkeypatch.delenv("HFMODEM_COEXIST_" + name, raising=False)


def test_hooks_inert_without_harness(monkeypatch, tmp_path):
    monkeypatch.delenv("HFMODEM_COEXIST_HARNESS", raising=False)
    monkeypatch.setenv("HFMODEM_COEXIST_FAULT", "after_mutex:kill")
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(tmp_path / "events"))
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(tmp_path))
    (tmp_path / "after_mutex").touch()
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("inert hook signalled"))
    coexist.event("test")
    coexist.barrier("after_mutex", timeout=0)
    coexist.fault("after_mutex")
    coexist.checkpoint("after_mutex")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["after_mutex"]


def test_event_log_identity_and_clock_pairs(monkeypatch, tmp_path):
    path = tmp_path / "events"
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_RUN", "unit")
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(path))
    coexist.event("claim_acquired", fd=9, dev=2, ino=42, nonce=17, start="now")
    coexist.event("claim_closed", fd=9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["claim_acquired", "claim_closed"]
    assert rows[0]["pid"] == os.getpid() and rows[0]["run"] == "unit"
    assert rows[0]["nonce"] == 17 and rows[0]["start"] == "now"
    assert rows[0]["monotonic_ns"] <= rows[1]["monotonic_ns"]
    assert rows[0]["time_ns"] > 0


def test_barrier_waits_for_release(monkeypatch, tmp_path):
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(tmp_path))
    (tmp_path / "before_open").touch()
    finished = threading.Event()
    errors = []
    def wait():
        try:
            coexist.barrier("before_open", timeout=3)
            finished.set()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=wait)
    thread.start()
    try:
        assert not finished.wait(.05)
        (tmp_path / "before_open.release").touch()
        assert finished.wait(3)
        assert (tmp_path / "before_open.reached").exists()
    finally:
        thread.join(3)
    assert not errors and not thread.is_alive()


def test_barrier_timeout_and_missing_point(monkeypatch, tmp_path):
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(tmp_path))
    coexist.barrier("missing", timeout=0)
    (tmp_path / "blocked").touch()
    with pytest.raises(TimeoutError):
        coexist.barrier("blocked", timeout=0)


@pytest.mark.parametrize("suffix", ["", ":raise", ":kill"])
def test_fault_points(monkeypatch, suffix):
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_FAULT", "after_mutex" + suffix)
    calls = []
    monkeypatch.setattr(os, "kill", lambda *args: calls.append(args))
    coexist.fault("elsewhere")
    with pytest.raises(RuntimeError, match="after_mutex"):
        coexist.fault("after_mutex")
    fatal = signal.SIGTERM if os.name == "nt" else signal.SIGKILL
    assert calls == ([(os.getpid(), fatal)] if suffix == ":kill" else [])


@pytest.mark.parametrize("role", ["event", "holder", "cleanup", "admission", "bootstrap"])
def test_windows_open_failure_contract(monkeypatch, tmp_path, role):
    # Exercise the Windows role boundary even on POSIX, without Win32 calls.
    error = PermissionError("injected access denied")
    error.winerror = 5

    def denied(path, opened_role):
        assert path == tmp_path / "events" and opened_role == role
        raise error

    monkeypatch.setattr(winclaim, "_open", denied)
    expected = OSError if role == "event" else claim.ClaimError
    with pytest.raises(expected) as caught:
        winclaim.open_role(tmp_path / "events", role)
    if role == "event":
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error


@pytest.fixture
def failed_log_open(monkeypatch, tmp_path):
    path = tmp_path / "events"
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_COEXIST_LOG", str(path))
    error = PermissionError("injected access denied")
    error.winerror = 5
    original_open = os.open

    def denied(opened_path, role_or_flags, *args, **kwargs):
        if os.name != "nt" and str(opened_path) != str(path):
            return original_open(opened_path, role_or_flags, *args, **kwargs)
        assert str(opened_path) == str(path)
        if os.name == "nt":
            assert role_or_flags == "event"
        raise error

    if os.name == "nt":
        monkeypatch.setattr(winclaim, "_open", denied)
    else:
        monkeypatch.setattr(os, "open", denied)
    return path


def test_event_open_failure_is_oserror(failed_log_open):
    with pytest.raises(OSError, match="injected access denied"):
        coexist.event("test")


def test_failed_logging_preserves_barrier(monkeypatch, tmp_path, failed_log_open):
    monkeypatch.setenv("HFMODEM_COEXIST_BARRIERS", str(tmp_path))
    (tmp_path / "after_mutex").touch()
    waits = []

    def release(delay):
        assert (tmp_path / "after_mutex.reached").exists()
        assert failed_log_open.with_suffix(".lost").exists()
        waits.append(delay)
        (tmp_path / "after_mutex.release").touch()

    monkeypatch.setattr(coexist.time, "sleep", release)
    coexist.barrier("after_mutex")
    assert waits == [.01]


@pytest.mark.parametrize("suffix", ["", ":raise", ":kill"])
def test_failed_logging_preserves_fault(monkeypatch, failed_log_open, suffix):
    monkeypatch.setenv("HFMODEM_COEXIST_FAULT", "after_mutex" + suffix)
    calls = []

    def kill(*args):
        assert failed_log_open.with_suffix(".lost").exists()
        calls.append(args)

    monkeypatch.setattr(os, "kill", kill)
    with pytest.raises(RuntimeError, match="coexist fault after_mutex"):
        coexist.fault("after_mutex")
    assert failed_log_open.with_suffix(".lost").exists()
    fatal = signal.SIGTERM if os.name == "nt" else signal.SIGKILL
    assert calls == ([(os.getpid(), fatal)] if suffix == ":kill" else [])
