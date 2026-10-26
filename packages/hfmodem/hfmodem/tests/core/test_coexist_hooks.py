# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
import signal
import threading

import pytest

from hfmodem.core import coexist


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
    assert calls == ([(os.getpid(), signal.SIGKILL)] if suffix == ":kill" else [])
