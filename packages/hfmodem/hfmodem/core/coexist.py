# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import json
from functools import wraps
import os
from pathlib import Path
import signal
import time


def enabled():
    return bool(os.environ.get("HFMODEM_COEXIST_HARNESS"))


def test_run():
    # Any one of these marks a test run, and a child that loses one still carries
    # another, so it fails closed instead of reaching the operator's station.
    return enabled() or any(os.environ.get(key) for key in (
        "HFMODEM_STATE_DIR", "PYTEST_CURRENT_TEST", "HERMETIC_WORK"))


def event(name, **fields):
    if not enabled():
        return
    path = os.environ.get("HFMODEM_COEXIST_LOG")
    if not path:
        return
    record = dict(fields, event=name, pid=os.getpid(),
                  run=os.environ.get("HFMODEM_COEXIST_RUN"),
                  monotonic_ns=time.monotonic_ns(), time_ns=time.time_ns())
    data = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError("short coexist event write")
    finally:
        os.close(fd)


def _lost():
    # The oracle refuses a run carrying this mark, so a check that an event is
    # absent cannot pass on an event that was dropped.
    path = os.environ.get("HFMODEM_COEXIST_LOG")
    if path:
        try:
            Path(path + ".lost").touch()
        except OSError:
            pass


def _note(name, **fields):
    # A lost record fails the row through the oracle, never by changing whether
    # the barrier holds or the fault fires.
    try:
        event(name, **fields)
    except OSError:
        _lost()


def barrier(point, *, timeout=30.0):
    if not enabled():
        return
    directory = os.environ.get("HFMODEM_COEXIST_BARRIERS")
    if not directory:
        return
    if not point or Path(point).name != point:
        raise ValueError("invalid barrier name")
    root = Path(directory)
    if not (root / point).exists():
        return
    _note("barrier", point=point)
    (root / (point + ".reached")).touch()
    deadline = time.monotonic() + timeout
    while not (root / (point + ".release")).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"coexist barrier {point}")
        time.sleep(0.01)


def fault(point):
    if not enabled():
        return
    selected = os.environ.get("HFMODEM_COEXIST_FAULT", "")
    if selected not in (point, point + ":raise", point + ":kill"):
        return
    _note("fault", point=point)
    if selected.endswith(":kill"):
        os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError(f"coexist fault {point}")


def checkpoint(point):
    if not enabled():
        return
    barrier(point)
    fault(point)


def _always(*args, **kwargs):
    return True


def _never(*args, **kwargs):
    return False


def _instrument(cls, name, report, failed, lowers=_never):
    """A raising checkpoint (an injected fault, a barrier timeout) takes a
    failure form through `failed` that production's handlers already meet. A call
    that lowers the line runs its body first: no fault point may stand between the
    line and low.
    """
    original = getattr(cls, name)

    def stop(self, when, result):
        try:
            checkpoint(f"{when}_{cls.__name__}_{name}")
        except Exception as exc:  # noqa: BLE001
            failed(self, exc, result)

    @wraps(original)
    def observed(self, *args, **kwargs):
        lowering = lowers(*args, **kwargs)
        if not lowering:
            stop(self, "before", None)
        result = original(self, *args, **kwargs)
        try:
            report(self, result, args, kwargs)
        except Exception:  # noqa: BLE001 — an event is best effort, never a keying fault
            _lost()
        if lowering:
            stop(self, "before", result)
        stop(self, "after", result)
        return result

    setattr(cls, name, observed)


def instrument_ptt(cls):
    if os.environ.get("HFMODEM_COEXIST_HARNESS") != "1" or os.environ.get("HFMODEM_COEXIST_INSTRUMENT") != "1":
        return
    from .ptt import PttError

    def refused(line, exc, result):
        raise PttError(str(exc)) from exc

    def unconfirmed(line, exc, result):
        line.released_low = False

    release = cls.release

    def unopened(line, exc, result):
        release(line)
        refused(line, exc, result)

    @wraps(release)
    def contained(line):
        try:
            release(line)
        except PttError:      # only an injected readback fault; release never raises
            line.released_low = False

    cls.release = contained

    def opened(line, result, args, kwargs):
        event("device_open", path=line.port, fd=line._fd)

    def driven(line, result, args, kwargs):
        on = args[0] if args else kwargs["on"]
        event("key" if on else "deassert", path=line.port, fd=line._fd)

    def sensed(line, result, args, kwargs):
        event("line_readback", path=line.port,
              verdict="low" if result is False else "high" if result is True else "unreadable")

    def released(line, result, args, kwargs):
        event("device_close", path=line.port, released_low=line.released_low)

    _instrument(cls, "open", opened, unopened)
    _instrument(cls, "assert_", driven, refused, lambda on: not on)
    # Production answers an unreadable line with None; the hook raises PttError
    # instead. That is louder and never less safe: every sense() caller meets
    # PttError as a failed readback, after the line is low or before it rises,
    # and release contains it as unconfirmed.
    _instrument(cls, "_sense_fd", sensed, refused)
    _instrument(cls, "release", released, unconfirmed, _always)


def instrument_rig(cls):
    if os.environ.get("HFMODEM_COEXIST_HARNESS") != "1" or os.environ.get("HFMODEM_COEXIST_INSTRUMENT") != "1":
        return
    from .rig import RigError

    def report(rig, result, args, kwargs):
        event("rig_line_state", state=rig.line_state, keyed=rig.keyed)

    def refused(rig, exc, token):
        raise RigError(str(exc)) from exc

    _instrument(cls, "key", report, refused)
    _instrument(cls, "unkey", report, refused, _always)
