# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The station kept out of every test and of what it starts.

Unset ports are filled in from HFMODEM_RIG's profile or HFMODEM_STATION's file,
which name the station's own radio, PTT_PORT is the line that keys it, and
HFMODEM_PYTHON picks the interpreter a launcher hands the modem. A suite run
where the station exports them, `onair.main(["--transmit", ...])` in process or
a launcher in a subprocess, would otherwise reach that radio. So every test not
marked `hardware` runs without them, and without a BASH_ENV for the shells it
starts, under hermetic_py's tripwire: in this process and in every Python it
starts, opening a serial device or a modem port fails the test, and so does
using a sound card the test did not script. A claim the test leaves behind is
cleared if it is named for the test's HERMETIC_TAG, and fails the test if not.
A `hardware` test runs with none of this, and only when its own variable is set.

WHAT IT DOES NOT STOP. Under pytest the tripwire is Python's alone: there are
no shims for lsof, pkill, pgrep, ps or rigctl as tools/tests/hermetic.sh has,
so a test that runs one of those reaches the real thing. A module that took
sounddevice before the test began keeps it, apart from the calls patched here;
a child started with an env= of its own, `-I` or `-E` carries no tripwire; and
the limits tools/tests/hermetic.sh states for Python hold here too.
"""
import importlib.util
import itertools
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import pytest

STATION = ("CAT_PORT", "PTT_PORT", "HFMODEM_RIG", "HFMODEM_STATION", "HFMODEM_PYTHON")
REAL_PTT = os.environ.get("HFMODEM_TEST_REAL_PTT")
GUARD = Path(__file__).with_name("hermetic_py")
CARD = ("query_devices", "query_hostapis", "check_input_settings", "check_output_settings",
        "InputStream", "OutputStream", "Stream", "RawInputStream", "RawOutputStream",
        "RawStream", "rec", "play", "playrec")
_runs = itertools.count()


def _guard():
    if not (GUARD / "hermetic_guard.py").is_file():
        return None
    spec = importlib.util.spec_from_file_location("hermetic_guard_in_pytest", GUARD / "hermetic_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _guard()


def _card_tripwire(name):
    if name.startswith("__"):
        raise AttributeError(name)
    guard.trip(f"sounddevice.{name}")


def _wire(name):
    return lambda *a, **k: guard.trip(f"sounddevice.{name}")


def _plant(monkeypatch):
    """The sound card behind the tripwire for this test, and what was planted.
    The card's own calls are wires that trip when used, so a test can still put
    a scripted one in their place."""
    card = sys.modules.get("sounddevice")
    if card is None:
        stub = importlib.util.module_from_spec(importlib.util.spec_from_loader("sounddevice", loader=None))
        stub.__getattr__ = _card_tripwire
        wires = {name: _wire(name) for name in CARD}
        vars(stub).update(wires)
        monkeypatch.setitem(sys.modules, "sounddevice", stub)
        return {id(stub), *map(id, wires.values())}
    planted = set()
    for name in CARD:
        if name in vars(card):
            wire = _wire(name)
            monkeypatch.setattr(card, name, wire)
            planted.add(id(wire))
    return planted


def _forget(imported, planted):
    """Modules first imported during the test that kept something planted are
    dropped, so a later test imports them again with its own sound card."""
    for name in imported:
        module = sys.modules.get(name)
        if module is not None and any(id(v) in planted for v in vars(module).values()):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def station_kept_out(request, monkeypatch):
    if request.node.get_closest_marker("hardware") is not None:
        yield
        return
    if guard is None:
        pytest.fail(f"no tripwire at {GUARD}: this test would run with nothing between it "
                    "and the station's radio and sound card", pytrace=False)
    for key in (*STATION, "BASH_ENV", "ENV"):
        monkeypatch.delenv(key, raising=False)
    work = tempfile.mkdtemp(prefix="hermetic-")
    tag = f"hermeticpy{os.getpid()}n{next(_runs)}"
    monkeypatch.setenv("HERMETIC_WORK", work)
    monkeypatch.setenv("HERMETIC_TAG", tag)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(GUARD), os.environ.get("PYTHONPATH")])))
    before = set(sys.modules)
    planted = _plant(monkeypatch)
    guard.arm()
    try:
        yield
    finally:
        guard.disarm()
        guard.sweep(work, tag)
        _forget(set(sys.modules) - before, planted)
        trips = Path(work, "trips")
        found = trips.read_text() if trips.exists() else ""
        shutil.rmtree(work, ignore_errors=True)
    if found:
        pytest.fail(f"HERMETIC TRIP -- this test reached past its own sandbox:\n{found}", pytrace=False)


@pytest.fixture
def fake_claim_audio_metadata(monkeypatch, station_kept_out):
    """Supply only harmless device metadata to tests that acquire a claim.

    Claim ownership resolves Linux audio endpoints to /dev/snd nodes. Tests that
    exercise unrelated preflight/on-air/capture paths still need that lookup to
    be deterministic, while stream APIs and the audit guard remain tripwired.
    An omitted endpoint has a stable default pair; explicit station devices stay
    absent from enumeration, and named fake endpoints resolve to a non-hardware
    server name that produces no /dev/snd node.
    """
    card = sys.modules["sounddevice"]

    def query_devices(device=None, kind=None):
        if device is None:
            return []
        return {"name": "hfmodem test fake audio server",
                "max_input_channels": 1, "max_output_channels": 1}

    monkeypatch.setitem(vars(card), "default", SimpleNamespace(device=(0, 1)))
    monkeypatch.setitem(vars(card), "query_devices", query_devices)
