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

The example station records into ~/radio/captures, the operator's own, and
every test that loads it would add a capture there. So a station whose
capture_dir lands there records under the test's tmp_path instead, a write
there from any Python is a TRIP, and the run fails if anything new appears
there by the end. A test that leaves a recording thread running fails too: its
recording is stored after the test, outside the sandbox that admitted it.

WHAT IT DOES NOT STOP. Under pytest the tripwire is Python's alone: there are
no shims for lsof, pkill, pgrep or ps as tools/tests/hermetic.sh has,
so supported inventory commands reach the real thing. Direct CAT/media command
arguments are checked before Popen/exec, including the JSON exec trampoline.
Opaque shells, scripts and wrappers are refused; the `flac` on PATH, which
stores and reads recordings, is admitted with no devices. Python children must
retain this interpreter, site initialization, the guard startup path and test
identity.
This tripwire trusts supported native tools; it is not an OS sandbox for arbitrary
native programs. A module that took sounddevice before the test began keeps it,
apart from the calls patched here. The limits tools/tests/hermetic.sh states for
Python hold here too.
"""
import importlib.util
import itertools
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

STATION = ("CAT_PORT", "PTT_PORT", "AUDIO_IN", "AUDIO_OUT", "AUDIO", "HFMODEM_AUDIO",
           "HFMODEM_RIG", "HFMODEM_STATION", "HFMODEM_PYTHON")
REAL_PTT = os.environ.get("HFMODEM_TEST_REAL_PTT")
GUARD = Path(__file__).with_name("hermetic_py")
CARD = ("query_devices", "query_hostapis", "check_input_settings", "check_output_settings",
        "InputStream", "OutputStream", "Stream", "RawInputStream", "RawOutputStream",
        "RawStream", "rec", "play", "playrec")
_runs = itertools.count()

# Unset, HFMODEM_DATA_DIR is the operator's installed data, where besra keeps its
# on-air logs. It is set for the whole process before anything is collected, as
# tools resolve paths under it once, on import; a run a hermetic test starts keeps
# that test's.
if not os.environ.get("HERMETIC_TAG"):
    os.environ["HFMODEM_DATA_DIR"] = tempfile.mkdtemp(prefix="hfmodem-data-")


def _guard():
    startup = sys.modules.get("hermetic_guard")
    if startup is not None and startup.armed:
        return startup
    if not (GUARD / "hermetic_guard.py").is_file():
        return None
    spec = importlib.util.spec_from_file_location("hermetic_guard_in_pytest", GUARD / "hermetic_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _guard()
if guard is not None:
    GUARD = Path(guard.__file__).parent


def admit_platform_probes(environment):
    if os.name == "posix":
        for argv in (["uname", "-p"], ["file", "-b", platform._follow_symlinks(sys.executable)]):
            guard.admit(argv, environment)
    elif os.name == "nt":
        # platform.win32_ver() falls back to `ver` through the shell subprocess derives.
        comspec = os.environ.get("ComSpec") or os.path.join(os.environ.get("SystemRoot", ""),
                                                             "System32", "cmd.exe")
        if os.path.isabs(comspec):
            guard.admit([comspec, "/c", "ver"], environment)


def allow_virtual_serial(*ports):
    from hfmodem.core import winclaim
    for port in ports:
        guard.own_com(port, winclaim.serial_target(port))


def _serial_tripwires(monkeypatch):
    guard.install_serial_guards(monkeypatch.setattr)


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


def _captures():
    root = Path(guard.CAPTURES) if guard else None
    return set(root.iterdir()) if root and root.is_dir() else set()


@pytest.fixture(scope="session", autouse=True)
def captures_untouched():
    before = _captures()
    yield
    added = sorted(map(str, _captures() - before))
    if added:
        pytest.fail(f"the run wrote into the operator's captures: {added}", pytrace=False)


def _own_captures(monkeypatch, tmp_path):
    """A station recording into the operator's captures records into the test's."""
    from hfmodem.core import config

    parse = config.parse
    def parsed(*args, **kwargs):
        cfg = parse(*args, **kwargs)
        where = cfg.audio.capture_dir and os.path.realpath(os.path.expanduser(cfg.audio.capture_dir))
        if where and (where == guard.CAPTURES or where.startswith(guard.CAPTURES + os.sep)):
            cfg = replace(cfg, audio=replace(cfg.audio, capture_dir=str(tmp_path / "captures")))
        return cfg
    monkeypatch.setattr(config, "parse", parsed)


@pytest.fixture(autouse=True)
def station_kept_out(request, monkeypatch):
    if request.node.get_closest_marker("hardware") is not None:
        yield
        return
    if guard is None:
        pytest.fail(f"no tripwire at {GUARD}: this test would run with nothing between it "
                    "and the station's radio and sound card", pytrace=False)
    saved = {key: os.environ.get(key) for key in STATION}
    for key in (*STATION, "BASH_ENV", "ENV"):
        monkeypatch.delenv(key, raising=False)
    # The recording resolver exports its result into os.environ; set first so teardown restores it.
    monkeypatch.setenv("HFMODEM_RECORDING", "")
    monkeypatch.delenv("HFMODEM_RECORDING")
    work = tempfile.mkdtemp(prefix="hermetic-")
    tag = f"hermeticpy{os.getpid()}n{next(_runs)}"
    monkeypatch.setenv("HERMETIC_WORK", work)
    monkeypatch.setenv("HERMETIC_TAG", tag)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(GUARD), os.environ.get("PYTHONPATH")])))
    _own_captures(monkeypatch, request.getfixturevalue("tmp_path"))
    before = set(sys.modules)
    threads = set(threading.enumerate())
    planted = _plant(monkeypatch)
    was_armed = guard.armed
    guard.arm()
    try:
        admit_platform_probes(os.environ)
        if flac := shutil.which("flac"):
            guard.admit([flac], os.environ)
        _serial_tripwires(monkeypatch)
        yield
    finally:
        guard.disarm()
        if was_armed:
            guard.arm()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        guard.sweep(work, tag)
        _forget(set(sys.modules) - before, planted)
        trips = Path(work, "trips")
        found = trips.read_text() if trips.exists() else ""
        shutil.rmtree(work, ignore_errors=True)
    if found:
        pytest.fail(f"HERMETIC TRIP -- this test reached past its own sandbox:\n{found}", pytrace=False)
    # A closed recorder may still be finishing on its own thread: an attached
    # child's control ends once its relay reads the pipe's end.
    deadline = time.monotonic() + 2
    for thread in threading.enumerate():
        if thread not in threads and thread.name.startswith("recording-"):
            thread.join(max(0, deadline - time.monotonic()))
    left = sorted(t.name for t in threading.enumerate()
                  if t not in threads and t.name.startswith("recording-") and t.is_alive())
    if left:
        pytest.fail(f"recording threads outlive the test, so a recorder was left open: {left}",
                    pytrace=False)


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
