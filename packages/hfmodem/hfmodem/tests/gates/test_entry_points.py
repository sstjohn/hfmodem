# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every declared entry point resolves and starts.

This exists because it caught something the moment it was written: the `hfmodem`
console script was declared in `pyproject.toml` pointing at `hfmodem.cli:main`, and
`hfmodem/cli.py` did not exist. Nothing noticed — a wheel would have installed
cleanly and produced a command that fails on first use, which is the same class of
defect as a package-data glob that ships no assets.

shrike's `validate_static` had this check and it was the best idea in it; it was
not carried into the shared gates until this file. It reads the declarations rather
than a hand-written list, so a new entry point is covered without anyone
remembering — with one exception, below.

`-m`-only modules are listed explicitly. Enumerating from `[project.scripts]`
*alone* would be a coverage regression: `shrike.ota` and `shrike.beacon` are
runnable only as `python -m`, and they are the two that key a transmitter.
"""
from __future__ import annotations

import ast
import importlib
import atexit
import fcntl
import io
import os
import subprocess
import sys
import threading
import tomllib
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
PACKAGES = REPO / "packages"


@pytest.fixture(autouse=True)
def checkout_tool_entry_points(monkeypatch):
    """Resolve standalone tools without another test changing the import path."""
    monkeypatch.syspath_prepend(str(REPO / "tools"))
    tool = importlib.import_module("kestrel_connect")
    assert Path(tool.__file__).resolve() == (REPO / "tools/kestrel_connect.py").resolve()


#: Runnable with `python -m` but not declared as console scripts. Named here so
#: the union of covered entry points cannot silently shrink.
MODULE_ENTRY_POINTS = [
    "hfmodem.brubru.rx",
    "hfmodem.brubru.tx",
    # The station itself. At the radio the venv is addressed by path rather
    # than activated, so this and the console script must agree.
    "hfmodem",
    "hfmodem.shrike.monitor",
    "hfmodem.shrike.live",
    "hfmodem.shrike.onair",
    "hfmodem.shrike.ota",
    "hfmodem.shrike.beacon",
    "hfmodem.shrike.ptc",
    "hfmodem.besra.monitor",
    "hfmodem.besra.host.run_server",
    "hfmodem.sabir.monitor",
    "hfmodem.sabir.offair",
    "hfmodem.sabir.onair",
    "hfmodem.sabir.host.run_server",
    "hfmodem.bateleur.offair",
    "hfmodem.kestrel.host.run_server",
    "hfmodem.lanner.qso",
]


def _scripts() -> list[tuple[str, str, str]]:
    """(dist, command, "module:function") for every declared console script."""
    out = []
    for py in sorted(PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(py.read_text(encoding="utf-8"))
        for cmd, target in data.get("project", {}).get("scripts", {}).items():
            out.append((py.parent.name, cmd, target))
    return out


def test_some_console_scripts_are_declared():
    """A gate that iterates an empty list passes for the wrong reason.

    Two, deliberately: `hfmodem` and `creance`. The four repos declared ten
    between them — `shrike-monitor`, `besra-modem`, `sabir-server` and the rest —
    and they collapse into one command with verbs rather than one command per
    modem, because a station is one thing. The modules themselves are still
    runnable and `MODULE_ENTRY_POINTS` is what keeps them covered.
    """
    declared = _scripts()
    assert len(declared) == 2, declared
    assert {c for _, c, _ in declared} == {"hfmodem", "creance"}


@pytest.mark.parametrize("dist,cmd,target", _scripts(),
                         ids=[f"{c}" for _, c, _ in _scripts()])
def test_a_declared_console_script_resolves(dist, cmd, target):
    """The defect this file was written for: a declared script whose module does
    not exist installs fine and fails on first use."""
    mod_name, _, func_name = target.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        pytest.fail(f"{dist} declares `{cmd} = {target}` and {mod_name} will not "
                    f"import: {exc}")
    fn = getattr(mod, func_name, None)
    assert callable(fn), f"{target} is not callable"


@pytest.mark.parametrize("mod", MODULE_ENTRY_POINTS)
def test_a_module_entry_point_starts(mod):
    """`--help` and exit 0. It proves the module imports, its argument parser is
    well formed, and nothing at import time needs a radio."""
    r = subprocess.run([sys.executable, "-m", mod, "--help"],
                       capture_output=True, text=True, timeout=120,
                       cwd=PACKAGES / "hfmodem")
    assert r.returncode == 0, (
        f"python -m {mod} --help exited {r.returncode}\n{r.stderr[-1500:]}")


#: Every entry point that must stay covered, by name. A `>=` on a count let an
#: audit delete `hfmodem.shrike.beacon` — one of the two that key a transmitter —
#: and the gate stayed green, which is the whole failure mode this file is about.
REQUIRED_COVERAGE = frozenset(MODULE_ENTRY_POINTS) | {"hfmodem.cli", "creance.cli"}


def test_the_covered_set_does_not_shrink():
    """Both halves, by name rather than by count. Enumerating only the
    declarations would drop the `-m`-only modules, two of which key a
    transmitter."""
    covered = {t.partition(":")[0] for _, _, t in _scripts()} | set(MODULE_ENTRY_POINTS)
    missing = REQUIRED_COVERAGE - covered
    assert not missing, f"entry points no longer covered: {sorted(missing)}"


MODEM_STARTUPS = [
    ("hfmodem.shrike.onair", "run"),
    ("hfmodem.sabir.onair", "run"),
    ("hfmodem.lanner.qso", "main"),
    ("kestrel_connect", "main"),
]


class FakeAdmission:
    def __init__(self, events):
        self.events = events
        self.held = True

    def opened(self):
        assert self.held
        self.events.append("opened")
        self.held = False

    def failed(self):
        self.events.append("failed")
        self.held = False


#: Where each modem's device work begins; sabir's admission follows its pure
#: refusals, rendering and profile selection inside `run`.
SEAMS = {"hfmodem.sabir.onair": "_keyed"}


def sabir_armed(monkeypatch, onair):
    monkeypatch.setattr(onair.config, "tx_drive", lambda value: 0.1)
    monkeypatch.setattr(onair, "_refusals", lambda args: None)
    monkeypatch.setattr(onair, "_rules", lambda args: None)
    monkeypatch.setattr(onair, "_burst", lambda args: SimpleNamespace(
        describe=lambda dial: "fake burst", audio=[], span_hz=(1000, 2000),
        emission=lambda *a: None, waveform="fake"))
    return onair.build_parser().parse_args(["--transmit", "--mycall", "N0CALL",
                                            "--dial", "7100000"])


def startup_call(monkeypatch, module, function):
    if module.__name__ == "hfmodem.sabir.onair":
        args = sabir_armed(monkeypatch, module)
        return lambda: module.run(args)
    if function == "run":
        return lambda: module.run(SimpleNamespace(
            transmit=True, listen_only=False, preflight=False, tune=False))
    if module.__name__ == "hfmodem.lanner.qso":
        return lambda: module.main([
            "--audio-in", "fake", "--audio-out", "fake", "--ptt-device", "fake",
            "--adif", "unused.adif"])
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AW",
                                      "--mycall", "N0CALL"])
    monkeypatch.setattr(module, "unspeakable", lambda *a: [])
    return module.main


@pytest.mark.parametrize("name,function", MODEM_STARTUPS)
@pytest.mark.parametrize("failure", [None, OSError, KeyboardInterrupt, SystemExit])
def test_modem_holds_admission_through_startup(monkeypatch, name, function, failure):
    from hfmodem.core import claim

    module = importlib.import_module(name)
    events = []
    admission = FakeAdmission(events)

    def inherited(fd, *, launch):
        assert (fd, launch) == (9, True)
        events.append("inherited")
        return admission

    def run(*args, **kwargs):
        startup = kwargs.get("admission") or args[-1]
        if hasattr(startup, "take"):   # shrike takes it at its first device open
            assert events == []
            startup.take()
            startup = startup._held
        assert startup is admission and admission.held
        events.append("device open")
        if failure:
            raise failure()
        admission.opened()
        return 17

    monkeypatch.setattr(claim, "inherited", inherited)
    monkeypatch.setattr(module, SEAMS.get(name, "_run"), run)
    call = startup_call(monkeypatch, module, function)
    if failure:
        with pytest.raises(failure):
            call()
        assert events == ["inherited", "device open", "failed"]
    else:
        assert call() == 17
        assert events == ["inherited", "device open", "opened", "failed"]
    assert not admission.held


@pytest.mark.parametrize("name,function", MODEM_STARTUPS)
def test_modem_refuses_before_any_open(monkeypatch, capsys, name, function):
    from hfmodem.core import claim

    module = importlib.import_module(name)

    def refused(*a, **kw):
        raise claim.ClaimError("cleanup in progress")

    def run(*args, **kwargs):
        startup = kwargs.get("admission") or args[-1]
        if hasattr(startup, "take"):
            startup.take()
        pytest.fail("device opened")

    monkeypatch.setattr(claim, "inherited", refused)
    monkeypatch.setattr(module, SEAMS.get(name, "_run"), run)
    try:
        code = startup_call(monkeypatch, module, function)()
    except SystemExit as exc:
        code = exc.code
    assert code == 2
    assert "CLAIM NOT ADMITTED: cleanup in progress" in capsys.readouterr().out


def test_rendering_needs_no_claim(monkeypatch):
    from hfmodem.core import claim
    from hfmodem.sabir import onair as sabir
    from hfmodem.shrike import onair as shrike

    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: pytest.fail("claim requested"))
    monkeypatch.setattr(shrike, "_run", lambda args, *, admission: admission)
    assert shrike.run(SimpleNamespace(transmit=False, listen_only=False,
                                      preflight=False, tune=False)) is None
    monkeypatch.setattr(sabir, "_keyed", lambda *a: pytest.fail("device opened"))
    assert sabir.main(["--mycall", "N0CALL", "--dial", "7100000"]) == 0


def test_a_pure_refusal_needs_no_claim(monkeypatch, capsys):
    from hfmodem.core import claim
    from hfmodem.sabir import onair as sabir
    from hfmodem.shrike import onair as shrike, ota

    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: pytest.fail("claim requested"))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(sys, "argv", ["onair", "--transmit", "--mycall", "N0CALL",
                                      "--dxcall", "K7ABC", "--dial", "7100000",
                                      "--serial", "/dev/null", "--max-cycles", "1"])
    monkeypatch.setattr(ota, "ptt_device", lambda rig, serial: "/dev/null")
    with pytest.raises(SystemExit, match="rigctl is not on PATH"):
        shrike.main()
    assert sabir.main(["--mycall", "N0CALL", "--dial", "7100000", "--transmit",
                       "--ptt-device", "/dev/null", "--line-ptt", "--rigctld", "127.0.0.1:1",
                       "--regulatory", "part97", "--licence", "general",
                       "--listen", str(sabir.MAX_LISTEN_S + 1)]) == 2
    assert "--listen" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [False, True])
def test_besra_acknowledges_after_radio_start(monkeypatch, failure):
    from hfmodem.besra.host import run_server
    from hfmodem.core import claim

    events = []
    admission = FakeAdmission(events)
    monkeypatch.setattr(sys, "argv", ["besra", "--radio", "ft891"])
    monkeypatch.setattr(claim, "inherited", lambda fd, *, launch:
                        admission if (fd, launch) == (9, True) else pytest.fail("bad claim"))

    def radio(args):
        assert admission.held
        events.append("radio start")
        if failure:
            raise OSError("fake open failed")
        return object(), SimpleNamespace(close=lambda: events.append("close"))

    def serve(*a, **kw):
        assert not admission.held
        return SimpleNamespace(serve_forever=lambda: events.append("serve"))

    monkeypatch.setattr(run_server, "_radio_modem", radio)
    monkeypatch.setattr(run_server, "HostServer", serve)
    monkeypatch.setattr(run_server.rates, "host_report", lambda: "fake report")
    if failure:
        with pytest.raises(OSError, match="fake open failed"):
            run_server.main()
        assert events == ["radio start", "failed"]
    else:
        assert run_server.main() == 0
        assert events == ["radio start", "opened", "failed", "serve", "close"]


@pytest.mark.parametrize("line", [[], ["--ptt-device", "/dev/null"]])
def test_kestrel_arm_refuses_the_line_then_line_ptt_before_claim_or_rig(
        monkeypatch, capsys, line):
    import kestrel_connect
    from hfmodem.core import claim
    from hfmodem.core.ptt import arming_refusal

    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AW",
                                      "--mycall", "N0CALL", "--arm", *line])
    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: pytest.fail("claim requested"))
    monkeypatch.setattr(kestrel_connect, "Rig", lambda *a, **kw: pytest.fail("rig opened"))
    assert kestrel_connect.main() == 2
    out = capsys.readouterr().out
    assert (arming_refusal(None) if not line else "--arm requires --line-ptt") in out, out


@pytest.mark.parametrize("failure", [False, True])
def test_kestrel_acknowledges_after_audio_start(monkeypatch, failure):
    import kestrel_connect as kestrel

    events = []
    admission = FakeAdmission(events)

    def start():
        assert admission.held
        events.append("audio start")
        if failure:
            raise OSError("fake audio failed")

    monkeypatch.setattr(kestrel.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(kestrel, "Session", lambda: None)
    monkeypatch.setattr(kestrel, "wait_clear_channel", lambda *a: False)
    io = SimpleNamespace(start=start)
    call = lambda: kestrel.connect("W1AW", "N0CALL", "2300", io,
                                   hs=object(), admission=admission)
    if failure:
        with pytest.raises(OSError, match="fake audio failed"):
            call()
        assert events == ["audio start"]
    else:
        assert call() is False
        assert events == ["audio start", "opened"]


class FakeNativeChild:
    pid = 123

    def __init__(self, tmp_path, *, stubborn=False):
        self.stdin = (tmp_path / "native-stdin").open("w+b")
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"error: fake diagnostic\n")
        self.live = True
        self.stubborn = stubborn
        self.events = []

    def alive(self):
        return self.live

    def terminate(self):
        self.events.append("terminate")
        if not self.stubborn:
            self.live = False

    def kill(self):
        self.events.append("kill")

    def wait_gone(self, timeout):
        self.events.append(("wait", timeout))
        return not self.live


@pytest.fixture
def besra_rig():
    from hfmodem.besra.radio import Rig

    rig = Rig(1, "fake-cat", 38400, rigctl="/bin/sh")
    atexit.unregister(rig._atexit_unkey)
    yield rig
    if rig._proc is not None:
        rig._proc.live = False
        rig._kill()


def test_besra_native_child_is_reused_and_shutdown_verified(monkeypatch, tmp_path, besra_rig):
    from hfmodem.core import native

    rig = besra_rig
    child = FakeNativeChild(tmp_path)
    calls = []

    def spawn(argv, **kwargs):
        assert argv == ["/bin/sh", "-m", "1", "-r", "fake-cat", "-s", "38400",
                        "-P", "NONE"]
        assert kwargs == dict(claim_fd=9, long_lived=True, device="fake-cat")
        calls.append(argv)
        return child

    monkeypatch.setattr(native, "spawn_native", spawn)
    assert rig._write("F", "7100000")
    assert rig._write("M", "PKTUSB", "2700")
    child.stdin.seek(0)
    assert child.stdin.read() == b"F 7100000\nM PKTUSB 2700\n"
    assert len(calls) == 1
    rig.stop()
    assert rig._proc is None and not child.live
    assert child.events == ["terminate", ("wait", 2)]
    assert all(s.closed for s in (child.stdin, child.stdout, child.stderr))
    assert not rig._write("F", "7200000")
    assert len(calls) == 1


@pytest.mark.parametrize("broken_pipe", [False, True])
def test_besra_reaps_previous_child_before_replacement(monkeypatch, tmp_path, besra_rig,
                                                      broken_pipe):
    from hfmodem.core import native

    rig = besra_rig
    first = FakeNativeChild(tmp_path)
    assert first.stdin is not None
    second = FakeNativeChild(tmp_path)
    calls = []

    def spawn(*a, **kw):
        if calls:
            assert not first.live
            assert all(s.closed for s in (first.stdin, first.stdout, first.stderr))
        calls.append(a)
        return first if len(calls) == 1 else second

    monkeypatch.setattr(native, "spawn_native", spawn)
    assert rig._write("F", "7100000")
    if broken_pipe:
        first.stdin.close()                      # dead pipe while the child still lives
    else:
        first.live = False
    assert rig._write("F", "7200000")
    assert rig._proc is second and len(calls) == 2


@pytest.mark.parametrize("with_child", [False, True])
def test_besra_failed_admission_never_retries(monkeypatch, tmp_path, besra_rig, with_child):
    from hfmodem.core import native

    rig = besra_rig
    child = FakeNativeChild(tmp_path, stubborn=True) if with_child else None
    calls = []

    def spawn(*a, **kw):
        calls.append(a)
        raise native.NativeSpawnError("cleanup in progress", child)

    monkeypatch.setattr(native, "spawn_native", spawn)
    assert not rig._write("F", "7100000")
    assert not rig._write("F", "7200000")
    assert len(calls) == 1 and rig._sick
    assert rig._proc is child


def test_besra_unresolved_child_blocks_replacement_and_shutdown(monkeypatch, tmp_path, besra_rig):
    from hfmodem.core import claim, native

    rig = besra_rig
    child = FakeNativeChild(tmp_path, stubborn=True)
    calls = []
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: calls.append(a) or child)
    assert rig._write("F", "7100000")
    # EPIPE does not establish death: keep the handle and all diagnostic pipes.
    original = child.stdin

    class BrokenPipe:
        def write(self, line):
            raise BrokenPipeError()

    child.stdin = BrokenPipe()
    assert not rig._write("F", "7200000")
    assert rig._proc is child and rig._sick and len(calls) == 1
    child.stdin = original
    assert not any(s.closed for s in (child.stdin, child.stdout, child.stderr))
    with pytest.raises(claim.ClaimError, match="termination unresolved"):
        rig.stop()
    assert not rig._stopped and rig._proc is child
    assert "terminate" in child.events and "kill" in child.events
    assert not rig._write("F", "7300000")
    assert len(calls) == 1


def test_besra_native_shutdown_escalates_to_kill(monkeypatch, tmp_path, besra_rig):
    from hfmodem.core import native

    child = FakeNativeChild(tmp_path, stubborn=True)

    def kill():
        child.events.append("kill")
        child.live = False

    child.kill = kill
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: child)
    assert besra_rig._write("F", "7100000")
    besra_rig.stop()
    assert child.events == ["terminate", ("wait", 2), "kill", ("wait", 2)]
    assert besra_rig._proc is None


@pytest.mark.parametrize("during_spawn", [False, True])
def test_besra_unkey_never_respawns_cat(monkeypatch, tmp_path, besra_rig, during_spawn):
    from hfmodem.core import native

    rig = besra_rig
    child = FakeNativeChild(tmp_path)
    calls = []

    def spawn(*a, **kw):
        calls.append(a)
        rig.unkey("overtook spawn")
        return child

    monkeypatch.setattr(native, "spawn_native", spawn)
    if not during_spawn:
        rig.unkey("before spawn")
    assert not rig._write("F", "7100000")
    assert rig._proc is None
    assert len(calls) == int(during_spawn)
    if during_spawn:
        assert not child.live and child.stdin.closed
    else:
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()


def test_besra_stop_unkeys_line_before_cat_and_hands_back_on_failure(monkeypatch, tmp_path,
                                                                  besra_rig):
    from hfmodem.core import claim, native

    rig = besra_rig
    child = FakeNativeChild(tmp_path, stubborn=True)
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: child)
    assert rig._write("F", "7100000")
    rig.line_ptt = True
    rig._keyer = SimpleNamespace(
        key=lambda on: child.events.append(("key", on)) or True,
        must_retire=False, hand_back=lambda: child.events.append("line closed"))
    with pytest.raises(claim.ClaimError, match="termination unresolved"):
        rig.stop()
    assert child.events[0] == ("key", False)
    assert child.events[-1] == "line closed"
    assert rig._keyer is None and rig._proc is child and not rig._stopped
    assert not rig._write("F", "7200000")


def test_besra_line_ptt_keeps_cat_until_stop(monkeypatch, tmp_path, besra_rig):
    from hfmodem.core import native

    rig = besra_rig
    child = FakeNativeChild(tmp_path)
    calls = []
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: calls.append(a) or child)
    rig.line_ptt = True
    rig._keyer = SimpleNamespace(
        key=lambda on: child.events.append(("key", on)) or True,
        must_retire=False, hand_back=lambda: child.events.append("line closed"))
    assert rig._write("F", "7100000")
    assert rig.ptt(True) and rig.ptt(False)
    assert rig._write("M", "PKTUSB", "2700")
    assert rig._proc is child and child.alive() and len(calls) == 1
    rig.stop()
    assert child.events == [("key", True), ("key", False), ("key", False),
                            "terminate", ("wait", 2), "line closed"]
    assert rig._proc is None and rig._keyer is None


def test_besra_netrigctl_uses_text_popen_without_native_admission(monkeypatch, besra_rig):
    from hfmodem.besra import radio
    from hfmodem.core import native

    rig = besra_rig
    rig.rigctld = "localhost:4532"
    child = SimpleNamespace(stdin=io.StringIO(), stdout=None, stderr=None,
                            poll=lambda: None, kill=lambda: None, wait=lambda **kw: 0)
    calls = []
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: pytest.fail("native admission"))

    def popen(argv, **kwargs):
        assert argv == ["/bin/sh", "-m", "2", "-r", "localhost:4532"]
        assert kwargs["text"] is True
        calls.append(argv)
        return child

    monkeypatch.setattr(radio.subprocess, "Popen", popen)
    assert rig._write("F", "7100000")
    assert rig._write("M", "PKTUSB", "2700")
    assert child.stdin.getvalue() == "F 7100000\nM PKTUSB 2700\n"
    assert len(calls) == 1
    rig.stop()
    assert rig._proc is None and child.stdin.closed


def fake_ota_rig():
    from hfmodem.shrike import ota

    rig = ota.Rig.__new__(ota.Rig)
    rig.rigctl, rig.model, rig.serial, rig.baud = "fake-rigctl", 1, "fake-cat", 38400
    rig._proc = None
    rig._sick = False
    rig._down = threading.Event()
    rig._errs = deque(maxlen=8)
    return rig


def test_shrike_native_child_is_reused_and_shutdown_verified(monkeypatch, tmp_path):
    from hfmodem.core import native

    rig = fake_ota_rig()
    child = FakeNativeChild(tmp_path)
    calls = []

    def spawn(argv, **kwargs):
        assert argv[-2:] == ["-P", "NONE"]
        assert kwargs == dict(claim_fd=9, long_lived=True, device="fake-cat")
        calls.append(argv)
        return child

    monkeypatch.setattr(native, "spawn_native", spawn)
    assert rig._open() is child
    assert rig._open() is child
    assert len(calls) == 1
    assert rig._put("F 7100000\n")
    rig._close()
    assert not child.live and rig._proc is None
    assert "terminate" in child.events
    assert all(s.closed for s in (child.stdin, child.stdout, child.stderr))


def test_shrike_cleanup_refusal_is_a_failed_command_without_respawn(monkeypatch):
    from hfmodem.core import claim, native

    calls = []

    def spawn(*a, **kw):
        calls.append(a)
        raise claim.ClaimError("cleanup in progress")

    monkeypatch.setattr(native, "spawn_native", spawn)
    rig = fake_ota_rig()
    assert not rig._cmd("F", "7100000")
    assert not rig._cmd("F", "7100000")
    assert rig._sick and len(calls) == 1


def test_shrike_unresolved_child_is_retained(monkeypatch, tmp_path):
    from hfmodem.core import claim, native

    rig = fake_ota_rig()
    child = FakeNativeChild(tmp_path, stubborn=True)
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: child)
    try:
        rig._open()
        with pytest.raises(claim.ClaimError, match="termination unresolved"):
            rig._close()
        assert rig._proc is child and rig._sick
        assert not child.stdin.closed
        assert "terminate" in child.events and "kill" in child.events
    finally:
        child.live = False
        rig._close()


def test_shrike_unresolved_reap_is_a_failed_command_not_a_raise(monkeypatch, tmp_path):
    from hfmodem.core import claim, native

    rig = fake_ota_rig()
    child = FakeNativeChild(tmp_path)
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: child)
    rig._open()

    def unresolved():
        raise claim.ClaimError("termination unresolved for pid 123")

    child.alive = unresolved
    assert rig._open() is None
    assert rig._sick and rig._proc is child
    child.alive = lambda: False
    rig._close()


def test_shrike_stop_reports_a_cat_child_it_could_not_end(monkeypatch, tmp_path, capsys):
    from hfmodem.core import native

    rig = fake_ota_rig()
    rig._keyer = None
    child = FakeNativeChild(tmp_path, stubborn=True)
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: child)
    rig._open()
    rig.stop()
    assert "CAT child not ended at stop" in capsys.readouterr().out
    assert rig._proc is child
    child.live = False
    rig._close()


def test_shrike_cat_one_shots_carry_their_own_admission(monkeypatch):
    from hfmodem.core import claim, native

    rig = fake_ota_rig()
    calls = []

    class OneShot:
        returncode = 0

        def communicate(self, timeout):
            calls.append(timeout)
            return b"7100000\n", b""

    def spawn(argv, **kw):
        calls.append((argv[-1], kw))
        return SimpleNamespace(_process=OneShot())

    monkeypatch.setattr(native, "spawn_native", spawn)
    assert rig.get_freq() == "7100000"
    assert rig.set_power_once(0.5)
    assert calls == [("f", dict(claim_fd=9, own_session=False)), 8,
                     ("0.50", dict(claim_fd=9, own_session=True)), 8]

    def refused(argv, **kw):
        raise claim.ClaimError("cleanup in progress (pid 1)")

    monkeypatch.setattr(native, "spawn_native", refused)
    assert rig.get_freq() == "" and rig.get_power() == "" and not rig.set_power_once(0.5)


@pytest.mark.parametrize("survives", [False, True])
def test_shrike_broken_pipe_replaces_only_a_reaped_child(monkeypatch, tmp_path, survives):
    from hfmodem.core import native

    rig = fake_ota_rig()
    first = FakeNativeChild(tmp_path)
    second = FakeNativeChild(tmp_path)
    children = iter([first, second])
    monkeypatch.setattr(native, "spawn_native", lambda *a, **kw: next(children))
    rig._open()

    def put(line):
        if rig._proc is first:
            first.live = survives
            raise BrokenPipeError()
        return True

    monkeypatch.setattr(rig, "_put", put)
    if survives:
        assert not rig._cmd("F", "7100000")
        assert rig._proc is first and rig._sick and not first.stdin.closed
        first.live = False
    else:
        assert rig._cmd("F", "7100000")
        assert rig._proc is second and first.stdin.closed
    rig._close()


@pytest.mark.parametrize("failed_open", [None, "line", "audio"])
def test_lanner_acknowledges_both_devices(monkeypatch, failed_open):
    from hfmodem.core import audio, claim, config, devices, ptt
    from hfmodem.lanner import qso, terminal
    from hfmodem.station import qsolog

    events = []
    admission = FakeAdmission(events)
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: admission)
    monkeypatch.setattr(devices, "find_device", lambda *a, **kw: "fake")
    monkeypatch.setattr(terminal, "claim_tty", lambda: None)
    monkeypatch.setattr(qso.signal, "signal", lambda *a: None)

    def opened(kind):
        assert admission.held
        events.append(kind)
        if failed_open == kind:
            raise OSError(f"fake {kind} failed")

    monkeypatch.setattr(ptt, "LineKeyer", lambda *a, **kw: SimpleNamespace(
        arm=lambda: opened("line"), hand_back=lambda: events.append("line closed")))
    monkeypatch.setattr(audio, "StationAudio", lambda **kw: SimpleNamespace(
        open=lambda: opened("audio"), close=lambda: events.append("audio closed")))
    monkeypatch.setattr(qsolog, "QsoLog", lambda *a, **kw: None)
    monkeypatch.setattr(qso, "Session", lambda *a, **kw: SimpleNamespace(
        close=lambda: None, finish=lambda: []))

    def run(session):
        assert not admission.held
        events.append("terminal")
        return 0

    monkeypatch.setattr(terminal, "run", run)
    call = startup_call(monkeypatch, qso, "main")
    if failed_open:
        with pytest.raises(OSError, match=f"fake {failed_open} failed"):
            call()
        assert "opened" not in events
    else:
        assert call() == 0
        assert events[:4] == ["line", "audio", "opened", "terminal"]
    assert events[-1] == "failed"
    assert not admission.held


@pytest.mark.parametrize("failed_open", [None, "line", "playback"])
def test_sabir_acknowledges_line_capture_and_playback(monkeypatch, failed_open):
    from hfmodem.core import audio, claim
    from hfmodem.sabir import onair

    events = []
    admission = FakeAdmission(events)
    monkeypatch.setattr(claim, "inherited", lambda *a, **kw: admission)
    args = sabir_armed(monkeypatch, onair)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: "fake")
    monkeypatch.setattr(onair, "_install_signals", lambda rig: None)

    def opened(kind):
        assert admission.held
        events.append(kind)
        if failed_open == kind:
            raise OSError(f"fake {kind} failed")

    rig = SimpleNamespace(identify=lambda: "fake", transmitting=lambda: False,
                          qsy=lambda dial: dial, set_mode=lambda: None, mode="PKTUSB",
                          arm=lambda: opened("line"),
                          hand_back=lambda: events.append("line closed"))
    monkeypatch.setattr(onair.radio, "Rig", lambda *a, **kw: rig)

    def listen(*a, **kw):
        opened("capture")
        return SimpleNamespace(verdict="clear")

    def transmit(*a, **kw):
        assert not admission.held
        events.append("transmit")
        return 1.0

    monkeypatch.setattr(onair, "listen", listen)
    monkeypatch.setattr(audio, "warm_output", lambda *a: opened("playback"))
    monkeypatch.setattr(onair.radio, "transmit", transmit)
    if failed_open:
        with pytest.raises(OSError, match=f"fake {failed_open} failed"):
            onair.run(args)
        assert "opened" not in events
    else:
        assert onair.run(args) == 0
        assert events[:5] == ["capture", "line", "playback", "opened", "transmit"]
    assert events[-1] == "failed"


@pytest.mark.parametrize("failed_open", [False, True])
def test_shrike_preflight_acknowledges_duplex_open(monkeypatch, failed_open):
    import numpy as np
    from hfmodem.shrike import onair

    events = []
    admission = FakeAdmission(events)

    def live(in_dev, out_dev=None, **kw):
        assert admission.held
        kind = "capture" if out_dev is None else "duplex"
        events.append(kind)
        if failed_open and out_dev is not None:
            raise OSError("fake duplex failed")
        return SimpleNamespace(read_ready=lambda: np.zeros(10), xruns=0, lost=0,
                               close=lambda: events.append("close"))

    class Finished(Exception):
        pass

    def sleep(seconds):
        if "duplex" in events:
            assert not admission.held
            raise Finished()

    monkeypatch.setattr(onair, "_LiveInput", live)
    monkeypatch.setattr(onair.time, "sleep", sleep)
    with pytest.raises(OSError if failed_open else Finished):
        onair._preflight("fake-in", "fake-out", None, 0.1, key=False,
                          admission=admission)
    if failed_open:
        assert "opened" not in events
    else:
        assert events == ["capture", "close", "duplex", "opened", "close"]


@pytest.mark.parametrize("failure", [False, True])
def test_startup_owns_real_temp_admission_until_open_or_failure(monkeypatch, tmp_path,
                                                              capsys, failure):
    from hfmodem.core import claim
    from hfmodem.sabir import onair

    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path))
    for key in ("HFMODEM_COEXIST_LOG", "HFMODEM_COEXIST_BARRIERS", "HFMODEM_COEXIST_FAULT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(claim, "process_started", lambda pid: "fake-started")
    monkeypatch.setattr(claim, "_command", lambda pid: "fake-modem")
    monkeypatch.setattr(claim, "check_devices", lambda *a, **kw: ())
    descriptor = claim.acquire("test", "startup")
    inherited = claim.inherited
    monkeypatch.setattr(claim, "inherited", lambda fd, *, launch:
                        inherited(descriptor, launch=launch) if fd == 9 else pytest.fail("bad fd"))

    def probe():
        return os.open(claim.mutex_path(), os.O_RDWR)

    def keyed(*args):
        admission = args[-1]
        fd = probe()
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        if failure:
            raise OSError("fake open failed")
        admission.opened()
        fd = probe()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        return 0

    monkeypatch.setattr(onair, "_keyed", keyed)
    args = sabir_armed(monkeypatch, onair)
    try:
        if failure:
            with pytest.raises(OSError, match="fake open failed"):
                onair.run(args)
        else:
            assert onair.run(args) == 0
        fd = probe()
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        assert ("CLAIM ADMITTED" in capsys.readouterr().out) is not failure
    finally:
        claim.release(descriptor)


def test_station_entry_point_inventory_is_explicit():
    """Every hardware-capable entry point the stage 0 claim fences, named by site:
    a rename or removal has to come back through this list."""
    sites = ["P/shrike/onair.py:run", "tools/kestrel_connect.py:main",
             "P/lanner/qso.py:main", "P/besra/host/run_server.py:main",
             "P/sabir/onair.py:run", "tools/consolelib/sabir.py:execute",
             "P/shrike/ota.py:Rig._open", "P/shrike/ota.py:Rig._close",
             "P/besra/radio.py:Rig._write", "tools/consolelib/arm.py:bring_down",
             "P/core/claim.py:acquire",
             "packages/hfhost/hfhost/supervisor.py:ModemProcess.start",
             "packages/hfhost/hfhost/audio.py:CommandSource._spawn"]
    for site in sites:
        path, _, name = site.replace("P/", "packages/hfmodem/hfmodem/", 1).partition(":")
        scope = ast.parse((REPO / path).read_text()).body
        for part in name.split("."):
            found = [node for node in scope if getattr(node, "name", None) == part]
            assert found, f"{site}: no {part}"
            scope = found[0].body
    assert "\nprove_down() {" in (REPO / "tools/lib/rig.sh").read_text()
    published = {"onair_session", "rigaudio", "rx_levels", "corpus_watch", "hfcapture",
                 "txwitness", "vara_monitor"}
    private = {"alc_knee", "drive_sweep", "callback_jitter", "loss_reporting",
               "callback_trace", "ptt_tail_check"}
    for name in published:
        assert (REPO / "tools" / f"{name}.py").is_file()
    from hfmodem.tests.evidence import private_source_tree
    if private_source_tree(REPO):
        for name in private:
            assert (REPO / "tools" / f"{name}.py").is_file()
