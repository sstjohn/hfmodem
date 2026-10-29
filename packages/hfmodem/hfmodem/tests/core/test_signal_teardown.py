# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the keying paths in this package do when the process is killed.

`finally:` runs for a Ctrl-C and for nothing else. SIGTERM is what `timeout(1)`,
a supervisor recycle and a cancelled tool call send; SIGHUP is a dropped ssh
session or a closed terminal. Neither reaches a `finally`, neither runs `atexit`,
and the last thing under both — RTS dropping when the kernel closes the port — is
recorded as UNPROVEN on this hardware in `core.ptt`. So a preflight, an `hfmodem
rig --arm` or a station killed with the line up used to leave it up.

`shrike.onair` is the fourth, and it answers the signal differently: it raises
rather than panicking, because the rig it would panic with is a local of the
session loop and the unkey it needs is three `finally` blocks down. What it owes
is therefore not an unkey from the handler but a BOUND on that unwind, and that
is what is asked of it here.

Every subject here runs in a child process: the fix ends in `Rig.panic`, which
arms a deadman and calls `os._exit`, and that would take the test session with
it. The keying line the child is given writes every edge it is asked for to a
file, so the parent reads what the wire did rather than what the child said.

Nothing here opens a serial port, a card or a socket to anything but a fake
rigctld on an ephemeral loopback port.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import config, wav
from hfmodem.station.process import Station
from hfmodem.tests.core.fakerig import FakeRigctld
from hfmodem.tests.core.test_host_teardown import loopback_bind as loopback_bind

CONFIG = """\
schema = 1

[station]
mycall     = "N0CALL"
control    = "local"
regulatory = "unregulated"
because    = "unit test, no radio present"
transmit   = {transmit}
id_interval_s = 600

[rig]
model     = "ft891"
cat       = "{cat}"
centre_hz = 7101500
host      = "127.0.0.1"
port      = {port}

[rig.profiles.ft891]
cat_port = "fake-cat"
ptt_port = "{ptt}"

[rig.ptt]
port     = "{ptt}"
settle_s = 0.04

[audio]
input      = "no such device"
output     = "no such device"
input_gain = 0.040

[protocols.vara]
enabled = true
host    = "vara"
"""

#: How long the parent waits for an edge it expects on the keying line. Generous:
#: the child imports numpy and the package on the way to the first one.
_APPEAR_S = 30.0

#: How long a signalled child gets to put the line down and go. `PANIC_BUDGET_S`
#: is 2 s and the deadman fires on the same clock.
_DIE_S = 8.0


@pytest.fixture
def rigctld():
    """A daemon with no PTT of its own — the one configuration the arm gate accepts,
    because this station owns the keying line."""
    fake = FakeRigctld(ptt_type="None")
    yield fake
    fake.close()


@pytest.fixture
def station_file(tmp_path, request):
    in_process = request.node.name in {
        'test_a_dropped_terminal_brings_the_stations_radio_down',
        'test_the_signals_are_taken_over_before_a_client_can_ask_for_the_key',
        'test_a_lane_that_will_not_come_down_does_not_keep_the_radio_up'}
    port = 1 if in_process else request.getfixturevalue('rigctld').port
    def write(*, transmit: bool = True) -> Path:
        path = tmp_path / "station.toml"
        path.write_text(CONFIG.format(transmit=str(transmit).lower(),
                                      cat="yaesu" if in_process else "rigctld",
                                      port=port,
                                      ptt=tmp_path / "no-such-tty"),
                        encoding="utf-8")
        return path
    return write


@pytest.fixture(autouse=True)
def _no_handler_outlives_its_test(monkeypatch):
    """Signal disposition is process-global and belongs to no one test. sabir's
    orphaned SIGTERM handler once caught a signal meant for pytest twenty minutes
    later; the in-process subjects here take over the same three."""
    monkeypatch.setattr(Station, "fatal_shutdown", lambda self, why: None)
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = {s: signal.getsignal(s) for s in signals}
    yield
    for sig, handler in before.items():
        signal.signal(sig, handler)


# -- the child -----------------------------------------------------------------------

_PRELUDE = """
import sys, time
from pathlib import Path

# On Linux a default audio endpoint resolves through PortAudio; no device here.
from hfmodem.core import claim
claim.audio_nodes = lambda *a, **k: ()

marks, port, cfg_path = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

from hfmodem.core import ptt as pttmod


class MarkingPtt:
    '''A keying line whose every edge lands in a file the parent can read.'''

    port = "/dev/fake-keying-line"

    def __init__(self, *_a, **_kw):
        self.line = False
        self.released = False
        self.released_low = None

    def _mark(self, word):
        with marks.open("a") as fh:
            fh.write(word + "\\n")

    def open(self):
        self._mark("OPEN")

    def assert_(self, on):
        if self.released:
            raise pttmod.PttError("not open")
        self.line = on
        self._mark("KEYED" if on else "DOWN")
        if on:
            # Keyed, and this thread is not coming back on its own: a settle wait,
            # a stalled CAT readback and a slow codec all look like this from here.
            time.sleep(60)

    def sense(self):
        return None if self.released else self.line

    def release(self):
        self.released = True
        self.line = False
        self.released_low = True
        self._mark("DOWN")


pttmod.RtsPtt = MarkingPtt
"""

_PREFLIGHT = """
from hfmodem.core import config
from hfmodem.station import preflight
preflight.RtsPtt = MarkingPtt
def _no_card(name, kind):
    raise preflight.AudioError("this child has no card")
preflight._device = _no_card
raise SystemExit(preflight.run(config.load(cfg_path)))
"""

_ARM = """
from hfmodem import cli
raise SystemExit(cli.main(["rig", cfg_path, "--arm"]))
"""

_STATION = """
import builtins
original_import = builtins.__import__
def no_serial(name, *args, **kwargs):
    if name == 'serial' or name.startswith('serial.'):
        raise AssertionError('station child must not import a serial driver')
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_serial
from contextlib import nullcontext
from hfmodem.core import config
from hfmodem.station import hosts, process

class FakeCat:
    cache_free = reads_ptt = True
    def __init__(self, *args, **kwargs):
        assert args[0] == 'fake-cat', 'resolved a real CAT endpoint'
    def open(self):
        pass
    def close(self):
        pass
    def identify(self):
        return 'FT-891'
    def freq(self):
        return 7100000
    def ptt_state(self, **kw):
        return False
    def sole_ptt_owner(self, notes):
        return True
    def hold(self):
        return nullcontext(True)
    def emergency_unkey(self):
        return True

process.YaesuCat = process.RigctldCat = FakeCat
class CheckedLine(MarkingPtt):
    def open(self):
        assert type(st._driver) is FakeCat, 'a real CAT driver was built'
        self._mark('FAKE_DRIVER')
        super().open()
process.RtsPtt = CheckedLine
hosts.build = lambda cfg, station=None, log=print, **kw: {}
st = process.Station(config.load(cfg_path))
process.claim.acquire = lambda *a, **kw: None
st.preflight_reference = lambda: None
st.preflight_measure = lambda reference: None
st.open_audio = lambda: setattr(st, 'audio', type('Card', (), {'close': lambda self: None})())
raise SystemExit(st.run())
"""

#: `shrike.onair`'s handler with the unwind under it INTACT: the SystemExit reaches
#: a `finally` that drops the line, which is what `run` does with `rig.stop()`.
_ONAIR = """
from hfmodem.shrike import onair

onair._unkey_on_signal()
line = MarkingPtt()
line.open()
try:
    line.assert_(True)          # marks KEYED, and sleeps rather than returning
finally:
    line.release()
"""

#: What the wedged child's deadman is cut to, so the parent can hold the death to
#: that clock rather than to `_DIE_S`. The shipped budget is asserted against the
#: launcher's grace below.
_WEDGED_DEADMAN_S = 1.5

#: The same handler with the unwind WEDGED. Not a contrivance: the handler's own
#: docstring names the PortAudio teardown and the WAV write it raises into, and a
#: session's `finally` also waits on a capture writer that may be blocked on a disk
#: that has stopped answering. Nothing but the deadman can end this process.
_ONAIR_WEDGED = f"""
import time
from hfmodem.shrike import onair, ota

ota.DEADMAN_BUDGET_S = {_WEDGED_DEADMAN_S}
onair._unkey_on_signal()
line = MarkingPtt()
line.open()
try:
    line.assert_(True)
finally:
    time.sleep(300)
"""


class Child(subprocess.Popen):
    """One keying path, running for real, with its keying line on paper."""

    marks: Path
    log: Path

    def edges(self) -> list[str]:
        return self.marks.read_text().split() if self.marks.exists() else []

    def wait_for(self, edge: str, timeout: float = _APPEAR_S) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if edge in self.edges():
                return True
            if self.poll() is not None:
                return edge in self.edges()
            time.sleep(0.005)
        return False

    def output(self) -> str:
        return self.log.read_text() if self.log.exists() else ""


@pytest.fixture
def child(tmp_path, request):
    started: list[Child] = []

    def start(body: str = "", cfg: Path | None = None, *,
              argv: tuple[str, ...] = ()) -> Child:
        marks = tmp_path / f"keying-line-{len(started)}.txt"
        log = tmp_path / f"child-{len(started)}.log"
        port = 1 if body == _STATION else request.getfixturevalue('rigctld').port
        cmd = ([sys.executable, *argv] if argv else
               [sys.executable, "-c", _PRELUDE + body,
                str(marks), str(port), str(cfg)])
        with log.open("w") as out:
            state = tmp_path / f"state-{len(started)}"
            state.mkdir()
            proc = Child(cmd, stdout=out, stderr=subprocess.STDOUT,
                         env={**{k: v for k, v in os.environ.items()
                                 if k not in {'HFMODEM_RIG', 'CAT_PORT', 'PTT_PORT'}},
                              'HFMODEM_STATE_DIR': str(state)})
        proc.marks, proc.log = marks, log
        started.append(proc)
        return proc

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _dies_with_the_line_down(proc: Child, sig: signal.Signals, keyed: str) -> None:
    assert proc.wait_for(keyed), f"the child never got that far:\n{proc.output()}"
    proc.send_signal(sig)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{sig.name} did not end it:\n{proc.output()}") from None
    assert "DOWN" in proc.edges(), (
        f"{sig.name} left the keying line {proc.edges()[-1] if proc.edges() else 'unopened'}"
        f" — a stuck transmitter:\n{proc.output()}")
    # The panic path and not an ordinary unwind: only it retires the rig first, arms
    # a deadman before the unkey, and says on fd 2 whether the rig confirmed.
    assert "!! unkeying" in proc.output(), (
        f"the line came down without the panic path:\n{proc.output()}")


# -- preflight, which keys six times -------------------------------------------------

@pytest.mark.realtime
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_preflight_killed_mid_key_leaves_the_line_down(child, station_file, sig):
    """`preflight` keys for the arm gate's proof and five times after it, and its
    sole teardown was a `finally:` neither of these signals reaches."""
    _dies_with_the_line_down(child(_PREFLIGHT, station_file()), sig, "KEYED")


@pytest.mark.realtime
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_hfmodem_rig_arm_killed_mid_proof_leaves_the_line_down(child, station_file, sig):
    """`hfmodem rig FILE --arm` keys the same proof behind the same bare `finally:`."""
    _dies_with_the_line_down(child(_ARM, station_file()), sig, "KEYED")


@pytest.mark.realtime
def test_a_dropped_terminal_brings_the_stations_radio_down(child, station_file):
    """`Station` answered SIGINT and SIGTERM and not SIGHUP, so a closed terminal or
    a dropped ssh mid-over skipped `shutdown()` and `Rig.panic()` alike."""
    proc = child(_STATION, station_file())
    _dies_with_the_line_down(proc, signal.SIGHUP, "KEYED")
    assert "FAKE_DRIVER" in proc.edges()


# -- shrike.onair, whose handler raises into an unwind ---------------------------------

@pytest.mark.realtime
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_shrike_onair_still_unkeys_down_the_unwind_it_raises_into(child, station_file, sig):
    """The deadman is a backstop UNDER the SystemExit, not a replacement for it.

    The unkey this path is named for is the `finally` chain's, and it stays the
    thing that lowers the line: a budget short enough to kill the interpreter
    before `ota.Rig.stop` has walked its ladder would remove the last software
    path to a dead transmitter rather than provide one.
    """
    proc = child(_ONAIR, station_file())
    assert proc.wait_for("KEYED"), f"the child never keyed:\n{proc.output()}"
    proc.send_signal(sig)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{sig.name} did not end it:\n{proc.output()}") from None
    assert "DOWN" in proc.edges(), (
        f"{sig.name} left the keying line up:\n{proc.output()}")
    assert "unkeying and stopping" in proc.output(), (
        f"the signal did not become the exception the `finally` blocks expect:"
        f"\n{proc.output()}")


@pytest.mark.realtime
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_wedged_shrike_onair_teardown_is_taken_down_anyway(child, station_file, sig):
    """A session whose unwind never finishes used to run on holding the ports.

    This is the path that left a transmitter keyed on 10.1 MHz on 2026-08-04, and
    the only one of the four here that armed nothing: it trusted three layers of
    `finally` to run, on a process that had just been asked to die. A wedge in any
    of them and the SIGTERM is spent -- the operator's next ask is SIGKILL, which
    no handler survives and which lowers nothing on this hardware.

    SIGINT is here because it was the one this handler did not take. Ctrl-C was
    left to CPython, which raises `KeyboardInterrupt` and arms nothing, so the
    operator's own most likely ask was the one ask with no backstop under it --
    and it is the ask most likely to land on a session that is already unhappy.
    """
    proc = child(_ONAIR_WEDGED, station_file())
    assert proc.wait_for("KEYED"), f"the child never keyed:\n{proc.output()}"
    asked_at = time.monotonic()
    proc.send_signal(sig)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"{sig.name} left a wedged session running:\n{proc.output()}") from None
    assert proc.returncode != 0, "a deadman that fires exits non-zero"
    # On the deadman's clock and not somewhere near it: the wedge is a 300 s sleep,
    # so anything ending this later than the budget ended it for another reason.
    assert time.monotonic() - asked_at < 2 * _WEDGED_DEADMAN_S, (
        f"{sig.name} outlived the deadman it armed:\n{proc.output()}")


#: A whole session with no radio in it: the shipped `python -m` entry point, a
#: recording in place of the card, and no `--transmit`, so there is no rig object
#: and nothing that could key. What is watched is the OUTPUT -- the interrupt has
#: to leave the summary as the last thing on the screen.
_ONAIR_DRY = ("-m", "hfmodem.shrike.onair", "--replay-realtime",
              "--mycall", "N0CALL", "--dxcall", "N0CALL", "--dial", "7100000")


def _connecting(proc: Child) -> bool:
    end = time.monotonic() + _APPEAR_S
    while "connecting" not in proc.output() and time.monotonic() < end:
        if proc.poll() is not None:
            return False
        time.sleep(0.01)
    return "connecting" in proc.output()


@pytest.mark.realtime
def test_ctrl_c_on_a_healthy_session_still_leaves_the_verdict_on_the_screen(
        child, tmp_path):
    """The other half of the same handler, and the reason it cannot answer SIGINT
    the way it answers the other two.

    `run` unkeys and writes the summary on the way out, and the entry point then
    answers the interrupt with 130 instead of a traceback printed after the
    `finally` that scrolls the verdict away -- every slot report this project has
    is built out of that block, and the launchers read 130 as a hand stop.
    """
    replay = tmp_path / "silence.wav"
    wav.write(replay, np.zeros(30 * 48000, dtype=np.float32), 48000)
    proc = child(argv=_ONAIR_DRY + ("--replay", str(replay),
                                    "--outdir", str(tmp_path / "captures")))
    assert _connecting(proc), f"the session never started:\n{proc.output()}"

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"Ctrl-C did not end it:\n{proc.output()}") from None

    out = proc.output()
    assert proc.returncode == 130, (
        f"an interrupted run exits with the shell's status for Ctrl-C, and a launcher "
        f"reads it:\n{out}")
    assert "unkeying and stopping" not in out, (
        f"Ctrl-C came out as the other two signals' SystemExit:\n{out}")
    assert "session ended: the operator interrupted it" in out, (
        f"the interrupt did not reach the clause that writes the summary:\n{out}")
    assert out.rstrip().splitlines()[-1].startswith("verdict:"), (
        f"the verdict was not the last thing on the screen:\n{out}")


@pytest.mark.realtime
@pytest.mark.parametrize("sig,status", [(signal.SIGTERM, 143), (signal.SIGHUP, 129)])
def test_the_other_stop_signals_exit_with_the_launchers_status(
        child, tmp_path, sig, status):
    """`tools/onair.sh` traps TERM to 143 and HUP to 129; the modem says the same."""
    replay = tmp_path / "silence.wav"
    wav.write(replay, np.zeros(30 * 48000, dtype=np.float32), 48000)
    proc = child(argv=_ONAIR_DRY + ("--replay", str(replay),
                                    "--outdir", str(tmp_path / "captures")))
    assert _connecting(proc), f"the session never started:\n{proc.output()}"
    proc.send_signal(sig)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{sig.name} did not end it:\n{proc.output()}") from None
    out = proc.output()
    assert proc.returncode == status, f"{sig.name} came out as {proc.returncode}:\n{out}"
    assert "unkeying and stopping" in out and "verdict:" in out, out


#: A healthy session whose captures take longer to store than either deadman
#: budget: the store holds no device, and the verdict must still be reached.
_ONAIR_SLOW_STORE = """
import sys, time
from pathlib import Path
from hfmodem.shrike import onair, ota

ota.DEADMAN_BUDGET_S = onair.UNWIND_REST_S = 0.5
stored = Path(sys.argv[-1], "stored")


class SlowStore:
    pending, unstored = 0, []

    def drain(self, timeout):
        pass

    def store(self):
        time.sleep(1.5)
        stored.touch()

    def report(self):
        return "captures stored slowly"


onair._WRITER = SlowStore()
onair._unkey_on_signal()
sys.argv = ["onair", *sys.argv[1:-1]]
sys.exit(onair.main_refusing())
"""


@pytest.mark.realtime
def test_a_slow_capture_store_after_ctrl_c_still_reaches_the_verdict(child, tmp_path):
    replay = tmp_path / "silence.wav"
    wav.write(replay, np.zeros(30 * 48000, dtype=np.float32), 48000)
    proc = child(argv=("-c", _ONAIR_SLOW_STORE, *_ONAIR_DRY[2:], "--replay", str(replay),
                       "--outdir", str(tmp_path / "captures"), str(tmp_path)))
    assert _connecting(proc), f"the session never started:\n{proc.output()}"
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"Ctrl-C did not end it:\n{proc.output()}") from None
    out = proc.output()
    assert (tmp_path / "stored").exists(), f"the store never ran:\n{out}"
    assert proc.returncode == 130, f"came out as {proc.returncode}:\n{out}"
    assert out.rstrip().splitlines()[-1].startswith("verdict:"), out


#: A rigctl that says it started, then waits on a FIFO the test opens only after
#: the terminal's Ctrl-C has gone to the whole foreground process group.
_FAKE_RIGCTL = """#!{python}
import pathlib, sys
here = pathlib.Path(sys.argv[0]).parent
(here / "started").touch()
open(here / "go").close()
(here / "done").write_text(" ".join(sys.argv[1:]))
"""

#: A launch claim on fd 9, as a launcher hands every modem one, in the state
#: directory the test gives it.
_CLAIMED = """
import os
from hfmodem.core import claim
claim.audio_nodes = lambda *a, **k: ()
fd = os.open(claim.bootstrap()[0], os.O_RDWR)
os.dup2(fd, 9)
os.close(fd)
claim.acquire("test", "teardown", fd=9)
"""

_TEARDOWN_ONESHOT = {
    "power restore": _CLAIMED + """
import sys
from pathlib import Path
from hfmodem.shrike import onair, ota
onair._unkey_on_signal()
ota.find_rigctl = lambda *_a: Path(sys.argv[1])
rig = ota.Rig(1036, "/dev/null", 38400)
with ota.signals_held():
    print("restored" if rig.set_power_once(0.5) else "NOT restored", flush=True)
""",
}


@pytest.mark.realtime
@pytest.mark.parametrize("step,said", [("power restore", "restored")])
def test_a_terminal_ctrl_c_does_not_reach_the_teardowns_rigctl(tmp_path, step, said):
    rigctl = tmp_path / "rigctl"
    rigctl.write_text(_FAKE_RIGCTL.format(python=sys.executable))
    rigctl.chmod(0o755)
    os.mkfifo(tmp_path / "go")
    log = tmp_path / "harness.log"
    env = dict(os.environ, HFMODEM_STATE_DIR=str(tmp_path / "state"),
               HFMODEM_COEXIST_HARNESS="1")
    with log.open("w") as out:
        proc = subprocess.Popen([sys.executable, "-c", _TEARDOWN_ONESHOT[step], str(rigctl)],
                                stdout=out, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
    try:
        end = time.monotonic() + _APPEAR_S
        while not (tmp_path / "started").exists():
            assert time.monotonic() < end and proc.poll() is None, log.read_text()
            time.sleep(0.005)
        os.killpg(proc.pid, signal.SIGINT)
        try:
            os.close(os.open(tmp_path / "go", os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        proc.wait(timeout=_DIE_S)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert (tmp_path / "done").exists(), f"Ctrl-C ended the {step}:\n{log.read_text()}"
    assert said in log.read_text().splitlines(), log.read_text()


#: The same session with a drain that never returns: past the rig, the unwind
#: is still bounded, by its own budget.
_ONAIR_HUNG_DRAIN = """
import sys, time
from hfmodem.shrike import onair

onair.UNWIND_REST_S = 0.5
onair._drain_captures = lambda **_kw: time.sleep(300)
onair._unkey_on_signal()
sys.argv = ["onair", *sys.argv[1:]]
sys.exit(onair.main_refusing())
"""


@pytest.mark.realtime
def test_a_hung_drain_after_ctrl_c_is_still_taken_down(child, tmp_path):
    replay = tmp_path / "silence.wav"
    wav.write(replay, np.zeros(30 * 48000, dtype=np.float32), 48000)
    proc = child(argv=("-c", _ONAIR_HUNG_DRAIN, *_ONAIR_DRY[2:], "--replay", str(replay),
                       "--outdir", str(tmp_path / "captures")))
    assert _connecting(proc), f"the session never started:\n{proc.output()}"
    asked_at = time.monotonic()
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"a hung drain outlived the deadman:\n{proc.output()}") from None
    assert proc.returncode == 1, f"came out as {proc.returncode}:\n{proc.output()}"
    assert "Traceback" not in proc.output(), proc.output()
    assert time.monotonic() - asked_at < 4, proc.output()


@pytest.fixture
def timed_stop_handler(monkeypatch):
    """The stop handler over a real interval timer, with the deadman's kill
    swapped for nothing and every handler put back afterwards."""
    from hfmodem.shrike import onair, ota

    before = {sig: signal.getsignal(sig)
              for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)}
    monkeypatch.setattr(ota, "arm_deadman", lambda budget: signal.setitimer(
        signal.ITIMER_REAL, budget))
    monkeypatch.setattr(ota, "_depth", 0)
    monkeypatch.setattr(ota, "_pending", None)
    onair._unkey_on_signal()
    signal.signal(signal.SIGALRM, lambda *_: None)
    yield ota
    signal.setitimer(signal.ITIMER_REAL, 0)
    for sig, handler in before.items():
        signal.signal(sig, handler)


def test_a_held_signal_delivered_after_the_hold_keeps_the_bound_it_was_given(
        timed_stop_handler):
    ota = timed_stop_handler
    signal.setitimer(signal.ITIMER_REAL, 600)
    with pytest.raises(KeyboardInterrupt):
        with ota.signals_held(after_s=300):
            signal.raise_signal(signal.SIGINT)
    left = signal.getitimer(signal.ITIMER_REAL)[0]
    assert 290 < left <= 300, f"the deadman has {left:.1f} s, not the 300 s handed over"


def test_a_signal_the_hold_takes_still_arms_the_deadman(timed_stop_handler):
    ota = timed_stop_handler
    with pytest.raises(KeyboardInterrupt):
        with ota.signals_held(after_s=300):
            signal.raise_signal(signal.SIGINT)
            left = signal.getitimer(signal.ITIMER_REAL)[0]
    assert ota.DEADMAN_BUDGET_S - 1 < left <= ota.DEADMAN_BUDGET_S, (
        f"a held Ctrl-C left the ladder under a {left:.1f} s deadman")


@pytest.mark.parametrize("unwinding", [False, True])
def test_a_stop_after_the_teardown_is_answered_at_once(timed_stop_handler, unwinding):
    """Once the hold has delivered its stop, the next one is not held for a
    teardown that has already happened."""
    from hfmodem.shrike import onair

    ota = timed_stop_handler
    ota._pending = [] if unwinding else None
    with pytest.raises(KeyboardInterrupt):
        with ota.signals_held(after_s=300):
            signal.raise_signal(signal.SIGINT)
    with pytest.raises(onair.Signalled):
        signal.raise_signal(signal.SIGTERM)


#: A stop whose ladder has used up the deadman's budget but for one second, then
#: a power restore into a rigctl that never answers.
_HUNG_RESTORE = _CLAIMED + """
import signal, sys
from pathlib import Path
from hfmodem.shrike import onair, ota

onair._unkey_on_signal()
ota.DEADMAN_BUDGET_S = 1.0
ota.find_rigctl = lambda *_a: Path(sys.argv[1])
power = onair._PhasePower.__new__(onair._PhasePower)
power.rig, power.baseline, power.level = ota.Rig(1036, "/dev/null", 38400), 0.25, 0.5
power.scale, power.clamp = 100.0, None


def session():
    try:
        signal.raise_signal(int(sys.argv[2]))
    finally:
        with ota.signals_held(after_s=5):
            power.restore()


onair.main = session
sys.exit(onair.main_refusing())
"""


@pytest.mark.realtime
@pytest.mark.parametrize("sig,status", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_a_hung_power_restore_is_a_reported_failure_not_a_deadman(tmp_path, sig, status):
    rigctl = tmp_path / "rigctl"
    rigctl.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    rigctl.chmod(0o755)
    env = dict(os.environ, HFMODEM_STATE_DIR=str(tmp_path / "state"),
               HFMODEM_COEXIST_HARNESS="1")
    done = subprocess.run([sys.executable, "-c", _HUNG_RESTORE, str(rigctl), str(int(sig))],
                          capture_output=True, text=True, env=env, timeout=_DIE_S)
    said = done.stdout + done.stderr
    assert done.returncode == status, f"came out as {done.returncode}:\n{said}"
    assert "RFPOWER NOT RESTORED" in said, said


#: A stop that never reached the teardown, with captures still queued: the drain
#: `atexit` runs knows nothing of the rig, so whatever deadman is armed stays.
_ATEXIT_STORE = """
import atexit, signal, sys, time
from hfmodem.shrike import onair, ota

ota.DEADMAN_BUDGET_S = 1.0


class Queued:
    pending, unstored = 1, ["capture"]

    def drain(self, timeout):
        self.pending = 0

    def store(self):
        time.sleep(5)
        print("store finished", flush=True)

    def report(self):
        return "captures stored"


onair._WRITER = Queued()
atexit.register(onair._drain_at_exit)
onair._unkey_on_signal()
signal.raise_signal(signal.SIGINT)
"""


@pytest.mark.realtime
def test_the_exit_time_drain_stays_under_the_armed_deadman(tmp_path):
    env = dict(os.environ, HFMODEM_STATE_DIR=str(tmp_path / "state"),
               HFMODEM_COEXIST_HARNESS="1")
    began = time.monotonic()
    done = subprocess.run([sys.executable, "-c", _ATEXIT_STORE], capture_output=True,
                          text=True, env=env, timeout=_DIE_S)
    said = done.stdout + done.stderr
    assert "store finished" not in said and done.returncode == 1, (
        f"the exit-time store outlived the deadman ({done.returncode}):\n{said}")
    assert time.monotonic() - began < 4, said


def test_the_deadman_fires_before_the_launcher_reaches_for_sigkill():
    """The budget is bounded ABOVE by what reaps this modem. `tools/onair.sh` ends
    its children through the claim, which gives each `REAP_GRACE_S` to unkey itself
    and then sends SIGKILL, ending the unwind, the handler and the deadman together
    -- so a budget at or past that grace is a deadman that never fires."""
    from hfmodem.core import native
    from hfmodem.shrike import ota

    ports = Path(__file__).resolve().parents[4].parent / "tools" / "lib" / "ports.sh"
    if not ports.exists():
        pytest.skip(f"{ports} is not present (installed-wheel run)")
    assert "from hfmodem.core.native import NativeChild, REAP_GRACE_S" in ports.read_text(), (
        "the launcher no longer reaps with the claim's grace")
    assert 0 < ota.DEADMAN_BUDGET_S < native.REAP_GRACE_S, (
        f"deadman {ota.DEADMAN_BUDGET_S} s against a {native.REAP_GRACE_S} s grace")


def test_the_deadman_outlasts_the_slowest_unkey_ladder(monkeypatch):
    """`Rig.stop` with every rung at its worst, on a clock that only the ladder
    moves: the keying line is dropped and handed back through the real
    `LineKeyer`, rigctl's stdin is then a full pipe that drains only at the real
    `_put`'s own deadline for the quit, and rigctl neither quits, ends nor dies
    inside `_close`'s waits. Each rung is held to the constant the deadman's
    budget is summed from, and the whole ladder to the budget, which ends a
    wedged unwind no more than a quarter second after it."""
    from types import SimpleNamespace

    from hfmodem.core import ptt
    from hfmodem.shrike import ota

    now = [0.0]

    def sleep(seconds):
        now[0] += max(0.0, seconds)

    def drained_at_the_deadline(_r, w, _x, timeout):
        sleep(timeout)
        os.read(read_end, 1 << 20)
        return [], w, []

    def fill():
        try:
            while True:
                os.write(write_end, bytes(65536))
        except BlockingIOError:
            pass

    clock = SimpleNamespace(monotonic=lambda: now[0], sleep=sleep, time=time.time)
    monkeypatch.setattr(ota, "time", clock)
    monkeypatch.setattr(ota, "select", SimpleNamespace(select=drained_at_the_deadline))

    read_end, write_end = os.pipe()
    os.set_blocking(write_end, False)
    rungs, waits = [], []

    class Line:
        released_low = None

        def __init__(self, port):
            self.high = False

        def open(self):
            pass

        def assert_(self, on):
            rungs.append(f"line {'up' if on else 'down'}")
            self.high = on

        def sense(self):
            return self.high

        def release(self):
            rungs.append("hand back")
            self.released_low = not self.high

    class Stuck:
        pid = 4242
        stdin = SimpleNamespace(fileno=lambda: write_end)
        stdout = stderr = None

        def alive(self):
            return True

        def wait_gone(self, timeout):
            waits.append(timeout)
            sleep(timeout)
            return False

        def terminate(self):
            pass

        def kill(self):
            pass

    armed = []
    monkeypatch.setattr(ptt, "RtsPtt", Line)
    monkeypatch.setattr(ota, "arm_deadman", lambda budget: armed.append((now[0], budget)))
    monkeypatch.setattr(ota, "find_rigctl", lambda *_a: Path("rigctl"))
    rig = ota.Rig(1036, "/dev/null", 38400, ptt_port="/dev/null")
    assert rig.ptt(True), rungs
    rig._proc, put, writes = Stuck(), rig._put, []

    def timed_put(line):
        fill()
        began = now[0]
        try:
            return put(line)
        finally:
            writes.append(now[0] - began)

    rig._put = timed_put
    try:
        ota.arm_unwind_deadman()
        rig.stop()
    finally:
        os.close(read_end)
        os.close(write_end)

    (began, budget), = armed
    took = now[0] - began
    assert rungs == ["line up", "line down", "hand back"], rungs
    assert rig._keyer.released_low is True
    assert writes == [ota.WRITE_TIMEOUT_S], (
        f"pipe writes took {writes}, against {ota.WRITE_TIMEOUT_S} s")
    assert waits == [ota.QUIT_WAIT_S, ota.TERM_WAIT_S, ota.KILL_WAIT_S], f"_close waited {waits}"
    assert took == ota.LADDER_WORST_S < budget <= took + 0.25, (
        f"the ladder took {took} s against a {budget} s deadman")


@pytest.mark.realtime
def test_a_real_ladder_spawns_nothing_and_ends_inside_the_deadman(monkeypatch):
    """The same ladder against a real rigctl stand-in that will not read its
    quit, end or die: each wait polls the child, and nothing else in the ladder
    may spend time the budget does not count."""
    from hfmodem.core import claim, native
    from hfmodem.shrike import ota

    stuck = subprocess.Popen([sys.executable, "-c", "import signal, time; "
                              "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"],
                             stdin=subprocess.PIPE)
    child = native.NativeChild(stuck.pid, claim.process_started(stuck.pid), stuck.stdin,
                               None, None, stuck)
    signal_child = native.NativeChild._signal
    monkeypatch.setattr(native.NativeChild, "_signal", lambda self, sig: signal_child(
        self, signal.SIGSTOP if sig == signal.SIGKILL else sig))
    os.set_blocking(stuck.stdin.fileno(), False)
    try:
        while True:
            os.write(stuck.stdin.fileno(), bytes(65536))
    except BlockingIOError:
        pass
    monkeypatch.setattr(ota, "find_rigctl", lambda *_a: Path("rigctl"))
    rig = ota.Rig(1036, "/dev/null", 38400)
    rig._proc = child
    spawned, run = [], subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: spawned.append(argv) or run(argv, **kw))
    try:
        began = time.monotonic()
        rig.stop()
        took = time.monotonic() - began
    finally:
        monkeypatch.undo()
        stuck.kill()
        os.kill(stuck.pid, signal.SIGCONT)
        stuck.wait()
    assert not spawned, f"the ladder ran {len(spawned)} processes: {spawned[:2]}..."
    assert ota.LADDER_WORST_S <= took < ota.DEADMAN_BUDGET_S, (
        f"the ladder took {took:.3f} s: constants {ota.LADDER_WORST_S} s, deadman {ota.DEADMAN_BUDGET_S} s")


def test_an_unkey_that_raises_still_hands_the_line_back_and_ends_rigctl(monkeypatch):
    from hfmodem.shrike import ota

    monkeypatch.setattr(ota, "find_rigctl", lambda *_a: Path("rigctl"))
    rig = ota.Rig(1036, "/dev/null", 38400)
    rungs = []

    class Keyer:
        def unkey(self):
            rungs.append("unkey")
            raise OSError("the line would not drop")

        def hand_back(self):
            rungs.append("hand back")

    rig._keyer = Keyer()
    monkeypatch.setattr(rig, "_close", lambda: rungs.append("close"))
    with pytest.raises(OSError, match="would not drop"):
        rig.stop()
    assert rungs == ["unkey", "hand back", "close"]


def test_only_the_power_restore_runs_in_a_session_of_its_own(monkeypatch):
    """The restore runs after a Ctrl-C, which must not reach it. A readback runs
    while the operator may still want a Ctrl-C to end it."""
    from types import SimpleNamespace

    from hfmodem.core import native
    from hfmodem.shrike import ota

    sessions = []

    class Answered:
        returncode = 0

        def communicate(self, timeout):
            return b"7101500\n", b""

    def spawn(argv, *, claim_fd, own_session=False):
        sessions.append((argv[9], own_session))
        return SimpleNamespace(_process=Answered())

    monkeypatch.setattr(native, "spawn_native", spawn)
    monkeypatch.setattr(ota, "find_rigctl", lambda *_a: Path("rigctl"))
    rig = ota.Rig(1036, "/dev/null", 38400)
    rig.get_freq()
    rig.get_power()
    assert rig.set_power_once(0.5)
    assert sessions == [("f", False), ("l", False), ("L", True)]


def test_a_later_signal_does_not_move_an_armed_deadman_later(monkeypatch):
    from hfmodem.shrike import ota

    armed = []
    monkeypatch.setattr(ota, "arm_deadman", armed.append)
    alarm = signal.signal(signal.SIGALRM, lambda *_: None)
    try:
        signal.setitimer(signal.ITIMER_REAL, 600)
        ota.arm_unwind_deadman()
        assert not armed
        signal.setitimer(signal.ITIMER_REAL, 0)
        ota.arm_unwind_deadman()
        assert armed == [ota.DEADMAN_BUDGET_S]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, alarm)


# -- the station, in process ---------------------------------------------------------

def _station(station_file, **kw) -> Station:
    return Station(config.load(station_file(**kw)),
                   replay=np.zeros(4800, dtype=np.float32))


def test_the_signals_are_taken_over_before_a_client_can_ask_for_the_key(
        monkeypatch, station_file):
    """The handlers went in after `hosts.build`, which is after the port is bound:
    in that window a client can attach and drive the arbiter to key with SIGTERM
    still at SIG_DFL."""
    from hfmodem.station import hosts

    seen: dict[str, object] = {}

    def build(cfg, *, station=None, log=print, **kw):
        seen.update({s.name: signal.getsignal(s)
                     for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)})
        return {}

    monkeypatch.setattr(hosts, "build", build)
    _station(station_file, transmit=False).run(on_ready=lambda st: st._stop.set())

    assert seen, "hosts.build was never reached"
    unowned = [name for name, handler in seen.items()
               if not getattr(handler, "__qualname__", "").startswith("Station.")]
    assert not unowned, f"{', '.join(unowned)} was still the default when the port opened"


def test_a_lane_that_will_not_come_down_does_not_keep_the_radio_up(capsys, station_file):
    """The lane loop sat unguarded in front of `rig.close()`, so an adapter raising
    from `stop()` — sabir's `_cmds.put`, or its 10 s join — skipped the radio and the
    card both. Report the dialect after the radio has come down, as documented."""
    class Stubborn:
        def stop(self):
            raise RuntimeError("adapter thread will not come down")

    class Lane:
        stopped = False

        def stop(self):
            self.stopped = True

    class Closable:
        closed = False

        def close(self):
            self.closed = True

    st = _station(station_file, transmit=False)
    st.lanes = {"sabir": Stubborn(), "vara": Lane()}
    st.rig, st.audio = Closable(), Closable()

    st.shutdown()

    assert st.rig.closed, "a stuck lane left the radio open"
    assert st.audio.closed, "a stuck lane left the card open"
    assert st.lanes["vara"].stopped, "one lane's failure stopped the others coming down"
    assert "sabir" in capsys.readouterr().err


# -- a panic already running, and a signal on top of it -------------------------------

#: A rig up and keyed under the guard, with a keying line that does not come down
#: at once. `RtsPtt.release()` is an ioctl, a readback and a close on a USB serial
#: device, and a dongle that has stopped answering blocks all three in the kernel —
#: which is the window every test in this section needs a signal to land inside.
#: `{watchdog}` is what the operator's Ctrl-C arrives on top of.
_KEYED_RIG = """
from hfmodem.core import config
from hfmodem.core.rig import Cat, Rig, unkey_on_signal


class SlowLine(MarkingPtt):
    def release(self):
        self._mark("RELEASING")
        time.sleep({release_s})
        super().release()


cfg = config.load(cfg_path)
rig = Rig(model=cfg.rig.model, cat=Cat(cfg.rig.host, cfg.rig.port), ptt=SlowLine(),
          profile=cfg.profile, control=cfg.control, mycall=cfg.station.mycall,
          transmit=cfg.station.transmit, max_key_s=cfg.rig.max_key_s)
with unkey_on_signal(lambda: rig):
    rig.ptt.open()
    {watchdog}
    rig.ptt.assert_(True)       # marks KEYED, and sleeps rather than returning
"""


def _the_ask_and_not_the_deadman_ends_it(proc: Child, asked_at: float) -> None:
    """Non-zero, and soon enough that it was this program that answered.

    The deadman fires on `PANIC_BUDGET_S` and exits non-zero itself, so a bare
    `returncode != 0` a few seconds later would pass just as well for a handler
    that ignored the signal outright.
    """
    from hfmodem.core.rig import PANIC_BUDGET_S

    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"the signal did not end it:\n{proc.output()}") from None
    assert proc.returncode != 0, (
        f"exit 0 with the unkey unconfirmed and the line last seen up: that is what "
        f"a supervisor restarts into:\n{proc.output()}")
    assert time.monotonic() - asked_at < PANIC_BUDGET_S, (
        f"the deadman ended this, not the ask — so nothing here would notice a "
        f"handler that answered the signal with a silent return:\n{proc.output()}")


@pytest.mark.realtime
def test_asking_twice_during_a_panic_is_not_a_clean_stop(child, station_file, rigctld):
    """The second Ctrl-C re-entered `panic()` while the first was still on the
    ladder, found nothing left to do, and exited 0 — with the release not yet at
    its `_clear()`, so the line was still asserted and only HUPCL under it.

    A wedged daemon because that is when this happens: the ladder is longest
    exactly when rigctld has stopped answering, which is the moment an operator
    reaches for Ctrl-C a second time.
    """
    rigctld.wedge = True
    proc = child(_KEYED_RIG.format(release_s=3.0, watchdog="pass"), station_file())
    assert proc.wait_for("KEYED"), f"the child never keyed:\n{proc.output()}"

    proc.send_signal(signal.SIGINT)
    assert proc.wait_for("RELEASING", timeout=_DIE_S), (
        f"the first panic never reached the keying line:\n{proc.output()}")
    proc.send_signal(signal.SIGINT)
    _the_ask_and_not_the_deadman_ends_it(proc, time.monotonic())


@pytest.mark.realtime
def test_one_signal_on_top_of_a_watchdog_panic_is_not_a_clean_stop(
        child, station_file, rigctld):
    """The worse form, and it needs no second ask at all.

    The watchdog firing IS the emergency — the transmitter has been up past its
    budget — and the operator's single Ctrl-C landed inside the panic it started.
    The rig's own `key()` arms the watchdog exactly as it is armed here.
    """
    rigctld.wedge = True
    proc = child(
        _KEYED_RIG.format(
            release_s=3.0,
            watchdog='rig._arm_watchdog(0.2, "a burst that never ended")'),
        station_file())
    assert proc.wait_for("KEYED"), f"the child never keyed:\n{proc.output()}"
    assert proc.wait_for("RELEASING", timeout=_DIE_S), (
        f"the watchdog never fired:\n{proc.output()}")

    proc.send_signal(signal.SIGINT)
    _the_ask_and_not_the_deadman_ends_it(proc, time.monotonic())


# -- the watchdog's panic, which has no deadman under it ------------------------------

def test_the_keying_lines_budget_leaves_the_panic_time_to_reach_cat():
    """`RELEASE_BUDGET_S` is bounded above by the deadman it runs underneath.

    On the signal path `panic()` arms the deadman for `PANIC_BUDGET_S` and then
    calls the ladder. A release budget at or near that spends the whole allowance
    on the leg that is stuck, so the process is killed with the independent CAT
    leg never attempted — which is the same failure this bound exists to end,
    reached from the other side.
    """
    from hfmodem.core.rig import PANIC_BUDGET_S, RELEASE_BUDGET_S

    assert 0 < RELEASE_BUDGET_S < PANIC_BUDGET_S / 2, (
        f"a {RELEASE_BUDGET_S} s release budget under a {PANIC_BUDGET_S} s deadman")


@pytest.mark.realtime
def test_a_watchdog_panic_goes_on_to_cat_when_the_keying_line_will_not_answer(
        child, station_file, rigctld):
    """The one panic with no deadman under it, and the one that fires BECAUSE the
    transmitter has already been up too long.

    `RtsPtt.release()` is an ioctl, a readback and a close on a USB serial adapter,
    and an adapter that has stopped answering blocks all three in the kernel with
    the line still asserted. `PANIC_BUDGET_S` bounds the CAT ladder and nothing
    bounded the release standing in front of it, so the panic never arrived at the
    one leg that does not go through the keying line at all. The signal path at
    least has the deadman to end it; a watchdog panic is not a request to end the
    process, so under this one there was nothing.

    The daemon here can answer `t`, which the arm gate would refuse and which is
    the point: with the keying line's own readback unreachable, CAT is the only
    witness left, and a panic that cannot reach it confirms nothing.
    """
    from hfmodem.core.rig import PANIC_BUDGET_S

    rigctld.ptt_type = "RIG"
    proc = child(
        _KEYED_RIG.format(
            release_s=300,
            watchdog='rig._arm_watchdog(0.2, "a burst that never ended")'),
        station_file())
    assert proc.wait_for("RELEASING"), f"the watchdog never fired:\n{proc.output()}"

    # Sharply: the whole ladder is meant to fit inside `PANIC_BUDGET_S`, and a
    # looser deadline here would pass just as well for a release that eventually
    # unwedged on its own — which is the case a stuck adapter never gives us.
    end = time.monotonic() + PANIC_BUDGET_S
    while "confirmed down" not in proc.output() and time.monotonic() < end:
        time.sleep(0.01)
    assert "confirmed down" in proc.output(), (
        f"the panic is still waiting on a keying line that will not answer, with "
        f"the transmitter up and CAT untried:\n{proc.output()}")
    assert "+T 0" in rigctld.log, (
        f"nothing reached the independent path:\n{proc.output()}")
    assert rigctld.log.index("+\\set_cache 0") < rigctld.log.index("+T 0") < rigctld.log.index("+t"), (
        f"CAT confirmation must follow an uncached unkey/readback: {rigctld.log}")
    assert proc.poll() is None, (
        f"the watchdog took the process down. A panic that confirmed the "
        f"transmitter down is not an ask to end the run, and the deadman that "
        f"would end it is the signal path's:\n{proc.output()}")


# -- preflight, whose report must not eat the signal ----------------------------------

#: preflight with a card that resolves and a reference dwell long enough to be
#: interrupted, which is the state the real one is in for two seconds of every run.
#: The duplex stream is refused rather than opened: nothing in this file may touch
#: a sound device, and what is being watched is the six keyings AFTER the dwell.
_PREFLIGHT_DWELL = """
from hfmodem.core import config, devices
from hfmodem.core.audio import AudioError
from hfmodem.station import preflight


def dwell(*_a, **_kw):
    with marks.open("a") as fh:
        fh.write("LISTENING\\n")
    time.sleep(60)


class NoCard:
    def __init__(self, **_kw):
        raise AudioError("no duplex stream is opened in this test")


devices.find_device = lambda *_a, **_kw: 0
preflight.RtsPtt = MarkingPtt
preflight.StationAudio = NoCard
preflight._listen = dwell
raise SystemExit(preflight.run(config.load(cfg_path)))
"""


@pytest.mark.realtime
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_signal_during_preflight_is_not_a_failed_measurement(child, station_file, sig):
    """Every phase before the rig runs with `self.rig` still None, so the guard's
    answer to a signal there is `SystemExit(128 + signum)` and nothing else. Three
    clauses in `preflight` name `SystemExit` — they were widened for `find_device`,
    which refuses that way — so all three signals were reported as a card that
    would not open, and the run went on to the arm gate's PTT proof and five more
    keyings.
    """
    proc = child(_PREFLIGHT_DWELL, station_file())
    assert proc.wait_for("LISTENING"), f"the dwell never began:\n{proc.output()}"
    proc.send_signal(sig)

    assert not proc.wait_for("KEYED", timeout=_DIE_S), (
        f"{sig.name} was read as a hardware fault and preflight keyed anyway:"
        f"\n{proc.output()}")
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{sig.name} did not end it:\n{proc.output()}") from None
    assert proc.returncode == 128 + sig, (
        f"{sig.name} came out as {proc.returncode}, which is a preflight verdict "
        f"rather than a signal:\n{proc.output()}")


def test_a_ctrl_c_in_preflights_keyed_window_unkeys_before_the_card_closes(monkeypatch):
    """Closing a stream waits on the driver, so the line comes down first."""
    from types import SimpleNamespace

    from hfmodem.shrike import onair

    events = []

    class Live:
        _lat = _blk = underruns = xruns = lost = pos = 0
        fs = 48000

        def __init__(self, in_dev, out_dev=None, **_):
            self.name = "capture-only" if out_dev is None else "duplex"

        def read_ready(self):
            return np.full(4800, 0.01, dtype=np.float32)

        def sample_now(self):
            return 0

        def close(self):
            events.append(f"{self.name} closed")

    class Rig:
        ptt_port = serial = "/dev/null"
        ptt_type = "LINE"
        up = False

        def ptt(self, on):
            self.up = on
            return True

        def stop(self):
            events.append("stopped keyed" if self.up else "stopped")
            self.up = False

    def sleep(s):
        if s == 0.25:
            raise KeyboardInterrupt

    monkeypatch.setattr(onair, "_LiveInput", Live)
    monkeypatch.setattr(onair, "time", SimpleNamespace(sleep=sleep, monotonic=time.monotonic,
                                                       perf_counter=time.perf_counter))
    with pytest.raises(KeyboardInterrupt):
        onair._preflight("in", "out", Rig(), 0.04, key=True)
    assert events == ["capture-only closed", "stopped keyed", "duplex closed"]
