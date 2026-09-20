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

import re
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
centre_hz = 7101500
host      = "127.0.0.1"
port      = {port}

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
def station_file(tmp_path, rigctld):
    def write(*, transmit: bool = True) -> Path:
        path = tmp_path / "station.toml"
        path.write_text(CONFIG.format(transmit=str(transmit).lower(),
                                      port=rigctld.port,
                                      ptt=tmp_path / "no-such-tty"),
                        encoding="utf-8")
        return path
    return write


@pytest.fixture(autouse=True)
def _no_handler_outlives_its_test():
    """Signal disposition is process-global and belongs to no one test. sabir's
    orphaned SIGTERM handler once caught a signal meant for pytest twenty minutes
    later; the in-process subjects here take over the same three."""
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = {s: signal.getsignal(s) for s in signals}
    yield
    for sig, handler in before.items():
        signal.signal(sig, handler)


# -- the child -----------------------------------------------------------------------

_PRELUDE = """
import sys, time
from pathlib import Path

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
raise SystemExit(preflight.run(config.load(cfg_path)))
"""

_ARM = """
from hfmodem import cli
raise SystemExit(cli.main(["rig", cfg_path, "--arm"]))
"""

_STATION = """
import numpy as np
from hfmodem.core import config
from hfmodem.station import hosts, process
process.RtsPtt = MarkingPtt
hosts.build = lambda cfg, station=None, log=print: {}
st = process.Station(config.load(cfg_path), replay=np.zeros(48000 * 120, np.float32))
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
#: that clock rather than to `_DIE_S`. The shipped 10 s is asserted against the
#: launcher's grace below.
_WEDGED_DEADMAN_S = 1.5

#: The same handler with the unwind WEDGED. Not a contrivance: the handler's own
#: docstring names the PortAudio teardown and the WAV write it raises into, and a
#: session's `finally` also waits on a capture writer that may be blocked on a disk
#: that has stopped answering. Nothing but the deadman can end this process.
_ONAIR_WEDGED = f"""
import time
from hfmodem.shrike import onair

onair.DEADMAN_BUDGET_S = {_WEDGED_DEADMAN_S}
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
def child(tmp_path, rigctld):
    started: list[Child] = []

    def start(body: str = "", cfg: Path | None = None, *,
              argv: tuple[str, ...] = ()) -> Child:
        marks = tmp_path / f"keying-line-{len(started)}.txt"
        log = tmp_path / f"child-{len(started)}.log"
        cmd = ([sys.executable, *argv] if argv else
               [sys.executable, "-c", _PRELUDE + body,
                str(marks), str(rigctld.port), str(cfg)])
        with log.open("w") as out:
            proc = Child(cmd, stdout=out, stderr=subprocess.STDOUT)
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
    _dies_with_the_line_down(child(_STATION, station_file()), signal.SIGHUP, "OPEN")


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


@pytest.mark.realtime
def test_ctrl_c_on_a_healthy_session_still_leaves_the_verdict_on_the_screen(
        child, tmp_path):
    """The other half of the same handler, and the reason it cannot answer SIGINT
    the way it answers the other two.

    `run` catches `KeyboardInterrupt` on purpose, so that an operator who stops a
    run early gets the summary last instead of a traceback printed after the
    `finally` that scrolls the verdict away -- every slot report this project has
    is built out of that block. `SystemExit` is not caught by that clause, so a
    handler that raised one here would buy the backstop above by spending the one
    thing the operator was reading, and would change the exit status under
    `tools/onair.sh` besides.
    """
    replay = tmp_path / "silence.wav"
    wav.write(replay, np.zeros(30 * 48000, dtype=np.float32), 48000)
    proc = child(argv=_ONAIR_DRY + ("--replay", str(replay),
                                    "--outdir", str(tmp_path / "captures")))
    end = time.monotonic() + _APPEAR_S
    while "connecting" not in proc.output() and time.monotonic() < end:
        assert proc.poll() is None, f"the session never started:\n{proc.output()}"
        time.sleep(0.01)
    assert "connecting" in proc.output(), f"the session never started:\n{proc.output()}"

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=_DIE_S)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"Ctrl-C did not end it:\n{proc.output()}") from None

    out = proc.output()
    assert proc.returncode == 0, (
        f"an interrupted run exits with the session's own status, and a launcher "
        f"reads it:\n{out}")
    assert "unkeying and stopping" not in out, (
        f"Ctrl-C came out as the other two signals' SystemExit, which `run` does "
        f"not catch:\n{out}")
    assert "session ended: the operator interrupted it" in out, (
        f"the interrupt did not reach the clause that writes the summary:\n{out}")
    assert out.rstrip().splitlines()[-1].startswith("verdict:"), (
        f"the verdict was not the last thing on the screen:\n{out}")


def test_the_deadman_fires_before_the_launcher_reaches_for_sigkill():
    """The budget is bounded ABOVE by what reaps this modem. `tools/onair.sh` gives
    an orphan `REAP_GRACE_S` to unkey itself and then sends SIGKILL, which ends the
    unwind, the handler and the deadman together -- so a budget at or past that
    grace is a deadman that never fires."""
    from hfmodem.shrike import onair

    launcher = Path(__file__).resolve().parents[4].parent / "tools" / "onair.sh"
    if not launcher.exists():
        pytest.skip(f"{launcher} is not present (installed-wheel run)")
    grace = re.search(r"^REAP_GRACE_S=(\d+)", launcher.read_text(), re.M)
    assert grace, "the launcher no longer states the grace it gives an orphan"
    assert 0 < onair.DEADMAN_BUDGET_S < int(grace.group(1)), (
        f"deadman {onair.DEADMAN_BUDGET_S} s against a {grace.group(1)} s grace")


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

    def build(cfg, *, station=None, log=print):
        seen.update({s.name: signal.getsignal(s)
                     for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)})
        station._stop.set()
        return {}

    monkeypatch.setattr(hosts, "build", build)
    _station(station_file, transmit=False).run()

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
    assert "T 0" in rigctld.log, (
        f"nothing reached the independent path:\n{proc.output()}")
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
