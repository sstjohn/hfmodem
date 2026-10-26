# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""beacon drives the same rig as onair, so it has to speak the same rig.

beacon keys a transmitter and plays into a sound card, so nothing real-hardware
runs here. The join is checked structurally: every method beacon calls on its
rig must exist on ``ota.Rig``. That is not a proxy for a real test -- it is the
exact defect that shipped. ``rig.start()`` was called on a class that has never
had a ``start``, so ``beacon.main`` raised AttributeError before it could key
anything, and no test noticed because no test imports beacon at all.

``main`` itself is driven whole below, with every hardware seam faked at
beacon's own imports, because the next two defects were behavioural: a missing
keying line surfaced as a traceback where ota converts it to a sentence, and a
rigctl that died mid-run let the remaining keyings count themselves as sent.
"""
from __future__ import annotations

import ast
import contextlib
import dis
import inspect
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import audio
from hfmodem.core.ptt import PttError
from hfmodem.core.rig import FATAL_SIGNALS
from hfmodem.shrike import beacon, ota

SOURCE = Path(inspect.getfile(beacon))


def _rig_calls(tree: ast.AST) -> set[str]:
    """Names invoked as ``rig.<name>(...)`` anywhere in the module."""
    return {n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "rig"}


def test_every_rig_method_beacon_calls_exists():
    called = _rig_calls(ast.parse(SOURCE.read_text()))
    assert called, "no rig.<method>() calls found -- has beacon been rewritten?"
    missing = sorted(m for m in called if not hasattr(ota.Rig, m))
    assert not missing, (
        f"beacon calls {missing} on ota.Rig, which does not have them -- "
        f"main() raises AttributeError before it can key. Rig offers: "
        f"{sorted(m for m in vars(ota.Rig) if not m.startswith('_'))}")


def test_the_key_is_always_dropped():
    """The finally: that unkeys is the one line here worth pinning. beacon keys in
    a loop, so an exception between keys must still reach rig.stop()."""
    tree = ast.parse(SOURCE.read_text())
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert any("stop" in _rig_calls(ast.Module(body=t.finalbody, type_ignores=[]))
               for t in tries), "no finally: drops the transmitter"


def test_the_sample_rate_is_the_constant():
    assert "48000" not in SOURCE.read_text(), (
        "a literal sample rate drifts on its own; spec.SAMPLE_RATE is the figure")


# -- main, whole, over fakes --------------------------------------------------------


#: A ramp rather than silence, so "the whole burst reached the air" is a claim
#: the audio can actually fail. Every sample is distinct and the last one is the
#: one a discarded tail loses.
_BURST = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)


class _Rig:
    def __init__(self):
        self.stopped = False
        self.failure = None
        self.serial = "/dev/null"

    def set_mode(self, m, passband):
        return True

    def set_freq(self, hz):
        return True

    def arm(self):
        pass

    def ptt(self, up):
        return True

    def key_failure(self):
        return self.failure

    def stop(self):
        self.stopped = True


def _fake_main(monkeypatch, rig, *, ptt_device=lambda r, s: "/dev/null", count=3,
               play=None, profile="ft891"):
    """Wire beacon.main to fakes at beacon's own names; returns the _play log.

    ``play`` substitutes the transmit call, which is how the one test that wants
    the real `ota._play` gets it — over a fake card rather than over a stub.
    """
    monkeypatch.setattr(beacon.session, "build_session", lambda *a, **kw: _BURST)
    monkeypatch.setattr(beacon, "find_device", lambda name, kind: 0)
    monkeypatch.setattr(beacon, "ptt_device", ptt_device)
    monkeypatch.setattr(beacon, "Rig", lambda *a, **kw: rig)
    played: list = []
    monkeypatch.setattr(beacon, "_play",
                        play or (lambda *a, **kw: played.append(a)))
    rig_args = ["--rig", profile] if profile else []
    monkeypatch.setattr(sys, "argv",
                        ["beacon", *rig_args, "--serial", "/dev/null",
                         "--mycall", "W9SSJ", "--dxcall", "W9SSJ", "--dial", "7100000",
                         "--count", str(count), "--gap", "0"])
    return played


def test_a_missing_keying_line_is_a_sentence_not_a_traceback(monkeypatch):
    """`ptt_device` raises PttError; ota.run converts it to NOT KEYING because
    the operator's fix is a flag, not a stack trace. beacon keys the same rig
    and owes the same conversion."""
    def refuse(r, s):
        raise PttError("PTT port /dev/cu.usbserial-XXXXB1 is not a character device")
    played = _fake_main(monkeypatch, _Rig(), ptt_device=refuse)
    with pytest.raises(SystemExit) as e:
        beacon.main()
    assert "NOT KEYING" in str(e.value), str(e.value)
    assert not played


@pytest.mark.parametrize("profile", [name for name, row in beacon.RIGS.items()
                                     if row.get("ptt_iface") is None])
def test_a_profile_without_a_keying_line_says_so_and_keys_nothing(monkeypatch, capsys, profile):
    played = _fake_main(monkeypatch, None, profile=profile)
    monkeypatch.setattr(beacon, "Rig", lambda *a, **kw: pytest.fail("rig opened"))
    beacon.main()
    assert ota.NO_KEYING_LINE in capsys.readouterr().out
    assert not played


def test_the_default_rig_is_the_station_rig_and_keys(monkeypatch, capsys):
    """A default run that cannot key is a beacon that radiates nothing."""
    keyed = []

    def ptt_device(rig, serial):
        keyed.append(rig)
        return "/dev/null"

    played = _fake_main(monkeypatch, _Rig(), ptt_device=ptt_device, count=2, profile=None)
    beacon.main()
    assert keyed == ["ftdx10"]
    assert beacon.RIGS["ftdx10"].get("ptt_iface") is not None
    assert len(played) == 2
    assert ota.NO_KEYING_LINE not in capsys.readouterr().out


def test_a_dead_channel_stops_the_count(monkeypatch):
    """A rigctl that dies mid-run must end the beacon, not let the remaining
    keyings count themselves as sent -- the record-of-intentions defect the
    2026-08-10 headline fix removed from ota.run."""
    rig = _Rig()
    rig.failure = "the rigctl that took the key-down exited (status 2)"
    played = _fake_main(monkeypatch, rig)
    with pytest.raises(SystemExit) as e:
        beacon.main()
    assert "NOT CONFIRMED" in str(e.value), str(e.value)
    assert len(played) == 1, "keyings after a dead channel counted themselves as sent"
    assert rig.stopped, "the finally: must still drop the key"


# -- the whole burst, through a card that is behind ---------------------------------

#: What the device is still holding when a write returns. This station's USB
#: codec reports 3.42 ms low / 12.75 ms high output latency, and the high figure
#: is 612 frames at 48 kHz. A card that emits everything the instant it is handed
#: over cannot tell a transmit path that drains from one that guesses, which is
#: how a beacon shipped for weeks throwing the end of every keying away.
_HELD = int(0.01275 * ota.FS)


class _Out:
    """An output stream that is genuinely behind, so draining it can be got wrong."""

    def __init__(self, card, channels):
        self.card, self.channels = card, channels
        self._played: list[np.ndarray] = []
        self._queued = np.zeros((0, channels), np.float32)
        card.calls.append("open")

    def start(self):
        self.card.calls.append("start")

    def write(self, block):
        n = max(len(block) - _HELD, 0)
        self._played.append(block[:n])
        self._queued = block[n:]

    def stop(self):
        """`Pa_StopStream`: returns only once the queue has reached the air."""
        self.card.calls.append("stop")
        self._played.append(self._queued)
        self._queued = self._queued[:0]

    def close(self):
        """Closing a stream nobody stopped is `paAbort` -- the queue is lost."""
        self.card.calls.append("close")
        self.card.air.append(np.concatenate(self._played)[:, 0] if self._played
                             else np.zeros(0, np.float32))


class _Card:
    """Enough of sounddevice to transmit through, and no convenience call."""

    def __init__(self):
        self.air: list[np.ndarray] = []
        self.calls: list[str] = []

    def OutputStream(self, *, channels=1, **kw):
        return _Out(self, channels)

    def play(self, *a, **kw):
        raise AssertionError(
            "sd.play raises CallbackAbort when the array runs out, which is "
            "PortAudio's paAbort: what the device still holds is discarded "
            "rather than played. Transmit through core.audio.play_drained.")


def _on_a_card(monkeypatch) -> _Card:
    card = _Card()
    monkeypatch.setitem(sys.modules, "sounddevice", card)
    monkeypatch.setattr(audio, "_WARMED", set())  # the warmup is per process, not per test
    return card


def test_the_beacon_puts_its_whole_burst_on_the_air(monkeypatch):
    """Every sample handed over reaches the card, tail included.

    The beacon exists to be watched on a remote SDR, and it was transmitting
    through `sd.play` -- whose callback aborts the stream when the array runs
    out, so the last of the audio was discarded before the key was even
    considered. Nothing above the sound card could see that: the sample count,
    the keyed duration and the log were all unchanged.
    """
    card = _on_a_card(monkeypatch)
    rig = _Rig()
    _fake_main(monkeypatch, rig, count=1, play=ota._play)
    beacon.main()
    assert np.array_equal(card.air[-1], _BURST), (
        f"{_BURST.size - card.air[-1].size} of {_BURST.size} samples never left "
        f"the card")


def test_the_stream_is_stopped_before_it_is_closed(monkeypatch):
    """`Stream.stop()` is `Pa_StopStream`, which waits for the pending buffers;
    closing without it is `paAbort`, which is the defect under another name."""
    card = _on_a_card(monkeypatch)
    _fake_main(monkeypatch, _Rig(), count=1, play=ota._play)
    beacon.main()
    assert card.calls[-2:] == ["stop", "close"]


@pytest.fixture
def stop_handler(monkeypatch):
    """The stop handler the hold works through, with no deadman and no hold
    state left over from another test."""
    from hfmodem.shrike import onair
    before = {sig: signal.getsignal(sig) for sig in FATAL_SIGNALS}
    monkeypatch.setattr(ota, "arm_deadman", lambda *_a: None)
    monkeypatch.setattr(ota, "_depth", 0)
    monkeypatch.setattr(ota, "_pending", None)
    onair._unkey_on_signal()
    yield
    for sig, handler in before.items():
        signal.signal(sig, handler)


def lines_after(function, marker: str) -> set[int]:
    """The source lines of `function` below the one holding `marker`."""
    source, first = inspect.getsourcelines(inspect.unwrap(function))
    at = next(i for i, line in enumerate(source) if marker in line)
    return set(range(first + at + 1, first + len(source)))


def checkpoint(code, offset: int) -> bool:
    """Whether CPython itself could answer a signal just before the instruction
    at `offset`: after a call or a backward jump, or at a function's entry. The
    tracer makes every opcode a checkpoint; these are the ones outside a test."""
    instructions = list(dis.get_instructions(code))
    before = {b.offset: a.opname for a, b in zip(instructions, instructions[1:])}
    here = next(i.opname for i in instructions if i.offset == offset)
    return here == "RESUME" or before.get(offset, "").startswith(
        ("CALL", "JUMP_BACKWARD", "RESUME"))


def kill_at_opcode(regions: dict, n: int, sig: int):
    """A tracer that sends `sig` to the whole process -- for any thread to take,
    as a terminal's Ctrl-C is -- at the `n`th opcode run on `regions`' lines,
    noting whether CPython could have answered it there on its own."""
    seen, real = [0], []

    def local(frame, event, _arg):
        if event == "opcode" and frame.f_lineno in regions[frame.f_code]:
            seen[0] += 1
            if seen[0] == n:
                real.append(checkpoint(frame.f_code, frame.f_lasti))
                os.kill(os.getpid(), sig)
                time.sleep(0.01)
        return local

    def tracer(frame, _event, _arg):
        if frame.f_code in regions:
            frame.f_trace_opcodes = True
            return local
        return None

    return tracer, seen, real


def every_opcode(regions: dict, sig: int, trial):
    """Run `trial` once for each opcode in `regions`, with `sig` sent at it and
    another thread alive to take it; `trial` gets the outcome, the count and
    whether the opcode is one CPython could answer a signal at."""
    idle = threading.Event()
    threading.Thread(target=idle.wait, daemon=True).start()
    n = 0
    try:
        while True:
            n += 1
            tracer, seen, real = kill_at_opcode(regions, n, sig)
            sys.settrace(tracer)
            try:
                outcome = trial.run()
            except (KeyboardInterrupt, SystemExit) as stop:
                outcome = stop
            finally:
                sys.settrace(None)
            if seen[0] < n:
                return n - 1
            trial.check(outcome, n, real[0])
    finally:
        idle.set()


def ladder_rig(monkeypatch, events, *, interrupt=True):
    """A real `ota.Rig` whose `stop` runs over a fake keying line and CAT child,
    asked to stop again between its line unkey and its hand-back unless told
    otherwise."""
    deadmen = []
    monkeypatch.setattr(ota, "find_rigctl", lambda *_a: Path("rigctl"))
    monkeypatch.setattr(ota, "arm_deadman", deadmen.append)

    class Keyer:
        def __init__(self, *_a, **_kw):
            pass

        def arm(self):
            pass

        def unkey(self):
            events.append("line unkey")
            if interrupt:
                signal.raise_signal(signal.SIGINT)

        def hand_back(self):
            events.append("hand back")

    monkeypatch.setattr(ota, "LineKeyer", Keyer)
    rig = ota.Rig(1036, "/dev/null", 38400, ptt_port="/dev/null")
    rig.set_mode = lambda *_a: True
    rig.set_freq = lambda _hz: True
    rig._close = lambda: events.append("close")
    return rig, deadmen


#: The whole ladder, in order, as `ladder_rig` records it.
LADDER = ["line unkey", "hand back", "close"]


def test_a_second_ctrl_c_inside_the_unkey_waits_for_the_rest_of_the_ladder(monkeypatch, stop_handler):
    events = []
    rig, deadmen = ladder_rig(monkeypatch, events)

    def ctrl_c(*_a, **_kw):
        raise KeyboardInterrupt

    _fake_main(monkeypatch, rig, play=ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        beacon.main()
    assert events == LADDER
    assert deadmen[0] == ota.DEADMAN_BUDGET_S
    assert ota._depth == 0 and ota._pending is None


def test_a_watchdog_stop_off_the_main_thread_still_runs_the_ladder(monkeypatch):
    """The PTT watchdog calls `stop` from its own thread. The ladder runs there,
    and neither the handlers nor the main thread's deadman are its to change."""
    events, failed = [], []
    rig, _ = ladder_rig(monkeypatch, events, interrupt=False)
    before = {sig: signal.getsignal(sig) for sig in FATAL_SIGNALS}

    def watchdog():
        try:
            rig.stop()
        except BaseException as exc:
            failed.append(exc)

    alarm = signal.signal(signal.SIGALRM, lambda *_: None)
    try:
        signal.setitimer(signal.ITIMER_REAL, 600)
        thread = threading.Thread(target=watchdog)
        thread.start()
        thread.join()
        left = signal.getitimer(signal.ITIMER_REAL)[0]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, alarm)
    assert not failed and events == LADDER
    assert {sig: signal.getsignal(sig) for sig in FATAL_SIGNALS} == before
    assert left > 590, "the watchdog's stop disarmed the session's deadman"


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_process_directed_signal_anywhere_in_the_hold_waits_for_the_ladder(
        monkeypatch, stop_handler, sig):
    """Signals sent at every opcode from the hold's entry, through the ladder, to
    the hold's release: each one is answered, and none before the CAT child's
    close."""
    regions = {
        inspect.unwrap(ota.signals_held).__code__:
            lines_after(ota.signals_held, "_depth += 1"),
        ota.Rig.stop.__code__: lines_after(ota.Rig.stop, "with signals_held():"),
        ota._release.__code__: lines_after(ota._release, "def _release"),
    }

    class Trial:
        def run(self):
            self.events = []
            rig, _ = ladder_rig(monkeypatch, self.events, interrupt=False)
            rig.stop()

        def check(self, outcome, n, real):
            assert self.events == LADDER, f"opcode {n}: {self.events}"
            assert isinstance(outcome, (KeyboardInterrupt, SystemExit)), f"opcode {n}"
            assert ota._depth == 0, f"opcode {n}"
            if real:
                assert ota._pending is None, f"opcode {n}: {ota._pending}"
            ota._pending = None

    assert every_opcode(regions, sig, Trial()) > 20


def test_a_stop_while_a_held_one_is_delivered_leaves_nothing_behind(monkeypatch, stop_handler):
    """A Ctrl-C held inside the ladder, then a SIGTERM at every opcode from the
    hold's end through the delivery: one stop comes out, and nothing is left
    waiting for the next hold to deliver as if it were its own."""
    die = signal.getsignal(signal.SIGINT)
    regions = {
        inspect.unwrap(ota.signals_held).__code__:
            lines_after(ota.signals_held, "    finally:"),
        ota._release.__code__: lines_after(ota._release, "def _release"),
        ota.hold_signal.__code__: lines_after(ota.hold_signal, "def hold_signal"),
        die.__code__: lines_after(die, "def _die"),
    }

    class Trial:
        def run(self):
            self.events = []
            rig, _ = ladder_rig(monkeypatch, self.events)
            rig.stop()

        def check(self, outcome, n, real):
            assert self.events == LADDER, f"opcode {n}: {self.events}"
            assert isinstance(outcome, (KeyboardInterrupt, SystemExit)), f"opcode {n}"
            assert ota._depth == 0, f"opcode {n}"
            if real:
                assert ota._pending is None, f"opcode {n}: {ota._pending}"
            ota._pending = None

    assert every_opcode(regions, signal.SIGTERM, Trial()) > 20



def stops(exc) -> list[str]:
    """The stop exceptions in an exception's context chain, outermost first."""
    found = []
    while exc is not None:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            found.append(type(exc).__name__)
        exc = exc.__context__
    return found


class Teardown:
    """`run`'s teardown in small: an outer hold with the rest of the unwind's
    budget, `Rig.stop` and its own hold inside it, then the power restore."""

    rest_s = 300.0

    def __init__(self, monkeypatch, *, interrupt, timer=0.0, unwinding=False):
        self.monkeypatch, self.interrupt = monkeypatch, interrupt
        self.timer, self.unwinding = timer, unwinding

    def run(self):
        self.events = []
        rig, _ = ladder_rig(self.monkeypatch, self.events, interrupt=self.interrupt)
        self.monkeypatch.setattr(ota, "arm_deadman", lambda budget: signal.setitimer(
            signal.ITIMER_REAL, budget))
        ota._pending = [] if self.unwinding else None
        signal.setitimer(signal.ITIMER_REAL, self.timer)
        with ota.signals_held(after_s=self.rest_s):
            rig.stop()
            self.events.append("restore")

    @staticmethod
    def regions(*codes):
        die = signal.getsignal(signal.SIGINT)
        return {fn.__code__: lines_after(fn, marker) for fn, marker in codes} | {
            die.__code__: lines_after(die, "def _die")}


@pytest.fixture
def quiet_alarm():
    alarm = signal.signal(signal.SIGALRM, lambda *_: None)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, alarm)


@pytest.mark.usefixtures("stop_handler", "quiet_alarm")
def test_a_first_stop_anywhere_in_a_teardown_comes_out_once(monkeypatch):
    """A session that ended on its own, and its first Ctrl-C at every opcode of
    the teardown: before the hold it is answered at once, inside it the whole
    ladder and restore run first, and either way exactly one stop comes out."""
    regions = Teardown.regions(
        (inspect.unwrap(ota.signals_held), "def signals_held"), (ota.Rig.stop, "def stop"),
        (ota._release, "def _release"), (ota.hold_signal, "def hold_signal"),
        (contextlib._GeneratorContextManager.__enter__, "def __enter__"),
        (contextlib._GeneratorContextManager.__exit__, "def __exit__"),
        (Teardown.run, "signal.setitimer(signal.ITIMER_REAL, self.timer)"))
    teardown = Teardown(monkeypatch, interrupt=False)

    class Trial:
        def run(self):
            teardown.run()

        def check(self, outcome, n, _real):
            assert stops(outcome) == ["KeyboardInterrupt"], f"opcode {n}: {outcome!r}"
            assert teardown.events in ([], LADDER + ["restore"]), (
                f"opcode {n}: {teardown.events}")
            assert ota._depth == 0, f"opcode {n}"
            ota._pending = None

    assert every_opcode(regions, signal.SIGINT, Trial()) > 100


@pytest.mark.usefixtures("stop_handler", "quiet_alarm")
@pytest.mark.parametrize("unwinding", [False, True])
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_second_stop_through_the_release_keeps_the_handover(monkeypatch, sig, unwinding):
    """A Ctrl-C held inside the ladder, then a second stop at every opcode from
    the hold's end through the delivery. Wherever CPython could answer it, the
    rest of the unwind's budget is still handed over, one stop comes out, and
    the hold is left empty."""
    regions = Teardown.regions(
        (inspect.unwrap(ota.signals_held), "    finally:"), (ota._release, "def _release"),
        (ota.hold_signal, "def hold_signal"),
        (contextlib._GeneratorContextManager.__exit__, "def __exit__"))
    teardown = Teardown(monkeypatch, interrupt=True, timer=600.0, unwinding=unwinding)

    class Trial:
        def run(self):
            teardown.run()

        def check(self, outcome, n, real):
            left = signal.getitimer(signal.ITIMER_REAL)[0]
            assert teardown.events == LADDER + ["restore"], f"opcode {n}"
            assert stops(outcome)[:1] and len(stops(outcome)) == 1, f"opcode {n}: {outcome!r}"
            assert ota._depth == 0, f"opcode {n}"
            if real:
                assert ota._pending is None, f"opcode {n}: {ota._pending}"
                assert Teardown.rest_s - 5 < left <= Teardown.rest_s, f"opcode {n}: {left}"
            ota._pending = None

    assert every_opcode(regions, sig, Trial()) > 20

#: `python -m hfmodem.shrike.ota`, with the session swapped for a stop whose
#: ladder is interrupted between its line unkey and its hand-back.
_OTA_AS_MAIN = """
import runpy, signal, sys
from pathlib import Path
import hfmodem.shrike.ota as ota

events = []


class Keyer:
    def __init__(self, *_a, **_kw):
        pass

    def unkey(self):
        events.append("line unkey")
        signal.raise_signal(signal.SIGINT)

    def hand_back(self):
        events.append("hand back")


def main():
    ota.find_rigctl = lambda *_a: Path("rigctl")
    ota.arm_deadman = lambda *_a: None
    ota.LineKeyer = Keyer
    rig = ota.Rig(1036, "/dev/null", 38400, ptt_port="/dev/null")
    rig._close = lambda: events.append("close")
    try:
        rig.stop()
    finally:
        print(events, flush=True)


ota.main = main
sys.argv = ["ota"]
runpy.run_module("hfmodem.shrike.ota", run_name="__main__", alter_sys=True)
"""


def test_run_as_a_module_ota_still_holds_a_ctrl_c_for_the_whole_ladder(tmp_path):
    env = dict(os.environ, HFMODEM_STATE_DIR=str(tmp_path / "state"),
               HFMODEM_COEXIST_HARNESS="1")
    done = subprocess.run([sys.executable, "-c", _OTA_AS_MAIN], capture_output=True,
                          text=True, env=env, timeout=60)
    said = done.stdout + done.stderr
    assert "['line unkey', 'hand back', 'close']" in said, said
    assert "KeyboardInterrupt" in said, said
