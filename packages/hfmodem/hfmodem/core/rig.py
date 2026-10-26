# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CAT through a driver; keying and its watchdog through an independent line.

Deassert-on-process-death remains UNPROVEN on this
   hardware; the line and CAT panic legs provide separate readback evidence.
"""
from __future__ import annotations

import contextlib
import math
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hfmodem.core import band, rigs
from hfmodem.core.operations import LeaseTimeout, OperationGate, leased
from hfmodem.core.ptt import Ptt, PttError
from hfmodem.core.regulatory import Control, Emission, NotPermitted, Profile

if TYPE_CHECKING:
    from .cat import CatDriver

#: How long `panic_unkey()` may take before we stop believing it will finish.
PANIC_BUDGET_S = 2.0

#: Attempts on the independent CAT path during a panic.
PANIC_ATTEMPTS = 4

#: How long the keying line gets in a panic before the ladder goes on without it.
#: A healthy `release()` is three ioctls and a close, in microseconds; this is the
#: budget for one that is not healthy, and it must stay well inside
#: `PANIC_BUDGET_S` so the CAT leg still has room under the signal path's deadman.
RELEASE_BUDGET_S = 0.5

#: Slack a burst gets beyond its own length before the watchdog fires.
_WATCHDOG_SLACK_S = 2.0
_WATCHDOG_FRACTION = 0.25

#: The ways a run is ended by something other than its operator's keyboard.
#: SIGTERM is timeout(1), a supervisor recycle and a cancelled tool call; SIGHUP
#: is a closed terminal or a dropped ssh session. Neither reaches a `finally:`.
FATAL_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def fatal_write(message: bytes) -> None:
    """Best-effort stderr output without changing the caller's tty mode."""
    try:
        blocking = os.get_blocking(2)
        try:
            os.set_blocking(2, False)
            os.write(2, message)
        finally:
            os.set_blocking(2, blocking)
    except OSError:
        pass


def arm_deadman(budget_s: float = PANIC_BUDGET_S, *, message: bytes | None = None) -> None:
    """Kill this interpreter shortly, whatever it is doing.

    setitimer, not threading.Timer. Starting a thread takes interpreter locks the
    interrupted thread may hold, so the one mechanism whose whole purpose is to be
    unblockable by that thread could be blocked by it; a SIGALRM handler needs
    nothing but the kernel. Off the main thread no handler is installable and a
    thread is what is left.
    """
    def terminate(*_):
        if message is not None:
            fatal_write(message)
        os._exit(1)

    try:
        signal.signal(signal.SIGALRM, terminate)
        signal.setitimer(signal.ITIMER_REAL, budget_s)
    except (ValueError, OSError):
        threading.Timer(budget_s, terminate).start()


@contextlib.contextmanager
def unkey_on_signal(rig: Callable[[], Rig | None]) -> Iterator[None]:
    """Hold the transmitter's fate for this block, however the process dies.

    A `finally:` runs for a Ctrl-C and for nothing else, and `atexit` does not run
    on a signal either — so a run killed while the line is up leaves it up, with
    only HUPCL-on-last-close under it, which `core.ptt` records as unproven on this
    hardware. The handler goes to `Rig.panic`: retire, arm a deadman, unkey, exit.

    The rig is read at signal time rather than passed, because the rig that needs
    bringing down is usually built inside the block being guarded. Dispositions are
    given back on the way out: they are process-global and belong to no one command.
    """
    def down(signum, _frame):
        current = rig()
        if current is not None:
            current.panic()             # retires, unkeys, exits — does not return
        raise SystemExit(128 + signum)

    prior = {sig: signal.signal(sig, down) for sig in FATAL_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in prior.items():
            signal.signal(sig, handler)


class RigError(Exception):
    """The rig could not be reached or refused a command."""


class StuckTransmitter(RigError):
    """Every path to unkey has been tried and none was confirmed.

    Loud on purpose. Raised only after `retired` is already set, so the program
    has stopped being able to key even if nobody catches this.
    """


@dataclass(frozen=True, slots=True)
class KeyToken:
    """Proof that this caller is the one holding the key."""

    seq: int
    at: float
    deadline: float | None
    why: str
    card_index: int | None = None
    card_deadline: int | None = None


@dataclass(frozen=True, slots=True)
class ArmReport:
    model: str
    dial_hz: int
    ptt_proven: bool
    ptt_owner_checked: bool
    notes: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [f"rig {self.model}, dial {self.dial_hz / 1e6:.6f} MHz",
                 "PTT PROVEN" if self.ptt_proven else "PTT UNPROVEN",
                 "PTT line ownership verified" if self.ptt_owner_checked
                 else "PTT line ownership NOT verified"]
        return "\n".join(lines + list(self.notes))


def __getattr__(name):
    if name == "Cat":
        from .cat import RigctldCat

        class Cat(RigctldCat):
            trusted = property(lambda self: self.cache_free)
            ptt_readable = property(lambda self: self.reads_ptt)
            model_name = RigctldCat.identify

        globals()[name] = Cat
        return Cat
    raise AttributeError(name)


class Rig:
    def __init__(self, *, model: str, cat: CatDriver | None, ptt: Ptt | None, profile: Profile,
                 control: Control, mycall: str = "", transmit: bool = False,
                 max_key_s: float = 30.0, identity: str | None = None,
                 dial_max_age_s: float = 30.0) -> None:
        if transmit and (cat is None or ptt is None):
            raise RigError("transmit requires CAT and a keying line")
        if (type(dial_max_age_s) not in (int, float)
                or not math.isfinite(dial_max_age_s) or dial_max_age_s <= 0):
            raise ValueError("dial_max_age_s must be finite and positive")
        self.identity = ("" if model == "generic" else model) if identity is None else identity
        self.dial_max_age_s = dial_max_age_s
        self.model = model
        self.cat = cat
        self.ptt = ptt
        self.profile = profile
        self.control = control
        self.mycall = mycall
        #: A hard interlock, not a preference. False and nothing can key.
        self.transmit = transmit
        self.max_key_s = max_key_s
        self.operations = OperationGate()
        if cat is not None:
            cat.operations = self.operations
        self.fatal = None
        self._fatal_handoff = False

        self._armed = False
        self._retired_why = ""
        self._panicked = False
        self._panic_confirmed: bool | None = None
        self._seq = 0
        self._keyed = threading.Event()
        self._keyed_since = 0.0
        #: "pending" while a key drives the line outside the lock, "overtaken"
        #: once an unkey lands in that window and leaves the key state to it.
        self._drive: str | None = None
        self._timer: threading.Timer | None = None
        self._dial_hz = 0
        self._dial_at: float | None = None
        self._dial_pins = 0
        self._protection = None
        self.line_state = "unconfirmed" if ptt is not None else "none"
        self.released_low: bool | None = None
        self._lock = threading.Lock()
        #: Held only by the panic path. A lock no caller and no CAT path takes
        #: cannot reintroduce the deadlock the panic path exists to avoid, and a
        #: check-then-set on a bare flag lets two threads both close the same fd.
        self._panic_lock = threading.Lock()

        if cat is not None:
            cat.keyed_probe = self._keyed.is_set

    # -- state -------------------------------------------------------------

    @property
    def keyed(self) -> bool:
        return self._keyed.is_set()

    @property
    def retired(self) -> bool:
        """A transmitter nobody has confirmed is down is not one we may key."""
        return bool(self._retired_why)

    @property
    def retired_why(self) -> str:
        return self._retired_why

    def _retire(self, why: str) -> None:
        if not self._retired_why:
            self._retired_why = why

    # -- arming ------------------------------------------------------------

    @leased
    def arm(self, *, prove_ptt: bool = True) -> ArmReport:
        if prove_ptt:
            if not self.transmit:
                raise RigError("proving the PTT line keys the radio, and transmit = false")
            if self.retired:
                raise RigError(f"this rig is retired: {self._retired_why}")
        if self.cat is None:
            return ArmReport("none", 0, False, False, ("No CAT or keying line owned.",))
        self.cat.open()
        notes: list[str] = []
        if not self.cat.cache_free:
            notes.append("rigctld would not disable its cache, so `t` reports "
                         "hamlib's memory of its own last command rather than the "
                         "radio. PTT readback is not evidence on this daemon.")

        want = rigs.RIGS.get(self.model, {}).get("note", self.model)
        got = self.cat.identify()
        if self.identity and not got:
            raise RigError(
                "rigctld would not say what radio it is driving. Keying an "
                "unidentified rig is the expensive mistake this check exists for.")
        if self.identity and not rigs.model_identity_matches(self.identity, got):
            raise RigError(
                f"CAT reports {got!r}, configured for {self.identity!r}. Keying "
                f"the wrong radio is the expensive mistake. ({want})")

        if not self.identity:
            notes.append("Rig identity is unchecked; no identity configured.")
        if self.model == "generic":
            notes.append("Transmit power is the operator's responsibility.")

        f = self.cat.freq()
        if not 1_000_000 <= f <= 60_000_000:
            raise RigError(
                f"CAT reports {f} Hz, which is not an HF dial. A wrong serial "
                "port answers plausibly enough to be worth checking.")
        self._cache_dial(f)

        if self.cat.ptt_state(_arming=True) is True:
            raise RigError("the radio is already transmitting. Someone else has it.")

        proven = owner_checked = False
        if prove_ptt:
            # No callsign is required for the proof: keying a sideband
            # transmitter with no audio emits nothing to identify — see
            # _prove_ptt.
            owner_checked = self._check_ptt_owner(notes)
            if not owner_checked:
                notes.append("PTT line ownership is unverified; continuing with the PTT proof.")
            with self.cat.hold() as held:
                if not held:
                    raise RigError("CAT hold refused")
                proven = self._prove_ptt(notes)
        else:
            notes.append("PTT proof skipped — the line has not been shown to "
                         "reach the radio.")

        if self.retired:
            # _prove_ptt's finally can panic, which retires. Returning a report
            # that reads like success for a rig that will refuse every burst is
            # worse than raising.
            raise RigError(f"the rig retired while arming: {self._retired_why}")
        if self.ptt is not None and self.ptt.sense() is False:
            self.line_state = "low"
        self._armed = self.transmit and self.ptt is not None
        return ArmReport(got or self.model, self._dial_hz, proven, owner_checked,
                         tuple(notes))

    def _check_ptt_owner(self, notes: list[str]) -> bool:
        return self.cat.sole_ptt_owner(notes)

    def _admission_pause(self, seconds=0):
        """Observe STOP between operations on the owning thread.

        Cancellation waits at most for the in-flight operation before normal
        unkey begins: CAT exchanges have a 1 s deadline; a line ioctl has no
        finite kernel deadline. Settle waits are interruptible by the event.
        """
        cancel = getattr(self, "admission_cancel", None)
        if cancel is None:
            if seconds:
                time.sleep(seconds)
        elif cancel.wait(seconds):
            raise RigError("ownership cancelled by STOP")

    def _prove_ptt(self, notes: list[str]) -> bool:
        if self.ptt is None:
            self.released_low = None
            return False
        self._admission_pause()
        if self.ptt.sense() is not False:
            self.line_state = "unconfirmed"
            raise RigError("keying line is not confirmed low before the proof")
        self._admission_pause()
        self.line_state = "low"
        hold = 0.20
        self._keyed.set()
        self._keyed_since = time.monotonic()
        self._arm_watchdog(hold + _WATCHDOG_SLACK_S, "arm-gate PTT proof")
        try:
            self.ptt.assert_(True)
            self._admission_pause(0.05)
            state = self.cat.ptt_state(_arming=True)
            self._admission_pause(max(0.0, hold - 0.05))
            if state is True and self.cat.cache_free:
                return True
            if state is False:
                notes.append(
                    "our line is up and the radio reports PTT down. Either the PTT "
                    "cable is on the wrong interface, or rigctld cannot see this "
                    "line — check which before believing either." if self.cat.cache_free
                    else "the radio reports PTT down, but this daemon serves `t` "
                         "from its own cache, so that is not evidence.")
            elif not self.cat.reads_ptt:
                notes.append(
                    "this daemon has no PTT of its own to report — the "
                    "configuration this station requires, so that the keying line "
                    "has one owner — and it answers `t` with ENAVAIL. The line was "
                    "asserted and CAT cannot say whether it arrived. Watching the "
                    "receiver mute is the measurement that can.")
            else:
                notes.append("the radio will not report PTT; the line was asserted "
                             "and nothing confirms it arrived.")
            return False
        finally:
            self.unkey()

    # -- tuning ------------------------------------------------------------

    @leased
    def tune(self, centre_hz: float) -> None:
        """Point the radio at a published channel centre.

        The dial is derived, never configured. Re-read afterwards because it does
        not stay where you left it — a real and repeated observation, not caution.
        """
        if self.cat is None:
            raise RigError("no CAT configured")
        with self.cat.hold(), self._lock:
            if self.keyed:
                raise RigError("refusing to retune while keyed")
            if self._dial_pins:
                raise RigError("refusing to retune a pinned dial")
            dial = band.dial_hz(centre_hz)
            self.cat.set_freq(dial)
            self._cache_dial(self.cat.freq())
            if abs(self._dial_hz - dial) > band.QSY_TOLERANCE_HZ:
                raise RigError(
                    f"asked for dial {dial} Hz, the radio reports {self._dial_hz}. "
                    "Believe the readback.")

    def _cache_dial(self, hz: int) -> None:
        self._dial_hz = hz
        self._dial_at = time.monotonic()

    @property
    def dial_age_s(self) -> float:
        return math.inf if self._dial_at is None else time.monotonic() - self._dial_at

    @property
    def panic_confirmed(self) -> bool | None:
        return self._panic_confirmed

    @contextlib.contextmanager
    def pin_dial(self):
        with self._lock:
            if self._dial_at is None:
                raise RigError("no dial reading has succeeded")
            self._dial_pins += 1
        try:
            yield self._dial_hz
        finally:
            with self._lock:
                self._dial_pins -= 1

    @leased
    def dial(self) -> int:
        if self.cat is None:
            raise RigError("no CAT configured")
        with self.cat.hold(), self._lock:
            if not self.keyed and not self._dial_pins:
                self._cache_dial(self.cat.freq())
            return self._dial_hz

    def hold(self, deadline_index=None):
        if self.cat is None:
            raise RigError("no CAT configured")
        return self.cat.hold(deadline_index)

    # -- keying ------------------------------------------------------------

    @leased
    def key(self, emission: Emission, *, why: str, duration_s: float | None = None,
            settle_s: float = 0.0, responding: bool = False,
            sample_now=None, deadline_index=None, attended: bool = False,
            open: bool = False) -> KeyToken:
        if duration_s is None and not open:
            raise TypeError("finite key requires duration_s")
        if not self.transmit:
            raise RigError("transmit is disabled for this station")
        with self.cat.hold(deadline_index) as held:
            if not held:
                raise RigError("CAT hold missed the key deadline")
            with self._lock:
                if self.retired:
                    raise RigError(f"this rig is retired: {self._retired_why}")
                if not self._armed:
                    raise RigError("arm() has not passed")
                if self.keyed:
                    raise RigError("already keyed")
                if self.line_state == "unconfirmed":
                    raise RigError("keying line is unconfirmed")
                if self._dial_at is None:
                    raise RigError("no dial reading has succeeded")
                if not (attended or self._dial_pins) and self.dial_age_s > self.dial_max_age_s:
                    raise RigError("cached dial is stale")
                checked = Emission(self._dial_hz, emission.audio_lo_hz, emission.audio_hi_hz,
                                   power_w=emission.power_w, designator=emission.designator,
                                   technique=emission.technique,
                                   symbol_rate_bd=emission.symbol_rate_bd,
                                   drift_hz=emission.drift_hz)
                try:
                    self.profile.check(checked, control=self.control, responding=responding)
                except NotPermitted as exc:
                    raise RigError(f"refused by {self.profile.name}: {exc}") from None
                deadline = None if open else self._deadline(duration_s, settle_s)
                self._seq += 1
                now = time.monotonic()
                index = int(self.cat.clock())
                end = None if open else index + int(deadline * 48000)
                self._protection = self.cat.protect(index, end)
                if not open:
                    self._arm_watchdog(deadline, why)
                self._keyed.set()
                self._keyed_since = now
                self._drive = "pending"
            try:
                self.ptt.assert_(True)
            except PttError as exc:
                self.panic_unkey(f"the keying line refused to assert: {exc}")
                raise RigError(f"cannot key: {exc}") from None
            finally:
                with self._lock:
                    overtaken, self._drive = self._drive == "overtaken", None
            if overtaken:
                self.unkey()
                raise RigError(f"an unkey overtook this key; the keying line is {self.line_state}")
            if self.retired:
                raise RigError(f"this rig is retired: {self._retired_why}")
            with self._lock:
                if self._keyed.is_set():
                    self.line_state = "high"
            card_index = int(sample_now()) if sample_now is not None else None
            card_end = (card_index + int(max(0, now + deadline - time.monotonic()) * 48000)
                        if card_index is not None and deadline is not None else None)
            return KeyToken(self._seq, now, deadline, why, card_index, card_end)

    def _deadline(self, duration_s: float, settle_s: float) -> float:
        """Per burst, not a flat ceiling.

        shrike derives it this way, and the reason is arithmetic: a flat 30 s cap
        lets a 1.25 s burst whose audio never arrives hold an unmodulated carrier
        for twenty-nine seconds of someone else's channel.
        """
        want = settle_s + duration_s + max(_WATCHDOG_SLACK_S,
                                           _WATCHDOG_FRACTION * duration_s)
        return min(want, self.max_key_s)

    def _clear_key(self) -> None:
        self._keyed.clear()
        self._cancel_watchdog()
        if self._protection is not None:
            self.cat.unprotect(self._protection)
            self._protection = None

    @leased
    def unkey(self, token: KeyToken | None = None) -> None:
        if self.ptt is None:
            self.released_low = None
            return
        with self._lock:
            if token is not None and token.seq != self._seq:
                return
            try:
                self.ptt.assert_(False)
                state = self.ptt.sense()
            except PttError as exc:
                self.panic_unkey(f"the keying line refused to deassert: {exc}")
                return
            if state is True:
                self.panic_unkey("the keying line reads high after deassert")
                return
            self.line_state = "low" if state is False else "unconfirmed"
            if self._drive:
                self._drive = "overtaken"
            else:
                self._clear_key()

    # -- panic -------------------------------------------------------------

    def panic_unkey(self, why: str) -> None:
        """Every independent path down, in order, holding no lock.

        No `keyed` guard anywhere on this path: a guard that can suppress the
        panic is not a safety mechanism.
        """
        if self.ptt is None:
            self.released_low = None
            return
        if self.fatal is not None and self.operations.active and not self._fatal_handoff:
            self._fatal_handoff = True
            self.fatal(f"panic overlapped an admitted operation: {why}")
        with self._panic_lock:
            first, self._panicked = not self._panicked, True
        if not first:
            # Nothing left to DO: the line is released and the rig is retired.
            # Re-entry is ordinary rather than catastrophic — a watchdog armed
            # before an earlier panic will still fire — so it does not take the
            # process down. But it must still SAY what the first panic found. A
            # bare return dropped that verdict, and `panic()` initialises
            # `confirmed = True` and only clears it by catching this exception:
            # a first panic that never confirmed the transmitter down, followed
            # by the operator's Ctrl-C, exited 0 in silence. Exit 0 is the one
            # status that invites a supervisor to restart into a keyed radio. A
            # panic still RUNNING reads `None` here rather than False, and that
            # case belongs to `panic()`, which exits non-zero on finding one.
            if self._panic_confirmed is False:
                raise StuckTransmitter(
                    f"{why}: an earlier panic never confirmed the transmitter "
                    "is down. Check the radio and remove power.")
            return
        self._retire(why)
        self._cancel_watchdog()
        started = time.monotonic()
        os.write(2, f"!! unkeying: {why}\n".encode())

        line = threading.Thread(target=self._release_line, daemon=True)
        line.start()
        line.join(RELEASE_BUDGET_S)
        if line.is_alive():
            os.write(2, b"   the keying line has not answered; going on to CAT\n")

        # The keying line answers first, because on this station it is the only
        # witness that can answer at all. `release()` reads it back between the
        # clear and the close; a driver reporting the line low is more direct
        # evidence than asking a daemon about a PTT it does not own. CAT stays as
        # the second path for a station wired the other way.
        confirmed = getattr(self.ptt, "released_low", None) is True
        if confirmed:
            os.write(2, b"   keying line reads low\n")
        self.released_low = getattr(self.ptt, "released_low", None)
        self.line_state = "low" if self.released_low is True else "unconfirmed"
        for attempt in range(PANIC_ATTEMPTS):
            backoff = 0.05 * attempt
            if time.monotonic() - started + backoff + 0.4 > PANIC_BUDGET_S:
                break
            if backoff:
                time.sleep(backoff)
            if time.monotonic() - started + 0.4 > PANIC_BUDGET_S:
                break
            try:
                if self.cat is not None and self.cat.emergency_unkey() is False:
                    confirmed = True
                    break
            except (OSError, RigError):
                pass
        self._clear_key()
        self._panic_confirmed = confirmed
        if not confirmed:
            os.write(2, b"!! UNKEY NOT CONFIRMED. CHECK THE RADIO AND REMOVE POWER.\n")
            if self.cat is not None and not self.cat.reads_ptt:
                # Naming the reason costs one write and stops the loudest message
                # in the program from meaning nothing: this daemon answers `T 0`
                # and `t` with ENAVAIL, so the CAT leg could not have confirmed
                # anything, on this configuration, ever.
                os.write(2, b"   (rigctld has no PTT of its own here, so `t` could "
                            b"not have confirmed it either way)\n")
            raise StuckTransmitter(
                f"{why}: no path confirmed the transmitter is down. "
                "Check the radio and remove power.")
        os.write(2, b"   confirmed down\n")

    def _release_line(self) -> None:
        """The keying line's leg of the ladder, on a thread so it can be left behind.

        `release()` is an ioctl, a readback and a close on a USB serial adapter, and
        an adapter that has stopped answering blocks all three in the kernel with the
        line still asserted — unbounded, uninterruptible, and standing in front of
        the one leg that does not go through it. A `setitimer` bound is no use here:
        `panic()` already has one pending for the deadman and setitimer replaces
        rather than stacks, and the watchdog arrives off the main thread, where no
        handler is installable at all.

        A thread, which `arm_deadman` refuses for itself, because this is not the
        last resort and does not have to be unblockable. On the signal path the
        deadman is still underneath. On the watchdog's there is none — a watchdog
        panic is not a request to end the process — and what must be reached instead
        is CAT.
        """
        if self.ptt is None:
            self.released_low = None
            return
        try:
            self.ptt.release()          # deassert, close, and do not reopen
        except Exception as exc:        # noqa: BLE001 — nothing may escape here
            os.write(2, f"   the keying line could not be released: {exc}\n".encode())

    def panic(self) -> None:
        """The signal path. Unkey first; drain nothing.

        An operator who reached for Ctrl-C is entitled to a dead transmitter and a
        dead process in about a second, whatever rigctld is doing — and that is
        exactly the moment rigctld is least likely to answer. The deadman is armed
        *before* the unkey, because a Python signal handler runs between bytecodes
        and cannot preempt a C extension holding the GIL.

        A second signal while the first panic is still working means the operator
        has asked twice and is done asking — and so does one signal arriving on top
        of a watchdog's panic, which is the emergency this whole path exists for.
        Either way the ladder below is already running and its verdict is not in
        yet, so there is nothing to wait for and nothing that can be called clean.
        """
        if self._panicked:
            os.write(2, b"!! asked again during a panic: exiting with the unkey "
                        b"unconfirmed. CHECK THE RADIO.\n")
            os._exit(1)
        self._retire("a signal arrived")
        arm_deadman(PANIC_BUDGET_S)
        confirmed = True
        try:
            self.panic_unkey("signal")
        except StuckTransmitter:
            confirmed = False
        # Non-zero on an unconfirmed unkey: exit 0 tells a supervisor everything is
        # well, which is the one status that invites a restart into a keyed radio.
        os._exit(0 if confirmed else 1)

    # -- watchdog ----------------------------------------------------------

    def _arm_watchdog(self, deadline_s: float, why: str) -> None:
        self._cancel_watchdog()
        t = threading.Timer(deadline_s, self._watchdog_fired, args=(why, deadline_s))
        t.daemon = True
        self._timer = t
        t.start()

    def _cancel_watchdog(self) -> None:
        t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    def _watchdog_fired(self, why: str, deadline_s: float) -> None:
        try:
            self.panic_unkey(f"keyed past {deadline_s:.2f} s ({why})")
        except StuckTransmitter:
            pass      # already reported on fd 2; excepthook would swallow it here

    # -- teardown ----------------------------------------------------------

    def close(self) -> bool | None:
        # A failed drain can leave PTT up even when no Station fatal handler is
        # installed (CLI and preflight use this object too). Always attempt the
        # bounded line release, and retain the watchdog until a clean drain.
        drain_error = None
        try:
            self.operations.close(self.max_key_s + 5)
        except LeaseTimeout as exc:
            drain_error = exc
            self._retire(f"close timed out: {exc}")

        # Even after a clean drain, a USB ioctl can wedge in release().  Bound
        # this leg so Station.shutdown can reach its deadman and independent CAT
        # unkey path.  Keep the watchdog until the line has actually answered.
        errors: list[Exception] = []

        def release() -> None:
            try:
                if self.ptt is not None:
                    self.ptt.release()
            except Exception as exc:  # noqa: BLE001 — report on the caller
                errors.append(exc)

        line = threading.Thread(target=release, daemon=True)
        line.start()
        line.join(RELEASE_BUDGET_S)
        if line.is_alive():
            self._retire("keying line did not release during close")
            raise LeaseTimeout("keying line did not release during close")
        if errors:
            self._retire(f"keying line release failed during close: {errors[0]}")
            raise LeaseTimeout(f"keying line release failed during close: {errors[0]}")
        self.released_low = getattr(self.ptt, "released_low", None)
        if self.ptt is not None:
            self.line_state = "low" if self.released_low is True else "unconfirmed"
            if self.released_low is not True:
                self._retire("keying line unconfirmed at close")
        if drain_error is not None:
            raise drain_error
        self._clear_key()
        if self.cat is not None:
            self.cat.close()
        return self.released_low


if os.environ.get("HFMODEM_COEXIST_HARNESS") == "1" and os.environ.get("HFMODEM_COEXIST_INSTRUMENT") == "1":
    from .coexist import instrument_rig
    instrument_rig(Rig)
