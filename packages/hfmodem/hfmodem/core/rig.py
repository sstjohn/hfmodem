# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The transmitter, and every way of making sure it stops.

Two transports, one job each, and never the other's:

    Cat     rigctld over one persistent socket. Frequency and mode. Never PTT.
    Ptt     a modem control line on a second serial interface. Never CAT.

That split is the whole design. `rigctld` owns the CAT port exclusively — station
policy, and the only reading of the incident record that survives scrutiny — while
`core.ptt` owns the keying line. One owner each, by construction rather than by
convention, so the two cannot contend and a wedged CAT transaction cannot hold the
key down.

## Keying requires a checked emission

`key()` takes an `Emission` and consults the station's regulatory profile before
the line goes up. It is not possible to key this object without one, which is
deliberate: an earlier design had a perfectly good regulatory gate that **nothing
ever called**, and a gate no transmit path consults is documentation. Making it an
argument means forgetting is a type error rather than an omission.

## Bringing it down

Ordered by how much of the machine has to still be working:

1. **`unkey()` in a `finally`** — one ioctl, microseconds. This is what fixes the
   *late* unkey, which a watchdog never could: a one-shot `rigctl` costs ~900 ms
   and that cost lands on the release.
2. **The watchdog**, armed *before* the key goes up, with a deadline derived from
   the burst rather than a flat ceiling. A 1.25 s PACTOR burst whose audio never
   reaches the card must not hold an unmodulated carrier for thirty seconds.
3. **`panic_unkey()`** — no lock, no `keyed` guard, no reopen. Then a *fresh*
   rigctld socket for `T 0`, read back, and `retired` either way.
4. **Process death** → fd close → the driver deasserts. **UNPROVEN on this
   hardware**; see `core.ptt`. It is the only mechanism that survives SIGKILL, and
   it is a hope until the witness dongle measures it.
5. The rig's own TX timeout, which is the operator's and not ours.

`retired` is a state, not an exception. A transmitter nobody has confirmed is down
is not one this program may key again, and `StuckTransmitter` raised on a watchdog
thread goes to `threading.excepthook` and dies there — so the refusal has to live
in an attribute the caller checks, not in a traceback nobody reads.

## Two rules with no exceptions and one with exactly one

**No CAT request while keyed.** It is the single point every incident account
agrees on, so it is an assertion in `Cat.command()` rather than a note. The one
exception is the arm gate's PTT proof, which is the only way to learn whether the
keying line reaches the radio at all; it passes `_arming=True` and says why.

**The panic path takes no lock.** The hazard is concrete: the arbiter holds its
lock, calls `key()`, blocks inside rigctld; the watchdog fires and waits on the
rig lock behind it — deadlock, transmitter keyed. An `RLock` is not the fix,
because it would let the handler interleave with a half-finished key-up.

**Nothing on the panic path calls `logging`.** Its lock may be held by the thread
we interrupted. `os.write(2, ...)` cannot deadlock.
"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from hfmodem.core import band, rigs
from hfmodem.core.ptt import Ptt, PttError
from hfmodem.core.regulatory import Control, Emission, NotPermitted, Profile

#: How long `panic_unkey()` may take before we stop believing it will finish.
PANIC_BUDGET_S = 2.0

#: Attempts on the independent CAT path during a panic, with backoff between.
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


def arm_deadman(budget_s: float = PANIC_BUDGET_S) -> None:
    """Kill this interpreter shortly, whatever it is doing.

    setitimer, not threading.Timer. Starting a thread takes interpreter locks the
    interrupted thread may hold, so the one mechanism whose whole purpose is to be
    unblockable by that thread could be blocked by it; a SIGALRM handler needs
    nothing but the kernel. Off the main thread no handler is installable and a
    thread is what is left.
    """
    try:
        signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
        signal.setitimer(signal.ITIMER_REAL, budget_s)
    except (ValueError, OSError):
        threading.Timer(budget_s, lambda: os._exit(1)).start()


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
    deadline: float
    why: str


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


class Cat:
    """rigctld over one persistent socket. Frequency and mode only.

    One connection, held open. A connection per command was measured at ~900 ms
    for a one-shot `rigctl`, and the cost lands on the unkey — which is the one
    place latency is unaffordable.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4532,
                 timeout: float = 2.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        #: False when rigctld would not take the setup above, so `t` is a cache
        #: read rather than a measurement. Callers that need evidence must check.
        self.trusted = True
        #: False once this daemon has declined to report PTT at all. A rig
        #: configured with no PTT of its own — which is what this station requires,
        #: since it keys through its own line — answers `t` with ENAVAIL. Then a
        #: missing confirmation is not a failed one, and anything reporting a
        #: verdict has to tell those apart.
        self.ptt_readable = True
        self._lock = threading.Lock()
        #: Set by the Rig that owns us. Read on every command.
        self.keyed_probe = lambda: False

    def open(self) -> None:
        if self._sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), self.timeout)
        except OSError as exc:
            raise RigError(
                f"no rigctld at {self.host}:{self.port} ({exc}). It owns the CAT "
                "port; nothing here talks to the serial line directly.") from None
        # The dial timeout must not survive as the read deadline — that bug cost
        # this project a silently-detaching host link.
        s.settimeout(self.timeout)
        self._sock = s
        self._buf = b""

        # Two things must be true before any answer from this socket is evidence,
        # and one command settles both.
        #
        # A reply must end in a `RPRT <n>` line, so it can be read to its end
        # rather than to a buffer size. Without a terminator one recv() of
        # `\dump_caps` — 10,206 bytes on a real rig — returns a fragment and
        # leaves the rest in the stream, after which every reply is attributed to
        # the wrong command. That is how `ptt_state()` comes to report a keyed
        # transmitter as down.
        #
        # This used to be requested with `\set_conf extended_resp 1`. Hamlib 5
        # has no such setting: it answers `RPRT -1` and then closes the
        # connection, so `open()` returned a Cat whose socket was already gone and
        # every later command said "CAT is not open" — from a caller that had done
        # nothing wrong. Terminators are per-command in this protocol (`+f`) and
        # ordinary replies carry one anyway, so the setting bought nothing and
        # cost the connection.
        #
        # `\set_cache 0` makes `t` and `f` reach the radio. hamlib otherwise
        # serves them from a 500 ms cache seeded by our own `T`/`F`
        # (`rig.c:3888`), so reading PTT back after commanding it down returns our
        # own command rather than the transmitter's state — a confirmation that
        # cannot fail, which is the same as no confirmation.
        #
        # It doubles as the terminator probe: `command()` reads until `RPRT `, so
        # an answer at all is proof the terminator is there.
        try:
            self.command("\\set_cache 0", _arming=True)
        except RigError:
            # An old rigctld may not know it, or may not terminate its replies.
            # Say so once; the caller can still work, but nothing downstream may
            # treat `t` as evidence.
            self.trusted = False
            # Refusing a setup command and hanging up on it are one event on this
            # daemon, and the refusal arrives first. One more command settles which
            # happened, here, rather than in a caller that has done nothing wrong.
            try:
                self.command("f", _arming=True)
            except RigError:
                pass
        if self._sock is None:
            raise RigError(
                f"rigctld at {self.host}:{self.port} closed the connection during "
                "setup. A Cat that reports success with a dead socket fails much "
                "later, somewhere that looks like the caller's fault.")

    def close(self) -> None:
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    def command(self, line: str, *, _arming: bool = False) -> str:
        """One rigctld command. Refuses to run while the transmitter is up.

        `_arming` is the single exception: the arm gate must read PTT back through
        CAT to learn whether the keying line reaches the radio, and there is no
        other way to learn it.
        """
        if self.keyed_probe() and not _arming:
            raise RigError(
                f"refusing CAT command {line!r} while keyed. Every account of a "
                "stuck transmitter here agrees on this one point.")
        with self._lock:
            if self._sock is None:
                raise RigError("CAT is not open")
            try:
                # `+` is how this protocol asks for a framed reply: a header
                # line, `Name: value` fields, then `RPRT <n>`. A plain reply is
                # the bare value with no terminator at all, so `_read_reply`
                # waits for an `RPRT ` that never comes and the command times out.
                self._sock.sendall(("+" + line + "\n").encode())
                data = self._read_reply()
            except OSError as exc:
                self.close()
                raise RigError(f"CAT command {line!r} failed: {exc}") from None
        return self._checked(line, data)

    def _read_reply(self) -> str:
        """One whole reply, to its `RPRT` terminator.

        Reads until the terminator rather than until the buffer is full, and keeps
        anything past it for the next call. A fixed-size read is what desynchronises
        the stream on the first large reply, and a desynchronised stream is worse
        than a closed one because every answer still looks like an answer.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            head, _, rest = self._buf.partition(b"RPRT ")
            term, sep, tail = rest.partition(b"\n")
            if sep:
                self._buf = tail
                return (head + b"RPRT " + term).decode(errors="replace")
            # A whole reply needs the terminator AND the newline ending its line.
            # Testing only for `RPRT ` and then RECURSING on a missing newline is
            # what this replaced: the recursion restored `_buf` byte for byte, so
            # its own guard was false at once and it took the same branch again --
            # a partial TCP read from rigctld, which is ordinary, became
            # RecursionError. That escapes `command`'s `except OSError` and every
            # `except RigError` above it, on the path that arms and keys. One
            # `deadline` spans the retries here; the recursion started a fresh one
            # each time and so could never expire.
            try:
                if time.monotonic() > deadline:
                    raise TimeoutError      # arriving, but never ending
                chunk = self._sock.recv(65536)
            except TimeoutError:
                # A daemon that answers plain sends `7097500` and stops. There is
                # no terminator coming, so the read ends at its own deadline rather
                # than in whatever the caller does next.
                self.close()
                raise RigError("rigctld did not finish a reply within "
                               f"{self.timeout:g} s") from None
            if not chunk:
                self.close()
                raise RigError("rigctld closed the connection")
            self._buf += chunk

    @staticmethod
    def _checked(line: str, data: str) -> str:
        """rigctld reports failure in band, as `RPRT <negative>`, and returns it
        instead of raising. An unchecked reply is an accepted write that did
        nothing."""
        for part in data.splitlines():
            if part.startswith("RPRT "):
                try:
                    code = int(part.split()[1])
                except (IndexError, ValueError):
                    continue
                if code < 0:
                    raise RigError(f"rigctld refused {line!r}: {part.strip()}")
        return data.strip()

    @staticmethod
    def _field(data: str, name: str) -> str | None:
        """The value of one `Name: value` line of a framed reply.

        By name rather than by position: `m` answers with both Mode and Passband,
        and an index into the lines is a promise about a reply's shape that the
        next hamlib is free to break.
        """
        for ln in data.splitlines():
            key, sep, value = ln.partition(":")
            if sep and key.strip() == name:
                return value.strip()
        return None

    # -- the things we ask a radio -----------------------------------------

    def model_name(self) -> str:
        out = self.command("\\dump_caps")
        for ln in out.splitlines():
            if "Model name:" in ln:
                return ln.split(":", 1)[1].strip()
        return ""

    def freq(self) -> int:
        got = self._field(self.command("f"), "Frequency")
        if got is None:
            raise RigError("rigctld answered `f` without a Frequency field")
        return int(float(got))

    def set_freq(self, hz: int) -> None:
        self.command(f"F {int(hz)}")

    def set_mode(self, mode: str, passband: int = 0) -> None:
        self.command(f"M {mode} {passband}")

    def ptt_state(self, *, _arming: bool = False) -> bool | None:
        """What the radio says about its own PTT, or None if it will not say.

        On a rig configured for a serial keying line, hamlib answers this from its
        own view of that line rather than by asking the radio, and if the port is
        closed it reports off unconditionally. So a `False` here is weak evidence
        and a `True` is strong.

        `None` is the ordinary answer on this station: a daemon started with
        `ptt_type=None` — the only kind `Rig._check_ptt_owner` accepts — has no PTT
        to report and answers `RPRT -11`. `ptt_readable` records that, so a caller
        can distinguish a daemon that says nothing from one that says down.
        """
        try:
            out = self.command("t", _arming=_arming)
        except RigError:
            self.ptt_readable = False
            return None
        state = {"0": False, "1": True}.get(self._field(out, "PTT") or "")
        self.ptt_readable = state is not None
        return state


class Rig:
    """One radio: CAT on rigctld, PTT on its own line, and no way to key
    without a checked emission."""

    def __init__(self, *, model: str, cat: Cat, ptt: Ptt, profile: Profile,
                 control: Control, mycall: str = "", transmit: bool = False,
                 max_key_s: float = 30.0) -> None:
        self.model = model
        self.cat = cat
        self.ptt = ptt
        self.profile = profile
        self.control = control
        self.mycall = mycall
        #: A hard interlock, not a preference. False and nothing can key.
        self.transmit = transmit
        self.max_key_s = max_key_s

        self._armed = False
        self._retired_why = ""
        self._panicked = False
        self._panic_confirmed: bool | None = None
        self._seq = 0
        self._keyed = threading.Event()
        self._keyed_since = 0.0
        self._timer: threading.Timer | None = None
        self._dial_hz = 0
        self._claimed = False
        self._lock = threading.Lock()
        #: Held only by the panic path. A lock no caller and no CAT path takes
        #: cannot reintroduce the deadlock the panic path exists to avoid, and a
        #: check-then-set on a bare flag lets two threads both close the same fd.
        self._panic_lock = threading.Lock()

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

    def arm(self, *, prove_ptt: bool = True) -> ArmReport:
        """Everything that must be true before this object may key.

        Ordered so the keyed proof happens last, after the radio has been
        identified, the dial read back, and the transmitter confirmed idle —
        keying an unverified radio is the expensive mistake the earlier checks
        exist to prevent.

        The CAT socket is opened here rather than by the caller. Three callers
        opened the keying line and none opened this one, so every path failed
        against a real radio with "CAT is not open" — a rig with nothing attached
        fails earlier, on the serial port, which is why it took a radio to see it.
        `Cat.open` is idempotent and says plainly when no daemon is listening.
        """
        self.cat.open()
        if prove_ptt:
            # arm() keys, so it is subject to every interlock a burst is.
            # This module claims "nothing keys while transmit = false"; that was
            # false of arm(prove_ptt=True), and closed only by both callers
            # happening to check first — precisely the convention it refuses to
            # rely on anywhere else.
            if not self.transmit:
                raise RigError(
                    "proving the PTT line keys the radio, and transmit = false. "
                    "Arm with prove_ptt=False, or enable it deliberately.")
            if self.retired:
                raise RigError(f"this rig is retired: {self._retired_why}")

        self.cat.open()
        notes: list[str] = []
        if not self.cat.trusted:
            notes.append("rigctld would not disable its cache, so `t` reports "
                         "hamlib's memory of its own last command rather than the "
                         "radio. PTT readback is not evidence on this daemon.")

        want = rigs.RIGS.get(self.model, {}).get("note", self.model)
        got = self.cat.model_name()
        if not got:
            raise RigError(
                "rigctld would not say what radio it is driving. Keying an "
                "unidentified rig is the expensive mistake this check exists for.")
        if self.model not in got.lower().replace("-", "").replace(" ", ""):
            raise RigError(
                f"rigctld reports {got!r}, configured for {self.model!r}. Keying "
                f"the wrong radio is the expensive mistake. ({want})")

        f = self.cat.freq()
        if not 1_000_000 <= f <= 60_000_000:
            raise RigError(
                f"rigctld reports {f} Hz, which is not an HF dial. A wrong serial "
                "port answers plausibly enough to be worth checking.")
        self._dial_hz = f

        if self.cat.ptt_state(_arming=True) is True:
            raise RigError("the radio is already transmitting. Someone else has it.")

        proven = owner_checked = False
        if prove_ptt:
            # No callsign is required for the proof: keying a sideband
            # transmitter with no audio emits nothing to identify — see
            # _prove_ptt.
            owner_checked = self._check_ptt_owner(notes)
            proven = self._prove_ptt(notes)
        else:
            notes.append("PTT proof skipped — the line has not been shown to "
                         "reach the radio.")

        if self.retired:
            # _prove_ptt's finally can panic, which retires. Returning a report
            # that reads like success for a rig that will refuse every burst is
            # worse than raising.
            raise RigError(f"the rig retired while arming: {self._retired_why}")
        self._armed = True
        return ArmReport(got or self.model, self._dial_hz, proven, owner_checked,
                         tuple(notes))

    def _check_ptt_owner(self, notes: list[str]) -> bool:
        """Whether rigctld also holds our keying line — asked, not probed.

        This used to find out by sending `T 1` and watching our own line move. Two
        things were wrong with that. It is a CAT *write*, which this module's
        foundational rule forbids; and on a daemon configured for CAT PTT it
        genuinely keys the transmitter, unidentified and with no watchdog armed —
        and if the process dies between that write and the `finally`, it is the one
        keyed state neither our ioctl nor HUPCL can reach, because rigctld owns
        that port.

        Asking costs nothing and says more: it distinguishes "not our line" from
        "no PTT at all".
        """
        try:
            out = self.cat.command("\\get_conf ptt_type", _arming=True)
        except RigError as exc:
            notes.append(f"rigctld would not say how it keys ({exc}); PTT line "
                         "ownership is unverified.")
            return False
        kind = ""
        for tok in out.split():
            if tok.startswith("ptt_type="):
                kind = tok.split("=", 1)[1].strip().lower()
        if kind in ("", "none"):
            return True
        raise RigError(
            f"rigctld is configured to key the radio itself (ptt_type={kind!r}). "
            "This station owns the keying line, and two owners of one transmitter "
            "will key it when nobody asked. Start rigctld with "
            "--set-conf=ptt_type=None.")

    def _prove_ptt(self, notes: list[str]) -> bool:
        """Assert the line and check the radio agrees it is transmitting.

        **This is a keying test, not an identification.** An earlier version sent
        the callsign in Morse on the RTS line and claimed that discharged §97.119.
        It does not: the rig's configured mode is PKTUSB, and keying a
        sideband transmitter with no audio produces no RF at all — so nothing
        identifiable went out, and nothing went out. Identification is the
        arbiter's job, through `cwid.audio()` and the transmit path.

        So the proof is short and its deadline comes from its own length, the way a
        burst's does. Arming two seconds and then transmitting for four is how the
        watchdog came to fire mid-proof, retire the rig, and leave the loop keying
        a released port.

        **It cannot come back proven on this station's daemon**, and that is not a
        fault in the wiring. `_check_ptt_owner` accepts only `ptt_type=None`, and a
        rig with no PTT of its own answers `t` with ENAVAIL — so the one
        configuration where the keying line has a single owner is the one where CAT
        has nothing to say about it. The note below says so in those words, because
        an unreadable PTT and a PTT that reads down mean opposite things.
        """
        hold = 0.20
        self._keyed.set()
        self._keyed_since = time.monotonic()
        self._arm_watchdog(hold + _WATCHDOG_SLACK_S, "arm-gate PTT proof")
        try:
            self.ptt.assert_(True)
            time.sleep(0.05)
            state = self.cat.ptt_state(_arming=True)
            time.sleep(max(0.0, hold - 0.05))
            if state is True:
                return True
            if state is False:
                notes.append(
                    "our line is up and the radio reports PTT down. Either the PTT "
                    "cable is on the wrong interface, or rigctld cannot see this "
                    "line — check which before believing either." if self.cat.trusted
                    else "the radio reports PTT down, but this daemon serves `t` "
                         "from its own cache, so that is not evidence.")
            elif not self.cat.ptt_readable:
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
            # Deassert first, and only cancel the watchdog once the line is
            # confirmed down. A failure here is the case this module exists for, so
            # it escalates rather than propagating out of a finally and leaving
            # `keyed` set with no timer left.
            try:
                self.ptt.assert_(False)
                if self.ptt.sense() is True:
                    self.panic_unkey("the line stayed up after the arm-gate proof")
                else:
                    self._keyed.clear()
                    self._cancel_watchdog()
            except PttError as exc:
                self.panic_unkey(f"the line refused to deassert during the proof: {exc}")

    # -- tuning ------------------------------------------------------------

    def tune(self, centre_hz: float) -> None:
        """Point the radio at a published channel centre.

        The dial is derived, never configured. Re-read afterwards because it does
        not stay where you left it — a real and repeated observation, not caution.
        """
        if self.keyed:
            raise RigError("refusing to retune while keyed")
        dial = band.dial_hz(centre_hz)
        self.cat.set_freq(dial)
        self._dial_hz = self.cat.freq()
        if abs(self._dial_hz - dial) > band.QSY_TOLERANCE_HZ:
            raise RigError(
                f"asked for dial {dial} Hz, the radio reports {self._dial_hz}. "
                "Believe the readback.")

    def dial(self) -> int:
        if self.keyed:
            return self._dial_hz
        self._dial_hz = self.cat.freq()
        return self._dial_hz

    # -- keying ------------------------------------------------------------

    def key(self, emission: Emission, *, why: str, duration_s: float,
            settle_s: float = 0.0, responding: bool = False) -> KeyToken:
        """Bring the transmitter up for one burst.

        The emission is checked against the station's profile first, on the dial
        the radio reports rather than the one we asked for.
        """
        # Under the lock, with the keyed flag, or two threads can both pass
        # "already keyed" and both assert — and the loser's unkey() then hits the
        # stale-token early return and silently does nothing while the line is up.
        # panic_unkey never takes this lock, so the documented deadlock hazard is
        # untouched.
        with self._lock:
            if not self.transmit:
                raise RigError(
                    "transmit is disabled for this station. Nothing keys while "
                    "[station] transmit = false, whatever else is configured.")
            if self.retired:
                raise RigError(f"this rig is retired: {self._retired_why}")
            if not self._armed:
                raise RigError("arm() has not passed")
            if self.keyed:
                raise RigError("already keyed")
            self._claimed = True

        checked = Emission(self.dial(), emission.audio_lo_hz, emission.audio_hi_hz,
                           power_w=emission.power_w, designator=emission.designator,
                           technique=emission.technique,
                           symbol_rate_bd=emission.symbol_rate_bd,
                           drift_hz=emission.drift_hz)
        try:
            self.profile.check(checked, control=self.control, responding=responding)
        except NotPermitted as exc:
            raise RigError(f"refused by {self.profile.name}: {exc}") from None

        deadline = self._deadline(duration_s, settle_s)
        with self._lock:
            self._seq += 1
            token = KeyToken(self._seq, time.monotonic(), deadline, why)
        # Armed before the key goes up, and the flag set before it too: a wedge
        # between the two must look keyed, not idle.
        self._arm_watchdog(deadline, why)
        self._keyed.set()
        self._keyed_since = time.monotonic()
        try:
            self.ptt.assert_(True)
        except PttError as exc:
            self._keyed.clear()
            self._cancel_watchdog()
            self._retire(f"the keying line refused to assert: {exc}")
            raise RigError(f"cannot key: {exc}") from None
        return token

    def _deadline(self, duration_s: float, settle_s: float) -> float:
        """Per burst, not a flat ceiling.

        shrike derives it this way, and the reason is arithmetic: a flat 30 s cap
        lets a 1.25 s burst whose audio never arrives hold an unmodulated carrier
        for twenty-nine seconds of someone else's channel.
        """
        want = settle_s + duration_s + max(_WATCHDOG_SLACK_S,
                                           _WATCHDOG_FRACTION * duration_s)
        return min(want, self.max_key_s)

    def unkey(self, token: KeyToken | None = None) -> None:
        """Bring the key down and confirm it went down."""
        if token is not None and token.seq != self._seq:
            return                      # a stale holder; someone else owns the key
        # Deassert first and cancel afterwards. Cancelling first opens a window
        # where the line is up and no timer is scheduled, which is the same
        # ordering mistake `key()` is careful to avoid three lines above — a
        # KeyboardInterrupt or a MemoryError landing in it leaves a keyed
        # transmitter with nothing left to bring it down. A spurious watchdog
        # after a confirmed-down line costs one extra `T 0`.
        try:
            self.ptt.assert_(False)
        except PttError as exc:
            self.panic_unkey(f"the keying line refused to deassert: {exc}")
            return
        if self.ptt.sense() is True:
            self.panic_unkey("the keying line reads high after deassert")
            return
        self._keyed.clear()
        self._cancel_watchdog()

    # -- panic -------------------------------------------------------------

    def panic_unkey(self, why: str) -> None:
        """Every independent path down, in order, holding no lock.

        No `keyed` guard anywhere on this path: a guard that can suppress the
        panic is not a safety mechanism.
        """
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
        for attempt in range(PANIC_ATTEMPTS):
            if time.monotonic() - started > PANIC_BUDGET_S:
                break
            try:
                # A *fresh* connection. The persistent one is what may be wedged.
                with socket.create_connection((self.cat.host, self.cat.port), 0.4) as s:
                    s.settimeout(0.4)
                    s.sendall(b"T 0\n")
                    s.recv(256)
                    s.sendall(b"t\n")
                    if s.recv(256).decode(errors="replace").strip().startswith("0"):
                        confirmed = True
                        break
            except OSError:
                time.sleep(0.05 * (attempt + 1))

        self._keyed.clear()
        self._panic_confirmed = confirmed
        if not confirmed:
            os.write(2, b"!! UNKEY NOT CONFIRMED. CHECK THE RADIO AND REMOVE POWER.\n")
            if not self.cat.ptt_readable:
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

    def close(self) -> None:
        self._cancel_watchdog()
        try:
            self.ptt.release()
        finally:
            self.cat.close()
