# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keying the transmitter, and bringing it down.

PTT is a modem control line on a serial port, asserted with one ioctl. On the
FT-891 the CP2105 presents two interfaces: CAT on the first, and the second one's
RTS wired to the rig's PTT input. That split is what lets `rigctld` own the CAT
port exclusively while this module owns the keying line, so there is exactly one
owner of each by construction rather than by convention.

Three things about serial ports will key a radio if you do not know them, and all
three are handled here explicitly rather than inherited.

**Opening a tty raises RTS and DTR.** hamlib carries this in two comments and two
code paths, and the XNU tty layer does it too. So the first ioctl after `open()`
is an unconditional deassert of both lines, before any other configuration. Not
doing this means `arm()` can key the transmitter, and it means anything that
reopens the port mid-session keys it again — which is why `release()` closes and
does not reopen. A "close and reopen to force a deassert" step sounds safe and is
the opposite.

**`HUPCL` governs deassert-on-last-close, and it is not safe to inherit.** hamlib
explicitly clears it (`serial.c`), so a port configured by hamlib and inherited
here would not drop the line when the process dies. We set it, and `CLOCAL` with
it. Note "last close": if anything else holds the port open — a forked child, a
terminal program — our death drops nothing. Hence `O_CLOEXEC`, and the rule that
the station process must not `fork()` without `exec()`.

**`CRTSCTS` gives RTS to the driver as a flow-control signal.** It is cleared, or
the kernel will move the line out from under us.

## What a readback does and does not prove

`sense()` reads `TIOCMGET`, which returns the driver's view of the output lines.
That proves the ioctl was accepted. It does **not** prove a pin moved, that a
cable is attached, or that a rig keyed. The only evidence about the far end is a
CAT readback through `Cat.ptt_state()`, and even that is only as good as the
radio's own reporting.

## Kill path status

Deasserting on our own fd is immediate and reliable. Deasserting **because the
process died** is the only mechanism here that survives SIGKILL, OOM and panic —
and it is **UNPROVEN on this hardware.** `HUPCL` is set, which is necessary and
may not be sufficient: the CP2105 driver's behaviour on last close has not been
measured, and it cannot be measured with a pty, because Darwin returns `ENOTTY`
for `TIOCMGET` on both ends of one. Proving it needs a second USB-serial adapter
with its CTS jumpered to this port's RTS, read from another process. Until that
exists, treat process death as a hope rather than a guarantee, and do not let the
documentation imply otherwise.
"""
from __future__ import annotations

import fcntl
import re
import os
import stat
import struct
import subprocess
import termios
import time
from dataclasses import dataclass
from typing import Callable, Protocol

#: The two output lines a serial PTT can use. RTS is the convention here.
_RTS = termios.TIOCM_RTS
_DTR = termios.TIOCM_DTR


class PttError(Exception):
    """The keying line could not be reached. Always fatal to a transmission."""


class Ptt(Protocol):
    """A transmit-enable line."""

    def assert_(self, on: bool) -> None:
        """Key or unkey. Raises `PttError` if the ioctl was refused."""
        ...

    def sense(self) -> bool | None:
        """Our line as the driver reports it; None if unknowable.

        Evidence that the ioctl landed, not that a rig keyed.
        """
        ...

    def release(self) -> None:
        """Drop the line and let go of the port. Never reopens, never raises."""
        ...


def require_char_device(path: str) -> None:
    """Refuse, loudly and by name, a path with no keying line behind it.

    A stat and not an open, because probing a keying line by opening it asserts
    RTS — that is a key-down. `RtsPtt.open` runs it so a wrong path dies naming
    itself rather than as a failed ioctl, and the transmit tools run it at their
    arm gates, before a session starts: on 2026-08-10 a placeholder default —
    ``/dev/cu.usbserial-XXXXB1``, a path that exists on no machine — rode through
    a full 27-cycle session that logged 22.1 s of carrier while never touching
    the radio, because nothing between the flag and the ioctl had ever asked
    whether the line was there.

    Construction stays permissive on purpose: preflight builds an `RtsPtt` on
    whatever the config names and reports the refusal from `open()` as one line
    of its findings, which a constructor that raises would turn into a crash.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise PttError(f"PTT port {path}: {exc.strerror or exc}") from None
    if not stat.S_ISCHR(mode):
        raise PttError(f"PTT port {path} is not a character device")


def arming_refusal(ptt_device: str | None) -> str | None:
    """Why `--arm` must be refused with this keying line, or None to proceed.

    The one gate every transmit tool runs before arming, kept here so the flag
    names, the stat and the refusal wording cannot drift apart per tool — each
    caller keeps its own exit convention and prints what this returns. On
    2026-08-10 a session was launched armed without `--ptt-device`: rigctld took
    the unkey and never answered, and the last-resort unkey had nothing to pull
    down, so the transmitter sat keyed until the operator was told to unkey by
    hand. A stat and never an open, as `require_char_device` says.
    """
    if not ptt_device:
        return ("--arm requires --ptt-device: without the keying line, nothing "
                "can unkey a transmitter once rigctld stops answering")
    try:
        require_char_device(ptt_device)
    except PttError as exc:
        return (f"REFUSING TO ARM: {exc} — the last-resort unkey would have "
                f"no line to pull down")
    return None


class RtsPtt:
    """RTS on a dedicated serial interface.

    The port is held from `open()` to `release()`. Holding it is deliberate:
    hamlib's model opens the PTT port on the session's *first* key-down, so the
    connect burst — the one burst a called station has to lock onto — pays a
    serial open that no later burst does. Here that cost is paid at arm time.

    `open()` refuses a path that is not a character device before touching it,
    so a wrong port dies naming itself instead of as a failed ioctl.
    """

    def __init__(self, port: str, *, line: str = "rts") -> None:
        if line not in ("rts", "dtr"):
            raise ValueError(f"line must be rts or dtr, got {line!r}")
        self.port = port
        self._bit = _RTS if line == "rts" else _DTR
        self._fd: int | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Open the port with both lines down and the hangup bit set."""
        if self._fd is not None:
            return
        require_char_device(self.port)
        try:
            fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError as exc:
            raise PttError(f"cannot open PTT port {self.port}: {exc}") from None
        try:
            # Before anything else. The open itself has already raised these.
            self._clear(fd, _RTS | _DTR)
            self._configure(fd)
            self._clear(fd, _RTS | _DTR)
        except Exception:
            # This close runs only when an ioctl has already failed — an
            # adapter dying mid-open — which is exactly when a close plausibly
            # raises OSError too. Bare, that OSError replaced the PttError and
            # escaped every `except PttError` in the never-raises contract.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        self._fd = fd

    def _configure(self, fd: int) -> None:
        try:
            attrs = termios.tcgetattr(fd)
        except termios.error as exc:
            raise PttError(f"{self.port} is not a tty: {exc}") from None
        cflag = attrs[2]
        cflag &= ~termios.CRTSCTS      # or the driver owns RTS
        cflag |= termios.HUPCL         # drop the lines on last close
        cflag |= termios.CLOCAL        # no modem-control on open/read
        attrs[2] = cflag
        # `termios.error` subclasses Exception, not OSError: left bare, it
        # sails past `except PttError` in every caller that trusts the
        # never-raises contract, `drop_rts` first among them.
        try:
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except termios.error as exc:
            raise PttError(f"cannot configure {self.port}: {exc}") from None

    #: What the lines read after the last `release()`: True when the driver
    #: confirmed *both* output lines low — the release clears RTS and DTR
    #: together, so a rig keyed by either is covered, and a DTR-keyed rig is
    #: never proved down off the one pin that did not key it. False when either
    #: read high, the readback was refused, *or* the deassert itself was; None
    #: until a release has happened. Unreadable and refused collapse into False
    #: deliberately — only a positive answer is confirmation, and a driver that
    #: will not answer TIOCMGET has not given one, so there is no third state
    #: for a caller to get wrong on the one path where being wrong keys a
    #: transmitter. The panic path needs that answer from *this* line rather
    #: than from CAT, because on the only daemon configuration this station
    #: accepts, rigctld owns no PTT and answers `t` with ENAVAIL — so the CAT leg
    #: could never confirm anything, and the loudest message in the program fired
    #: on every ordinary shutdown.
    released_low: bool | None = None

    def release(self) -> None:
        """Deassert both lines, read them back, then close. **Never reopens** — a
        reopen raises RTS — and **never raises**: it is the last act on a port
        whose adapter may already be gone, and the callers that reach it from
        `finally` and atexit have their own account of the failure to protect.
        A deassert the driver refuses collapses into `released_low = False`,
        the same collapse an unreadable readback gets.

        The readback happens between the clear and the close because that is the
        only moment it can: it needs the descriptor, and the descriptor is gone
        a line later. It covers RTS and DTR both, since both were cleared and
        either can be the one wired to PTT — which line keys a given rig is a
        measurement (2026-07-30: this station's receiver went deaf on RTS and
        not on DTR), not something this readback may assume.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            try:
                self._clear(fd, _RTS | _DTR)
            except PttError:
                self.released_low = False   # a refused deassert confirmed nothing
            else:
                self.released_low = self._sense_fd(fd, _RTS | _DTR) is False
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    # -- the line ----------------------------------------------------------

    def assert_(self, on: bool) -> None:
        if self._fd is None:
            raise PttError(f"{self.port} is not open")
        if on:
            self._set(self._fd, self._bit)
        else:
            self._clear(self._fd, self._bit)

    def sense(self) -> bool | None:
        if self._fd is None:
            return None
        return self._sense_fd(self._fd)

    def _sense_fd(self, fd: int, bits: int | None = None) -> bool | None:
        try:
            packed = fcntl.ioctl(fd, termios.TIOCMGET, struct.pack("I", 0))
        except OSError:
            return None
        return bool(struct.unpack("I", packed)[0]
                    & (self._bit if bits is None else bits))

    # -- ioctls ------------------------------------------------------------

    @staticmethod
    def _set(fd: int, bits: int) -> None:
        try:
            fcntl.ioctl(fd, termios.TIOCMBIS, struct.pack("I", bits))
        except OSError as exc:
            raise PttError(f"TIOCMBIS failed: {exc}") from None

    @staticmethod
    def _clear(fd: int, bits: int) -> None:
        try:
            fcntl.ioctl(fd, termios.TIOCMBIC, struct.pack("I", bits))
        except OSError as exc:
            raise PttError(f"TIOCMBIC failed: {exc}") from None

    def __repr__(self) -> str:
        state = "closed" if self._fd is None else ("asserted" if self.sense() else "down")
        return f"RtsPtt({self.port!r}, {state})"


#: Daemon money. An unkey may spend this long asking a transport before the
#: verdict has to come from somewhere else; the key-up clock below is tighter
#: because failing a key-up costs a burst where failing it quietly costs the
#: slot. One pair of numbers for every modem — each of the three keying paths
#: once carried its own answer, and the answers diverged by accident. These are
#: the bridge's, the pair most recently sized against a real incident: on
#: 2026-07-28 an unkey spent three seconds waiting on one blocked read and so
#: asked exactly once in six — a budget spent that way is a single attempt
#: wearing a retry loop's clothes, and it held an FT-891 keyed 7.84 s past its
#: last sample.
UNKEY_BUDGET_S = 6.0
KEY_BUDGET_S = 3.0
#: Unlike ``T 0``, re-sending ``T 1`` is not free: a rig that answered a refusal
#: has answered, and asking it fifteen more times inside the budget -- the loop
#: sleeps 0.2 s a pass -- buries the
#: refusal it already gave in its own log. Three passes covers a dropped packet
#: or a serial line busy with the last command, which is what a retry is for.
KEY_TRIES = 3


@dataclass(slots=True)
class Attempt:
    """One drive of a transport, reported as what was read back, never what was sent.

    ``confirmed`` — the far end was read back in the commanded state. ``taken`` —
    the transport accepted the command and nothing on that path can read the
    result; real for a one-shot whose whole verdict is its exit status, and said
    out loud rather than dressed as confirmation. ``failed`` carries the split
    the ladder turns on: ``quiet`` is a transport that did not answer — refused,
    silent, cut off mid-verdict — where not-quiet is a transport that answered
    and reported the rig still up, which is worth asking again. ``why`` is one
    sentence naming the failure; ``detail`` is the wire record behind it — the
    bytes or argv, per-read timings, errno, exit status — the instrument that can
    say afterwards which way a daemon went, where the 2026-08-09 log's one
    sentence covered half a dozen failures and grew two days of theory.
    """

    verdict: str                # "confirmed" | "taken" | "failed"
    why: str = ""
    detail: str = ""
    state: bool | None = None   # PTT as read back, where the transport can ask
    state_read: bool = False
    quiet: bool = False


class LineTransport:
    """RTS on a port this process holds: one ioctl, and the driver's own readback.

    Not paced — there is no reply to wait for, and repeating an ioctl the driver
    already applied would only spend time with the rig up — and never the drop
    rung's target: `drop_rts` OPENS a port to work, an open raises RTS, and this
    port is already open with the descriptor in hand.
    """

    place = "line"
    peer = "the keying line"
    paced = False
    reads_back = True

    def __init__(self, device: str) -> None:
        self.device = device
        self._ptt = RtsPtt(device)

    def arm(self) -> None:
        self._ptt.open()

    def drive(self, on: bool, timeout: float | None = None, *,
              confirm_silence: bool = True) -> Attempt:
        try:
            self._ptt.assert_(on)
            state = self._ptt.sense()
        except PttError as exc:
            return Attempt("failed", detail=str(exc), quiet=True)
        if state is on:
            return Attempt("confirmed", state=state, state_read=True)
        return Attempt("failed", state=state, state_read=state is not None)

    def release(self) -> bool | None:
        self._ptt.release()
        return self._ptt.released_low


class OneShotRigctl:
    """``T 0`` as a fresh rigctl per attempt; the exit status is the whole verdict.

    A zero exit is ``taken``, never ``confirmed`` — nothing on this path reads
    PTT back, and the caller is told so. A nonzero exit is how rigctl reports a
    dead daemon, in milliseconds, and `subprocess.run` does not raise for it:
    reading that as success is how a dead daemon once read as a clean unkey, so
    the returncode check lives here for every modem at once. Every failure is
    ``quiet``: with no read-back there is no way to see a transport that is
    alive with the rig merely slow, so nothing a retry could use ever arrives.
    """

    place = "rig"
    peer = "rigctl"
    paced = True
    reads_back = False
    attempt_timeout = UNKEY_BUDGET_S
    taken_note = "rigctl took the command; nothing on this path reads PTT back"

    def __init__(self, argv: list) -> None:
        self.argv = [str(a) for a in argv]

    def drive(self, on: bool, timeout: float, *,
              confirm_silence: bool = True) -> Attempt:
        t0 = time.monotonic()
        name = self.argv[0].rsplit("/", 1)[-1]
        ran = " ".join(self.argv[-2:])
        try:
            r = subprocess.run(self.argv, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return Attempt("failed", quiet=True,
                           why=f"{name} gave no verdict within {timeout:.1f}s",
                           detail=f"ran {ran!r}; killed at {timeout * 1000:.0f}ms")
        except OSError as exc:
            return Attempt("failed", quiet=True,
                           why=f"{name} could not be run",
                           detail=f"{exc}; total "
                                  f"{(time.monotonic() - t0) * 1000:.0f}ms")
        if r.returncode == 0:
            return Attempt("taken")
        said = " ".join((r.stderr or r.stdout or "").split())[-160:]
        return Attempt("failed", quiet=True,
                       why=f"{name} exited {r.returncode}",
                       detail=f"ran {ran!r}; said {said!r}; total "
                              f"{(time.monotonic() - t0) * 1000:.0f}ms")

    def release(self) -> bool | None:
        return None


class Keyer:
    """The keying mechanism, whole: drive a transport, believe only what reads
    back, and bring the transmitter down by every path there is, in order of
    directness.

    Three modems each grew a private answer to the same four questions — how
    many times to retry, whether to believe a write, what an unconfirmed unkey
    means, when to stop trusting the rig — and the stuck-key defect was found
    and fixed in each separately, on different days, because the answers lived
    in three places. They live here now. A Rig supplies only what is genuinely
    its own: the transport (a serial line, a rigctld socket, one-shot rigctl)
    and, through ``on_retire``, what being unfit to key is called in its world.

    THE UNKEY LADDER — try the transport; confirm by reading back, because an
    accepted write is not a transmitter that dropped; retry inside one bounded
    budget; fall to the local ioctl that needs no daemon and no reply; alarm in
    the words an operator reads at 3 a.m.; and retire the rig. With one split,
    measured rather than designed:

    **A transport that answers keeps the budget. A transport that goes quiet
    forfeits it after one ask, when a keying line is on hand.** A daemon that
    answers ``T 0`` and still reports the rig keyed is demonstrably alive with
    the delay at the radio — repeating the write is the only medicine, and the
    budget bounds how long. But on 2026-08-13 every VARA attempt held ~8.4 s of
    carrier while four ``T 0``s in a row were accepted and answered with
    silence, 751 ms each and a further 751 ms apiece chasing them with ``t`` — and
    the line then read LOW in microseconds, first try, four of four across that
    night's rig-session records, two drops each in sessions 191812 and 192045.
    Silence is not a slow answer: nothing a retry could use ever comes back,
    and the ioctl sits right there. So one quiet ask spends the daemon's turn;
    the line is dropped, read back on RTS *and* DTR, and the rig is retired —
    whatever keyed it has stopped answering, and a transmitter nobody can
    confirm down is not one to key again. Only when the drop fails, or there
    is no line, does the budget go back to repeating the write, which is then
    the only move left.

    `key` and `unkey` never raise and take no lock: a forced unkey runs on
    whatever thread caught the signal, and waiting there is a deadlock with the
    transmitter keyed. Good news goes to `log`, alarms to `alarm` — one
    callable for a tool that prints, two for a Rig that logs at levels.
    ``held`` reports when the key went up, so a failed attempt states how long
    the carrier has been standing; ``abort`` lets an owner end a key-up that a
    forced unkey has overtaken.
    """

    def __init__(self, transport, log: Callable[[str], None],
                 alarm: Callable[[str], None] | None = None, *,
                 ptt_device: str | None = None,
                 held: Callable[[], float] | None = None,
                 abort: Callable[[], bool] | None = None,
                 on_retire: Callable[[], None] | None = None) -> None:
        self.transport = transport
        self._log = log
        self._alarm = alarm if alarm is not None else log
        #: The line `drop_rts` can pull when the transport is not the line
        #: itself. None for a CAT-keyed rig — no line exists — and None for a
        #: line transport, which unkeys on the descriptor it already holds.
        self.ptt_device = ptt_device
        self._held = held
        self._abort = abort
        self._on_retire = on_retire
        #: Set once an unkey could not be confirmed, or the transport that keys
        #: this rig stopped answering. The Rig that owns this keyer keeps its
        #: own retire semantics; this is the verdict it acts on.
        self.must_retire = False

    def arm(self) -> None:
        """Open the line — `RtsPtt.open` deasserts before it does anything
        else — and say who keys and who tunes. Raises `PttError` as `open`
        does: arming is the one moment a refusal is cheap."""
        self.transport.arm()
        self._log(f"PTT is the keying line on {self.transport.device}; "
                  f"CAT tunes only")

    def _retire(self) -> None:
        self.must_retire = True
        if self._on_retire is not None:
            self._on_retire()

    # -- key ---------------------------------------------------------------

    def key(self, on: bool, note: str = "") -> bool:
        """Drive; True only for a readback that confirms, or an acceptance on a
        path with nothing to read back — and then the log says unverified.

        An unconfirmed key-UP ends by dropping PTT to be sure: the command may
        have raised the key even though nothing confirmed it, and a watchdog
        gated on the caller's own bookkeeping would never fire on a line it
        does not believe is up. On the line that is one deassert; through a
        daemon it is the whole unkey ladder, because a socket that went silent
        under ``T 1`` says nothing about what the rig did with it. A failed
        *pipe* write needs no drop and gets none — proof of non-delivery is the
        one failure that leaves nothing keyed — which is why the pipe-keyed
        Rigs answer their own key-ups and come here for the ladder.
        """
        if not on:
            return self.unkey()
        tag = f" ({note})" if note else ""
        if not self.transport.paced:
            a = self.transport.drive(True)
            if a.verdict == "confirmed":
                self._log(f"PTT ON -> line{tag}")
                return True
            if a.detail:
                self._alarm(f"keying line: {a.detail}")
            self._alarm(f"*** PTT ON NOT CONFIRMED — the line did not go high; "
                        f"NOT TRANSMITTING{tag}; dropping PTT to be sure ***")
            self.key(False)
            return False
        end = time.monotonic() + KEY_BUDGET_S
        attempt, last = 0, ""
        while True:
            if self._abort is not None and self._abort():
                self._log("PTT ON abandoned — this rig is being taken down")
                return False
            attempt += 1
            per = max(0.05, min(self.transport.attempt_timeout,
                                end - time.monotonic()))
            a = self.transport.drive(True, per)
            if a.verdict in ("confirmed", "taken"):
                self._log(f"PTT ON -> {self.transport.place}{tag}")
                return True
            last = a.why
            self._log(f"PTT ON attempt {attempt}: {last} [{a.detail}]")
            if attempt >= KEY_TRIES or time.monotonic() >= end:
                break
            time.sleep(0.2)
        # An accepted-looking silence is not a key-up, but it is not proof the
        # rig stayed down either, so this puts it down rather than walking away.
        self._alarm(f"*** PTT ON NOT CONFIRMED after {attempt} attempt(s) "
                    f"({last}) — NOT TRANSMITTING{tag}; dropping PTT to be "
                    f"sure ***")
        self.unkey()
        return False

    # -- the unkey ladder --------------------------------------------------

    def unkey(self, budget: float | None = None,
              attempt_timeout: float | None = None) -> bool:
        """The ladder. True when the transmitter is down or was confirmed down
        on the line itself; False is the alarm already raised and the rig
        already retired. Never raises, never waits on a lock.
        """
        if not self.transport.paced:
            a = self.transport.drive(False)
            if a.verdict == "confirmed":
                self._log("PTT OFF -> line (reads low)")
                return True
            if a.detail:
                self._alarm(f"keying line: {a.detail}")
            self._alarm("*** PTT OFF NOT CONFIRMED — the line will not go low; "
                        "TRANSMITTER MAY BE STUCK. Unkey it manually now. ***")
            self._retire()
            return False
        budget = UNKEY_BUDGET_S if budget is None else budget
        per_cap = (self.transport.attempt_timeout if attempt_timeout is None
                   else attempt_timeout)
        end = time.monotonic() + budget
        attempt, last = 0, ""
        drop_left = bool(self.ptt_device)
        while True:
            attempt += 1
            per = max(0.05, min(per_cap, end - time.monotonic()))
            # When a line is still in hand, a silent ``T 0`` is not chased with
            # a read-back: the ``t`` would spend another timeout of carrier
            # asking the component that just went quiet, and the line's own
            # readback is moments away and more direct.
            a = self.transport.drive(False, per, confirm_silence=not drop_left)
            if a.verdict == "confirmed":
                self._log(f"PTT OFF -> {self.transport.place}")
                return True
            if a.verdict == "taken":
                self._log(f"PTT OFF -> {self.transport.place} "
                          f"({self.transport.taken_note})")
                return True
            last = a.why
            parts = [a.detail]
            if self.transport.reads_back and a.state_read:
                parts.append(f"t -> {'1' if a.state else 'no answer'}")
            if self._held is not None:
                since = self._held()
                parts.append(f"PTT held {time.time() - since:.1f}s" if since
                             else "PTT assert time unknown")
            self._log(f"PTT OFF attempt {attempt}: {last} "
                      f"[{'; '.join(p for p in parts if p)}]")
            if a.quiet and drop_left:
                drop_left = False
                if self._drop(last):
                    return True
            if time.monotonic() >= end:
                if drop_left and self._drop(last):
                    return True
                self._retire()
                self._alarm(f"*** PTT OFF NOT CONFIRMED in {budget:.1f}s "
                            f"({last}) — TRANSMITTER MAY BE STUCK. Unkey it "
                            f"manually now. ***")
                return False
            time.sleep(0.05)

    def _drop(self, why: str) -> bool:
        """The local rung, and the retire that goes with it either way: whatever
        keys this rig has stopped answering, and that ends the session whatever
        the line reads. When the line reads LOW the message must not send the
        operator to the radio — the ioctl went to the driver, not over the link
        that just failed, and reading the line low is better evidence than the
        reply that never came."""
        dropped = drop_rts(self.ptt_device, self._log)
        self._retire()
        if dropped:
            self._alarm(f"*** {self.transport.peer} stopped answering ({why}) "
                        f"— PTT CONFIRMED DOWN on the line itself. Nothing "
                        f"further will be transmitted through this rig. ***")
        return dropped

    # -- hand back ---------------------------------------------------------

    def hand_back(self) -> None:
        """Deassert, read back, close — `RtsPtt.release` never reopens, because
        a reopen raises RTS and that is key-down — and give the verdict. A
        transport with no line of its own releases silently."""
        low = self.transport.release()
        if low is None:
            return
        if low:
            self._log("keying line handed back — reads LOW (PTT down)")
        else:
            self._alarm("*** keying line handed back NOT CONFIRMED LOW — "
                        "THE TRANSMITTER MAY STILL BE UP ***")


class LineKeyer(Keyer):
    """A `Keyer` on the line itself — the shape both line-keyed Rigs hold for a
    session. The safety wordings and the drop-to-be-sure live in `Keyer`; both
    Rigs carried their own copies once, until the copies began to diverge and
    one of them lost the deassert on an unconfirmed key-up doing it."""

    def __init__(self, device: str, log: Callable[[str], None],
                 alarm: Callable[[str], None] | None = None) -> None:
        super().__init__(LineTransport(device), log, alarm)
        self.device = device


#: A CP2105 enumerates one device node per interface, distinguished by a trailing
#: interface letter and index on a shared serial number: `...usbserial-<serial>B0`
#: carries CAT (Enhanced), `...B1`'s RTS is the rig's PTT input (Standard).
#:
#: The pattern is deliberately narrow. A looser rule that advanced any trailing
#: digit would turn `/dev/ttyUSB0` into `/dev/ttyUSB1`, and on Linux those are two
#: *different adapters* — quite possibly another radio's CAT port. Pointing a
#: keying line at hardware nobody asked about is worse than refusing to guess.
_CP210X = re.compile(r"^(?P<head>.*usbserial-[0-9A-Za-z]+B)(?P<iface>\d)$")


def derive_ptt_port(cat_port: str) -> str:
    """The PTT interface of a two-interface adapter, given its CAT interface.

    Only derives for the naming convention it recognises; anything else must be
    named explicitly. Even when it does derive, the result is a suggestion a human
    should confirm — the arm gate proves the line before anything transmits.
    """
    m = _CP210X.match(cat_port or "")
    if m:
        return f"{m['head']}{int(m['iface']) + 1}"
    raise ValueError(
        f"cannot derive a PTT port from {cat_port!r} — name the keying port "
        "explicitly. Guessing which line keys a transmitter is not a default, "
        "and a wrong guess can key a different radio.")


def drop_rts(device: str | None, log: Callable[[str], None]) -> bool:
    """Take the keying line down without asking a daemon to. True when it reads low.

    The escape hatch from a key that went through hamlib, and the record of the
    2026-08-09 incident it exists for. Four unkeys timed out and one session hung
    after its first burst, all at 50 W into a real antenna, and none of it
    reproduced off the air -- CAT alone answered 40 of 40, key/unkey without audio
    15 of 15, CAT under a streaming codec 30 of 30. The same daemon on the same
    hardware has since answered ``T 0`` in 58 ms and ``t`` in 3 ms with zero
    failures, holding a 64 ms median under 351 concurrent CAT operations -- so the
    daemon is not fragile, CAT traffic does not starve PTT, and transmitting is
    the only condition that has ever produced the failure. Why remains
    unestablished. Whatever the mechanism, the path that has to work when the
    daemon has stopped answering must not travel the link that fails only then,
    and this one does not: `RtsPtt.open` deasserts before it does anything else,
    and `release` deasserts again, reads the line back, and closes.

    Never raises. It runs after everything else has already failed, and an exception
    here would lose the caller's own account of why. `release` shares the promise,
    so an adapter that vanishes between the open and the deassert is a False and a
    log line, not a traceback over the original failure.

    Both output lines are taken down and both are read back: which one is wired to
    PTT differs per rig, and a rig keyed by DTR must not be proved down off RTS.

    Only for that case. Opening the port is what makes this work and also what makes
    it unsuitable as a routine unkey: a station that owns its keying line should hold
    an `RtsPtt` open and use `assert_` instead.
    """
    if not device:
        log("no PTT device known — cannot take the keying line down directly")
        return False
    ptt = RtsPtt(device)
    try:
        ptt.open()
    except PttError as exc:
        log(f"keying line: {exc}")
        return False
    ptt.release()
    # Two verdicts, two stems: the failure is read at the moment a transmitter
    # may be stuck, and it must not open by asserting the outcome it retracts.
    if ptt.released_low:
        log(f"keying line on {device} taken down directly — RTS and DTR read "
            f"LOW (PTT down)")
        return True
    log(f"NOT CONFIRMED LOW: the keying line on {device} was commanded down "
        f"but did not read back low — the transmitter may still be keyed")
    return False
