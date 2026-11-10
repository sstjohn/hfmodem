#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""This tree's rigctld client, with bounded keying."""
from __future__ import annotations

import atexit
import os
import queue
import signal
import socket
import sys
import threading
import time

from hfmodem.core.config import ConfigError, station_profile
from hfmodem.core.ptt import LineKeyer, drop_unheld_line
from hfmodem.core.rig import FATAL_SIGNALS, arm_deadman

# A VARA over is seconds long (the longest observed BW2300 DATA over is ~8.6 s), so
# a continuous key beyond this is a stuck transmitter, not traffic.
MAX_KEY_S = 25.0

# What the rig's own meters say while keyed, read through rigctld beside the keying
# line: radiated power in watts and ALC. Until 10-23 they were read only at rest,
# where they say nothing, and power was an estimate.
TX_METERS = ('RFPOWER_METER_WATTS', 'ALC')
METER_PERIOD_S = 0.25
# A key-down shorter than this ends inside the rig's power attack at the CAT read
# cadence: 0.34 s bursts all read ~0.1 W and 11-symbol ones scattered 1.3-22 W
# on a 50 W station, so their power is unmeasured rather than a number.
MIN_METERED_KEY_S = 0.5


_ID_INTERVAL_S = 600.0            # identification reminder interval

_DEADMAN_S = 2.0


def _id_reminders_owed(log) -> bool:
    """Whether the selected profile enables identification reminders.

    Selecting `unregulated` disables this reminder. An absent or unreadable
    station file keeps the reminder enabled.
    """
    try:
        declared = station_profile()
    except ConfigError as exc:
        log(str(exc))
        return True
    return declared is None or declared.name != "unregulated"


class Rig:
    def __init__(self, hostport: str | None, armed: bool, log,
                 ptt_device: str | None = None, line_ptt: bool = False):
        if armed and not (line_ptt and ptt_device):
            raise ValueError("armed rig requires --line-ptt and --ptt-device")
        self.log = log
        # The serial line whose RTS IS the PTT, when there is one.
        self.ptt_device = ptt_device
        self.released_low: bool | None = None
        self._keyer = None
        self._armed = False
        self._first_key = None        # when this station first keyed
        self._id_last = None          # last identification reminder
        self._id_owed = _id_reminders_owed(log)
        self.addr = None
        if hostport:
            host, _, port = hostport.partition(":")
            self.addr = (host or "127.0.0.1", int(port or 4533))
        self.keyed = False
        self.key_since = 0.0
        self._unkey_at = 0.0
        self._lock = threading.Lock()     # serialises key-ups; the unkey path never takes it
        self._no_tx = threading.Event()   # set once this rig is being taken down for good
        self._panicked = False
        self._stop = threading.Event()
        self._metering, self._unkeyed = threading.Event(), threading.Event()
        self._key_ups = 0
        self.meter_reports: queue.SimpleQueue = queue.SimpleQueue()
        self.armed = armed
        threading.Thread(target=self._watchdog, daemon=True).start()
        if self.addr is not None:
            threading.Thread(target=self._meter, daemon=True).start()

    # -- rigctld ---------------------------------------------------------------
    def _cmd(self, text: str, timeout: float = 3.0, want: int = 4096, *, whole: bool = False) -> str:
        """One command on its own connection; return the reply received.

        Every byte this program sends rigctld goes through here — tests rely on
        that to prove a path never touched the daemon. A read timeout returns
        the reply so far; other socket errors propagate to the caller.

        rigctld can reset a connection instead of replying: on 2026-10-22 one
        reset the mode set of a VarAC slot QSY just after an unkey, and the
        session died with the slot unreached. Every command sent here is a read
        or an idempotent set, so a reset is asked once more on a fresh connection.
        `whole` reads to the close, for a batch that ends in `q`.
        """
        if self.addr is None:
            return ""
        try:
            return self._ask(text, timeout, want, whole)
        except ConnectionResetError:
            self.log(f"rigctld reset the connection on {text.strip()!r}; asking once more")
            return self._ask(text, timeout, want, whole)

    def _ask(self, text: str, timeout: float, want: int, whole: bool = False) -> str:
        received = bytearray()
        s = socket.create_connection(self.addr, timeout=timeout)
        try:
            s.sendall(text.encode())
            s.settimeout(timeout)
            while True:
                try:
                    chunk = s.recv(want)
                except TimeoutError:
                    break
                if not chunk:
                    break
                received.extend(chunk)
                if len(chunk) < want and not whole:
                    break
            return received.decode(errors="replace")
        finally:
            s.close()

    def model(self) -> str:
        """The rig's own name for itself, found by label in ``\\dump_caps``.

        One command and nothing else on the wire: the arm gate wants a name, not
        a frequency, and CAT traffic on that path is not free. ``"?"`` when the
        caps come back short or not at all — the caller decides what an
        unreadable rig means.
        """
        for line in self._cmd("\\dump_caps\n").splitlines():
            if "Model name:" in line:
                return line.split(":", 1)[1].strip()
        return "?"

    def identify(self) -> tuple[str, str]:
        """(model, frequency) as reported by the rigctld we were actually given.

        Either can come back ``"?"``: a daemon that takes the command and never
        answers — CAT wedged, radio off — used to make the frequency read an
        IndexError, which took a live vara-mail attempt down as a traceback on
        2026-08-03. The caller decides what an unreadable rig means; this only
        reports it.
        """
        model = self.model()
        lines = self._cmd("f\n").strip().splitlines() if self.addr else []
        return model, (lines[0] if lines else "?")

    def state(self) -> dict:
        """Frequency, mode and passband, asked in rigctld's one-line form (';').

        Bare ``m`` answers on two lines — mode, then passband — which desynchronises
        a reader that takes one line per command: the stray line becomes the next
        command's reply.
        """
        def ask(command: str) -> dict:
            fields = self._cmd(f";{command}\n").split(";")
            return dict(f.split(": ", 1) for f in fields if ": " in f)

        freq, mode = ask("f").get("Frequency"), ask("m")
        passband = mode.get("Passband", "")
        return {
            "frequency_hz": int(freq) if freq and freq.lstrip("-").isdigit() else None,
            "mode": mode.get("Mode"),
            "passband_hz": int(passband) if passband.isdigit() else None,
        }

    def batch(self, text: str, timeout: float = 5.0) -> str:
        """An extended-framing batch closed by `q`, on one connection, read to its end."""
        return self._cmd(text, timeout, whole=True)

    def set_freq(self, hz: float) -> str:
        return self._cmd(f"F {int(hz)}\n")

    def set_mode(self, mode: str, passband: int = 0) -> str:
        return self._cmd(f"M {mode} {int(passband)}\n")

    def power_w(self, watts: float | None = None, scale_w: float = 100.0) -> float | None:
        """RFPOWER in watts, 1.0 being `scale_w`, read back after setting it to `watts`
        when given; None when the rig does not answer. The readback is the verdict on
        the write, so a refused set shows there."""
        if watts is not None:
            self._cmd(f"L RFPOWER {watts / scale_w:g}\n", want=64)
        try:
            return float(self._cmd("l RFPOWER\n", want=64).split()[0]) * scale_w
        except (OSError, ValueError, IndexError):
            return None

    # -- PTT -------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed

    @armed.setter
    def armed(self, on: bool) -> None:
        # onair_session arms only once the rig has named itself, so arming opens the line.
        if on and self._no_tx.is_set():
            raise ValueError("cannot arm a rig that has been taken down")
        if not on and self._keyer is not None:
            self._unkey()
            self._keyer.hand_back()  # deassert, read back, close before disarming
            self.released_low = self._keyer.released_low
            self._keyer = None
        if on and self._keyer is None:
            if not self.ptt_device:
                raise ValueError("armed rig requires --line-ptt and --ptt-device")
            self._keyer = LineKeyer(self.ptt_device, self.log)
            self._keyer.arm()         # opens deasserted, announces the split
        self._armed = on

    def key(self, on: bool, why: str = "") -> bool:
        if not self.armed:
            self.log(f"PTT {'ON' if on else 'OFF'} (DISARMED — not keying)")
            return not on
        if on and self._no_tx.is_set():
            self.log("PTT ON refused — this rig has been taken down")
            return False
        if on:
            self._note_identification_due()
        with self._lock:
            ok = self._key_locked(on, why)
            # A forced unkey can overtake key-up; the losing key must leave the line low.
            if on and ok and self._no_tx.is_set():
                self._key_locked(False, "overtaken by a forced unkey")
                return False
        return ok

    def _note_identification_due(self) -> None:
        """Warn when this station has been transmitting for ten minutes.

        The clock runs from this station's first key when reminders are enabled
        by :func:`_id_reminders_owed`. This prints a reminder; the operator chooses
        how to identify the station.
        """
        if not self._id_owed:
            return
        now = time.time()
        if self._first_key is None:
            self._first_key = self._id_last = now
            return
        if now - self._id_last >= _ID_INTERVAL_S:
            self._id_last = now
            self.log(f"*** {int((now - self._first_key) // 60)} min of transmitting: "
                     "station identification reminder is due ***")

    def _key_locked(self, on: bool, why: str = "") -> bool:
        if not on:
            return self._unkey()
        keyer = self._keyer
        if keyer is None:
            self.log("PTT ON refused — no keying line available")
            return False
        if keyer.key(True, why):
            self.keyed = True
            self.key_since = time.time()
            self._key_ups += 1
            self._unkeyed.clear()
            self._metering.set()
            return True
        self.keyed = False
        if keyer.must_retire:
            self._no_tx.set()
        return False

    def _unkey(self) -> bool:
        # A signal or watchdog must not wait on the lock held by the stalled key-up.
        self._unkey_at = time.time()
        self._metering.clear()
        self._unkeyed.set()
        if self._keyer is None:
            return drop_unheld_line(self.ptt_device, self.log)
        if self._keyer.key(False):
            self.keyed = False
            self.key_since = 0.0
            return True
        self._no_tx.set()
        return False

    @property
    def retired(self) -> bool:
        """This rig has been taken down — by a signal, a shutdown, or an unkey that
        never confirmed. Nothing may transmit through it again."""
        return self._no_tx.is_set()

    def unkey_hard(self, why: str) -> bool:
        """Force the transmitter down now, from any thread, holding nothing."""
        if not self.armed:
            return True
        self.log(f"force unkey ({why})")
        return self._unkey()

    def drop(self, why: str) -> bool:
        self.log(f"unkey ({why})")
        return self._unkey()

    def panic(self, why: str) -> None:
        """Signal-handler entry: retire the rig, unkey it, and guarantee an exit.

        Runs on the interrupted thread, so it must complete without waiting on
        anything that thread might already own. Order matters: retire first so a
        racing key-up cannot undo the unkey, arm the deadman before the unkey rather
        than after so that even a hung unkey cannot outlive it, and only then drop
        the keying line. A second signal while the first is still working means
        the operator has asked twice and is done asking.
        """
        if self._panicked:
            os._exit(1)
        self._panicked = True
        self._no_tx.set()
        self._arm_deadman()
        self.unkey_hard(why)

    def _arm_deadman(self) -> None:
        """Kill the interpreter if it has not exited shortly after a panic.

        ``sys.exit`` from a signal handler still has to unwind PortAudio callbacks and
        a ``finally:`` that writes WAV files, and any of that can wedge. By the time
        this fires the transmitter is already down, so a process that will not die is
        strictly worse than one killed rudely.

        The mechanism is `core.rig`'s, not a second one: this used to start a thread,
        which takes interpreter locks the interrupted thread may hold.
        """
        arm_deadman(_DEADMAN_S)

    @property
    def line_held(self) -> bool:
        return self._keyer is not None

    def line_state(self) -> bool | None:
        """Read the held keying line without asking CAT or opening a port."""
        keyer = self._keyer
        if keyer is None:
            return None
        try:
            return keyer.transport.sense()
        except OSError:
            return None

    def ptt_state(self, timeout: float = 2.0) -> bool | None:
        """Ask the rig, rather than believing our own bookkeeping. None = unknown.

        The unknown case is kept distinct: reading it as "down" is how a program talks
        itself into believing a transmitter it cannot see is off.
        """
        try:
            r = self._cmd("t\n", timeout=timeout, want=32).strip().splitlines()
            return bool(int(r[0])) if r and r[0].lstrip("-").isdigit() else None
        except (OSError, ValueError):
            return None

    def _watchdog(self) -> None:
        while not self._stop.wait(0.5):
            if self.armed and self.keyed and self.key_since:
                held = time.time() - self.key_since
                if held > MAX_KEY_S:
                    self.log(f"WATCHDOG: keyed {held:.0f}s > {MAX_KEY_S:.0f}s")
                    self.unkey_hard("watchdog")

    def _meter(self) -> None:
        """Peak power and ALC over each key-down, said once the key is up.

        Its own thread and its own rigctld connections: the key is the serial line,
        never CAT, so nothing here can hold a key-up or an unkey. A read that fails
        is named in the summary and costs nothing else.
        """
        while not self._stop.is_set():
            if not self._metering.wait(0.5):
                continue
            started, peaks, reads, unread, this = self.key_since or time.time(), {}, 0, {}, self._key_ups
            # The count tells a quick unkey and key-up apart, which the flag alone cannot.
            while self._metering.is_set() and self._key_ups == this and not self._stop.is_set():
                for name in TX_METERS:
                    try:
                        value = float(self._cmd(f"l {name}\n", timeout=1.0, want=64).split()[0])
                    except (OSError, ValueError, IndexError) as error:
                        unread.setdefault(name, type(error).__name__)
                        continue
                    peaks[name] = max(value, peaks.get(name, value))
                reads += 1
                self._unkeyed.wait(METER_PERIOD_S)
            ended = self._unkey_at if self._unkey_at >= started else time.time()
            held = ended - started
            watts, alc = peaks.get('RFPOWER_METER_WATTS'), peaks.get('ALC')
            short = held < MIN_METERED_KEY_S and 'RFPOWER_METER_WATTS' not in unread
            if short:
                watts = None
            report = dict(utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)),
                          seconds=round(held, 2), reads=reads,
                          peak_watts=watts, peak_alc=alc, unread=unread or None,
                          unmeasured=['RFPOWER_METER_WATTS'] if short else None)
            power = (f"power unmeasured ({held:.2f} s key-down)" if short
                     else "power unread" if watts is None else f"{watts:.1f} W")
            self.log("TX METERS: peak " + power + ", "
                     + ("ALC unread" if alc is None else f"ALC {alc:.2f}")
                     + f" over {reads} reads in {report['seconds']:.1f} s"
                     + (f" ({', '.join(f'{k} {v}' for k, v in unread.items())})" if unread else ""))
            self.meter_reports.put(report)

    def shutdown(self) -> None:
        """Unkey, then stand the watchdog down — never the other way round.

        Clearing the stop flag first retires the last line of defence at precisely the
        moment it is most likely to be needed, since the unkey it is standing behind is
        the one that has just been asked to run.
        """
        self._no_tx.set()
        self.unkey_hard("shutdown")
        keyer, self._keyer = self._keyer, None
        if keyer is not None:
            keyer.hand_back()         # deassert, read back, close — never reopens
            self.released_low = keyer.released_low
        self.retire()

    def retire(self) -> None:
        """Nothing may key through this rig again, and the watchdog may stand down."""
        self._no_tx.set()
        self._stop.set()


def _install_unkey_handlers(rig: Rig) -> None:
    """An interrupted transmit tool must leave the transmitter down, however it dies.

    `finally:` alone does not survive a Ctrl-C mid-over, and neither it nor atexit
    runs on a signal at all — a session killed mid-transmission on 2026-08-04 stayed
    keyed until the operator cut power. SIGTERM is what timeout(1), a supervisor and
    a cancelled tool call send; SIGHUP is a dropped terminal, which is how a session
    started over ssh ends. The handler goes through the rig's own panic path:
    retire, arm the deadman, unkey, and leave non-zero.

    `kestrel_connect` takes these rather than installing its own: it drives this
    same `Rig`, and a second copy is four signals to keep in step by hand.
    """
    atexit.register(rig.shutdown)

    def on_signal(signum, _frame):
        rig.panic(signal.Signals(signum).name)
        sys.exit(1)

    for sig in FATAL_SIGNALS:
        signal.signal(sig, on_signal)
