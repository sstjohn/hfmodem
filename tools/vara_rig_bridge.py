#!/usr/bin/env python3
"""Put a Wine VARA on the radio, and record everything it hears.

VARA under Wine talks to BlackHole, not to the rig. This bridges the two —
rig RX audio -> VARA's input, VARA's TX audio -> rig, with PTT keyed off the
``PTT ON``/``PTT OFF`` messages VARA already publishes on its command port —
and writes both directions to WAV alongside a log of VARA's host traffic.

The point is high-speed capture. Our own VARA is unregistered and so transmits
at the bottom of the ladder, but the speed cap follows the *transmitting*
station: a registered RMS gateway answers at whatever level the link supports.
So a session against a real gateway yields genuine high-SL frames, recorded,
*and* labelled — VARA reports the level it decoded (``BITRATE (N) x bps``) and
delivers the payload bytes, which is the ground truth a bench loopback can
never produce (spec/06 §6.1c).

    # bench dry run — bridges audio, logs PTT, never keys the rig:
    python3 tools/vara_rig_bridge.py --record /tmp/cap

    # on air (keys the transmitter — needs a tuned antenna). 4533 is the FT-891
    # on this bench; 4532 is a different radio. --arm demands the model match,
    # a keying line, and --line-ptt: the line is driven directly and rigctld
    # only tunes, because on 2026-08-13 the daemon answered every command until
    # the first key-down and then answered nothing, deterministically.
    python3 tools/vara_rig_bridge.py --record /tmp/cap --rigctld 127.0.0.1:4533 \
        --expect-model FT-891 --arm --ptt-device /dev/cu.usbserial-XXXXB1 \
        --line-ptt --gateway W9XYZ --mycall MYCALL

:class:`Rig` is this tree's rigctld client — ``hfcapture`` and ``kestrel_connect``
open it disarmed, for frequency and mode only.
"""
from __future__ import annotations

import argparse
import atexit
import os
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
from hfmodem.core.busy import ReceiverFault, level_range, receiver_fault
from hfmodem.core.busy import is_busy as _is_busy
from hfmodem.core.config import ConfigError, station_profile
from hfmodem.core.occupied import occupied_hz
from hfmodem.core.ptt import (Attempt, Keyer, LineKeyer, arming_refusal,
                              KEY_BUDGET_S, UNKEY_BUDGET_S)
from hfmodem.core.rig import FATAL_SIGNALS, arm_deadman

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "working" / "vara" / "oracle"))

FS = 48000
BLOCK = 1024


class _Leg:
    """One audio direction: capture from ``src`` and replay into ``dst``."""

    def __init__(self, src: str, dst: str, name: str, gain: float = 1.0):
        import sounddevice as sd  # PortAudio: only the audio legs need it

        self.name = name
        self.src = src
        self.gain = gain
        self.q: queue.Queue = queue.Queue(maxsize=64)
        self.recorded: list[np.ndarray] = []
        self.peak = 0.0
        self.clipped = 0
        self.dst_channels = int(sd.query_devices(dst, "output")["max_output_channels"])
        self._in = sd.InputStream(device=src, channels=1, samplerate=FS,
                                  blocksize=BLOCK, dtype="float32",
                                  callback=self._on_in)
        self._out = sd.OutputStream(device=dst, channels=self.dst_channels,
                                    samplerate=FS, blocksize=BLOCK,
                                    dtype="float32", callback=self._on_out)

    def _on_in(self, indata, frames, t, status):
        mono = indata[:, 0].copy()
        self.peak = max(self.peak, float(np.abs(mono).max()))
        self.clipped += int((np.abs(mono) > 0.99).sum())
        self.recorded.append(mono)
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            pass                                    # drop rather than back up the rig

    def _on_out(self, outdata, frames, t, status):
        try:
            mono = self.q.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            return
        if len(mono) < frames:
            mono = np.concatenate([mono, np.zeros(frames - len(mono), np.float32)])
        block = np.zeros((frames, self.dst_channels), np.float32)
        block[:, 0] = mono[:frames] * self.gain
        if self.dst_channels > 1:
            block[:, 1] = block[:, 0]
        outdata[:] = block

    def start(self):
        self._in.start(); self._out.start()

    def stop(self):
        for s in (self._in, self._out):
            try:
                s.stop(); s.close()
            except Exception:
                pass

    def save(self, path: Path):
        from scipy.io import wavfile

        a = np.concatenate(self.recorded) if self.recorded else np.zeros(1, np.float32)
        wavfile.write(str(path), FS, a.astype(np.float32))
        return len(a) / FS, float(np.abs(a).max())


# A VARA over is seconds long (the longest observed BW2300 DATA over is ~8.6 s), so
# a continuous key beyond this is a stuck transmitter, not traffic.
MAX_KEY_S = 25.0


_ID_INTERVAL_S = 600.0            # identification reminder interval

# A forced unkey runs against the clock, not against a retry count. An operator who
# reached for Ctrl-C is entitled to a dead transmitter and a dead process in about a
# second, whatever rigctld is doing — and the moment they reach for it is exactly the
# moment rigctld is least likely to answer.
_UNKEY_BUDGET_S = UNKEY_BUDGET_S  # ordinary ``PTT OFF``, where there is time to be patient
_PANIC_BUDGET_S = 1.0             # forced unkey: signal, watchdog, watcher death, shutdown

# Per command, and it decides what the budgets above are worth. A budget spent
# waiting on one blocked read is a single attempt wearing a retry loop's clothes:
# that is what held this station's FT-891 on the air for 7.84 s past its last sample
# on 2026-07-28, when the ordinary unkey used three seconds per command and so asked
# exactly once in six. Round trips to this rig measure 0.27 s for ``T 0`` and 0.05 s
# for ``t``, so 0.75 s is threefold generous and still buys four passes; the forced
# path keeps a tighter one because an operator reaching for Ctrl-C is owed a dead
# transmitter in about a second. Re-sending ``T 0`` costs nothing — the write is what
# drops the transmitter, and the read is only evidence.
_UNKEY_TIMEOUT_S = 0.75
_PANIC_TIMEOUT_S = 0.4
_DEADMAN_S = 2.0                  # ... after which the interpreter is killed outright

# Key-up runs to a clock as well, and a tighter one — the clock itself, and the
# reasons for its size, live with the ladder in `core.ptt`.
_KEY_BUDGET_S = KEY_BUDGET_S


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"


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


def _rprt_codes(reply: str) -> list[int] | None:
    """Every status code in the reply, or None if an ``RPRT`` arrives without one.

    The framing is not fixed: the same daemon has answered with one ``RPRT 0`` per
    line and, on the extended command form, with ``RPRT 0 RPRT 0`` on a single
    line. Reading tokens rather than lines takes both, and an ``RPRT`` with
    nothing readable after it — a reply cut off mid-verdict — is None, not a pass.
    """
    toks = reply.split()
    codes = []
    for i, tok in enumerate(toks):
        if tok == "RPRT":
            if i + 1 >= len(toks) or not toks[i + 1].lstrip("-").isdigit():
                return None
            codes.append(int(toks[i + 1]))
    return codes


class _Wire:
    """What actually crossed the socket for one rigctld command.

    The unkey timeouts of 2026-08-09 went into the log as one sentence — "rigctld
    took the command and did not answer" — and that sentence covers half a dozen
    failures with different causes: a connection refused, a connection accepted and
    then held silent, a peer that closed without a byte, a reply cut off mid-line,
    a verdict the parser could not read. Two days of theory grew where the
    distinction should have been. So :meth:`Rig._cmd` fills one of these as it
    works — the bytes sent, every read with its own elapsed time, the connect
    time, the errno if the OS gave one — and the PTT paths log it whenever a
    command does not get the reply it expected.
    """

    def __init__(self):
        self.sent = b""
        self.received = b""
        self.connect_s: float | None = None   # None: the connection never completed
        self.reads: list[str] = []            # one per recv: bytes, EOF, or timeout
        self.eof = False
        self.timed_out = False
        self.exc: OSError | None = None
        self.elapsed = 0.0

    def failure(self) -> str:
        """Which way rigctld failed to answer; meaningful once a reply is rejected.

        Each mode points somewhere different: a refused connection is a daemon not
        listening; accepted-then-silent is a daemon alive on the socket and dead to
        the radio; a reply cut short is the transport failing mid-sentence; an
        in-band refusal is a rig that heard and said no; a truncated or off-script
        verdict is a framing problem, and that one is ours.
        """
        if self.exc is not None:
            if isinstance(self.exc, ConnectionRefusedError):
                return "refused the connection"
            if self.connect_s is None:
                return ("never completed the connection"
                        if isinstance(self.exc, TimeoutError)
                        else f"failed connecting: {self.exc}")
            return f"failed mid-exchange: {self.exc}"
        if self.connect_s is None:
            return "was never asked (no rigctld address)"
        if not self.received:
            return ("accepted the connection, then closed it without a byte"
                    if self.eof else "accepted the command and went silent")
        if self.eof:
            return "closed the connection mid-reply"
        if self.timed_out:
            return "went silent mid-reply"
        text = self.received.decode(errors="replace").strip()
        codes = _rprt_codes(text)
        if codes is None:
            return f"answered a truncated verdict: {text!r}"
        if any(codes):
            return f"refused in-band: {text!r}"
        return f"answered off-script: {text!r}"

    def detail(self) -> str:
        parts = [f"sent {self.sent!r}", f"got {self.received!r}",
                 "connect " + (_ms(self.connect_s) if self.connect_s is not None
                               else "incomplete")]
        if self.reads:
            parts.append("read " + ", then ".join(self.reads))
        if self.exc is not None and getattr(self.exc, "errno", None) is not None:
            parts.append(f"errno {self.exc.errno}")
        parts.append(f"total {_ms(self.elapsed)}")
        return "; ".join(parts)

    def answered(self) -> bool:
        """Whether a readable verdict came back at all — the split the unkey
        ladder turns on. An answer, even a refusal, is a daemon alive and worth
        asking again inside the budget; silence in any shape is not, and on
        this station it has never once turned back into an answer
        (2026-08-09, 2026-08-13: every retry of a quiet ``T 0`` stayed quiet).
        """
        return (self.exc is None and bool(self.received) and not self.eof
                and not self.timed_out
                and _rprt_codes(self.received.decode(errors="replace")) is not None)


class _RigctldPtt:
    """The daemon as a keying transport for `core.ptt.Keyer`.

    Every byte still goes through :meth:`Rig._cmd` — tests rely on that to prove
    a path never touched the daemon — and every attempt fills a :class:`_Wire`,
    so a failed command reaches the log with the bytes, timings and errno that
    say which way the daemon went. The verdict is what the rig read back: a
    ``T 0`` is believed only when ``t`` reads the PTT down, or when the rig has
    no PTT to report and the accepted write is all the evidence there is —
    taken, never silently. The key-up is not read back, and that is the same
    argument the other way up: everything between the key and the first sample
    is unmodulated carrier, so the cheapest confirmation there is the one not
    asked for.
    """

    place = "rig"
    peer = "rigctld"
    paced = True
    reads_back = True
    attempt_timeout = _UNKEY_TIMEOUT_S
    taken_note = "rig will not report PTT; unverified"

    def __init__(self, rig: "Rig") -> None:
        self._rig = rig

    def drive(self, on: bool, timeout: float, *,
              confirm_silence: bool = True) -> Attempt:
        wire = _Wire()
        try:
            reply = self._rig._cmd(f"T {'1' if on else '0'}\n", timeout=timeout,
                                   want=64, wire=wire)
        except OSError:
            reply = ""                # the wire holds the errno and the story
        accepted = bool(reply.strip()) and Rig._rprt_ok(reply)
        quiet = not wire.answered()
        if on:
            if accepted:
                return Attempt("taken")
            return Attempt("failed", why=f"rigctld {wire.failure()}",
                           detail=wire.detail(), quiet=quiet)
        if quiet and not confirm_silence:
            # The ``T 0`` got no verdict and the caller holds a keying line: a
            # ``t`` here spends another timeout of carrier asking the component
            # that just went silent, for an answer the line gives directly.
            return Attempt("failed", why=f"rigctld {wire.failure()}",
                           detail=wire.detail(), quiet=True)
        state = self._rig.ptt_state(timeout)
        if state is False:
            return Attempt("confirmed", state=False, state_read=True)
        if accepted and state is None:
            return Attempt("taken", state_read=True)
        return Attempt("failed",
                       why=("rig still reports PTT on" if accepted
                            else f"rigctld {wire.failure()}"),
                       detail=wire.detail(), state=state, state_read=True,
                       quiet=quiet)

    def release(self) -> bool | None:
        return None


class Rig:
    """rigctld PTT with positive identification and an independent unkey watchdog.

    A failed unkey is retried and escalated rather than logged and forgotten — including
    an in-band ``RPRT <negative>``, which rigctld returns instead of raising — the
    watchdog force-unkeys on its own thread if the key outlives a plausible over, and
    SIGINT/SIGTERM/atexit all unkey, since ``finally:`` does not survive a Ctrl-C
    mid-over. It cannot help if rigctld itself wedges: that failure took a power cycle
    on 2026-07-24 and wants a transport the daemon is not on.

    Both directions run to a clock rather than a retry count, so every caller has a
    worst case it can state. Every unkey is confirmed by reading the PTT line back — a
    write rigctld accepts is not evidence the transmitter dropped, since a wedged daemon
    still accepts connections and still takes the bytes. The key-up is not read back,
    and that is the same argument the other way up: everything between the key and the
    first sample is unmodulated carrier, so the cheapest confirmation there is the one
    not asked for.

    The forced unkey takes no lock. It runs on whatever thread caught the signal, which
    is normally the thread already inside :meth:`key` — so waiting for ``_lock`` there
    is a deadlock with the transmitter keyed, which is the worst state this program has.
    Nor is a reentrant lock the answer: that just lets the handler interleave its ``T 0``
    with a half-finished key-up. Going around the lock cannot corrupt a transaction
    anyway, because :meth:`_cmd` gives every command its own connection; the only real
    risk is that our ``T 0`` and an in-flight ``T 1`` reach rigctld out of order, and
    ``_no_tx`` settles that — it aborts the interrupted key-up, and makes whoever holds
    the lock put the rig back down on the way out.
    """

    def __init__(self, hostport: str | None, armed: bool, log,
                 ptt_device: str | None = None, line_ptt: bool = False):
        self.log = log
        self.armed = armed
        # The serial line whose RTS IS the PTT, when there is one.
        self.ptt_device = ptt_device
        # With ``line_ptt`` we drive that line ourselves and rigctld is left to tune
        # the radio, which it does with the transmitter down. Without it,
        # every key goes through the daemon and the line is touched only when the
        # daemon has stopped answering with the rig possibly still up.
        self._keyer = None
        if line_ptt and ptt_device and armed:
            self._keyer = LineKeyer(ptt_device, log)
            self._keyer.arm()         # opens deasserted, announces the split
        self._first_key = None        # when this station first keyed
        self._id_last = None          # last identification reminder
        self._id_owed = _id_reminders_owed(log)
        self.addr = None
        if hostport:
            host, _, port = hostport.partition(":")
            self.addr = (host or "127.0.0.1", int(port or 4533))
        self.keyed = False
        self.key_since = 0.0
        self._lock = threading.Lock()     # serialises key-ups; the unkey path never takes it
        self._no_tx = threading.Event()   # set once this rig is being taken down for good
        # The daemon path's half of the one keying mechanism (`core.ptt.Keyer`):
        # same ladder, same wordings, same retire as every other modem, with
        # this Rig supplying only the transport and what retiring means here.
        self._daemon = Keyer(_RigctldPtt(self), log,
                             ptt_device=ptt_device,
                             held=lambda: self.key_since,
                             abort=self._no_tx.is_set,
                             on_retire=self._no_tx.set)
        self._panicked = False
        self._stop = threading.Event()
        threading.Thread(target=self._watchdog, daemon=True).start()

    # -- rigctld ---------------------------------------------------------------
    def _cmd(self, text: str, timeout: float = 3.0, want: int = 4096,
             wire: _Wire | None = None) -> str:
        """One command on its own connection; fills ``wire`` with what crossed it.

        Every byte this program sends rigctld goes through here — tests rely on
        that to prove a path never touched the daemon. Raises what the socket
        raises, as it always has, but records the exception (and everything up to
        it) in ``wire`` first, so a caller that must not lose the story passes one
        in and reads it back whatever happened.
        """
        if self.addr is None:
            return ""
        w = wire if wire is not None else _Wire()
        w.sent = text.encode()
        t0 = time.monotonic()
        try:
            try:
                s = socket.create_connection(self.addr, timeout=timeout)
            except OSError as e:
                w.exc = e
                raise
            w.connect_s = time.monotonic() - t0
            try:
                s.sendall(w.sent)
                s.settimeout(timeout)
                while True:
                    t_read = time.monotonic()
                    try:
                        chunk = s.recv(want)
                    except TimeoutError:
                        w.timed_out = True
                        w.reads.append(f"timeout at {_ms(time.monotonic() - t_read)}")
                        break
                    if not chunk:
                        w.eof = True
                        w.reads.append(f"EOF at {_ms(time.monotonic() - t_read)}")
                        break
                    w.reads.append(f"{len(chunk)}B at {_ms(time.monotonic() - t_read)}")
                    w.received += chunk
                    if len(chunk) < want:
                        break
                return w.received.decode(errors="replace")
            except OSError as e:
                w.exc = e
                raise
            finally:
                s.close()
        finally:
            w.elapsed = time.monotonic() - t0

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

    def set_freq(self, hz: float) -> str:
        return self._cmd(f"F {int(hz)}\n")

    def set_mode(self, mode: str, passband: int = 0) -> str:
        return self._cmd(f"M {mode} {int(passband)}\n")

    # -- PTT -------------------------------------------------------------------
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
            if on and ok and self._no_tx.is_set():
                # A forced unkey overtook us while our own ``T 1`` was in flight, so
                # rigctld may have applied them in that order. Put the rig back down —
                # but only if the key-up worked, because a failed one has already
                # dropped PTT itself and set this flag doing it, and dropping twice
                # spends the unkey budget twice over.
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

    @staticmethod
    def _rprt_ok(reply: str) -> bool:
        """rigctld reports failure in-band as ``RPRT <negative>``; it does not raise.

        Every code must be zero, wherever it sits — see :func:`_rprt_codes` for the
        framings this has to take. The line-based reading before it stopped at the
        first ``RPRT`` it saw, so ``RPRT 0 RPRT -9`` on one line read as success.
        No verdict at all is still a pass: nothing to contradict.
        """
        codes = _rprt_codes(reply)
        return codes is not None and not any(codes)

    def _key_locked(self, on: bool, why: str = "") -> bool:
        """One keying mechanism for both configurations: the line keyer holds
        the descriptor, the daemon keyer holds the socket ladder, and both are
        `core.ptt.Keyer` — the budgets, the retries, the drop-to-be-sure on an
        unconfirmed key-up and the wordings all live there. What is this Rig's
        own is the bookkeeping the watchdog reads (`keyed`, `key_since`) and
        what retiring is called here (`_no_tx`)."""
        if not on:
            return self._unkey(_UNKEY_BUDGET_S, _UNKEY_TIMEOUT_S)
        keyer = self._keyer if self._keyer is not None else self._daemon
        if keyer.key(True, why):
            self.keyed = True
            self.key_since = time.time()
            return True
        self.keyed = False
        if keyer.must_retire:
            self._no_tx.set()
        return False

    def _unkey(self, budget: float, timeout: float) -> bool:
        """The unkey ladder — `core.ptt.Keyer.unkey`, the same one every modem
        runs. Deliberately lock-free (see the class docstring) and bounded by a
        deadline, so the emergency callers get a predictable worst case. An
        accepted write is not proof; the rig is read back, and a daemon that
        answers with the rig still up is asked again inside the budget — where
        one that goes silent forfeits it after a single ask, because on
        2026-08-13 four silent ``T 0``s in a row held ~4.7 s of unintended
        carrier while the keying line, consulted at last, read LOW at once.
        Every failed pass is logged with its `_Wire` record — the bytes both
        directions, per-read timing, the ``t`` read-back, how long PTT has been
        up — which is the instrument the 2026-08-09 timeouts were missing.
        """
        if self._keyer is not None:
            # The budgets are daemon money — the keyer has no reply to wait for
            # and spends nothing, alarming and setting `must_retire` on its own
            # if the line will not go low.
            if self._keyer.key(False):
                self.keyed = False
                self.key_since = 0.0
                return True
            self._no_tx.set()
            return False
        if self.addr is None:
            return True
        if self._daemon.unkey(budget, timeout):
            self.keyed = False
            self.key_since = 0.0
            return True
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
        return self._unkey(_PANIC_BUDGET_S, _PANIC_TIMEOUT_S)

    def drop(self, why: str) -> bool:
        """Put PTT down to the ordinary budget, armed or not.

        The arm gate stops this program putting a signal on the air; it is not a reason
        to walk away from a transmitter that is already up. A caller that has read the
        PTT line back and found it high has found something to drop — a dry run
        included, which is exactly when it is somebody else's mistake being cleaned up.
        """
        self.log(f"unkey ({why})")
        return self._unkey(_UNKEY_BUDGET_S, _UNKEY_TIMEOUT_S)

    def panic(self, why: str) -> None:
        """Signal-handler entry: retire the rig, unkey it, and guarantee an exit.

        Runs on the interrupted thread, so it must complete without waiting on
        anything that thread might already own. Order matters: retire first so a
        racing key-up cannot undo the unkey, arm the deadman before the unkey rather
        than after so that even a hung unkey cannot outlive it, and only then talk to
        rigctld. A second signal while the first is still working means the operator
        has asked twice and is done asking.
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
        self.retire()

    def retire(self) -> None:
        """Nothing may key through this rig again, and the watchdog may stand down."""
        self._no_tx.set()
        self._stop.set()


def channel_busy(leg: _Leg, secs: float = 6.0,
                 bw: str = "") -> tuple[bool, float, ReceiverFault | None]:
    """Listen and decide whether someone is already using the frequency.

    Blind transmitting is not acceptable operating practice.

    This used to compare the loudest 0.5 s block-RMS against the median one. That
    test is blind to exactly the signal it most needs to see: a wideband data mode
    fills the passband, lifting the median with the peak, so the ratio stays low
    precisely when the channel is fully occupied. It reported CLEAR on a 40 m
    frequency carrying an audible QSO. `core.busy.is_busy` scores the signal's
    own structure instead — burst, spectral shape and narrow features — and refuses
    a verdict when the receiver is muted or deaf. Returns (busy, margin_dB,
    fault), the margin being how far the strongest evidence sits above its
    threshold.

    The fault comes first and is not a verdict about the channel: an input that
    delivered nothing, or that collapses to the codec floor part of the window,
    cannot have heard the station we would be transmitting over. `--rig-device`
    naming a device that opens and stays silent — the wrong index after a USB
    re-enumeration, BlackHole with nothing routed into it — leaves `recorded`
    empty, and an empty capture is that input, not a quiet band.
    """
    time.sleep(secs)
    a = (np.concatenate(leg.recorded[-int(secs * FS / BLOCK) - 1:])
         if leg.recorded else np.zeros(0, np.float32))
    fault = receiver_fault(*level_range(a, FS), seconds=secs, device=leg.src)
    busy, _per_sec, margin = _is_busy(a, FS, occupied_hz("vara", bw))
    return busy, margin, fault


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig-device", default="USB Audio Device")
    ap.add_argument("--vara-in", default="BlackHole 64ch",
                    help="device VARA LISTENS on (we write rig audio here)")
    ap.add_argument("--vara-out", default="BlackHole 16ch",
                    help="device VARA TRANSMITS into (we read it and feed the rig)")
    ap.add_argument("--rigctld", help="host:port for PTT (FT-891 bench rig = 127.0.0.1:4533; "
                                      "4532 is a DIFFERENT radio)")
    ap.add_argument("--expect-model", help="required with --arm: refuse to key unless "
                                           "rigctld reports this model (e.g. FT-891)")
    ap.add_argument("--arm", action="store_true",
                    help="actually key the transmitter (omit for a safe dry run)")
    ap.add_argument("--ptt-device", metavar="DEV",
                    help="serial line whose RTS is the PTT, used as the last "
                         "resort in an unkey rigctld has stopped answering. "
                         "Required with --arm")
    ap.add_argument("--line-ptt", action="store_true",
                    help="key --ptt-device directly and leave rigctld to tune. "
                         "Required with --arm: the daemon's rig side dies on "
                         "the first key-down on this station (2026-08-13)")
    ap.add_argument("--cmd-port", type=int, default=8300)
    ap.add_argument("--data-port", type=int, default=8301)
    ap.add_argument("--gateway", help="callsign to connect to (omit to just listen)")
    ap.add_argument("--mycall", required=True,
                    help="your callsign — you are the licensee for these emissions")
    ap.add_argument("--bw", default="2750", choices=["500", "2300", "2750"])
    ap.add_argument("--tx-gain", type=float, default=0.3, help="drive level into the rig")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--connect-timeout", type=float, default=90.0)
    ap.add_argument("--listen-first", type=float, default=6.0,
                    help="seconds to sense the channel before keying (0 to skip)")
    ap.add_argument("--force", action="store_true",
                    help="transmit even if the channel sounds occupied")
    ap.add_argument("--record", default="/tmp/vara_rig_capture")
    args = ap.parse_args()

    out = Path(args.record); out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str):
        stamp = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(stamp, flush=True); lines.append(stamp)

    if args.arm and not args.rigctld:
        log("--arm needs --rigctld"); return 2
    if args.arm and (refusal := arming_refusal(args.ptt_device)):
        log(refusal); return 2
    if args.arm and not args.line_ptt:
        # Not a preference. 2026-08-13, eight of eight: rigctld answered every
        # command until the first key-down, then answered nothing — with the
        # socket connecting in a millisecond the whole time. A daemon-keyed
        # armed run transmits one over and then cannot unkey through the thing
        # that keyed it.
        log("--arm requires --line-ptt: rigctld may tune this rig, but it may "
            "not hold the key — its rig side stops answering the moment PTT "
            "is asserted (2026-08-13, every armed attempt)")
        return 2
    log(f"TX {'ARMED' if args.arm else 'DISARMED (dry run)'}")

    rig = Rig(args.rigctld, args.arm, log, ptt_device=args.ptt_device,
              line_ptt=args.line_ptt)
    if args.rigctld:
        model, freq = rig.identify()
        log(f"rigctld {args.rigctld}: model={model!r} freq={freq}")
        if args.arm:
            if not args.expect_model:
                log("--arm requires --expect-model (never key a radio you cannot name)")
                return 2
            if args.expect_model.lower() not in model.lower():
                log(f"REFUSING TO ARM: rigctld reports {model!r}, expected "
                    f"{args.expect_model!r} — this is probably the wrong radio")
                return 2
            log(f"rig identified as {model} — arming")

    _install_unkey_handlers(rig)

    rx = _Leg(args.rig_device, args.vara_in, "rig->vara")
    tx = _Leg(args.vara_out, args.rig_device, "vara->rig", gain=args.tx_gain)
    ptt = rig.key

    from transcript import Transcript
    from vara_client import VaraClient

    t0 = time.monotonic()
    tr = Transcript(str(out / "host.jsonl"), t0, echo=False)
    cli = VaraClient("rig", args.mycall, tr, cmd_port=args.cmd_port,
                     data_port=args.data_port)
    # VARA publishes every host message, PTT included, on this queue; the watcher
    # turns PTT ON/OFF into rig keying and logs the rest (BITRATE, CONNECTED, ...).
    events = cli.subscribe()
    stop = threading.Event()

    def watch():
        # This thread is the only thing that unkeys during a session, so its death
        # must be loud and must not leave the transmitter keyed.
        try:
            while not stop.is_set():
                try:
                    msg = events.get(timeout=0.2)
                except queue.Empty:
                    continue
                log(f"VARA: {msg}")
                if msg == "PTT ON":
                    ptt(True)
                elif msg == "PTT OFF":
                    ptt(False)
        except BaseException as e:                    # noqa: BLE001 - must not be silent
            log(f"*** PTT WATCHER DIED ({e!r}) — force unkeying and aborting ***")
            rig.unkey_hard("watcher died")
            stop.set()

    threading.Thread(target=watch, daemon=True).start()
    rx.start(); tx.start()
    log("bridge up")
    try:
        cli.connect_tcp()
        cli.set_bandwidth(args.bw)
        if args.gateway and args.listen_first > 0:
            busy, margin, fault = channel_busy(rx, args.listen_first, args.bw)
            if fault is not None:
                # Not a channel verdict, so --force does not weigh on it: what the
                # operator can hear says nothing about whether our input is attached.
                log(fault.reason)
                return 3
            log(f"channel sense: {margin:+.1f} dB against the occupancy thresholds "
                f"({'OCCUPIED' if busy else 'clear'})")
            if busy and not args.force:
                log("channel in use — NOT transmitting (pass --force to override)")
                return 3
        if args.gateway:
            log(f"connecting {args.mycall} -> {args.gateway} at BW{args.bw}")
            ok = cli.connect(args.gateway, timeout=args.connect_timeout)
            log(f"connect result: {ok}")
            if ok:
                got = cli.recv_data(65536, timeout=args.seconds)
                log(f"received {len(got)} payload bytes from {args.gateway}")
                (out / "payload.bin").write_bytes(got)
        else:
            log(f"listening {args.seconds:.0f}s (no connect)")
            time.sleep(args.seconds)
    except Exception as e:
        log(f"session error: {e!r}")
    finally:
        stop.set()
        try:
            cli.close()
        except Exception:
            pass
        rig.shutdown()
        rx.stop(); tx.stop()
        d1, p1 = rx.save(out / "rig_rx.wav")
        d2, p2 = tx.save(out / "vara_tx.wav")
        log(f"saved rig_rx.wav {d1:.1f}s peak={p1:.3f}, vara_tx.wav {d2:.1f}s peak={p2:.3f}")
        if rx.clipped:
            # The high speed levels carry data in the amplitude of a QAM
            # constellation; a clipped capture cannot be demodulated at all.
            log(f"WARNING: receive audio CLIPPED on {rx.clipped} samples "
                f"(peak {rx.peak:.3f}) — turn the rig/codec input gain down until "
                f"peak sits near 0.3-0.5, or the high-SL capture is unusable")
        elif rx.peak < 0.02:
            log(f"WARNING: receive audio very quiet (peak {rx.peak:.3f}) — check the "
                f"rig is on the right frequency and the codec is selected")
        else:
            log(f"receive level OK (peak {rx.peak:.3f})")
        (out / "bridge.log").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
