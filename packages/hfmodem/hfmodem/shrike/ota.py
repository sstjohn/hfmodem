# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""On-air test harness: drive an HF rig to probe real PACTOR/Winlink stations.

shrike renders a session (our connect, optionally a data packet), keys the rig,
transmits the audio, then records and decodes the station's reply. The reply is
the reverse direction, and no monitor can supply it: a monitor decodes what it
happens to hear and cannot make a station answer. Only a transmission of our own
produces the connect-answer, the control signals in live context, ACK/NAK and the
cycle timing of a link that has us on the other end.

Rig-agnostic: pass --rig {ft891,g90,x6100} plus the serial/audio devices. Runs
Hamlib rigctl for frequency/mode/PTT and sounddevice for TX playback / RX capture.
A PTT watchdog force-unkeys after --max-key seconds so a hung playback can't cook
the finals.

Decode of the reply is delegated to an external reference decoder, named by
`HFMODEM_DECODER` (see `_decode`): record -> 48k stereo WAV -> that decoder ->
the station's PACTOR bursts, read out.

Dry run (--dry-run) renders and saves the TX wav and prints the plan without any
rig, so the whole flow is verifiable with no hardware attached.
"""
from __future__ import annotations

import argparse
import os
import re
import select
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


from . import session, spec
from ..core.audio import play_drained, warm_output
from ..core.devices import find_device, list_devices
from ..core.occupied import FILTER_HZ
from ..core.rxreadiness import parse_mode_readback, parse_frequency_readback, ReceiverReadbackError
from ..core.ptt import (Keyer, OneShotRigctl, PttError, derive_ptt_port,
                        require_char_device)
from ..core.rigs import RIGS

FS = spec.SAMPLE_RATE

# How long a command may take to reach rigctl before the channel is given up on.
# Generous next to a pipe write (microseconds) and short next to the 1.25 s cycle
# it has to fit inside, so an unkey that is going to fail fails while there is
# still time to do something about it.
WRITE_TIMEOUT_S = 0.25

# PACTOR-3 is centred at spec.CENTER_FREQ_HZ (1500 Hz) in the audio passband, so on
# USB the emitted RF centre = dial + 1500 Hz. Winlink's RMSChannels list quotes the
# CHANNEL CENTRE; the operator tunes dial = centre - 1500. Keep that offset explicit
# here so the rig is never silently mistuned by 1500 Hz.
def center_to_dial(center_hz: float) -> int:
    return int(round(center_hz - spec.CENTER_FREQ_HZ))

# One cost of keying on `RIGS[...]["ptt_type"] == "RTS"`, to watch on the air:
# hamlib closes the PTT port after configuring it and reopens it on the session's
# FIRST key-down, so the first burst -- the connect, the one burst a called station
# has to lock onto -- pays a serial open that no later burst does. `--preflight`
# measures it, as PTT actuation against the rig's settle.
def ptt_device(rig: str, serial: str) -> str:
    """The port whose RTS keys `rig`, derived from the CAT port it shares a chip
    with -- the two device names differ only in the trailing interface index
    (...B0 Enhanced/CAT, ...B1 Standard/PTT). Derived rather than listed so
    --serial stays the single source of truth for both. Rigs that key over CAT
    have no second port and answer with the one they already have.

    The derivation is `core.ptt.derive_ptt_port`'s, deliberately narrow: a
    loose "advance the last digit" turns /dev/ttyUSB0 into /dev/ttyUSB1, and on
    Linux those are two different adapters -- quite possibly another radio's
    CAT port -- so a name that carries a trailing digit outside the CP210x
    convention is refused rather than guessed at. A name with no interface
    index at all (a bench port, a symlink) has nothing to advance and is used
    as it stands.

    Either way, a path that is not a character device is REFUSED here, naming
    what was derived and from what. A stat and never an open: opening a keying
    line asserts RTS on a port something else may be driving. See
    `core.ptt.require_char_device` for the 2026-08-10 placeholder-device
    session this check exists for.
    """
    iface = RIGS[rig].get("ptt_iface")
    if iface is None:
        require_char_device(serial)
        return serial
    if re.search(r"\d$", serial):
        try:
            derived = derive_ptt_port(serial)
        except ValueError as exc:
            raise PttError(str(exc)) from None
    else:
        derived = serial
    try:
        require_char_device(derived)
    except PttError as exc:
        raise PttError(f"{exc} -- derived from CAT port {serial}; fix --serial, "
                       f"or name the keying port explicitly") from None
    return derived

# PACTOR-3-capable Winlink RMS within 40m/20m range of Milwaukee (EN63). These are
# CHANNEL CENTRE frequencies in Hz, as winlink.org/RMSChannels lists them; the dial
# is derived (centre - 1500). VERIFY both the callsigns and that these are centres,
# not dials, against the live RMSChannels list before keying — they were entered as
# centres but the list is the source of truth.
GATEWAYS = {
    "AB0DK":  {"40m": 7101500,  "30m": 10148400, "20m": 14110000, "grid": "EN30"},
    "KI0BK":  {"40m": 7102500,  "20m": 14105000, "grid": "EM28"},
    "KC0TPS": {"40m": 7104000,  "30m": 10145900, "20m": 14098700, "grid": "EM48"},
    "W9OTR":  {"30m": 10145500, "20m": 14108000, "10m": 28305900, "grid": "EM68"},
    "K0NTS":  {"40m": 7104900,  "20m": 14104900, "grid": "DM79"},
}


def find_rigctl(rigctl_dir=None) -> Path:
    """Hamlib's `rigctl`, or a refusal naming what is missing and where we looked.

    No shipped directory default: rigctl comes off PATH unless a caller names the
    install, because a home-directory path here was one station's. What PATH does
    not carry, the LAUNCHER supplies -- `tools/lib/ports.sh` puts a private
    Hamlib build there for every child it starts at once, which is one fact in
    one place rather than a --rigctl-dir flag threaded through each consumer.

    The refusal is here, at construction, because a bare "rigctl" handed to
    `Popen` fails as a raw FileNotFoundError from inside a keying path -- with
    the audio device already open, and at the radio that reads as an audio
    fault. Nothing in `station/preflight.py` builds one of these (its CAT is a
    socket to rigctld), so refusing in the constructor costs it nothing.
    """
    if rigctl_dir:
        exe = Path(rigctl_dir).expanduser() / "rigctl"
        if not os.access(exe, os.X_OK):
            raise SystemExit(f"NOT KEYING: no executable rigctl at {exe} -- "
                             f"check the directory that was named, or leave it "
                             f"unset and let PATH answer")
        return exe
    found = shutil.which("rigctl")
    if found is None:
        raise SystemExit(
            "NOT KEYING: hamlib's rigctl is not on PATH, and it is what tunes "
            "and keys this rig.\n"
            f"  PATH: {os.environ.get('PATH', '')}\n"
            "  Install hamlib (`brew install hamlib`, or your package manager). "
            "For a build of your own, name its bin directory as HAMLIB_BIN and "
            "put its bin on PATH before starting hfmodem.\n"
            "  Check with: command -v rigctl rigctld")
    return Path(found)


def drain(proc: subprocess.Popen, errors: deque | None = None) -> None:
    """Empty rigctl's stdout so a long session cannot fill the pipe.

    A pipe nobody reads holds about 64 KiB, and rigctl blocks writing once it is
    full -- at which point it stops reading its stdin, and every command we send
    it, unkey included, goes into a pipe that never drains. A session is
    thousands of commands and prints something for many of them, so this is a
    route to a permanently keyed transmitter with no error anywhere.

    Not all of it is noise: rigctl reports a refused command in-band
    (``set_ptt: error = ...``), and a channel that discards those passes every
    liveness check while nothing reaches the radio -- a rig powered off behind a
    still-enumerated adapter keys nothing and says so on every command, to a
    reader that did not exist. Lines carrying an error go into `errors` for
    `Rig.key_failure` to read back; the rest is dropped as before.
    """
    def pump():
        try:
            for line in iter(proc.stdout.readline, ""):
                if errors is not None and "error" in line.lower():
                    errors.append(line.strip())
        except (ValueError, OSError):
            pass
    threading.Thread(target=pump, daemon=True).start()


class Rig:
    """Hamlib control via direct one-shot rigctl commands.

    Talks to the rig directly (``rigctl -m <model> -r <serial>``) rather than
    through a persistent rigctld daemon: the daemon path silently dropped PTT on
    the FT-891 at 38400 (a startup/timing race), whereas direct rigctl keys it
    reliably. A CAT PTT command persists at the rig after the process exits, so
    T 1 / play / T 0 across separate calls holds the key for the whole burst.
    Verified on-air on the FT-891 (W9SSJ, 2026-07-23).

    WHICH PATH FOR WHICH OPERATION -- the two rules that look contradictory are
    about different directions, and both hold:

      * SET/latching operations (T 1, T 0, F, M) -- USE ONE-SHOT rigctl. They
        latch at the rig, so nothing is lost when the process exits, and the
        daemon's startup race is what dropped PTT.
      * READ-back operations (f, m) -- also fine one-shot. An earlier note here
        claimed they race the CAT and return "", but that was retested on this rig
        (2026-07-24): five consecutive one-shot reads returned 7100000 identically.
        The contradicting evidence was contention with a concurrently running
        rigctld plus a shell quoting bug, not a CAT race. `get_freq` still tolerates
        "" because contention remains possible if something else opens the port.

    Exclusive CAT is required either way: this bypasses rigctld and opens the
    serial port directly, so any corpus-collector rigctld (and the scanner) must
    be stopped before an on-air run. Do not start a rigctld to compensate.

    KEYING IS NOT A CAT OPERATION on a rig that offers a PTT line (`ptt_type`).
    That changes one thing above: RTS is held by the open port rather than
    latched at the rig, so where a CAT PTT certainly survives the process, RTS
    *can* fall with it. It is not known to. Deassert-on-last-close is unmeasured
    on this adapter (`core.ptt`), and the port here is opened by hamlib, which
    clears `HUPCL` outright -- so of all the ways this station holds the keying
    line, this is the one least likely to drop it. On 2026-08-04 a session keying
    RTS through this class was killed mid-over and kept transmitting on 10.1 MHz
    until the operator powered the radio down. So `stop` kills a wedged rigctl
    before its one-shot rather than leaving it holding the port: the one-shot
    needs the port, and nothing waits on the close to lower anything.
    """

    def __init__(self, model, serial, baud, *, ptt_type="RIG", ptt_port=None,
                 rigctl_dir=None):
        self.rigctl = find_rigctl(rigctl_dir)
        self.model, self.serial, self.baud = model, serial, baud
        self.ptt_type = ptt_type
        self.ptt_port = ptt_port or serial
        self._proc = None
        self._sick = False
        self._keyer = None       # the rigctl that took the last key-down
        self._errs: deque = deque(maxlen=8)   # rigctl's own in-band refusals
        self._down = threading.Event()        # a stop() has begun; see stop()
        self.retired = False                  # the unkey ladder ran out; see retire()

    def _argv(self, *extra):
        """Every invocation, long-lived or one-shot, carries the SAME keying
        configuration. A one-shot that fell back to CAT PTT would send `TX0;`
        while RTS was still asserted, and the rig would stay keyed -- which is
        exactly the failure the one-shot in `stop` exists to prevent."""
        argv = [str(self.rigctl), "-m", str(self.model), "-r", self.serial,
                "-s", str(self.baud), "-P", self.ptt_type]
        if self.ptt_port != self.serial:
            argv += ["-p", self.ptt_port]
        return argv + list(extra)

    def _open(self):
        """One long-lived rigctl in interactive mode, commands over its stdin.

        A one-shot `rigctl` costs about 900 ms here -- process start plus opening
        and configuring the CAT port -- and that lands on the UNKEY, so the rig
        stays keyed for the best part of a second after the audio stops. That is
        precisely the slot a PACTOR peer answers in (~0.29 s into the 1.25 s
        cycle), so every reply was being transmitted over. Measured on the FT-891,
        2026-07-25. Holding the process open makes a PTT change a pipe write.

        Once `stop()` has begun (`_down`), a dead child is never replaced:
        None instead. The refusal lives HERE, at the spawn, because a caller
        that checked before calling could pass while the child was alive and
        arrive after the watchdog's stop() had killed it -- and a fresh rigctl
        spawned then re-opens the CAT port stop just released and blocks its
        one-shot unkey on it for seconds, with the rig possibly keyed. A live
        child is still returned; nothing respawns.

        Check-and-spawn is still not atomic, and stop() deliberately takes no
        lock -- no unkey path may ever wait on another thread. So a stop()
        landing while `Popen` is in flight is caught on the way out: the gate
        is read again once the spawn returns, and a child born into a stopped
        channel is killed here rather than handed to anyone -- stop()'s
        `_close` ran before this child existed and would never have seen it.
        """
        if self._proc is None or self._proc.poll() is not None:
            if self._down.is_set():
                return None
            # stderr folds into stdout so the drain sees rigctl's refusals
            # whichever stream carries them; nothing parses this pipe for data.
            proc = subprocess.Popen(
                self._argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            if self._down.is_set():
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                return None
            self._proc = proc
            # Writes go out through `_put` on this fd, never through the buffered
            # wrapper, so that a child which has stopped reading costs a deadline
            # rather than a caller that never returns.
            os.set_blocking(proc.stdin.fileno(), False)
            drain(proc, self._errs)
            self._sick = False
        return self._proc

    def _put(self, line: str) -> bool:
        """One line into the child's stdin, or False within WRITE_TIMEOUT_S."""
        # A local reference, because a concurrent `_close` nulls `_proc` and an
        # AttributeError here is not the OSError the caller catches -- it ended
        # a session with a traceback where the summary should have been. A
        # child killed under the local reference is a write that raises EPIPE,
        # which is the failure `_cmd` already speaks.
        proc = self._proc
        if proc is None:
            return False
        fd, buf = proc.stdin.fileno(), line.encode()
        deadline = time.monotonic() + WRITE_TIMEOUT_S
        while buf:
            try:
                buf = buf[os.write(fd, buf):]
            except BlockingIOError:
                pass
            if buf and not select.select([], [fd], [],
                                         max(0.0, deadline - time.monotonic()))[1]:
                return False
        return True

    def _cmd(self, *a) -> bool:
        """Write one command to the long-lived rigctl. True if it went.

        Bounded, and it does not retry into a pipe that did not take it. `rigctl`
        stops reading its stdin whenever it is busy -- a CAT transaction the rig
        is slow to answer, its own stdout backing up -- and a blocking write into
        that pipe never returns and never raises, which defeats every unkey in
        the program at once. So a write that misses its deadline marks the
        channel dead and says so; the caller stops, and `stop`'s one-shot, which
        shares neither the pipe nor the process, is what unkeys.

        A write that FAILS is different from one that is not taken: the child has
        exited, so one fresh one is worth trying before giving up.
        """
        if self._sick:
            return False
        line = " ".join(a) + "\n"
        for _ in range(2):
            if self._open() is None:
                # stop() has begun its ladder and the child is gone; `_open`
                # is what refuses to replace it, so the gate and the spawn
                # cannot be interleaved by a watchdog stop() between them.
                return False
            try:
                if self._put(line):
                    return True
            except OSError:
                self._proc = None
                continue
            break
        self._sick = True
        return False

    def set_freq(self, hz):    return self._cmd("F", str(int(hz)))
    def set_mode(self, m, passband=FILTER_HZ):
        return self._cmd("M", m, str(int(passband)))
    def set_power(self, level): return self._cmd("L", "RFPOWER", f"{level:.2f}")

    def set_power_once(self, level) -> bool:
        """RFPOWER through a one-shot, for after `stop` has closed the channel.

        `stop` latches `_down`, so the live path is gone for good by the time a
        session restores the level it borrowed -- and that is the right order:
        nothing may hold the CAT port while the unkey ladder may still need it.
        """
        self._close()
        return subprocess.run(self._argv("L", "RFPOWER", f"{level:.2f}"),
                              capture_output=True, text=True,
                              timeout=8).returncode == 0

    def get_power(self):
        """RFPOWER read-back, via a one-shot, on `get_freq`'s rule and for its
        reason: the long-lived process owns the port, so a read closes it first
        and the next PTT change opens a new one."""
        self._close()
        r = subprocess.run(self._argv("l", "RFPOWER"), capture_output=True,
                           text=True, timeout=8)
        return r.stdout.strip()

    def _devices_present(self) -> None:
        """Raise `PttError` unless the keying path's devices are still there.

        A stat and never an open -- opening a keying line asserts RTS. Both
        ports, because the command travels the CAT port and the RTS travels the
        other, and either can be the one that was never plugged in.
        """
        require_char_device(self.serial)
        if self.ptt_port != self.serial:
            require_char_device(self.ptt_port)

    def ptt(self, on):
        """Key or unkey. A key-down REFUSES, by `PttError`, a device that is
        not there -- the check nothing made before the 2026-08-10 placeholder
        session (`core.ptt.require_char_device`) -- and refuses, by returning
        False, a rig that `stop` has already taken down. The unkey checks
        nothing: it must stay possible in every state the channel can be in.
        """
        if not on:
            return self._cmd("T", "0")
        if self._down.is_set() or self.retired:
            # The watchdog's stop() can land while the main thread is still in
            # its loop; a key-down after it would put a fresh rigctl on the CAT
            # port the stop just released. And a rig whose unkey ladder ran out
            # is one nobody has confirmed is down -- not one to key again.
            return False
        self._devices_present()
        self._errs.clear()       # what key_failure reads must postdate this key
        took = self._cmd("T", "1")
        self._keyer = self._proc if took else None
        return took

    def retire(self) -> None:
        """Nothing may key through this rig again; unkeys stay allowed.

        The last rung of `core.ptt.Keyer.unkey`, reached when every path down
        has been tried and none answered. `_down` is not the same state: that
        one says a stop is in progress, and a rig that stopped cleanly may be
        keyed by a later session. This one says the transmitter's state is
        unknown, which no session may build on.
        """
        self.retired = True

    def key_failure(self):
        """Why the last key-down cannot have reached the rig, or None.

        Asked AFTER the burst, because that is the only time the question can
        be answered: the write that keys is a pipe write, and a pipe write
        succeeds into a child that is already dying on a serial port it could
        not open. That is how the 2026-08-10 session came to count carrier
        against a device that did not exist. What remains checkable once the
        burst is over: the rigctl that took the command is still standing, the
        channel is not sick, and the keying devices are still there.

        Evidence, not proof -- a live rigctl on a live port says nothing about
        a pin having moved (see `core.ptt` on what a readback proves). But
        every one of these failing is a burst that CANNOT have keyed, which is
        exactly the claim a carrier count must be grounded in.
        """
        if self._sick:
            return "the rigctl channel went dead -- a command write timed out"
        if self._keyer is None:
            return "no key-down was ever taken by a rigctl"
        if self._keyer.poll() is not None:
            return (f"the rigctl that took the key-down exited (status "
                    f"{self._keyer.returncode}) -- the command went into a "
                    f"pipe nothing was reading")
        if self._errs:
            # A rigctl that stays alive and refuses every command in-band --
            # radio off, CAT cable out, adapter still enumerated -- passes all
            # the liveness checks above. Its own words are the evidence.
            return "rigctl reported errors after the key-down: " + \
                   "; ".join(self._errs)
        try:
            self._devices_present()
        except PttError as exc:
            return str(exc)
        return None
    def get_mode(self):
        """Read mode/width after releasing our interactive CAT owner, before RF.

        This getter never opens the separate PTT port. Unlike a successful
        pipe write, its process status and actual response are required.
        The established get_freq close/read behavior is otherwise unchanged.
        """
        self._close()
        argv = [str(self.rigctl), "-m", str(self.model), "-r", self.serial,
                "-s", str(self.baud), "-P", "NONE", "m"]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReceiverReadbackError("mode readback transport failed or timed out") from exc
        return parse_mode_readback(result.stdout, result.returncode, result.stderr)

    def get_frequency_readback(self):
        """Strict CAT-only getter for an already assessed non-retuning child."""
        self._close()
        argv = [str(self.rigctl), "-m", str(self.model), "-r", self.serial,
                "-s", str(self.baud), "-P", "NONE", "f"]
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReceiverReadbackError("frequency readback transport failed or timed out") from exc
        return parse_frequency_readback(result.stdout, result.returncode, result.stderr)

    def get_freq(self):
        """Frequency read-back, via a one-shot.

        The long-lived process owns the serial port exclusively, so a concurrent
        one-shot BLOCKS on it -- which hangs rather than errors, with the rig
        possibly keyed. Reads are never latency-critical, so the interactive
        process is closed first and reopened by the next PTT change.
        """
        self._close()
        r = subprocess.run(self._argv("f"), capture_output=True, text=True,
                           timeout=8)
        return r.stdout.strip()

    def _close(self):
        """Give the process the chance to quit, then take it away.

        A dead channel is not asked politely: it is the case where the child is
        not reading, and it is holding the serial port that the one-shot unkey
        needs next.
        """
        if self._proc is None:
            return
        try:
            if not self._sick:
                self._put("q\n")
                self._proc.wait(timeout=3)
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.kill()
            try: self._proc.wait(timeout=2)
            except Exception: pass
        self._proc = None

    def stop(self):
        """Unkey and release the port -- by every path there is, in order of
        directness, because a rig left transmitting is the one failure that
        must not happen: once down the live pipe, then the one ladder
        (`core.ptt.Keyer.unkey`) -- a one-shot `T 0` judged by its exit
        status, the keying line taken down with our own ioctl when the
        one-shot cannot answer, the alarm, and a rig that stays down.

        The one-shot runs whatever the pipe write did. It shares nothing with
        it -- not the pipe, not the process, and not the serial port, which
        `_close` has just released -- so it is the unkey that holds when the
        live path is the thing that went wrong. That is also why `_close` runs
        before the ladder and not after: the live process owns the port
        exclusively, and a one-shot racing it would wait on the port instead
        of keying it down. Only a rig keyed by a serial line gives the ladder
        a line to fall to; CAT PTT has no line to drop, and for it repeated
        one-shots are the last word, then the alarm.

        `stop` runs on the watchdog's thread as well as the main one, and the
        first thing it does is close the door behind itself: `_down` stops any
        other thread's `ptt` from spawning a fresh rigctl onto the CAT port
        this ladder is about to need. Without that, the main thread's routine
        unkey could land between `_close` and the one-shot and hold the port
        against it -- the unkey that must always work, blocked by the unkey
        that was merely polite. The ladder itself takes no lock, so nothing
        here can ever wait on another thread with the transmitter keyed;
        running it twice costs a duplicate `T 0`, which costs nothing.
        """
        self._down.set()
        try: self.ptt(False)
        except Exception: pass
        try: self._close()
        except Exception: pass
        # Every invocation carries the SAME keying configuration (`_argv`), so
        # the one-shot cannot fall back to CAT PTT while RTS is still asserted.
        Keyer(OneShotRigctl(self._argv("T", "0")),
              lambda line: print(line, flush=True),
              alarm=lambda line: print(f"!! {line}", flush=True),
              ptt_device=self.ptt_port if self.ptt_type != "RIG" else None,
              on_retire=self.retire).unkey()


def _warm_output(device) -> None:
    """`core.audio.warm_output`, and a session is not worth losing to it.

    A warm-up that raises has cost this run nothing yet -- nothing is keyed and
    the burst's own open is still to come, which is where a device that is really
    gone will say so.
    """
    try:
        warm_output(device)
    except Exception:
        pass


def _play(audio_f32, device, max_key, rig, settle=0.10):
    """Key PTT, let the codec settle, play, unkey — watchdog force-unkeys on overrun.

    `settle` is the pause between keying and pushing samples, and it is per rig --
    `core.rigs.RIGS[...]["settle"]`, which is also where the reason each value is
    what it is lives.

    The watchdog guards against a HUNG playback, so it is sized to the audio, not
    to a fixed PACTOR-shaped default: a transmission longer than `max_key` is
    REFUSED up front rather than keyed and silently cut short. Truncating a
    legitimate transmission is worse than not sending it -- a clipped WSPR frame
    (110.6 s, against a 40 s default) once read as a dead antenna and cost a real
    hardware misdiagnosis.
    """
    _warm_output(device)
    import threading
    dur = len(audio_f32) / FS
    if dur > max_key:
        raise SystemExit(f"refusing to transmit: {dur:.1f}s of audio exceeds "
                         f"--max-key {max_key:.0f}s. Raise --max-key to at least "
                         f"{dur + 5:.0f} for this transmission.")
    stop = threading.Event()
    limit = dur + max(5.0, 0.25 * dur)          # generous margin over the real length

    def watchdog():
        if not stop.wait(limit + settle):
            # The full ladder, not just a write into the pipe: a playback that
            # hangs and a rigctl that stops answering are one failure seen from
            # two sides, so the force-unkey must not depend on the channel that
            # is suspect. `stop` ends at `drop_rts` -- the local ioctl that
            # needs no daemon and no reply, and it takes the transmitter off the
            # air whether or not the playback below ever comes back.
            try:
                if rig: rig.stop()
            except Exception: pass
            print(f"!! PTT watchdog fired at {limit:.0f}s — force unkey")
    wd = threading.Thread(target=watchdog, daemon=True); wd.start()
    try:
        # Inside the try, so a refused key-down still stops the watchdog and
        # still runs the unconditional unkey. `ptt(True)` raises on a device
        # that is not there; False is a channel that did not take the command.
        # Either way, audio into an unkeyed transmitter is a burst spent
        # talking to nobody -- and counted afterwards as if it were carrier.
        if rig and rig.ptt(True) is False:
            raise PttError(f"the key-down was not taken -- rigctl on "
                           f"{rig.serial} has stopped answering")
        if rig and settle:
            time.sleep(settle)
        # Per-burst open/close does stall the input occasionally (6% of blocks);
        # holding the device open starves it continuously, which is worse. Both
        # measured on the air 2026-07-28, and `play_drained` carries the rest.
        play_drained(audio_f32, FS, device)
    finally:
        if rig: rig.ptt(False)
        unkey = time.monotonic()
        stop.set()
    # The instant the carrier dropped, so a caller can tell audio it captured
    # while transmitting from audio that arrived after. A third-party receiver
    # puts the far end's answer 70-190 ms behind this moment; anything that
    # discards "whatever is queued" after the fact eats the front of it.
    return dur, unkey


def _record(seconds, device, out_wav):
    import sounddevice as sd
    n = int(seconds * FS)
    rec = sd.rec(n, samplerate=FS, channels=1, dtype="float32", device=device)
    sd.wait()
    session.write_wav(out_wav, rec[:, 0])   # stereo 48k int16, what decoders open
    return out_wav


def _decode(wav_path, run_sh=None):
    if run_sh is None:
        # The reference decoder is a separate program, and it is not
        # redistributable — so the operator names it rather than this guessing at
        # a path that exists only on the machine where the modem was written.
        run_sh = os.environ.get("HFMODEM_DECODER")
        if not run_sh:
            raise RuntimeError(
                "no reference decoder configured: set HFMODEM_DECODER to a runner "
                "that takes a WAV and prints a decode. Obtain your own copy — it "
                "cannot be shipped with this.")
    sh = Path(run_sh).expanduser()
    if not sh.exists():
        print(f"(decode skipped — {sh} not found; wav saved at {wav_path})")
        return ""
    print(f"decoding {wav_path} through {sh.name} (~2-5 min)...")
    out = subprocess.run(["bash", str(sh), str(wav_path)], capture_output=True,
                         text=True, timeout=600).stdout
    for line in out.splitlines():
        if line.startswith("###") or "CONNECT" in line or "STATUS" in line:
            print("  decoder>", line)
    return out


def run(args):
    outdir = Path(args.outdir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    dx = args.dxcall.upper()
    if args.dial:                                    # operator gives the dial directly
        dial = args.dial
        center = dial + int(spec.CENTER_FREQ_HZ)
    else:
        center = args.center or GATEWAYS.get(dx, {}).get(args.band)
        if center is None:
            raise SystemExit("no freq: pass --center or --dial, or a known "
                             f"--dxcall/--band. Known gateways: {', '.join(GATEWAYS)}")
        dial = center_to_dial(center)

    # Build the TX: a PACTOR-1 connect ADDRESSED TO THE GATEWAY (the connect frame
    # carries the *called* station's callsign — verified against the DL6MAA capture),
    # plus optional data packets. NB our callsign in the connect handshake is TBD:
    # the caller-ID exchange is part of the reverse-direction protocol this OTA test
    # is meant to reveal; --mycall is logged and used for your station identification.
    payloads = [args.message.encode()] if (args.data and args.message) else []
    audio = session.build_session(dx, payloads, sl=args.sl)
    tx_wav = str(outdir / f"tx_{args.mycall}_{dx}.wav")
    session.write_wav(tx_wav, audio)
    dur = len(audio) / FS
    print(f"TX: connect({args.mycall}->{dx})"
          f"{' + %d data pkt' % len(payloads) if payloads else ''}  |  {dur:.1f}s  |  {tx_wav}")
    print(f"    centre {center/1e6:.4f} MHz -> dial {dial/1e6:.4f} MHz {RIGS[args.rig]['mode']}"
          f"  (signal centre = dial + {spec.CENTER_FREQ_HZ:.0f} Hz; station {dx}, band {args.band})")

    if args.dry_run:
        print("DRY RUN — rendered TX only; no rig keyed. Plug a rig in and drop --dry-run to transmit.")
        return

    out_dev = find_device(args.audio_out, "out")
    in_dev = find_device(args.audio_in, "in")
    r = RIGS[args.rig]
    # The arm gate: refuse a keying path that is not there BEFORE the session
    # exists to be miscounted. Converted to a clean exit because the operator's
    # fix is a flag, not a stack trace.
    try:
        ptt_port = ptt_device(args.rig, args.serial)
    except PttError as exc:
        raise SystemExit(f"NOT KEYING: {exc}") from None
    rig = Rig(r["model"], args.serial, args.baud or r["baud"],
              ptt_type=r.get("ptt_type", "RIG"), ptt_port=ptt_port)
    try:
        if not (rig.set_mode(r["mode"]) and rig.set_freq(dial)):
            raise SystemExit(f"NOT KEYING: rigctl on {args.serial} did not take "
                             f"the mode/frequency commands -- the CAT channel "
                             f"is not standing")
        # The readback is the readback, never the number we asked for: `f`
        # answering nothing used to become `int("" or dial)`, the intended dial
        # printed as though the rig had confirmed it -- on the same CAT channel
        # the session was about to trust for every key-down.
        readback = rig.get_freq()
        if not readback.isdigit():
            raise SystemExit(f"NOT KEYING: the rig did not answer a frequency "
                             f"readback on {args.serial} (got {readback!r}) -- "
                             f"a radio that cannot be read cannot be trusted "
                             f"to key (powered off, or the port is contended)")
        readback = int(readback)
        print(f"rig dial {readback} Hz (signal centre ~{readback + int(spec.CENTER_FREQ_HZ)} Hz).")
        # A beat between tuning and the first key. No channel check runs here:
        # whether to transmit on a shared band is the operator's judgement.
        time.sleep(3)
        for attempt in range(1, args.attempts + 1):
            print(f"--- attempt {attempt}/{args.attempts}: keying {dur:.1f}s ---")
            _play(audio, out_dev, args.max_key, rig)
            # The burst is only a transmission if the channel that keyed it is
            # still standing. Stopping is cheaper than every later line of this
            # loop reporting a probe that never touched the radio.
            fail = rig.key_failure()
            if fail is not None:
                raise SystemExit(f"TRANSMISSION NOT CONFIRMED: {fail} -- "
                                 f"stopping rather than decoding replies to a "
                                 f"burst that never went out")
            rx_wav = str(outdir / f"rx_{dx}_{attempt}.wav")
            print(f"receiving {args.listen}s for {dx}'s reply...")
            _record(args.listen, in_dev, rx_wav)
            _decode(rx_wav)
            time.sleep(args.gap)
    finally:
        rig.stop()
        print("PTT off.")


def main():
    p = argparse.ArgumentParser(description="shrike on-air PACTOR probe")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--rig", choices=list(RIGS), default="ft891")
    p.add_argument("--serial",
                   help="CAT serial port, e.g. /dev/cu.usbserial-XXXXB0 -- "
                        "`ls /dev/cu.*` to find yours")
    p.add_argument("--baud", type=int, default=0, help="override CAT baud")
    p.add_argument("--audio-out", help="TX audio device (name substring or index)")
    p.add_argument("--audio-in", help="RX audio device (name substring or index)")
    p.add_argument("--mycall", default="N0CALL", help="YOUR callsign (identifies you)")
    p.add_argument("--dxcall", help="the gateway to call -- no default, this "
                                    "callsign goes on the air")
    p.add_argument("--band", default="40m")
    p.add_argument("--center", type=int, help="explicit channel CENTRE freq Hz (dial derived: centre-1500)")
    p.add_argument("--dial", type=int, help="explicit DIAL freq Hz (tuned as-is; overrides --center/table)")
    p.add_argument("--data", action="store_true", help="also send a data packet (tests ACK/NAK)")
    p.add_argument("--message", default="TEST DE SHRIKE")
    p.add_argument("--sl", type=int, default=2)
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--listen", type=float, default=15.0, help="RX window after each TX")
    p.add_argument("--gap", type=float, default=5.0)
    p.add_argument("--max-key", type=float, default=40.0, help="PTT watchdog force-unkey")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--outdir", default="captures/ota")
    args = p.parse_args()
    if args.list_devices:
        list_devices(); return
    # Checked here rather than by `required=True` so --list-devices still runs
    # without one. The connect carries the CALLED station's callsign, so a
    # default is this probe calling a stranger on the operator's behalf.
    if not args.dxcall:
        p.error("--dxcall is required: the station to call. "
                f"Known gateways: {', '.join(GATEWAYS)}")
    # Same shape as --dxcall: required by hand so --dry-run, which never opens
    # the rig, still runs without naming a port.
    if not args.serial and not args.dry_run:
        p.error("--serial is required: the CAT port, e.g. "
                "/dev/cu.usbserial-XXXXB0 -- `ls /dev/cu.*` to find yours")
    run(args)


if __name__ == "__main__":
    # A signal must unwind, not just kill: every unkey here hangs off a
    # `finally`. See onair._unkey_on_signal for what this cost once.
    from hfmodem.shrike.onair import _unkey_on_signal
    _unkey_on_signal()
    main()
