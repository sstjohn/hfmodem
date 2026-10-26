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
Hamlib rigctl tunes, LineKeyer keys, and sounddevice plays and captures audio.
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
import contextlib
import os
import re
import select
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path


from . import session, spec
from ..core.audio import play_drained, warm_output
from ..core.claim import ClaimError
from ..core.devices import find_device, list_devices
from ..core.occupied import FILTER_HZ
from ..core.rxreadiness import RadioMode, parse_mode_readback, parse_frequency_readback, ReceiverReadbackError
from ..core.ptt import (LineKeyer, PttError, derive_ptt_port,
                        require_char_device)
from ..core.rig import arm_deadman
from ..core.rigs import RIGS, flight_defaults, mode_passband

FS = spec.SAMPLE_RATE

# How long a command may take to reach rigctl before the channel is given up on.
# Generous next to a pipe write (microseconds) and short next to the 1.25 s cycle
# it has to fit inside, so an unkey that is going to fail fails while there is
# still time to do something about it.
WRITE_TIMEOUT_S = 0.25

# How long `_close` waits for rigctl to quit when asked, to end when terminated,
# then to die when killed.
QUIT_WAIT_S = 3.0
TERM_WAIT_S = 2.0
KILL_WAIT_S = 2.0

# What the unwind gets before the interpreter is taken down under it. Every other
# deadman in this tree arms with the key ALREADY DOWN -- `Rig.panic` unkeys first
# -- and can afford `PANIC_BUDGET_S`. Here the key is still up when a stop signal
# arms it, and `Rig.stop` then walks its whole ladder: the keying line dropped and
# handed back, ioctls on a port this process already holds, then rigctl taken off
# the serial port -- the quit written to its pipe, and the waits for it to quit,
# to end and to die. Firing before that ladder finishes would leave the CAT child
# to the reaper and the line to the close of a dying process, rather than to the
# ladder that reads it back.
#
# Bounded above by what reaps this modem: `tools/onair.sh` ends its children
# through the claim, which gives each `REAP_GRACE_S` to unkey itself and then
# sends SIGKILL, ending the handler and the deadman with it.
LADDER_WORST_S = WRITE_TIMEOUT_S + QUIT_WAIT_S + TERM_WAIT_S + KILL_WAIT_S
DEADMAN_MARGIN_S = 0.25
DEADMAN_BUDGET_S = LADDER_WORST_S + DEADMAN_MARGIN_S

# PACTOR-3 is centred at spec.CENTER_FREQ_HZ (1500 Hz) in the audio passband, so on
# USB the emitted RF centre = dial + 1500 Hz. Winlink's RMSChannels list quotes the
# CHANNEL CENTRE; the operator tunes dial = centre - 1500. Keep that offset explicit
# here so the rig is never silently mistuned by 1500 Hz.
def center_to_dial(center_hz: float) -> int:
    return int(round(center_hz - spec.CENTER_FREQ_HZ))

NO_KEYING_LINE = "transmit unavailable: no keying line in this profile"


def ptt_device(rig: str, serial: str) -> str | None:
    iface = RIGS[rig].get("ptt_iface")
    if iface is None:
        return None
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


def drain(stream, errors: deque | None = None) -> None:
    def pump():
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode(errors="replace")
                if errors is not None and "error" in line.lower():
                    errors.append(line.strip())
        except (ValueError, OSError):
            pass
    threading.Thread(target=pump, daemon=True).start()


def cat_once(argv, timeout, *, own_session=False):
    """One rigctl invocation under the launch admission, answered as
    `subprocess.run(..., capture_output=True, text=True)` answers."""
    from ..core.native import spawn_native

    child = spawn_native(argv, claim_fd=9, own_session=own_session)
    try:
        out, err = child._process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        child._process.kill()
        child._process.communicate()
        raise
    return subprocess.CompletedProcess(argv, child._process.returncode,
                                       out.decode(errors="replace"), err.decode(errors="replace"))


class Rig:
    def __init__(self, model, serial, baud, *, ptt_type="LINE", ptt_port=None,
                 rigctl_dir=None):
        if ptt_type != "LINE":
            raise ValueError("ptt_type must be LINE")
        self.rigctl = find_rigctl(rigctl_dir)
        self.model, self.serial, self.baud = model, serial, baud
        self.ptt_type = ptt_type
        self.ptt_port = ptt_port
        self._proc = None
        self._sick = False
        self._errs: deque = deque(maxlen=8)
        self._down = threading.Event()
        self.retired = False
        self._keyer = (LineKeyer(ptt_port, lambda line: print(line, flush=True),
                                alarm=lambda line: print(f"!! {line}", flush=True),
                                abort=self._down.is_set, on_retire=self.retire)
                       if ptt_port else None)

    def _argv(self, *extra):
        return [str(self.rigctl), "-m", str(self.model), "-r", self.serial,
                "-s", str(self.baud), "-P", "NONE", *extra]

    def _open(self):
        from ..core.native import spawn_native

        try:
            if self._proc is None or not self._proc.alive():
                if self._proc is not None:
                    self._close()        # reap and close the dead child before any replacement
                if self._down.is_set():
                    return None
                proc = self._proc = spawn_native(self._argv(), claim_fd=9, long_lived=True,
                                                 device=self.serial)
                if self._down.is_set():
                    self._close()
                    return None
                # Writes go out through `_put` on this fd, never through the buffered
                # wrapper, so that a child which has stopped reading costs a deadline
                # rather than a caller that never returns.
                os.set_blocking(proc.stdin.fileno(), False)
                drain(proc.stdout, self._errs)
                drain(proc.stderr, self._errs)
                self._sick = False
        except ClaimError as exc:
            if getattr(exc, "child", None) is not None:
                self._proc = exc.child   # kept, so that `_close` can still end it
            self._errs.append(str(exc))
            self._sick = True
            return None
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
        if self._sick:
            return False
        line = " ".join(a) + "\n"
        for _ in range(2):
            if self._open() is None:
                return False
            try:
                if self._put(line):
                    return True
            except OSError:
                # The handle stays: a broken pipe does not prove the child is
                # gone, and only `_close` may decide it is.
                continue
            break
        self._sick = True
        return False

    def set_freq(self, hz):    return self._cmd("F", str(int(hz)))
    def set_mode(self, m, passband=FILTER_HZ):
        return self._cmd("M", m, str(int(passband)))
    def set_power(self, level): return self._cmd("L", "RFPOWER", f"{level:.2f}")

    def _once(self, *cmd):
        """One invocation of its own, after the interactive child hands the port back."""
        self._close()
        return cat_once(self._argv(*cmd), 8)

    def set_power_once(self, level) -> bool:
        """RFPOWER through a one-shot, for after `stop` has closed the channel.

        In its own session, so the operator's Ctrl-C does not reach it, and over
        before an armed deadman, so a rig that does not answer is a restore
        reported as failed and not a process killed.
        """
        try:
            self._close()
            timeout = within_deadman(8)
            if timeout <= 0:
                return False
            return cat_once(self._argv("L", "RFPOWER", f"{level:.2f}"), timeout,
                            own_session=True).returncode == 0
        except (ClaimError, subprocess.TimeoutExpired):
            return False

    def get_power(self):
        try:
            return self._once("l", "RFPOWER").stdout.strip()
        except ClaimError:
            return ""

    def arm(self) -> None:
        if self._keyer is None:
            raise PttError(NO_KEYING_LINE)
        if self._down.is_set() or self.retired:
            raise PttError("the rig has been taken down")
        try:
            self._keyer.arm()
        except BaseException:
            self.stop()
            raise

    def ptt(self, on):
        if self._keyer is None:
            return not on
        if on:
            if self._down.is_set() or self.retired:
                return False
            # WHY: stop can land after this check; arm then raises PttError
            # instead of returning False. Both refuse the key safely, so let
            # the exception propagate to the caller's transmit-abort path.
            self.arm()
        return self._keyer.key(on)

    def retire(self) -> None:
        self.retired = True

    def key_failure(self):
        if self._keyer is None:
            return NO_KEYING_LINE
        key = self._keyer.last_key
        if key is None or key.verdict != "confirmed" or not key.state_read or key.state is not True:
            return "no key-down was confirmed on the keying line"
        if self._down.is_set() or self.retired or self._keyer.must_retire:
            return "the keying line has been stopped or retired"
        drive = self._keyer.last_drive
        if (drive is None or drive.verdict != "confirmed" or not drive.state_read
                or drive.state is not self._keyer.last_on):
            return "the last keying line drive was not confirmed by readback"
        return None

    def get_mode(self):
        """Read mode/width after releasing our interactive CAT owner, before RF.

        This getter never opens the separate PTT port. Unlike a successful
        pipe write, its process status and actual response are required.
        The established get_freq close/read behavior is otherwise unchanged.
        """
        try:
            result = self._once("m")
        except (OSError, ClaimError, subprocess.TimeoutExpired) as exc:
            raise ReceiverReadbackError("mode readback transport failed or timed out") from exc
        return parse_mode_readback(result.stdout, result.returncode, result.stderr)

    def get_receiver_filters(self):
        from ..core.ftdx10 import read_filters
        try:
            result = self._once("w", "VS;", "w", "MD0;", "w", "MD1;", "w", "SH0;", "w", "RF0;")
        except (OSError, ClaimError, subprocess.TimeoutExpired) as exc:
            raise ReceiverReadbackError("raw receiver readback transport failed or timed out") from exc
        if (result.returncode or "error =" in result.stdout or "error =" in result.stderr
                or "RPRT -" in result.stdout or "RPRT -" in result.stderr):
            raise ReceiverReadbackError("raw receiver readback failed")
        vfos = re.findall(r"\bVS([01]);", result.stdout)
        modes = (re.findall(rf"\bMD{vfos[0]}([0-9A-F]);", result.stdout)
                 if len(vfos) == 1 else [])
        width, roof = read_filters(result.stdout)
        if len(modes) != 1 or width is None or roof is None:
            raise ReceiverReadbackError("missing or unknown raw receiver readback")
        return RadioMode("PKTUSB" if modes[0] == "C" else modes[0], width,
                         result.stdout), roof

    def get_frequency_readback(self):
        """Strict CAT-only getter for an already assessed non-retuning child."""
        try:
            result = self._once("f")
        except (OSError, ClaimError, subprocess.TimeoutExpired) as exc:
            raise ReceiverReadbackError("frequency readback transport failed or timed out") from exc
        return parse_frequency_readback(result.stdout, result.returncode, result.stderr)

    def get_freq(self):
        try:
            return self._once("f").stdout.strip()
        except ClaimError:
            return ""

    def _close(self):
        proc = self._proc
        if proc is None:
            return
        try:
            if not self._sick:
                self._put("q\n")
                proc.wait_gone(QUIT_WAIT_S)
        except Exception:
            pass
        if proc.alive():
            proc.terminate()
            if not proc.wait_gone(TERM_WAIT_S):
                proc.kill()
                if not proc.wait_gone(KILL_WAIT_S):
                    self._sick = True
                    raise ClaimError(f"native CAT termination unresolved for pid {proc.pid}")
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        self._proc = None

    def stop(self):
        with signals_held():
            self._down.set()
            try:
                if self._keyer is not None:
                    self._keyer.unkey()
            finally:
                try:
                    if self._keyer is not None:
                        self._keyer.hand_back()
                finally:
                    try:
                        self._close()
                    except Exception as exc:
                        print(f"!! CAT child not ended at stop: {exc}", flush=True)


def within_deadman(timeout: float) -> float:
    """`timeout`, cut to end a margin before an armed deadman would fire."""
    left = signal.getitimer(signal.ITIMER_REAL)[0]
    return timeout if left == 0 else min(timeout, left - DEADMAN_MARGIN_S)


def arm_unwind_deadman() -> None:
    """Arm the unwind's deadman, unless one is already counting: a later signal
    must not move an armed deadman later."""
    if signal.getitimer(signal.ITIMER_REAL)[0] == 0:
        arm_deadman(DEADMAN_BUDGET_S)


_depth = 0                          # `signals_held` blocks running on the main thread
_pending: list[int] | None = None   # stop signals held; None while each is answered


def hold_signal(signum: int) -> bool:
    """Whether a stop signal waits for the end of a hold rather than being
    answered now. The handler asks, so holding never swaps a handler.

    The first stop signal of an unwind is answered, and every one after it is
    held until the teardown's hold has ended: a second Ctrl-C landing between a
    `finally:` and the hold it opens would otherwise skip `Rig.stop` entirely.
    """
    global _pending
    if _pending is not None:
        _pending.append(signum)
        return True
    _pending = []
    return False


@contextlib.contextmanager
def signals_held(after_s: float | None = None):
    """Let the unkey ladder finish before a stop signal is answered.

    A Ctrl-C landing inside `Rig.stop` could cut its ladder short between the
    keying line's unkey, its hand-back with the readback, and the CAT child's
    close, each of which a stopped session still owes the rig. A signal
    the hold takes still arms the deadman, so a wedged ladder stays bounded, and
    is delivered when the outermost hold ends, so a caller can keep the rig's
    own restore inside it. That end gives an armed deadman `after_s` for the
    rest of the caller's unwind, or disarms it when there is none.

    Whether stops are held is the one global `_pending`, which the stop handler
    reads and each end of a hold changes in a single store. A signal sent to the
    process may be taken by any thread and answered on the main one at whatever
    bytecode it has reached; there is no swap of handlers for it to land in.
    """
    global _depth, _pending
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if _pending is None:
        _pending = []
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if not _depth:
            _release(after_s)


def _release(after_s: float | None) -> None:
    """End the outermost hold: the deadman handed over while stops are still
    held, then one store that answers every later stop at once, then the first
    held stop delivered. One landing while that is delivered is dropped, since
    the unwind already has its stop."""
    global _pending
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        signal.setitimer(signal.ITIMER_REAL, after_s or 0)
    taken, _pending = _pending, None
    try:
        for sig in taken[:1]:
            signal.raise_signal(sig)
    finally:
        _pending = None


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


def _play(audio_f32, device, max_key, rig, settle=0.10, *, admit=None):
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
    if admit is not None:
        admit()
    _warm_output(device)
    if admit is not None:
        admit()
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
        if admit is not None:
            admit()
        if rig and rig.ptt(True) is False:
            raise PttError("the key-down was not confirmed on the keying line")
        if rig and settle:
            time.sleep(settle)
        # Per-burst open/close does stall the input occasionally (6% of blocks);
        # holding the device open starves it continuously, which is worse. Both
        # measured on the air 2026-07-28, and `play_drained` carries the rest.
        if admit is not None:
            admit()
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
    if not args.dry_run and RIGS[args.rig].get("ptt_iface") is None:
        print(NO_KEYING_LINE, flush=True)
        in_dev = find_device(args.audio_in, "in")
        _record(args.listen, in_dev, str(outdir / "listen.wav"))
        return
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
              ptt_type="LINE", ptt_port=ptt_port)
    try:
        if not (rig.set_mode(r["mode"], mode_passband(r, FILTER_HZ)) and rig.set_freq(dial)):
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
        rig.arm()
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
    p.add_argument("--rig", choices=list(RIGS))
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
    args = flight_defaults(p.parse_args())
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
    # `finally`. See onair._unkey_on_signal for what this cost once. Run as
    # `-m`, this file is a second copy beside the `hfmodem.shrike.ota` that
    # onair imports, so the session runs from that one, whose hold the stop
    # handler reads.
    from hfmodem.shrike import ota
    from hfmodem.shrike.onair import _unkey_on_signal
    _unkey_on_signal()
    ota.main()
