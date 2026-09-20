# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Real-radio backend: key a rig and move audio between besra and a sound card.

  * **PTT on the keying line, CAT over a persistent hamlib `rigctl` connection.**
    With `line_ptt=True` and a `ptt_device`, `Rig` holds that serial port open for
    the session and keys it with one ioctl (`core.ptt.RtsPtt`), reading the line
    back both ways; `rigctl` is left to set frequency and mode, which it does
    before there is any RF about. That split is the answer to 2026-08-09, when
    rigctld stopped answering on the first key-down of a session on the air — see
    `core.ptt.drop_rts` for the incident: a transmit path that has to work while
    the antenna radiates must not travel the link that fails only then.

    Without `line_ptt` every key goes through hamlib, and a per-command `rigctl`
    spawn costs ~900 ms on the FT-891 (opening and configuring the CAT port) — a
    cost that lands on the *unkey*, so the rig trails the audio by most of a
    second and ARDOP's turnaround transmits over the peer's reply. So a PTT change
    is a pipe write to one long-lived process. The transport lives behind the `Rig`
    interface: pass `rigctld="host:port"` to route through the shared daemon
    (netrigctl) instead of opening the CAT port directly — the station's rule, so
    nothing contends for the port. Either way it is one long-lived `rigctl`. And
    the emergency unkey does **not** use the persistent pipe — a wedged pipe blocks
    like a wedged socket, so a kill path must not share the transport with the
    thing that hangs.
  * **`sounddevice` for TX playback and RX capture** at the card's 48 kHz,
    resampled to and from **ARDOP's 12 kHz, which is normative** — `core.resample`
    is the only rate conversion in besra and it happens at this edge only. The
    12 kHz core does not move; harmonising it to 48 kHz would break the
    cross-decode gate against the real ardopcf binary.

`RadioLink` binds all this to a `BesraModem` in place of the virtual air: the
modem renders a frame → key → play → unkey; between transmissions the capture
stream is decoded in a rolling window — no energy gate anywhere, see
`RollingDecoder` — and each frame is handed to the modem's session. Nothing here
runs without a radio, so `sounddevice` is imported lazily and every entry point
has a hardware-free dry path.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

from .phy.demodulator import SAMPLE_RATE, Demodulator, RxFunnel, frame_span  # 12000
from ..core import band, levels, rates
from ..core.audio import play_drained, warm_output
from ..core.occupied import FILTER_HZ
from ..core.ptt import Keyer, LineKeyer, OneShotRigctl
from ..core.resample import from_card, to_card
from ..core.rigs import RIGS
from ..core.wav import CaptureRecorder

FS_RADIO = rates.CARD_RATE_HZ               # sound-card rate

#: ARDOP audio is centred at 1500 Hz, so on USB the RF centre = dial + 1500.
#: The dial convention itself is one fact for the whole station.
CENTRE_OFFSET_HZ = band.DIAL_OFFSET_HZ
center_to_dial = band.dial_hz


def find_rigctl(rigctl: str = "rigctl") -> Path:
    """Hamlib's `rigctl`, or a refusal naming what is missing and where we looked.

    A mirror of `shrike.ota.find_rigctl` rather than an import of it — protocols
    do not import each other (`tests/gates/test_import_direction.py`) — and the
    refusal lives at construction for the same reason as there: a bare "rigctl"
    handed to `Popen` fails as a raw FileNotFoundError from inside a keying path,
    with the audio device already open, and at the radio that reads as an audio
    fault.
    """
    if "/" in rigctl:
        exe = Path(rigctl).expanduser()
        if not os.access(exe, os.X_OK):
            raise SystemExit(f"NOT KEYING: no executable rigctl at {exe} -- "
                             f"check the path that was named, or pass a bare "
                             f"'rigctl' and let PATH answer")
        return exe
    found = shutil.which(rigctl)
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


class Rig:
    """Hamlib freq/mode over a persistent `rigctl`; PTT on the keying line or hamlib.

    With `line_ptt`, every key and unkey is an ioctl on a descriptor this object
    holds for the session, read back both ways, and no transmit-time path touches
    hamlib at all — the daemon is asked for frequency and mode at arm time and
    nothing after. That is the configuration this station transmits in.

    Without it, set and PTT commands are pipe writes to one long-lived interactive
    `rigctl`, so a key change is not a ~900 ms process spawn (see the module
    docstring). The transport is deliberately isolated here — persistence is
    settled, the shared backend is not — and `set`/`ptt` never read back, so that
    path needs none of the close-before-read machinery a read-back path would.

    `unkey` is the kill path, and what it is independent *of* depends on which one
    is in use. On the line it is one ioctl on the descriptor already open: nothing
    to spawn, no reply to wait for, and no way for a wedged daemon to delay it.
    Through hamlib it kills the long-lived process to release the CAT port and keys
    down through a *fresh* one-shot, because the persistent pipe can block if the
    CAT wedges and the one path that must always work cannot share a transport with
    the thing that hangs. The watchdog, `stop`, and the registered atexit handler
    all go through it.
    """

    def __init__(self, model: int, serial: str, baud: int,
                 rigctl: str = "rigctl", rigctld: str | None = None,
                 settle: float = 0.0, ptt_device: str | None = None,
                 line_ptt: bool = False) -> None:
        """`settle` is the pause between key-down and the first sample. It is a
        property of the radio, so it lives on the radio; `core.rigs.RIGS` holds the
        value and the reason each one is what it is, and `named` is the only thing
        that should set it. Zero is what a rig built from a raw model number gets,
        because nobody said."""
        # A bare command name resolves on PATH; pass a full path for a private
        # build. Either way `find_rigctl` refuses here, before any audio device
        # is open, when there is no rigctl to be had.
        self.rigctl = str(find_rigctl(rigctl))
        self.model, self.serial, self.baud = model, serial, baud
        self.settle = settle
        # A rigctld address (host:port) routes through the shared daemon (netrigctl,
        # model 2) instead of opening the CAT port directly — the station's rule, so
        # nothing contends for the port. rigctld itself is started by the operator
        # against the real rig; besra only ever talks to it.
        self.rigctld = rigctld
        # The serial line whose RTS is the PTT. Without `line_ptt` it is the
        # emergency unkey's only route — the hamlib one-shot travels the link that
        # failed on 2026-08-09 (`core.ptt.drop_rts`), and a last resort that shares
        # the failure mode of the thing it is backing up is not one. With `line_ptt`
        # it is how this rig keys, full stop; see `hfmodem.core.ptt`.
        self.ptt_device = ptt_device
        self.line_ptt = bool(line_ptt)
        self._keyer: LineKeyer | None = None
        if self.line_ptt:
            if not ptt_device:
                # Falling back to the daemon here would be the defect wearing the
                # fix's name: the operator asked for the line and would be told
                # nothing while every key went back over the link RF disrupts.
                raise ValueError("line_ptt needs a ptt_device — the line to key")
            self._keyer = LineKeyer(ptt_device, log.info, alarm=log.error)
            self._keyer.arm()           # opens deasserted, announces the split
        self.retired = False
        self._stopped = False
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        #: An emergency unkey has begun. `_write` then stops replacing a dead
        #: child, because a fresh rigctl would re-open the CAT port the ladder's
        #: one-shot needs next and block it for seconds with the rig possibly
        #: keyed — the same door `shrike.ota.Rig._open` closes with its `_down`.
        self._down = threading.Event()
        atexit.register(self._atexit_unkey)

    @classmethod
    def named(cls, name: str, serial: str, **kw) -> "Rig":
        r = RIGS[name]
        return cls(r["model"], serial, r["baud"], settle=r["settle"], **kw)

    def _base(self) -> list[str]:
        if self.rigctld:
            return [self.rigctl, "-m", "2", "-r", self.rigctld]      # netrigctl -> daemon
        return [self.rigctl, "-m", str(self.model), "-r", self.serial, "-s", str(self.baud)]

    def _write(self, *a: str) -> bool:
        """One line to the persistent rigctl. True when the write was taken —
        which is evidence the command left here, not that a rig obeyed it —
        False once both attempts fail, and False is proof the command cannot
        have gone."""
        with self._lock:
            for attempt in (1, 2):               # reopen once on a broken pipe
                if self._proc is None or self._proc.poll() is not None:
                    if self._down.is_set():      # see `_down`: nothing respawns
                        return False
                    self._proc = subprocess.Popen(
                        self._base(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
                try:
                    self._proc.stdin.write(" ".join(a) + "\n")
                    self._proc.stdin.flush()
                    return True
                except (BrokenPipeError, OSError) as e:
                    self._proc = None
                    if attempt == 2:
                        log.error("rigctl %s failed: %s", " ".join(a), e)
        return False

    def set_freq(self, hz: int) -> None: self._write("F", str(int(hz)))
    def set_mode(self, mode: str, passband: int = FILTER_HZ) -> None:
        self._write("M", mode, str(int(passband)))

    def qsy(self, hz: int) -> int | None:
        """Set the dial and read it back — one one-shot — or None when the rig
        will not say.

        The set and its readback travel in the same rigctl invocation: the
        persistent pipe's writes are asynchronous, so a set through it followed
        by a kill-and-read could overtake the set and verify nothing. Reads need
        exclusive access, so the persistent process is killed first, as
        `identify` does. Unlike the model — which netrigctl hides, so `identify`
        answers "?" through the daemon — the frequency read passes through to
        the rig, so None here means nobody is talking to the radio, not that
        the question cannot be asked.
        """
        with self._lock:
            self._kill()
        try:
            r = subprocess.run(self._base() + ["F", str(int(hz)), "f"],
                               capture_output=True, text=True, timeout=8)
        except subprocess.TimeoutExpired:
            return None
        out = r.stdout.split()
        return (int(out[-1]) if r.returncode == 0 and out and out[-1].isdigit()
                else None)

    def ptt(self, on: bool) -> bool:
        """Key or unkey, and say whether it can have happened: on the line the
        verdict is the keyer's readback, through hamlib it is whether the pipe
        took the command. False on a key-up is proof the burst cannot have gone
        out, and `RadioLink._transmit` keys nothing into it."""
        # A retired rig refuses key-UP but never key-down: after a signal-time
        # unkey, a racing transmit thread must not put the line back up in the
        # window before the interpreter exits.
        if on and self.retired:
            log.warning("PTT up refused: rig retired")
            return False
        if self.line_ptt:
            return self._key_line(on)
        return self._write("T", "1" if on else "0")

    # -- the keying line ---------------------------------------------------

    def _key_line(self, on: bool) -> bool:
        """Key or unkey by ioctl; `LineKeyer` says what the line read back and
        deasserts on an unconfirmed key-up. What is ours is the retire: a
        transmitter nobody has confirmed is down is not one this rig may key
        again."""
        if self._keyer is None:                  # released; the port is closed
            if on:
                log.error("PTT up refused: the keying line has been handed back")
                return False
            return True                          # nothing is up to take down
        took = self._keyer.key(on)
        if self._keyer.must_retire:
            self.retire()
        return took

    def _release_line(self) -> None:
        keyer, self._keyer = self._keyer, None
        if keyer is not None:
            keyer.hand_back()

    def retire(self) -> None:
        """Nothing may key through this rig again; unkeys stay allowed."""
        self.retired = True

    def identify(self) -> str:
        """The model name `rigctl` reports for the port we were given — for an
        arm-time check, since keying the wrong radio is the expensive mistake. Reads
        need exclusive access, so the persistent process is killed first.

        `"?"` means unreadable, and a timeout is one more way of being unreadable.
        Through rigctld the answer is already "?" every time — the daemon owns the
        rig and ``\\dump_caps`` names the netrigctl backend rather than the model — so
        the caller's unverified path is the ordinary path here, and the dial
        readback is what actually decides whether this process may key. Raising
        instead cost arm 03 of the 2026-08-29 night slot, which died before it keyed
        on a ``\\dump_caps`` that answered a minute later; a check whose failure is
        already survivable must not be able to end the run."""
        with self._lock:
            self._kill()
        try:
            r = subprocess.run(self._base() + ["\\dump_caps"], capture_output=True,
                               text=True, timeout=8)
        except subprocess.TimeoutExpired:
            return "?"
        for line in r.stdout.splitlines():
            if "Model name:" in line:
                return line.split(":", 1)[1].strip()
        return "?"

    def unkey(self, why: str) -> None:
        """Independent emergency unkey — the watchdogs, the refused key-up and the
        signal handlers.

        `why` is required and the warning is unconditional, because the value of
        the word is that it is rare: an operator who reads it at the end of every
        clean run is one who reads past it on the run that meant it. The ordinary
        teardown is `stop`, which runs the same ladder and says so at INFO.
        """
        log.warning("emergency unkey: %s", why)
        self._unkey()

    def _unkey(self) -> None:
        """The ladder, whatever brought us here.

        On the keying line that is one ioctl on the descriptor already open — there
        is no process to kill, no reply to wait for, and nothing a wedged daemon can
        delay. Otherwise: close the door (`_down`), kill the long-lived process
        (freeing the CAT port), and run the one ladder — `core.ptt.Keyer.unkey`,
        where the one-shot's returncode check, the fall to the keying line, the
        alarm wordings and the retire live for every modem at once.

        Lock-free, deliberately: this used to wait on `_lock` to kill the pipe,
        and `_write` holds that lock across a blocking `stdin.write` — a rigctl
        that stops reading is precisely the failure being escaped, so the
        emergency unkey could queue behind the wedge it exists to escape, with
        the transmitter keyed. Killing the child without the lock breaks any
        such write with EPIPE instead, and `_down` keeps the CAT port free for
        the one-shot.
        """
        if self.line_ptt:
            self._key_line(False)
            return
        self._down.set()
        self._kill()
        Keyer(OneShotRigctl(self._base() + ["T", "0"]), log.info,
              alarm=log.error, ptt_device=self.ptt_device,
              on_retire=self.retire).unkey()

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
            self._proc.wait(timeout=2)
        except Exception:
            pass
        self._proc = None

    def stop(self) -> None:
        """The ordinary end of a session. The ladder logs what it did at INFO —
        `PTT OFF -> …`, then the line handed back — which is the whole record a
        clean shutdown needs."""
        self._unkey()
        self._release_line()
        self._stopped = True

    def _atexit_unkey(self) -> None:
        """The backstop for the exits `stop` never reaches: an uncaught
        exception out of a keyed session, a `SystemExit` past the `finally`.

        Nothing can key through a stopped rig again — the line has been handed
        back, `_down` keeps `_write` from respawning — so after `stop` there is
        nothing here to put down, and running the ladder anyway spent a second
        one-shot at a CAT port already given up and pushed the reason the
        session ended out of the operator's last screen.
        """
        if self._stopped:
            return
        try:
            self.unkey("atexit")
            self._release_line()
        except Exception:
            pass


def tune_atu(rig: Rig, out_device=None, *, seconds: float = 4.0, freq_hz: float = 1500.0,
             amplitude: float = 0.15, max_key_s: float = 8.0) -> None:
    """Key a short steady tone for ATU tuning — the safe way; never hand-roll a keyed
    carrier.

    Everything that stuck the finals once is designed out: the audio device is warmed
    up *before* PTT so the tone is flowing the instant we key (no keyed-into-dead-air),
    PTT settles before audio, a watchdog force-unkeys through `Rig.unkey`'s independent
    kill path if the bounded play overruns, and PTT is released in `finally`. The tone
    is short and low so a lost process self-recovers fast — and the rig's own TX
    time-out timer is the hardware backstop. Let it complete; do not interrupt a keyed
    carrier."""
    n = int(seconds * FS_RADIO)
    tone = (amplitude * np.sin(2 * np.pi * freq_hz * np.arange(n) / FS_RADIO)).astype(np.float32)

    warm_output(out_device)

    stop = threading.Event()

    def watchdog():
        if not stop.wait(max_key_s):
            rig.unkey("tune watchdog")           # independent, kill-proof
            log.error("tune watchdog fired at %ss — force unkey", max_key_s)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        rig.ptt(True)
        time.sleep(rig.settle)
        play_drained(tone, FS_RADIO, out_device)
    finally:
        # Same verdict, same ladder as `RadioLink._transmit`: a False through
        # hamlib is a T 0 that never left a rig whose CAT PTT latches.
        if rig.ptt(False) is False and not rig.line_ptt:
            rig.unkey("the persistent pipe did not take the key-up")
        stop.set()


_TAIL = np.zeros(4800, dtype="<i2")      # the demodulator's own flush pad


class RollingDecoder:
    """Decode a continuous capture in overlapping windows, reporting each frame once.

    The receive path for both `RadioLink` and the monitor: nothing is gated. Every
    `STEP_S`, the last `OVERLAP_S + STEP_S` seconds are decoded whole and any frame
    already reported is dropped by its position in the stream.

    An energy gate is what this replaces, and on a real channel it could not work.
    The gate opened above an absolute 200 int16 RMS and closed on quiet, but the
    station's own quiet channel measures **3935 int16 RMS** (10th percentile of 0.1 s
    frames) at the verified 0.040 codec working point — capture peak −7.0 dBFS, 0.00%
    railed, a properly levelled receiver and not a hot one. Twenty times the gate: it
    opened on the first block and never closed, and the modem got nothing (measured:
    0 bursts across 44 s of `rf-corpus/7102k_065457.wav`). The other end of the gate
    was as wrong — a frame still decodes at a 0.1 s RMS of 2.2 int16, −84 dBFS — so
    the demodulator is level-blind and the gate could only ever discard sensitivity.
    Windowing on the clock is level-blind too, and it makes the live path at least as
    complete as a whole-capture pass — on `7102k_065457.wav` it is better, finding a
    third ConReq (on the repeat grid, so real) that a 44 s pass buries: the
    demodulator's silence gate is a fraction of the *capture's* peak, and a 6 s
    window sets a far lower bar than a 44 s one.

    `STEP_S` is the receive latency, and ARQ sets it. A ConAck arrives ~0.7 s after
    our 1.745 s ConReq ends and the confirming ConAck must be keyed before the ConReq
    repeats (`ArqSession._connect_interval`, 2.0 s from the end of the last one), so
    the budget is ~0.6 s; the data ACKs run to the same 2.0 s repeat. A window costs
    26 ms to decode on `rf-corpus/7102k_065457.wav` and 103 ms on the 2026-08-05
    KE8LVA session (median per 6.25 s window; a busy channel offers the acquisition
    walk far more to try), so 0.25 s steps — a frame reports within a step of
    ending, measured well inside it — spend 10-41% of one core. The 3-4 ms and 2%
    this used to claim reproduce on nothing in the corpus.

    Audio arrives on the sound card's thread and is only queued there; `pump` does
    the decoding on the caller's thread and returns the audio it took, so a caller
    that also records or levels the capture needs no second queue."""

    STEP_S = 0.25
    #: Carry-over between windows. Longer than the longest ARDOP frame
    #: (4FSK.2000.600, 5.52 s), so no frame can straddle a window edge and be lost
    #: the way a burst gate loses one straddling its open or close. It also puts the
    #: 48->12 kHz resampler's edge transient inside another window's interior.
    OVERLAP_S = 6.0
    #: Room past a failed frame's last sample before its verdict is believed. Two
    #: things live in it: the 48->12 kHz resampler's edge, where `resample_poly`'s
    #: Kaiser FIR runs 40 taps at the card rate against zero padding (0.83 ms), and
    #: the demodulator's symbol-timing searches, which read a little past
    #: `_body_span`. Measured by cutting a window at the card rate at frame-end + m
    #: and sweeping m: over every ARDOP frame class at 30/6/3 dB and three noise
    #: seeds, 20 ms is the most any of them needs to settle on the verdict it gets
    #: with a second to spare (8PSK.2000.100 at 3 dB wants 5 ms through the card
    #: edge; ConAck500 and 4FSK.500.100 want 20 at the native rate). 0.1 s is five
    #: times that and is also the sound card's own block, the grain the newest edge
    #: advances in, so the headroom costs no step.
    FRAME_MARGIN_S = 0.1
    #: Re-finding a frame in the next window's overlap reproduces its stream offset
    #: exactly (measured: identical under every buffer alignment tried), so this
    #: guard only has to absorb resampler rounding. Far shorter than any frame.
    #:
    #: It has to be matched per position rather than carried as a high-water mark.
    #: Acquisition does not find frames in stream order — a frame whose leader is
    #: weak surfaces windows after a later one — so a mark set by the later frame
    #: silently swallows the earlier one for good. Measured on the 2026-08-05
    #: KE8LVA session: 11 frames, five of them ConReq2000M, reported only once the
    #: mark stopped running ahead of them.
    #:
    #: The slot carries the verdict as well as the position, because the hold-back
    #: that keeps a cut frame out of it is only as good as the span it is handed.
    #: `frame_span` reads the type off the header, and a `header_only` frame's type
    #: is a guess at one rather than a sighting of one (`Demodulator._scan_header`)
    #: — guessed as a shorter class, the sighting is released while the real frame
    #: is still arriving and the complete decode two windows later finds its own
    #: position taken by the guess. So a decode displaces a failure at the same
    #: position and is reported; nothing displaces a decode.
    DEDUP_S = 0.05

    def __init__(self, demod, on_frame, *, resample: bool = True) -> None:
        self._demod = demod
        self._on_frame = on_frame
        self._resample = resample                # capture at 48 k (sound card) vs native 12 k
        self._rate = FS_RADIO if resample else SAMPLE_RATE
        self._q: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._buf = np.empty(0, dtype=np.float32)
        self._n = 0                              # capture samples taken
        self._decoded_to = 0
        self._seen: list[tuple[int, bool]] = []   # positions reported, and their verdict
        self._frames = 0

    @property
    def funnel(self) -> RxFunnel:
        """The acquisition funnel for this capture: the demodulator's counters, the
        frames that survived dedup, and the audio all of it ran on."""
        return replace(self._demod.funnel, capture_s=self._n / self._rate,
                       frames=self._frames)

    def push(self, block: np.ndarray) -> None:
        """The sound-card callback. Queue only — never decode on the audio thread."""
        with self._lock:
            self._q.append(np.asarray(block, dtype=np.float32))

    def pump(self) -> np.ndarray:
        with self._lock:
            blocks, self._q = self._q, []
        au = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)
        if au.size:
            self._buf = np.concatenate([self._buf, au])
            self._n += au.size
        if self._n - self._decoded_to >= self.STEP_S * self._rate:
            self._decode()
        return au

    def flush(self) -> None:
        """Decode the tail the step cadence never reached."""
        self._decode(final=True)

    def _decode(self, *, final: bool = False) -> None:
        if not self._buf.size:
            return
        at = self._span(self._n - self._buf.size)
        # The latest sample a frame may end on and still count as wholly arrived.
        horizon = self._span(self._n) - int(self.FRAME_MARGIN_S * SAMPLE_RATE)
        window = np.concatenate([self._at_ardop_rate(), _TAIL])
        guard = int(self.DEDUP_S * SAMPLE_RATE)
        for f in self._demod.decode(window, at=at):
            pos = at + f.offset
            slot = next((s for s in self._seen if abs(pos - s[0]) < guard), None)
            if slot is not None and (slot[1] or not f.ok):
                continue
            # A frame cut short by the newest window edge cannot validate, so a clean
            # decode is whole by definition and reports at once — a gateway answering
            # appears within a step. A failed one waits for its own last sample, which
            # the header already names (`frame_span`), so a frame reported failed is
            # one the channel failed and not one the edge cut. Release it unheld and
            # the truncated first sighting wins the dedup and suppresses the complete
            # frame: measured on rf-corpus/7102k_065457.wav, where 3 of the 4 ConReqs
            # are first seen with ~1.1 s of themselves still to come and arrive
            # ok=False carrying no callsign at all.
            #
            # The wait is the frame's own, not a flat OVERLAP_S, because a flat one is
            # the frame's length over again for nothing and the peer is not waiting.
            # Against WM4RB on 2026-08-14 the gateway ran a measured 6.42 s metronome
            # — a 4.41 s frame, then 2.01 s to answer in — and our DATANAKs keyed at
            # frame-end +2.25 to +2.70 s, four of four, every one landing after the
            # repeat had already begun. The cadence cannot say whether they were
            # heard: that 2.01 s runs off the gateway's own frame end and a NAK
            # neither starts it nor stops it (`ARQ.c:2143` and `2210-2212`; only an
            # ACK stops the repeat). What IS observed is that the gearshift never
            # fired and the whole data phase failed. Replayed off that capture
            # and the W6IDS one of the same night, the failed data frames report at
            # end +0.23 to +0.55 s where the flat wait left them at +1.83 to +2.11 s —
            # inside both gateways' windows rather than outside every one of them.
            # Two of WM4RB's four still arrive ~2.1 s late, and that is acquisition
            # finding them late, not this wait.
            #
            # The span is read off the type, and a `header_only` frame's type is a
            # guess — but not a guess about how long the frame is. No valid type
            # sits one header symbol from another with matching parity, and
            # `Demodulator._scan_header` reports nothing from its fallback loop
            # without leader behind the header, so a guessed type is a real
            # transmission's own header read on the grid its own leader set.
            # Holding every guess for the longest frame anyone can send instead
            # (4FSK.2000.600, 5.26 s) is the conservative-looking rule, and W6IDS's
            # cadence refuses it: 4.14 s between headers for a 2.04 s frame on
            # 2026-08-29, so fourteen of the nineteen header-only sightings of that
            # slot would have been answered from inside the gateway's next
            # transmission (`test_a_guessed_type_is_still_released_at_its_own_length`).
            if not f.ok and not final and pos + frame_span(f.type) > horizon:
                continue
            if slot is not None:
                self._seen.remove(slot)
            self._seen.append((pos, f.ok))
            self._frames += 1
            self._on_frame(pos, f)
        self._decoded_to = self._n
        self._seen = [s for s in self._seen if s[0] >= at]   # no window reaches back further
        # Keeps OVERLAP_S plus a step. The longest ARDOP frame runs 5.52 s and its
        # hold-back ends FRAME_MARGIN_S past that, so a frame released from the
        # hold-back still has lead-in ahead of it rather than starting at sample zero
        # of its window — acquisition triggers on a rise out of silence and has none
        # without it.
        self._buf = self._buf[-int((self.OVERLAP_S + self.STEP_S) * self._rate):]

    def _at_ardop_rate(self) -> np.ndarray:
        return from_card(self._buf, SAMPLE_RATE) if self._resample else self._buf.astype("<i2")

    def _span(self, n_card: int) -> int:
        """A capture index as a 12 kHz index — the card is the timebase."""
        return rates.to_native(n_card, SAMPLE_RATE) if self._resample else n_card


class RadioLink:
    """Bind a `BesraModem` to a real rig's sound card, in place of the virtual air.

    TX: the modem's rendered frame is played to the output device with PTT keyed
    around it (watchdog-guarded). RX: the capture stream feeds a `RollingDecoder`
    whose frames go straight to the modem's session — half-duplex, so audio captured
    while transmitting (our own signal) is dropped at the callback and never reaches
    the decoder.

    Decoding on our own pump thread rather than through the modem's audio queue is
    what keeps the ARQ turnaround inside `RollingDecoder.STEP_S`; the session's lock
    serialises us against the modem's own timer pump."""

    _PUMP_S = 0.05           # decoder wake-up, well inside RollingDecoder.STEP_S

    def __init__(self, modem, *, in_device=None, out_device=None, rig: Rig | None = None,
                 max_key_s: float = 30.0, record=None,
                 drive: float = levels.TX_DRIVE) -> None:
        self.modem = modem
        self.in_device = in_device
        self.out_device = out_device
        self.rig = rig
        self.max_key_s = max_key_s
        self.drive = drive
        self.muted = False                       # true while keyed: drop our own audio
        self.recorder = (CaptureRecorder(record, rate=FS_RADIO, source=str(in_device))
                         if record is not None else None)
        self._rec_fault: str | None = None
        self._rx = RollingDecoder(Demodulator(expect_session=modem.expected_session,
                                              rx_epoch=modem.rx_epoch),
                                  lambda pos, frame: modem.receive_frames([frame]))
        self._stream = None
        self._pump: threading.Thread | None = None
        self._running = False
        modem.audio_out = self._transmit

    def _transmit(self, samples: np.ndarray) -> None:
        # AFTER the rate conversion, because that is where the samples past the
        # rail come from: ardopcf's own soft clip leaves the 12 kHz arrays at
        # 0.9979 and the interpolation to 48 kHz overshoots it, with nothing
        # downstream limiting. Composed scale is no protection either -- it is
        # whatever the modulator happened to sum to, which is the one path here
        # that had no drive at all.
        audio = levels.at_drive(to_card(samples, SAMPLE_RATE), self.drive)
        self.muted = True
        stop = threading.Event()

        def watchdog():
            if not stop.wait(self.max_key_s):
                if self.rig:
                    self.rig.unkey(f"watchdog: keyed past {self.max_key_s}s")
                log.error("PTT watchdog fired at %ss — force unkey", self.max_key_s)
        threading.Thread(target=watchdog, daemon=True).start()
        try:
            # THE DEVICE IS DRIVEN BEFORE THE KEY: one that will not open must
            # refuse with the transmitter still down, and the first open of a
            # session is the expensive one. `core.audio.warm_output` carries the
            # on-air measurement behind both. The burst's own stream still opens
            # under the carrier, because `play_drained` opens AND CLOSES per
            # burst — a held-open output stream keeps a rig whose PTT follows data
            # audio transmitting between frames, with CAT reporting PTT down the
            # whole time. Observed on air.
            warm_output(self.out_device)
            # PTT is inside the try: a rigctl CAT timeout on key (the very failure
            # the module docstring warns about) would otherwise skip the finally
            # and leave the receiver muted and the watchdog leaked.
            if self.rig:
                if self.rig.ptt(True) is False:
                    # A key-up the rig refused -- retired, unconfirmed and
                    # already dropped to be sure, or a command that never
                    # left -- is a burst that cannot have gone out. Settle
                    # plus audio into the codec then is a transmission in
                    # the log and silence on the band, and the ARQ waits
                    # out an answer nobody was asked for. Play nothing;
                    # the session's own repeat ladder decides what is
                    # next, and the next key-up is decided afresh.
                    log.error("NOT TRANSMITTING -- the key-up was refused; "
                              "%.2f s of audio not played",
                              len(audio) / FS_RADIO)
                    return
                # The RIG's settle, from `core.rigs`, not a number of our own.
                # This was a hardcoded 0.35 — no rig's table figure; the X6100's is
                # 0.40 — on a station
                # that keys an FT-891 whose table entry is 0.04, and it went out
                # as dead carrier at the head of every burst. Measured off the
                # 2026-08-06 recordings, where each modem's own capture goes to
                # digital silence from key-down until its first sample: besra led
                # by 0.43 s (median, n=31) and kestrel, minutes later on the same
                # rig and band, by 0.085 s (median, n=19). `tools/ptt_tail_check`
                # instrumented the same rig that hour at lead 0.040 s — so 0.04 is
                # the figure, and the 0.39 s above it was ours. That instrument is
                # this call, key for key: it keys, plays through `play_drained` and
                # unkeys, with no settle of its own, so its 0.040 s is CAT plus a
                # cold stream open plus codec latency and is a ceiling on what the
                # open below adds here, where the device is already warm.
                time.sleep(self.rig.settle)
            # Returns on Pa_StopStream, once the device has played what was
            # written — so the unkey below cannot cut the tail.
            play_drained(audio, FS_RADIO, self.out_device)
        finally:
            # The verdict, read at the one moment False means a stuck
            # transmitter: through hamlib it is proof T 0 never left, and CAT
            # PTT latches at the rig, so the ladder — one-shot, drop_rts,
            # alarm, retire — is what stands between here and going back to
            # listening keyed. The line path has already alarmed and retired
            # itself inside `_key_line`.
            if self.rig and self.rig.ptt(False) is False and not self.rig.line_ptt:
                self.rig.unkey("the persistent pipe did not take the key-up")
            stop.set()
            self.muted = False

    def _capture(self, block: np.ndarray) -> None:
        # Recorded ahead of the mute, so the file keeps a continuous timebase across
        # our own transmissions; only the decoder is half-duplex.
        if self.recorder is not None:
            self.recorder.push(block)
        if not self.muted:
            self._rx.push(block)

    def _run(self) -> None:
        while self._running:
            time.sleep(self._PUMP_S)
            self._drain()
            try:
                self._rx.pump()
            except Exception:
                # One bad window must not end the pump: a dead receive thread is a
                # station that has gone deaf for the rest of the session.
                log.exception("receive pump")

    def _drain(self) -> None:
        """Written on the pump thread, and stopped at the first fault.

        A recording that resumes after a failed write is a shorter file that
        reads as a whole one — the loss the continuous stream exists to make
        visible — so the reason is kept for `stream_report` and the writing does
        not restart. The link outranks the recording, so this never raises into
        the pump.
        """
        if self.recorder is None or self._rec_fault is not None:
            return
        try:
            self.recorder.drain()
        except Exception as exc:                 # noqa: BLE001
            self._rec_fault = str(exc)
            log.error("session stream stopped writing -- %s", exc)

    def stream_report(self) -> str:
        """What this session left on the disk, named in full: the line an
        operator reads at three in the morning is the one they paste into the
        analysis.

        The sample clock is claimable here for the reason `_capture` records
        ahead of the mute — sample k of the file is sample k of what the card
        delivered, our own keyed intervals included, which is what
        `tools/rehear --ours` reads back out of it. What is NOT claimed is that
        the card delivered everything the band did: nothing here counts xruns,
        so there is no wholeness to assert and none is asserted.
        """
        if self.recorder is None:
            return "session stream: not taken -- this session ran with no recorder"
        if self._rec_fault is not None:
            return (f"session stream: {self.recorder.path} is INCOMPLETE after "
                    f"{self.recorder.seconds:.1f} s -- {self._rec_fault}. What is "
                    f"in it is still on the session's sample clock; what is "
                    f"missing is everything after that.")
        return (f"session stream: {self.recorder.path} -- "
                f"{self.recorder.seconds:.1f} s, "
                f"{self.recorder.path.stat().st_size / 1e6:.1f} MB, on the sample "
                f"clock every index this session logged is an offset into")

    def start(self) -> None:
        self._stream = capture_stream(self.in_device, self._capture)
        self._stream.start()
        self._running = True
        self._pump = threading.Thread(target=self._run, name="besra-rx", daemon=True)
        self._pump.start()

    def close(self) -> None:
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
            if self.rig:
                self.rig.stop()
        finally:
            # In the finally because this is the half of teardown that has to happen:
            # a session that ends by failing to unkey is exactly one worth listening
            # back to. The pump thread does the writing, so it is joined first —
            # closing the WAV under a drain in flight would race it.
            pump, self._pump = self._pump, None
            if pump is not None:
                pump.join(timeout=2.0)
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:         # noqa: BLE001
                    # `CaptureRecorder.close` finalises the WAV and the sidecar
                    # before it raises, and it raises for the owner to report —
                    # which is the line below, not a traceback over the top of it.
                    self._rec_fault = self._rec_fault or str(exc)
            # ardopcf's `LogStats`, said at the end of the listening rather than per
            # connection: this decoder never stops, and an arrival nobody answered
            # is most often outside a connection.
            funnel = self._rx.funnel
            log.info("%s", funnel)
            self.modem.status(str(funnel))
            # Last, because it is what the run leaves behind.
            print(self.stream_report(), flush=True)


def capture_stream(device, on_block):
    """A started-on-`__enter__` sounddevice input stream handing each mono float32
    block to `on_block`. `RadioLink` passes its half-duplex capture hook; the monitor
    passes its own recorder."""
    import sounddevice as sd
    return sd.InputStream(
        device=device, channels=1, samplerate=FS_RADIO, dtype="float32",
        blocksize=int(0.1 * FS_RADIO),
        # Copied because sounddevice reuses the callback buffer and the monitor
        # queues the block for another thread rather than consuming it here.
        callback=lambda indata, *a: on_block(indata[:, 0].copy()))
