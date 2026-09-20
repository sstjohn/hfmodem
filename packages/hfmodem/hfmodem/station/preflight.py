# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station facts a bench cannot settle, measured in about fifteen seconds.

`hfmodem station` opens a card, arms a rig and runs four protocols, and almost
everything it depends on is checked against something a bench can produce. These
are not, and each one shows up as a bad session rather than as an error:

  * whether this codec tolerates a persistently open duplex stream. A persistent
    output-only stream starved its input 16x on 2026-07-28 and was reverted; the
    argument that one stream owning the device cannot contend with itself is
    structural, and structural arguments are what this check is for.
  * whether the card runs at the blocksize it was asked for, rather than at its
    own 4096-frame default.
  * the ADC->DAC offset, which every transmit index is arithmetic on and which
    has only ever been seen on a loopback.
  * how long the rig takes to actuate after the line goes up. That is what
    `[rig.ptt] settle_s` exists to cover, and it was not measured at all until
    2026-07-28.
  * how long the rig stays deaf after unkeying, which is the earliest peer answer
    this station can hear.
  * which wire is about to be keyed, and by what method.

Keying happens with **no modulation**. On a data interface the audio is the
drive, so PTT with silence puts no power into the antenna: it exercises the T/R
path without transmitting. That is the only reason this may key without an ATU
match or a listen-first, and it is why `Rig.key()` is not the path taken here —
`key()` requires an `Emission` and consults the regulatory profile, and an
unmodulated key is not an emission. The arm gate's own PTT proof keys the line
directly for exactly the same reason. What the arm gate cannot give is the sample
index the line went up at, and that index *is* the actuation measurement, so the
line is held here rather than borrowed.

A line nothing could measure says so and is not a pass. A clean sheet printed for
a station with no rig attached is the one outcome this file exists to prevent.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from hfmodem.core import devices
from hfmodem.core.audio import AudioError, StationAudio, StreamLane
from hfmodem.core.config import StationConfig
from hfmodem.core.ptt import PttError, RtsPtt
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.rig import Cat, Rig, RigError, unkey_on_signal

#: How long each stream is listened to before its level is believed.
DWELL_S = 2.0

#: Keyings, and how long the line is held for each. Five because the figure that
#: matters is a median: one keying that lands in a noise dip is not a rig.
KEYINGS = 5
HOLD_S = 0.25

#: Quiet band read before each keying, to have a floor to compare against.
FLOOR_S = 0.3

#: How long the receiver is given to come back before recovery is called lost.
RECOVER_S = 0.6

#: Down to a tenth of the quiet-band level is unambiguous: the mute measured
#: -50 dB, and band noise does not fade 20 dB inside one block.
MUTE_FRACTION = 0.1

#: Below this there is no band noise to watch collapse, so nothing can be timed.
AUDIBLE = 1e-5

#: The starvation gate. The regression it exists to catch was 16x down, so half
#: is a wide gate that a starved input still cannot pass.
STARVED_FRACTION = 0.5

#: What this station's turnaround assumes the rig's T/R recovery is — shrike's
#: measured figure for the FT-891, with the slack its own gate used. It lives
#: here because the unified station has no cycle budget yet and this is the only
#: thing that checks the number; it belongs beside that budget when there is one.
TR_RECOVERY_S = 0.055
TR_RECOVERY_SLACK_S = 0.015

#: A crystal is out by tens of ppm. Three orders of magnitude past that is not a
#: crystal — it is a card running at a rate nobody asked for, and every sample
#: index in this station is arithmetic on the rate we think it has. A device
#: delivering 44100 where 48000 was requested reads about -81000 ppm.
#:
#: It is the second question and not the first. A large figure on this station has
#: never once been the crystal: the card reads +3.8 to +6.7 ppm live, and −867251
#: under eight pure-Python threads, where 87% of a 90 s capture never reached the
#: interpreter and `input_overflow` was false on every callback that ran. Every
#: reading past ±20 ppm in the whole record is negative, which a crystal has no
#: reason to be. So `StationAudio.lost` — capture the converter timestamped and
#: Python was never handed — is asked first, and answers directly what a rate fit
#: can only be read backwards into.
#:
#: This gate is only sound because `StationAudio.clock_ppm` anchors its fit on
#: the converter's own timestamps. Anchored on callback-entry time it inherited
#: the driver's delivery jitter — load-dependent, and observed as far as
#: -14660 ppm across real sessions — so a preflight run on a loaded machine
#: could refuse a healthy station.
MAX_PPM = 1000.0

#: Beyond its own hold, how long the line may stay up before the rig's panic path
#: takes it down. Long enough that a scheduler hiccup is not an incident.
DEADMAN_SLACK_S = 1.0

OK = "ok"
STOP = "STOP"
UNKNOWN = "cannot measure"

_NAME_W = 13
_VERDICT_W = 14


@dataclass(frozen=True, slots=True)
class Line:
    name: str
    verdict: str
    detail: str
    note: str = ""


def run(cfg: StationConfig) -> int:
    """Measure the station. 0 only when every line was measured and passed."""
    return Preflight(cfg).run()


class Preflight:
    """One pass over the station, printing as it goes.

    Printed per line rather than collected and reported, because the operator is
    watching a radio while this runs and the interesting failures are the ones
    where the next line never arrives.
    """

    def __init__(self, cfg: StationConfig) -> None:
        self.cfg = cfg
        self.lines: list[Line] = []
        self.tap: StreamLane | None = None
        # Held on the instance, not in `run`'s frame: `_rig` keys inside the arm
        # gate's proof, and a signal arriving there must find the rig that is up.
        self.rig: Rig | None = None

    # -- the run -----------------------------------------------------------

    def run(self) -> int:
        print("== preflight ==", flush=True)
        audio = None
        with unkey_on_signal(lambda: self.rig):
            try:
                audio = self._card(self._reference())
                self._keying(audio, self._rig())
                # Last because it needs a baseline: the fit wants seconds of stream,
                # and by here the keying loop has supplied a dozen of them.
                self._clock(audio)
            finally:
                if self.rig is not None:
                    self.rig.close()
                if audio is not None:
                    audio.close()
        return self.verdict()

    def say(self, name: str, verdict: str, detail: str, note: str = "") -> None:
        self.lines.append(Line(name, verdict, detail, note))
        print(f"  {name:<{_NAME_W}}: {verdict:<{_VERDICT_W}} {detail}", flush=True)
        if note:
            print(f"  {'':<{_NAME_W}}  !! {note}", flush=True)

    def revise(self, name: str, verdict: str, detail: str) -> None:
        """Replace a line that a later measurement settled, and print it again.

        The operator watched the first verdict go by, so amending one silently
        would leave the report they read contradicting the exit code.
        """
        self.lines = [ln for ln in self.lines if ln.name != name]
        self.say(name, verdict, detail)

    # -- the card ----------------------------------------------------------

    def _reference(self) -> float | None:
        """Capture-only rms, taken before any duplex stream exists.

        Opened and closed here rather than kept: two PortAudio streams on one
        CoreAudio device is the incident this comparison exists to detect, so a
        reference overlapping the duplex stream would be measuring it.
        """
        cfg = self.cfg.audio
        try:
            seg, overflows = _listen(_device(cfg.input, "in"),
                                     cfg.blocksize, cfg.latency)
        except Exception as exc:      # noqa: BLE001
            self.say("capture-only", UNKNOWN, f"the input device did not open: {exc}")
            return None

        rms = _rms(seg)
        detail = (f"rms {rms:.5f} over {len(seg) / CARD_RATE_HZ:.1f} s on "
                  f"{cfg.input or '(default)'}, {overflows} overflow(s)")
        if overflows:
            self.say("capture-only", STOP, detail,
                     "the input dropped audio with nothing else running, so the "
                     "duplex figures below are not comparable to anything")
            return None
        if rms <= AUDIBLE:
            self.say("capture-only", UNKNOWN, detail,
                     "the reference is silent, so the starvation gate cannot run. "
                     "Re-run with the rig on and a live band before trusting this.")
            return None
        self.say("capture-only", OK, detail)
        return rms

    def _card(self, reference: float | None) -> StationAudio | None:
        """The station's own stream, held open, and what it reports about itself."""
        cfg = self.cfg.audio
        try:
            audio = StationAudio(
                input_device=_device(cfg.input, "in"),
                output_device=_device(cfg.output, "out"),
                gain=cfg.input_gain, blocksize=cfg.blocksize)
            audio.open()
        except Exception as exc:      # noqa: BLE001
            for name in ("duplex", "xruns", "ADC->DAC"):
                self.say(name, UNKNOWN, f"the card did not open: {exc}")
            return None

        self.tap = audio.subscribe(StreamLane(CARD_RATE_HZ))
        time.sleep(DWELL_S)
        seg, _ = self._read()
        self._duplex(seg, reference)
        self._xruns(audio)
        self._offset(audio)
        return audio

    def _duplex(self, seg: np.ndarray, reference: float | None) -> None:
        rms = _rms(seg)
        detail = f"rms {rms:.5f} over {len(seg) / CARD_RATE_HZ:.1f} s"
        if reference is None:
            self.say("duplex", UNKNOWN, f"{detail}, with nothing to compare it to")
        elif rms < STARVED_FRACTION * reference:
            self.say("duplex", STOP,
                     f"{detail}, {reference / max(rms, 1e-9):.1f}x down on the reference",
                     "INPUT STARVED: holding the duplex stream open is costing this "
                     "codec its input. Do not transmit.")
        else:
            self.say("duplex", OK, f"{detail}, {rms / reference:.2f}x the reference")

    def _xruns(self, audio: StationAudio) -> None:
        detail = (f"{audio.underruns} underrun(s), {audio.overflows} overflow(s) "
                  f"at block {audio.blocksize}")
        if audio.underruns or audio.overflows:
            self.say("xruns", STOP, detail,
                     "the stream is dropping audio at this blocksize. A dropped "
                     "block moves every later sample index off the air by an "
                     "unknown amount, and the grid is made of sample indices.")
        else:
            self.say("xruns", OK, detail)

    def _offset(self, audio: StationAudio) -> None:
        """The ADC->DAC offset the card actually has.

        `StationAudio.open()` medians this over its settling callbacks and
        asserts the spread, so reaching here at all means it is a constant on
        this hardware. What is left to check is that there is one and that the
        DAC is ahead of the ADC, which is the direction a duplex device has.
        """
        lat = audio.lat
        if lat is None:
            self.say("ADC->DAC", STOP, "None",
                     "the offset never settled, so every transmit index is an "
                     "estimate rather than arithmetic")
        elif lat <= 0:
            self.say("ADC->DAC", STOP, f"{lat} samples",
                     "the DAC is not ahead of the ADC, which no duplex device "
                     "does — the timestamps this card reports cannot be used")
        else:
            self.say("ADC->DAC", OK,
                     f"{lat} samples ({lat / CARD_RATE_HZ * 1e3:.1f} ms), "
                     "median over the settling callbacks with the spread asserted")

    def _clock(self, audio: StationAudio | None) -> None:
        if audio is None:
            self.say("card clock", UNKNOWN, "the card did not open")
            return
        ppm, sigma = audio.clock_ppm()
        if audio.lost >= audio.blocksize:
            share = audio.lost / (audio.samples + audio.lost) * 100
            self.say("card clock", STOP,
                     f"{audio.lost} samples "
                     f"({audio.lost / CARD_RATE_HZ * 1e3:.0f} ms, {share:.1f}% of "
                     "what was captured) never reached this program",
                     "the converter timestamped them and the interpreter was too "
                     "busy to take them, which nothing else reports — the stream "
                     "is spliced, and every index this station computes assumes "
                     "it is not. Give the machine less to do and run this again.")
        elif not np.isfinite(ppm):
            self.say("card clock", UNKNOWN,
                     "too little of the stream to fit a rate against the system clock")
        elif abs(ppm) > MAX_PPM:
            self.say("card clock", STOP, f"{ppm:+.0f} ppm",
                     "no lost capture was detected, so this is the rate itself: "
                     f"the card is not running at {CARD_RATE_HZ} Hz, and every "
                     "sample index in this station assumes it is")
        else:
            self.say("card clock", OK, f"{ppm:+.1f} ppm (1 sigma {sigma:.1f})")

    # -- the radio ---------------------------------------------------------

    def _rig(self) -> Rig | None:
        """Arm the rig, and say which wire that puts under this program's control.

        `prove_ptt=True`: the proof is what distinguishes a keying line that
        reaches the radio from one that moves a pin on a cable nobody plugged in,
        and it is 200 ms of unmodulated key like everything else here.
        """
        cfg = self.cfg
        wire = (f"{cfg.rig.ptt.backend.upper()} on {cfg.rig.ptt.port}, "
                f"CAT via rigctld at {cfg.rig.host}:{cfg.rig.port}")
        rig = None
        try:
            # The constructors are inside the block on purpose. All three are
            # lazy today, but nothing states or tests that, and a hardware fault
            # surfacing at construction must be a line that could not be
            # measured, not the end of the run.
            rig = self.rig = Rig(
                model=cfg.rig.model, cat=Cat(cfg.rig.host, cfg.rig.port),
                ptt=RtsPtt(cfg.rig.ptt.port), profile=cfg.profile,
                control=cfg.control, mycall=cfg.station.mycall,
                transmit=cfg.station.transmit, max_key_s=cfg.rig.max_key_s)
            rig.ptt.open()
            report = rig.arm(prove_ptt=True)
        except Exception as exc:      # noqa: BLE001
            if rig is not None:
                rig.close()
            # After the close and not before: until then this is still the thing
            # that would bring a line down, and after it a Ctrl-C reaching it
            # spends the whole CAT ladder to print UNKEY NOT CONFIRMED about a
            # radio that never keyed.
            self.rig = None
            self.say("keying", UNKNOWN, f"{wire} — {exc}")
            return None

        detail = (f"{wire}; {report.model} on dial "
                  f"{report.dial_hz / 1e6:.6f} MHz, "
                  + ("PTT proven" if report.ptt_proven else "PTT UNPROVEN")
                  + (", line ownership verified" if report.ptt_owner_checked
                     else ", ownership unverified"))
        note = " ".join(report.notes)
        if report.ptt_proven:
            self.say("keying", OK, detail, note)
        elif not rig.cat.trusted or not rig.cat.ptt_readable:
            # Two ways for the readback to be worth nothing. The daemon serves `t`
            # from its own cache, so "the radio says down" is its memory of our own
            # last command; or it has no PTT of its own to report, which is the
            # configuration this station requires and the one it runs. Neither is
            # evidence either way, and calling either a failure would be as wrong
            # as calling it a pass. The keying measurement below is what can tell.
            self.say("keying", UNKNOWN, detail, note)
        else:
            self.say("keying", STOP, detail,
                     note or "the line is up and the radio does not report "
                             "transmitting — keying is not under this program's "
                             "control")
        return rig

    def _keying(self, audio: StationAudio | None, rig: Rig | None) -> None:
        """Key five times, unmodulated, and time the receiver either side of it."""
        if audio is None or rig is None:
            why = ("the card did not open" if audio is None
                   else "the rig is not usable")
            for name in ("PTT actuation", "T/R recovery"):
                self.say(name, UNKNOWN, f"nothing was keyed: {why}")
            return

        actuation: list[float] = []
        recovery: list[float] = []
        try:
            for _ in range(KEYINGS):
                # A dead stream and a quiet band are the same picture from here,
                # and one of them makes every number below a fiction.
                audio.alive()
                a, r = self._one_keying(audio, rig)
                actuation.append(a)
                recovery.append(r)
                if rig.retired:
                    raise RigError(f"the rig retired: {rig.retired_why}")
        except (AudioError, RigError, PttError) as exc:
            for name in ("PTT actuation", "T/R recovery"):
                self.say(name, STOP, f"after {len(recovery)} of {KEYINGS} keyings",
                         str(exc))
            return

        self._actuation(actuation)
        self._recovery(recovery)
        self._settle_keying(actuation)

    def _settle_keying(self, actuation: list[float]) -> None:
        """Let the actuation figure settle the keying line CAT could not.

        CAT readback is worth nothing on the daemon this station requires, which
        has no PTT of its own to report, so `keying` goes by unmeasured before
        anything is timed. Our own receiver cannot mute unless the line reached
        the radio, so a keying the mute was timed against is the evidence the
        readback could not give. With nothing measurable the line stays unmeasured
        — a station that cannot prove it keys is not one that can.
        """
        keyed = [v for v in actuation if not np.isnan(v)]
        line = next((ln for ln in self.lines if ln.name == "keying"), None)
        if line is None or line.verdict != UNKNOWN or not keyed:
            return
        self.revise("keying", OK,
                    f"{line.detail}; the receiver muted on {len(keyed)} of "
                    f"{len(actuation)} keyings")

    def _one_keying(self, audio: StationAudio, rig: Rig) -> tuple[float, float]:
        """One key-down, and (actuation, recovery) in ms — nan for either if the
        band is too quiet to see the receiver mute.

        The line going up is timed from the capture, not from the call: asserting
        RTS is one ioctl and returns in microseconds, so timing the call measures
        the ioctl. Keying mutes our own receiver, so the sample at which the band
        noise collapses is the sample the rig went to transmit.
        """
        self._read()                    # drop what is buffered; the floor is next
        time.sleep(FLOOR_S)
        base, _ = self._read()
        floor = float(np.median(np.abs(base))) if len(base) else 0.0

        keyed_at = audio.sample_now()
        self._key(rig)
        unkeyed_at = audio.sample_now()
        time.sleep(RECOVER_S)
        # One read, starting back at the floor read: the whole keyed interval is
        # still in front of it, and both edges are indexed into the same buffer.
        seg, at = self._read()
        return _edges(seg, at, keyed_at, unkeyed_at, floor)

    def _key(self, rig: Rig) -> None:
        """Hold the line up for `HOLD_S` with nothing modulating it.

        The deadman is the rig's own panic path, armed before the line goes up
        and cancelled only once the line is confirmed down — the ordering
        `Rig.unkey` is careful about, for the same reason: a window where the
        line is up and no timer is scheduled has nothing left to bring it down.
        """
        deadman = threading.Timer(
            HOLD_S + DEADMAN_SLACK_S, rig.panic_unkey,
            args=(f"preflight held the key past {HOLD_S + DEADMAN_SLACK_S:.2f} s",))
        deadman.daemon = True
        deadman.start()
        try:
            rig.ptt.assert_(True)
            time.sleep(HOLD_S)
        finally:
            try:
                rig.ptt.assert_(False)
                if rig.ptt.sense() is True:
                    rig.panic_unkey("the keying line reads high after the preflight "
                                    "deassert")
            except PttError as exc:
                rig.panic_unkey(f"the keying line refused to deassert: {exc}")
            finally:
                deadman.cancel()

    def _actuation(self, values: list[float]) -> None:
        settle_ms = self.cfg.rig.ptt.settle_s * 1e3
        median = _median_ms(values)
        if np.isnan(median):
            self.say("PTT actuation", UNKNOWN,
                     f"{len(values)} keyings, none of them measurable",
                     "the band is too quiet to see the receiver mute, so the one "
                     "number settle_s exists to cover is unknown")
        elif median > settle_ms:
            self.say("PTT actuation", STOP,
                     f"{median:.0f} ms median of {len(values)} "
                     f"(settle_s is {settle_ms:.0f} ms)",
                     "the rig keys after the audio starts, so the head of every "
                     "burst is being cut. Raise [rig.ptt] settle_s to at least "
                     f"{np.ceil(median / 10) * 10 / 1e3:.2f}.")
        else:
            self.say("PTT actuation", OK,
                     f"{median:.0f} ms median of {len(values)} "
                     f"(settle_s is {settle_ms:.0f} ms)")

    def _recovery(self, values: list[float]) -> None:
        budget_ms = TR_RECOVERY_S * 1e3
        median = _median_ms(values)
        if np.isnan(median):
            self.say("T/R recovery", UNKNOWN,
                     f"{len(values)} keyings, none of them measurable",
                     "the band is too quiet to see the receiver come back, so the "
                     "earliest answer this station could hear is unknown")
        elif median > budget_ms + TR_RECOVERY_SLACK_S * 1e3:
            self.say("T/R recovery", STOP,
                     f"{median:.0f} ms median of {len(values)} "
                     f"(the turnaround assumes {budget_ms:.0f} ms)",
                     "this rig is deaf longer than the turnaround assumes, so a "
                     "peer answering promptly answers into a receiver that is not "
                     f"back yet. Raise TR_RECOVERY_S to {np.ceil(median / 5) * 5 / 1e3:.3f}.")
        else:
            self.say("T/R recovery", OK,
                     f"{median:.0f} ms median of {len(values)} "
                     f"(the turnaround assumes {budget_ms:.0f} ms)")

    # -- the answer --------------------------------------------------------

    def verdict(self) -> int:
        stopped = [ln.name for ln in self.lines if ln.verdict == STOP]
        blind = [ln.name for ln in self.lines if ln.verdict == UNKNOWN]
        if stopped:
            print("== preflight FAILED ==", flush=True)
            print(f"   stopped on: {', '.join(stopped)}", flush=True)
            return 1
        if blind:
            print("== preflight INCOMPLETE ==", flush=True)
            print(f"   not measured: {', '.join(blind)}. A line nothing could "
                  "measure is not a line that passed.", flush=True)
            return 1
        print("== preflight OK ==", flush=True)
        return 0

    # -- the tap -----------------------------------------------------------

    def _read(self) -> tuple[np.ndarray, int]:
        """What the card has delivered since the last read, and where it starts.

        A `StreamLane` rather than a rolling one: every sample once, in order,
        with a card index attached — which is what both edges are measured
        against, and what a window-based reader cannot offer.
        """
        got = self.tap.poll() if self.tap is not None else []
        return (got[0][1], got[0][0]) if got else (np.zeros(0, np.float32), 0)


def _device(name, kind: str):
    """The card this station was told to use, or a failed measurement.

    `find_device` refuses with `SystemExit`, which is right for a CLI and wrong
    here: the guard this whole run sits inside raises `SystemExit` too, for a
    signal. A clause wide enough to report the first as a line that could not be
    measured swallowed the second, and preflight went on to key five more times.
    """
    try:
        return devices.find_device(name, kind, required=True)
    except SystemExit as exc:
        raise AudioError(str(exc)) from exc


def _listen(device, blocksize: int, latency: str) -> tuple[np.ndarray, int]:
    """`DWELL_S` of capture-only audio, and the overflows it took to get it."""
    import sounddevice as sd

    got: list[np.ndarray] = []
    overflows = 0

    def callback(indata, frames, t, status) -> None:
        nonlocal overflows
        if status and getattr(status, "input_overflow", False):
            overflows += 1
        got.append(np.array(indata[:, 0], np.float32))

    with sd.InputStream(samplerate=CARD_RATE_HZ, blocksize=blocksize,
                        dtype="float32", channels=1, latency=latency,
                        device=device, callback=callback):
        time.sleep(DWELL_S)
    return (np.concatenate(got) if got else np.zeros(0, np.float32)), overflows


def _edges(seg: np.ndarray, at: int, keyed_at: float, unkeyed_at: float,
           floor: float) -> tuple[float, float]:
    """(actuation, recovery) in ms out of one keying's capture.

    Both are nan when the band is too quiet to see the receiver mute, which is a
    reading rather than a zero: a station that cannot see its own transmitter
    knows nothing about how long it takes to key.
    """
    if floor <= AUDIBLE or not len(seg):
        return float("nan"), float("nan")
    env = np.abs(seg)
    down = env[max(0, int(keyed_at - at)):] <= MUTE_FRACTION * floor
    # Recovery runs from the unkey to where the level climbs back to the
    # quiet-band median it had before we keyed. Below that the receiver is still
    # muted, and a peer answering inside that window is one we cannot hear.
    back = env[max(0, int(unkeyed_at - at)):] >= floor
    return _first_ms(down), _first_ms(back)


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def _first_ms(mask: np.ndarray) -> float:
    """Where a condition first holds, in ms from the start of the window."""
    hit = np.nonzero(mask)[0]
    return float(hit[0]) / CARD_RATE_HZ * 1e3 if hit.size else float("nan")


def _median_ms(values: list[float]) -> float:
    """The median of what could be measured, or nan if nothing could be."""
    if not values or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmedian(values))
