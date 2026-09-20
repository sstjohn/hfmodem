# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Render or transmit one bounded Sabir beacon or prepared DATAGRAM proof.

Without --transmit, only local rendering and description occur. Transmission
uses explicit audio devices, CAT readback, channel sensing, line PTT, a keyed-time
watchdog and station regulatory checks. Every burst includes Morse identification.
A DATAGRAM proof is one-way reception evidence and does not establish live ARQ.
"""
from __future__ import annotations

import argparse
import atexit
import signal
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hfmodem.core import band, config, cwid, regulatory
from hfmodem.core.busy import FULL_BAND, is_busy, level_range, receiver_fault
from hfmodem.core.devices import find_device, list_devices
from hfmodem.core.ptt import PttError, arming_refusal
from hfmodem.core.rigs import RIGS
from hfmodem.sabir import offair, radio
from hfmodem.sabir.floor import beacon as wspr
from hfmodem.sabir.floor.mfsk import BASE_HZ, GRID
from hfmodem.sabir.phy.rate import FS

#: Between the beacon and the identification. Long enough that a decoder reading
#: the beacon is not still hearing Morse, short enough to stay one transmission
#: -- an identification in a separate key-up is a second emission to account for.
_ID_GAP_S = 0.5

#: The identification's own keyed span, from `core.cwid`: a 20 WPM envelope with
#: a 5 ms raised-cosine edge is tens of hertz wide and 150 errs generously.
_ID_SPAN_HZ = (1500.0 - cwid.BANDWIDTH_HZ / 2, 1500.0 + cwid.BANDWIDTH_HZ / 2)

#: Default listen window. The pactor verb's 8 s, which is twice `MIN_LIVE_S` and
#: the window `core.busy` set `shape` and `tone` — the two that decide — against.
LISTEN_S = 8.0

#: `--transmit` says a licensed operator is on frequency, and this path makes one
#: bounded transmission and exits -- it answers nothing and waits for nothing, so
#: there is no moment at which it is the station deciding on its own. §97.221 and
#: its equivalents are about the other case.
CONTROL = regulatory.Control.LOCAL


# -- what would go on the air ------------------------------------------------

@dataclass(frozen=True, slots=True)
class Burst:
    """A rendered transmission and the physical facts about it."""

    audio: np.ndarray
    waveform: str
    audio_lo_hz: float
    audio_hi_hz: float

    @property
    def seconds(self) -> float:
        return self.audio.size / FS

    @property
    def span_hz(self) -> tuple[float, float]:
        """The audio passband, identification included. The Morse goes out
        inside the same key-up and at 1500 Hz is the wider signal of the two, so
        a span quoted off the beacon alone understates what is emitted."""
        return (min(self.audio_lo_hz, _ID_SPAN_HZ[0]),
                max(self.audio_hi_hz, _ID_SPAN_HZ[1]))

    def emission(self, dial_hz: int, power_w: float | None = None) -> regulatory.Emission:
        """What the regulatory gate is asked about -- the same span `describe`
        shows the operator, so what is checked and what is printed cannot drift
        apart."""
        lo, hi = self.span_hz
        return regulatory.Emission(dial_hz, lo, hi, power_w=power_w,
                                   technique=self.waveform)

    def describe(self, dial_hz: int | None) -> str:
        lo, hi = self.span_hz
        where = ""
        if dial_hz is not None:
            where = (f"\n  on the air  {(dial_hz + lo) / 1e6:.6f} - "
                     f"{(dial_hz + hi) / 1e6:.6f} MHz")
        return (f"  waveform    {self.waveform}\n"
                f"  audio       {lo:.0f} - {hi:.0f} Hz "
                f"({hi - lo:.0f} Hz occupied, identification included)\n"
                f"  duration    {self.seconds:.1f} s keyed{where}")


def _identified(payload: np.ndarray, mycall: str) -> np.ndarray:
    """The burst with the station's callsign in Morse behind it.

    Each half is peak-normalised before it is joined, so the identification goes
    out at the drive the waveform does rather than at whatever amplitude
    `cwid.audio` happens to render at.
    """
    def unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x / (float(np.abs(x).max()) or 1.0)

    return np.concatenate([unit(payload), np.zeros(int(_ID_GAP_S * FS)),
                           unit(cwid.audio(mycall.upper()))])


def presence_burst(mycall: str, profile: int = 1) -> Burst:
    """The M9 presence beacon on the floor MFSK waveform.

    Seven tones on a `GRID` lattice from `BASE_HZ`; the span charges a symbol
    rate either side of the outermost, which is where the main lobes end.
    """
    audio = _identified(offair.render_presence(mycall, profile=profile), mycall)
    return Burst(audio, "M9 presence beacon, floor MFSK",
                 BASE_HZ - GRID, BASE_HZ + 6 * GRID + GRID)


def wspr_burst(mycall: str, grid: str, gear: str) -> Burst:
    """The M6b narrow-tone beacon: four tones a few hertz apart, self-timed.

    Tone unit 3 sits at `wspr.F_CENTER`, so the span is three spacings either
    side of it plus a spacing of skirt.
    """
    spacing = FS / wspr.BEACON_GEARS[gear].sym
    audio = _identified(offair.render_wspr(mycall, grid, gear), mycall)
    return Burst(audio, f"M6b weak-signal beacon, gear {gear}",
                 wspr.F_CENTER - 4 * spacing, wspr.F_CENTER + 4 * spacing)


# -- listening before transmitting -------------------------------------------

@dataclass(frozen=True, slots=True)
class Sense:
    """What the receiver heard, and whether the operator may overrule it."""

    verdict: str            # "clear" | "busy" | "deaf" | "intermittent"
    margin_db: float

    @property
    def overridable(self) -> bool:
        """Only a judgement about the band is. A receiver `core.busy` will not
        trust has judged nothing, so there is nothing for `--force` to weigh --
        and `--force` past it once keyed over an occupant nobody here could have
        heard."""
        return self.verdict == "busy"


def listen(seconds: float, device, name="", log=print, *,
           band=FULL_BAND) -> Sense:
    """Record and score one window over `band`. Biased toward busy, as
    `core.busy` is.

    `name` is the input as the operator named it, for the fault sentence -- the
    device this ends up recording from is an index, which tells him nothing.
    """
    import sounddevice as sd
    audio = sd.rec(int(seconds * FS), samplerate=int(FS), channels=1,
                   device=device, blocking=True).ravel().astype(float)
    busy, _db, margin = is_busy(audio, int(FS), band)
    fault = receiver_fault(*level_range(audio, int(FS)), seconds=seconds,
                           device=name or device)
    if fault is not None:
        log(f"  {fault.reason}")
        return Sense(fault.kind, margin)
    log(f"  channel sense: {margin:+.1f} dB -> "
        f"{'OCCUPIED' if busy else 'clear'}")
    return Sense("busy" if busy else "clear", margin)


# -- the run -----------------------------------------------------------------

def _install_signals(rig: radio.Rig) -> None:
    """Retire, then unkey, then leave non-zero.

    atexit does not run on SIGTERM or SIGHUP and a `finally` is never reached by
    either, which is exactly how an unattended run gets stopped: `timeout(1)`, a
    supervisor, a dropped terminal. Retiring first closes the window in which a
    racing thread puts the line back up before the interpreter exits.
    """
    def down(signum, _frame):
        rig.retire()
        rig.unkey(signal.Signals(signum).name)
        rig.hand_back()
        raise SystemExit(1)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, down)
    atexit.register(rig.hand_back)


def _burst(args) -> Burst:
    if getattr(args, "proof_manifest", None):
        if args.tone:
            raise ValueError("choose a proof manifest or a tone")
        from hfmodem.sabir.proof import load_burst
        burst, _ = load_burst(args.proof_manifest, callsign=args.mycall)
        return burst
    if args.tone:
        # Identified like any other emission. A bare tune-up tone would have to
        # be identified afterwards anyway, so carrying the Morse behind it is
        # the least RF that answers the question.
        return Burst(_identified(radio.tone(args.tone), args.mycall),
                     "steady 1500 Hz tone, drive check", 1400.0, 1600.0)
    if args.beacon == "presence":
        return presence_burst(args.mycall, profile=args.profile)
    return wspr_burst(args.mycall, args.grid, args.gear)


def _rules(args) -> regulatory.Profile:
    """The regime this station operates under, in the station file's vocabulary.

    Built here rather than defaulted anywhere: `core.regulatory` ships no default
    profile on purpose, and a beacon is exactly the emission an operator is least
    likely to be watching when it lands outside a segment.
    """
    settings = ({"licence": args.licence} if args.regulatory == "part97"
                else {"because": args.because})
    return regulatory.profile(args.regulatory, **settings)


def _refusals(args) -> str | None:
    """Everything that must be true before a radio is opened, in the order a
    refusal is cheapest. Returns the operator's sentence, or None to proceed."""
    if refusal := arming_refusal(args.ptt_device):
        return refusal
    if not args.line_ptt:
        return ("--transmit requires --line-ptt: this station's rigctld stopped "
                "answering on the first key-down of a session under RF, so the "
                "daemon may not hold the key. Start it ptt_type=None and let it "
                "tune")
    if args.channel is None and args.dial is None:
        return (f"NOT KEYING: --radio {args.radio} with no --channel or --dial "
                f"-- the dial would be whatever the rig was last left on, and "
                f"nothing here would know")
    if not (args.serial or args.rigctld):
        return ("NOT KEYING: --transmit needs --rigctld or --serial. A dial "
                "nothing can read back is not a verified dial")
    if not args.regulatory:
        return ("NOT KEYING: --transmit needs --regulatory. 'part97' for the US "
                "amateur service (with --licence), or 'unregulated' if you "
                "operate under rules hfmodem does not model (with --because), "
                "in which case you are answerable for every emission. There is "
                "no default: a station that has not said which rules it "
                "operates under has not said enough")
    if args.regulatory == "part97" and not args.licence:
        return ("NOT KEYING: --regulatory part97 needs --licence -- which HF "
                "segments carry data depends on the class, and guessing it in "
                "the permissive direction is the failure that matters")
    return None


def run(args) -> int:
    # Resolved here rather than in `main`, because this is where the transmitter
    # is built and `main` is not its only caller.
    args.gain = config.tx_drive(args.gain)
    if args.transmit and (refusal := _refusals(args)):
        print(refusal)
        return 2
    burst = _burst(args)
    dial = (band.dial_hz(args.channel) if args.channel is not None
            else args.dial)
    print(f"sabir on the air, or not:\n{burst.describe(dial)}")
    if args.wav:
        offair.wav_write(args.wav, burst.audio)
        print(f"  rendered    {args.wav}")

    if not args.transmit:
        print("NOT ARMED -- pass --transmit to key the transmitter")
        return 0
    try:
        rules = _rules(args)
    except ValueError as exc:
        print(f"NOT KEYING: {exc}")
        return 2

    # `required=True` because this run transmits and listens. An unset device is
    # the system default, which is the laptop's own speakers and microphone --
    # a beacon into the room, and a channel sense that judged the room.
    tx_device = find_device(args.audio_out, "out", required=True)
    rx_device = find_device(args.audio_in, "in", required=True)

    rig = radio.Rig(args.radio, args.ptt_device, profile=rules, control=CONTROL,
                    serial=args.serial, rigctld=args.rigctld)
    try:
        model = rig.identify()
        # One transceiver, several agents, and no station-wide notion of who
        # holds PTT. Only a `True` is an answer -- a daemon with no PTT of its
        # own answers ENAVAIL, which is this station's ordinary case and is what
        # the receiver-intermittent leg of the channel sense covers instead.
        if rig.transmitting() is True:
            print("the radio is ALREADY transmitting -- something else holds "
                  "PTT; not keying")
            return 2
        got = rig.qsy(dial)
    except radio.CatError as exc:
        # Before the line is opened and before anything is armed, so a station
        # with no hamlib finds out here rather than at a key-down.
        print(exc)
        return 2
    print(f"rig: {args.radio}, hamlib reports {model!r}"
          + (" (the daemon owns the radio, so this verifies nothing -- the dial "
             "readback decides)" if model == "?" else ""))

    if got is None or abs(got - dial) > band.QSY_TOLERANCE_HZ:
        print(f"QSY FAILED: asked {dial} Hz, rig reports "
              f"{got if got is not None else 'nothing readable'} -- not keying")
        return 2
    rig.set_mode()
    print(f"dial {got / 1e6:.6f} MHz verified, mode {rig.mode}, "
          f"TX drive {args.gain:.2f} peak")

    heard = listen(args.listen, rx_device, args.audio_in, band=burst.span_hz)
    if heard.verdict != "clear":
        if not heard.overridable:
            return 2
        if not args.force:
            print("  channel in use -- NOT transmitting (pass --force to "
                  "override, after listening yourself)")
            return 1
        print("  --force: transmitting over an occupant")

    try:
        rig.arm()
    except PttError as exc:
        # The line stat'd as a character device and will not drive modem lines.
        # Nothing is open and nothing has been keyed.
        print(f"NOT KEYING: {exc}")
        return 2
    _install_signals(rig)
    try:
        sent = radio.transmit(rig, burst.audio, burst.emission(got, args.power),
                              device=tx_device, gain=args.gain,
                              max_key_s=args.max_key, note=burst.waveform)
    except regulatory.NotPermitted as exc:
        print(f"NOT KEYING: refused by {rules.name}: {exc}")
        return 2
    except ValueError as exc:
        print(f"NOT KEYING: {exc}")
        return 2
    finally:
        rig.hand_back()
    if not sent:
        print("nothing transmitted -- the line would not confirm the key-up")
        return 1
    decode = (f"python -m hfmodem.sabir.proof verify {args.proof_manifest} CAPTURE.wav"
              if getattr(args, "proof_manifest", None)
              else "python -m hfmodem.sabir.offair decode CAPTURE.wav")
    print(f"transmitted {sent:.1f} s on {band.centre_hz(got) / 1e6:.6f} MHz "
          f"centre. Decode the far end's recording with: {decode}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m hfmodem.sabir.onair",
        description="render or transmit one identified Sabir burst")
    ap.add_argument("--proof-manifest", type=Path,
                    help="prepared current-protocol DATAGRAM proof; validate its source and waveform hashes")
    ap.add_argument("--beacon", choices=("presence", "wspr"), default="presence",
                    help="presence: the M9 block on the floor waveform, ~4.5 s, "
                         "decodes at the floor's ~-15 dB. wspr: the M6b "
                         "narrow-tone beacon, 17-132 s by gear, decodes far "
                         "below the noise. Identification is not optional, so "
                         "keyed time is roughly double the block")
    ap.add_argument("--mycall", help="this station's callsign, sent in Morse")
    ap.add_argument("--grid", default="", help="4-char locator (--beacon wspr)")
    ap.add_argument("--gear", choices=tuple(wspr.BEACON_GEARS), default="beacon_deep")
    ap.add_argument("--profile", type=int, default=1,
                    help="advertised profile in the presence beacon")
    ap.add_argument("--tone", type=float, metavar="SECONDS",
                    help="a steady 1500 Hz tone instead of a beacon, to confirm "
                         "audio reaches the rig and to set drive against ALC")

    rg = ap.add_argument_group("the radio")
    rg.add_argument("--radio", choices=tuple(RIGS), default="ft891")
    rg.add_argument("--serial", help="rig CAT serial device, when no daemon holds it")
    rg.add_argument("--rigctld", metavar="HOST:PORT", nargs="?",
                    const=radio.DEFAULT_RIGCTLD,
                    help=f"tune through a running rigctld (default "
                         f"{radio.DEFAULT_RIGCTLD}) -- the station rule, since "
                         f"the daemon already holds the CAT port")
    rg.add_argument("--ptt-device", metavar="DEV",
                    help="the serial line whose RTS keys the rig")
    rg.add_argument("--line-ptt", action="store_true",
                    help="key that line directly and leave hamlib to tune. "
                         "Required with --transmit")
    rg.add_argument("--channel", type=int, metavar="HZ",
                    help="channel CENTRE; the dial is 1500 Hz below it")
    rg.add_argument("--dial", type=int, metavar="HZ", help="explicit dial")
    rg.add_argument("--audio-in", help="receiver audio device, for the channel sense")
    rg.add_argument("--audio-out", help="transmit audio device")
    rg.add_argument("--list-audio", action="store_true")

    rl = ap.add_argument_group("the rules this station operates under")
    rl.add_argument("--regulatory", metavar="PROFILE",
                    help="'part97' (US amateur, needs --licence) or "
                         "'unregulated' (no constraint modelled, you are "
                         "answerable for every emission, needs --because). "
                         "Required with --transmit and has no default")
    rl.add_argument("--licence", metavar="CLASS",
                    help="technician|general|advanced|extra (--regulatory part97)")
    rl.add_argument("--because", metavar="REASON",
                    help="why this station is unchecked, e.g. \"dummy load, no "
                         "antenna connected\" (--regulatory unregulated)")
    rl.add_argument("--power", type=float, metavar="WATTS",
                    help="transmitter PEP. Optional except on a band that "
                         "carries its own limit, where the gate refuses an "
                         "emission that does not state its power rather than "
                         "let an unknown amount into a capped band")

    tx = ap.add_argument_group("transmitting")
    tx.add_argument("--transmit", action="store_true",
                    help="ARM the rig (licensed operator on frequency). Without "
                         "it this renders and describes and keys nothing")
    tx.add_argument("--force", action="store_true",
                    help="transmit over an occupied channel. Your own ears "
                         "outrank the detector; a deaf receiver is not "
                         "overridable")
    tx.add_argument("--listen", type=float, default=LISTEN_S, metavar="SECONDS",
                    help=f"channel sense window (default {LISTEN_S:g})")
    tx.add_argument("--gain", type=float, default=None,
                    help="playback peak, which is what sets RF power on a data "
                         "interface. Defaults to [audio] tx_drive from the "
                         f"station file named by {config.STATION_ENV}, else "
                         f"{radio.DEFAULT_TX_GAIN}")
    tx.add_argument("--max-key", type=float, default=None, metavar="SECONDS",
                    help="hard ceiling on keyed time (default: the burst plus 5 s)")
    tx.add_argument("--wav", type=Path, metavar="FILE",
                    help="write exactly what would be transmitted, for the "
                         "record and for a bench decode")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_audio:
        list_devices()
        return 0
    # Payload the burst cannot be rendered without, and the rehearsal renders.
    if not args.mycall:
        print("--mycall names the station this transmission identifies as")
        return 2
    if args.beacon == "wspr" and not args.grid and not args.tone:
        print("the wspr beacon carries a grid square: pass --grid")
        return 2
    try:
        return run(args)
    except ValueError as exc:
        print(f"invalid transmission: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
