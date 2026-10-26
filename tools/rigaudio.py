#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Put a capture device on the laptop speakers, so the operator can still hear.

WHY THIS EXISTS. The station has one rig and one pair of ears. Feeding the rig's
speaker output into a capture interface gives this project two things it has
never had locally -- received audio on a second device, independent of the codec
the modems own, and the rig's MONITOR, which is our own transmission as the rig
actually emits it. What it takes away is the operator's ability to hear any of
it, and their ears have overturned more conclusions than any log in this tree.
So the audio goes to the speakers as well, live, while everything else records.

It is a monitor path and nothing else. It opens no rig, holds no serial port and
cannot key: the only devices it touches are an input and an output.

    ./tools/rigaudio.py                     # one capture device -> both ears
    ./tools/rigaudio.py --list              # what this machine has
    ./tools/rigaudio.py --in "KT USB" --out "MacBook Pro Speakers" --gain 2.0

TWO SOURCES, ONE EAR EACH. The station hears the same air twice and the two are
not the same evidence: the rear data jack is what the modem decodes, and the
speaker jack carries the rig's MONITOR, so it holds our own transmission as well
as the band. Put one in each ear and a difference between them is audible
immediately rather than after an arm and an analysis -- which is how the operator
found a receive fault on 2026-09-16 by switching between them by hand.

    ./tools/rigaudio.py --left "Line IN" --right "USB Audio Device" \
                        --out "Headphone OUT"

Each source gets its own ring, because they are separate crystals. Keys while it
runs: s swaps the ears, l/r solo one, b restores both, +/- gain, q quits.

TWO STREAMS, NOT ONE DUPLEX. Two USB devices run on two crystals, so a single
duplex stream across them would have to resample or glitch. A ring buffer between
an input callback and an output callback absorbs the difference, and the drift --
tens of parts per million, a sample every few seconds -- is taken out by letting
the buffer run one block long or short rather than by resampling. What that costs
is one block of audio every few minutes, inaudible on a monitor path and not a
path anything is decoded from.
"""
from __future__ import annotations

import argparse
import contextlib
import queue
import sys
import termios
import tty

import numpy as np
import sounddevice as sd
from hfcapture import device_claim, device_opened
from hfmodem.core import claim

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import codec_gain  # noqa: E402  -- a sibling tool, not a package

#: 5.3 ms at 48 kHz. Small enough that the operator hears the band rather than a
#: recording of it -- a peer's answer lands 70 ms after our carrier drops, and a
#: monitor that lagged that far would put the two in the wrong order by ear.
BLOCK = 256
#: Blocks of slack between the two clocks. Four is ~21 ms: deep enough to ride a
#: scheduling hiccup, shallow enough that the total path stays under ~30 ms.
DEPTH = 4
#: One key press, in amplitude. ~2 dB, so a dozen presses cross the useful range.
STEP = 1.26

#: Two rows of keys, laid out as the ears are: left hand moves the left ear.
#: Volume is this monitor's own and affects nothing else; gain is the device's
#: capture scalar and changes what every other process on it receives.
KEYS = """  volume  left 1 down 2 up      right 9 down 0 up     (this monitor only)
  gain    left 3 down 4 up      right 7 down 8 up     (the device: the modem hears it too)
  s swap ears   l/r solo   b both   m mute   q quit"""


#: key -> (which source, multiplier)
VOLUME_KEYS = {"1": (0, 1 / STEP), "2": (0, STEP), "9": (1, 1 / STEP), "0": (1, STEP)}
GAIN_KEYS = {"3": (0, 1 / STEP), "4": (0, STEP), "7": (1, 1 / STEP), "8": (1, STEP)}


def _pick(name: str, want_input: bool) -> int:
    """The one device whose name contains `name`, or a refusal naming the rest.

    EXACTLY ONE. `"USB Audio"` matched both the rig's codec and the interface
    added on 2026-08-14 (`KT USB Audio` contains it), and the launcher's own
    matcher took the first by enumeration order -- which is to say by luck, since
    USB order moves across a replug. A monitor picking the wrong device is a
    confusing evening; the same bug in a transmit path is a station modulating
    something nobody is listening to.
    """
    key = "max_input_channels" if want_input else "max_output_channels"
    hits = [(i, d) for i, d in enumerate(sd.query_devices())
            if name.lower() in d["name"].lower() and d[key] > 0]
    if not hits:
        raise SystemExit(f"no {'input' if want_input else 'output'} device "
                         f"matching {name!r} -- try --list")
    if len(hits) > 1:
        named = ", ".join(f"{i}:{d['name']}" for i, d in hits)
        raise SystemExit(f"{name!r} matches more than one device ({named}) -- "
                         "name it precisely enough to pick one")
    return hits[0][0]


class _Source:
    """One capture device and the ring that absorbs its clock against the output.

    Per source, not shared: two USB interfaces run on two crystals and neither is
    the output's, so a single ring fed by both would have to drop a block for one
    device and starve the other in the same instant.
    """

    def __init__(self, device: int, label: str):
        self.device = device
        self.label = label
        self.name = sd.query_devices()[device]["name"]
        self.ring: queue.Queue[np.ndarray] = queue.Queue(maxsize=DEPTH * 4)
        self.peak = 0.0
        self.over = 0
        self.under = 0
        self.volume = 1.0                      # software, this ear only
        self.codec = None                      # CoreAudio capture scalar, or None
        try:
            self.codec_id = codec_gain.find(self.name)
            self.codec = codec_gain.get_gain(self.codec_id)[1]
        except (SystemExit, OSError, RuntimeError, TypeError, IndexError):
            self.codec_id = None

    def set_codec(self, step: float) -> None:
        """Move this device's own capture scalar.

        This is the hardware gain, not a monitor level: it changes what every
        other process on that device receives, including a running arm's modem.
        Hence the separate key and the separate readout -- an operator adjusting
        what they hear should never move what the decoder hears by accident.
        """
        if self.codec_id is None or self.codec is None:
            return
        want = min(1.0, max(0.001, self.codec * step))
        if codec_gain.set_gain(self.codec_id, want):
            self.codec = codec_gain.get_gain(self.codec_id)[1]

    def on_in(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        self.peak = max(self.peak, float(np.abs(block).max()))
        try:
            self.ring.put_nowait(block)
        except queue.Full:
            # This device's clock is ahead, or the output stalled. Drop the
            # OLDEST so what is heard stays current: a monitor that lags is worse
            # than one that skips, and a peer answers 70 ms after our carrier
            # drops -- lag that far and the two arrive in the wrong order by ear.
            with contextlib.suppress(queue.Empty):
                self.ring.get_nowait()
                self.ring.put_nowait(block)
                self.over += 1

    def take(self, frames: int) -> np.ndarray:
        try:
            return self.ring.get_nowait()
        except queue.Empty:
            self.under += 1
            return np.zeros(frames, dtype="float32")

    def stream(self, rate: int) -> sd.InputStream:
        return sd.InputStream(device=self.device, channels=1, samplerate=rate,
                              blocksize=BLOCK, dtype="float32", callback=self.on_in)


@contextlib.contextmanager
def _keys():
    """Single keypresses without waiting for Enter, and a terminal restored after.

    Degrades to no keys at all when stdin is not a terminal, so the monitor still
    runs from a script or with its output piped.
    """
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        import select

        def pressed():
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            return sys.stdin.read(1) if ready else None

        yield pressed
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _list() -> None:
    for i, d in enumerate(sd.query_devices()):
        io = f"{d['max_input_channels']} in, {d['max_output_channels']} out"
        print(f"  {i}: {d['name']:34s} {io:16s} {d['default_samplerate']:.0f} Hz")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", default="KT USB",
                   help="one capture device into both ears (default: KT USB)")
    p.add_argument("--left", help="capture device for the left ear")
    p.add_argument("--right", help="capture device for the right ear")
    p.add_argument("--out", dest="dst", default="MacBook Pro Speakers",
                   help="where the operator listens")
    p.add_argument("--rate", type=int, default=48000)
    p.add_argument("--gain", type=float, default=1.0,
                   help="linear, applied to the monitor only")
    p.add_argument("--left-gain", type=float, help="linear, left ear only")
    p.add_argument("--right-gain", type=float, help="linear, right ear only")
    p.add_argument("--quiet", action="store_true", help="no level meter")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        _list()
        return 0
    if bool(args.left) != bool(args.right):
        raise SystemExit("name both --left and --right, or neither")

    with device_claim("rigaudio", devices=claim.Devices(
            capture=args.left or args.src, playback=args.dst)):
        if args.left:
            sources = [_Source(_pick(args.left, True), "L"),
                       _Source(_pick(args.right, True), "R")]
        else:
            sources = [_Source(_pick(args.src, True), "both")]

        dst = _pick(args.dst, False)
        dst_name = sd.query_devices()[dst]["name"]
        out_ch = min(2, sd.query_devices()[dst]["max_output_channels"])
        if len(sources) == 2 and out_ch < 2:
            raise SystemExit(f"{dst_name!r} is mono; two sources need a stereo output")

        sources[0].volume = args.left_gain if args.left_gain is not None else args.gain
        if len(sources) == 2:
            sources[1].volume = args.right_gain if args.right_gain is not None else args.gain
        state = {"swap": False, "solo": None, "muted": False}

        def on_out(outdata, frames, time_info, status):
            blocks = [src.take(frames) * src.volume for src in sources]
            if len(blocks) == 1:
                ears = [blocks[0]] * out_ch
            else:
                ears = list(blocks)
                if state["swap"]:
                    ears.reverse()
                if state["solo"] is not None:
                    ears = [ears[state["solo"]]] * 2
            if state["muted"]:
                outdata[:] = 0.0
            else:
                outdata[:] = np.clip(np.column_stack(ears[:out_ch]), -1.0, 1.0)

        for src in sources:
            codec = "gain unreadable" if src.codec is None else f"gain {src.codec:.3f}"
            print(f"  {src.label:>4s}: {src.name}  ({codec})")
        print(f"    ->  {dst_name}, {args.rate} Hz, {BLOCK} frames "
              f"({1000 * BLOCK / args.rate:.1f} ms)")
        print("  nothing here can key a transmitter.")
        print(KEYS if len(sources) == 2 else "  keys: 1/2 volume, 3/4 gain, m mute, q quit")
        print()

        try:
            with contextlib.ExitStack() as open_streams:
                for src in sources:
                    stream = src.stream(args.rate)
                    open_streams.callback(stream.close)
                    open_streams.enter_context(stream)
                output = sd.OutputStream(
                    device=dst, channels=out_ch, samplerate=args.rate,
                    blocksize=BLOCK, dtype="float32", callback=on_out)
                open_streams.callback(output.close)
                open_streams.enter_context(output)
                device_opened()
                with _keys() as pressed:
                    settled = False
                    while True:
                        sd.sleep(200)
                        if not settled:
                            # The rings fill before the output starts pulling, so the
                            # first tick always shows overruns that never happened in
                            # anyone's ears. Count drift from a running monitor only.
                            for src in sources:
                                src.over = src.under = 0
                            settled = True
                        key = pressed()
                        if key in ("q", "\x03"):
                            break
                        if key == "s":
                            state["swap"] = not state["swap"]
                        elif key in ("l", "r"):
                            want = 0 if key == "l" else 1
                            state["solo"] = None if state["solo"] == want else want
                        elif key == "b":
                            state["solo"] = None
                        elif key == "m":
                            state["muted"] = not state["muted"]
                        elif key in VOLUME_KEYS:
                            which, step = VOLUME_KEYS[key]
                            if which < len(sources):
                                src = sources[which]
                                src.volume = min(16.0, max(0.06, src.volume * step))
                        elif key in GAIN_KEYS:
                            which, step = GAIN_KEYS[key]
                            if which < len(sources):
                                sources[which].set_codec(step)
                        if args.quiet:
                            continue
                        meters = []
                        for src in sources:
                            peak, src.peak = src.peak, 0.0
                            bar = "#" * min(14, int(peak * 14 / 0.5))
                            codec = "  --  " if src.codec is None else f"{src.codec:5.3f}"
                            meters.append(f"{src.label} {peak:6.4f}|{bar:<14s}| "
                                          f"vol {src.volume:4.2f} gain {codec}")
                        drift = sum(s.over for s in sources), sum(s.under for s in sources)
                        said = "  ".join(meters)
                        if state["swap"]:
                            said += " SWAPPED"
                        if state["solo"] is not None:
                            said += f" SOLO {sources[state['solo']].label}"
                        if state["muted"]:
                            said += " MUTED"
                        if any(drift):
                            said += f"  drift {drift[0]}+/{drift[1]}-"
                        print(f"\r  {said}   ", end="", flush=True)
        except KeyboardInterrupt:
            pass
        except Exception as exc:                                # noqa: BLE001
            print(f"\n!! monitor failed: {exc}", file=sys.stderr)
            return 1
        print("\n  monitor stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
