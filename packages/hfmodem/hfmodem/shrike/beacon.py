# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""TX-only beacon: key shrike's connect repeatedly so a remote SDR can catch it.

No listen, no decode -- just N keyings of the connect on the 1.25 s grid with a
gap between, for verifying on a KiwiSDR waterfall that the rig keys, holds the
dial, and lands the connect's two tones on the right centre, independent of any
gateway. That is all this radiates: the payload list handed to `build_session`
is empty, so the PACTOR-3 loop never runs and nothing 18-tone is in the audio.
A beacon verifies only what it transmits -- checking the 18-tone P3 waveform
itself takes `shrike.ota --data`, which renders a real data packet.

    python -m hfmodem.shrike.beacon --rig x6100 --serial <port> --mycall <call> \
        --dxcall <call> --band 40m --count 12 --gap 2 --audio-out "USB Audio"

The connect is ADDRESSED, so `--dxcall` has no default: the station named hears
itself called on every one of the N keyings. Point it at your own callsign unless
you mean to call somebody.
"""
from __future__ import annotations

import argparse
import time

from . import session, spec
from ..core.devices import find_device
from ..core.ptt import PttError
from ..core.rigs import RIGS
from .ota import Rig, _play, center_to_dial, ptt_device, GATEWAYS


def main():
    p = argparse.ArgumentParser(description="shrike on-air TX beacon")
    p.add_argument("--rig", choices=list(RIGS), default="x6100")
    p.add_argument("--serial", required=True,
                   help="CAT serial port, e.g. /dev/cu.usbserial-XXXXB0 -- "
                        "`ls /dev/cu.*` to find yours")
    p.add_argument("--baud", type=int, default=0)
    p.add_argument("--mycall", required=True)
    p.add_argument("--dxcall", required=True,
                   help="the station to call -- no default, this callsign "
                        "goes on the air")
    p.add_argument("--band", default="40m")
    p.add_argument("--center", type=int)
    p.add_argument("--dial", type=int)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--gap", type=float, default=2.0)
    p.add_argument("--max-key", type=float, default=6.0)
    p.add_argument("--audio-out")
    args = p.parse_args()

    if args.dial:
        dial = args.dial
    else:
        center = args.center or GATEWAYS.get(args.dxcall.upper(), {}).get(args.band)
        if center is None:
            raise SystemExit("pass --center or --dial")
        dial = center_to_dial(center)

    audio = session.build_session(args.dxcall.upper(), [], sl=2)
    dur = len(audio) / spec.SAMPLE_RATE
    out_dev = find_device(args.audio_out, "out")
    r = RIGS[args.rig]
    # The same arm gate and the same conversion ota.run makes: the operator's
    # fix is a flag, not a stack trace.
    try:
        ptt_port = ptt_device(args.rig, args.serial)
    except PttError as exc:
        raise SystemExit(f"NOT KEYING: {exc}") from None
    rig = Rig(r["model"], args.serial, args.baud or r["baud"],
              ptt_type=r.get("ptt_type", "RIG"), ptt_port=ptt_port)
    try:
        rig.set_mode(r["mode"]); rig.set_freq(dial)
        centre = dial + int(spec.CENTER_FREQ_HZ)
        print(f"beacon: dial {dial} Hz, signal centre ~{centre} Hz {r['mode']}  "
              f"({args.count} x {dur:.1f}s connect, {args.gap}s gap ~= "
              f"{args.count*(dur+args.gap):.0f}s total). Watch a KiwiSDR on "
              f"{centre/1e6:.4f} MHz.")
        for i in range(1, args.count + 1):
            print(f"  key {i}/{args.count}", flush=True)
            _play(audio, out_dev, args.max_key, rig)
            # A keying is only a keying if the channel that took it is still
            # standing; without this, a rigctl that died mid-run let every
            # remaining cycle count itself as sent.
            fail = rig.key_failure()
            if fail is not None:
                raise SystemExit(f"TRANSMISSION NOT CONFIRMED: {fail} -- "
                                 f"stopping rather than counting keyings that "
                                 f"never went out")
            time.sleep(args.gap)
    finally:
        rig.stop()
        print("beacon done, PTT off.")


if __name__ == "__main__":
    # A signal must unwind, not just kill: every unkey here hangs off a
    # `finally`. See onair._unkey_on_signal for what this cost once.
    from hfmodem.shrike.onair import _unkey_on_signal
    _unkey_on_signal()
    main()
