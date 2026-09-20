# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two Sabir CBOR host servers connected through simulated audio.

No radio is used. Station A listens on 8400 and station B on 8410.
"""

from __future__ import annotations

import argparse
import time

from hfmodem.sabir.sim.m4 import hf_channel, make_pair


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="moderate",
                    choices=["good", "moderate", "poor", "nvis_disturbed",
                             "polar_disturbed", "clean"],
                    help="Watterson profile between the two stations")
    ap.add_argument("--snr", type=float, default=15.0, help="SNR dB in 3 kHz")
    ap.add_argument("--ports", type=int, nargs=2, default=[8400, 8410],
                    metavar=("A", "B"))
    ap.add_argument("--realtime", action="store_true",
                    help="pace the virtual air to the wall clock, so an "
                         "external client's throughput/PTT-duty figures are real")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    profile = None if args.profile == "clean" else args.profile
    log = (lambda *a: None) if args.quiet else (
        lambda who, msg: print(f"[{time.strftime('%H:%M:%S')}] {who}: {msg}",
                               flush=True))
    pair = make_pair(hf_channel(profile, args.snr), ports=tuple(args.ports), log=log,
                     realtime=args.realtime)
    for name, srv in zip("AB", pair.servers):
        print(f"station {name}: host-api 127.0.0.1:{srv.port}", flush=True)
    paced = " (real-time paced)" if args.realtime else ""
    print(f"channel: {args.profile} at {args.snr:+.0f} dB{paced}; Ctrl-C to stop",
          flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pair.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
