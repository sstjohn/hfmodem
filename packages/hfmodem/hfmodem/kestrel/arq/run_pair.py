# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Launch two kestrel host-API servers sharing one AudioChannel.

Each server runs a real :class:`KestrelModem` (ARQ FSM + proven VARA HF 500 burst
codec). Point any VARA host-API client (Pat, VarAC, or the test client) at
each port pair; a CONNECT on side A establishes an over-the-air ARQ session to
side B, and data written to A's data port arrives on B's — every frame a real
synthesised burst decoded by the frozen receiver.

    python3 -m hfmodem.kestrel.arq.run_pair
        A: cmd=8300 data=8301   B: cmd=8310 data=8311

This is the standalone wiring of the ARQ FSM to the host-API server (the test
`tests/kestrel/test_arq_loopback.py` drives the same setup programmatically).
"""
from __future__ import annotations

import argparse
import sys
import time

from hfmodem.kestrel.arq.modem import AudioChannel, KestrelModem  # noqa: E402
from hfmodem.kestrel.host.server import VaraServer  # noqa: E402  host_api


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--a-cmd", type=int, default=8300)
    ap.add_argument("--a-data", type=int, default=8301)
    ap.add_argument("--b-cmd", type=int, default=8310)
    ap.add_argument("--b-data", type=int, default=8311)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def mklog(tag):
        if args.quiet:
            return None
        return lambda who, msg: print(f"[{tag}] {who:>10}: {msg}", flush=True)

    ch = AudioChannel()
    srvA = VaraServer(host=args.host, cmd_port=args.a_cmd, data_port=args.a_data,
                      modem_factory=lambda: KestrelModem(ch.a), log=mklog("A"))
    srvB = VaraServer(host=args.host, cmd_port=args.b_cmd, data_port=args.b_data,
                      modem_factory=lambda: KestrelModem(ch.b), log=mklog("B"))
    srvA.start_background()
    srvB.start_background()
    print(f"kestrel pair up:  A cmd={srvA.cmd_port} data={srvA.data_port}   "
          f"B cmd={srvB.cmd_port} data={srvB.data_port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srvA.stop(); srvB.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
