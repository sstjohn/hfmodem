# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CLI: run the kestrel host-side VARA-protocol server.

Point a real client at it (VarAC / Pat / Winlink Express) or this project's own
VARA host-API client. With the default loopback modem, a client can connect, transfer
data (echoed back), and disconnect -- no radio required.

    python3 run_server.py --cmd-port 8300 --data-port 8301

Then, e.g. in Pat, configure a VARA HF transport pointing at 127.0.0.1:8300.
Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time

from .protocol import DEFAULT_CMD_PORT, DEFAULT_DATA_PORT
from .server import VaraServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    ap.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    ap.add_argument("--iamalive-interval", type=float, default=60.0,
                    help="seconds between IAMALIVE heartbeats")
    ap.add_argument("--quiet", action="store_true", help="do not log traffic")
    args = ap.parse_args(argv)

    def log(who, msg):
        if not args.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {who:>10}: {msg}", flush=True)

    srv = VaraServer(host=args.host, cmd_port=args.cmd_port,
                     data_port=args.data_port,
                     iamalive_interval=args.iamalive_interval, log=log)
    print(f"kestrel host-API server listening: cmd={srv.host}:{srv.cmd_port} "
          f"data={srv.host}:{srv.data_port} (loopback modem)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
