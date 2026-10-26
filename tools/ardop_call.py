# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Call an ARDOP gateway through besra, driven by hfhost's ARDOP client.

The client was written from the protocol document rather than from besra's
server, which is the only reason its agreement means anything. `tools/onair.sh
ardop` uses it, so an on-air call also exercises that independence.

    python3 tools/ardop_call.py NS0A [seconds] [--bandwidth 200]

besra must already be bound to the radio and serving on 8515/8516 — `onair.sh`
starts it.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

for package in ("hfhost", "hfmodem"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / package))

from hfmodem.core import config   # noqa: E402
from hfhost.ardop import ModemError      # noqa: E402
from hfhost.config import ModemConfig      # noqa: E402
from hfhost.link import open_link          # noqa: E402
from hfhost.transcript import Transcript   # noqa: E402


def parser() -> argparse.ArgumentParser:
    """Built separately from `main` so `--help` costs nothing.

    Everything below used to run at import: asking this tool to describe itself
    opened a socket to a modem that was not there.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("gateway", help="callsign to call")
    p.add_argument("seconds", nargs="?", type=float, default=90.0,
                   help="how long to wait for the outcome (default 90)")
    p.add_argument("--mycall", default=None,
                   help="default: $MYCALL, else the station file's")
    p.add_argument("--cmd-port", type=int, default=8515,
                   help="besra's command port; data is +1 (default 8515)")
    # THE SAME BANDWIDTH THE MODEM WAS STARTED AT. `ARQBW` pushed after start()
    # governs the next ConReq, so a client carrying its own default overrules
    # the server's ceiling: a call launched at BW200 went out `ARQBW now
    # 2000MAX`, and 200 is the only bandwidth that has moved mail from here.
    p.add_argument("--bandwidth", type=int, default=500, choices=(200, 500, 1000, 2000),
                   help="ARQ bandwidth in Hz; besra's own default is 500")
    return p


def retained_call(link) -> str:
    """The callsign the modem already holds. A bare `MYCALL` is a query: sending one
    in place of a callsign would leave whatever was there."""
    try:
        return link.client.command("MYCALL").fields["value"]
    except (ModemError, TimeoutError):
        return ""


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    tr_path = os.path.join(tempfile.mkdtemp(), "ardop-call.jsonl")
    cfg = ModemConfig(name="besra", cmd_port=a.cmd_port, data_port=a.cmd_port + 1,
                      dialect="ardop", bandwidth=str(a.bandwidth))
    link = open_link(cfg, "127.0.0.1", Transcript(tr_path, "ardop-call"))

    q: queue.Queue = queue.Queue()
    link.subscribe(q)
    call = a.mycall or config.mycall()
    if call:
        link.configure(call)
    link.attach()
    print(f"attached: {link.attached}  modem: {link.request_version()}", flush=True)
    if not call:
        call = retained_call(link)
        if not call:
            print("the modem holds no callsign either -- nothing to call as", flush=True)
            link.close()
            return 1
        print(f"calling as the modem's retained {call}", flush=True)
        link.configure(call)
    print(f"ARQCALL {a.gateway} -- keying", flush=True)
    link.connect(a.gateway)

    t0 = time.time()
    seen: list[str] = []
    while time.time() - t0 < a.seconds:
        try:
            ev = q.get(timeout=1.0)
        except queue.Empty:
            if link.connected:
                break
            continue
        seen.append(ev.kind)
        print(f"  [{time.time() - t0:6.1f}s] {ev.kind:<12} "
              f"{getattr(ev, 'fields', None) or ''}", flush=True)
        if ev.kind in ("connected", "disconnected", "fault"):
            break

    print(f"\nconnected: {link.connected}", flush=True)
    print(f"event kinds seen: {sorted(set(seen))}", flush=True)
    print(f"transcript: {tr_path}", flush=True)
    link.close()
    return 0 if link.connected else 1


if __name__ == "__main__":
    raise SystemExit(main())
