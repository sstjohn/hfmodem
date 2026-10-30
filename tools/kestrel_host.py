#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Serve VARA host ports over the stock-compatible audio driver.

Uses kestrel_connect's audio transport. PTT ON/OFF is reported to the host;
this launcher does not configure or key CAT. Virtual cables are the defaults.

    python tools/kestrel_host.py --tx-device 'BlackHole 16ch' \
        --rx-device 'BlackHole 64ch' --cmd-port 8300 --data-port 8301

The application selects MYCALL, bandwidth, COMPRESSION and CONNECT/LISTEN.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packages' / 'hfmodem'))
from hfmodem.kestrel.host.server import VaraServer
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime



def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--chat']:
        from varac_chat import main as chat
        return chat(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--cmd-port', type=int, default=8300)
    ap.add_argument('--data-port', type=int, default=8301)
    ap.add_argument('--kiss-port', type=int, help="stock's KISS port (8100 when enabled); off unless given")
    ap.add_argument('--tx-device', default='BlackHole 16ch')
    ap.add_argument('--rx-device', default='BlackHole 64ch')
    ap.add_argument('--amplitude', type=float, default=0.5)
    ap.add_argument('--low-speed-only', action='store_true')
    args = ap.parse_args(argv)
    # Importing/parsing --help never opens a device.
    from kestrel_connect import AudioVaraIO
    transport = AudioVaraIO(args.tx_device, args.rx_device, args.amplitude)
    runtime = AudioHostRuntime(transport, high_speed=not args.low_speed_only)
    server = VaraServer(host=args.host, cmd_port=args.cmd_port, data_port=args.data_port,
                        kiss_port=args.kiss_port, modem_factory=runtime.modem_factory)
    try:
        transport.start()
        server.start_background()
        print(f'VARA audio host: {args.host}:{server.cmd_port}/{server.data_port}'
              + (f' KISS {server.kiss_port}' if server.kiss_port else ''), flush=True)
        while True:
            runtime.step()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        if runtime.modem:
            runtime.modem.stop()
            runtime.modem.poll()
        transport.stop()
    return 0


if __name__ == '__main__':
    from kestrel_connect import exit_status
    raise SystemExit(exit_status(main))
