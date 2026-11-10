# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel<->kestrel ARQ end-to-end over the BW2300 wideband waveform.

Same integration proof as ``test_arq_loopback.py`` (host API -> ArqFsm -> TX synth
-> audio -> RX decode -> ArqFsm -> host API), but the connection selects
**bandwidth 2300**, so every frame — connect handshake, stop-and-wait DATA overs
with per-over ACK, and the 3-burst disconnect — is a real synthesised BW2300 base
(rec3) OFDM burst recovered by the BW2300 receiver. A payload written to A's data
port must arrive byte-exact on B's data port.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

from hfmodem.kestrel.arq.modem import AudioChannel, KestrelModem
from hfmodem.kestrel.host.server import VaraServer
from hfmodem.tests.kestrel.corpora import harness

Transcript = harness("transcript").Transcript
VaraClient = harness("vara_client").VaraClient


def _transcript():
    tmp = tempfile.mkdtemp(prefix="kestrel-arq2300-")
    return Transcript(path=os.path.join(tmp, "t.jsonl"),
                      epoch=time.monotonic(), echo=False)


def run_session(payload: bytes, bandwidth: str = "2300", verbose: bool = False):
    ch = AudioChannel()
    log = (lambda who, msg: print(f"{who:>12}: {msg}", flush=True)) if verbose else None

    srvA = VaraServer(host="127.0.0.1", cmd_port=0, data_port=0,
                      iamalive_interval=3600.0,
                      modem_factory=lambda: KestrelModem(ch.a), log=log)
    srvB = VaraServer(host="127.0.0.1", cmd_port=0, data_port=0,
                      iamalive_interval=3600.0,
                      modem_factory=lambda: KestrelModem(ch.b), log=log)
    srvA.start_background()
    srvB.start_background()

    ta, tb = _transcript(), _transcript()
    A = VaraClient("A", mycall="AAAA1", transcript=ta, host="127.0.0.1",
                   cmd_port=srvA.cmd_port, data_port=srvA.data_port, scheme="varahf")
    B = VaraClient("B", mycall="BBBB2", transcript=tb, host="127.0.0.1",
                   cmd_port=srvB.cmd_port, data_port=srvB.data_port, scheme="varahf")
    try:
        B.connect_tcp(); A.connect_tcp()
        B.set_bandwidth(bandwidth); A.set_bandwidth(bandwidth)

        qB = B.subscribe()
        B.listen(True)
        assert A.connect("BBBB2", timeout=60.0, p2p=True), "A did not reach CONNECTED"
        got_b_connected = B.wait_for("CONNECTED", timeout=30.0)
        B.unsubscribe(qB)
        assert got_b_connected, "B host never got CONNECTED"

        A.send_data(payload, label="e2e-2300")
        received = B.recv_data(len(payload), timeout=300.0)

        assert A.flush(timeout=90.0), "A TX buffer did not drain to 0"
        assert A.disconnect(timeout=60.0), "A did not cleanly DISCONNECT"
        return received
    finally:
        A.close(); B.close()
        srvA.stop(); srvB.stop()
        ta.close(); tb.close()


class TestArqBw2300(unittest.TestCase):
    def test_bw2300_payload_transfers_byte_exact(self):
        payload = bytes((i * 37 + 11) & 0xFF for i in range(180))   # 3 BW2300 blocks
        received = run_session(payload, bandwidth="2300")
        self.assertEqual(received, payload, "BW2300 payload not byte-exact through ARQ")


if __name__ == "__main__":
    pl = bytes((i * 37 + 11) & 0xFF for i in range(180))
    t0 = time.time()
    got = run_session(pl, verbose=True)
    dt = time.time() - t0
    ok = got == pl
    print(f"\n=== BW2300 {'PASS' if ok else 'FAIL'}: {len(got)}/{len(pl)} byte-exact="
          f"{ok} in {dt:.1f}s ===", flush=True)
    sys.exit(0 if ok else 1)
