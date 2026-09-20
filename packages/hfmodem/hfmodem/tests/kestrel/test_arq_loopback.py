# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel<->kestrel ARQ end-to-end over the host API (the integration proof).

Two independent kestrel modems, each behind its own host-API TCP server, share
one :class:`AudioChannel`. Two oracle clients drive them exactly as VarAC / Pat
would:

    host-API(A)  ->  ArqFsm(A)  ->  TX synth  ->  audio  ->  RX decode  ->
    ArqFsm(B)  ->  host-API(B)          (and the reverse for ACKs)

A payload written to A's data port must arrive byte-exact on B's data port,
through the connect handshake, stop-and-wait ARQ (per-over ACK + gear-shift) and
3-burst disconnect of `spec/05` — every frame a real synthesised VARA HF 500
burst recovered by the frozen receiver.

Run:  python3 -m pytest kestrel/tests/kestrel/test_arq_loopback.py -s
      python3 kestrel/tests/kestrel/test_arq_loopback.py        (standalone, prints trace)
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
    tmp = tempfile.mkdtemp(prefix="kestrel-arq-")
    return Transcript(path=os.path.join(tmp, "t.jsonl"),
                      epoch=time.monotonic(), echo=False)


def run_session(payload: bytes, verbose: bool = False):
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
        B.set_bandwidth("500"); A.set_bandwidth("500")

        # Responder arms LISTEN and waits for the inbound connect.
        qB = B.subscribe()
        B.listen(True)

        # Initiator dials the responder.
        assert A.connect("BBBB2", timeout=60.0, p2p=True), "A did not reach CONNECTED"

        # Responder's host must see CONNECTED too.
        got_b_connected = B.wait_for("CONNECTED", timeout=30.0)
        B.unsubscribe(qB)
        assert got_b_connected, "B host never got CONNECTED"

        # Payload A -> B over the air.
        A.send_data(payload, label="e2e")
        received = B.recv_data(len(payload), timeout=180.0)

        assert A.flush(timeout=60.0), "A TX buffer did not drain to 0"
        assert A.disconnect(timeout=60.0), "A did not cleanly DISCONNECT"
        return received
    finally:
        A.close(); B.close()
        srvA.stop(); srvB.stop()
        ta.close(); tb.close()


class TestArqLoopback(unittest.TestCase):
    def test_payload_transfers_byte_exact(self):
        payload = bytes(range(200))          # 5 blocks -> exercises 1->2 gear-shift
        received = run_session(payload)
        self.assertEqual(received, payload, "payload not byte-exact through ARQ")


if __name__ == "__main__":
    pl = bytes(range(200))
    t0 = time.time()
    got = run_session(pl, verbose=True)
    dt = time.time() - t0
    ok = got == pl
    print(f"\n=== {'PASS' if ok else 'FAIL'}: {len(got)}/{len(pl)} bytes byte-exact="
          f"{ok} in {dt:.1f}s ===", flush=True)
    sys.exit(0 if ok else 1)
