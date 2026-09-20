# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel ARQ speaking VARA's real control tokens end-to-end.

Same byte-exact loopback as test_arq_loopback, but with ``vara_compat`` on: the
per-over ACKs, the connect-answer and keepalives go out as the control burst a
real VARA keys, not kestrel's own control frames. The session must still complete
byte-exact, and the short bursts on the wire must be readable by the reader that
bandwidth uses — four DBPSK tokens at BW500 [spec 02 §2.6], one index-modulated
answer at BW2300 [spec 04 §4.2C].
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.arq.fsm import ArqConfig
from hfmodem.kestrel.arq.modem import AudioChannel, KestrelModem
from hfmodem.kestrel.host.server import VaraServer
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.corpora import harness
from hfmodem.kestrel.vara import vara_control as vc

Transcript = harness("transcript").Transcript
VaraClient = harness("vara_client").VaraClient


def _transcript():
    tmp = tempfile.mkdtemp(prefix="kestrel-vara-")
    return Transcript(path=os.path.join(tmp, "t.jsonl"),
                      epoch=time.monotonic(), echo=False)


def _tap(channel):
    """Record every burst each endpoint transmits, so we can inspect the wire."""
    bursts = []
    for ep in (channel.a, channel.b):
        orig = ep.transmit
        def make(orig):
            def t(samples):
                bursts.append(np.asarray(samples, float))
                return orig(samples)
            return t
        ep.transmit = make(orig)
    return bursts


def run_vara_session(payload: bytes, bandwidth: str = "500"):
    ch = AudioChannel()
    bursts = _tap(ch)
    cfg = lambda: ArqConfig(vara_compat=True)

    srvA = VaraServer(host="127.0.0.1", cmd_port=0, data_port=0,
                      iamalive_interval=3600.0,
                      modem_factory=lambda: KestrelModem(ch.a, cfg()))
    srvB = VaraServer(host="127.0.0.1", cmd_port=0, data_port=0,
                      iamalive_interval=3600.0,
                      modem_factory=lambda: KestrelModem(ch.b, cfg()))
    srvA.start_background(); srvB.start_background()

    ta, tb = _transcript(), _transcript()
    A = VaraClient("A", mycall="AAAA1", transcript=ta, host="127.0.0.1",
                   cmd_port=srvA.cmd_port, data_port=srvA.data_port, scheme="varahf")
    B = VaraClient("B", mycall="BBBB2", transcript=tb, host="127.0.0.1",
                   cmd_port=srvB.cmd_port, data_port=srvB.data_port, scheme="varahf")
    try:
        B.connect_tcp(); A.connect_tcp()
        B.set_bandwidth(bandwidth); A.set_bandwidth(bandwidth)
        qB = B.subscribe(); B.listen(True)
        assert A.connect("BBBB2", timeout=60.0, p2p=True), "A did not reach CONNECTED"
        assert B.wait_for("CONNECTED", timeout=30.0), "B host never got CONNECTED"
        B.unsubscribe(qB)
        A.send_data(payload, label="vara")
        received = B.recv_data(len(payload), timeout=180.0)
        assert A.flush(timeout=60.0), "A TX buffer did not drain"
        assert A.disconnect(timeout=60.0), "A did not cleanly DISCONNECT"
        return received, bursts
    finally:
        A.close(); B.close(); srvA.stop(); srvB.stop(); ta.close(); tb.close()


class TestArqVaraTokens(unittest.TestCase):
    def test_session_byte_exact_with_vara_tokens_on_the_wire(self):
        payload = bytes(range(200))          # 5 blocks -> multiple ACKed overs
        received, bursts = run_vara_session(payload)
        self.assertEqual(received, payload, "payload not byte-exact in vara_compat")

        shorts = [b for b in bursts if len(b) < phy.CONTROL_TOKEN_MAX_SAMPLES]
        self.assertTrue(shorts, "no short control-token bursts were emitted")

        names = [m.name for b in shorts if (m := vc.detect_token(b)) is not None]
        # every short burst must be a recognised VARA token...
        self.assertEqual(len(names), len(shorts),
                         "a short burst was not a recognisable VARA token")
        # ...and the data-ACK (the per-over acknowledgement) must appear.
        self.assertIn("data-ack", names, "no VARA data-ACK token on the wire")
        self.assertIn("connected-ack", names, "no VARA connected-ack token")

    def test_bw2300_session_byte_exact_with_the_answer_burst(self):
        """BW2300 vara_compat: every short burst is the index-modulated answer.

        One waveform for the connect-answer and the per-over acknowledgement
        alike [spec 04 §4.2C]. The DBPSK table this used to assert reads none of
        them, and reads none of a real VARA's either — the keepalive and the NAK
        have no BW2300 waveform any recording fixes, so they stay native.
        """
        payload = bytes((i * 5 + 1) & 0xFF for i in range(300))
        received, bursts = run_vara_session(payload, bandwidth="2300")
        self.assertEqual(received, payload, "payload not byte-exact in BW2300 vara_compat")

        shorts = [b for b in bursts if len(b) < phy.CONTROL_TOKEN_MAX_SAMPLES]
        self.assertTrue(shorts, "no short control bursts were emitted")
        names = [m.name for b in shorts
                 if (m := phy.detect_token(b, "2300")) is not None]
        self.assertEqual(len(names), len(shorts),
                         "a short BW2300 burst was not the answer waveform")
        self.assertEqual(set(names), {phy.ANSWER_2300})
        self.assertTrue(all(vc.detect_token(b) is None for b in shorts),
                        "the BW500 table named a burst no real VARA keys wide")


class TestRealAcknowledgement(unittest.TestCase):
    """The receive path at BW2300, against a gateway's own transmission.

    Everything above is kestrel against kestrel, which stays green with the
    waveform wrong. This is the same code path — `_rx_loop`'s short-burst branch —
    handed the acknowledgement a real Winlink gateway keyed on 80 m.
    """

    def _burst(self, path, t, lead=0.05, secs=0.75):
        x = corpora.wav_mono(path)
        i = int((t - lead) * 48000)
        return x[i:i + int(secs * 48000)]

    def _read(self, samples, bw):
        ch = AudioChannel()
        m = KestrelModem(ch.a)
        m.set_bandwidth(bw)
        m._running = True
        ch.a.inbox.put(np.asarray(samples, float))
        ch.a.inbox.put(None)
        m._rx_loop()
        out = []
        while not m._events.empty():
            out.append(m._events.get_nowait())
        return out

    def test_a_real_gateways_acknowledgement_reaches_the_worker(self):
        for path, t in corpora.ONAIR_BW2300_ACKS:
            if not path.exists():
                self.skipTest(f"{path.name} not present")
            burst = self._burst(path, t)
            self.assertLess(len(burst), phy.CONTROL_TOKEN_MAX_SAMPLES)
            evs = self._read(burst, "2300")
            self.assertEqual(evs, [("rx_token", phy.ANSWER_2300)],
                             f"{path.name} at {t} s was dropped by the rx loop")
            self.assertIsNone(vc.detect_token(burst),
                              "the DBPSK table named a real BW2300 burst")

    def test_live_band_audio_around_it_is_not_read_as_one(self):
        """The floor, on the recordings the acknowledgements came off: 200 random
        windows of the same length in each, away from the burst itself."""
        rng = np.random.default_rng(0)
        for path, t in corpora.ONAIR_BW2300_ACKS:
            if not path.exists():
                self.skipTest(f"{path.name} not present")
            x = corpora.wav_mono(path)
            n = int(0.75 * 48000)
            for _ in range(200):
                i = int(rng.integers(0, len(x) - n))
                if abs(i - int(t * 48000)) < 2 * 48000:
                    continue
                self.assertEqual(self._read(x[i:i + n], "2300"), [],
                                 f"{path.name} read band noise at {i / 48000:.1f} s "
                                 "as an acknowledgement")


if __name__ == "__main__":
    unittest.main()
