# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Tests for the kestrel VARA-interop connection modules (spec 04 §4.2, 05 §5.3).

Covers:
  * the VB6-Rnd LCG core against the spec KAT (spec 04 §4.2.3),
  * MFSK handshake-tone generation (determinism, spec-range, regression),
  * recognizers (accept correct / reject wrong / no false accepts),
  * MFSK synth -> demod round-trip (spec 04 §4.2.1),
  * OFDM link-setup TX -> RX caller round trip (the off-air proof is in
    test_vara_mail_rx),
  * the VARA-station MFSK handshake driver (kestrel<->kestrel, mfsk_only).

Run:  python3 -m pytest kestrel/tests/kestrel/test_vara_interop.py -q
"""
from __future__ import annotations

import unittest

import numpy as np

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OFDM

# A spread of callsigns (lengths 4-6, with SSID) for matrix tests.
_CALLS = ["K7ABC", "W2XYZ", "AAAA1", "BBBB2", "W1AW", "N0DX",
          "GATE1", "ZZZZ9", "VK3XYZ", "W1AW-7"]

# Frozen REGRESSION vectors. NOTE: spec 04 §4.2.4 gives only match-COUNTS, not
# concrete numeric tone vectors, so these are self-derived from the spec closed
# form and frozen here as a regression guard  [ours — see FLAG in the report].
_KAT_PAYLOAD = {
    ("K7ABC", "connect-request"):
        [36, 53, 86, 69, 52, 85, 76, 61, 98, 49, 64, 71, 68, 73, 52, 61, 74, 85,
         90, 45, 46, 89, 44, 37, 82, 83, 36, 61, 38, 43, 32],
    ("K7ABC", "connect-response"):
        [80, 71, 84, 77, 70, 33, 56, 79, 56, 95, 98, 59, 50, 29, 76],
    ("W2XYZ", "connect-request"):
        [44, 33, 64, 49, 62, 33, 70, 53, 72, 35, 94, 89, 80, 77, 60, 89, 88, 81,
         40, 73, 78, 61, 72, 95, 78, 77, 70, 61, 78, 93, 78],
}


class TestLcgCore(unittest.TestCase):
    def test_vb6_rnd_reference_sequence(self):
        # spec 03 §3.5.1: default-seed Rnd sequence.
        s = 0x50000
        got = []
        for _ in range(5):
            s = VF._lcg(s)
            got.append(round(s / 2**24, 7))
        self.assertEqual(
            got, [0.7055475, 0.5334240, 0.5795186, 0.2895625, 0.3019480])


class TestHandshakeGenerator(unittest.TestCase):
    def test_preambles_match_spec(self):
        # spec 04 §4.2.2 fixed preamble tones.
        self.assertEqual(VF.CR.preamble[:10],
                         (74, 68, 70, 60, 60, 77, 50, 76, 78, 74))
        self.assertEqual(VF.CONNECT_RESPONSE.preamble,
                         (62, 67, 55, 66, 59, 72, 68, 55))
        # spec 04 §4.2C: the ack's preamble is four two-tone symbols.
        self.assertEqual(VF.CONNECTED_ACK_PREAMBLE,
                         ((64, 67), (56, 74), (64, 69), (68, 78)))

    def test_payload_counts(self):
        for cs in _CALLS:
            self.assertEqual(len(VF.payload_bins(cs, VF.CR)), 31)
            self.assertEqual(len(VF.payload_bins(cs, VF.CONNECT_RESPONSE)), 15)

    def test_payload_in_spec_range(self):
        # spec 04 §4.2.3: bin = (29 + parity + 14*P + 2*D) & 0xFF, P in 0..4,
        # D in 0..6 -> bins in [29, 98].
        for cs in _CALLS:
            for k in (VF.CR, VF.CONNECT_RESPONSE):
                for b in VF.payload_bins(cs, k):
                    self.assertTrue(29 <= b <= 98, f"{cs} {k.name} bin {b}")

    def test_determinism(self):
        for cs in _CALLS:
            self.assertEqual(VF.payload_bins(cs, VF.CR),
                             VF.payload_bins(cs.lower(), VF.CR))  # uppercased

    def test_regression_vectors(self):
        for (cs, kname), expected in _KAT_PAYLOAD.items():
            self.assertEqual(VF.payload_bins(cs, VF.BURSTS[kname]), expected,
                             f"{cs}/{kname} regression drift")


class TestRecognizers(unittest.TestCase):
    def test_accept_correct_reject_wrong(self):
        for kind in (VF.CR, VF.CONNECT_RESPONSE):
            for cs in _CALLS:
                tones = VF.handshake_tones(cs, kind)
                self.assertTrue(VF.recognize(tones, cs, kind))
                for other in _CALLS:
                    if other != cs:
                        self.assertFalse(VF.recognize(tones, other, kind),
                                         f"false accept {other} vs {cs} {kind.name}")

    def test_no_false_accepts_matrix(self):
        # spec 04 §4.2.4: 0 false accepts.
        for kind in (VF.CR, VF.CONNECT_RESPONSE):
            fa = 0
            for a in _CALLS:
                tones = VF.handshake_tones(a, kind)
                for b in _CALLS:
                    if a != b and VF.recognize(tones, b, kind):
                        fa += 1
            self.assertEqual(fa, 0, f"{kind.name}: {fa} false accepts")

    def test_best_match_picks_addressed_call(self):
        tones = VF.handshake_tones("GATE1", VF.CR)
        call, m, n = VF.best_match(tones, ["N0DX", "GATE1", "W1AW"], VF.CR)
        self.assertEqual(call, "GATE1")
        self.assertEqual(m, n)


class TestMfskRoundTrip(unittest.TestCase):
    def test_synth_demod_recovers_tones(self):
        for kind in (VF.CR, VF.CONNECT_RESPONSE):
            for cs in _CALLS:
                tones = VF.handshake_tones(cs, kind)
                audio = MK.synth_tones(tones)
                rec = MK.demod_tones(audio, len(tones))
                self.assertEqual(rec, tones, f"{cs} {kind.name} round-trip")

    def test_carrier_to_hz(self):
        # spec 04 §4.2.1: f = carrier * 48000/2048.
        self.assertAlmostEqual(MK.carrier_to_hz(64), 64 * 48000 / 2048)


class TestOfdmLinkSetup(unittest.TestCase):
    def test_link_setup_round_trips_the_caller(self):
        # Self-consistency only — the off-air arbiter is test_vara_mail_rx's
        # decode of a real captured VARA link-setup.
        for caller in ("K7ABC", "W1AW-7"):
            audio = OFDM.link_setup_tx(caller)
            self.assertGreater(len(audio), 4 * 48000)      # a ~4.4 s wideband over
            self.assertEqual(OFDM.link_setup_rx(audio), caller)

    def test_link_setup_rx_returns_none_on_noise(self):
        rng = np.random.default_rng(2)
        self.assertIsNone(OFDM.link_setup_rx(rng.standard_normal(48000 * 5)))


# --------------------------------------------------------------------------- #
class _CaptureIO(VA.VaraIO):
    """Test IO: transmit audio into the peer's inbox; capture events."""
    def __init__(self, name, peer_inbox, log):
        self.name = name
        self.peer_inbox = peer_inbox
        self.log_list = log
        self.connected_result = None
        self.pending_seen = False

    def key(self, on):
        pass

    def tx(self, samples):
        self.peer_inbox.append(np.asarray(samples, float))

    def pending(self):
        self.pending_seen = True

    def connected(self, caller, called, bw):
        self.connected_result = (caller, called, bw)

    def log(self, msg):
        self.log_list.append((self.name, msg))


def _run_mfsk_handshake(initiator_call, gateway_call, listen_calls):
    """Drive a kestrel<->kestrel MFSK handshake to quiescence. Returns
    (init_driver, resp_driver)."""
    i_inbox, r_inbox, log = [], [], []
    io_i = _CaptureIO("init", r_inbox, log)      # initiator TX -> responder inbox
    io_r = _CaptureIO("resp", i_inbox, log)      # responder TX -> initiator inbox
    init = VA.VaraStationHandshake([initiator_call], io_i, mfsk_only=True)
    resp = VA.VaraStationHandshake(listen_calls, io_r, mfsk_only=True)
    resp.listen(True)
    init.originate(gateway_call)
    # Pump: FIFO, responder inbox first so its reply bursts queue for the initiator.
    guard = 0
    while (r_inbox or i_inbox) and guard < 100:
        guard += 1
        if r_inbox:
            resp.on_rx_audio(r_inbox.pop(0))
        elif i_inbox:
            init.on_rx_audio(i_inbox.pop(0))
    return init, resp


class TestVaraHandshakeDriver(unittest.TestCase):
    def test_mfsk_handshake_completes(self):
        init, resp = _run_mfsk_handshake("MYCALL", "GATE1", ["GATE1", "N0DX"])
        self.assertEqual(init.state, VA.VaraState.CONNECTED)
        self.assertEqual(resp.state, VA.VaraState.CONNECTED)
        # initiator learns both (caller=self, called=gateway)  [spec 05 §5.3 step6]
        self.assertEqual(init.io.connected_result, ("MYCALL", "GATE1", "2300"))
        # mfsk_only skips the link-setup in both directions, so the responder
        # completes the handshake with the caller unlearned (spec 05 §5.3.2);
        # test_vara_mail_rx covers the full path where it decodes the caller.
        self.assertEqual(resp.io.connected_result,
                         (VA.CALLER_UNKNOWN, "GATE1", "2300"))
        self.assertTrue(resp.io.pending_seen)

    def test_responder_ignores_cr_for_other_call(self):
        # Responder listens for GATE1 only; initiator dials N0DX -> no match.
        init, resp = _run_mfsk_handshake("MYCALL", "N0DX", ["GATE1"])
        self.assertEqual(resp.state, VA.VaraState.LISTENING)
        self.assertNotEqual(init.state, VA.VaraState.CONNECTED)
        self.assertFalse(resp.io.pending_seen)

    def test_real_vara_linksetup_is_synthesised(self):
        # Against a real VARA station (mfsk_only=False) the initiator's step-4
        # link-setup is a real BW2300 rec3 over carrying the caller callsign — it
        # is keyed and played, not stubbed.  [spec 04 §4.2A, spec 05 §5.3.2]
        inbox = []
        io = _CaptureIO("init", inbox, [])
        d = VA.VaraStationHandshake(["MYCALL"], io, mfsk_only=False)
        d.caller = "MYCALL"
        d._tx_link_setup()
        self.assertTrue(inbox, "link-setup must be transmitted")
        self.assertGreater(len(inbox[-1]), 4 * 48000, "a ~4.4 s wideband over")


if __name__ == "__main__":
    unittest.main(verbosity=2)
