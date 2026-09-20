# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel reads a real gateway's mail — the BW2300 receive chain, wired end to end.

Two live Winlink RMS sessions recorded at this station carry independent ground
truth: ``payload.bin`` is the plaintext a real VARA delivered on its host port
during the same session ``rig_rx.wav`` recorded off air. The tests here hold the
whole receive path — segmenter bracket, payload-blind alignment, turbo decode,
CRC, VARA over framing, duplicate suppression, host delivery — to *byte
equality* against that output, through the same ``on_rx_audio`` entry point a
live session drives.

A round trip through our own transmitter proves only self-consistency (an
encoder/decoder pair can cancel a shared error and stay green), so the arbiter
throughout is the recorded gateway audio; the self-loop tests live elsewhere
(``test_bw2300_roundtrip``). Each planted counterexample below corrupts one
stage of the receive chain on the *real* audio and must go red — a gate that
cannot fail is not measuring anything.

False-accept exposure of the acceptance rule (delivering an over to the host):
  1. payload-blind reference-column guard, >= 16 of 24. Measured populations
     (``test_data_over_gate``): real overs 20-24, everything else in 246 s of
     off-air HF <= 8, synthetic noise <= 9 over 400 blobs — no overlap observed.
  2. turbo decode whose CRC-16/GENIBUS passes twice on identical bits. A single
     16-bit check accepts ~2**-16 of noise candidates per decode (~1/4400 per
     multi-candidate scan, measured project-wide); the double-pass rule is
     measured ~12x stricter than checking once per iteration (``coding.turbo``).
  3. duplicate suppression cannot create a false accept (it only drops).
The gates are serial: noise pays for a decode only after clearing a guard it
has never been observed to clear, so the per-bracket exposure is bounded by
(P(guard >= 16 on noise) x 2**-16) — the first factor unobserved in every
measured population, the second ~1.5e-5. A link-setup accept further requires
the frame's three fixed structure bytes (an extra 2**-24 against random bytes).

Measured rather than argued, on the link-setup rule: 232 five-second windows of
real off-air HF at 1 s steps yield exactly one accept, and it is the genuine
one (see ``test_link_setup_rx_reads_the_off_air_connect_and_nothing_else``).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF
from hfmodem.tests.kestrel import corpora

FS = rx.FS
_MYCALL = "W9SSJ"
_SESSIONS = (("KC9GHZ_2300", "KC9GHZ"), ("NS0A_2300", "NS0A"))


class _IO(VA.VaraIO):
    def __init__(self):
        self.datas: list[bytes] = []
        self.keys = 0
        self.conn = None
        self.lines: list[str] = []

    def key(self, on):
        self.keys += bool(on)

    def tx(self, samples): ...

    def pending(self): ...

    def connected(self, caller, called, bw):
        self.conn = (caller, called, bw)

    def data(self, payload):
        self.datas.append(bytes(payload))

    def log(self, msg):
        self.lines.append(msg)


def _connected(called: str) -> tuple[VA.VaraStationHandshake, _IO]:
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", called, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


@lru_cache(maxsize=4)
def _session(name: str):
    """(normalised audio, detected over brackets, host-port ground truth)."""
    base = corpora.OFFAIR / name
    if not (base / "rig_rx.wav").exists():
        pytest.skip(f"off-air recording for {name} not present")
    x = corpora.wav_mono(base / "rig_rx.wav")
    x = x / (np.abs(x).max() or 1.0)
    return x, tuple(rx.detect_overs(x)), (base / "payload.bin").read_bytes()


def _on_air(x, pad: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(int(pad * FS)), np.asarray(x, float),
                        np.zeros(int(pad * FS))])
    return y + rng.standard_normal(len(y)) * 1e-4


# --------------------------------------------------------------------------- #
# The arbiter: real gateway audio against the gateway's own host-port output.
@pytest.mark.parametrize("session,called", _SESSIONS)
def test_real_gateway_mail_reassembles_byte_exact(session, called):
    """Every DATA over of a live session decodes CRC-clean and the extracted
    payloads join to exactly the bytes the gateway's own modem delivered."""
    x, overs, truth = _session(session)
    assert overs, "the recording should hold wideband overs"
    got = b""
    for s, e in overs:
        fr = rx.decode_over(x, s, e)
        assert fr.crc_ok, f"the over at {s / FS:.1f} s of {session} did not decode"
        got += phy.vara_payload(fr.payload)
    assert got == truth, f"{session}: reassembled {len(got)} bytes != host-port output"


@pytest.mark.parametrize("session,called", _SESSIONS)
def test_live_fsm_delivers_the_mail(session, called):
    """The same recordings through the live path — ``on_rx_audio`` with the
    segmenter's brackets — must answer every over and hand the host the same
    bytes. This is the wiring test: decode results must reach ``VaraIO.data``,
    not stop at the keying decision."""
    x, overs, truth = _session(session)
    hs, io = _connected(called)
    for s, e in overs:
        hs.on_rx_audio(x[max(0, s - FS // 4):e + FS // 4])
    assert io.keys == len(overs), "an over went unanswered"
    assert b"".join(io.datas) == truth, (
        f"{session}: host received {sum(map(len, io.datas))} bytes, "
        f"expected {len(truth)}")


# --------------------------------------------------------------------------- #
# Planted counterexamples, on the real audio: corrupt one stage, watch it go red.
def test_wrong_whitener_reads_nothing():
    """De-whitening with an inverted PN must fail the CRC on a real over — if it
    still passes, the CRC gate is not connected to the whitener."""
    x, overs, _ = _session(_SESSIONS[0][0])
    s, e = overs[0]
    assert rx.decode_over(x, s, e).crc_ok, "sanity: the over decodes unmodified"
    orig = rx._PN
    try:
        rx._PN = orig ^ 1
        assert not rx.decode_over(x, s, e).crc_ok, (
            "a decode with the WRONG whitener PN passed CRC — the gate can pass "
            "without looking")
    finally:
        rx._PN = orig


def test_wrong_channel_interleave_reads_nothing():
    """A channel de-interleave one position out of phase must fail the CRC on a
    real over."""
    x, overs, _ = _session(_SESSIONS[0][0])
    s, e = overs[0]
    assert rx.decode_over(x, s, e).crc_ok, "sanity: the over decodes unmodified"
    orig = rx.chan_perm
    try:
        rx.chan_perm = lambda level: np.roll(orig(level), 1)
        assert not rx.decode_over(x, s, e).crc_ok, (
            "a decode with a shifted channel interleaver passed CRC")
    finally:
        rx.chan_perm = orig


# --------------------------------------------------------------------------- #
# Link-setup RX: the burst that names the caller  [spec 04 §4.2A; spec 05 §5.3.2].
@corpora.requires_bw2300_capture
def test_link_setup_rx_reads_a_real_vara_burst():
    """The staged capture's first segment is a real VARA link-setup; the decoded
    caller must be the transmitting station's callsign."""
    x = corpora.wav_mono(corpora.BW2300_CAPTURE)
    x = x / np.abs(x).max()
    s, e = rx.detect_overs(x)[0]
    assert OF.link_setup_rx(x[s:e]) == "AAAA1"


def test_link_setup_rx_reads_the_off_air_connect_and_nothing_else():
    """The off-air arbiter, and the acceptance rule's exposure, in one sweep.

    Every 5 s window at 1 s steps across the two recorded gateway sessions and
    the verified clear channel — 232 windows of real HF through this station's
    own rig. Exactly one accepts: t = 13.0 s of the NS0A session, where a real
    VARA opened the session by transmitting its link-setup, and the caller it
    decodes to is this station's own callsign. Nothing else in the material,
    the five gateway DATA overs included, is accepted as a link-setup.
    """
    seen, scanned = [], 0
    for path in (corpora.OFFAIR / "KC9GHZ_2300" / "rig_rx.wav",
                 corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav",
                 corpora.CLEAR_CHANNEL):
        if not path.exists():
            continue
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        win = 5 * FS
        for i in range(0, max(0, len(x) - win), FS):
            scanned += 1
            caller = OF.link_setup_rx(x[i:i + win], tries=2)
            if caller is not None:
                seen.append((path.parent.name, round(i / FS, 1), caller))
    if not scanned:
        pytest.skip("off-air gateway recordings not present")
    assert seen == [("NS0A_2300", 13.0, _MYCALL)], (
        f"{scanned} windows scanned, accepted {seen}")


def test_link_setup_rx_refuses_a_data_over():
    """A CRC-clean DATA over is not a link-setup: the structure gate, not the
    CRC, is what says so."""
    body = phy.vara_body(b"not a connect", _MYCALL)
    assert OF.link_setup_rx(_on_air(tx.synth_burst(body, over=0))) is None


def test_link_setup_rx_refuses_noise():
    rng = np.random.default_rng(7)
    for _ in range(3):
        assert OF.link_setup_rx(rng.standard_normal(int(4.6 * FS))) is None


# --------------------------------------------------------------------------- #
# The responder learns the caller before it acks  [spec 05 §5.3 steps 2, 4, 5].
def _pump(init, resp, i_in, r_in, rounds=50):
    for _ in range(rounds):
        if r_in:
            resp.on_rx_audio(r_in.pop(0))
        elif i_in:
            init.on_rx_audio(i_in.pop(0))
        else:
            return


def _pair():
    i_in, r_in = [], []
    io_i, io_r = _IO(), _IO()
    io_i.tx = lambda x: r_in.append(np.asarray(x, float))
    io_r.tx = lambda x: i_in.append(np.asarray(x, float))
    init = VA.VaraStationHandshake(["MYCALL"], io_i, mfsk_only=False)
    resp = VA.VaraStationHandshake(["GATE1"], io_r, mfsk_only=False)
    return init, resp, io_i, io_r, i_in, r_in


def test_responder_learns_the_caller_from_the_link_setup():
    """Full kestrel<->kestrel handshake with the real link-setup over: the
    responder must decode it and report the CALLER's callsign — not
    ``CALLER_UNKNOWN`` — and only then ack."""
    init, resp, io_i, io_r, i_in, r_in = _pair()
    resp.listen(True)
    init.originate("GATE1")
    _pump(init, resp, i_in, r_in)
    assert init.state == VA.VaraState.CONNECTED
    assert resp.state == VA.VaraState.CONNECTED
    assert io_r.conn == ("MYCALL", "GATE1", "2300"), (
        f"responder reported {io_r.conn}; the caller must come from the decoded "
        "link-setup")
    assert io_i.conn == ("MYCALL", "GATE1", "2300")


def test_responder_does_not_ack_what_it_cannot_decode():
    """Noise of over length in the link-setup window must leave the responder
    CONNECTING, its ack unsent: acking an unidentified caller reports a
    connection nobody made."""
    init, resp, io_i, io_r, i_in, r_in = _pair()
    resp.listen(True)
    init.originate("GATE1")
    _pump(init, resp, i_in, r_in, rounds=2)         # CR over, response back
    assert resp.step == VA._R_RESP_SENT and resp.state == VA.VaraState.CONNECTING
    resp.on_rx_audio(np.random.default_rng(1).standard_normal(int(4.6 * FS)))
    assert resp.state == VA.VaraState.CONNECTING and io_r.conn is None


def test_responder_answers_a_repeated_cr():
    """An initiator that missed our connect-response repeats its CR; the repeat
    is answered again instead of stalling the connect."""
    init, resp, io_i, io_r, i_in, r_in = _pair()
    resp.listen(True)
    init.originate("GATE1")
    _pump(init, resp, i_in, r_in, rounds=2)
    keys_before = io_r.keys
    resp.on_rx_audio(MK.synth_burst("GATE1", VF.CR))
    assert io_r.keys == keys_before + 1, "the repeated CR went unanswered"
    assert resp.state == VA.VaraState.CONNECTING


# --------------------------------------------------------------------------- #
# Delivery semantics at the host boundary.
def test_a_repeated_over_is_answered_but_delivered_once():
    """A gateway that missed our response repeats the over; the repeat must be
    answered (or the gateway stops) and must NOT reach the host twice (a
    duplicate corrupts the mail byte stream)."""
    hs, io = _connected("KC9GHZ")
    burst = _on_air(tx.synth_burst(phy.vara_body(b"once only\r", _MYCALL), over=1))
    hs.on_rx_audio(burst)
    hs.on_rx_audio(burst)
    assert io.keys == 2, "the repeated over went unanswered"
    assert io.datas == [b"once only\r"], f"delivered {io.datas}"


def test_distinct_overs_are_each_delivered():
    hs, io = _connected("KC9GHZ")
    for i, text in enumerate((b"first over\r", b"second over\r")):
        hs.on_rx_audio(_on_air(tx.synth_burst(phy.vara_body(text, _MYCALL),
                                              over=i + 1), seed=i))
    assert io.datas == [b"first over\r", b"second over\r"]


def test_a_strangers_link_setup_is_not_delivered():
    """The one wideband frame that is not session data: it must neither key nor
    reach the host's data port."""
    hs, io = _connected("KC9GHZ")
    hs.on_rx_audio(_on_air(tx.synth_frame(VF.link_setup_frame("K7ABC"), over=0)))
    assert io.keys == 0 and io.datas == []
