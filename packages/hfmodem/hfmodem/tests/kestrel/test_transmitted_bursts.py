# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the handshake puts on the air: whose call it names, and how loud it is.

Nothing in the suite looked at either. Every other handshake test asks only
whether a peer running this same code recognises what we sent, and two defects
walked through a mutation audit on that blind spot:

  * the step-4 link-setup naming the CALLED station instead of us. It decodes, its
    CRC is good, and a gateway answers the station it names — which is not the one
    transmitting. The kestrel<->kestrel tests cannot see it, because a kestrel
    responder leaves the caller UNKNOWN until the OFDM link-setup RX lands
    [spec 05 §5.3.2], so nothing on the far side ever reads the field.
  * every burst emitted 40 dB down. We key the rig, put out near-silence, hear no
    answer, and report it exactly as we report a gateway that never replied.

Levels, measured on the audio captured here: the MFSK handshake bursts peak at 0.5
(-9.0 dBFS rms) and the wideband link-setup at 1.0 (-3.0 dBFS rms). Full scale is
1.0 — tools/vara_rig_bridge.py hands what it is given straight to the sound card
and calls |x| > 0.99 clipped — so these are absolute levels, not relative ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import VaraIO, VaraState, VaraStationHandshake

# A burst quieter than this is not driving a transmitter that was set up on the
# levels above: 11 dB below the quietest burst the modem emits, and 29 dB above a
# 40 dB attenuation of it.
_MIN_RMS = 0.1                                   # -20 dBFS
_FULL_SCALE = 1.0
_WIDEBAND_MIN = int(2.5 * MK.FS)                 # the 4.4 s OFDM over vs the 1.7 s CR


class _Capture(VaraIO):
    """Keeps every burst handed to the transmitter."""

    def __init__(self, inbox: list | None = None):
        self.bursts: list[np.ndarray] = []
        self.inbox = inbox

    def key(self, on): pass

    def tx(self, s):
        s = np.asarray(s, float)
        self.bursts.append(s)
        if self.inbox is not None:
            self.inbox.append(s)

    def pending(self): pass

    def connected(self, caller, called, bw): pass

    def log(self, m): pass


def _initiator_through_link_setup(mycall: str, gateway: str = "W1AW"):
    """An initiator driven to step 4: CR out, gateway's answer in, link-setup out."""
    io = _Capture()
    station = VaraStationHandshake([mycall], io, bw="2300")
    station.originate(gateway)
    station.on_rx_audio(MK.synth_burst(gateway, VF.CONNECT_RESPONSE))
    assert len(io.bursts) == 2, (
        f"expected the CR and the link-setup, got {len(io.bursts)} bursts")
    return io


@pytest.mark.parametrize("mycall", ["W9SSJ", "W9SSJ-5"])
def test_the_link_setup_we_transmit_names_us(mycall):
    io = _initiator_through_link_setup(mycall, gateway="W1AW")
    got = rx.decode_burst(io.bursts[1], rx.BASE_LEVEL)
    assert got is not None and got.crc_ok, "the link-setup we transmitted will not decode"
    caller = VF.caller_from_link_setup(bytes(got.frame_bytes))
    assert caller == mycall, (
        f"we transmitted a link-setup naming {caller}. A gateway takes the caller "
        f"identity from this frame, so it has to be MYCALL ({mycall}) — naming the "
        "station we called makes it answer someone who is not on the air")


def _stream(station, audio):
    """Hand the peer's audio over the way the receiver hands it over."""
    for at in range(0, len(audio), VA._STREAM_BLOCK):
        station.on_rx_stream(audio[at:at + VA._STREAM_BLOCK])


def _peer_over_idle(station):
    """One `session-responder-over-idle` on this session's alphabet, with the lead
    behind it that the answer is keyed into."""
    _stream(station, np.concatenate([
        MK.synth_burst(station.called,
                       VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, station.bw)),
        np.zeros(int(0.13 * MK.FS))]))


def test_every_transmitted_burst_is_at_drive_level():
    """Sweep one whole session's transmit audio, both stations, every burst kind."""
    a_in: list[np.ndarray] = []
    b_in: list[np.ndarray] = []
    A = VaraStationHandshake(["W9SSJ"], _Capture(b_in), bw="2300")
    B = VaraStationHandshake(["W1AW"], _Capture(a_in), bw="2300")
    B.listen(True)
    A.originate("W1AW")
    for _ in range(6):
        while b_in:
            B.on_rx_audio(b_in.pop(0))
        while a_in:
            A.on_rx_audio(a_in.pop(0))
    assert A.state == VaraState.CONNECTED and B.state == VaraState.CONNECTED
    # The keepalives, keyed the way this station keys them: in the peer's turn its
    # own clock keys nothing at all, so each one is drawn by a 745 over-idle of the
    # peer's and goes out in the gap behind it  [see _answer_peer_idle].
    assert A.turn == VA._TURN_PEER
    _peer_over_idle(A)
    _peer_over_idle(A)
    # A gateway DATA over, answered. It has to be the real waveform: silence of the
    # right length no longer keys anything (kestrel/tests/kestrel/test_data_over_gate.py),
    # and without a genuine one the per-over response never reaches this sweep.
    _stream(A, np.concatenate([
        tx.synth_burst(b"x" * rx.payload_bytes(rx.BASE_LEVEL), over=0),
        np.zeros(int(0.3 * MK.FS))]))

    bursts = A.io.bursts + B.io.bursts
    wide = [b for b in bursts if len(b) >= _WIDEBAND_MIN]
    # CR, session-confirm, two keepalive-A and the per-over response from A; the
    # connect-response, the connected-ack and its repeat from B; A's link-setup is
    # the one wideband burst. An exact count, so a burst kind dropping out of the
    # sweep fails here rather than quietly going unmeasured.
    assert len(wide) == 1 and len(bursts) - len(wide) == 8, (
        f"the session covered {len(bursts) - len(wide)} MFSK and {len(wide)} "
        "wideband bursts, not the 8 and 1 this sweep is written over")
    for b in bursts:
        rms = float(np.sqrt((b * b).mean()))
        peak = float(np.abs(b).max())
        assert rms >= _MIN_RMS, (
            f"a {len(b) / MK.FS:.2f} s burst went out at {20 * np.log10(rms):.1f} "
            "dBFS rms. The rig keys and emits nothing usable, which the log reports "
            "as a peer that did not answer")
        assert peak <= _FULL_SCALE, (
            f"a {len(b) / MK.FS:.2f} s burst peaks at {peak:.3f}, past full scale — "
            "the sound card clips it and the transmitter splatters")


@pytest.mark.parametrize("form,n_sym", [("full", 41), ("stock", 32)])
def test_the_re_keyed_request_goes_out_in_the_form_asked_for(form, n_sym):
    """A stock 4.9.0 caller keys 41 symbols once and 32 every time after — one
    preamble tone in front of the same 31 payload tones — and this station keys 41
    every time, as the 2026-09-09 VARA HF 4.9.0 lattice bench measured against a
    stock caller keyed the same way. Whichever form goes out, the tones the
    gateway is keyed to are the same ones."""
    io = _Capture()
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300", cr_retry_form=form)
    hs.originate("W1AW")
    hs.originate("W1AW", retry=True)
    first, again = io.bursts
    assert MK.demod_burst(first, VF.CR) == VF.handshake_tones("W1AW", VF.CR)
    kind = VF.connect_request_retry("2300") if form == "stock" else VF.CR
    assert len(kind.preamble) + kind.n_payload == n_sym
    assert MK.demod_tones(again, n_sym) == VF.handshake_tones("W1AW", kind)
    assert VF.payload_bins("W1AW", kind) == VF.payload_bins("W1AW", VF.CR)
