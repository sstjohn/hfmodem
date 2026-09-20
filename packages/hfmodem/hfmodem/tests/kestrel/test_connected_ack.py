# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Step 5 must be falsifiable: only a genuine connected-ack completes the connect.

This branch previously set CONNECTED on *any* audio arriving in
``_I_LINKSETUP_SENT`` — ten zero samples, an empty array, a carrier from someone
tuning up, or our own link-setup tail handed back late by the burst segmenter. On
a live band the segmenter supplies such a trigger almost immediately, so an
on-air connect attempt could not fail: ``kestrel_connect`` would print CONNECTED
carrying no evidence the gateway had answered.

An experiment that cannot fail is worse than one that fails, so each spoof below
is a case that used to succeed.

The ack carries no callsign — it is eleven two-tone symbols whose first four are
a fixed preamble and whose last seven are session state [spec 04 §4.2C] — so what
is checked is that preamble, and the wrong-station cases below are the other
handshake bursts, which do name a station and are not it.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import _I_LINKSETUP_SENT, VaraState, VaraStationHandshake

_CALLED = "NS0A"


class _IO:
    def key(self, on): ...
    def tx(self, samples): ...
    def log(self, msg): ...
    def pending(self): ...
    def connected(self, *a): ...


def _awaiting_ack() -> VaraStationHandshake:
    hs = VaraStationHandshake(["W9SSJ"], _IO(), bw="2300")
    hs.originate(_CALLED, "W9SSJ")
    hs.step = _I_LINKSETUP_SENT
    return hs


def _spoofs():
    rng = np.random.default_rng(0)
    t = np.arange(48000) / 48000
    # The ack's preamble pairs read as single tones — what a station that got the
    # waveform wrong would emit, and what the old model of this burst thought it was.
    half = MK.synth_tones([lo for lo, _ in VF.CONNECTED_ACK_2300])
    return [
        ("ten zero samples", np.zeros(10)),
        ("empty array", np.zeros(0)),
        ("white noise", rng.standard_normal(48000)),
        ("carrier from someone tuning up", 0.5 * np.sin(2 * np.pi * 1000 * t)),
        ("a connect-response, not an ack", MK.synth_burst(_CALLED, VF.CONNECT_RESPONSE)),
        ("a keepalive, not an ack", MK.synth_burst(_CALLED, VF.SESSION_KEEPALIVE_A)),
        ("one tone per symbol instead of two", half),
    ]


@pytest.mark.parametrize("label,audio", _spoofs(), ids=[s[0] for s in _spoofs()])
def test_spoof_does_not_complete_the_connect(label, audio):
    hs = _awaiting_ack()
    hs.on_rx_audio(audio)
    assert hs.state is not VaraState.CONNECTED, (
        f"{label} completed the connect — step 5 accepts anything again, and an "
        "on-air run would report a connect that never happened")


def test_the_genuine_ack_does_complete_it():
    """Guard against the obvious over-correction: rejecting everything."""
    hs = _awaiting_ack()
    hs.on_rx_audio(MK.synth_tone_pairs(VF.CONNECTED_ACK_2300))
    assert hs.state is VaraState.CONNECTED, "genuine connected-ack was rejected"


def test_the_preamble_is_what_decides_not_the_state_symbols():
    """The seven symbols behind the preamble vary per session and per bandwidth
    [spec 04 §4.2C], so an ack carrying state we have never seen must still be
    recognised — otherwise kestrel connects to the one gateway whose frame we
    captured and to nobody else."""
    rng = np.random.default_rng(4)
    alphabet = sorted(VF.TONE_ALPHABET)
    for _ in range(20):
        state = [tuple(sorted(rng.choice(alphabet, 2, replace=False))) for _ in range(7)]
        hs = _awaiting_ack()
        hs.on_rx_audio(MK.synth_tone_pairs(list(VF.CONNECTED_ACK_PREAMBLE) + state))
        assert hs.state is VaraState.CONNECTED, f"ack with state {state} was rejected"
