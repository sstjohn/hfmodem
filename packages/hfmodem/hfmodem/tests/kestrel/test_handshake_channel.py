# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The connect handshake through a channel, rather than through a wire.

`test_vara_connect.test_two_kestrels_complete_the_connect_handshake` hands each
burst to the peer as a pristine array starting at sample 0. That is the shape of
test that let two on-air defects survive: an acceptance threshold calibrated on
clean loopback, and a receiver that demodulated from wherever an envelope detector
thought a burst began. Both are invisible when the burst arrives perfect and
aligned, and both stopped kestrel completing a connect on the air.

So this runs the same exchange with the two things a real path adds:

  * **finite SNR** — fading costs tones, which is what broke the 0.8 threshold; and
  * **padding** — the burst sits somewhere inside a longer window of noise rather
    than starting at sample 0, which is what broke demodulate-from-zero.

Measured waterfall (12 seeds per point): 12/12 down to -15 dB, 1/12 at -18 dB.
Recorded on-air SN across eight gateway sessions ran -10.6 to +5.9 dB, so the
handshake carries roughly 5 dB of margin at the worst conditions observed.

The assertion sits at -6 dB — well inside the flat region, so it is a regression
test and not a coin flip.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara.vara_arq import VaraState, VaraStationHandshake

_FS = 48000
_ASSERT_SNR_DB = -6.0        # deep in the flat part of the waterfall
_PAD_S = 0.35


class _IO:
    def __init__(self, sink: list) -> None:
        self.sink = sink
        self.conn = None

    def key(self, on: bool) -> None: ...
    def tx(self, samples) -> None: self.sink.append(np.asarray(samples, float))
    def log(self, msg: str) -> None: ...
    def pending(self) -> None: ...
    def connected(self, caller, called, bw) -> None: self.conn = (caller, called, bw)


def _channel(burst: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Embed the burst in noise, at a finite SNR — what a receiver actually gets."""
    pad = int(_PAD_S * _FS)
    sig = np.concatenate([np.zeros(pad), burst, np.zeros(pad)])
    noise_pw = (burst ** 2).mean() / (10 ** (snr_db / 10))
    return sig + rng.normal(0.0, np.sqrt(noise_pw), len(sig))


def _exchange(snr_db: float, seed: int) -> tuple[bool, bool]:
    rng = np.random.default_rng(seed)
    to_b: list = []
    to_a: list = []
    a = VaraStationHandshake(["W9SSJ"], _IO(to_b), bw="2300")
    b = VaraStationHandshake(["W1AW"], _IO(to_a), bw="2300")
    b.listen(True)
    a.originate("W1AW")
    for _ in range(8):
        while to_b:
            b.on_rx_audio(_channel(to_b.pop(0), snr_db, rng))
        while to_a:
            a.on_rx_audio(_channel(to_a.pop(0), snr_db, rng))
    return a.state == VaraState.CONNECTED, b.state == VaraState.CONNECTED


@pytest.mark.parametrize("seed", range(4))
def test_handshake_completes_through_a_noisy_padded_channel(seed):
    ini, resp = _exchange(_ASSERT_SNR_DB, seed)
    assert ini, f"initiator did not reach CONNECTED at {_ASSERT_SNR_DB} dB SNR"
    assert resp, f"responder did not reach CONNECTED at {_ASSERT_SNR_DB} dB SNR"


def test_padding_alone_does_not_break_it():
    """Isolates the alignment half: no noise, but the burst does not start at
    sample 0. This is the case that demodulate-from-zero could not handle."""
    ini, resp = _exchange(60.0, 0)
    assert ini and resp, "handshake fails on padding alone, with no noise at all"


def test_the_channel_helper_is_actually_degrading():
    """Guard against the test quietly becoming the pristine one it replaced."""
    rng = np.random.default_rng(0)
    burst = rng.normal(0, 1, 4096)
    out = _channel(burst, _ASSERT_SNR_DB, rng)
    assert len(out) > len(burst) + _FS // 2, "padding is not being applied"
    noise_only = out[:int(_PAD_S * _FS) - 100]
    assert noise_only.std() > 0.05, "no noise is being added"
