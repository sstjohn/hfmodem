# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The connect handshake, driven by real gateway audio through the real state machine.

This is the on-air test minus the transmitter: a recorded VARA gateway answer is fed
to :class:`VaraStationHandshake` exactly as ``kestrel_connect`` feeds it live, and the
handshake must advance. It exists because kestrel had never completed an on-air
connect and every other handshake test used synthesised or loopback audio, which hides
both of the defects it now covers.

Two things had to be fixed for this to pass, and either alone leaves it failing:

  * The acceptance threshold was calibrated on clean loopback, where a correct match
    scores N/N. On a real channel fading costs tones — a genuine NS0A response scores
    11/15 — and the 0.8 cut rejected it by one tone. See ``test_handshake_offair``.
  * The receiver demodulated from wherever the audio-envelope segmenter decided a
    burst began. On real audio the segmenter handed over a 0.11 s fragment 176 ms
    before the actual response; demodulating that from sample 0 matches 0/8 preamble
    tones. The burst is now *located* by its fixed preamble instead.

Both failure modes report identically in the log — "did NOT match dialled call" — which
reads as the gateway never answering, and is why this went unnoticed across eight
on-air sessions.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake

_FS = 48000


class IO:
    """Records what the handshake did, and never transmits anything."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.connected_as = None
        self.keyed = 0

    def key(self, on: bool) -> None:
        self.keyed += int(bool(on))

    def tx(self, *a, **k) -> None: ...

    def log(self, msg: str) -> None:
        self.lines.append(msg)

    def pending(self) -> None:
        self.lines.append("PENDING")

    def connected(self, caller, called, bw) -> None:
        self.connected_as = (caller, called, bw)


def _window(session: str, t0: float, t1: float) -> np.ndarray:
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    from scipy.io import wavfile
    fs, x = wavfile.read(str(path))
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    x = x / (np.abs(x).max() or 1.0)
    return x[int(t0 * fs):int(t1 * fs)]


def test_real_gateway_response_advances_the_handshake():
    """The whole point: a real NS0A connect-response, off air, must be accepted and
    must move the initiator on to sending its link-setup."""
    audio = _window("NS0A_2300", 9.0, 11.5)
    io = IO()
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.originate("NS0A", "W9SSJ")
    hs.on_rx_audio(audio)

    joined = " | ".join(io.lines)
    assert "did NOT match dialled call" not in joined, (
        f"genuine gateway answer still rejected: {joined}")
    assert any("confirmed for NS0A" in m for m in io.lines), (
        f"connect-response not confirmed: {joined}")
    assert any("link-setup" in m for m in io.lines), (
        f"handshake did not advance to link-setup: {joined}")


def test_the_burst_is_found_by_preamble_not_by_envelope():
    """Pins the mechanism: the response sits ~176 ms inside the window handed over,
    so a receiver that demodulates from sample 0 cannot see it."""
    audio = _window("NS0A_2300", 9.0, 11.5)
    io = IO()
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.originate("NS0A", "W9SSJ")
    hs.on_rx_audio(audio)
    locked = [m for m in io.lines if "preamble locked at" in m]
    assert locked, f"no preamble lock happened: {io.lines}"
    at = int(locked[0].split("+")[1].split()[0])
    assert at > 10_000, (
        f"lock offset {at} is near zero — this test is no longer exercising the case "
        "it was written for")
