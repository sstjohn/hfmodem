# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""TX self-loop acceptance: kestrel-TX audio -> proven RX -> byte-exact.

Renders each payload to audio with `kestrel/tx/varahf500_tx.py`, then decodes it
back with the frozen receiver `kestrel/rx/varahf500.py`, and asserts the payload
comes through byte-exact with the constant frame offset c0 = 9. This is the
NECESSARY condition on the transmitter (a wrong TX cannot round-trip through the
independently-validated RX). Cross-payload (counter / PRBS15 / random) guards
against payload-structure fitting.

Stronger waveform evidence is in ``test_bw500_tx_native.py``: held-out native
stock recordings are compared against the actual renderer after only timing
and carrier gain/phase alignment. Those offline checks catch pulse and subband
stagger errors that a self-loop can tolerate.
"""
import os

import pytest

from hfmodem.kestrel.rx import varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx


def _prbs15(n=256):
    state, mask = (1 << 15) - 1, (1 << 15) - 1   # taps at bits 15, 14
    out = bytearray(n)
    for i in range(n):
        b = 0
        for _ in range(8):
            fb = ((state >> 14) & 1) ^ ((state >> 13) & 1)
            b = ((b << 1) | fb) & 0xFF
            state = ((state << 1) | fb) & mask
        out[i] = b
    return bytes(out)


def _frames(payload):
    fr = []
    for i in range(6):
        chunk = payload[i * 43:(i + 1) * 43]
        chunk = chunk + bytes(43 - len(chunk))            # pad last frame to 43
        marker = (0x99 - 4 * i) & 0xFF if i < 5 else 0x82  # cosmetic; RX checks CRC only
        fr.append(tx.build_frame(chunk, marker))
    return fr


def _selfloop(payload):
    onset, out, c0s, oks = 3000, b"", [], []
    for fb in _frames(payload):
        audio = tx.synth_burst(fb, onset=onset)
        # timing search (a real RX recovers burst timing; here we sweep a symbol)
        best = None
        for st in range(onset - 460, onset - 40, 40):
            r = rx.decode_burst(audio, start=st)
            if r.crc_ok:
                best = r
                break
            best = best or r
        out += best.frame_bytes[:43]
        c0s.append(best.c0)
        oks.append(best.crc_ok)
    return out[:256], c0s, all(oks)


@pytest.mark.parametrize("name,payload", [
    ("counter", bytes(range(256))),
    ("prbs15", _prbs15(256)),
    ("random", os.urandom(256)),
])
def test_tx_selfloop_byte_exact(name, payload):
    decoded, c0s, all_crc = _selfloop(payload)
    assert all_crc, f"{name}: not all frames CRC-passed"
    assert all(c == 9 for c in c0s), f"{name}: c0 not constant 9: {c0s}"
    assert decoded == payload, f"{name}: not byte-exact"
