# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""VARA BW500 control-token codec — the responder/initiator ACK & keepalive bursts.

VARA's control channel is a small vocabulary of **fixed tokens** (spec 02 §2.6),
not parametric frames: the data-ACK is one constant waveform (no sequence number —
stop-and-wait, §5.4), and connected-ack / keepalive share a template differing only
in a repetition-coded flag. Each token is **raw differential BPSK on the 1350 Hz
sub-band**, 512-sample columns.

This module synthesises those tokens (TX) and recognises them in received audio
(RX, by matched correlation). It is what lets kestrel's ARQ emit acknowledgements a
real VARA decodes, and read VARA's — the piece its own invented control frames
(``kestrel/arq/frames``) stood in for before these tokens were promoted to spec
facts.

The BW500 bit strings are promoted spec facts (spec 02 §2.6), read from the
2026-07-13 loopback corpus and confirmed against it: more than forty data-ACKs
at Hamming 0 off one logged session. Nothing here reads VARA internals.

**Nothing here is BW2300's, and a table here used to be.** What that table said
a BW2300 control burst is, no real VARA has ever been recorded keying: on the one BW2300
recording held with a real VARA at both ends it matched none of the thirteen
short control bursts, and a bit search over both whole sides at sixteen column
phases matched none of them either. The reason is that 44 DBPSK columns of 512
samples and 11 two-tone MFSK symbols of four columns each are the same 22528
samples read two ways, and §4.2C's reading is the measured one: every short
frame a real VARA keys at BW2300 is that shape, and ``vara_arq._ack_plateau``
is what reads it. So this module is BW500's, whole, and says nothing about the
wide bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, firwin

FS = 48000
H = 512                 # 512-sample column (two 256-sample half-symbols)
CARRIER = 1350.0        # the BW500 DBPSK sub-band; every BW500 token rides it
_RAMP = 64              # raised-cosine key-up/down, samples
_LP = firwin(801, 120 / (FS / 2))

# Canonical token patterns (spec 02 §2.6). Every BW500 token rides 1350 Hz;
# data-ACK is constant, and connected-ack / alive share a template and differ only
# in the 4-bit repetition-coded tail flag.
_BODY = "1111111111111110000011110000111100000000"
TOKENS = {
    "data-ack":      "01110111111101110000100001111000",
    "nak":           "01111000111101111000100010001111",
    "connected-ack": _BODY + "0000",
    "alive":         _BODY + "1111",
    "ready":         "01110111011101110111111110000000011111110000",
}


def _bits_to_symbols(bits: str) -> np.ndarray:
    """Differential bits -> unit symbols. bit=1 flips phase by π, bit=0 holds."""
    phase = 0.0
    syms = [1.0 + 0j]
    for b in bits:
        phase += np.pi if b == "1" else 0.0
        syms.append(np.exp(1j * phase))
    return np.asarray(syms)


def synth_token(name: str, amplitude: float = 0.5) -> np.ndarray:
    """Render a control token to real 48 kHz audio (DBPSK on :data:`CARRIER`)."""
    syms = _bits_to_symbols(TOKENS[name])
    n = len(syms) * H
    t = np.arange(n)
    baseband = syms[t // H]
    sig = (baseband * np.exp(2j * np.pi * CARRIER * t / FS)).real
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(_RAMP) / _RAMP))
    sig[:_RAMP] *= ramp
    sig[-_RAMP:] *= ramp[::-1]
    return sig * (amplitude / np.abs(sig).max())


def _baseband(audio: np.ndarray, carrier: float) -> np.ndarray:
    """Heterodyne to DC and low-pass. The 801-tap convolution dominates control
    demod, and every BW500 token shares one carrier, so callers hoist this out of
    their token loop rather than paying for it per token."""
    nn = np.arange(len(audio))
    return fftconvolve(audio * np.exp(-2j * np.pi * carrier * nn / FS), _LP, "same")


def _demod_bits_bb(bb: np.ndarray, ncol: int
                   ) -> tuple[np.ndarray | None, float, int]:
    """Onset-locked differential demod of an already-heterodyned burst.

    The onset search is over all H column phases at once: within one burst the
    usable column count takes at most two values, so onsets group into at most two
    rectangular gathers and the whole sweep is a couple of array ops instead of H
    Python iterations.
    """
    nbb = len(bb)
    onsets = np.arange(H)
    counts = np.minimum(ncol, np.maximum(0, -(-(nbb - onsets) // H)))
    q_all = np.full(H, -1.0)
    d_keep: dict[int, np.ndarray] = {}
    for n in np.unique(counts):
        if n < ncol * 0.8 or n < 2:
            continue
        sel = np.flatnonzero(counts == n)
        idx = sel[:, None] + np.arange(n)[None, :] * H          # (G, n)
        seg = bb[idx]
        d = seg[:, 1:] * np.conj(seg[:, :-1])                   # (G, n-1)
        ad = np.abs(d)
        m = ad > 0.3 * np.median(ad, axis=1, keepdims=True)
        cnt = m.sum(axis=1)
        ok = cnt >= 6
        if not ok.any():
            continue
        u = np.exp(2j * np.angle(d))
        q = np.abs((u * m).sum(axis=1) / np.maximum(cnt, 1))
        q_all[sel[ok]] = q[ok]
        for r, o in enumerate(sel):
            if ok[r]:
                d_keep[int(o)] = d[r]
    best_onset = int(np.argmax(q_all))
    if q_all[best_onset] < 0:
        return None, -1.0, 0
    d = d_keep[best_onset]
    return (d.real < 0).astype(int), float(q_all[best_onset]), best_onset


def _demod_bits(audio: np.ndarray, ncol: int,
                carrier: float = CARRIER) -> tuple[np.ndarray | None, float, int]:
    """Raw-differential-BPSK bits over ``ncol`` columns, timing locked to max collapse."""
    return _demod_bits_bb(_baseband(audio, carrier), ncol)


@dataclass
class TokenMatch:
    name: str
    hamming: int
    quality: float          # DBPSK collapse at the locked onset (>~0.8 = clean)


def detect_token(audio: np.ndarray, max_hamming: int = 2) -> TokenMatch | None:
    """Recognise a control token in one burst of audio, or None.

    Demodulates at each token's length and picks the nearest canonical pattern;
    accepts it only if within ``max_hamming`` bits (allowing a
    1-column rotation, which a best-onset lock can introduce). The default 2 is
    safely below the token separations, so they never cross-classify; real clean
    tokens land at Hamming 0–1.
    """
    best: TokenMatch | None = None
    bb = _baseband(audio, CARRIER)
    for name, patt in TOKENS.items():
        ref = np.array([int(c) for c in patt])
        bits, q, _ = _demod_bits_bb(bb, len(ref) + 1)
        if bits is None:
            continue
        for shift in (0, 1, -1):
            b = np.roll(bits, shift)[:len(ref)] if len(bits) >= len(ref) else None
            if b is None or len(b) != len(ref):
                continue
            hd = int((b != ref).sum())
            hd = min(hd, len(ref) - hd)      # DBPSK sign ambiguity: whole-stream invert
            if hd <= max_hamming and (best is None or hd < best.hamming):
                best = TokenMatch(name=name, hamming=hd, quality=q)
    return best
