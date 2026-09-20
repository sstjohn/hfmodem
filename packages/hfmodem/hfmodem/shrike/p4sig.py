# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-4 robust-mode signaling, named from its published sequences.

Both granting gateways answer, on the arms where they never grant, in a waveform
this station had no reader for: 1.6-2.3 kHz bursts once a cycle in the answer
slot, a 112.5 Hz carrier comb, `nothing heard` printed under every one. SCS's
own PACTOR-4 description puts those numbers on paper: robust-mode packets are
DQPSK spread to a chip rate of 1800/s by a published 16-chip sequence -- a
16-chip periodicity at 1800 chips/s IS a 112.5 Hz comb -- and every packet opens
on a 19-symbol Chu sequence spread the same way, 304 chips, 168.9 ms, which is
the measured burst length. Correlating the recordings against that construction
is what turned the matched number into an identification: the published
spreading sequence scores where a random one at the same geometry does not,
one Chu root carries 25 of 28 bursts across two gateways and two bands, and
genuine PACTOR-1/2/3 audio through the same pipeline scores at the noise floor.
The knee is 0.28: measured against noise at 0.148-0.186, this station's own
PACTOR-1 at 0.098-0.206, and genuine PACTOR-3 at every speed level at
0.083-0.135, where the two real emissions score 0.263-0.501.

Receive-side only. Nothing here renders a waveform, and a score is a name for a
burst, not a decode: no field, no CRC, no counter. What it buys is a log that
says `the peer left PACTOR-1` instead of spending a retry budget against a
station that is answering every cycle.
"""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window, hilbert, resample_poly

#: The spread symbol rate of every robust- and normal-mode packet [SCS-P4] §6.2.
CHIP_RATE = 1800.0

#: Audio centre of the single-carrier waveform, shared with PACTOR-3.
CENTRE_HZ = 1500.0

CHU_LEN = 19
SPREAD_FACTOR = 16
HEADER_CHIPS = CHU_LEN * SPREAD_FACTOR

#: The published SF-16 spreading sequence, [SCS-P4] §6.2's `scsSpread16`
#: shorts as interleaved re/im over 32767.
_SPREAD16_IQ = (
    31159, -10139, 32319, 5401, 32217, 5976, 29448, 14370,
    32609, -3209, 11128, 30820, -30546, -11858, -12215, -30405,
    32636, 2929, -31477, 9102, 31522, 8948, -27518, 17789,
    32524, -3981, 12148, -30432, -32393, -4939, -986, 32752,
)
SPREAD16 = (np.array(_SPREAD16_IQ[0::2], np.float64)
            + 1j * np.array(_SPREAD16_IQ[1::2], np.float64)) / 32767.0

#: Complex baseband rate the correlations run at: 8 samples per chip exactly.
_BB_FS = CHIP_RATE * 8

#: The naming knee. Measured populations at 48 kHz off the modem's own codec:
#: 28 signaling bursts (two gateways, two bands) score 0.185-0.419 and the same
#: emissions read through the sessions' own 30 listen windows score 0.185-0.393
#: with 24 of 30 at or above the knee; the negatives -- ten PACTOR-1 control
#: bursts, ten matched noise windows, six genuine PACTOR-1/3 reference bursts
#: and the nineteen windows those sessions' reader answered in -- all sit at or
#: below 0.183. A burst under the knee goes unnamed, and over a fifteen-cycle
#: emission that is a missed cycle, never a false name.
SPREAD_KNEE = 0.28


def _baseband(seg: np.ndarray, fs: float = 48000.0) -> np.ndarray:
    z = hilbert(np.asarray(seg, np.float64))
    z *= np.exp(-2j * np.pi * CENTRE_HZ * np.arange(z.size) / fs)
    up, down = int(round(_BB_FS)), int(round(fs))
    g = gcd(up, down)
    return resample_poly(z, up // g, down // g).astype(np.complex64)


def _template(chips: np.ndarray) -> np.ndarray:
    t = np.repeat(chips, 8)
    return (t / np.linalg.norm(t)).astype(np.complex64)


def _spread_template() -> np.ndarray:
    return _template(np.tile(SPREAD16, CHU_LEN))


def _header_template(root: int, shift: int) -> np.ndarray:
    k = np.arange(CHU_LEN)
    chu = np.exp(-1j * np.pi * root * k * (k + 1) / CHU_LEN)
    chips = np.repeat(chu[(k + shift) % CHU_LEN], SPREAD_FACTOR)
    return _template(chips * np.tile(SPREAD16, CHU_LEN))


def _scan(z: np.ndarray, templates: dict, cfos: np.ndarray):
    length = len(next(iter(templates.values())))
    if z.size < length:
        return 0.0, None, 0.0
    n = z.size
    tt = np.arange(n) / _BB_FS
    csum = np.concatenate(([0.0], np.cumsum(np.abs(z) ** 2)))
    energy = np.sqrt(np.maximum(csum[length:] - csum[:-length], 1e-12))
    spectra = {key: np.fft.fft(np.conj(tp[::-1]), 2 * n)
               for key, tp in templates.items()}
    best = (0.0, None, 0.0)
    for cfo in cfos:
        zz = np.fft.fft(z * np.exp(-2j * np.pi * cfo * tt), 2 * n)
        for key, tf in spectra.items():
            c = np.abs(np.fft.ifft(zz * tf))[length - 1:n]
            r = float((c / energy[:c.size]).max())
            if r > best[0]:
                best = (r, key, float(cfo))
    return best


#: The comb screen's naming knee. Measured over the same populations as
#: `SPREAD_KNEE`, burst by burst off the session captures: all 27 raster bursts
#: of the two emissions score 5.11-13.28 with the spacing landing in the 112.5
#: family, while the same sessions' PACTOR-1 bursts, listen windows and matched
#: noise sit at or under 5.00 or pin elsewhere. Genuine PACTOR-3 leaks the odd
#: burst past this screen -- its 120 Hz carrier grid is a comb too -- which is
#: priced in: the screen buys cheapness, and `spread_score` stays the arbiter.
COMB_KNEE = 5.0


def comb_family(spacing: float) -> bool:
    """Whether a measured comb spacing belongs to 112.5 Hz or a subharmonic.

    The cepstral peak of a genuine 112.5 Hz comb lands on 112.5/k as often as
    on 112.5 -- the recorded emissions split between 112.2-113.6 and 28.05-28.22
    -- so the family, not the fundamental alone, is what pins a burst.
    """
    return any(abs(k * spacing - 112.5) <= 1.6 for k in (1, 2, 3, 4))


def comb_screen(seg: np.ndarray, fs: float = 48000.0) -> tuple[float, float]:
    """(z at the 112.5 Hz comb, best spacing over 24-200 Hz), peak-free.

    The pre-filter `spread_score` is too expensive to be: one magnitude
    spectrum, flattened against a 120 Hz running median with the mains
    harmonics blanked, read through its own Fourier transform -- a comb of
    spacing D maximised over phase is the cepstral modulus at quefrency 1/D,
    so no peak is ever picked and no spacing is ever bandwidth arithmetic.
    z is the modulus at 1/112.5, normalised so matched noise sits near 1;
    a burst screens in when z clears `COMB_KNEE` *and* the best spacing
    satisfies `comb_family` -- PACTOR-3's 120 Hz carrier grid clears the
    first test alone often enough to need the second.
    """
    x = np.asarray(seg, np.float64)
    n = x.size
    if n < int(0.13 * fs):
        return 0.0, 0.0
    npad = 1 << max(16, int(np.ceil(np.log2(n * 4))))
    mag = np.abs(np.fft.rfft(x * get_window("hann", n), npad))
    f = np.fft.rfftfreq(npad, 1 / fs)
    df = f[1] - f[0]
    db = 20 * np.log10(mag + 1e-12)
    flat = db - median_filter(db, size=int(round(120 / df)) | 1, mode="nearest")
    for m in range(1, 44):
        flat[np.abs(f - 60 * m) < 4] = 0.0
    band = (f >= 300) & (f <= 2600)
    s = flat[band] - flat[band].mean()
    ceps = np.abs(np.fft.rfft(s, 4 * s.size)) / np.sqrt((s ** 2).sum())
    q = np.arange(ceps.size) / (4 * s.size * df)
    sel = (q >= 1 / 200.0) & (q <= 1 / 24.0)
    zb, qb = ceps[sel], q[sel]
    near = (qb >= 1 / 114.0) & (qb <= 1 / 111.0)
    z = float(zb[near].max()) if near.any() else 0.0
    return z, float(1 / qb[zb.argmax()])


def spread_score(seg: np.ndarray, fs: float = 48000.0) -> float:
    """Peak normalised correlation against the bare SF-16 spreading train.

    The cheap statistic, one template: it sees the chip-level periodicity every
    robust-mode burst carries whatever its Chu variant, and its response to a
    carrier offset repeats every 112.5 Hz, so +/-56 Hz of search covers all of
    them. This is the number `SPREAD_KNEE` is a knee for.
    """
    z = _baseband(seg, fs)
    r, _, _ = _scan(z, {0: _spread_template()}, np.arange(-56.0, 57.0, 4.0))
    return r


def header_score(seg: np.ndarray, fs: float = 48000.0):
    """Best (score, root, shift, cfo) over the full Chu19 header family.

    The confirming statistic, 342 templates: on the recorded emissions it runs
    ~0.1 above `spread_score` and pins the root, at a cost that belongs offline
    -- rescans and reports, not the cycle loop.
    """
    z = _baseband(seg, fs)
    templates = {(u, s): _header_template(u, s)
                 for u in range(1, CHU_LEN) for s in range(CHU_LEN)}
    r, key, cfo = _scan(z, templates, np.arange(-30.0, 31.0, 4.0))
    root, shift = key if key else (0, 0)
    return r, root, shift, cfo
