# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Offline carrier-placement comparison with explicit noise and rate references.

python -m hfmodem.sabir.sim.design --frames 64 --output results.json

Both placements use identical payloads, pilot/coding choices and receiver.
The offset comparator reconstructs integer audio bins and the asymmetric FFT
filter; center1500 uses the current mixer and linear FIR. Noise is referenced
to transmitted active-region mean power before fading. Each frame uses new,
seeded Watterson taps; results are finite-sample estimates, not guarantees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.phy.rate import FS
from hfmodem.sabir.sim.m2 import add_noise_snr3k
from hfmodem.sabir.sim.watterson import Watterson


class OffsetPhy(Phy):
    """Comparison waveform: 18..41, 6..61 or 7..30 positive audio bins."""
    def __init__(self, gear, clip=True):
        super().__init__(gear, clip)
        first = 7 if gear.n_fft == 512 else 6 if gear.n_carriers == 56 else 18
        self._carriers = np.arange(first, first + gear.n_carriers)
        self._bins = self._carriers % self.n_fft
        self.mixer_hz = 0.0
        self.band = ((600., 2900.) if gear.n_fft == 512 else
                     (240., 2925.) if gear.n_carriers == 56 else (750., 2280.))

    def _bandpass(self, x):
        n = x.size
        m = next_fast_len(n)
        f = np.fft.fftfreq(m, 1 / FS)
        lo, hi = self.band
        mask = np.zeros(m)
        mask[(f >= lo) & (f <= hi)] = 1
        a = (f >= lo - 100) & (f < lo)
        b = (f > hi) & (f <= hi + 100)
        mask[a] = .5 * (1 - np.cos(np.pi * (f[a] - lo + 100) / 100))
        mask[b] = .5 * (1 + np.cos(np.pi * (f[b] - hi) / 100))
        return ifft(fft(x, m) * mask)[:n]


def spectrum(x):
    """Whole real-audio burst, no analysis window; includes preamble/guards."""
    a = Phy.to_audio(x)
    p = abs(np.fft.rfft(a)) ** 2
    f = np.fft.rfftfreq(a.size, 1 / FS)
    cumulative = np.cumsum(p) / p.sum()
    def bounds(tail):
        return f[np.searchsorted(cumulative, [tail, 1 - tail])].tolist()
    active = abs(x[GUARD_HEAD:-GUARD_TAIL]) ** 2
    return dict(band99_hz=bounds(.005), band999_hz=bounds(.0005),
                outside_100_2900_db=float(10 * np.log10(max(
                    p[(f < 100) | (f > 2900)].sum() / p.sum(), 1e-30))),
                papr_db=float(10 * np.log10(active.max() / active.mean())))


def trial(phy, fc, profile, snr, seed, cfo=0., drift=0.):
    rng = np.random.default_rng(seed)
    payload = rng.bytes(4 * fc.data_bytes)
    bits = fc.encode(payload)
    tx = phy.transmit(bits)
    y = Watterson(profile, FS, rng)(tx) if profile else tx.copy()
    t = np.arange(y.size) / FS
    y *= np.exp(2j * np.pi * (cfo * t + .5 * drift * t*t))
    y = add_noise_snr3k(y, snr, rng, reference=tx)
    # Include real sound-card projection and analytic ingress, not just IQ.
    y = phy.from_audio(phy.to_audio(y))
    try:
        _, llr, _ = phy.receive(y, n_symbols=phy.n_symbols_for(bits.size), dd=1)
        got, _ = fc.decode(llr)
    except ValueError:
        got = None
    return got == payload, tx


CASES = (
    ('robust', 'poor', -2., 0., 0.),
    ('workhorse', 'poor', 8., 0., 0.),
    ('workhorse34', 'poor', 12., 0., 0.),
    ('fast', 'moderate', 16., 0., 0.),
    ('max', 'good', 24., 0., 0.),
    ('sparse34', 'good', 10., 0., 0.),
    ('doppler', 'polar', 14., 0., 0.),
    ('wide256', None, 32., 0., 0.),
    ('workhorse', None, 10., -75., -3.5),
    ('workhorse', None, 10., 75., 3.5),
    # Negative controls: an unsuitable Doppler waveform and inadequate SNR.
    ('workhorse', 'polar', 14., 0., 0.),
    ('wide256', None, 12., 0., 0.),
)


def measure(frames, seed):
    rows = []
    for case, (name, channel, snr, cfo, drift) in enumerate(CASES):
        gear = GEARS[name]
        fc = FrameCodec(QCLDPC(gear.code), gear.repeat)
        for placement, constructor in [('offset', OffsetPhy), ('center1500', Phy)]:
            phy = constructor(gear)
            good = 0
            for i in range(frames):
                ok, tx = trial(phy, fc, channel, snr, seed + 1000*case + i, cfo, drift)
                good += ok
            rows.append(dict(gear=name, channel=channel or 'AWGN', snr3k_db=snr,
                             cfo_hz=cfo, drift_hz_s=drift, placement=placement,
                             successes=good, trials=frames,
                             payload_bytes=4*fc.data_bytes,
                             burst_seconds=tx.size / FS,
                             payload_burst_bps=32*fc.data_bytes / (tx.size / FS),
                             spectrum=spectrum(tx)))
    return dict(seed=seed, frames_per_case=frames,
                comparison_scope='bundled carrier placement and filter change; common current receiver and coding, not a center-only ablation',
                noise_reference='transmitted active-region mean power, 3000 Hz',
                receiver='paired-chirp acquisition, scattered pilots, one decision-directed pass, real audio ingress',
                spectrum_scope='last trial of each row; whole real-audio burst without analysis window',
                rate_scope='four codewords + CRC/length/FEC/pilots/CP/preamble/guards; excludes session controls, turnaround, retries',
                rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=int, default=64)
    ap.add_argument('--seed', type=int, default=1909)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.frames < 1:
        ap.error('--frames must be positive')
    result = measure(args.frames, args.seed)
    body = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(body)
    print(body)


if __name__ == '__main__':
    main()
