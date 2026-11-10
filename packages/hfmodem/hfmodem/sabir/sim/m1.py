# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Workhorse acquisition, coding and spectral measurements.

Run ``python -m hfmodem.sabir.sim.m1`` from the repo root to produce
``tests/sabir/m1_results.txt`` (and ``tests/sabir/m1_ber.png`` if matplotlib is present).

Eb/N0 convention: energy per *information* bit per data subcarrier at the FFT,
excluding CP / pilot / preamble overhead (the textbook normalisation, so the
uncoded curve is directly comparable to Q(sqrt(2 Eb/N0))). Uncoded QPSK
carries 2 info bits per cell; the rate-1/2 coded gear carries 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc, erfcinv

from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.dsp import channel
from hfmodem.sabir.phy import Phy
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL


def qfunc(x: float) -> float:
    return 0.5 * erfc(x / math.sqrt(2))


def qfuncinv(p: float) -> float:
    return math.sqrt(2) * erfcinv(2 * p)


def impair(x: np.ndarray, cfo_hz: float = 0.0, drift_hz_s: float = 0.0,
           ebn0_db: float | None = None, info_bits_per_cell: int = 1,
           rng: np.random.Generator | None = None) -> np.ndarray:
    if cfo_hz or drift_hz_s:
        t = np.arange(x.size) / FS
        x = x * np.exp(2j * np.pi * (cfo_hz * t + 0.5 * drift_hz_s * t * t))
    if ebn0_db is not None:
        x = channel.awgn(x, ebn0_db, info_bits_per_cell, rng)
    return x


# -- gate (a): byte-exact loopback -------------------------------------------
def loopback(payload: bytes, ebn0_db: float | None = None, cfo_hz: float = 0.0,
             drift_hz_s: float = 0.0, via_audio: bool = False, clip: bool = True,
             seed: int = 0):
    phy = Phy(clip=clip)
    fc = FrameCodec()
    tx = phy.transmit(fc.encode(payload))
    rx = impair(tx, cfo_hz, drift_hz_s, ebn0_db, 1, np.random.default_rng(seed))
    if via_audio:
        rx = phy.from_audio(phy.to_audio(rx))
    _, llr, res = phy.receive(rx)
    got, stats = fc.decode(llr)
    return got, res, stats


# -- gate (b): uncoded QPSK BER vs the Q-function ----------------------------
def uncoded_ber(ebn0_db: float, n_bits: int, seed: int = 0,
                syms_per_burst: int = 90) -> float:
    phy = Phy(clip=False)
    rng = np.random.default_rng(seed)
    bps = phy.capacity(syms_per_burst) // syms_per_burst
    errors = total = 0
    while total < n_bits:
        bits = rng.integers(0, 2, syms_per_burst * bps)
        tx = phy.transmit(bits)
        rx = channel.awgn(tx, ebn0_db, 2, rng)      # uncoded QPSK: 2 info bits/Es
        hard, _, _ = phy.receive(rx)
        errors += int((hard[: bits.size] != bits).sum())
        total += bits.size
    return errors / total


def uncoded_sweep(ebn0_list, n_bits: int, seed: int = 0):
    rows = []
    for i, ebn0 in enumerate(ebn0_list):
        ber = uncoded_ber(ebn0, n_bits, seed + i)
        theory = qfunc(math.sqrt(2 * 10 ** (ebn0 / 10)))
        implied = 10 * math.log10(qfuncinv(ber) ** 2 / 2) if 0 < ber < 0.5 else float("nan")
        rows.append((ebn0, ber, theory, ebn0 - implied))
    return rows


# -- gate (c): coded FER/BER waterfall, clipping ON --------------------------
def coded_fer(ebn0_db: float, n_frames: int, seed: int = 0,
              payload_len: int = 61) -> tuple[float, float]:
    phy = Phy(clip=True)
    fc = FrameCodec()
    rng = np.random.default_rng(seed)
    frame_errs = bit_errs = 0
    n_bits = 0
    for _ in range(n_frames):
        payload = rng.integers(0, 256, payload_len, dtype=np.uint8).tobytes()
        tx = phy.transmit(fc.encode(payload))
        rx = channel.awgn(tx, ebn0_db, 1, rng)      # rate 1/2: 1 info bit/Es
        _, llr, _ = phy.receive(rx)
        got, _ = fc.decode(llr)
        if got != payload:
            frame_errs += 1
            a = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
            if got is not None and len(got) == payload_len:
                b = np.unpackbits(np.frombuffer(got, dtype=np.uint8))
                bit_errs += int((a != b).sum())
            else:
                bit_errs += a.size // 2
        n_bits += 8 * payload_len
    return frame_errs / n_frames, bit_errs / n_bits


def coded_sweep(ebn0_list, n_frames: int, seed: int = 0):
    return [(e, *coded_fer(e, n_frames, seed + 100 * i))
            for i, e in enumerate(ebn0_list)]


# -- gate (d): CFO/drift acquisition sweep -----------------------------------
def cfo_sweep(ebn0_db: float = 5.0, trials: int = 3, seed: int = 0,
              payload_len: int = 180):
    rows = []
    base = np.random.default_rng(seed)
    for cfo in (-75.0, -40.0, 0.0, 40.0, 75.0):
        for drift in (-3.5, 0.0, 3.5):
            ok = 0
            err_max = 0.0
            for tr in range(trials):
                payload = base.integers(0, 256, payload_len, dtype=np.uint8).tobytes()
                got, res, _ = loopback(payload, ebn0_db, cfo, drift,
                                       seed=int(base.integers(1 << 30)))
                ok += got == payload
                # true CFO at acquisition vs the coarse estimate
                t_acq = (GUARD_HEAD + 5120) / FS
                err = abs(res.cfo_hz - (cfo + drift * t_acq))
                err_max = max(err_max, err)
            rows.append((cfo, drift, ok, trials, err_max))
    return rows


# -- PAPR + occupied bandwidth ----------------------------------------------
def papr_bandwidth(seed: int = 0):
    fc = FrameCodec()
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, 180, dtype=np.uint8).tobytes()
    out = {}
    for clip in (False, True):
        phy = Phy(clip=clip)
        tx = phy.transmit(fc.encode(payload))
        body = tx[GUARD_HEAD : tx.size - GUARD_TAIL]
        out["papr_on" if clip else "papr_off"] = float(
            10 * np.log10(np.max(np.abs(body) ** 2) / np.mean(np.abs(body) ** 2)))
        if clip:
            spec = np.abs(np.fft.rfft(phy.to_audio(tx))) ** 2
            freqs = np.fft.rfftfreq(tx.size, d=1 / FS)
            csum = np.cumsum(spec) / spec.sum()
            lo = freqs[np.searchsorted(csum, 0.005)]
            hi = freqs[np.searchsorted(csum, 0.995)]
            out["occupied_99_hz"] = float(hi - lo)
            out["band_lo_hz"], out["band_hi_hz"] = float(lo), float(hi)
    return out


def main():
    lines = ["Sabir workhorse acquisition and coding results", "=" * 60, ""]

    lines.append("Gate (a): byte-exact loopback (real ZC acquisition + pilot EQ)")
    rng = np.random.default_rng(42)
    for n, via_audio, ebn0 in ((200, True, None), (61, True, None),
                               (1, False, 15.0), (500, False, 10.0)):
        payload = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        got, _, _ = loopback(payload, ebn0, via_audio=via_audio, seed=n)
        chan = "clean audio" if via_audio else f"AWGN {ebn0} dB"
        lines.append(f"  {n:4d} bytes, {chan:>12}: "
                     f"{'EXACT' if got == payload else 'FAIL'}")
    lines.append("")

    lines.append("Gate (b): uncoded QPSK BER, clipping OFF (vs Q(sqrt(2 Eb/N0)))")
    lines.append("  Eb/N0 dB      BER       theory    offset dB")
    unc = uncoded_sweep([0, 2, 4, 6, 8], n_bits=400_000, seed=1)
    for ebn0, ber, theory, off in unc:
        lines.append(f"  {ebn0:7.1f}  {ber:.3e}  {theory:.3e}   {off:+.2f}")
    lines.append("")

    lines.append("Gate (c): coded FER/BER, rate-1/2 QC-LDPC (1024,512), clipping ON")
    lines.append("  Eb/N0 dB     FER        BER")
    cod = coded_sweep([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n_frames=200, seed=2)
    for ebn0, fer, ber in cod:
        lines.append(f"  {ebn0:7.1f}  {fer:8.3f}   {ber:.3e}")
    lines.append("")

    lines.append("Gate (d): acquisition + decode across +/-75 Hz, +/-3.5 Hz/s "
                 "(Eb/N0 5 dB, clip ON)")
    lines.append("   CFO Hz  drift Hz/s   exact   max CFO err Hz")
    for cfo, drift, ok, n, err in cfo_sweep(5.0, trials=3, seed=3):
        lines.append(f"  {cfo:+7.1f}  {drift:+9.1f}   {ok}/{n}      {err:.2f}")
    lines.append("")

    pb = papr_bandwidth()
    lines.append("PAPR / occupied bandwidth (workhorse gear, clip depth 4 dB)")
    lines.append(f"  PAPR clip off : {pb['papr_off']:.2f} dB")
    lines.append(f"  PAPR clip on  : {pb['papr_on']:.2f} dB")
    lines.append(f"  99% occupied  : {pb['occupied_99_hz']:.0f} Hz "
                 f"({pb['band_lo_hz']:.0f}..{pb['band_hi_hz']:.0f} Hz)")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m1_results.txt").write_text(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    eb = np.linspace(0, 8.5, 200)
    ax.semilogy(eb, [qfunc(math.sqrt(2 * 10 ** (e / 10))) for e in eb],
                "k--", lw=1, label="QPSK theory Q(sqrt(2 Eb/N0))")
    ax.semilogy([r[0] for r in unc], [r[1] for r in unc], "o-",
                label="uncoded, chain (clip off)")
    ax.semilogy([r[0] for r in cod if r[2] > 0], [r[2] for r in cod if r[2] > 0],
                "s-", label="rate-1/2 QC-LDPC (clip on), BER")
    ax.semilogy([r[0] for r in cod if r[1] > 0], [r[1] for r in cod if r[1] > 0],
                "^-", label="rate-1/2 QC-LDPC (clip on), FER")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("error rate")
    ax.set_title("Sabir workhorse: 24-carrier QPSK, AWGN")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(here / "m1_ber.png", dpi=120)


if __name__ == "__main__":
    main()
