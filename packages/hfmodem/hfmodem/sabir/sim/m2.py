# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M2a acceptance sweeps: the OFDM ladder over Watterson fading, with numbers.

Run ``python -m hfmodem.sabir.sim.m2`` from the repo root to produce
``tests/sabir/m2_results.txt`` (and ``tests/sabir/m2_fer.png`` if matplotlib is present).

SNR convention: SNR in a 3 kHz reference bandwidth, normalized to each
received burst's active-region power. Fading is drawn fresh per frame. This
conditions out overall fade depth; use a transmitted reference for ensemble
SNR measurements (as in sim.design).
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.sim.watterson import Watterson

# each rung's demonstration profile and sweep range (dB, SNR in 3 kHz)
RUNG_PLAN = {
    "robust":      ("poor",     (-8, -6, -5, -4, -3, -2, 0)),
    "workhorse":   ("poor",     (0, 2, 3, 4, 5, 6, 8)),
    "workhorse34": ("poor",     (4, 6, 7, 8, 9, 10, 12)),
    "fast":        ("moderate", (8, 10, 12, 13, 14, 16, 18)),
    "max":         ("good",     (16, 18, 20, 21, 22, 24, 26)),
}


def add_noise_snr3k(x: np.ndarray, snr_db: float,
                    rng: np.random.Generator, *,
                    reference: np.ndarray | None = None) -> np.ndarray:
    """Complex AWGN with active-region power referenced to x or reference.

    Pass the unfaded transmitted burst as reference to retain fade depth.
    The default specifies SNR relative to this received realization.
    """
    ref = x if reference is None else reference
    body = ref[GUARD_HEAD : ref.size - GUARD_TAIL]
    p = np.mean(np.abs(body) ** 2)
    sigma2 = p * FS / (3000.0 * 10 ** (snr_db / 10))
    n = np.sqrt(sigma2 / 2) * (rng.standard_normal(x.shape)
                               + 1j * rng.standard_normal(x.shape))
    return x + n


def notch(x: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """Zero out [f_lo, f_hi] Hz -- a static dead band (interferer null,
    filter defect) for the carrier-masking demonstration."""
    spec = np.fft.fft(x)
    f = np.fft.fftfreq(x.size, d=1 / FS)
    spec[(f >= f_lo) & (f <= f_hi)] = 0
    return np.fft.ifft(spec)


class Rung:
    """One gear + its frame codec, bound together for the sweeps."""

    def __init__(self, gear_name: str, n_codewords: int = 2):
        self.gear = GEARS[gear_name]
        self.phy = Phy(self.gear)
        self.fc = FrameCodec(QCLDPC(self.gear.code), self.gear.repeat)
        self.payload_len = n_codewords * self.fc.data_bytes

    def frame(self, rng: np.random.Generator,
              tx_mask: np.ndarray | None = None):
        payload = rng.integers(0, 256, self.payload_len,
                               dtype=np.uint8).tobytes()
        bits = self.fc.encode(payload)
        tx = self.phy.transmit(bits, tx_mask)
        n_syms = self.phy.n_symbols_for(bits.size, tx_mask)
        return payload, tx, n_syms

    def info_bps(self) -> float:
        """Net payload rate of this rung's demonstration frame."""
        _, tx, _ = self.frame(np.random.default_rng(0))
        return 8 * self.payload_len / (tx.size / FS)


def rung_fer(gear_name: str, profile: str | None, snr_db: float,
             n_frames: int, seed: int = 0,
             tx_mask: np.ndarray | None = None,
             rx_masking: bool = True,
             notch_hz: tuple[float, float] | None = None) -> float:
    rung = Rung(gear_name)
    if not rx_masking:
        # disable the dead-carrier erasure by pushing its threshold to zero
        import hfmodem.sabir.dsp.fading as fading
        orig = fading.estimate
        def est_no_mask(*a, **kw):
            kw["mask_cap"] = 0.0
            return orig(*a, **kw)
        fading.estimate = est_no_mask
    rng = np.random.default_rng(seed)
    errs = 0
    try:
        for _ in range(n_frames):
            payload, tx, n_syms = rung.frame(rng, tx_mask)
            y = tx
            if profile is not None:
                y = Watterson(profile, FS, rng)(y)
            if notch_hz is not None:
                y = notch(y, *notch_hz)
            y = add_noise_snr3k(y, snr_db, rng)
            try:
                _, llr, _ = rung.phy.receive(y, n_symbols=n_syms,
                                             tx_mask=tx_mask)
                got, _ = rung.fc.decode(llr)
            except ValueError:
                got = None
            errs += got != payload
    finally:
        if not rx_masking:
            fading.estimate = orig
    return errs / n_frames


def rung_sweep(gear_name: str, profile: str, snr_list, n_frames: int,
               seed: int = 0):
    return [(snr, rung_fer(gear_name, profile, snr, n_frames,
                           seed + 1000 * i))
            for i, snr in enumerate(snr_list)]


# -- Watterson tap statistics -------------------------------------------------
def watterson_stats(seed: int = 0):
    from hfmodem.sabir.sim.watterson import PROFILES, tap_process
    rows = []
    rng = np.random.default_rng(seed)
    for name, (delays, spread) in PROFILES.items():
        fs = max(64.0 * spread / 2, 0.25)
        n = int(fs * max(3000.0, 400.0 / max(spread, 1e-3)))
        h = tap_process(n, fs, spread, rng=rng)
        p = float(np.mean(np.abs(h) ** 2))
        kurt = float(np.mean(np.abs(h) ** 4) / p**2)
        H = np.fft.fft(h * np.hanning(n))
        f = np.fft.fftfreq(n, 1 / fs)
        P = np.abs(H) ** 2
        mu = np.sum(f * P) / np.sum(P)
        sig2 = float(2 * np.sqrt(np.sum((f - mu) ** 2 * P) / np.sum(P)))
        rows.append((name, delays, spread, p, kurt, sig2))
    return rows


# -- dead-carrier masking demo ------------------------------------------------
def masking_demo(snr_db: float = 8.0, n_frames: int = 50, seed: int = 0):
    """Rate-3/4 rung through a static dead band (4 carriers wide) + AWGN --
    the code rate where ~17% dead cells actually bite.

    Three receivers: |H|^2/sigma^2 weighting only; + RX-side erasure of the
    detected dead carriers; and + TX-side masking (the M3 loading loop: TX
    skips the dead carriers, so its peak-limited power and code rate go
    entirely into live ones at 20/24 throughput).
    """
    gear = GEARS["workhorse34"]
    # kill relative carriers 8..11 = 4 of 24 (a ~187 Hz dead band mid-channel)
    dead = np.arange(8, 12)
    f_lo = gear.carrier_hz[dead[0]] - FS / gear.n_fft / 2
    f_hi = gear.carrier_hz[dead[-1]] + FS / gear.n_fft / 2
    tx_mask = np.ones(gear.n_carriers, dtype=bool)
    tx_mask[dead] = False
    out = {}
    out["unmasked"] = rung_fer("workhorse34", None, snr_db, n_frames, seed,
                               rx_masking=False, notch_hz=(f_lo, f_hi))
    out["rx_masked"] = rung_fer("workhorse34", None, snr_db, n_frames, seed,
                                rx_masking=True, notch_hz=(f_lo, f_hi))
    out["tx_masked"] = rung_fer("workhorse34", None, snr_db, n_frames, seed,
                                tx_mask=tx_mask, notch_hz=(f_lo, f_hi))
    out["snr_db"] = snr_db
    out["notch_hz"] = (f_lo, f_hi)
    return out


def main():
    lines = ["sabir M2a acceptance results", "=" * 64, ""]

    lines.append("Watterson tap statistics (unit power; Rayleigh kurtosis 2;")
    lines.append("measured two-sided 2-sigma Doppler width vs profile spec)")
    lines.append("  profile           delays ms   spread Hz   power  kurt   "
                 "measured Hz")
    for name, delays, spread, p, kurt, sig2 in watterson_stats(1):
        d = "/".join(f"{x:g}" for x in delays)
        lines.append(f"  {name:16s}  {d:9s}   {spread:7.1f}   {p:5.3f}  "
                     f"{kurt:5.3f}   {sig2:7.3f}")
    lines.append("")

    lines.append("OFDM ladder: FER vs SNR (3 kHz reference bandwidth), fresh")
    lines.append("Watterson realisation per frame, clip ON, real acquisition")
    sweeps = {}
    for i, (name, (profile, snrs)) in enumerate(RUNG_PLAN.items()):
        rung = Rung(name)
        bps = rung.info_bps()
        sweeps[name] = rung_sweep(name, profile, snrs, n_frames=60,
                                  seed=100 * i)
        lines.append(f"  {name} ({rung.gear.name}, {bps:.0f} bit/s net, "
                     f"{profile} profile)")
        lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in sweeps[name]))
        lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in sweeps[name]))
    lines.append("")

    lines.append("Robust rung on NVIS-disturbed (7 ms exceeds the 5.3 ms CP)")
    nvis = rung_sweep("robust", "nvis", (-2, 0, 2, 4), n_frames=40, seed=7000)
    lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in nvis))
    lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in nvis))
    lines.append("")

    md = masking_demo(snr_db=8.0, n_frames=60, seed=9000)
    lines.append("Dead-carrier masking: rate-3/4 rung, static dead band "
                 f"{md['notch_hz'][0]:.0f}-{md['notch_hz'][1]:.0f} Hz "
                 f"(4 of 24 carriers), AWGN {md['snr_db']:.0f} dB SNR")
    lines.append(f"  CSI weighting only    : FER {md['unmasked']:.2f}")
    lines.append(f"  + RX erasure masking  : FER {md['rx_masked']:.2f}")
    lines.append(f"  + TX masking (loading): FER {md['tx_masked']:.2f} "
                 "(20/24 carriers -> 83% throughput)")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m2_results.txt").write_text(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name, rows in sweeps.items():
        prof = RUNG_PLAN[name][0]
        snr = [s for s, _ in rows]
        fer = [max(f, 1e-3) for _, f in rows]
        ax.semilogy(snr, fer, "o-", label=f"{name} ({prof})")
    ax.set_xlabel("SNR in 3 kHz (dB)")
    ax.set_ylabel("frame error rate")
    ax.set_ylim(8e-3, 1.2)
    ax.set_title("sabir M2a: OFDM ladder over Watterson fading")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(here / "m2_fer.png", dpi=120)


if __name__ == "__main__":
    main()
