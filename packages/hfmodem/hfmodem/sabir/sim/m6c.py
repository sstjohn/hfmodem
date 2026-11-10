# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6c acceptance sweeps: the PHY-refinement bundle, with before/after numbers.

Run ``python -m hfmodem.sabir.sim.m6c`` from the repo root to produce
``tests/sabir/m6c_results.txt``. Five demonstrations, each an audit gap:

1. decoder-aided iterative channel estimation (CRC-clean codewords re-enter
   as exact virtual pilots, failed ones as decoder-refreshed references);
2. impulsive noise (Poisson sferic bursts) and the pre-FFT blanker + robust
   noise estimate + LLR weight cap that survives it;
3. the "doppler" mid-tier gear (512-FFT / 93.75 Hz / checkerboard lattice)
   between coherent OFDM and the noncoherent floor -- plus additional
   missing fast/max numbers on the Poor profile, honestly reported;
4. ACE (Krongold-Jones clip-and-project) PAPR on the QAM tiers, measured at
   fixed *peak* power against both the backed-off and the plain deep clip;
5. the "sparse34" 1-in-12 pilot gear leaning on (1) for a net-rate gain.

SNR convention matches ``sim.m2`` (3 kHz reference bandwidth).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Gear, Phy
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.sim.impulse import add_impulse_noise
from hfmodem.sabir.sim.m2 import add_noise_snr3k
from hfmodem.sabir.sim.watterson import Watterson


def decoder_feedback(fc: FrameCodec):
    """Phy.receive ``feedback`` adapter: LLRs -> (coded stream bits, known).

    Runs the frame's LDPC decode, then re-whitens/re-interleaves each
    codeword's hard output back into transmit order. Every cell gets the
    decoder's (post-50-iteration) opinion as its re-estimation reference --
    These are tentative decisions, including failed codewords. This optional
    experiment differs from the session receiver, which uses CRC-clean words
    only. The returned mask selects decoder references; it is not a CRC result.
    """
    def fb(llr: np.ndarray):
        L = fc.code.n
        n_cw = llr.size // (fc.repeat * L)
        if n_cw == 0:
            return None
        stream = n_cw * L
        cw_llr = llr[: fc.repeat * stream].reshape(
            fc.repeat, stream).sum(axis=0).reshape(L, n_cw).T
        deint = cw_llr[:, fc.inv_perm] * (1 - 2 * fc.pn)
        hard, _, _ = fc.code.decode(deint)
        bits = (hard.astype(np.int64) ^ fc.pn[None, :])[:, fc.perm]
        return (np.tile(bits.T.ravel(), fc.repeat),
                np.ones(fc.repeat * stream, dtype=bool))
    return fb


def add_noise_peak3k(x: np.ndarray, snr_db: float,
                     rng: np.random.Generator) -> np.ndarray:
    """AWGN at an SNR referenced to the burst's *peak* power (3 kHz bandwidth)
    -- the fair yardstick for PAPR work, where the transmitter is peak-power
    limited and every dB of PAPR saved is a dB of mean power gained."""
    body = x[GUARD_HEAD : x.size - GUARD_TAIL]
    p = np.max(np.abs(body) ** 2)
    sigma2 = p * FS / (3000.0 * 10 ** (snr_db / 10))
    n = np.sqrt(sigma2 / 2) * (rng.standard_normal(x.shape)
                               + 1j * rng.standard_normal(x.shape))
    return x + n


def fer(gear: Gear | str, profile: str | None, snr_db: float, n_frames: int,
        seed: int = 0, n_cw: int = 4, dd: int = 0, fb_iters: int = 0,
        blank: float = 0.0, impulses: dict | None = None,
        peak_ref: bool = False) -> float:
    gear = GEARS[gear] if isinstance(gear, str) else gear
    phy = Phy(gear)
    fc = FrameCodec(QCLDPC(gear.code), gear.repeat)
    feedback = decoder_feedback(fc) if fb_iters else None
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_frames):
        payload = rng.integers(0, 256, n_cw * fc.data_bytes,
                               dtype=np.uint8).tobytes()
        bits = fc.encode(payload)
        tx = phy.transmit(bits)
        n_syms = phy.n_symbols_for(bits.size)
        y = Watterson(profile, FS, rng)(tx) if profile else tx
        y = (add_noise_peak3k if peak_ref else add_noise_snr3k)(y, snr_db, rng)
        if impulses is not None:
            y = add_impulse_noise(y, rng, **impulses)
        try:
            _, llr, _ = phy.receive(y, n_symbols=n_syms, dd=dd, blank=blank,
                                    feedback=feedback, fb_iters=fb_iters)
            got, _ = fc.decode(llr)
        except ValueError:
            got = None
        errs += got != payload
    return errs / n_frames


def net_bps(gear: Gear | str, n_cw: int = 4) -> float:
    gear = GEARS[gear] if isinstance(gear, str) else gear
    phy = Phy(gear)
    fc = FrameCodec(QCLDPC(gear.code), gear.repeat)
    payload_len = n_cw * fc.data_bytes
    bits = fc.encode(bytes(payload_len))
    return 8 * payload_len / (phy.transmit(bits).size / FS)


def papr_db(gear: Gear, n_cw: int = 4, seed: int = 0) -> float:
    phy = Phy(gear)
    fc = FrameCodec(QCLDPC(gear.code), gear.repeat)
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, n_cw * fc.data_bytes,
                           dtype=np.uint8).tobytes()
    tx = phy.transmit(fc.encode(payload))
    p = np.abs(tx[GUARD_HEAD : tx.size - GUARD_TAIL]) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


IMP = dict(rate_hz=20.0, imp_db=25.0, dur_ms=(0.05, 0.3))


def main():
    lines = ["sabir M6c acceptance results (PHY-refinement bundle)",
             "=" * 64, "",
             "All frames: 4 codewords, fresh Watterson realisation per frame,",
             "real acquisition, SNR in 3 kHz referenced to each received burst.", ""]

    # 1. decoder-aided iterative channel estimation --------------------------
    lines += ["1. Decoder-aided iterative channel estimation",
              "   workhorse34 (24c-qpsk-r34) on Poor (2 ms / 1 Hz), 80 frames.",
              "   plain = scattered-pilot estimate only; dd1 = one",
              "   decision-directed pass (the session default receiver); fb2 = two",
              "   decoder-aided passes (LDPC hard output as the virtual-pilot",
              "   reference for every cell).",
              "   SNR dB    plain   dd1     fb2"]
    for snr in (7.0, 8.0, 9.0):
        s = int(10 * snr)
        f0 = fer("workhorse34", "poor", snr, 80, seed=s)
        f1 = fer("workhorse34", "poor", snr, 80, seed=s, dd=1)
        f2 = fer("workhorse34", "poor", snr, 80, seed=s, fb_iters=2)
        lines.append(f"   {snr:+5.1f}    {f0:5.2f}   {f1:5.2f}   {f2:5.2f}")
    lines.append("")

    # 2. impulsive noise + blanker -------------------------------------------
    lines += ["2. Impulse noise and the pre-FFT blanker",
              f"   fast (56c-16qam-r34) on Moderate + Poisson bursts"
              f" ({IMP['rate_hz']:.0f}/s,",
              f"   +{IMP['imp_db']:.0f} dB, {IMP['dur_ms'][0]}-"
              f"{IMP['dur_ms'][1]} ms), 50 frames. blanker = 3.5 x local",
              "   median envelope zeroing + median noise estimate + LLR"
              " weight cap.",
              "   SNR dB    no blanker   blanker"]
    for snr in (12.0, 14.0, 16.0):
        s = int(10 * snr)
        f0 = fer("fast", "moderate", snr, 50, seed=s, impulses=IMP)
        f1 = fer("fast", "moderate", snr, 50, seed=s, impulses=IMP, blank=3.5)
        lines.append(f"   {snr:+5.1f}      {f0:5.2f}      {f1:5.2f}")
    s = 140
    f0 = fer("fast", "moderate", 14.0, 50, seed=s)
    f1 = fer("fast", "moderate", 14.0, 50, seed=s, blank=3.5)
    lines += [f"   control (no impulses, 14 dB): blanker off {f0:.2f},"
              f" on {f1:.2f} -- inert on clean air", ""]

    # 3. doppler mid-tier + additional Poor-channel measurements -----------------
    dop = GEARS["doppler"]
    lines += ["3a. Doppler mid-tier gear",
              f"   doppler ({dop.name}: 512-FFT, 93.75 Hz spacing, 57.7 baud,",
              "   checkerboard pilot lattice = 28.8 Hz track on every"
              " carrier),",
              f"   {net_bps('doppler'):.0f} bit/s net vs workhorse"
              f" {net_bps('workhorse'):.0f} bit/s. 40 frames.",
              "   profile          SNR dB   workhorse   doppler"]
    for prof, snrs in (("subpolar (2 ms/5 Hz)", (6.0, 8.0, 12.0)),
                       ("polar (3 ms/10 Hz)", (10.0, 14.0, 18.0))):
        pname = prof.split()[0]
        for snr in snrs:
            s = int(10 * snr)
            fw = fer("workhorse", pname, snr, 40, seed=s)
            fd = fer("doppler", pname, snr, 40, seed=s)
            lines.append(f"   {prof:20s} {snr:+5.1f}    {fw:5.2f}     {fd:5.2f}")
    lines.append("")

    lines += ["3b. fast/max rungs on the Poor profile (2 ms / 1 Hz) --"
              " additional",
              "   channel coverage beyond"
              " moderate/good. 40 frames,",
              "   plain receiver / with the decoder-aided passes (fb2)."]
    for name, snrs in (("fast", (14.0, 18.0, 22.0, 26.0)),
                       ("max", (22.0, 26.0, 30.0))):
        row = "  ".join(
            f"{snr:+.0f}: "
            f"{fer(name, 'poor', snr, 40, seed=int(10 * snr)):.2f}"
            f"/{fer(name, 'poor', snr, 40, seed=int(10 * snr), fb_iters=2):.2f}"
            for snr in snrs)
        lines.append(f"   {name:5s}  {row}")
    lines += ["   FER is a finite-sample whole-frame result, not an operating threshold.", ""]

    # 4. ACE PAPR on the QAM tiers -------------------------------------------
    lines += ["4. Active constellation extension (Krongold-Jones"
              " clip-and-project)",
              "   FER at fixed *peak* power (add_noise_peak3k), 40 frames:"
              " the backed-off",
              "   baseline clip vs the same deep clip plain vs deep clip"
              " with ACE.",
              "   rung  peak-SNR |  base clip: PAPR/FER |  deep plain |"
              "  deep ACE"]
    for name, prof, snrs, deep in (("fast", "moderate", (21.0, 23.0), 4.5),
                                   ("max", "good", (27.0, 29.0), 6.0)):
        g0 = GEARS[name]
        gp = replace(g0, clip_papr_db=deep)
        ga = replace(g0, clip_papr_db=deep, ace=True)
        p0, pp, pa = papr_db(g0), papr_db(gp), papr_db(ga)
        for snr in snrs:
            s = int(10 * snr)
            f0 = fer(g0, prof, snr, 40, seed=s, peak_ref=True)
            fp = fer(gp, prof, snr, 40, seed=s, peak_ref=True)
            fa = fer(ga, prof, snr, 40, seed=s, peak_ref=True)
            lines.append(
                f"   {name:5s} {snr:+5.1f}   |  {g0.clip_papr_db:.1f} dB:"
                f" {p0:.2f}/{f0:.2f}  |  {deep:.1f} dB: {pp:.2f}/{fp:.2f}"
                f"  |  {pa:.2f}/{fa:.2f}")
    lines += ["   These compare equal peak-power limits. Stock profiles retain their",
              "   listed clipping depths and disable ACE.", ""]

    # 5. sparse pilots on slow channels --------------------------------------
    sp = GEARS["sparse34"]
    lines += ["5. Adaptive pilot density: sparse34"
              f" ({sp.name}, 1-in-12 pilots)",
              f"   net rate {net_bps('sparse34'):.0f} vs workhorse34"
              f" {net_bps('workhorse34'):.0f} bit/s"
              f" (+{100 * (net_bps('sparse34') / net_bps('workhorse34') - 1):.0f}%),"
              " 40 frames.",
              "   profile   SNR dB   w34 plain   sp34 plain   sp34 fb2"]
    for prof, snrs in (("good", (8.0, 12.0)), ("poor", (9.0, 11.0))):
        for snr in snrs:
            s = int(10 * snr)
            fw = fer("workhorse34", prof, snr, 40, seed=s)
            f0 = fer("sparse34", prof, snr, 40, seed=s)
            f2 = fer("sparse34", prof, snr, 40, seed=s, fb_iters=2)
            lines.append(f"   {prof:8s} {snr:+5.1f}     {fw:5.2f}      "
                         f"{f0:5.2f}       {f2:5.2f}")
    lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m6c_results.txt").write_text(text)


if __name__ == "__main__":
    main()
