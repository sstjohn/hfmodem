# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M2b acceptance sweeps: the noncoherent floor family, with numbers.

Run ``python -m hfmodem.sabir.sim.m2b`` from the repo root to produce
``tests/sabir/m2b_results.txt`` (and ``tests/sabir/m2b_fer.png`` if matplotlib is there).

Four demonstrations:
- floor FER vs SNR on AWGN (the sensitivity ladder: repeat x1/x2/x4);
- the dual-waveform gate: polar_disturbed (7 ms / 30 Hz) kills the OFDM
  rungs at the tested SNRs while the floor can decode below 0 dB;
- CA-TBCC vs an equal-length (256,128) QC-LDPC on BPSK/AWGN -- the reason
  the floor code is convolutional;
- the control burst's occupied bandwidth (<= 500 Hz for the ACDS role).

SNR is measured after fading in a 3 kHz reference bandwidth, averaged over
the padded receive window. This historical benchmark is not an ensemble
transmitted-power reference; use ``sim.m3.run_pair`` for that convention.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.arq import wire

from hfmodem.sabir.fec import CATBCC, QCLDPC
from hfmodem.sabir.floor import FLOOR_GEARS, FloorModem
from hfmodem.sabir.frame import crc16
from hfmodem.sabir.frame.codec import crc_ok
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.m2 import Rung, add_noise_snr3k
from hfmodem.sabir.sim.watterson import Watterson

CONTROL = wire.Control(wire.DATA, 0x81, seq=1, gear=4,
                       mask=wire.cw_mask([0]), aux=wire.data_aux(154, 1, (0,) * 8))
BLOCK = CONTROL.pack()
PAD = 48000                       # 1 s of listening noise either side


def floor_fer(gear_name: str, profile: str | None, snr_db: float,
              n_frames: int, seed: int = 0) -> float:
    gear = FLOOR_GEARS[gear_name]
    modem = FloorModem(gear)
    tx = modem.transmit(BLOCK)
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_frames):
        y = np.concatenate([np.zeros(PAD), tx, np.zeros(PAD)])
        if profile is not None:
            y = Watterson(profile, FS, rng)(y)
        y = add_noise_snr3k(y, snr_db, rng)
        block, _ = modem.receive(y, wire.BLOCK_BYTES)
        errs += wire.Control.unpack(block) != CONTROL if block else 1
    return errs / n_frames


def floor_sweep(gear_name: str, profile: str | None, snr_list, n_frames: int,
                seed: int = 0):
    return [(snr, floor_fer(gear_name, profile, snr, n_frames,
                            seed + 1000 * i))
            for i, snr in enumerate(snr_list)]


def ofdm_fer(rung_name: str, profile: str, snr_db: float, n_frames: int,
             seed: int = 0) -> float:
    rung = Rung(rung_name, n_codewords=1)
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_frames):
        payload, tx, n_syms = rung.frame(rng)
        y = add_noise_snr3k(Watterson(profile, FS, rng)(tx), snr_db, rng)
        try:
            _, llr, _ = rung.phy.receive(y, n_symbols=n_syms)
            got, _ = rung.fc.decode(llr)
        except ValueError:
            got = None
        errs += got != payload
    return errs / n_frames


# -- CA-TBCC vs short LDPC ------------------------------------------------------
def code_duel(ebn0_db: float, n_frames: int, seed: int = 0):
    """(256,128) CA-TBCC vs (256,128) QC-LDPC, BPSK on AWGN, same payload
    (14 bytes + CRC-16). Returns (tbcc_bler, ldpc_bler)."""
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, 14, dtype=np.uint8).tobytes()
    block = payload + crc16(payload).to_bytes(2, "big")
    info = np.unpackbits(np.frombuffer(block, dtype=np.uint8)).astype(np.int64)
    tb = CATBCC()
    ldpc = QCLDPC("r12", z=16)
    cw_tb = tb.encode(block)
    cw_ld = ldpc.encode(info).astype(np.int64)
    sigma = np.sqrt(1 / 10 ** (ebn0_db / 10))     # rate 1/2, unit-energy BPSK
    e_tb = e_ld = 0
    for _ in range(n_frames):
        y = (1.0 - 2.0 * cw_tb) + sigma * rng.standard_normal(256)
        got, _ = tb.decode(2 * y / sigma**2, len(block), crc_ok)
        e_tb += got != block
        y = (1.0 - 2.0 * cw_ld) + sigma * rng.standard_normal(256)
        hard, ok, _ = ldpc.decode(2 * y / sigma**2)
        e_ld += not (ok[0] and (hard[0] == cw_ld).all())
    return e_tb / n_frames, e_ld / n_frames


# -- spectrum -------------------------------------------------------------------
def occupied_bw(x: np.ndarray, frac: float = 0.99):
    """Equal-tail occupied-power band of the complete real-audio burst."""
    audio = np.sqrt(2.0) * np.asarray(x).real
    spec = np.abs(np.fft.rfft(audio)) ** 2
    f = np.fft.rfftfreq(audio.size, 1 / FS)
    tail = (1 - frac) / 2
    bounds = np.searchsorted(np.cumsum(spec) / spec.sum(), [tail, 1 - tail])
    return tuple(float(f[b]) for b in bounds)


def main():
    lines = ["sabir M2b acceptance results (the noncoherent floor)",
             "=" * 64, ""]

    lines.append("Floor bursts carry the 44-byte session control block "
                 "(42-byte header + CRC-16),")
    lines.append("CA-TBCC (704 coded bits per copy), 4-FSK. Header rate = "
                 "336 non-CRC bits / burst (no user data):")
    for name in ("floor", "floor2", "floor4"):
        m = FloorModem(FLOOR_GEARS[name])
        dur = m.duration_s(wire.BLOCK_BYTES)
        lines.append(f"  {name:7s} repeat x{m.gear.repeat}: {dur:5.2f} s, "
                     f"{336 / dur:5.1f} bit/s header")
    lines.append("")

    lines.append("Floor FER vs SNR (3 kHz ref) on AWGN, real Costas sync, "
                 "1 s noise pads; noise set from post-channel padded-window power")
    plan = {
        "floor": (-16, -15, -14, -13, -12, -10),
        "floor2": (-18, -17, -16, -15, -14, -12),
        "floor4": (-20, -19, -18, -17, -16, -14),
    }
    sweeps = {}
    for i, (name, snrs) in enumerate(plan.items()):
        sweeps[name] = floor_sweep(name, None, snrs, n_frames=40, seed=100 * i)
        lines.append(f"  {name}")
        lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in sweeps[name]))
        lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in sweeps[name]))
    lines.append("")

    lines.append("The dual-waveform gate: polar_disturbed (7 ms delay, 30 Hz "
                 "Doppler spread).")
    lines.append("Coherent OFDM results at the tested SNRs:")
    for name, snr in (("robust", 10.0), ("robust", 20.0),
                      ("workhorse", 15.0), ("workhorse", 25.0)):
        fer = ofdm_fer(name, "polar_disturbed", snr, 20, seed=int(snr))
        lines.append(f"  OFDM {name:10s} @ {snr:+5.1f} dB : FER {fer:.2f}")
    lines.append("The floor decodes the same channel far below 0 dB:")
    pd = {}
    for name, snrs in (("floor", (-8, -6, -4, -2, 0)),
                       ("floor2", (-10, -8, -6, -4))):
        pd[name] = floor_sweep(name, "polar_disturbed", snrs, n_frames=30,
                               seed=7000)
        lines.append(f"  floor {name}")
        lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in pd[name]))
        lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in pd[name]))
    lines.append("")

    lines.append("Floor on the polar profile (3 ms / 10 Hz):")
    pol = floor_sweep("floor", "polar", (-10, -8, -6, -4), n_frames=30,
                      seed=8000)
    lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in pol))
    lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in pol))
    lines.append("")

    lines.append("CA-TBCC vs equal-length QC-LDPC, (256,128), same 14-byte+"
                 "CRC payload, BPSK/AWGN:")
    lines.append("  Eb/N0 dB :  TBCC   LDPC")
    for ebn0 in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        bt, bl = code_duel(ebn0, 400, seed=int(10 * ebn0))
        lines.append(f"  {ebn0:8.1f} : {bt:6.3f} {bl:6.3f}")
    lines.append("")

    tx = FloorModem().transmit(BLOCK)
    lo, hi = occupied_bw(tx)
    body = tx[np.abs(tx) > 0]
    p = np.abs(body) ** 2
    lines.append(f"Control burst spectrum: 99% occupied {lo:.0f}-{hi:.0f} Hz "
                 f"= {hi - lo:.0f} Hz (<= 500 Hz ACDS limit), "
                 f"PAPR {10 * np.log10(p.max() / p.mean()):.2f} dB")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m2b_results.txt").write_text(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name, rows in sweeps.items():
        ax.semilogy([s for s, _ in rows], [max(f, 1e-3) for _, f in rows],
                    "o-", label=f"{name} (AWGN)")
    for name, rows in pd.items():
        ax.semilogy([s for s, _ in rows], [max(f, 1e-3) for _, f in rows],
                    "s--", label=f"{name} (polar_disturbed)")
    ax.set_xlabel("SNR in 3 kHz (dB)")
    ax.set_ylabel("frame error rate")
    ax.set_ylim(8e-3, 1.2)
    ax.set_title("sabir M2b: the noncoherent floor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(here / "m2b_fer.png", dpi=120)


if __name__ == "__main__":
    main()
