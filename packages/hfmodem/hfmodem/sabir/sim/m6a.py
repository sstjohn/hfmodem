# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6a acceptance: the fast adaptive control tier, with before/after numbers.

The M3 control plane rode every ACK/header/handshake on the ~4.5 s floor
burst regardless of link quality, wrapping each 7-11 s data body in ~9.8 s
of non-data air (42-53% link efficiency). M6a's fast coherent OFDM control
burst (~0.5 s) plus the ACK-piggybacked turn handover closes that gap; the
floor burst remains the connect/fallback safety net.

``python -m hfmodem.sabir.sim.m6a`` runs the sweeps -> ``tests/sabir/m6a_results.txt``:
per-rung effective-throughput table (old vs new, from rendered waveforms),
full sessions over Poor Watterson against the M3/M4 baselines, and the
low-SNR / deep-fade cases proving the floor fallback still protects the
link.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.arq import ArqConfig, LinkModem
from hfmodem.sabir.arq.fsm import DATA_LADDER
from hfmodem.sabir.fec import QCLDPC
from hfmodem.sabir.frame import FrameCodec
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import FS
from hfmodem.sabir.sim.m3 import run_pair


def rung_table(turnaround_s: float = 0.25) -> list[dict]:
    """Per-rung one-way cycle timing, old (floor control) vs new (fast),
    from rendered waveform durations: info bits / (header + body + 2
    turnarounds + ACK)."""
    lm = LinkModem(ArqConfig())          # renders both control tiers
    floor_s = lm.floor.transmit(bytes(22)).size / FS
    fast_s = lm.fast.n_samples(1) / FS
    rows = []
    rng = np.random.default_rng(0)
    for rc in DATA_LADDER:
        gear = GEARS[rc.name]
        phy = Phy(gear)
        codec = FrameCodec(QCLDPC(gear.code))
        coded = rng.integers(0, 2, (rc.frame_cws, codec.code.n))
        from hfmodem.sabir.arq import layout
        bits, _ = layout.assemble(phy, coded, rc.grouped, gear.repeat, None)
        body_s = phy.transmit(bits).size / FS
        info = 8 * rc.frame_cws * codec.data_bytes
        old = floor_s + body_s + 2 * turnaround_s + floor_s
        new = fast_s + body_s + 2 * turnaround_s + fast_s
        rows.append(dict(rung=rc.name, n_cw=rc.frame_cws, info_bits=info,
                         body_s=body_s, old_cycle_s=old, new_cycle_s=new,
                         old_bps=info / old, new_bps=info / new,
                         old_eff=body_s / old, new_eff=body_s / new))
    return rows


def fade_snr(t: float) -> float:
    """12 dB with a deep 0 dB fade from t = 60 s to 120 s."""
    return 0.0 if 60.0 <= t < 120.0 else 12.0


def main():
    lines = ["sabir M6a acceptance results", "=" * 64, ""]

    # 1. per-rung effective throughput, floor control vs fast control
    rows = rung_table()
    floor_s = (rows[0]["old_cycle_s"] - rows[0]["body_s"] - 0.5) / 2
    fast_s = (rows[0]["new_cycle_s"] - rows[0]["body_s"] - 0.5) / 2
    lines += [
        "Control-overhead gate: one data round = header + body + 2x0.25 s"
        " turnaround + ACK, waveform-rendered durations, uniform loading",
        f"  control burst: floor {floor_s:.2f} s -> fast {fast_s:.2f} s"
        f" per block ({2 * floor_s + 0.5:.2f} s -> {2 * fast_s + 0.5:.2f} s"
        " non-data air per round)",
        "",
        "  rung         cws  body_s   old cyc  eff   bit/s | new cyc  eff"
        "   bit/s | gain"]
    for r in rows:
        lines.append(
            f"  {r['rung']:12s} {r['n_cw']:3d}  {r['body_s']:6.2f}"
            f"  {r['old_cycle_s']:6.2f}  {100 * r['old_eff']:3.0f}%"
            f"  {r['old_bps']:5.0f} | {r['new_cycle_s']:6.2f}"
            f"  {100 * r['new_eff']:3.0f}%  {r['new_bps']:5.0f}"
            f" | x{r['new_bps'] / r['old_bps']:.2f}")
    lines.append("")

    # 2. full sessions over Poor Watterson vs the M3/M4 baselines
    payload = np.random.default_rng(1).integers(
        0, 256, 3000, dtype=np.uint8).tobytes()
    new3 = run_pair(payload, "poor", 8.0, seed=2)
    old3 = run_pair(payload, "poor", 8.0, seed=2,
                    cfg_kw={"fast_ctrl": False})
    lines += [
        "Session gate (M3 condition): 3000 B one-way, Poor Watterson"
        " (2 ms / 1 Hz), 8 dB",
        f"  floor-only control: {old3.throughput_bps:.0f} bit/s"
        f" ({old3.wall_s:.0f} s), byte-exact {old3.ok}   [M3 recorded:"
        " 213 bit/s]",
        f"  fast control      : {new3.throughput_bps:.0f} bit/s"
        f" ({new3.wall_s:.0f} s), byte-exact {new3.ok}"
        f"  -> x{new3.throughput_bps / old3.throughput_bps:.2f}",
        ""]

    rng = np.random.default_rng(4)
    pa = rng.integers(0, 256, 6000, dtype=np.uint8).tobytes()
    pb = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    new4 = run_pair(pa, "poor", 10.0, seed=7, payload_b=pb,
                    max_exchanges=900)
    old4 = run_pair(pa, "poor", 10.0, seed=7, payload_b=pb,
                    max_exchanges=900, cfg_kw={"fast_ctrl": False})
    n_new = 8 * (len(pa) + len(pb)) / new4.wall_s
    n_old = 8 * (len(pa) + len(pb)) / old4.wall_s
    ok_new = new4.ok
    lines += [
        "Session gate (M4 condition): 6000 B + 4000 B bidirectional, Poor"
        " Watterson, 10 dB",
        f"  floor-only control: {n_old:.0f} bit/s ({old4.wall_s:.0f} s),"
        f" byte-exact both ways {old4.ok}   [M4 recorded: 233 bit/s]",
        f"  fast control      : {n_new:.0f} bit/s ({new4.wall_s:.0f} s),"
        f" byte-exact both ways {ok_new}  -> x{n_new / n_old:.2f}",
        f"  ACKs piggybacked on reverse DATA headers:"
        f" {new4.stats_a['pb_acks'] + new4.stats_b['pb_acks']}"
        f" (turn handover without a control over)",
        ""]

    # 3. robustness: the floor safety net still carries the marginal link
    small = np.random.default_rng(9).integers(
        0, 256, 488, dtype=np.uint8).tobytes()
    lo = run_pair(small, "poor", -3.0, seed=3, start_rung=0)
    lines += [
        "Floor-fallback gate A: 488 B over Poor at -3 dB (below the fast"
        " tier's reach)",
        f"  byte-exact: {lo.ok}; control tier never left the floor:"
        f" {not lo.stats_a['ctrl'] and not lo.stats_b['ctrl']}",
        ""]

    big = np.random.default_rng(11).integers(
        0, 256, 4000, dtype=np.uint8).tobytes()
    fd = run_pair(big, "poor", fade_snr, seed=6, max_exchanges=900)
    a_tiers = [tier for _, tier in fd.stats_a["ctrl"]]
    lines += [
        "Floor-fallback gate B: 4000 B over Poor, 12 dB with a 0 dB fade"
        " from t=60 s to 120 s",
        f"  byte-exact: {fd.ok}; A's control tier trajectory:"
        f" {[(f'{t:.0f}s', tier) for t, tier in fd.stats_a['ctrl']]}",
        f"  fast engaged, fell back to floor in the fade, re-earned fast:"
        f" {'floor' in a_tiers[1:] and a_tiers[0] == 'fast' and a_tiers[-1] == 'fast'}",
        ""]

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m6a_results.txt").write_text(text)


if __name__ == "__main__":
    main()
