# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6b acceptance sweeps: the narrow-tone beacon, with numbers.

Run ``python -m hfmodem.sabir.sim.m6b`` from the repo root to produce
``tests/sabir/m6b_results.txt`` (and ``tests/sabir/m6b_fer.png`` if matplotlib is there).

Three demonstrations:
- beacon FER vs SNR on AWGN per depth rung, with the FER-0.5 crossing,
  integration time and net bit rate -- the bottom of the gear ladder;
- the deepest SNR at which a single burst decodes field-exact;
- the Doppler trade: the narrow tones under Watterson good/moderate/poor,
  where the wide interactive floor keeps working and the beacon cannot.

SNR convention: 2.5 kHz reference bandwidth (the WSJT convention), unlike
``sim.m2``'s 3 kHz -- subtract 0.79 dB to compare against the floor sweeps.
"""

from __future__ import annotations

import numpy as np

from hfmodem.sabir.floor import BEACON_GEARS, BeaconModem, BeaconPayload, send_beacon
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.sim.watterson import Watterson

PAYLOAD = BeaconPayload("W1AW", "FN31", status=173)
PAD = 48000                       # 1 s of listening noise either side


def add_noise_snr2k5(x: np.ndarray, snr_db: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Complex AWGN at a given SNR in a 2.5 kHz reference bandwidth."""
    body = x[GUARD_HEAD : x.size - GUARD_TAIL]
    p = np.mean(np.abs(body) ** 2)
    sigma2 = p * FS / (2500.0 * 10 ** (snr_db / 10))
    n = np.sqrt(sigma2 / 2) * (rng.standard_normal(x.shape)
                               + 1j * rng.standard_normal(x.shape))
    return x + n


def beacon_trial(modem: BeaconModem, tx: np.ndarray, profile: str | None,
                 snr_db: float, rng) -> bool:
    """One TX -> channel -> RX pass; True iff the decode is field-exact."""
    y = np.concatenate([np.zeros(PAD), tx, np.zeros(PAD)])
    if profile is not None:
        y = Watterson(profile, FS, rng)(y)
    y = add_noise_snr2k5(y, snr_db, rng)
    block, _ = modem.receive(y)
    return block is not None and BeaconPayload.unpack(block) == PAYLOAD


def beacon_fer(gear_name: str, profile: str | None, snr_db: float,
               n_frames: int, seed: int = 0) -> float:
    gear = BEACON_GEARS[gear_name]
    modem = BeaconModem(gear)
    tx = send_beacon(PAYLOAD, gear)
    rng = np.random.default_rng(seed)
    errs = sum(not beacon_trial(modem, tx, profile, snr_db, rng)
               for _ in range(n_frames))
    return errs / n_frames


def beacon_sweep(gear_name: str, profile: str | None, snr_list, n_frames: int,
                 seed: int = 0):
    return [(snr, beacon_fer(gear_name, profile, snr, n_frames,
                             seed + 1000 * i))
            for i, snr in enumerate(snr_list)]


def crossing(rows) -> float | None:
    """Linear FER-0.5 crossing of a (snr, fer) sweep, descending SNR order."""
    for (s1, f1), (s2, f2) in zip(rows, rows[1:]):
        if f1 <= 0.5 <= f2 and f2 > f1:
            return s1 + (0.5 - f1) * (s2 - s1) / (f2 - f1)
    return None


def deepest_decode(gear_name: str, start_db: float, seed: int = 0,
                   step: float = 0.5) -> tuple[float, BeaconPayload]:
    """Walk down from start_db until a single burst no longer decodes
    field-exact; returns the deepest SNR that did, and the decoded fields."""
    gear = BEACON_GEARS[gear_name]
    modem = BeaconModem(gear)
    tx = send_beacon(PAYLOAD, gear)
    best = None
    snr = start_db
    while True:
        rng = np.random.default_rng((seed, round(-10 * snr)))
        y = add_noise_snr2k5(
            np.concatenate([np.zeros(PAD), tx, np.zeros(PAD)]), snr, rng)
        block, _ = modem.receive(y)
        got = BeaconPayload.unpack(block) if block else None
        if got != PAYLOAD:
            break
        best = (snr, got)
        snr -= step
    return best


def main():
    lines = ["sabir M6b acceptance results (the narrow-tone beacon)",
             "=" * 64, ""]

    lines.append("Beacon bursts carry callsign+grid+status (55 payload bits "
                 "in a 9-byte")
    lines.append("CRC-terminated block), CA-TBCC rate 1/2, 4-FSK on narrow "
                 "tones, 7x7")
    lines.append("Costas self-sync (no GPS/NTP slotting). SNR in 2.5 kHz "
                 "(WSJT convention).")
    lines.append("")
    for name, gear in BEACON_GEARS.items():
        m = BeaconModem(gear)
        dur = m.duration_s(9)
        lines.append(f"  {name:13s} {gear.name:18s} tones {m.grid:5.3f} Hz, "
                     f"{dur:6.1f} s, {55 / dur:4.2f} bit/s net")
    lines.append("")

    lines.append("FER vs SNR on AWGN, real Costas sync, 1 s noise pads:")
    plan = {
        "beacon_short": (-19, -20, -21, -22, -23, -24),
        "beacon_med":   (-22, -23, -24, -25, -26, -27),
        "beacon_deep":  (-25, -26, -27, -28, -29, -30),
        "beacon_deep2": (-27, -28, -29, -30, -31, -32),
    }
    frames = {"beacon_short": 40, "beacon_med": 30,
              "beacon_deep": 25, "beacon_deep2": 20}
    sweeps = {}
    for i, (name, snrs) in enumerate(plan.items()):
        sweeps[name] = beacon_sweep(name, None, snrs, frames[name],
                                    seed=100 * i)
        cross = crossing(sweeps[name])
        lines.append(f"  {name} ({frames[name]} frames/point)")
        lines.append("    SNR dB : " + "  ".join(f"{s:+5.1f}" for s, _ in sweeps[name]))
        lines.append("    FER    : " + "  ".join(f"{f:5.2f}" for _, f in sweeps[name]))
        lines.append("    FER-0.5 crossing: "
                     + (f"{cross:+.1f} dB" if cross else "outside sweep"))
    lines.append("")

    snr, got = deepest_decode("beacon_deep2", -28.0, seed=42)
    lines.append(f"Deepest single-burst field-exact decode (beacon_deep2, "
                 f"0.5 dB steps): {snr:+.1f} dB")
    lines.append(f"  decoded: callsign={got.callsign!r} grid={got.grid!r} "
                 f"status={got.status}")
    lines.append("")

    lines.append("The Doppler trade (Watterson fading; the narrow tones "
                 "trade the wide")
    lines.append("floor's 30 Hz Doppler immunity for sensitivity -- spread "
                 "comparable to")
    lines.append("the tone spacing smears the energy out of the detection "
                 "bins):")
    dop = {}
    for name, snrs in (("beacon_med", (-14, -18, -22)),
                       ("beacon_deep", (-14, -18, -22))):
        for profile in ("good", "moderate", "poor"):
            rows = beacon_sweep(name, profile, snrs, 12, seed=5000)
            dop[(name, profile)] = rows
            lines.append(f"  {name:12s} {profile:9s}: " + "  ".join(
                f"{s:+.0f} dB->{f:.2f}" for s, f in rows))
    lines.append("(For reference the AWGN crossings above sit 6-14 dB lower;")
    lines.append(" under spread >= ~tone spacing the deep rungs never "
                 "converge -- use the")
    lines.append(" wide interactive floor there. The beacon is for quiet, "
                 "stable paths.)")

    text = "\n".join(lines) + "\n"
    print(text)
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[2] / "tests" / "sabir"
    (here / "m6b_results.txt").write_text(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for name, rows in sweeps.items():
        ax.semilogy([s for s, _ in rows], [max(f, 1e-3) for _, f in rows],
                    "o-", label=name)
    ax.set_xlabel("SNR in 2.5 kHz (dB)")
    ax.set_ylabel("frame error rate")
    ax.set_ylim(8e-3, 1.2)
    ax.set_title("sabir M6b: narrow-tone beacon on AWGN")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(here / "m6b_fer.png", dpi=120)


if __name__ == "__main__":
    main()
