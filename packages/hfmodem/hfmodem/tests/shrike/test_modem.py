# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Validate the multitone DPSK modulator against synthetic ground truth.

Proves: differential mapping inverts; tones land on the published frequencies;
the whole TX->RX loopback is bit-exact in the clear and degrades gracefully in
noise. Also checks the synthesised crest factor against the values SCS publishes
per speed level -- an INDEPENDENT cross-check that our waveform construction is
shaped like the real thing, not merely self-consistent.

Run:  .venv/bin/python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_modem.py
"""

from __future__ import annotations

import contextlib

import numpy as np


from hfmodem.shrike import modem, placement, spec  # noqa: E402

rng = np.random.default_rng(7)
FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)


@contextlib.contextmanager
def time_domain_only():
    """Force the per-tone sum in time, through the same public entry point."""
    real = modem._transform_length
    modem._transform_length = lambda cfg, n_out: None
    try:
        yield
    finally:
        modem._transform_length = real


def test_all() -> None:
    print("Multitone DPSK modem validation\n")

    # --- 1. differential mapping ----------------------------------------------
    print("1. differential encode -> decode, noiseless")
    for bps, name in [(1, "DBPSK"), (2, "DQPSK")]:
        bits = rng.integers(0, 2, 400, dtype=np.uint8)
        syms = modem.differential_encode(bits, bps)
        check(syms.size == bits.size // bps + 1,
              f"{name}: N/{bps} symbols + 1 phase-reference symbol")
        check(np.allclose(np.abs(syms), 1.0), f"{name}: symbols are unit-modulus")
        soft = modem.differential_decode(syms, bps)
        hard = (soft < 0).astype(np.uint8)
        check(np.array_equal(hard, bits), f"{name}: bits recovered exactly")

    # a differential system must be immune to a constant phase rotation
    print("\n2. differential decoding is invariant to a constant phase offset")
    for bps, name in [(1, "DBPSK"), (2, "DQPSK")]:
        bits = rng.integers(0, 2, 200, dtype=np.uint8)
        syms = modem.differential_encode(bits, bps)
        for rot_deg in (17.0, 90.0, 200.0):
            rot = syms * np.exp(1j * np.deg2rad(rot_deg))
            hard = (modem.differential_decode(rot, bps) < 0).astype(np.uint8)
            check(np.array_equal(hard, bits),
                  f"{name}: immune to a {rot_deg:g} deg rotation")

    # --- 3. tones land where the spec says -------------------------------------
    print("\n3. synthesised tones land on the published frequencies")
    # Use ALL-ZERO data: with DBPSK a zero bit means "no phase change", so each tone
    # collapses to a pure CW carrier and its spectral line is unambiguous. (Testing
    # this with random data is a trap -- a 100 Bd DPSK tone has a ~100 Hz-wide, noisy
    # periodogram whose peak bin wanders by tens of Hz, which says nothing.)
    cfg = modem.ModConfig()
    for sl in (1, 6):
        s = spec.SPEED_LEVELS[sl]
        n_sym = 128
        tones = {cn: modem.differential_encode(
                     np.zeros((n_sym - 1) * s.bits_per_symbol, dtype=np.uint8),
                     s.bits_per_symbol)
                 for cn in s.channels}
        sig = modem.modulate_tones(tones, cfg)

        freqs = np.fft.rfftfreq(sig.size, 1 / cfg.sample_rate)
        psd = np.abs(np.fft.rfft(sig)) ** 2
        worst = 0.0
        for cn in s.channels:
            f0 = spec.channel_freq_hz(cn)
            band = (freqs > f0 - 50) & (freqs < f0 + 50)
            peak_f = freqs[band][np.argmax(psd[band])]
            worst = max(worst, abs(peak_f - f0))
        check(worst < 2.0,
              f"SL{sl}: all {s.n_tones} carriers within {worst:.2f} Hz of the published "
              f"frequencies ({spec.channel_freq_hz(s.channels[0]):.0f}"
              f"-{spec.channel_freq_hz(s.channels[-1]):.0f} Hz)",
              f"worst offset {worst:.2f} Hz")

        # essentially no energy outside the published 2.2 kHz mask
        inband = (freqs >= 400) & (freqs <= 2600)
        frac = psd[inband].sum() / psd.sum()
        check(frac > 0.99,
              f"SL{sl}: {frac * 100:.2f}% of energy inside the 400-2600 Hz mask")

    # --- 4. crest factor vs the published table --------------------------------
    # The spec tabulates CFR per speed level (1.9 dB at SL1 rising to 5.7 dB at SL6).
    # We should land in the same ballpark and, crucially, reproduce the TREND -- more
    # tones means a peakier sum. If we were building the signal wrongly (e.g. all
    # tones phase-aligned) the crest factor would blow up.
    print("\n4. crest factor vs the published table -- MEASURES THE U7 GAP")
    # This is not a pass/fail on our code; it is a measurement that tells us how much
    # of the waveform we are still missing.
    #
    # A naive modulator gives every tone the same starting phase, so the 18 carriers
    # sum like independent Gaussians and the peak-to-RMS ratio lands around 11-12 dB.
    # SCS publishes 1.9-5.7 dB and states outright that PACTOR-III "is designed to
    # provide the benefits of both modes by MINIMIZING the CF". So the real waveform
    # carries a per-tone phase schedule we do not yet have (unknowns.U7_PHASE_REF).
    #
    # The useful consequence: the published CFR column is an OBJECTIVE FUNCTION. Any
    # candidate phase schedule can be scored against it offline, with no external
    # decoder and
    # no captures. Record the gap so progress on U7 is visible.
    print("       SL  tones   published    ours     gap")
    gaps = []
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        n_sym = 128
        tones = {cn: modem.differential_encode(
                     rng.integers(0, 2, (n_sym - 1) * s.bits_per_symbol, dtype=np.uint8),
                     s.bits_per_symbol)
                 for cn in s.channels}
        sig = modem.modulate_tones(tones, cfg)
        core = sig[cfg.sps * 8: -cfg.sps * 8]        # drop the pulse-shaping ramps
        cf_db = 20 * np.log10(np.max(np.abs(core)) / np.sqrt(np.mean(core ** 2)))
        gap = cf_db - s.crest_factor_db
        gaps.append(gap)
        print(f"       {sl}   {s.n_tones:2d}     {s.crest_factor_db:>4.1f} dB   "
              f"{cf_db:>4.1f} dB   {gap:>+5.1f} dB")

    check(all(np.isfinite(g) for g in gaps), "crest factor is measurable at every SL")
    # SL1 has only two tones, so even a naive schedule cannot be far off; if SL1's
    # crest factor were wild, the modulator itself would be broken.
    check(gaps[0] < 9.0,
          f"SL1 (2 tones) crest factor is sane -- {gaps[0]:+.1f} dB from published",
          "a 2-tone sum cannot legitimately have a huge crest factor; "
          "the modulator is wrong")
    print(f"\n       => mean gap {np.mean(gaps):+.1f} dB. This is the U7 phase-schedule")
    print("          deficit, and it is the objective function for closing it.")

    # --- 5. full loopback ------------------------------------------------------
    print("\n5. TX -> RX loopback, bit-exact in the clear (all speed levels)")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        n_data_sym = 71                       # short-cycle data field (see below)
        truth: dict[int, np.ndarray] = {}
        tones: dict[int, np.ndarray] = {}
        for cn in s.channels:
            b = rng.integers(0, 2, n_data_sym * s.bits_per_symbol, dtype=np.uint8)
            truth[cn] = b
            tones[cn] = modem.differential_encode(b, s.bits_per_symbol)

        sig = modem.modulate_tones(tones, cfg)
        rx = modem.demodulate_tones(sig, s.channels, n_data_sym + 1, cfg)

        errs = 0
        total = 0
        for cn in s.channels:
            soft = modem.differential_decode(rx[cn], s.bits_per_symbol)
            hard = (soft < 0).astype(np.uint8)
            n = min(hard.size, truth[cn].size)
            errs += int(np.sum(hard[:n] != truth[cn][:n]))
            total += n
        check(errs == 0,
              f"SL{sl} ({s.n_tones} tones, {s.modulation}): {total} bits, "
              f"{errs} errors",
              f"{errs}/{total} bit errors -- inter-tone leakage?")

    # --- 6. graceful degradation in noise --------------------------------------
    print("\n6. BER degrades monotonically with noise (SL6, the hardest case)")
    s = spec.SPEED_LEVELS[6]
    n_data_sym = 71
    truth, tones = {}, {}
    for cn in s.channels:
        b = rng.integers(0, 2, n_data_sym * s.bits_per_symbol, dtype=np.uint8)
        truth[cn] = b
        tones[cn] = modem.differential_encode(b, s.bits_per_symbol)
    sig = modem.modulate_tones(tones, cfg)

    bers = []
    for snr_db in (30.0, 20.0, 12.0, 6.0):
        noise = rng.normal(0, np.sqrt(np.mean(sig ** 2) / 10 ** (snr_db / 10)), sig.size)
        rx = modem.demodulate_tones(sig + noise, s.channels, n_data_sym + 1, cfg)
        errs = total = 0
        for cn in s.channels:
            hard = (modem.differential_decode(rx[cn], s.bits_per_symbol) < 0).astype(np.uint8)
            n = min(hard.size, truth[cn].size)
            errs += int(np.sum(hard[:n] != truth[cn][:n]))
            total += n
        bers.append(errs / total)
        print(f"       SNR {snr_db:>4.0f} dB -> raw BER {errs / total:.4f}")
    check(all(bers[i] <= bers[i + 1] + 1e-9 for i in range(len(bers) - 1)),
          "BER increases monotonically as SNR falls", f"{bers}")
    check(bers[0] < 1e-3, "BER is essentially zero at 30 dB SNR", f"{bers[0]}")

    # --- 7. the two ways of summing the tones agree ----------------------------
    # `modulate_tones` sums the tones as spectra and inverts once; the per-tone sum in
    # time is the definition of the waveform. The fast form is worth 3-8x a frame, and
    # it is only allowed to be there because it is the SAME waveform -- so render both
    # ways and measure, rather than asserting it in a comment. The long cycle is here
    # because it is where the two drift furthest apart: 4x the transform, 4x the
    # rounding.
    print("\n7. frequency-domain and time-domain synthesis are the same waveform")
    TOL = 1e-9                                    # on a 0.5-peak waveform

    s = spec.SPEED_LEVELS[6]
    for label, cycle_s in [("short", spec.CYCLE_SHORT_S), ("long", spec.CYCLE_LONG_S)]:
        n_sym = round(cycle_s * spec.SYMBOL_RATE_BD)
        tones = {cn: modem.differential_encode(
                     rng.integers(0, 2, (n_sym - 1) * s.bits_per_symbol, dtype=np.uint8),
                     s.bits_per_symbol)
                 for cn in s.channels}
        fast = modem.modulate_tones(tones, cfg)
        with time_domain_only():
            slow = modem.modulate_tones(tones, cfg)
        d = np.abs(fast - slow).max()
        check(d < TOL, f"SL6 {label} cycle ({cycle_s:g} s, {fast.size} samples): "
              f"max |diff| {d:.1e}", f"{d:.3e} exceeds {TOL:.0e}")

    # A whole frame, not just a tone set: FEC, interleave, differential map and
    # synthesis, rendered twice through the transmitter's own entry point.
    info = bytes((0x41 + 3 * i) & 0xFF for i in range(placement.HEADER.crc_bytes - 2))
    fast = placement.data_packet(info)
    with time_domain_only():
        slow = placement.data_packet(info)
    d = np.abs(fast - slow).max()
    check(d < TOL, f"SL2 header packet, whole frame: max |diff| {d:.1e}",
          f"{d:.3e} exceeds {TOL:.0e}")

    # The pulse the transmitter ACTUALLY USES on the air. `onair._render_cs` builds its
    # ModConfig with matched_pulse=True, and everything above this line runs the default
    # raised cosine -- so without this the gate covered every speed level and both cycle
    # lengths in a configuration the radio never emits. A reviewer measured it correct
    # at 3.4e-12 before it was guarded; that is the wrong order to do it in.
    for cn_set, label in ((placement.HEADER.tones, "SL2 header tones"),
                          (tuple(range(18)), "all 18 tones")):
        cfg_mf = modem.ModConfig(matched_pulse=True)
        n_sym = 73
        tones = {cn: modem.differential_encode(
                     rng.integers(0, 2, n_sym - 1, dtype=np.uint8), 1)
                 for cn in cn_set}
        fast = modem.modulate_tones(tones, cfg_mf)
        with time_domain_only():
            slow = modem.modulate_tones(tones, cfg_mf)
        d = np.abs(fast - slow).max()
        check(d < TOL, f"matched pulse (the on-air shape), {label}: max |diff| {d:.1e}",
              f"{d:.3e} exceeds {TOL:.0e}")

    # Two pulses with the SAME TAP COUNT must not share a cached spectrum: a raised
    # cosine is 3841 taps at rolloff 0.30 and at 0.15 alike, and keying the cache on
    # the count alone silently rendered the second with the first's spectrum.
    tones = {cn: modem.differential_encode(
                 rng.integers(0, 2, 160, dtype=np.uint8), 2) for cn in s.channels}
    for rolloff in (0.30, 0.15, 0.30):
        c = modem.ModConfig(rolloff=rolloff)
        fast = modem.modulate_tones(tones, c)
        with time_domain_only():
            slow = modem.modulate_tones(tones, c)
        d = np.abs(fast - slow).max()
        check(d < TOL, f"rolloff {rolloff:.2f} after a different pulse: "
              f"max |diff| {d:.1e}", f"{d:.3e} -- cached spectrum of the wrong pulse?")

    # Off the 120 Hz grid the rotation identity does not hold, and the answer must be
    # a slower waveform rather than a wrong one. At 8 kHz no transform length puts a
    # tone spacing on a whole bin (fs/120 = 66.67 samples), and rounding the step
    # would move every carrier by tens of Hz with nothing raised.
    slow_cfg = modem.ModConfig(sample_rate=8000)
    check(modem._transform_length(slow_cfg, 100000) is None,
          "8 kHz is refused the frequency-domain sum (fs/120 = 66.67 samples)")
    n_sym = 128
    tones = {cn: modem.differential_encode(np.zeros(n_sym - 1, dtype=np.uint8), 1)
             for cn in spec.SPEED_LEVELS[1].channels}
    sig = modem.modulate_tones(tones, slow_cfg)
    freqs = np.fft.rfftfreq(sig.size, 1 / slow_cfg.sample_rate)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    worst = 0.0
    for cn in spec.SPEED_LEVELS[1].channels:
        f0 = spec.channel_freq_hz(cn)
        band = (freqs > f0 - 50) & (freqs < f0 + 50)
        worst = max(worst, abs(freqs[band][np.argmax(psd[band])] - f0))
    check(worst < 2.0, f"8 kHz still lands its carriers ({worst:.2f} Hz worst offset)",
          f"worst offset {worst:.2f} Hz")

    # Tones of unequal length stay a modem error with a modem message, rather than
    # whatever the arithmetic underneath happens to raise first.
    try:
        modem.modulate_tones({3: np.ones(10, complex), 5: np.ones(11, complex)}, cfg)
    except ValueError as exc:
        check("same symbol count" in str(exc), "unequal tone lengths raise ValueError",
              f"wrong message: {exc}")
    except Exception as exc:
        check(False, "unequal tone lengths raise ValueError",
              f"raised {type(exc).__name__}: {exc}")
    else:
        check(False, "unequal tone lengths raise ValueError", "nothing raised")

    print("\n" + "-" * 68)
    assert not FAILURES, (f"{len(FAILURES)} FAILED of {CHECKS}: "
                          + "; ".join(FAILURES))
    print(f"RESULT: ALL PASS ({CHECKS} checks)")
    print("\nNOTE: a clean loopback proves our modulator and OUR demodulator agree.")
    print("      It says NOTHING about whether a real receiver will agree.")
