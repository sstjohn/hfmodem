# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Self-consistency checks on the transcribed PACTOR-III constants.

The point: a transcription error in a constant would otherwise surface much later
as "no receiver will decode our signal", which is an appallingly bad error
message. The
spec is redundant enough to check itself, so we do that here, before any DSP
exists.

Run:  .venv/bin/python -m pytest packages/hfmodem/hfmodem/tests/shrike/test_spec.py
"""

from __future__ import annotations

from hfmodem.shrike import spec, unknowns  # noqa: E402

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


def test_all() -> None:
    print("PACTOR-III constant self-consistency\n")

    # --- 1. Physical data rate must equal tones * bits/symbol * baud -----------
    # This is the check that independently confirmed the tone maps.
    print("1. PDR == n_tones * bits_per_symbol * symbol_rate")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        computed = s.n_tones * s.bits_per_symbol * spec.SYMBOL_RATE_BD
        check(computed == s.pdr_bps,
              f"SL{sl}: {s.n_tones} tones x {s.bits_per_symbol} bit x 100 Bd = "
              f"{computed:.0f} == PDR {s.pdr_bps}",
              f"got {computed}, spec says {s.pdr_bps}")

    # --- 2. Net data rate must equal the LONG-cycle payload over 3.75 s --------
    # NDR is quoted against the data mode, not the short cycle. If this holds, our
    # reading of both the payload table and the cycle durations is right.
    print("\n2. NDR == long-cycle payload / 3.75 s")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        computed = s.payload_long * 8 / spec.CYCLE_LONG_S
        check(abs(computed - s.ndr_bps) < 0.15,
              f"SL{sl}: {s.payload_long} B * 8 / 3.75 s = {computed:.1f} "
              f"== NDR {s.ndr_bps}",
              f"got {computed:.2f}, spec says {s.ndr_bps}")

    # --- 3. Every tone map is symmetric about 1500 Hz --------------------------
    print("\n3. tone maps symmetric about channel 8.5 (= 1500 Hz)")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        mirrored = tuple(sorted(17 - c for c in s.channels))
        check(mirrored == tuple(sorted(s.channels)),
              f"SL{sl}: {list(s.channels)} is mirror-symmetric")

    mid = (spec.channel_freq_hz(8) + spec.channel_freq_hz(9)) / 2
    check(mid == spec.CENTER_FREQ_HZ,
          f"midpoint of channels 8/9 = {mid:.0f} Hz == stated centre "
          f"{spec.CENTER_FREQ_HZ:.0f} Hz")
    check(spec.channel_freq_hz(0) == 480.0 and spec.channel_freq_hz(17) == 2520.0,
          "channel 0 = 480 Hz and channel 17 = 2520 Hz")

    # --- 4. Every speed level must carry the header/CS channels ---------------
    # Channels 5 and 12 carry the variable headers and the control signals, so a
    # speed level that omitted them could not be signalled at all.
    print("\n4. every speed level contains the header/CS channels 5 and 12")
    for sl in sorted(spec.SPEED_LEVELS):
        s = spec.SPEED_LEVELS[sl]
        check(all(c in s.channels for c in spec.VH_CHANNELS),
              f"SL{sl} contains channels {list(spec.VH_CHANNELS)}")

    # --- 5. Carrier swap is an involution covering all 18 channels -------------
    print("\n5. carrier swap is an involution over all 18 channels")
    swap = spec.CARRIER_SWAP
    check(set(swap) == set(range(18)), "swap map covers channels 0..17",
          f"covers {sorted(swap)}")
    check(all(swap[swap[c]] == c for c in swap), "swap is its own inverse")
    check(swap[5] == 12 and swap[12] == 5,
          "swap maps 5 <-> 12, so the header/CS channels stay header/CS channels")
    # A speed level's tone SET does not move under the swap -- only which virtual
    # carrier, and so which symbol clock, each channel is on. That is what lets a
    # receiver read a packet's header block without first knowing the arrangement,
    # which `p3rx.header_of` relies on.
    for sl, s in sorted(spec.SPEED_LEVELS.items()):
        check(set(s.channels) == {swap[c] for c in s.channels},
              f"SL{sl}'s channels are closed under the swap",
              f"{sorted(s.channels)} -> {sorted({swap[c] for c in s.channels})}")

    # --- 6. Header tables ------------------------------------------------------
    print("\n6. header tables")
    check(len(spec.VARIABLE_HEADERS) == 16, "16 variable headers")
    check(len(spec.CONSTANT_HEADERS) == 16, "16 constant headers")
    check(all(h <= 0xFFFFFFFF for h in spec.VARIABLE_HEADERS),
          "variable headers are 32-bit")
    check(all(h <= 0xFFFF for h in spec.CONSTANT_HEADERS),
          "constant headers are 16-bit")
    check(len(set(spec.VARIABLE_HEADERS)) == 16,
          "all 16 variable headers are distinct")

    # The one that is NOT ok -- and we assert the anomaly rather than hide it, so
    # that if someone "helpfully" edits the table this test tells them why not.
    n_distinct_ch = len(set(spec.CONSTANT_HEADERS))
    check(n_distinct_ch == 15,
          "constant headers contain exactly ONE duplicate, as published "
          "(CH7 == CH11 == 0x5a3c) -- see unknowns.CH_DUPLICATE",
          f"expected 15 distinct values, found {n_distinct_ch}")
    check(spec.CONSTANT_HEADERS[7] == spec.CONSTANT_HEADERS[11] == 0x5A3C,
          "the duplicate is specifically CH7/CH11 (not something we mistyped)")

    # 16 constant headers for the 16 channels that are not 5 or 12.
    non_vh = [c for c in range(18) if c not in spec.VH_CHANNELS]
    check(len(non_vh) == len(spec.CONSTANT_HEADERS),
          f"{len(non_vh)} non-header channels == {len(spec.CONSTANT_HEADERS)} constant headers")

    # --- 7. Number of channels -------------------------------------------------
    print("\n7. channel count and bandwidth")
    check(spec.SPEED_LEVELS[6].n_tones == spec.N_CHANNELS,
          "SL6 uses all 18 channels")
    occupied = spec.channel_freq_hz(17) - spec.channel_freq_hz(0)
    check(occupied == 2040.0,
          f"outermost tone spacing = {occupied:.0f} Hz (2.2 kHz occupied incl. skirts)")

    # --- 8. Status byte --------------------------------------------------------
    print("\n8. status byte packing")
    check(spec.status_byte(0, spec.DataType.ASCII_8BIT) == 0x00,
          "packet 0, ASCII => 0x00")
    check(spec.status_byte(3) == 0x03, "packet counter is bits 0-1")
    check(spec.status_byte(0, spec.DataType.PMC_ENGLISH) == (0b110 << 2),
          "data type occupies bits 2-4",
          f"got {spec.status_byte(0, spec.DataType.PMC_ENGLISH):#010b}")
    check(all((spec.status_byte(0, dt) >> 2) & 0b111 == dt for dt in range(8)),
          "every data type round-trips through bits 2-4")
    check(spec.status_byte(0, long_cycle_request=True) == 0x20,
          "cycle-length suggestion is bit 5")
    check(spec.status_byte(0, changeover_request=True) == 0x40,
          "changeover request is bit 6")
    check(spec.status_byte(0, qrt=True) == 0x80, "QRT is bit 7")

    # --- 9. Sub-band lead ------------------------------------------------------
    print("\n9. sub-band lead")
    # Length is the whole invariant: `Path.subband_lead` hands the table straight
    # out and `clock_offsets` zips it against the tone tuple, so an entry of the
    # wrong length would truncate silently rather than raise -- half a comb on the
    # wrong clock and no error anywhere.
    from hfmodem.shrike import placement
    for sl, lead in spec.SUBBAND_LEAD.items():
        for tbl, tag in ((placement.SPEED_PATHS, "short"),
                         (placement.LONG_PATHS, "long")):
            n = len(tbl[sl].tones)
            check(len(lead) == n,
                  f"SL{sl} {tag}: one lead per tone ({n})",
                  f"table has {len(lead)}")
    check(set(spec.SUBBAND_LEAD) == {1, 2},
          "the two narrow levels stagger their combs and no other does")
    for sl, lead in spec.SUBBAND_LEAD.items():
        check(min(lead) == 0.0,
              f"SL{sl}: the lead is relative -- some carrier gives up nothing")
    check(spec.SUBBAND_LEAD[1] == (0.5, 0.0),
          "SL1 leads with channel 5, by half a symbol",
          "measured on the reference entry packet at +0.514 symbol and on both "
          "of its changeover frames at +0.512 and +0.505")

    # --- report ---------------------------------------------------------------
    print("\n" + "-" * 68)
    print("Unknown register (assumptions are NOT facts):\n")
    print(unknowns.summary())
    print("-" * 68)

    assert not FAILURES, (f"{len(FAILURES)} FAILED of {CHECKS}: "
                          + "; ".join(FAILURES))

    print(f"\nRESULT: ALL PASS ({CHECKS} checks)")
