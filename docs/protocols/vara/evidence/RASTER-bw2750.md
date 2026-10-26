<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# BW2750 raster by speed level, off the high speed capable bench tapes

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

`raster_bw2750.py` (this folder) produces every number below; it reads the tapes and writes nothing.
Bursts are cut on `measured_on/off` from `overs.jsonl`. Instances are ISS DATA keyings: the manifest
tape's A leg where the level is there, otherwise ladder `__b2a` keyings whose floor is −200 and whose
burst RMS equals the clean value. Their PSK cells come out constant-modulus to <1e-4, so the noise gate
was off for the whole burst. Two instances are noisy and flagged: L9's second and both of L14's (no clean
L14 exists). "Tape" is shortened to the manifest name or the ladder timestamp.

Three methods do most of the work:

- **CP fold.** Correlate the burst with itself at lag Tu over a 64-sample window, then fold at every period from 1030 to 1600. The fold's peak gives the symbol period and its half-height width gives the CP.
- **Cross-frame product.** Every cell carries a phase twiddle that is fixed per cell position, so its raw phases are uniform. The product of two frames' cells at the same position cancels the twiddle and leaves the modulation alphabet. The class counts after snapping to a 2M grid show the order.
- **Header ramp removal.** The first symbols carry no twiddle, so fitting their linear phase puts a burst into a canonical frame. That makes twiddle comparisons across levels possible.

## Controls

**L4 (ISS, record 103).** Manifest A, on 1955838, 208364 samples. There are 406.96 blocks of 512 samples. Each is one-hot (median energy share of the top bin 0.999), and the lit bins are 6..25 (562.5–2343.75 Hz, all 20 used). Autocorrelation shows no CP: +0.004 at 1280 and +0.054 at 1024, which is the one-hot structure repeating, not a prefix. The block count is 407 = 12 training + 395 columns, and the silent lead-in is not in the cut. kestrel's `decode_over(level=103)` returns a CRC-clean 92-byte frame. The pipeline reproduces record 103 exactly.

**IRS ack.** 16377 samples, 15 on the manifest tape. None of them has a 512/1024/1152 structure (lag +0.11/−0.05/+0.08, top lag 698) or a one-hot one (top-bin share 0.42). The acks are not copies of each other: the lowest similarity to the first is 0.12. No record describes the ack, so there is nothing to reproduce. It is not on any data raster measured here.

## Per level

Tone levels (L5–L10) share one geometry. Each is a set of single QPSK tones centred on 1500 Hz, one symbol every 512 samples: a period scan over 488–536 is best at 512 for every level. There is no CP, since the lag-512 correlation is under ±0.011. Adjacent tones are sampled half a symbol apart; at L5 the three tones are a third apart. The tones are not on the 46.875/93.75 Hz grid. Raw and symbol-to-symbol phases are uniform (M4 resultant ≤0.05), and the cross-frame product is on a 4-grid (M2 ≤0.07). That makes the modulation coherent QPSK under a fixed twiddle, not differential. The burst does not open with L4's training: normalised cross-correlation against L4's first 7168 samples is 0.10–0.18. No symbol is the same in two frames, so no fixed preamble shows either.

### L4 (control, above)

### L5
| | |
|---|---|
| instances | 050512Z B 2377737; 050336Z B 2136503 |
| burst | 195823 median (n 11) |
| symbol | 512, CP 0, 93.75 Bd; sampling phases 159/335/495 |
| tones | 3 at 600.0 / 1500.0 / 2400.0 Hz, spacing 900; −3 dB lobe 97 Hz |
| symbols | 383 live per tone |
| modulation | QPSK: product M4 0.998, M2 0.024; amplitude cv 0.011 |
| cells | 1086 = 362 × 3 (derived, rec4 coded 2172) |

### L6
| | |
|---|---|
| instances | 050204Z B 2277439; 050336Z B 2359418 |
| burst | 190299 (n 34) |
| symbol | 512, CP 0; phases 161/417 alternating |
| tones | 4 at 510 / 1170 / 1830 / 2490 Hz, spacing 660 |
| symbols | 372 live per tone |
| modulation | QPSK: M4 0.998, M2 0.065; cv 0.011 |
| cells | 1416 = 354 × 4 (derived) |

### L7
| | |
|---|---|
| instances | 050204Z B 1186505; 050512Z B 1192602 |
| burst | 185236 (n 10) |
| symbol | 512, CP 0; phases 279/23 alternating |
| tones | 6 at 449.8 … 2550.5 Hz, spacing 420 |
| symbols | 362 live per tone |
| modulation | QPSK: M4 0.923, M2 0.045; cv 0.068 (ISI from closer tones) |
| cells | 2082 = 347 × 6 (derived) |

### L8
| | |
|---|---|
| instances | manifest A 2195449; 050336Z B 1194765 |
| burst | 183974 (n 6) |
| symbol | 512, CP 0; phases 271/15 alternating |
| tones | 8 at 380.4 … 2620.3 Hz, spacing 320 |
| symbols | 359 live per tone |
| modulation | QPSK: M4 0.925, M2 0.051; cv 0.067 |
| cells | 2760 = 345 × 8 (derived) |

### L9
| | |
|---|---|
| instances | 045941Z B 1142793; 045304Z B 695690 (noisy, floor −24 dBFS) |
| burst | 182185 (n 11) |
| symbol | 512, CP 0; phases 267/11 alternating |
| tones | 10 at 374.9 … 2624.9 Hz, spacing 250 |
| symbols | 356 live per tone |
| modulation | QPSK: M4 0.714, M8 0.272, M2 0.024 (second instance noisy); cv 0.045 |
| cells | 3430 = 343 × 10 (derived) |

### L10
| | |
|---|---|
| instances | 045556Z B 1118866; 045139Z B 1067991 |
| burst | 182418 (n 23) |
| symbol | 512, CP 0; phases 254/510 alternating |
| tones | 13 at 240.2 … 2760.4 Hz, spacing 210 |
| symbols | 356 live per tone |
| modulation | QPSK: M4 0.968, M2 0.049; cv 0.045 |
| cells | 4472 = 344 × 13 (derived, assumes rate 1/2) |

The 1152 family (L11–L17) is one raster at every level:

- **Symbol.** Tu = 1024, CP = 128 (fold half-width 127–131), period 1152 = 41.67 Bd. The lag correlation is 0.103–0.117 at 1024 against |r| ≤ 0.012 at 1280 and at 512.
- **Carriers.** 59 on bins 3..61 of the 1024-point FFT, 140.625–2859.375 Hz at 46.875 Hz spacing. Occupancy is flat to 0.00 dB at the PSK levels. No bin stands out, and no unmodulated tone is present.
- **FFT window.** It must sit ≥16 samples inside the CP. PSK amplitude cv is 0.023 / 0.004 / 0.0005 / 1e-5 at 0 / 8 / 12 / 16 samples in, so symbol edges are blended over about 16 samples.
- **Header.** The first 4 symbols (PSK) or 5 (QAM) have no twiddle. Their raw cells lie on the level's own grid (R 0.98–1.00 per symbol, QPSK / 8PSK / QAM diagonals), and they change from frame to frame. The two L14 frames were both the 5th DATA frame of their sessions, and their symbols 0–4 were identical while the data were not. Every other pair differed. No level opens with L4's training (cross-correlation 0.12–0.20).
- **Fixed reference (PSK).** Symbol 4 is identical in every PSK frame compared, at every level and in every session.
- **Twiddle.** From symbol 4 (PSK) or 5 (QAM) every cell is multiplied by a twiddle on the 2π/128 grid: 100% of L12's cells, mod 90°, lie within 0.1 step, against 20% by chance. The sequence is the same at every level. At QAM it starts one symbol later, and the best cell shift against L12 is exactly +59 (R 1.000, median 0.02). It restarts every frame.
- **Pilots (QAM only).** At symbols t ≡ 5 (mod 8) the odd-index carriers (bins 4, 6 … 60, 29 cells) are pilots, and at t ≡ 6 the even-index ones (bins 3, 5 … 61, 30 cells). That is 20 pairs, t = 5…158, so 1180 = 20 × 59 cells, all hits 1.00 with 0.004 elsewhere. They have unit LUT modulus (0.909 of the 16QAM corner, 0.808 of the 32-cross corner) and are identical across frames and across L16/L17.
- **Frame period.** The double L17 burst is 313 symbols: the pilot pairs run on 157/158 → 159/160, and the twiddle restarts at +154. So it is 5 header symbols + 2 × 154 per frame.

### L11
| | |
|---|---|
| instances | 045304Z B 1117460; 060325Z B 3293984 |
| burst | 176196 (n 15) → 153 symbols |
| modulation | QPSK: product 4 classes of 8 (2377/0/2198/0/2322/0/2130/0), rms 0.12°; constant modulus |
| cells | 8673 = 147 × 59 (derived) |

### L12
| | |
|---|---|
| instances | manifest A 2408974; 045513Z B 1121093 |
| burst | 176179 (n 14) → 153 |
| modulation | QPSK: 4 of 8 classes, rms 0.10° |
| cells | 8673 |

### L13
| | |
|---|---|
| instances | 055838Z B 3444906; 060010Z B 3314418 |
| burst | 176194 (n 11) → 153 |
| modulation | QPSK: 4 of 8 classes, rms 0.05° |
| cells | 8673 |

### L14
| | |
|---|---|
| instances | 060010Z B 1323766; 060143Z B 1321284 (both noisy, amplitude cv 0.066) |
| burst | 176156 (n 14) → 153 |
| modulation | 8PSK: against clean L12, 8 even classes of 16 hold 90% (≈1000 each vs ≈100), rms 2.3° |
| cells | 8673 |

### L15
| | |
|---|---|
| instances | manifest A 2618120; 055838Z B 1485478 |
| burst | 176204 (n 17) → 153 |
| modulation | 8PSK: 8 of 16 classes (≈1130 each, 0 odd), rms 0.06°; the header is raw 8PSK |
| cells | 8673 |

### L16
| | |
|---|---|
| instances | manifest A 2828308; 055838Z B 1695671 |
| burst | 183101 (n 18) → 159 |
| modulation | 16QAM: rings 0.33 / 0.75 / 1 plus pilot 0.91; 16 points used, EVM 0.3% after twiddle removal |
| cells | 7847 = 133 × 59 data + 1180 pilots = 153 × 59 |

### L17
| | |
|---|---|
| instances | manifest A 3047432; 055838Z B 1914290; double 4380167, 4781078 |
| burst | 183113 (n 21) → 159; double 360530 (n 4) → 313 |
| modulation | 32-cross: rings 0.246 / 0.546 / 0.729 / 0.871 / 1 (LUT 0.3 / 0.671 / 0.9 / 1.082 / 1.237 ÷ 1.237) plus pilot 0.81; 32 points, EVM 0.3% |
| cells | 7847 + 1180 per frame |

Overhead: PSK 153 = 5 + 147 + 1, QAM 159 = 5 + 153 + 1, double 313 = 5 + 2 × 154. So one 59-cell symbol per frame is unaccounted for by coded cells and pilots, and where it sits is not measured.

## Proposed Record rows

M = measured, D = derived from `payload_len` + 3 (CRC + control byte, calibrated on BW2300 L4/L10/L12/L14/L16), from the BW2300 `coded` law and from the measured cell counts, U = unknown. Keys follow the existing 100 + record convention.

| key (L) | law | n_info | coded | frame_bytes | bpc | dw50 | cp | span | first_bin | stride | ncols | lead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 103 (4) | index M | 736 | 1484 | 92 M | 4 | 512 M | 0 M | 20 M | 6 M | 1 | 395 M | 14 (12 M + 2) |
| 104 (5) | tone (new) M | 1080 D | 2172 D | 135 D | 2 M | 512 M | 0 M | 3 tones M | — (1500 ± k·900 Hz) M | 900 Hz M | 383 live M | U |
| 105 (6) | tone | 1410 D | 2832 D | 176 D | 2 M | 512 | 0 | 4 M | 660 Hz grid M | 660 Hz | 372 M | U |
| 106 (7) | tone | 2076 D | 4164 D | 259 D | 2 M | 512 | 0 | 6 M | 420 Hz grid M | 420 Hz | 362 M | U |
| 107 (8) | tone | 2754 D | 5520 D | 344 D | 2 M | 512 | 0 | 8 M | 320 Hz grid M | 320 Hz | 359 M | U |
| 108 (9) | tone | 3424 D | 6860 D | 428 D | 2 M | 512 | 0 | 10 M | 250 Hz grid M | 250 Hz | 356 M | U |
| 109 (10) | tone | 4466 D | 8944 D | 558 D | 2 M | 512 | 0 | 13 M | 210 Hz grid M | 210 Hz | 356 M | U |
| 110 (11) | const M | 8667 D | 17346 D | 1083 D | 2 M | 1152 M | 128 M | 59 M | 3 M | 1 M | 153 M | 5 M |
| 111 (12) | const | 11556 D | 17346 D | 1444 D | 2 M | 1152 | 128 | 59 | 3 | 1 | 153 | 5 |
| 112 (13) | const | ≈13867 D (puncture U) | 17346 D | 1733 D | 2 M | 1152 | 128 | 59 | 3 | 1 | 153 | 5 |
| 113 (14) | const | 17338 D | 26019 D | 2167 D | 3 M | 1152 | 128 | 59 | 3 | 1 | 153 | 5 |
| 114 (15) | const | ≈20805 D (puncture U) | 26019 D | 2600 D | 3 M | 1152 | 128 | 59 | 3 | 1 | 153 | 5 |
| 115 (16) | const | ≈26147 D (puncture U) | 31388 D | 3268 D | 4 M | 1152 | 128 | 59 | 3 | 1 | 159 M | 5 M |
| 116 (17) | const | ≈31378 D (puncture U) | 39235 D | 3922 D | 5 M | 1152 | 128 | 59 | 3 | 1 | 159 / 313 M | 5 M |

`pos` follows from bpc (1 / 2 / 3 / 4). The 16QAM corner 1.1 and the 32-cross moduli are MEASURED. For `il`: L5–L9 carry exactly BW2300 rec4–8's frame sizes, so `il` 4–8 is D (untested). L10–L17 have no seeds on hand (U). Twiddle, header, pilot map and the extra per-frame symbol are fields the Record type does not have yet.

The rows match existing records. L4 is record 103, confirmed. L5–L9 carry BW2300 rec4–8's coding on a new tone raster. L10 matches no existing record. L11–L15 are BW2300 rec9–13's structure with 59 carriers instead of 49 (8673 = 147 × 59 against 7203 = 147 × 49), and the prediction in RECORD-SEMANTICS §4 (L11: 8673 / 17346 / 8667) holds. L16/L17 are rec14/rec15 with 59 carriers: 133 × 59 data cells, and the 980 = 20 × 49 "forced cells" become 1180 = 20 × 59 pilots.

## Disagreements with sigidwiki

No cell of the wiki row is contradicted by the tape. L4 20 bins, L5–L10 94 Bd with 3/4/6/8/10/13 carriers of 4PSK, L11–L17 42 Bd with 59 carriers, and 4PSK / 8PSK / 16QAM / 32QAM at L11–13 / 14–15 / 16 / 17 are all confirmed. Four points go beyond the table:

- "FSK" at L4 is one-hot index modulation.
- "42 Bd" is 41.67 Bd, a 1152-sample symbol.
- The L5–L10 "carriers" are QPSK tones at non-grid spacings (900 … 210 Hz) with staggered timing, not OFDM subcarriers.
- The net rates fit frame bits over ≈3.58 s, while the bursts last 3.67 s (L11–15) and 3.81 s (L16–17). That is a throughput convention, not a raster disagreement.

## Disagreements with our current records

1. **01 §2750, `varahf2300.py` comment at the BW2750 rows, and review correction 3.** These say Tu ≈ 1280, Ts ≈ 1400–1450 and "dw50 = 1152 does not carry over to 2750". The tape says otherwise: Tu 1024, CP 128, period 1152 at L11–L17, with correlation 0.00 at lag 1280. The 46.875 Hz reading was right.
2. **01 §2750 occupied bandwidth "~2.7 k (2742/2713)".** The carrier centres span 140.6–2859.4 Hz, about 2.81 kHz edge to edge.
3. **RECORD-SEMANTICS §1 / 01 const law "no pilots", with forced cells ignored.** The QAM levels carry 1180 pilots per frame on a period-8, alternating-parity map.
4. **const `burst_length` = ceil(cells/span) · dw50, no preamble.** Real bursts add 5 header/reference symbols plus one symbol per frame, and L17 keys two frames in one burst (313 symbols).
5. **The const path as implemented** has no twiddle, header, pilots or edge guard, and `_demod_const` windows at sample 0. The measured twiddle is the doc'd exp(j2πS/128). It is per cell position, shared across levels, offset by one symbol at QAM and restarted per frame. Without all of these the path cannot match.
6. **06:114 (rec4–8 "dw50 512, span 16").** At BW2750, L5–L9 carry rec4–8's frame sizes on 3–10 non-grid tones, not a 16-bin comb. This applies at BW2750 only; BW2300's L5–L9 raster was not measured here.
7. **RECORD-SEMANTICS §4 "carrier count and bins … unknown" for BW2750 above base.** Measured above, except the extra per-frame symbol, the header's content, `il` and the puncturing.
