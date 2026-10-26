<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# BW500 physical raster, levels 4-13, from the high speed capable bench tapes

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

`raster_bw500.py` produces every number below (about 40 s; prints tables, writes nothing). Bursts are cut on
`overs.jsonl` `measured_on/off`. A burst is used only if its floor is digital silence and its `burst_rms` sits at
the level's modal value. A burst hit by gated noise reads louder, so that test removes it even on a noisy leg.
Levels 5, 6, 7, 9 and 10 exist only as B (ISS) keyings on the ladders' b2a legs, so their instances come from
there, all passing that test. Levels 4, 8, 11, 12 and 13 also have manifest-tape instances.

Two waveform families are on tape. **L4-L6** are the base family: single-carrier sub-bands, 512-sample columns,
no CP, de-rotated by `grid480`. **L7-L13** are CP-OFDM: 1024 + 128 samples, 11 carriers on bins 27..37 of the
1024-point FFT, a 5-symbol lead, and 154-symbol frames with scattered pilots at the QAM levels.

## Controls

- **L4 ISS data** (`N0CALL-W9SSJ-bw500-counter65536` A @1948805): `varahf500.decode_burst` returns a CRC-clean
  46-byte frame (43 bytes delivered). The sub-band measurement below reproduces the record: 93.75 Bd, sub-bands at
  1350/1650, and DBPSK collapse 0.996 on the matching `grid480` row against ≤0.15 on the wrong row.
- **IRS ack** (16376 samples, same tape, B @2158633 and @2364457). A 1500 Hz carrier dominates (the −3 dB width is
  one 11.7 Hz bin), −20 dB occupancy is 1148-1805 Hz, and the 1024/1152 leakage is 27 dB against 103 dB for data.
  That is the control waveform in 01 §1.5, not the data raster. The 22520-sample keyings some ISS stations make at
  L7-L13 (for example L11 A @20760349) are the same control waveform (28 dB).

## Per-level measurements

"Leakage" is the median power in bins 27..37 over the power in the neighbouring bands, read on a given (Tu, P)
grid at the best offset. A clean OFDM raster reads about 100 dB. The wrong raster reads about 20 dB.

### L4 (record 3), control

| item | value |
|---|---|
| instances | counter65536 A @1948805; 053832Z-ladder B @8058673 |
| burst | 205389 / 205387 (modal 205400, n 3) |
| symbol | 512 samples, no CP; \|z\|² line 93.75 Hz (×254 median) in both sub-bands |
| carriers | 2 sub-bands at 1350 / 1650 Hz |
| symbols | 401 live columns (403 = 9 lead + 394 frame, per varahf500) |
| modulation | grid480 row 0 at 1350 Hz, row 1 at 1650 Hz: M2 0.996, M4 0.986; wrong row ≤0.15 → **DBPSK** |

### L5 (record 4)

| item | value |
|---|---|
| instances | 053832Z-ladder B @2237489, @3404588 (and @5771825) |
| burst | modal 205600 (n 24, 203964-205905) |
| symbol | 512, no CP; 93.75 Hz line ×229-256 |
| carriers | 2 sub-bands at 1350 / 1650 Hz; spectrum identical in shape to L4 |
| symbols | 402 live columns in both sub-bands |
| modulation | same grid480 rows and index law as L4: **M4 0.986, M2 0.03-0.07** → **DQPSK** |

### L6 (record 5)

| item | value |
|---|---|
| instances | 053832Z-ladder B @2472384, @3639021 (also @4500692, @6963682) |
| burst | modal 196000 (n 19, 194959-196446) |
| symbol | 512, no CP; 93.75 Hz line ×236-303 in all three sub-bands |
| carriers | **3 sub-bands at 1312.5 / 1500 / 1687.5 Hz** (187.5 Hz apart; −3 dB humps about 140 Hz wide against L4's 250 Hz) |
| symbols | 383 live columns |
| modulation | 1312.5 on grid480 row 0 and 1500 on row 1: **M4 0.999**, M2 ≤0.11 → **DQPSK**. 1687.5 locks on neither row (≤0.17): its reference row is not in grid480 |

### L7-L13, the shared raster

| item | value (every instance, 2 per level) |
|---|---|
| leakage at (Tu,P) = (1024,1152) / (1024,1280) / (1280,1408) / (1008,1152) / (512,512) / (2048,2304) | **103-104** / 18-20 / 19-20 / 16-18 / 17 / 19-22 dB |
| CP-autocorrelation fold at lag 1024 | period 1151.5-1152.5 samples, contrast 0.09 |
| symbol | **Tu 1024, CP 128, P 1152 → 41.667 Bd**. The ISI-free window is 99 of the 128 CP samples, so about 29 samples are shared with the previous symbol's taper (the overlap-add in 01) |
| FFT origin | the ISI-free window's last position (the boundary) + 127 puts every PSK cell on the 2π/128 lattice |
| carriers | **bins 27..37**, 1265.6-1734.4 Hz, spacing 46.875 Hz, all within 1 dB. Bins 24-26 and 38-40 sit 47-63 dB down. −3 dB occupancy 1254-1746 Hz. No carrier stands out |
| first symbol | already on the raster (ISI 100-102 dB at symbol 0); nothing on the 512 grid ahead of it |
| preamble vs L4 | the opening 4608 samples cross-correlate with L4's opening at 0.25-0.36. That is the chance floor: L4 vs L4 reads 0.34, and same-level pairs read 0.25-0.31. No shared preamble |
| lead | 5 symbols. Symbols 0-3 carry the level's own constellation under the PSK twiddle and vary per over. **Symbol 4 is identical in every PSK burst at every PSK level** (L7-L11, 8 ladders plus the manifest tape). At L12/L13 symbol 4 varies and is under the body twiddle |
| frame | **154 symbols at QAM, MEASURED**. In L13's double-length bursts the fixed-cell map repeats at a shift of exactly 154 (253 of 253 cells; 44 at ±1). No lead or reference comes before frame 2 |

### L7 (record 6)

| item | value |
|---|---|
| instances | 052917Z-ladder B @2366113; 053832Z-ladder B @4725719 (the only 2 clean) |
| burst | modal 176200 (n 11) = 153 symbols |
| modulation | 1 ring; 100% on the lattice; residual mod 90° after L8's twiddle = {0} in lead and body → **QPSK** |

### L8 (record 7)

| item | value |
|---|---|
| instances | counter65536 A @2181689; 053528Z B @1172518; 053832Z B @1184473 (10 in the census) |
| burst | modal 176200 (n 19; one 181464 with a noise edge) = 153 symbols |
| modulation | **QPSK**, as L7; defines the reference twiddle mod 90° |

### L9 (record 8)

| item | value |
|---|---|
| instances | 053136Z B @1164311; 053357Z B @1172582 (6 in the census) |
| burst | 176200 = 153 symbols |
| modulation | **QPSK**; residual {0} |

### L10 (record 9)

| item | value |
|---|---|
| instances | 052707Z B @1108748; 052607Z B @1109156 (8 in the census) |
| burst | 176200 = 153 symbols |
| modulation | 1 ring; residual {0, 45}° (877/795) → **8PSK** |

### L11 (record 10)

| item | value |
|---|---|
| instances | counter65536 A @20549166; 052812Z B @1136888 (5 in the census) |
| burst | 176200 = 153 symbols |
| modulation | **8PSK**, residual {0, 45}° |

### L12 (record 11)

| item | value |
|---|---|
| instances | counter65536 A @20336939; 052202Z B @1108495 (12 in the census) |
| burst | 183100 (n 16, 183093-183131) = 159 symbols |
| modulation | data rings 0.3333 : 0.7454 : 1 = 1 : √5 : 3 (**square 16QAM**, corner 1.1), plus a unit ring at 0.9091 = 1.0/1.1. Body twiddle mod 90° voted from inner-ring cells of 11 bursts; the held-out manifest burst de-rotates **100% onto the odd-integer lattice, 16 points** |
| pilots | frame rows 0 and 1 mod 8: odd bins, then even bins, 11 cells per 8 symbols, 20 pairs = **220 per frame**. Modulus 1.0 in LUT units, values identical across sessions and identical to L13's |
| extra | **44 fixed unit-modulus cells** per frame at level-specific positions |

### L13 (record 12)

| item | value |
|---|---|
| instances | counter65536 A @2600000, @2813230 (6 single, 6 double in the census; 28 for the twiddle vote) |
| burst | **360500 (n 42) = 313 symbols** = 5 + 2×154; single 183100 (n 7) = 159 |
| modulation | data rings 0.2425 : 0.5423 : 0.7276 : 0.8745 : 1 = √2 : √10 : √18 : √26 : √34 (**32-cross**), unit ring 0.8085 = 1/1.237. The held-out burst de-rotates **100% onto the lattice, 32 points, max coordinate 5** |
| pilots / extra | 220 pilots and 44 fixed unit cells per frame. L13's extra positions share 13 of 44 with L12's |

**Differential vs coherent.** On every level the raw cells and their symbol-to-symbol differences are uniform in
phase (|E e^{jMθ}| ≤ 0.07 for M = 1..8). The per-cell twiddle causes that. Once a fixed per-cell rotation is
removed, one learned from other bursts in other sessions, the cells fall on the absolute constellation. The QAM
pilots are one per carrier every 8 symbols, the pattern a coherent equaliser needs, so coherent is the reading the
tape supports. The tape alone cannot exclude a quadrant-differential labelling; a decode settles that.

## Cell accounting and proposed Record rows

| L | rec | burst sym | lead | frame sym | frame cells | pilots | extra | data = coded/bpc | payload max | N/8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4 | 402 cols | 9? | 394? | 788? | - | 38? | 1500/2 = 750 | 90 | 93 |
| 6 | 5 | 383 cols | 9? | 374? | 1122? | - | 36? | 2172/2 = 1086 | 132 | 135 |
| 7-9 | 6-8 | 153 | 5 | 148 | 1628 | 0 | 44 | 3168/2 = 1584 | 194/260/312 | 197.25/263/315.5 |
| 10-11 | 9-10 | 153 | 5 | 148 | 1628 | 0 | 44 | 4752/3 = 1584 | 392/471 | 395/474 |
| 12 | 11 | 159 | 5 | 154 | 1694 | 220 | 44 | 5720/4 = 1430 | 591 | 594.4 |
| 13 | 12 | 159 / 313 | 5 | 154 | 1694 | 220 | 44 | 7150/5 = 1430 | 710 per frame | 713.5 |

At every level the delivered payload is N/8 rounded down, minus 3: payload + marker + CRC, the BW500 convention.
The 1430 data cells + 220 pilots + 44 extras fill the QAM frame exactly. At PSK, 1584 + 44 fills 148 symbols only
if the lead is 5 there too, and no double-length PSK burst is on tape to show it.

Marks: **M** measured, **D** derived, **U** unknown. `n_info`/`coded` are the held tablegen values (D) and are
not re-measured.

| rec (L) | law | n_info | coded | frame_bytes | bpc | dw50 | cp | span | first_bin | stride | ncols | lead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 (5) | sub-band DQPSK, grid480 rows 0/1 (M) | 744 | 1500 | 93 (M via payload) | 2 (M) | 512 (M) | 0 (M) | 2 sub-bands 1350/1650 (M) | - | - | 394 (D, from L4) | 9 (D, from L4) |
| 5 (6) | sub-band DQPSK, 3 sub-bands (M); 3rd reference U | 1080 | 2172 | 135 (M) | 2 (M, 2 of 3 sub-bands) | 512 (M) | 0 (M) | 3 at 1312.5/1500/1687.5 (M) | - | - | 374 (D) | 9 (D) |
| 6 (7) | const QPSK (M) | 1578 | 3168 | 197 (M) | 2 (M) | 1152 (M) | 128 (M) | 11 (M) | 27 of 1024 (M) | 1 (M) | 148 (D) | 5 (D) |
| 7 (8) | const QPSK (M) | 2104 | 3168 | 263 (M) | 2 (M) | 1152 | 128 | 11 | 27 | 1 | 148 (D) | 5 (D) |
| 8 (9) | const QPSK (M) | 2524 | 3168 | 315 (M) | 2 (M) | 1152 | 128 | 11 | 27 | 1 | 148 (D) | 5 (D) |
| 9 (10) | const 8PSK (M) | 3160 | 4752 | 395 (M) | 3 (M) | 1152 | 128 | 11 | 27 | 1 | 148 (D) | 5 (D) |
| 10 (11) | const 8PSK (M) | 3792 | 4752 | 474 (M) | 3 (M) | 1152 | 128 | 11 | 27 | 1 | 148 (D) | 5 (D) |
| 11 (12) | const 16QAM + 220 pilots + 44 fixed (M) | 4755 | 5720 | 594 (M) | 4 (M) | 1152 | 128 | 11 | 27 | 1 | 154 (M) | 5 (M) |
| 12 (13) | const 32-cross + 220 pilots + 44 fixed (M) | 5708 | 7150 | 713 (M) | 5 (M) | 1152 | 128 | 11 | 27 | 1 | 154 (M) | 5 (M) |

Still U at every const level: the twiddle's absolute value (only mod 90° is held, per cell), the lead symbols'
content, the PSK extras' positions and values, fill order, puncture patterns, and whether the fixed symbol 4 at PSK
belongs to the lead or the frame.

## Disagreements with sigidwiki

| L | wiki cell | tape |
|---|---|---|
| 4 | BPSK | differential BPSK across columns (naming) |
| 5 | 4PSK | differential QPSK on grid480-rotated cells (naming) |
| 6 | 4PSK | DQPSK on two of three sub-bands; the third is not verified |
| 7-11 | 588 / 705 / 884 / 1060 / 441 bps | N/bps puts the frame at 3.576 s = 149 symbols. The tape's 153-symbol burst minus the 5-symbol lead measured at QAM leaves 148 (3.552 s), a 0.7% gap. The wiki's rate fits only if symbol 4 counts in the PSK frame |

The rest agrees: carriers (2, 3, 11), 94 and 42 Bd (41.67 on tape), and QPSK/8PSK/16QAM/32QAM at L7-L13 (the
32QAM is the cross). At L12/L13 the wiki's rate gives 3.696 s = 154 symbols exactly. L4-L6 give 4.18/4.20/4.00 s:
394 and 374 columns (4.20 s, 3.99 s), consistent with a 9-column lead. The wiki does not mention the lead, the
twiddle or the pilots.

## Disagreements with our current records

1. **Const-law burst length** (RECORD-SEMANTICS §2: `ceil(ceil(coded/bpc)/span)·dw50`, no preamble, pilots or
   forced cells). At BW500, L13 would get 130 symbols; the tape has 159 (5 lead + 130 data + 20 pilot + 4 extra
   symbols' worth of cells). PSK would get 144; the tape has 153.
2. **"kestrel implements none of twiddle, OLA, pilots or preamble"** (§1). All four are on tape at BW500: a
   per-cell 2π/128 twiddle, fixed across sessions, with one table at PSK and another in the QAM body; a ~29-sample
   overlap inside the CP; 220 scattered unit pilots per QAM frame; and a 5-symbol lead.
3. **What makes the 3.58 → 3.70 s step** (§4 and review item 1). At BW500 the QAM frame is 6 symbols longer than
   the PSK frame because 220 pilots are added while the data drops 154 cells: 1430 + 220 + 44 = 1694 against
   1584 + 44 = 1628. Pilots make the step here, not forced cells.
4. **§4 "BW500 rec4-12: 750/2, 1086/3, 1584/11 ×5, 1430/11 ×2"**. The carrier divisors are right, but the frames
   are 148/154 symbols, not 144/130. L5/L6 have 2 and 3 sub-bands (402 and 383 columns), not per-carrier
   one-offs.
5. **rec9-16 placeholders span 32 / first_bin 9** (§4, BW2300). The BW500 const law measures span 11 from bin 27
   of 1024 with dw50 1152 / cp 128, which confirms the dw50 and cp; BW2300 is not measured here.
6. **BW2300 rec16 (506 = 46×11) as a BW500 row.** Not supported. Every clean BW500 L12/L13 data burst is 159 or
   313 symbols. Final frames with short payloads (L8 167/213/227 bytes, L11 151) are padded to full length, and no
   51-symbol data burst is on tape. It stays U.
7. **BW500 `IndexRecord`/base path.** L5 and L6 are not index law and not const law. They are the L4 sub-band
   family in DQPSK, and they reuse grid480 rows 0/1 by sub-band index (row 0 at 1350 and at 1312.5 Hz). L6's third
   sub-band needs a reference row grid480 does not hold.
8. **01 §1.1 (500 Hz)** describes only the sub-band family. BW500 L7-L13 is standard CP-OFDM on bins 27..37,
   which the document does not cover.

A lead to test, not a finding: BW500's Rnd gaps at rec6-11 are 44-46 draws (review item 6), and every const frame
carries 44 extra cells.
