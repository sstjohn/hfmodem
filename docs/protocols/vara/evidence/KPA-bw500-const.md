# BW500 levels 7-13 closed from known plaintext

`kpa_bw500_const.py` prints every table below (about 25 s, nothing written). It takes the BW2300 law
(`KPA-bw2300-const.md`) onto BW500's 11 carriers and demodulates 60 frames off the high speed capable bench tapes
(53 overs, three of them double L13 overs): 2 at L7, 13 at L8, 6 at L9, 8 at L10, 5 at L11, 16 at L12 and
10 at L13. All 60 are CRC-clean and byte-exact against the transcript's delivered bytes, with **0 raw
on-air bit errors before FEC on every frame**. Every interior cell equals the chain's predicted cell to
3.2e-3 of the reference amplitude (0 at PSK); only the burst's first and last symbol deviate (edge taper).

## Findings

The BW2300 law holds unchanged except for the carrier set and the number of gap cells per carrier.

1. **Level to record.** Level L is record L−1 (`tablegen.BW500[L−1]`). The delivered payload on a full
   frame is `N//8 − 3`: 194, 260, 312, 392, 471, 591, 710 bytes at L7-L13, each seen on tape. The frame
   is payload, one control byte, then the CRC-16/GENIBUS, as at BW2300 (BW500's "marker" is that control
   byte). Control bytes seen: 0x81, 0x89, 0x8d, 0x91, 0x95, 0x99. A short frame reads data, `14 09`,
   zero fill, `07`, `82`, CRC. At BW2300 the byte before `82` is `06`. When only two bytes are spare
   (L8 @2183411), the tail is `14 07 82`.
2. **Carriers and window.** Bins 27..37 of the 1024-point FFT, 1152-sample symbols. Cells come from
   `raster_bw2300.ofdm_cells(backoff=16)`, the same window 112 samples into the symbol. The PSK reference
   reads exactly +1 on every carrier.
3. **Scramble.** The same S table: `Int(Rnd*129)` from whitener seed 29318, 409 columns per FFT-bin row
   from bin 0, so rows 27..37 (bin 27 column 0 is draw 1 + 27·409 = 11044). Column c = s − 4 at PSK
   (column 0 = the reference symbol 4) and c = s − 5 − 154f at QAM, restarting in frame 2 of a double
   L13. Lead symbols are unscrambled. This includes QAM symbol 4, which is lead, not body; RASTER-bw500
   reads it otherwise.
4. **Frame.**
   - PSK (L7-L11): symbols 0-3 lead, 4 reference +1, 5..152 data. That is 148 cells per carrier:
     144 data + **4 repeat cells** (differential step +1). 11 × 144 = 1584 = coded/bpc at every PSK record.
   - QAM (L12-L13): symbols 0-4 lead, then 154-symbol frames. Pilots +1: absolute-even bins at frame row
     8j, absolute-odd bins at 8j+1 (the BW2300 rule; here bin 27 is odd, so its pilots are on row 8j+1),
     20 per carrier. There are **4 extra +1 cells** per carrier, each on the row right after one of that
     carrier's pilot rows j. That leaves 130 data cells per carrier: 11 × 130 = 1430 = coded/bpc.
5. **The 44-46 draw gap is the 44 gap cells.** Draws come bin-major from the record's gap, after its two
   permutations. PSK uses `7 + Int(Rnd*142)` as a symbol and QAM uses `Int(Rnd*19)` as a pilot row. A
   draw that repeats an earlier value on the same carrier is thrown away and drawn again. That makes 44
   cells plus the repeats: 46 / 44 / 44 / 45 / 44 / 46 / 49 draws at records 6-12, closing exactly on the
   next record's seed. BW2300 is the same law with 1 cell per carrier and 49 carriers.
6. **Modulation.** PSK is time-differential per carrier from the +1 reference. QAM is coherent on the
   pilots. The LUT is `tablegen._column(pos, v)` indexed by the raw bit group, MSB first: pos 1 QPSK
   (L7-L9), pos 2 8PSK (L10-L11), pos 3 16QAM (L12), pos 4 32-cross (L13). Fill is bin-major.
7. **Puncture.** The BW2300 group law applies at every rate, with k = round(N/(coded−N−12)). k = 1 (L7) is
   the rate-1/2 mux itself. k is 2/4/2/4/5/4 at L8-L13. Zero pad 0/0/1/0/0/2/3 bits at records 6-12.
   Whitener, turbo, il1 (gather) and il2 are the held BW500 columns.
8. **Lead.** `LUT[Int(Rnd·2^bpc)]` drawn bin-major, symbol fastest, from the session's continuing Rnd
   stream. That is 44 draws per PSK over, 55 per QAM frame, and 110 for a double L13 over, although only
   one lead is sent. The state is recovered on all 53 overs. It sits about 3.335 M draws past the L4
   training seed 4107106, as at BW2300. Consecutive decoded overs of one tape differ by sums of 44 and 55,
   except where non-const keyings sit between them (the 524 and 4345 rows).
9. **Residual.** None in the data path. One L13 over (@2600000) shows a pilot phase ramp of −9° to −13°
   across the carriers. That is the grid estimate, and the per-carrier pilot mean absorbs it. Open, as at
   BW2300: the control byte's meaning, the lead stream's origin in a session, and the edge taper for TX.

### Proposed Record rows (BW500 const law)

M = measured on tape by this decode; D = derived (held tablegen value or arithmetic). Every row: dw50 1152
(M), cp 128 (M), FFT 1024, span 11 (M), first_bin 27 (M), window 112 samples into the symbol (M), scramble
rows 27..37 (M), 4 gap cells per carrier (M), frame_bytes = N//8 = payload + ctl + CRC (M).

| L | rec | law | pos | bpc | N (D) | coded (D) | frame_bytes | k (M) | pad (D) | lead sym (M) | frame sym (M) | gap cell (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 6 | PSK diff | 1 | 2 | 1578 | 3168 | 197 | 1 | 0 | 4 + ref | 148 | 7+Int(Rnd*142) |
| 8 | 7 | PSK diff | 1 | 2 | 2104 | 3168 | 263 | 2 | 0 | 4 + ref | 148 | 7+Int(Rnd*142) |
| 9 | 8 | PSK diff | 1 | 2 | 2524 | 3168 | 315 | 4 | 1 | 4 + ref | 148 | 7+Int(Rnd*142) |
| 10 | 9 | PSK diff | 2 | 3 | 3160 | 4752 | 395 | 2 | 0 | 4 + ref | 148 | 7+Int(Rnd*142) |
| 11 | 10 | PSK diff | 2 | 3 | 3792 | 4752 | 474 | 4 | 0 | 4 + ref | 148 | 7+Int(Rnd*142) |
| 12 | 11 | QAM coh | 3 | 4 | 4755 | 5720 | 594 | 5 | 2 | 5 | 154 × 1 | after pilot row Int(Rnd*19) |
| 13 | 12 | QAM coh | 4 | 5 | 5708 | 7150 | 713 | 4 | 3 | 5 | 154 × 1 or 2 | after pilot row Int(Rnd*19) |

Burst length: PSK 153 × 1152 = 176256, QAM (5 + 154·frames) × 1152 = 183168 / 360576, matching the
measured 176200 / 183100 / 360500 less the taper.

### Differences from BW2300

| item | BW2300 | BW500 |
|---|---|---|
| carriers | 49, bins 8..56 | 11, bins 27..37 (S rows 27..37) |
| gap cells per carrier | 1 | 4, redrawn on a repeat within the carrier |
| record gap | 49 draws | 44 + repeats (44-49) |
| lead draws | 196 PSK / 245 per QAM frame | 44 PSK / 55 per QAM frame |
| records | 9-15, `tablegen.BW2300` | 6-12, `tablegen.BW500` |
| short-frame tail before ctl `82` | `06` | `07` |

Everything else is identical: the S table itself, the reference and pilot rules by absolute bin parity,
the frame symbol counts (153 PSK, 5 + 154·f QAM), the LUT columns, the group puncture, the lead law and
the CRC.

## Evidence

- First probe: with S rows 27..37 and column s − 4, symbol 4 of L7, L8 and L10 descrambles to 0° on all
  11 carriers. Symbol-to-symbol differences give M4 resultant 1.0000 (QPSK) and M8 1.0000 at L10. At QAM,
  row 5 descrambled with column s − 5 puts every even-bin pilot at 0°.
- Cells whose differential step is +1 in all 13 clean L8 overs: exactly 4 per carrier. Fitting `a +
  Int(Rnd*m)` over the rec-7 gap, bin-major, hits a = 7, m = 142 alone, 44/44 (the BW2300 constants). At
  rec 8 (L10, gap 45) carrier 3 draws 79 twice, and the repeat is skipped.
- QAM +1 cells off the pilots in all 16 L12 and 7 L13 overs: 4 per carrier, each one row after one of
  the carrier's pilot rows. The rows equal the gap's `Int(Rnd*19)` in draw order, with repeats skipped.
- With the layout, scramble, LUT and group puncture above and no other change, every frame decodes with
  0 raw bit errors (table below).

### Per over

| level | tape | measured_on | frame | payload (first, len) | ctl bytes | raw bit errors / on-air bits | CRC | byte-exact | max dev interior / edge | lead Rnd state |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 20260926T052917Z-ladder | 2366113 | 0 | (241, 15) | 1409 +176x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.045 | 5901591 |
| 7 | 20260926T053832Z-ladder | 4725719 | 0 | (35, 194) | 99 | 0 / 3168 | True | True | 0.0000 / 0.053 | 9942433 |
| 8 | N0CALL-W9SSJ-bw500-counter65536 | 2181689 | 0 | (43, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.010 | 13265963 |
| 8 | 20260926T052707Z-ladder | 2156809 | 0 | (89, 167) | 1409 +90x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.087 | 13475962 |
| 8 | 20260926T052812Z-ladder | 2183411 | 0 | (254, 258) | 140782 | 0 / 3168 | True | True | 0.0000 / 0.440 | 7103400 |
| 8 | 20260926T052917Z-ladder | 2161075 | 0 | (237, 260) | 81 | 0 / 3168 | True | True | 0.0000 / 0.059 | 1738267 |
| 8 | 20260926T053136Z-ladder | 2213650 | 0 | (54, 260) | 95 | 0 / 3168 | True | True | 0.0000 / 0.028 | 13975206 |
| 8 | 20260926T053357Z-ladder | 2210920 | 0 | (8, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.632 | 6081586 |
| 8 | 20260926T053357Z-ladder | 3454582 | 0 | (29, 227) | 1409 +30x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.026 | 13344351 |
| 8 | 20260926T053528Z-ladder | 1172518 | 0 | (14, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.032 | 3538917 |
| 8 | 20260926T053528Z-ladder | 2211880 | 0 | (212, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.044 | 6081586 |
| 8 | 20260926T053528Z-ladder | 3464232 | 0 | (43, 213) | 1409 +44x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.056 | 11246616 |
| 8 | 20260926T053700Z-ladder | 2224492 | 0 | (8, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.041 | 6081586 |
| 8 | 20260926T053700Z-ladder | 3467643 | 0 | (29, 227) | 1409 +30x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.026 | 13344351 |
| 8 | 20260926T053832Z-ladder | 1184473 | 0 | (223, 260) | 99 | 0 / 3168 | True | True | 0.0000 / 0.009 | 3538917 |
| 9 | 20260926T053026Z-ladder | 2215410 | 0 | (176, 312) | 89 | 0 / 3168 | True | True | 0.0000 / 0.032 | 15830618 |
| 9 | 20260926T053026Z-ladder | 2420717 | 0 | (232, 280) | 1409 +29x00 0782 | 0 / 3168 | True | True | 0.0000 / 0.048 | 9850598 |
| 9 | 20260926T053136Z-ladder | 1164311 | 0 | (56, 312) | 99 | 0 / 3168 | True | True | 0.0000 / 0.058 | 14324316 |
| 9 | 20260926T053246Z-ladder | 2226841 | 0 | (134, 312) | 8d | 0 / 3168 | True | True | 0.0000 / 0.019 | 5793147 |
| 9 | 20260926T053357Z-ladder | 1172582 | 0 | (14, 312) | 99 | 0 / 3168 | True | True | 0.0000 / 0.034 | 3538917 |
| 9 | 20260926T053700Z-ladder | 1185642 | 0 | (14, 312) | 99 | 0 / 3168 | True | True | 0.0000 / 0.034 | 3538917 |
| 10 | 20260926T052424Z-ladder | 1109268 | 0 | (192, 392) | 91 | 0 / 4752 | True | True | 0.0000 / 0.069 | 12254096 |
| 10 | 20260926T052516Z-ladder | 1109724 | 0 | (192, 392) | 91 | 0 / 4752 | True | True | 0.0000 / 0.069 | 12254096 |
| 10 | 20260926T052607Z-ladder | 1109156 | 0 | (140, 392) | 91 | 0 / 4752 | True | True | 0.0000 / 0.063 | 12254096 |
| 10 | 20260926T052707Z-ladder | 1108748 | 0 | (74, 392) | 91 | 0 / 4752 | True | True | 0.0000 / 0.043 | 12254096 |
| 10 | 20260926T052917Z-ladder | 1114794 | 0 | (222, 392) | 91 | 0 / 4752 | True | True | 0.0000 / 0.029 | 13165481 |
| 10 | 20260926T053026Z-ladder | 1156843 | 0 | (98, 392) | 95 | 0 / 4752 | True | True | 0.0000 / 0.039 | 9591511 |
| 10 | 20260926T053246Z-ladder | 1166498 | 0 | (56, 392) | 95 | 0 / 4752 | True | True | 0.0000 / 0.037 | 14324316 |
| 10 | 20260926T053246Z-ladder | 2432930 | 0 | (190, 322) | 1409 +67x00 0782 | 0 / 4752 | True | True | 0.0000 / 0.050 | 1861239 |
| 11 | N0CALL-W9SSJ-bw500-counter65536 | 20549166 | 0 | (105, 151) | 1409 +317x00 0782 | 0 / 4752 | True | True | 0.0000 / 0.010 | 12085723 |
| 11 | 20260926T052115Z-ladder | 1322818 | 0 | (50, 462) | 1409 +6x00 0782 | 0 / 4752 | True | True | 0.0000 / 0.021 | 8792809 |
| 11 | 20260926T052250Z-ladder | 1323982 | 0 | (50, 462) | 1409 +6x00 0782 | 0 / 4752 | True | True | 0.0000 / 0.021 | 8792809 |
| 11 | 20260926T052812Z-ladder | 1136888 | 0 | (160, 471) | 8d | 0 / 4752 | True | True | 0.0000 / 0.043 | 10227278 |
| 11 | 20260926T053136Z-ladder | 2418953 | 0 | (58, 454) | 1409 +14x00 0782 | 0 / 4752 | True | True | 0.0000 / 0.090 | 12728434 |
| 12 | N0CALL-W9SSJ-bw500-counter65536 | 2387249 | 0 | (47, 591) | 99 | 0 / 5720 | True | True | 0.0005 / 0.010 | 7911975 |
| 12 | N0CALL-W9SSJ-bw500-counter65536 | 20336939 | 0 | (26, 591) | 81 | 0 / 5720 | True | True | 0.0023 / 0.049 | 14260474 |
| 12 | 20260926T052115Z-ladder | 1111368 | 0 | (227, 591) | 81 | 0 / 5720 | True | True | 0.0030 / 0.066 | 12254096 |
| 12 | 20260926T052202Z-ladder | 1108495 | 0 | (68, 591) | 8d | 0 / 5720 | True | True | 0.0020 / 0.044 | 12254096 |
| 12 | 20260926T052250Z-ladder | 1110740 | 0 | (227, 591) | 81 | 0 / 5720 | True | True | 0.0030 / 0.066 | 12254096 |
| 12 | 20260926T052337Z-ladder | 1109842 | 0 | (147, 591) | 89 | 0 / 5720 | True | True | 0.0014 / 0.030 | 12254096 |
| 12 | 20260926T052337Z-ladder | 1322818 | 0 | (226, 542) | 1409 +46x00 0782 | 0 / 5720 | True | True | 0.0028 / 0.061 | 8792809 |
| 12 | 20260926T052424Z-ladder | 1315352 | 0 | (72, 591) | 81 | 0 / 5720 | True | True | 0.0010 / 0.022 | 13447100 |
| 12 | 20260926T052516Z-ladder | 1314784 | 0 | (72, 591) | 81 | 0 / 5720 | True | True | 0.0010 / 0.022 | 13447100 |
| 12 | 20260926T052607Z-ladder | 1313704 | 0 | (20, 591) | 81 | 0 / 5720 | True | True | 0.0032 / 0.069 | 13447100 |
| 12 | 20260926T052707Z-ladder | 1314576 | 0 | (210, 591) | 89 | 0 / 5720 | True | True | 0.0028 / 0.061 | 13447100 |
| 12 | 20260926T052812Z-ladder | 1341444 | 0 | (119, 591) | 89 | 0 / 5720 | True | True | 0.0007 / 0.016 | 4600218 |
| 12 | 20260926T052917Z-ladder | 1320358 | 0 | (102, 591) | 89 | 0 / 5720 | True | True | 0.0007 / 0.014 | 6075589 |
| 12 | 20260926T053357Z-ladder | 2415970 | 0 | (12, 591) | 81 | 0 / 5720 | True | True | 0.0025 / 0.055 | 1395518 |
| 12 | 20260926T053528Z-ladder | 2416930 | 0 | (216, 591) | 89 | 0 / 5720 | True | True | 0.0011 / 0.023 | 1395518 |
| 12 | 20260926T053700Z-ladder | 2430054 | 0 | (12, 591) | 81 | 0 / 5720 | True | True | 0.0025 / 0.055 | 1395518 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 2600000 | 0 | (126, 710) | 99 | 0 / 7150 | True | True | 0.0009 / 0.018 | 2751532 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 2813230 | 0 | (68, 710) | 99 | 0 / 7150 | True | True | 0.0015 / 0.029 | 13089589 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 3026737 | 0 | (10, 710) | 99 | 0 / 7150 | True | True | 0.0003 / 0.005 | 7040018 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 3879464 | 0 | (34, 710) | 99 | 0 / 7150 | True | True | 0.0000 / 0.000 | 15284671 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 3879464 | 1 | (232, 710) | 99 | 0 / 7150 | True | True | 0.0018 / 0.035 |  |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 4271165 | 0 | (174, 710) | 99 | 0 / 7150 | True | True | 0.0000 / 0.000 | 9668109 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 4271165 | 1 | (116, 710) | 99 | 0 / 7150 | True | True | 0.0013 / 0.025 |  |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 4662835 | 0 | (58, 710) | 99 | 0 / 7150 | True | True | 0.0000 / 0.000 | 8911339 |
| 13 | N0CALL-W9SSJ-bw500-counter65536 | 4662835 | 1 | (0, 710) | 99 | 0 / 7150 | True | True | 0.0020 / 0.038 |  |
| 13 | 20260926T052202Z-ladder | 1321475 | 0 | (147, 621) | 1409 +86x00 0782 | 0 / 7150 | True | True | 0.0028 / 0.055 | 8792809 |

### Per level

| level | frames | raw bit errors (sum) | CRC clean | byte-exact |
|---|---|---|---|---|
| 7 | 2 | 0 | 2 | 2 |
| 8 | 13 | 0 | 13 | 13 |
| 9 | 6 | 0 | 6 | 6 |
| 10 | 8 | 0 | 8 | 8 |
| 11 | 5 | 0 | 5 | 5 |
| 12 | 16 | 0 | 16 | 16 |
| 13 | 10 | 0 | 10 | 10 |

Interior cells against the chain, frames with no raw bit error: max |z - expected| = 0.0032, median 0.0000.
### Lead: Rnd draws between consecutive decoded overs of one tape

| tape | from (level @on) | to (level @on) | draws |
|---|---|---|---|
| 20260926T052917Z-ladder | L10 @1114794 | L12 @1320358 | 44 |
| 20260926T052917Z-ladder | L12 @1320358 | L8 @2161075 | 198 |
| 20260926T052917Z-ladder | L8 @2161075 | L7 @2366113 | 44 |
| 20260926T053832Z-ladder | L8 @1184473 | L7 @4725719 | 524 |
| N0CALL-W9SSJ-bw500-counter65536 | L8 @2181689 | L12 @2387249 | 44 |
| N0CALL-W9SSJ-bw500-counter65536 | L12 @2387249 | L13 @2600000 | 55 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @2600000 | L13 @2813230 | 55 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @2813230 | L13 @3026737 | 55 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @3026737 | L13 @3879464 | 275 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @3879464 | L13 @4271165 | 110 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @4271165 | L13 @4662835 | 110 |
| N0CALL-W9SSJ-bw500-counter65536 | L13 @4662835 | L12 @20336939 | 4345 |
| N0CALL-W9SSJ-bw500-counter65536 | L12 @20336939 | L11 @20549166 | 55 |
| 20260926T052707Z-ladder | L10 @1108748 | L12 @1314576 | 44 |
| 20260926T052707Z-ladder | L12 @1314576 | L8 @2156809 | 198 |
| 20260926T052812Z-ladder | L11 @1136888 | L12 @1341444 | 44 |
| 20260926T052812Z-ladder | L12 @1341444 | L8 @2183411 | 198 |
| 20260926T053136Z-ladder | L9 @1164311 | L8 @2213650 | 242 |
| 20260926T053136Z-ladder | L8 @2213650 | L11 @2418953 | 44 |
| 20260926T053357Z-ladder | L9 @1172582 | L8 @2210920 | 231 |
| 20260926T053357Z-ladder | L8 @2210920 | L12 @2415970 | 44 |
| 20260926T053357Z-ladder | L12 @2415970 | L8 @3454582 | 231 |
| 20260926T053528Z-ladder | L8 @1172518 | L8 @2211880 | 231 |
| 20260926T053528Z-ladder | L8 @2211880 | L12 @2416930 | 44 |
| 20260926T053528Z-ladder | L12 @2416930 | L8 @3464232 | 242 |
| 20260926T053700Z-ladder | L9 @1185642 | L8 @2224492 | 231 |
| 20260926T053700Z-ladder | L8 @2224492 | L12 @2430054 | 44 |
| 20260926T053700Z-ladder | L12 @2430054 | L8 @3467643 | 231 |
| 20260926T053026Z-ladder | L10 @1156843 | L9 @2215410 | 253 |
| 20260926T053026Z-ladder | L9 @2215410 | L9 @2420717 | 44 |
| 20260926T053246Z-ladder | L10 @1166498 | L9 @2226841 | 253 |
| 20260926T053246Z-ladder | L9 @2226841 | L10 @2432930 | 44 |
| 20260926T052424Z-ladder | L10 @1109268 | L12 @1315352 | 44 |
| 20260926T052516Z-ladder | L10 @1109724 | L12 @1314784 | 44 |
| 20260926T052607Z-ladder | L10 @1109156 | L12 @1313704 | 44 |
| 20260926T052115Z-ladder | L12 @1111368 | L11 @1322818 | 55 |
| 20260926T052250Z-ladder | L12 @1110740 | L11 @1323982 | 55 |
| 20260926T052202Z-ladder | L12 @1108495 | L13 @1321475 | 55 |
| 20260926T052337Z-ladder | L12 @1109842 | L12 @1322818 | 55 |

## What must change

Paths under `packages/hfmodem/hfmodem/kestrel/`.

| where | change |
|---|---|
| `rx/tablegen.py:99` (`whitener`), with the BW2300 item | a shared scramble table `Int(Rnd*129)` from seed 29318, 409 columns per bin row from bin 0. BW500 reads rows 27..37. |
| `rx/tablegen.py:510` (`_bw500_gap`) | add the gap-cell generator for records 6-12: 4 cells per carrier, bin-major, repeats redrawn, `7 + Int(Rnd*142)` (PSK) or pilot row `Int(Rnd*19)` (QAM). It must close on `BW500[rec+1][0]`. Make it one function with the BW2300 per-carrier cell (count 1, 49 carriers). |
| `rx/tablegen.py:117-122` (`BW500`) | no change: seeds, N and coded for records 6-12 are right as held. |
| `rx/varahf500.py:200-228` (`IndexRecord`, `INDEX_RECORDS`) | add a const-law record type for host levels 7-13 (the rows above: rec, pos, bpc, n_info, coded, frame = N//8, span 11, first_bin 27, dw 1152, cp 128, lead 4+ref or 5, QAM flag). `frame` stays payload + 3. |
| `rx/varahf500.py:249-254` (`BASE_LEVEL`, `_PAYLOAD`) | extend the level numbering and payload map to 7-13 (194/260/312/392/471/591/710). |
| `rx/varahf500.py:287` (`frames_carried`), `:759` (`decode_burst_frames`) | const-law demod: CP grid (1152), window 112 into the symbol, descramble, differential from the reference (PSK) or pilot-equalised (QAM), bin-major gather skipping gap cells, then il1 gather, group depuncture, turbo on il2, dewhiten, CRC. Frame count from the burst length (1 or 2 at L13). |
| `tx/varahf500_tx.py:81-99` (`build_frame`, `encode_coded`) | frame = payload + control byte + CRC at N//8 bytes per record. Encode via the group puncture. |
| `tx/varahf500_tx.py:139-160` (`synth_burst`) | a const-law branch: lead `LUT[Int(Rnd·2^bpc)]` from the session stream (44 draws per PSK over, 55 per QAM frame), reference or pilots and gap cells, bin-major data, `× exp(−j2πS/128)` on bins 27..37, 1024 IFFT + 128 CP, edge taper. |
| `coding/turbo.py:226-260` (`_punct_keep`, `encode_punctured`, `decode_punctured`) | the group law, shared with BW2300 (see its change list). At BW500, records 7-12 need it. |
| `vara/vara_ofdm.py:71-72, 114-118` (`_BW500_RECORDS`) | map BW500 records 6-12 to the new levels 7-13. |
