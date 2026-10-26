<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# BW2300 physical raster, levels 4-16, off the high speed capable bench tapes

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

`raster_bw2300.py` prints every number below (about 3 s, no plots). Levels are host `BITRATE (N)` = sigidwiki
L; record = L-1. Bursts are cut on `overs.jsonl` `measured_on/off`. All samples are at 48 kHz.

Level to record is settled by bytes. For every level, `payload_len` + 3 equals the frame bytes that
`tablegen.BW2300`'s N implies for record L-1: 89+3=92, 132+3=135, 173+3=176, 256+3=259, 341+3=344, 425+3=428,
896+3=899, 1196+3=1199, 1436+3=1439, 2156+3=2159, 2710+3=2713 and 3254+3=3257. The double-length L16 over
delivers 6508 = 2x3254.

The tape shows three families:

- **L4: one-hot index law**, the control.
- **L5-L9: frequency-division sub-bands.** Each band carries one complex sample per 512-sample symbol. No
  CP, and not on any FFT grid.
- **L10-L16: CP-OFDM.** Tu 1024, CP 128, 49 carriers.

## Control: level 4 data (ISS) and the IRS ack

| instance (tape, who, measured_on) | samples | 512-blocks | one-bin energy | bins | ac512 / ac1024 / ac1152 |
|---|---|---|---|---|---|
| N0CALL-W9SSJ-counter4096, A, 1965294 | 208365 | 407 | 1.000 | 9..24 | +0.049 / +0.052 / +0.000 |
| N0CALL-W9SSJ-counter65536, A, 1954906 | 208365 | 407 | 1.000 | 9..24 | +0.049 / +0.052 / +0.000 |
| W9SSJ-N0CALL-counter4096, A, 1963541 | 208366 | 407 | 1.000 | 9..24 | +0.052 / +0.052 / +0.002 |

407 = 12 training + 395 columns, and each block carries one bin of 9..24 of a 512-point FFT with no CP. That is
record 3 exactly. The lead-in silence is not in the cut. L4's first 6144 samples from two sessions at the same
over position cross-correlate at 1.000, so the training is session-deterministic.

IRS ack (B, counter4096, 454393 and 2181369): 22520 samples = 11 symbols of 2048, and 16375 = 8. The two-bin
energy fraction is 1.000. The first pairs are {64,67} {56,74} {64,69} {68,78}, which is the 02/04 §4.2C
two-tone token. The pipeline reproduces what the records and docs already say.

## Levels 5-9: sub-bands, 93.75 Bd

Envelope line at 93.75 Hz in every level (period 512). Global autocorrelation at 512/1024/1152 is within
±0.016, so there is no CP. Carrier centres are symmetric about 1500 Hz, `1500 + step·(k - (n-1)/2)`.

| L | carriers | step Hz | centres Hz (first..last) | -20 dB width Hz | stagger between adjacent bands | symbols/band | cells | rec cells (coded/2) | surplus |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 3 | 600 | 900, 1500, 2100 | 290 | +176, +336 samples (thirds) | 383 | 1149 | 1086 | 63 (21/band) |
| 6 | 4 | 500 | 750..2250 | 293 | alternate +256 | 372 | 1488 | 1416 | 72 (18) |
| 7 | 6 | 320 | 700..2300 | 170 | alternate +240 | 362* | 2172 | 2082 | 90 (15) |
| 8 | 8 | 1600/7 = 228.57 | 700..2300 | 170 | alternate +256 | 359 | 2872 | 2760 | 112 (14) |
| 9 | 10 | 210 | 555..2445 | 175 | alternate +258 | 356 | 3560 | 3430 | 130 (13) |

\*L7: floor(185786/512). The clean instance's tail carries noise, and the noisy instance counts 362-363.

Centroids measured: L8 701.9 ... 2298.3, L9 555.3 ... 2445.5, both within 2 Hz of the model. Adjacent
centres are 228.57 Hz and 1020 Hz apart, which puts them on no 512-, 1024- or 2048-point grid. These are
separate sub-band carriers, like the BW500 base (1350/1650 Hz, H 512), not DFT-grid OFDM. At L5/L6 the
per-band energy profile falls to 1/2000 of peak once a symbol. At L7-9 it falls to 1/4 (eye max/min 2000 against 4).

Modulation, one sample per symbol per band at the energy peak:

| L | instances (tape, who, measured_on[, clean from sample]) | raw R2/R4/R8 | symbol diff R1/R2/R4/R8 | cross-instance R1/R2/R4/R8 | conclusion |
|---|---|---|---|---|---|
| 5 | 051938Z B 2313402; 051938Z B 3274308 (30000); 051737Z B 941139 (60000) | .045/.040/.065 | .04/.05/.04/.03 | .03/.02/**.999**/.996 | QPSK |
| 6 | 051826Z B 2299764; 051938Z B 3497388; 051425Z B 940544 (60000); 051513Z B 936853 (60000) | .04/.05/.06 | .05/.05/.03/.03 | .05/.04/**.999**/.996 | QPSK |
| 7 | 051938Z B 3717213 (noise in last 0.5 s); 051002Z B 704213 (noisy, -19 dB) | .05/.08/.06 | .04/.05/.06/.06 | .05/.05/**.56**/.17 | QPSK (noise-limited) |
| 8 | counter4096 A 2205917; counter65536 A 2194259; 051826Z B 1208324 | .05/.05/.05 | .08/.05/.05/.06 | .03/.03/**.917**/.70 | QPSK |
| 9 | 051737Z B 1165906; 050835Z B 908850 (72000) | .05/.05/.04 | .06/.04/.06/.05 | .06/.04/**.953**/.82 | QPSK |

Constant modulus at the sampling point: |sample| std 0.009 at L5/L6 and 0.05-0.07 at L7-9, where the
filter lets in some ICI. Raw and symbol-to-symbol phases are uniform. Between two overs with different
payloads, though, the ratio sits on four points 90° apart. So each (band, symbol) carries a fixed,
payload-independent phase scramble, shared across sessions, and the data are QPSK. 8PSK would put R4 near 0.

Opening and overhead: the first 12 blocks are not index-law (one-bin fraction 0.13-0.25). Their best
correlation with L4's training is 0.11-0.22, against 1.000 for L4 with L4. Every band is at full amplitude
from its first symbol. Over six L6 pairs, no symbol in the second half of the over holds one quadrant in all
pairs. That rules out a mid-over reference there. The first half is covered by one clean pair only. The
surplus (21/18/15/14/13 symbols per band) is not located.

## Levels 10-16: CP-OFDM

| L | instances (tape, who, measured_on) | samples | ac lag peak | fold period | CP fold peak | symbols | bins (1024-FFT) | Hz | edge |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 051337Z B 1138938, 1348084; 051425Z B 1159559 | 176202 | 1024 | 1152 | 0.969 | 153 | 8..56 | 375.0-2625.0 | -94 dB |
| 11 | 050835Z B 1120668; 050919Z B 1121832; 051045Z B 1128816 | 176213 | 1024 | 1152 | 0.968 | 153 | 8..56 | same | -94 |
| 12 | counter4096 A 2420736; counter65536 A 2408299; W9SSJ-N0CALL A 2417190 | 176209 | 1024 | 1152 | 0.971 | 153 | 8..56 | same | -94 |
| 13 | counter4096 A 2839297; W9SSJ-N0CALL A 2835744 | 176216 | 1024 | 1152 | 0.971 | 153 | 8..56 | same | -94 |
| 14 | counter4096 A 2629884; counter65536 A 2616680; W9SSJ-N0CALL A 2625329 | 176217 | 1024 | 1152 | 0.969 | 153 | 8..56 | same | -94 |
| 15 | counter65536 A 2826094, 6762860 | 183128 | 1024 | 1152 | 0.972 | 159 | 8..56 | same | -95 |
| 16 | counter65536 A 3044455 ... 4145516 (6), 4367207 ... 6365037 (6, double) | 183132 / 360544 | 1024 | 1152 | 0.969 / 0.976 | 159 / 313 | 8..56 | same | -95 |

- **Tu 1024, CP 128, 41.667 Bd.** Global autocorrelation peaks at lag 1024 (+0.09 to +0.12) against ±0.02
  at 512, 1152 and 1280. The windowed CP metric folds best at period 1152, peak 0.97. The grid starts 0-18
  samples before `measured_on`.
- **49 carriers, bins 8..56, 46.875 Hz apart, 375-2625 Hz**, flat to ±1 %. Nothing outside reaches within
  94 dB when the FFT window starts mid-CP.
- **Symbol edges are tapered.** With the window at the CP's end (backoff 0) the cells smear: raw R128 falls
  to 0.1 and the edge to -30 dB. At backoff 16-96 everything is exact (R128 = 1.00). That points to about 10
  samples of window/overlap at each symbol boundary.
- **Per-cell phase scramble on a 2π/128 lattice.** Raw cells at PSK levels have R128 = 1.00 per bin, while
  R2..R64, symbol-to-symbol and bin-to-bin differences all sit at chance (0.06-0.13). The scramble is the same
  at every (symbol, bin) across overs, sessions and all seven levels. Cross-instance ratios land on exact
  lattice points (max residual 0.0018 units), and QAM cells divided by an L12 cell keep QAM geometry (below).
  Nothing is differential.

| L | cross-instance steps (units 2π/128) | amplitude rings (rms = 1) | ring shares | modulation |
|---|---|---|---|---|
| 10 | 4, gcd 32 (L10×L10 twice, L10×L11) | 1 | 1 | QPSK |
| 11 | 4, gcd 32 | 1 | 1 | QPSK |
| 12 | 4, gcd 32 | 1 | 1 | QPSK |
| 13 | 8, gcd 16 (L13×L12, L13×L14) | 1 | 1 | 8PSK |
| 14 | 8, gcd 16 | 1 | 1 | 8PSK |
| 15 | — | 0.438 / 0.988 / 1.338 | .246 / .496 / .259 | 16QAM square |
| 16 | — | 0.312 / 0.713 / 0.938 / 1.138 / 1.312 | .124 / .244 / .124 / .264 / .244 | 32QAM cross |

For the QAM levels, divide each cell by the L12 cell at the same position and read the angle mod 90°. In
16QAM, 100 % of the inner and corner rings sit on the integer lattice and 100 % of the middle ring sits at
±0.446 units (18.43°). In 32-cross, rings 1, 3, 4 and 5 sit on the lattice (45°, 45°, 11.31° = 4.02 units,
30.96° = 11.01 units) and ring 2 sits at ±0.446. Radii and shares match square 16QAM (0.447/1/1.342, 1:2:1)
and the 6x6-minus-corners cross (0.316/0.707/0.949/1.140/1.304, 4:8:4:8:8) to 0.01.

**Over structure.**

- **Lead: 5 symbols.** At PSK levels, symbols 0-4 are identical between sessions at the same over
  position. Between different overs, symbols 0-3 differ and symbol 4 does not: it is identical across every
  PSK over tried, L10 through L14. That is session-driven training, like L4's Rnd preamble, plus one fixed
  reference symbol. At QAM levels, symbols 0-4 all differ between overs. The lead does not correlate with
  L4's training (0.14-0.17) and is not index-law.
- **PSK levels: no pilots and no mid-over reference.** After symbol 5, no symbol is identical between overs
  with different payloads.
- **QAM levels: scattered pilots.** Even bins of symbol 5+8j and odd bins of symbol 6+8j, j = 0..19, repeat
  per 154-symbol frame. That is 49 cells per pair and 20 pairs, **980 cells**, fixed across all six L16 overs
  and across L15/L16. They have constant modulus at 1.054x the 32QAM rms (1.213x the 16QAM rms), which is
  1.00 in LUT units at both levels.
- **L16: 49 more fixed cells.** One per carrier, scattered over symbols 6..151, at pilot amplitude, and at
  the same positions again in the second frame of a double over. They are not fixed at L15.
- **Frame arithmetic.** PSK: 153 = 5 + 148, and 148x49 = 7252 = 7203 + 49. QAM: 159 = 5 + 154, and
  154x49 = 7546 = 6517 + 980 + 49. Double L16: 313 = 5 + 2x154, with the second frame's pilots at 159+8j and
  no second lead.

## Proposed Record rows

M = MEASURED, D = DERIVED, U = UNKNOWN. n_info and coded are tablegen's held values (D: consistent with the
delivered bytes). `span` = carriers. At L5-9 `first_bin`/`stride` are replaced by centre and spacing in Hz,
because the carriers sit on no FFT grid. `ncols` = symbols per over (per band at L5-9). `lead` = lead symbols.

| L | rec | law | n_info | coded | frame_bytes | bpc | dw50 | cp | span | first_bin / stride | ncols | lead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4 | sub-band QPSK (M) | 1080 D | 2172 D | 135 D | 2 M | 512 M | 0 M | 3 M | 900 Hz / 600 Hz M | 383 M | U (21-symbol surplus) |
| 6 | 5 | sub-band QPSK (M) | 1410 D | 2832 D | 176 D | 2 M | 512 M | 0 M | 4 M | 750 / 500 M | 372 M | U (18) |
| 7 | 6 | sub-band QPSK (M, noisy) | 2076 D | 4164 D | 259 D | 2 M | 512 M | 0 M | 6 M | 700 / 320 M | 362 D | U (15) |
| 8 | 7 | sub-band QPSK (M) | 2754 D | 5520 D | 344 D | 2 M | 512 M | 0 M | 8 M | 700 / 228.57 M | 359 M | U (14) |
| 9 | 8 | sub-band QPSK (M) | 3424 D | 6860 D | 428 D | 2 M | 512 M | 0 M | 10 M | 555 / 210 M | 356 M | U (13) |
| 10 | 9 | const, pos 1 | 7197 D | 14406 D | 899 D | 2 M | 1152 M | 128 M | 49 M | 8 / 1 M | 153 M | 5 M |
| 11 | 10 | const, pos 1 | 9596 D | 14406 D | 1199 D | 2 M | 1152 M | 128 M | 49 M | 8 / 1 M | 153 M | 5 M |
| 12 | 11 | const, pos 1 | 11512 D | 14406 D | 1439 D | 2 M | 1152 M | 128 M | 49 M | 8 / 1 M | 153 M | 5 M |
| 13 | 12 | const, pos 2 | 14398 D | 21609 D | 1799 U (74 B final only) | 3 M | 1152 M | 128 M | 49 M | 8 / 1 M | 153 M | 5 M |
| 14 | 13 | const, pos 2 | 17276 D | 21609 D | 2159 D | 3 M | 1152 M | 128 M | 49 M | 8 / 1 M | 153 M | 5 M |
| 15 | 14 | const, pos 3 + 980 pilots | 21710 D | 26068 D | 2713 D | 4 M | 1152 M | 128 M | 49 M | 8 / 1 M | 159 M | 5 M |
| 16 | 15 | const, pos 4 + 980 pilots + 49 fixed | 26056 D | 32585 D | 3257 D | 5 M | 1152 M | 128 M | 49 M | 8 / 1 M | 159 (313 double) M | 5 M |

Still U at every level: the per-cell scramble table (measured only as lattice-valued, shared and
payload-independent), the cell fill order, the puncture patterns, pos=4 labels, and the role of the 49 extra
cells per frame at L10-14 (at L16 they are fixed and pilot-like).

## Disagreements with sigidwiki

None of the Bd, carrier count or modulation cells is contradicted at any level from L4 to L16. L4's "16c FSK"
is the 16-bin one-hot comb. L5-9 are 94 Bd with 3/4/6/8/10 carriers and QPSK. L10-16 are 41.67 Bd with 49
carriers, and the modulation matches. The bps column is not contradicted: N over the keyed duration comes to
264/355/536/718/900/1961/2614/3136/3922/4706/5690/6829 bps, 2-3 % below every wiki figure, so the wiki divides
by a slightly shorter time. The review's point stands in one respect: the QAM step on tape is exactly 6
symbols (+0.144 s, a ratio of 1.039), not the wiki's implied 1.033.

## Disagreements with our current records

1. **rec9-15 carriers and band.** The rows hold span 32 on bins 9..40 (422-1875 Hz). The tape has **49 on
   bins 8..56** (375-2625 Hz). 147x49 and 153x49 from RECORD-SEMANTICS §4 hold, with the 980 identified as
   pilots.
2. **rec9-15 burst length.** `burst_length` gives 226 symbols (260352 samples) at rec9-13 and 204 at
   rec14/15. The tape gives **153 (176256) and 159 (183168)**. 06's "over 4.32 s" is really 3.672 / 3.816 s.
3. **Lead and pilots.** The const rows assume no preamble and no pilots. The tape has a 5-symbol lead at
   every level, plus 980 scattered pilot cells at rec14/15 (and 49 more fixed cells at rec15). kestrel's
   "forced cells ignored" discards the pilots.
4. **Phase law.** kestrel's `_synth_const` applies no twiddle. The tape has a payload-independent per-cell
   phase on the 2π/128 lattice at every const level, consistent with 01's `exp(j2πS/128)`. It is the same
   table at every level, so it depends only on (symbol, bin). 01's raised-cosine OLA fits the ~10-sample edge
   taper.
5. **rec4-8.** 06 gives dw50 512, span 16, "LUT constellation, QPSK (unit)". dw50 512, CP 0 and QPSK hold. The
   16-bin comb does not: these are 3-10 staggered sub-band carriers at 600/500/320/228.57/210 Hz spacing, off
   any FFT raster, with a fixed per-symbol phase scramble. Our ladder has no record type for them. Their
   nearest relative is the BW500 base (varahf500's sub-band DBPSK).
6. **rec16 ("extra rung").** No keying uses a 213-byte frame. The L16 double over is two rec15 frames (6508 B),
   which supports the review: rec16 is not a speed level.
