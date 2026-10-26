<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# BW2750 levels 5-10 and BW500 levels 5-6: the tone family, closed from known plaintext

`kpa_tone_bw2750_bw500.py` prints every table below in about 12 minutes. It reuses
`kpa_bw2300_subband.py` (unchanged) for the chain and imports that ladder's rows 2-9 and pulses, cached in
`$KPA_CACHE`. Tapes are read for measurement only. The level is the host `BITRATE (N)`.

## Result

**124 of the 126 data overs decode byte-exact on all bands. Every clean over has zero channel bit errors
before FEC.** The waveform is BW2300 L5-9's (KPA-bw2300-subband.md) with no change of law. Only
per-row parameters differ, and BW2750 L10 adds a record and three grid rows.

| row | overs | byte-exact | clean overs (floor digital silence or residual < -45 dB): raw bit errors | noisy overs | failures |
|---|---|---|---|---|---|
| BW2750 L5 | 11 | 11 | 0/2172 (x2) | 0-105 | - |
| BW2750 L6 | 32 | 31 | 0/2832 (x4) | 0-239 | 050336Z 1912763 (lock 0.10, floor -12 dB) |
| BW2750 L7 | 5 | 5 | 0/4164 (x3, one a short frame) | 45-77 | - |
| BW2750 L8 | 5 | 5 | 0/5520 (x3, one short) | 15-65 | - |
| BW2750 L9 | 9 | 9 | 0/6860 (x1, the only clean one) | 2-187 | - |
| BW2750 L10 | 21 | 21 | 0/8944 (x9, three short) | 0-136 | - |
| BW500 L5 | 24 | 24 | 0/1500 (x3); short 8292144 4/1500 | 0-22 | - |
| BW500 L6 | 19 | 18 | 0/2172 (x4); 6005989 43/2172 (burst rms off modal) | 0-8 | 053832Z 7601552 (lock 0.07, floor -10 dB) |

Both failures are the noisiest overs of their rows (floor -12 and -10 dB, lock 0.10 and 0.07).

## Differences from BW2300 L5-9

1. **Centres.** BW2750: `1500 + step (b - (nb-1)/2)` with steps 900/660/420/320/250/210 (RASTER-bw2750.md,
   confirmed to 0.001 Hz by the de-rotated phase ramp). BW500 L5: 1350/1650. **BW500 L6: 1320/1500/1680
   (step 180), not RASTER-bw500's 1312.5/1687.5.** At 187.5, bands 0 and 2 ramp +-7.500 Hz.
2. **Pulse is per level, not per bandwidth.** L5/L6 (BW2750) and BW500 L5 use BW500's 513-tap pulse
   (-64.6 dB). L7/L8 (BW2750) and **BW500 L6** use BW2300's L8 fit (P8, -54.2 to -55.1 dB, 2e-3 energy
   beyond +-512). L9/L10 use a confined pulse (P9w, -73 to -75 dB, 7e-5 beyond +-512). Crossed, P8 and P9
   give each other -27 dB, which is exactly P8's tail energy. So they are two pulses, not two fits of one.
   P9w is BW2300's P9 refitted on BW2750 L9. It sits -45.7 dB from P9, and P9 reads BW2750 L9 at only
   -46 dB. Whether BW2300 L9's pulse differs from BW2750's at that level or P9 was not fully converged is
   UNKNOWN. BW2300's P7 fit is poor (-21 dB here), so P8 serves L7.
3. **Stagger.** BW2750 L5 0/176/336 and L6-10 0/256 alternating, as at BW2300. BW500 L5 0/256. **BW500 L6
   0/176/336** (-54.2 dB against -34.8 at 0/171/341 and -7.7 at 0/256).
4. **Records.** BW2750 L5-9 are BW2300 records 4-8 (seeds and lengths). **BW2750 L10 is a new record
   (15908083, 8944, 4466)**, and its stage-1 seed is where L9's pilot draws close, the same state as BW2300
   record 9's seed. It decoded CRC-clean at the first try with bands 10-12 erased. BW500 L5/L6 are
   **BW500** records 4/5 (9007905 / 11166657), not BW2300's.
5. **Pilots.** The same law at every row: per band, NPIL distinct `t = 13 + Int(Rnd*(ncols-6))`,
   continuing the record's stream. Every held chain closes exactly on the next record's seed. The draws
   are BW2750 37/36/36/40/41 (identical to BW2300) and BW500 L5 38 (2x18, span 387) and L6 36 (3x12, span
   368). L10's 39 draws (13x3, span 341) end on **11548315**, the predicted stage-1 seed of BW2750's next
   record (L11). That is DERIVED and untested.
6. **Grid rows.** Band b reads grid row b at `t - 8` at every row: 1.000 collapse on its own row, <= 0.107 on
   any other. BW2750 reuses rows 0-9 in BW2300's order. **BW500 L6's third sub-band is row 2**, the row
   RASTER-bw500 could not lock (it tried only grid480's rows 0/1). **L10's bands 10-12 need rows 10-12**,
   measured here as literals (t 8..355, up to a per-row constant). Six clean overs agree on every cell,
   the worst cell sits 0.19 unit off the lattice, and no shift or reversal of rows 0-9 or grid480 matches
   them (best 0.14, chance). **BW500 L5 runs to t = 401** (402 symbols), one past grid480's rows. Index
   393 of rows 0 and 1 is measured (3 overs agree), and t = 400 matches grid480 index 392.
7. **Lead.** 8 symbols at every row: cells at t = -2, -1 sit 57-120 dB below the data. t 0..7 are at data
   power. The content is UNKNOWN, as at BW2300.
8. **Frame.** payload | control | CRC-16/GENIBUS, floor(N/8) bytes, unchanged. Control bytes seen are
   0x19 on the first over, then 0x99, 0x95, 0x91, 0x8d, 0x89 and 0x81, and 0x82 on short frames. Every short
   frame ends `14 09`, zeros, then **`06` (BW2750) or `07` (BW500 L5)** as the last byte before the
   control byte. Meaning UNKNOWN.

## Proposed rows

M = MEASURED here, D = DERIVED, U = UNKNOWN. Every row has law `subband` (M), differential QPSK with Gray
[0,1,3,2] (M), dw50 512 (M), cp 0 (M), lead 8 plus reference at t = 8 (M; content U), grid row b = band b
(M), pilot law `13 + Int(Rnd*(ncols-6))` (M), rate 1/2 with no puncture (M), and band-major fill (M).

| row | record (seed, coded, N) | frame B | bands | centres Hz | stagger | nsym | npil | pulse | grid rows |
|---|---|---|---|---|---|---|---|---|---|
| BW2750 L5 | BW2300 rec4 M | 135 M | 3 M | 600/1500/2400 M | 0,176,336 M | 383 M | 12 M | P513 M | 0-2 M |
| BW2750 L6 | BW2300 rec5 M | 176 M | 4 M | 510 + 660b M | 0/256 M | 372 M | 9 M | P513 M | 0-3 M |
| BW2750 L7 | BW2300 rec6 M | 259 M | 6 M | 450 + 420b M | 0/256 M | 362 M | 6 M | P8 M (-54.6) | 0-5 M |
| BW2750 L8 | BW2300 rec7 M | 344 M | 8 M | 380 + 320b M | 0/256 M | 359 M | 5 M | P8 M (-55) | 0-7 M |
| BW2750 L9 | BW2300 rec8 M | 428 M | 10 M | 375 + 250b M | 0/256 M | 356 M | 4 M | P9w M (-75) | 0-9 M |
| BW2750 L10 | (15908083, 8944, 4466) M; next seed 11548315 D | 558 M | 13 M | 240 + 210b M | 0/256 M | 356 M | 3 M | P9w M (-73) | 0-12 M (10-12 literals) |
| BW500 L5 | BW500 rec4 M | 93 M | 2 M | 1350/1650 M | 0,256 M | 402 M | 18 M | P513 M (-64.7) | 0-1 M (+ index 393) |
| BW500 L6 | BW500 rec5 M | 135 M | 3 M | 1320/1500/1680 M | 0,176,336 M | 383 M | 12 M | P8 M (-54.2) | 0-2 M |

Still UNKNOWN: the lead content, the control-byte semantics and the short-frame fill, the grid file
layout for rows >= 2 (and whether they are generated at all), P8's true extent beyond +-1024, and whether P9
and P9w are one pulse.

## Evidence per over

The full per-over table (floor, lock, model residual, raw bit errors, CRC, payload match, frame tail) is the
script's last section. The clean overs the rows rest on:

- BW2750 L5 050512Z 2377737, 050336Z 2136503: -64.6 dB, 0/2172.
- BW2750 L6 050204Z 2277439, 050336Z 2359418 / 3449724, 050512Z 3356427: -64.6 dB, 0/2832.
- BW2750 L7 050204Z 1186505, 050512Z 1192602, 050512Z 4658862 (short, 180 B): -54 dB, 0/4164.
- BW2750 L8 manifest 2195449, 050336Z 1194765, 050204Z 3463376 (short, 276 B): -55 dB, 0/5520.
- BW2750 L9 045941Z 1142793: -75.2 dB, 0/6860. The second clean-looking instance, 045304Z 695690, is noisy
  (floor -24 dB) and still decodes with 2/6860.
- BW2750 L10 045556Z 1118866 / 1330308 (short 470 B), 045853Z 1130349, 050029Z 1138415, 050116Z 1135522,
  050204Z 2495703, 050512Z 3574938, 045139Z 1067991 and 045221Z 1115380 (short 240 B): 0/8944.
- BW500 L5 053832Z 2237489 / 3404588 / 5771825: -64.5 dB, 0/1500.
- BW500 L6 053832Z 2472384 / 3639021 / 4500692 / 6963682: -54.2 dB, 0/2172.

How it fell: the BW2300 chain and rows decoded BW2750 L5-9 byte-exact at the first attempt. The RASTER
centres and staggers were used, and the pulses were then chosen by residual. BW500 L5 did the same once
the 0/256 stagger was read off a per-band sampling scan. BW500 L6 decoded with 12 raw errors on the
wrong pulse. Row 2 showed at collapse 0.87 against <= 0.10, and the phase ramp then moved the centres to
180 Hz spacing, which P8 fits at -54 dB. L10 decoded on the chained seed with its three new bands erased,
and those rows were then read off six clean overs.

## What must change in the product (not touched here)

- `rx/tablegen.py` `BW500`/`BW2300` (lines 141ff): BW2750 needs its own record table. Records 4-8 are
  BW2300's, and record 9 is (15908083, 8944, 4466). Its record 10 seed is predicted at 11548315.
- `rx/tablegen.py` grid480 (`grid480`, line 880): add row 0/1 index 393 and rows 10-12 as measured literals
  (print them with this script). Rows 2-9 as KPA-bw2300-subband.md says.
- `rx/tablegen.py` pilots: the one generator serves BW500 records 3-5, BW2300 records 4-8 and BW2750
  records 4-9, given (nb, npil, ncols) per row.
- `rx/varahf2300.py:214` ("Records 4-8 are absent") and `:233` `RECORDS`: the `SubbandRecord` proposed in
  KPA-bw2300-subband.md needs per-row centres (not a single f0/step law at BW500 L6's 180 Hz), a
  stagger tuple and a pulse name. BW2750 L5-10 need keys 104-109.
- `rx/varahf500.py:127` `SUBBANDS` and `:251` `BASE_LEVEL`: BW500 L5/L6 are the sub-band family on BW500
  records 4/5, with 2 or 3 sub-bands, DQPSK, lead 8 and 18 or 12 pilots per sub-band. They are not DBPSK
  and use no `usedmap`.
- `tx/bw500_pulse.py`: reused unchanged at BW2750 L5/L6 and BW500 L5. The P8 and P9w taps need a
  measured-taps module.
- RASTER-bw500.md L6 row ("1312.5 / 1500 / 1687.5", "third reference U") is superseded by 1320/1500/1680 and
  row 2. RASTER-bw2750.md's "coherent QPSK" reading is superseded: it is differential, as at BW2300.
