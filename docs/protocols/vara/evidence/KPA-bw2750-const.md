# BW2750 levels 11-17 closed from known plaintext

`kpa_bw2750_const.py` prints every table below (about 3 min, nothing written). It runs the BW2300 chain
(`KPA-bw2300-const.md`, now shared through `kpa_common.py`) on 59 carriers and demodulates 69 frames off
the high speed capable bench tapes: 65 overs, four of them double L17 overs. **All 69 frames are CRC-clean and
byte-exact** against the transcript's delivered bytes: 8 at L11, 6 at L12, 3 at L13, 4 at L14, 14 at L15,
11 at L16 and 23 at L17. Raw on-air bit errors before FEC are 0 on 61 frames. The other 8 are the four
L14 overs (529 / 166 / 78 / 173 bits; every L14 over on tape carries the sweep's injected noise) and four
L17 overs (1 and 2 bits on 4156426 and 2135986; 530 and 670 on the noisy 2183670 and 2181946). On 57 of the
59 clean-floor frames with no raw error, every interior cell equals the chain's prediction to 1e-3 of the
reference. The two exceptions (L15 1530250 and 1529086) carry a constant −0.35°/bin phase ramp, which is
the raster grid one sample late on that tape and invisible to differential PSK.

## Findings

Everything below the carrier set is the BW2300 law. What is BW2750's own is the carrier set, the
interleaver seeds (one chain, no new generator) and the Rnd footprints that scale with the carrier count.

1. **Carriers and window.** 59 carriers on FFT bins 3..61 (1024-point, 46.875 Hz), 1152-sample symbol.
   Cells come from `raster_bw2300.ofdm_cells(backoff=16, bins=slice(3, 62))`, the window 112 samples into
   the symbol as at BW2300. The PSK reference reads exactly +1 on every carrier.
2. **Scramble.** The same S table: `Int(Rnd*129)` from the whitener seed 29318, 409 columns per FFT-bin
   row from bin 0. BW2750 reads rows 3..61, so bin 3 column 0 is draw 1228 = 1 + 3·409. The table needs
   62 rows (BW2300 used 57). Column origin and the per-frame restart are unchanged.
3. **Interleavers: one Rnd chain, continuous with BW2300's.** BW2750 shares `tablegen.BW2300` up to record
   9's seed 15908083. At BW2750 that seed opens L10 (the 13-tone level) with stage 1 over 8944 and stage 2
   over 4466, then a 39-draw gap. The state left is **11548315, L11's stage-1 seed**. A 2^24 search of
   L12's and L11's first carrier against the known systematic bits found it on tape independently, and
   L10's (8944, 4466, 39) is the only neighbouring choice that lands on it (4465, 4467, 8943 and 8946 do
   not). From L11, each level runs stage 1 (`coded`), stage 2 (`N`), then **a 59-draw gap** (one per
   carrier), and the state left is the next level's seed. Every seed through L17 then decodes. So the N
   values below are confirmed by the chain as well as by the decode: a wrong N moves every later seed.
   L17's chain ends on 4061299. That is the analogue of BW2300's record 16 seed, and no level uses it.
4. **Gap cells.** One per carrier, bin-major from bin 3. PSK: repeat cell at symbol `7 + Int(Rnd*142)`.
   QAM: the +1 cell right after pilot row `Int(Rnd*19)`. The same law as BW2300 (the BW500 closure has
   4 per carrier with a redraw on repeats; `kpa_common.Level.per_bin` covers all three).
5. **Pilots (L16-17).** Absolute-even bins at frame row 8j, absolute-odd bins at 8j+1. That is the BW2300
   rule by absolute bin, so bin 3 (index 0) carries its pilots on the odd row. With 20 pilot rows plus the
   extra cell, each carrier has 133 data cells, and 59 × 133 × bpc = `coded`.
6. **Coding per level.** The frame is payload + control byte + CRC-16/GENIBUS. Coded length is 59 × 147 ×
   bpc (PSK) or 59 × 133 × bpc (QAM). The group puncture is the BW2300 law with k = 1/2/4/2/4/5/4 at
   L11-17. The rate and modulation pairs are therefore L11 QPSK-1/2, L12 QPSK-2/3, L13 QPSK-4/5,
   L14 8PSK-2/3, L15 8PSK-4/5, L16 16QAM-5/6 and L17 32QAM-4/5. These are BW2300's records 9-15, shifted
   one level up and widened to 59 carriers. LUT columns are pos 1/1/1/2/2/3/4, indexed by the raw bit
   group.
7. **Lead.** `LUT[Int(Rnd·2^bpc)]` bin-major from the session stream: 4 × 59 = **236 draws** per PSK over
   and 5 × 59 = **295** per QAM frame. A double L17 over consumes 590, although only one lead is keyed.
   Consecutive decoded overs of a tape differ by 236, 295, 590 or sums of them (472, 531), except where
   other keyings sit between them. The state is recovered on 62 of 65 overs. The misses are one noisy
   L13 over and two overs whose symbol 0 carries the edge taper (L15 1533662, L16 1741375).
8. **Frame tail.** Full frames end with one control byte: 0x99, 0x95, 0x91, 0x89, 0x85, 0x81 seen. Short
   frames read data, `14 09`, zero fill, `06`, `82`, CRC, as at BW2300.
9. **RASTER-bw2750's open "extra symbol"** is the reference symbol 4 (PSK), plus 147 data cells and 1
   repeat per carrier, or the extra +1 cell (QAM). Its derived N for L13, L15, L16 and L17 (≈13867,
   ≈20805, ≈26147, ≈31378) resolve to 13864, 20804, 26145 and 31376. Its derived rows for L11, L12 and
   L14 hold exactly.

## Proposed Record rows (BW2750 const law)

M = measured by this decode (the chain reproduces the tape bit-exactly); D = derived. Every row: law const,
dw50 1152 (M), cp 128 (M), FFT 1024, span 59 (M), first_bin 3 (M), window 112 into the symbol (M), scramble
rows 3..61 (M), 1 gap cell per carrier (M), lead 4 PSK / 5 QAM (M), frame_bytes = payload + ctl + CRC (M).
Keys follow the 100 + record convention.

| key (L) | name | n_info | coded | frame_bytes | bpc | pos | k | pad | stage-1 seed | symbols |
|---|---|---|---|---|---|---|---|---|---|---|
| 109 (10) | 13-tone QPSK-1/2 | 4466 D | 8944 D | 558 D | 2 | 1 | 1 | 0 | 15908083 D | tone raster (RASTER-bw2750) |
| 110 (11) | QPSK-1/2 | 8667 M | 17346 M | 1083 M | 2 M | 1 M | 1 M | 0 | 11548315 M | 153 M |
| 111 (12) | QPSK-2/3 | 11556 M | 17346 M | 1444 M | 2 M | 1 M | 2 M | 0 | 12091152 M | 153 M |
| 112 (13) | QPSK-4/5 | 13864 M | 17346 M | 1733 M | 2 M | 1 M | 4 M | 4 | 8109825 M | 153 M |
| 113 (14) | 8PSK-2/3 | 17338 M | 26019 M | 2167 M | 3 M | 2 M | 2 M | 0 | 15712339 M | 153 M |
| 114 (15) | 8PSK-4/5 | 20804 M | 26019 M | 2600 M | 3 M | 2 M | 4 M | 2 | 367472 M | 153 M |
| 115 (16) | 16QAM-5/6 | 26145 M | 31388 M | 3268 M | 4 M | 3 M | 5 M | 2 | 2608703 M | 5 + 154 M |
| 116 (17) | 32QAM-4/5 | 31376 M | 39235 M | 3922 M | 5 M | 4 M | 4 M | 3 | 10769080 M | 5 + 154·(1 or 2) M |

The L10 row is D: only its Rnd footprint is tested (it must be 8944/4466 with a 39-draw gap for L11's
seed to follow), and its tone raster has not been decoded. The 39-draw gap is 3 × 13 tones, so it
presumably holds three cells per tone. L5-L9 at BW2750 reuse BW2300 records 4-8's seeds, since the chain
runs through them unchanged (D). Suggested `tablegen.BW2750` = `BW2300[:9]` + the rows above as
`(seed, coded, N)`, closing on 4061299.

## Evidence per over

`dev` = max |rx/g − expected| over interior symbols / first-and-last symbol (edge taper). `draws` = Rnd
draws from the previous decoded over of the same tape. Noisy-floor overs (sweep noise on): L11 3293984,
L13 3237068, all of L14 except 4139783 (and 4139783 carries noise over 67 symbols despite a silent floor),
L17 2183670 / 2181946.

| level | tape | measured_on | frame | payload (first, len) | raw bit errors / on-air bits | dev interior / edge | CRC / byte-exact | lead Rnd state | draws |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 20260926T045304Z-ladder | 1117460 | 0 | (129, 895) | 0 / 17346 | 0.0001 / 0.092 | True/True | 9662054 |  |
| 11 | 20260926T045347Z-ladder | 1117856 | 0 | (129, 895) | 0 / 17346 | 0.0001 / 0.092 | True/True | 9662054 |  |
| 11 | 20260926T045430Z-ladder | 1118432 | 0 | (129, 895) | 0 / 17346 | 0.0001 / 0.092 | True/True | 9662054 |  |
| 11 | 20260926T045644Z-ladder | 1118472 | 0 | (129, 895) | 0 / 17346 | 0.0001 / 0.092 | True/True | 9662054 |  |
| 11 | 20260926T045853Z-ladder | 1341567 | 0 | (50, 974) | 0 / 17346 | 0.0001 / 0.621 | True/True | 884220 |  |
| 11 | 20260926T050029Z-ladder | 1349389 | 0 | (9, 1015) | 0 / 17346 | 0.0001 / 0.099 | True/True | 5220357 |  |
| 11 | 20260926T050116Z-ladder | 1347277 | 0 | (9, 1015) | 0 / 17346 | 0.0001 / 0.099 | True/True | 5220357 |  |
| 11 | 20260926T060325Z-ladder | 3293984 | 0 | (220, 1080) | 0 / 17346 | 0.0001 / 0.095 | True/True | 3044100 |  |
| 12 | N0CALL-W9SSJ-bw2750-counter65536 | 2408974 | 0 | (174, 1441) | 0 / 17346 | 0.0001 / 0.079 | True/True | 7405647 |  |
| 12 | 20260926T060143Z-ladder | 3317048 | 0 | (105, 1441) | 0 / 17346 | 0.0001 / 0.083 | True/True | 6996161 |  |
| 12 | 20260926T045513Z-ladder | 1121093 | 0 | (171, 1109) | 0 / 17346 | 0.0001 / 0.022 | True/True | 2982307 |  |
| 12 | 20260926T045727Z-ladder | 1122270 | 0 | (86, 1194) | 0 / 17346 | 0.0001 / 0.058 | True/True | 10558545 |  |
| 12 | 20260926T045810Z-ladder | 1123475 | 0 | (2, 1278) | 0 / 17346 | 0.0001 / 0.087 | True/True | 14024591 |  |
| 12 | 20260926T045941Z-ladder | 1354782 | 0 | (94, 1186) | 0 / 17346 | 0.0001 / 0.154 | True/True | 10558545 |  |
| 13 | 20260926T055838Z-ladder | 3444906 | 0 | (133, 1659) | 0 / 17346 | 0.0001 / 0.118 | True/True | 10206749 |  |
| 13 | 20260926T060010Z-ladder | 3314418 | 0 | (96, 1730) | 0 / 17346 | 0.0001 / 0.131 | True/True | 6996161 |  |
| 13 | 20260926T055838Z-ladder | 3237068 | 0 | (195, 1730) | 0 / 17346 | 0.4587 / 0.818 | True/True | none |  |
| 14 | 20260926T060325Z-ladder | 4139783 | 0 | (31, 1761) | 529 / 26019 | 0.7785 / 0.584 | True/True | 4906929 |  |
| 14 | 20260926T055838Z-ladder | 858036 | 0 | (216, 2164) | 166 / 26019 | 0.4986 / 0.444 | True/True | 6476916 |  |
| 14 | 20260926T055838Z-ladder | 1276592 | 0 | (113, 2164) | 78 / 26019 | 0.4872 / 0.299 | True/True | 2873612 |  |
| 14 | 20260926T055838Z-ladder | 3028462 | 0 | (79, 2164) | 173 / 26019 | 0.5030 / 0.447 | True/True | 7175077 |  |
| 15 | N0CALL-W9SSJ-bw2750-counter65536 | 2618120 | 0 | (79, 2597) | 0 / 26019 | 0.0001 / 0.060 | True/True | 3508683 |  |
| 15 | 20260926T055838Z-ladder | 1485478 | 0 | (229, 2597) | 0 / 26019 | 0.0001 / 0.235 | True/True | 1866296 |  |
| 15 | 20260926T060010Z-ladder | 1533662 | 0 | (130, 2597) | 0 / 26019 | 0.0001 / 0.634 | True/True | none |  |
| 15 | 20260926T060010Z-ladder | 3522801 | 0 | (34, 2526) | 0 / 26019 | 0.0001 / 0.216 | True/True | 10206749 |  |
| 15 | 20260926T060143Z-ladder | 1531438 | 0 | (175, 2597) | 0 / 26019 | 0.0001 / 0.171 | True/True | 1866296 |  |
| 15 | 20260926T060143Z-ladder | 3524661 | 0 | (10, 2597) | 0 / 26019 | 0.0001 / 0.114 | True/True | 10206749 |  |
| 15 | 20260926T060143Z-ladder | 3733809 | 0 | (47, 2597) | 0 / 26019 | 0.0001 / 0.077 | True/True | 14897081 |  |
| 15 | 20260926T060143Z-ladder | 3943482 | 0 | (84, 2476) | 0 / 26019 | 0.0001 / 0.059 | True/True | 10523029 |  |
| 15 | 20260926T060325Z-ladder | 1537026 | 0 | (208, 2597) | 0 / 26019 | 0.0001 / 0.159 | True/True | 1866296 |  |
| 15 | 20260926T060325Z-ladder | 3500816 | 0 | (20, 2597) | 0 / 26019 | 0.0001 / 0.092 | True/True | 13838512 |  |
| 15 | 20260926T060325Z-ladder | 3930126 | 0 | (250, 2597) | 0 / 26019 | 0.0001 / 0.211 | True/True | 10523029 |  |
| 15 | 20260926T060511Z-ladder | 1530250 | 0 | (82, 2597) | 0 / 26019 | 0.3721 / 0.439 | True/True | 1866296 |  |
| 15 | 20260926T060511Z-ladder | 3501208 | 0 | (150, 2597) | 0 / 26019 | 0.0001 / 0.129 | True/True | 13838512 |  |
| 15 | 20260926T060706Z-ladder | 1529086 | 0 | (82, 2597) | 0 / 26019 | 0.3721 / 0.439 | True/True | 1866296 |  |
| 16 | N0CALL-W9SSJ-bw2750-counter65536 | 2828308 | 0 | (116, 3265) | 0 / 31388 | 0.0006 / 0.144 | True/True | 7280007 |  |
| 16 | N0CALL-W9SSJ-bw2750-counter65536 | 5982988 | 0 | (135, 2937) | 0 / 31388 | 0.0003 / 0.101 | True/True | 11830518 |  |
| 16 | 20260926T055838Z-ladder | 1695671 | 0 | (10, 3265) | 0 / 31388 | 0.0003 / 0.052 | True/True | 10988196 |  |
| 16 | 20260926T060010Z-ladder | 1743611 | 0 | (167, 3265) | 0 / 31388 | 0.0005 / 0.107 | True/True | 10988196 |  |
| 16 | 20260926T060143Z-ladder | 1741375 | 0 | (212, 3265) | 0 / 31388 | 0.0005 / 0.444 | True/True | none |  |
| 16 | 20260926T060325Z-ladder | 1747219 | 0 | (245, 3265) | 0 / 31388 | 0.0001 / 0.052 | True/True | 10988196 |  |
| 16 | 20260926T060325Z-ladder | 3711527 | 0 | (57, 3265) | 0 / 31388 | 0.0003 / 0.098 | True/True | 10741148 |  |
| 16 | 20260926T060511Z-ladder | 1740699 | 0 | (119, 3265) | 0 / 31388 | 0.0003 / 0.092 | True/True | 10988196 |  |
| 16 | 20260926T060511Z-ladder | 3710896 | 0 | (187, 3265) | 0 / 31388 | 0.0003 / 0.120 | True/True | 10741148 |  |
| 16 | 20260926T060511Z-ladder | 3929482 | 0 | (124, 3265) | 0 / 31388 | 0.0003 / 0.077 | True/True | 10523029 |  |
| 16 | 20260926T060706Z-ladder | 1739023 | 0 | (119, 3265) | 0 / 31388 | 0.0003 / 0.092 | True/True | 10988196 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 3047432 | 0 | (53, 3919) | 0 / 39235 | 0.0003 / 0.065 | True/True | 13654652 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 3268881 | 0 | (132, 3919) | 0 / 39235 | 0.0003 / 0.106 | True/True | 9341941 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 3490325 | 0 | (211, 3919) | 0 / 39235 | 0.0005 / 0.107 | True/True | 8172482 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 3712017 | 0 | (34, 3919) | 0 / 39235 | 0.0008 / 0.203 | True/True | 6284787 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 3934734 | 0 | (113, 3919) | 0 / 39235 | 0.0002 / 0.057 | True/True | 8963544 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 4156426 | 0 | (192, 3919) | 1 / 39235 | 0.0010 / 0.242 | True/True | 14864129 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 4380167 | 0 | (15, 3919) | 0 / 39235 | 0.0001 / 0.000 | True/True | 8869535 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 4380167 | 1 | (94, 3919) | 0 / 39235 | 0.0005 / 0.126 | True/True |  |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 4781078 | 0 | (173, 3919) | 0 / 39235 | 0.0001 / 0.000 | True/True | 1888333 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 4781078 | 1 | (252, 3919) | 0 / 39235 | 0.0005 / 0.140 | True/True |  |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 5182218 | 0 | (75, 3919) | 0 / 39235 | 0.0001 / 0.000 | True/True | 12170123 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 5182218 | 1 | (154, 3919) | 0 / 39235 | 0.0003 / 0.083 | True/True |  |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 5583112 | 0 | (233, 3919) | 0 / 39235 | 0.0001 / 0.000 | True/True | 5536729 |  |
| 17 | N0CALL-W9SSJ-bw2750-counter65536 | 5583112 | 1 | (56, 3919) | 0 / 39235 | 0.0005 / 0.136 | True/True |  |  |
| 17 | 20260926T055838Z-ladder | 1914290 | 0 | (203, 3919) | 0 / 39235 | 0.0005 / 0.125 | True/True | 3365437 |  |
| 17 | 20260926T055838Z-ladder | 2135986 | 0 | (26, 3919) | 2 / 39235 | 0.0005 / 0.331 | True/True | 9311914 |  |
| 17 | 20260926T060010Z-ladder | 1962486 | 0 | (104, 3919) | 0 / 39235 | 0.0005 / 0.126 | True/True | 3365437 |  |
| 17 | 20260926T060010Z-ladder | 2183670 | 0 | (183, 3919) | 530 / 39235 | 0.5702 / 0.447 | True/True | 9311914 |  |
| 17 | 20260926T060143Z-ladder | 1960250 | 0 | (149, 3919) | 0 / 39235 | 0.0004 / 0.113 | True/True | 3365437 |  |
| 17 | 20260926T060143Z-ladder | 2181946 | 0 | (228, 3919) | 670 / 39235 | 0.5940 / 0.624 | True/True | 9311914 |  |
| 17 | 20260926T060325Z-ladder | 1965838 | 0 | (182, 3919) | 0 / 39235 | 0.0002 / 0.050 | True/True | 3365437 |  |
| 17 | 20260926T060511Z-ladder | 1959062 | 0 | (56, 3919) | 0 / 39235 | 0.0005 / 0.126 | True/True | 3365437 |  |
| 17 | 20260926T060706Z-ladder | 1957898 | 0 | (56, 3919) | 0 / 39235 | 0.0005 / 0.126 | True/True | 3365437 |  |

How it was found: the S table was tested first with BW2300's rows 3..61, and 100 % of differential steps
landed on the QPSK grid with the reference at +1. For the interleaver, the group law says which coded
positions are systematic and what they carry. A 2^24 vectorised filter over stage-1 seeds, matching
`coded[π(k)]` against the first carrier's hard bits for each possible repeat symbol, returned one seed
per level (12091152 at L12, 11548315 at L11). The L12 repeat cells on bins 3-5 (108/109, 93/94, 41/42)
then fixed N = 11556 and the gap origin at bin 3. Chaining L11 → L12 with N = 8667 and 59 draws hit
12091152 exactly, and every later level decoded off the chain.

## What must change

BW2300's list in `KPA-bw2300-const.md` applies unchanged (scramble, gap cells, differential PSK, pilots,
group puncture, LUT without clsparm, lead, control byte). BW2750 adds the following:

| where | change |
|---|---|
| `rx/tablegen.py:124-130` | add `BW2750` = `BW2300[:9]` + `(15908083, 8944, 4466)` + the seven `(seed, coded, N)` rows above. |
| `rx/tablegen.py:133`, `:152-176` (`_ROWS`, `interleaver`) | BW2750 L17's stage 1 is 39235 long, beyond the 32585-row grid, so BW2750's permutations come from `permutation()` on its own spec rather than from the 17-column BW2300 array. |
| `rx/tablegen.py` (scramble table, BW2300 item) | generate at least 62 rows (bins 0..61), not 57. Gap cells: 59 draws per record at BW2750, 49 at BW2300. |
| `rx/bw2750_allocations.py` | no change: the const levels carry no allocation table. The module docstring (:4-20) can say the high levels need none. |
| `rx/varahf2300.py:99-103` | add keys 110-116 (and 109 for L10 once its tone raster is closed). |
| `rx/varahf2300.py:227-231` (comment) | delete "dw50=1152 / cp=128 geometry below is BW2300's … ~1280 samples": BW2750 high levels are 1024 + 128 on bins 3..61. |
| `rx/varahf2300.py:233-254` (`RECORDS`) | add the rows above: `span 59`, `first_bin 3`, `dw50 1152`, `cp 128`, with the pos, bpc and frame_bytes shown. |
| `rx/varahf2300.py:169-191` (`Record`), `:304-323` (`turbo_perm`, `chan_perm`) | `il` as a column index into BW2300's 17-column array cannot express BW2750's records. Give the record its interleaver spec (seed, coded, N) or a bandwidth-keyed table, and build the permutations from it. |
| `rx/varahf2300.py:256-270` (`BASE_LEVELS`, `INDEX_LEVELS_BW`) | add a BW2750 const family so the receiver can try 110-116. |
| `rx/varahf2300.py:851-857` (`CP_SAMPLES`, `CP_USEFUL`) | the "BW2750 useful 1258-1394" premise is refuted: 1024 at every const level. Narrow `CP_USEFUL` around 1024. |
| `rx/varahf2300.py:39-52` (module docstring) | the BW2750 high levels are now described. Drop "needs the exact per-cell carrier raster". |
| `tx/varahf2300_tx.py:154-172` (`frame_to_onair`) | route through the record's own permutations and the group puncture, so BW2750 keys work. |
| `tx/varahf2300_tx.py:202-226` (`_synth_const`) | the BW2300 rewrite, parametrised by `first_bin` and `span`, so the same code renders 59 carriers on bins 3..61 with 59 gap cells from the record's gap. |
| `tx/varahf2300_tx.py:142-151` (`build_frame`) | as at BW2300: explicit control byte, short-frame `14 09 … 06` fill. |
