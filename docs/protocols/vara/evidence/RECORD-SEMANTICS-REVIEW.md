# Review of RECORD-SEMANTICS.md against the code

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

Numbers recomputed from `RECORDS`, `tablegen.BW2300/BW500` and the LCG (`permutation`, then `step` to the next seed); claims read against the cited code and 01/03/06.

| § | claim | verdict |
|---|---|---|
| head | record = BITRATE−1 at every bandwidth | CONFIRMED (2300 :195-197, 500 `IndexRecord`, 2750 key 103 = BITRATE 4) |
| 1 | index law, reference positions (tablegen :179/:248/:310): one-hot −1j, no CP/window/twiddle; rec3 512 bins 9..24, rec1/2 1024 17..48, rec0 2048 33..96 | CONFIRMED |
| 1 | preamble 2/4/6/12, seed 4107106, 5 extra draws at 2750 | CONFIRMED |
| 1 | preamble draw is `Int(Rnd·16)` | WRONG at BW2750 rec100/101: `Int(Rnd·8)` (tx :96-97) |
| 1 | const law 1152 = 1024+128; twiddle/OLA documented, not implemented | CONFIRMED |
| 1 | BW500 base 1350/1650, H 512, 394 cols, 748 cells; L1-3 bins 27..37 / 54..74 | CONFIRMED |
| 1 | burst start: 750-2350 Hz gate at 0.12·max, 6 turbo tries, lead as silence | CONFIRMED |
| 2 | field meanings, `burst_length` formulas, `IndexRecord` mapping, `L3_BPC` 3 | CONFIRMED |
| 3 | order frame→pad→whiten→turbo→gather-interleave | CONFIRMED (tx `frame_to_onair`); 03:14-19 differs as stated |
| 3 | whitener seed 29318, `s>>23`, 150000 bits, reset per frame | CONFIRMED |
| 3 | turbo (13,15), 8 states, tail [s1,p1,s2,p2]×3, even/odd parity, invented puncture; pos=4 22/32 labels adopted | CONFIRMED |
| 3 | interleavers byte-exact to VARA `.dat` | CONFIRMED by tablegen docstring and its gate test; not rerun |
| 4 | rec9-13 7203 = 147×49; rec14/15 6517+980 = 7497 = 153×49; none ÷32; 147 sym = 3.53 s | CONFIRMED arithmetic |
| 4 | rec4-8 362/354/347/345/343 symbols | CONFIRMED |
| 4 | BW500 rec4-12 750/2, 1086/3, 1584/11×5, 1430/11×2 | CONFIRMED |
| 4 | Rnd gap 49 after each of rec9-15; rec4-8 37/36/36/40/41 | CONFIRMED |
| 4 | N/bps 3.58 s at 42-Bd PSK, 3.70 s at QAM, both bandwidths | CONFIRMED (3.575-3.580; 3.696-3.699) |
| 4 | the 3.58→3.70 step "is the step 980 forced cells add" | WRONG (see 1) |
| 4 | 59×46.875 = 2766 Hz, 59×37.5 = 2212 Hz | CONFIRMED arithmetic; inference conflicts with 01 (see 3) |
| 4 | L11 BW2750 prediction 8673 / 17346 / 8667 | CONFIRMED arithmetic; the prediction itself is UNVERIFIABLE without a tape |
| 5 | `KEYABLE_LEVELS` picks up const rows; `decode_burst` ±2048 first CRC-clean; `decode_overs` index-only; `_CLSPARM` bpc 2-4; `_demod_const` at sample 0; `candidate_levels` BW2300 only | CONFIRMED |
| 5 | "rec9/10/11 share a length, as do 12/13, and so would 14/15" | WRONG (see 2) |
| 5 | `data_over_tx` passes `level` as the record index at 2300/2750 | IMPRECISE (see 4) |
| 6 | rec9 226-symbol over ≈1320 bps | CONFIRMED (897 B·8 / 5.424 s = 1323) |
| 6 | rec16: 429+77 = 506 = 46×11; coded/N = BW500 rec15/16 | CONFIRMED |
| 6 | "Everywhere … ≈3.58 s per frame (3.70 at QAM)" | WRONG (see 5) |
| 6 | sigidwiki table values themselves | UNVERIFIABLE here (no source in repo cited) |

## Corrections that matter

1. **The forced-cell step does not produce 3.70 s.** At BW2300, 7497/7203 = 153/147 = 1.041. That would give 3.58 × 1.041 = 3.72 s; the sigidwiki figure is 3.70 s, a ratio of 1.033. Six extra 1152-sample symbols add 0.144 s against an observed step of 0.119 s. BW500 shows the same 3.70 s at L12/13 while its data cells *fall* from 144 to 130 symbols of 11, and no forced-cell count is held there. The claim reads as a conclusion, and the numbers do not support it. Two more points: 6517 is already 133×49, and 980 is 20×49. Divisibility by 49 therefore holds with or without the forced cells and cannot confirm them.

2. **Burst lengths in §5.** rec9-13 all have 7203 cells and all render 226 symbols (260352 samples). rec14 and rec15 already share 204 symbols (235008). By length alone, `decode_burst` sees three const groups: {9-13}, {14, 15} and {16}.

3. **The BW2750 carrier-spacing inference.** 46.875 Hz spacing implies Tu = 1024. 01 §2750 (:632-648) explicitly favours Tu ≈ 1280 over the Tu = 1024 / CP ≈ 256 reading and says not to assume 1024. 42 Bd on a 1024 Tu also conflicts with the measured Ts of 1400-1450. The 2766 Hz match is suggestive, but the brief should frame it as a conflict with 01 rather than as support. Occupancy also includes skirts and anything other than data carriers.

4. **`data_over_tx` at 2750.** It passes `level` through unchanged and ignores `bw` unless it is "500". At 2750 the caller must therefore pass a 100-103 key. The default `level=3` with `bw="2750"` renders BW2300's 16-bin comb.

5. **"Everywhere" in §6.** The 94-Bd levels do not fit 3.58 s. BW2300 L5-9 run 4.00 / 3.88 / 3.78 / 3.75 / 3.71 s, and BW500 L4-6 run 4.18 / 4.20 / 4.00 s. The rate/modulation agreement holds at the 42-Bd levels only.

6. **The Rnd-gap reading.** The 49-draw gap matches BW2300's 49 carriers. BW500's gaps for rec3-11 are 37/38/36/46/44/44/45/44/46, not 11, so "gap = carrier count" is not a law. Treat it as a coincidence to test, not as evidence.

7. **A doc inconsistency the brief inherits indirectly.** 06:250-255 says the top level is "validated against a licensed peer". 06:80-92 and the `varahf2300` docstring say no const-law burst has ever been demodulated or accepted. The brief follows the latter, which is correct, but 06 contradicts itself and should be fixed.
