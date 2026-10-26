<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# BW2300 levels 10-16 closed from known plaintext

`kpa_bw2300_const.py` prints every table and result below (about 80 s, nothing written). It demodulates
46 frames off the high speed capable bench tapes (40 overs, six of them double L16 overs) with our coding chain
plus the raster described here. All 46 are CRC-clean and byte-exact against the transcript's delivered
bytes: 10 frames at L10, 5 at L11, 6 at L12, 2 at L13, 3 at L14, 2 at L15 and 18 at L16. Raw on-air bit
errors before FEC are 0 on 45 frames. The exception is 67 of 14406 on L11 909025, whose first 57 symbols
carry the sweep's injected noise. On every clean frame, every interior cell equals the chain's predicted
cell to 8e-4 of the reference amplitude. Only the first and last symbol of a burst deviate (up to 0.42),
which is the edge taper.

## Findings

Cells are indexed by FFT bin k (1024-point, occupied bins 8..56, 46.875 Hz) and by OFDM symbol s (1152
samples, counted from the burst's first symbol). "Rnd" is the VB6 LCG already in `tablegen`
(`step`/`draw`).

1. **The 3 bytes.** A frame is the data bytes, then one control byte, then the CRC-16/GENIBUS
   (big-endian) over everything before it. So `frame_bytes` = payload + 3, and the encoder input is
   exactly what `build_frame` would produce if it were handed data + control byte. The control byte
   varies with session state, not with level: 0x89 (L10), 0x81 (L11, L14, L16), 0x95/0x99 (L12), 0x99
   (L14, L15, L16). L4 shows 0x99/0x19. A short frame reads data, `14 xx` (09 or f6), zero fill, then
   `06`, control byte `82`, CRC. L13's 1799-byte frame is confirmed.
   Whitener, turbo interleaver (il2 column = record) and channel interleaver (il1 column = record,
   gather) are all as held. The whitener, turbo (13,15) constituents and 12-bit dual tail need no
   change.

2. **Fill order: bin-major.** Coded cell n goes to bin `8 + n // cells_per_bin`, and within a bin the
   cells run in symbol order, skipping reserved cells. Bits are grouped MSB first, `bpc` per cell.
   - PSK (L10-14): data occupies symbols 5..152, 147 cells per bin. One **repeat cell** per bin sits at
     symbol `7 + Int(Rnd*142)`, where the draw is draw b of the record's own 49-draw gap (after its two
     permutations, closing on the next record's seed). This is what the 49-draw gap is.
   - QAM (L15-16): each 154-symbol frame starts at symbol 5 (+154 per frame in a double over). It holds
     20 pilot rows: even bins at 5+8j, odd bins at 6+8j. One **extra +1 cell** per bin sits on the symbol
     right after pilot row `j = Int(Rnd*19)`, from the same 49-draw gap. That leaves 133 data cells per
     bin, and 49×133 = 6517 = coded/bpc.

3. **Scramble.** `S[k][c] = Int(Rnd*129)`, drawn row-major from the **whitener seed 29318**. There are
   409 columns per FFT-bin row, starting at k = 0, so bin 8 column 0 is draw 3273. The transmitted cell is
   `value · exp(-j·2π·S[k][c]/128)`. The column is c = s − 4 at PSK levels, where column 0 is the reference
   symbol 4, and c = s − 5 − 154f at QAM levels, restarting in every frame of a double over. Lead symbols
   are not scrambled. The table is the same at every level and in every over. The generator reproduces
   the tape's S mod 32 on 100 % of cells, and full S through the decodes. The 2π/128 lattice is this table.
   With the FFT window 112 samples into the raster's 1152-sample symbol (`raster_bw2300.ofdm_cells`,
   backoff 16), the reference cell is exactly +1 on every bin and no residual ramp remains.

4. **Modulation.**
   - PSK is **time-differential per carrier**: `cell[s] = cell[s−1] · LUT[v]`. Symbol 4 (column 0) is
     +1, and a repeat cell steps by +1.
   - QAM is coherent: `cell = LUT[v]`. Pilots and extra cells are +1 (LUT units), which is 1.215× the
     16QAM rms and 1.053× the 32-cross rms, matching the raster's pilot levels.
   - **LUT = `tablegen._column(pos, v)` indexed directly by the raw bit group v**, with no clsparm pass
     in front at any column. Our pos 3 (16QAM) and **pos 4 (32-cross, all 32 labels, including the 22
     "adopted" ones)** are exact on the tape.

5. **Puncture.** Rate 1/2 (L10) is our mux unchanged. At every other rate the info bits go in groups of
   k (k = 2 at L11/L13, 4 at L12/L14/L16, 5 at L15):
   `s[kg], P, s[kg+1] … s[kg+k−1]` with `P = p1[kg]` for even g and `p2[kg]` for odd g. After the groups
   come the 12-bit dual tail `[s1,p1,s2,p2]×3`, then zero padding to `coded`: 0/0/4/0/2/4/3 bits at
   records 9..15. Rate 2/3 is therefore `s p1 s | s p2 s`. Our evenly spaced keep-set with systematic bits
   first (`turbo._punct_keep`) is wrong at every punctured level.

6. **Lead.** Symbols 0-3 (PSK) or 0-4 (QAM) are `LUT[Int(Rnd·2^bpc)]` drawn bin-major, symbol fastest,
   from the session's continuing Rnd stream. They are unscrambled and carry no payload. Consecutive overs
   consume exactly 196 draws (PSK) or 245 draws (QAM), and a double L16 over consumes 490. The same
   session state gives the same lead whatever the payload: L10 1118613 and 1115809 have different bytes
   and identical leads. In the sweep sessions the state sits about 3.335 M draws past the L4
   training seed 4107106. The script recovers the state for 39 of 40 overs (not the noisy 909025).
   Symbol 4 at PSK is a fixed +1 reference, scrambled.

7. **Residual.** None in the data path. Open items: what the control byte and the short-frame
   `14 xx … 06` bytes mean (ARQ layer, not PHY), and the exact start of the lead stream in a session. For
   TX, the ~10-sample edge taper (01's raised-cosine OLA) and whether the 16-sample offset between the
   raster grid and the phase-exact window is the CP start or a transmitted ramp. A rendered burst
   cross-correlated against a tape over would settle it.

## Evidence (in order found)

- Symbols 0-3 of an L10 over read `16 − 6b (mod 32)` on every bin: a pure channel ramp, so no scramble.
  Removing the ramp leaves S mod 32 on data cells. A 2^24 search of `−Int(Rnd·n) mod 32` along bin 8
  hit n = 129 alone. Continuing that stream, bin 9 appears 409 draws later, and so on, and the start lies
  3273 = 1 + 8·409 draws past seed 29318.
- After descrambling, symbol 4 is one constant on all bins, but coherent labels at data cells scored
  chance (0.27) for every natural order. A payload-histogram test finds the counter bytes' XOR signatures
  only in the **symbol-to-symbol difference**, at 340 against a predicted 348. Relative two-bit cell
  signatures across 8 overs then match 138/138 known systematic cells, and the unique hits give
  `bin = n // 147`, `symbol = 5 + n mod 147 (+1 past a gap)`.
- Scanning the control byte (256 values) with full re-encoding matched 7203/7203 cells at L10 (0x89),
  pinned each carrier's repeat cell, and fitted `2 + Int(Rnd*142)` on the gap draws 49/49.
- L11 systematic bits sit at coded positions 0,2,3,5,6,8… (the triplets); a control-byte scan then gives
  the parity as `p1[2m]` for even m (0.997) and `p2[2m]` for odd m. The same group law fits L12 and L14
  exactly, including the tail and the zero pad.
- At QAM, the S table fits with the column origin at symbol 5 (100 % on inner and corner rings). The
  second frame fits only with a restart at symbol 159 (980/980 pilots at one phase). A contingency table
  of predicted labels against measured points is 100 % diagonal for our pos-3 and pos-4 columns.
- The lead is a 2-D window of one pattern: row j+4 equals row j shifted by one bin. That makes it a 1-D
  sequence laid out bin-major, 4 symbols per bin, and a 2^24 search finds `Int(Rnd*4)` through the LUT
  on 196/196 cells.

## What must change

| where | change |
|---|---|
| `rx/tablegen.py` (with `whitener`, :99) | add the scramble table: `Int(Rnd*129)` continuing from seed 29318, 409 columns per FFT-bin row, rows from bin 0. It shares its states with the whitener's first draws. Add the per-record gap cell `7 + Int(Rnd*142)` (PSK) or pilot row `Int(Rnd*19)` (QAM) from the 49-draw gap after each of records 9-15. |
| `rx/tablegen.py:429-441` (`_cross32` docstring) | the 22 inferred pos-4 labels are now observed; drop "not observed". |
| `rx/varahf2300.py:246-252` (records 9-15) | span 49, first_bin 8. Lead 4 (PSK) or 5 (QAM). 153 symbols at PSK; 5 + 154·frames at QAM, where L16 carries 1 or 2 frames per over. `frame_bytes` stays payload + 3, with the third byte the control byte (fix the "payload + 2-byte CRC" comment at :299-301 accordingly). Record 16 (:253) is not a level. |
| `rx/varahf2300.py:326-341` (`const_points`) | use `range(2**bpc)` at every pos. The PSK clsparm pass is a second Gray and is wrong on air. |
| `rx/varahf2300.py:523` (`_demod_const`), `:548` (`burst_length`) | rewrite to the raster above: descramble, per-carrier differential at PSK, pilot-referenced at QAM, bin-major gather, frame length from the lead + frame rows. |
| `coding/turbo.py:226-260` (`_punct_keep`, `encode_punctured`, `decode_punctured`) | replace with the group law: k from N/(coded−N−12), parity alternating p1/p2 by group, tail after the groups, zero pad at the end. |
| `tx/varahf2300_tx.py:164-170` | route rates other than 1/2 and 1/3 through the group puncture. |
| `tx/varahf2300_tx.py:202-226` (`_synth_const`) | emit the lead (session Rnd, `LUT[Int(Rnd·2^bpc)]`, bin-major), then the reference or pilots, and the bin-major data (differential at PSK, plus repeat/extra cells), all times `exp(−j2πS/128)` on bins 8..56. Add the edge taper. |
| `tx/varahf2300_tx.py:142-151` (`build_frame`) | take the control byte explicitly, and the `14 xx … 06` short-frame fill when the ARQ layer supplies it. |
