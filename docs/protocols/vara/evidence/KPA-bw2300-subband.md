# BW2300 levels 5-9 (records 4-8): the sub-band waveform, closed from known plaintext

`kpa_bw2300_subband.py` prints every table below in about 90 s (`--rows` adds the measured grid rows).
Tapes are read for measurement only. The level is the host `BITRATE (N)`, and the record is L-1.

## Result

**52 of the 54 data overs at L5-9 decode byte-exact through our coding chain on all bands.** Every one
delivers the transcript's payload, and every clean over has zero channel bit errors before FEC.

| L | overs | byte-exact | clean overs: raw bit errors | noisy overs: raw bit errors | failures |
|---|---|---|---|---|---|
| 5 | 15 | 15 | 0/2172 | 0-138/2172 | - |
| 6 | 15 | 14 | 0/2832 (x2) | 0-226/2832 | 051938Z 3056366 (lock 0.10, floor -10 dB) |
| 7 | 5 | 4 | none clean | 1-150/4164 | 051337Z 705068 (lock 0.12) |
| 8 | 9 | 9 | 0/5520 (x5) | 11-37/5520 | - |
| 9 | 10 | 10 | 0/6860 | 0-31/6860 | - |

Both failures are the two noisiest overs on the tapes. The L7 over at 3717213 is a short final frame and
decodes byte-exact despite its noisy last 0.5 s.

## The waveform

```
frame  = payload | control byte | CRC-16/GENIBUS(payload | control)      floor(N/8) bytes, zero pad to N
bits   = whiten -> turbo (13,15) rate 1/2 -> il1 gather                   record L-1, unchanged from records 0-3
cells  = QPSK increment per on-air bit pair (MSB first): Gray [0,1,3,2] -> 0/90/180/270 deg
fill   = band-major: band 0's data columns in time order, then band 1, ...
band b = t 0..7 lead (training), t 8 reference (cell 1), t 9..nsym-1 data + pilot columns
         cell(t) = cell(t-1) * i^inc; a pilot column repeats cell(t-1)
tx     = cell(t) * conj(grid[b][t-8])
audio  = Re sum_b,t tx * p(n - n0 - stagger_b - 512 t) * exp(j 2 pi f_b n / 48000)
```

| L | rec | bands | f_b (Hz) | stagger (samples) | nsym | data cols/band | pilots/band | pilot t | pulse |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 4 | 3 | 900, 1500, 2100 | 0, 176, 336 | 383 | 362 | 12 | 13 + Int(Rnd*368) | BW500 513-tap |
| 6 | 5 | 4 | 750 + 500b | 0, 256, 0, 256 | 372 | 354 | 9 | 13 + Int(Rnd*357) | BW500 513-tap |
| 7 | 6 | 6 | 700 + 320b | 0/256 alternating | 362 | 347 | 6 | 13 + Int(Rnd*347) | 1025-tap (below) |
| 8 | 7 | 8 | **702 + 228b** | 0/256 alternating | 359 | 345 | 5 | 13 + Int(Rnd*344) | 1025-tap |
| 9 | 8 | 10 | 555 + 210b | 0/256 alternating | 356 | 343 | 4 | 13 + Int(Rnd*341) | 1025-tap |

`nsym = 9 + data + pilots`. The pilot span is always `ncols - 6`, with `ncols = nsym - 9`. Pilot column
`co = t - 8` therefore falls in `5 .. ncols-2`. Data columns x bands x 2 = coded exactly at every level.

**Coding.** All five records are `coded = 2N + 12`, so `turbo.encode` works unchanged at rate 1/2 and no
puncture pattern is involved. The il1/il2 columns and seeds tablegen already holds (`BW2300`, rx/tablegen.py:124)
are correct for records 4-8. That is now MEASURED, because the plaintext decodes through them.

**The 3 non-payload bytes** are a control byte and a CRC-16/GENIBUS over the payload and the control byte,
in that order. N is not a byte multiple at L6/7/8 (1410, 2076, 2754), so 2/4/2 zero bits pad the frame.
Control bytes seen: 0x19 on each session's first data over, then 0x99, 0x95, 0x91 and 0x8d. The same
0x19/0x99 appear on our rec3 decodes. The short final frame (L7 3717213, 184 of 256 bytes) carries 0x82,
with `14 09` after the payload, zeros, and `06` in the last payload byte. What these values mean is UNKNOWN.

**Pilots: a generator, not a table.** For each band in order, VARA draws `t = 13 + Int(Rnd*(ncols-6))` with
rejection until NPIL values are distinct (tablegen's binary32 `draw`). The stream starts where record L-1's
stage-2 permutation ends, band 0's draws first. The draw counts are 37/36/36/40/41 (one
duplicate each at L5 and L9), and every count **closes exactly on record L's stage-1 seed**. That accounts
for the Rnd gaps RECORD-SEMANTICS §4 left open. Blind check: at every generated pilot position on every
level, the de-rotated increment is 0. The same law regenerates **BW500 record 3's 36 `_PILOTS`** (tablegen:586)
bit for bit: co = 5 + Int(Rnd*386), 18 per sub-band, 37 draws, closing on BW500 record 4's seed 9007905.

**The scramble is grid480, one row per band.** Band b reads row b at index `t - 8`. Rows 0 and 1 are
tablegen's grid480 at `[0..392]` and `[471..863]`, the same index law as BW500's `grid480_index`
(rx/varahf500.py:192) with c0 = 8. At L5 all 375 cells of rows 0/1 match the tape to 0.09 unit
(1 unit = 2π/128). The table is shared across levels: row 2, read at L5, de-rotates L6's band 2 with zero
off-lattice increments, and rows 0-7 read at L5/L6/L9 put every cell of five clean L8 overs outside
training within 0.21 unit of the lattice. Rows 2-9 are MEASURED (`--rows`): row 2 at L5 (t 8..382),
row 3 at L6 (to 371), rows 4-9 at L9 (to 355), row tails 4-7 at L8 (to 358), and row tails 4-5 at L7
(359..361, noisy: UNRELIABLE). Each new row carries a constant non-integer phase against a straight-line
carrier convention (−0.50, +0.32, … units). That constant is invisible to any receiver. None of the tried
Rnd quantisations (Int(Rnd*128), rounded, lags 1-63) generates grid480 row 0. Whether rows 2-9 sit at
stride 471 in VARA's file is UNKNOWN.

**Differential, not coherent.** The information is in the increment `tx(t)·conj(tx(t-1))` after de-rotation,
exactly like BW500's DBPSK across columns, here QPSK. The "fixed per-(band, symbol) phase scramble" in
RASTER-bw2300.md is grid480. The "QPSK" is increments.

**Lead (t 0..7): training, generator UNKNOWN.** The lead is QPSK on the lattice and is not payload data.
Overs at the same position in a session match across sessions (all five first-over L5 leads, and
counter4096/65536 L8), while different positions differ. So it is session-deterministic, like L4's Rnd
preamble. It is not `Int(Rnd*4)` from the 4107106 preamble stream at any offset up to 40000, in either
band-major or time-major order. A receiver never reads it.

**Pulse.** L5/L6: BW500's measured 513-tap pulse (tx/bw500_pulse.py) at the table's centres and
staggers, with no free parameter. A clean over's full-band model residual is **−64.6 dB**. L7-9 need a
longer pulse, fitted here by alternating least squares, with the cells held to plaintext on the bands whose
rows are known. Both sides show a half-amplitude at ±211, a first zero at ±385 and a minimum of −0.07 at
±481. At L9 it is confined to ±512 (energy beyond 6e-5) with a residual of **−77.0 dB**. The L8 fit
reaches −54.9 dB and L7 −48.4 dB (clean head only), and both leave 2e-3 of their energy beyond ±512.
Whether that is a per-level difference or fit convergence is UNKNOWN. The pulse is a measured set of taps,
not a closed form.

**Corrections to RASTER-bw2300.md.** The L8 spacing is **228 Hz, centres 702..2298**, not 1600/7. The
1600/7 model drifts ±2.0 Hz at the edge bands, linear in band, and the raster's own centroids (701.9, 2298.3)
had it. The L7 stagger is 256, not 240 (model residual 1.5e-5 at 256 against 8.8e-5 at 252 and 1.2e-3
at 240). The L5 stagger is exactly 176/336. The "surplus" symbols per band are 9 + NPIL: 8 lead, 1
reference and the pilots.

## Evidence trail (how each item fell)

1. Cross-instance ratios, with the scramble cancelled, set against the plaintext's QPSK. A DP alignment
   scored 48 % of cells band-major/differential at the first try (chance 25 %). Local offsets then walked
   9 → 21 across the over: 9 lead symbols plus 12 held columns per band.
2. Raw cell phases sit on the 2π/128 lattice (R128 0.93). The first-differences of band 1 match grid480
   `463 + t` on 374/374 cells, and band 0 matches `t − 8` on 374/374. After de-rotation the increments are
   {0, 90, 180, 270}° to 0.003 rad.
3. With bands 0/1 de-rotated and band 2 erased, L5 turbo-decodes CRC-clean to `payload | 0x99 | 1f3e`.
4. The DP pilot positions were `13 + Int(Rnd*368)` for 27 of 37 raw draws on a (span, base) brute force.
   The rest were DP ambiguity next to zero increments. The generator then held blind at all five levels.
5. The ladder (bootstrap L5 → L9, then pulses) reads rows 2-9. The final pass decodes everything on the
   full model.

## What must change in the product (not touched here)

- `rx/tablegen.py:586-610`: replace the `_PILOTS` literal with the pilot generator (`co = 5 + Int(Rnd*(ncols-6))`,
  per sub-band, from the record's gap). One function serves BW500 record 3 and BW2300 records 4-8.
- `rx/tablegen.py:612, 758`: grid480 gains rows 2-9 as measured literals (from `--rows`), keyed by
  (row, column) until the file stride is known. Don't count the L7 tails 359..361.
- `rx/varahf2300.py:169-190`: `Record` has no sub-band law. Add a `SubbandRecord` (below).
  `:214-217` ("Records 4-8 are absent") is obsolete. `:233` `RECORDS` gains records 4-8. `:548`
  `burst_length` for these is `nsym*512` plus the pulse tails (measured 196231 = 383·512 + 135 at L5,
  184103 = 359·512 + 295 at L8). `:565` `_decode_at` needs a sub-band demod:
  a band-0 lattice lock on grid row 0 (payload-blind, `lock()` here), then least squares on the band model
  (`design`/`cells_ls`), then de-rotate, increments, LLRs, band-major. `:299` `payload_bytes` stays
  frame − 2, with the control byte as the last "payload" byte, as at rec3.
- `tx/varahf2300_tx.py:154`: `frame_to_onair` already works. `:229` `synth_frame` needs a `_synth_subband`:
  cumulative increments with pilots held, × conj(grid), the table's pulse, centres and staggers. The lead
  can be a stand-in, as BW500 record 2 does (03 §3.6.6).
- `tx/bw500_pulse.py`: reused unchanged at L5/L6. The L7-9 pulse needs its own measured-taps module, the
  L9 fit (`PULSES[9]`).
- `docs/protocols/vara/06-speed-gearshift.md:114` ("4-8 | LUT constellation | 512 | 16 | QPSK (unit)") is
  wrong on span and law.
- Prediction, untested: BW500 L5/L6 (94 Bd, 2/3 carriers) are this family on BW500 record 4/5 seeds.

## Proposed rows

M = MEASURED here, D = DERIVED, U = UNKNOWN. Every row has law `subband` (M), bpc 2 as differential QPSK
(M), dw50 512 (M), cp 0 (M), lead 8 plus reference at t = 8 (M; lead content U), grid row b = band b
(M), and pilot law `13 + Int(Rnd*(ncols-6))` (M).

| rec (L) | n_info | coded | frame_bytes | bands | f0 / step Hz | stagger | nsym | npil | pulse | il |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 (5) | 1080 M | 2172 M | 135 M | 3 M | 900 / 600 M | 0,176,336 M | 383 M | 12 M | P500 M | 4 M |
| 5 (6) | 1410 M | 2832 M | 176 M | 4 M | 750 / 500 M | 0,256 alt M | 372 M | 9 M | P500 M | 5 M |
| 6 (7) | 2076 M | 4164 M | 259 M | 6 M | 700 / 320 M | 0,256 alt M | 362 M | 6 M | P79 M (fit −48 dB, tail U) | 6 M |
| 7 (8) | 2754 M | 5520 M | 344 M | 8 M | 702 / 228 M | 0,256 alt M | 359 M | 5 M | P79 M (fit −55 dB, tail U) | 7 M |
| 8 (9) | 3424 M | 6860 M | 428 M | 10 M | 555 / 210 M | 0,256 alt M | 356 M | 4 M | P79 M (−77 dB) | 8 M |

Still UNKNOWN: the lead generator, the control byte's semantics and the short-frame fill, the grid file
stride for rows ≥ 2 (and whether the grid is generated at all), the L7/L8 pulse tails beyond ±512, and the
L7 row tails at t 359..361.
