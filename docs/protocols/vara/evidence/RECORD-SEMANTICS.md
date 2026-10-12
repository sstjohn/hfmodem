# VARA speed-level records: what a row means, and what a tape must supply

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

Paths: `rx/`, `tx/`, `coding/` are under `packages/hfmodem/hfmodem/kestrel/`; `01`..`06` are `docs/protocols/vara/`. Record index = host `BITRATE (N)` − 1 at every bandwidth (rx/varahf2300.py:195-197), so sigidwiki level L is our record L−1.

## 1. Physical layer as held

48 kHz real audio (rx/varahf2300.py:97).

- **Index law (rec0-3, BW2300/2750).** One lit bin per symbol, value `−1j`, plain `dw50`-point IFFT, **no CP, no window, no twiddle** (01:65-76, 01:88-94). Bin k = k·48000/dw50 Hz. rec3: dw50 512 (93.75 Bd, 93.75 Hz bins), comb 9..24; rec2/rec1: 1024 (46.875 Bd), 17..48; rec0: 2048 (23.44 Bd), 33..96 (01:247-256). Sigidwiki's "23/47 Bd FSK" (L1-3) is this one-hot law; "94 Bd" = 512-sample symbols.
- **Reference scheme, index law.** No pilots. 24 of each over's columns are reference columns lighting a payload-independent bin (`map1≠0`, bin = `alloc + stride·map2`), at `(279k+263)//17` (rec3), `(104k+93)//11` (rec1/2), `(41k+33)//8` (rec0) (rx/tablegen.py:179, 248, 310).
- **Preamble, index law.** 2 silent blocks + 2/4/6/12 training symbols at rec0..3, bin = `alloc[s] + stride·Int(Rnd·16)` from a session-continuous VB6 Rnd stream (seed 4107106 at BW2300 over 0; 5 extra draws at BW2750) (01:151-245; tx/varahf2300_tx.py:80-126).
- **Const law (rec9-16, derived).** `dw50 = 1152` = 1024 IFFT + 128 CP (41.67 Bd, 46.875 Hz spacing), cells from a 64-entry LUT, doc'd per-cell twiddle `exp(j2π·S/128)` on a 128×469 raster plus raised-cosine overlap-add (01:444-548). **kestrel implements none of twiddle, OLA, pilots or preamble** on this path (tx/varahf2300_tx.py:202-226).
- **BW500 base (L4).** Two sub-bands at 1350/1650 Hz, H = 512, WOLA pulse, DBPSK across columns, `grid480` de-rotation, 394 columns, 748 cells (rx/varahf500.py:125-137; 03:188-210). BW500 L1-3 are one-hot index records on bins 27..37 of 1024 / 54..74 of 2048 (rx/varahf500.py:226-228).
- **Burst start, BW2300.** Envelope gate 750-2350 Hz, 0.12·max (rx/varahf2300.py:601-619), then a payload-blind onset × start-column scan scored by reference-column argmax hits, best 6 turbo-decoded, `lead` prepended as silence (rx/varahf2300.py:622-712). **BW500 base:** differential-collapse lock over [−9,+2) columns, c0 by turbo self-consistency (rx/varahf500.py:17-35). No const-law burst has ever been located on a tape (rx/varahf2300.py:52-78).

## 2. Record fields (`rx/varahf2300.py:169-190`)

| field | meaning, units |
|---|---|
| `level` | record index = host BITRATE−1; also the default `il1`/`il2` column. BW2750 rows are keyed 100-103 so they never enter `LEVELS` (:103, :256) |
| `law` | `"index"` one-hot, `"const"` LUT constellation (06:101 `b00`: 2/1) |
| `n_info` | turbo info bits N; frame bits zero-padded to N (rec2: 400+2) |
| `coded` | turbo output bits; `2N+12` → rate-1/2, `3N+12` → 1/3, else punctured (:368-375) |
| `frame_bytes` | bytes coded = payload + 2-byte CRC; `payload_bytes = frame_bytes−2` (:299); ARQ control byte is inside payload |
| `bpc` | bits per column (index) or per cell (const) |
| `pos` | LUT column, `cell = LUT[pos+6·coded]` (const only) |
| `dw50` | samples per symbol including CP |
| `cp` | cyclic-prefix samples |
| `span` | index: comb width in bins; const: carriers per symbol |
| `first_bin` | lowest bin, in the dw50 FFT (index) or the (dw50−cp) FFT (const) |
| `stride` | bins per Gray step (index) |
| `ncols` | index: emission columns incl. 24 reference |
| `lead` | index: blocks before column 1 = 2 lead-in + training symbols |
| `il` | override interleaver column (`il_col`, :188-190); BW2750 base uses 3 |

Index tables live outside the row, in `_BASE_TABLES[level] = (alloc, map1, map2)` (:106-137).

`IndexRecord` (rx/varahf500.py:202-223) has the same meanings under other names: `level` is the host BITRATE, `rec` the interleaver column, `dw`, `lead` = training columns only, `n_src` = N, `payload` bytes, `frame = payload+3` (marker + CRC; BW2300 has no marker), `rate13` iff `coded = 3N+12`; bpc fixed at 3 (`L3_BPC`, :234).

`burst_length` (rx/varahf2300.py:548-555): index = `(lead+ncols)·dw50`; const = `ceil(ceil(coded/bpc)/span)·dw50`, with no preamble, no pilots, no forced cells.

## 3. Coding chain (as implemented; validated at rec0-3 and BW500 L1-4)

Frame → MSB-first bits, zero-pad to N → **XOR whitener** → **turbo** → **channel interleave (gather)** `onair[k] = coded[π[k]]` → value law (tx/varahf2300_tx.py:154-172); RX inverts (rx/varahf2300.py:347-386). 03:14-19 gives another order (turbo first, twiddle, WOLA); the code's order is the one that decodes VARA audio.

- **Whitener:** VB6 Rnd LCG, seed 29318, bit = `s>>23`, one 150000-bit stream shared by all bandwidths, reset per frame (rx/tablegen.py:99-111).
- **Turbo:** two (13,15)-octal RSCs, 8 states, 12-bit dual tail `[s1,p1,s2,p2]×3`. Rate 1/2: parity1 at even i, parity2 at odd i (coding/turbo.py:69-83). Rate 1/3 unpunctured (:193-205). **Rates 2/3, 4/5, 5/6 use an invented evenly-spaced puncture** (:216-250). VARA's matrix is unknown (03:124).
- **Interleavers:** per record, stage-1 (channel) and stage-2 (turbo) permutations from `permutation(n, seed)` (Rnd rejection sampling), stage 2 continuing stage 1's stream. Seeds and lengths for all 17 BW2300 and BW500 records are at rx/tablegen.py:117-130. Built into 17-column arrays (`il[col::17]`) (:152-176). They are byte-exact against VARA's `.dat` tables, but used on air only at rec0-3 and BW500 rec0-3.
- **CRC:** CRC-16/GENIBUS over the frame minus its last 2 bytes, big-endian (rx/varahf2300.py:399-404).
- **Symbol mapping:** `clsparm_gray` closed form, rows bpc 2/3/4 (rx/tablegen.py:385-396). Index law: `bin = ((alloc + stride·gray − first_bin) mod span) + first_bin`. Const law: PSK columns go through Gray, QAM columns are per-axis labelled (rx/varahf2300.py:326-341).
- **LUT:** `constellation_lut` is reconstructed bit-exact to VARA's shipped 64 entries (rx/tablegen.py:454-466). The 64 entries truncate six period-6 columns (BPSK, QPSK, 8PSK, 16QAM corner 1.1, 32-cross 1.5, 64QAM 1.6); `_column` gives each full alphabet. **pos=4 quadrant 2/3 labels (22 of 32) are adopted, not observed** (:429-441).
- **Regenerated tables:** whitener, both interleavers and `alloc_col3`/`map2_col3` come from the Rnd stream. `map1`, reference positions and Gray are closed form. `base_bins_col0/1/2`, `bw500_alloc(0)`, `grid480` and the BW2750 rec0-2 allocations (rx/bw2750_allocations.py) are measured literals.

## 4. What rows 9-16 assume, and what is unknown

rec9-16 (rx/varahf2300.py:246-253): const law, dw50 1152, cp 128, **span 32 on bins 9..40 of 1024 (422-1875 Hz)**, sequential cell fill, no pilots. rec14-16 carry `word[+0x3e]` forced cells, 980/980/77 (06:257-291), which kestrel ignores. `span=32` and `first_bin=9` are placeholders, not measurements (06:88-93).

**Evidence in our own tables:** every record's `coded/bpc` (plus forced cells) divides exactly by sigidwiki's carrier count, and not by 32. rec9-13: 7203 cells = 147×49. rec14/15: 6517 + 980 = 7497 = 153×49. rec4-8: 1086/3, 1416/4, 2082/6, 2760/8, 3430/10 give 362/354/347/345/343 symbols. BW500 rec4-12: 750/2, 1086/3, 1584/11 ×5, 1430/11 ×2. The Rnd gap after each of rec9-15 is exactly **49 draws** (rec4-8: 37/36/36/40/41). `N/bps` from sigidwiki is 3.58 s at every 4/8PSK 42-Bd level and 3.70 s at 16/32QAM, at both BW2300 and BW500, which is the step 980 forced cells add.

**rec4-8 (BW2300 L5-9):** coded/N are known (2172/1080 … 6860/3424, all rate 1/2) 06:114 claims dw50 512, span 16, QPSK; 06:160-170 calls them open. Unknown: carrier count and bins, CP, per-symbol carrier hopping (the 36-41 draw gaps are too short for a per-column table), pilots, training, twiddle or differential reference, fill order, over length, frames per over.

**BW2750 above base:** no row. 01:632-648 holds Tu ≈ 1280 and Ts ≈ 1400-1450; sigidwiki says 42 Bd × 59. 59 × 46.875 Hz = 2766 Hz, close to the measured 2742/2713 Hz occupancy, while 59 × 37.5 = 2212 Hz does not fill it. No BW2750 coded/N/seed is held for records 4-16. Prediction to check: L11 = 147×59 = 8673 cells → coded 17346, N 8667.

**BW500 L5-13:** no record type exists (`IndexRecord` is one-hot, the base path hard-wired DBPSK); seeds and coded/N are held (rx/tablegen.py:117-122).

**A tape must yield, per level:** (a) symbol and CP length (CP autocorrelation at 1024/128 vs 1280); (b) occupied bins, from a long-term spectrum and per-symbol occupancy; (c) pilot or forced-cell positions and values; (d) preamble length and structure; (e) over length in samples, and frames per over (reconcile 06:108-118's "4.32 s" against 147 symbols = 3.53 s); (f) phase reference: twiddle `S`, `derotation_reference` or differential; (g) cell fill order (test π against sym-major and carrier-major fills); (h) puncture pattern at 2/3, 4/5, 5/6; (i) pos=4 labels; (j) the host `BITRATE (N)` stamp for each over. (h) and (i) need known plaintext or a CRC-guided search, so record bench overs with known payloads, both cables separately.

## 5. Adding a record

**TX:** `synth_frame` → `frame_to_onair` (generic over any row) → `_synth_index` / `_synth_const` (tx/varahf2300_tx.py:154-243). Index rows also need `_BASE_TABLES` and `lead`. `KEYABLE_LEVELS` picks up const rows automatically (rx/varahf2300.py:276-277). `vara_ofdm.data_over_tx` passes `level` as the record index at 2300/2750 and maps BW500 records through `_BW500_RECORDS` (vara/vara_ofdm.py:71-121).

**RX:** `decode_burst(level=None)` trial-decodes every `LEVELS` row whose `burst_length` lies within 2048 samples of the audio, and returns the first CRC-clean result (rx/varahf2300.py:558-598). Length separates only families. rec9/10/11 share a length, as do 12/13, and so would 14/15. The CRC decides.

`decode_overs` tries only the index family (rx/varahf2300.py:715-734). A const row is invisible to off-air decoding until an alignment method exists; `cp_repetition` (:887) is the start of one.

**Gotchas:**
- `_CLSPARM` holds bpc 2-4 only (:141), so a BPSK or bpc-1 const row breaks `const_points`.
- `_demod_const` assumes the burst begins at sample 0.
- `candidate_levels` covers BW2300 only.
- New BW2750 rows need keys ≥ 100 and their own interleaver columns via `il`.

## 6. Sigidwiki against our table

### Sigidwiki reference (Bd / carriers / mod / net bps); L1-3 FSK 23/47 Bd

| L | BW500 | BW2300 | BW2750 |
|---|---|---|---|
| 4 | 94/2/BPSK/88 | — | — |
| 5 | 94/2/4PSK/177 | 94/3/4PSK/270 | = 2300 |
| 6 | 94/3/4PSK/270 | 94/4/4PSK/363 | = 2300 |
| 7 | 42/11/4PSK/441 | 94/6/4PSK/549 | = 2300 |
| 8 | 42/11/4PSK/588 | 94/8/4PSK/735 | = 2300 |
| 9 | 42/11/4PSK/705 | 94/10/4PSK/922 | = 2300 |
| 10 | 42/11/8PSK/884 | 42/49/4PSK/2011 | 94/13/4PSK/1203 |
| 11 | 42/11/8PSK/1060 | 42/49/4PSK/2682 | 42/59/4PSK/2423 |
| 12 | 42/11/16QAM/1286 | 42/49/4PSK/3219 | 42/59/4PSK/3230 |
| 13 | 42/11/32QAM/1543 | 42/49/8PSK/4025 | 42/59/4PSK/3877 |
| 14 | — | 42/49/8PSK/4830 | 42/59/8PSK/4848 |
| 15 | — | 42/49/16QAM/5872 | 42/59/8PSK/5817 |
| 16 | — | 42/49/32QAM/7050 | 42/59/16QAM/7074 |
| 17 | — | — | 42/59/32QAM/8489 |

### Against our BW2300 rows

| L | rec | sigidwiki (Bd/carriers/mod/bps) | ours | agrees? |
|---|---|---|---|---|
| 5-9 | 4-8 | 94 / 3,4,6,8,10 / 4PSK / 270,363,549,735,922 | no row; 06:114 says 512-sample (94 Bd), span 16, QPSK; coded r1/2 | Bd and mod agree; **carriers 16 vs 3-10** |
| 10 | 9 | 42/49/4PSK/2011 | 41.67 Bd, **32**, QPSK r1/2, 899 B | **carriers**; our 226-symbol over gives ≈1320 bps |
| 11 | 10 | 42/49/4PSK/2682 | 32, QPSK r2/3 | **carriers** |
| 12 | 11 | 42/49/4PSK/3219 | 32, QPSK r4/5 | **carriers** |
| 13 | 12 | 42/49/8PSK/4025 | 32, 8PSK r2/3 | **carriers** |
| 14 | 13 | 42/49/8PSK/4830 | 32, 8PSK r4/5 | **carriers** |
| 15 | 14 | 42/49/16QAM/5872 | 32, 16QAM r5/6, forced cells ignored | **carriers, forced cells** |
| 16 | 15 | 42/49/32QAM/7050 | 32, 32-cross r4/5, forced ignored | **carriers, forced cells** |
| — | 16 | (BW2300 tops out at L16) | "32QAM-4/5", 213 B, in `LEVELS` | **extra rung.** 429+77 = 506 cells = 46×11, and coded/N 2145/1704 duplicate BW500 rec15/16: not a speed level |

Everywhere, sigidwiki's rates match our coded/N at ≈3.58 s per frame (3.70 s at QAM). The disagreements are carrier count, band placement and over structure, never code rate or modulation. Also: BW2750 L11-17 42 Bd vs our 01 §2750 Tu≈1280/33-34 Bd; BW500 L4 "BPSK" is our DBPSK (naming only).
