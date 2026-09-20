# 03 — Coding layer (scrambler / interleaver / FEC / CRC)

Two coding regimes exist and must not be conflated:

- **MFSK handshake bursts** (CR / connect-response / connected-ack) carry **no
  FEC, interleaving or whitening** — their tones are a public-LCG (VB6 `Rnd`) PRNG
  stream keyed to the destination callsign, specified in
  [`04 §4.2`](04-frame-block-formats.md). They do not use the stages in this file.
  The one shared piece is the CRC (§3.4), which derives the PRNG seed.
- **OFDM DATA waveform** (BW2300 link-setup and bulk data, host
  `BITRATE (N) … bps`): §3.0–§3.5 apply. §3.6 specifies the BW500 DATA path, at
  its base level (record 3) and the gear below it (record 2).

## 3.0 Pipeline order

| Stage order | Value |
|-------------|-------|
| BW2300 OFDM DATA, TX (bytes → cells) | `frame bytes → CRC-16/GENIBUS append → serialize to bits → turbo FEC encode → whiten (XOR PN) → block interleave → map to 64-entry constellation LUT → per-cell twiddle ⊗exp(j2π·S/128) on the 469·bin+col_rel raster → OFDM cell grid → WOLA synth`. RX is the exact inverse. The relative order of whitening and interleaving — and whether turbo encoding precedes both — is not confirmed here. |
| MFSK handshake bursts | no FEC, interleave or whitener stage — PRNG-tone generator (see [`04 §4.2`](04-frame-block-formats.md)) |

## 3.1 Scrambler / randomizer (OFDM DATA)

| Attribute                         | Value |
|-----------------------------------|-------|
| Present?                          | Yes (OFDM DATA waveform) |
| Type (additive/multiplicative)    | Additive bit whitener (XOR with a fixed PN sequence) |
| Polynomial / sequence             | Fixed **BW2300-specific PN table** (bandwidth-dependent; the 500 Hz PN differs). Bits: `bw2300/whitener_pn.u8` (0/1 per byte; valid ≈first 197958; see §3.5.4 for where the tables live). Reset to table start per coded block. |
| Seed / init state                 | Reset to table start per coded block |
| Reset boundary (per frame/block)  | Per coded block |

## 3.2 Interleaver (OFDM DATA)

| Attribute                          | Value |
|------------------------------------|-------|
| Present?                           | Yes (OFDM DATA waveform) |
| Type (block/convolutional)         | Block, symbol-major (fills OFDM symbol carrier-slots in a fixed permutation) |
| Geometry / depth                   | Maps coded bits onto the 65 active carriers × ~100 symbols of a BW2300 burst; **BW2300-specific map**, `bw2300/interleave_stage1.i32` (32585 rows × 17 seg-columns, row-major `[i·17+seg]`; column `[seg::17][:coded_len]` = exact bijection of `0..coded_len−1`) and the sibling `bw2300/interleave_stage2.i32` (26056 rows × 17; column `[seg::17][:N]` = exact bijection of `0..N−1`) — **full columns for all 17 records**. Cell-usage: `bw2300/usedmap.u8` (782 ones). Table locations: §3.5.4. |
| Read/write order                   | Write coded bits sequentially, read out by the map into carrier-slots (frequency+time spread) |
| Per-speed-level / -bandwidth variation | **Yes** — the map is bandwidth-dependent (BW500/BW2300/BW2750 each have their own) and level-dependent |

## 3.3 Forward error correction (FEC) (OFDM DATA)

| Attribute                                    | Value |
|----------------------------------------------|-------|
| Family (conv / LDPC / Turbo / RS / none)     | **Turbo** (parallel-concatenated convolutional) |
| Code rate(s) per speed level                 | ≈**1/3** base; higher levels reach higher rate via puncturing |
| Constituent generators (octal)               | **(13, 15)** octal constituent RSC encoders |
| Puncturing pattern                           | Per-level puncture matrix (BW/level-dependent); base rate ~1/3 unpunctured |
| Termination                                  | Not specified here for BW2300 (see §3.5.4); BW500 in §3.6.4 |
| Soft-decision handling                       | log-MAP (soft) decode |

## 3.4 CRC / error detection

| Attribute                         | Value |
|-----------------------------------|-------|
| Present? scope (frame/block)      | Yes — per frame |
| Width                             | **16 bits** |
| Polynomial                        | **CRC-16/GENIBUS**, poly **0x1021** |
| Init / final-XOR / reflection     | init **0xFFFF**, xorout **0xFFFF**, refin=false, refout=false (i.e. GENIBUS = CCITT-FALSE with output inversion) |
| Coverage (which bytes)            | Frame body — for the MFSK handshakes, the destination-callsign ASCII, whose CRC seeds the tone PRNG; for OFDM, frame body + appended CRC |

## 3.5 BW2300 OFDM coding

The BW2300 OFDM waveform carries both the caller-ID link-setup burst and bulk
data. §3.5.3 gives the chain and its framing, §3.5.4 the constant tables, §3.5.5
the caller-ID link-setup frame.

### 3.5.3 BW2300 OFDM link-setup / DATA chain

The BW2300 OFDM waveform (host `BITRATE (N) … bps`; VARA's general bulk-data
waveform) uses the §3.0 pipeline with the **BW2300-specific coding tables**
(interleave map, whitening PN, used-carrier map, twiddle-phase sequence `S`) —
the 500 Hz assets do **not** apply. The forward (MOD) and inverse (DEMOD) chains,
in functional terms (the raster index law is in [`01 §2300`](01-physical-layer.md)):

```
DEMOD:  grid cell ⊗ reference[469·bin + col_rel]              # de-rotate (index law, spec/01)
        → constellation demap (nearest of the 64-entry non-uniform-magnitude LUT)
        → de-interleave (BW2300 block interleave map)
        → de-whiten    (XOR BW2300 PN)
        → turbo (13,15) log-MAP decode
        → CRC-16/GENIBUS check  → frame → caller callsign + session params
MOD  =  the exact inverse (forward) chain.
```

The modulation is index/position type — the coded value selects the carrier bin —
so RX is an argmax over the received energy grid.

- **Framing.** A frame is **92 bytes = 736 bits**:
  `[89 payload bytes][1 trailer/block byte][2-byte CRC, big-endian]`, with **no
  header**. CRC-16/GENIBUS over `body[0:90]` equals `frame[90:92]` big-endian. The
  link-setup burst uses the same 92-byte framing law as a DATA over.
- **Per-over payload slicing.** 89 consecutive payload bytes per over
  (`over_i = payload[89·i : 89·i+89]`); a 512-byte payload therefore occupies 6 overs.
- **Whitener.** XOR with the BW2300 whitener PN table (0/1 per byte, MSB-first),
  reset per block.
- **Turbo.** Parallel (13,15)-octal RSC, **rate 1/2, N = 736 → 1484 coded bits**
  (= 2·736 + 12 dual-tail) at records 3 and 2, and **rate 1/3, `3N + 12`** at
  records 1 and 0 — 612 from 200 and 300 from 96. The internal interleaver is
  `interleave_stage2.i32` column 3,
  `perm[i] = IL2[3 + 17·i][:736]` — a native `0..735` permutation, used directly with
  **no pruning**.
- **Channel interleaver.** `interleave_stage1.i32`
  column 3 (a 1484-entry permutation), applied as a GATHER: `onair[k] = coded[π[k]]`.
- **Twiddle.** `exp(j2π·m/128)` — divisor 128, per
  [`01 §2300`](01-physical-layer.md).

### 3.5.4 BW2300 coding data tables

The BW2300 constant tables needed to build a VARA-accurate BW2300 link-setup /
DATA transmitter are produced in code by the companion implementation's
generator, `hfmodem/kestrel/rx/tablegen.py`, under a byte-exactness gate; they
are not shipped as data files, and the file names in the table below identify
the tables the generator produces. The generator records how each is known. The
forward (MOD) chain is fully specified by these tables plus the
[`01 §2300`](01-physical-layer.md) raster.

| Element | Value / pointer |
|---------|-----------------|
| Constellation LUT (bandwidth-independent) | 64-entry complex LUT `constellation_lut_64.c64`, value law `cell = LUT[pos + 6·coded]` — six constellation columns interleaved with period 6 (BPSK / QPSK / 8-PSK on the unit circle; 16-QAM, **32-QAM cross** and 64-QAM with outermost moduli 1.1 / 1.5 / 1.6). The table is a **truncated prefix of the raster** — column `pos` present only up to `coded = ⌊(63−pos)/6⌋` — so its zero entries are padding, not nulls, and entry 13 is QPSK `coded=2` (`−j`), not a dedicated pilot. (An earlier revision read the table as complete: "14 distinct moduli" and per-column APSK rings; both were truncation artifacts.) Demap = nearest point of the record's column. Full geometry and label law: [`06 §6.1c`](06-speed-gearshift.md) and `hfmodem/kestrel/rx/tablegen.py`. Byte-identical BW500↔BW2300. |
| Whitener PN | `bw2300/whitener_pn.u8` |
| Block-interleave map (stage 1 / stage 2) | `bw2300/interleave_stage1.i32`, `bw2300/interleave_stage2.i32`; cell-usage `bw2300/usedmap.u8` |
| Twiddle-phase sequence `S` | `bw2300/twiddle_phase_S.u8` (values 0..254; valid ≈first 73744). `twiddle[k] = exp(j·2π·S[k]/128)` on the `469·bin + col_rel` raster ([`01 §2300`](01-physical-layer.md)). The divisor is **128** even though `S` spans 0..254. |
| De-rotation reference (128×469) | `bw2300/derotation_reference.c64`. Unit-modulus (`|z|=1`, phases quantized `2π/128`); logical size 128×469 = 60032 (first ≈59904 clean; use table values directly — it is **not** index-aligned to `exp(±j2π·S/128)`). RX de-rotate: `cell ⊗ reference[469·bin + col_rel]`. Byte-identical BW500↔BW2300. |
| Base-record bin-placement tables | Record 3's is `9 + Int(Rnd*16)`, 395 draws from the gap after its own two permutations, closing on record 4's stage-1 seed ([`01 §2300`](01-physical-layer.md)). **Every record's gap holds exactly its own `ncols + 24` draws and closes on the next record's stage-1 seed** — 148 / 252 / 252 / 419 at rec0..rec3 — which is what fixes 124 / 228 / 228 / 395 emission columns and 24 reference columns at each. The stream is therefore identified at all four. **The table is generated only at record 3.** Below the base level `first_bin + Int(Rnd*span)` off the record's own gap lands within **−1..+3** of the true bin and never reliably on it — measured at all three, `{−1: 11, 0: 21, +1: 36, +2: 25, +3: 7}` over rec0's 100 data columns, `{3, 56, 83, 50, 12}` over rec1's 204 and `{9, 57, 77, 48, 13}` over rec2's — so records 0, 1 and 2 are demodulated from a measured table or not at all. All three are now measured (`tablegen.base_bins_col0/1/2`), read off the 2026-09-04 gear-down tapes. Records 0 and 1 need no known plaintext to be read: with `bpc` 3 at stride 2 an allocation reaches a quarter of a 64-bin comb and half of a 32-bin one, so overs of differing payloads intersect down to one bin a column. Record 2 spends 16 values at stride 2 on 32 bins, which is a whole parity of its comb, and had to be read from its payload. Reference columns: 24 at every record, at `(41k+33)//8` on rec0's 124 columns and `(104k+93)//11` on rec1's and rec2's 228 — records 1 and 2 share theirs position for position. |
| Turbo (BW2300) constituents / termination | Same family as BW500 (§3.6): parallel-concatenated **(13,15)-octal RSC**, log-MAP; base rate ≈1/3, per-level puncturing. The BW2300 interleave and repetition tables above realize the level's rate. The exact BW2300 **puncture matrix and termination are not specified here.** In their place the reference implementation reuses BW500's (13,15) constituent and 12-bit dual-tail termination (§3.6); that is an implementation stopgap standing in for an unspecified value, not a property of VARA stated here. |

### 3.5.5 Caller-ID link-setup coding

> **Reading the burst.** It is easy to take this burst for a narrowband 16-ary
> MFSK waveform with 1024-sample symbols. It is the **ordinary BW2300 rec3
> wideband OFDM burst**: the apparent "16 tones, one tone per symbol" is rec3's
> one-hot index modulation (one lit bin in a 16-bin span, `bpc = 4`), the apparent
> "371 symbols" are the 371 data emission columns, and symbols are 512 samples.
> The coding below is independent of that geometry. See
> [`04 §4.2A`](04-frame-block-formats.md).

The caller-identity link-setup burst is coded with the **same (13,15)-octal turbo
family** as the BW500 DATA path (§3.6.4), configured **rate 1/2, N = 736**:

```
TX:  caller frame (90 body bytes + 2-byte CRC-16/GENIBUS BE, big-endian, over body[0:90])
     → serialize to 736 bits (MSB-first) → XOR PN whitener (BW2300 whitener PN)
     → turbo (13,15) rate-1/2 encode (+12-bit dual tail)      → 1484 coded bits (encoder-native)
     → CHANNEL interleave (GATHER): onair[k] = coded[ π[k] ], π = interleave_stage1 col-3 [:1484]
     → group 4 on-air bits MSB-first → Gray σ symbol map (LUT [0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10])
     → 371 rec3 data emission columns (Gray value → one-hot bin, per [`01 §2300`](01-physical-layer.md))
RX = the inverse: σ⁻¹ demap → channel de-interleave (coded[π[k]]=onair[k]) → turbo log-MAP
     decode → de-whiten → CRC-16/GENIBUS → frame.
```

| Item | Value |
|------|-------|
| FEC family | parallel-concatenated turbo, two identical **(13,15)-octal RSC** (8-state) — identical to §3.6.4 |
| Rate / budget | **rate 1/2**, `N = 736` info → `2N+12 = 1484` coded (= 371 × 4-bit symbols) |
| Whitener | XOR PN over the 736 info bits, MSB-first |
| Symbol map | 4 coded bits → σ Gray LUT → 16-ary tone (see [`04 §4.2A`](04-frame-block-formats.md)) |
| **Channel interleaver** π | **`interleave_stage1` column 3, first 1484 entries** — a clean bijection of `0..1483`, applied GATHER: `onair[k] = coded[π[k]]`. The link-setup path reads the same table and column as a DATA over. |
| **Internal turbo interleaver** (N=736) | **native `interleave_stage2` column 3**, `perm[i] = IL2[3+17·i][:736]` — a clean `0..735` permutation with **no pruning**. It is one waveform and one interleaver, shared with the DATA path (§3.5.3), not two. |
| CRC / frame | CRC-16/GENIBUS(body[0:90]) == frame[90:92] BE; caller callsign 6-bit packed at frame start (frame layout: [`04 §4.4.7`](04-frame-block-formats.md)) |

> **N-parametric rule (both interleavers).** Both permutations come from VARA's two
> 17-column int32 tables, accessed as `perm[i] = TABLE[i·17 + col]`:
> **channel/carrier-placement** from `interleave_stage1.i32`
> (column chosen per *coded* length; col-3 → 1484), and **internal turbo** from
> `interleave_stage2.i32` (column chosen per
> *info* block size). A column is used directly when its first-N entries already form
> a `0..N-1` permutation. **Every level's column does**, including the caller-ID
> `N=736` case, so the column is always taken directly:
> `perm[i] = TABLE[i·17 + level][:N]`.
>
> Do **not** derive the permutation by pruning a larger column — dropping entries ≥ N
> from the next larger column, e.g. building `N=736` out of the `N=744` column. That
> yields a different permutation, and a modem built to it will not interoperate.

## 3.6 BW500 DATA-path coding

This section specifies the BW500 DATA path so it can be built from spec alone:
level 4 (SHORT, record 3) in §3.6.1–§3.6.5 and the level below it (record 2) in
§3.6.6, plus the rate-1/3 variant used by control levels 0/1 in §3.6.4. The BW500
constant tables are produced in code by the companion implementation's generator,
`hfmodem/kestrel/rx/tablegen.py` (§3.5.4), rather than shipped as data. Nothing
ships: the tables that follow from the algorithms below are computed at the point
of use, and the two the algorithms do not reach — the measured de-rotation
integers and BW2300 record 2's base bins — are integer literals in the same
module.

### 3.6.1 Physical / demod frame (2-subband, L4 SHORT)

| Attribute | Value |
|-----------|-------|
| Sample rate | 48000 Hz |
| Sub-bands | **2**, centers **1350 Hz** (sub 0) and **1650 Hz** (sub 1) |
| Samples/symbol `H` | **512** |
| Pulse | **WOLA** Nyquist prototype (one sample/symbol at the symbol center; ~120 Hz per-subband LPF) |
| Columns/frame | **394** (`NSYM`); coded cells/frame **748** (`CODED`) |
| Modulation | differential-BPSK across columns on de-rotated data cells (`bit = 1 if Re(v_k · conj(v_{k−1})) < 0`) |

Analysed as a single band on a 256-sample symbol, the 500 Hz waveform does not
resolve into a recoverable constellation. It is two sub-bands at `H = 512`, as
above and in [`01 §500`](01-physical-layer.md).

### 3.6.2 De-rotation reference + cell selection

| Attribute | Value |
|-----------|-------|
| Reference | `grid480` unit-modulus, `derotation_reference_grid480.c64`. De-rotate `cell ⊗ grid480[idx]`. |
| Reference read index | sub 1: `471 + (462 + col) mod 471`; sub 0: `(460 + col) mod 469` |
| Data-cell selection | `usedmap.u8`: cell `(s, co)` is data iff `USED[17·(128·co + s) + off] != 1`, `off = 3` (L4 SHORT), `co ∈ 1..392` |
| Frame-start column `c0` | `9` for L4 SHORT (blindly locked by turbo self-consistency; payload-independent) |

### 3.6.3 Interleavers (Stage-1 raster + turbo interleaver)

| Attribute | Value |
|-----------|-------|
| Stage-1 (coded-cell → coded-bit) | `interleave_stage1.i32`; coded bits placed at `P = IDX1[off :: 17][:748]`, `off = 3` (1-based; 0 = pilot/pad) |
| Turbo (inner) interleaver | the BW500 stage-2 table `IL`; the 368-bit permutation = `IL[3 :: 17][:368]` (a full `0..367` permutation) |

### 3.6.4 Whitener + turbo FEC + CRC

| Attribute | Value |
|-----------|-------|
| Whitener (PN) | additive bit whitener over the **368 input bits**: `frame_bits = decoded XOR PN[:368]`. Bits `whitener_pn.u8` (regenerable from the VB6-`Rnd` LCG — see [`04 §4.2.3`](04-frame-block-formats.md) for the LCG). |
| FEC family | parallel-concatenated turbo, **two identical (13,15)-octal RSC** constituents, 8 states |
| Constituent (RSC) definition | state `s = reg0 \| reg1<<1 \| reg2<<2`; feedback `a = (u ⊕ reg1 ⊕ reg2)`; parity `p = (a ⊕ reg0 ⊕ reg2)`; next `s' = a \| reg0<<1 \| reg1<<2` |
| Rate / cell budget | **rate-1/2** for L4 (systematic + punctured parity: `coded[0:2N:2]=sys`, `coded[1:2N:2]` = parity1 at even i / parity2 at odd i); `N = 368` input → `2N+12 = 748` coded |
| Rate-1/3 variant (control levels 0/1) | no puncture: `coded[0:3N:3]=sys, [1::3]=par1, [2::3]=par2`, `3N+12` coded; same constituents + tail |
| Termination | **dual-tail, 12 trailing coded bits**: each constituent is flushed to state 0 with 3 tail steps (`u = reg1 ⊕ reg2`); tail bits interleaved `[s1,p1,s2,p2]×3` |
| Soft-decision | log-MAP (max*) BCJR, ~12 iterations |
| CRC | **CRC-16/GENIBUS** (poly 0x1021, init/xorout 0xFFFF, no reflection); over `frame[:44]`, appended big-endian as `frame[44:46]` |
| Frame payload | **43 payload bytes** per frame (byte 43 = marker, 44:46 = CRC); message length is signaled out-of-band (trim the final short frame) |

### 3.6.5 TX/RX pipeline order (BW500 L4)

```
TX:  frame(43B + marker) → CRC-16/GENIBUS append (2B, big-endian)
     → serialize to 368 bits → XOR PN whitener
     → turbo (13,15) rate-1/2 encode (+12-bit dual-tail)   → 748 coded bits
     → Stage-1 raster place  P = IDX1[3::17][:748]
     → constellation/diff-BPSK map onto data cells (usedmap-selected)
     → ⊗ grid480 reference (twiddle)  → 2-subband WOLA synth (H=512, 1350/1650 Hz)
RX = exact inverse (de-rotate → diff-BPSK → Stage-1 de-interleave
     → turbo log-MAP decode → de-whiten → CRC check).
```

### 3.6.6 Record 2 (host `BITRATE (3)`) — one-hot index modulation

The gear below the base level, 61 bps, and the level a stock BW500 pair opens a
delivery at. It is not §3.6.1's waveform slowed down: it is one-hot index
modulation, the family BW2300 records 2 and 3 use (§3.5.4), on this bandwidth's
own tables.

| Attribute | Value |
|-----------|-------|
| Samples/column `dw50` | **1024** (46.875 baud); no CP, no pulse shaping — a column is one steady bin |
| Band | bins **27..37** of the column's own 1024-point DFT; `span = 11`, `stride = 1`, `first_bin = 27` |
| Burst | **4 training columns**, then **226 frame columns** (4.91 s) |
| Frame columns | **202 data**, 3 bits each (606 coded), + **24 reference** |
| Reference columns | `(75k + 67) // 8`, `k = 0..23` |
| Base bin `alloc` | `27 + Int(Rnd·11)`, the first 226 draws of the record-2 interleaver gap |
| Reference class `map2` | `Int(Rnd·8)`, the 24 draws after them; the state then lands on record 3's stage-1 seed `6437280`, which is what fixes the 226/24 split of the 250-draw gap |
| Lit bin | data column: `((alloc + clsparm[3][v] − 27) mod 11) + 27`, `v` = the column's 3 coded bits MSB-first; reference column: `((alloc + map2 − 27) mod 11) + 27` |
| Unreachable offsets | 8 values on an 11-bin wheel, so 3 bins of every data column can never be lit — a wrong alignment shows up as impossible readings before any decode |
| Stage-1 (coded-cell → coded-bit) | `P = IDX1[2 :: 17][:606]` |
| Turbo (inner) interleaver | `IL[2 :: 17][:297]` |
| Rate / cell budget | rate-1/2 as §3.6.4; `N = 297` input → `2N+12 = 606` coded |
| Whitener | `frame_bits = decoded XOR PN[:297]` |
| CRC | CRC-16/GENIBUS over `frame[:35]`, appended big-endian as `frame[35:37]` |
| Frame payload | **34 payload bytes** (byte 34 = marker, 35:37 = CRC) |

Known from the two record-2 overs of the 2026-07-13 `bw500-zeros256` loopback
session: every one of the 226 columns of both overs lights the bin these tables
predict, and both decode CRC-clean to the 34 and 7 payload bytes the session's
host log records. Reference columns 73 and 74 satisfy every constraint two overs
can raise; the closed form above takes 74, and either choice decodes both overs
to the same bytes.

**The 4 training columns are open.** They differ between the two overs, so they
carry a per-over draw, and no `Rnd` stream anchored anywhere reproduces both. The
wide bandwidth's law — the record's own `alloc` read from index 1, offset by
`Int(Rnd·16)` per column (§3.5.4) — has no solution here at the gap that law is
defined with, and the two overs' eight bins cannot fix a 24-bit stream position
and a gap besides: 32 of 40 invented bin pairs admit a solution on the same
search. BW500 offers no anchor either, its base level being sub-band DBPSK with
no one-hot preamble to invert (best single-bin share of a base lead-in, over
every phase and both combs: 0.32, against 1.00 for these columns).

Nothing reads them for correctness — a receiver cannot know the transmitter's
stream position, the two ends of a link sitting at different states — so they are
training: energy in the comb for gain, timing and frequency. A transmitter keys
the allocation with the draw at zero, and the 226 frame columns behind it are
exact: rendered against the recording, the body correlates **0.99997** and the
whole over **0.987**, the difference being those four columns alone.
