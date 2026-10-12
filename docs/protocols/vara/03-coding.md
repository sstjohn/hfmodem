# 03 — Coding layer (scrambler / interleaver / FEC / CRC)

Two coding regimes exist and must not be conflated:

- **MFSK handshake bursts** (CR / connect-response / connected-ack) carry **no
  FEC, interleaving or whitening** — their tones are a public-LCG (VB6 `Rnd`) PRNG
  stream keyed to the destination callsign, specified in
  [`04 §4.2`](04-frame-block-formats.md). They do not use the stages in this file.
  The one shared piece is the CRC (§3.4), which derives the PRNG seed.
- **DATA levels** of every family and bandwidth (index, tone and OFDM,
  [`01 §1.0`](01-physical-layer.md); host `BITRATE (N) … bps`): §3.0–§3.4 apply to
  all of them. §3.5 is the BW2300 index levels and the link-setup, §3.6 the BW500
  base path, §3.7 the tone and OFDM levels.

## 3.0 Pipeline order

One chain serves every DATA level; only the last stage, bits to cells, belongs to
the waveform family.

```
frame    payload | control byte | CRC-16/GENIBUS(payload | control), big-endian
bits     MSB first, zero-padded to the record's N info bits
whiten   XOR the PN (§3.1) from index 0
turbo    (13,15) on the record's stage-2 permutation (§3.2, §3.3)
puncture group law to the record's coded length (§3.3); none at rate 1/2 or 1/3
gather   onair[k] = coded[π1[k]], π1 the record's stage-1 permutation
cells    index: bpc bits → Gray → one lit bin ([01 §2300])
         tone:  2 bits → Gray quarter-turn increment per band ([01 §Tone family])
         OFDM:  bpc bits → LUT column pos, then × exp(−j2π·S/128) ([01 §OFDM family])
```

RX is the exact inverse. The MFSK handshake bursts have no stage of this chain
(see [`04 §4.2`](04-frame-block-formats.md)).

## 3.1 Scrambler / randomizer (DATA)

| Attribute                         | Value |
|-----------------------------------|-------|
| Present?                          | Yes, at every DATA level |
| Type (additive/multiplicative)    | Additive bit whitener (XOR with a fixed PN sequence) over the N info bits, before the turbo encoder |
| Polynomial / sequence             | VB6 `Rnd` from seed **29318**, `bit = state ≫ 23` (`tablegen.whitener()`, `whitener_pn.u8`). One sequence at every bandwidth |
| Seed / init state                 | Restarts at index 0 for every frame |
| Reset boundary (per frame/block)  | Per frame |

The OFDM levels add a second, cell-domain scramble after modulation, drawn from
the same seed (§3.7).

## 3.2 Interleaver (DATA)

| Attribute                          | Value |
|------------------------------------|-------|
| Present?                           | Yes, at every DATA level |
| Type                               | Two permutations per record: **stage 1** over the coded length (the channel interleaver, applied as a gather `onair[k] = coded[π1[k]]`) and **stage 2** over N (the turbo encoder's internal interleaver) |
| Generator                          | One VB6 `Rnd` chain per bandwidth (`tablegen.permutation`, `tablegen.BW500` / `BW2300` / `BW2750` as `(stage-1 seed, coded, N)` per record). Stage 2 continues stage 1's stream; after both come the record's **gap** draws (the index levels' bin tables, the tone pilots, the OFDM gap cells), and the gap closes exactly on the next record's stage-1 seed |
| Tables                             | The BW2300 stage-1 / stage-2 columns are also VARA's 17-column `interleave_stage1.i32` / `interleave_stage2.i32` (`[i·17 + record]`); every column is used directly, never pruned (§3.5.5) |
| Per-speed-level / -bandwidth variation | **Yes** — per record and per bandwidth. BW2750 shares BW2300's chain through record 8 (L1–9) and runs its own from L10, closing on 4061299 after L17 |

## 3.3 Forward error correction (FEC) (DATA)

| Attribute                                    | Value |
|----------------------------------------------|-------|
| Family (conv / LDPC / Turbo / RS / none)     | **Turbo** (parallel-concatenated convolutional) |
| Constituent generators (octal)               | **(13, 15)** octal RSC encoders, 8 states (§3.6.4) |
| Rates                                        | **1/3** (`coded = 3N + 12`, unpunctured) at records 0–1; **1/2** (`2N + 12`) at the base levels and every tone row; **2/3, 4/5, 5/6** by the group puncture at the OFDM rows |
| Termination                                  | 12-bit dual tail `[s1,p1,s2,p2]×3`, both constituents flushed to state 0 |
| Soft-decision handling                       | log-MAP (soft) decode |

**Group puncture.** Wherever `coded < 3N + 12`:

```
q = coded − N − 12,  k = round(N / q)
group g carries info bits s[kg .. kg+k−1] as   s[kg], P_g, s[kg+1], …, s[kg+k−1]
P_g = p1[kg] for even g, p2[kg] for odd g      (constituent 1 / 2 parity at the group's first bit)
then the 12-bit dual tail, then zeros up to coded
```

`k = 1` is the rate-1/2 mux (`s p1 s p2 …`); rate 2/3 reads `s p1 s | s p2 s`.
`coding/turbo.py` `encode_punctured` / `decode_punctured`. Per OFDM level
(the tone rows are all `k = 1`, pad 0):

| bw | levels | k | zero pad (bits) |
|---|---|---|---|
| 2300 | L10–16 | 1 / 2 / 4 / 2 / 4 / 5 / 4 | 0 / 0 / 4 / 0 / 2 / 4 / 3 |
| 2750 | L11–17 | 1 / 2 / 4 / 2 / 4 / 5 / 4 | 0 / 0 / 4 / 0 / 2 / 2 / 3 |
| 500 | L7–13 | 1 / 2 / 4 / 2 / 4 / 5 / 4 | 0 / 0 / 1 / 0 / 0 / 2 / 3 |

## 3.4 CRC / error detection

| Attribute                         | Value |
|-----------------------------------|-------|
| Present? scope (frame/block)      | Yes — per frame |
| Width                             | **16 bits** |
| Polynomial                        | **CRC-16/GENIBUS**, poly **0x1021** |
| Init / final-XOR / reflection     | init **0xFFFF**, xorout **0xFFFF**, refin=false, refout=false (i.e. GENIBUS = CCITT-FALSE with output inversion) |
| Coverage (which bytes)            | For the MFSK handshakes, the destination-callsign ASCII, whose CRC seeds the tone PRNG. For DATA, every frame byte before it — the payload and the control byte — appended big-endian |

## 3.5 BW2300 index levels and the link-setup

The BW2300 index levels carry both the caller-ID link-setup burst and the base
level's data. §3.5.3 gives the chain and its framing, §3.5.4 the constant tables,
§3.5.5 the caller-ID link-setup frame.

### 3.5.3 BW2300 index-level link-setup / DATA chain

The BW2300 index levels (host L1–4, records 0–3), the link-setup included, use
the §3.0 chain with the index-family cell stage. In functional terms (the bin law
is in [`01 §2300`](01-physical-layer.md)):

```
DEMOD:  per emission column, the lit bin → Gray⁻¹ → bpc on-air bits
        → channel de-interleave (the record's stage-1 column, gather)
        → turbo (13,15) log-MAP decode (the record's stage-2 column)
        → de-whiten (XOR the PN, §3.1)
        → CRC-16/GENIBUS check  → frame → caller callsign + session params
MOD  =  the exact inverse (forward) chain.
```

The modulation is index/position type — the coded value selects the carrier bin —
so RX is an argmax over the received energy grid.

- **Framing.** A frame is **92 bytes = 736 bits**:
  `[89 payload bytes][1 control byte][2-byte CRC, big-endian]`, with **no
  header**. CRC-16/GENIBUS over `body[0:90]` equals `frame[90:92]` big-endian. The
  link-setup burst uses the same 92-byte framing law as a DATA over.
- **Per-over payload slicing.** 89 consecutive payload bytes per over
  (`over_i = payload[89·i : 89·i+89]`); a 512-byte payload therefore occupies 6 overs.
- **Whitener.** XOR with the PN of §3.1, MSB-first, from index 0 per frame.
- **Turbo.** Parallel (13,15)-octal RSC, **rate 1/2, N = 736 → 1484 coded bits**
  (= 2·736 + 12 dual-tail) at records 3 and 2, and **rate 1/3, `3N + 12`** at
  records 1 and 0 — 612 from 200 and 300 from 96. The internal interleaver is
  `interleave_stage2.i32` column 3,
  `perm[i] = IL2[3 + 17·i][:736]` — a native `0..735` permutation, used directly with
  **no pruning**.
- **Channel interleaver.** `interleave_stage1.i32`
  column 3 (a 1484-entry permutation), applied as a GATHER: `onair[k] = coded[π[k]]`.

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
| Constellation LUT (bandwidth-independent) | 64-entry complex LUT `constellation_lut_64.c64`, value law `cell = LUT[pos + 6·coded]` — six constellation columns interleaved with period 6 (BPSK / QPSK / 8-PSK on the unit circle; 16-QAM, **32-QAM cross** and 64-QAM with outermost moduli 1.1 / 1.5 / 1.6). The table is a **truncated prefix of the raster** — column `pos` present only up to `coded = ⌊(63−pos)/6⌋` — so its zero entries are padding, not nulls, and entry 13 is QPSK `coded=2` (`−j`), not a dedicated pilot. (An earlier revision read the table as complete: "14 distinct moduli" and per-column APSK rings; both were truncation artifacts.) At the OFDM levels the raw bit group indexes a column directly, with no Gray pass (§3.7); all 32 pos-4 labels are observed on tape. Full geometry: [`06 §6.1c`](06-speed-gearshift.md) and `hfmodem/kestrel/rx/tablegen.py`. Byte-identical BW500↔BW2300. |
| Whitener PN | `whitener_pn.u8`, `tablegen.whitener()`: VB6 `Rnd` from seed 29318, one sequence at every bandwidth (§3.1) |
| Block-interleave map (stage 1 / stage 2) | `bw2300/interleave_stage1.i32`, `bw2300/interleave_stage2.i32`; cell-usage `bw2300/usedmap.u8` |
| Cell scramble `S` (OFDM levels) | `tablegen.scramble()`: `Int(Rnd·129)` from seed 29318, 62 bin rows × 409 columns; cells go on air `× exp(−j2π·S/128)` (§3.7) |
| Tone grid rows | `tablegen.grid_row(b)`: rows 0/1 are the BW500 base's `grid480` reference, rows 2–12 measured literals ([`01 §Tone family`](01-physical-layer.md)) |
| Base-record bin-placement tables | Record 3's is `9 + Int(Rnd*16)`, 395 draws from the gap after its own two permutations, closing on record 4's stage-1 seed ([`01 §2300`](01-physical-layer.md)). **Every record's gap holds exactly its own `ncols + 24` draws and closes on the next record's stage-1 seed** — 148 / 252 / 252 / 419 at rec0..rec3 — which is what fixes 124 / 228 / 228 / 395 emission columns and 24 reference columns at each. The stream is therefore identified at all four. **The table is generated only at record 3.** Below the base level `first_bin + Int(Rnd*span)` off the record's own gap lands within **−1..+3** of the true bin and never reliably on it — measured at all three, `{−1: 11, 0: 21, +1: 36, +2: 25, +3: 7}` over rec0's 100 data columns, `{3, 56, 83, 50, 12}` over rec1's 204 and `{9, 57, 77, 48, 13}` over rec2's — so records 0, 1 and 2 are demodulated from a measured table or not at all. All three are now measured (`tablegen.base_bins_col0/1/2`), read off the 2026-09-04 gear-down tapes. Records 0 and 1 need no known plaintext to be read: with `bpc` 3 at stride 2 an allocation reaches a quarter of a 64-bin comb and half of a 32-bin one, so overs of differing payloads intersect down to one bin a column. Record 2 spends 16 values at stride 2 on 32 bins, which is a whole parity of its comb, and had to be read from its payload. Reference columns: 24 at every record, at `(41k+33)//8` on rec0's 124 columns and `(104k+93)//11` on rec1's and rec2's 228 — records 1 and 2 share theirs position for position. |
| Turbo constituents / termination / puncture | Same as BW500 (§3.6.4): parallel-concatenated **(13,15)-octal RSC**, 12-bit dual tail, log-MAP. Rate 1/3 at records 0–1, 1/2 at records 2–8, the group puncture of §3.3 above that |

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
| Frame payload | **43 payload bytes** per frame (byte 43 = the control byte, 44:46 = CRC); message length is signaled out-of-band (trim the final short frame) |

### 3.6.5 TX/RX pipeline order (BW500 L4)

```
TX:  frame(43B + control byte) → CRC-16/GENIBUS append (2B, big-endian)
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
| Frame payload | **34 payload bytes** (byte 34 = the control byte, 35:37 = CRC) |

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

## 3.7 Tone and OFDM levels

The coding half of the tone family (BW2300 L5–9, BW2750 L5–10, BW500 L5–6) and
the OFDM family (BW2300 L10–16, BW2750 L11–17, BW500 L7–13). Their physical half
is [`01 §1.0`](01-physical-layer.md) and the sections it points to; per-level
`N`, `coded` and frame sizes are [`06`](06-speed-gearshift.md). Closed from
known plaintext on the high speed capable bench tapes: every frame of every row decodes
CRC-clean and byte-exact against the transcript
(`evidence/KPA-bw2300-const.md`, `KPA-bw2300-subband.md`,
`KPA-bw2750-const.md`, `KPA-bw500-const.md`, `KPA-tone-bw2750-bw500.md`).

**Frame.** `floor(N/8)` bytes: payload, one control byte, then CRC-16/GENIBUS
over both, big-endian; N bits are the frame zero-padded where N is not a byte
multiple (2/4/2 bits at BW2300 L6/7/8). The control byte is the ARQ layer's
per-frame field ([`04 §4.2B`](04-frame-block-formats.md)): `0x81` to `0x99` in
steps of 4 on full frames, `0x19` on a session's first data over,
`0x82` on a short one. A payload short of the row's capacity is filled ahead of
the control byte:

```
spare ≥ 3   payload | 14 <caller CRC high> | 00 … | T | 82 | CRC
spare = 2   payload | 14 T | 82 | CRC
T = 06 at BW2300 and BW2750, 07 at BW500
```

`<caller CRC high>` is the high byte of CRC-16/GENIBUS of the session caller's
callsign, as in every short over ([`04 §4.2B`](04-frame-block-formats.md)); the
high speed capable tapes carry `09` (`N0CALL` calling) and `f6` (`W9SSJ`).

**Chain.** §3.0 unchanged: whiten, turbo on the record's stage-2 permutation,
group puncture where the rate is above 1/2 (§3.3; every tone row is rate 1/2),
gather by the stage-1 permutation. The permutations are the bandwidth chain's own
records (§3.2): BW2750 L5–9 are BW2300 records 4–8 outright, and BW500 L5–6 are
BW500 records 4–5.

**Tone cells.** Two on-air bits MSB first → Gray `[0,1,3,2]` → a quarter-turn
increment, filled band-major (band 0's data columns in time order, then band 1's),
skipping pilot columns ([`01 §Tone family`](01-physical-layer.md)).

**Tone pilots** (`tablegen.tone_pilots`). Per band in order, draw
`co = 5 + Int(Rnd·(ncols − 6))` from the record's gap, rejecting repeats, until the
band has its count; `t = 8 + co`. The draws close on the next record's stage-1 seed:
37 / 36 / 36 / 40 / 41 draws at L5–9 (BW2300 and BW2750 alike), 39 at BW2750 L10,
38 and 36 at BW500 L5 and L6. BW500 record 3's 36 pilots (L4) are the same law.

**OFDM cells.** `bpc` on-air bits MSB first as `v`, `cell = LUT[v]` from LUT
column `pos` indexed by `v` directly (`tablegen._column(pos, v)`). There is no Gray
pass in front at any column: the `clsparm` Gray rows belong to the index levels
only, and all 32 labels of the pos-4 cross are observed on tape. Fill is
bin-major, symbol order within a carrier.

**Scramble** (`tablegen.scramble()`), OFDM only:

```
state ← 29318 (the whitener's seed)
for k in 0..61 (FFT bin), c in 0..408 (column): state ← Rnd step; S[k][c] = Int(Rnd·129)
on air: cell · exp(−j2π·S[k][c] / 128)
c = s − 4 (PSK, column 0 the reference symbol) or s − 5 − 154·f (QAM, frame f)
```

So bin k column c is draw `1 + 409·k + c`: bin 8 column 0 is draw 3273 (BW2300),
bin 3 draw 1228 (BW2750), bin 27 draw 11044 (BW500). The lead is not scrambled.

**Gap cells** (`tablegen.gap_cells`), OFDM only. From the record's gap, carrier
by carrier from the first bin, draw `7 + Int(Rnd·142)` (PSK: the symbol of a repeat
cell, a +1 differential step) or `Int(Rnd·19)` (QAM: the pilot row whose next row
holds an extra +1), redrawing a value its carrier already holds. One per carrier
at BW2300 (49 draws) and BW2750 (59), four at BW500 (44 plus the redraws, 44–49).
The draws close exactly on the next record's stage-1 seed, which is what fixes
the gap as these cells.
