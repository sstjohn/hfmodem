# 01 — Physical layer (OFDM parameters)

This section specifies the transmitted waveform: symbol geometry, carrier layout,
cell values and burst structure, one block per bandwidth mode. The coding chain
that feeds it — whitening, FEC, interleaving, CRC — is [`03`](03-coding.md);
frame layouts are [`04`](04-frame-block-formats.md); the per-level ladder is
[`06`](06-speed-gearshift.md).

In the 500 Hz mode the two directions carry different waveforms. The initiator
sends long DATA bursts, the responder sends only short control/ACK bursts of
roughly 0.34–0.68 s; §1.1 through §1.4 describe the DATA waveform and §1.5
separates the two populations. Fields left empty are not specified here.

## 1.1 Global / per-bandwidth parameters

### 500 Hz mode

The waveform is **two sub-bands at 1350 Hz and 1650 Hz**, one symbol every **512**
samples at 48 kHz, carrying **differential BPSK** across columns. The full
demodulation chain — de-rotation reference, cell selection, interleavers, FEC — is
[`03 §3.6`](03-coding.md); this table is its physical half.

| Parameter                     | Value | Unit |
|-------------------------------|-------|------|
| Audio sample rate (host)      | 48000 | Hz   |
| Occupied bandwidth            | ~500–560 (99 % power; energy spans ~1150–1850 Hz) | Hz   |
| Sub-bands                     | **2**, centres **1350.0** (sub 0) and **1650.0** (sub 1) | Hz   |
| Samples per symbol `H`        | **512** | —    |
| Symbol duration / rate        | **10.667** ms (= 512/48000); **93.75** baud | ms   |
| Cyclic-prefix / guard length  | **none** (0) | ms   |
| Pulse / windowing             | **WOLA** Nyquist prototype, one sample per symbol at the symbol centre (~120 Hz per-sub-band low-pass) | —    |
| Modulation                    | **differential BPSK** across columns on de-rotated data cells: `bit = 1 if Re(v_k · conj(v_{k−1})) < 0` | —    |
| Columns per frame             | **394**; coded cells per frame **748** (L4 SHORT, `off = 3`) | —    |

That is the base level, host `BITRATE (4)`. **The level below it is a different
waveform family**: record 2 (host `BITRATE (3)`, 61 bps, 34 payload bytes a block,
the level a stock pair opens a delivery at) is one-hot index modulation, not
sub-band DBPSK — one 1024-sample column lights exactly one bin of **27..37**
(1265.6–1734.4 Hz, 46.875 baud), 4 training columns then 226 frame columns of
which 202 carry three bits each. Its energy fills the band evenly where the base
level shows two humps with a notch at 1500 Hz, and its coding half is
[`03 §3.6.6`](03-coding.md). Reversed off the two record-2 overs of the
2026-07-13 `bw500-zeros256` loopback session: the 226 frame columns exactly and
keyable, the 4 training columns not — those carry a per-over draw whose
stream this bandwidth gives no way to locate, and nothing reads them for
correctness.

**A 256-point analysis window misreads this mode.** On a 256-point FFT the
waveform presents as a 187.5 Hz carrier grid — 5.333 ms symbols at 187.5 baud, an
alternating ±150 Hz tone apparently *off* the integer bin grid, a continuous pilot
at k=0, and roughly 785 symbols in a 4.35 s burst. That model does not recover a
frame. The off-grid tone and the accompanying half-subcarrier / +180°-per-symbol
anomaly are consequences of analysing a 512-sample symbol on a 256-point window:
at H=512 the two sub-band centres are ordinary carriers and the anomaly
disappears, and the symbol count halves accordingly — 394 columns per frame, not
~785 symbols per burst. §1.2 through §1.5 describe the waveform as it appears in
that 256-point window.

### 2300 Hz mode

The base DATA level (**rec3**) is **index modulation**: one lit bin per
512-sample symbol, its *position* carrying four bits. It is not a multi-carrier
constellation. Info rate at this level is **175 bps**.

| Parameter | Value | Unit |
|-----------|-------|------|
| Audio sample rate (host) | 48000 | Hz |
| Samples per symbol (`dw50`) | **512**; no cyclic prefix, no window, no overlap-add | samples |
| Symbol rate | **93.75** baud (= 48000/512) | Bd |
| Transform | plain length-**512** IFFT per symbol, concatenated | — |
| Emission | **one-hot**: exactly one lit bin per symbol, `−1j`-phased, constant magnitude | — |
| Bin span | bins **9 .. 24** of the 512-point grid (`first_bin = 9`, `span = 16`) | — |
| Bits per column (`bpc`) | **4** — carried by *which* bin is lit, Gray-mapped | — |
| Columns per over | **395** = 371 data + 24 reference; 371 × 4 = 1484 coded bits | — |
| Frame container | **92** bytes (payload + CRC), 736 turbo input bits | — |
| Occupied bandwidth | ~**735 – 2350** | Hz |

#### Base level — burst structure

A DATA over is **210217 samples, 4.380 s**, laid out in 512-sample blocks:

| Blocks | Content |
|---|---|
| 0–1 | silent lead-in |
| 2–13 | the 12-symbol reference preamble |
| 14 … | the 395 emission columns, in time order (371 data + 24 reference) |

Each 512-sample block is one index-modulated column: exactly one lit bin in
9..24 of the 512-point FFT, the bin position being the coded value. The four
512-sample sub-blocks of a 2048-sample span are decorrelated (inter-block
correlation ≈ 0), not replicated (512-sample periodicity correlation ≈ 0.33).
Every cell is the constant one-hot `−1j` at `|cell| = 97.7329` (phase −90°); the
magnitude carries nothing. Synthesis is a length-512 IFFT per symbol,
concatenated — no window, no cyclic prefix, no overlap-add, no twiddle.

A DATA over carries **one record**, not a sum of layers. The rec3 synthesis alone
accounts for the whole over: it matches at **0.9851** in the time domain and
**0.9851** on complex 2048-point cells across all 65 carriers, at one global real
scale (**0.1909** in the wav-float domain), identically across all seven DATA
overs of a session — a deterministic match, not a fit. The remaining **2.95 %** is
the 12-symbol preamble; it does not correlate with a rec0, rec1 or rec2 synthesis
(`|corr| ≤ 0.027` over all guard and lag offsets). Records rec0/rec1/rec2 are
*other speed levels*, not simultaneously transmitted layers.

#### Base level — bin placement law and tables

The coded value selects the OFDM **bin**; the cell value is a constant one-hot
`−1j`. rec3 geometry constants: `first_bin = 9`, `span = 16`, `stride = 1`,
`bpc = 4`.

**Tables** (all seg=3 columns, indexed by `ai = (col−1)·2` for emission columns
`col = 1..395`):

- `alloc_col3.i32` — per-column bin-allocation offset.
- `map1_col3.i32` — reference gate: `0` ⇒ data cell, `≠0` ⇒ reference cell.
- `map2_col3.i32` — reference class value (reference cells only).
- `clsparm_gray.i32` — Gray bit-map, row `(bpc−2)·16` (bpc4 row for rec3).

All four are produced, under a byte-exactness gate, by the companion
implementation's generator `hfmodem/kestrel/rx/tablegen.py`; they are not
shipped as data files, and the `.i32` names above identify the tables rather
than files.

The bpc=4 Gray row is `[0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]`.

**Forward (TX)** — walk `col = 1..395` in order, `ai = (col−1)·2`:

- if `map1_col3[ai] ≠ 0`: **reference** column — emit one one-hot `−1j` cell, consume
  no payload bits (its class is `map2_col3[ai]`).
- else **data** column — consume `bpc = 4` on-air bits MSB-first as `coded (0..15)`; then
  ```
  gray = clsparm_gray[(bpc−2)·16 + coded]
  bin  = ((alloc_col3[ai] + stride·gray − first_bin) % span) + first_bin      # ∈ 9..24
  ```
  emit one-hot `−1j` at internal grid `[(col−1) + bin·940]` (equivalently audio
  carrier `= (64/span)·bin = 4·bin`).

**RX (inverse)** — per data column in order: `raw = bin − first_bin`;
`gray = (raw − (alloc_col3[ai] − first_bin)) % span`; `coded = clsparm⁻¹[gray]`; emit
4 bits MSB-first. Concatenate the **371 data columns** ⇒ 1484 on-air bits →
un-interleave (`interleave_stage1.i32` col 3, gather `coded[il1[3::17][k]] = onair[k]`)
→ turbo decode (rate-1/2, N=736, turbo interleaver `interleave_stage2.i32` col 3)
→ de-whiten (XOR `whitener_pn.u8`, MSB-first, from index 0) → 92-byte frame →
CRC-16/GENIBUS over `frame[0:90]` == `frame[90:92]`.

The receive front end for real audio is: energy-segment the burst → 512-sample
blocks → per-block 512-point FFT → `argmax|·|` over bins 9..24 = the internal bin
→ block index, after the 12 preamble blocks, = the emission column in order → the
inverse law above.

#### Base level — the 12-symbol reference preamble

Blocks 2–13 carry a 12-symbol training preamble. Each symbol is a single one-hot
`−1j` tone (`|cell| = 97.7329`, phase −90°) on internal bin `b ∈ 9..24`
(`first_bin = 9`, `span = 16`), at the same 512-sample OFDM symbol length as the
data columns. It is emitted from the reference grid (stride 100,
`grid[(sym−1) + 100·bin]`); the per-cell phase field for rec3 is `0`, giving the
constant `−1j`.

**Bin law**, per symbol `s = 1..12`:

```
bin(s) = ((ALLOC[s] + stride·S_rand − first_bin) mod span) + first_bin        # ∈ 9..24
first_bin = 9, span = 16, stride = 1
ALLOC[1..12] = [24, 18, 18, 9, 12, 22, 11, 22, 17, 23, 12, 14]
             = alloc_col3[2·s]
S_rand = Int(Rnd()·16), a VB6 Rnd draw
```

The additive `ALLOC` term is the same allocation table
(`alloc_col3.i32`, above) the data columns use, at the
per-symbol offset `ai = 2·s`. `map2` is `0` in this region, so unlike the 24
in-band reference columns — which are fixed at `bin = alloc` — the preamble adds a
live PRNG term. This is a data column's own law with the draw where the Gray value
would be, and it is the record's own geometry throughout — which is what lets the
same three lines render rec2's preamble (below).

**The PRNG is payload-independent and session-deterministic.** It is the VB6 `Rnd`
generator, a 24-bit LCG `state ← (state·0x43FD43FD + 0xC39EC3) mod 2²⁴`,
advance-then-read, with `Int(Rnd()·16) = state ≫ 20`. `Randomize` is never called,
so the stream is identical on every run: sessions with different payloads and
different connect timing produce byte-identical preambles for every over. The
overs draw from one continuous stream — 12 consecutive draws per over,
back-to-back, with a single extra draw between over 0 and over 1. Over 0 begins at
LCG state **4107106**, which is the state left by the fixed connect and link-setup
`Rnd` consumption of this ARQ scenario. A licensed peer, or different pre-data
traffic, shifts that starting state; the per-over draw structure and the bin law
are invariant, and this document does not fix the starting state for other
scenarios.

**Reference vectors** (canonical for implementers):

| over | 12 preamble bins (blocks 2–13) |
|---|---|
| 0 | 14 24 10 21 23 21 22 24 15 24 23 15 |
| 1 | 17 16 9 16 12 9 12 22 18 21 19 14 |
| 2 | 12 21 19 20 20 22 22 10 18 19 13 18 |
| 3 | 9 16 11 17 10 19 10 9 14 9 9 16 |
| 4 | 12 22 23 15 10 18 11 22 20 19 9 18 |
| 5 | 10 12 17 24 20 22 23 17 12 23 19 19 |
| 6 | 18 10 24 13 16 22 18 14 14 18 22 10 |

**Synthesis:** for each of the 12 symbols, `X[b] = −1j·97.7329`,
`X[512−b] = +1j·97.7329`, length-512 IFFT (real), placed at blocks 2..13; the 395
data columns follow at block 14. The preamble carries no payload — a receiver
decodes the 92-byte frame CRC-clean from the 395 data columns alone, and VARA
decodes a transmission that omits it. Emitting it makes the base-level transmit
waveform time-domain byte-identical.

#### Record 2 — the same preamble, six symbols long

**The preamble occupies 6144 samples at rec3 and at rec2**, so its symbol count is
`6144 / dw50`: twelve at rec3, **six at rec2**. It is a coincidence of those two
records and not a constant span: rec1 and rec0 were recorded on 2026-09-04 and
their preambles are **four symbols and two**, 4096 samples at both, against the
6144-sample prediction of six and three (§"Records rec0 and rec1 on tape").
The symbol count is the record table's own field — 2 / 4 / 6 / 12 at rec0..rec3 —
rather than a sample span divided out. The bin law above is unchanged — rec2
supplies `first_bin = 17`, `span = 32`, `stride = 2`, and

```
ALLOC[1..6] = [35, 40, 39, 36, 31, 24] = rec2_bins[s]
```

which is rec2's own 228-entry column bin table
(→[`table-derivations §5a`](../table-derivations.md)) read from index 1, exactly as
rec3's twelve are `alloc_col3` read from index 1. The preamble therefore has no
table of its own at either record.

The draws come from the one continuous stream: a rec2 over consumes its six back to
back with no gap after the preceding over, whatever record that over used.
Measured on the 2026-08-14 two-sided bench session, both directions — each
station's own stream located by inverting its base-level preambles, and the same
six allocations falling out of both:

| side | 6 preamble bins | Rnd state at the session's first captured preamble |
|---|---|---|
| a2b | 43 48 43 20 47 44 | 2767643 (2 base overs ahead of it) |
| b2a | 17 38 43 32 17 20 | 3653944 (1 base over ahead of it) |

**Synthesis** is the base level's with rec2's geometry: `X[b] = −1j·|cell|`,
`X[1024−b] = +1j·|cell|`, length-1024 IFFT, six symbols, then the 228 data columns.
Rendered that way one whole over correlates **0.99997** against the recording and
its preamble alone **1.000000**, against **0.987** for the same body with the six
symbols left silent.

#### Base-level records rec0–rec3

The base levels differ only in geometry, rate and source length:

| record | `dw50` | first_bin | span | stride | `bpc` | coded | info | symbols/block | preamble symbols | spread = 64/span |
|---|---|---|---|---|---|---|---|---|---|---|
| rec0 | 2048 | 33 | 64 | 2 | 3 | 300 | 96 | 25 | 2 | ×1 |
| rec1 | 1024 | 17 | 32 | 2 | 3 | 612 | 200 | 51 | 4 | ×2 |
| rec2 | 1024 | 17 | 32 | 2 | 4 | 816 | 402 | 51 | 6 | ×2 |
| rec3 | 512 | 9 | 16 | 1 | 4 | 1484 | 736 | 102 | 12 | ×4 |

**Emission columns are `coded/bpc + 24` at every record** — 124 / 228 / 228 / 395 —
and the 24 are the reference columns, the same count at all four. The tenth column
above is the training preamble's symbol count, not that reference count; it was
read as the reference count once, and the two agree at no record.

The first eight records of the record table are the speed ladder, at
**24 / 41 / 82 / 175 / 270 / 363 / 549 / 735 bps**; rec3 is the 175 bps level,
reported as `BITRATE (4)`.

Derived record fields: bins per column = `2·span` (rec3: 32); two spread fields
equal `32·span` and `17·span` (rec3: 512 and 272); and `symbols/block · span` is a
constant ≈ 1620. Each record also carries a DATA-eligible flag (0 for rec3) and one
further field whose rec3 value is 158; neither is specified here.

The internal bin maps into audio at `spread = 64/span`, so rec3's internal bins
9..24 land on audio carriers 36..96.

**Per-record coding source.** Every record turbo-encodes a **prefix of the same
frame buffer** — one payload, re-encoded at each rate, each record truncated to its
own info length:

| seg | record | bytelen | `N` (info bits) | coded | rate | source |
|----:|----:|----:|----:|----:|:---:|:---|
| 0 | rec0 | 12 | 96  | 300  | 1/3 | `frame[0:12]` → 96 bits (exact) |
| 1 | rec1 | 25 | 200 | 612  | 1/3 | `frame[0:25]` → 200 bits (exact) |
| 2 | rec2 | 50 | 402 | 816  | 1/2 | `frame[0:50]` → 400 bits **+ 2 zero-pad** = 402 |
| 3 | rec3 | 92 | 736 | 1484 | 1/2 | `frame[0:92]` = whole frame → 736 bits (exact) |

`frame` = `payload + trailer + CRC16-GENIBUS`, and **each record frames its own**:
the last two bytes a record encodes are a CRC-16/GENIBUS over the bytes before
them, and the byte before those is the ARQ layer's control byte. Measured on the
2026-09-04 ladder tapes — 22 of 22 record-0 overs and 8 of 8 record-1 overs end in
a valid CRC over their own bytes, and the counter payload their peer was pushing
reads back in consecutive non-overlapping chunks of 9 and 22 bytes. So a record is
not a prefix of the base frame with the trailer and CRC cut off; it is a shorter
frame of the same shape, carrying `bytelen − 3` payload bytes an over. rec2's 402
info bits are the 400 bits of its 50 frame bytes plus 2 trailing zero bits, then
whitened.

Per record the source is prepared as: copy the first `bytelen` bytes of the shared
frame; unpack MSB-first, 8 bits per byte, into an `N`-long bit array, so any
surplus is zero-pad; whiten in place, `src_bits[i] ^= PN[i]`, `i = 0..N−1`, with the
same PN from index 0; turbo-encode to `coded` bits, `coded = 3N+12` for rec0/rec1
and `2N+12` for rec2/rec3.

Per-record interleavers are the record's own columns of the shared maps: channel
interleave `il1[seg::17][:coded]`, turbo interleave `il2[seg::17][:N]`. Each is a
full permutation of its length for all four records (300/96, 612/200, 816/402,
1484/736).

#### Records rec0 and rec1 on tape

Both were recorded on 2026-09-04 by walking a stock pair down the ladder under
band-limited noise on the caller's input — the responder's level follows the
*receiver's* feedback, so the overs that drop a gear are the responder's. Under
the noise that commands the drop roughly half the columns' argmax is wrong, so the
first runs cut the noise mid-session and read the two or three overs the gear-shift
takes to climb back. That precondition is gone: on a bench the noise is the
harness's own, a seeded buffer written to the caller's input by the stream that
records it, so the tape holds it sample for sample and one integer lag takes it off
— a −10 dB session reads 0.132 to 0.161 RMS throughout and, subtracted, reads
numerically zero between overs and 0.048 during them. Every over of every session
is then readable, which is where the frame counts below come from.

One caution about the manifests: a key-up is stamped with the last level its host
announced, and on a gear change that announcement can be one over ahead of the
waveform. Three of the release tape's key-ups stamped rec0 are 232 and 236 blocks
of 1024 long, which is a rec1 over. The emission length settles it.

Geometry, from three clean overs of each. The symbol length is the analysis window
that concentrates a column's whole energy in one bin, at the record's own comb:

| | rec0 | rec1 |
|---|---|---|
| host level | `BITRATE (1)` | `BITRATE (2)` |
| symbol length, one-bin energy fraction | **2048**, 1.000 (0.773 at 512, 0.705 at 1024) | **1024**, 1.000 (0.710 at 512, 0.274 at 2048) |
| bins occupied | 33..96 | 17..48 |
| preamble symbols | 2 | 4 |
| emission columns | 124 | 228 |
| over length | 258 048 samples (5.376 s) | 237 568 (4.951 s) |

The column count is fixed twice over, and the two agree: the last emitted column
ends the burst exactly (`last lit block = preamble + ncols − 1`, no tail), and each
record's PRNG gap holds exactly `ncols + 24` draws (below).

**The bin table is not the base level's stream requantised.** That is the one field
BW2750 changes, so it was the hypothesis: record 3's 395 draws read to 64 bins from
bin 33 (rec0) or 32 bins from bin 17 (rec1). It fails. With `bpc = 3` and
`stride = 2` a data column's offset `(bin − first_bin − alloc) mod span` can only be
`2·gray`, `gray ∈ 0..7`, so a correct allocation confines every column to a narrow
window whatever the payload — and record 3's stream puts 42 of 124 rec0 columns
there against a null of 36 (chance 37), and 140 of 228 rec1 columns against a null
of 139 (chance 135). Nothing.

**Each record's own PRNG gap is the stream, and it closes.** Continuing the
interleaver stream past a record's two permutations, the next draws land on the
*following* record's stage-1 seed after exactly `ncols + 24`:

| record | gap after | draws to the next seed | = ncols + 24 |
|---|---|---|---|
| rec0 | `perm(300, 12600710)`, `perm(96, ·)` | **148** | 124 + 24 |
| rec1 | `perm(612, 7910701)`, `perm(200, ·)` | **252** | 228 + 24 |
| rec2 | `perm(816, 14955604)`, `perm(402, ·)` | **252** | 228 + 24 |
| rec3 | `perm(1484, 9947396)`, `perm(736, ·)` | **419** | 395 + 24 |

**But the stream is not the table, at any of the three records below the base.**
`first_bin + Int(Rnd*span)` off a record's own gap lands within **−1..+3** of the
true bin and no closer. The displacement is a property of the family, measured on
each record's own table:

| record | −1 | 0 | +1 | +2 | +3 | over |
|---|---:|---:|---:|---:|---:|---|
| rec0 | 11 | 21 | 36 | 25 | 7 | 100 data columns |
| rec1 | 3 | 56 | 83 | 50 | 12 | 204 |
| rec2 | 9 | 57 | 77 | 48 | 13 | 204 |

What displaces it is not a function of the draw, of the column or of the cell
value, and no `Int(Rnd*n)` stream anywhere in the 2**24 orbit reproduces it. So
these tables stand as read off the air.

**The tables are measured, and they need no plaintext to be read.** A data column
lights `first_bin + (base + stride·gray) mod span` with `gray` under `2**bpc`, so
one over admits `2**bpc` bins of the comb and the true base is in every over's
set. Records 0 and 1 spend 8 values at stride 2 on combs of 64 and 32 bins —
16 of 64 and 16 of 32 — so each further over of a different payload cuts what
survives and the intersection closes on its own. That is what record 2 could not
do: 16 values at stride 2 on 32 bins is the whole comb of one parity, and its
three tape frames leave 0 of 228 columns unique, which is why that record had to
be read from its payload and these two need not be.

The reference columns are the ones every over of a different payload agrees on.
Requiring *all* of them to agree loses one the moment a single over's argmax slips
a bin, so the count of overs holding the modal bin is the test: over eleven
record-1 overs the 24 score 11 against a next-best 6 and a median of 3, and over
eleven record-0 overs 11 against 8, median 3. **rec1's 24 are rec2's, position for
position**, `(104k+93)//11`; **rec0's** are `(41k+33)//8` on its 124 columns —
`4, 9, 14, 19, 24, 29, 34, 40, 45, 50, 55, 60, 65, 70, 75, 81, 86, 91, 96, 101,
106, 111, 116, 122`.

With the layout fixed, the frames fall out of the counter payload the responder
was pushing, and every over gives the same table: nine record-1 frames and ten
record-0 frames, identical column for column. Scored the way the base level's was
— the table and its frame against the tape — each over reads **228 of 228** and
**124 of 124** at one alignment and chance at every other, nulls 14 to 26 and 4 to
45, with 24 of 24 on the reference columns the frame cannot flatter. 14 of 15
record-1 overs and 49 of 51 record-0 overs decode to a clean CRC.

Confirmed on a second run of the same night that neither table was measured from:
7 of 8 record-1 overs and 50 of 53 record-0 overs at full score, all of them 24 of
24 on the reference columns, and the layout recovered from that capture alone is
the same one.

#### Base level — column order, blocks and amplitude

Emission columns are consumed **in order**, one column per `dw50`-length OFDM
symbol. The column counter starts at minus the record's reference-column count and
advances by one per symbol; a block boundary falls at each multiple of the
record's symbols/block, so an over is 4 blocks.

For rec0/rec1/rec2 the emitted data-column count is `coded/bpc = 4·symbols`
exactly — **100, 204, 204** — and `n_data_cols·dw50` = 204800, 208896, 208896
against the 210217-sample over: one data column per `dw50` block, sequentially,
`4·symbols` blocks filling the over. For rec3, `4·102 = 408` exceeds its 371 data
columns by the ~37 guard and reference columns; how those are accounted per block
is not specified here.

Two grids carry the emission: the internal one-hot column grid, indexed
`[(col−1) + bin·940]`, and the audio symbol grid, indexed `[(sym−1) + bin·100]`.
Against a 2048-sample frame a column belongs to frame `(col−1) // (2048/dw50)` —
one column per frame for rec0, two for rec1/rec2, four for rec3.

**Amplitude.** Each column's `dw50` samples are scaled by
`(1/1.41)·10^(agc(3))`; both constants are identical for all four base records,
which is the uniform per-cell amplitude **97.73**. The base DATA path applies no
spread and no overlap-add.

In the 2048-point receive domain, `|audio_cell| ≈ 61.081·|LUT[idx]|` per copy, so
`97.73 = 61.081·1.6` for the LUT magnitude 1.6, and a 0.82 LUT point gives
`0.82·61.08 = 50.1` against 50.3 observed. 61.081 is a composite
transmit/ADC/2048-point-FFT measurement gain, not a constant of the protocol.

**AGC.** A per-burst normalisation runs over a magnitude half-spectrum buffer,
f32, indexed `[bin + col·513]`, dimensions 513 × 469, where `513 = 1024/2 + 1`:
accumulate a level, form `norm = level/(2^e − 1.0)`, then divide per bin. It
carries no payload.

#### High-throughput levels rec9–rec16

Records 9–16 are the high-throughput DATA levels, emitted by the constellation
modulator, whose value law is a **dense constellation** rather than the base index
modulation. Link setup does not use them: the caller-identity burst is an ordinary
rec3 burst — the 175 bps level of the speed ladder above — and not a record of its
own.

| Stage | Law |
|---|---|
| bits → coded (Gray) | `v` = `bpc` on-air bits; `coded = clsparm_gray[(bpc−2)·16 + v]` (rows 0/1/2 = standard **2/3/4-bit Gray**) |
| coded → constellation | `cell = LUT[pos + 6·coded]`, `pos` = the record's constellation column |
| twiddle | `cell ·= twiddle[469·bin + (sym − block_start)]` = `exp(j·2π·S/128)` |
| placement | `grid[(sym−1) + 100·bin] = cell` (identity in sym, bin) |
| OFDM symbol | `dw50 = 1152` = **1024-pt IFFT + 128-sample cyclic prefix** |

**Constellation ladder** (LUT column per `pos`; the LUT interleaves six standard
constellations with period 6):

- `pos=1` **QPSK** (bpc 2): `(1,0)(1,90)(1,−90)(1,180)` (`(‖z‖, deg)`, coded order)
- `pos=2` **8-PSK** (bpc 3): unit circle, Gray order
- `pos=3` **16-QAM square** (bpc 4): corner modulus 1.1, moduli `{0.367, 0.82, 1.1}`
- `pos=4` **32-QAM cross** (bpc 5): corner modulus 1.5 — the 6×6 grid of odd
  multiples of `u = 0.3/√2` minus the four `(±5,±5)` corners; mean power 0.900,
  `d_min = 0.424264`, moduli `{0.3, 0.671, 0.9, 1.082, 1.237}`
- `pos=5` **64-QAM square** (bpc 6): corner modulus 1.6 (selected by no BW2300
  record)

Each column is a complete `2^bpc` alphabet; the 512-byte LUT is a
**truncated prefix** of the raster (column `pos` present only up to
`coded = ⌊(63−pos)/6⌋`), which is why an earlier revision read the `pos ≥ 3`
columns as sparse "APSK rings" with a bpc of 3. `bpc` is **4** at rec14 and **5**
at rec15/rec16; the divisibility of each record's coded-bit count by its column
count settles it, and the full columns, the QAM label law and the marking of the
pos=4 labels inferred beyond the table are in
[`06 §6.1c/§6.1e`](06-speed-gearshift.md) and the constant-table generator
`hfmodem/kestrel/rx/tablegen.py`.

**Coding chain** — same (13,15) turbo core throughout, interleaver column = record
index, whitened with the shared PN:

`rec9` QPSK r1/2 (N=7197, coded=14406) · `rec10` QPSK r2/3 · `rec11` QPSK r4/5 ·
`rec12` 8-PSK r2/3 (N=14398) · `rec13` 8-PSK r4/5 · `rec14` **16-QAM** r5/6 ·
`rec15` **32-QAM cross** r4/5 · `rec16` **32-QAM cross** r4/5. Rates follow from `N` and
`coded` (`2N+12` = r1/2, and so on).

> **Implementation note — this ladder is not validated against VARA.** No level
> above the base has been demodulated from a VARA transmission or accepted by a
> VARA receiver. What has been shown is that the chain above is *invertible*
> against this repository's own encoder, which is not the same as matching VARA:
> payload (897 B) + CRC-16 → whiten → turbo r1/2 (turbo interleaver column 9) →
> channel interleave (column 9) → 2-bit Gray → QPSK `LUT[1 + 6·coded]` → 7203
> cells → 226 OFDM symbols (1024-point IFFT + 128-sample CP) → audio → receive
> (strip CP, 1024-point FFT, nearest point, de-interleave, turbo, de-whiten) →
> CRC-clean, payload byte-identical. The per-cell `(sym, bin)` carrier schedule is
> derived rather than observed, and the round trip above substitutes a
> self-consistent sequential grid for it. See [`06 §6.1c`](06-speed-gearshift.md).

#### Cell raster, twiddle and overlap-add (records 4–16)

The constellation modulator lays modulated cells onto an internal
`128 (bin) × 940 (col)` complex grid, then applies a position-keyed twiddle. The
raster constants (**940**, **469**, twiddle divisor **128**) are geometry shared
with the 500 Hz mode; only the tables differ by bandwidth. `bin` is the **slow**
axis (stride 940), `col` the **fast** axis (0..939). **469 is the twiddle-grid
stride; 940 is the placement-grid stride** — they are different grids.

| Quantity | Value |
|----------|-------|
| Forward (TX) cell placement | `grid[(col−1) + 940·bin] = constellationLUT[sym] ⊗ twiddle[469·bin + col_rel]` |
| Grid axes | **bin = slow (stride 940), col = fast (0..939)**; the grid reshapes `(128, 940)` with `row = bin` |
| Twiddle / de-rotation index | `idx(bin,col) = 469·bin + col_rel`, `col_rel = col − block_start` (per emission block: block A starts at col 2; block B starts one past the block-A end) |
| Twiddle definition | `twiddle[k] = exp(j·2π·S[k]/128)`, `S[k]` a fixed per-position phase sequence (`twiddle_phase_S.u8`); the RX de-rotation reference is its conjugate partner on the same raster |
| Reference-sequence table | `128 × 469` unit-modulus, phases quantized to `2π/128`; values in `derotation_reference.c64` (use table values directly — not index-aligned to `exp(±j2π·S/128)`) |
| Twiddle grid | per-burst complex64 grid, `grid[469·bin + col] = exp(j·2π·m/128)`; **pure unit-modulus** (60032 cells all \|z\|=1.0000, phase/(2π/128) residual RMS **1.8e‑6**) |
| Twiddle grid shape/stride | **[128 bin][469 col]**, row-major stride **469** (60032 = 128·469 = 480256 B / 8) |
| Twiddle divisor | **128** exactly — `θ = m·2π/128` |
| Phase index `m` | `m = S[k]` (uint8), from a 65000-byte table read as a **flat stream** by a running counter, the grid filled in raster emission order; a conjugate-mirror column is filled alongside. The table's values span 0..254; the phase indices realised in the grid are `m ∈ {0..127}` |

`derotation_reference.c64` is exactly `exp(j2π·n/128)`, `n∈ℤ`
(phase·128/2π fractional RMS = 1.8e-6).

**RX de-rotation:** `cell ⊗ reference[469·bin + col_rel]`, then nearest-point
demap against the 64-entry constellation LUT. The forward chain (index law above +
interleave/whiten/turbo/CRC) is specified in [`03 §3.5.3`](03-coding.md).

**Overlap-add.** This path applies a raised-cosine overlap-add per output symbol.
The window is `w1[i] = (cos θ + 1)·0.5` with `w2` its complementary half, and a
previous-symbol state buffer supplies the overlap memory; the per-symbol blend is
`out[k] = w2[k]·cur[k] + w1[k]·prev[k]`. Neither window applies on the base DATA
path.

The BW2300 coding assets (interleave map, used-carrier map, whitening PN,
twiddle-phase sequence `S`) are **bandwidth-dependent** and differ from the 500 Hz
set; the constellation LUT and the reference sequence are bandwidth-independent.
The numeric constant tables are produced in code by the companion
implementation's generator, `hfmodem/kestrel/rx/tablegen.py`, which records how
each is known; they are not shipped as data files. Full coding-chain facts are
in [`03 §3.5.4`](03-coding.md).

For these levels the per-carrier modulation follows the constellation ladder
above — unit-circle PSK at pos ≤ 2 and non-uniform-magnitude QAM at pos ≥ 3
(outermost moduli 1.1 / 1.5 / 1.6), not plain unit-circle QPSK — whitened by a
unit-modulus reference sequence and a per-cell twiddle `exp(j·2π·S/128)`. Records
9–16 use `span = 32` and `dw50 = 1152` with a 128-sample cyclic prefix.

#### Reading the 2300 Hz waveform through a 2048-point window

A 2048-point analysis FFT does not match the base level's 512-sample symbol, and
the mismatch produces a consistent set of apparent structures that are not
physical content. Analysed at 2048 the base over presents as 65 contiguous
carriers in bins 34..98 at 23.4375 Hz spacing, a 2048-point IFFT, a ~2020–2029
sample stride (~23.7 baud), 100 symbols per link-setup burst, WOLA pulse shaping,
and a four-layer multi-rate sum of records rec0..rec3.

The cause is that a rec3 512-sample symbol fills one quarter of the window: its
single tone leaks a sinc skirt into neighbouring bins, and four *different*
consecutive columns land in one window. That is what produces the apparent
"65 contiguous carriers in four mod-4 classes", the apparent pilots, and the
469-stride twiddle raster. Synthesising rec3 alone and analysing it the same way
reproduces the per-class magnitudes at k0 = 0.96, k1 = 0.95, k2 = 0.97, k3 = 0.95.

Concretely, in that window:

- `k≡0` (bins 34,38,…,98; 17 carriers) appears as continuous real BPSK pilots
  (purely real ±1, raw M4=M2=1.00, channel phase ≈ 0);
- `k≡2` (16 carriers) appears ~77 % null or sparse;
- `k≡1,3` (32 carriers) appears to be the primary data, on a varied-modulus
  constellation with ~29 % null cells — matching the 19 zero-padding entries of
  the truncated 64-entry LUT — and an 8-PSK phase family;
- the data-symbol count reads as ~99–102 per over, with symbol 0 a low-energy guard;
- `k≡2` cell magnitudes quantise to `{0, 97.73, 195.5, 293.2}` = `N·97.73`,
  `N ∈ {0,1,2,3}` — the count of 512-sample columns landing on that
  (symbol, carrier) — and every data symbol has `Σ N = 4`, i.e. exactly 4 lit
  cells per symbol over 100 symbols, accounting for the 395 columns with
  collisions (335 single, 31 double, 1 triple);
- observed per-class magnitude maxima are `k≡2` ≈ 293 > `k≡0` ≈ 248 > `k≡1,3`
  ≈ 165;
- pilots read perfectly real after removing the per-symbol common phase with
  `est = ½·angle(Σ_pilots e^{2jθ})` (`|imag|/|z| = 0.0000`), and every `k≡2` lit
  cell is exactly `−1j` (phase −π/2, std 0.00).

None of the above is separate physical content in a base-level over. A 512-point
FFT per block recovers one column per block in time order directly.

### 2750 Hz mode

The 2750 Hz mode is specified here at the base level and only partly above it.

**The base level is fixed.** Host `BITRATE (4)`, and it is the 2300 Hz base level's
index law on a wider comb — nothing else about it differs:

| Parameter | BW2300 base | **BW2750 base** |
|-----------|-------------|-----------------|
| Samples per symbol `dw50` / cyclic prefix | 512 / none | **512 / none** |
| One-hot comb | 16 bins at 9..24 (843.75–2250 Hz) | **20 bins at 6..25 (562.5–2343.75 Hz)**, 93.75 Hz spacing |
| Emission columns / reference columns | 395 / 24 | **395 / 24, the same positions** |
| Turbo info / coded bits, frame bytes, bits per column | 736 / 1484, 92, 4 | **the same** |
| Bin-allocation table | `9 + Int(Rnd*16)`, 395 draws | **`6 + Int(Rnd*20)` — the same 395 draws of the same stream** |
| Reference-cell class map / channel + turbo interleavers / whitener | rec3's | **rec3's, unchanged** |
| Training preamble | 12 symbols, `Int(Rnd*16)` offsetting the record's own columns | **the same law on the 2750 columns, reduced into the 20-bin span** |
| Preamble draws before the session's first over | 0 | **5** |
| Burst length | 4.363 s | **4.363 s** |

A 20-bin comb carries 16 values, so **four bins of every column carry nothing**; a
demodulator that weighs all 20 against each other reads noise into every column.

How it is known: the one loopback BW2750 capture (2026-07-21) holds a single
wideband OFDM burst, the link-setup, whose 92 bytes are known in advance. Solving
each data column's allocation against them puts all **395 of 395** in the two-value
window a *shared* uniform deviate quantised to 20 instead of 16 allows, and nothing
else; the 24 reference columns pin **24/24** with no plaintext at all; the best any
other frame alignment on the same audio reaches is 31 of 371 in one residual class.
The preamble's 12 implied draws are `Int(Rnd*16)` — **not** `Int(Rnd*20)` — 12/12,
and they begin five draws into the stream rather than at its head.
The same tape holds an unlogged BW2300 session 50 s later, same modems and
callsigns, that the 16-bin record reads and the 20-bin one does not — and whose
own link-setup preamble begins at draw 0, so the five are the bandwidth's lead
and not the history of a process that had already keyed a session. One
observation each. Off air, seven
overs from three Winlink gateways that VARA itself logged as `CONNECTED … 2750`
(NS0A, KC9GHZ, KO2F, 2026-07-24) decode CRC-clean at this record and at no other,
carrying continuous B2F greeting text that was not known in advance.

**BW2750's gear-down record is not specified.** BW2300's rec2 bin table is measured
rather than generated, so there is no stream to re-quantise, and no recording held
here carries a BW2750 one.

Above the base level:

| Parameter | Value | Unit |
|-----------|-------|------|
| Occupied bandwidth | **~2.7 k** (2742 / 2713, ±30) | Hz |
| High-SL waveform has a cyclic prefix | **YES** (non-coherent block autocorrelation, peak/median 1.5–1.9×) | — |
| High-SL CP repetition period | **Not determined.** A control-validated correlator puts the peak anywhere in 1258–1394 depending on window, at a metric of 0.5–0.63 against 0.975 on a known-geometry synthetic. Consistent with ≈1280, not resolved to it; see [`06 §6.1b`](06-speed-gearshift.md). Observed near 1280 (26.7 ms), centre stable 1266–1284 | samples |
| High-SL symbol stride `Ts` | ≈ 1400–1450 (~33–34 baud) — **not** on the 23/42/47/94 grid | samples |
| Base / low-SL layer | **same no-CP structure as BW2300** (autocorrelation null at 1024/2048, no CP peak) | — |
| Useful symbol `Tu` | **Not determined** — the split between useful symbol and CP is unresolved. Observed near 1280 (37.5 Hz grid), favoured over the `Tu=1024, CP≈256` reading. Replicated across two stations, two bands and two receivers: two further long bursts score the CP correlation highest at lag 1280 (1.40 and 1.33 over a wide-baseline floor) against 1.12 and 1.03 at lag 1024. **Not a power of two — do not assume a 1024/2048 IFFT.** 1280 = 256·5 is a valid mixed-radix size | samples / Hz |
| CP length and exact `Ts` | **Not determined.** The carrier-comb test that would separate them — symbol-synchronous FFT accumulation scored against a random-offset baseline — scores 4.2× on a synthetic control and 1.0×, no gain at all, on every observed burst, even after removing the fractional CFO. `Ts` and the CP length are therefore not fixed here; only `Tu` is | samples |
| Modulation order / constellation / FEC rate / level index | Not specified here | — |

**One falsification.** BW2300's high-SL symbol is
`dw50 = 1152` (1024 + 128 CP). BW2750's is **≈ 1280 — about 11 % longer — so
`dw50 = 1152` does not carry over to 2750.** Anything that assumes it does is
wrong.

`Tu ≈ 1280` rests on two stations, two receivers and two bands, measured by a
method that recovers the geometry of a synthesised BW2300 rec9 burst exactly.
Everything else in this block rests on a single observation. Two properties of the
measurement matter to anyone repeating it:

- The CP correlation peak is *triangular with half-width equal to the CP length.*
  Normalising it against a nearby lag divides the peak out and makes a real CP
  look absent — the floor has to come from several hundred samples away.
- The discrimination margin is ~1.3×, against 1.6× on the control. It is
  directionally consistent across bursts, but it is not a margin that settles the
  question. A `rec` table from a BW2750 session would end it outright.

> **§1.2–§1.5 below describe the 500 Hz waveform as seen through a 256-point
> analysis window.** That model locks and produces coherent constellations but
> never recovers a frame; §1.1 gives the geometry that does. The model that
> decodes real audio is §1.1 plus [`03 §3.6`](03-coding.md). The 256-point view is
> retained here because its numbers are self-consistent and will be re-derived by
> anyone who analyses this waveform at that resolution.

## 1.2 Pilot / reference subcarriers

**k=0 (1500 Hz, FFT bin 8) is a continuous centre carrier**, present in every
DATA symbol. It is a usable common-phase / CFO / timing reference — the demod
derotates all carriers by `angle(k0)` — **but it is not an independent
per-carrier channel pilot**: the three inner tones (k=−1, 0, +1) are mutually
**phase-locked** (fixed relative phases; `k=+1·conj(k0)` collapses to a single
point at a fixed angle across all payloads/sessions). So k0 tracks the same
common phase the edge tones carry; it does not give a second, independent look at
the channel. A differential scan on the 23.4 Hz grid finds no continuous pilot at
all; that grid is the wrong one.

| Attribute                         | Value |
|-----------------------------------|-------|
| Pilot subcarrier index            | k=0 (1500 Hz, bin 8), continuous every symbol |
| Pilot value / sequence            | constant-modulus; phase-locked to the active edge tone (fixed offset ≈ π) — not an independent reference |
| Pilot insertion rate (time/freq)  | continuous in time on one fixed carrier |
| Purpose (channel est. / AFC / …)  | common-phase / CFO / symbol-timing reference (AFC) |

## 1.3 Per-carrier modulation, by speed level

Modulation order can vary per speed level (see [`06`](06-speed-gearshift.md)).
One row per speed level within the 500 Hz mode. The link characterised here ran at
a **low speed level** (CONNECTED …500), on a ladder reaching ~88 bps.

**The alphabet is PSK-family, not QAM.** A multi-amplitude reading of this
waveform is an artifact of demodulating an **alternating-tone** structure without
recognising it: the two band-edge carriers k=−1 and k=+1 **alternate
one-per-symbol** (only one is strongly active in a given symbol; the other is ~0),
so pooling one carrier's symbols mixes an "active" ring (|·|≈1) with an "idle"
blob (|·|≈0) and looks like two amplitude rings. When the active tone is isolated
per symbol its **amplitude is constant** — constant-modulus EVM ≈ **0.063**
(±6 %, 16/16 bursts). k=±2,±3 sit 16–20 dB down and read as noise-like at this
SNR.

Structure at this resolution, one **OFDM symbol** = 256 samples, 187.5 baud:

- k=0: continuous phase-reference carrier (every symbol).
- k=−1 / k=+1: **alternate** as the active data-bearing tone (frequency
  interleaving — consistent with dodging inter-symbol/adjacent-carrier
  interference given there is no guard interval). Constant modulus.
- The residual data phase of the active tone *after* common-phase (k0) removal is
  small and does not resolve into clean discrete QPSK/BPSK clusters under blind
  equalisation. The evidence weakly favours a low-order **differential** PSK,
  possibly with a scrambler; the exact order and mapping are not fixed here.

| Speed level | Modulation | Bits/carrier/symbol | Gray/mapping |
|-------------|------------|---------------------|--------------|
| low ("500") | **constant-modulus PSK** on an alternating band-edge tone + continuous k0 reference (not QAM; exact PSK order, differential form and scrambler not fixed here) | not fixed here; ≤2 likely | not fixed here |

## 1.4 Items the 256-point model leaves open

- **Exact PSK order / mapping of the active tone.** Under this model an
  exhaustive probe finds the data **not recoverable as per-carrier phase** for any
  variant: no FFT size (256, or **512 which resolves the 93.75 Hz sub-grid
  orthogonally**), symbol stride, timing offset, or CFO collapses any carrier's
  phase into a discrete BPSK/QPSK/8PSK alphabet — the phase-resultant
  `|E[e^{jMθ}]|` stays **≤0.13** for M=2,4,8 under a full ±half-baud CFO search,
  where the same extractor scores **1.00** and recovers the exact CFO on synthetic
  constant-modulus QPSK. Differential phase (same-side m↔m−2 and across-hop) is
  likewise non-discrete (≤0.06). The data is **not** in tone frequency (per-symbol
  frequency is unimodal, not M-FSK) **nor** in active-tone amplitude (only a
  burst-common framing envelope reproduces; cross-payload envelope correlation
  0.90 against same-payload 0.73). The per-symbol waveform is a **rigid phasor**
  (real vector after one global-phase removal): ~92 % in one component plus a
  ~8 %, ~4-D perturbation subspace whose energy grows with payload entropy
  (zeros < counter < prbs9 < prbs15) — the payload sitting in a low-index,
  low-power perturbation, and/or a phase-whitened constellation. That negative is
  exhaustive over a model that is itself wrong: [`03 §3.6`](03-coding.md) decodes
  this waveform byte-exact at H=512, one of the sizes the probe names as ruled
  out. Re-aligning the off-grid tone on-grid does not fix the model either — the
  tone is indeed off-grid at this resolution, but on-grid re-alignment still
  yields uniform phase.
- **Whether k=±2,±3 carry data** in higher-throughput bursts; they are 16–20 dB
  down and read as noise at this speed.
- **Absence of a cyclic prefix is a positive result** — the demod locks with a
  full-symbol FFT and no CP strip, and gives constant-modulus tones.
- **Native sample rate (12 kHz) and absolute FFT size (64 vs 256)** rest on the ÷4
  inference. The demod runs at 48 k / 256-pt regardless, and the Hz facts (187.5
  spacing, 1500 centre, 5.333 ms symbol) do not depend on it.

## 1.5 Two burst populations (DATA vs CONTROL)

Every 500 Hz session carries **two distinct waveforms**:

- **DATA bursts** — long (≈4.35 s ≈785 sym, or ≈8.53 s ≈1570 sym), sent by the
  initiator. The 256/187.5/no-CP alternating-tone OFDM characterised above.
- **CONTROL / ACK bursts** — short (≈0.34–0.68 s, ~65–130 sym), sent by the
  responder between DATA bursts (≈85 ms turnaround). Same ~1150–1850 Hz band and
  256-sample symbol grid, but a **different tone structure**: the centre carrier
  (1500 Hz, bin 8) dominates and varies while bins 7/9 sit lower and roughly
  equal — i.e. **no** band-edge alternation. These carry the ARQ handshake and
  acquisition preamble. This document characterises them only briefly and does not
  specify their demodulation.

| Attribute | DATA | CONTROL |
|-----------|------|---------|
| Duration | 4.35 / 8.53 s | 0.34–0.68 s |
| Tone structure | k=±1 alternate + k0 pilot | k0 (1500 Hz) dominant, no alternation |
