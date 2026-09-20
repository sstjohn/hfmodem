# 06 — Speed level & gear-shift logic

VARA adapts its throughput by stepping a **speed level** up and down. This
section specifies the level ladder for each bandwidth, the per-level physical
descriptor, and the up-/down-shift policy; it builds on the per-level physical
and coding detail in [`01`](01-physical-layer.md) and [`03`](03-coding.md).

The level index and its rate are host-visible as `BITRATE (N) x bps TX` (the
level this station will use for its next over) and `BITRATE (N) x bps RX` (the
level of the frame just decoded). Registration gates the top of the ladder: an
unregistered build caps at `BITRATE (4)` and reports `LINK UNREGISTERED`.

## 6.1 Speed-level ladder (500 Hz)

The BW500 levels below are those of **VARA HF 4.9.0**.

| Level | Rate (`BITRATE (N)`) | Net payload / block | Modulation (→[01](01-physical-layer.md)) | FEC rate (→[03](03-coding.md)) |
|-------|----------------------|---------------------|------------------------------------------|--------------------------------|
| 0–2 (more robust) | (not fixed here) | | | |
| **3** | **61 bps** | **34 bytes** | (→01) | (→03) |
| **4** (unregistered cap) | **88 bps** | **43 bytes** | (→01) | (→03) |
| >4 (registered) | (gated by registration; not reachable on an unregistered build) | | | |

> Net payload per block is what the `BUFFER` ledger clears: each ACKed block
> clears exactly 43 raw bytes at L4 and 34 at L3. `BUFFER` counts raw host bytes
> irrespective of `COMPRESSION TEXT`; payload content does not change the figure
> (43 B/block at L4 for every payload). This document does not fix the mapping of
> a BW500 level index to its **modulation order + FEC rate** — that is a
> physical-layer matter for [`01`](01-physical-layer.md)/[`03`](03-coding.md),
> pending per-level DATA constellations.

## 6.1a Full mode ladder — bandwidths, level counts, peak rates

VARA HF has **exactly three bandwidths**; 2750 is the widest. Peak rates below
are *uncompressed user* rates at the top level of each band. Per-level
`(constellation, code-rate, carrier count)` detail is not part of the published
mode ladder and is not tabulated here
(→[`01`](01-physical-layer.md)/[`03`](03-coding.md)).

| Bandwidth | Name | Speed levels | Peak bps (top level) |
|---|---|---|---|
| 500 Hz | Narrow | 13 | **1543** |
| 2300 Hz | Standard | 16 | **7050** |
| **2750 Hz** | **Tactical** | **17** | **8489** |

- **Per-carrier constellation climbs with level:** FSK → BPSK → 4-PSK/8-PSK →
  16-QAM/32-QAM, per the shared constellation LUT (`constellation_lut_64.c64`,
  regenerable from `hfmodem/kestrel/rx/tablegen.py` — see §6.1c).
- **2750 ("Tactical") is free-selectable** (a client checkbox, no separate
  charge); the `VARAHF2750.dat` mode is present in every build. Registration
  ($69, lifetime, all VARA products) lifts a **speed cap only** (an unregistered
  build runs ~10 % of max ≈ level 2–3, ~175 bps).

  The **cap follows the registration, not the role.** A widely mirrored claim
  that gateway operation runs at full speed without a licence does not hold:
  with the answering station in the gateway role on BW2300, both endpoints
  report the same 175 bps ceiling. Unregistered operation tops out at 175 bps in
  every direction, every bandwidth, and both roles.

## 6.1b The widest, fastest mode

The top of VARA's mode space:

| Parameter | Value |
|---|---|
| Bandwidth | **2750 Hz (Tactical)**, widest |
| Speed level | **top (level 17)** |
| Nominal bps (uncompressed) | **≈ 8489** |
| Per-carrier constellation | **32-QAM cross** (corner modulus 1.5) — top of the ladder |
| Symbol grid | Per-level OFDM symbol = `dw50` samples @48 kHz, **plain IFFT / no WOLA**. BW2300's high levels use `dw50=1152` (1024 + **128-sample cyclic prefix**). **BW2750 does not reuse it**: a high-level BW2750 burst carries a cyclic-prefix repetition of **≈1280 samples (~11 % longer)**, so BW2750 has its own `rec` table — see [`01 §2750`](01-physical-layer.md). This document **does not fix** BW2750's high-level symbol split: the period is not 1152, and estimates of the useful length span `useful ≈ 1258–1394` |
| Payload compression | static order-0 Huffman ([`08`](08-compression.md)) |

> The encoder is **one level-parametric code path** — a single function for every
> level, selected by the per-level descriptor plus the shared constellation LUT,
> differing only in constellation size and code rate. The same law that holds at
> the low levels therefore drives the top gear. A registration only affects
> *emitting* max speed on air.

## 6.1c Level ladder from the per-level record table

> **Scope of §6.1c–§6.1e: derived, not observed.** No level above the base has
> been demodulated from a VARA transmission, and no transmission from this
> implementation above the base level has been accepted by a VARA receiver. The
> ladder below is closed by a round trip against **this implementation's own
> encoder**, which shows the chain is invertible — not that it matches VARA.
>
> What is interop-validated is the base level (rec3), both directions, against
> VARA and against off-air gateway audio. The carrier raster for rec9–16 is
> further a **self-consistent choice made by this implementation**
> (`kestrel/tx/varahf2300_tx.py`), not a recovered VARA parameter: this is not
> what VARA specifies, and it stands in until a high-level capture pins the real
> raster. A correct constellation and code rate alone will not produce an
> interoperable signal.

The per-level physical descriptor is the **232-byte (`0xe8`) speed-level record
table**. The table is per-bandwidth: a BW2300 session holds **BW2300's** ladder,
a BW2750 session loads a different table — of which one row is now known, its
**record 3**, `BITRATE (4)`: the row below with `span` 20 and the one-hot band at
bins 6..25, everything else identical, and the bin-allocation table the same 395
draws of the same stream quantised to 20 instead of 16
(→[`01 §2750`](01-physical-layer.md)). Fields used: `b00` = value-law selector
(2 = index-mod, 1 = constellation-mapped), `b01` = `pos+1` (LUT column),
`i32[0x50]` = `dw50` (OFDM symbol samples), `i32[0x54]` = cyclic-prefix samples,
`byte[0x88]/2` = span (carriers), `word[0x8e]` = block size (over =
`4·w8e·dw50` samples), `byte[0x11]` = frame bytes, `f32[0x40]/[0x48]` = AGC,
`f32[0x4c]` = per-bin twiddle. Records 17+ are terminator/garbage (`b00=255`).

| rec | value law | dw50 (CP) | span | pos | constellation | frame B | over s | note |
|----|-----------|-----------|------|-----|---------------|---------|--------|------|
| 0 | **index-mod −1j** | 2048 (0) | 64 | — | one-hot bin (LUT overflows) | 12 | 4.27 | robust DATA level |
| 1 | index-mod −1j | 1024 (0) | 32 | — | one-hot bin | 25 | 4.35 | |
| **2** | **index-mod −1j** | **1024 (0)** | **32** | — | **one-hot bin (4-bit, stride 2)** | **50** | **5.03** | **BITRATE-3; 228 columns, bins 17–48, validated against VARA audio (→below)** |
| **3** | **index-mod −1j** | **512 (0)** | **16** | — | **one-hot bin (4-bit)** | **92** | **4.35** | **BITRATE-4; base level, validated against VARA audio (→01)** |
| 4–8 | LUT constellation | 512 (0) | 16 | 1 | **QPSK (unit)** | 135–214 | 4.31 | `b95=1` = LINK-SETUP bursts |
| 9–11 | LUT constellation | **1152 (128 CP)** | 32 | 1 | **QPSK** | 224–239 | 4.32 | high-throughput DATA |
| 12–13 | LUT constellation | 1152 (128 CP) | 32 | 2 | **8-PSK (unit circle)** | 224–239 | 4.32 | high-throughput DATA |
| 14 | LUT constellation | 1152 (128 CP) | 32 | 3 | **16-QAM (corner 1.1)** | 246 | 4.32 | high-throughput DATA |
| 15–16 | LUT constellation | 1152 (128 CP) | 32 | 4 | **32-QAM cross (corner 1.5)** | 213–250 | 4.32 | top DATA levels |

**The value law splits into two families**, and the constellation law for the top
levels is separate from the base:

- **Base DATA levels (rec0–3, `b00=2`): index / position modulation, value
  `= −1j` one-hot.** The coded value picks the *bin*
  (`bin=((alloc+stride·gray−first_bin)%span)+first_bin`), not a constellation
  point; `LUT[pos+6·coded]` **overflows** its 64 entries at these bpc, which is
  exactly why the index-mod branch applies. No twiddle (`f4c=0`), no CP
  (`dw54=0`). rec3 is byte-exact against VARA audio
  (→[`01`](01-physical-layer.md)), and so, since 2026-08-25, is **rec2**.

  **rec2 closed (2026-08-25).** Host `BITRATE (3)`, 82 bps, the level a VARA pair
  drops to below base — both stations of the 2026-08-14 two-sided BW2300 bench
  session used it for the 37-byte tail of their 126-byte transfer. Measured off
  that recording: **228 emission columns** of 1024 samples (4.86 s of columns,
  5.03 s PTT-to-PTT with its 6 training symbols and lead-in), one lit bin in
  **17..48** of a 1024-FFT, **stride 2** so the 4-bit cell reaches 16 of the 32
  bins, **24 reference columns** at `(104k+93)//11`, and the same coding chain as
  rec3 on its own `il1`/`il2` columns (402 info, 816 coded, rate 1/2). The 228/24
  split is independently fixed by the interleaver stream: rec2's gap holds 252
  `Rnd` draws and closes on rec3's seed, and 204 data columns + 2×24 is the only
  split that spends it. Three recorded overs — two DATA tails and one link-setup
  — decode CRC-clean and reproduce bin-for-bin.

  **The 6 training symbols closed 2026-09-02**, off the same recording, which is
  what makes rec2 keyable and not only readable. The preamble runs 6144 samples at
  both records — twelve symbols at rec3, six at rec2 — under the base level's own
  bin law at rec2's geometry, and its allocations are the record's own bin table
  read from index 1, so it costs no table of its own
  (→[`01`](01-physical-layer.md)). One whole over rendered that way correlates
  0.99997 against the tape, its preamble alone 1.000000. Nothing keys it yet: the
  VARA station engine sends every over at base, and the gear-shift that would use
  this is a separate decision.

  What is still **not** closed at rec2: the bin table's generator.
  `17 + Int(Rnd*32)` off the same gap lands within −1..+3 of the measured bin on
  203 of 204 data columns and no arrangement of the VB6 stream closes the
  difference, so kestrel ships the measured table
  (→[`table-derivations §5a`](../table-derivations.md)).

  **rec1 and rec0 are closed, 2026-09-04**; **rec4–8 remain open**. rec1 has the
  same 228/24 shape by the same gap arithmetic (612 coded at 3 bpc = 204 cells,
  gap 252) and rec0 the same at 124 columns (300 coded at 3 bpc = 100 cells, gap
  148), and both tables are now measured off gear-down tapes — with no known
  plaintext, since 3 bpc at stride 2 leaves most of the comb unreachable from an
  allocation and overs of differing payloads intersect down to one bin a column.
  rec1's 24 reference columns are rec2's, position for position; rec0's are
  `(41k+33)//8`. Both take the rate-1/3 turbo branch (`3N+12`), and kestrel reads
  and renders them (→[`table-derivations §5a`](../table-derivations.md)).
  rec4–8's gaps are 36–41 draws, far too few for a per-column table, which is the
  structural reason those levels place their cells some other way.
- **High-throughput DATA levels (rec9–16, `b00=1`): a constellation alphabet,
  value `= LUT[pos + 6·coded]`, `pos = b01−1`.** These use a **different value
  law** from the base — a genuine amplitude+phase alphabet, plus a **128-sample
  cyclic prefix** (`dw50=1152=1024+128`) and a **per-carrier twiddle
  `f4c=0.125`**. The constellation columns of the LUT are six standard
  constellations, interleaved with period 6:
  - `pos=1` QPSK: `(1,0)(1,90)(1,−90)(1,180)` (`(magnitude, degrees)`, coded order)
  - `pos=2` 8-PSK: unit circle at `0,±45,±90,±135,180`
  - `pos=3` 16-QAM square, corner modulus 1.1 (moduli `{0.367, 0.82, 1.1}`)
  - `pos=4` **32-QAM cross**, corner modulus 1.5: the 6×6 grid of odd multiples
    of `u = 0.3/√2` per axis minus the four `(±5,±5)` corners — 32 points, mean
    power exactly 0.900, `d_min = 2u = 0.424264`, moduli
    `{0.3, 0.671, 0.9, 1.082, 1.237}`
  - `pos=5` 64-QAM square, corner modulus 1.6

  An earlier revision read these columns off the 512-byte LUT as
  complete tables ("3-ring APSK", "dense APSK, 10 points", "outer-ring APSK").
  The table is a **truncated prefix**: `LUT[pos + 6·coded]` runs off its 64
  entries at `coded ≥ ⌊(64−pos)/6⌋`, so pos=3 shows 11 of 16 points and pos=4
  only 10 of 32 — the "dense APSK" and its moduli list were artifacts of that
  truncation, and no sub-alphabet exists: all 32 pos=4 points are used. The full
  columns, the QAM label law (per-axis `clsparm_gray` Gray rows; pos=4's
  `coded = 16·sx + 8·sy + 4·my + mx` with the `mx=2` overflow folded onto the
  `|Q|=5u` arm, rows reversed) and the byte-exact regeneration of the table are in
  the constant-table generator `hfmodem/kestrel/rx/tablegen.py`
  (gate: `tests/gates/test_tablegen.py`). One caveat is inherited from the
  truncation: only quadrants 0 and 1 of the pos=4 column appear in the table, so
  22 of its 32 labels rest on an adopted independent-sign-bit assignment —
  chosen because every other column shows per-axis independence and it is the
  Gray-best candidate (1.154 mean bit-flips over the 52 nearest-neighbour pairs,
  against 1.269 and 1.577 for the competing quadrant assignments). A capture
  from a licensed peer at rec15 would settle it. Bits per cell for rec14/15/16
  are settled in
  [§6.1e](#61e-rec141516-bits-per-cell-and-the-forced-cell-schedule).

**The top levels are on-air-validatable.** An unregistered build's speed cap
limits only VARA *emitting* the top levels locally; the same decode arbiter works
against a **licensed remote VARA peer** — a licensed station emits the top level
and this implementation decodes it (RX); this implementation emits the top level
and the licensed VARA decodes it (TX). The top levels are therefore derived
**pending a licensed on-air partner**, not beyond validation.

### 6.1d High-throughput Gray map and coding rates

**Gray map**, from the constellation modulator and the Gray-permutation table
`clsparm`: on-air bits → constellation is `v` = `bpc` bits →
`coded = clsparm[(bpc−2)·16 + v]` → `cell = LUT[pos+6·coded]`, `pos=byte[+1]−1`.
`clsparm` rows are the standard Gray permutations: row0 (bpc2) `[0,1,3,2]`, row1
(bpc3) `[0,1,3,2,7,6,4,5]`, row2 (bpc4)
`[0,1,3,2,7,6,4,5,15,14,12,13,8,9,11,10]`. **This holds for rec9–13** (QPSK bpc2,
8-PSK bpc3). For `pos≥3` (rec14/15/16) `bpc` is **not** 3 — `coded` divisibility
rules it out (rec15 ⇒ bpc5/32-ary) — and those records carry forced/reference
cells (`word[+0x3e]=980/980/77`); see
[§6.1e](#61e-rec141516-bits-per-cell-and-the-forced-cell-schedule).

**Coding rate per high record** (`N=i32[0x20]` info, `coded=i32[0x1c]`; il column
= record index; whitening from the PN table; frame = `byte[0x11]` bytes incl.
CRC16):

| rec | constellation | rate | N info | coded | frame B |
|----|---------------|------|--------|-------|---------|
| 9  | QPSK  | 1/2 | 7197  | 14406 | 899 |
| 10 | QPSK  | 2/3 | 9596  | 14406 | 1199 |
| 11 | QPSK  | 4/5 | 11512 | 14406 | 1439 |
| 12 | 8-PSK | 2/3 | 14398 | 21609 | 1799 |
| 13 | 8-PSK | 4/5 | 17276 | 21609 | 2159 |
| 14 | 16-QAM (pos3) | 5/6 | 21710 | 26068 | 2713 |
| 15 | 32-QAM cross (pos4) | 4/5 | 26056 | 32585 | 3257 |
| 16 | 32-QAM cross (pos4) | 4/5 | 1704  | 2145  | 213 |

**Round trip through this implementation** (rec9 QPSK r1/2): payload+CRC16 →
whiten → turbo r1/2 → channel-interleave → 2-bit Gray → QPSK `LUT[1+6·coded]` →
7203 cells → 226 OFDM symbols (1024-IFFT + 128 CP) → audio → RX inverts →
**CRC-clean, payload byte-identical**. This is a working high-throughput modem
built on the law above (physical detail in [`01`](01-physical-layer.md)); the
round trip is self-consistent and is not by itself a demonstration of
interoperability with VARA.

**Reference-modem validation:** the unregistered VARA build limits the speed
levels available for transmit measurements. Bit-exactness
against VARA is validated against a licensed peer (VARA emits the top level and
this implementation decodes; this implementation emits and the licensed VARA
decodes). Two items remain open for VARA bit-exactness: the exact per-cell
`(sym,bin)` carrier raster and the `pos≥3` bpc confirmation — both resolvable
from one high-level on-air capture.

### 6.1e rec14/15/16 bits per cell and the forced-cell schedule

Data cells consume `bpc` on-air bits each, so a record's `coded` length must be
divisible by its true `bpc`. Checking each high record's `coded` against
candidate `bpc` **rules out `bpc = 3`** for the top three records:

| rec | pos | coded | `word[+0x3e]` (forced cells) | `coded` divisible by bpc | ⇒ bpc |
|----|----|-------|------|--------------------------|-------|
| 9–11 | 1 | 14406 | 0 | 2, 3 | **2** (QPSK, clean) |
| 12–13 | 2 | 21609 | 0 | 3 only | **3** (8-PSK, clean) |
| **14** | 3 | 26068 | **980** | 2, 4 (**not 3**) | **4 ⇒ 16-ary** (record byte `[+1]=4`; `26068 = 6517·4`) |
| **15** | 4 | 32585 | **980** | **5 only** (not 2,3,4) | **5 ⇒ 32-ary** (record byte `[+1]=5`) |
| **16** | 4 | 2145 | **77** | 3, 5 | **5 ⇒ 32-ary** (record byte `[+1]=5`; `2145 = 429·5`) |

- **rec9–13 are clean** (`word[+0x3e]=0`, `coded` divisible by bpc): QPSK bpc2
  (rec9–11), 8-PSK bpc3 (rec12/13) — the self-consistent round trip in §6.1d
  covers these.
- **rec15's `coded=32585` is divisible only by bpc=5 ⇒ a 32-ary constellation**,
  which matches the published top level ([§6.1b](#61b-the-widest-fastest-mode)).
  The top DATA level is therefore **not** an 8-point APSK — the `pos=4` column
  is the full 32-QAM cross (§6.1c).
- **bpc for all three comes from the record byte `[+1]` (= constellation order =
  `pos+1`), corroborated by `coded` divisibility and the ladder progression:**
  **rec14 = bpc 4 (16-ary)** (byte `[+1]=4`, `26068=6517·4`; the 8→16→32-ary
  rungs are rec12/13 → rec14 → rec15), **rec15 = bpc 5 (32-ary)**, **rec16 =
  bpc 5 (32-ary)** (byte `[+1]=5`, `2145=429·5`). Byte `[+1]` is trusted here
  because it equals the known bpc on every validated record (rec3=4, 16-ary,
  validated against VARA audio; rec0=3 8-ary; rec12/13=3 8-PSK).
- **rec14/15/16 are the only DATA/high records with forced/reference cells**
  (`word[+0x3e]` = 980/980/77 > 0). The forced-cell schedule for these lives in
  the same 17-column maps (`map1`/`map2`/`map26c`, seg columns 14/15/16). The
  **exact bit-loading** — how the 980/77 forced cells interleave with the data
  cells — follows the `word[+0x3e]>0` branch of the constellation modulator,
  which this document does not specify. (The alphabets themselves are settled:
  16-QAM at pos=3, the 32-QAM cross at pos=4 — see §6.1c.)

**What this section fixes, and what it does not.**

- **Fixed:** the constellation order of all three top records (rec14 = bpc 4 /
  16-QAM, rec15 = bpc 5 / 32-QAM cross, rec16 = bpc 5 / 32-QAM cross), the
  forced-cell **count** (`word[+0x3e]` = 980/980/77), and the **full-length
  channel + internal interleaver columns for every record**, including `il1` seg
  12/13/14/15 and `il2` seg 14/15, taken from the BW2300 mode `.dat` and verified
  as exact bijections (regenerable, with the constellation LUT, from
  `hfmodem/kestrel/rx/tablegen.py`). **No fallback interleaver is used on any
  level.**
- **Not fixed:** the exact bit-loading of the forced-cell records — how the
  980/77 forced/reference cells interleave with the data cells — plus the
  per-cell carrier raster, and the pos=4 quadrant-2/3 label assignment (adopted
  on Gray-optimality, not observed — §6.1c). An unregistered VARA build cannot
  emit these levels, so there is no local audio to bit-match; pinning the
  modulator's `word[+0x3e]>0` branch needs a high-level on-air capture from a
  licensed peer. rec9–13 are round-trip validated.

## 6.2 Up-shift policy (go faster)

| Attribute                             | Value |
|---------------------------------------|-------|
| Metric driving up-shift (SNR/BER/ACK) | Decode success (successful over) — `SN` reports high (>20 dB) around up-shifts; not directly exposed as a threshold |
| Threshold to up-shift                 | Not fixed by this document; pinning it needs a controlled SNR sweep |
| Hysteresis / dwell time               | Up-shift on the **next over after** a successful lower-level over (≥1 good over) |
| Step size (one level / jump)          | One level per shift (L3 → L4) |
| Throughput axis: blocks-per-over      | Separate from level: at fixed L4 the modem grows 1 → 2 blocks/over after several consecutive successes (over 4.31 s → 8.51 s) |

## 6.3 Down-shift policy (go more robust)

| Attribute                             | Value |
|---------------------------------------|-------|
| Metric driving down-shift             | Over failure: `BUFFER` fails to advance (block not ACKed) + overlapping/collision PTT (retransmit); `SN` had dipped (~16.6 dB) beforehand |
| Threshold to down-shift               | Not fixed by this document; pinning it needs a controlled SNR/error sweep |
| Trigger on consecutive NAKs?          | Triggered after a single failed/retransmitted over (L4 → L3); NAK not host-visible |
| Step size                             | One level (L4 → L3) |

## 6.4 Negotiation / signaling of level

| Attribute                                  | Value |
|--------------------------------------------|-------|
| How the level is signaled to the peer      | Carried in the DATA waveform: the RX side reports the decoded frame's level as `BITRATE (N) x bps RX` (matching the TX side's `… TX`), so the level is embedded per-over, not host-negotiated |
| Who decides (TX-driven / RX-requested)     | **TX-driven**: the sending (ISS) station announces `BITRATE (N) x bps TX` before keying and chooses the level; the IRS only reports what it decoded |
| Registration/gateway caps affecting levels | Unregistered build capped at `BITRATE (4)` (`LINK UNREGISTERED`); higher levels need registration |

## 6.5 Limits of this description

- **The gear-shift policy of §6.2/§6.3 is specified by direction only.** Down on
  failure, up on success is fixed; the **thresholds, the hysteresis, and the
  exact triggering metric are not**. Fixing them needs an AWGN/impairment sweep
  that forces repeatable shifts and reaches levels 0–2.
- **The BW500 level → (modulation, FEC-rate) mapping is empty here**, pending the
  per-level DATA demod ([`01`](01-physical-layer.md)/[`03`](03-coding.md)); only
  the host-visible `bps` and net-bytes/block are given.
- **Two adaptation axes** exist and must not be conflated: (a) the `BITRATE`
  speed level, (b) blocks-per-over. Only (a) is the gear shift; (b) is an
  efficiency knob at a fixed level.
- **The registered ladder above L4** is not characterised here; it needs a
  registered modem.
