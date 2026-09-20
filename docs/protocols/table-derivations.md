# Constant-table generation and verification

The receiver tables use deterministic generators, closed-form expressions and
measured waveform constants. This guide describes their definitions and checks.
Comparisons cover the active receiver range; incomplete measurements are marked.

Citations use two roots:

- `gen` = the generator: `hfmodem/kestrel/rx/tablegen.py` (VARA HF / kestrel) and
  `hfmodem/shrike/tablegen.py` (PACTOR / shrike).
- `gate` = `hfmodem/tests/gates/test_tablegen.py`, which regenerates each table
  and holds it against the bytes the receiver loads.

Strength is marked per fact:

- **[computed]** — regenerated from a published algorithm, byte-identical.
- **[closed form]** — a formula, no algorithm state, byte-identical.
- **[reconstructed]** — reproduced to the bit, including the required floating-point
  arithmetic.
- **[reduced]** — the shipped blob is larger than what the receiver reads; the
  read part is one small documented constant, and the boundary is asserted.
- **[designed]** — a filter of our own, computed at the point of use from a
  published waveform parameter. There is no byte claim because there are no
  bytes: nothing ships, and what the receiver runs is the design.
- **[measured]** — recovered by known-plaintext inversion of our own recordings,
  over the extent a burst emits; the rest is stated as underived.

---

## 1. Computed from a published algorithm — the VB6 `Rnd` stream

The whiteners and all four interleavers are draws from one generator: the
Visual Basic 6 runtime PRNG in `msvbvm60`. It is a 24-bit linear congruential
generator,

```
s  <-  (s * 0x43FD43FD + 0xC39EC3)  mod  2^24        (full period)
Rnd =  Single(s / 2^24)                              -- IEEE binary32
draw = Int(Rnd * n)                                  -- 0 .. n-1
```

The subtlety is the last line, and getting it wrong is invisible until it
desynchronises a table. `Rnd * n` is a **binary32** multiply. `s / 2^24` is exact
(a power-of-two divide of a 24-bit integer), but the product rounds to 24
significant bits, round-half-to-even, before `Int` floors it. Truncating the
exact product instead — the tempting `(s * n) >> 24` — disagrees with the runtime
about once per 100,000 draws, and each disagreement shifts every later value in
that column, desynchronising 9 of the 68 interleaver columns. `gen`'s `draw()`
does the binary32 multiply in exact integers. **[computed]**

- **Witness.** Turbo column 7, `n = 2104`, `s = 5613669`: shift-and-truncate
  gives 703, the binary32 multiply gives 704. `gate` pins both numbers.
- **The whiteners** (`gen.whitener`): one additive pre-FEC PN sequence, `bit =
  s >> 23`, seed 29318. The same 150,000-bit run whitens BW500 and BW2300; each
  consumes only a short prefix. `gate` matches the live 150,000-bit extent of
  both reference vectors; their unused tails are outside the receiver range.
- **The four interleavers** (`gen.interleaver`): each record is a rejection-sampled
  permutation whose stage-2 (turbo) draw stream *continues* from the state stage 1
  ends on, no gap. BW2300 uses a 32585×17 (stage 1) or 26056×17 (stage 2)
  array in row-major order. BW500 replaces each column's active prefix with its
  shorter permutation and retains BW2300 values in the remaining rows.
  The generator reproduces both the active entries and the inactive tail.


**The gaps between records were never gaps.** The interleaver seeds looked like a
list of unrelated magic numbers with 36-to-419-draw holes between the tables. They
are one unbroken stream. Record 3's interleaver ends at state 1746102; continuing
that same stream yields the record-3 base tables — 395 allocation draws followed by
24 class draws, 419 in all — and lands exactly on state **15597635**, which is
record 4's stage-1 seed. `gen.map2_col3` raises if the stream does not close on that
number; it closes. A reseed cannot land on its own successor's seed by luck, so this
is the proof the seeds are draws from one generator, not 35 independent constants.

---

## 2. Derived in closed form

No algorithm state — each is a formula in an index, byte-identical under `gate`.

- **`alloc_col3`** (`gen.alloc_col3`): per-column bin-allocation offset
  `9 + (state >> 20)`, the 395 draws taken from the record-3→4 gap above. **[closed form]**
- **`map2_col3`** (`gen.map2_col3`): the reference-cell class at each of the 24 pilot
  columns, `Int(Rnd * 8)` stored as `2 * class`, the 24 draws that follow the 395 and
  close on record 4's seed. **[closed form]**
- **`map1_col3`** (`gen.map1_col3`): a 1 at each pilot column, 0 elsewhere. The pilot
  columns are `(279k + 263) // 17` for `k = 0..23` — 24 pilots Bresenham-spaced over
  the 395 columns. No PRNG. **[closed form]**
- **`clsparm_gray`** (`gen.clsparm_gray`): a 2-D reflected Gray map,
  `4·gray(i>>2) + (gray(i&3) XOR 3·(gray(i>>2)&1))`, laid down for 2, 3 and 4 bits
  per cell. The plain 1-D Gray code `i ^ (i>>1)` does **not** match — it gives
  `[…,6,7,5,4]` where the table has `[…,7,6,4,5]`; the row-parity reflection is
  what reproduces it. **[closed form]**
- **`hdr_raster_src`** (PACTOR header, `shrike gen.hdr_raster_src`): the source
  symbol index for each of the 288 header cells is `52 + 563·(3 - i%4) + i//4` — four
  interleaved columns 563 apart, a 4-deep block interleave written as an affine map.
  **[closed form]**

### 2a. The constellation LUT — reconstructed to the bit

The shared 64-entry complex LUT (`gen.constellation_lut`, `cell = LUT[pos + 6·coded]`)
is six constellation columns interleaved with period 6: three PSK columns, a 4-QAM
square, a 32-QAM cross, and a 64-QAM square. The 64 entries are a truncated prefix
of that raster — the zeros in it are where a short PSK column has run out, not
constellation nulls. **[reconstructed]**

Byte-exact generation specifies both the geometry and floating-point arithmetic. Two constants carry the near-quadrant PSK residues onto exact float32 values: VARA's 15-significant-digit `Pi` literal (`3.14159265358979`, seven
ulp short of the true double), and the 66-bit constant the x87 `FSIN`/`FCOS` reduce
their argument against (0xC90FDAA22168C234C, taken as a fraction). Compute the PSK
phases against the true `math.pi` and the low bits differ. `gate` holds the whole
LUT byte-for-byte in its on-disk `.c64` form (512 bytes).

**Inference, marked as inference.** In the 32-QAM cross column, only the 10 labels
in quadrants 0 and 1 were read off real captures; the other 22 of the 32 are
*inferred* from the constellation's sign symmetry, not observed. The geometry the
inference rests on (the `mx = 2` column folding onto the `|Q| = 5u` arm with its two
rows reversed) reproduces the shipped bytes, which is corroboration, not observation.

---

## 3. The PACTOR acquisition markers — signed Hadamard-32 rows

The 16 P2 acquisition marker words (`shrike gen.marker_codes`) are length-16 QPSK
words whose 32 bipolar chips are signed rows of a 32×32 Sylvester-Hadamard matrix
under one fixed column permutation (`_PTS`, the labeling points in F2^5). They are
**real-orthogonal**: the 16×16 Gram matrix is `32·I + i·K` with `K` integer
antisymmetric, so the real parts are exactly orthogonal — what a differential
correlator needs. Their XOR-span is the Reed–Muller code RM(1,5) up to a coordinate
permutation. The P3 `pre800` acquisition preamble is the same 32 words with the two
band halves swapped and scaled to ±800 taps. **[closed form / reconstructed]** —
`gate` checks the codes and `pre800` byte-for-byte and re-derives the `32·I` Gram.

**Correction to an earlier reading of our own.** An earlier note concluded the
markers were "not an orthogonal code." That was an artifact of measuring correlation
at length 8 — over the half-word the orthogonality is not visible; over the full
length-16 word (or the 32-chip real vector) it is exact. The conclusion was about the
measurement window, not the code.

---

## 4. Reduced to a documented constant — `usedmap`

`usedmap.u8` is 900,000 bytes. The receiver reads one of 17 interleaved planes from
it — the off=3 stride, `17·(128·col + sub) + 3` — and that plane is a set of 36
pilot `(column, sub-band)` pairs: the reference cells the BW500 L4 receiver skips.
32 pilots serve one sub-band; columns 177 and 323 serve both. 392 columns × 2
sub-bands − 36 = 748 coded cells. `gen.usedmap_plane` reproduces exactly that stride;
`gate` compares it over the off=3 record and asserts the count is 36. The other 16
records' planes on the other strides are never consulted by this receiver and are out
of scope. One constant serves both bandwidths. **[reduced]**

---

## 5. Recovered by measurement — `grid480`

`grid480` is the per-cell differential-phase de-rotation reference,
`exp(j·2π·k/128)` at each cell the receiver fetches. It was **not** derived from a
formula. It was recovered by known-plaintext inversion of our own BlackHole loopback
recordings: a BW500 SHORT burst emits only its data body, so inverting the recorded
phase against the known payload reads the reference `k` off each emitted cell.
**[measured]**

Be precise about what the inversion recovered and what it did not:

- The receiver fetches **939** cells — 470 columns per sub-band, headroom for the
  blind `c0` scan. Of these, **786** were recovered: exactly the 393 consecutive
  columns one alignment consults.
- Every recovered value landed on the 128-point phase lattice, which is why the
  measured input is small integers — `assets/grid480_k.npy`, 786 `(index, k)` pairs,
  the one genuinely measured file here and labeled as such.
- Recovery was **redundant**: five unrelated payloads each recovered all 786
  independently and agreed pairwise, holding across 714 frames.
- The remaining **153** are **underived**. The columns that carry them are not
  emitted by a short burst, so no amount of this corpus reaches them; they are left
  zero. That costs the receiver nothing here — the 786 are exactly the span one
  alignment reads — but it is a gap in the derivation, not a solved value, and is not
  filed as one. `gate` asserts 786 recovered, all on the lattice, all within the
  939 the receiver fetches.

### 5a. Recovered by measurement — BW2300's three sub-base bin tables

The three speed levels below the BW2300 base one — hosts `BITRATE (3)`, `(2)` and
`(1)`, 82 / 41 / 24 bps — place their columns the way the base level does: one lit
bin per OFDM symbol, at `((base + stride·gray − first_bin) mod span) + first_bin`.
Each `base` table is one number per emission column — 228, 228 and 124 — and all
three are **measured**, not generated. **[measured]**

What recovered it was known-plaintext inversion of recorded VARA audio, not a
round trip through our own modulator:

- Three recorded overs at that level. Two are the 37-byte tails of the 126-byte
  transfers in the 2026-08-14 two-sided bench session — different plaintext each
  way, which is what cancels the unknown table and separates the 24 reference
  columns from the 204 data columns. The third is a link-setup over off an
  unrelated 2026-07 recording whose frame was not used in the fit.
- The three agree on every column they share (228/228 and 208/208 twice) and the
  table reproduces all three overs **bin-for-bin**, 228 of 228 columns.
- The column count is independently fixed by arithmetic: record 3's bin tables
  are the 419 `Rnd` draws between its interleaver permutations and record 4's
  seed (395 allocations + 24 reference classes, §1). Record 2's gap is 252 draws
  and closes on record 3's seed the same way, and 204 data columns + 2×24 is the
  only split that spends it.
- What is **not** derived: the values themselves. `17 + Int(Rnd*32)` taken off
  that same gap lands within −1..+3 of the measured bin on 203 of 204 data
  columns — near enough to be the same table, not near enough to be it. The
  displacement is not a function of the draw, of the column, or of the cell
  value, and no `Int(Rnd*n)` stream anywhere in the 2²⁴ orbit reproduces it. It
  is filed as measured for that reason rather than fitted to a formula that does
  not close.
- Derived after all, 2026-09-02: the record's 6 training symbols. They are the
  same 6144-sample preamble the base level fills with twelve, at rec2's geometry,
  and their allocation is this very table read from index 1 — so the preamble
  costs no numbers of its own and kestrel now keys the level as well as reads it
  (→[`vara/01 "Record 2 — the same preamble, six symbols long"`](vara/01-physical-layer.md)).
  The count is the record's own field, not that sample span divided out: it is 2 /
  4 / 6 / 12 at rec0 / rec1 / rec2 / rec3, so 4096 samples at the two low records.

**Records 1 and 0, 2026-09-04.** The two below it, recovered off gear-down tapes
made by putting band-limited noise on a stock caller's input — a transmitter's
level follows its receiver's feedback, so the overs that drop a gear are the
responder's. They needed **no known plaintext**: with `bpc` 3 at stride 2 an
allocation reaches only 16 bins of record 0's 64 and 16 of record 1's 32, so one
over admits `2**bpc` bins a column, the true base is in every over's set, and
overs of differing payloads intersect down to one. Record 2 could not be read that
way — 16 values at stride 2 on 32 bins is a whole parity of its comb, and its
three tape frames leave 0 of 228 columns unique — which is exactly why it needed
its payload and these two did not.

- The reference columns are the ones the overs agree on, ranked by how many hold
  the modal bin: 11 of 11 for the 24, against a next-best 6 (rec1) and 8 (rec0)
  and a median of 3. **Record 1's 24 are record 2's, position for position**
  (`(104k+93)//11`); record 0's are `(41k+33)//8` on its 124 columns.
- With the layout fixed the frames fall out of the counter payload the peer was
  pushing — each record frames its own, `[payload][1 ARQ control byte][CRC-16/
  GENIBUS]`, so the search is 256 starts × 256 control bytes and the right one
  reads 204 of 204 and 100 of 100 data columns against medians of 58 and 33.
  Nine record-1 frames and ten record-0 frames give the same table column for
  column.
- Scored against the tape, each over reads 228 of 228 and 124 of 124 at one
  alignment and chance at every other, with 24 of 24 on the reference columns the
  frame cannot flatter; 14 of 15 and 49 of 51 overs decode to a clean CRC. A
  second capture of the same night that neither table was measured from reads 7 of
  8 and 50 of 53 the same way, and gives back the same reference layout.
- The same near miss holds at both: `first_bin + Int(Rnd*span)` off the record's
  own gap lands within −1..+3 of the measured bin, `{−1: 11, 0: 21, +1: 36, +2:
  25, +3: 7}` over rec0's 100 data columns and `{3, 56, 83, 50, 12}` over rec1's
  204. It is a property of the family, not of one record, and none of the three is
  fitted to it.

---

## 6. Signal-processing kernels

The PACTOR symbol pulse and acquisition filters are generated at the point of
use. The VARA synthesis prototype is a measured waveform constant.

- **The PACTOR symbol pulse** — `shrike gen.symbol_pulse`, a **root-raised cosine
  at 100 Bd, 8 samples per symbol, 31 taps, normalised to unit sum**. It is both
  ends of the chain: `shrike/modem.py` shapes with it, `shrike/rx.py` and
  `shrike/p2rx.py` match against it, and `resample_poly` carries it from 8
  samples per symbol to whatever the working rate is. Odd tap count so the group
  delay is the integer 15. **[designed]**

  **The roll-off is SCS's, not a fit of ours.** `[SCS-P4]` §11.8 prints the
  symbol filter for the PACTOR-2-derived speed level — 32 coefficients, 8 per
  symbol, normalised to unit coefficient sum — and calls it "very similar to the
  RRC pulse, but optimized concerning spectral side lobes and orthogonality". A
  true RRC at roll-off **2/3** tracks that published pulse to **3.3e-3** peak.
  The same fit run against §10.1's 129-tap 1800 Bd filter returns **0.330**
  against a printed roll-off of **0.33**, which is what says the fit is reading
  the pulse and not itself. So the design is: the shape SCS documents, computed
  from the closed form rather than transcribed.

  **And the air agrees with the spec.** Swept across roll-off against the six
  packet headers of a real PACTOR-III session (`occ15.wav`), the constant-header
  fit peaks exactly at 2/3 — **0.8510** worst and 0.8759 mean, falling away on
  both sides and down to 0.8412 by 0.4.

  It also decided against a better pulse. Truncating an RRC to 31 taps costs it
  the Nyquist property it was chosen for, and minimising out-of-band energy under
  exact orthogonality — §11.8's own two criteria — yields a pulse with a third
  less stopband energy and orthogonality at 1e-17. That pulse reads the same real
  session at **0.8481**. Better in theory, measurably worse on what SCS
  transmitters actually put on the band, so it is not what ships.

  `gate` checks 31 taps, symmetry, unit sum, and half
  amplitude at the 50 Hz Nyquist frequency (the RRC's defining property, at any
  roll-off), and a matched pair whose residual ISI at the symbol instants is
  under 1.3e-2.

- **The P2 acquisition front end** — `shrike gen.acq_lowpass` and
  `gen.acq_window`, both **[designed]**, and four configuration numbers that are
  now module constants in `shrike/p2rx.py` rather than keys in a file.

  - **The window is not the pulse.** The 32-point transform runs at 800 Hz, which
    is `symbol_pulse`'s own 8 samples per symbol, and that coincidence makes
    reuse the plausible wrong answer. It is wrong: the window is a **root-raised
    cosine at unit roll-off on 27 taps**, laid at indices 0..26 of the ring with
    its peak at 13. At β = 1 the RRC numerator collapses to `4t·cos(2πt)`, whose
    zeros are exact and land three quarters and five quarters of a symbol out —
    samples 6 and 10 either side of the peak, which is the signature that tells
    this kernel from the symbol pulse's, whose crossings sit near 7.5 and 11.5.

    The distinction is expensive and quiet. Reusing the symbol pulse here still
    recovers an off-air marker at 0.997 — the score this section used to cite —
    while costing the P2 receive path its whole single-copy margin (decode holds
    to σ 0.60 with the right window and fails at σ 0 with the wrong one) and
    stopping burst copies summing at any noise level at all. A correlation score
    does not report that; `gate` pins the shape.

    Against `PACTOR-II_FEC.wav`, `tests/shrike/test_p2rx.py` recovers all three of
    an independent decoder's frame markers at their times and codeword indices,
    and a Hann of the same support finds **no marker at all**. The interior
    zeros are the β = 1 zeros at ±6 samples from the peak.
  - **The low-pass** carries the 9600 Hz input down by 12 to the 800 Hz analysis
    rate: `firwin(96, 400 Hz, Kaiser β = 6.5)`. Cutoff at the output Nyquist,
    β = 6.5 for about a 68 dB stopband by Kaiser's rule; the design puts the
    first band that folds into the retained one — 800 Hz up — **76 dB** down,
    and `gate` asserts 70.
  - `ACQ_FS = 9600`, `ACQ_DECIM = 12`, `ACQ_LO_HZ = 1500`, `FRAMES_PER_SYMBOL = 8`:
    the input rate, decimation factor, baseband mixing frequency, and
    differential span (8 frames = one symbol at 100 Bd against the 800 Hz frame
    rate). The transform size is the length of the window.

- **`prototype_pulse.npz` — the VARA BW500 synthesis prototype.** Keys `dspan`
  (int64 offsets −800..+800, 1601 taps about a symbol centre), `p0`/`p1` (complex128
  prototype for sub-band 0 at 1350 Hz / sub-band 1 at 1650 Hz), and `f0` = [1350,
  1650]. `kestrel/tx/varahf500_tx.py` overlap-adds it at hop `H = 512` to render
  symbols to audio. It is a fixed, payload-independent WOLA Nyquist prototype pulse
  (spec/03 §3.6.1) — Nyquist to within an ISI ratio of 0.02–0.08 at ±512, which is
  why the receiver's single-sample-per-symbol demod is exact — **estimated from our
  own recordings of real BW500 transmit audio** by per-symbol deconvolution, which
  also defines `grid480`. It ships as
  a file for one reason, and it is not that the estimate cannot be repeated: the
  deconvolution re-runs unmodified in 3.4 s and returns these bytes to 4.4e-16
  relative, and substituting `tablegen.grid480()` for the archived de-rotation
  reference still reproduces the pulse at 0.99996 correlation. What does not ship is
  the 4.3-second recording it reads. (A *first-principles* rebuild from the WOLA
  window generator is a separate and open question — the complex per-sub-band pulse
  does not factor into a standard window times a carrier to the float grid.)
  **[measured]** Validated end to end: with it, TX audio self-loops
  byte-exact through `kestrel/rx/varahf500.py` and reproduces VARA's own
  transmitted waveform for the same payloads.

---

## 7. Corrections owed back to the published record

Two published or internal statements are wrong, and re-deriving these tables is what
showed it. Recorded here so the correction travels with the tables.

- **SCS's PACTOR interleaver recurrence.** SCS's published pseudocode reads
  `S=1; P=0; OUT[i]=P; P+=M; if P > PACKET_SIZE then P=S; S+=1`. Read literally,
  `P > PACKET_SIZE` emits index `N` and is **never a bijection**: across the 18,804
  `(N, M)` pairs the recurrence is defined over, exhaustively tested, only
  `P >= PACKET_SIZE` yields a permutation for every table. The generator uses the
  `>=` branch (`shrike gen._walk`); the published `>` is a typo that produces an
  out-of-range index wherever the two readings differ.
- **Constellation extent.** The 64-entry complex LUT is a truncated raster.
  Trailing zeros are exhausted short PSK columns, not a further APSK alphabet.

---

## 8. PACTOR-III interleaver recurrence

`shrike gen.INTERLEAVERS` defines eight permutations from SCS's published
recurrence: five single walks and three two-dimensional walk products.
They are generated at the point of use. Optional reference-vector checks run
when the development corpus is available.

The full-frame interleaver geometry remains unresolved
(`shrike/unknowns.py`, `U3_INTERLEAVER`). The per-speed-level data frames do not
consult these full-frame tables.

---

### Summary

Generated tables are built at the point of use. The measured synthesis prototype
is packaged as an asset:

| File (key) | Category | How known | Proved |
|---|---|---|---|
| `prototype_pulse.npz` | Measured | WOLA synthesis prototype, deconvolved from our own recordings | §6 |

| Constant | How known | Held by |
|---|---|---|
| Whiteners (BW500, BW2300) | Computed — VB6 `Rnd`, `bit=s>>23` | byte-identical over live extent |
| Interleavers (4 arrays) | Computed — VB6 `Rnd` permutation stream | byte-identical incl. overwritten tail |
| `alloc_col3`, `map2_col3` | Closed form — draws from the record-3→4 gap | byte-identical; closes on seed 15597635 |
| `map1_col3`, `clsparm_gray`, `hdr_raster_src` | Closed form — index formulae | byte-identical |
| `constellation_lut` | Reconstructed — VB6 `Pi` + x87 reduction | byte-identical; 22/32 cross labels inferred |
| Acquisition markers, `pre800` | Signed Hadamard-32 rows, RM(1,5) span | byte-identical; `32·I` real Gram |
| `usedmap` | Reduced — 36 pilots on one of 17 planes | matches the off=3 record; count 36 |
| `grid480` | Measured — inversion of loopback recordings | 786/939 on the 128-lattice; 153 underived |
| `symbol_pulse` | Designed — RRC at `[SCS-P4]` §11.8's roll-off, 31 taps; real-air sweep peaks there | does not ship; `gate` holds the shape and the Nyquist ISI |
| `acq_window` | Designed — RRC at unit roll-off, 27 taps, peak at 13 of 32 | does not ship; 3/3 real markers at 0.997–0.998, Hann finds none |
| `acq_lowpass` | Designed — `firwin(96, 400 Hz, Kaiser β 6.5)` | does not ship; first image band 76 dB down |
| Eight P3 interleavers | Computed — SCS's published recurrence | byte-identical against the archived bytes |
