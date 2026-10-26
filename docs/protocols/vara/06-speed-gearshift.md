<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# 06 — Speed level & gear-shift logic

VARA adapts its throughput by stepping a **speed level** up and down. This
section specifies the level ladder for each bandwidth, the per-level physical
descriptor, and the up-/down-shift policy; it builds on the per-level physical
and coding detail in [`01`](01-physical-layer.md) and [`03`](03-coding.md).

The level index and its rate are host-visible as `BITRATE (N) x bps TX` (the
level this station will use for its next over) and `BITRATE (N) x bps RX` (the
level of the frame just decoded). High speed capability gates the top of the
ladder: a speed limited build caps at `BITRATE (4)` and reports
a speed limited link notification; a high speed capable link, marked in the caller's
link-setup ([`04 §4.2A`](04-frame-block-formats.md)), runs every level to the top.

## 6.1 Speed-level ladder (500 Hz)

The BW500 levels below are those of **VARA HF 4.9.0**.

| Level | Rate (`BITRATE (N)`) | Net payload / block | Modulation (→[01](01-physical-layer.md)) | FEC rate (→[03](03-coding.md)) |
|-------|----------------------|---------------------|------------------------------------------|--------------------------------|
| 0–2 (more robust) | (not fixed here) | | | |
| **3** | **61 bps** | **34 bytes** | (→01) | (→03) |
| **4** (speed limited cap) | **88 bps** | **43 bytes** | (→01) | (→03) |
| 5–13 (high speed capable) | see §6.1c | | tone at 5–6, OFDM at 7–13 | 1/2 to 5/6 |

> Net payload per block is what the `BUFFER` ledger clears: each ACKed block
> clears exactly 43 raw bytes at L4 and 34 at L3. `BUFFER` counts raw host bytes
> irrespective of `COMPRESSION TEXT`; payload content does not change the figure
> (43 B/block at L4 for every payload). Levels 4–13 are tabulated with their
> modulation and rate in §6.1c.

## 6.1a Full mode ladder — bandwidths, level counts, peak rates

VARA HF has **exactly three bandwidths**; 2750 is the widest. Peak rates below
are *uncompressed user* rates at the top level of each band; per-level
constellation, code rate and carrier count are in §6.1c.

| Bandwidth | Name | Speed levels | Peak bps (top level) |
|---|---|---|---|
| 500 Hz | Narrow | 13 | **1543** |
| 2300 Hz | Standard | 16 | **7050** |
| **2750 Hz** | **Tactical** | **17** | **8489** |

- **The waveform climbs through three families with level:** one-hot index
  modulation at the base, differential BPSK/QPSK tones in the middle, then OFDM
  with differential QPSK/8-PSK and coherent 16-QAM/32-QAM cross, the last from the
  shared constellation LUT (`constellation_lut_64.c64`, regenerable from
  `hfmodem/kestrel/rx/tablegen.py` — see §6.1c).
- **BW2750** uses the level table below. High speed capable links can use
  the full table; speed limited peers negotiate a lower ceiling. These are
  link capabilities, independent of whether the station calls or answers.

Kestrel’s current `arq/phy.py` sets `TX_CEILING` and `RX_CEILING` to each
bandwidth’s highest implemented level: 13 at BW500, 16 at BW2300 and 17 at
BW2750. Limits reported by speed limited peers in the bench campaigns below
do not restrict kestrel’s codec coverage.

## 6.1b The widest, fastest mode

The top of VARA's mode space:

| Parameter | Value |
|---|---|
| Bandwidth | **2750 Hz (Tactical)**, widest |
| Speed level | **top (level 17)** |
| Nominal bps (uncompressed) | **≈ 8489** |
| Per-carrier constellation | **32-QAM cross**, coherent on pilots — top of the ladder |
| Symbol grid | OFDM, 1024-point IFFT + **128-sample cyclic prefix** = 1152 samples (41.67 Bd), as at BW2300, with a 32-sample raised-cosine blend at each symbol edge; **59 carriers on bins 3..61** — see [`01 §OFDM family`](01-physical-layer.md) |
| Frame | 3922 bytes (3919 payload), rate 4/5; one or two frames an over |
| Payload compression | static order-0 Huffman ([`08`](08-compression.md)) |

> Within a family the encoder is **one level-parametric code path**, differing
> between rows only in a handful of integers (carriers or bands, constellation,
> code rate). The negotiated link capability determines which levels a link may use.

## 6.1c Levels by bandwidth

Every level is one of three waveform families ([`01 §1.0`](01-physical-layer.md)):
index at the base, tone in the middle, OFDM at the top. Level L is record L − 1 of
the bandwidth's interleaver chain. `frame` is `floor(N/8)` bytes: payload, the
control byte and the CRC, so the payload is `frame − 3`. Rates follow from `N` and
`coded` (`2N + 12` is rate 1/2; above it, the group puncture of
[`03 §3.3`](03-coding.md)). Symbols are per over; a tone symbol is 512 samples, an
OFDM symbol 1152, and the index base level carries 395 columns behind its
preamble. The top level of each bandwidth keys one or two frames an over (§6.2).

**BW2300**

| L | family | modulation | rate | N | coded | frame B | payload B | carriers | symbols |
|---|---|---|---|---|---|---|---|---|---|
| 4 | index | one-hot, 16 bins | 1/2 | 736 | 1484 | 92 | 89 | 16 bins | 12 + 395 |
| 5 | tone | DQPSK | 1/2 | 1080 | 2172 | 135 | 132 | 3 bands | 383 |
| 6 | tone | DQPSK | 1/2 | 1410 | 2832 | 176 | 173 | 4 | 372 |
| 7 | tone | DQPSK | 1/2 | 2076 | 4164 | 259 | 256 | 6 | 362 |
| 8 | tone | DQPSK | 1/2 | 2754 | 5520 | 344 | 341 | 8 | 359 |
| 9 | tone | DQPSK | 1/2 | 3424 | 6860 | 428 | 425 | 10 | 356 |
| 10 | OFDM | QPSK, differential | 1/2 | 7197 | 14406 | 899 | 896 | 49 | 153 |
| 11 | OFDM | QPSK, differential | 2/3 | 9596 | 14406 | 1199 | 1196 | 49 | 153 |
| 12 | OFDM | QPSK, differential | 4/5 | 11512 | 14406 | 1439 | 1436 | 49 | 153 |
| 13 | OFDM | 8PSK, differential | 2/3 | 14398 | 21609 | 1799 | 1796 | 49 | 153 |
| 14 | OFDM | 8PSK, differential | 4/5 | 17276 | 21609 | 2159 | 2156 | 49 | 153 |
| 15 | OFDM | 16QAM, coherent | 5/6 | 21710 | 26068 | 2713 | 2710 | 49 | 159 |
| 16 | OFDM | 32-cross, coherent | 4/5 | 26056 | 32585 | 3257 | 3254 | 49 | 159, or 313 with two frames |

**BW2750**

| L | family | modulation | rate | N | coded | frame B | payload B | carriers | symbols |
|---|---|---|---|---|---|---|---|---|---|
| 4 | index | one-hot, 20 bins | 1/2 | 736 | 1484 | 92 | 89 | 20 bins | 12 + 395 |
| 5–9 | tone | DQPSK | 1/2 | as BW2300 L5–9 | | | | 3 / 4 / 6 / 8 / 10 bands | 383 / 372 / 362 / 359 / 356 |
| 10 | tone | DQPSK | 1/2 | 4466 | 8944 | 558 | 555 | 13 bands | 356 |
| 11 | OFDM | QPSK, differential | 1/2 | 8667 | 17346 | 1083 | 1080 | 59 | 153 |
| 12 | OFDM | QPSK, differential | 2/3 | 11556 | 17346 | 1444 | 1441 | 59 | 153 |
| 13 | OFDM | QPSK, differential | 4/5 | 13864 | 17346 | 1733 | 1730 | 59 | 153 |
| 14 | OFDM | 8PSK, differential | 2/3 | 17338 | 26019 | 2167 | 2164 | 59 | 153 |
| 15 | OFDM | 8PSK, differential | 4/5 | 20804 | 26019 | 2600 | 2597 | 59 | 153 |
| 16 | OFDM | 16QAM, coherent | 5/6 | 26145 | 31388 | 3268 | 3265 | 59 | 159 |
| 17 | OFDM | 32-cross, coherent | 4/5 | 31376 | 39235 | 3922 | 3919 | 59 | 159, or 313 with two frames |

BW2750 L5–9 are BW2300's records 4–8 on wider band spacings; L10 onwards is its
own chain, which closes on 4061299 after L17.

**BW500**

| L | family | modulation | rate | N | coded | frame B | payload B | carriers | symbols |
|---|---|---|---|---|---|---|---|---|---|
| 4 | tone | DBPSK | 1/2 | 368 | 748 | 46 | 43 | 2 bands | 9 + 394 |
| 5 | tone | DQPSK | 1/2 | 744 | 1500 | 93 | 90 | 2 bands | 402 |
| 6 | tone | DQPSK | 1/2 | 1080 | 2172 | 135 | 132 | 3 bands | 383 |
| 7 | OFDM | QPSK, differential | 1/2 | 1578 | 3168 | 197 | 194 | 11 | 153 |
| 8 | OFDM | QPSK, differential | 2/3 | 2104 | 3168 | 263 | 260 | 11 | 153 |
| 9 | OFDM | QPSK, differential | 4/5 | 2524 | 3168 | 315 | 312 | 11 | 153 |
| 10 | OFDM | 8PSK, differential | 2/3 | 3160 | 4752 | 395 | 392 | 11 | 153 |
| 11 | OFDM | 8PSK, differential | 4/5 | 3792 | 4752 | 474 | 471 | 11 | 153 |
| 12 | OFDM | 16QAM, coherent | 5/6 | 4755 | 5720 | 594 | 591 | 11 | 159 |
| 13 | OFDM | 32-cross, coherent | 4/5 | 5708 | 7150 | 713 | 710 | 11 | 159, or 313 with two frames |

An OFDM over is 3.672 s (153 symbols), 3.816 s (159) or 7.512 s (313); a tone over
3.80–4.30 s. The transcripts' delivered bytes agree with
the payload column at every level on tape.

**Constellation columns.** The OFDM levels take column `pos` of the 64-entry LUT,
indexed by the raw bit group `v` (`tablegen._column(pos, v)`), with no Gray pass:

- `pos=1` **QPSK**: `(1,0)(1,90)(1,−90)(1,180)` (`(magnitude, degrees)`, `v` order)
- `pos=2` **8-PSK**: unit circle at `0,±45,±90,±135,180`
- `pos=3` **16-QAM** square, corner modulus 1.1 (moduli `{0.367, 0.82, 1.1}`)
- `pos=4` **32-QAM cross**: the 6×6 grid of odd multiples of `u = 0.3/√2` per
  axis minus the four `(±5,±5)` corners (the corners' modulus is 1.5); mean power
  0.900, `d_min = 0.424264`, moduli `{0.3, 0.671, 0.9, 1.082, 1.237}`
- `pos=5` **64-QAM** square, corner modulus 1.6, used by no level

The 512-byte LUT is a truncated prefix of these columns — column `pos` is present
only up to `v = ⌊(63−pos)/6⌋` — so it shows 11 of the 16-QAM points and 10 of
the 32-cross points. All 32 cross labels, including the 22 the table does not
hold, are exact on tape (`evidence/KPA-bw2300-const.md` §4).
The full columns are generated by `hfmodem/kestrel/rx/tablegen.py` (gate:
`tests/gates/test_tablegen.py`).

**What is validated.** Every tone and OFDM row decodes byte-exact from the
high speed capable bench tapes of two stock VARA HF 4.9.0 installations, at all three
bandwidths: 52 of 54 BW2300 tone overs, 124 of 126 BW2750 and BW500 tone overs,
and every OFDM frame (46 at BW2300, 69 at BW2750, 60 at BW500); the four failures
are the noisiest overs on the tapes (`evidence/KPA-*.md`).
Rendered from the transcript's bytes, kestrel's OFDM transmitter matches the tape
to its 16-bit quantisation floor (−84 to −85 dB) and its tone transmitter to
about −50 dB, the rest being sub-mHz per-band carrier offsets no receiver sees
(`TAPER-WINDOW.md`, `ofdm_render_gate.py`, `tone_render_residual.py`). These
waveform comparisons establish the recorded bench results. They do not measure
RF performance at every level; current implementation coverage is stated above.

## 6.1d Base levels (records 0–3)

BW2300 and BW2750 L1–4 and BW500 L1–3 are the index family
([`01 §2300`](01-physical-layer.md)): **index / position modulation, value `= −1j`
one-hot.** The coded value picks the *bin*
(`bin=((alloc+stride·gray−first_bin)%span)+first_bin`), not a constellation
point, with no scramble and no CP. rec3 is byte-exact against VARA audio
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

**rec1 and rec0 are closed, 2026-09-04**. rec1 has the
same 228/24 shape by the same gap arithmetic (612 coded at 3 bpc = 204 cells,
gap 252) and rec0 the same at 124 columns (300 coded at 3 bpc = 100 cells, gap
148), and both tables are now measured off gear-down tapes — with no known
plaintext, since 3 bpc at stride 2 leaves most of the comb unreachable from an
allocation and overs of differing payloads intersect down to one bin a column.
rec1's 24 reference columns are rec2's, position for position; rec0's are
`(41k+33)//8`. Both take the rate-1/3 turbo branch (`3N+12`), and kestrel reads
and renders them (→[`table-derivations §5a`](../table-derivations.md)).
rec4–8's gaps of 36–41 draws are the tone family's pilots
([`03 §3.7`](03-coding.md)).

## 6.2 The gear-shift on a high speed capable link

**The receiver drives.** After every delivered frame the IRS answers with a step
code, the frame index of its 8-symbol answer
([`04 §4.2C`](04-frame-block-formats.md), *Level requests*): 0 hold, 6 +1, 4 +2,
7 +3, 8 +4, 5 −1, 1 NAK. The ISS keys exactly the level asked, 428 of 428 delivered
full frames, and the IRS's `BITRATE (N) … RX` line equals the ISS's `… TX` line on
all 489 delivered frames; an undelivered frame gets no RX line. So every rule below
but the NAK fall and the double frame runs in the IRS. As observed on 5 clean
sessions and 57 noise ladders of a high speed capable stock pair
(`evidence/GEARSHIFT-OBSERVED.md`, `ACK-LEVEL-REQUEST.md`):

| rule | observed |
|---|---|
| **Start** | The caller's link-setup is level 4 on every tape, and no first data frame is above 6. A callee ISS starts where the caller's session-confirm puts it — frame 3, 4 or 5 for level 4, 5 or 6 — and the caller chooses from the noise on the callee's connect answer (BW2300: 6 at a floor of −26..−23 dBFS, 5 at −22..−17, 4 at −16..−10). A caller ISS starts at 4. |
| **Next delivery** | The short frame that closes a delivery names, in the byte ahead of its `82`, the level its sender will read the peer's next delivery at ([`04 §4.2A`](04-frame-block-formats.md)); that delivery's first frame comes at that level (B keyed 6, 4 and 5 as A's closes named on the 2026-09-26 bench) and the step codes move it from there. A delivery keyed at any other level is not read: KB8AY's reply closed `05 82` and `03 42` on 2026-09-26 and our level-4 overs behind them drew only NAKs. |
| **Up** | No dwell: the level moves on the frame after any delivered frame, by +1 to +4 and never more (240 up-steps: +1 ×100, +2 ×39, +3 ×35, +4 ×66). The step is the number of SN thresholds the frame clears, capped at 4; the fit leaves 0–4 misfits per level. |
| **Jump cap** | At BW2300 and BW2750 a jump of more than one never lands above top − 2: the top two levels are entered only from the level below (16 ← 15 nine of nine, 17 ← 16 seven of seven), and BW2300 goes 12 → 14 over 13. BW500 jumps straight onto 13 (9 → 13, 10 → 13). |
| **Frame count** | When the rest of the delivery fits in one or two frames, the IRS asks for the lowest level that still finishes it in the fewest frames, moving at most one level down per frame. It shortens the last frame (14 → 13 with 74 B, 17 → 16 with 2937 B), holds a level the SN would lift and raises one it would not (12 → 13 to fit 621 B). The IRS therefore knows how much of the delivery is left; the field that tells it is not identified. |
| **Down** | A delivered frame read at low SN takes the next frame one level down (frame 5), never more: 0 of 373 delivered moves fell further. |
| **NAK** | Frame 1. The ISS falls 1–4 levels by its own per-level rule (−1 ×35, −2 ×19, −3 ×3, −4 ×7): BW500 12 → 11 and 11 → 9, BW2750 17 → 16 and 16 → 14; a level entered by a jump that fails at once falls further (BW500 13 → 9 after 9 → 13, 12 → 9 after 8 → 12). A second failure falls again. A frame the IRS does not answer draws a poll from the ISS, which the IRS answers with its 16-symbol over-NAK (`SESSION_OVER_NAK_SHORT`, the caller's 61/541 twin of the responder's 289/541), and the resend is 2–4 levels down. On the bench nothing goes below 4, and the lowest level after a failure run is 5; on air K9KDJ (80 m, 2026-09-27) re-sent at 3, 2 and 1 behind our over-NAKs. |
| **Re-climb** | The same rules, with a brake near the bottom: the first clean frame at level 4–6 after a failure climbs +1 (11 of 13), and the next usually +4. |
| **Double frames** | After exactly six consecutive delivered frames at the top level, every frame is two frames long while at least two single loads remain (52 of 52, none before the sixth). The IRS answers frame 0 and the ISS doubles unasked. When less than two loads remain the frame-count rule steps down one. |
| **Time** | Single frames cycle in 4.3–5.0 s and double ones in 8.2–8.35 s, so doubling buys 9–11 % at the top. 65536 bytes took 88 s at BW2750 (top 17), 104 s at BW2300 (16) and 392 s at BW500 (13). |

The SN thresholds themselves are fitted only for levels 5–7
(GEARSHIFT-OBSERVED §4) and are not fixed here.

## 6.3 kestrel's policy

What stock does as ISS is fixed by the IRS's codes; what the IRS asks for, and how
far a sender falls after a NAK, are each station's own. kestrel's choices below
are ours, not VARA's (`vara/vara_arq.py`, `arq/phy.py`):

- **As ISS** it obeys every step frame on the peer's stream (`_LEVEL_STEP`),
  within 4..top and within `arq/phy.TX_CEILING`, the highest level its renderer
  keys at that bandwidth, and logs the level asked against the level keyed. After
  a NAK it re-sends one level down from a level a single step reached, one more
  after a second failure in a row, at the last level that delivered when a jump
  reached it, two down after an unanswered poll, down to level 1 (`_fall`), where
  a step frame then never lifts it back to 4 unasked (`_level_space`). It keys two frames an
  over after six delivered singles at the top level. It starts each delivery at
  the level the peer's last close named (`_note_close_level`), or where the last
  step left the link when no close came between.
- **As IRS** it answers NAK on a CRC failure and one level down when the frame
  came in above `arq/phy.RX_CEILING`; a resend that fails too draws the
  16-symbol data NAK (`SESSION_DATA_NAK_CALLER_SHORT`, 61/31, or the
  responder's 289/31), as a stock IRS keys it, and the next resend is expected
  one more level down, and read at the level over that too (`_read_levels`;
  KC9GHZ's greeting came clean at 5 behind our 61/31 on 2026-09-27). The peer's 8-symbol poll (frame 12, `over_poll`) or
  683 idle while an over of its is due draws the short over-NAK (`SESSION_OVER_NAK_SHORT`) and
  never a repeat of the last positive answer, which the peer would take as the
  acknowledgement of the over this station did not read; the resend is then
  expected two levels down (`_nak_owed_over`), under 4 when the peer's last over
  was (`_irs_floor`). An idle that begins sooner behind
  our answer than the shortest over (3.67 s) has no over ahead of it: the peer
  did not hear the answer, and the same answer is keyed again at the same step
  (`_answer_unheard`; K9KDJ, 80 m, 2026-09-27, whose 745s came 1.5-2.6 s behind
  our continues and whose resends of delivered overs our NAKs drew). Any other
  idle in the peer's turn with an over of its due, our last answer a NAK or
  nothing at all, draws the over-NAK too and never keepalive-a
  (`_reack_on_idle`; KC9GHZ and K7EK-10, 2026-09-27). At BW500 the caller's
  over-NAK is 60/1117, the frame the stock caller keyed behind the responder's
  short idles on the 2026-09-27 cables; 61/541 is the only one on tape at the
  wide bandwidths. Near the end of a delivery it applies the
  frame-count rule to the peer's per-frame field, which counts overs rather than
  bytes and so approximates stock's rule. Otherwise it asks up by the number of
  required-SNR rows above the frame's level that its cell SNR clears by 3 dB
  (`_REQUIRED_SNR`, `_CLIMB_MARGIN`), at most four and never by a jump onto the
  top two levels; one down when the SNR is under the level's own row; and holds
  between (`_irs_frame`). Level 4 and the records under it are read for their
  channel SNR off the index-law frame itself (the lit bin of every column
  against the rest, `rx/varahf2300.quality`, calibrated on the 2026-09-26
  bench overs), and a clean frame there asks level 5 from −2 dB, 6 from −1 dB
  (never more than two up), record 3 asks 4 from −4 dB, and a base read under
  −5.8 dB asks record 2 (`_INDEX_CLIMB`, `_INDEX_FALL`); with no measure it
  asks one up. Its own close names the
  level its last step asked, which is where its receiver then expects the
  peer's next delivery (`_close_byte`).
- **Start.** As caller it confirms level 5 or 6 when the connect-response's tone
  measure clears 15 or 18 dB and the receiver reads that level, otherwise 4
  (`_START_FLOORS`, `_CONFIRM_START`). As callee it starts where the confirm says,
  within its TX ceiling.
- **High speed capability.** A link runs high speed capable when this station is
  configured as high speed capable: as caller it then sends the high speed
  capable link-setup bytes (`link_setup_frame(..., high_speed=True)`); as callee
  it logs what the caller's link-setup is marked (`link_setup_marks`). A speed
  limited link keeps levels 1–4 and its own paths (§6.4).

## 6.4 Speed limited links

A free build caps at `BITRATE (4)`, prints a speed limited link notification, and asks at most
`04` in its link-setup ([`04 §4.2A`](04-frame-block-formats.md)). Its levels are
the index family's 1–4 and, at BW500, the base tone level. The step codes are the
same (0, 6, 4, 5 and the NAK), a BW500 sender doubles after six singles at level 4
as the high speed capable top does, and the level that closes a delivery drops a record
([`05 §5.4`](05-arq-session-state-machine.md)). Recovery from NAKs at levels 1–4 is
[`29`](../../implementation/vara/29-low-level-data-nacks.md); the link-setup's level negotiation is
[`30`](30-connect-speed-negotiation.md).

## 6.5 Limits of this description

- **Stock's SN thresholds** are fitted only at levels 5–7; above that, the IRS's
  choice of step is described by its outcome, not by a threshold table.
- **The remaining-bytes field** a stock IRS reads for the frame-count rule is not
  identified.
- **Which link-setup byte** — `body[7]` bit 0 or the ask `06` — a responder reads to
  print a high speed capability notification is not separable on the tapes, and levels 1–3 of a
  high speed capable link-setup were not captured.
- **Whether a stock IRS reads a level below the one it asked for** is unmeasured.
  kestrel's ISS keys below the request whenever its TX ceiling is lower.
- **RF performance** at every level is not measured by the bench waveform
  comparisons in §6.1c. The current codec coverage is stated above.
