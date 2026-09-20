# PACTOR-2 — waveform, frame geometry, coding and acquisition

This document specifies the PACTOR-2 physical and link layer: the two-carrier
DPSK waveform, the speed-level ladder, the frame geometry, the coding chain, and
the acquisition and frame-detection rules a receiver applies. The description is
materially incomplete; §7 states what is not specified here.

Public sources are cited inline: `[M.1798]` for ITU-R Recommendation M.1798,
`[sigidwiki]` for the sigidwiki PACTOR-2 signal page and its two published
sample recordings (one ARQ, one FEC broadcast), `[SCS]` for SCS's *The PACTOR-2
Protocol — A Technical Description* (1996), `[SCS-P3]` for *The PACTOR-3
Protocol* (2004), and `[SCS-P4]` for *The PACTOR-4 Protocol*.

`[SCS-P4]` is a PACTOR-2 source and not an analogy. Its §11 speed level is a
two-carrier mode it states outright is "based on the PACTOR-2 speedlevel 1";
§11.5 and §11.6 reprint that level's convolutional code and interleaver loop,
§11.7 gives its differential phases as "just like P2", and §11.3 its header as
"same as PACTOR-2". Where it and `[SCS]` describe the same thing they agree, and
between them they specify several things `[SCS]` alone leaves open — the carrier
stagger (§1.4), the phase alphabet (§1.1) and the CRC (§3).

Section headings carry an evidence tag saying what stands behind them and how
much of it; the tags are defined in `EVIDENCE.md`. Almost every one of them here
reads `1 clip`, and §7.8 is where that is stated as the binding limit on the
whole document rather than as a caveat on a paragraph.

---

## 1. The waveform

### 1.1 Carriers and modulation

- **Two DPSK carriers**, nominal spacing **200 Hz** = twice the symbol rate;
  measured spacing on off-air recordings is **193–208 Hz**. Nominal centre is
  **1500 Hz**, giving carriers at **1400 / 1600 Hz**. The absolute centre
  follows the receiver's tuning: the two published recordings sit at ≈1008 Hz
  (ARQ) and ≈1718 Hz (FEC) `[sigidwiki]`.
- **Symbol rate 100 Bd** (10 ms per symbol) `[M.1798]`. At a 48 kHz sample rate
  the symbol integrator spans L = 480 samples.
- **Occupied bandwidth ≈450 Hz** `[sigidwiki]`, consistent with two 100 Bd
  carriers 200 Hz apart — and PUBLISHED: `[SCS]` §1 calls PACTOR-2 "a two-tone
  DPSK system with **raised cosine pulse shaping**, which reduces the required
  bandwidth to less than 500 Hz", §3 giving "around 450 Hz at minus 50 dB".
  `pactor2._render` applies `tablegen.symbol_pulse` on this rate's grid, which is
  the same kernel PACTOR-3 transmits through. Rectangular symbols went out until
  2026-09-02 and put **5.0 %** of the radiated power outside 1250–1750 Hz, with
  1899 Hz between the −26 dB edges and the whole audio band above −50 dB; shaped,
  the same six bursts measure **360 Hz at −26 dB, 366 at −30, 401 at −40, 559 at
  −50 and 0.005 % out of band**. The independent monitor reads the shaped
  waveform at speed levels 1, 2 and 3 exactly as it read the unshaped one, and
  returns the HB9AK link's own payload text from the replica arm.
- **Both carriers carry independent data.** The pair is a throughput doubling,
  not a diversity repeat: the frame geometry of §2 counts a frame's coded bits
  as `symbols × 2 tones × bits per cell`, and in the FEC recording the two
  carriers' recovered DQPSK dibit streams agree only ≈25 %, which is chance for
  a 4-ary alphabet, across all four relative rotations.
- **Modulation order follows the speed level.** Measured by the Mth-power
  concentration statistic `C_M = |mean((d/|d|)^M)|`, `d = sym[k]·conj(sym[k-1])`:
  - the FEC recording is **DQPSK** on both carriers, and stays DQPSK for the
    whole clip once the symbol period of §1.3 is right. Packet by packet, with
    the period and phase fitted inside each 72-symbol packet, C4 reads median
    **0.759** and **0.835** on the two tones against C2 at **0.090** and
    **0.091**, and **M = 4 is the argmax in 34 of 34 packet-carrier pairs** —
    including inside the three frames a reference decoder parameterises as level
    0, which is DBPSK (§5.12);
  - **the ARQ recording is not read at all** (§7.7). It was previously recorded
    here as opening DBPSK and climbing to DQPSK, "a live speed-level
    progression". That reading was taken on a front end fixed at `fs/100` with
    one grid for both carriers, and it does not survive a front end that measures
    the period: the clip does not lock at any rate from 60 to 160 Bd.
- Carriers lock cleanly with a ±30 Hz fine search. A coarse FFT peak lands a few
  Hz off, which is enough to smear a 100 Bd differential constellation.
- **The symbol alphabet sits on odd multiples of 45°**, not on multiples of 90°.
  It is `p2rx`'s own soft demapper read backwards — `soft_bits(d, 2) =
  (Im d, −Re d)` with positive meaning a zero bit puts its decision boundaries on
  the axes, so the constellation is on the diagonals — and it agrees with the
  π/4-rotated marker alphabet §5.5 measures independently. `pactor2.modulate`
  emitted multiples of 90° until 2026-08-01; against that alphabet the demapper
  recovers **0.545** of the bits at the best of all eight half-step rotations,
  because the two label the constellation in opposite rotational senses and no
  rotation composes one into the other. Both now pass
  `test_p2.py::test_soft_demapper_inverts_the_modulator`, which the old pair
  fails on both DQPSK lanes.

The two-carrier signature discriminates PACTOR-2 from PACTOR-3 without ambiguity:
PACTOR-2's ~200 Hz carrier spacing against PACTOR-3 SL1's 840 Hz between tones 5
and 12, and against PACTOR-3's 14-, 18-, 6- and 2-tone sets.

### 1.2 Speed levels

Four levels, one per modulation order, on the two 100 Bd carriers:

| level | modulation | bits per cell | raw rate |
|---|---|---|---|
| 0 | DBPSK | 1 | 200 bit/s |
| 1 | DQPSK | 2 | 400 bit/s |
| 2 | 8-DPSK | 3 | 600 bit/s |
| 3 | 16-DPSK | 4 | 800 bit/s |

The ladder this table gives is the document's own. `[SCS]` §1: "The maximum
absolute transfer rate is 800 bits per second" — which is level 3's rung, and
pins the other three by the modulation order. The per-level code rate is
published as well, at Annex I p. 6 (§3), so the coded-bit count varies with
modulation order alone and the rate changes underneath it; the 100/200/400/600
ladder once recorded here as the document's own is in no edition of it.

The speed level is a parameter *within* the mode, distinct from the protocol
level (PACTOR-1 / -2 / -3); a status field reporting "SL: 2" is a speed level,
not PACTOR-2.

### 1.3 A recording's symbol clock is its own  [MEASURED, 2026-08-01]

`fs / 100` is the nominal symbol period, not the recording's. On
`PACTOR-II_FEC_c1500.wav` the two tones independently measure **476.3 samples per
symbol — 100.77 Bd on that file's 48 kHz grid**, a 0.77 % error; the 4th-power
concentration over 900 symbols reads 0.73 and 0.81 there against 0.45 and 0.49 at
the nominal 480. Re-fitting carrier and phase per 60-symbol window makes the
concentration uniformly 0.6–0.87 on both tones and the optimum phase walks
steadily, which is a rate error and not an offset.

The consequence is structural: over a 72-symbol frame, 3.7 samples per symbol
accumulates to **0.43 of a symbol** from one end of the frame to the other, so a
receiver integrating on a fixed `fs/100` grid straddles symbol boundaries by the
end of every frame it starts cleanly. It cannot be recovered by searching the
symbol phase, and it is the reason the alternating per-tone C4 in earlier
readings of this clip looked like a property of the data: it is the fixed grid
walking through the true one. `p2rx.measure_symbol_period` measures it.

### 1.4 The two carriers are staggered by T/2  [MEASURED + PUBLISHED, 2026-08-01]

Two halves, and they are not equally strong. That the carriers **do not share a
symbol boundary** is MEASURED on real PACTOR-2 audio, twice, by independent
estimators (§1.4.1). That the offset is **exactly T/2** is published — for the
PACTOR-4 speed level derived from PACTOR-2 speed level 1, not for PACTOR-2 in so
many words, so for PACTOR-2 itself it is an inference, and a strong one.

`[SCS-P4]` §11.7, describing the speed level §11 states is based on PACTOR-2
speed level 1 and whose phase alphabet it calls "just like P2":
"Symbols on the carrier with the lower frequency always appear delayed by T / 2."
It is load-bearing in that document's own arithmetic — §11.11 gives the packet
length as `216*T + 4*T + T/2`, the half symbol being the low carrier finishing
late — and §11.5 and §11.6 inherit the convolutional code and the interleaver
from the same P2 level outright, reprinting both.

**The sign does not transfer**, and is not established for PACTOR-2 here.
PACTOR-3 runs the same mechanism on its speed level 2 with the LOW carriers
LEADING (`spec.SUBBAND_LEAD`, measured), which is the opposite of the sentence
above. A T/2 offset is also its own inverse modulo T, so the two carriers share
one sampling grid at half-symbol spacing either way and the sign cannot show up
in the sampling at all — only in which high-tone symbol pairs with which low-tone
one. §7.6 carries that pairing as an explicit arm rather than assuming it.

**And on an ARQ link the sign is not constant.** The two virtual carriers exchange
tones every cycle and take the stagger with them, so both signs appear on the air
in alternate bursts of the same QSO — §7.11, measured. Receive reads which one a
burst is in rather than assuming either.

**The published crest factor is consistent with it.** `[SCS]` §3: "The Crest
Factor of the PACTOR-2 signal is therefore only 1.45." `[SCS-P3]` Figure 2 gives
PACTOR-3 speed level 1 — also two carriers, also DBPSK, and with no lead recorded
against it in `spec.SUBBAND_LEAD` — as **1.9 dB**, a gap of 0.45 dB in PACTOR-2's
favour between two modes that are otherwise the same shape. Synthesized here,
staggering the two carriers by T/2 lowers peak/RMS by **0.33 dB** under a raised
cosine and **0.70–0.83 dB** under the 32-tap symbol filter `[SCS-P4]` §11.8
prints: the right sign, and bracketing the published gap. The absolute numbers do
not match — peak/RMS is 6.9–8.3 dB for every two-carrier arrangement tried, and
exactly 2.000 for two unmodulated tones — so SCS's crest-factor convention is
something other than peak/RMS and is not recovered here. Only the difference is
evidence, and it is offered as corroboration rather than as the measurement.

### 1.4.1 What the two estimators said before the document was read  [P+S: 1 clip]

With the period of §1.3 held fixed so only the phase is free, the two tones of
`PACTOR-II_FEC_c1500.wav` want **different symbol phases**: 60 and 324 of 476
samples, a difference of **264 samples = 0.554 of a symbol** where half a symbol
is 238. Both phase curves are sharply peaked — concentration runs 0.075 to 0.697
on the low tone and 0.074 to 0.823 on the high one across the phase — so the
optima are not noise.

The 10 ms integrator has a sinc null at exactly 200 Hz, which is the tone
spacing, so this is not the other carrier leaking into the estimate; and a
constant carrier offset moves the constellation's rotation, not the integration
phase.

A second estimator agrees that the boundaries differ and disagrees on by how
much. Asking where the phase transitions physically are — the magnitude of the
baseband's sample-to-sample difference peaks at a DPSK phase step, and the phase
of that waveform's component at the symbol rate is the boundary — puts the two
tones at 135.3 and 285.3 samples, a difference of **150 = 0.315 of a symbol**.
Both tones carry a real symbol-rate line (relative strength 0.032 and 0.028
against 0.0036 for the largest of twenty noise runs), and no decoding enters that
statistic at all.

So **the two carriers do not share a symbol boundary** was measured twice by
independent means before §1.4's sentence was found, at 0.315 and 0.554 of a
symbol — one estimator either side of the published T/2, which is what one fixed
offset looks like through two windows of different quality. The published value
is used; neither measurement is precise enough to refine it.

Every decode scan before this one integrated **both tones on one grid**, which
puts one of them across its own symbol boundaries from end to end of every frame
— half the coded bits of every candidate. The ARQ clip cannot corroborate it: its
whole-clip concentration is only 0.13, because it carries two stations, control
bursts and gaps, so its phase curve has no usable peak.

---

## 2. Frame geometry

The speed level and cycle length determine the frame geometry:

| level | bits/sym/tone | short: coded bits | symbols | Field bytes | long: coded bits | symbols | Field bytes |
|---|---|---|---|---|---|---|---|
| 0 | 1 (DBPSK) | 144 | 72 | 8 | 640 | 320 | 39 |
| 1 | 2 (DQPSK) | 288 | 72 | 17 | 1280 | 320 | 79 |
| 2 | 3 (8-DPSK) | 432 | 72 | 35 | 1920 | 320 | 159 |
| 3 | 4 (16-DPSK) | 576 | 72 | 62 | 2560 | 320 | 279 |

Every coded-bit count factors exactly as `symbols × 2 tones × bits_per_cell` —
the PACTOR-II speed ladder of 200/400/600/800 bit/s raw on two 100 Bd tones.

**The 72 and the 320 are published**: `[SCS]` Figure 1, of the packet the de-interleaver hands back — "The
original data field consists of 72 or 320 pulses in standard or data mode,
respectively." The two packet durations the same document gives are those fields
plus the header and nothing else: 8 + 72 = 80 pulses at 10 ms is the stated
0.8 s, and 8 + 320 = 328 is the stated 3.28 s. The phase-reference pulse ahead of
each packet is outside both figures, so a frame occupies 81 or 329 slots of its
cycle — 0.81 s of a 1.25 s standard cycle, 3.29 s of a 3.75 s data cycle.

**Field bytes** gives the packet field length. All eight values are `(coded bits × code rate − 8 flush
bits) / 8` at the code rates SCS publishes per speed level (1/2, 1/2, 2/3, 7/8;
Annex I), and all eight equal that document's Table 1 data bytes plus the status
byte and the two CRC bytes: 5+3, 14+3, 32+3, 59+3 short and 36+3, 76+3, 156+3,
276+3 long. The interleaver depths are published separately, one set of four for
both frame lengths — SL1: 16, SL2: 8, SL3: 4, SL4: 2 (Annex II) — and every one
of them divides every packet size in the table. `shrike.pactor2` used the field-length
column as its helical stride until this was checked; it now carries the published
depths. The level-2 coincidence that supported the old reading — 432 coded bits
with 35, matching the stride PACTOR-3 measures on its own 432-soft header — is a
coincidence: PACTOR-3's header holds 26 bytes at that stride, so the two numbers
are not the same quantity.

The two candidate tables permute every P2 buffer size, so no self-consistent
round trip can separate them, and an A/B against an independent decoder cannot
either: 8 spliced arms ×
2 repeats (both tables, SL1 and SL2, each with a matched bad-CRC negative) had
it acquire the P1 connect in 16 of 16 runs and print nothing for the P2 data
in any of them — the §5 frame-marker wall, not a verdict.

The CRC-covered field length is derived rather than tabulated:
`crc_bytes = (trellis steps − 8 flush bits)/8`, where the step count is the coded
bits opened back out through the level's puncture. It divides exactly on all
eight paths and reproduces the Field bytes column and Table 1 on all eight. The
derivation used to assume rate 1/2 and 6 flush bits at every level, which agreed
at levels 0 and 1 and was short by 9 and 27 bytes at levels 2 and 3 — see §3.

---

## 3. Coding chain

PACTOR-2 and PACTOR-3 share their coding backend, interleaver family, CRC and
status-byte semantics. PACTOR-2 does not have a separate codec. The backend holds
two convolutional codes; PACTOR-2 uses the K = 9 one at every speed level, which
is also the code PACTOR-3 runs on its own case 0.

- **Convolutional code:** K = 9, `G1 = 111101011`, `G2 = 101110001` (Annex I,
  p. 6), written newest-bit-first as **`(0o657, 0o435)`**; bits packed
  **LSB-first**, trellis zero-flushed with 8 bits.
- **Code rate:** 1/2 at SL1 and SL2, punctured to 2/3 at SL3 (vector `1011`) and
  7/8 at SL4 (vector `10100110101011`), all four from Annex I p. 6.
- **CRC:** published. `[SCS-P4]` §5, "Same as P3": "CCITT-CRC16 with 0xFFFF
  preassigned CRC register and CRC complement at the end of the CRC calculation.
  The lower 8 bit of the 16 bit CRC will be transmitted first." Initial register
  0xFFFF, final complement, little-endian trailer — which with the reflected
  CCITT table is CRC-16/X-25, and is what `coding.crc16` computes and
  `pactor2.build_field` appends. `[SCS]` and `[M.1798]` name only "16-bit CCITT
  Cyclic Redundancy Check", so this was carried from PACTOR-3 until the P4
  document was read; it is no longer an assumption.
- **Whitening: there is none.**  [MEASURED, 2026-08-01 — §5.10]
  No PACTOR document in the reference library — `[SCS]`, `[SCS-P3]`, `[SCS-P4]`,
  PT-III, `[M.1798]` -0/-1/-2, or the Wavecom tables — describes a scrambler,
  randomiser or whitener for any PACTOR mode, and PACTOR-2 does not have one. The
  unwhitened test vector in §5.10 is accepted; the whitened variant is not.
- **Interleaver:** a helical block interleaver of depth M over a packet of
  PACKET_SIZE bits, PACKET_SIZE being the punctured coded-bit count with the
  status byte, the CRC and the 8 flush bits in it. Output position I draws input
  position P(I), where P starts at 0 and steps forward by M, and on reaching
  PACKET_SIZE restarts at the next unused offset S = 1, 2, 3, … — each restart one
  place further in than the last. That recurrence is the whole of the permutation,
  and it is the fact SCS's Annex II routine (`[SCS]` Annex II, `[SCS-P4]` §11.6)
  states; `pactor2.interleave_pointer` generates the permutation from it and
  `channel_of_code` inverts it, which is what a receiver gathers by. All four
  depths divide all eight packet sizes.

  A published routine fixes the permutation, not the boundary convention: its
  restart test reads `P > PACKET_SIZE`, which for a size divisible by the depth
  lets P reach PACKET_SIZE and index one past the end, so a 0-based pointer wants
  `P >= PACKET_SIZE`, which is the implemented condition.

  This file carried PACTOR-3's **backward** helical walk until 2026-08-01 — down
  from the end by the stride, emitting each column reversed. That is a different
  permutation at every one of the four depths and no start offset relates them;
  `test_p2.py::test_published_interleaver_listing` fails on all four with the old
  walk. Both permute every buffer size, so no round trip against ourselves could
  ever have separated them (§2), and both were carried as separate arms through
  the scans of §7.6 — neither decoded.
- **Status byte:** one byte, immediately before the CRC, carrying `FRNR` in bits
  0-1 and `TYPE` in bits 2-4 — measured against a receiver's own report of bytes
  we chose (§5.10). The other fields that report names — `SL`, `CYC`, `RQ`,
  `REV`, `LSB`, `dF` — are not in it. `[SCS]` Figure 1 publishes the same layout
  and fills it in: bits 0-1 the modulo-4 packet number; bits 2-4 the data type,
  `000` 8-bit ASCII, `001` Huffman, `010` Huffman swapped ("upper case"), `011`
  reserved, `100`/`101` PMC German normal/swapped, `110`/`111` PMC English
  normal/swapped — so bit 4 set means the PMC family, the compression PACTOR-1
  does not have; bits 5, 6, 7 the cycle-length suggestion, changeover request
  and QRT flag, the same three PACTOR-3 carries.

### The code family was the wrong one of the two  [CORRECTED, 2026-08-01]

`shrike.pactor2` encoded with K = 7 and `(0o171, 0o133)` at rate 1/2 on every
level. The published polynomials bit-reversed are `(0o657, 0o435)` — exactly
`shrike.placement.CASE0_CODE`, the K = 9 code PACTOR-3 uses on its own slowest
path. So the shared backend the reuse argument rests on holds two codes, and
PACTOR-2 had been pointed at the K = 7 one. `coding.CODE_K9` now owns the pair
and both protocols take it from there.

Puncturing follows: SL3 and SL4 had none, so their fields were short by 9 and 27
bytes on the short cycle. Every one of the eight paths now divides exactly and
reproduces Table 1's payload without rounding — the check
`test_p2.py::test_published_geometry` makes. The old configuration fails that
check on all four short paths, so it is not a rubber stamp: K = 7 leaves 8.25,
17.25, 26.25 and 35.25 bytes, never a whole number of them.

The arbiter cannot yet judge the change. The two punctured levels ride 8-DPSK and
16-DPSK, and `pactor2.modulate` carries only DBPSK and DQPSK, so no arm can be
built for SL3 or SL4 at all; the SL1 and SL2 arms that can be built run into the
§5 frame-marker wall as before — 4 of 4 runs on the corrected chain acquired the
P1 connect and printed nothing for the P2 data.

---

## 4. Control signals

The six 20-bit control-signal codewords are shared with PACTOR-3: Plotkin-unique,
minimum distance 12, sent DPSK on the two carriers.

`[SCS]` §2: "All CS are always sent in DBPSK in order to obtain a maximum of
robustness", 20 pulses, and every CS is preceded by its own phase reference
pulse. The bit mapping onto those 20 pulses, the burst's position within the
cycle and the repetition are not specified there.

The same section counts them differently and does not disagree: "six different
CS, each consisting of 40 bits, all having exactly the maximum possible mutual
hamming distance of 24 bits to each other. They thus reach exactly the Plotkin
boundary". Twenty pulses carrying one DBPSK bit on each of two carriers is 40
bits on the air, and `spec.CONTROL_SIGNALS` is equidistant at 12 over 20 in all
fifteen pairs, so the on-air word is at 24 over 40 — the Plotkin bound at both
lengths, which is 12 for six words of 20 bits and 24 for six of 40. What the
document's count settles is that the set is 40 bits WIDE and not 40 bits LONG:
no six-word code of 40 consecutive bits is being looked for, and the PACTOR-1
codewords `0x06A9` and `0x059A` are 12-bit words that cannot be two of these
(`pactor1-control-signals.md` §3).

An FEC broadcast carries no control signals at all. The excess of low-distance
codeword matches previously recorded here for the ARQ recording — 15 at distance
≤2 against ~4 expected — was taken through the same front end §1.1 now retires,
on a clip that does not demodulate (§7.7), so it is withdrawn rather than
weakened: it is not evidence either way.

### 4.1 What is keyed and read, and on what basis  [HYPOTHESIS, 2026-09-02]

`pactor2.control_signal` and `p2rx.control_signal_at` fill the three gaps above
from **PACTOR-3**, which carries the same six codewords and whose keying of them
is measured on a stranger's tape (`pactor3.md` §7):

| gap | what is keyed | where it comes from |
|---|---|---|
| bit → pulse | the codeword least significant bit first, one bit per pulse, DBPSK | `rx._cs_bits` and `placement.control_signal` read `spec.CONTROL_SIGNALS` this way for PACTOR-3; `p2rx.cs_table` is now the one orientation for the whole modem |
| alphabet | this waveform's own measured DBPSK diagonals (`pactor2._DBPSK_STEP`), not PACTOR-3's on-axis pair | the P2 alphabet is measured at SL1 through an independent monitor (§7.12); the codeword rides the same carriers the field does |
| position | `pactor2.cs_slot` — the packet's own keying plus a 70 ms turnaround, so **0.880 s** short and **3.360 s** long, from the answered packet's phase reference | PACTOR-3's is 0.890 s and 3.390 s over a keying one pulse longer, measured over sixteen short cycles of `PIII_Complete_1` |
| stagger | the same T/2 the two virtual carriers carry everywhere else in PACTOR-2, and the swap with them | §1.4, §7.11 |

**What is proven and what is not.** Our own renderer and our own reader close:
every codeword reads at zero bit errors in both arrangements at 48 and 12 kHz, no
codeword is manufactured out of 400 windows of noise or out of 40 SL3 data
fields, and a whole rendered link — connect, announcement, twelve PACTOR-2
packets at three levels with the peer answering on the raster — returns 12 of 12
fields byte-exact and 12 of 12 codewords at zero errors. That is a round trip and
it says nothing about whether a PACTOR-2 station would answer this way.

**An independent monitor cannot arbitrate it, and that is now measured rather
than assumed.**
The same link was put to an independent monitor three times, in three arms —
codewords at 0.880 s, codewords at 0.890 s, and no codewords at all — and it read
the three identically: the connect, `Level: 2`, the level at every packet and an
unbroken frame count, with the run-to-run scatter larger than any difference
between the arms. The monitor is receive-only and never answers, so a reverse
channel is outside what it can judge. What the arms do settle is that a codeword
in the answer slot at either instant costs the packets nothing — no frame and no
level is lost to it, which is the same result the PACTOR-3 two-sided arm returned.

**A PACTOR-1 connect is what makes the monitor track a PACTOR-2 link**, and that
correction belongs here: free-running PACTOR-2 bursts were previously read as
"no `###CONNECT` and no `Level` line", and the missing piece was the connect
rather than the reverse channel. With a PACTOR-1 connect and an announcement
packet in front of them the same bursts read `###CONNECT`, `###PLISTEN: Level: 2`
and a frame number that runs unbroken from the PACTOR-1 packet through all three
PACTOR-2 speed levels.

---

## 5. Acquisition and frame detection

### 5.1 The correlator front end

A per-sample quadrature downconverter feeding a 32-point FFT — a tone
correlator, not a pattern matcher. The internal rate is **9600 Hz**.

| stage | operation |
|---|---|
| NCO / LO | 512-entry sine table with linear interpolation; f_LO = **1500.0 Hz** |
| mix | sample × (cos, sin) → complex baseband |
| lowpass | **96-tap** float FIR, applied to I and Q |
| decimate | **D = 12** → 800 Hz |
| ring + window | 32-deep ring × a 32-point window |
| FFT | **32-point complex** |
| phase | `atan2(re, −im) × 32768/π` → int16 angle, **2π = 65536** |
| differential | phase **minus the same bin 8 FFT frames ago** |

Two consequences fall out exactly: FFT bin spacing = 9600/12/32 = **25.000 Hz**,
and the 8-frame differential spans 96 samples = **10.0 ms = exactly one symbol at
100 Bd**. Each FFT frame therefore yields, per bin, the DPSK differential phase
over one symbol.

### 5.2 The bin grid

The analysis grid has 25 Hz spacing. Detector bin `b` corresponds to the
carrier-pair centre `1200 + 25*b` Hz for `b = 0..23`. Bin 12 therefore centres
the 1400/1600 Hz pair at 1500 Hz. The fractional-bin frequency estimate refines
that centre.

### 5.3 The detection gates

Acquisition scores the differential-phase codewords described in §5.4.1.
A strong phase match identifies a carrier pair, codeword and timing candidate.
The codeword comparison uses unit phasors, so its score measures phase agreement
rather than received power.

### 5.4 The frame parameters

The marker supplies speed level, cycle length and carrier-pair information.
These parameters determine the coded-bit count and field length in §2.

### 5.4.1 The frame marker, and the codebook it is drawn from  [MEASURED, 2026-08-01]

The marker is an **eight-chip complex codeword carried on the per-symbol
differential phase of each carrier**, and the acquisition path is specified end to
end below. The implementation is checked against recorded markers and
independent reception results.

**The codebook is complex, not real.** The 512 chips of ±1 read as 64 rows of 8
are scored in pairs: row `2k` against the *cosine* of a carrier's differential
phase, row `2k+1` against the *sine*, summed. That is one complex codeword

```
c[k][t] = chip[2k][t] + i·chip[2k+1][t],   k = 0…15,  t = 0…7
```

per bank, every chip an odd multiple of 45°. Sixteen codewords per bank, two
banks — one per carrier — and that is where the whole protocol's 45° phase
alphabet comes from. Read as 64 real rows the table has only 51 distinct
patterns, which is what made it look like a broken codebook; read as 32 complex
ones the objection disappears.

**The correlator.** For bin `n`, over `k = 0…15`,

```
z = Σ_t e^{i d_t[n]}·c_lo[k][t]  +  Σ_t e^{i d'_t[n+8]}·c_hi[k][t]
mag = |800·z|² >> 15
```

where `d_t` is the one-symbol differential phase of bin `n`, tap `t` being `t`
symbols back, and `d'` the same for bin `n+8` read from the phasor row **four FFT
frames — half a symbol — earlier**. That lag is the T/2 carrier stagger of §1.4,
and it says the **lower carrier leads**, which is §5.12's reading of the same
stagger arrived at from the data field.

**Bin `n` is a PAIR, not a tone.** Slots `n` and `n+8` are 200 Hz apart, so a
detector bin is a two-carrier hypothesis and `1200 + 25n` Hz is the pair's
**centre**. Bin 12 is the 1400/1600 pair. That resolves §5.2's grid: the reported
carrier was never one tone's frequency.

**The ceiling is 10000 and there is no slack.** Both sums saturate at
`8·800·√2`, so `|z| ≤ 18102` and `mag ≤ 10000`. A real marker off
`PACTOR-II_FEC_c1500.wav` reaches **9967** — 99.7 % — and arming needs 8800, i.e.
**88 %** of a perfect match on both carriers at once.

**The transmit side follows.** Tap `t` is `t` symbols back, so symbol `j` of a
nine-pulse burst — Figure 1's eight header pulses plus its "single phase reference
pulse", which is what the differential is taken against — carries differential
step `-arg(c[7-j])`, and the two carriers are staggered T/2. `pactor2.frame_marker`
emits exactly that.

**`k` identifies the frame parameters.** The level and length follow
`sl = (k>>1)&3`, `long = (k>>3)&1`. So the marker is how a frame's parameters
reach a receiver that has not demodulated anything yet, and §7.2's "what it fails
to carry is the frame's parameters" is answered.

**Both ends measured.** Sixteen markers built by `pactor2.frame_marker`, played
into a conforming receiver: **13 frame descriptors formed, every one on bin 12,
every one carrying back the codeword index that was sent** (the other three fell
before that receiver's capture started). Peak magnitudes 9457–9785 against the
8800 needed. And in the other direction, `p2rx.find_markers` reads the three real
markers on the published FEC recording at 2.485, 6.077 and 9.670 s with codeword
indices 1, 0, 1 — the same times and the same indices that receiver's own
correlator reports, at 0.998–0.999 of the noiseless ceiling. The two gaps are
3.592 s, which is §5.9's 362-symbol long cycle measured a different way entirely.

### 5.5 Marker phase alphabet

Each complex codeword chip has real and imaginary components of ±1, so its
phase is an odd multiple of 45°. A plain DQPSK alphabet at multiples of 90°
is offset from every target. The marker alphabet is π/4-rotated.

### 5.6 Front-end conventions

Mix the audio at 1500 Hz, apply the acquisition low-pass filter, and decimate
from 9600 Hz to 800 Hz. A 32-point windowed transform gives 25 Hz bin spacing.
The differential spans eight analysis frames, one symbol at 100 Bd. Use the
acquisition window defined by `shrike.tablegen.acq_window`, rather than the
transmit symbol pulse. `p2rx.find_markers` checks the resulting codeword scores
on recorded audio.

### 5.7 Recording limits

The published FEC recording supplies acquisition evidence but does not provide
a CRC-valid payload reference. Acquisition and payload decoding are separate
checks; the later off-air comparisons in §7.11–§7.13 provide byte-level results.

### 5.8 The header is 8 pulses, and its codeword table is recovered  [P+S: 1 clip]

SCS Figure 1: "Header consisting of 8 pulses, supports QRG-Tracking,
Listen-Mode and Memory-ARQ", and "Every packet and CS is preceded by a single
phase reference pulse", all pulses 10 ms. That is what the detector's chip table
of §5.3 scores, and the table is now recovered in full: **512 consecutive int16
of ±800**, the only such run available, read as 64 rows of 8. Row 3 is
`+ − − − + − − +`, reproducing §5.5's known-good codeword exactly. 51 of the 64
rows are distinct, so the set is not an orthogonal code.

A codeword hunt on real audio needs a joint constraint, not a peak: the maximum
over 64 rows at one position has a **median of 0.71**, because eight complex
values almost always suit one of 64 sign patterns. That is the same shape of
error as screening a dense tone comb against the in-band median.

`[SCS-P4]` §11.11 counts "16 header symbols (8 header symbols per carrier)", so
the header rides **both** carriers and requiring them to score the same row at
the same position is a far narrower target. Scored that way on the FEC clip, with
a null built by rolling one carrier against the other: **nothing**, in either
carrier pairing — joint maximum 0.788 against a null maximum of 0.887.

So the 8-pulse header of Figure 1 is not detectable in the time domain on this
recording by correlating against the table, and the reason may be that the two
are not the same object. What §5.3's detector correlates its 8 chips against is
the **phasor row, which is indexed by FFT bin** — and §5.5's accepting hit is
eight consecutive *bins* sharing one index, not eight consecutive *symbols*. A
frequency-domain signature and a time-domain pulse train are both "eight of
something"; only the first is what that table scores.

**PACTOR-2 has no sideband anchor, and cannot have one.** For PACTOR-3 the
published header codeword settles the sideband because its tone comb is not
symmetric about 1500 Hz, so a mirrored recording only anchors mirrored.
PACTOR-2's two carriers are 1400 and 1600 about a centre of 1500: **a mirror maps
the pair onto itself**. And the codeword chips are real ±1 while the published
phase alphabet is antipodal (π/4 and 5π/4), so `|Σ c·d|` is identical for `d` and
`conj(d)` — a BPSK codeword is conjugation-symmetric. Neither the geometry nor
the header can tell the sidebands apart.

The sideband therefore shows up **only** in the sense of the data field's
differential, where §7.6 already carries it as an explicit axis alongside the
carrier order. For PACTOR-2 that is not a check to run before trusting a
negative; it is a dimension the search has to span, and does.

### 5.9 The FEC broadcast repeats on a 362-symbol period  [P: 1 clip, 12.5 s]

On the corrected front end — measured period, published T/2 stagger — the
differential phases of `PACTOR-II_FEC_c1500.wav`, quantised to the four
diagonals, match a lagged copy of themselves best at **lag 362 symbols (3.59 s)
and at its double, 724**. Both carriers put those two lags first, at both span
lengths tried:

| carrier | span | best lags | agreement | median over all lags | 99.9th pct |
|---|---|---|---|---|---|
| 1400 | 72 sym | 724, 362 | 0.597, 0.528 | 0.361 | 0.521 |
| 1600 | 72 sym | 724, 362 | 0.667, 0.667 | 0.361 | 0.643 |
| 1400 | 320 sym | 362 | 0.353 | 0.278 | 0.347 |
| 1600 | 320 sym | 362, 724 | 0.384, 0.362 | 0.278 | 0.365 |

Chance is 0.25 for a four-way symbol. Four readings taken independently — two
carriers × two span lengths — agreeing on one lag and its double is the result;
no single one of them would be worth much.

**362 symbols is the long cycle, not the standard one.** At the measured
100.756 Bd a 1.25 s standard cycle is 125.9 symbols and a 3.75 s data-mode cycle
is 377.8. The period is 4 % short of the long cycle and nowhere near a multiple
of the standard one (2.87×). Four detected frames in a 12.51 s clip says the same
thing: 3.3 long cycles, or 10 standard ones.

The agreement is partial, 0.35 to 0.67 rather than the ~0.9 an identical repeat
would give at this signal quality, so what recurs at that period is not
established — a retransmitted packet whose status byte and counter differ, a
header, or the cycle structure itself would all show this way.

**Corroborated independently, 2026-08-01.** The three frame markers §5.4.1 locates
on this clip sit at 2.485, 6.077 and 9.670 s: two gaps of 3.592 s, the same period
reached here from symbol-quantised autocorrelation and there from a codeword
correlator that knows nothing about it.

**It is not the packet that repeats.** §7.9 correlates the three anchored
72-symbol data windows against each other directly, with the per-lane rotation
fitted and a rolled null: all twelve pairings sit below their own null. So the
period is the cycle structure, and there is nothing to soft-combine.

### 5.10 Coding chain and accepted test vector  [MEASURED, 2026-08-01]

The packet decoding stages are:

| stage | what it does |
|---|---|
| geometry | level and long flag → coded-bit count and field length, §2's table |
| de-interleave | Annex II's pointer walk, run on the output side |
| depuncture | mask-driven; at level ≤ 1 a straight copy |
| trellis | **K = 9** convolutional decoding |
| whiten | none for PACTOR-2 |
| CRC | reflected 0x1021, preset 0xFFFF, over the field length |

The Annex II interleaver depths are 16, 8, 4 and 2 by speed level.
The restart condition is `P >= PACKET_SIZE`. PACTOR-2 does not apply a
whitener. At level 0, the short geometry has 144 coded bits and 72 trellis
steps, including the 8-bit flush.

The accepted test vector and two negative conventions give:

| arm | its trellis returned | its CRC |
|---|---|---|
| **unwhitened, LSB-first** | `534852494b45fee1` — our field, byte-exact | **0 = PASS** |
| whitened, LSB-first | `534843c06857cc7a` | fail |
| unwhitened, MSB-first | `ca124a92d2a27f87` — our bytes bit-reversed | fail |

The accepted vector is the frame `pactor2.build_field(b"SHRIKE", PATHS[0])`
builds, encoded and interleaved at depth 16 into 144 channel bits:
`a8d658b05d31530c117c8811668ef58dbbe9`.
`test_p2.py::test_accepted_vector` keeps this vector, covering the K = 9 taps,
LSB-first packing, 8-bit flush, interleaver, geometry, CRC byte order and
positive-soft-means-zero convention.

**The status byte carries two of the reported fields and not the other three.**
Four frames were put through, with status bytes `0x45`, `0x01`, `0x42` and
`0xa3`; all four passed the CRC and all four were reported.

| reported | tracks | evidence |
|---|---|---|
| `FRNR` | **status bits 0-1** | 1, 1, 2, 3 across the four, as chosen |
| `TYPE` | **status bits 2-4** | 1 on `0x45`, 0 on the three others |
| `CYC` | *not* status bit 5 | stayed 0 with bit 5 set |
| `REV` | *not* status bit 6 | stayed 1 across bit 6 clear and set |
| `RQ` | *not* status bit 7 | stayed 0 with bit 7 set |

So `spec.status_byte`'s low five bits are confirmed and its top three are not
this receiver's `CYC`, `REV` and `RQ`; those, with `SL`, `LSB` and `dF`, come
from the descriptor and the detector rather than from the field. An earlier
reading here took all five as confirmed off the single byte `0x45`, which had
them constant — three arms with those bits varied is what separates them.

**The field is `[data][status][2 CRC]`**, observed rather than derived: with the
type set to 8-bit ASCII the reported payload length is **5** where the field is
8, so the status byte and the CRC are outside it. That is §2's split, from the
receiver. The payload text itself is not emitted — §6's confirmed-packet rule
applies to a listen-mode frame however valid it is.

### 5.11 The detector does fire on a frame we transmit  [MEASURED, 2026-08-01]

§7.2 recorded that a conforming receiver will not detect a transmitter built from
this document. Against the corrected π/4 alphabet that is too strong. A P1
connect followed by eighteen synthesized P2 bursts on the 1.25 s grid — SL1 and
SL2, staggered and not, both tone orders, 24.6 s — drove **six** packet decodes
with the PACTOR-2 coding chain (§5.10).
The same connect followed by silence drove **none**: its only two decodes are on
another protocol's path.

What those six frames are not is our frames. Their descriptors carry carriers of
1359, 1442, 1514, 1662, 1711 and 1762 Hz — the band is 1400–1600 — and levels and
cycle lengths of `(144, 8)`, `(1280, 79)`, `(432, 35)`, `(640, 39)`, `(640, 39)`
and `(1920, 159)`, which is the geometry table sampled at random. So the detector
arms on our energy and mis-parameterises the frame; the gap is a marker that
carries the level, the cycle length and the carrier, not detection as such.

### 5.12 Carrier order and symbol alignment

`pactor2.channel_buffer` places the upper carrier at channel position `2k`
and the lower carrier at `2k+1` for symbol `k`. Each carrier is sampled on its
own fitted symbol grid; the lower carrier leads by half a symbol (§1.4.1).
There is no additional whole-symbol lane lag.

`tests/shrike/test_p2rx.py` checks the differential demodulator against fixed
reference vectors on the published FEC recording. The comparison searches
carrier offset, symbol period and alignment, and includes shuffled-vector and
wrong-recording controls. Carrier order does not settle constellation polarity:
a four-phase alphabet permits quarter-turn rotations, which the frame CRC must
resolve.

## 6. Session structure

- A PACTOR-2 session is entered through a **PACTOR-1 connect**. The receiver
  acquires on the SELCALL and thereafter tracks that session's cycle timing on the
  **1.25 s cycle grid**. A mid-stream excerpt carries no acquisition point of its
  own and is not acquired.
- Frame acceptance requires the measured carrier within **±5 Hz** of the acquired
  session's nominal (§5.4).
- Payload emission requires a **confirmed packet**: the frame's cycle counter and
  memory-ARQ pairing must continue the established session. A mid-stream excerpt
  spliced onto an unrelated connect passes the frequency gate and forms frames,
  but never confirms a packet, and so emits no payload.

---

## 7. What this document does not specify

1. **Front-end conventions** are specified in §5.6 and implemented in `p2rx`.
2. ~~**The frame-marker waveform.**~~ **SETTLED, 2026-08-01,** and specified in
   §5.4.1: eight chips of a complex 45° codeword on each carrier's differential
   phase, nine pulses, the lower carrier leading by T/2. Measured both ways —
   thirteen frame descriptors formed by a conforming receiver off markers this
   package builds, each on the right bin pair and carrying back the codeword
   index sent; and the three real markers on the published FEC recording read by
   `p2rx.find_markers` at the same times and with the same indices that receiver
   reports. What remains open above it is the packet-confirm state below, not
   acquisition.
3. ~~**Packet-confirm state.**~~ **NOT A GATE, 2026-08-03.** This item read that
   the fields gating payload emission, and the ARQ cycle/counter state behind
   them, were what stood between a conformant transmission and a printed payload.
   They are not: a plain stream of data bursts on the 1.25 s grid, with no
   PACTOR-1 connect in front of it and no session to continue, prints from its
   FIRST frame once the constellation origin is right (§7.12). The confirm state
   was never reached because every CRC was failing. Everything between
   demodulation and CRC is PACTOR-3 backend reuse (§3).
4. **Control-signal framing** (§4).
5. **Soft-bit layout:** channel-order values alternate between the two
   carriers. A level-0 short frame contains 144 coded bits (§2, §5.10).
6. **Payload decoding from off-air audio.** Still open, and the negative is now a
   much stronger one — the front end of §1.3 and §1.1 and the interleaver of §3
   were all wrong in the earlier rounds, and the scanner is now checked against a
   planted frame rather than trusted.

   The scan enumerates, per level and cycle length: every symbol start; both tone
   orders and both lane lags — two axes §5.12 has since **measured** on the FEC
   clip, the upper carrier feeding channel position 0 and the lag being zero on
   the fitted grids; **both senses of the differential** (a sideband inversion, and the axis
   that separates the demapper's labelling from the modulator's — this was never
   searched before); 2M constellation rotations rather than M, so a half-step is
   reachable; the residual constellation angle removed per window from
   `arg(mean(u^M))/M`, since a carrier offset appears only as that rotation and
   the concentration statistic is blind to it; Annex II's permutation in both
   boundary readings and both directions, plus identity as a control; and
   whitening on and off, the SCS description carrying no scrambler. Three of
   those axes are now closed by §5.10 — the interleaver's direction and boundary,
   and whitening — so a rerun spans a third of the space these counts cover.

   **The scanner is validated by a plant**: a frame built with the Annex II
   permutation, transmitted at the FEC clip's measured 100.77 Bd with a 7.3 Hz
   carrier offset and noise, is recovered byte-exact at its planted start. It was
   not recovered at M rotations, nor at 2M; it came back only once the sense of
   the differential was added as an axis. That is what put the demapper fault of
   §1.1 in view, and the fault itself was then isolated directly, by demodulating
   known bits through `soft_bits` and finding no rotation that inverts
   `modulate`.

| recording | arm | trials | expected false | CRC-valid frames |
|---|---|---|---|---|
| FEC broadcast | one grid, whitened | 284,400 | 4.34 | 1, non-ASCII, singleton |
| FEC broadcast | one grid, unwhitened | 284,400 | 4.34 | 3, non-ASCII, singletons |
| ARQ | one grid, whitened | 383,520 | 5.85 | 5, non-ASCII |
| ARQ | one grid, unwhitened | 383,520 | 5.85 | 8, non-ASCII |
| FEC broadcast | per-tone grids, both whitenings, both lane lags | 909,696 | 13.88 | 5, non-ASCII, singletons |
| ARQ | per-tone grids, both whitenings, both lane lags | 1,227,648 | 18.73 | 14, non-ASCII |
| FEC broadcast | **published T/2 grid**, both cell layouts, both pairings | 909,696 | 13.88 | 14, non-ASCII, singletons |
| FEC broadcast | **long cycle**, SL2, published T/2 grid | 479,488 | 7.32 | 8, non-ASCII, singletons |
| FEC broadcast | **long cycle**, SL1, published T/2 grid | 239,744 | 3.66 | 13, non-ASCII, singletons |

The middle arms put each carrier on its own fitted phase; the rest put them on
the published T/2 grid exactly, and add the cell layout as an axis —
**carrier-major**, all of one carrier's coded bits then all of the other's,
against the symbol-major interleaving every earlier scan assumed. The last two
answer §5.9's 362-symbol repeat by scanning the **long cycle**, which every
attempt before had skipped. All of them land where chance puts them.

**The scanner recovers plants through all of it.** Frames at both cycle lengths
and both cell layouts, DBPSK and DQPSK, built with the Annex II permutation and
transmitted at 100.77 Bd with the low carrier T/2 late, a 7.3 Hz carrier offset
and noise, each come back byte-exact at the planted start.

The long arm is only affordable because the trellis is run **batched over symbol
starts** — the same decode for every candidate frame at one parameter setting,
paid once instead of 512 times per start. That is bit-exact against the scalar
decoder, tie-breaking included, which is the thing to check about a faster
decoder: one that decodes differently would quietly invalidate every count taken
with it. It turned a three-hour arm with no output until the end into four
minutes with a line per parameter combination.

So the stagger is compensated, on the published value, and it does not decode.
That does not make it wrong — it is published, and §1.4's crest-factor gap
corroborates it — but it is no longer the outstanding suspect.

   Every accept is a singleton at one start. The pairs that report a count of two
   are one start reached by two settings that are the same map for a one-bit
   cell — conjugation with a half-turn is the identity on a real projection — so
   they are one accept counted twice, not a confirmation. Being singletons is the
   clock-shift result stated the other way round: the scan offers every symbol
   start, so an accept whose neighbours reject is one that does not survive a
   one-symbol shift. Nothing repeats across cycles, and the totals sit where
   chance puts them.

   **The ARQ rows are not evidence** — see §7.7. Only the FEC rows are a
   negative, and every conclusion above rests on them alone.

### 7.6.1 The front end was blind to the two densest levels  [T+P: 1 clip]

Speed levels 3 and 4 were recorded here as unscannable because `pactor2.modulate`
emits only DBPSK and DQPSK, so no plant could be built. Building one directly
from a phase table exposed a second and larger reason: **the front end could not
have locked them either.**

An M-ary differential concentrates under the **Mth** power and under no lower
one. The concentration statistic driving `measure_symbol_period` tried only the
2nd and 4th, so on a synthesized 8-DPSK carrier it reads **0.052** and on a
16-DPSK one **0.016**, against 0.99 once widened to M ∈ {2, 4, 8, 16}. The symbol
period it returned at those levels was fitted to nothing at all — at 16-DPSK it
came back 476.16 samples for a signal generated at exactly 480.

The planted 8-DPSK frame was not recovered until this was widened, and is
recovered after. `test_p2.py::test_symbol_period_at_every_dpsk_order` fails on
three of its eight checks against the old statistic. The FEC clip's own fit is
unchanged by the widening — 476.40 samples, same phases — so §7.6's earlier rows
stand.

The punctured levels' coding chain was never the problem: SL3 and SL4 round-trip
CRC-valid noiselessly through the puncturing, the batched depuncture, the
interleaver at depths 4 and 2 and both cell layouts. What was missing was a front
end that could see them.

### 7.7 The ARQ recording does not demodulate, and is not evidence  [P: 2 clips]

Both published recordings had been treated as two negatives. Only one of them is.

At packet granularity — 72 symbols, the standard cycle's data field — with the
symbol period and phase fitted freely inside each packet, and a null built by
phase-randomising the same audio so the comparison keeps its power spectrum:

| recording | tone | packets | concentration, median / max | null, median / max | packets above the null's max |
|---|---|---|---|---|---|
| FEC | 1400 | 17 | 0.784 / 0.850 | 0.292 / 0.356 | **17 of 17** |
| FEC | 1600 | 17 | 0.838 / 0.902 | 0.297 / 0.365 | **17 of 17** |
| ARQ | 1400 | 23 | 0.321 / 0.441 | 0.281 / 0.349 | 6 of 23 |
| ARQ | 1600 | 23 | 0.310 / 0.429 | 0.286 / 0.384 | 4 of 23 |

The same instrument, the same null, one clip through it cleanly and the other
barely off its own noise floor. It is not a wrong rate: a sweep from 60 to
160 Bd over the whole clip peaks at 0.224 on one tone and 0.211 on the other, at
rates the two tones do not agree on (98.4 and 89.1 Bd). It is not a wrong band:
the two carriers are where they should be, 1403.3 and 1602.5 Hz by PSD, 199.2 Hz
apart. And it is not turnaround gaps: in-band power runs 17–24 dB above an
off-carrier reference in every 50-symbol block of the clip, with no silence.

Why it does not read is open. What is settled is that a clip that cannot be
demodulated cannot support a negative, so the ARQ rows of §7.6 are withdrawn from
the evidence, and so is everything else taken through that front end — the
speed-level progression of §1.1 and the control-signal excess of §4.

### 7.8 The evidence base is one 12.5-second clip, and that is the binding limit  [P: 1 clip]

Every negative in §7.6 rests on `PACTOR-II_FEC_c1500.wav` alone, because it is
the only recording that demodulates. That is a thin base for a claim about a
protocol, and two things make it thinner.

**No decoder is known to read it.** §5.7 measures an independent decoder building
an invalid packet field for every frame of it. So there is no recording anywhere
in this project whose correct PACTOR-2 bytes are known, and a receive result can
therefore only ever rest on our own CRC — which over a search space of this size
needs a repeat, or ASCII, to be worth anything.

**There is other material, and it is better in one respect.**
`captures/p2cand_14110k_150453.wav`, captured off air by this station, is 116 s
against 12.5, and through the instrument of §7.7 its high carrier clears the null
in 23 of 28 packets. Measured per packet on both carriers, **39 of 160 packets
have both above 0.55**, including a continuous run from t ≈ 38 s to 48 s reaching
0.96 and 0.91 — as good as the published clip.

It is not yet usable, for two reasons, and both are stated rather than worked
around. Its carriers fade against each other, so no single symbol period and
phase holds across more than a few seconds: fitted over 34 s the second carrier
falls to 0.307, over 3.5 s it reaches 0.680. A scan of it needs a front end
fitted per window, which §7.6's does not do. And it is a **candidate**, not an
identified signal: two carriers 197.7 Hz apart at ≈99 Bd with high differential
concentration is PACTOR-2's shape, but it is also PACTOR-3 speed level 1's, which
is likewise two carriers of DBPSK at 100 Bd.

Measured on it, the carrier separation does **not** corroborate §1.4: over four
spans where both carriers lock it reads 0.414, 0.346, 0.819 and 0.921 of a
symbol, which is scatter across the whole range rather than the published T/2.
That neither confirms nor refutes the published value — the phase argmax over a
3-second fading span is a blunt instrument, and the recording is unidentified —
but it means T/2 still rests on one clip and one document.

### 7.9 Anchored at the frame markers, and calibrated  [MEASURED, 2026-08-02]

§7.6's scan is blind: it offers every symbol start on the clip, so it buys one
false CRC accept per 65,536 trials and no single accept can carry anything.
Acquisition (§5.4.1) removes the need for that. `p2rx.find_markers` reads three
real markers on `PACTOR-II_FEC_c1500.wav` at 2.485, 6.0775 and 9.670 s, at 0.998
to 0.999 of the noiseless ceiling against nothing else above 0.87, and all three
carry `k` = 1, 0, 1 — speed level 0, short cycle. At an anchored start what is
left open is a few dozen labellings, not a million alignments.

The anchored scan runs off the marker times `find_markers` reports. The marker
end that correlator reports leads the first data cell by **2.3 symbols**, calibrated on synthesized
frames at two pad lengths (2.375 and 2.250, an analysis-frame quantum apart); the
shift window spans several symbols either side of it, and the two short plants
below land at +1 and 0.

**The constellation rotation is per lane.** `normalise` puts each carrier's
alphabet on its own canonical axis and the two carriers have independent phase
offsets, so one shared rotation cannot serve both — a planted DQPSK frame is not
recovered with a shared one, which is how this was found. §5.12's level-1 arm
already ran it that way ("all sixteen pairs of quarter-turn rotations").

**Every arm is plant-validated first.** SL1 and SL2, short and long, built by
`pactor2`, transmitted at the clip's measured 100.77 Bd with a 7.3 Hz carrier
offset and noise, found through `find_markers` rather than at a handed-over start,
and recovered **byte-exact** at their anchored starts, four for four.

| arm | trials | expected false | CRC-valid |
|---|---|---|---|
| SL1 short — the level the descriptor carries | 3,456 | 0.053 | 0 |
| SL2 short — the order the audio measures (§5.12) | 13,824 | 0.211 | 1 |
| SL1/SL2 long cycle | 11,520 | 0.176 | 0 |
| SL1/SL2 short, lane lag ±1 | 34,560 | 0.527 | 0 |
| **total** | **63,360** | **0.97** | **1** |

The one accept is a 15-byte non-ASCII body at 2.485 s, SL2 short, shift +2. It is
a **singleton**: the same enumeration at ±1 symbol either side of it rejects, and
it recurs at neither of the other two anchors. One accept against 0.97 expected is
where chance puts it. The long arm is short of 17,280 trials because a 320-symbol
frame at the third anchor runs off the end of a 12.512 s clip.

Levels 3 and 4 are not scanned. §5.12 measures **M = 4 in 34 of 34
packet-carrier pairs** on this clip, so 8-DPSK and 16-DPSK are excluded by the
waveform; scanning them would buy false-accept exposure against a hypothesis the
audio already refuses.

**And an arm with no false-accept budget at all.** The CRC tests sixteen bits.
Re-encoding what the trellis returned and correlating it back against the softs it
was given tests every coded bit, and needs no budget, so it can be offered as many
starts as one likes:

```
metric = Σ_k s_k·(1 − 2·c_k) / Σ_k |s_k|,   c = encode(viterbi(s))
```

The null is the identical sweep on the same window with its **symbols permuted** —
same constellation, same amplitudes, no code. Over ±40 symbols around each anchor,
81 starts per anchor, 40 null draws:

| path | anchor | best | median | null max | null median |
|---|---|---|---|---|---|
| SL1 short | 2.485 | 0.816 | 0.749 | 0.826 | 0.757 |
| SL1 short | 6.078 | 0.817 | 0.756 | 0.799 | 0.762 |
| SL1 short | 9.670 | 0.815 | 0.750 | 0.768 | 0.733 |
| SL2 short | 2.485 | 0.905 | 0.880 | 0.900 | 0.880 |
| SL2 short | 6.078 | 0.922 | 0.884 | 0.905 | 0.882 |
| SL2 short | 9.670 | 0.918 | 0.880 | 0.908 | 0.874 |

The two maxima are not comparable — the real arm draws 81 samples and the null 40 —
so read the medians, which lie within 0.017 of the null's on every row — below it
on two of the six and equal on a third.

**The instrument is calibrated, and that is what makes this a result.** A planted
SL2 frame demodulated **no better than this recording** decodes outright. The
clip's anchored windows concentrate at 0.802 and 0.842; a plant at 1.5 dB
concentrates at 0.807 and 0.814, and returns metric **1.000**, null max 0.895, and
the payload byte-exact through the CRC. The same holds at 2.0, 2.5, 4, 6, 8 and
12 dB, and for SL1 down to 1 dB. So a conforming frame at this recording's own
demodulation quality would have been found by this instrument, at the starts
acquisition hands over, without a shift of the clock.

What that leaves is a two-way split, stated as such. Either the material does not
carry a frame this chain can read — which is what §5.7's independent decoder
finding no valid field on the same clip already says from the other side — or
there is an axis the search does not span. It is no longer SNR, timing, the front
end, the carrier assignment, the interleaver, the whitener, the geometry or the
alignment: every one of those is now closed by measurement, and the labelling
choices that were left — polarity, cell layout, differential sense, carrier
pairing, lane lag — are enumerated exhaustively above.

**The anchored frames do not repeat, so nothing is combined.** §5.9's 362-symbol
period is corroborated exactly by the marker spacing — both gaps are 3.5925 s =
361.998 symbols on the measured clock — but it is not the data field repeating.
Correlating the three anchored 72-symbol windows against each other, with the
per-lane rotation fitted by coherent agreement (never by a CRC, so it costs no
trials) and a null built by rolling one copy against the other:

| | frames 0-1 | 0-2 | 1-2 |
|---|---|---|---|
| upper carrier, M = 2 | 0.049 | 0.155 | 0.171 |
| lower carrier, M = 2 | 0.087 | 0.075 | 0.226 |
| upper carrier, M = 4 | 0.053 | 0.076 | 0.186 |
| lower carrier, M = 4 | 0.091 | 0.117 | 0.240 |

Every one of the twelve is below its own rolled null, which runs 0.227 to 0.325.
So what recurs at 362 symbols is the cycle structure, not the packet, and §8.1's
reading — that a repeat helps this receiver find its clock and not its bits — is
corroborated from the signal as well as from the decoder.

---

### 7.10 Real speed-level-3 material, and what it measures  [MEASURED, 2026-08-02]

Three off-air recordings of one 7.051 MHz keyboard QSO — `hb9ak_055156`,
`_055208`, `_055246`, shifted so the carriers land at 1400/1600 Hz — carry
**speed level 3 throughout**, with an independent monitor reading 9, 26 and 30
frames off them and printing the text. That is the first PACTOR-2 material in
this project with a known answer, and it moves several things out of the
inferred column.

**Speed level 3 is 8-DPSK, measured.** On `_055246`, over the 72 symbols after a
frame marker, the differential phase concentrates at **C8 = 0.90–0.94** with
**C1, C2 and C4 all at 0.01–0.16**. An eighth-power-only concentration is what an
eight-phase alphabet gives and what nothing else does, so the ladder's third rung
is read off the air rather than off a table.

**The data field needs the shipped kernel, not a boxcar.** This is the fault that
had held receive. Every scan before this one integrated one symbol of baseband at
the carrier — the textbook DPSK detector — and §5.1 had already measured that
detector recovering **0.36** of the frame-marker correlator's output where the
shipped window gets **1.00**, because the transmitted pulse is shaped and spans
some three symbols. The same ratio applies to the data: demodulated with a boxcar
on a per-window fitted clock and grid phase, these fields concentrate at
**0.33–0.55**; through `p2rx.bin_phasors` they concentrate at **0.90–0.94**,
which is what a plant at 10 dB reads. A frame that is 3 dB-equivalent in one
front end and 10 dB-equivalent in the other is the difference between decoding
and not.

**The clock is nominal, and the marker grid is the data's.** The strong markers
sit on an exact 2.5 s grid across 40 s, so capture and transmitter agree to
better than 5e-5 and the symbol period is 480 samples at 48 kHz;
`measure_symbol_period` fitted 481.92 over the whole file, which is a fit to the
gaps and the second station as much as to the bursts. Bursts run every 1.25 s,
0.82 s long, 32 of them.

**The frames repeat, and the symbols are recovered faithfully.** Correlating the
76 symbols after each marker between every pair of bursts — `|mean(a·conj(b))|`,
which a constant rotation leaves alone, so there is nothing to fit — gives
**0.94–0.975** for many pairs against **0.35** for chance. Both carriers group
the bursts identically, into families of up to seven. Nothing that mangled the
symbols would make two bursts 40 s apart agree at 0.97.

**And the repeats measure the field's geometry directly.** Two families whose
symbols agree everywhere except a short run put every differing cell in four
blocks at symbols **16–17, 34–35, 52–53 and 70–71** — 18 symbols apart. That is
108 channel positions, so the field is **six positions per symbol in symbol-major
order**, 432 of them, and the interleaver is the **depth-4 block interleaver**
whose four columns are contiguous 108-position blocks. The differing run is
**48 positions** = 12 in each column, at the END of each column: 48 punctured
positions is **32 trellis steps at rate 2/3**, and 32 steps at the end of a
terminated K = 9 trellis is the last 24 information bits plus the 8 flush bits —
the **status byte, the two CRC bytes and the flush**, which is exactly what two
copies of one idle packet differ by. A second pair, whose payload also differs in
its last two bytes, puts its blocks one symbol earlier, as it must.

Six independent things are pinned by that one pattern, all from the signal:
symbol-major layout, six positions per symbol, interleaver depth 4, the column
direction (the run lands at the end of each column, not the start), the gather
direction (a contiguous code run maps to contiguous channel runs, which is
`channel_of_code` and not its transpose), and status-then-CRC-last at rate 2/3
over 288 steps with 8 flush bits.

**And it decodes.** `p2rx.decode_bursts` now returns 31 CRC-valid fields from
the 32 bursts (29 when this section was first written; §7.13), and **all 31 are
byte-exact** against the fields an independent decoder's CRC accepted for the
same bursts — Hamming distance 0 of 280 graded bits, thirty-one times over.
~~The twenty-ninth, at 0.650 s, differs in 11 bits and is counted as the false
accept it is.~~ **RETRACTED, 2026-08-03:** the 0.650 s decode was byte-exact all
along; the truth it differed from was mis-parsed (§7.14). The run offers 8,192
alignments, which buys 0.125 expected false CRC accepts. A CRC accept is worth
sixteen bits and is therefore not the result; 280 bits of agreement are.
The comparison runs through the shipped receiver, and `rf-corpus/regress` carries
the recording as `pos_p2_sl3_hb9ak`, graded against those bytes.

That was 14 and 13 when this section was first written, every one of them on an
even cycle; §7.11 is the other half.

**Two faults had held it, and the exhaustive sweeps are what a wrong chain looks
like.** The first is the boxcar demodulator above. The second is the puncture's
PHASE: Annex I prints `1011`, and starting that at trellis step 0 gives
`((1,1),(0,1))` where the truth is `((1,0),(1,1))` — the same four-position
pattern rotated one place right. A printed vector fixes the pattern, not where it
starts, and no other rotation returns a single frame. One convention came with it:
`channel_buffer` takes a cell's three bits LEAST SIGNIFICANT FIRST, the order the
field's bytes are packed in. That is why sweeping every one of the 40320 maps from
a symbol's eight phases to its three bits came back at chance, and why 720
arrangements of the six channel positions did too: they were all run against a
chain whose puncture was one place out, and no labelling repairs that.

**What is still open.** Three bursts of the 32 return nothing in either carrier
arrangement (§7.11). And the 8-DPSK alphabet's ORIGIN is still inferred: receive
enumerates every half-sector rotation, so it costs nothing there, but an SL3
transmitter has no measured alphabet behind it.

**Every arm is plant-validated first.** An SL3 frame built by `pactor2`,
transmitted with a 7.3 Hz carrier offset and noise, found through
`p2rx.find_markers` rather than at a handed-over start, is recovered byte-exact
at its anchored start at 20, 14, 10 and 8 dB, through the same `bin_phasors`
front end.

### 7.11 The two carriers exchange tones every ARQ cycle  [PUBLISHED; the mechanics MEASURED, 2026-08-02]

**SCS states it outright for PACTOR-2.** [SCS] §5: *"In the PACTOR-2 system, the
transferred information is swapped from one channel (tone) to the other in every
cycle. Unlike FSK systems, the link is thus not blocked when strong narrow band
QRM completely overpowers one channel (e.g. CW or carriers), but only its maximum
speed is reduced."* So the swap is read, not inferred, and PACTOR-3's
`spec.CARRIER_SWAP` is the same mechanism on eighteen carriers rather than two —
which is also why `spec.SUBBAND_LEAD` is indexed by a carrier's rank and not by
its tone.

What the recordings supply is everything the sentence does not: that the swap
takes the channel RANK and the T/2 STAGGER with it, that the two changes are one
fact, and that acquisition can read which arrangement a burst is in. The
alternate bursts of §7.10 — the half that returned nothing at either differential
sense — are the swapped cycles.

**It is one fact that appears as two changes, and neither works alone.** Reading
the field with the stagger reversed but the lanes in their home order, or with the
lanes swapped but the stagger where it was, is not half right — it is nothing.
On the 32 bursts of `_055246`:

| what the demodulator reads                       | CRC-valid fields |
|--------------------------------------------------|-----------------:|
| home arrangement (lower tone leads, upper first)  | 14 |
| stagger reversed alone                            |  0 |
| lanes swapped alone                               |  1 |
| **stagger reversed AND lanes swapped**            | **15** |

The 14 and the 15 are disjoint sets on alternating cycles, and no burst decodes in
both arrangements.

**Acquisition reads which arrangement a burst is in.** A frame marker rides the
same two virtual carriers the data field does, so the swap moves its two
codebooks and its stagger together and a correlator pinned to the home
arrangement is deaf on every other cycle — which is why §7.10 measured these
recordings' marker spacing as 2.5 s where the link runs a 1.25 s cycle. Scoring
both arrangements separates them cleanly. Over the 30 anchors of `_055246` that
arm at all — two of the 32 fade below the threshold in both arrangements —

    arrangement the burst is in      0.966 - 0.999
    the other arrangement            0.520 - 0.660

and the arrangement the marker names is the
arrangement that decodes at **all 29** anchors that decode at all. `find_markers`
therefore reports it, and `decode_bursts` takes it rather than offering the
decoder a second alignment: reading both arms per burst would double the scan from
8,192 alignments to 16,384 — 0.125 expected false CRC accepts to 0.25 — to recover
bursts acquisition has already placed. That is how `p3rx.header_of` reads
PACTOR-3's swap as well.

**The alternation is strict and per-cycle**, so it is one bit for a recording
rather than a reading per burst: normal and swapped alternate without exception
across every anchor of all three recordings that arms at all, and `burst_grid`
votes on the parity of grid index 0 and carries it from there. That covers the two
anchors of `_055246` whose marker fails to arm in either arrangement.

**What it returns, across the three recordings of the QSO.** The independent
monitor reads 9, 26 and 30 frames off them:

| recording  | bursts | home arrangement only | arrangement read | independent monitor |
|------------|-------:|----------------------:|-----------------:|--------------------:|
| `_055156`  |      9 |                     5 |            **9** |                   9 |
| `_055208`  |     27 |                    13 |           **26** |                  26 |
| `_055246`  |     32 |                    14 |           **29** |                  30 |

`_055156` grades **9 of its 9** byte-exact against the union of the reference runs:
the 4.409 s burst that first "matched none of them" carries a field the one
archived run had missed, and the reference payload that "matched no burst" was one
that decoder's own CRC had REJECTED (§7.14). `_055246` grades 31 of 31.
`_055208` has no byte-level reference, so its 26 is an agreement of counts and not of
bytes. `rf-corpus/regress` carries `_055156` as the `pos_p2_sl3_hb9ak_055156`
fixture beside the longer one; pinning the receiver back to the home arrangement
takes them to 14 and 5 lines, below both fixtures' gates.

**Which arrangement §1.4's published sentence describes is INFERRED.** "Symbols on
the carrier with the lower frequency always appear delayed by T / 2" names one
sign; the home arrangement measured here is the other one, with the lower carrier
leading, and the swapped arrangement matches the sentence. On a link that
alternates every cycle both signs are on the air, so §1.4's "the sign does not
transfer" need not mean either source has it wrong. Nothing in receive rests on
this: the arrangement is read per burst.

### 7.12 Transmit is read by an independent decoder — the gap was the constellation origin  [MEASURED, 2026-08-03]

An independent monitor now reads PACTOR-2 **transmitted by this package** at speed
levels 1, 2 and 3, and prints the payload. The 32 bursts of
`hb9ak_055246_c1500.wav` re-rendered field for field by `pactor2.data_burst` —
same anchors, same arrangement, nothing of the recording's audio in them — come
back as the same payload stream that recording's own audio produces through the
same monitor: *is a / pointless / exercise / as / they / collect / every week /
of the / year! / the / moment with a*, in order, 32 frames of 32. The one payload
missing from the replica is the one carried by a burst our receiver cannot read
out of the recording, so there was nothing to re-render there.

**The whole gap was where each level's differential alphabet starts.** Every other
convention in the chain was already graded by a decode; an origin cannot be,
because `p2rx.decode_burst` enumerates every half-sector rotation of each carrier
and `p2rx.burst_window` measures the data grid from the marker time our own
transmitter put there. Transmit and receive cancelled a constant turn of the
alphabet and the round trip stayed green while the waveform was wrong.

| level | origin as this package emitted it | measured origin | how |
|---|---|---|---|
| SL1 | 45° / 225° (`[SCS-P4]` §11.7 as printed) | **135° / 315°** | eight transmissions 45° apart, read at 90/135/180, silent at 45/225/270/315/0 |
| SL2 | 225°, 315°, 135°, 45° for dibits 00/01/10/11 | **+270°: 135°, 225°, 45°, 315°** | four transmissions 90° apart, then four at 22.5°: read at 247.5/270/292.5, silent at 225 and 315 |
| SL3 | `gray[c] ↦ −2π(c+1)/m` | **`gray[c] ↦ −2πc/m`** | measured off air against the frame marker, then confirmed through the monitor |
| SL4 | — | the SL3 rule, **unmeasured** | no level-4 material exists here |

Each window is the demapper's own acceptance region, so its edges are the answer
rather than a fit: DBPSK accepts within 90° of its axis and the open window is
exactly (45°, 225°), which puts the axis at 135° with no room either side; DQPSK
accepts within 45° and the open window is exactly (225°, 315°). The published π/4
of §11.7 is **the DBPSK decision boundary itself**, which is why an earlier feed's
channel softs came back three-quarters ZEROED rather than inverted — a symbol on
the boundary decides nothing. What survives from the page is the geometry: odd
multiples of 45°, one Gray step per sector, which all three measured alphabets
keep.

SL3's origin was measured off the air first, with no outside decoder in the
loop. The
recording decodes byte-exact, so the cell value behind every data symbol of a real
transmitter's frame is known; the marker's eight chips are known too, and they
measure the rotation a carrier offset puts on every one-symbol differential, which
is the only thing between a differential phase and its alphabet. De-rotate by it
and each of the eight cell values lands on one phase at |r| = 0.995–0.997 over
183–364 symbols, one whole sector from what this package emitted.
`test_p2.py:test_dense_alphabet_origin_is_measured_off_air` is that measurement,
and `test_p2_oracle.py` is the monitor gate. Against the old alphabets that gate
reads **no level at all**.

**Speed level 4 is still not read from outside, and its origin is not what stands
in the way.** [MEASURED, 2026-09-17] Sixteen turns of the 16-DPSK ladder, four
cycles each, inside a link the same monitor was reading at levels 1, 2 and 3:
none of the sixteen came back, on either reading of the rate-7/8 puncture. The
same session read levels 1–3 on **both** cycle lengths and the entry packet cold,
so what is silent is one level rather than the transmitter. No PACTOR-2 level-4
material exists on this bench, so nothing can show the monitor has a level-4
decoder to begin with, and the verdict there is CANNOT SAY rather than a
failure.

### 7.13 The last graded gap was the receiver's own edges, not the channel  [MEASURED, 2026-08-03]

Three receiver edges capped the graded decode of `_055246`, each a receiver fact
and none of them a threshold:

- *The comb phase belongs to the anchor.* Both carriers ride one transmitter
  clock, so the burst grid fixes each lane's analysis comb outright; a per-lane
  concentration search agreed with that phase on every strong burst and betrayed
  the weak ones — at 33.15 s it put the faded lane three frames off its own
  clock, a 3/8-symbol error that read as unexplainable phase noise (agreement
  0.854 where the clock's own phase gives 0.921).
- *Magnitude is confidence.* The two-tone waveform fades one carrier at a time
  — at 4.40 s the upper carrier alone, at 38.15 s a sweep that takes each lane
  for a complementary half of the field — and unit-normalised softs hand the
  trellis a faded symbol's phase at full weight. `soft_bits` now reads the
  differential's magnitude, and the dense levels take the max-log metric on the
  phasor rather than a sector count.
- *A P2 trellis ends at zero.* The flush is inside `n_buf`, so both boundary
  states are the receiver's to claim; the free-boundary decoder (right for
  PACTOR-3's high levels, `coding.viterbi_decode`) let an impostor path outscore
  the true field at 33.15 s — re-encoded from zero the impostor's metric is
  698.5 against the true field's 734.4.

With the three together: **31 decodes, 31 byte-exact, 13 of 13 distinct
payloads** — 16 of 16 distinct fields counting the idle's status-byte variants —
and `_055156` at 9/9, graded against the corrected reference truth (§7.14). The
ARQ packet counter (status-byte bits 0–1) runs a strict stop-and-wait sequence
across the decodes and brackets each undecoded slot as a single transmission
acked first try, so Memory-ARQ combining is refuted for them: there is nothing to
combine. 38.15 s stays unread on both sides — its fade crosses the pair mid-field,
which no single-copy decoder in this comparison survives. How the undecoded slots
were identified and how the reference decoder was driven are in the project's
working notes, which are not part of this distribution.

### 7.14 The reference truth was corrected  [MEASURED, 2026-08-03]

The byte-level ground truth both PACTOR-2 fixtures grade against was regenerated:
whole 35-byte fields — data, status byte and the two CRC bytes — CRC-accepted
only, taken as the union of several reference runs rather than one, since that
decoder's yield on identical audio is not deterministic. The earlier truth was
incorrect. **There is no nine-byte grouping or `0x44` group marker in the
packet field.** The corrected reference contains complete 35-byte fields and
excludes CRC-rejected frames; §7.13 gives the resulting comparison.

---

## 8. What receive needs that we do not have

The coding chain is checked by the fixed vector in §5.10. The front end
implements the carrier and symbol-grid conventions in §5.12 and is checked
against reference vectors and synthetic frames.

Acquisition is closed as well, as of 2026-08-01: §5.4.1 specifies the marker,
§5.6 the front end that scores it, and both directions are measured against a
conforming receiver rather than against ourselves. §5.11's gap — a receiver
arming on our energy and taking the frame's parameters from noise — is what the
marker fills.

The specification side is closed as far as the published material goes. The code,
its polynomials, the puncturing vectors, the interleaver routine, the four
depths, the payload table, the geometry arithmetic, the CRC's register preset and
byte order, the phase alphabet and the carrier stagger are all published and all
implemented as published. The front end measures its own symbol period, gives
each carrier its own phase, removes the residual constellation angle per window,
and locks the published recording in 17 of 17 packets. The scan spans every
discrete unknown that remains — start, carrier order, differential sense,
rotation, interleaver direction and boundary, cell layout, carrier pairing,
whitening, level and cycle length — and is validated by planted frames at every
level and both cycle lengths. It returns accepts at exactly the chance rate.

What is missing is not another parameter. It is **material**:

1. **A PACTOR-2 recording whose correct bytes are known.** There is none. The one
   recording that demodulates is not decoded by any independent decoder either
   (§5.7), so nothing can grade a candidate output but our own CRC, and a CRC
   over a space this large is only believable if a hit repeats. A capture taken
   alongside a decoder that prints its payload would end this in an afternoon.
2. **A recording long enough, and stable enough, to carry repeats.** The usable
   clip is 12.5 s and holds about four frames. `p2cand_14110k_150453.wav` is
   116 s and has stretches as clean, but fades between its carriers and is not
   identified (§7.8); a front end fitted per window would make it usable.
3. ~~**Failing both, a transmit-side route.**~~ **OPEN, and now reachable.**
   shrike emits the frame marker and a conforming receiver forms a descriptor off
   it with the right parameters (§5.4.1), so its frames are acquired rather than
   mis-parameterised. What stands between that and a decoder printing our payload
   is the packet-confirm state of §7.3 — cycle counter and memory-ARQ pairing —
   not the waveform.
4. **An independent transmitter for receive testing.** A monitor can compare
   decoding results, but a known-payload transmission from another implementation
   is needed to test receive without sharing encoder assumptions.

### 8.1 Repetition and combining

A periodic burst does not establish that its payload repeats. The anchored
window comparison in §7.9 finds no matching repeated packet in the FEC clip.
The period measured in §5.9 therefore does not by itself justify combining
successive bursts before decoding.

---

## References

- `[SCS]` SCS GmbH & Co. KG, *The PACTOR-2 Protocol — A Technical Description*
  (1996). Table 1 p. 5 the payload ladder, Annex I the code and the puncturing
  vectors, Annex II the interleaver routine and its four depths.
- `[SCS-P3]` SCS, *The PACTOR-3 Protocol* (2004). Figure 2 the per-level table
  including crest factors.
- `[SCS-P4]` SCS, *The PACTOR-4 Protocol*. §5 the CRC, §11 the PACTOR-2-derived
  two-carrier speed level: §11.5 the code, §11.6 the interleaver, §11.7 the
  phase alphabet and the T/2 carrier stagger, §11.8 the symbol filter, §11.11
  and §11.12 the symbol counts and the packet arithmetic.
- `[M.1798]` ITU-R Recommendation M.1798, *Characteristics of HF radio equipment
  for the exchange of digital data and electronic mail in the maritime mobile
  service*, editions -0 (2007), -1 (2010) and -2 (2021).
- `[sigidwiki]` sigidwiki, PACTOR-2 signal page and sample recordings (ARQ, FEC
  broadcast).
- Wavecom, *PACTOR Advanced Protocols*, third-party geometry tables.
- hf-pactor, Sailer (HB9JNX), `hfkernel/fsk/pactor.c`.
