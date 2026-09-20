# PACTOR-III: waveform, coding and framing

Outbound mail delivery over PACTOR-3 through WS8EOC is confirmed by the operator
for message `9WQNVWBDLVOY`. See [implementation evidence](../../STATUS.md#pactor).
Dated experiments below describe the observations available at the time.

The PACTOR-III waveform is published as ITU-R Recommendation M.1798, and facts this
document shares with the Recommendation are cited to it. What is recorded here is
everything a conforming modem also needs and the Recommendation does not give: the
codeword tables, the interleaver recurrences and their strides, the soft-decision
mapping, the whitening transform, the acquisition references, and the geometry that
places coded bits onto the (tone, symbol) grid. The PACTOR-1 connect frame is
described in §17 because a PACTOR-III session opens with one.

Values are normative unless the text says otherwise. Where two readings of a fact
disagree, or where a value is genuinely not settled, that is said in place; §18
collects the points this document does not specify. The companion register of open
questions is `hfmodem.shrike.unknowns`.

Evidence tags — what stands behind a given claim, and how much of it — are defined
in `EVIDENCE.md`. How a link comes to be running PACTOR-III at all, as against how
PACTOR-III works once it is, is `pactor-capability.md`.

---

## 1. Channel grid and symbol timing

| Parameter | Value |
|---|---|
| Symbol rate | 100 Bd on every speed level (10 ms symbol) [M.1798 §2] |
| Channels | 18, numbered 0..17 [M.1798 §2] |
| Channel spacing | 120 Hz |
| Channel 0 | 480 Hz, so channel *n* is 480 + 120*n* Hz; highest tone 2520 Hz |
| Signal centre | 1500 Hz (480..2520 Hz, midpoint 480 + 120 × 8.5) |
| Maximum occupied bandwidth | 2.2 kHz, 400 to 2600 Hz [M.1798 §2] |
| Variable-header and control-signal channels | 5 (1080 Hz) and 12 (1920 Hz) [M.1798 §2] |
| Emission designator | 2K20J2D |
| Short ARQ cycle | 1.25 s [M.1798 §3 "Cycle Duration"] |
| Long ARQ cycle | 3.75 s [M.1798 §3 "Cycle Duration"] |

Every speed level's channel set contains channels 5 and 12, and every set is
symmetric about channel 8.5.

Each ARQ cycle swaps the virtual data carriers onto different tones for frequency
diversity: in the swapped state carrier 0 takes tone 17, 1 takes 16, 2 takes 9,
3 takes 10, 4 takes 11, 5 takes 12, 6 takes 13, 7 takes 14 and 8 takes 15
[M.1798 §2].

At a 9600 Hz processing rate, downconversion at 1080 Hz and 1920 Hz followed
by decimation by 12 gives 800 Hz: eight samples per 100 Bd symbol.

---

## 2. Speed levels and frame descriptors

Speed levels are indexed from 1 in the published table [M.1798 §3] and from 0
internally: **case = SL − 1**. Each case fixes a channel mask, a number of soft
bytes per cell (L), and a frame descriptor.

| Case | SL | Channel mask | Channels | n | L | Coded softs | Field bytes | Modulation |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0x01020 | {5, 12} | 2 | 1 | 144 | 8 | DBPSK |
| 1 | 2 | 0x054a8 | {3, 5, 7, 10, 12, 14} | 6 | 1 | 432 | 26 | DBPSK |
| 2 | 3 | 0x0fffc | {2..15} | 14 | 1 | 1008 | 62 | DBPSK |
| 3 | 4 | 0x0fffc | {2..15} | 14 | 2 | 2016 | 125 | DQPSK |
| 4 | 5 | 0x1fffe | {1..16} | 16 | 2 | 2304 | 215 | DQPSK |
| 5 | 6 | 0x3ffff | {0..17} | 18 | 2 | — | — | DQPSK |

Every soft count factors as **72 × channels × L**, the 72 being the number of symbol
rows a gather window holds. The field byte count is the CRC length; the payload it
carries is `field − 3` bytes, which reproduces the published usable-payload table
exactly (5, 23, 59, 122, 212, 284 bytes on the short cycle) [M.1798 §4].

Each case has a short-cycle and a long-cycle descriptor, selected by a cycle-length
argument (0 short, 1 long):

| Case | Short: coded / field | Long: coded / field | Interleave span |
|---|---|---|---|
| 1 | 432 / 26 | 1920 / 119 | 740 (short) |
| 2 | 1008 / 62 | 4480 / 279 | 1520 (short) |
| 4 | 2304 / 215 | 10240 / 959 | 280 (short) |

The long-cycle field sizes likewise sit three bytes above the published long-cycle
payloads, which are 36, 116, 276, 556, 956 and 1276 bytes [M.1798 §4]. A 4480-soft
long-cycle frame spans about 3.75 s of audio.

On the rate-1/2 cases (0–3) the Viterbi emits exactly one byte more than the CRC
covers — 9 bytes against a CRC length of 8, 27 against 26, 63 against 62 — so the
frame is `[info][CRC-16/X-25 LE][one unread byte]`. Only case 4 is punctured
(1720 information bits into 2304 coded bits).

### Speed-level detection

The acquisition correlator reports which of its reference sequences matched. That
index is the speed level: with the winning reference numbered *m*,

    case = (m >> 1) & 3

and the reported speed level is `case + 1`. Bit 3 of the reference index is a
separate flag carried into the demodulator setup; reference indices 8 and 9 give
`(m >> 1) & 3 = 0` but do not produce a case-0 gather.

A decoder that fails at case *c* retries the same look at case *c* + 4, so a single
look presents at most two frame hypotheses.

---

## 3. Packet structure in time

A packet is an acquisition burst followed, with no gap, by a coherent symbol stream
on the channel set:

    [acquisition burst] [pilot] [detection block] [data rows]

That is the shape a receiver acquires a LINK on. Inside an established cycle there
is no burst and the whole grid is on the air — *Packet extent on the air*, below.

- The acquisition burst and the body **cannot overlap in time**. Channels 5 and 12
  are the burst's own two bands, so starting the body even one symbol before the
  burst ends costs acquisition.
- Measured on a real SL2 packet: the data-carrier symbol grid begins less than 3 ms
  after the burst ends, and channel 5 carries exactly 176 symbols before its power
  collapses.
- Data symbols per carrier, from the frame descriptors: **176 on the short cycle,
  207 on the long cycle**. A 16-symbol long-header mode also exists.
- The pilot and detection block is **8 + 8 symbols**. Those 16 symbols are the first
  16 rows of the decoder's gather window, not a preamble ahead of it: spending them
  on a pilot and a variable header costs 16 of the frame's 72 rows.
- A lag-1 differential needs only one reference symbol, so a body that carries data
  in the detection block and shortens the pilot to a single symbol reaches 72-row
  coverage of 0.92; the residual rows fall in burst time.

### Packet extent on the air  [P+S: 1 session, plus our own levels read back]

A packet inside an established cycle is `[phase reference][8-symbol header block]
[one row per symbol]` and carries no acquisition burst: a coded grid of **81
symbols — 0.810 s** on the short cycle and 329 on the long one, from which a peer
times its answer (§7).

Every station recorded keys **one further symbol** behind the grid, and it is no
part of the coded field: at speed level 3 it is the same 14-bit word on all
fourteen carriers across seven packets, both carrier swaps, whatever the payload
and whatever the last data row carried (`11010010101010` in virtual-carrier
order, `01111000011110` on the long cycle), and at the levels above it varies
packet to packet. It is not the block's first or last header dibit, not a bit of
the channel's constant-header word, and not the last data row: two channels
carrying the same constant header carry opposite trailer bits. Their control
signals run to 22 symbols the same way.

The speed-level-1 entry packet of §17.1 keys it too. Read two ways that agree, a
matched filter and a bare one-symbol DFT against the unlit channels, on both the
reference recording and its independently encoded copy: power holds through
symbol 81 — within 0.6 dB of the packet body on both carriers — and collapses at
82, and the step onto that last symbol is a modulated **−45°** on both carriers,
where a transmitter's key-down ramp would hold the previous phase. An earlier
read of the same packet stopped at symbol 80 and was one symbol short; nothing 81
symbols long exists on this tape, so whether a receiver will take a packet
without the trailer is unmeasured either way. This station keys it at every
speed level (`placement.ENTRY_TRAILER`), and an independent monitor reads levels
2 to 6 back with it there, twenty frames of twenty at the level and field length
each was sent at.

### The gather window anchor

Row 0 of the 72-row gather window sits a fixed **19 symbols after the acquisition
burst starts**, whatever the burst's length:

| Burst length | Row offset |
|---|---|
| 37.5 symbols | 18 rows |
| 26 symbols | 7 rows |
| 21 symbols | 2 rows |
| 20 symbols | 1 row |

Two soft positions — the two channels of one row — are lost per symbol of burst
length, confirmed at five lengths. Ahead of row 0 the demodulator discards a
warm-up of **48 symbol extractions**; sample consumption begins 1206 samples before
the reported detection position. The window does not wrap: rows past 71 produce no
response.

Acquisition needs about **20 symbols of burst**. At 37.5 and 32 symbols a look
yields two detections, at 26, 22 and 20 symbols one detection with a working
gather, and at 18 symbols and below none at all.

Detection commits 0x37 = 55 events after the correlation score last clears
threshold, and the reported position is that instant less a fixed latency, so the
anchor sits a fixed distance before the last score peak.

---

## 4. The acquisition burst

The burst occupies the two bands 1080 Hz and 1920 Hz. Its reference is a table of
512 int16 values which are **32 complex 8-chip QPSK reference sequences**, 16 per
block — not 256 chips of one long sequence.

The acquisition front end mixes each band to baseband, filters and decimates
from 9600 Hz by 12, then applies a windowed 32-point FFT. The resulting
spectrum is evaluated over the candidate frequency offsets.

The reference comparison uses eight taps at one-symbol intervals. At the
800 Hz analysis rate and 100 Bd symbol rate, adjacent taps are eight samples
apart. All eight timing phases are scored.

The scorer pairs channels 2*r* and 2*r*+1 as the real and imaginary parts of FFT
bin *r* — **16 frequency bins, 25 Hz apart** — and pairs reference groups (2*m*,
2*m*+1) into one complex 8-chip reference `c_m[j]`. Each row is scored over 16
references × 4 variants and tagged `v = 4m + variant`; the variants are the two
band-to-block assignments and their reference-conjugated forms. The peak detector
stores *v*, and the decoder reads the speed level out of it as in §2.

The reported phase is a **lag-8 differential** in decimated samples, which is lag 1
in symbols. For a carrier on the band oscillator the reported phasor is
`z_r = exp(i(dphi − 90°(r − 7)))`; a 25 Hz frequency offset slides the row pattern
by one row; absolute carrier phase and carrier amplitude do not affect it (0° to
within the table's 0.7° quantisation; 2000 and 8000 counts identical).

### Generating a burst

Two carriers at 1080 and 1920 Hz, 100 Bd, whose per-symbol differential phases are
the negated arguments of the two reference chip sequences, newest chip last in time,
preceded by idle (zero-differential) symbols filling the correlator's history:

    dphi_band0[s] = −arg(c_block1_m[7 − s])
    dphi_band1[s] = −arg(c_block0_m[7 − s])

Such a burst scores 9991 against a ceiling of 10000 at the row whose phase is 0.6°;
a real off-air burst scores 9948 at the same kind of row. Silence, noise,
unmodulated 1080 + 1920 Hz carriers and random QPSK on both bands all produce no
detection.

The variant matters beyond acquisition. For case 0, reference 1 unconjugated (either
band assignment) leaves 13 fixed soft positions and a full-rank constraint set;
reference 0 leaves 15 fixed positions whose constraints impose one parity; both
conjugated forms acquire but place nothing (125–140 fixed positions).

---

## 5. Pilot, detection block and the per-tone phase spread

The transmitted symbol stream per carrier is a sequence of fixed-pattern segments
followed by data, all sharing one running symbol index:

- **Pilot**, 8 symbols, phases 0, 180, 0, 90, 180, 180, 180, 90 degrees, scaled by
  1/8. The first 16 symbols are this pattern twice.
- **Data-field reference**, 8 symbols, phases 0, 22.5, 90, 202.5, 0, 202.5, 90,
  22.5 degrees, scaled by 1/8.
- **Data**, per §3.

Every replica symbol has magnitude 4095 = 32767/8, i.e. uniform amplitude.

Every packet and every control signal is preceded by a single phase-reference
pulse, and all pulses occupy a 10 ms time slot [M.1798 §4]. The definition of that
pulse is not specified here.

A per-tone phase spread rides on the whole stream. The seed is 19 complex
unit-magnitude values, palindromic, a CAZAC/Zadoff-Chu sequence; one row's angles
are 0, −18.95, −56.84, −113.68, 170.53, 75.79, −37.89, −170.53, 37.89, −132.63,
37.89, −170.53, −37.89, 75.79, 170.53, −113.68, −56.84, −18.95, 0 degrees. The
spread advances **once per 8-symbol block**, not per symbol:

    symbol n = base[n] × seedblock[n // 8],  seedblock[b] = prod(seed_row[0 .. b−1])

with the row index taken mod 19. The resulting block rotations are 0, 0, 18.9, 75.8,
189.5, 18.9, 303.1, 341.0, 151.6, 113.7 degrees.

---

## 6. Header codewords

### Variable headers (VH)

Sixteen codewords of 32 bits, each sent as 8 symbols alternately on channels **12
and 5, channel 12 first** [P: 5 packets], coding four bits: bit 0 is the request status
(a repeated packet), bits 2 and 3 give speed levels 1 to 4 by modulo-4 logic —
levels 5 and 6 are distinguished by additionally analysing the constant headers —
and bit 4 is the cycle duration, 0 short and 1 long [M.1798 §4].

Two things in that sentence are the Recommendation's and not the air's, and both
are load-bearing for anyone reading code against this page. The carrier order is
the first: this page used to say channel 5 then channel 12, and on five real
packets the other order fits at 0.94–0.97 where that one reaches 0.49–0.55.
`p3frame.VH_ORDER = (12, 5)` is the measurement. The second is the bit numbering,
which above is the Recommendation's 1-based count: read 0-based off the word, the
fields are bit 0 for the request status, **bits 1–2** for the speed level and
**bit 3** for the cycle duration, which is what `p3frame.variable_header` builds
and what reads the reference's long cycles correctly.

The request bit does **not** identify the physical carrier arrangement. In the
September 11 WS8EOC capture, 35 short retries retain sequence 1 and VH=1 while
the physical arrangement alternates. In `PIII_Complete_1`, advancing short SL3
frames carry sequence 1/2/3/0 with VH 5/4/5/4; the first long frames carry
sequence 0/1/2 with VH 12/13/14. Bit 0 follows status-byte sequence parity in
these recordings. The transmitter derives it from that status byte, while
header acquisition retains its separately measured carrier/clock ordering.
`test_cs6_rx.py` contains portable independent short/long/retry crops;
`test_cs6_waveform.py` checks the transmitter across sequence wrap and both
physical arrangements.

Repeating a packet does not move it. W4DNA keyed one speed-level-1 field —
`512d362e3021413d`, status `0x21`, counter 1 — three times, at 21.185, 22.435 and
23.683 s of `7101k_234600.wav`, on the home,
swapped and home carrier orders, and bit 0 reads 1 on all three, the first keying
included. The Recommendation's repeat status would read 0, 1, 1 there; the tape
reads 1, 1, 1, and across `PIII_Complete_1` bit 0 equals the status byte's low bit
on all 34 CRC-valid packets. `placement.data_packet`'s
`request_status=bool(info[-1] & 1)` is that low bit, so a first keying on an odd
counter declaring itself a repeat is the Recommendation's sentence and not a fault
Measured on three keyings of one held-counter field, and on all 34
CRC-valid packets of the reference recording.

The transmitted form is the published logical code with
bit-fields [12:15] and [16:19] exchanged — the two middle hex nibbles — all other
24 bits identity; the mapping is a bijection over all 16 entries. Logical
0x1873174F transmits as 0x1871374F. That nibble swap is the interleaving between
channels 5 and 12.

### Constant headers (CH)

16-bit codewords under the same swap one nibble narrower (bit-fields [4:7] and
[8:11] exchanged; logical 0xC324 transmits as 0xC234). The full table has **18**
entries, in logical form:

    c324, f987, b1c8, f370, 801d, c328, 7c3d, d8f1, 5a3c,
    792d, 8397, 33aa, aa33, 5a3c, 823c, 073f, f798, d801

The published 16-entry list is this table without index 5 (0xc328) and index 12
(0xaa33). **0x5a3c genuinely appears twice** (indices 8 and 13): the CH7/CH11
duplicate in the published list is real, not a transcription error.

### The 8-symbol header router

An 8-symbol header structure lays a variable header onto channels 5 and 12 while the
remaining 16 channels carry their constant header. Per position it takes one byte of
the 32-bit variable header and emits four dibits with shifts [2, 0, 6, 4], indexed by
`counter & 3`:

    VHbyte = row[counter >> 2];  dibit = (VHbyte >> shift[counter & 3]) & 3

so each header byte becomes four dibits in bit order [2:3], [0:1], [6:7], [4:5].
The emission order is 2 outer blocks × 4 bit-planes × channel columns; within a
plane the two variable-header columns are reached as channel 5 then channel 12,
giving 16 dibits, the 32 header bits. The output grid is **symbol-major,
channel-inner**: channel *t* of symbol *s* at output dibit *s*×18 + *t*, modulo the
variable-header insertions; 8 symbols × 18 channels = 144 base dibits plus 6
insertions at channels 5 and 12. Reconstructing each non-variable channel's 8 dibits
as a 16-bit word (dibit bits LSB first) reproduces the published constant-header
codewords.

This 8-symbol structure is a fixed reference pattern used for detection and
correlation. It is not the field that carries the speed level, cycle and frame
number — that is the coded frame of §2, §8 and §11.

---

## 7. Control signals

Six 20-bit codewords:

| CS | Value |
|---|---|
| CS1 | 0x52D56 |
| CS2 | 0xAABA2 |
| CS3 | 0xAD45A |
| CS4 | 0x95AC5 |
| CS5 | 0x74339 |
| CS6 | 0x4B4AD |

All 15 pairwise Hamming distances are exactly 12 of 20, so the set meets the Plotkin
bound and is a perfect code — which is what makes the cross-correlation soft-decision
decoding the Recommendation describes possible. Every control signal is sent
**DBPSK on both channel 5 and channel 12** for maximum robustness, the same 20 bits
on each, so the transmitted block is 40 bits with minimum distance 24. Selection is
by index × 3 into the table, the bytes are expanded to bit symbols through a
bit-complement step, and the modulator is invoked with a bit count of 40.

The meanings are published [M.1798 §4]: CS1 and CS2 acknowledge and request packets,
CS3 forces a break-in, CS4 demands an increase to the next higher speed level, CS5 is
a NACK asking for a repetition and a reduction to the next lower level, and CS6
toggles the cycle length between short and long.

The PACTOR-1 control codewords — 0x4D5, 0xAB2, 0x34B, 0xD2C — are a separate,
adjacent table.

### Control pulse shape and physical carrier ordering  [P: 1 session, 2 stations]

Bare controls on `PIII_Complete_1` carry a phase reference, twenty DBPSK
symbols, and one trailing symbol that repeats its predecessor. The two physical
carriers are staggered by half a symbol (5 ms), with the leading carrier changing
on successive cycles. At 11.17, 13.67 and 16.17 seconds channel 5 leads; at 12.42,
14.92, 17.42 and 54.92 seconds channel 12 leads. The last two are CS6 commands.
The change in sign distinguishes this from a fixed recording-channel delay.

`placement.control_signal` reproduces the stagger and trailing repeat. Its
`swapped` argument describes physical ordering, independently of the data
header's request-status bit. The live driver projects ordering from the latest
CRC-valid packet and its observed cycle. `control_pulse_lead` accounts for filter
padding removed before playback so the leading phase-reference center, rather
than the first audio sample, lands on the reply boundary.

The nominal codeword duration used for cycle rotation remains 210 ms. Physical
pulse extent, including the runout and shaping, is measured separately. Portable
recorded-waveform comparisons and missing-stagger/runout negative controls are
in `tests/shrike/test_cs6_waveform.py`; actual SCS acceptance remains a rig test.

### The acknowledgement is the alternation

CS1 and CS2 are not two meanings. As in AMTOR and in PACTOR-1
(`pactor1-control-signals.md` §4.1) the receiving station acknowledges by
**alternating** between them, and repeating the previous codeword is how it asks
for the packet again. What PACTOR-III pins the alternation to is the status byte's
modulo-4 packet counter (§14): over the reference session's 26 answered packets an
**even counter is answered CS1 and an odd one is never CS1** — no exceptions, at
zero bit errors of twenty, across the whole speed-level ladder, both cycle lengths
and both sides of every changeover.

CS4 and CS6 substitute in that slot rather than adding a transmission: the IRS
emits exactly one codeword a cycle, so a gear command carries that cycle's
acknowledgement with it. What never appears against an odd counter is CS1.

A station joining that train out of step finds the phase in one cycle, because the
counter is the phase. A station that answers a peer with the same codeword every
cycle is telling it, correctly, to send the same packet forever.

### The answer slot  [P: 1 session, both directions]

The **answering station's** responses sit approximately **0.890 s after the
caller's packet phase reference** on the short cycle, and 3.390 s on the long
one. Sixteen short cycles of `PIII_Complete_1` give 889.4–891.9 ms, at speed
levels 1, 3 and 6 and including the entry packet of §17.1. This measurement
does not specify the caller's reply timing after the direction reverses.

The same recording has approximately **0.954 s** from the answering station's
packet to the caller's response. Independently fitted CS3-to-control pulse
references give 954.500 and 954.750 ms in that direction, versus 895.625 ms in
the other direction. These pulse fits and the older coarse header references
use different markers; neither supports applying 890 ms to both directions.
These values come from independent pulse fits to the same recorded exchange.

For the first reversal, the caller's entry header at 4.670 s and its response
at 6.521125 s differ by one 1.250 s cycle plus approximately 0.600 s. That
agrees with the packet-minus-control rotation, 0.810 − 0.210 s. Applying a
fixed peer-plus-0.890 s rule instead predicts 6.456625 s, about 64.5 ms before
the recorded response. This is a counterexample to a universal reply delay,
not a measurement of a receiver's timing tolerance. The offline reproduction
compares the recorded pulse positions with those predicted by that rule.

The original answerer-direction measurement decomposes as the 0.820 s keying of §3 — the 0.810 s coded grid and its
trailer symbol — plus a **70 ms turnaround**, beside the 100–105 ms a commercial
PACTOR-1 pair spends in the same gap. The long cycle's
3.390 s is the same geometry over the long packet.

Against PACTOR-1's 960 ms packet that is a **sign, not a tolerance**. Anchored on
the longer packet the same answers imply a turnaround of −71 to −68 ms — they
arrive before a PACTOR-1 packet would have ended — so a station reading for them
there looks at 1.00–1.09 s and no width of search brings the two together.

### The slot moves when the LINK does, not when the transmitter does  [P: 5 recordings]

The answer above is where a peer answers a packet it **read**. A peer that has not
read one goes on answering where PACTOR-1 put it, and the two recordings pin both
halves.

`PIII_Complete_1`'s peer read the entry packet: its two PACTOR-1 answers sit at
3.156 and 4.406 s, a 1.250 s raster that puts the next at 5.656, and its answer to
the entry packet is at **5.5638 s** — 890.8 ms after that packet's phase reference
at 4.673, and 92 ms ahead of its own PACTOR-1 slot. One packet, one cycle, and the
whole cycle geometry has moved.

The four arms of `captures/onair-0825-21{01,05,08,11}` — two gateways, 40 m and
80 m — are the other half. Each granted the upgrade, each was keyed four
speed-level-1 entry packets, and none of the four peers ever read one: every cycle
came back with the same `0x59A` at zero bit errors, the word a station repeats
while it waits. Their bursts sit **1031–1043, 1030–1047, 1054–1066 and
1048–1101 ms after the caller's slot boundary in the PACTOR-1 cycles either side,
and 1034–1036, 1036, 1054 and 1052–1088 ms after it inside the upgraded ones**.
The packet changed length by 150 ms and the answer did not move at all.

So the receive window follows the **link's** protocol, and an upgrade is the
interval in which the two ends disagree about what that is. Until the far end
acknowledges the entry packet, its answer is still on the raster the link was
opened on.

### CS3 is never sent alone

CS3 is the **first twenty symbols of a packet**. A station taking the channel keys a
phase reference, the codeword and a short frame as one transmission with no gap and
no re-key, in the answer slot above — so a peer reading for its acknowledgement
finds the codeword exactly where it was looking, keyed the way a bare one is keyed,
and the same 0.81 s of carrier already carries the first bytes of what the new
sender has to say. §17.1 has the geometry of the frame behind the head and what the
two measured instances carry.

A bare CS3 would leave the peer nothing to switch to receive *for*. The same holds
in PACTOR-1, where the head and the control signal carry the same twelve bits in the
same order (`pactor1-control-signals.md` §2, §4).

---

## 8. Placement: coded bits onto the (channel, symbol) grid

Channel-order values are arranged symbol by symbol, with active channels in
ascending order and `L` coded bits per channel:

    channel_index = (symbol * n_active + tone_rank) * L + j,   j = 0 .. L−1

where `tone_rank` is the channel's rank among the active channels, ascending. The
positions are contiguous. Case 4 has 72 symbols × 16 active channels × 2
bits = 2304 coded positions.

Transmit placement is the inverse of that, composed with the inverse of the
interleave of §10:

    code index --(inverse interleave)--> channel index --(above)--> (channel, symbol)

Measured, with a single-cell probe, on case 0 with a burst of *B* symbols:

    look 1:  channel = 2 * (row + B − 19) + (1 − tone_rank)      (rank mirrored)
    look 2:  channel = 2 * (row + B − 15) +      tone_rank

The two hypotheses differ by a spectral mirror, which exchanges the carrier
rank. Ascending (5, 12) order is the transmit order. Either hypothesis reaching
a valid CRC is accepted.

On case 1 the gather is a uniform shift: each grid cell drives exactly one channel
position,

    channel_position = grid_cell + 30        (30 = 5 rows × 6 channels)

verified across rows 1..50 on all six channels. Rows 0 and 1 show extra coupling
from the differential's startup, and cells whose target exceeds the 432-position
window (rows 67 and above) fall outside it and drive nothing.

---

## 9. Soft demapping and the differential convention

The per-symbol soft metric is **phase only**: the matched-filter output is reduced
to a 16-bit angle and its amplitude discarded.

    phi16 = round(atan2(I, −Q) × 10430.0596)      (10430.0595703125 = 65536 / 2π)

The negated Q is the receiver's conjugate convention, which a transmitter must
match: a symbol phase θ lands at (I, Q) = (cos, −sin)(θ + reference). The angle is
then differenced against the same channel's earlier symbol from a rotating history
(index advanced `(idx + 1) & 7` per decimated sample, i.e. lag 8 in samples = lag 1
in symbols), continuous across the field and reset only at frame start.

The difference is mapped to soft bytes through a 512-entry sine table holding one
full period, `LUT[u] = 32767 sin(2πu/512)`:

    u  = (phi >> 7) & 0x1ff
    s0 = clip(LUT[u]),  s1 = clip(LUT[(u + 0x80) & 0x1ff])       (+0x80 = +90°)
    clip(v): m = min((|v| / 2) >> 7, 0x7f)

A **positive soft byte means the coded bit is 0.**

Two-soft cells (cases 3–5, QPSK) take both samples and give

    bit0 = [sin(delta) < 0],   bit1 = [cos(delta) > 0]

so the constellation sits on the π/4 diagonals:

| Dibit | Differential phase |
|---|---|
| (0, 0) | 135° |
| (0, 1) | 45° |
| (1, 0) | 225° |
| (1, 1) | 315° |

One-soft cells (cases 0–2, DBPSK) take a single sample at
`(int16)(phase + 0x2000) >> 7 − 0x80`, that is sin(delta − 45°). Cases 1 and 2 emit
a positive soft (bit 0) when that is negative — 315° for a zero, 135° for a one —
and **case 0 inverts the rule**, emitting a positive soft when it is positive.

The data grid is differential against the **previous symbol** (lag 1): one flipped
data cell moves exactly one channel position. The lag-8 structure belongs to the
pilot and variable-header replica on channels 5 and 12; transmitting the data grid
at lag 8 makes a single cell light 8–10 channel positions and collapses coverage
from 0.90 to 0.06.

The receiver's per-channel matched filter is the matched pair of the transmit pulse:
31 taps, float32, summing to 1.0, at 8 samples per symbol. Its tap bank is selected
by speed level, one bank for SL below 2 and another for SL 2 and above, with a
second bank added above SL 3.

The header symbol stream is continuous with the data: the first data symbol's
differential reference is the last header symbol, and the per-channel phase spread
of §5 runs unbroken across the boundary.

A magnitude helper (a CORDIC-style |I + jQ|) exists for automatic gain control and
is not in the soft path.

### Measuring a burst's carrier frequency  [T: derivation, no off-air clip]

The diagonals above are also a frequency-measurement hazard, and two instruments
have already disagreed by 12.5 Hz on the same recording because of it.

**Squaring is valid on control signals and invalid on packets.** A control signal
is textbook DBPSK: its differential steps are 0 and 180°, they double to 0 and
360°, and the squared baseband carries a clean line at twice the carrier. A data
or entry packet is not: its steps are 135° and 315°, they double to 270° and 630°
— the *same* angle — so squaring leaves a deterministic 270° ≡ −90° per symbol
riding on the line. At 100 Bd that is a −25 Hz line in the squared domain and
**−12.5 Hz after halving, on every packet, whatever the payload**. It is stable to
0.02 Hz across dozens of bursts, which is exactly what makes it convincing. The
two candidates −90° and +270° are one discrete-time tone, so the reading is also
ambiguous by 50 Hz: a packet keyed 40 Hz high reads 22.5 Hz *low*, 62.5 Hz from
the truth.

**The spectral peak is not the carrier at all.** A suppressed-carrier signal has no
line there; the peak is the data's own spectral ripple and it moves 15–45 Hz, in
either direction, between two renders of the same modulation.

**What to use.** Matched-filter the burst, sample it on its symbol clock, difference
consecutive symbols, remove the data by decision *at the constellation that burst
is keyed on* — 135° for cases 0–2, {0, π} for a control, the π/4 dibits for cases
3–5 — and read the mean residual rotation per symbol as a frequency. It is
unambiguous only over ±25 Hz at 100 Bd, so a burst known to be far off frequency is
measured seeded at its coarse hypothesis and the residual added back.
`p3acquire._refined` is this estimator in the live receiver, decision-directed off
the codeword it just decoded; the coarse sweep is what puts a far-off peer inside
its unambiguous span before it is asked.

---

## 10. Interleaving

Two distinct mechanisms, selected by path.

### The walking de-interleave (cases 1–4)

    stride = STRIDE[case − 1]
    index = start = N − stride
    for k in 0 .. N−1:
        out[k] = in[index]
        index -= stride
        if index < 0: start += 1; index = start

The short- and long-cycle stride values are:

    t1 = [35, 69, 250, 563, 909, 0]
    t2 = [181, 321, 1122, 2407, 3876, 0]

Case 1 takes 35 (N = 432), case 2 takes 69 (N = 1008), case 4 takes 563 (N = 2304).
The transmit interleave is the inverse scatter, `chan[idx[k]] = code[k]`.
Case 0 uses the helical permutation below.


### The helical de-interleave (case 0)

    out[k] = in[idx];  idx += stride;  if idx > N−1: wrap += 1; idx = wrap

with idx = 0 and wrap = 1 initially. Its table is {16, 8, 4, 2} indexed by case.
Measured end to end on case 0 (N = 144), consecutive code positions draw channels
nine apart and the walk makes **16 passes of 9 elements**, so the table entry is the
interleaver *depth* and the stride is N / depth = 9. Interleave and de-interleave
compose to the identity in both directions.

### Data-field geometry

`placement.LONG_ROWS` uses 320 data rows behind a nine-symbol head. Multiplying
by each level's active carriers and bits per cell, then accounting for puncturing
and trellis flush, reproduces all six published long-cycle payload sizes:
36, 116, 276, 556, 956 and 1276 bytes.

The corresponding coded-bit counts are 640, 1920, 4480, 8960, 10240 and 11520.
Levels 3–6 decode the ten long-cycle fields of `PIII_Complete_1` CRC-valid.
The generated permutation definitions and their checks are described in
[table generation](../table-derivations.md).

---

## 11. Convolutional coding and puncturing

M.1798 specifies full-frame bit-interleaving of the whole packet followed by "an
optimum rate 1/2 convolutional code with a constraint length of 7 or 9", with the
rate-3/4 and rate-8/9 codes derived from it by puncturing, and punctured bits
replaced by null symbols at the receiver [M.1798 §3]. It does not give the
polynomials or the patterns.

### Case 0 (SL1): rate 1/2, K = 9

    256 states, v = (state << 1) | bit, next = v & 0xff
    g0 = 0o753   applied to the soft at the EVEN index of each pair
    g1 = 0o561   the odd index
    taps counted with the newest bit at bit 0: (0o657, 0o435)
    a positive soft means the coded bit is 0; the decoder maximises the metric
    8 zero flush bits: 64 message bits -> 72 trellis steps -> 144 code bits
    traceback starts at state 0 and packs the field LSB-first per byte

0o753/0o561 is the standard maximum-free-distance K = 9 pair. Encoding a field with
this trellis and handing it back to the decoder returns the field in all six cases
tried, including all-zeros, all-ones and two arbitrary fields; under MSB-first
packing only the palindromic cases survive, which is what pins the bit order. Across
fifteen live decodes the field a receiver validates equals `whiten(this model's
output)` byte for byte.

The decoder's input is 144 consecutive int8 softs, two per trellis step.

### Cases 1–3: rate 1/2, K = 7

    64 states, v = (state << 1) | bit, next = v & 0x3f
    g0 = 0o171   applied to the soft at the EVEN index of each pair
    g1 = 0o133   the odd index
    taps counted with the newest bit at bit 0: (0o117, 0o155)
    a positive soft means the coded bit is 0
    terminated; the whitened field is packed LSB-first per byte

0o171/0o133 is the standard maximum-free-distance K = 7 pair, and both the pair and
the order it is transmitted in are settled by interoperability rather than by
argument. Frames built under six candidate pairs against both bit orders were offered
to an independent PACTOR-III decoder: exactly one combination — this one — passes that
decoder's CRC and reads the chosen 26-byte field back byte-exact, against thirteen
matched negatives including a random codeword. In the receiving direction, the 26
PACTOR-III packets of `PIII_Complete_1` (§17.1) decode CRC-valid under it and under
nothing else tried — exchanging the two generators, or substituting either of the
other candidate pairs, leaves the whole 79 s with a single readable packet, the
speed-level-1 entry packet that runs on the K = 9 code above. Those 26 are speed
levels 3 to 6, so the off-air evidence covers the punctured cases as well; the session
carries no speed level 2, so case 1 itself rests on the frame offered above.

### Puncturing

Only cases 4 and above are punctured; below that the depuncture stage is a copy.
Erasures are inserted as a neutral soft (0x3fff) before the accumulate-compare-select.
The stored patterns are (period, keep-mask) pairs consumed LSB-first, 1 = keep:
masks {0x5, 0xB366, 0x1003}, periods {3, 16}.

| Level | Rate | Pattern |
|---|---|---|
| SL5 | 3/4 | period 3, G1 = [1, 1, 0], G2 = [0, 1, 1] (mask 0x5) |
| SL6 | 8/9 | period 16 mother bits, keep 9 at positions {1, 2, 5, 6, 8, 9, 12, 13, 15}; G1 = [0, 1, 0, 1, 1, 0, 1, 0], G2 = [1, 0, 1, 0, 1, 0, 1, 1] (mask 0xB366) |

Case 4 uses the rate-3/4 pattern: keep, puncture, keep repeating, so depuncture
inserts a neutral soft value at every output index *j* with *j* mod 3 == 1.
The rate arithmetic agrees: 215 information bytes are 1720 bits, rate 1/2 gives
3440 coded bits, and 2304/3440 = 0.670 = two kept of every three.

---

## 12. Whitening

    word[i] ^= reflected_CRC_CCITT_table[i]

over the information bytes as 16-bit words indexed by word position, the table laid
down **big-endian** (high byte first), the position counter reset per frame. The
table is the same 256 × uint16 reflected CRC table used for the CRC (polynomial
0x8408). The transform is its own inverse.

Measured rather than inferred: an all-zero 9-byte case-0 field is reconstructed by
the receiver as `00 00 11 89 23 12 32 9b`, which is table entries 0x0000, 0x1189,
0x2312, 0x329B high byte first. The whitening is applied to the field before the CRC
reads it, so a transmitter whitens first and the receiver's own pass restores the
field.

This stage is not described in M.1798, and a coding chain without it does not decode.

---

## 13. CRC

**CRC-16/X-25**: polynomial 0x1021, init 0xFFFF, RefIn true, RefOut true, XorOut
0xFFFF, good-frame residue 0xF0B8, check value 0x906E. The two CRC bytes are
appended little-endian, and the verifier tests the residue.

The implementation is a 256 × uint16 reflected table (polynomial 0x8408). The same
CRC serves every case — field lengths 8, 26, 62, 125, 215 — and the PACTOR-1 path
shares it.

Duplicate-frame detection is separate and is not a CRC verdict: it reads the field's
trailing two bytes, compares them with the previous frame's, and reports whether
they differ.

---

## 14. Field layout, status byte and payload types

A field is

    [payload bytes][status byte][CRC-16/X-25, little-endian]

so the reported length is `field length − 3` and the payload sizes are the published
ones. The status byte carries a modulo-4 packet counter in bits 0 and 1, the data
type in bits 2 to 4, a cycle-length suggestion in bit 5, a changeover request in
bit 6 and the QRT protocol in bit 7 [M.1798 §4], so the type is `(status >> 2) & 7`:

| Type | Meaning |
|---|---|
| 000 | ASCII, 8-bit |
| 001 | Huffman, normal |
| 010 | Huffman, swapped (upper case) |
| 011 | reserved |
| 100 | PMC German, normal |
| 101 | PMC German, swapped |
| 110 | PMC English, normal |
| 111 | PMC English, swapped |

Header fields carry a fixed template rather than user data. A 62-byte header field
observed in full reads:

    field[0:59]   walking-bit template, period 15:
                  0f 8f 87 c7 c3 e3 e1 f1 f0 78 78 3c 3c 1e 1e
                  (consecutive XOR of the bytes gives single walking bits)
    field[59]     FRNR (frame number)
    field[60:62]  CRC-16/X-25, little-endian

CRC-16/X-25 over field[0:60] gives 0xa90a, stored as `0a a9`, and the residue over
all 62 bytes reflects to 0xF0B8. The case-0 field is the same template truncated to
five bytes, then the status byte, then the CRC: a real station's entry packet
(§17.1) reads `0f 8f 87 c7 c3 1a 66 89`, status 0x1a, so counter 2.

Type 0 is raw text; types 1, 2 and 4–7 are compressed, and real PACTOR-III traffic
is compressed: the reference session of §17.1 runs its data train at type 6 and its
mailbox greeting at type 7, both PMC English. The compressed formats are not
specified here.

Nothing carries a length: the field is the fixed size the speed level and cycle
length fix, both ends know it, and a station that has fewer bytes than that pads
the field out. What it pads with is not published; §18 records that.

---

## 15. The data field: cross-tone spreading

The data field is **not** independent per-channel streams. Each transmitted symbol
is a linear combination across the data channels, which is why a per-channel
demodulation of the data channels lands about 38° off the 0/180 grid at every symbol
rate while channels 5 and 12 demodulate cleanly to within 4–6°.

The spreading sequence is Zadoff-Chu, N = 19 (prime), root 1:

    a_k = exp(−j π k (k+1) / 19),   k = 0 .. 17

stored as 19 complex int16 values scaled by 32767 with a (0, 0) pad, matching the
closed form to 2e-5. The spreading matrix is the 18 × 18 circulant
`M[i, j] = a[(i − j) mod 18]`; over a level's data channels the active operator is
the corresponding square submatrix, and the receiver applies its inverse. De-spread,
the SL2 data channels demodulate to 9.7° clean QPSK.

For SL2 the measured chain is:

| Stage | Value |
|---|---|
| Data channels | {3, 7, 10, 14} (channels 5 and 12 are the pilots) |
| Spreading | the 4 × 4 submatrix of M |
| Constellation | absolute QPSK, 2 bits/symbol, b0 = sign(I), b1 = sign(−Q); a positive soft is bit 0 |
| Symbol rate | 100 Bd (480 samples at 48 kHz) |
| Field | 88 symbols × 4 channels × 2 bits = 704 coded bits |
| Interleaver | the 704-entry table |
| FEC | rate-1/2 K = 7 Viterbi, unterminated: 352 bits = 44 bytes |
| Then | de-whiten, CRC-16/X-25 |

This decodes a real SL2 packet CRC-clean, giving identical information bytes at
three window offsets, and the inverse transmitter round-trips byte-exact through
audio. Rectangular symbol pulses round-trip cleanly through an integrating
demodulator; raised-cosine shaping introduces inter-symbol interference that it does
not follow.

The 44-byte frame this produces is not the 26-byte SL2 field of §2, and which frame
of the packet it is is not settled here.

---

## 16. Session behaviour

- Packets are **phase-coherent and cadence-locked** across a session: a receiver
  tracks the 1.25 s cycle continuously from the connect. A 1.6 s gap inserted into a
  recording stops the decode at the gap, though the audio after it is valid; a single
  extracted cycle repeated at 1.25 s does not decode either, because copying one
  cycle resets the inter-packet phase.
- Successive packets are not byte-identical: only one preamble of a real session
  cross-correlates with a template taken from another.
- **Memory-ARQ.** Frames are recovered from redundancy across looks, so replacing or
  silencing one occurrence of a packet does not remove its content from the decode;
  it costs later frame numbers.
- Each look anchors its own 72-row window to its own detection, so rows before the
  anchor always fall on that packet's own acquisition burst.
- A short-cycle packet fits as burst + 8 pilot + 8 detection + 66 data rows = 82
  symbols = 1.2337 s, leaving 16 ms of margin in the 1.25 s cycle. Seventy-two data
  rows do not fit in one short cycle.

---

## 17. The PACTOR-1 connect frame

A PACTOR-III session opens with a PACTOR-1 connect. The frame is a **single
dual-rate FSK burst**, continuous phase throughout including across the rate change:

| Section | Content | Rate | Duration |
|---|---|---|---|
| Address | 9 bytes | 100 Bd | 0.720 s |
| Redundancy | 6 bytes | 200 Bd | 0.240 s |

96 bit periods at each rate, 0.960 s in total, and nothing follows the redundancy
section. MARK (bit 1) is 1400 Hz and SPACE (bit 0) is 1600 Hz; bits are sent
LSB-first. Real off-air frames measure 0.964 and 0.965 s of key-down including their
own envelope edges.

The address image is

    T = [0x55] + ASCII(callsign) + 0x0F fill to 9 bytes

and goes on air as-is: for DL6MAA, `55 44 4c 36 4d 41 41 0f 0f`. The redundancy
section is `T[1:7]` — the callsign again, literally — `44 4c 36 4d 41 41`, so a
receiver's cross-check `secondary[i] == primary[i+1]` holds by construction. The
callsign is plain ASCII, each character validated as 0x2C < c ≤ 0x5A, and there is
**no frame CRC**: the character-validation path writes the callsign with no CRC gate.

A receiver byte-locks one bit early and rotates each received byte,
`T[i] = (raw[i] >> 1) | (bit0(raw[i+1]) << 7)`, before matching. That rotation is the
receiver's alignment; a transmitter does not pre-compensate for it. Envelope onset
does not settle the question either way, because stations key up on an idle MARK
carrier ahead of the frame — about 1.4 bit periods on one measured station — so onset
measures keying rather than encoding.

Three connect variants exist: Normal Call, Longpath Call and Robust Call.

Full adjudication, including the reference receiver and the measurement series that
settled the on-air bit phase, is in `pactor1-link-request.md`.

---

## 17.1 The PACTOR-1 → PACTOR-3 upgrade  [P: 1 session, 79 s, 2 stations]

The connect carries no capability field, the connect answer is one bit and it is a
rate rather than a mode, and no codeword in the PACTOR-1 table means "I cannot"
(`pactor-connect-frames.md`, `pactor1-control-signals.md` §5). What does carry a
station's ceiling, and what the published record does and does not say about it, is
`pactor-capability.md`; this section describes one recorded transition and not the
general mechanism.

The transition happens afterwards, on the established PACTOR-1 link, and in this
recording it is three transmissions long: the caller announces, **the answering
station commands the change**, and the caller keys a speed-level-1 entry packet in
its next transmit slot. The station that ends up receiving PACTOR-3 is not surprised
by it — it ordered it.

Measured off `rf-corpus/PIII_Complete_1`, the whole 79 s of one DL6MAA session
(`ref_occ15` is its first fifteen seconds and holds the transition). Times are
phase references rather than envelope onsets:

    1.0, 2.0 s   PACTOR-1 connect bursts, address DL6MAA
    3.157        PACTOR-1 CS1, zero bit errors — the connect answer, 200 Bd
    3.387        the caller's one PACTOR-1 data packet — the announcement
    4.412        0x59A in the answer slot — the grant
    4.6725       the caller's SPEED-LEVEL-1 ENTRY PACKET
    5.5637       CS3 BREAK-IN — the answerer takes the channel
    6.820        the answerer's greeting, 212 bytes at speed level 5:
                 "C-II DSP/QUICC System - Maildrop QRV"
    9.0225 on    the caller's data train, speed level 3, one packet a cycle

The greeting carries a changeover request (§14, status bit 6); the caller answers
CS3 in its turn and takes the link back, so the entry is bracketed by two
changeovers.

### The announcement  [T: inference from 2 stations and our own record]

The transmission a grant answers is an ordinary PACTOR-1 data packet, and what
distinguishes it is **bits 4 and 5 of the PACTOR-1 status byte** — the two the 1990
description leaves *"noch nicht belegt"*. Both third-party stations observed being
granted PACTOR-3 set bit 5: `0x35` in this session and `0x31` at W4DNA. This
station's own record says the same by its absence — every session of its 269 that
drew a grant is dated to the period when it set both bits, and none has arrived in
any session since it stopped.

That the bits *are* the capability field is inference from a correlation, not a
reading of a specification. The competing account of bit 4 is that it is the top of
a three-bit data type, which is how a modern monitor reads it — so a station that
sets both misdeclares its payload type, and is decoded correctly regardless.

### The grant

`0x59A` is one of the twelve-bit words tabulated beside the four PACTOR-1 control
signals, to which PACTOR-1 assigns no meaning (`pactor1-control-signals.md` §3). It
arrives where the codeword answering the announcement packet would, it acknowledges
that packet, and the granted station's very next burst is PACTOR-3 — **one
turnaround later, not one cycle**. Neither station moves its slot timing to do it.

A second exchange says the same thing away from this session:
`rf-corpus/pos_pactor1_local`, recorded at a different gateway on a different day,
carries `0x59A` at 12.443 s and PACTOR-3 carrier energy about 0.16 s behind it.

A station waiting for an entry packet repeats the word once a cycle for as long as
it waits. A run of the word on the raster is therefore not a silence — it is the
peer still asking, once a cycle, and it says that whatever the peer was waiting for
had not arrived by the last of them.

**That is all it says.** An earlier revision of this section read the run as "the
peer's PACTOR-3 receiver is open and it has read nothing keyed at it", which turns
the peer's *emission* into a claim about the peer's *reception*, and no recording
here carries that step. Only a fraction of what this station keys is known to arrive
at a gateway at all — in the one session where both directions were caught by a
third receiver, this station's own reader took sixteen of the thirty-six codewords
the gateway sent while both ends were on the channel — and in that regime "its
receiver was open and read nothing" and "little of what we keyed reached it" are the
same observation. Separating them needs a recording made at the far end, and there
is none.

### The entry packet

**Speed level 1 is an entry level, not a traffic level.** Its field is eight bytes,
five of them payload, so a link that stayed there would carry five bytes a cycle.
What it buys is acquisition: it lights only channels 5 and 12, which are the two
the acquisition burst, the variable header and every control signal already use
(§1), so a receiver tracking a PACTOR-1 raster can find it cold. The real IRS did:
it acquired that burst and answered it in the turnaround.

A monitor is not that receiver and cannot stand in for one. Speed levels 1 and 2
carry none and four of the sixteen constant headers, so nothing locked to constant
headers ever sees an entry packet; the 32-bit **variable** header is what places
it, on the two carriers every level lights.

Measured on the reference session's entry packet:

- **82 symbols** — one phase reference, the header block, 72 rows, and the
  trailer symbol every recorded station keys (§3) — 0.820 s.
- **Channels 5 and 12 only**, every other channel at the noise floor.
- **Nothing keyed ahead of it.** The two carriers come up within half a symbol of
  the phase reference, 4.7 ms before it, against 65:1 between packet and gap —
  where the same station's PACTOR-1 packets key idle carrier ahead of their frame
  (§17).
- **In the next transmit slot, on the raster already being kept.** Its phase
  reference falls 1.2854 s after the key-down of the PACTOR-1 packet it answers,
  against that station's own 1.2493 s raster — part of the excess being the idle
  carrier a PACTOR-1 station keys ahead of its frame and this packet does not.
- The field is a **header template**, not user text: `0f 8f 87 c7 c3 1a 66 89` —
  the walking-bit template of §14 truncated to five bytes, the status byte, the
  CRC.

That field is also what settles the case-0 chain (§9, §10, §11) against a station
that is not ours. It decodes CRC-valid at five consecutive eighth-symbol alignments
and nowhere else in 79 s; its variable header reads the published speed-level-1
word, unswapped, short cycle, at 0.977 over the two carriers; and its five payload
bytes are the first five of the repeating pattern the same station's speed-level-3
packets carry four seconds later, through a different code, a different interleave
and a different anchor.

### The answer is a break-in, and a break-in is an answer

The entry packet is answered **CS3 BREAK-IN**, in the turnaround an acknowledgement
would have come back in — 891.2 ms after the phase reference, the ordinary answer
slot of §7. It is not an acknowledgement: the IRS reads the entry packet, decides
its own greeting comes first, and takes the channel.

It is still an answer, and it carries everything the entry packet was keyed to
learn — that the peer has the waveform. A station that treats only an
acknowledgement as proof of the upgrade runs its whole PACTOR-3 phase at five bytes
a cycle, having been told in the first turnaround that it could stop.

One entry packet is one observation and not yet a rule about entry packets. What it
settles is the direction of the inference: a break-in in that slot is an answer,
and the entry is spent on it.

### A changeover restarts the counter

The status byte's modulo-4 counter (§14) belongs to the **direction**. It runs
unbroken through every speed-level and cycle-length change, and it starts again at
every changeover:

    entry packet   counter 2    the last of the numbering the PACTOR-1 phase kept,
                                behind that phase's one data packet
    greeting       counter 1    the answerer's own, after its break-in
    data train     1, 2, 3, 0   the caller's, after it takes the link back

The general rule is worth more than the observation. **Stop-and-wait cannot skip a
counter**: the ISS holds its packet until something answers it, so the only two
values a working link can put on the air are the one the receiving end is waiting
for and the one it just took. A third value is not a gap around a lost packet — it
is a peer whose numbering started again. Read as a gap it costs the packet carrying
it, the packet behind that, and the one after that to duplicate detection, and the
stream resumes only when the counter comes round.

### The ladder

    9.11 - 16.61 s    SL3, short cycle, 59-byte fields, counter 1,2,3,0,1,2,3
    17.86 - 21.61 s   SL3, long cycle, 276-byte fields, 3.75 s
    25.36 - 29.11 s   SL4 long, 556 bytes
    32.87 - 44.12 s   SL5 long, 956 bytes
    47.87 - 51.62 s   SL6 long, 1276 bytes
    55.37 s on        SL6 back to the short cycle, 284 bytes

The counter runs straight through every change above: it is not the speed level's
and not the cycle length's, and only a changeover restarts it. The IRS drives the
ladder — CS4 asks for the next level up and CS6 flips the cycle length, both in the
odd-counter slot that also acknowledges (§7), and here the first CS6 arrives on the
seventh data cycle. A station that holds the short raster through a CS6
desynchronises the session.

Status bit 5 is not the cycle length's. The Recommendation's cycle-length
suggestion (§14) is already set at 16.613 s, on the *short* cycle of the run above
and on the reference's first loaded packet — status `0x33`, a full 59-byte field —
and stays set through 44.115 s, clearing at 47.865 s (`0x10`) on the partial field
that empties the buffer. Every template-filled field in the session has it clear:
`0x19`/`0x1a` idle, `0x5d` changeover, `0x98` QRT. What raises it is a loaded
packet with bytes queued behind it, whatever the cycle length; CS6 is what moves
the raster. Measured across the reference caller's whole session, from its
first loaded packet to the partial field that empties its buffer.

### The changeover packet  [P: 1 session, 2 bursts]

The CS3 of §7 is a packet's head. **One keying, 81 symbols, 0.810 s** — a data
packet's length (§3) — on channels 5 and 12 alone, whatever speed level the traffic
either side of it runs at:

    symbol 0        phase reference
    symbols 1-20    the CS3 codeword, keyed as a bare control signal is keyed
    symbols 21-24   run-in, four symbols carrying no bit anything reads
    symbols 25-80   56 rows of a case-0 frame — a six-byte field, three of payload

That is **the ordinary case-0 grid with sixteen rows given away**, and the sixteen
are what the head costs. An ordinary packet spends symbols 1–8 on the header block
and 9–80 on 72 rows; this one spends 1–24 on the codeword and its run-in and 25–80
on the 56 rows left. 56 rows × 2 carriers is 112 cells, the sixteen-column helical
transpose of §10 runs seven rows deep instead of nine, and `cells / 16 − 1` gives
six field bytes — three of payload, the status byte and the CRC (§14) — which is
the rule every unpunctured case obeys.

The head and the frame do **not** share a sign rule. Both ride the ±45° diagonal, as
both stations key them, but the codeword keeps the ordinary bit phase — a zero at
315°, where a receiver slicing the real axis needs it — while the frame behind it
takes case 0's inverted convention (§9). Keying the head through the frame's map
complements every bit of it and lands eight from the nearest codeword: a break-in
nobody reads as one, on a channel the peer goes on transmitting into.

Nothing separates the head from the frame. The quietest of the 81 symbols stands at
0.96 and 0.76 of its own burst's median, where a re-key would take it to the noise
floor, and no variable header anywhere in either burst reaches 0.75 against 0.977
for the same station's entry packet — so there is no gap, no second keying, and no
packet header block.

**Where it sits.** In the control-signal slot, where the acknowledgement it replaces
would have gone, and its own phase reference becomes the new ISS's raster:

    5.5637 s   the answerer's head, 0.8912 s after the entry packet it answers,
               which is the ordinary answer slot of §7
    6.8200 s   the answerer's first packet as ISS, 1.2563 s later — one cycle on
               the raster its own head just set
    7.7681 s   the caller's head, 1.2488 s after its own last transmission and
               0.9481 s after the packet it answers — where its codeword went the
               cycle before, 0.9557 s after the head at 5.5637. Its own slot

The station that yields is the one that moves. Its next transmission falls one cycle
plus 0.597 s after its own last packet's phase reference, and 0.594 s in the other
direction — a packet less a codeword, which is what carries a station that has just
become the IRS from the packet slot to the codeword slot. It also settles the packet
in flight rather than requeueing it, because the break-in answered it.

**What the two carry.** Three payload bytes, a status byte and a CRC:

    5.5637 s   0d 50 54   status 0x20   counter 0   the IRS taking the link
    7.7681 s   0f 8f 87   status 0x18   counter 0   the ISS taking it back

Both counters are 0 and both are followed by counter 1 — the restart recorded
above. Neither field is opaque: `0d 50 54` is carriage return, `P`, `T`, and
`0f 8f 87` is the first three bytes of the header template of §14, at the data type
the same station's speed-level-3 train declares.

**What the geometry rests on.** It was fitted to these two bursts jointly — head
length, row count, transpose depth, carrier order and sign rule together, a search
of some 300,000 combinations — and exactly one decodes both CRC-valid. At that
width a CRC is not worth much on its own, and what stands behind it is semantic
rather than statistical: the 212-byte greeting the answerer sends in the very next
cycle decompresses to "C-II DSP/QUICC System - Maildrop QRV", and `0d 50 54` is the
missing head of that sentence and nothing else — which is not something a search
over geometries could have arranged. Rendering that payload and status back through
the geometry reproduces the answerer's field byte for byte on both carrier
arrangements. As a negative control, 60,086 quarter-symbol head positions across
four FT8 captures yield no CRC-valid field at all.

**It rests on one session.** Both bursts are in `PIII_Complete_1`, and there is no
second recording behind them: the session's companion capture `PIII_Complete_2`
holds no PACTOR-III packet and no CS3 in 63 s, and `ref_occ15` is the first fifteen
seconds of this same QSO re-encoded — the same events at the same times, different
sample values. What the session does hold is two independent transmitters keying
the construction in opposite directions with different fields. What it cannot
settle is whether the four-symbol run-in and the 56-row field are fixed for every
implementation, or whether some stations key a different head length and give away
a different number of rows. A recording of a different station pair changing hands
in PACTOR-III would.

---

## 18. Not specified here

- What the 82nd symbol carries on the stations that key one (§3). It is fixed per
  level at speed level 3 and varies packet to packet above it, and the packet that
  omits it is answered in the same slot as the ones that do not.
- Whether the helical table is a depth or a literal stride at N = 432: at N = 144 it
  measures as a depth (stride 9), which would make the 432-bit frame's stride 27
  rather than 16.
- Confirmation that a frame encoded under the case-4 rate-3/4 pattern validates
  against a clean case-4 frame; the pattern itself is read and the rate arithmetic
  agrees, but no CRC-passing case-4 frame has been produced from it.
- The compressed payload formats (Huffman and PMC) that carry real payload content.
- What fills a data field a station has not filled. Every off-air PACTOR-III packet
  in the corpus is compressed and fills its field exactly, so none of them shows
  padding; `hfmodem.shrike.unknowns:P3_IDLE_PADDING` holds the candidates and what
  would settle it, which is one short field from a real station.
- The mapping of the 8-symbol header router's field selector onto (SL, CYC, RQ).
- The definition of the 10 ms phase-reference pulse, and the role of the ±410
  bipolar sequence that is no longer carried here.
- Which frame of the packet the 44-byte spread-QPSK field of §15 is.
- On case 1 at a dense operating point, about 18 soft positions that no change to the
  data grid moves. They lie on 8-periodic rows (15, 23, 31, 47, 53, 55) and are
  consistent with the demodulator's mod-8 reference pinning every eighth position to
  the fixed pilot block, which would make them frame constraints rather than
  transmittable cells. The gather map itself is affine over GF(2) and exact for
  sparse grids; the residual is a phase-boundary non-linearity at the dense point.
