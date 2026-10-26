<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Lead generators: one callsign-keyed Rnd stream per station per session

`lead_study.py` prints every table below (a few minutes; it cuts every data over and fits the L7-9 sub-band
pulse blind). The tapes are read for measurement only.

## Findings

1. **The stream is reseeded at every session, keyed to the caller's callsign.** Each station runs the
   control-burst stream that `vara_frames.control_bursts` already generates: `Randomize((CRC(caller) + 852)
   & 0x7FFF)` at the caller and `+ 1080` at the responder (both keyed to the CALLER), advanced by
   `G(caller) + carry`. It then advances past preadv 1 and control frames 0 and 1. Call that state **E1**.
   Every lead in the corpus is drawn from this stream after E1. Nothing carries over from the process's
   history: every ladder session, keyed about 45 s after the last, starts from the same E1.
2. **4107106 is AAAA1's E1.** The "training seed" is E1 for caller AAAA1 at BW2300. AAAA1's BW2750 E1 sits
   exactly 5 draws later, which is the documented `STREAM_LEAD = 5`. The two bandwidths differ because the
   control frames' Pb rejections run over n_p = 5 and 6 respectively. W9SSJ's caller E1 at BW2300 is
   2767643, the a2b state in `01-physical-layer.md`'s rec2 table. Its responder E1 on the n_p = 6 alphabet
   is 3653944, that table's b2a state.
3. **The draw count per over depends only on bandwidth, level and frame count.** Payload, retries and
   acks do not change it. 269 leads equal the rule's draws cell for cell,
   and 59 noisy leads agree best at the predicted position. None sits elsewhere. That covers three
   bandwidths, both roles, three callers and all 59 tapes that have a full over.
4. **Sub-band lead (BW2300 L5-9): `Int(Rnd*4)` from the same stream, 9 per band.** The draws are
   time-major, 9 rows of `nb`, Gray-mapped to quarter turns. Row 0 is the reference's phase (t = 8), and
   rows 1..8 are lead t = 0..7. Each cell is multiplied by `conj(grid[b][t])`. The lead uses grid indices
   0..7, the same cells the data at t = 8..15 uses, so no new table is needed. Hypothesis (b) holds. (a),
   a fixed pattern per level, is excluded: the same level at different states differs, and at the same
   state it is identical across sessions. (c) and (d) were not needed. 52 of 55 overs match, and the 3
   unreadable ones have lock 0.10-0.13.
5. **Two-frame OFDM overs draw two lead blocks and send the second.** This holds at all three bandwidths
   (490 at BW2300 L16, 110 at BW500 L13, 590 at BW2750 L17).

## The rule

```
E1(station) = control_end(advance(start((crc(caller) + {A: 852, B: 1080}) & 0x7FFF), G(caller) + carry),
                          n_p = {2300: 5, 2750: 6, 500: 2}[bw])       # preadv 1 + frames 0 and 1
s = E1
for each keying of this station, in order:
    if it is the first data over of a delivery (one host write): s += 1
    lead of this over = s (two-frame OFDM: s + one block)
    s += cost(keying)

cost = 0                  ack, connect frames, disconnect, anything under 3 s (short/idle frames)
     = 12                 L4 at BW2300 and BW2750 (the one-hot preamble, Int(Rnd*16) per symbol)
     = 9 * bands          sub-band levels: BW2300 L5-9 (3,4,6,8,10), BW2750 L5-10 (3,4,6,8,10,13),
                          BW500 L4-6 (2,2,3)
     = lead * carriers * frames   OFDM: lead 4 (PSK) / 5 (QAM) x 49 / 59 / 11 carriers
```

At the caller, the link-setup L4 over is keyed at E1 itself, so the documented "one extra draw between
over 0 and 1" is the first delivery's draw. The responder's first data over sits at E1 + 1. Every later
delivery's first over draws one more again (§ Second deliveries); retries and the later overs of a
delivery do not.

## Evidence

### Tally (`lead_study.py`, § Tally)

| bw | family | exact | hit | anchor | miss | unreadable |
|---|---|---|---|---|---|---|
| 2300 | L4 preamble | 22 | 0 | - | 0 | 0 |
| 2300 | OFDM L10-16 | 40 | 1 | - | 0 | 0 |
| 500 | OFDM L7-13 | 129 | 24 | - | 0 | 17 |
| 2750 | OFDM L11-17 | 78 | 34 | - | 0 | 0 |
| 2300 | sub-band L5 | 0 | 14 | 1 | 0 | 0 |
| 2300 | sub-band L6 | 1 | 12 | 1 | 0 | 2 |
| 2300 | sub-band L7 | 0 | 3 | 1 | 0 | 1 |
| 2300 | sub-band L8 | 2 | 6 | 1 | 0 | 0 |
| 2300 | sub-band L9 | 0 | 9 | 1 | 0 | 0 |

- **exact:** every lead cell equals the predicted draw.
- **hit:** the predicted position's agreement beats every other position within ±300 draws by more than
  0.1. For OFDM this is 0.83-0.996 against 0.06-0.35.
- **miss:** another position agrees at 0.9 or better. There are none.
- **unreadable:** no position stands out, so the cells are noise.

23 of the 24 unreadable overs are followed, on the same station and tape, by an exact or hit over, so the
rule's count through them is confirmed anyway. The exception is the last over of BW500 053832Z. The noisy
BW2300 L11 over 909025, which KPA-bw2300-const could not recover, reads 0.995 at its predicted state.

Sub-band scores compare the lead cells t = 0..7 with the prediction and leave no free phase. `abs` checks
bands 0-1 against grid480 rows 0/1: it is 1.00 on every clean over and 0.81-1.00 on the readable noisy
ones. `pair` checks every band, differenced against the level's anchor over. The anchors are the clean
overs 2313402 (L5), 2299764 (L6), 3717213 (L7), 2194259 (L8) and 1165906 (L9), each with abs 1.00. Their
states are predicted by the rule, not fitted.

### Origin (`lead_study.py`, § Origin)

| caller | role | BW2300 E1 | BW2750 E1 | distance from 4107106 |
|---|---|---|---|---|
| AAAA1 | caller | 4107106 | 12338801 | 0 / +5 |
| N0CALL | caller / responder | 15762747 / 2380025 | 6648348 / 2596114 | - |
| W9SSJ | caller / responder | 2767643 / 14503988 | 5232089 / 3653944 | - |

### One session walked (N0CALL-W9SSJ-bw2300-counter65536, station A)

The session opens with an MFSK connect request (0 draws). Then:

| step | over | lead at (draws past Randomize + G) | draws spent |
|---|---|---|---|
| 1 | link-setup L4 | 59 = E1 | 12 |
| 2 | short L4 frames, 0.7 and 1.4 s | - | 0 |
| 3 | the first-data draw | - | 1 |
| 4 | L4 | 72 | 12 |
| 5 | L8 | 84 | 72 |
| 6 | L12 | 156 | 196 |
| 7 | L14 | 352 | 196 |
| 8 | L15 | 548 | 245 |
| 9 | L16 ×6 | 793 … 2018 | 245 each |
| 10 | L16 two-frame ×6 | 2508 … 4958 | 490 each, lead 245 into the block |
| 11 | L15 | 5203 | |

The ladder responder sessions separate the sub-band costs. For example:

- L6 + L9 before the first OFDM lead: 58 + 36 + 90 = 184.
- L5 + L7 + L9: 58 + 27 + 54 + 90 = 229.
- L4 ×3 + L8: 58 + 36 + 72 = 166.

The same arithmetic gives BW2750 L4 = 12 and BW500 L4 = 18 (2 bands).

### Second deliveries (`handover_bench.py --close-holding`, 2026-09-26)

Three high speed capable stock-vs-stock sessions, W9SSJ calling WI9YZA, one cable per direction: B's greeting,
A's login, B's reply, then A's second delivery. Every short burst on both cables is a named frame on the
62/290 or 852/1080 streams (control frame 0, the 62/13 and 290/13 turn grants, the step continues), so
nothing keyed between overs touches the lead stream. The overs read against E1 of their own station,
`abs` on bands 0-1 for L6 and the one-hot draws for L4:

| session | station | over | level | rule | read | cells |
|---|---|---|---|---|---|---|
| 2300 short | A | link-setup | 4 | +0 | +0 | 12/12 |
| 2300 short | A | login | 6 | +13 | +13 | 1.00 |
| 2300 short | A | second delivery | 6 | +49 | **+50** | 1.00 |
| 2300 short | B | greeting | 6 | +1 | +1 | 1.00 |
| 2300 short | B | reply | 6 | +37 | **+38** | 1.00 |
| 2300 pad300 | A | link-setup | 4 | +0 | +0 | 12/12 |
| 2300 pad300 | A | login, over 1 of 2 | 6 | +13 | +13 | 1.00 |
| 2300 pad300 | A | second delivery | 6 | +103 | **+104** | 0.94 |
| 2300 pad300 | B | greeting, over 1 of 2 | 6 | +1 | +1 | 1.00 |
| 2300 pad300 | B | reply | 6 | +91 | **+92** | 0.94 |

The L7 second overs of the pad300 deliveries need the blind pulse fit and were not read; the sums
through them close. The BW2750 session's sub-band leads were not read (the law is untested there).
Nothing else within ±1500 draws scores above 0.75.

What the bench does not settle: KB8AY's second delivery on the air (arms 2 and 4 of 2026-09-26) read
at its E1 + 42 / + 41 against this rule's + 38, and the 2026-08-14 ref128 responder's first delivery
after receiving sits at E1 + 4 against + 1. In every one of those the station had been through a turn
negotiation (turn-request, idle, drained, keepalive answers) that the bench never keys; which of those
frames draw, and how many, is open.

### How the sub-band law fell

1. Two clean L6 overs at different predicted states differ by exact quarter turns on every lead cell, so
   the lead is QPSK times a fixed per-cell scramble. Over pairs at the same predicted state agree to about
   5 units of 128.
2. The scramble is grid480 row b at index t. A window search over the table put the lead of bands 0/1 on
   the QPSK lattice at offset 0 and 471 exactly.
3. A 2^24 search over layouts (band-major 8/9, time-major; identity/Gray; ±) found one state, for
   time-major Gray only: the predicted state + 4, which is one row of 4 bands.
4. The per-band residual rotation varied from over to over and equals minus row 0's Gray value on every
   clean over. That makes row 0 the reference.

## What remains unknown

- What the per-delivery draw is. It is measured, not named.
- Why a two-frame over sends its second block.
- Whether the sub-band lead law holds at BW2750 L5-10 and BW500 L4-6. Their draw counts (9 per band) are
  confirmed by the OFDM leads after them; their lead cells were not read here.
- BW2750 and BW500 L4 were not inverted directly; their 12 and 18 come from the sums.
- How the full-length disconnect over draws. It is always last on these tapes, so nothing follows it to
  check.
- Callsigns with an SSID: only N0CALL, W9SSJ and AAAA1 are exercised. Normalisation follows whatever
  `control_bursts` uses.
- Why the responder's E1 in the 08-14 b2a tape sits on the n_p = 6 alphabet.

## What must change (not touched here)

- The levels design note, §"Lead state" (line 85). "Seeded at 4107106" is AAAA1's E1 only.
  The TX session state becomes E1(caller, role, bw) plus the cost table above. The draw counts it pins
  are right, and the start is no longer open.
- The same document's tone paragraph (line 81) says the lead generator is unknown and a stand-in. It now
  has a generator (finding 4), so only the L7-9 pulse remains a stand-in.
- `tx/varahf2300_tx.py:56-114`: `PREAMBLE_SEED`, `STREAM_LEAD` and `over_preamble_bins` are AAAA1's
  constants. They become derived from the caller's callsign, role and bandwidth.
- `docs/protocols/vara/01-physical-layer.md` §2300: "Over 0 begins at LCG state 4107106 … does not fix the
  starting state for other scenarios" is now fixed by the rule. The rec2 table's two states are E1 values.
