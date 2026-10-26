<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# The level request is the answer's frame index

This is a measurement record of the source and recordings examined for that
campaign. Implementation gaps described below refer to that source revision.
For current codec and link coverage, see the [protocol index](../README.md)
and [implementation evidence](../../../STATUS.md#vara).

A high speed capable VARA IRS names the ISS's next level by **which frame of its answer stream it keys**. The stream
is the one we already generate (`vara_frames.answer_tail`: keyed to the called station, SEED_OFF 62 when the
caller answers and 290 when the responder does). The frame index is a step code: 0 hold, 6 up one, 4 up two,
7 up three, 8 up four, 5 down one, 1 NAK. The ISS keys exactly the level the code asks for on all 428
delivered full frames, with no misfits at any bandwidth. That includes every move GEARSHIFT-OBSERVED.md put
down to the frame-count rule, the jump cap and the SN ladder, so all three of those rules run in the IRS and
the ISS just obeys. Nothing in the answer is an SN figure or an absolute level. The same code rides two
other carriers. The 32-symbol single-tone session frame at SEED_OFF 60, PREADV 1 + 62·index, answers a
callee ISS's first data frame. The caller's 16-symbol session-confirm at SEED_OFF 1551, PREADV 1 + 30·index,
sets the callee's starting level: frame 3 → level 4, 4 → 5, 5 → 6.

Every one of the 707 IRS answer keyings in `overs.jsonl` decodes to exactly one frame of one stream,
**100 %**, with our existing tables and generator and nothing new. `ack_level_study.py` prints every table here.

## Mapping

| answer frame (8-symbol, and 32-symbol 60/1+62·i) | ISS's next frame | n (8-sym + 32-sym) |
|---|---|---|
| 0 | same level (full 69+9, or double-length 52) | 121 + 9 |
| 6 | +1 | 86 + 14 |
| 4 | +2 | 30 + 9 |
| 7 | +3 | 20 + 15 |
| 8 | +4 | 58 + 8 |
| 5 | −1 (frame delivered) | 58 + 0 |
| 1 | NAK: frame not delivered; ISS falls 1–4 on its own rule | 64 + 0 |

| caller's session-confirm (16-sym, 1551/1+30·i) | callee ISS's first data level | n (500 / 2300 / 2750) |
|---|---|---|
| frame 3 (PREADV 91, our `SESSION_CONFIRM`) | 4 | 1 / 7 / 3 |
| frame 4 (PREADV 121) | 5 | 8 / 6 / 3 |
| frame 5 (PREADV 151) | 6 | 8 / 4 / 17 |

The step is in host `BITRATE` numbers. At BW2300, 12 + frame 4 → 14 (3 of 3), and 14 + frame 5 → 13 on the
last frame of the 4096-byte tapes, so 13 is a real level there that requests happen to step over. Frames 2, 3
and 9 of the answer stream never occur in this corpus, so no −2 or −3 code has been observed. A fall of more
than one only ever follows a NAK.

## Evidence

### Decode

| answer | keyings | frame(s) read | unread |
|---|---|---|---|
| 8-symbol (16376 samples) | 438 | answer stream 0/1/4/5/6/7/8 | 0 |
| 11-symbol (22521) | 142 | control stream 0 (66), 1 (71); answer stream 13 (5) | 0 |
| 16-symbol (≈32.7k) | 69 | 1551/3,4,5 (57), 61/18 (10), 61/1 (2) | 0 |
| 32-symbol (≈65.5k) | 58 | 60/0,4,6,7,8 (55), 60/18 (3) | 0 |

Two-tone bursts were read with `vara_mfsk.demod_tone_pairs` from `measured_on`, using no alignment search, and
the seven tail pairs were looked up exactly in `answer_tail` for both calls, both roles and frames 0–63. The
look-up is exact, so a symbol error would show up as unread, and there are none. The one thing needed at
BW2750 is to search the bandwidth's band (`MK.band_for("2750")`, bins 22–105). With the default BW2300 band,
190 of the 240 BW2750 answers read 1–4 symbols wrong. Single-tone frames were read with `demod_tones`, run
backwards through `payload_states`, and located on each (call, SEED_OFF) stream by walking the lattice.

Whose stream it is follows the existing rule exactly. In the ladders the IRS is the caller (N0CALL), and its
8-symbol answers are W9SSJ's frames at SEED_OFF 62. In the manifests the IRS is the responder, and they are
the called station's frames at 290 (W9SSJ, or N0CALL on the W9SSJ-caller tape). The 11-symbol bursts are the
caller's control stream (852 caller, 1080 responder).

### Obedience, by the rule the gear-shift study credited

| bw | SN | SN, down one | frame count | jump cap | then long | misfits |
|---|---|---|---|---|---|---|
| 500 | 52 | 6 | 44 | 18 | 42 | 0 |
| 2300 | 44 | 4 | 21 | 20 | 6 | 0 |
| 2750 | 65 | 23 | 30 | 49 | 4 | 0 |

The frame-count moves (95) include raises, holds and falls chosen by how many bytes remain. The IRS asks
for each one, so **the IRS knows how much of the delivery is left**. Something in the data frame must carry
that, and this study does not identify the field. Double-length frames always follow frame 0 (52 of 52), and
frame 0 also precedes 78 ordinary holds, so the ISS chooses to double the frame without being asked.

### The two lengths

- **16376 (8 symbols: lead `(64,67)` + 7)**: the answer to every data frame that has another behind it, and
  to every failed one. It is the step code above.
- **22521 (11 symbols: 4-pair preamble + 7)**: the answer to the **last** delivered frame of a delivery
  (control frame 0, 18/20/23 by bandwidth), to the ISS's end-of-delivery polls (frame 1 to the first, frame 0
  to the second, 17/17/22), and, as answer-stream frame 13, to the turn request that starts a callee's
  delivery (5). It carries no step. The next delivery starts from the confirm or the first answer.
- The 65526-sample answer to a ladder's first data frame is the 32-symbol session frame, carrying the same
  step code (55 of 55 read, step matched every time).

### NAK and poll

After frame 1 the ISS falls −1 ×35, −2 ×19, −3 ×3, −4 ×7. The NAK is one frame for all four, so the size is
the ISS's own choice. It follows a per-level table. At BW500 the step from 12 goes to 11 ×7 and from 11 to
9 ×11, and 17 → 16 ×6 and 16 → 14 ×5 at BW2750. A level that was reached by a jump and failed on its first
try falls lower: BW500 13 → 9 after 9 → 13 (2 of 2), 12 → 9 after 8 → 12 (2 of 2). When the IRS does not
answer a frame, the ISS polls and gets a 16-symbol 61/18 (the caller's counterpart of our
`SESSION_OVER_NAK_RESPONDER_SHORT` 289/541) or a 32-symbol 60/18 (`SESSION_OVER_NAK`, 60/1117). Both are
followed by a resend −2 to −4. A 16-symbol 61/1 answered a failed full frame twice, and the ISS fell −1 both
times.

## What the code must do

The table already in `vara_arq.py:570`, `_LEVEL_STEP = {4: 2, 5: -1, 6: 1, 7: 3, 10: -1}`, is this code as
read off speed limited BW500 pairs. What is missing is **frame 8 = +4**, plus the level space the steps
operate in.

**Our IRS, to request a level.** Key the 8-symbol continue at frame `{0:0, +1:6, +2:4, +3:7, +4:8, −1:5}`
of `answer_tail(called, 62 if we called else 290, bw, frame)`, and frame 1 for a frame we could not read.
- `vara_arq.py:4656 _continue_tail` picks a frame from the step only at BW500 as initiator
  (`_answer_step`, :4667, which returns only 0 or 6). At BW2300/2750 it always keys frame 0 (:4665), which asks
  the peer to hold its level forever. Nothing ever keys 4, 7 or 8.
- `vara_arq.py:4777` keys the 32-symbol `SESSION_OVER_RESPONSE` (60/1, frame 0) when that form is selected.
  To carry a step, it would be `replace(kind, preadv=1 + 62 * frame)`.
- `vara_arq.py:4584 _tx_session_confirm` always keys `SESSION_CONFIRM` (1551/91 = frame 3), so a high speed capable
  callee always starts at 4. Frames 4 and 5 (PREADV 121, 151) ask for a start at 5 or 6.
- The policy itself (SN ladder, +4 cap, jump cap, frame count) belongs on the IRS side. For the frame-count
  part our IRS would need the remaining-bytes field that stock evidently reads out of the data frame, and
  that field has not been identified.

**Our ISS, to obey a gateway.** Read frames 0, 1, 4, 5, 6, 7 and 8 on the peer's stream and move by the
table, clamped to the bandwidth's high speed capable levels 4..top (13/16/17).
- `vara_arq.py:3505 _peer_continue_tails` builds its set from `_LEVEL_STEP` plus 0 and 5. **Frame 8 is not in
  it**, so a high speed capable peer's +4 continue (58 of 438 8-symbol answers here, 13 %) is not read as a
  continue at all.
- `vara_arq.py:4240 _took_over_continue` applies a step only when `tx_level is None` and the pending record
  is in `_LADDER` (:572, four speed limited records per bandwidth). Otherwise it falls back to
  `_rerecord_500` on that four-record ladder. There is no path from a step to a high speed capable level 4–17. The
  data path keys at `_over_level` (:2265), which is the base level.
- `vara_arq.py:4231` reads the 32-symbol step answer only at BW500 as initiator, and only on the 288 stream.
  A callee ISS has to read the caller's 60/(1 + 62·i) answer to its first frame (55 of 57 ladders).
- A callee ISS must read the caller's confirm frame index for its start level. `_NSYM`/`_CANDIDATES`
  (:1550–1580) hold only `SESSION_CONFIRM` at PREADV 91, so frames 4 and 5 are not recognised.
- The NAK fall is ours to choose (`_data_nak_level`, :2957). Stock falls 1–4 by level. It never falls below
  4 and never below 5 after a failure run (GEARSHIFT-OBSERVED rule 3).

**Readings to revise.** `vara_frames.py:574 _CALLER_500_FRAME` treats frames 4 and 6 as the BW500 caller
*reporting* a level-2 or level-1 over. `vara_frames.py:626 RETRIED_OVER_FRAMES` treats them as the continue
to a *retried* over. `docs/protocols/vara/04-frame-block-formats.md` §4.2C (answer-tail table) holds both
readings. On this corpus 4 and 6 are the +2 and +1 requests on any delivered frame (30 and 86 here, none of
them retries), and 5 is −1 whether or not a `0x81` record was involved.
