# High speed capable VARA gear-shift, as observed

What a high speed capable VARA HF 4.9.0 pair does with its speed level, read from transcripts only: the 5 clean
manifest sessions and the 57 noise ladders on the bench tapes, joined with `overs.jsonl`.
`gearshift_study.py` prints every table below. Levels are host `BITRATE (N)` numbers; "delivered" means the
IRS printed `data_rx` bytes for the frame; SN is what the IRS printed for that frame. In the ladders B is the
ISS and only its leg (B→A) is noisy, so anything B does in step with that noise it learned from A.

## Rules

**1. Start.** The caller's link-setup is keyed at level 4 on every tape. The first data frame is never above
level 6. A callee ISS that has data at CONNECTED starts at 6, 5 or 4 depending on the noise on its own leg,
cleanly separated by floor (BW2300: 6 at −26..−23 dBFS, 5 at −22..−17, 4 at −16..−10). The only thing that crossed
that leg before the first data frame was B's connect answer, so A measured it and told B. A caller ISS that idled
for 31 s first (the manifests) starts at 4, 5 of 5, then goes 4→8→12.

**2. Up-shift.** No dwell: the level moves on the very next frame after any delivered frame. Steps are +1 to +4,
never more (240 up-steps: +1 ×100, +2 ×39, +3 ×35, +4 ×66). The size is set by the IRS's reading of the frame:
on BW2750 a ladder of SN thresholds (table 4) fits all but 0–4 moves per level, and the step is the number of
thresholds the SN clears, capped at 4. Two caps sit on top of it:
- *Jump cap.* On BW2300/2750 a jump of more than one never lands above top−2: 16 and 17 at BW2750 and 15 and 16
  at BW2300 are entered only from the level below (16 ← 15 nine of nine, 17 ← 16 seven of seven). BW2300 goes 12→14
  three times out of three at SN 32–43, skipping 13. BW500 jumps straight onto 13 (9→13 ×3, 10→13 ×2).
- *Frame count.* When the rest of the delivery fits in one or two frames, the ISS keys the lowest level that
  still finishes it in the fewest frames, and moves at most one level down per frame to get there. This is what
  shortens the last frame (14→13 with 74 B, 17→16 with 2937 B), holds a level the SN would lift (15 held at SN 43 with
  5073 B left, `060143Z`), and raises one it would not (12→13 to fit 621 B, `052202Z`).

The IRS's answer is 16376 samples before an up-shift, a hold, a fall and a failed frame alike (table 8); 22521
only after the last delivered frame and the end polls, 65526 once per ladder as the answer to the first data frame
(55 of 57). Whatever carries the step rides inside the fixed-length answer.

**3. Down-shift.** A delivered frame read at low SN takes the next frame one level down, never more (0 of 373
delivered moves fell more than one). An undelivered frame the IRS answers falls 1 to 4 levels (table 6); a second
failure falls again. When the IRS does not answer at all, the ISS keys a 0.23–0.38 s poll, the IRS answers it with
0.68 s — on `20260926T051938Z-ladder` (BW2300, N0CALL → W9SSJ, 63.9 s) the caller's 16-symbol over-NAK, 61/541
keyed to W9SSJ, 16 of 16 tones (`vara_frames.SESSION_OVER_NAK_SHORT`) — and the resend is 2 to 4 levels down
(BW500: −2 eight times of eight). Nothing ever goes below 4, even at
0 dB wideband SNR; level 4 is reached only from 5, and the lowest level after any failure run is 5.

**4. Re-climb.** Same thresholds and caps as the first climb, with a brake near the bottom: when the first frame
read clean after a failure sits at level 4–6 (SN 31–41 wide, 19–22 on BW500), 11 of 13 such frames climb only +1,
and the frame after that usually +4 (BW2750 5→6→10 three times, BW2300 5→6→10). The other two went +4. Higher up
there is no brake: in the quiet BW2750 ladders, after 16x→14→13→12→11 on low SN, 11 read at SN 45 went straight to
15 (twice).

**5. Double-length frames.** After exactly 6 consecutive delivered frames at the top level, every frame is double
length while at least two single loads remain: 52 of 52, carrying exactly twice the single load (1420, 6508,
7838 B), and none before the sixth. It ends when less than two loads remain, and the frame-count rule then steps
down one level. No frame at any other level is long; no ladder held its top for six frames, so this rests on the
three 65536-byte manifests.

**6. Time budget.** Single frames cycle in 4.3–5.0 s, long frames in 8.2–8.35 s, so the long frame buys
9–11 % at the top (table 12). End to end, 65536 bytes took 88 s at BW2750 (top 17, 1.35 ms/byte,
5947 bps), 104 s at BW2300 (16, 1.59 ms/byte), 392 s at BW500 (13, 5.97 ms/byte). Idle ISS: at level 4, one
0.68 s keying, then four of 1.37 s at intervals of 1.4, 2.1, 12.1 and 12.1 s, each answered with 0.47 s. After a
delivery, two 0.47 s keyings, 4.4 s and 5.5 s after the last data frame's PTT ON, each answered with 0.47 s.

**7. What the IRS reports.** `BITRATE (N) … RX` equals the ISS's TX on all 489 delivered frames; an undelivered
frame gets no RX line (80). The IRS never reports a lower level. What moves the ISS is the reading the IRS sends
back (rule 2), not the RX line.

## Evidence

### 1. Every session

| tape | bw | B-leg floor | ISS data frames (x = nothing delivered, p = short keying: poll or end of delivery, L = double length) |
|---|---|---|---|
| N0CALL-W9SSJ-bw2300-counter4096 | 2300 | clean | p 4 8 12 14 13 p |
| W9SSJ-N0CALL-bw2300-counter4096 | 2300 | clean | p 4 8 12 14 13 p |
| N0CALL-W9SSJ-bw2750-counter65536 | 2750 | clean | p 4 8 12 15 16 17 17 17 17 17 17 17L 17L 17L 17L 16 p |
| N0CALL-W9SSJ-bw2300-counter65536 | 2300 | clean | p 4 8 12 14 15 16 16 16 16 16 16 16L 16L 16L 16L 16L 16L 15 p |
| N0CALL-W9SSJ-bw500-counter65536 | 500 | clean | p 4 8 12 13 13 13 13 13 13 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 13L 12 11 p |
| 20260926T050709Z | 2300 | -26.36 | 6 9 10 10 p p |
| 20260926T050752Z | 2300 | -25.38 | 6 9 11 10 p p |
| 20260926T050835Z | 2300 | -24.38 | 6 9 9 11 p p |
| 20260926T050919Z | 2300 | -23.33 | 6 9 9 11 p p |
| 20260926T051002Z | 2300 | -22.34 | 5 7 9 12 p p |
| 20260926T051045Z | 2300 | -21.34 | 5 8 9 11 p p |
| 20260926T051128Z | 2300 | -20.34 | 5 8 9 11 p p |
| 20260926T051211Z | 2300 | -19.33 | 5 7 8 12 p p |
| 20260926T051254Z | 2300 | -18.32 | 5 7 8 12 p p |
| 20260926T051337Z | 2300 | -17.4 | 5 7 6 10 10 p p |
| 20260926T051425Z | 2300 | -16.34 | 4 6 6 10 10 p p |
| 20260926T051513Z | 2300 | -15.35 | 4 6 6 10 10 p p |
| 20260926T051601Z | 2300 | -14.38 | 4 5 6 10 10 p p |
| 20260926T051649Z | 2300 | -13.36 | 4 5 6 10 10 p p |
| 20260926T051737Z | 2300 | -12.33 | 4 5 5 9 12 p p |
| 20260926T051826Z | 2300 | -11.37 | 4 4 4 8 10x 6 5 5 6 10 p p |
| 20260926T051938Z | 2300 | -10.35 | 4 4 4 8 10x 6x 5 4 5 9 10x p 6 5 6 7 p p |
| 20260926T045139Z | 2750 | -26.35 | 6 10 11 10 p p |
| 20260926T045221Z | 2750 | -25.34 | 6 10 11 10 p p |
| 20260926T045304Z | 2750 | -24.29 | 6 9 10 11 p p |
| 20260926T045347Z | 2750 | -23.35 | 6 9 10 11 p p |
| 20260926T045430Z | 2750 | -22.33 | 6 9 10 11 p p |
| 20260926T045513Z | 2750 | -21.35 | 6 8 9 12 p p |
| 20260926T045556Z | 2750 | -20.45 | 6 9 9 10 10 p p |
| 20260926T045644Z | 2750 | -19.34 | 6 9 10 11 p p |
| 20260926T045727Z | 2750 | -18.36 | 6 7 9 12 p p |
| 20260926T045810Z | 2750 | -17.37 | 6 7 8 12 p p |
| 20260926T045853Z | 2750 | -16.48 | 6 6 6 10 11 p p |
| 20260926T045941Z | 2750 | -15.35 | 5 6 5 9 12 p p |
| 20260926T050029Z | 2750 | -14.38 | 5 6 6 10 11 p p |
| 20260926T050116Z | 2750 | -13.4 | 5 6 6 10 11 p p |
| 20260926T050204Z | 2750 | -12.37 | 4 5 4 7 11x 7x 6 5 6 10 10x p 7x 6 8 p p |
| 20260926T050336Z | 2750 | -11.41 | 4 4 4 8 12x 8x p 6 5 6 10x 7x 6 5 6 10 p p |
| 20260926T050512Z | 2750 | -10.41 | 4 4 4 7 11x 7x p 5 4 5 9x p 6x 5 6 10 9x 7x 6x 5 7 p p |
| 20260926T055838Z | 2750 | -32.34 | 6 10 14 15 14 15 16 17 17 17x 16 15 14 13 13 p p |
| 20260926T060010Z | 2750 | -30.41 | 6 10 13 14 14 15 16 17 17 17x 16 15 14 13 15 p p |
| 20260926T060143Z | 2750 | -29.37 | 6 10 12 13 14 15 16 17 17 17x 16x 14 13 12 15 15 15 p p |
| 20260926T060325Z | 2750 | -28.36 | 6 10 13 14 13 15 16 17 17x 16x 14 13 12 11 15 16 15 14 p p |
| 20260926T060511Z | 2750 | -27.38 | 6 10 12 11 12 15 16 17 17x 16x 14 13 12 11 15 16 16 16x 14 13 p p |
| 20260926T060706Z | 2750 | -26.38 | 6 10 11 12 12 15 16 17 17x 16x 14x p |
| 20260926T052115Z | 500 | -26.4 | 6 10 11 12 11 p p |
| 20260926T052202Z | 500 | -25.39 | 6 9 10 12 13 p p |
| 20260926T052250Z | 500 | -24.41 | 6 10 11 12 11 p p |
| 20260926T052337Z | 500 | -23.43 | 6 9 11 12 12 p p |
| 20260926T052424Z | 500 | -22.33 | 6 9 8 10 12 11 p p |
| 20260926T052516Z | 500 | -21.32 | 6 9 8 10 12 11 p p |
| 20260926T052607Z | 500 | -20.33 | 6 8 8 10 12 11x 9 8 p p |
| 20260926T052707Z | 500 | -19.31 | 6 7 8 10 12 12x 11x 9 8 p p |
| 20260926T052812Z | 500 | -18.38 | 5 6 7 11 12 12x 11x 9 8 p p |
| 20260926T052917Z | 500 | -17.3 | 5 7 7 10 12 12x 11x 9 8 7 p p |
| 20260926T053026Z | 500 | -16.37 | 5 6 6 10 13 13x 12x 11x 9 9 p p |
| 20260926T053136Z | 500 | -15.3 | 5 6 5 9 13 12x 11x 9x 8 11 p p |
| 20260926T053246Z | 500 | -14.32 | 5 5 6 10 13 13x 12x 11x 9 10 p p |
| 20260926T053357Z | 500 | -13.32 | 5 5 5 9 13x 9x 8x 7 8 12 11x 9x 8x 7 8 p p |
| 20260926T053528Z | 500 | -12.31 | 5 5 5 8 12x 9x 8x 7 8 12 12x 11x 9x 8 8 p p |
| 20260926T053700Z | 500 | -11.41 | 5 5 5 9 13x 9x 8x 7 8 12 11x 9x 8x 7 8 p p |
| 20260926T053832Z | 500 | -10.38 | 4 5 5 8 12x 9x p 7x p 5 6 10x 8x 7x p 5 6 10x p 8x p 6 7 11x 9x p 7x p 5 6 8x 7x p 5 6 9x 7x 6 5 4 5 p p |

### 2. Start level against the noise on the ISS's leg (ladders)

| bw | first data level | ladders | B-leg floor (dBFS) |
|---|---|---|---|
| 500 | 4 | 1 | -10.38 .. -10.38 |
| 500 | 5 | 8 | -18.38 .. -11.41 |
| 500 | 6 | 8 | -26.4 .. -19.31 |
| 2300 | 4 | 7 | -16.34 .. -10.35 |
| 2300 | 5 | 6 | -22.34 .. -17.4 |
| 2300 | 6 | 4 | -26.36 .. -23.33 |
| 2750 | 4 | 3 | -12.37 .. -10.41 |
| 2750 | 5 | 3 | -15.35 .. -13.4 |
| 2750 | 6 | 17 | -32.34 .. -16.48 |

### 3. What decided each move after a delivered single frame
"jump cap" and "frame count" are the caps of rule 2; "SN" is everything else.

| bw | decided by | delta | n |
|---|---|---|---|
| 500 | SN | 0 | 16 |
| 500 | SN | 1 | 19 |
| 500 | SN | 2 | 9 |
| 500 | SN | 3 | 8 |
| 500 | SN, down one | -1 | 6 |
| 500 | frame count | -1 | 13 |
| 500 | frame count | 0 | 9 |
| 500 | frame count | 1 | 9 |
| 500 | frame count | 2 | 5 |
| 500 | frame count | 3 | 3 |
| 500 | frame count | 4 | 4 |
| 500 | jump cap | 1 | 1 |
| 500 | jump cap | 4 | 17 |
| 2300 | SN | 0 | 15 |
| 2300 | SN | 1 | 13 |
| 2300 | SN | 2 | 10 |
| 2300 | SN | 3 | 6 |
| 2300 | SN, down one | -1 | 4 |
| 2300 | frame count | -1 | 3 |
| 2300 | frame count | 0 | 6 |
| 2300 | frame count | 1 | 2 |
| 2300 | frame count | 2 | 4 |
| 2300 | frame count | 3 | 2 |
| 2300 | frame count | 4 | 3 |
| 2300 | jump cap | 1 | 2 |
| 2300 | jump cap | 2 | 3 |
| 2300 | jump cap | 4 | 15 |
| 2750 | SN | 0 | 25 |
| 2750 | SN | 1 | 27 |
| 2750 | SN | 2 | 4 |
| 2750 | SN | 3 | 9 |
| 2750 | SN, down one | -1 | 23 |
| 2750 | frame count | -1 | 6 |
| 2750 | frame count | 0 | 7 |
| 2750 | frame count | 1 | 7 |
| 2750 | frame count | 2 | 3 |
| 2750 | frame count | 3 | 3 |
| 2750 | frame count | 4 | 3 |
| 2750 | jump cap | 1 | 20 |
| 2750 | jump cap | 2 | 1 |
| 2750 | jump cap | 3 | 4 |
| 2750 | jump cap | 4 | 24 |

| bw | from | to (count) |
|---|---|---|
| 500 | 4 | 5 (2), 8 (1) |
| 500 | 5 | 6 (8), 7 (1), 8 (2), 9 (3) |
| 500 | 6 | 7 (3), 8 (2), 9 (5), 10 (6) |
| 500 | 7 | 8 (6), 10 (1), 11 (2) |
| 500 | 8 | 10 (4), 11 (1), 12 (6) |
| 500 | 9 | 10 (2), 11 (1), 13 (3) |
| 500 | 10 | 11 (2), 12 (6), 13 (2) |
| 500 | 11 | 12 (4) |
| 500 | 12 | 13 (2) |
| 2300 | 4 | 5 (4), 6 (2), 8 (5) |
| 2300 | 5 | 6 (4), 7 (4), 8 (2), 9 (2) |
| 2300 | 6 | 7 (1), 9 (4), 10 (6) |
| 2300 | 7 | 8 (2), 9 (1) |
| 2300 | 8 | 9 (2), 10 (2), 12 (5) |
| 2300 | 9 | 10 (2), 11 (5), 12 (2) |
| 2300 | 12 | 14 (3) |
| 2300 | 14 | 15 (1) |
| 2300 | 15 | 16 (1) |
| 2750 | 4 | 5 (2), 7 (2), 8 (2) |
| 2750 | 5 | 6 (7), 7 (1), 9 (2) |
| 2750 | 6 | 7 (2), 8 (2), 9 (5), 10 (15) |
| 2750 | 7 | 8 (1), 9 (1), 11 (2) |
| 2750 | 8 | 9 (1), 12 (3) |
| 2750 | 9 | 10 (5), 12 (3) |
| 2750 | 10 | 11 (10), 12 (2), 13 (2), 14 (1) |
| 2750 | 11 | 12 (2), 15 (2) |
| 2750 | 12 | 13 (1), 15 (4) |
| 2750 | 13 | 14 (3), 15 (2) |
| 2750 | 14 | 15 (4) |
| 2750 | 15 | 16 (9) |
| 2750 | 16 | 17 (7) |

### 4. SN thresholds
`req`: the SN a delivered frame needs for the next frame to reach that level, fitted from table 3's moves.

| bw | level | req SN (dB) | bounds | misfits | highest SN that stayed below | lowest SN that reached it |
|---|---|---|---|---|---|---|
| 500 | 5 | -8.2 | 2 | 1 | 19.4 | -8.2 |
| 500 | 6 | -3.6 | 19 | 0 | -3.6 | -3.1 |
| 500 | 7 | 0.6 | 13 | 4 | 21.9 | 0.7 |
| 500 | 8 | 2.6 | 14 | 1 | 20.6 | 3.6 |
| 500 | 9 | 17.4 | 20 | 4 | 17.4 | 5.6 |
| 500 | 10 | 8.7 | 17 | 2 | 21.7 | 7.7 |
| 500 | 11 | 14.5 | 10 | 4 | 18.3 | 7.5 |
| 500 | 12 | 9.6 | 7 | 0 | 9.5 | 15.9 |
| 500 | 13 | 16.0 | 5 | 0 | 15.9 | 18.4 |
| 2300 | 5 | -4.9 | 9 | 0 | -4.9 | -4.2 |
| 2300 | 6 | -2.1 | 14 | 1 | 12.6 | -1.7 |
| 2300 | 7 | -0.2 | 13 | 3 | 34.6 | 1.6 |
| 2300 | 8 | 1.9 | 13 | 2 | 3.0 | 2.0 |
| 2300 | 9 | 4.2 | 13 | 1 | 4.2 | 3.7 |
| 2300 | 10 | 8.1 | 17 | 1 | 10.4 | 9.7 |
| 2300 | 11 | 36.0 | 4 | 1 | 36.0 | 12.2 |
| 2300 | 12 | 12.2 | 4 | 0 | 12.2 | 33.9 |
| 2300 | 14 | 32.4 | 3 | 0 |  | 32.4 |
| 2300 | 15 | 32.1 | 1 | 0 |  | 32.1 |
| 2300 | 16 | 29.4 | 1 | 0 |  | 29.4 |
| 2750 | 5 | -4.7 | 8 | 0 | -4.7 | -4.3 |
| 2750 | 6 | -1.6 | 13 | 2 | 15.4 | -2.7 |
| 2750 | 7 | 1.4 | 15 | 4 | 40.7 | 2.1 |
| 2750 | 8 | 2.3 | 8 | 2 | 15.9 | 3.4 |
| 2750 | 9 | 4.1 | 11 | 0 | 4.1 | 4.8 |
| 2750 | 10 | 6.2 | 27 | 2 | 7.4 | 4.8 |
| 2750 | 11 | 7.5 | 10 | 1 | 13.9 | 9.2 |
| 2750 | 12 | 10.1 | 11 | 2 | 11.3 | 10.8 |
| 2750 | 13 | 13.2 | 11 | 0 | 13.2 | 13.9 |
| 2750 | 14 | 14.0 | 13 | 3 | 15.5 | 13.9 |
| 2750 | 15 | 15.4 | 18 | 0 | 15.3 | 16.5 |
| 2750 | 16 | 14.7 | 11 | 0 | 14.6 | 41.0 |
| 2750 | 17 | 39.7 | 7 | 0 |  | 39.7 |

Misfits (all levels):

| tape | bw | move | SN | step into the frame | frame before |
|---|---|---|---|---|---|
| 20260926T045556Z | 2750 | 9 > 10 | 13.9 | 0 | delivered |
| 20260926T045644Z | 2750 | 9 > 10 | 4.8 | 3 | delivered |
| 20260926T045727Z | 2750 | 7 > 9 | 7.4 | 1 | delivered |
| 20260926T050029Z | 2750 | 5 > 6 | -2.7 |  | - |
| 20260926T050204Z | 2750 | 4 > 7 | 12.5 | -1 | delivered |
| 20260926T050204Z | 2750 | 5 > 6 | 35.9 | -1 | delivered |
| 20260926T050336Z | 2750 | 5 > 6 | 38.8 | -1 | delivered |
| 20260926T050336Z | 2750 | 5 > 6 | 40.7 | -1 | delivered |
| 20260926T050512Z | 2750 | 4 > 5 | 15.4 | -1 | delivered |
| 20260926T050512Z | 2750 | 4 > 7 | 15.9 | 0 | delivered |
| 20260926T050512Z | 2750 | 5 > 6 | 6.6 | -1 | failed |
| 20260926T050752Z | 2300 | 6 > 9 | 10.4 |  | - |
| 20260926T050752Z | 2300 | 9 > 11 | 12.2 | 3 | delivered |
| 20260926T051002Z | 2300 | 5 > 7 | 3.0 |  | - |
| 20260926T051128Z | 2300 | 8 > 9 | 3.7 | 3 | delivered |
| 20260926T051254Z | 2300 | 5 > 7 | 2.3 |  | - |
| 20260926T051513Z | 2300 | 6 > 6 | 2.2 | 2 | delivered |
| 20260926T051826Z | 2300 | 5 > 6 | 34.6 | 0 | delivered |
| 20260926T051938Z | 2300 | 4 > 5 | 12.6 | -1 | delivered |
| 20260926T051938Z | 2300 | 5 > 6 | 31.6 | -1 | delivered |
| 20260926T052115Z | 500 | 10 > 11 | 8.7 | 4 | delivered |
| 20260926T052202Z | 500 | 6 > 9 | 8.6 |  | - |
| 20260926T052202Z | 500 | 9 > 10 | 7.7 | 3 | delivered |
| 20260926T052250Z | 500 | 10 > 11 | 7.5 | 4 | delivered |
| 20260926T052337Z | 500 | 6 > 9 | 7.8 |  | - |
| 20260926T052337Z | 500 | 9 > 11 | 9.5 | 3 | delivered |
| 20260926T052424Z | 500 | 6 > 9 | 6.2 |  | - |
| 20260926T052516Z | 500 | 6 > 9 | 5.6 |  | - |
| 20260926T052917Z | 500 | 7 > 10 | 18.3 | 0 | delivered |
| 20260926T053832Z | 500 | 5 > 4 | 19.4 | -1 | delivered |
| 20260926T053832Z | 500 | 5 > 6 | 19.4 | -2 | failed |
| 20260926T053832Z | 500 | 5 > 6 | 20.6 | -2 | failed |
| 20260926T053832Z | 500 | 5 > 6 | 21.4 | -2 | failed |
| 20260926T053832Z | 500 | 5 > 6 | 21.9 | -2 | failed |
| 20260926T053832Z | 500 | 6 > 7 | 20.6 | -2 | failed |
| 20260926T053832Z | 500 | 6 > 9 | 21.7 | 1 | delivered |
| 20260926T060010Z | 2750 | 10 > 13 | 15.0 | 4 | delivered |
| 20260926T060325Z | 2750 | 10 > 13 | 15.5 | 4 | delivered |
| 20260926T060325Z | 2750 | 12 > 11 | 11.1 | -1 | delivered |
| 20260926T060325Z | 2750 | 13 > 14 | 13.9 | 3 | delivered |
| 20260926T060706Z | 2750 | 10 > 11 | 11.3 | 4 | delivered |

### 5. Climb after a failure

| bw | level of the clean frame | frames since the failure | next step | n |
|---|---|---|---|---|
| 500 | <= 6 | 1 | 1 | 5 |
| 500 | <= 6 | 2 | -1 | 1 |
| 500 | >= 7 | 1 | 0 | 2 |
| 500 | >= 7 | 1 | 1 | 3 |
| 500 | >= 7 | 1 | 3 | 1 |
| 500 | >= 7 | 2 | -1 | 1 |
| 500 | >= 7 | 2 | 4 | 3 |
| 2300 | <= 6 | 2 | 1 | 1 |
| 2300 | <= 6 | 3 | 1 | 1 |
| 2300 | <= 6 | 3 | 4 | 1 |
| 2750 | <= 6 | 1 | 2 | 2 |
| 2750 | <= 6 | 2 | 1 | 4 |
| 2750 | <= 6 | 2 | 4 | 1 |
| 2750 | >= 7 | 3 | 3 | 1 |
| 2750 | >= 7 | 4 | 0 | 1 |
| 2750 | >= 7 | 4 | 2 | 1 |
| 2750 | >= 7 | 4 | 4 | 2 |

### 6. Failures

| bw | answer to the failed frame | delta to the next frame (count) |
|---|---|---|
| 500 | NAK | -4 (2), -3 (2), -2 (13), -1 (24) |
| 500 | poll between | -2 (8) |
| 2300 | NAK | -4 (2), -1 (1) |
| 2300 | poll between | -4 (1) |
| 2750 | NAK | -4 (3), -3 (1), -2 (6), -1 (12) |
| 2750 | poll between | -3 (2), -2 (2) |

| bw | failed levels > first delivering level | polls in the run | n |
|---|---|---|---|
| 500 | 10 > 8 > 6 | 2 | 1 |
| 500 | 10 > 8 > 7 > 5 | 1 | 1 |
| 500 | 11 > 9 | 0 | 1 |
| 500 | 11 > 9 > 7 > 5 | 2 | 1 |
| 500 | 11 > 9 > 8 > 7 | 0 | 2 |
| 500 | 12 > 11 > 9 | 0 | 3 |
| 500 | 12 > 11 > 9 > 8 | 0 | 2 |
| 500 | 12 > 9 > 7 > 5 | 2 | 1 |
| 500 | 12 > 9 > 8 > 7 | 0 | 1 |
| 500 | 13 > 12 > 11 > 9 | 0 | 2 |
| 500 | 13 > 9 > 8 > 7 | 0 | 2 |
| 500 | 8 > 7 > 5 | 1 | 1 |
| 500 | 9 > 7 > 6 | 0 | 1 |
| 2300 | 10 > 6 | 0 | 1 |
| 2300 | 10 > 6 | 1 | 1 |
| 2300 | 10 > 6 > 5 | 0 | 1 |
| 2750 | 10 > 7 > 6 | 0 | 1 |
| 2750 | 10 > 7 > 6 | 1 | 1 |
| 2750 | 11 > 7 > 5 | 1 | 1 |
| 2750 | 11 > 7 > 6 | 0 | 1 |
| 2750 | 12 > 8 > 6 | 1 | 1 |
| 2750 | 16 > 14 | 0 | 1 |
| 2750 | 17 > 16 | 0 | 2 |
| 2750 | 17 > 16 > 14 | 0 | 3 |
| 2750 | 17 > 16 > 14 (end) | 1 | 1 |
| 2750 | 9 > 6 > 5 | 1 | 1 |
| 2750 | 9 > 7 > 6 > 5 | 0 | 1 |

### 7. Double-length frames

| bw | delivered top-level frames before it (6 = 6+) | remaining >= 2 x single capacity | shape | n |
|---|---|---|---|---|
| 500 | 0 | False | full | 3 |
| 500 | 0 | True | full | 4 |
| 500 | 1 | False | full | 2 |
| 500 | 1 | True | full | 1 |
| 500 | 2 | True | full | 1 |
| 500 | 3 | True | full | 1 |
| 500 | 4 | True | full | 1 |
| 500 | 5 | True | full | 1 |
| 500 | 6 | True | long | 42 |
| 2300 | 0 | True | full | 1 |
| 2300 | 1 | True | full | 1 |
| 2300 | 2 | True | full | 1 |
| 2300 | 3 | True | full | 1 |
| 2300 | 4 | True | full | 1 |
| 2300 | 5 | True | full | 1 |
| 2300 | 6 | True | long | 6 |
| 2750 | 0 | True | full | 7 |
| 2750 | 1 | True | full | 7 |
| 2750 | 2 | True | full | 4 |
| 2750 | 3 | True | full | 1 |
| 2750 | 4 | True | full | 1 |
| 2750 | 5 | True | full | 1 |
| 2750 | 6 | True | long | 4 |

| tape | bw | top level | single frames at it before the first long | long frames | after the last long | next frame |
|---|---|---|---|---|---|---|
| N0CALL-W9SSJ-bw2300-counter4096 | 2300 | 14 | 1 | 0 | - | - |
| W9SSJ-N0CALL-bw2300-counter4096 | 2300 | 14 | 1 | 0 | - | - |
| N0CALL-W9SSJ-bw2750-counter65536 | 2750 | 17 | 6 | 4 | 2937 left, cap 3919 | 16 (2937 B) |
| N0CALL-W9SSJ-bw2300-counter65536 | 2300 | 16 | 6 | 6 | 232 left, cap 3254 | 15 (232 B) |
| N0CALL-W9SSJ-bw500-counter65536 | 500 | 13 | 6 | 42 | 742 left, cap 710 | 12 (591 B) |
| 20260926T050709Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T050752Z-ladder | 2300 | 11 | 1 | 0 | - | - |
| 20260926T050835Z-ladder | 2300 | 11 | 1 | 0 | - | - |
| 20260926T050919Z-ladder | 2300 | 11 | 1 | 0 | - | - |
| 20260926T051002Z-ladder | 2300 | 12 | 1 | 0 | - | - |
| 20260926T051045Z-ladder | 2300 | 11 | 1 | 0 | - | - |
| 20260926T051128Z-ladder | 2300 | 11 | 1 | 0 | - | - |
| 20260926T051211Z-ladder | 2300 | 12 | 1 | 0 | - | - |
| 20260926T051254Z-ladder | 2300 | 12 | 1 | 0 | - | - |
| 20260926T051337Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T051425Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T051513Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T051601Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T051649Z-ladder | 2300 | 10 | 2 | 0 | - | - |
| 20260926T051737Z-ladder | 2300 | 12 | 1 | 0 | - | - |
| 20260926T051826Z-ladder | 2300 | 10 | 1 | 0 | - | - |
| 20260926T051938Z-ladder | 2300 | 10 | 0 | 0 | - | - |
| 20260926T045139Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045221Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045304Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045347Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045430Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045513Z-ladder | 2750 | 12 | 1 | 0 | - | - |
| 20260926T045556Z-ladder | 2750 | 10 | 2 | 0 | - | - |
| 20260926T045644Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045727Z-ladder | 2750 | 12 | 1 | 0 | - | - |
| 20260926T045810Z-ladder | 2750 | 12 | 1 | 0 | - | - |
| 20260926T045853Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T045941Z-ladder | 2750 | 12 | 1 | 0 | - | - |
| 20260926T050029Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T050116Z-ladder | 2750 | 11 | 1 | 0 | - | - |
| 20260926T050204Z-ladder | 2750 | 11 | 0 | 0 | - | - |
| 20260926T050336Z-ladder | 2750 | 12 | 0 | 0 | - | - |
| 20260926T050512Z-ladder | 2750 | 11 | 0 | 0 | - | - |
| 20260926T055838Z-ladder | 2750 | 17 | 2 | 0 | - | - |
| 20260926T060010Z-ladder | 2750 | 17 | 2 | 0 | - | - |
| 20260926T060143Z-ladder | 2750 | 17 | 2 | 0 | - | - |
| 20260926T060325Z-ladder | 2750 | 17 | 1 | 0 | - | - |
| 20260926T060511Z-ladder | 2750 | 17 | 1 | 0 | - | - |
| 20260926T060706Z-ladder | 2750 | 17 | 1 | 0 | - | - |
| 20260926T052115Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052202Z-ladder | 500 | 13 | 1 | 0 | - | - |
| 20260926T052250Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052337Z-ladder | 500 | 12 | 2 | 0 | - | - |
| 20260926T052424Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052516Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052607Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052707Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052812Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T052917Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T053026Z-ladder | 500 | 13 | 1 | 0 | - | - |
| 20260926T053136Z-ladder | 500 | 13 | 1 | 0 | - | - |
| 20260926T053246Z-ladder | 500 | 13 | 1 | 0 | - | - |
| 20260926T053357Z-ladder | 500 | 13 | 0 | 0 | - | - |
| 20260926T053528Z-ladder | 500 | 12 | 1 | 0 | - | - |
| 20260926T053700Z-ladder | 500 | 13 | 0 | 0 | - | - |
| 20260926T053832Z-ladder | 500 | 12 | 0 | 0 | - | - |

### 8. IRS answers by what the ISS does next

| bw | ISS frame | IRS answer | n |
|---|---|---|---|
| 500 | down next | ack16 | 20 |
| 500 | hold next | ack16 | 63 |
| 500 | hold next | idle65 | 4 |
| 500 | last delivered frame | ack22 | 18 |
| 500 | poll | ack22 | 36 |
| 500 | poll | idle32 | 5 |
| 500 | poll | idle65 | 3 |
| 500 | undelivered | ack16 | 41 |
| 500 | undelivered | idle32 | 5 |
| 500 | undelivered | idle65 | 3 |
| 500 | up next | ack16 | 62 |
| 500 | up next | idle65 | 13 |
| 2300 | down next | ack16 | 8 |
| 2300 | hold next | ack16 | 25 |
| 2300 | hold next | idle65 | 2 |
| 2300 | last delivered frame | ack22 | 20 |
| 2300 | poll | ack22 | 40 |
| 2300 | poll | idle32 | 1 |
| 2300 | undelivered | ack16 | 3 |
| 2300 | undelivered | idle32 | 1 |
| 2300 | up next | ack16 | 45 |
| 2300 | up next | idle65 | 15 |
| 2750 | down next | ack16 | 30 |
| 2750 | hold next | ack16 | 33 |
| 2750 | hold next | idle65 | 3 |
| 2750 | last delivered frame | ack22 | 23 |
| 2750 | poll | ack22 | 46 |
| 2750 | poll | idle32 | 4 |
| 2750 | poll | none | 1 |
| 2750 | undelivered | ack16 | 21 |
| 2750 | undelivered | idle32 | 6 |
| 2750 | up next | ack16 | 87 |
| 2750 | up next | idle65 | 18 |

| bw | tapes | IRS answer to the first data frame | n |
|---|---|---|---|
| 500 | ladder (callee is ISS) | idle65 | 17 |
| 500 | manifest (caller is ISS) | ack16 | 1 |
| 2300 | ladder (callee is ISS) | idle65 | 17 |
| 2300 | manifest (caller is ISS) | ack16 | 3 |
| 2750 | ladder (callee is ISS) | ack16 | 2 |
| 2750 | ladder (callee is ISS) | idle65 | 21 |
| 2750 | manifest (caller is ISS) | ack16 | 1 |

### 9. RX line against TX line

| bw | IRS BITRATE RX vs ISS TX | frame | n |
|---|---|---|---|
| 500 | equal | delivered | 180 |
| 500 | no RX line | undelivered | 49 |
| 2300 | equal | delivered | 115 |
| 2300 | no RX line | undelivered | 4 |
| 2750 | equal | delivered | 194 |
| 2750 | no RX line | undelivered | 27 |

### 10. Bytes per frame
Single-frame loads. BW2300 levels 10–16 are the doc 06 §6.1d frame bytes less 3 (899, 1199, 1439, 2159, 2713, 3257);
level 13 was never keyed full.

| level | bw500 | bw2300 | bw2750 |
|---|---|---|---|
| 4 | 43 | 89 | 89 |
| 5 | 90 | 132 | 132 |
| 6 | 132 | 173 | 173 |
| 7 | 194 | 256 | 256 |
| 8 | 260 | 341 | 341 |
| 9 | 312 | 425 | 425 |
| 10 | 392 | 896 | 555 |
| 11 | 471 | 1196 | 1080 |
| 12 | 591 | 1436 | 1441 |
| 13 | 710 |  | 1730 |
| 14 |  | 2156 | 2164 |
| 15 |  | 2710 | 2597 |
| 16 |  | 3254 | 3265 |
| 17 |  |  | 3919 |

### 11. Idle before data

| tape | bw | idle ISS keyings before data | lengths (samples) | keying-to-keying gaps (s) | IRS answers |
|---|---|---|---|---|---|
| N0CALL-W9SSJ-bw2300-counter4096 | 2300 | 5 | 33k x1, 66k x4 | 1.37 2.06 12.09 12.06 | ack22 x5 |
| W9SSJ-N0CALL-bw2300-counter4096 | 2300 | 5 | 33k x1, 66k x4 | 1.37 2.04 12.07 12.05 | ack22 x5 |
| N0CALL-W9SSJ-bw2750-counter65536 | 2750 | 5 | 33k x1, 66k x4 | 1.36 2.06 12.05 12.07 | ack22 x5 |
| N0CALL-W9SSJ-bw2300-counter65536 | 2300 | 5 | 33k x1, 66k x4 | 1.36 2.05 12.05 12.06 | ack22 x5 |
| N0CALL-W9SSJ-bw500-counter65536 | 500 | 5 | 33k x1, 66k x4 | 1.37 2.05 12.05 12.04 | ack22 x5 |

### 12. Cycle per level and time budget

| bw | level | frame | n | cycle (s, median) | bytes | cycle bps |
|---|---|---|---|---|---|---|
| 500 | 4 | full | 3 | 4.86 | 43 | 71 |
| 500 | 5 | full | 23 | 4.87 | 90 | 148 |
| 500 | 6 | full | 19 | 4.7 | 132 | 225 |
| 500 | 7 | full | 10 | 4.27 | 194 | 363 |
| 500 | 8 | full | 14 | 4.27 | 260 | 487 |
| 500 | 9 | full | 13 | 4.28 | 312 | 583 |
| 500 | 10 | full | 10 | 4.28 | 392 | 732 |
| 500 | 11 | full | 4 | 4.3 | 471 | 876 |
| 500 | 12 | full | 15 | 4.43 | 591 | 1067 |
| 500 | 13 | full | 9 | 4.44 | 710 | 1278 |
| 500 | 13 | long | 42 | 8.16 | 1420 | 1392 |
| 2300 | 4 | full | 15 | 5.0 | 89 | 142 |
| 2300 | 5 | full | 15 | 4.68 | 132 | 226 |
| 2300 | 6 | full | 15 | 4.57 | 173 | 303 |
| 2300 | 7 | full | 4 | 4.48 | 256 | 458 |
| 2300 | 8 | full | 9 | 4.47 | 341 | 610 |
| 2300 | 9 | full | 11 | 4.42 | 425 | 770 |
| 2300 | 10 | full | 6 | 4.31 | 896 | 1664 |
| 2300 | 11 | full | 1 | 4.31 | 1196 | 2222 |
| 2300 | 12 | full | 3 | 4.34 | 1436 | 2648 |
| 2300 | 14 | full | 3 | 4.37 | 2156 | 3951 |
| 2300 | 15 | full | 1 | 4.55 | 2710 | 4770 |
| 2300 | 16 | full | 6 | 4.6 | 3254 | 5659 |
| 2300 | 16 | long | 6 | 8.33 | 6508 | 6252 |
| 2750 | 4 | full | 10 | 4.98 | 89 | 143 |
| 2750 | 5 | full | 12 | 4.67 | 132 | 226 |
| 2750 | 6 | full | 32 | 4.57 | 173 | 303 |
| 2750 | 7 | full | 4 | 4.48 | 256 | 457 |
| 2750 | 8 | full | 4 | 4.46 | 341 | 612 |
| 2750 | 9 | full | 9 | 4.4 | 425 | 773 |
| 2750 | 10 | full | 18 | 4.4 | 555 | 1010 |
| 2750 | 11 | full | 6 | 4.31 | 1080 | 2003 |
| 2750 | 12 | full | 9 | 4.31 | 1441 | 2673 |
| 2750 | 13 | full | 9 | 4.33 | 1730 | 3197 |
| 2750 | 14 | full | 12 | 4.38 | 2164 | 3956 |
| 2750 | 15 | full | 15 | 4.38 | 2597 | 4744 |
| 2750 | 16 | full | 12 | 4.56 | 3265 | 5732 |
| 2750 | 17 | full | 15 | 4.62 | 3919 | 6787 |
| 2750 | 17 | long | 4 | 8.35 | 7838 | 7506 |

| tape(s) | bw | top level reached | tapes | bytes | ms per byte (median) | net bps |
|---|---|---|---|---|---|---|
| sweep-bw500 | 500 | 12 | 11 | 22528 | 17.41 | 347 |
| N0CALL-W9SSJ-bw500-counter65536 | 500 | 13 | 1 | 65536 | 5.97 | 1339 |
| sweep-bw500 | 500 | 13 | 6 | 12288 | 22.41 | 334 |
| sweep-bw2300 | 2300 | 10 | 8 | 16384 | 11.51 | 513 |
| sweep-bw2300 | 2300 | 11 | 5 | 10240 | 9.04 | 884 |
| sweep-bw2300 | 2300 | 12 | 4 | 8192 | 9.15 | 819 |
| N0CALL-W9SSJ-bw2300-counter4096 | 2300 | 14 | 1 | 4096 | 5.44 | 1471 |
| W9SSJ-N0CALL-bw2300-counter4096 | 2300 | 14 | 1 | 4096 | 5.43 | 1473 |
| N0CALL-W9SSJ-bw2300-counter65536 | 2300 | 16 | 1 | 65536 | 1.59 | 5021 |
| sweep-bw2750 | 2750 | 10 | 1 | 2048 | 11.2 | 714 |
| sweep-bw2750 | 2750 | 11 | 11 | 22528 | 9.01 | 532 |
| sweep-bw2750 | 2750 | 12 | 5 | 10240 | 9.07 | 544 |
| N0CALL-W9SSJ-bw2750-counter65536 | 2750 | 17 | 1 | 65536 | 1.35 | 5947 |
| sweep-bw2750-quiet | 2750 | 17 | 6 | 178311 | 2.49 | 3383 |

## Contradictions

With `packages/hfmodem/hfmodem/kestrel/vara/vara_arq.py`:
- **:2265–2288 `_over_level`** keys every data frame at the base level (or the one-lower close). A high speed capable
  ISS starts at 4–6 and climbs by up to 4 a frame (rules 1, 2).
- **:570 `_LEVEL_STEP`** maps the receiver's continue to steps of +2, −1, +1, +3, −1. A +4 step occurs 66 times
  (4→8→12 on every clean tape), and nothing in the table produces it.
- **:561 `_SINGLES_BEFORE_PAIR = 6` with :2402** pairs only at BW500's base level. The same six-then-double rule
  holds at the top level of all three bandwidths (rule 5). The :2399 docstring lets the close ride as a pair's second
  frame; 52 of 52 long frames here were full and the close came as a single frame one level down.
- **:2957–2963 `_data_nak_level`** drops one record at the wide bandwidths and to the bottom at BW500. High speed capable
  falls are 1–4 levels at every bandwidth, and BW500 never reaches its bottom (4) from a failure.
- **:3228–3241**, the explicit-level entry ladder, descends one level per full frame at the start of each delivery
  (the clamp at :3241). VARA never descends at the start of a delivery; it descends one level per frame only on a
  low SN reading, which is the same rate.
- **`arq/phy.py:153 close_level`** treats the close's drop as a fixed record behind `0x81`. High speed capable, it is one case
  of the frame-count rule, which applies at every level and also holds or raises the level (rule 2).

With `docs/protocols/vara/06-speed-gearshift.md`:
- §6.2 "one level per shift": +1 to +4, with the jump and frame-count caps.
- §6.2 "decode success, SN not a threshold": a per-level SN threshold ladder fits BW2750 (table 4); at BW500 the fit is
  loose (non-monotone at 9–11), so the printed SN is close to the metric but not the metric itself.
- §6.2 blocks per over "after several successes": exactly six, at the top level, and only while two loads remain.
- §6.3 "one level after a single failed over": −1 needs no failure (a delivered low-SN frame), and a failure falls
  1–4.
- §6.4 "the IRS only reports what it decoded": the IRS's reading sets the ISS's step and its starting level.
- §6.1c rec4–8 "frame B 135–214": levels 6–9 carry 173, 256, 341 and 425 bytes at BW2300, so rec5–8 frames are
  176–428 bytes if the frame-less-3 relation of rec9–15 holds.
- §6.5 "an impairment sweep reaching levels 0–2": a high speed capable pair never goes below 4 in these ladders, down to
  about 0 dB wideband SNR.
