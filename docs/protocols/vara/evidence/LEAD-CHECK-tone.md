# The tone-family lead law at BW2750 and BW500

`lead_check_tone.py` prints every table behind this note in about 3 minutes. It reads the high speed capable
bench tapes for measurement only. Each over's stream state is `lead_study.sessions`' prediction from the
session history, with nothing fitted. The tone rows are read by `rx.varatone` and rendered by the
committed `tx.varatone_tx.cells` from `VB6Rnd(state)`. A lead cell matches when tape / render,
normalised at the reference cell, rounds to quadrant 0. That comparison is taken after the grid
de-rotation that the render carries. Cells below 0.3 of their reference's amplitude are absent from the
tape, because the capture opened a symbol late, and are not counted.

## Result

**The law holds at BW2750 L5-10 and BW500 L5-6 unchanged. BW500 L4 keys the same 18 draws at the same
E1 state in the same layout, but it maps each draw as `Int(Rnd*2)` half turns (BPSK), not
`Gray(Int(Rnd*4))` quarter turns.** Neither QPSK at E1 nor any state near the old `PREAMBLE_SEED`
constant fits BW500 L4.

| row | overs (floor silent) | CRC-clean overs with every present cell matched | cells matched / present | render residual, lead fixed at the state (lead free) |
|---|---|---|---|---|
| BW2750 L5 | 11 (1) | 7 | 246/257 | 2377737: -48.9 dB (-49.2); 2136503: -48.9 (-49.2) |
| BW2750 L8 | 6 (3) | 3 | 310/320 | 2195449: -49.3 (-49.5); 3463376: -49.2 (-49.4) |
| BW2750 L10 | 23 (11) | 18 | 2168/2177 | 1067991: -50.9 (-51.2); 1115380: -50.9 (-51.2) |
| BW500 L5 | 24 (4) | 22 | 379/382 | 2237489: -52.0 (-52.3); 3404588: -52.0 (-52.3) |
| BW500 L6 | 19 (5) | 17 | 453/456 | 2472384: -49.7 (-49.9); 3639021: -49.7 (-49.8) |
| BW500 L4, BPSK | 21 (20) | 20 | 318/318 | 1948805: -52.0; 170629: -30.5 (its lead t = 0 is absent) |
| BW500 L4, QPSK (the tone law) | 21 (20) | 0 | 267/318 | 1948805: -17.6; 170629: -24.7 |
| BW2750 L6 (extra) | 34 (4) | 25 | 948/981 | -50.9 (-51.2) on two |
| BW2750 L7 (extra) | 10 (3) | 4 | 239/240 | -24.8 (-24.8) on two, see below |
| BW2750 L9 (extra) | 11 (3) | 4 | 702/716 | 1142793: -51.6 (-51.9) |

- **Every floor-silent CRC-clean over matches on every present cell.** Every miscounted cell is on an over
  whose floor is -23.6 dBFS or louder. On all of them the predicted state still scores above every
  other state within ±300 draws (0.69-0.99 against at most 0.62). No over in any row reads better
  anywhere else in ±300. The fallback layouts were never needed: ref row last, band-major, and no Gray.
- Fixing the lead at the predicted state costs 0.1-0.3 dB against a free least-squares lead. That
  is the whole over's residual. So the rendered lead is the tape's lead to the model's floor.
- The CRC-failed overs have lock ≤ 0.13, apart from BW2750 L9 2602406 / 3786650 and L10 2578187, where
  lock is 0.39-0.69. Those three still read their lead at the state (0.96-1.00).
- The BW500 L5 and L6 overs are all responder (b2a) keyings on the ladder legs. The BW2750 L8 set includes
  one caller over, 2195449.

## BW500 L4

The DBPSK base level's lead cells lie on the half-turn lattice: no present cell is off it, over 20 clean
overs. These are 18 link-setups at one state (draws 70, E1) and data overs at draws 89 and 1128. The law
is otherwise the tone law. It uses 9 time-major rows over the two sub-bands. Row 0 turns the reference
column 9. Rows 1..8 are lead columns 1..8, each times conj(grid480) at the index that data column 9 + t
reads. The alphabet is `2·Int(Rnd*2)` quarter turns.

Over the three distinct states jointly, the predicted position scores 0.979 (one absent cell). The best
other offset within ±300 scores 0.708, at +132. A 16-cell binary lead cannot single out a position
alone, since the best alternative on one over reaches 0.94. The joint score does. Against the old
constant, `PREAMBLE_SEED` + 0..6000 draws, the best position reaches 0.88-0.94 per over, which is the
chance level for that many positions. It differs from over to over (+921, +2325, +613), so the constant
is excluded. The QPSK tone law at E1 matches only where the Gray value happens to be even (267/318). The
residual shows the same thing: -52.0 dB with BPSK against -17.6 dB with QPSK on 1948805.

## Side finding

BW2750 L7 renders at -24.8 dB whether the lead is fixed or free, so the lead is not the cause.
KPA-tone-bw2750-bw500 reached -54.6 dB with its P8 fit. The product's `ROWS[2750, 7]` pulse or geometry
therefore departs from that fit. Not investigated here.

## What must change (not touched here)

- `tx/varahf500_tx.synth_burst(lead=)` (working tree, uncommitted) keys `Gray(Int(Rnd*4))`. It should be
  `2 * rnd.draw(1)` per cell, still 18 draws. Its session state is E1 plus the lead_study cost table, not
  `PREAMBLE_SEED`.
- `tx/varatone_tx.py` docstring: "at BW2750 and BW500 only the draw count is confirmed". The lead
  content is now confirmed cell by cell at BW2750 L5-10 and BW500 L5-6.
- LEAD-GENERATORS.md "What remains unknown", third item: closed by this note.
