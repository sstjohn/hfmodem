# What the caller keys during the peer's turn — measurements, 2026-09-16

The figures the peer's-turn code is written against. Cited from `vara_arq` by path
so the constants are checkable against something in the tree.

## Bench: what a stock caller answers an idle with

Stock VARA HF 4.9.0 as the caller on a cable, 2026-09-16, four idle trains.
**45 of 45 idles answered**, one burst each, keyed 0.11-0.14 s after the idle's
last sample:

| peer frame | caller's answer | BW2300 | BW2750 |
|---|---|---|---|
| `session-responder-idle` (288/683) | one `session-over-nak`, keyed to the CALLED station | 31/31 tones x20 | 28/31 tones x20 |
| `session-responder-over-idle` (288/745) | one `session-keepalive-a` | 31/31 tones x20 | 24/31 tones x20 |

With no peer frames at all the caller keys **nothing for 60 s and then
disconnects**, and every idle it reads restarts that timer. So the caller's own
clock produces no unprompted burst in the peer's turn; what it answers is the
peer's cadence, and what bounds a stalled link is a silence timer rather than a
keying budget.

This is what `_answer_peer_idle` implements, and why `_has_over_nak_frame` is
bandwidth-agnostic while `_has_idle_over_nak` — the BW2750-only *retransmission*
ladder — is not.

## K0SI 40 m, `logs/onair/20260916T023202Z-W9SSJ-K0SI.wav`

Arm log: `vara-morning-2300-l4-tx-k0si-2300-20260916T023155.412560Z.log`

| what | when | notes |
|---|---|---|
| `session-drained-responder` | 67.334 s, 76.734 s | 31/31 tones, 0.12 s after our ACK |
| `session-responder-idle` x3 | 56.639, 60.386, 64.135 s | 3.747 s apart, 2.38 s of gap between bursts |
| our final over, last sample | 95.648 s | |
| ACK-slot burst | 95.80 s | |
| responder turn-request | 101.262 s (0.74) | **+5.61 s** after our over |
| responder turn-request | 104.922 s (0.94) | **+9.27 s**; train cycles 3.66 s |

Two things come out of this tape:

* The three 683 idles 3.747 s apart, with our ACK held across them, are the case
  `_release_held_answer` releases on — the ACK went out 11.6 s late without it.
  A window's own second block is a 4.4 s DATA over, so it cannot present as a
  pair of short idles a cadence apart; that is the whole discriminator, and it is
  why `_IDLE_PAIR_MIN_S` is 3.0 s and why the pair is judged at naming time.
* The turn-requests at **+5.6 s and +9.3 s** are why `_TURN_REQUEST_ANSWER_S` is
  12.0 s and not `_FINAL_QUERY_REPLY_S`. That constant is 2.0 s and measures a
  different interval — a *query's* unkey to its reply — so bounding the
  turn-request with it would never fire on this, the case it was written for. At
  the stock bench the responder's turn-request train starts +5.7 s behind the
  frame it follows and cycles every 3.64 s; three of those slots is ~12 s, which
  covers the first three requests and stops short of the peer's own next cadence.

## KD7UHR, `logs/onair/20260916T041101Z-*`

* `session-responder-idle` x10 on a 3.262 s grid, 1.9 s of gap between bursts.
* The gateway's greeting over runs 50.53-54.75 s. Our keepalive — keyed on our
  own clock, unprompted — went out 0.7 s before that over ended and flushed it.
  The over decodes clean through the live path when nothing keys.

This is the loss that removing the unprompted keepalive cadence fixes. It is
*not* fixed by a guard on the over-search buffer: during an over's arrival the
search has no whole frame to report, so no read-only predicate over its state can
see the arrival at all.

## NS0A, `logs/onair/20260916T043220Z-*`

* `session-responder-over-idle` x3, 2.0 s of gap between bursts, all three inside
  a hold while we sat on a decoded ACK — the same 11.6 s late answer as K0SI, on
  the 745 frame instead of the 683.
