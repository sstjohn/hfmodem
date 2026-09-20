# DATA acknowledgements and lost-answer recovery

Measured with VARA HF 4.9.0, including new BW2300 stock-to-stock and
Kestrel-to-stock experiments on 2026-09-19. This section supersedes any earlier
suggestion that silence after DATA means the DATA should be repeated.

A missing answer is ambiguous: the peer may have received DATA and advanced, or
may still be missing it. Preserve the outstanding bytes and ask for state.
Only a decoded negative answer licenses a DATA retransmission. Energy, duration,
and a turn request alone do not establish delivery.

## Observed exchanges

The descriptors below use the existing `vara_frames` seed offset / preadvance
notation. These are HF observations; the VARA FM connection-request article by
Billy Penley describes a different exchange.

| Situation | Caller action | Stock responder answer | Result |
| --- | --- | --- | --- |
| Intact intermediate answer | Decode the eight-symbol two-tone continue | — | Retire exactly one pending block, send the next |
| Lost intermediate answer | Called-keyed 60/683 or 60/745, alternating with successful full DATA boundaries | Called-keyed 288/1 | Pending DATA was received; advance exactly once |
| Lost intermediate answer with a speed decrease | Same phased query | Called-keyed 288/311 | Pending DATA was received; lower the next record one step and repack unsent bytes |
| DATA itself erased | Same query, pending bytes retained | Called-keyed 288/1117 (`SESSION_OVER_NAK_RESPONDER`) | Retain the block; retransmission is now justified |
| Lost short-final answer, no further blocks queued | Called-keyed 60/683 | Fresh caller-keyed responder turn request | Send called-keyed 60/807 (`SESSION_DRAINED`); retire pending only after that handover transmits |
| Query or its answer also unreadable | Repeat the same pending block's query, bounded | No valid positive answer | Never advance or blindly replay DATA; close with delivery unconfirmed when the budget expires |

Both wide bandwidths, 2300 and 2750, have controlled delivered-DATA and
missing-DATA evidence. The new BW2300 four-loss experiment reproduced the prior
BW2750 sequence: queries 683, 745, 683, 745; replies 1, 1, 1, 311.
The full-DATA phase advances on every successfully keyed full frame, including
a NAK retry. Queries, refused writes and unrelated idles do not advance it.
A two-stock double-loss experiment exposed the retry case; see
[lower-record recovery](24-lower-record-recovery.md).

The implementation starts the intermediate query deadline 1.8 seconds after
DATA unkey. It allows 4.5 seconds per intermediate query reply and at most three
successfully transmitted queries per pending block. These are bounded recovery
policy values, not a claim that stock uses exactly these retry limits. Refused
transport writes spend no transmission budget. An explicit NAK retry starts a
new DATA reply window without replenishing either budget.

Final-query confirmation retains the existing two-second wall/sample freshness
bounds and requires the raw receive stream's complete reply. A pre-query or stale
turn request can prompt a query but cannot itself retire DATA.

## Why yesterday's short replies were missed

Earlier September 18 changes already measured the link's fractional frequency
offset and added a whole-frame matcher for damaged eleven-symbol final ACKs.
The remaining eight-symbol continuation failures have a separate cause: the
receiver mute and the transport's post-transmit guard remove the opening symbol.
A generic shape detector correctly refuses a burst without that symbol.

Three native KC9GHZ BW2300 reply windows from September 18 have all seven tail
pairs intact. They agree with three independently host-confirmed stock replies
on September 19. `OVER_CONTINUE_RESPONDER_BY_LINK` now includes that full link.
The matcher still requires all seven measured tail pairs live, clear and exact,
an unavailable lead, an outstanding full intermediate block, and more queued
DATA. NAK and final-control vetoes precede it.

The receive cursor can begin after the missing lead. The matcher pads one symbol
of unavailable time before its analysis, allowing the intact tail to align; this
adds no evidence and reconstructs no received tone. Without it, the 05:29 UTC
session's third DATA reply could not align even with the correct tail table.
The 23:17 and 23:50 UTC sessions also recover. Tests feed the exact transport
cursor crops through 512-, 4096- and 4800-sample callbacks and verify one distinct
next DATA frame, with no second advancement on a replayed bracket.

This tail exception is scoped to measured full links. Other unreadable short
replies take the query exchange instead of receiving an assumed ACK.

## Weak known eight-symbol answers (September 20)

K0SI's 15:05 UTC BW2300 session supplied all eight expected pairs after DATA
#4 and #6. The generic shape gate rejected them because the lead of #4 and
two pairs of #6 did not clear its 2 dB threshold. Each subsequent 288/1 query
answer confirmed the same pending block without a DATA retransmission.

For independently measured full-link ACKs, a separate matcher requires all
seven tail pairs exact and live, the final pair clear, and at least six clear
pairs overall. An unclear lead can be ignored; a clear contradictory lead,
missing interior pair, wrong link, or unknown tail cannot. Eight consecutive
alignment offsets (5.3 ms) are required. The native #6 has nine such offsets.
Existing NAK and final-control vetoes still run first. This does not relax the
generic detector or teach unknown replies an ACK meaning.

The exact native receive-cursor crops, replay tests, and provenance ship under
`tests/kestrel/fixtures/k0si-weak-continue-0920/`.

## ACK with a speed decrease, and unnecessary idles (September 20)

KC9GHZ's 14:51 UTC BW2750 session answered DATA #3 and every query. Its
immediate short reply was damaged; the first query reply was an exact 288/311.
The receiver recognized it, but the state machine wrongly restricted acceptance
to one remaining short block. Four queued blocks therefore stalled the link.

Independent stock-caller replay establishes the missing meaning. With 438 bytes
still buffered, both generated 288/311 and the actual first recorded query reply
produced `BUFFER 349`, then `BITRATE (3)`. The next native DATA frame had a valid
CRC and contained the next 47 bytes, starting at byte 178 of the queued message.
A generated short DOWN-pair ACK produced the same result without a query.
Repeating 288/311 descended from level 3 to 2 at BW2750, and from levels 4 through
1 at BW2300, retiring exactly 89, 47, and 22 bytes at successive boundaries.
Thus 288/311 acknowledges the pending DATA and requests a relative speed decrease;
it is not a NAK or merely a closing-block marker.

The wideband sender now repacks only the unsent current delivery, preserves later
host writes, and holds the requested lower record through that delivery. This
also honors a peer decrease when the operator requested initial level 4. Explicit
lower-target ladders retain their existing behavior. A refused transmission
retains the repacked bytes and their planned records. Strict query freshness,
full-payload matching, and pending-block identity checks remain required.
The partial long-answer reader now schedules its next scan at the reply's tail;
its earlier return used to bypass the strict query reader's fine scan schedule.

Earlier in the same session, noise behind two CRC-clean greeting blocks exceeded
the window detector's 0.4 relative spectral-peakiness threshold (about 0.53 and
0.52), falsely holding their ACKs. The gateway's first subsequent idles each fit
31/31 tones. A complete, fresh, called-keyed idle at the early cadence, before a
second DATA frame could finish, now releases the held ACK. It requires at least
24 exact clear payload tones and a clear last tone, and is limited to 3.0–3.71 s
of held audio. A partial name or later idle keeps the existing two-idle/window
protection. This reduces the observed 7/11 s delays to the first idle; it does
not claim that the initial noise-based hold has been eliminated.

Native reply fixtures and stock replay evidence are in
`tests/kestrel/fixtures/kc9ghz-downshift-0920/`; early idle fixtures are in
`tests/kestrel/fixtures/kc9ghz-early-idle-0920/`. The bounded virtual-only stock
harness and full cable recordings remain in the private station archive.
The installed stock modem is limited to the four low levels; this does not
establish higher-level or BW500 downshift behavior.

## Evidence and limits

Native, shipped control-only fixtures and source sample boundaries are under
`tests/kestrel/fixtures/data-replies-0919/`. The full development artifacts
remain in the private station archive and are not required at runtime.

- Stock BW2300 clean and four-intermediate-ACK-loss runs: 393 bytes caller to
  responder and 230 bytes back, exactly, empty buffers and clean disconnects.
- Stock BW2300 DATA-loss run: responder host had zero bytes and both peers were
  connected at the intact query; its 288/1117 reply matched all 32 symbols.
  The source DATA was live while actual responder input was erased for 217088
  samples. Stock recovered all bytes, initially retransmitting at a lower record.
- Stock BW2300 short-final ACK-loss run: damaged initial ACK, then exact
  query/request/drained controls and 79/230 exact host bytes.
- Kestrel BW2300 B2F clean and two-ACK-loss runs: 235 outgoing and 645 incoming
  message bytes, exact in both directions. Independent cable/actual-RX audit
  verifies both ACK erasures, phased queries, intact positive replies and three
  distinct CRC-clean body frames. No DATA repeats or audio errors.

The two-ACK-loss B2F run used the production CLI defaults. Existing
`--no-data-retries` and `--probe-intermediate-query` options remain accepted for
script compatibility; wideband state queries no longer require them.

These runs used virtual audio, not RF. Recovery includes the base-record wideband caller states measured here and
the [qualified lower records](24-lower-record-recovery.md).
Unsupported bandwidth/record/queue states close with bytes unconfirmed rather
than speculatively retransmitting or advancing them. Fresh solicited NAKs at a full base frame can now trigger the measured
[lower-record repacketization](23-nak-repacketization.md); other NAK retries
retain the current record and block geometry.
