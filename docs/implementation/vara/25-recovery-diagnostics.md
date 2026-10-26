<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# DATA recovery diagnostics

Intermediate queries have distinct diagnostic names:
`session-intermediate-answer-query-a` (60/683) and
`session-intermediate-answer-query-b` (60/745). The short-final query retains
`session-final-answer-query` (60/683). These labels distinguish caller intent;
the underlying measured waveforms are unchanged.

Query, refusal, retry and unconfirmed-close logs report the pending over number,
record, pending payload-byte count, queued-byte count, query attempts and NAK
retry count. Payload contents are not printed. A refused lower retry therefore
reports the original retained geometry, rather than implying that repacketization
already committed.

Positive query replies report the matched descriptor and seed/preadv, including
288/1, the scoped 288/311 last-full reply, and 288/249 retry/base continuation.
Previously all three could be logged as `session-intermediate-query-answer`.
The context is captured before the pending frame is retired, so the log identifies
which bytes that answer confirmed.

Examples from the same recovery boundary:

```text
tx session-intermediate-answer-query-b keyed-by=KC9GHZ
intermediate query 2/3 (60/745) — retaining over #1, record 101, 22 pending bytes, 72 queued bytes, intermediate queries 2/3, NAK retries 1/3
rx session-retry-query-answer (288/249) — acknowledged over #1, record 101, 22 pending bytes, 72 queued bytes, intermediate queries 2/3, NAK retries 1/3
```

The names do not broaden recognition or change delivery ownership. Waveform
identity and native reply/action diagnostics are checked by regression tests.

Interactive CHAT uses at most four unanswered status queries per transmitted
DATA over, in either connection role. It then closes with delivery unconfirmed.
Queued host writes and rejected receive candidates do not restart that budget;
refused transmissions do not consume it. A qualified NAK can authorize a DATA
retry under the separate, existing three-retry bound. Non-chat BW500 retains
the stock eighteen-query allowance. This is a local fail-fast policy, not a
claim that the stock protocol requires four queries.

W1HPC on 2026-10-25 reached seventeen of the old eighteen queries while our QSY
invitation remained unconfirmed. Its recorded DATA rejection has all 29 leading
payload tones correct, with the final two unreadable. Requiring a clear final
tone caused us to ignore the rejection and query again instead of retransmitting.
For a fresh, solicited BW500 DATA NAK, all 29 leading payload tones may now
establish identity when the tail is erased; any clear wrong tone still rejects
the candidate. The entire nominal frame and turnaround guard must elapse before
we transmit. The stream reader retains the prefix through that guard. Direct
unsolicited replies and positive acknowledgements retain their existing rules.

Replaying the recorded receive gaps now causes the third query's reply to
authorize a lower-rate resend, retaining every unconfirmed byte. A paired-modem
test then completes delivery of the 28-byte QSY message exactly once. Negative
tests cover insufficient identity, wrong calls, stale queries, incorrect turn
ownership, and premature transmission. The query limit is a separate safeguard,
not the synchronization repair. These offline results do not establish an
on-air recovery or explain why the recorded tail tones were unreadable.

## Receive-side state and queries

Connection role and transmit-turn ownership are different. A caller can be
receiving and an answering station can be sending. Role selects the control
waveform; ownership and the outstanding delivery obligation select the action.

The answering station's query dispatch follows this table:

| State | Valid peer query | Transition |
| --- | --- | --- |
| Fresh setup, before a data grant | Setup query | Repeat setup idle; no payload credit |
| Peer granted the turn, DATA missing | Any supported DATA query | Send NAK; retain missing-data obligation |
| Valid DATA received, ACK owed | Any supported DATA query | Repeat saved ACK; do not deliver again |
| Receiving an unfinished window | Query-like input | Do not interrupt the window |
| Our transmit turn, closing, or no receive obligation | DATA query | No receive-side response |

A successfully transmitted grant creates the missing-data obligation. It must
not depend on detecting a damaged packet: the channel can erase that packet
completely. A refused grant creates no obligation. A new grant discards the
previous delivery's saved ACK so it cannot acknowledge unread new data.

Valid retransmitted DATA replaces the missing-data obligation with its ACK.
Queries and repeated answers do not reset the progress clock. Successful
responses consume the recovery budget; refused transmissions do not. Exhausted
query recovery closes without claiming delivery. Locally queued chat does not
alter these receive-side decisions.

The W5JAB recording from 2026-10-25 exposed a missing grant-to-DATA transition:
the modem granted the caller's turn but retained no receive obligation, then
silently ignored complete W9SSJ-addressed queries. Tests replay the recorded
post-grant window and queries, erase DATA entirely, exercise all supported
query encodings, and verify successful resend, duplicate suppression, refused
transmission, wrong-call rejection, and bounded recovery. These are offline
checks; they do not establish a subsequent on-air QSO.

## Recovery invariants

The reliability review treats recovery as the continuation of one outstanding
transaction, rather than a new transaction on every received control frame:

| Event | Required behavior |
| --- | --- |
| Repeated grant request while DATA is missing | Repeat the grant; preserve the rate, receive obligation, recovery count and progress deadline |
| Turn request while our DATA is unacknowledged | Retain our turn and pending bytes; no grant |
| Turn request during a held receive window or close | Preserve that transaction; no grant |
| Failed DATA, including a refused NAK transmission | Replace any older positive answer with a missing-DATA obligation; adapt rate only after successful transmission |
| Accepted DATA, but ACK transmission refused | Save this DATA's ACK for the next query; never repeat an older delivery's ACK |
| Short or long caller NAK after our query | Require a fresh successfully sent query for this exact pending frame; consume it once; no delivery credit |
| Peer poll or idle during unfinished work | Respond as appropriate without restarting the work's progress deadline |

Healthy idle chat can still use peer contact to maintain the link. An unexpected
drained frame is not progress; an actual requested handover is. Regression tests
in `test_recovery_state_invariants.py` exercise these transitions with generated
control audio and recorded queries, including interleaved grant/query cycles,
transmission refusals and stale responses.

KQ4QWH's 2026-10-25 BBS transfer exposed a separate receive-rate disagreement:
after predicting level 1, the active session located a complete CRC-clean
level-4 retransmission and then discarded it solely because of the prediction.
The peer's next idle drew another NAK. A transmitted NAK does not prove that
the peer heard it. The BW500 base reader now accepts complete CRC-clean DATA
at an earlier rate in the active retry ladder, subject to the same receive-turn,
broadcast, echo and duplicate gates. Unrequested rates and failed off-level
high-speed candidates remain rejected. The locator log reports the actual and
predicted levels separately when they differ. Recorded replay tests cover both
stream callback sizes and the bracket reader; delivery occurs once.
Requested-rate history also survives replacement of an unheard direct NAK by
a missing-DATA query response. A new recovery transaction starts fresh history.

The same KQ4QWH contact ended locally at 21:42:56 UTC after duplicate bytes
corrupted application framing. A 332-byte delivered block was resent in smaller
pieces; 209 matching bytes were buffered. Another NAK caused that prefix to
restart. The old filter appended the restart to the buffered prefix, saw a
mismatch, and released duplicate bytes as new data. The captured file stream
contained 541 duplicate bytes and crossed its declared message length while
the actual download was still incomplete.

Replay matching now retains an outer obligation while matching retransmissions
of its buffered sub-overs. Repeated queries for the same sub-over do not add
obligations. An exact repeated sub-over does not erase the outer prefix. A
mismatch or early final boundary still releases genuinely different data.
Recorded payload tests exercise OFF and TEXT compression and replay the actual
application failure: corrected bytes remain a valid incomplete message; they
do not claim a complete or checksum-verified file.

KQ4QWH's subsequent 22:00 UTC attempt exposed an acquisition deadline bug.
Its response to the sixth request decoded at 15/15 tones. Link setup transmitted
22:01:01.744–22:01:06.320 UTC, crossing the original request deadline. The host
reported DISCONNECTED at 22:01:06.392, before listening for the setup reply.
The recorded reply at samples 12860000–12963328 establishes CONNECTED when
replayed into the still-pending handshake; it is a valid connected ACK.

The audio owner now gives the first successfully transmitted outgoing setup at
least 30 seconds from unkey to complete acquisition. This is a one-time stage
allowance, not an activity timer: retries and restarted requests cannot renew
it, refused setup output cannot earn it, and operator abort remains immediate.
An early response does not shorten the existing attempt window. Inbound
acquisition retains its separate PENDING-based deadline. The recorded ACK and
owner-clock regressions are in `test_late_connect_setup.py`.

KL7QR's later 2026-10-25 transfer revealed two additional recovery mistakes.
A CRC-clean L3 frame (samples 35620864–35939328) was rejected while L1 was
predicted. A quick query had replaced a direct NAK in the inferred history:
6,M,M predicts L1, whereas a peer hearing the direct NAK can take 6,N,M to L3.
Replacement now retains that alternative rate for complete CRC-clean base DATA;
failed candidates still must match the prediction. No transmit ladder changes.

At 22:10:29 and 22:11:00 UTC, peer queries after acknowledged L6 overs drew
unnecessary NAKs. The subsequent L4/L5 retries contained the already delivered
143 and 145 bytes respectively. The queries began roughly 3.68 seconds after
our ACK, past the global shortest-over threshold (3.672 s), but before a new
L6 DATA frame (4.117 s) could fit. The ACK-recovery decision now uses the
expected high-speed frame duration. These early queries repeat the ACK;
a query late enough to follow unread DATA still takes the recovery path.
Recorded tests cover both queries at three callback sizes and the clean L3
retry, along with duplicate delivery prevention and fresh-history isolation.


N5YYF's 2026-10-25 queued 28-byte QSY response exposed a receive-side final-ACK
recovery error. Repeated peer idles drew short turn requests even though the
peer had not retired its last DATA. The isolated stock/stock BW500 CHAT bench
repeats caller-keyed long turn-idle (850/63) until delivery is confirmed, then
sends the long turn request (850/1), and sends DATA only after a separate grant.
Losing eight recovery replies does not change that sequence.

The caller now preserves that distinction explicitly: pending ACK confirmation,
pending turn grant, and owned transmit turn. Repeated ACKs do not count as
progress or unanswered turn requests. A refused ACK keeps the delivery debt;
a refused request retains the confirmed state for a long-request retry.
Confirmation alone never authorizes DATA. This measured path is scoped to
BW500 CHAT callers with completed receive delivery and queued output.

Recorded fixtures in `test_n5yyf_ack_recovery.py` cover long idle-response,
11-pair confirmation (called-station responder stream at pair 135), and short
responder turn-idle confirmation. The last form was observed on the native/stock
bench after stock retired its buffer. Separate recorded grants complete the
exchange. Wrong-peer and duplicate confirmations cannot deliver bytes or
refresh progress. The existing no-progress budget still bounds failed recovery.

The recovery confirmation must also be complete before requesting the turn.
The existing long-frame recognizer can identify a frame from a partial payload;
using that identity immediately caused a request to overlap the remaining
confirmation in the first native/stock bench. Recovery now waits through the
last symbol. Prefix-only fixture tests require silence and unchanged progress
until the rest of each confirmation arrives.
