# Missing short-final DATA also needs a lower retry

The K0SI 2026-09-19 22:48 recording exposes two independent problems. Document 27
covers the NACK scan deadline. This document covers what to transmit after its
288/1117 negative answer to a final-answer query.

With a 79-byte BW2300 base-record final DATA frame erased on the virtual cable,
stock VARA HF 4.9.0 answers the final query with 288/1117. Three unchanged
record-3 retransmissions deliver zero bytes. An experimental record-2 retry also
delivers zero. A record-1 retry, holding that record through the remaining
22/22/22/13-byte delivery, delivers all 79 bytes exactly. The clean control
succeeds. Cable audits independently check the erasure, control payloads, every
retry CRC and concatenated application bytes.

The existing two-record NACK recovery now includes a freshly queried BW2300
short-final base frame, with no later host write queued and no explicit TX
ladder. The pending short frame is itself the delivery boundary. Repacketization
retains every unconfirmed byte and the original over number; it commits only
when the retry actually transmits. Full lower frames hold record 1 through the
closing frame. A payload fitting a short lower frame needs no additional close.
A full lower frame still needs its closing block, including an empty one when
necessary. Query and retry limits remain bounded. No NACK retires bytes.

Production clean/loss pairs pass for both 79-byte and 17-byte final payloads:
exact application bytes, 225 reverse bytes, no pending DATA or queued blocks,
healthy independent cable captures, unchanged sources and owned cleanup.
No receive or retransmission policy is patched in these production cases.
The comparative lower-rate candidates were explicitly bench-only experiments.

The private station archive identifies the experiments as follows:
- 20260919T231333Z: clean and unchanged-retry negative, 79 bytes.
- 20260919T231638Z: experimental record-1 success, 79 bytes.
- 20260919T231815Z: experimental record-2 negative, 79 bytes.
- 20260919T232159Z: production clean/loss pair, 79 bytes.
- 20260919T232621Z: production clean/loss pair, 17 bytes.

251 focused regressions cover native NACK replay, payload/boundary preservation,
refused transmission, query freshness, retry limits and previous recovery cases.
This extension does not qualify short-final retry geometry at other bandwidths
or records; those are part of the separate NACK matrix campaign.
