<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Missing short-final DATA also needs a lower retry

The K0SI 2026-09-19 22:48 recording exposes two independent problems. Document 27
covers the NACK scan deadline. This document covers what to transmit after its
288/1117 negative answer to a final-answer query.

When a 79-byte BW2300 base-record final DATA frame goes unread,
a VARA HF station answers the final query with 288/1117. Three unchanged
record-3 retransmissions deliver zero bytes. A record-2 retry also
delivers zero. A record-1 retry, holding that record through the remaining
22/22/22/13-byte delivery, delivers all 79 bytes exactly. The clean control
succeeds. Independent recordings check the missing DATA, control payloads, every
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
healthy independent captures, unchanged sources and owned cleanup.
No receive or retransmission policy is patched in these production cases.
The record-1 and record-2 candidates were comparisons only, not production.

Recordings:
- 20260919T231333Z: clean and unchanged-retry negative, 79 bytes.
- 20260919T231638Z: candidate record-1 success, 79 bytes.
- 20260919T231815Z: candidate record-2 negative, 79 bytes.
- 20260919T232159Z: production clean/loss pair, 79 bytes.
- 20260919T232621Z: production clean/loss pair, 17 bytes.

251 focused regressions cover native NACK replay, payload/boundary preservation,
refused transmission, query freshness, retry limits and previous recovery cases.
This extension does not qualify short-final retry geometry at other bandwidths
or records; those are part of the separate NACK matrix campaign.
