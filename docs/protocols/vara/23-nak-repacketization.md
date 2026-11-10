<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# NAK-triggered repacketization

For a missing full base-record DATA frame, a VARA HF responder answers the
caller's state query with called-keyed 288/1117. Independent recordings between
VARA HF stations at BW2300 (2026-09-19) and BW2750 (2026-09-12) show the
caller restarting with the first **22 unconfirmed bytes** at record 1 / 101,
then returning to base records with the remaining stream. All three decoded
frames in each capture have clean CRCs; the lower frame's full-body field is
0x99 in the 393-byte recording.

Kestrel now prepares that smaller retry for a fresh solicited NAK while a full
base frame is pending. It combines the original pending payload with the unsent
remainder of the same host delivery, takes 22 bytes for the lower frame, and
repartitions the remaining bytes into base frames when at least one full base
frame remains. A smaller remainder stays at record 1 / 101 through its closing
frame, with full bodies announcing 0x89 (hold record). Later host writes remain
separate, and an exact multiple retains its empty closing block.

No acknowledgement is inferred from the NAK. The new pending body and unsent
queue replace the old ones only after the retry transmits. A refused write
leaves both unchanged. The over number and accumulated retry budget are retained;
only a subsequent validated positive reply retires the smaller pending body.
A subsequent host delivery starts at the normal base record.

The scope is explicit: a fresh query, an outstanding full base frame, a complete
queued delivery boundary, and the normal sender. An unsolicited or short NAK,
an explicit transmit-rate ladder, or another record retains the original retry
geometry. Continuous SNR-based rate selection remains unimplemented.

The production sender completed clean and missing-first-DATA exchanges with
a VARA HF station at both wide bandwidths: 393 bytes sent and 225 bytes received,
byte-for-byte, with no pending bytes or queued blocks at completion. Independent
recordings verify the unread first DATA, intact query and 288/1117 reply,
CRC-clean 22-byte retry, and the exact concatenated continuation. No RF validation is
claimed by these tests.

Recordings: `20260919T164044Z-2300-*`
and `20260919T164404Z-2750-*`. The initial failed trial is retained separately:
its diagnostic driver used an expired idle deadline after synchronous DATA TX,
queried across the ACK and duplicated data. The corrected driver restored the
production scheduling order; subsequent runs use `mail_session` directly.

Short 94-byte clean/missing-DATA pairs also pass at both bandwidths, with independent
CRC and exact-byte audits. A rejected earlier candidate jumped from the 22-byte
retry to a short base closing frame; the VARA HF responder did not accept that
unannounced jump. The revised sender holds the lower record through
22/22/22/22/6-byte frames. Recordings: `20260919T165735Z-2300-*` and
`20260919T165938Z-2750-*`.