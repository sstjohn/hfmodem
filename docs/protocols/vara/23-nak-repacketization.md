# NAK-triggered repacketization

For a missing full base-record DATA frame, stock's response to the caller's
state query is called-keyed 288/1117. Independent stock-to-stock recordings at
BW2300 (2026-09-19) and BW2750 (2026-09-12) show the caller restarting with the
first **22 unconfirmed bytes** at record 1 / 101, then returning to base records
with the remaining stream. All three decoded frames in each capture have clean
CRCs; the lower frame's full-body field is 0x99 in the 393-byte experiment.

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

The production sender completed clean and first-DATA-erasure experiments against
VARA HF 4.9.0 at both wide bandwidths: 393 bytes sent and 225 bytes received,
byte-for-byte, with no pending bytes or queued blocks at completion. Independent
cable audits verify the erased first waveform, intact query and 288/1117 reply,
CRC-clean 22-byte retry, and the exact concatenated continuation. These were
virtual-audio tests, with no RF or physical codec use.

Experiment IDs: `20260919T164044Z-2300-*`
and `20260919T164404Z-2750-*`. The initial failed trial is retained separately:
its diagnostic driver used an expired idle deadline after synchronous DATA TX,
queried across the ACK and duplicated data. The corrected driver restored the
production scheduling order; subsequent runs use `mail_session` directly.

Short 94-byte clean/erasure pairs also pass at both bandwidths, with independent
CRC and exact-byte audits. A rejected earlier candidate jumped from the 22-byte
retry to a short base closing frame; stock did not accept that unannounced jump.
The revised sender holds the lower record through 22/22/22/22/6-byte frames.
Experiment IDs: `20260919T165735Z-2300-*` and `20260919T165938Z-2750-*`.
Their captures and failed candidates remain in the private station archive.
