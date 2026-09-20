# Lost answers after rate changes

The pending DATA record determines recovery geometry. The configured target rate
can differ from the record actually on air while descending through the ladder,
and a NAK can replace a base frame with a smaller pending frame. Checking either
the base record alone or the target's payload capacity excluded valid recovery
states.

Full frames now use the existing phased intermediate state query when the actual
pending body is full at its recorded level and more DATA is queued. Short final
frames use the query/request/drained exchange when no further DATA is queued.
Both retain the existing bounded attempts, complete-reply checks, same-pending
identity and wall-clock/sample freshness requirements. Neither silence nor a
query by itself retires DATA. Unsupported states still close unconfirmed.

The seven-symbol clipped-continuation exception remains scoped to the measured
base-record links; broader query recovery does not extend that exception.

## Qualification

| Bandwidth | Pending record | Payload capacity | Delivery tested |
| --- | --- | --- | --- |
| 2750 | 102 | 47 bytes | 196 bytes, full-ACK and final-ACK erasure |
| 2750 | 101 | 22 bytes | 185 bytes, full-ACK and final-ACK erasure |
| 2750 | 100 | 9 bytes | 181 bytes, full-ACK and final-ACK erasure |
| 2300 | 2 | 47 bytes | 94 bytes, short-final ACK erasure only |
| 2300 / 2750 | 1 / 101 after NAK | 22 bytes | 94 and393 bytes, DATA then retry-ACK erasure |

Full BW2300 record 2 remains outside the query scope. The ordinary sender uses
that record for a short close; no full-record-2 query exchange was qualified.
Each test also receives the same exact 225-byte greeting from stock.

BW2750 runs: `20260919T170553Z` (record102), `20260919T170930Z` (record101),
`20260919T171254Z` (record100). BW2300 closing-record run: `20260919T171639Z`.
Their captures remain in the private station archive.


The virtual-audio harness runs the production `mail_session` against VARA HF
4.9.0. Each rate has a clean baseline, an erased intermediate ACK, and an erased
short-final ACK. Independent cable recordings retain the original stock answer;
Kestrel's receive recording contains zeros across the selected callback range.
The subsequent query and complete reply are decoded independently, as are the
CRC and payload of every transmitted DATA frame. Success requires exact bytes
in both directions, no pending/queued DATA, no audio loss, unchanged source
hashes throughout the run, and completed process cleanup.

Native control-only excerpts, pending-frame metadata and exact source boundaries
are shipped under `tests/kestrel/fixtures/lower-answer-recovery/`. Tests replay
those excerpts through multiple callback sizes and reject unsolicited, stale
and wrong-pending answers. Full captures and audits remain in the private
station archive.

These are virtual-link measurements. They do not establish RF fading performance,
near-capacity short-trailer qualification, or general SNR-based rate adaptation.

## Retry phase and the double-loss boundary

A full retransmission advances query phase too. The sender's rate-ladder index
still counts new full frames, so a separate retry parity tracks this protocol
phase without accidentally skipping a rate. A refused retry advances neither.
The next successfully sent short frame and a new session reset the phase.

In the native two-stock BW2750 double-loss experiment, the first base DATA was
erased, then the ACK of stock's 22-byte record101 retry was erased. Stock keyed
60/683 before its first NAK and **60/745 after the full retry**. The latter drew
288/249 (31/31 payload tones), then stock sent the remaining72 bytes as a base
record103 short frame. Both hosts received exactly94/230 bytes; buffers emptied.
Stock's lower retry carried field0x91 in this experiment.

Kestrel's short-remainder policy holds record1/101 with field0x89. With the correct
745 query phase its missing retry ACK draws the ordinary 288/1 answer and it can
continue at that lower record. The measured288/249 answer also has a scoped
handler: it confirms the lower retry and regroups the unsent short remainder at
base, preserving later host deliveries and bytes across a refused next TX.

Reusing683 after the retry instead drew a different16-symbol289/541 frame.
Treating that response as an ACK failed to complete delivery. It is explicitly
rejected; its waveform is retained only as a negative regression fixture.
Failed experimental interpretations remain in the development captures.

Stock experiment IDs: clean
`20260919T175448Z` and double-loss `20260919T180025Z`. The DATA fault was a
restored mute on the virtual BlackHole16 input; the ACK fault was an exact
sample erasure in the established reverse relay. Actual receiver input is
recorded, and both virtual levels and Wine input settings were restored.

Kestrel double-loss and erased-retry negative controls exercise both paths.
An erased lower retry draws a
NAK and preserves the pending frame; losing its ACK solicits positive state and
advances exactly once. No16-symbol frame is accepted as this positive answer.

The final Kestrel short double-loss pairs (`20260919T180917Z` BW2300 and
`20260919T181125Z` BW2750) pass94/225 bytes exactly. Longer393-byte double-loss
runs (`20260919T181535Z` and`20260919T181658Z`) pass393/225 bytes exactly and
exercise288/249 before the remaining base frames. Each has independently decoded
CRC-clean DATA, intact phased query/reply, verified receive erasure, empty
pending/queue at completion, unchanged runtime hashes and completed cleanup.
Earlier393-byte clean runs establish the normal transfer baseline; the phase
change is exercised only after a full retry.
