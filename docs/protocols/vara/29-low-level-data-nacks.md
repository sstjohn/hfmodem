# DATA rejection at host speed levels 1–4

The September 19–20 stock VARA HF 4.9.0 campaign found that the eight-symbol
continue shape also carries a DATA NACK. Its common first tone pair is not
positive acknowledgment. In the preserved `20260919T233737Z-2300-crc-observe`
run, the old detector retired 47 rejected bytes and sent the next frame; stock
received only the preceding 89 bytes. An independent cable decode confirms that
the rejected frame had the intended body and deliberately incorrect CRC.

## Measured replies and actions

Measurements use W9SSJ calling KC9GHZ, plus an independent N0XYZ peer absent
from the captured reply tables, on fixed BlackHole virtual audio.
The two stock installations are unregistered, so **nothing above host level 4
is qualified by this campaign**. ARQ record numbers differ from host levels:

| Bandwidth | Host levels 4, 3, 2, 1: ARQ records | Full body bytes |
|---|---|---|
| 500 | 4 (legacy alias 3), 2, 1, 0 | 44, 35, 23, 10 |
| 2300 | 3, 2, 1, 0 | 90, 48, 23, 10 |
| 2750 | 103, 102, 101, 100 | 90, 48, 23, 10 |

A complete queued delivery whose full pending frame was CRC-rejected retries
one supported record lower at the wide bandwidths, retaining every unconfirmed
byte and holding the lower record for the remaining delivery. BW500 uses host
levels 4 → 3 → 1 for CRC-rejection recovery: stock rejected repeated level-2
DATA in the consecutive-failure experiment, whereas the level-1 candidate
delivered all bytes. Bench-selected announced clean descents can still use all four levels.
At the lowest record a NACK repeats that record. A positive answer to a newly transmitted frame resets the NACK retry
budget; repeated NACKs for the same pending data do not.

Before any DATA has been accepted, stock can send the called-keyed 16-symbol
`289/31` rejection, including at lower records. After earlier DATA was accepted,
it can instead send an eight-symbol two-tone rejection. N0XYZ also sends the
long called-keyed `288/63` rejection directly after DATA. These are observed
link/state differences, not a rule assigning one NACK shape to each speed.
The long form is accepted in a fresh direct-DATA window or a solicited query
window; an abandoned query cannot reuse the direct window. When a direct
CRC-rejection reply is erased, the called-keyed query answer is also `288/63`.

The three complete eight-symbol negative tails are recorded in
`DATA_NAK_RESPONDER_BY_LINK`. These are measured **full-link** vectors, not a
general callsign generator. Complete clear matches authorize a retry only in
the pending full-DATA window. Near matches veto an ACK but do not authorize a
retry. Native fixtures retain source hashes and sample boundaries in
`tests/kestrel/fixtures/data-nak-pairs-0919/manifest.json`.

Positive hold and next-lower-record replies have different full tails, now
recorded separately and covered by the native `data-ack-pairs-0919` fixtures.
A different callsign or unfamiliar tail must obtain called-keyed positive query
confirmation before retiring bytes. A positively confirmed tail can be learned
for that connection only; the observation is bound to the exact pending frame
and its transmit timestamp, and the cache clears on reconnect. A NACK never
teaches a positive tail.

## Evidence and scope

The full development recordings do not ship. The mapping experiments
deliberately select entry records and suppress automatic
NACK action to measure replies. Its clean descents deliver exact bytes through
all four records at every bandwidth. CRC failures and lost frames are separate
arms; an arm being a valid measurement does not imply transport succeeded.
`transport/` exercises production ARQ, with faults applied at the waveform or
actual receiver callback, and captures an independent cable in each direction.
The early BW500 fault renderer preserved exact encoded body/CRC bytes but
used different leading padding from production. From `20260920T004942Z` onward,
it uses production onset and column settings too; prior runs are retained as
negative/candidate evidence rather than padding-identical CRC-only controls.

| Production run | Result |
|---|---|
| `20260920T000722Z-2300-{clean,three-crcs}` | 94 bytes exact despite three successive CRC rejections, descending to record 0 |
| `20260920T001827Z-2300-later-three-crcs` | 183 bytes exact; CRC rejections after earlier DATA was accepted exercise the eight-symbol form |
| `20260920T003211Z-2750-{clean,later-ladder-crcs}` | 183 bytes exact with CRC rejection at each of host levels 4, 3, 2 and 1; independent wire decode confirms the intended bad CRCs and exact good payload concatenation |
| `20260920T010654Z-2300-later-three-crcs` | N0XYZ: 183 bytes exact with direct long NACK detection enabled; the first two rejections are handled with no query |
| `20260920T010026Z-500-floor-data-drop` | Production: 94 bytes exact after two CRC rejections and an erased lowest-record frame; queried `288/1117` authorizes a same-record repeat |
| `20260920T010249Z-500-crc-retry-ack-drop` | Production: 94 bytes exact after erasing the level-3 retry ACK in all 13 actual receiver callbacks |
| `20260920T005339Z-2300-{clean,later-three-crcs}` | N0XYZ: 183 bytes exact in both arms; unknown positive tails require query confirmation, and called-keyed NACKs recover three rejected frames down to record 0 |
| `20260920T004942Z-500-{clean,retry-nak}` | Production: 94 bytes exact; first rejection uses host 4 → 3, another rejection after an acknowledged level-3 frame uses host 3 → 1 |
| `20260920T004529Z-500-500-nak-skip` | Bench-only rate candidate: 137 bytes exact through repeated CRC failures at host levels 4, 3 and 1 |
| `20260920T002428Z-2300-crc-retry-ack-drop` | 94 bytes exact after erasing the lower retry's ACK; all 13 affected receiver callbacks are zero, while the independent answer remains recorded |

The missing-DATA answer `288/1117` is distinct from CRC rejection. A preliminary
`289/541` reply sometimes precedes it in the mapping arms; its meaning is not
qualified as permission to retire or retransmit DATA. BW500 missing-DATA fallback above its lowest record and short-final CRC
rejection remain unqualified by this change. Explicit TX
ladder configurations are outside the automatic NACK repacketization path.
The previously qualified wideband missing-DATA and BW2300 short-final actions
remain documented in [23](23-nak-repacketization.md),
[24](24-lower-record-recovery.md), and [28](28-short-final-nak-retry.md).

## Regression checks

The final focused ACK/NACK, native reply, repacketization and final-answer
suite passes 259 tests. Additional BW500 codec tests pass 55 tests and
session/continue/window tests pass 41. Source hashes are frozen within each
stock arm; cleanup and independent wire audits are recorded beside each run.


## K0SI at 01:14 UTC September 20: ACK query scan timing

The capture `logs/onair/20260920T011453Z-W9SSJ-K0SI.wav` contains exact called-keyed
288/1 answers to both queries after DATA over 2: all 31 payload tones match.
The receive cursors are samples 4198048 and 4511904 at 48 kHz. Native excerpts
and their hashes are in `fixtures/intermediate-query-answer/k0si-20260920-provenance.json`.

Uneven receive callbacks at 1.37, 1.46 and 1.89 seconds reproduce the lost
answer: the 1.37-second scan finds an unfinished ACK, schedules the usual
half-second step and ignores the 1.46-second callback. By 1.89 seconds the
complete answer is beyond the 0.5-second freshness limit. An exact partial
payload now schedules the next scan at its fitted tail. The full final symbol,
called-key match, pending-message identity and freshness checks still apply.
Both native excerpts advance the pending frame exactly once at 1.46 seconds;
wrong calls and missing tails cannot advance it.

The original short DATA response is also audible in the recording. The rig's
receive mute extends into its opening, and several interior two-tone symbols
have insufficient separation from competing peaks. Several peaks fall at
third-order intermodulation frequencies (for example, bin 63 beside 77/91);
the capture does not establish whether transmitter or receiver caused this.
At the time of this QSO, the table had no qualified W9SSJ/K0SI ACK template. This evidence does not
justify accepting the short shape as an ACK: the query remains necessary for
this capture, and its first complete answer should resolve the pending frame.


## K0SI at 01:36 UTC September 20: a qualified short ACK

In `logs/onair/20260920T013613Z-W9SSJ-K0SI.wav`, DATA over 2 draws a complete
short reply at receive cursor 3301888. The first query response, at cursor
3467904, matches all 31 payload tones of K0SI's 288/1 ACK. There is no DATA
retransmission between the short reply and this confirmation. The pending 89
bytes are then retired, the remaining 55 sent, and the mail exchange completes.

The short reply's eight pairs are `(64,67) (58,92) (77,91) (48,64) (65,71)
(38,92) (81,85) (58,98)`. They now qualify the W9SSJ → K0SI BW2300 entry;
no other callsign or bandwidth inherits this measurement. Native fixtures,
cursors and hashes are in `fixtures/k0si-data-ack-0920/provenance.json`.
The intact reply advances exactly once without a query with 512, 4096, 4800
and 8192-sample callbacks. The earlier distorted reply still cannot meet the
existing tone checks, even with the known template, and remains a query case.
