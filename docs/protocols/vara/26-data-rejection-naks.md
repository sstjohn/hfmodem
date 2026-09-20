# DATA rejection NACKs: 289/31 and 288/63

K0SI answered an 89-byte BW2300 DATA frame on September 19 with a 16-symbol
single-tone control, then answered all three state queries with a 32-symbol
control. All 15 and 31 payload tones were readable in the post-TX capture.
The session failed because these NACK forms had no connected DATA handler.

| Context | Called-keyed seed / preadvance | Payload symbols | Meaning at this boundary |
| --- | --- | ---: | --- |
| Direct response to DATA | 289 / 31 | 15 | NACK; retry one record lower |
| Response to state query | 288 / 63 | 31 | Same NACK; retry one record lower |

The short payload is identical to `SESSION_CONNECT_CONFIRM`. The connection
interpretation remains confined to the outstanding link-setup state. Recognizing
that waveform during a DATA turnaround must neither acknowledge bytes nor reset
the connection.

## Independent stock comparison

Stock VARA HF 4.9.0, BW2300, W9SSJ calling KC9GHZ, virtual audio only:

- Clean 144-byte delivery: 89 + 55 bytes delivered exactly.
- Same first DATA body with one encoded CRC bit flipped: zero host bytes;
  exact 289/31 followed by exact 288/63 after each query. Independent cable
  decoding recovered the intended body and the deliberately wrong CRC exactly.
- Recognize either NACK and retry at the same record: no delivery.
- Repacketize at record 1 (two steps down): no delivery.
- Repacketize at record 2 (one step down): exact delivery of all 144 bytes,
  as 47 + 47 + 47 + 3, through either the direct or queried NACK path.

The retry preserves the pending over number. Each later distinct block advances
that number normally. The sender announces and holds the lower record with 0x89
through the remainder of this delivery, then returns to the ordinary policy for
the next host delivery. A NACK never retires bytes.

This differs from the qualified 288/1117 response to missing DATA, whose measured
retry is two records lower. The two NACK forms must not share an assumed record
change merely because both are negative feedback.

## Implementation boundary

The new action is scoped to a default BW2300 initiator with a full base-record
frame pending and a complete delivery boundary queued. The short NACK requires a
fresh post-DATA window; the long NACK requires a successful, fresh query for the
same pending frame. Unsupported bandwidths, explicit transmit ladders, short
final frames, and further NACKs at a lower record do not acquire this action.

Recognition uses the called station's payload, at least 13 exact clear symbols
for the short form or 20 for the long form, no contradictory clear symbols, and
a clear last payload symbol. The complete fitted frame plus 60 ms must have
arrived. If a stream scan finds the full payload before that guard expires, it
schedules the next scan at the guard instead of waiting another half-second and
missing the fresh reply window. This matters for 4800-sample callbacks in all
three K0SI query replies.

Repacketization prepares a replacement before TX and commits it only when the
transport confirms transmission. It preserves later host writes separately and
uses the existing bounded retry budget. Logs name the actual NACK descriptor,
record, pending and queued byte counts, and retry/query budgets.

Native regression fixtures contain K0SI's direct NACK and all three queried
NACKs, plus independently recorded stock examples of both. Tests exercise three
callback sizes, wrong calls and states, expired or absent queries, incomplete
tails, refused transmission, repeated actions, separate host deliveries, and
budget exhaustion. Fixtures and source provenance are in
`tests/kestrel/fixtures/data-naks-0919/` within the hfmodem package.

The experiments compare clean, CRC-corrupted, unchanged-record and
lower-record inputs. No RF validation
is claimed by these offline and virtual-cable experiments.
