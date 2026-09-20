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
