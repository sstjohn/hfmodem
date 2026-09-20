# A NACK tail must schedule its next scan

The 2026-09-19 22:48 UTC K0SI attempt sent one short-final 79-byte record-3
DATA frame, retried it once, then closed with delivery unconfirmed. In the receive
recording there is no recognizable control reply immediately after either DATA
transmission. All three final-answer queries receive the called station's
288/1117 NACK, with 31/31 payload tones decoded independently. These are negative
answers, not acknowledgements. An independent KiwiSDR
recording decodes both our DATA transmissions with clean CRCs and 79-byte payloads.
That receiver was about 140 miles from us and 436 miles from K0SI; this establishes
valid on-air DATA there, not successful reception at K0SI. The recordings do not
establish why K0SI did not accept DATA or the correct retry format.

The live receiver acted on the first query reply. It rejected the other two as
stale at +0.51 s. A partial payload fit had already identified the NACK just before
its last symbol, but the next search was scheduled half a second later. Polling
and callback boundaries could put that next search past the freshness limit.

A matched, incomplete responder NACK now schedules a search at its predicted
last symbol. It still cannot trigger a reply before the tail is present, and
stale frames still cannot trigger transmission. This changes receive scheduling,
not the NACK's meaning or retry geometry.

Three unmodified recording slices, source hash and exact live-cursor bounds are
in tests/kestrel/fixtures/k0si-2248-0919. Replay covers regular chunks and uneven
poll groups with a scan milliseconds before the tail, reproducing the former
stale rejection. It requires exactly one retry, unchanged pending bytes, no
premature transmission and no duplicate action through the bracket route.
