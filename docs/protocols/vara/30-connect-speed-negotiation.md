<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Connect responses select the setup speed

Observed VARA HF caller transmissions on 2026-09-20 correct the earlier reading
of lower lattice positions as damaged-request rejections. They are connect offers
selecting the speed of the caller-ID setup transmission.

| Bandwidth | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|
| 2750 | 0 | 1 | 2 | 3 |
| 2300 | 14 | 15 | 16 | 17 |
| 500 | 21 | 22 | 23 | 24 |

Positions use the bandwidth's existing connect-response descriptor, advancing
30 PRNG draws per position. Each payload remains keyed to the called station.
This table concerns connection setup, not DATA ACK/NAK interpretation.

## Independent evidence

KD7UHR's responses after requests 4, 5 and 6 in
`20260920T050126Z-W9SSJ-KD7UHR.wav` each match BW2750 position 0, 15/15 tones.
A VARA HF caller answers that response with an unencrypted level-1 (18 bps)
transmission: a CRC-valid caller-ID frame for W9SSJ:

```
5e44d32800000080048bd63e
```

Observed caller frames cover the remaining lower positions at all three
bandwidths and the BW2750 level-4 control. Decoded frames, source hashes and
speed announcements are in
`tests/kestrel/fixtures/kd7uhr-response-zero-0920/peer-setup-frames.json`.
Level-1 frames from KD9ZZZ and W9SSJ-10 check the identity format.

A VARA HF responder also connects at **every bandwidth/level combination in the
table** when our transmitter renders setup at the speed it offered, for clean,
corrupted-tone and noisy requests: all 12 cases are in
`peer-setup-accepted.json` beside the frame fixture. Labels such as
`accept-500-2` are case names: `offered_level` and the measured tones, not the
name, determine which speed was tested.

These are setup-interoperability tests; they do not establish
live KD7UHR mail delivery or later DATA speed negotiation.

September 9 recordings established that a damaged request changes the
response position. They did **not** establish rejection semantics: our own caller
ignored those positions, and no VARA HF caller had been seen to answer them. That circular
inference caused Kestrel to discard valid lower-speed offers.

## Setup formats

The first eight body bytes are the existing packed caller/SSID/setup header.
Levels 2..4 append `14 <caller CRC high>`, zero padding, and `04 82`. Level 1
instead ends the ten-byte body directly with `04 8b`. All frames append the
big-endian CRC-16/GENIBUS of their body.

| Bandwidth | Level 1 body | Level 2 body | Level 3 body | Level 4 body |
|---|---:|---:|---:|---:|
| 2300 / 2750 | 10 | 23 | 48 | 90 |
| 500 | 10 | 23 | 35 | 44 |

The setup uses the corresponding DATA waveform. The chosen host level is kept
separately from the operator's DATA-level setting. Repeated offers may select a
different setup speed, within the existing transmitted-setup budget. Receiving
an offer alone never marks the session CONNECTED; confirmation is still owed.

All four positions retain the called-station, bandwidth, tone-count and
comparability checks. The stream and segmented-audio paths accept short-preamble
responses using their payloads. No new unkeyed or cross-bandwidth acceptance is
introduced.
