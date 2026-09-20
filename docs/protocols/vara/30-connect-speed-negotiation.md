# Connect responses select the setup speed

Stock VARA HF 4.9.0 caller replays on 2026-09-20 correct the earlier reading of
lower lattice positions as damaged-request rejections. They are connect offers
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
Each recording was replayed at original gain to a separately launched stock
caller. Every trial reported `UNENCRYPTED LINK`, `BITRATE (1) 18 bps TX`, then
transmitted a CRC-valid caller-ID frame for W9SSJ:

```
5e44d32800000080048bd63e
```

Fresh stock-caller trials also cover the remaining lower positions at all three
bandwidths and the BW2750 level-4 control. Decoded frames, source hashes, speed
announcements and capture status are retained in
`tests/kestrel/fixtures/kd7uhr-response-zero-0920/stock-setup-frames.json`.
Level-1 trials with KD9ZZZ and W9SSJ-10 independently check the identity format.
Full local audio and host/PTT ledgers remain in the private station archive.

An earlier multi-case experiment reused one caller after ABORT. Some later
CONNECT commands returned WRONG, so those trials are excluded. Every reported
caller result above uses a fresh Wine instance, virtual cables, recorded input
and output, and verified owned-process cleanup.

The reverse interoperability test also reached stock's `CONNECTED W9SSJ KD7UHR`
status at **every bandwidth/level combination in the table**. A fresh stock
responder heard a clean, corrupted-tone or noisy request; its actual offer was
measured before our transmitter rendered setup at that speed. All 12 cases had
clean capture status and verified cleanup. Exact run IDs and input/output/source
hashes are in `stock-setup-accepted.json` beside the frame fixture. Labels such as
`accept-500-2` are experiment names: `offered_level` and the measured tones, not
the name, determine which speed was tested.

These are setup-interoperability tests on virtual cables; they do not establish
live KD7UHR mail delivery or later DATA speed negotiation.

The September 9 damage experiments established that request damage changes the
response position. They did **not** establish rejection semantics: our own caller
ignored those positions, and no stock caller was fed them. That circular
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
