<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-3 mail through K4MSU — 2026-11-14

W9SSJ (EN63bc) called the K4MSU Winlink gateway at Murray State University
(EM56UO) on 40 m with hfmodem's PACTOR modem. The link went to PACTOR-3 at
SL1, the gateway sent its banner and SID, and W9SSJ answered. The gateway
accepted the outbound proposal with `FS Y` and acknowledged the 460-byte
compressed message. It then proposed one message, which W9SSJ accepted with
`FS +` and received complete. W9SSJ sent `FF`, the gateway answered `FQ`, and
the link closed with QRT and a normal disconnect. Every packet used the short
cycle. The operator confirmed delivery in both directions.

| Field | Value |
|---|---|
| Session | 2026-11-14 15:30:36 to 15:36:23 UTC |
| Channel | 7101.0 kHz USB dial; 7102.5 kHz centre |
| Transmit power | 40 W setting |
| Mail | 1 sent, acknowledged by the gateway; 1 received complete |
| Close | `FF`, `FQ`, QRT and disconnect |
| Replay | 41 CRC-valid peer data frames from the station receiver |

## Timeline

| UTC | Event |
|---|---|
| 15:31:08 | W9SSJ calls K4MSU |
| 15:31:23 | Connected; PACTOR-3 at SL1 |
| 15:31:52 | Gateway banner |
| 15:32:00 | Gateway SID; W9SSJ answers with its SID and one proposal |
| 15:32:51 | Gateway accepts the proposal (`FS Y`); the message follows |
| 15:35:27 | Message acknowledged; gateway proposes one message, W9SSJ accepts (`FS +`) |
| 15:36:15 | Inbound message received complete |
| 15:36:17 | W9SSJ sends `FF` |
| 15:36:23 | Gateway answers `FQ`; QRT and disconnect follow |

## Transcript excerpt

The gateway's text, as decoded on air, with private fields replaced by `#`:

```
RMS Trimode 1.4.4.0 Welcome to K4MSU Winlink Gateway at Murray State University
W9SSJ has 101 daily minutes remaining with K4MSU (EM56UO)
[WL2K-5.0-B2FWIHJM$]
;PQ: ########
CMS via K4MSU >
```

W9SSJ's reply and the rest of the exchange:

```
;FW: W9SSJ
[Pat-1.0.0-B2FHM$]
;PR: ########
; K4MSU DE W9SSJ (EN63bc)
FC EM ############ 622 460 0
F> 73
                                  ;PM: W9SSJ ############ 487 ##########
                                  FS Y
(outbound message, 460 bytes compressed)
                                  FC EM ############ 587 487 0
                                  F> A0
FS +
                                  (inbound message, 487 bytes compressed)
FF
                                  FQ
```

Three recordings ship with the [site example](../../site/examples/pactor3-mail/index.html):

| Recording | Source | Rate | Length | SHA-256 |
|---|---|---|---|---|
| [Receiver](../../site/examples/pactor3-mail/audio/receiver.flac) | Station receive audio | 48 kHz | 326.8 s | `de6c90a396c9051cbf6e2efe971e209e08b6ddee164b3bf1d9623b3d731ad4f1` |
| [Monitor](../../site/examples/pactor3-mail/audio/monitor.flac) | Local transmit-monitor tap, −0.03 s | 48 kHz | 326.8 s | `8515c3ad203b01c08988e80771be5b1c37f5e11281916c7778dbd2fe1b9dbf9a` |
| [Witness](../../site/examples/pactor3-mail/audio/witness.flac) | Remote KiwiSDR, +1.29 s | 12 kHz | 328.1 s | `bf649763d5e1d3340c131081fa587df7ae0932a54994703b50c05daef0ed0b39` |

The receiver recording starts at 15:31:08 UTC. The monitor is aligned to it by
120 unique CRC-valid packet payloads and the witness by 135, with maximum
residuals of 3 ms and 42 ms. The witness covers the whole session and hears both
stations. A 0.3 s gap in the witness stream near its start is filled with
silence. The [example manifest](../../site/examples/pactor3-mail/manifest.json)
records offsets, residuals and replay limits.

## Limits

- Gateway acknowledgement is the recorded outbound evidence. Delivery in both
  directions rests on the operator's confirmation; no receipt record is retained.
- Mail data moved at SL1 to SL3. SL4 to SL6 appear only in empty packets before
  the greeting.
- Transmit intervals come from the station's own transmit records, not measured
  RF edges.
- Message identifiers, addresses, subjects, the login challenge and response,
  and message contents are not published. The decoded bytes in the example show
  them as `#`.
