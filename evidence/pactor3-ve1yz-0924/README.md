<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-3 mail through VE1YZ — 2026-09-24

W9SSJ called VE1YZ on 40 m with hfmodem's PACTOR modem, logged in, sent one
Winlink message and received two. The gateway accepted outbound message
`PV5D0KLB6XUN`, and the session closed with `FF` and `FQ`. The operator reported
its arrival in the destination inbox; the audio does not establish delivery.
Acknowledged local transmit reached PACTOR-3 SL6; peer data received here
reached SL2. The session lasted 11 minutes 45 seconds.

| Field | Value |
|---|---|
| Start | 2026-09-24 04:10:20 UTC |
| Channel | 7095.0 kHz USB dial; 7096.5 kHz centre |
| Transmit power | 60 W |
| Mail | 1 sent, accepted by the gateway; 2 received complete |
| Replay | 110 CRC-valid peer data frames from the station receiver |

Three recordings ship with the [site example](../../site/examples/pactor3-send/index.html):

| Recording | Source | Rate | Length | SHA-256 |
|---|---|---|---|---|
| [Receiver](../../site/examples/pactor3-send/audio/receiver.flac) | Station receive audio | 48 kHz | 603.0 s | `914101c090bcaa8a21b529c1541b5ef70d0ad0c2eb12b07f3c9a666f42a2e8ac` |
| [Monitor](../../site/examples/pactor3-send/audio/monitor.flac) | Local transmit-monitor tap, +36.09 s | 48 kHz | 693.4 s | `2ebac0630d50ed829bd632337b2048e568d76587fcf98905358378a3cc8d5937` |
| [Witness](../../site/examples/pactor3-send/audio/witness.flac) | Remote SDR, +3.50 s | 12 kHz | 613.9 s | `a6de8dcc8e990e754a6f983970800dbf0a70fe6bc5bb9027ebc9e91cbc00a163` |

The monitor is aligned to the receiver clock by 12 receive-window envelopes and
the witness by 30 unique CRC-valid packet payloads; each shares 48 distinct
CRC-valid payloads with the receiver. The witness hears both stations, so it shows
that this station's PACTOR-3 transmissions were on the air; the monitor alone
shows only what was sent. The [example manifest](../../site/examples/pactor3-send/manifest.json)
records offsets, residuals and replay limits.

## Limits

- Gateway acceptance is the recorded outbound evidence. Destination receipt
  rests on the operator's report; no receipt record is retained.
- SL6 is the highest level the peer acknowledged from this station. Peer data
  decoded here did not exceed SL2.
- The 110–151 s homepage excerpt was rechecked with audio-acquired control
  timing. The rest of the event list keeps the original export and may omit
  controls or misplace them on the session grid.
- Message contents are not published.
