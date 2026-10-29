<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-3 mail from WS8EOC — 2026-10-03

W9SSJ called WS8EOC on 80 m with hfmodem's PACTOR modem, logged in and
received one complete Winlink message. Peer data reached PACTOR-3 SL3, both in
the original session and when the recording is replayed. The session lasted
3 minutes 26 seconds.

| Field | Value |
|---|---|
| Start | 2026-10-03 05:00:49 UTC |
| Channel | 3595.0 kHz USB dial; 3596.5 kHz centre |
| Transmit power | 60 W |
| Mail | 1 received complete; none sent |
| Replay | 80 CRC-valid peer data frames from the station receiver |

Two recordings ship with the [site example](../../site/examples/pactor3-receive/index.html):

| Recording | Source | Rate | Length | SHA-256 |
|---|---|---|---|---|
| [Receiver](../../site/examples/pactor3-receive/audio/receiver.flac) | Station receive audio | 48 kHz | 168.8 s | `ca7c10386adc5757124640c4c384cedb22142a0f06a7280ba7482672853cd192` |
| [Witness](../../site/examples/pactor3-receive/audio/witness.flac) | Remote SDR, +1.19 s | 12 kHz | 174.3 s | `dfb5ab1652411e50df8524d0f390f7b8e51f27f688f9a7206f8a98accd85b01e` |

The witness is aligned to the receiver clock by 40 unique CRC-valid packet
payloads, with a maximum residual of 32 ms. The
[example manifest](../../site/examples/pactor3-receive/manifest.json) records
the alignment and replay limits.

## Limits

- This is the strongest retained complete receive, not full-speed PACTOR-3.
  Acknowledged local transmit stayed at SL1.
- Transmit intervals come from software session timing, not a monitor tap.
- Message contents are not published.
