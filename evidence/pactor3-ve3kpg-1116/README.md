<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-3 long-cycle mail through VE3KPG — 2026-11-16

W9SSJ (EN63bc) called the VE3KPG Winlink gateway (FN04VE) on 40 m with
hfmodem's PACTOR modem. The link went to PACTOR-3 at SL1, and the gateway sent
its banner and SID on long-cycle packets. W9SSJ answered with one proposal. The
gateway accepted it with `FS Y`, and W9SSJ sent the 450-byte compressed message
on long-cycle packets at SL1 to SL3. The gateway acknowledged it and proposed
one message. W9SSJ accepted with `FS +` and received the complete 475-byte
compressed message on long-cycle packets at SL1 and SL2. W9SSJ sent `FF`, the
gateway answered `FQ`, and the link closed with QRT. The operator confirmed
delivery of the sent message.

| Field | Value |
|---|---|
| Session | 2026-11-16 22:02:53 to 22:06:58 UTC |
| Channel | 7089.5 kHz USB dial; 7091.0 kHz centre |
| Transmit power | 47 W setting |
| Mail | 1 sent, acknowledged by the gateway; 1 received complete |
| Cycle | Mail data on long-cycle (3.75 s) packets in both directions |
| Close | `FF`, `FQ`, QRT and terminal marker |
| Replay | 26 CRC-valid peer data frames from the station receiver |

## Long-cycle packets

The session log names each PACTOR-3 data packet. First copies of long-cycle
data packets:

| Direction | SL1 | SL2 | SL3 |
|---|---|---|---|
| Received from VE3KPG | 11 | 6 | — |
| Sent by W9SSJ | 9 | 2 | 2 |

One of the two SL3 packets that W9SSJ sent is empty. The log also shows two
repeated received copies (one at SL1, one at SL2) and one repeated sent packet
at SL1. Each turn starts with short-cycle packets. The gateway's proposal and
the `FS +` answer moved on short-cycle packets. SL4 to SL6 appear only in
empty packets before the greeting.

## Timeline

| UTC | Event |
|---|---|
| 22:03:25 | W9SSJ calls VE3KPG |
| 22:03:46 | Connected; PACTOR-3 at SL1 |
| 22:04:08 | Gateway banner, on long-cycle packets |
| 22:04:27 | Gateway SID; W9SSJ answers with its SID and one proposal |
| 22:04:57 | Gateway accepts the proposal (`FS Y`) |
| 22:05:00 | Outbound message starts, on long-cycle packets |
| 22:05:38 | Message acknowledged; gateway proposes one message |
| 22:05:46 | W9SSJ accepts (`FS +`) |
| 22:05:50 | Inbound message starts, on long-cycle packets |
| 22:06:32 | Inbound message received complete |
| 22:06:34 | W9SSJ sends `FF` |
| 22:06:40 | Gateway answers `FQ`; W9SSJ sends QRT |

## Transcript excerpt

The gateway's text, as decoded on air, with private fields replaced by `#`:

```
RMS Trimode 1.4.4.2
W9SSJ has 120 daily minutes remaining with VE3KPG (FN04VE)
{SFI = 122 On 2026-11-16 19:00 UTC}
*****************************************************
Kawartha Packet Group
Hiawatha, Ontario
Sysop is Harmen,VA3OU/VE3EMA
MTD Connections: VARA=154 PACTOR=30
******************************************************
[WL2K-5.0-B2FWIHJM$]
;PQ: ########
CMS via VE3KPG >
```

W9SSJ's reply and the rest of the exchange:

```
;FW: W9SSJ
[Pat-1.0.0-B2FHM$]
;PR: ########
; VE3KPG DE W9SSJ (EN63bc)
FC EM ############ 608 450 0
F> 5B
                                  ;PM: W9SSJ ############ 475 ##########
                                  FS Y
(outbound message, 450 bytes compressed)
                                  FC EM ############ 569 475 0
                                  F> 91
FS +
                                  (inbound message, 475 bytes compressed)
FF
                                  FQ
```

Three recordings ship with the [site example](../../site/examples/pactor3-long/index.html):

| Recording | Source | Rate | Length | SHA-256 |
|---|---|---|---|---|
| [Receiver](../../site/examples/pactor3-long/audio/receiver.flac) | Station receive audio | 48 kHz | 205.6 s | `342b58944cf055ef7f6c0061232ee616a077e88a22dc86914b354f16b0d6b2a3` |
| [Monitor](../../site/examples/pactor3-long/audio/monitor.flac) | Local transmit-monitor tap, −0.03 s | 48 kHz | 205.6 s | `332f2a3e6fffe34fe83b3862a9e8d71a8b4734398509bb76a8d317d3156bde5c` |
| [Witness](../../site/examples/pactor3-long/audio/witness.flac) | Remote KiwiSDR, +1.21 s | 12 kHz | 206.8 s | `fcbd2cfac182befc6bd244d7f5f155daa0c1b7dfc8a1d1be74469b71519d3ed7` |

The receiver recording starts at 22:03:25 UTC. Sixteen unique CRC-valid packet
payloads align the monitor to it, with a maximum residual of 3 ms. Thirty-five
align the witness, with a maximum residual of 27 ms. The witness covers the
whole session and hears both stations, but it decodes mostly W9SSJ packets.
The [example manifest](../../site/examples/pactor3-long/manifest.json) records
offsets, residuals and replay limits.

## Limits

- Gateway acknowledgement is the recorded outbound evidence. Delivery of the
  sent message rests on the operator's confirmation. This record includes no
  receipt.
- The packet counts above come from the session log. The replay did not decode
  every packet: it does not show the gateway's `FQ` packet or two SL2 packets of
  the inbound message.
- Times in the timeline come from the session log and the receiver clock. They
  are accurate to about one second.
- Transmit intervals come from the station's own transmit records, not measured
  RF edges.
- The decoded bytes in the example and the excerpt above show message
  identifiers, addresses, subjects, the login challenge and response, and
  message contents as `#`.
