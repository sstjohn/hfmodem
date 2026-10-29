<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-4 greeting from VE3KPG — 2026-09-29

W9SSJ connected to VE3KPG on 40 m with hfmodem's experimental PACTOR-4 entry.
The gateway answered in PACTOR-4, stepped its robust long packets from SL2 to
SL3, and sent its complete Winlink greeting, which decoded here CRC-valid
through the `CMS via VE3KPG >` prompt. When this station took its first data
turn, the gateway answered once and then fell silent. No mail moved.

| Field | Value |
|---|---|
| Recording | [Download FLAC](audio/receiver.flac) |
| Format | 48 kHz mono, 16-bit; 5,578,752 samples; 116.224 s; whole session, uncut |
| Start | 2026-09-29 15:24:49 UTC |
| Channel | 7089.5 kHz USB dial; 7091.0 kHz centre |
| Source | Station receive stream; no witness |
| Transmit power | 50 W setting |

## What the recording shows

Times are seconds from the first sample.

| Time | Event |
|---|---|
| 0–29 s | PACTOR-1 connect and robust PACTOR-4 entry from this station |
| 29.7–31.0 s | Gateway changeover words X27 and Y27, then a 31-byte break-in body, CRC-valid |
| 36–51 s | Gateway speed-ladder shorts answering this station's X27 acknowledgements |
| 56.0, 59.7 s | Robust long packets at SL2, counters 1 and 2 |
| 63.5, 67.2 s | Robust long packet at SL3, counter 3, and its repeat; it ends with the prompt |
| 71–90 s | CRC-valid idle long packets at SL3 |
| 93–100 s | This station's first data bodies; the gateway answers with X27, Y27 and an idle body |
| 101–116 s | This station keeps sending; no further answer; local retry budget exhausted |

The peer's packet counter advanced from 0 to 3 while this station sent X27
acknowledgements, which implies the gateway accepted them. Nine peer frames decoded
CRC-valid in all. [Provenance](provenance.json) lists each one with its status
byte and speed level, plus the source and audio hashes.

## Limits

- No PACTOR-4 Winlink mail transfer. The login and proposal were queued but
  never answered, and nothing shows they reached the gateway.
- The decoded greeting printed one malformed line, `W9SSJ has RMS Trimode
  1.4.4.0`, because the counter-1 packet repeats the counter-0 body as its
  prefix. Each field is CRC-valid on its own; how the gateway meant the repeat
  is unresolved.
- No witness recording, so this station's transmissions are known from its
  own session log, not from an independent receiver.
- The login and proposal contents are not published.

Verify the shipped audio from this directory:

```sh
shasum -a 256 audio/receiver.flac
```

Expected SHA-256:
`e8eede4ab40838e1519ecafe9dadd48009a5177053d84f0ca36030d543dfa5f7`.
