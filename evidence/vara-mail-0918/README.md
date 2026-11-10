<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# VARA mail through KB8AY — 2026-09-18

W9SSJ sent message `NL03TXWQ8SGL` through KB8AY using hfmodem's VARA modem.
The gateway accepted the proposal with `FS Y` and returned `FF` after the body.
The operator reported receipt at the destination; the audio does not establish
inbox delivery.

| Field | Value |
|---|---|
| Recording | [Download WAV](audio/20260918T154839Z-W9SSJ-tap.wav) |
| Format | 48 kHz mono; 8,231,424 samples; 171.488 s |
| Start | 2026-09-18 15:48:46 UTC |
| Channel | 7100.0 kHz USB dial; 7101.5 kHz centre; BW2300 |
| Source | Rig speaker audio with transmit monitor enabled |
| Message | 633 bytes rendered; 441 bytes compressed |

Local transmit-monitor audio and received signals share one recording clock.
A local monitor signal alone does not prove reception at the peer.

[Provenance](provenance.json) records the recording hash and source.
Verify the shipped audio from this directory:

```sh
shasum -a 256 audio/20260918T154839Z-W9SSJ-tap.wav
```

Expected SHA-256:
`c35119771eff9a5a5d2041750063a55dc00dbec7108ea754f130a38bf4645bcd`.
