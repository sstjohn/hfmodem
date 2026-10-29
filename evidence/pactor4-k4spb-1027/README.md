<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# PACTOR-4 greeting from K4SPB — 2026-10-27

W9SSJ connected to K4SPB with hfmodem's experimental PACTOR-4 entry and
received the gateway's complete Winlink greeting over PACTOR-4. Forty-three
CRC-valid frames, 13 of them distinct, arrived in all four packet formats this
receiver handles: one break-in body, eight normal, eight long and 26 chirp
frames. Several chirps had damaged headers and were recovered by a timing
sweep. The greeting reassembles once, without duplicates, through the
`CMS via K4SPB >` prompt at about 183.5 s. About two minutes of idle chirps
follow. No turn change to this station and no mail is recorded.

| Field | Value |
|---|---|
| Start | 2026-10-27 06:43:59 UTC |
| Source | Station receive stream, SHA-256 `7ae4edaacd229a4fa7a6759d6add3e0042d5bb0cd26f664c95e34bfd1a4332b0`; no witness |
| Frames | 43 CRC-valid, 24.8–309.8 s from the first sample; frequency offset +4.95 Hz |
| Shipped audio | Six 3.55 s crops at 8 kHz around the damaged-header chirps |

The full session recording does not ship. What ships is the receive
regression and its fixture:

- `packages/hfmodem/hfmodem/tests/shrike/fixtures/k4spb-1027/evidence.json`:
  decoded frames with status bytes, data and onset sample for every frame, plus the expected
  chirp decodes. SHA-256
  `ecedbff286820406fe01d41b714016a1a3d11903c10991e6e7da356a5ea0d082`.
- `chirps.npz`, beside it: the six crops, 16-bit PCM with their source
  origins. SHA-256
  `c24892a3d21d076995437c8a2b45fa080b9af7e686ad107ba4970d0a2bea065a`.
- [Regression](../../packages/hfmodem/hfmodem/tests/shrike/test_p4_k4spb_1027.py):
  checks that the greeting is delivered exactly once across all formats and
  that each recorded sweep recovers its CRC-valid chirp.

## Limits

- Receive only. The recording shows no data from this station acknowledged
  and no PACTOR-4 Winlink mail transfer.
- The gateway repeated most frames several times, which is consistent with it
  waiting on acknowledgements; how many of this station's acknowledgements it
  read is not established.
- The band and dial frequency are not in the shipped fixture.
