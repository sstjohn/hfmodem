<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# WS8EOC, 2026-09-20 23:58 UTC, 80 metres

Unmodified PCM16 crops in `fixtures/ws8eoc-long-recovery-0920/`, from the local receive stream of the 18:58 CDT contact.
`metadata.json` records the source SHA-256, absolute sample ranges, control
phases, and each crop's SHA-256. Samples run at 48 kHz. Apply -13.1 Hz receive
correction; no normalization or noise reduction was applied to these crops.

- `first-long.wav`: first long SL1 frame after CS6, including collection slack.
  Original data row zero: 712962; original reply audio epoch: 872591 (slot 14).
- `return-short.wav`: first short SL1 frame after the downshift, status 0x42,
  requesting changeover. Original data row zero: 1792962.
- `repeat-controls.wav`: 17 concatenated, independently indexed answer windows.
  Each is 16800 samples, beginning 2400 samples before its decoded CS2 phase.
  The concatenation is not a continuous recording; use the manifest phases.

The tests exercise real packet/control decoders, ARQ callbacks, placement and
in-memory transmission. They initialize the established link geometry from the
recorded coordinates; this is not a simulation of the full RF contact, and it
cannot prove WS8EOC received an emitted waveform. Deadline tests charge decoder
work against a simulated converter clock rather than relying on wall-time races.

## Follow-up, 2026-09-21 00:46 UTC (19:46 CDT)

`outbound-cs6.wav` is an unmodified PCM16 crop from
`captures/onair-0920-1946/stream.wav`: samples 2535198–2548798, with CS6 at
2537598. This contact used 0 Hz receive correction. Its source hash and crop hash
are recorded separately in the manifest. TX21's sidecar places its ordinary
header phase at 2493442, slot 41. The next packet belongs at phase 2553442,
slot 42. Actual TX25 instead emitted at phase 4953442, slot 82: 50 seconds late.

The outbound tests replay this grant against the recorded TX21 geometry. The
return-to-short cases reuse the recorded codeword on a constructed long-cycle
reply interval; that is a counterfactual, not a later recorded exchange.
