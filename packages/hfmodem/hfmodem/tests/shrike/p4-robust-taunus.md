<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Four recorded robust-mode short packets, as dibits

`fixtures/p4-robust-taunus/dibits.npz` holds four rows of 84 quarter-turn steps, `int8`, one per packet:
`T0`–`T3`, the four short packets of
[signals.taunus.de PACTOR-IV](http://www.signals.taunus.de/WAV/PACTOR-IV.HTML),
`PACTOR-IV_112-5_DQPSK.WAV`, SHA-256
`c4e191d4888f0d15ebd2cfecdb5f6503c99a34b29c40fd8285c04eb58d0f47a7`, 11025 Hz mono,
at 0.257 s, 1.507 s, 2.757 s and 4.007 s. The file is kept under the ignored
`working/pactor4-decode-0916/normal-short/local-audio/` with the rest of that
page's provenance. The three packets of a second station are in
`fixtures/sigidwiki/p4-robust`.

Reproduce as `sigidwiki-p4-robust.md` says. Every row is unchanged across four
analysis windows, and the locks land at 0.905–0.912 despread fraction.

These are symbol decisions, not audio: they validate the datablock reading, the
two header words and the mask, and nothing about acquisition from a raw window.
