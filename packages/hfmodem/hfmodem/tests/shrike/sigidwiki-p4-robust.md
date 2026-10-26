<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Three recorded robust-mode short packets, as dibits

`fixtures/sigidwiki/p4-robust/dibits.npz` holds three rows of 84 quarter-turn steps, `int8`, one per packet:
`B4`, `B5` and `B6`, the three caller bursts of
[Signal Identification Wiki PACTOR IV](https://www.sigidwiki.com/wiki/PACTOR_IV),
[PACTORIVaudio1.mp3](https://www.sigidwiki.com/images/c/ce/PACTORIVaudio1.mp3),
SHA-256 `e9d14a178426ad9c82a62e5bc7272bbcb152c13a48821ceb89a02965d6816a76`, at
4.863 s, 6.113 s and 7.364 s. Attribution: Signal Identification Wiki
contributors, source and adapted excerpts
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The page does
not identify the operator or recording date. Four more packets, from a second
station, are in `fixtures/p4-robust-taunus`.

Reproduce: decode to mono float, take each burst with 0.12–0.23 s of lead, mix to
complex baseband at 28800 Hz, lock chip phase and carrier on the SF-16 despread
energy fraction over all 85 symbols, refine the carrier on the fourth-power tone
of the despread symbols, then hard-decide the 84 differentials with the residual
common rotation removed. Every row is unchanged across four such windows; the
tape locks at 0.35 despread fraction.

These are symbol decisions, not audio: they validate the datablock reading, the
two header words and the mask, and nothing about acquisition from a raw window.
