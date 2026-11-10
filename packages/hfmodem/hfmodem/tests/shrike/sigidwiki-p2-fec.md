<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# Reference softs for one recorded PACTOR-2 FEC frame

`fixtures/sigidwiki/p2-fec/softs.json` holds 144 `int8` channel-order softs at level 0, 72 symbols on each
of two carriers interleaved one symbol at a time, as an independent decoder
reported them for one frame of
[Signal Identification Wiki PACTOR II](https://www.sigidwiki.com/wiki/PACTOR_II),
[PACTOR-II_FEC.mp3](https://www.sigidwiki.com/images/e/ed/PACTOR-II_FEC.mp3),
SHA-256 `bd456a0e013f18c12d7104433230553de7b80ff9b931a0e001d2f3171ba2c9d7`.
Attribution: Signal Identification Wiki contributors, source and adapted data
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The page does
not identify the operator or recording date.

`test_p2rx.py` correlates its own demodulator against them on the same
recording; `docs/protocols/pactor/pactor2.md` section 5.12 has the comparison.
