<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!-- SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ) -->
# CML offline generator attribution

`cml/cml_generate.c` adapts `CreateSRandomInterleaver` from the **Iterative
Solutions Coded Modulation Library** (CML) `srandom.h`, copyright 2005–2006
Matthew C. Valenti and Yufei Wu, last updated 11 January 2006, under the GNU
Lesser General Public License version 2.1 or later. The adaptation is by Saul
St John (W9SSJ), 2026, under the same licence; `cml/COPYING.LESSER` holds its
text.

Upstream source: <https://github.com/wvu-wcrl/CML/blob/master/source/include/srandom.h>,
SHA-256 `75a5c211deb8b119d7c3a0bb5c6b5c4648a22bf74bc8106e271dac60d99c8cfe`.

That header is not distributed here. Its `ran2` is the Numerical Recipes
routine, whose terms do not allow redistribution. The uniform generator in
`cml_generate.c` is written from the published algorithm instead: L'Ecuyer's
combination of two multiplicative congruential generators (Communications of
the ACM 31(6):742–751, 1988) with a Bays–Durham shuffle (ACM Transactions on
Mathematical Software 2(1):59–64, 1976), using the moduli, multipliers, warm-up
and output mapping the recipes were drawn with. It produces the same stream
draw for draw, so every recovered table regenerates unchanged.

The adaptation uses explicit seeds and accepted-swap budgets instead of local
wall-clock seeding and the upstream default 10N budget. It keeps the column
walk and the literal pre-swap acceptance guards, and adds bounded counters,
incomplete-cap reporting, binary output, and a candidate cache with
scalar/audit modes. `python -m hfmodem.shrike.c2proof self-test --upstream
srandom.h` compares it with a fetched copy of the upstream header.

Normal regeneration compiles `cml/cml_generate.c`. It is not linked into the
modem runtime. The Python CLI is AGPL-3.0-only, as marked in its source.
