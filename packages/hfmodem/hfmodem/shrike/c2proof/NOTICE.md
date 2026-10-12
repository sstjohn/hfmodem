# CML offline generator attribution

`upstream/srandom.h` is the original **Iterative Solutions Coded Modulation
Library** S-random interleaver implementation, copyright2005–2006 Matthew C.
Valenti and Yufei Wu, last updated January11,2006. Its original copyright and
LGPL-2.1-or-later notice are retained without modification.

Upstream source: <https://github.com/wvu-wcrl/CML/blob/master/source/include/srandom.h>

Bundled source SHA-256:
`75a5c211deb8b119d7c3a0bb5c6b5c4648a22bf74bc8106e271dac60d99c8cfe`.

`cml_generate.c` is an LGPL-2.1-or-later adaptation by Saul St John (W9SSJ),2026.
It uses explicit seeds and accepted-swap budgets instead of local wall-clock
seeding and the upstream default10N budget. It retains the original double
precision ran2 mapping, discarded initialization value, column walk and literal
pre-swap acceptance guards. Additions provide bounded counters, incomplete-cap
reporting, binary output, and a candidate cache with scalar/audit modes.

The upstream header is included for source inspection and independent tiny
controls. Normal regeneration compiles `cml_generate.c`. Neither file is linked
into the modem runtime. The Python CLI is AGPL-3.0-only, as marked in its source.
`COPYING.LESSER` contains the GNU Lesser General Public License version2.1.
