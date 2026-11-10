# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The recordings of sabir's first transmission, and why they are the bench.

2026-08-29 00:20Z, 40 m at 7086.000 kHz, one arm per rung, heard by two public
KiwiSDRs this station does not own: Dayton OH at 302 mi and Empire MI at 103 mi.

Until these, every sabir receive result came off a signal this tree had made
itself. `test_offair.py` renders a burst, offsets it 30 Hz, adds stationary
noise, round-trips it through an exactly-integer 12 kHz clock and decodes it --
a path with no fading in it at all. A real 40 m evening path swings the level
10-15 dB inside one burst, and that is what `beacon_short` missed on, on both
receivers at once, at 21 dB peak SNR, while the deeper rungs decoded either side
of it. Nothing generated here reproduces it, so the recordings are the gate.

Written by a run into gitignored `captures/`, so no clone has them and only this
station can make more. `tests/gates/test_corpus_present.py` warns by name when
they are absent, rather than letting the gate become a smaller number.
"""
from __future__ import annotations

from hfmodem.tests import evidence

SLOT = evidence.CAPTURES / "sabir-0829-0020"

#: Each kept clip and the rung its decode has to name. `dc` is Dayton, `ec` is
#: Empire; Dayton's `beacon_short` clip is the one widened to 65 s around the
#: key-up before it was established that the cut was never the problem.
HEARD = {
    "dc-01-presence": "presence",
    "dc-02-short-wide": "beacon_short",
    "dc-02-med": "beacon_med",
    "dc-02-deep": "beacon_deep",
    "ec-01-presence": "presence",
    "ec-02-short": "beacon_short",
    "ec-02-med": "beacon_med",
    "ec-02-deep": "beacon_deep",
}

KEPT = tuple(SLOT / f"{name}.wav" for name in HEARD)
