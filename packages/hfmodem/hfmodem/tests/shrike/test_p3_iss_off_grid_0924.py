# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A cycle that keyed from its own decode is not reported off the grid.

WS8EOC-80 0924-0344 (`captures/onair-0923-2244`), holds 40-44: each ISS packet
answered a CS4 decoded mid-window and keyed on its slot, RF -52..-60 ms from the
boundary, and the peer climbed SL2 to SL6 on five consecutive CS4s. The hold
line after each read `off-grid +869 ms`: measured once `transmit` had returned
at the burst's PTT drop, 0.87 s past the key. Hold 45, keyed by the tick after
the line, read -27.3 ms.
"""
from types import SimpleNamespace

from hfmodem.shrike import onair

FS = 48000
SLOT, BOUNDARY, SETTLE = 123, 7_395_000, 1920
KEYED_S = 0.882                                      # TX[49]'s PTT window


def at(sample, used):
    live = SimpleNamespace(sample_now=lambda: sample)
    return live, SimpleNamespace(slots_used=used)


def test_a_slot_keyed_from_the_decode_reads_as_keyed_not_as_869_ms_late():
    live, tx = at(BOUNDARY - SETTLE + round((KEYED_S - .013) * FS), [SLOT])
    line = onair._off_grid(live, tx, SLOT, BOUNDARY - SETTLE)
    assert "off-grid" not in line and "keyed" in line


def test_a_cycle_still_to_key_reads_its_offset():
    live, tx = at(BOUNDARY - SETTLE - round(.0273 * FS), [SLOT - 1])
    assert onair._off_grid(live, tx, SLOT, BOUNDARY - SETTLE) == "off-grid -27.3 ms"
