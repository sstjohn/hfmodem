# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A cycle the grid admitted and that keyed nothing charges nothing to the next.

WS8EOC-80 0924-0344 (`working/pactor-level-air-0916/arms/
pactor-current-ws8eoc-80-pounce-20260924T034419Z`, `captures/onair-0923-2244`).
Hold 12, long ISS behind our unconfirmed changeover: the grid admitted slot 40
and the ISS held its packet. Hold 13: the peer's CS1 decoded mid-window and the
SL1 packet keyed at slot 43, before that cycle's own admission, so it closed
against slot 40's -- `breakin_cost_n` took a whole long cycle, 3.7 s. Every
changeover cycle after it carried that cost past the key: SLOT n IS GONE at
+3712..3840 ms with the bare clamp at 0, REGRID GAVE UP, one key in four slots
against a gateway keying short 0x43 packets in every one (holds 33-39), and 39
slots given away in the arm.
"""
import numpy as np

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_p3_missed_slots_0922 import station, stand_late

FS = 48000
EMPTY = np.zeros(0, np.float32)


def test_an_admission_that_keyed_nothing_is_not_the_next_keys_cost(tmp_path, capsys):
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    live = tx.live
    tx.breakin_due = True                       # `taking_link`: the changeover is inflight
    stand_late(live, g, tx, 40, 0)
    got, _, _ = onair._regrid(live, g, tx, s.host, s.rx, 40, EMPTY, live.pos, 1920)
    assert got == 40 and not tx.slots_used      # admitted, and the ISS held its packet

    tx.aim(g, 43)                               # hold 13: CS1 read inside the window
    live.now = live.pos = tx.key_instant(g, 43) - live.key_notice - 2 * live._blk
    tx.send_packet(1, b': W9S', 0x21)
    assert tx.slots_used == [43] and not tx.refused

    capsys.readouterr()
    slot = 50                                   # hold 33: a changeover cycle on time
    stand_late(live, g, tx, slot, 0)
    got, _, _ = onair._regrid(live, g, tx, s.host, s.rx, slot, EMPTY, live.pos, 1920)
    out = capsys.readouterr().out
    assert got == slot and onair.SLOT_GONE not in out
    assert tx.breakin_cost_n < round(onair.BREAKIN_CLAMP_TOL_S * FS)
