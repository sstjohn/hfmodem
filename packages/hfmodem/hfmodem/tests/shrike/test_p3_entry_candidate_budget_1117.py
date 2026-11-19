# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KC0TPS 30 m 2026-11-16 23:04Z: the candidate answer to the ninth entry, and QRT.

`operator-data/working/onair-20261116T230455857737Z-bd3e9552-…`: the ninth entry (attempt 9,
retry 8, log 281) drew the arm's only P3 answer, `[p3] candidate ACK at 22.285 s`
(287), held for a corroborating cycle; the next tick spent the retry budget and
sent QRT (292). WM4RB 40 m 22:47Z did the same after a candidate BREAK-IN to its
tenth entry.
"""
import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.p3_arq_support import granted


def entries_after_candidate(cycles, candidates=1):
    host, keyed = granted()
    host.no_p3_fallback = True                       # as flown: the silence rule keeps P3
    a = host.arq
    a.cfg.p3_entry_candidate_cycles = cycles
    a._inflight.retries = a.cfg.max_retries          # the ninth entry is on the air
    a.note_entry_candidate()                         # 22.285 s
    before = len(keyed.packets)
    for _ in range(16):
        if candidates > 1:
            a.note_entry_candidate()                 # a candidate every cycle after
        host.tick()
        if not a.entry_pending or a.state is not arq.State.CONNECTED:
            break
    return len(keyed.packets) - before, host


def test_a_candidate_answer_keeps_three_more_entries():
    keyed, host = entries_after_candidate(3)
    assert keyed == 3
    assert any('max retries' in line for line in host.log_lines)


def test_further_candidates_do_not_extend_the_entry_again():
    keyed, _ = entries_after_candidate(3, candidates=12)
    assert keyed == 3


def test_zero_restores_the_flown_budget():
    keyed, _ = entries_after_candidate(0)
    assert keyed == 0


def test_entry_candidate_cycles_configuration():
    assert arq.ArqConfig().p3_entry_candidate_cycles == 3
    with pytest.raises(ValueError, match='p3_entry_candidate_cycles'):
        arq.ArqConfig(p3_entry_candidate_cycles=-1)
