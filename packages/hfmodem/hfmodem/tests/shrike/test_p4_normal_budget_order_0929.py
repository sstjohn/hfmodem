# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.shrike import p4short
from hfmodem.shrike.p4hold import BreakinHold
from hfmodem.tests.shrike.test_p4_normal_budget_0929 import fields, FS


@pytest.mark.parametrize('allowance,calls_expected', [(.056, [8]), (.070, [1, 8])])
def test_quick_miss_cannot_take_away_a_full_decode_that_would_fit(monkeypatch, allowance, calls_expected):
    hold = BreakinHold(FS)
    clock, calls = [0.], []
    hold.now = lambda: clock[0]
    frame, scores = fields()
    decode = p4short.decode_scores

    def timed(scores, *, iterations=8):
        calls.append(iterations)
        clock[0] += .008 if iterations == 1 else .055
        return (None, None) if iterations == 1 else decode(scores)

    monkeypatch.setattr(p4short, 'decode_scores', timed)
    assert hold.normal(FS, 0, 0., .7, scores, None, deadline=allowance) == frame
    assert calls == calls_expected
    assert clock[0] <= allowance
