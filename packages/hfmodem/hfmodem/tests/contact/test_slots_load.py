# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A watch that cannot keep up stops reading whole channels, not the audio."""
from types import SimpleNamespace

import numpy as np

from hfmodem.contact import slots as S

CF = 14_105_000


def _notices(records):
    return [(r[1], r[2].text) for _, r in records if r[0] == 'qso' and r[2].text in (S.SHED, S.RESUMED)]


def test_shedding_keeps_the_modems_channel_and_resumes_with_fresh_readers():
    m = S.SlotMonitor(CF, CF, mycall='W9SSJ')
    assert [(ch.slot, ch.home) for ch in m._channels] == [(1, False), (None, True), (11, False)]
    assert [r[2].text for r in m.shed()] == [S.SHED] and m._shed == {11}
    assert m.shed()[0][1] == 1 and m.shed() == []
    unread = {ch.slot: ch for ch in m._channels if ch.slot in m._shed}
    for ch in unread.values():
        ch.mixer = None  # Reading a shed channel would fail here.
    m.feed(np.zeros(S.BLOCK))
    assert m.resume()[0][1:] == (1, S.Said(1, m.now, 'coverage', S.RESUMED))
    assert m._channels[0] is not unread[1] and m._shed == {11}
    m.feed(np.zeros(S.BLOCK))


def test_the_live_watch_sheds_when_behind_and_takes_the_channel_back(monkeypatch):
    clock, cost = [0.], [1.]
    monkeypatch.setattr(S, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(S, '_lower_priority', lambda increment: None)

    def feed(self, samples):
        clock[0] += cost[0] * len(samples) / S.FS
        self.samples += len(samples)
        return []
    monkeypatch.setattr(S.SlotMonitor, 'feed', feed)
    block = np.zeros(S.BLOCK, np.float32)
    items = [block] * 240 + [None]

    def get():
        if len(items) == 121:
            cost[0] = 0.
        return items.pop(0)
    out = []
    S._watch(SimpleNamespace(get=get), SimpleNamespace(put=out.append), CF, CF, dict(mycall='W9SSJ'))
    assert _notices(out) == [(11, S.SHED), (11, S.RESUMED)]
    shed, resumed = (r[2].at for _, r in out if r[0] == 'qso')
    assert 30 < shed < 60 < resumed
