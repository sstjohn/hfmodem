# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import queue

from hfmodem.contact import slots as s


def test_new_band_discards_measurement_and_station_evidence():
    monitor = s.SlotMonitor(14_105_000, 14_105_000, mycall='W9SSJ')
    monitor.passband = (100, 2000)
    monitor._reported = object()
    monitor._evidence[1] = object()
    monitor._judged[1] = object()
    monitor._cqs['K7QB'] = object()
    monitor.heard('K7QB', 1)
    monitor.retune(7_105_000, 7_105_000)
    assert monitor.passband == s.PASSBAND_HZ and monitor._reported is None
    assert monitor._evidence == monitor._judged == monitor._cqs == {}
    assert monitor.calls == ['W9SSJ']
    assert monitor._spectra == [] and monitor._measure_from == monitor.now
    assert monitor.calling == 7_105_000


def test_queued_old_band_results_cannot_repopulate_new_band_history():
    watch = s.SlotWatch.__new__(s.SlotWatch)
    watch._in, watch._out = queue.Queue(), queue.Queue()
    watch._calling, watch._generation = 14_105_000, 0
    watch._table, watch._found, watch._now = ['old'], [('cq', 1, 'old')], 3
    watch._out.put((0, ('table', ['old'], 4)))
    watch._out.put((0, ('cq', 1, 'old')))
    watch.retune(7_105_000, 7_105_000)
    assert watch.table()[0] == [] and watch.poll() == []
    assert watch._in.get_nowait() == ('tune', 7_105_000, 7_105_000, 1)
    watch._out.put((1, ('table', ['new'], 5)))
    watch._out.put((1, ('cq', 1, 'new')))
    assert watch.table() == (['new'], 5)
    assert watch.poll() == [('cq', 1, 'new')]
