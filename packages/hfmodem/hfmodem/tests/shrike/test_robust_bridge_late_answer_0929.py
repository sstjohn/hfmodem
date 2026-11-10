# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC's entry-3 answer in the final call listen, onair-0929-1913.

It answered the 21st robust call from the hushed slot after it: entry 3
(095AC5), inverted, 0.981 s past that slot's boundary. The cycle that read it
then overran slots 24, 25 and 26, and the bridge keyed on slot 28. The tape
holds no second read of the answer; the held answers here only show where the
bridge reads.
"""
import json

import numpy as np

from hfmodem.shrike import onair, session, spec
from hfmodem.shrike.robust import Bridge
from hfmodem.tests.shrike.test_robust_answer_0923 import _air, _burst, _keyed

FS = 48000
SLOT = 60000
PHASE = round(.980875 * FS)


def _answer_in_the_final_listen(tmp_path, hushed, held=()):
    """The answer once, then held in `held`, complemented on alternate slots."""
    x = np.zeros((hushed + 16) * SLOT, np.float32)
    for slot in (hushed, *held):
        answer = _burst(spec.CONTROL_SIGNALS[3], invert=(slot - hushed) % 2 == 0, amp=.11)
        at = slot * SLOT + PHASE
        x[at:at + answer.size] = answer
    session.write_wav(str(tmp_path / 'peer.wav'), x)


def _overrun_three_slots(monkeypatch):
    regrid, jumped = onair._regrid, []

    def late(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n):
        if host.arq.robust_answer is not None and not jumped:
            jumped.append(slot)
            slot += 3
            seg = np.r_[seg, live.take_until(raster.boundary(slot) - settle_n)]
        return regrid(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n)

    monkeypatch.setattr(onair, '_regrid', late)
    return jumped


def test_an_answer_read_before_an_overrun_keeps_its_own_slot(tmp_path, monkeypatch):
    _answer_in_the_final_listen(tmp_path, hushed=4, held=range(8, 12))
    jumped = _overrun_three_slots(monkeypatch)
    keyed = _keyed(monkeypatch)
    log = _air(tmp_path, '--after-robust', 'p2', '--robust-p2-max', '4', '--retries', '3')
    assert 'FINAL CALL LISTEN' in log, log
    assert 'robust answer: 3 (095AC5) inverted' in log, log
    assert jumped == [5], log
    bridge = [s for s, w, _ in keyed if 'ROBUST BRIDGE' in w]
    assert bridge == [8, 9, 10, 11], log
    assert 'P2 call packet never taken: 4 cycles' in log, log
    result = json.loads((tmp_path / 'out' / 'robust-bridge.json').read_text())
    assert result['answer']['entry'] == 3
    assert abs(result['answer']['phase'] - PHASE / FS) < .005
    assert [(r['slot'], r['form'], r['entry']) for r in result['reads']] == [
        (slot, 'FSK', 3) for slot in (8, 9, 10, 11)], result


def test_a_window_without_the_answer_instant_is_reported_not_read():
    bridge = Bridge('W9SSJ', PHASE)
    for audio in (np.zeros(0, np.float32), np.zeros(SLOT // 4, np.float32)):
        granted, report = bridge.read(audio, 3 * SLOT, SLOT, 1)
        assert not granted
        assert report.startswith('no audio at the answer instant'), report
    assert bridge.silent == 0 and bridge.cycles == 2
    assert [r['unread'] for r in bridge.heard] == [True, True]
