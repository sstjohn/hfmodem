# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One late cycle must not make every later window longer.

The Windows PACTOR arm's capture pipeline starved the session loop: a stall
left the reader seconds behind the converter, the next window held all of it,
and the per-window work then cost more than the window lasted. The bench here
charges that work per sample of window, at a rate above real time, after one
stall in the dead time behind a key.
"""
import numpy as np

from hfmodem.shrike import onair, rxfront
from hfmodem.shrike.arq import State
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_grid import SLOT_N, _Bench, _off_raster, _session

FS = rxfront.FS
STALL_N = round(4.0 * FS)
# Per-window work dearer than the window, which is what a starved capture path
# is. Below one a stall drains on its own and there is nothing to bound.
RECORD_RATE = 1.2


def _starved_session(monkeypatch):
    windows = []
    stalled = []
    record = onair._CycleEvidence.record

    def charged(self, name, seg, seg_start):
        windows.append(seg.size)
        if not stalled:
            stalled.append(True)
            self.live.spend(STALL_N)
        self.live.spend(round(seg.size * RECORD_RATE))
        return record(self, name, seg, seg_start)

    monkeypatch.setattr(onair._CycleEvidence, 'record', charged)
    return _session(cycles=8, hold=4, charge=0, peer=False, seconds=400.0), windows


def test_a_stalled_cycle_does_not_snowball_into_the_next(monkeypatch):
    got, windows = _starved_session(monkeypatch)
    carry = onair.CARRY_MAX_CYCLES * SLOT_N
    after = windows[1:]
    assert after, windows
    # The carried arrears plus the stretch in front of the key the slot search
    # chose: at most a second slot when the listen floor steps one.
    assert max(after) <= carry + 2 * SLOT_N, [w / FS for w in windows]
    assert onair.BEHIND_THE_GRID in got['log'] and 'skipped unread' in got['log']
    assert max(abs(x) for x in _off_raster(got)) <= round(.001 * FS)


class _Rx:
    def __init__(self):
        self.fed = self.held = 0
        self.skipped = 0.0

    def feed(self, chunk):
        self.fed += chunk.size

    def bridge(self, chunk):
        self.held += chunk.size

    def skip(self, seconds):
        self.skipped += seconds


class _Host:
    protocol = Protocol.PACTOR1
    arq = type('Arq', (), {'state': State.CONNECTED})()


def test_only_delivered_arrears_past_the_cap_are_skipped(capsys):
    bench = _Bench(seconds=20)
    bench.now = 6 * FS                    # six seconds delivered and unread
    ahead = round(.2 * FS)
    rx = _Rx()
    seg = onair._listen_until_answer(bench, bench.now + ahead, _Host(), rx,
                                     SLOT_N, carry_n=SLOT_N)
    assert seg.size == SLOT_N + ahead
    assert bench.pos == bench.now == 6 * FS + ahead
    assert rx.fed + rx.held == seg.size
    assert np.isclose(rx.skipped, (6 * FS - SLOT_N) / FS)
    assert 'skipped unread' in capsys.readouterr().out


def test_a_window_inside_the_cap_is_untouched(capsys):
    bench = _Bench(seconds=20)
    bench.now = SLOT_N // 2
    want = 2 * SLOT_N                      # long, but mostly still to come
    rx = _Rx()
    seg = onair._listen_until_answer(bench, want, _Host(), rx, SLOT_N,
                                     carry_n=SLOT_N)
    assert seg.size == want and rx.skipped == 0
    assert 'skipped unread' not in capsys.readouterr().out


def test_no_cap_holds_everything(capsys):
    bench = _Bench(seconds=20)
    bench.now = 6 * FS
    rx = _Rx()
    seg = onair._listen_until_answer(bench, bench.now, _Host(), rx, SLOT_N)
    assert seg.size == 6 * FS and rx.skipped == 0

