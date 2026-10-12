# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Work admission is bounded independently by sample and elapsed wall time."""

import pytest

from hfmodem.shrike.p4budget import WorkBudget


class Clock:
    def __init__(self, sample=0, wall=100.):
        self.sample = sample
        self.wall = wall

    def budget(self, key_sample=1000, **kwargs):
        return WorkBudget(1000, key_sample, lambda: self.sample,
                          lambda: self.wall, **kwargs)


def test_sample_jump_exhausts_budget_with_wall_clock_stationary():
    clock = Clock()
    budget = clock.budget()
    assert budget.admits('body')
    clock.sample = 850
    assert budget.remaining() == pytest.approx(.040)
    assert not budget.admits('detect')
    assert budget.deferred == ['detect']


def test_wall_jump_exhausts_budget_with_sample_clock_stationary():
    clock = Clock()
    budget = clock.budget()
    deadline = budget.wall_deadline
    clock.wall += .850
    assert budget.remaining() == pytest.approx(.040)
    assert not budget.admits('detect')
    assert budget.wall_deadline == deadline
    clock.sample = -1000
    assert not budget.admits('refine')
    assert budget.wall_deadline == deadline


def test_sample_deadline_still_binds_when_wall_clock_moves_back():
    clock = Clock()
    budget = clock.budget()
    clock.sample = 850
    clock.wall -= 10
    assert not budget.admits('detect')
    assert budget.remaining() == pytest.approx(.040)


def test_measured_stage_costs_preserve_render_and_settle_reserve():
    clock = Clock()
    costs = dict(detect=.04220, refine=.21675, body=.57731)
    budget = clock.budget(costs=costs)
    for stage in ('detect', 'refine', 'body'):
        assert budget.admits(stage)
        clock.wall += costs[stage]
    assert budget.remaining() == pytest.approx(.05374)
    assert not budget.admits('body')
    assert budget.sample_deadline == 890
    assert budget.wall_deadline == pytest.approx(100.890)
    assert 101.-clock.wall >= .110


def test_default_costs_are_conservative_measured_reserves():
    budget = Clock().budget()
    assert budget.costs == dict(head=.010, detect=.050, refine=.250, body=.660,
                                packet=.660, long=.030, window=.660)
    assert budget.costs['detect'] > .04220
    assert budget.costs['refine'] > .21675
    assert budget.costs['body'] > .57731


def test_exact_boundary_is_admitted_and_override_is_local():
    clock = Clock(wall=0.)
    budget = clock.budget(reserve_s=0., costs={'detect': 1.})
    assert budget.remaining() == 1.
    assert budget.admits('detect')
    assert budget.deferred == []
    assert Clock().budget().costs['detect'] == .050


def test_reserve_rounds_to_sample_grid():
    budget = Clock().budget(reserve_s=.1106)
    assert budget.sample_deadline == 889
    assert budget.wall_deadline == pytest.approx(100.8894)
    assert budget.remaining() == pytest.approx(.889)


@pytest.mark.parametrize('word', ['X27', 'Y27'])
@pytest.mark.parametrize('cfo', [-144., -84., -28., 28., 84., 144.])
def test_hold_nomination_preserves_offset_words(word, cfo):
    import numpy as np
    from scipy.signal import hilbert
    from hfmodem.shrike import p4robusttx
    from hfmodem.shrike.p4receiver import P4Receiver
    fs = 48000
    wave = p4robusttx.robust_word(word)
    wave = (hilbert(wave)*np.exp(2j*np.pi*cfo*np.arange(len(wave))/fs)).real
    audio = np.zeros(round(.4*fs), np.float32)
    first = round(.05*fs)
    audio[first:first+len(wave)] = wave
    receiver = P4Receiver(fs, admitted=True)
    work = WorkBudget(fs, 3*fs, lambda: fs, lambda: 0.)
    receiver.read_hold_word(audio, fs, due=fs+first, carrier_end=fs, work=work)
    assert any(o.word == word and not o.provisional and not o.weak
               for o in receiver.observations)


def test_sample_jump_expires_a_head_evicted_from_retained_audio():
    from types import SimpleNamespace
    import numpy as np
    from hfmodem.shrike import p4robusttx
    from hfmodem.shrike.p4holdwork import HoldAudio
    from hfmodem.shrike.p4receiver import P4Receiver, Registered
    fs = 48000
    wave = p4robusttx.robust_word('Y27')
    audio = np.zeros(round(.265*fs), np.float32)
    audio[round(.05*fs):] = wave[:round(.215*fs)]
    receiver = P4Receiver(fs, admitted=True)
    retained = HoldAudio(fs)
    retained.retain(audio, 0)
    raster = SimpleNamespace(sending=True, cycle_n=60000, ticks=1,
        d=round(.05*fs), d_max_n=6240, rx_due=lambda slot: slot*60000+round(.05*fs))
    now = [len(audio), 0.]
    work = WorkBudget(fs, 2*fs, lambda: now[0], lambda: now[1])
    retained.read(receiver, raster, 0, work, lambda: now[1])
    assert retained.pending and any(o.provisional for o in receiver.observations)
    retained.retain(np.zeros(8*fs), 6*fs)
    now[0] = 14*fs
    retained.read(receiver, raster, 0, work, lambda: now[1])
    assert not retained.pending and not any(o.provisional for o in receiver.observations)
    assert receiver.registrations[-1].outcome == Registered.EXPIRED
