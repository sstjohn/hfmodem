# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import CodeType, FunctionType, SimpleNamespace
from inspect import unwrap

from hfmodem.shrike import onair
from hfmodem.shrike.p4budget import watch_read_admission


def read_history():
    code = next(c for c in unwrap(onair._run_p4_link).__code__.co_consts
                if isinstance(c, CodeType) and c.co_name == 'remember_full_read')
    fs, now = 48000, [0.]
    costs = {'normal': {}, 'sf8': {}}
    live = SimpleNamespace(pos=0, sample_now=lambda: round(now[0]*fs))
    bindings = dict(full_read_costs=costs, live=live)

    def cell(value):
        return (lambda: value).__closure__[0]

    remember = FunctionType(code, dict(onair.__dict__, FS=fs,
        time=SimpleNamespace(perf_counter=lambda: now[0])), closure=tuple(
            cell(bindings[name]) for name in code.co_freevars))
    return remember, costs, now, fs


def test_cached_full_reads_cannot_erase_fresh_work_reserve():
    remember, costs, now, fs = read_history()
    now[0] = .080
    remember('normal', 0, 0, 0.)
    assert watch_read_admission(2, 0, 0, 1920, fs, costs['normal'].values()) == (False, 3840)
    for _ in range(8):
        started = now[0]
        now[0] += .001
        remember('normal', 0, round(started*fs), started)
    assert watch_read_admission(2, 0, 0, 1920, fs, costs['normal'].values()) == (False, 3840)
    assert costs['sf8'] == {} and costs['normal'] == {0: 3840}


def test_eight_fresh_reads_age_out_expensive_full_read():
    remember, costs, now, fs = read_history()
    now[0] = .080
    remember('normal', 0, 0, 0.)
    for onset in range(1, 9):
        started = now[0]
        now[0] += .001
        remember('normal', onset*fs, round(started*fs), started)
        if onset < 8:
            assert watch_read_admission(2, 0, 0, 1920, fs,
                costs['normal'].values()) == (False, 3840)
    assert len(costs['normal']) == 8 and 0 not in costs['normal']
    assert set(costs['normal'].values()) == {48}
    assert watch_read_admission(2, 0, 0, 1920, fs, costs['normal'].values())[0]
    assert costs['sf8'] == {}
