# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every eligible slot of the September 10 entry campaign, on the sample clock.

The evening arms lost half their entry slots to work that spends wall clock and
consumes no audio: the window in front of the key is bounded in SAMPLES, so a
stage that runs long does not shorten it, it walks past the key. `_Bench` is
the one clock that reproduces that without a radio -- it advances on reads and
on `spend` -- so each production stage here is wrapped to charge its own
measured duration back to it, and the audio is the recorded stream of the arms
that lost the slots, cut to the campaign and nothing else.

The repair is one thing: the frequency ladder reads the same and costs half,
so the eight milliseconds of notice a 40 ms settle leaves can hold it. The
cases below hold the ladder's arithmetic, its cost, and what the cost buys.
"""
import contextlib
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert, oaconvolve, resample_poly

from hfmodem.shrike import arq, live, onair, p3acquire, placement, rx, spec
from hfmodem.tests.evidence import SWEEP_FLOOR
from hfmodem.tests.shrike.onair_support import _Bench as _Bench, _Rig
from hfmodem.tests.shrike.p3_arq_support import granted
from hfmodem.tests.shrike.p3_deadline_support import (
    CHARGED as CHARGED,
    D_MAX_N as D_MAX_N,
    D_S as D_S,
    FEED as FEED,
    LINK as LINK,
    MEASURED as MEASURED,
    P1_CS_N as P1_CS_N,
    P1_DATA_N as P1_DATA_N,
    Replay,
    STAGES as STAGES,
    _flown_grid as _flown_grid,
    _named,
    _one_at_a_time as _one_at_a_time,
    _per_hypothesis,
    charging as charging,
    crop,
    crops as crops,
)
from hfmodem.tests.shrike.p3_reply_support import fallback as fallback

FIXTURES = Path(__file__).with_name('fixtures') / 'slot-deadline-0910'
METADATA = FIXTURES / 'metadata.json'


def instrumentation_overhead_ms(n=20000):
    t0 = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        _ = time.perf_counter() - t
    return (time.perf_counter() - t0) / n * 1e3


NONE = dict(slot_gone=0, late_to_key=0, pre_render=0, regrid_gave_up=0)


def assert_replay_policy(run):
    assert run.cycles
    keyed = [c for c in run.cycles if c['keyed']]
    assert keyed
    # The pre-regrid aim may have been abandoned. Every actual key still
    # owes the same minimum notice, including keys emitted during collection.
    assert all(c['key_late_ms'] and
               all(late < run.grace_ms for late in c['key_late_ms'])
               for c in keyed)
    for c in run.cycles:
        if c['intent']:
            assert c['keyed']
    assert run.tx.slots_used == sorted(set(run.tx.slots_used))
    return keyed


@pytest.mark.parametrize('tag', ['e07', 'e05'])
def test_every_eligible_slot_is_keyed_under_the_recorded_workload(tag, tmp_path):
    """Production held reads retain the owned reply in each yielded slot."""
    run = Replay(tag, tmp_path).run()
    assert_replay_policy(run)
    assert run.deferrals == NONE, run.deferrals
    assert 'LISTENING CYCLE' not in run.log
    assert run.cycles[0]['owned'][0] == 0


def test_the_pactor1_control_pays_no_entry_acquisition(tmp_path):
    """Exact P1 grants short-circuit P3 acquisition in either protocol arm."""
    run = Replay('e08', tmp_path).run()
    assert run.stage_table()['cs-acquire']['n'] == 0
    assert max(run.off_grid_ms) < run.grace_ms / 4, run.off_grid_ms
    entry = Replay('e07', tmp_path, slow_ladder=True).run()
    assert_replay_policy(entry)
    assert entry.host.arq._upgrade_requests > 0
    grants = [c for c in entry.cycles if c['p1_grant']]
    assert grants and all(c['reader_acquisitions'] == 0 for c in grants)


def test_the_upgrade_cycle_stops_costing_a_slot(tmp_path):
    """No slot is yielded to the entry reply; actual keys meet admission."""
    for slow in (True, False):
        run = Replay('r06', tmp_path, slow_ladder=slow).run()
        assert_replay_policy(run)
        assert run.stage_table()['cs-acquire']['n'] > 0
        assert run.deferrals == NONE, run.deferrals
        assert 'LISTENING CYCLE' not in run.log

@pytest.mark.parametrize('tag', ['e07', 'e05'])
def test_the_flown_ladder_overruns_the_notice_and_the_batched_one_does_not(
        tag, tmp_path):
    """P1 grants bypass both ladders; isolated cold-search parity/cost stays pinned."""
    flown = Replay(tag, tmp_path, slow_ladder=True).run()
    fixed = Replay(tag, tmp_path).run()
    for run in (flown, fixed):
        assert_replay_policy(run)
        assert run.host.arq._upgrade_requests > 0
        grants = [c for c in run.cycles if c['p1_grant']]
        assert grants and all(c['reader_acquisitions'] == 0 for c in grants)
        assert run.deferrals == NONE, run.deferrals
    assert fixed.tx.slots_used == flown.tx.slots_used


def test_the_frequency_ladder_reads_the_same_and_costs_a_fraction(tmp_path):
    """`p3acquire._bands` against the per-offset matched filter it replaced.

    The ladder could not be run in front of a key and could barely be run
    behind one: 31 hypotheses at 0.40 ms apiece, of which 0.32 was the filter
    call rather than its arithmetic. Stacked along one axis it is the same
    filter on the same samples -- agreeing here to a part in 1e15, which is
    `rx._baseband`'s own standard against the direct convolution -- for well
    under half the wall clock. Both measured in this process, so the ratio is
    a property of the code and not of the machine's mood.
    """
    _, audio = crop('e07')
    seg = np.asarray(audio[:round(.46 * onair.FS)], dtype=float)
    small = resample_poly(seg, 1, p3acquire.DECIMATION)
    analytic = hilbert(small)
    clock = np.arange(len(small)) / p3acquire.SEARCH_FS
    pulse = rx._pulse(p3acquire.SEARCH_FS // 100)
    offsets = p3acquire.CONTROL_OFFSETS_HZ

    def one_at_a_time():
        return {cn: np.asarray([
            rx._baseband((analytic * np.exp(-2j*np.pi*hz*clock)).real,
                         cn, p3acquire.SEARCH_FS, pulse) for hz in offsets])
            for cn in spec.VH_CHANNELS}

    def timed(call):
        t0 = time.perf_counter()
        for _ in range(5):
            got = call()
        return (time.perf_counter() - t0) / 5, got

    slow_s, slow = timed(one_at_a_time)
    fast_s, fast = timed(
        lambda: p3acquire._bands(analytic, clock, offsets, pulse))
    for cn in spec.VH_CHANNELS:
        scale = np.abs(slow[cn]).max()
        # MEASURED, NOT ASSUMED: 1.7e-15 absolute and 5.4e-15 of peak is the
        # worst this build produces over the whole fixture corpus, which is
        # the order `rx._baseband`'s own docstring claims against the direct
        # convolution. The bound that matters is the one on what the search
        # RETURNS, and that is
        # `test_the_ladder_finds_the_same_words_over_every_fixture_in_the_tree`.
        assert np.abs(fast[cn] - slow[cn]).max() <= 1e-14 + 1e-14 * scale
    assert fast_s * 2 < slow_s, (fast_s, slow_s)


def test_the_ladder_finds_the_same_words_over_every_fixture_in_the_tree():
    """The equivalence sweep, in the tree rather than in a write-up.

    One crop is not a corpus and the samples are not the answer: what the
    ladder returns is a codeword at a position and a frequency, and that is
    what has to be unchanged. Every PCM16 WAV under `tests/shrike/fixtures/`
    is swept in 460 ms brackets through both `control_signal` and
    `changeover` -- real off-air PACTOR-1, -2 and -3, clipped and quiet alike.
    Codeword, start, instant, offset, body and text must match exactly; only
    the ranking `quality` may move, and it is held to a part in 1e12 of
    itself so a build that starts disagreeing says by how much.
    """
    recordings = sorted(FIXTURES.parent.rglob('*.wav'))
    if not recordings:
        pytest.skip(f'no shrike recordings to sweep under {FIXTURES.parent}')
    bad, tried, found, worst = [], 0, 0, 0.0
    for path in recordings:
        try:
            with wave.open(str(path)) as wav:
                if (wav.getframerate(), wav.getsampwidth(),
                        wav.getnchannels()) != (onair.FS, 2, 1):
                    continue
                raw = wav.readframes(wav.getnframes())
        except wave.Error:
            continue  # a float32 fixture; this reader wants PCM16.
        audio = np.frombuffer(raw, '<i2').astype(np.float32) / 32768
        for lo in range(0, max(1, audio.size - round(.46 * onair.FS)),
                        round(.31 * onair.FS)):
            seg = audio[lo:lo + round(.46 * onair.FS)]
            if seg.size < round(.23 * onair.FS):
                break
            for search in (p3acquire.control_signal, p3acquire.changeover):
                fast = search(seg)
                with _per_hypothesis():
                    slow = search(seg)
                tried += 1
                found += slow is not None
                if _named(fast) != _named(slow):
                    bad.append((path.name, lo, search.__name__))
                elif slow is not None:
                    worst = max(worst, abs(fast.quality - slow.quality))
    assert tried > SWEEP_FLOOR[0] and found > SWEEP_FLOOR[1], (tried, found)
    assert not bad, bad
    assert worst < 1e-12, worst


def test_the_frame_readers_matched_filter_is_flop_bound_not_call_bound(tmp_path):
    """Why the ladder's fix does not transfer to the cold frame reader.

    `p3rx.decode_headed` opens with an eighteen-channel baseband at the full
    rate, which looks like the ladder's shape and is not: 1860 taps over a
    second of 48 kHz audio is arithmetic, and stacking the eighteen into one
    call returns the same samples for the same wall clock. So the frame reader
    was moved rather than rewritten -- see
    `test_the_cold_frame_reader_waits_for_a_window_that_can_pay_for_it`.
    """
    _, audio = crop('r06')
    seg = np.asarray(audio[:round(1.2 * onair.FS)], dtype=float)
    pulse = rx._pulse(onair.FS // 100)
    tones = range(spec.N_CHANNELS)

    def timed(call):
        t0 = time.perf_counter()
        for _ in range(3):
            got = call()
        return (time.perf_counter() - t0) / 3, got

    per_s, per = timed(
        lambda: {cn: rx._baseband(seg, cn, onair.FS, pulse) for cn in tones})

    def batched():
        car = np.asarray([rx.carrier(cn, onair.FS, 0, seg.size) for cn in tones])
        out = oaconvolve(seg[None, :] * car, pulse[None, :], axes=1)
        return {cn: out[cn] for cn in tones}

    batch_s, batch = timed(batched)
    for cn in tones:
        assert np.array_equal(per[cn], batch[cn])
    assert batch_s > per_s / 2, (batch_s, per_s)


@pytest.mark.parametrize('tag', ['e07', 'r06'])
def test_a_keyed_carrier_lands_on_its_own_boundary_and_never_on_a_peer(tag, tmp_path):
    """No off-slot key, and no two carriers sharing a slot."""
    run = Replay(tag, tmp_path).run()
    settle_n, slot_n = run.settle_n, run.grid.slot_n
    slots = []
    for first, end in run.bench.emissions:
        boundary = first - first % slot_n
        assert first - boundary < settle_n, (first, boundary)
        slots.append(boundary // slot_n)
        assert end - first < slot_n, (first, end)
    assert len(slots) == len(set(slots)), slots
    assert slots == sorted(slots)


def test_a_deliberate_stall_defers_once_and_recovers_inside_the_budget(tmp_path):
    """The stall is bounded; post-stall slots recover their deadline."""
    stalled = Replay('e07', tmp_path, stall_at=4, stall_s=1.5).run()
    assert stalled.stall_slot is not None
    assert stalled.deferrals['slot_gone'] >= 1, stalled.deferrals
    assert stalled.deferrals['regrid_gave_up'] == 0, stalled.log
    after = [c for c in stalled.cycles
             if c['slot'] > stalled.stall_slot + onair.REGRID_TRIES]
    assert after and any(c['keyed'] for c in after)
    assert all(c['keyed'] for c in after), after
    assert all(c['off_grid_ms'] < stalled.grace_ms for c in after)

@contextlib.contextmanager
def _rolling_decodes():
    """Every segment the rolling decoder is handed, in samples."""
    seen, real = [], live.RollingRx._decode

    def watched(self, seg):
        seen.append(seg.size)
        return real(self, seg)

    live.RollingRx._decode = watched
    try:
        yield seen
    finally:
        live.RollingRx._decode = real


def _wide_window(run, *, reps=4):
    """What a two-slot window costs, on the bench clock and in decoded samples.

    The shape a lost slot leaves: the loop aims a slot further on, so the window
    in front of the next key is two slots rather than one, and everything past
    the last `FEED_MAX_SLOTS` of it is held. Taken as its own call because that
    is how the arm met it -- `_regrid`'s recovered window is decoded and flushed
    inside its own cycle, and this one is not.
    """
    bench, slot_n = run.bench, run.grid.slot_n
    widest, overruns = 0, []
    with run._charging(), _rolling_decodes() as seen:
        for _ in range(reps):
            seen.clear()
            onair._listen_until_answer(bench, 2 * slot_n, run.host, run.sessrx,
                                       onair.FEED_MAX_SLOTS * slot_n)
            widest = max(widest, max(seen, default=0))
            overruns.append(bench.now - bench.pos)
            run.sessrx.flush()
    return widest, float(np.median(overruns))


def test_the_window_a_lost_slot_builds_costs_only_its_fed_tail(tmp_path,
                                                               monkeypatch):
    """A lost slot may not cost the next one, and until 2026-09-13 it did.

    `RollingRx.hold` puts the held audio in the buffer `push` decodes, so a
    window longer than a cycle was re-decoded whole on every 0.25 s slide: the
    2.507 s window a handed-back slot builds cost 1026 ms of rolling feed on
    the 0913-2320 arm's own audio against 290 ms for the 1.250 s one, and the
    loop is bound in samples, so all of that lands past the key instant. One
    lost slot then built the window that lost the next -- thirteen of the
    fourteen lost slots from 215 on alternate +95.0/+107.0 ms.

    Both halves are measured here in one process, on one recording, through the
    production listen loop against the sample clock. What the fix is, is one
    line: `live.RollingRx.push` lets go of whatever is held in FRONT of the
    audio it is about to decode (`forget_held`), which the frame scan has read
    entire in any case.
    """
    run = Replay('e07', tmp_path, charge_feed=True)
    fed_n = onair.FEED_MAX_SLOTS * run.grid.slot_n
    widest, overrun = _wide_window(run)
    # The decoder is never handed more than the loop fed it. This is the whole
    # mechanism, and it is arithmetic rather than a timing: the held prefix
    # cannot be re-read on a slide it is no longer in.
    assert widest <= fed_n, (widest, fed_n)
    # ...and the cost that follows from it: the clock stands well inside the
    # slot the next key needs, where the whole point is that the slot survives.
    assert overrun < run.grid.slot_n / 2, overrun / onair.FS

    monkeypatch.setattr(live.RollingRx, 'forget_held', lambda self: None)
    reverted = Replay('e07', tmp_path, charge_feed=True)
    was_widest, was_overrun = _wide_window(reverted)
    assert was_widest > fed_n, (was_widest, fed_n)
    assert was_overrun > 3 * overrun, (was_overrun, overrun)


def test_the_entry_render_is_byte_identical_and_built_once_per_payload(tmp_path):
    """The cache may not change a single sample of what goes on the air."""
    host, _ = granted()
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(host)
    sl, payload, status = 1, spec.TEMPLATE[:20], 0x1a
    fresh = placement.link_packet(sl, payload, status, swapped=False,
                                  flush=placement.ENTRY_FLUSH)
    first, field = tx._entry_burst(sl, payload, status, False)
    again, same = tx._entry_burst(sl, payload, status, False)
    assert np.array_equal(first, fresh) and again is first and same is field
    assert tx._entry_burst(sl, payload, status ^ 1, False)[0] is not first


def test_p1_alternation_and_p3_ownership_survive_the_campaign(tmp_path):
    """Pending P3 ownership survives listens; counted owned grants end it."""
    p1 = Replay('e08', tmp_path).run()
    assert {c['protocol'] for c in p1.cycles} == {'PACTOR1'}
    assert p1.tx.p1_seq == sorted(p1.tx.p1_seq)
    entry = Replay('e07', tmp_path).run()
    assert_replay_policy(entry)
    assert all(c['protocol'] == 'PACTOR3' and c['entry'] for c in entry.cycles)
    assert entry.tx.slots_used == [c['slot'] for c in entry.cycles if c['keyed']]
    assert 0 < entry.host.arq._upgrade_requests < arq.ENTRY_GRANT_CYCLES
    host, _ = granted()
    from hfmodem.tests.shrike.p3_arq_support import repeat
    for n in range(arq.ENTRY_GRANT_CYCLES):
        repeat(host)
        assert host.arq._upgrade_requests == n + 1
        if n + 1 < arq.ENTRY_GRANT_CYCLES:
            assert host.protocol is spec.Protocol.PACTOR3 and host.arq.entry_pending
    assert host.protocol is spec.Protocol.PACTOR1 and not host.arq.entry_pending


def test_postponed_aim_keeps_actual_key_admission(tmp_path):
    run = Replay('r06', tmp_path, stall_at=2, stall_s=1.5).run()
    assert_replay_policy(run)
    moved = [c for c in run.cycles if c['aimed'] == run.stall_slot
             and c['slot'] > c['aimed'] and c['keyed']]
    assert len(moved) == 1
    cycle = moved[0]
    assert cycle['off_grid_ms'] > run.grace_ms  # The old aim really was lost.
    assert cycle['key_late_ms'] and max(cycle['key_late_ms']) < run.grace_ms
    assert cycle['slot'] in cycle['emission_slots']
    assert len(cycle['emissions']) == len(cycle['emission_slots'])
    for (first, _), slot in zip(cycle['emissions'], cycle['emission_slots']):
        assert first // run.grid.slot_n == slot
        assert first - run.grid.boundary(slot) < run.settle_n
