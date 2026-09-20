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
import io
import json
import time
import wave
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert, oaconvolve, resample_poly

from hfmodem.shrike import arq, live, onair, p3acquire, placement, rx, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_granted_entry_retry import granted
from hfmodem.tests.shrike.test_p3_morning_timing import fallback
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FIXTURES = Path(__file__).with_name('fixtures') / 'slot-deadline-0910'
METADATA = FIXTURES / 'metadata.json'


@cache
def crops():
    if not METADATA.exists():
        pytest.skip(f'September 10 slot recordings absent: {METADATA}')
    return {row['file'].split('-')[0]: row
            for row in json.loads(METADATA.read_text())['fixtures']}

# What the arms measured for the peer's turnaround, and the PACTOR-1 geometry
# the grid is seeded with before the entry upgrades it.
D_S = {'e07': .1043, 'e05': .0981, 'r06': .1016, 'e08': .1035}
# The link each arm was holding when it lost its slots. The three PACTOR-3 arms
# start on a consumed grant with the entry still pending, which is the state
# every one of them was in for the whole campaign; the mail arm is the PACTOR-1
# positive control and starts where its own entry budget had already run out.
LINK = {'e07': granted, 'e05': granted, 'r06': granted,
        'e08': lambda: (fallback(), None)}
P1_DATA_N, P1_CS_N, D_MAX_N = 46080, 5760, 6240
# Each stage charges its own measured duration back to the sample clock. The
# four below it are measured only -- they are nested inside a charged stage and
# are here to say which part of it costs what.
CHARGED = (('scan', onair, '_scan_frame'),
           ('control', onair._SessionRx, 'control_signal_in'),
           ('flush', onair._SessionRx, 'flush'),
           ('upgrade', onair._SessionRx, 'upgrade_scan'),
           ('render', placement, 'link_packet'))
MEASURED = (('cs-acquire', p3acquire, 'control_signal'),
            ('cs-changeover', p3acquire, 'changeover'),
            ('compensate', p3acquire, 'compensate'),
            ('field', placement, 'build_field'))
# The rolling feed, charged only where a scene asks for it. Its cost is a
# property of the WINDOW rather than of the cycle -- `_SessionRx.feed` decodes
# the whole rolling buffer on every 0.25 s slide -- so a campaign that never
# hands a slot back pays the same figure every cycle, and charging it across the
# scenes above would re-baseline all of them for nothing they are about.
FEED = (('feed', onair._SessionRx, 'feed'),)
STAGES = tuple(name for name, _, _ in CHARGED + MEASURED + FEED)


@contextlib.contextmanager
def _per_hypothesis():
    """`p3acquire._bands` as it flew, for the length of a `with`."""
    batched = p3acquire._bands
    p3acquire._bands = _one_at_a_time
    try:
        yield
    finally:
        p3acquire._bands = batched


def _named(got):
    """Everything a `Candidate` decides with; `quality` only ranks."""
    if got is None:
        return None
    ev = got.event
    return (ev.cs, ev.start, ev.t, got.offset_hz, repr(ev.packet), ev.text)


def _one_at_a_time(analytic, clock, offsets, pulse):
    """`p3acquire._bands` as it flew: one matched-filter call per hypothesis."""
    return {cn: np.asarray([
        rx._baseband((analytic * np.exp(-2j*np.pi*hz*clock)).real, cn,
                     p3acquire.SEARCH_FS, pulse) for hz in offsets])
        for cn in spec.VH_CHANNELS}


def crop(tag):
    row = crops()[tag]
    return row, recorded_pcm({'file': f"slot-deadline-0910/{row['file']}",
                              'sha256': row['pcm_sha256']})


@contextlib.contextmanager
def charging(bench, cost, *, slow_ladder=False, charge_feed=False):
    """Wrap each production stage to charge its own duration to `bench`.

    The only clock that reproduces work which spends wall time and consumes no
    audio. `cost` collects each stage's measured milliseconds, keyed by name;
    the four MEASURED stages are nested inside charged ones and are timed but
    not charged, so they say which part of a charge costs what.
    """
    saved = []
    cost.update({stage: cost.get(stage, []) for stage in STAGES})

    def wrap(stage, owner, name, charge):
        fn = getattr(owner, name)
        saved.append((owner, name, fn))

        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                took = time.perf_counter() - t0
                cost[stage].append(took * 1e3)
                if charge:
                    bench.spend(round(took * onair.FS))
        setattr(owner, name, timed)

    for stage, owner, name in CHARGED + (FEED if charge_feed else ()):
        wrap(stage, owner, name, True)
    for stage, owner, name in MEASURED:
        wrap(stage, owner, name, False)
    if slow_ladder:
        saved.append((p3acquire, '_bands', p3acquire._bands))
        p3acquire._bands = _one_at_a_time
    try:
        yield
    finally:
        for owner, name, fn in saved:
            setattr(owner, name, fn)


class Replay:
    """One arm's entry campaign, cycle by cycle, against the deadline."""

    def __init__(self, tag, tmp_path, *, stall_at=None, stall_s=1.5,
                 slow_ladder=False, charge_feed=False):
        self.row, audio = crop(tag)
        self.host, _ = LINK[tag]()
        self.tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path,
                                settle=.04)
        self.host.peer = self.tx
        self.tx.attach(self.host)
        self.sessrx = onair._SessionRx(self.host)
        self.tx.sessrx = self.sessrx
        self.bench = _Bench(seconds=self.row['samples'] / onair.FS + 10)
        self.bench.audio[:audio.size] = audio
        # Our own transmitter is in this recording and is not in the live
        # stream: `flush_to` drops every sample of our carrier the moment it
        # ends, so the loop never reads one. The spans come off the arm's own
        # capture sidecars -- the gaps between the windows it actually read.
        for lo, hi in self.row['own_carrier_spans']:
            self.bench.audio[lo:hi] = 0.0
        self.tx.live = self.bench
        self.grid = onair._MasterGrid(0, 60000, round(.04 * onair.FS),
                                      packet_n=P1_DATA_N, cs_n=P1_CS_N,
                                      d_max_n=D_MAX_N)
        self.grid.d_n, self.grid.d_ref_n = D_S[tag] * onair.FS, P1_DATA_N
        self.grid.keyed_slot = 0
        self.settle_n = round(.04 * onair.FS)
        # On the grid from the first cycle: the replay opens where the arm's own
        # first packet in the crop ended, which is where `flush_to` would have
        # left the read cursor and where a receive window begins.
        self.bench.now = self.bench.pos = next(
            end for _, end in self.row['own_carrier_spans']
            if end > round(.5 * onair.FS))
        # The crop starts after a recorded emission. This initial record comes
        # from its measured carrier span; subsequent records come only from
        # the real RadioTx emission path exercised below.
        if self.host.protocol == onair.Protocol.PACTOR3:
            first, end = next((lo, hi) for lo, hi in self.row['own_carrier_spans']
                              if hi == self.bench.pos)
            self.grid._p3_keyed_reply = (0, first, end, 60000)
        self.first_slot = self.bench.pos // 60000 + 1
        self.stall_at, self.stall_n = stall_at, round(stall_s * onair.FS)
        # The negative control: the frequency ladder one hypothesis at a
        # time, as it flew, with everything else identical. Same machine, same
        # audio, same run -- so the comparison is of two implementations of
        # one search and not of two machines.
        self.slow_ladder = slow_ladder
        self.charge_feed = charge_feed
        self.cost = {stage: [] for stage in STAGES}
        self.cycles, self.log = [], ''

    # -- charging ---------------------------------------------------------
    def _charging(self):
        return charging(self.bench, self.cost, slow_ladder=self.slow_ladder,
                        charge_feed=self.charge_feed)

    # -- the cycle --------------------------------------------------------
    def _cycle(self, slot):
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            slot = self._run_cycle(slot)
        self.log += said.getvalue()
        self.cycles[-1]['said'] = said.getvalue()
        return slot

    def _run_cycle(self, slot):
        bench, grid, tx, host = self.bench, self.grid, self.tx, self.host
        sessrx, settle_n = self.sessrx, self.settle_n
        # A burst can leave from inside the collection too -- the FSM answers a
        # decode and `_tx` puts it on the grid itself -- so the cycle's keying
        # is counted over the whole of it and not around the tick.
        keyed = len(bench.emissions)
        mark = {stage: len(v) for stage, v in self.cost.items()}
        sessrx.new_cycle()
        tx.defer_p3_cs = True
        slot, _ = onair._regear_next_slot(grid, slot, host.arq.cycle_long)
        slot = onair._keyable_slot(bench, grid, slot, settle_n)
        tx.aim(grid, slot)
        key_at = tx.key_instant(grid, slot)
        win = (key_at - settle_n - bench.holdback
               - round(onair.PREKEY_RESERVE_S * onair.FS) - bench.pos)
        whole = onair._listen_until_answer(
            bench, win, host, sessrx, onair.FEED_MAX_SLOTS * grid.slot_n)
        seg_start = bench.pos - whole.size
        whole, seg_start, _ = onair._collect(
            bench, sessrx, whole, seg_start,
            sessrx.control_bridge_until(
                key_at - max(grid.d, settle_n),
                key_at - onair._prekey_lead(bench, settle_n),
                max(bench.pos, int(bench.sample_now())), grid))
        if whole.size and grid.sending:
            onair._scan_frame(sessrx, whole, seg_start)
        whole, seg_start, _ = onair._collect(
            bench, sessrx, whole, seg_start,
            sessrx.control_collect_until(
                key_at - onair._prekey_lead(bench, settle_n),
                max(bench.pos, int(bench.sample_now())), grid))
        if slot == self.stall_at:
            bench.spend(self.stall_n)
        sessrx.control_signal_in(whole, seg_start, grid)
        # Where the clock stands against the PTT instant of the slot this cycle
        # aimed at, before anything is handed back. This is the margin the whole
        # question turns on: past it the carrier cannot come up on the boundary.
        off = (bench.sample_now() - (key_at - settle_n)) / onair.FS
        # ...unless this cycle's burst already went out from inside the
        # collection, the FSM having answered a decode: `_regrid`'s ALREADY
        # SPENT branch. The clock is then a whole burst past the instant and
        # measures the carrier, not the work in front of it.
        aimed = slot
        spent = bool(tx.slots_used) and tx.slots_used[-1] >= slot
        slot, whole, seg_start = onair._regrid(
            bench, grid, tx, host, sessrx, slot, whole, seg_start, settle_n)
        host.tick()
        tx.emit_pending_cs()
        sessrx.flush()
        if whole.size:
            onair._scan_frame(sessrx, whole, seg_start, upgrade=True)
        self.cycles.append(dict(aimed=aimed, slot=slot,
                                window_s=whole.size / onair.FS,
                                off_grid_ms=off * 1e3,
                                keyed=len(bench.emissions) > keyed,
                                intent=not spent and (
                                    len(bench.emissions) > keyed
                                    or aimed != slot),
                                protocol=host.protocol.name,
                                entry=host.arq.entry_pending,
                                spent_ms={
                                    stage: round(sum(v[mark[stage]:]), 2)
                                    for stage, v in self.cost.items()
                                    if sum(v[mark[stage]:]) > .05}))
        return grid.next_slot(slot)

    def run(self, cycles=None):
        # The last slot whose own carrier still fits inside the crop.
        slot = self.first_slot
        last = self.row['last_slot'] - self.row['first_slot'] - 1
        with self._charging():
            while slot <= last and (cycles is None or len(self.cycles) < cycles):
                slot = self._cycle(slot)
        return self

    # -- what the transcript says -----------------------------------------
    @property
    def deferrals(self):
        """Slots the runtime's own words say it gave up, on cycles it keyed.

        A cycle the ARQ had nothing for keys nothing, and an aim moved on such
        a cycle costs no transmission -- the operator's standard is the
        ELIGIBLE slot. A cycle that gave a slot back and then keyed is a miss
        whatever it went on to do with the next one.
        """
        said = ''.join(c['said'] for c in self.cycles if c['keyed'])
        return dict(slot_gone=said.count(onair.SLOT_GONE),
                    late_to_key=said.count(onair.LATE_KEY),
                    pre_render=said.count('The unrendered burst would miss'),
                    regrid_gave_up=said.count('REGRID GAVE UP'))

    @property
    def missed(self):
        return sum(self.deferrals.values())

    @property
    def forgiven(self):
        """Carriers the grid kept by keying them INSIDE their own boundary.

        The other price of an overrun, and since 2026-09-13 it is the one a
        sub-symbol overrun actually pays: `onair._clamp_forgives` leaves an
        overrun the reader forgives to the emission path instead of spending a
        slot on it, so a cycle that used to appear here as `SLOT ... IS GONE`
        appears as `SLOT ... IS KEPT` and its carrier goes out. The work is
        just as late either way, which is what these scenes measure.
        """
        said = ''.join(c['said'] for c in self.cycles if c['keyed'])
        return said.count(onair.SLOT_KEPT)

    @property
    def overran(self):
        """Cycles whose overrun cost the campaign something -- a slot given
        away, or a carrier that had to be keyed late into its boundary."""
        return self.missed + self.forgiven

    @property
    def off_grid_ms(self):
        """The margin on the cycles that actually took the slot they aimed at.

        A cycle the ARQ had nothing for keys nothing and moves no clock, so its
        window opens wherever the last one left off and its margin measures the
        gap rather than the work.
        """
        return [c['off_grid_ms'] for c in self.cycles if c['intent']]

    @property
    def idle(self):
        return sum(not c['intent'] for c in self.cycles)

    @property
    def grace_ms(self):
        """How far past the PTT instant a carrier can still make the boundary."""
        return (self.settle_n - self.bench.key_notice) / onair.FS * 1e3

    def stage_table(self):
        return {stage: dict(n=len(v), median=float(np.median(v)) if v else 0.0,
                            maximum=max(v) if v else 0.0, total=float(sum(v)))
                for stage, v in self.cost.items()}


def instrumentation_overhead_ms(n=20000):
    t0 = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        _ = time.perf_counter() - t
    return (time.perf_counter() - t0) / n * 1e3


NONE = dict(slot_gone=0, late_to_key=0, pre_render=0, regrid_gave_up=0)


@pytest.mark.parametrize('tag', ['e07', 'e05'])
def test_every_eligible_slot_is_keyed_under_the_recorded_workload(tag, tmp_path):
    """Zero deferrals across the campaign, and every margin inside the notice."""
    run = Replay(tag, tmp_path).run()
    assert run.cycles, 'the replay collected no cycles'
    assert run.deferrals == NONE, run.deferrals
    assert max(run.off_grid_ms) < run.grace_ms, run.off_grid_ms


def test_the_pactor1_control_pays_no_entry_acquisition(tmp_path):
    """The arm that lost nothing, and the reason it lost nothing.

    E08 is the PACTOR-1 mail call of the same evening: same loop, same guards,
    same 8 ms of notice, nine deferrals in 197 keys on the air and none of them
    an entry cycle. Its control read is the two anchored words and nothing
    else, and its margin sits an order inside the notice where an entry arm
    reading in front of its key sits most of the way through it. That
    contrast is the measurement.

    Its deferral COUNT is not asserted, and that is about this machine rather
    than about the station. The replay charges each stage's real duration to
    the sample clock, and E08's post-key sweep for the protocols it is not in
    measures 62 ms at rest against the 244 ms of dead time behind a 0.96 s
    PACTOR-1 carrier -- but 88 ms in one run here and 107 in the next, and
    above about 100 it takes the slot. An instrument that is being shaken
    cannot be asked that question; it can be asked this one.
    """
    run = Replay('e08', tmp_path).run()
    assert run.stage_table()['cs-acquire']['n'] == 0, 'a PACTOR-1 cycle acquired'
    assert max(run.off_grid_ms) < run.grace_ms / 4, run.off_grid_ms
    entry = Replay('e07', tmp_path, slow_ladder=True).run()
    assert entry.stage_table()['cs-acquire']['n'] > 0, 'entry never acquired'
    # Earlier collection intentionally removes work from the old PTT margin;
    # compare the actual charged work, not that obsolete placement ratio.
    assert entry.stage_table()['control']['median'] > run.stage_table()['control']['median']


def test_the_upgrade_cycle_stops_costing_a_slot(tmp_path):
    """R06's upgrade cycle, which was the last slot any arm lost.

    Its link upgrades mid-crop and the cycle before the upgrade keys nothing,
    so on the flown ladder the next window opened over a second long instead
    of a third of one and the cold PACTOR-3 frame reader ran over all of it --
    132 ms, of which 124 is the blind alignment scan run twice. That reader
    was never touched. With the ladder halved the cycle keys, the long window
    never forms, and the arm holds its cadence from the grant on: sixteen
    carriers against twelve, zero deferrals against two.

    THE ABSOLUTE COUNT IS NOT THE ASSERTION HERE AND THE OTHER ARMS' IS.
    R06's worst margin is +7.1 ms of the 8.0 a 40 ms settle leaves, measured
    on a build host at load 3.7 where the ladder runs at twice its quiet
    cost; E07 and E05 sit at +5.1 and clear it by three. Run inside the whole
    directory, against nine hundred neighbours, R06's charged clock tips and
    one slot goes -- the measuring machine, not the station, but not
    something to assert through either. What is asserted is what the change
    did, measured both ways in this process.

    AND THE SLOT IS NO LONGER WHAT IT COSTS. `onair._clamp_forgives` leaves a
    sub-symbol overrun to the emission path rather than handing the slot back,
    so R06's upgrade cycle now keys on the flown ladder too -- late into its own
    boundary, and said out loud as `SLOT ... IS KEPT`. What the ladder costs is
    therefore counted as `overran` rather than as carriers, which is what this
    docstring already said the assertion had to be.
    """
    flown = Replay('r06', tmp_path, slow_ladder=True).run()
    fixed = Replay('r06', tmp_path).run()
    assert flown.stage_table()['cs-acquire']['n'] > 0
    assert fixed.stage_table()['cs-acquire']['n'] > 0
    assert fixed.overran == 0, (fixed.deferrals, fixed.forgiven)
    assert len(fixed.bench.emissions) >= len(flown.bench.emissions)
    # THE NOTICE IS THE LINE AND THE HALVING WAS A PROXY FOR IT. Measured here
    # the upgrade cycle is +10.7 ms flown and +6.6 ms batched, so the ratio is
    # 1.6 where it used to read 2 -- the admission guard is block-quantised now
    # (`_Bench.samples`) and the flown figure no longer runs away into a second
    # lost slot. What the change actually buys is the same both ways: the
    # batched ladder's worst margin is inside the 8.0 ms a 40 ms settle leaves
    # and the flown one's is not. The 7.1 ms this file's header records from a
    # build host at load 3.7 is the headroom that leaves.
    assert max(fixed.off_grid_ms) < fixed.grace_ms, fixed.off_grid_ms
    assert flown.stage_table()['cs-acquire']['median'] > fixed.stage_table()['cs-acquire']['median']


@pytest.mark.parametrize('tag', ['e07', 'e05'])
def test_the_flown_ladder_overruns_the_notice_and_the_batched_one_does_not(
        tag, tmp_path):
    """Both implementations of one search, same machine, same audio, same run.

    The comparison is of two ways of calling the same matched filter and not
    of two machines: one hypothesis at a time it runs past the notice a 40 ms
    settle leaves, on nearly every aim of the campaign; stacked into one call it
    fits, and no cycle comes within the notice of its own key instant.

    WHAT THE OVERRUN COSTS MOVED, AND THE OVERRUN DID NOT. Until 2026-09-13 a
    cycle past the notice handed its slot back, so the flown ladder emitted
    twelve carriers here against sixteen. `onair._clamp_forgives` now leaves a
    sub-symbol overrun to the emission path, which keys it INSIDE its boundary
    -- the campaign keeps its carriers and pays in placement instead. So the
    count this scene pins is `overran`, which is the same cycles under either
    rule, and the measurement is the population of margins it samples.
    """
    flown = Replay(tag, tmp_path, slow_ladder=True).run()
    fixed = Replay(tag, tmp_path).run()
    # THE MARGIN IS THE MEASUREMENT AND THE COUNT IS A SAMPLING OF IT. The
    # admission guard runs on the codec's DELIVERED count, which arrives one
    # `_blk` at a time and so lags the converter by up to 2.67 ms; a budget
    # sitting 0.2-1.4 ms past the notice therefore loses its slot only on the
    # cycles where a block boundary falls inside that overshoot. The flown
    # ladder is past the notice on 13 of 16 aims here and on 14 of 16 on E05,
    # and pays for it on 3 to 4 cycles of the campaign -- forgiven into their
    # boundaries now, spent outright before that, and lost every one of them to
    # a guard taken on the converter's own position. So the campaign is pinned
    # on the population of margins, and the count carries only that the overrun
    # does still cost the cycle something.
    assert flown.stage_table()['cs-acquire']['n'] > 0
    assert fixed.stage_table()['cs-acquire']['n'] > 0
    # A FRACTION RATHER THAN A SLACK OF TWO, because the campaign got longer:
    # the four cycles the forgiveness gives back are four more margins, and an
    # absolute slack written against a twelve-cycle run is a different demand on
    # a sixteen-cycle one. Three quarters is what the batched ladder misses by
    # the whole population -- it is past the notice on none of them.
    assert flown.stage_table()['cs-acquire']['median'] > fixed.stage_table()['cs-acquire']['median']
    assert fixed.deferrals == NONE, fixed.deferrals
    assert fixed.overran == 0, (fixed.deferrals, fixed.forgiven)
    assert max(fixed.off_grid_ms) < fixed.grace_ms, fixed.off_grid_ms
    assert len(fixed.bench.emissions) >= len(flown.bench.emissions)


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
    assert tried > 500 and found > 20, (tried, found)
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
    """An injected stall must cost its own slot, say so, and then converge."""
    stalled = Replay('e07', tmp_path, stall_at=4, stall_s=1.5).run()
    assert stalled.deferrals['slot_gone'] >= 1, stalled.deferrals
    assert stalled.deferrals['regrid_gave_up'] == 0, stalled.log
    after = [c for c in stalled.cycles if c['slot'] > 4 + onair.REGRID_TRIES]
    assert after and all(c['keyed'] for c in after), after
    assert all(c['off_grid_ms'] < stalled.grace_ms for c in after), after


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


def test_p1_alternation_and_p3_ownership_survive_the_campaign(tmp_path, monkeypatch):
    """The control arm stays PACTOR-1; the entry arm owns P3, then falls back.

    THE FALLBACK IS DRIVEN AT A BUDGET THE CROP REACHES. A pending entry gets
    `arq.ENTRY_GRANT_CYCLES` requests under the repeated-grant rule, and E07 is
    sixteen cycles: at the shipping fourteen the budget outlives the recording,
    so the arm owns PACTOR-3 to the end of it with the entry still pending. Run
    at eight -- the number of requests this crop actually contains, and what the
    limit was when it was cut -- the same audio falls back at cycle twelve. Both
    halves are asserted, because what the case is about is that the entry owns
    the protocol while it is pending and gives it up when the budget is spent,
    and not what the budget's value happens to be.
    """
    p1 = Replay('e08', tmp_path).run()
    assert {c['protocol'] for c in p1.cycles} == {'PACTOR1'}, p1.cycles
    assert p1.tx.p1_seq == sorted(p1.tx.p1_seq), p1.tx.p1_seq
    entry = Replay('e07', tmp_path).run()
    assert all(c['protocol'] == 'PACTOR3' for c in entry.cycles if c['entry'])
    assert all(c['entry'] for c in entry.cycles), 'the budget ran out in the crop'
    assert entry.tx.slots_used == list(range(entry.first_slot,
                                             entry.first_slot + len(entry.cycles)))
    monkeypatch.setattr(arq, 'ENTRY_GRANT_CYCLES', 8)
    spent = Replay('e07', tmp_path).run()
    assert all(c['protocol'] == 'PACTOR3' for c in spent.cycles if c['entry'])
    assert spent.cycles[-1]['protocol'] == 'PACTOR1', 'the budget never ran out'
    assert not spent.cycles[-1]['entry']
    assert spent.tx.slots_used == entry.tx.slots_used
