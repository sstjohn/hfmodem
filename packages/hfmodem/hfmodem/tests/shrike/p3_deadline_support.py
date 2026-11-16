# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import io
import json
import time
from functools import cache
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.shrike import onair, p3acquire, pactor1, placement, rx, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.onair_support import _Bench, _Rig
from hfmodem.tests.shrike.p3_arq_support import granted
from hfmodem.tests.shrike.p3_reply_support import fallback

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
    """The search's bands as they flew, for the length of a `with`."""
    batched = p3acquire._grid
    p3acquire._grid = _flown_grid
    try:
        yield
    finally:
        p3acquire._grid = batched


def _named(got):
    """Everything a `Candidate` decides with; `quality` only ranks."""
    if got is None:
        return None
    ev = got.event
    return (ev.cs, ev.start, ev.t, got.offset_hz, got.coarse_hz,
            repr(ev.packet), ev.text)


def _one_at_a_time(analytic, clock, offsets, pulse):
    """`p3acquire._bands` as it flew: one matched-filter call per hypothesis."""
    return {cn: np.asarray([
        rx._baseband((analytic * np.exp(-2j*np.pi*hz*clock)).real, cn,
                     p3acquire.SEARCH_FS, pulse) for hz in offsets])
        for cn in spec.VH_CHANNELS}


def _flown_grid(analytic, offsets, first, count):
    """`p3acquire._grid` read off the bands as they flew."""
    bands = _one_at_a_time(analytic, np.arange(len(analytic)) / p3acquire.SEARCH_FS,
                           offsets, rx._pulse(p3acquire.SEARCH_FS // 100))
    return {cn: band[:, first::p3acquire.STEP][:, :count]
            for cn, band in bands.items()}


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
        saved.append((p3acquire, '_grid', p3acquire._grid))
        p3acquire._grid = _flown_grid
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
        self.stall_slot = None
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
        if whole.size and grid.sending and host.protocol != spec.Protocol.PACTOR3:
            onair._scan_frame(sessrx, whole, seg_start)
        whole, seg_start, _ = onair._collect(
            bench, sessrx, whole, seg_start,
            sessrx.control_collect_until(
                key_at - onair._prekey_lead(bench, settle_n),
                max(bench.pos, int(bench.sample_now())), grid))
        extent = sessrx._entry_p1_reply_extent(grid)
        owned = getattr(grid, '_p3_keyed_reply', None)
        acquisitions = len(self.cost['cs-acquire'])
        read_trace = []
        reply_key = (host.arq._link_generation, owned)

        def proof():
            row = sessrx._entry_reply_results.get(reply_key, {})
            return dict(complete=bool(row.get('complete')), grant=bool(row.get('grant')),
                        current=host.arq._entry_p1_current)

        supported_at_entry = bool(extent is not None and seg_start <= extent[0]
                                  and seg_start + whole.size >= extent[1])
        initial_origin, initial_end = seg_start, seg_start + whole.size
        p1_at_entry = grid.rx_due_in(initial_origin, initial_end)
        initial_reply = proof()
        acquire, deliver = p3acquire.control_signal, sessrx._on

        def traced_acquire(*args, **kwargs):
            read_trace.append(dict(event='cold-acquire', **proof()))
            return acquire(*args, **kwargs)

        def traced_delivery(ev, **kwargs):
            result = deliver(ev, **kwargs)
            if (ev.protocol == onair.Protocol.PACTOR1 and ev.kind == 'unassigned'
                    and ev.spare == pactor1.CS_59A):
                read_trace.append(dict(event='full-p1-grant', at=round(ev.t*onair.FS),
                                       **proof()))
            return result

        with patch.object(p3acquire, 'control_signal', traced_acquire), \
                patch.object(sessrx, '_on', traced_delivery):
            heard, _, whole, seg_start = onair._read_held_window(
                bench, grid, tx, sessrx, slot, whole, seg_start)
        reader_acquisitions = len(self.cost['cs-acquire']) - acquisitions
        # Preserve both the retry credit and its current, complete P1 proof.
        p1_candidate = host.arq._upgrade_asked
        p1_grant = host.arq._upgrade_asked
        p1_current = host.arq._entry_p1_current
        if (self.stall_at is not None and self.stall_slot is None
                and slot >= self.stall_at):
            bench.spend(self.stall_n)
            self.stall_slot = slot
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
        slot, whole, seg_start, _ = onair._regrid(
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
                                emissions=bench.emissions[keyed:],
                                emission_slots=tx.slots_used[keyed:],
                                key_late_ms=[(self.settle_n / onair.FS - lead) * 1e3
                                             for lead in bench.leads[keyed:]],
                                owned=owned, extent=extent,
                                p1_grant=p1_grant,
                                p1_candidate=p1_candidate,
                                p1_current=p1_current, heard=heard,
                                initial_reply=initial_reply,
                                supported_at_entry=supported_at_entry,
                                initial_origin=initial_origin, initial_end=initial_end,
                                p1_at_entry=p1_at_entry,
                                read_trace=read_trace,
                                reader_acquisitions=reader_acquisitions,
                                grants=host.arq._upgrade_requests,
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
        """Pre-regrid margins, including aims abandoned for a later slot.

        Actual emission admission is measured separately by ``key_late_ms``.
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
