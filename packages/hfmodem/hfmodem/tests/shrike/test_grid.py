# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the CARRIER came up, against where the grid said to key.

Every other timing check in this repository measures an intention. The replay
harness reconstructs each recorded window onto the 1.25 s raster, so a
transmission that slips behind a live receive window cannot appear in it at all;
the session summary's slot increment is computed from slot INDICES, so it reads
1.25 s while the air is anything at all. Both are working as designed, and
neither can see a burst that went out in the wrong slot.

captures/onair-0801-2156 is what they missed. Four consecutive cycles of a live
session against a real gateway, hold offset against the next transmission's
offset from the boundary:

    +1467.2 ms  ->  +1455.8 ms        +31.2 ms  ->  +17.8 ms
    +1422.3 ms  ->  +1409.8 ms        +23.1 ms  ->  +11.8 ms

Within 12 ms every time, over a range of 1.44 s: the emission was chained to
where the CAPTURE sat rather than to the grid. Two of those are more than a whole
1250 ms slot, so the burst went out in the next slot -- on top of the station we
were calling, where nothing is listening.

The mechanism is that `_LiveInput.transmit` cannot refuse. A DAC has no way to
emit into the past, so an aim point that has gone by is clamped to the earliest
instant the stream can manage, which is wherever the loop is standing. The
never-transmit-late guard is evaluated once, at the top of the cycle, against the
READ position; everything between it and the key costs wall-clock time and no
samples, so the converter walks past the boundary after the guard has approved
it and nothing looks again.

So this file drives the whole session loop over a duplex stream whose clock is
ARITHMETIC rather than wall time -- reads advance it, and decoding is charged to
it -- and asks two questions of the result: did the carrier come up ON a boundary
of the grid the session was running, and IN THE SHIFT that boundary calls for?
Both, because they are one fact about one cycle: a station that keys on the right
boundary in the wrong shift is unreadable to a peer counting cycles, and
unreadable in a way that looks exactly like a dead band.

THE DECODE IS CHARGED WHERE IT RUNS, and where each one runs is the other half of
this file. The frame scan is spent out of the listen window, in front of the
bridge; the anchored control-signal read is spent out of the keying settle,
because it wants audio the bridge is still collecting; the flush and the scan for
a protocol we are not in are spent behind our own carrier. Charged anywhere below
that carrier it is all free: the next cycle's guard absorbs it, and the loop that
pays it twice over -- once in the cycle and again on every retry of the re-grid --
looks like a loop that converges. `FLUSH_N`, `SCAN_N`, `CS_N` and `UPGRADE_N` are
what those four calls measure at on this machine and are what EVERY cycle pays;
`OVERRUN_N` is a cycle that lost its slot outright, which is a different claim.

THE PRE-KEY WINDOW IS ITSELF A MEASUREMENT HERE. `_Bench` records where its clock
stood for every charge, so what a cycle spends between the bridge and the key is
a number rather than an argument -- and `scan_late` puts the frame scan back in
that window, which is the placement this file was written against.

The counterexample is planted rather than assumed. Run against onair.py as it
flew on 2026-08-01, with nothing patched but the stream, the same bench puts the
carrier 1142 ms from the slot it was aimed at -- of a 1250 ms slot -- and 324 ms
off the raster, while a cycle that fits inside its slot is exact to the sample.
The negative controls below reproduce that from inside the suite.

Run: python -m hfmodem.tests.shrike.test_grid
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from hfmodem.shrike import (arq, modem, onair, pactor1, placement, ptc, rxfront,
                            spec)
from hfmodem.shrike.arq import State
from hfmodem.shrike.spec import Protocol

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
# What the on-air session lost per cycle: the log's own off-grid readings ran
# +1422 to +1467 ms on the cycles that overran, against a 1250 ms slot. 1.4 s is
# inside that range and comfortably more than the ~1.0 s listening window a
# one-slot cadence leaves, which is what makes the slot unreachable.
OVERRUN_N = round(1.40 * FS)
# ...and what the decodes a keyed cycle really runs cost, MEASURED on this
# machine, this repo, over a 1.30 s window: `_SessionRx.flush` 26.7 ms over its
# 0.75 s cap, `deep_scan` 20.3 ms with nothing in the channel, the anchored
# control-signal read 0.3 ms, and `upgrade_scan`'s blind PACTOR-3 pass 48.0 ms.
#
# These are the numbers the overrun above cannot stand in for. The bulk models a
# cycle that lost its slot outright; these are the ordinary cost of every cycle,
# and they are charged to `_regrid`'s OWN flush and scan as well -- which is what
# makes a re-grid that cannot converge reachable from inside the suite.
FLUSH_N = round(0.027 * FS)
SCAN_N = round(0.021 * FS)
CS_N = round(0.0003 * FS)
UPGRADE_N = round(0.048 * FS)
# ...and how those costs SCALE with the audio handed to them, which a fixed
# figure cannot carry and which is what the 2026-08-02 session ran away on. A
# decode charged per cycle makes a longer window free; charged per SAMPLE it
# makes a longer window dearer, and a loop that chooses its window from how far
# behind it already is then feeds itself.
#
# MEASURED ON THE RUNAWAY'S OWN AUDIO -- captures/onair-0802-1609/hold_08.wav,
# the 59.8 s window the session collected -- through `_SessionRx`'s own rolling
# parameters, which are a 4 s window re-decoded every 0.25 s slide, so a second
# of channel is looked at sixteen times. CPU seconds per second of audio:
#
#     window   1.25 s   2.50 s   5.00 s   10.0 s
#     feed      2.99 s  15.52 s  68.68 s   187 s
#     per s      2.4      6.2     13.7      18.7
#
# The same call over synthetic noise reads 0.11, 0.27 and 0.81 across the same
# lengths -- under one throughout -- which is why nothing offline had ever seen
# this: an empty channel gives the decoder nothing to run its scans on, and it is
# REAL AUDIO that makes it dear.
#
# 2.4 is the smallest of those and it is the one charged here. Above one is the
# whole of what matters -- a receiver that cannot keep up with the band falls
# further behind every cycle, and a window measured from how far behind it is
# then grows without bound. The frame scans are minor beside it: 0.014 and 0.070
# per second measured on the same file.
FEED_RATE = 2.4
SCAN_RATE = 0.05
UPGRADE_RATE = 0.07
# `_LiveInput.read` takes the capture queue one 128-frame block at a time and
# concatenates: 4.4 ms to drain a 20 s backlog, 10.2 ms for 60 s, measured. Small
# -- and it is spent inside the keying settle, by `RadioTx._tx`'s own
# `take_until`, which is where a millisecond is not small.
DRAIN_RATE = 0.00022
# An overrun that loses the slot and NOTHING MORE. Comfortably past the boundary
# so it is genuinely gone, and inside `slot - settle - key_notice` = 1178 ms so
# the very next boundary is still reachable with a full settle. That is what
# makes it the shift's own gate: with the next slot available on every other
# ground, the only thing that can push a burst past it is the polarity its
# samples are in.
NUDGE_N = round(0.60 * FS)
# The class itself, not the module attribute: `_session` rebinds
# `onair._LiveInput` to the bench factory for the length of a run, and the
# bench must keep reaching the PRODUCTION notice and clamp through it -- a
# bench that re-derives them is a model of the code under test, and a model
# cannot disagree with what it models. That is how a stubbed `clamp_late`
# once sailed through every scene in this file.
_LIVE = onair._LiveInput
ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


class _Bench:
    """The duplex stream's contract, on a clock made of arithmetic.

    `_LiveInput` is the session's master clock: the converter runs at 48 kHz
    whatever the OS is doing, readers ask for absolute sample positions, and the
    transmitter schedules against the same counter. All of that is reproduced
    here exactly, with one substitution -- the converter advances when audio is
    read or when work is charged against it, rather than when wall time passes.
    That makes an overrun a number in the test rather than a race, and it is the
    only way this measurement can be taken without a radio.

    The notice and the clamp are `_LiveInput`'s own methods, run on this clock,
    NOT copies: the bench supplies `samples`, `_lat` and `_blk` and borrows the
    arithmetic, so breaking the production placeability breaks these scenes.

    The peer is REACTIVE, which the corpus cannot be: it answers `d` after our
    carrier drops, in the shift we sent in, so a session that transmits in the
    wrong slot is answered in the wrong slot too.
    """

    # Measured on this machine and hard-coded in `_LiveInput`: 1152 samples from
    # a sample's ADC to its DAC, at a 128-frame block, and three blocks of notice
    # before the callback that fills the buffer after next.
    LAT = 1152

    # The turnaround the schedule is designed for, and it is measured: 105 ms is
    # what `D_NOMINAL_S` reads off a W4DNA -> KE5YTA exchange between two
    # commercial modems. It stood at 55 here, which is `TR_SWITCH_S` exactly --
    # a peer answering at the precise instant this rig stops being deaf, so
    # every tolerance derived from the gap collapsed to nothing and the bench
    # was measuring the edge of audibility rather than a working link.
    D_S = 0.105

    def __init__(self, *, seconds: float = 200.0, blk: int = 128,
                 holdback: int = 608, d: float = D_S, peer=None,
                 lat_in: int = 0):
        self.fs = FS
        self.audio = np.zeros(int(seconds * FS), np.float32)
        # A session that never keys again is the failure this file exists to
        # catch and it does not announce itself: the loop keeps handing slots
        # back, `host.tick()` is never reached, so the ARQ never advances and
        # `--max-cycles` never decrements. Nothing but a clock cap turns that
        # into a result. The scenes here run to ~95 s of bench clock; a
        # non-converging re-grid burns a slot an iteration, so the headroom is
        # about eighty iterations of one.
        self.limit = self.audio.size
        self.now = 0            # the converter's position
        self.pos = 0            # the next sample a reader will get
        self.floor = 0
        # The counters a capture source carries, which the session loop reads
        # straight into every sidecar. `lost` is the blocks a starved
        # interpreter never accepted; a bench is never starved, and the loop
        # still asks.
        self.xruns, self.xrun_at, self.underruns, self.lost = 0, None, 0, 0
        self.holdback, self._blk, self._lat = holdback, blk, self.LAT
        self.tx_latency_n = 0     # a bench station has measured no residue
        self.lat_in = lat_in
        self.keyed_s = 0.0
        self.keyed_at: int | None = None
        self.d_n = round(d * FS)
        self.peer = peer
        self.emissions: list[tuple[int, int]] = []
        self.leads: list[float] = []      # PTT lead each burst actually got
        self.taken: list[tuple[int, int]] = []      # spans handed to the reader
        self.drained: list[tuple[int, int, int, int]] = []
        # (read start, read end, charge instant, charged samples), retained so
        # the final ID's all-at-once drain is accounted separately from CS work.
        self.charge = 0         # samples of clock the cycle's decode overruns by
        self.flush_n = 0        # ...and what one flush costs
        self.scan_n = 0         # ...and one frame scan
        self.cs_n = 0           # ...and the anchored control-signal read
        self.upgrade_n = 0      # ...and the blind scan for a protocol we are not in
        # ...and the same costs charged PER SAMPLE of the audio handed to them,
        # which is the only way a window's length can pay for itself. Zero unless
        # a scene asks for it, so the fixed-cost scenes above measure what they
        # always measured.
        self.feed_rate = 0.0
        self.scan_rate = 0.0
        self.upgrade_rate = 0.0
        self.drain_rate = 0.0
        # Where the clock stood for every charge since the last carrier, so what
        # a cycle spends INSIDE THE KEYING SETTLE is a measurement rather than an
        # inference. That window is the one this file exists about: work in it is
        # charged to the rig's PTT lead, and past `key_notice` it costs the slot.
        self.spent: list[tuple[int, int]] = []
        self.prekey: list[int] = []
        self.armed = False      # the bulk is one cycle's, not one decode's
        self.answered = False   # ...and so is the peer's transmission
        self.always_placeable = False

    # -- the session's view of the codec ----------------------------------
    @property
    def samples(self) -> int:
        """What the CALLBACK HAS HANDED OVER -- not where the converter stands.

        THE TWO NUMBERS THE ADMISSION GUARD IS THE DIFFERENCE OF. `_LiveInput`
        hands Python one `_blk`-frame block at a time, so the delivered count is
        always a multiple of the block and always lags `sample_now()`; the guard
        (`clamp_late`) is taken on this one while the cycle's last read is
        scheduled on the other. This returned `self.now` -- one continuous
        counter standing in for both -- so the quantisation at the heart of the
        2026-09-11 entry-slot losses did not exist here, and no test in this
        repository could fail the way that arm did. 60000 samples a slot is
        468.75 blocks, so the room a cycle has before the guard trips takes four
        values and returns; a budget sitting on the threshold then loses one slot
        in four, deterministically, which is what
        `test_granted_entry_slots` reproduces.
        """
        return max(0, (self.now - self.lat_in) // self._blk * self._blk)

    @property
    def key_notice(self) -> int:
        return _LIVE.key_notice.fget(self)

    def _advance(self, to: int) -> None:
        self.now = max(self.now, to)
        if self.now > self.limit:
            raise RuntimeError(
                f"the session ran past {self.limit / FS:.0f} s of bench clock "
                f"without finishing: {len(self.emissions)} carriers on the air. "
                f"A loop that hands slots back forever occupies nothing and "
                f"reports nothing wrong.")

    def spend(self, n: int) -> None:
        """Charge `n` samples of clock to work that consumed no audio."""
        self.spent.append((self.now, n))
        self._advance(self.now + n)

    def _grab(self, a: int, b: int) -> np.ndarray:
        seg = np.zeros(max(0, b - a), np.float32)
        lo, hi = min(max(a, 0), self.audio.size), min(max(b, 0), self.audio.size)
        if hi > lo:
            seg[:hi - lo] = self.audio[lo:hi]
        return seg

    def _hand_over(self, a: int, b: int) -> np.ndarray:
        self.pos = b
        if b > a:
            self.taken.append((a, b))
        return self._grab(a, b)

    def read(self, count: int) -> np.ndarray:
        a = max(self.pos, self.floor)
        b = a + max(0, int(count))
        self._advance(b)                 # live audio cannot be read early
        got = self._hand_over(a, b)
        # A read is not free of the backlog it drains -- see DRAIN_RATE. Charged
        # here rather than at the decode because `RadioTx._tx` runs one of these
        # after the bridge sleep, inside the keying settle, and decodes nothing.
        if self.drain_rate and got.size:
            cost = round(got.size * self.drain_rate)
            self.drained.append((a, b, self.now, cost))
            self.spend(cost)
        return got

    def take(self, seconds: float) -> np.ndarray:
        return self.read(int(seconds * self.fs))

    def take_until(self, index: int) -> np.ndarray:
        return self.read(max(0, index - self.pos))

    def read_ready(self) -> np.ndarray:
        return self.read(max(0, self.samples - max(self.pos, self.floor)))

    def sample_now(self) -> float:
        return float(self.now)

    def wait_until(self, index: int) -> float:
        slept = max(0, index - self.now)
        self._advance(index)
        return slept / self.fs * 1e3

    def clamp_late(self, at: int) -> int:
        # `always_placeable` is the assumption the loop used to run on -- that a
        # boundary aimed at is a boundary reachable -- and it is set only by the
        # negative controls below, to put the session back the way it flew.
        if self.always_placeable:
            return 0
        return _LIVE.clamp_late(self, at)

    def flush_to(self, index: int) -> None:
        self.floor = self.pos = index

    def flush(self, before: float | None = None) -> None:
        self.floor = self.pos = self.now

    def transmit(self, audio, *, at=None, settle: float = 0.0, key=None,
                 max_key: float = 40.0) -> tuple[int, int]:
        earliest = self.samples + 3 * self._blk
        start = earliest if at is None else max(at - self.LAT, earliest)
        first = start + self.LAT
        # PTT cannot be asserted in the past either, so an aim point accepted
        # with less notice than the rig's settle comes out as a SHORT SETTLE
        # rather than as a late burst. That is silent on the air -- the carrier is
        # where it was asked for and the log says so -- and it is what a slot
        # search blind to the settle buys: on the x6100, 32 ms of 400.
        lead = min(settle, max(0.0, (first - self.now) / self.fs))
        self.leads.append(lead)
        # What this burst spent between the bridge and its own key. The bridge
        # runs to `onair._prekey_lead` in front of the boundary -- the PTT instant
        # plus the measured admission reserve -- so a charge recorded at or after
        # that instant is work done inside the window the whole cycle is arranged
        # to keep empty, and the window the admission guard adjudicates.
        # THE OVERLAP, not the charges that START inside it. A decode that begins
        # two milliseconds before the key instant and runs for twenty is twenty
        # milliseconds of work in front of a keyed transmitter, and counting it as
        # none is a gate that cannot see the thing it was written for.
        keyed_at = (None if at is None else
                    at - onair._prekey_lead(self, round(settle * self.fs)))
        self.prekey.append(
            0 if keyed_at is None else
            sum(min(pos + n, first) - max(pos, keyed_at) for pos, n in self.spent
                if pos + n > keyed_at))
        del self.spent[:]
        self._advance(first + audio.size)
        self.keyed_s = lead + audio.size / self.fs
        # Where the carrier came up, which is the audio less the lead the rig
        # actually got rather than the one it was promised.
        self.keyed_at = first - round(lead * self.fs)
        self.emissions.append((first, first + audio.size))
        if key is not None:
            key(True)
            key(False)
        if self.peer is not None:
            self.peer(self, first + audio.size)
        return first, first + audio.size

    def clock_report(self) -> str:
        return f"bench clock: {self.now} samples, no wall time involved"

    def stream_report(self) -> str:
        return "session stream: not taken -- this session had no sound card"

    def close(self) -> None:
        pass


class _Rig:
    """A rig that keys nothing. No serial device is opened by this file."""

    def __init__(self, *a, **kw):
        self.edges: list[bool] = []
        self.dial: int | None = None
        self.mode: str | None = None
        self.passband: int | None = None
        self.closed = False

    def set_freq(self, hz):
        self.dial = int(hz)
        return True

    def get_freq(self):
        # A working rig, which is what these cycle-timing cases are about. It
        # answered None until 2026-08-15, when that stopped meaning "no opinion"
        # and started meaning "the dial this would key on is unknown" -- shrike
        # refuses on it now, the way besra and sabir already did.
        return self.dial

    def set_mode(self, mode, *, passband=3000):
        self.mode = mode
        self.passband = int(passband)
        return True

    def get_mode(self):
        from hfmodem.core.rxreadiness import RadioMode
        return RadioMode(self.mode, self.passband,
                         f"{self.mode}\n{self.passband}\n")

    def _close(self):
        self.closed = True

    def ptt(self, on):
        self.edges.append(bool(on))
        return True

    def key_failure(self):
        # A rig that keys nothing has nothing to doubt: this file measures
        # where carriers land, not whether a keying channel stood.
        return None

    def stop(self):
        pass


def _answer(shift, cs=pactor1.CS_ACK_A, quiet_after: int | None = None,
            takeover: int | None = None):
    """A station that replies `d` after our carrier drops, in our own shift.

    `quiet_after` is a gateway that answers the call, takes the link, and then
    goes off the air -- the commonest way a real session ends and the scene the
    2026-08-02 runaway happened in. Everything the loop does about a peer that
    is not there then runs every cycle with nothing to interrupt it.

    `takeover` is the 2026-08-13 gateway: at that emission count it answers
    with the CS3-headed changeover packet instead of a codeword -- taking the
    link, so the session holds the IRS side -- and then fades, which leaves
    every IRS cycle sweeping a window that is not decoding.

    "Die Shiftlage ... wird invertiert" applies to the cycle, and within one
    cycle both directions share it -- so the peer's sense is ours, and a session
    whose bursts land in the wrong slot gets its answers in the wrong slot.

    ONCE PER CYCLE, because that is what a PACTOR station is: "PACTOR arbeitet
    als bitsynchrones System mit einem festen Zeitraster" -- it runs a clock and
    transmits on it, rather than replying to whatever it hears. Answering every
    carrier makes a ping-pong the protocol has no room for: our acknowledgement
    of its answer draws another answer inside the same cycle, the FSM answers
    THAT, and the second burst is a whole slot late through no fault of the grid.
    Reactive to WHERE our carrier fell, which is the property the shift and slot
    checks need, and not to how many times it fell.
    """
    def put(bench: _Bench, rf_end: int) -> None:
        if bench.answered:
            return
        bench.answered = True
        n = len(bench.emissions)
        if quiet_after is not None and n > quiet_after:
            return
        if takeover is not None and n >= takeover and (
                bench.emissions[-1][1] - bench.emissions[-1][0] < FS // 2):
            return          # the session answered as IRS: the new ISS has faded
        if takeover is not None and n >= takeover:
            # Repeated until the session's own emissions turn codeword-sized,
            # which is what a real station does with a changeover nobody
            # acknowledged: it sends it again, byte for byte.
            burst = onair._trim_silence(np.asarray(pactor1.breakin_signal(
                b"BK DE K7ABC", 100, invert=bool(shift()),
                lead_s=0, tail_s=0), np.float32))
        else:
            word = (pactor1.CS_SPEED if n == 1 else
                    (cs if n % 2 else pactor1.CS_ACK_B))
            burst = onair._trim_silence(np.asarray(
                pactor1.control_signal(word, invert=bool(shift())), np.float32))
        at = rf_end + bench.d_n
        end = min(at + burst.size, bench.audio.size)
        if end > at:
            bench.audio[at:end] += burst[:end - at]
    return put


def _session(*, cycles: int, hold: int, charge: int, peer: bool,
             decode: bool = False, regrid=None, as_flown: bool = False,
             scan_late: bool = False, scale: bool = False,
             quiet_after: int | None = None, takeover: int | None = None,
             seconds: float = 200.0, d: float = _Bench.D_S,
             before_render: bool = True, answer=None,
             keep_upgrade: bool = False, prekey_n: int = 0,
             extra_argv: tuple[str, ...] = ()) -> dict:
    """One whole `onair` session over the bench, with --transmit armed.

    Armed, because the defect lives in the duplex transmit path and a dry run
    never reaches it. Nothing here touches a device: the rig is a stub whose PTT
    is a list, and the stream is `_Bench`.

    THE DECODE IS CHARGED WHERE IT RUNS, each call at its own measured cost and
    at the point in the cycle the loop makes it. Charging the cycle's overrun
    anywhere else (it was charged inside `_peer_bursts`, in the dead time AFTER
    the carrier) hides the whole question, and hides the fact that a re-grid pays
    the same costs again on every iteration of its own loop.

    `scan_late` moves the frame scan's COST to the anchored read, which runs
    after the bridge -- the placement this file was written against, reproduced
    without touching the loop. The scan itself still runs where it now runs, so
    what the control measures is the clock and nothing else.

    `scale` charges each decode BY THE AUDIO IT IS GIVEN as well, at the measured
    rates. Nothing else in this file can see what that changes: a fixed charge
    makes every window cost the same, so a loop whose window grows with its own
    lateness looks free.

    `before_render=False` isolates the historical post-render backstop: the
    newer recovery must not repair its deliberately missed slot before that
    backstop receives already-rendered PCM. Normal scenes exercise both guards.
    """
    made: list = []
    bench_box: list[_Bench] = []
    aimed: list[tuple[int, int] | None] = []      # (boundary, anchor) per burst
    rendered: list[bool | None] = []              # ...and the shift it was in
    labels: list[str] = []                        # protocol carrier or final ID
    # ...and one row per ATTEMPT, taken after the emission path has finished with
    # it, so a refused burst and a re-aimed one are each their own fact. The
    # three lists above are indexed by attempt and `bench.emissions` by carrier;
    # those are different sequences the moment anything is refused.
    bursts: list[dict] = []

    class _Tx(onair.RadioTx):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

        def _advance_aim(self, **kwargs):
            if not before_render:
                kwargs.pop("before_render", None)
            return super()._advance_aim(**kwargs)

        def _tx(self, audio, what, drive=None, **kwargs):
            # The grid AS IT STOOD when this burst went out. Taken here because
            # the anchor moves -- `reverse` slides it 840 ms at a changeover,
            # `_place` up to half a cycle after a hush -- so the grid read at the
            # end of a session is not the grid any given burst was aimed at, and
            # measuring against it would fabricate an error or hide one.
            aimed.append(None if len(made) < 2
                         else (self.boundary, made[1].anchor,
                               made[1].shift_slot))
            # ...and the polarity the samples in hand are ALREADY IN, read before
            # `_tx` consumes it. The bench peer cannot supply this: it answers in
            # whatever shift we sent, so it agrees with us however wrong we are.
            rendered.append(self._sent_invert)
            labels.append(what)
            if prekey_n and bench_box:
                # THE WORK A CYCLE DOES AFTER ITS LAST READ, charged where the
                # arm paid it: in front of the placement backstop and the
                # admission guard both. See `onair.TX_ADMIT_RESERVE_S`.
                bench_box[0].spend(prekey_n)
            super()._tx(audio, what, drive=drive, **kwargs)
            b = bench_box[0] if bench_box else None
            grid = made[1] if len(made) > 1 else None
            bursts.append({
                "what": what,
                "slot": self.slot,
                "boundary": self.boundary,
                "rx_due": (None if grid is None or self.slot is None
                           else grid.rx_due(self.slot)),
                "refused": bool(self.refused),
                "first": (b.emissions[-1][0]
                          if b and b.emissions and not self.refused else None),
            })

    class _Grid(onair._MasterGrid):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    def _live(*a, **kw):
        put = (answer(lambda: made[0].invert) if answer is not None else
               _answer(lambda: made[0].invert, quiet_after=quiet_after,
                       takeover=takeover) if peer else None)
        b = _Bench(seconds=seconds, peer=put, d=d)
        b.charge, b.always_placeable = charge, as_flown
        if decode:
            b.flush_n, b.upgrade_n = FLUSH_N, UPGRADE_N
            b.scan_n, b.cs_n = ((0, SCAN_N + CS_N) if scan_late
                                else (SCAN_N, CS_N))
        if scale:
            b.feed_rate, b.scan_rate = FEED_RATE, SCAN_RATE
            b.upgrade_rate, b.drain_rate = UPGRADE_RATE, DRAIN_RATE
        bench_box.append(b)
        return b

    _flush, _scan, _cycle = (onair._SessionRx.flush, onair._SessionRx.deep_scan,
                             onair._SessionRx.new_cycle)
    _cs, _up, _feed = (onair._SessionRx.control_signal,
                       onair._SessionRx.upgrade_scan, onair._SessionRx.feed)

    def _spend(fixed: int, rate: float = 0.0, audio: int = 0,
               bulk: bool = False) -> None:
        if not bench_box:
            return
        b = bench_box[0]
        b.spend(fixed + round(audio * rate))
        if bulk and b.armed:
            # The cycle's overrun, and it is one cycle's rather than one call's:
            # the bulk models a cycle whose work walked past its own boundary,
            # which happened once per cycle on the air.
            b.armed = False
            b.spend(b.charge)

    def _charged_flush(self):
        # Capped at FLUSH_CONTEXT_S by `_SessionRx.flush`, so it is one of the
        # two decodes a long window does NOT make dearer.
        _spend(bench_box[0].flush_n if bench_box else 0)
        return _flush(self)

    def _charged_scan(self, audio):
        b = bench_box[0] if bench_box else None
        _spend(b.scan_n if b else 0, b.scan_rate if b else 0.0, audio.size,
               bulk=True)
        return _scan(self, audio)

    def _charged_cs(self, seg, seg_start, at, **kwargs):
        # Anchored: it reads one codeword at one instant, whatever the window
        # around it is long -- the other decode a long window does not charge for.
        _spend(bench_box[0].cs_n if bench_box else 0)
        return _cs(self, seg, seg_start, at, **kwargs)

    def _charged_upgrade(self, audio):
        b = bench_box[0] if bench_box else None
        _spend(b.upgrade_n if b else 0, b.upgrade_rate if b else 0.0, audio.size)
        return _up(self, audio)

    def _charged_feed(self, chunk):
        # The listen loop's own decode, per slice as it arrives. Charged nowhere
        # before this, because with a fixed per-cycle cost there was nothing for
        # it to be proportional TO.
        b = bench_box[0] if bench_box else None
        _spend(0, b.feed_rate if b else 0.0, chunk.size)
        return _feed(self, chunk)

    def _charged_cycle(self):
        if bench_box:
            b = bench_box[0]
            b.armed, b.answered = True, False
        return _cycle(self)

    # THE PEER SPEAKS PACTOR-1 AND NOTHING ELSE, so the link is held there.
    # `_answer` renders control signals and no data frames, which is a station
    # that cannot follow an upgrade -- and a real one says so by answering the
    # offer in PACTOR-1, which `ptc.PtcHost._follow_peer` reads as a
    # contradiction and falls back on. Modelling that round trip would put an ARQ
    # path this file does not measure between the scenes, and it already does:
    # the scene whose cycle fits its slot stays in PACTOR-1 while the scene that
    # overruns collects enough audio to take a second acknowledgement and upgrade,
    # after which the two are no longer the same session being compared.
    _upgrade = ptc.PtcHost.upgrade

    class _Host(onair.PtcHost):
        """A station with something to say for the whole session.

        An ISS whose outbound buffer has drained transmits NOTHING -- not a
        packet, not a control signal (`arq._start_next_packet` returns on an
        empty buffer, where the IRS branch a few lines above sends CS_REQUEST
        every cycle whatever it heard). `--message` reaches only the dry-run
        branch, so the bench link went quiet the moment an acknowledgement
        cleared its one in-flight packet -- and it cleared in the scenes that
        collected the most audio, which is exactly the scenes under test. That
        put the carrier count under measurement at the mercy of how fast the FSM
        was answered, which is not what this file is measuring.
        """

        def tick(self, **kwargs) -> None:
            # Per cycle rather than once, because `on_host_data` is a no-op until
            # the link is CONNECTED and the link comes up mid-window. Three
            # packets' worth against the one a cycle carries, so the buffer never
            # runs dry and the QRT path is never reached.
            self.arq.on_host_data(b"the quick brown fox ")
            super().tick(**kwargs)

    saved = (onair.RadioTx, onair._MasterGrid, onair._LiveInput,
             onair._SessionRx.flush, onair._SessionRx.deep_scan,
             onair._SessionRx.new_cycle, onair.find_device, onair.ota.Rig,
             onair._regrid, onair._save_capture_async, onair.PtcHost,
             onair._SessionRx.control_signal, onair._SessionRx.upgrade_scan,
             onair._SessionRx.feed, onair._HoldBudget)
    if peer and quiet_after is None and takeover is None:
        # This peer ACKs an endlessly replenished payload stream. Successful
        # ACK processing now renews the real idle timeout, so use the real
        # budget's configurable ceiling to bound this geometry-only scene.
        # Idle renewal and the normal 480-slot ceiling have their own tests.
        budget = onair._HoldBudget
        onair._HoldBudget = lambda cycles, **kwargs: budget(cycles, ceiling=hold, **kwargs)
    if not keep_upgrade:
        ptc.PtcHost.upgrade = lambda self, payload_waiting: None
    onair.PtcHost = _Host
    onair.RadioTx, onair._MasterGrid = _Tx, _Grid
    onair._LiveInput = _live
    onair._SessionRx.flush = _charged_flush
    onair._SessionRx.deep_scan = _charged_scan
    onair._SessionRx.new_cycle = _charged_cycle
    onair._SessionRx.control_signal = _charged_cs
    onair._SessionRx.upgrade_scan = _charged_upgrade
    onair._SessionRx.feed = _charged_feed
    onair.find_device = lambda name, kind, required=False: 0
    onair.ota.Rig = _Rig
    # The per-cycle capture write runs on its own thread and outlives the
    # session; nothing here reads the files, and leaving it on races the
    # temporary directory away from under it.
    onair._save_capture_async = lambda *a, **kw: None
    if regrid is not None:
        onair._regrid = regrid
    argv, log = sys.argv, io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        # --serial names a real character device because the arm gate now
        # refuses one that is not there -- the default is a placeholder path,
        # and a session against a placeholder is test_keyed_truth's business.
        sys.argv = ["shrike.onair", "--transmit", "--hold", str(hold),
                    "--max-cycles", str(cycles), "--mycall", "W9SSJ",
                    "--dxcall", "K7ABC", "--dial", "7100000",
                    "--serial", "/dev/null", "--outdir", tmp, *extra_argv]
        try:
            with contextlib.redirect_stdout(log):
                onair.main()
        finally:
            (onair.RadioTx, onair._MasterGrid, onair._LiveInput,
             onair._SessionRx.flush, onair._SessionRx.deep_scan,
             onair._SessionRx.new_cycle, onair.find_device, onair.ota.Rig,
             onair._regrid, onair._save_capture_async, onair.PtcHost,
             onair._SessionRx.control_signal, onair._SessionRx.upgrade_scan,
             onair._SessionRx.feed, onair._HoldBudget) = saved
            ptc.PtcHost.upgrade = _upgrade
            sys.argv = argv
    return {"bench": bench_box[0], "grid": made[1], "tx": made[0],
            "aimed": aimed, "rendered": rendered, "labels": labels,
            "bursts": bursts, "log": log.getvalue()}


def _missed(got: dict) -> list[int]:
    """Samples from each carrier to the SLOT IT WAS AIMED AT.

    The figure the on-air log printed, and the one that decides whether the peer
    is reading: 1456 ms of a 1250 ms slot is a burst in the next slot. The
    connect is excluded -- it goes out before a grid exists, and is the thing the
    anchor is measured from.
    """
    return [int(first - aim[0]) for (first, _e), aim in
            zip(got["bench"].emissions, got["aimed"]) if aim]


def _off_raster(got: dict) -> list[int]:
    """...and signed samples to the nearest boundary of the grid as it stood.

    Both, because neither alone is enough. `_missed` compares a burst to the
    number the session handed its own transmitter, and a loop that moved its aim
    to match would satisfy it; this compares the burst to the raster itself, and
    a burst one whole slot late would satisfy THAT. The anchor is snapshotted per
    burst because it moves -- `reverse` slides it 840 ms at a changeover -- so
    the grid at the end of a session is not the grid every burst was aimed at.
    """
    out = []
    for (first, _e), aim in zip(got["bench"].emissions, got["aimed"]):
        if not aim:
            continue
        err = (first - aim[1]) % SLOT_N
        out.append(int(err - SLOT_N if err > SLOT_N // 2 else err))
    return out


def _landed(got: dict) -> list[tuple[int, bool, bool | None]]:
    """Per carrier: the slot it came up in, the shift that slot calls for, and
    the shift its samples were actually rendered in.

    All three snapshotted per burst, because both the anchor and `shift_slot`
    move during a session -- `reverse` slides one, `align` flips the other -- and
    reading either off the grid at the end would fabricate an error or hide one.
    """
    out = []
    for (first, _e), aim, inv in zip(got["bench"].emissions, got["aimed"],
                                     got["rendered"]):
        if aim is None:
            continue
        slot = int(round((first - aim[1]) / SLOT_N))
        out.append((slot, bool((slot - aim[2]) & 1), inv))
    return out


def _prekey(got: dict) -> list[float]:
    """Per carrier: milliseconds of decode charged between the bridge and the key.

    THE INVARIANT NOTHING ELSE HERE CAN SEE. Every other check reads where the
    carrier came up, and a cycle that spends 65 ms inside the keying settle still
    puts one on a boundary -- the next one, a slot later, which reads as a grid
    working correctly. This reads the window itself: the bridge runs to the key
    instant, so what is charged past it is charged to the rig's PTT lead, and
    past `key_notice` it costs the slot.
    """
    return [n / FS * 1e3 for n, aim in
            zip(got["bench"].prekey, got["aimed"]) if aim]


def _lead_shortfall(got: dict) -> list[float]:
    """Per carrier: milliseconds of the keying settle the rig did not get.

    PTT cannot be asserted in the past either, so a boundary accepted with less
    notice than the settle comes out as a SHORT SETTLE rather than as a late
    burst -- the carrier is exactly where it was asked for and the log says so.
    A slot search blind to the settle can pick a boundary `key_notice` ahead and
    hand an x6100 32 ms of the 400 it is configured for; nothing else in this
    file would notice.

    Charged against the lateness the grid deliberately allowed, because that is
    spent out of the same settle and spending it is the point.
    """
    settle = got["tx"].settle
    return [max(0.0, settle - lead - (first - aim[0]) / FS) * 1e3
            for lead, (first, _e), aim in zip(got["bench"].leads,
                                              got["bench"].emissions,
                                              got["aimed"]) if aim]


def _re_aimed(got: dict) -> int:
    """How many carriers the emission path moved to another slot.

    A whole slot or more, so the tens of milliseconds a cycle's own decode
    spends inside the turnaround are not counted as a re-aim.
    """
    return sum(1 for m in _missed(got) if m >= SLOT_N)


def _placed_well(got: dict) -> list[int]:
    """Carriers that are neither on their own boundary nor a clean re-aim.

    Two ways for a burst to be correctly placed and they are different claims.
    Either the carrier came up ON the boundary it was aimed at -- nothing
    allowed for, which is what the grid is owed once the decode is out of the
    settle -- or the emission path moved it, in which case it must have gone a
    WHOLE, EVEN number of slots, because `shift` is slot parity and the audio was
    rendered before `_tx` ever saw it. Anything else is a burst nobody placed.
    """
    return [m for m in _missed(got)
            if not (m <= 0 or (m > 0 and m % (2 * SLOT_N) == 0))]


def _largest_gap(got: dict) -> int:
    """The longest span of channel the session never handed to a reader.

    Our own carrier does not count -- a keyed transmitter hears nothing -- and
    neither does the keying settle in front of it. Anything else is audio that
    reached the codec and was dropped, which is how the far end's opening
    symbols were once thrown away and the log then recorded that nobody had
    answered.
    """
    bench = got["bench"]
    spans = sorted(bench.taken + [(a, b) for a, b in bench.emissions])
    worst, at = 0, bench.emissions[0][0]
    for a, b in spans:
        worst = max(worst, a - at)
        at = max(at, b)
    return worst


# The two loops the fix has to hold in, and they are different shapes. The setup
# loop calls into a channel that has not answered, so the grid free-runs and the
# hush eventually takes it off the air; the hold loop runs a live link, where the
# peer's answer moves the receive window and a changeover can move the anchor.
# The defect was measured in the second and lives in both.
SCENES = (("calling, no answer", dict(cycles=14, hold=10, peer=False), 4),
          ("a link held, a peer answering", dict(cycles=4, hold=10, peer=True), 2))


def _worst(xs: list[int]) -> float:
    return max((abs(x) for x in xs), default=float("inf")) / FS * 1e3


def the_emission_is_on_the_grid() -> None:
    """The measurement no other check in this repository takes."""
    for name, scene, least in SCENES:
        print(f"\nThe carrier against the grid -- {name}")
        easy = _session(charge=0, **scene)
        check("with the cycle inside its slot, every carrier comes up on the "
              "boundary it was aimed at",
              len(_missed(easy)) >= least and _worst(_missed(easy)) == 0
              and _worst(_off_raster(easy)) == 0,
              f"{len(_missed(easy))} emissions")

        # THE MEASURED COST, ON ITS OWN. No overrun at all -- just the flush, the
        # frame scan and the anchored read at what they cost on this machine,
        # charged where they run. This is the FT-891 as it stands, and what the
        # grid owes here is THE CADENCE: the protocol is one packet every 1.25 s
        # in every cycle, and a rule that hands a slot back for the cost every
        # cycle pays turns that into 2.5 s, which stalls a link on its own
        # (§3, §8.1).
        real = _session(charge=0, decode=True, **scene)
        check(f"...the {(FLUSH_N + SCAN_N + CS_N) / FS * 1e3:.0f} ms the cycle's "
              f"own decode really costs takes nothing from the key: every "
              f"carrier comes up on the boundary it was aimed at",
              len(_missed(real)) >= least and not _placed_well(real),
              f"{len(_missed(real))} emissions, {_re_aimed(real)} re-aimed, "
              f"{real['log'].count(onair.SLOT_GONE)} slots given up")
        # ...and the window that cost is kept OUT of, which is the invariant the
        # placement was violating and which nothing here could see. The anchored
        # read is what remains: it wants the block the bridge is still
        # collecting, and it measures 0.3 ms against the 8 ms the FT-891's settle
        # leaves past `key_notice`.
        check("...and the pre-key window holds nothing but the anchored "
              "control-signal read",
              max(_prekey(real), default=1e9) <= CS_N / FS * 1e3,
              f"worst {max(_prekey(real), default=float('nan')):.1f} ms charged "
              f"inside the settle, against {CS_N / FS * 1e3:.1f} for the read")
        check("...and every one the emission path had to move said so",
              _re_aimed(real) == real["log"].count(onair.LATE_KEY),
              f"{_re_aimed(real)} moved, "
              f"{real['log'].count(onair.LATE_KEY)} alarms")
        # THE MEDIAN, not every increment. At the top of the turnaround band the
        # schedule has 15 ms in hand and a cycle that jitters costs a slot, which
        # is the grid working; what must not happen is the SYSTEMATIC two, which
        # is what handing a slot back for the cost every cycle pays produces --
        # every increment was 2 before the decode had a deadline of its own.
        inc = [b - a for a, b in zip(real["tx"].slots_used,
                                     real["tx"].slots_used[1:])]
        check("...and the rate it flies is the protocol's own one slot, not the "
              "two that giving a slot back every cycle would make",
              bool(inc) and float(np.median(inc)) == 1.0,
              f"median {np.median(inc):g}, increments {inc}")

        # THE COUNTEREXAMPLE FOR THE WINDOW, planted the same way the raster's
        # is: the same session, the same total cost, the frame scan charged after
        # the bridge instead of in front of it. That is where it stood, and it is
        # what a tolerance for lateness was added to absorb. The grid answers it
        # by handing the slot back -- correctly, and every cycle, which is the
        # 2.5 s cadence that stalls a link.
        blind = _session(charge=0, decode=True, scan_late=True, **scene)
        binc = [b - a for a, b in zip(blind["tx"].slots_used,
                                      blind["tx"].slots_used[1:])]
        if scene["peer"]:
            check("NEGATIVE CONTROL: with the same decode charged after the "
                  "bridge, the grid gives the slot back for it every cycle",
                  bool(binc) and float(np.median(binc)) == 2.0
                  and onair.SLOT_GONE in blind["log"],
                  f"median {np.median(binc):g}, "
                  f"{blind['log'].count(onair.SLOT_GONE)} slots given up")
            # ...and the instrument is shown the thing it is meant to read, which
            # the control above cannot do: a slot handed back never becomes a
            # carrier, so the window it filled is not one `_prekey` has a burst to
            # report it on. With the stream saying every boundary is reachable --
            # the loop as it flew -- every cycle keys anyway, and the window is
            # measurably full.
            seen = _session(charge=0, decode=True, scan_late=True, as_flown=True,
                            **scene)
            # ...on the carriers whose cycle had a receive instant to read. The
            # anchored read is aimed at our own data end plus `d`, so a cycle with
            # no transmission of ours behind it makes no read and is charged
            # nothing for it -- the session's first, and the one a hush ends in.
            # See `_MasterGrid.rx_due_in`.
            filled = [x for x in _prekey(seen) if x]
            check("...and the window it fills is measurable, not merely argued",
                  len(filled) >= least and min(filled, default=0.0)
                  >= SCAN_N / FS * 1e3,
                  f"least {min(filled, default=float('nan')):.1f} ms charged "
                  f"inside the settle over {len(filled)} of "
                  f"{len(_prekey(seen))} carriers")
        else:
            # THERE IS NO ANCHORED READ TO CHARGE IT ON IN THIS SCENE, and
            # `scan_late` moves the cost onto that read. A station that has not
            # linked does not make one: `_SessionRx.control_signal` refuses at its
            # first line while the state is CONNECTING, and always has -- the
            # calling phase reads with `p1rx.acquire_control_signal` instead, over
            # the acquisition window and in the dead time AFTER the carrier drops,
            # where it costs the slot nothing. `arm-post-v31-B-assessed-40-kb5lzk
            # -20260915T234606Z` is that path connecting: `RX candidate CS1 at
            # d = 55 ms` on cycle 4, the corroborated CS1 delivered on cycle 5,
            # CONNECTING -> CONNECTED.
            #
            # So the control here is the complement, and it is the stronger of the
            # two: the cost the linked scene above pays before its key is one this
            # phase cannot be made to pay at all, whatever it is charged.
            charged = [x for x in _prekey(blind) if x]
            check("NEGATIVE CONTROL: the calling phase makes no anchored read, "
                  "so the same decode never reaches its pre-key window and the "
                  "cadence stays on the protocol's one slot",
                  bool(binc) and float(np.median(binc)) == 1.0
                  and onair.SLOT_GONE not in blind["log"],
                  f"median {np.median(binc):g}, "
                  f"{blind['log'].count(onair.SLOT_GONE)} slots given up")
            # ...and it is the READ that is absent rather than the charge: a cycle
            # whose burst detector found an onset still reads a codeword there
            # (`_onset_control_signal`, one cycle late and on the onsets the cycle
            # already has), and that read is charged in full where it runs.
            check("...and what little is charged there is one whole read of "
                  "`_onset_control_signal`'s, not a fraction of one",
                  len(charged) < least
                  and all(abs(x - (SCAN_N + CS_N) / FS * 1e3) < 1e-6
                          for x in charged),
                  f"{len(charged)} of {len(_prekey(blind))} carriers charged, "
                  f"{sorted(set(charged))} ms")

        got = _session(charge=OVERRUN_N, decode=True, **scene)
        check(f"...and still lands on the raster when the cycle overruns its "
              f"slot by {OVERRUN_N / FS * 1e3:.0f} ms, so the grid was not moved "
              f"to meet it", len(_missed(got)) >= least
              and _worst(_off_raster(got)) == 0,
              f"{len(_missed(got))} emissions, worst "
              f"{_worst(_off_raster(got)):.1f} ms from the nearest boundary")
        # A burst can still have to move: the FSM answers a decode from
        # `on_rx_event`, which reaches `RadioTx._tx` having never been past the
        # loop's check, and by then the overrun has already been spent. What it
        # may NOT do is move quietly, or move by an odd number of slots -- the
        # shift is slot parity and the audio is already rendered.
        steps = sorted({m / SLOT_N for m in _missed(got)})
        check("...and every burst that had to move went a whole, even number of "
              "slots -- the step that keeps the shift it is rendered in",
              all(s == int(s) and int(s) % 2 == 0 for s in steps),
              f"steps {[f'{s:g}' for s in steps]} slots")
        check("...and said so every time it moved, rather than moving quietly",
              _re_aimed(got) == got["log"].count(onair.LATE_KEY),
              f"{_re_aimed(got)} moved, "
              f"{got['log'].count(onair.LATE_KEY)} alarms")
        check("...and the overrun really did cost slots, so that was not simply "
              "an easy run", onair.SLOT_GONE in got["log"],
              f"{got['log'].count(onair.SLOT_GONE)} slots given up")
        check("...and the re-grid always converged, so the loop never handed the "
              "burst on with nowhere left to put it",
              "REGRID GAVE UP" not in got["log"])
        check("...and every burst still got the keying settle its rig is set "
              "for, less only the lateness the grid allowed",
              max(_lead_shortfall(got), default=0.0) < 1e-6,
              f"worst {max(_lead_shortfall(got), default=0.0):.1f} ms short of "
              f"{got['tx'].settle * 1e3:.0f}")

        # The backstop, on its own, and it guarantees strictly less. `_regrid`
        # answers for the transmissions the loop schedules, and gives the slot
        # back before the audio for it is collected; the FSM can also key
        # straight out of a decode, and that reaches `RadioTx._tx` having never
        # been past the loop's check. All the backstop can still do there is put
        # the burst on a LATER boundary -- on the raster, and in the shift the
        # samples it is holding were rendered in, which is the difference between
        # a late packet and a collision.
        held = _session(charge=OVERRUN_N, decode=True, regrid=_no_regrid,
                        before_render=False,
                        **scene)
        check("with the loop's own check removed, the emission path still holds "
              "the carrier to the raster and says that it had to",
              len(_missed(held)) >= least and _worst(_off_raster(held)) == 0
              and onair.LATE_KEY in held["log"],
              f"{held['log'].count(onair.LATE_KEY)} late keys, worst "
              f"{_worst(_off_raster(held)):.1f} ms off the raster, "
              f"{_worst(_missed(held)):.0f} ms from the slot aimed at")

        # THE COUNTEREXAMPLE, planted rather than assumed. Every check above
        # would pass on a gate that is not looking, so the gate is shown the
        # defect it was built for: the same bench and the same overrun, with the
        # session put back the way it flew -- no re-aim, and a stream that says
        # every boundary is reachable, which is what the loop assumed.
        flown = _session(charge=OVERRUN_N, decode=True, regrid=_no_regrid,
                         before_render=False,
                         as_flown=True, **scene)
        check("NEGATIVE CONTROL: as it flew, the carrier follows the receive "
              "window instead of the grid, by most of a slot",
              len(_missed(flown)) >= least
              and _worst(_missed(flown)) > 0.5 * SLOT_N / FS * 1e3,
              f"worst {_worst(_missed(flown)):.0f} ms of a "
              f"{SLOT_N / FS * 1e3:.0f} ms slot, over {len(_missed(flown))} "
              f"emissions")
        check("...and it is off the raster too, so this is not one clean slot "
              "late", _worst(_off_raster(flown)) > 0,
              f"worst {_worst(_off_raster(flown)):.0f} ms from the nearest "
              f"boundary")


def the_carrier_keeps_the_shift_it_was_rendered_in() -> None:
    """A burst goes out in the polarity its own samples carry, or not at all.

    `aim` sets where a transmission keys and which shift it keys in together,
    because they are one fact about one cycle. But every renderer has consumed
    that answer by the time the audio reaches `RadioTx._tx` -- `_flip` is
    evaluated while the burst is being built -- so a re-aim inside the emission
    path moves the carrier onto a correct boundary carrying the OPPOSITE shift.
    `shift(slot)` is slot parity, so advancing one slot flips it every time.

    Unreadable to a station counting cycles, and unreadable in a way that looks
    exactly like a dead band. The bench peer cannot see it: it answers in
    whatever shift we sent, so it agrees with us however wrong we are. This asks
    the GRID, per burst, against the anchor and the shift epoch as they stood.
    """
    print("\nThe shift on the air, against the shift the slot calls for")
    for name, scene, _least in SCENES:
        got = _session(charge=OVERRUN_N, decode=True, **scene)
        bad = [(s, w, i) for s, w, i in _landed(got) if i is not None and w != i]
        check(f"every carrier is in the shift its slot calls for -- {name}",
              not bad, f"{len(bad)} of {len(_landed(got))} in the wrong shift")

        # The path the backstop exists for is exactly the one that gets this
        # wrong: a burst the FSM keys straight out of a decode has never been
        # past the loop's check, and re-aiming it freely inverts it.
        #
        # THE OVERRUN IS THE SMALL ONE HERE, and that is the whole point. Lose a
        # slot by 1.4 s and the next boundary is unreachable anyway, so the
        # search steps two slots for reasons that have nothing to do with the
        # shift and a backstop that ignores polarity passes by luck. Lose it by
        # 600 ms and the next boundary is available on every other ground.
        held = _session(charge=NUDGE_N, decode=True, regrid=_no_regrid,
                        before_render=False,
                        **scene)
        moved = [(s, w, i) for s, w, i in _landed(held) if i is not None]
        bad = [r for r in moved if r[1] != r[2]]
        check(f"...and still is with the loop's check removed and the next slot "
              f"free to take, so it is the polarity holding it -- {name}",
              not bad and onair.LATE_KEY in held["log"],
              f"{len(bad)} of {len(moved)} in the wrong shift over "
              f"{held['log'].count(onair.LATE_KEY)} late keys")


def _no_regrid(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n):
    """The loop as it flew on 2026-08-01: whatever the window did, key anyway."""
    return slot, seg, seg_start


_REGRID = onair._regrid


def _fed_regrid(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n):
    """The re-grid as it flew on 2026-08-02: the recovered slot is rolling-decoded.

    The same function, with one number put back: `_regrid` asks for its window to
    be HELD, and this asks for it to be FED, which is what the loop did. Nothing
    else about the re-grid changes, so what the control measures is the cost of
    that decode and nothing else.
    """
    listen = onair._listen_until_answer
    onair._listen_until_answer = (
        lambda live_, want, host_, sessrx_, feed_n, *a, **kw:
        listen(live_, want, host_, sessrx_, 10 ** 9, *a, **kw))
    try:
        return _REGRID(live, raster, tx, host, sessrx, slot, seg, seg_start,
                       settle_n)
    finally:
        onair._listen_until_answer = listen


def _deaf_regrid(live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n):
    """The cheap fix: hand the slot back, then sleep through it.

    `flush_to` IS the sleeping, and it has to be written out: a stream nobody
    reads keeps filling its queue, and the next thing to touch it starts from
    wherever it is told to. Without it this control would be rescued by the
    emission path, which reads its own wait out -- and a control that the code
    under test can rescue is not measuring anything.
    """
    while live.clamp_late(tx.boundary):
        slot += 1
        tx.aim(raster, slot)
    live.flush_to(tx.boundary - settle_n)
    return slot, seg, seg_start


def the_recovered_slot_is_listened_through() -> None:
    """A slot handed back to the grid is spent on the channel, not slept through.

    The second-order failure, and it has happened once already in another form:
    dropping the capture queue after a transmission threw away the far end's
    opening symbols, after which our own log recorded that nobody had answered.
    A slot given back is a whole cycle of the peer's channel and `flush_to` would
    take every sample of it.
    """
    print("\nThe channel across a slot the grid took back")
    scene = dict(cycles=4, hold=10, peer=True)
    got = _session(charge=OVERRUN_N, decode=True, **scene)
    # The keying settle is ours by construction -- PTT is up and the receiver is
    # muted -- and a block is the granularity the codec delivers in. Everything
    # past that is channel that reached the converter and was thrown away.
    floor = round(got["tx"].settle * FS) + 2 * got["bench"]._blk
    check("nothing beyond the keying settle is dropped from the stream",
          _largest_gap(got) <= floor,
          f"{_largest_gap(got) / FS * 1e3:.0f} ms dropped, against "
          f"{floor / FS * 1e3:.0f} allowed")

    deaf = _session(charge=OVERRUN_N, decode=True, regrid=_deaf_regrid,
                    **scene)
    check("NEGATIVE CONTROL: re-aiming and then waiting it out loses the "
          "channel across the recovered slot", _largest_gap(deaf) > floor,
          f"{_largest_gap(deaf) / FS * 1e3:.0f} ms dropped")


def _cs_carriers(got: dict) -> list[tuple[int, float]]:
    """(slot, prekey ms) of every control-signal-length carrier.

    A session that placed the call transmits packets and the connect; the only
    codeword-sized bursts it can key are the IRS's answers after a reversal, so
    length is the role, read off the emissions themselves.
    """
    return [(slot_, pre / FS * 1e3)
            for (first, end), (slot_, _dur), pre
            in zip(got["bench"].emissions, got["tx"].keyed,
                   got["bench"].prekey)
            if end - first < FS // 2]


def the_reversed_link_answers_out_of_channel_time() -> None:
    """An IRS cycle keys its answer without spending the frame sweep in the settle.

    THE 2026-08-13 SESSION, from the inside. As IRS against a real gateway
    whose packets were not resolving, the pre-key frame scan swept the whole
    window every cycle -- 20.3 ms measured on an empty channel, against the
    8 ms the keying settle leaves past `key_notice` -- so every cycle reached
    its key 11-15 ms late and four slots in 23 cycles went to `SLOT GONE`. On
    a live link every lost slot is a missing acknowledgement, and a lost slot
    per four cycles is a 25% ack outage: more than enough to stop a gateway's
    counter advancing.

    The scene is the failure's own shape: the peer answers the call, takes the
    link with the CS3-headed changeover packet, and fades. What the grid owes
    from there is an answer in EVERY cycle -- the repeat request is the
    reverse channel of a stalled link -- with nothing but the anchored read
    spent between the bridge and the key. The one cycle allowed to overrun is
    the stall's onset, where the sweep was armed by a frame that did decode
    and met a window with nothing in it.
    """
    print("\nThe IRS cycles, against a peer that took the link and faded")
    scene = dict(cycles=7, hold=16, peer=True, takeover=8)
    got = _session(charge=0, decode=True, **scene)
    rev = got["log"].find("-> IRS")
    check("the peer's break-in reversed the link", rev >= 0)
    irs = _cs_carriers(got)
    check("...and the reversed link kept answering through the fade",
          len(irs) >= 4, f"{len(irs)} codeword carriers keyed as IRS")
    # UP TO THE GOODBYE, which is where the answering stops. A faded link is
    # signed off rather than abandoned (`arq._give_up`), and every cycle of that
    # teardown changes role -- the peer breaks in, we take the link back to
    # re-send the QRT -- so it spends a slot apiece by design. The cadence this
    # check is about is the IRS's, and it ends where the QRT starts.
    answered = got["log"][max(rev, 0):]
    bye = answered.find("-> QRT")
    gone = answered[:bye if bye >= 0 else None].count(onair.SLOT_GONE)
    check("at most the stall-onset cycle loses its slot; the fade is answered "
          "in every cycle after it", 0 <= gone <= 1,
          f"{gone} slots given up after the reversal")
    hot = [f"{p:.1f}" for _s, p in irs[1:] if p > CS_N / FS * 1e3 + 0.05]
    check("...and the pre-key window of every IRS carrier holds nothing but "
          "the anchored read", not hot,
          f"{len(hot)} carriers charged {hot} ms inside the settle")
    # THE CHANGEOVER CYCLE IS THE ONE EXCEPTION AND IT IS BOUNDED, not exempt.
    # It reads the packet behind the CS3 head, which at this turnaround ends
    # 15 ms in front of the key, so the scan behind it necessarily runs into the
    # settle -- and may only run as far into it as `key_notice` leaves, which is
    # what decides whether the carrier still comes up on the boundary. See
    # `the_changeover_cycle_keys_the_slot_the_rotation_bought`.
    room = (got["tx"].settle * FS - got["bench"].key_notice) / FS * 1e3
    check("...and the changeover cycle's own read of the peer's packet stays "
          "inside what `key_notice` leaves of the settle", irs[0][1] < room,
          f"{irs[0][1]:.1f} ms of {room:.1f}")
    inc = [b - a for (a, _), (b, _) in zip(irs, irs[1:])]
    check("...and the answers fly at the protocol's own one slot",
          bool(inc) and float(np.median(inc)) == 1.0,
          f"median {np.median(inc):g}, increments {inc}")
    # A SESSION THAT CHANGES DIRECTION IS WHERE A BURST STOPS STANDING FOR A
    # CYCLE, so the summary's two percentages are asked here. This one keys
    # 960 ms packets before the reversal and 120 ms codewords after it, and no
    # arithmetic on the cycle count reaches the second figure from the first.
    lens = sorted({round(d, 2) for _, d in got["tx"].keyed})
    line = next((ln for ln in got["log"].splitlines()
                 if ln.startswith("keyed ")), "")
    pct = [int(x) for x in re.findall(r"\((\d+)%", line)] or [0, 0]
    check("the summary counts cycles and air separately across a changeover",
          len(lens) > 1 and len(pct) == 2 and pct[0] != pct[1],
          f"burst lengths {lens}, {line}")

    # THE COUNTEREXAMPLE, planted the file's own way: the same session with
    # the sweep's cost charged after the bridge -- the placement that flew on
    # 2026-08-13 -- and every swept IRS cycle hands its slot back for it.
    flown = _session(charge=0, decode=True, scan_late=True, **scene)
    frev = flown["log"].find("-> IRS")
    fgone = flown["log"][max(frev, 0):].count(onair.SLOT_GONE)
    check("NEGATIVE CONTROL: with the sweep charged after the bridge, the "
          "reversed link loses a slot per cycle -- the ack outage as flown",
          frev >= 0 and fgone >= 3, f"{fgone} slots given up after the reversal")


#: VE1YZ's turnaround on 7096500, 2026-09-02: the arm tracked `d 100.5 ms` at
#: the cycle before the changeover and the whole session inside 99-108. Set at
#: the gap the bench renders, which the edge statistic then reads 5 ms wide.
VE1YZ_D_S = 0.0955


def _first_irs_answer(got: dict) -> tuple[float, float]:
    """The first codeword keyed after the link reverses: milliseconds from the
    peer's changeover onset to our carrier, and from our carrier to the boundary
    it was aimed at.

    The onset is where the bench PUT the peer's burst -- `d` past our own carrier
    dropping -- rather than anything the session reported about it, so the first
    figure is a distance on the air. The session's own `[timing]` line is not:
    `send_p1_cs` prints it while the codeword is being rendered, so it reads the
    read position rather than the carrier, and `_tx` can still move the burst
    after it. On 2026-08-29 it said 0.903 s INSIDE over a burst the emission path
    then took two slots on.
    """
    b = got["bench"]
    for i, (first, end) in enumerate(b.emissions):
        if i and end - first < FS // 2:
            onset = b.emissions[i - 1][1] + b.d_n
            aim = got["aimed"][i]
            return ((first - onset) / FS * 1e3,
                    (first - aim[0]) / FS * 1e3 if aim else float("nan"))
    return float("nan"), float("nan")


def the_changeover_cycle_keys_the_slot_the_rotation_bought() -> None:
    """The first answer to a peer's changeover goes out inside its own cycle.

    `reverse` slides the transmit anchor 840 ms, which is where the station
    becoming IRS owes its codeword: a codeword and a turnaround short of a whole
    cycle past the peer's changeover onset, on the instant that peer's read is
    latched to. Reaching it means reading the packet behind the CS3 head first,
    and that read ran to the key instant itself -- so the frame scan behind it
    came out of the keying settle, against the 8 ms the settle holds past
    `key_notice`, and the cycle lost its slot.

    MEASURED, six cycles a decoded CS3 head reversed at three gateways, and the
    four that overran did it by `scan + key_notice - settle` exactly: VE1YZ
    2026-09-02 at +6.6 and +8.6 ms, where the arm's only two `SLOT ... IS GONE`
    of 141 cycles are its two changeovers and the one CS1 it ever keyed to one
    went out 2.136 s after the peer's onset, in no slot the peer could read it
    in; WS8EOC 2026-08-29 at +13.8 and +15.7, where the emission path took the
    burst two slots on for its shift. KB5LZK 2026-08-22 is the pair that did
    not, and it says what the read costs: at a 78.9 ms turnaround the packet
    ends far enough in front of the key that the read stops on the packet
    instead, and both changeovers there keyed +1.9 and +0.0 ms from the
    boundary. The read is capped `PREKEY_RESERVE_S` short of the key now, and it
    is the tail guard that gives it up rather than the packet.
    """
    print("\nThe changeover cycle, against the slot the rotation bought")
    scene = dict(cycles=7, hold=16, peer=True, takeover=8, d=VE1YZ_D_S)
    got = _session(charge=0, decode=True, **scene)
    log = got["log"]
    rev = log.find("-> IRS")
    check("the peer's break-in reversed the link", rev >= 0)
    cycle = log[max(rev, 0):log.find("[ack]", max(rev, 0))]
    check("...and the cycle that reversed it kept its own slot",
          onair.SLOT_GONE not in cycle and onair.LATE_KEY not in cycle,
          cycle.strip().splitlines()[-1] if cycle.strip() else "")
    gap, off = _first_irs_answer(got)
    # A codeword and a turnaround short of a whole cycle past the peer's onset,
    # which is the rotation seen from the far end. Anything a slot on reads as
    # 1130 ms more.
    due = (SLOT_N - round(spec.P1_CS_S * FS)) / FS * 1e3 - VE1YZ_D_S * 1e3
    check("...so the answer keyed INSIDE the 1.25 s cycle the peer's changeover "
          "opened", abs(gap - due) < 1.0,
          f"{gap:.0f} ms after the peer's onset, against {due:.0f}")
    check("...on the boundary the rotation moved, not a slot on",
          abs(off) < 5.0, f"{off:+.1f} ms off the aimed boundary")

    flown = onair.PREKEY_RESERVE_S
    try:
        # NEGATIVE CONTROL, the file's own way: the reserve at zero is the read
        # as it flew, running to the key instant, with the packet's decode still
        # in front of the carrier. Disable the later pre-render recovery too,
        # so the miss reaches the historical post-render alarm being checked.
        onair.PREKEY_RESERVE_S = 0.0
        was = _session(charge=0, decode=True, before_render=False, **scene)
    finally:
        onair.PREKEY_RESERVE_S = flown
    wrev = was["log"].find("-> IRS")
    wcycle = was["log"][max(wrev, 0):was["log"].find("[ack]", max(wrev, 0))]
    wgap, _woff = _first_irs_answer(was)
    check("NEGATIVE CONTROL: read to the key instant, the changeover cycle "
          "hands its slot back and the answer misses the cycle",
          wrev >= 0 and (onair.SLOT_GONE in wcycle or onair.LATE_KEY in wcycle),
          f"{wgap:.0f} ms after the peer's onset")


def the_regrid_window_keeps_its_origin_across_its_own_key() -> None:
    """A key inside the recovered slot must not corrupt the window's origin.

    `_regrid`'s recovered slot is scanned before it is bridged, and a decode in
    it reaches the FSM, which keys: `RadioTx._tx` flushes the capture queue to
    the end of our own burst and moves the read position with it. An origin for
    the recovered audio computed afterwards, by subtracting sizes from that
    position, lands inside our own carrier -- and everything downstream that
    takes an offset from `seg_start` (the capture sidecar's end sample, the
    onset fold) is then a whole burst out, on exactly the cycles a post-mortem
    reads them for.

    The stream's audio here is the sample index itself, so the claim is asked
    of the content rather than of the bookkeeping: whatever window `_regrid`
    hands back, sample `k` of it must carry the value `seg_start + k`. The FSM
    is a stub with one behaviour -- answer the first frame scan -- but the
    transmission it keys goes through the production `RadioTx._tx`, backstop,
    take, flush and all.
    """
    print("\nthe re-grid's window against a transmission of its own")
    settle_n = round(0.04 * FS)
    raster = onair._MasterGrid(anchor=0, slot_n=SLOT_N,
                               offset_n=round(0.185 * FS),
                               packet_n=round(spec.P1_PACKET_S * FS),
                               cs_n=round(spec.P1_CS_S * FS),
                               d_max_n=round(0.13 * FS))
    bench = _Bench(seconds=60.0)
    bench.audio = np.arange(bench.audio.size, dtype=np.float32)

    class _Arq:
        state = State.CONNECTED

    class _Host:
        protocol = Protocol.PACTOR1
        arq = _Arq()

    class _Rx:
        """The session receiver, reduced to the one move that matters here:
        the first deep scan answers, like the FSM keying from `on_rx_event`."""

        def __init__(self, tx):
            self.tx, self.keyed = tx, False

        def bridge(self, chunk):
            pass

        def flush(self):
            pass

        def skip(self, seconds):
            pass

        def deep_scan(self, audio):
            if not self.keyed:
                self.keyed = True
                self.tx.send_p1_cs(0)

    with tempfile.TemporaryDirectory() as tmp:
        tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=Path(tmp),
                           settle=0.04)
        rx = _Rx(tx)
        tx.live, tx.sessrx = bench, rx
        tx.aim(raster, 1)
        # The cycle as `_regrid` finds it: a window read to 100 ms past its own
        # boundary, so the slot is demonstrably gone.
        seg = bench.read(raster.boundary(1) + round(0.10 * FS))
        with contextlib.redirect_stdout(io.StringIO()):
            slot, out, out_start = onair._regrid(
                bench, raster, tx, _Host(), rx, 1, seg, 0, settle_n)
    check("the recovered slot's scan really keyed, through the production "
          "transmit path", len(bench.emissions) == 1 and bool(tx.slots_used),
          f"{len(bench.emissions)} carriers, slots {tx.slots_used}")
    check("the window handed back is the stream where it says it is: sample k "
          "carries the value seg_start + k",
          out.size > 0 and np.array_equal(
              out, np.arange(out_start, out_start + out.size,
                             dtype=np.float32)),
          f"seg_start {out_start}, first value {out[0] if out.size else '-'}, "
          f"size {out.size}")
    check("...and it ends before our own carrier, not inside it",
          out_start + out.size <= bench.emissions[0][0],
          f"window ends {out_start + out.size}, carrier up at "
          f"{bench.emissions[0][0]}")


def the_decode_does_not_choose_the_window() -> None:
    """A decode that costs more than the audio it reads must not set its own size.

    THE 2026-08-02 FAILURE, and every check above passed while it was live. The
    session was calling a gateway on a quiet 30 m channel with nothing answering:

        SLOT 18 IS GONE  -- overran its own key instant by   +1399.8 ms
        SLOT 20 IS GONE  -- overran its own key instant by  +12835.8 ms
        SLOT 31 IS GONE  -- overran its own key instant by  +17517.8 ms
        !! REGRID GAVE UP after 3 tries
        hold 8 slot 46 ... captured 59.848 s, off-grid +23761.2 ms

    and it ran 18 -> 20 -> 31 -> 46 -> 66 -> 70 -> 103 -> ... -> 274, with each
    cycle's capture growing 2.456 -> 59.848 -> 100.413 -> 153.726 s. Every burst
    that keyed came up +0.0 and +1.8 ms from its boundary, which is what the rest
    of this file measures and why none of it noticed: the emission was exactly
    right, and the schedule around it was taking minutes to the slot.

    THE CAPTURE IS NOT THE WINDOW, and reading it as one sends you after the
    wrong quantity. No listening window in that session was longer than about two
    slots -- the slot search will not aim further than the next boundary it can
    place a carrier on. What grew is the audio the cycle SWEPT UP behind its
    window, because a re-grid iteration that spent twelve seconds decoding one
    slot then found twelve seconds of channel waiting in the queue. The listen is
    the cost; the capture is the receipt.

    WHAT MAKES IT REACHABLE HERE is the cost model and the peer. A fixed charge
    per cycle makes a longer window free, so the loop that reads more when it is
    late looks free too; charged per sample at the measured rates (`FEED_RATE`)
    the window pays for itself, and above one it cannot. And the peer has to
    STOP: a station that answers keeps the re-grid honest by ending the cycle,
    where one that has gone off the air lets it run every cycle unopposed, which
    is the difference between the sessions this file already ran and that one.

    Both halves are asked. That the loop stays bounded, and that it stays bounded
    without going deaf -- the whole reason `_regrid` listens through a recovered
    slot rather than sleeping it out is that dropping the channel there once
    threw away the far end's opening symbols and then reported that nobody had
    answered. Holding audio is not dropping it.
    """
    print("\nThe cycle against a decode that costs more than the channel")
    scene = dict(cycles=4, hold=10, peer=True, decode=True, quiet_after=3,
                 scale=True, charge=0)
    got = _session(**scene)
    keyed = [s for s, _w, _i in _landed(got)]
    inc = [b - a for a, b in zip(keyed, keyed[1:])]
    # A BOUND, not the protocol's own cadence. At the measured rates a cycle's
    # own listening window costs more than it lasts, so this machine cannot hold
    # the one-slot raster on this material and no rule here can make it -- what
    # it must do is degrade to a fixed multiple and stay there. Four slots is 5 s
    # to the packet, which is a slow link; forty is the session that flew.
    check("with the decode charged by the audio it reads, the cadence stays "
          "bounded", bool(inc) and max(inc) <= 4,
          f"worst {max(inc, default=0)} slots between carriers, increments {inc}")
    # ...measured on the spans actually handed to a reader, which is the other
    # half of the same fact. A window runs from the read position to the next
    # boundary a carrier can be placed on, so a reader standing behind its own
    # converter finds the arrears in front of it and pays to read them too. Two
    # slots is what the slot search can legitimately reach past.
    widest = max((b - a) / FS for a, b in got["bench"].taken)
    check("...and no window handed to a reader outgrows the raster",
          widest <= 2 * SLOT_N / FS,
          f"widest {widest:.2f} s against {2 * SLOT_N / FS:.2f}")
    check("...and the re-grid always converged", "REGRID GAVE UP" not in got["log"])
    # The settle, the measured pre-key reserve in front of it
    # (`onair.TX_ADMIT_RESERVE_S` -- audio the cycle deliberately stops short of
    # so its own admission check is taken against a budget somebody measured),
    # and two blocks of quantisation.
    floor = (onair._prekey_lead(got["bench"], round(got["tx"].settle * FS))
             + 2 * got["bench"]._blk)
    check("...and it is not bounded by going deaf: nothing beyond the keying "
          "settle is dropped from the stream", _largest_gap(got) <= floor,
          f"{_largest_gap(got) / FS * 1e3:.0f} ms dropped, against "
          f"{floor / FS * 1e3:.0f} allowed")
    # THE PTT LEAD, which the same session lost 11 ms of on every burst it
    # managed to key -- "PTT LEAD ERODED to 29 ms of the 40 this rig is set for".
    # A drain is charged here, so the allowance is the anchored read's own 0.3 ms
    # rather than zero: `_Bench.read` hands a whole span over at once, where the
    # live one blocks a block at a time and spreads the same cost across the wait
    # in front of it. What the gate can say is that nothing of the SIZE the air
    # reported gets inside the settle.
    shortfalls = _lead_shortfall(got)
    named = [(label, aim) for label, aim in zip(got["labels"], got["aimed"])
             if aim is not None]
    protocol = [n for n, (label, _aim) in zip(shortfalls, named)
                if not label.startswith("ID ")]
    check("...and every protocol burst still got the settle its rig is set for",
          max(protocol, default=0.0) <= CS_N / FS * 1e3,
          f"worst {max(protocol, default=0.0):.2f} ms short of "
          f"{got['tx'].settle * 1e3:.0f}, against 11 on the air")

    # The final CW ID also uses the guarded transmit path, after the protocol
    # loop stops. Its final read is not an anchored CS decode: this bench
    # conservatively charges the WHOLE drain after the blocking read returns.
    # The measured 66117-sample read costs15 samples (0.3125 ms), not CS_N=14.
    # Account that exact charge, without widening the protocol allowance or
    # making an arbitrarily long post-wait drain pass the physical lead check.
    ids = []
    for short, (label, aim) in zip(shortfalls, named):
        if not label.startswith("ID "):
            continue
        key_at = aim[0] - onair._prekey_lead(got["bench"],
                                            round(got["tx"].settle * FS))
        reads = [(b - a, cost) for a, b, at, cost in got["bench"].drained
                 if at == key_at]
        ids.append((round(short * FS / 1e3), reads))
    check("...and the final ID's drain is bounded by two slots of audio",
          bool(ids) and all(len(reads) == 1 and 0 < reads[0][0] <= 2 * SLOT_N
                            for _short, reads in ids), f"ID drain records {ids}")
    # AT MOST, because `TX_ADMIT_RESERVE_S` is room in front of the key for
    # exactly this: the drain now finishes before the PTT instant instead of
    # standing on it, so a charge smaller than the reserve reaches the lead not at
    # all. What the gate holds is that nothing BUT the charged drain can.
    check("...and its lead loss is no more than the explicitly charged drain",
          bool(ids) and all(len(reads) == 1
                            and short <= reads[0][1]
                            == round(reads[0][0] * DRAIN_RATE)
                            for short, reads in ids),
          f"{[(short, reads) for short, reads in ids]}")

    # THE COUNTEREXAMPLE, planted rather than assumed: the same session with the
    # recovered slot fed to the rolling decoder, which is what it did. Nothing
    # else differs but the clock it is given -- a peer that has gone quiet no
    # longer takes a linked session off the air (`_MasterGrid._blind`), so the
    # runaway is left to run rather than being interrupted by a hush, and it
    # needs 278 s of bench clock to reach the same demonstration it used to
    # reach in 154. Longer, not looser: every gate below is the one it flew.
    flown = _session(regrid=_fed_regrid, seconds=400.0, **scene)
    fkeyed = [s for s, _w, _i in _landed(flown)]
    finc = [b - a for a, b in zip(fkeyed, fkeyed[1:])]
    check("NEGATIVE CONTROL: with the recovered slot rolling-decoded, the slot "
          "handed back costs more than the slot it saves",
          bool(finc) and max(finc) >= 2 * max(inc) and "REGRID GAVE UP"
          in flown["log"],
          f"worst {max(finc, default=0)} slots against {max(inc, default=0)}, "
          f"{flown['log'].count('REGRID GAVE UP')} re-grids gave up, "
          f"{flown['bench'].now / FS:.0f} s of clock against "
          f"{got['bench'].now / FS:.0f}")


def the_window_is_fed_only_where_an_answer_can_still_be_used() -> None:
    """Audio further back than a cycle is collected and held, never fed.

    The bound `the_decode_does_not_choose_the_window` rests on, asked of the one
    function that turns a window into audio. A session cannot demonstrate it once
    the loop is bounded -- the arrears it exists for never build up -- so it is
    asked directly, which is also the only way to see that the held audio is
    still IN the stream rather than skipped over.
    """
    print("\nThe rolling decode against the length of the window")
    fed, held = [], []

    class _Rx:
        def feed(self, chunk):
            fed.append(chunk.size)

        def bridge(self, chunk):
            held.append(chunk.size)

    class _Arq:
        state = State.CONNECTED

    class _Host:
        protocol = Protocol.PACTOR1
        arq = _Arq()

    bench = _Bench()
    want = 5 * SLOT_N
    # The session's own number, so the check reads the CONSTANT and not a copy of
    # it: `FEED_MAX_SLOTS` is what the three call sites pass, and it is the value
    # -- one cycle -- that the bound rests on.
    feed_n = onair.FEED_MAX_SLOTS * SLOT_N
    got = onair._listen_until_answer(bench, want, _Host(), _Rx(), feed_n)
    check("the whole window is collected", got.size == want,
          f"{got.size / FS:.2f} s of {want / FS:.2f}")
    # Against the CYCLE rather than against `feed_n`, because a slot is the
    # protocol's own answer time and is the reason the cap is where it is. One
    # slice of slack: the split is decided per slice, not per sample.
    check("...and no more than the cycle a station answers in went through the "
          "rolling decode", sum(fed) <= SLOT_N + int(0.25 * FS),
          f"{sum(fed) / FS:.2f} s fed of {want / FS:.2f} collected, against a "
          f"{SLOT_N / FS:.2f} s cycle")
    check("...and everything in front of it was held, not skipped",
          sum(fed) + sum(held) == want,
          f"{(sum(fed) + sum(held)) / FS:.2f} s reached the decoder of "
          f"{want / FS:.2f} collected")
    line = next((ln for ln in _capture_listen(
        bench, want, feed_n, _Host(), _Rx()).splitlines()
        if onair.BEHIND_THE_GRID in ln), "")
    check("...and it said so", bool(line))
    # WHAT THE LINE IS ALLOWED TO CLAIM. As flown it read "only the 1.25 s in
    # front of the key can be decoded as it arrives ... every cycle it could
    # have answered has gone", and onair-0811-1215 -- where it fired once, on a
    # hushed cycle, over a 32 ms sliver straddling a boundary nothing keyed on
    # -- was read live as the receive window discarding the gateway's answer.
    # The checks above are the facts: every sample collected, every sample
    # decoded. The line must state the excess it is about and that holding it
    # drops nothing, not narrate a loss that is not happening.
    check("...naming the excess it is holding",
          f"{(want - feed_n) / FS * 1e3:.0f} ms" in line, line)
    check("...and claiming no loss, because there is none",
          "discards nothing" in line and "have answered has gone" not in line,
          line)


def _capture_listen(bench, want, feed_n, host, rx) -> str:
    log = io.StringIO()
    bench.pos = bench.now = 0
    with contextlib.redirect_stdout(log):
        onair._listen_until_answer(bench, want, host, rx, feed_n)
    return log.getvalue()


def the_slot_search_leaves_room_for_the_settle() -> None:
    """A boundary counts as keyable only if the PTT settle still fits in front.

    Placeability alone accepts a boundary `key_notice` ahead -- 32 ms, the
    converter's latency plus the three blocks of notice its callback needs --
    and `transmit` then finds the PTT instant already past and asserts the key
    immediately. The carrier comes up exactly where it was asked for and every
    figure in the log agrees, so nothing downstream can tell: what is lost is the
    settle, and on the x6100 that is 32 ms of the 400 the rig is set for.

    Asked of the search directly rather than through a session, because the rig
    it bites hardest is one whose cycle budget does not close at all -- 400 ms of
    settle plus a 960 ms packet plus a 120 ms answer will not fit in 1250, and
    `run` refuses to key rather than transmit on it.
    """
    print("\nThe slot search, against the settle it still owes the rig")
    raster = onair._MasterGrid(anchor=0, slot_n=SLOT_N,
                               offset_n=round(0.185 * FS),
                               packet_n=round(spec.P1_PACKET_S * FS),
                               cs_n=round(spec.P1_CS_S * FS),
                               d_max_n=round(0.13 * FS))
    for rig, settle in (("ft891", 0.04), ("g90", 0.10), ("x6100", 0.40)):
        bench = _Bench()
        # Standing exactly far enough back that slot 1 is placeable and no
        # further: `clamp_late` is zero there and one sample later it is not.
        bench.now = bench.pos = raster.boundary(1) - bench.key_notice
        got = onair._keyable_slot(bench, raster, 1, round(settle * FS),
                                  listen=False)
        lead = (raster.boundary(got) - bench.now) / FS
        check(f"{rig}: a {settle * 1e3:.0f} ms settle is still in hand at the "
              f"slot it picks", lead >= settle,
              f"slot {got}, {lead * 1e3:.0f} ms of notice")


def the_cadence_instrument_reads_the_carriers() -> None:
    """What the session REPORTS as its cadence, against what it flew.

    `median slot increment 1 = 1.25 s` was printed by the session that put half
    its bursts in the next slot. The instrument read the slot the loop intended,
    appended before the emission path could still move the burst -- so it
    reported the plan and called it the measurement. Two things are asked here
    and the first is the one with teeth: the number the summary is built from has
    to be the number the carriers came up on.
    """
    print("\nThe cadence as flown, counted from the carriers")
    got = _session(cycles=4, hold=10, charge=OVERRUN_N, decode=True,
                   peer=True)
    keyed = [s for s, _w, _i in _landed(got)]
    # `slots_used` carries the connect too, which goes out before there is a grid
    # to place it on and is the thing the anchor is measured from.
    reported = got["tx"].slots_used[-len(keyed):]
    check("the cadence instrument records the slot each carrier came up in",
          reported == keyed, f"reported {reported}, flew {keyed}")
    check("no two carriers share a slot", len(set(keyed)) == len(keyed),
          f"slots {keyed}")

    # With the cycle inside its slot there is no room to interpret: the protocol
    # is one packet every 1.25 s, in every cycle, and the grid has no reason to
    # skip one. "Slower is allowed" cannot fail against a loop whose own comment
    # reads ONE SLOT, ALWAYS.
    #
    # Six call cycles let this peer supply its three corroborating answers
    # before the final-call listening tail. Four intentionally starts that tail
    # before connection; its receive-only slot is not steady linked cadence.
    # Correct ACK handling now keeps this peer advancing without retry yields.
    easy = _session(cycles=6, hold=10, charge=0, peer=True)
    inc = [b - a for a, b in zip(easy["tx"].slots_used,
                                 easy["tx"].slots_used[1:])]
    check("a cycle that fits in its slot keys in every one of them",
          bool(inc) and set(inc) == {1}, f"increments {inc}")


def _quiet_run(got: dict) -> int:
    """The longest run of consecutive slots the session put no carrier in.

    Off `RadioTx.keyed`, which records the slot of every burst the rig was keyed
    for. Slots rather than samples because the anchor MOVES -- `reverse` slides
    it 840 ms at a changeover -- so a gap measured against the anchor as it
    finished would fabricate one. `the_cadence_instrument_reads_the_carriers`
    is what holds those slots to the carriers that actually came up in them.
    """
    slots = sorted({s for s, _ in got["tx"].keyed})
    return max((b - a - 1 for a, b in zip(slots, slots[1:])), default=0)


def a_linked_session_holds_its_cadence() -> None:
    """A gateway that answers and then fades must not be given up on by silence.

    The hush takes the transmitter off the air for HUSH_CYCLES so a grid that
    has never been answered can be placed by what it hears. Armed on a session
    that HAS been answered it is the opposite of a repair: a PACTOR IRS times
    its acknowledgement against a 1.25 s raster, so six cycles of nothing is
    several cycles past the point it gives up, and the station we are trying to
    hold is the only one the silence can reach.

    The scene is the commonest way a real session ends -- an answer, a link, and
    then a peer that fades -- and the two runs differ in nothing but the rule.

    THE COUNTERS ARE ASKED HERE TOO, against the bench's own record of the RF.
    The summary is the line every duty-cycle argument gets quoted off, and the
    number it carried counted RENDERED bursts: a 120 ms control signal and a
    960 ms packet alike, and in a dry run bursts that were written to a file.
    """
    print("\nA link held through a peer that goes quiet")
    HUSH = onair._MasterGrid.HUSH_CYCLES
    # FIVE ANSWERED CYCLES, because a connect is no longer one codeword: the
    # search reads this station on the odd cycles and `_ConnectEvidence` wants
    # three of them before it will call it a link. Three answers left the scene
    # calling into cycle six and never reaching the hold this is about.
    scene = dict(cycles=6, hold=40, charge=0, peer=True, quiet_after=5)
    got = _session(**scene)
    check("a session that has been answered is never taken off the air for a hush",
          _quiet_run(got) < HUSH,
          f"longest silence {_quiet_run(got)} slot(s), a hush is {HUSH}")
    check("...and it does not report going quiet either",
          "OFF THE AIR" not in got["log"])

    # THE COUNTEREXAMPLE, PLANTED: remove both the hush-arm guard and the
    # connected-session cancellation in update, leaving their other work intact.
    saved = onair._MasterGrid._blind
    saved_update = onair._MasterGrid.update

    def as_flown(self, why="nothing heard", *, linked=False):
        self.acquired = False
        return saved(self, why, linked=False)

    def unlinked_update(self, *args, **kwargs):
        kwargs["linked"] = False
        return saved_update(self, *args, **kwargs)

    onair._MasterGrid._blind = as_flown
    onair._MasterGrid.update = unlinked_update
    try:
        flown = _session(**scene)
    finally:
        onair._MasterGrid._blind = saved
        onair._MasterGrid.update = saved_update
    check("NEGATIVE CONTROL: without the rule the same peer is left in silence",
          _quiet_run(flown) >= HUSH,
          f"longest silence {_quiet_run(flown)} slot(s) against "
          f"{_quiet_run(got)} with the rule")

    # -- and what the summary says about it -------------------------------
    bench, tx = got["bench"], got["tx"]
    air = sum(b - a for a, b in bench.emissions) / FS
    check("the carrier count and the air time the session reports are the "
          "bench's own record of the RF",
          len(tx.keyed) == len(bench.emissions)
          and abs(sum(d for _, d in tx.keyed) - air) < 1e-3,
          f"{len(tx.keyed)} keyed against {len(bench.emissions)} carriers, "
          f"{sum(d for _, d in tx.keyed):.2f} s against {air:.2f} s on the air")
    line = next((ln for ln in got["log"].splitlines()
                 if ln.startswith("keyed ")), "")
    pct = [int(x) for x in re.findall(r"\((\d+)%", line)] or [0, 0]
    check("...and says both what fraction of the cycles it keyed and of the air",
          len(pct) == 2 and f"{air:.1f} s of carrier" in line,
          line)
    # The two are not the same number. That they cannot be DERIVED from one
    # another wants burst lengths that run eight to one, and this scene no
    # longer carries any: a peer that answers the call and then never
    # acknowledges a packet is not yielded to, so every burst here is a packet
    # and the session ends on the retry budget rather than as an IRS. The
    # eight-to-one half is asked where a link really does change direction --
    # see `the_reversed_link_answers_out_of_channel_time`.
    check("...which are different numbers, because a burst is not a cycle",
          pct[0] != pct[1],
          f"{pct[0]}% of cycles, {pct[1]}% of the air, burst lengths "
          f"{sorted({round(d, 2) for _, d in tx.keyed})}")
    # ...AND THE HALF THAT DISCRIMINATES. In an armed session every burst that is
    # rendered is also keyed, so the check above cannot tell a count of carriers
    # from a count of bursts and does not claim to. The case that can is a run
    # with the rig unarmed: it renders every burst to a file, keys nothing, and
    # counted them as transmissions all the same.
    with tempfile.TemporaryDirectory() as tmp:
        dry = onair.RadioTx(None, transmit=False, out_dev=None,
                            outdir=Path(tmp))
        with contextlib.redirect_stdout(io.StringIO()):
            dry._tx(np.ones(FS // 10, np.float32), "a burst nobody hears")
    check("a burst the rig was never keyed for is not counted as one",
          dry.n == 1 and dry.keyed == [], f"n {dry.n}, keyed {dry.keyed}")


def a_hushed_cycle_stands_on_its_own_boundary() -> None:
    """The off-grid figure measures the instant the cycle actually aimed at.

    A keyed cycle aims at `boundary - settle` -- PTT has to be up before the
    audio starts -- and a hushed one keys nothing and reads to the boundary
    itself. Measured against the key instant regardless, every hushed cycle
    reported the keying settle as distance off the grid: onair-0811-1215
    printed a walk from +0.1 to +133.8 ms that way, 40 ms of it the FT-891's
    settle and the rest delivery latency and decode time, while its own
    sidecars show the windows holding at one cycle all session (59904 +/- 384
    samples of 60000) -- a grid that never drifted a sample. The operator read
    the figure as the receive window walking away from the gateway's answer.

    The bench charges nothing and its clock is arithmetic, so a session that
    measures each cycle against its own aim point prints zero on every line,
    hushed and keyed alike. The peer is silent, which is what arms the hush.
    """
    print("\nWhere a hushed cycle says it stands")
    got = _session(cycles=12, hold=0, charge=0, peer=False)
    offs: dict[bool, list[float]] = {True: [], False: []}
    for ln in got["log"].splitlines():
        m = re.search(r"off-grid ([+-][\d.]+) ms", ln)
        if m:
            offs["HUSHED" in ln].append(float(m.group(1)))
    check("a peer that never answers takes the session into a hush",
          len(offs[True]) >= onair._MasterGrid.HUSH_CYCLES,
          f"{len(offs[True])} hushed lines, {len(offs[False])} keyed")
    worst = max(map(abs, offs[True]), default=float("inf"))
    check("a hushed cycle standing on its boundary says so",
          worst < 1.0, f"worst {worst:+.1f} ms against the +40.0 it read "
          f"as flown, with nothing charged and nothing late")
    # ...ONE RESERVE IN FRONT OF IT, and deliberately. A keyed cycle stops
    # reading `onair.TX_ADMIT_RESERVE_S` before the PTT instant so the tick, the
    # render and the drain have a budget somebody measured, and standing that far
    # early is what the reserve IS. What the gate holds is that the distance is
    # the reserve and nothing else.
    reserve = onair.TX_ADMIT_RESERVE_S * 1e3
    worst_keyed = max((abs(o + reserve) for o in offs[False]),
                      default=float("inf"))
    check("...and a keyed cycle still measures against its key instant, one "
          "measured reserve in front of it",
          worst_keyed < 1.0,
          f"worst {worst_keyed:+.1f} ms off a reserve of {reserve:.0f}")


def _burst_pair(*, remember: bool) -> tuple[list[int], list[bool], str]:
    """Two PACTOR-1 packets rendered against ONE aim, as the flush path does.

    `remember` is whether the transmitter still knows the first slot is spent.
    Cleared, and with the later pre-render lateness recovery disabled, this is
    the aim as it flew: `_flip` has nothing to tell it that the cycle it names
    has already been keyed, so it hands the second burst the first slot's
    polarity and the emission path inherits the miss.
    """
    grid = onair._MasterGrid(0, SLOT_N, round(onair.TX_OFFSET_S * FS),
                             packet_n=round(spec.P1_PACKET_S * FS),
                             cs_n=round(spec.P1_CS_S * FS),
                             d_max_n=onair._d_max_n(spec.CYCLE_SHORT_S, 0.04))
    rendered: list[bool] = []

    class _Tx(onair.RadioTx):
        def _tx(self, audio, what, drive=None, **kwargs):
            rendered.append(self._sent_invert)
            super()._tx(audio, what, drive=drive, **kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        tx = _Tx(_Rig(), transmit=True, out_dev=0, outdir=Path(tmp), settle=0.04)
        tx.live = _Bench()
        with contextlib.redirect_stdout(io.StringIO()) as log:
            tx.aim(grid, 4)
            tx.send_p1_packet(b"the quick brown fox ", 100, 1)
            if not remember:
                del tx.slots_used[:]
                # Recreate the historical renderer, before unused late slots
                # gained their own recovery. Otherwise that independent guard
                # repairs the deliberately forgotten spent slot as well.
                advance = tx._advance_aim
                tx._advance_aim = lambda **kwargs: advance()
            tx.send_p1_packet(b"jumps over the lazy ", 100, 2)
    slots = [round((a - grid.anchor) / SLOT_N) for a, _e in tx.live.emissions]
    return slots, rendered, log.getvalue()


def a_second_burst_in_a_cycle_belongs_to_the_next_one() -> None:
    """A burst rendered after this cycle's slot is spent is aimed one slot on.

    `arq.on_rx_cs` answers a control signal in the cycle it arrived in, and
    `_SessionRx.flush` -- where a codeword the anchored read missed finally
    surfaces -- runs behind `host.tick()`. So the FSM's answer is built while
    the aim still names the slot the tick has just keyed, and `_flip` bakes THAT
    slot's polarity into the samples. The emission path is then holding finished
    audio in the spent cycle's shift, and
    `the_carrier_keeps_the_shift_it_was_rendered_in` is it doing the right thing
    with a burst that should never have reached it: it steps in TWOS to keep the
    polarity it was handed, so the answer lands a whole cycle after the one the
    peer is timing us against.

    MEASURED, working/t6-pactor-ws8eoc-force.log, WS8EOC on 3596500,
    2026-08-19. Every answer keyed +1507.4 ms into a slot boundary and went out
    on slots 10, 14, 18, 22, 26, 30 against a gateway keying every slot -- one
    whole 1.25 s cycle behind, and never drifting off it, because slot parity
    quantises the miss to two. WS8EOC answered CS1 twenty-one times without once
    alternating to CS2: the packet counter never left #1, no greeting arrived,
    and the operator on a monitor heard us answering every other one.
    """
    print("\nA second burst in one cycle, against the cycle it belongs to")
    slots, rendered, log = _burst_pair(remember=True)
    check("the answer goes out in the slot after the one already keyed",
          slots == [4, 5], f"slots {slots}")
    check("...in that slot's own shift, so nothing has to move it",
          rendered[1] != rendered[0] and onair.LATE_KEY not in log,
          f"shifts {rendered} over {log.count(onair.LATE_KEY)} late keys")

    flown, flown_shift, flown_log = _burst_pair(remember=False)
    check("NEGATIVE CONTROL: an aim that does not know its slot is spent loses "
          "a whole cycle",
          flown == [4, 6] and onair.LATE_KEY in flown_log,
          f"slots {flown} over {flown_log.count(onair.LATE_KEY)} late keys")
    check("...and it is the spent cycle's polarity that costs it the slot "
          "between",
          flown_shift[1] == flown_shift[0], f"shifts {flown_shift}")


# KB5LZK on 30 m, 2026-09-03, off the arm's own log
# (working/onair-0903-1051/pactor-day-05-kb5lzk-mail.log) and the session
# capture behind it. The anchor is solved from `hold 5 slot 19, CS due @
# 1192113`; the peer's break-in head is the sample the QRM guard quoted, and its
# 1.25 s raster carries every one of the 27 packets the gateway then sent.
KB5LZK_ANCHOR = 2620
KB5LZK_D_N = round(0.0711 * FS)
KB5LZK_PEER_AT = 1132113
P1_PACKET_N = round(spec.P1_PACKET_S * FS)
P1_CS_N = round(spec.P1_CS_S * FS)


def _kb5lzk_grid() -> onair._MasterGrid:
    """The arm's grid at the changeover, through both reversals."""
    g = onair._MasterGrid(KB5LZK_ANCHOR, SLOT_N, 0, packet_n=P1_PACKET_N,
                          cs_n=P1_CS_N, d_max_n=round(0.130 * FS))
    g.d_n, g.d_ref_n, g.sending = float(KB5LZK_D_N), P1_PACKET_N, True
    g.note_peer_codeword(KB5LZK_PEER_AT, P1_CS_N, "CS3/break-in", "KB5LZK")
    g.reverse(to_iss=False)                # the peer broke in
    g.reverse(to_iss=True)                 # ...and asked us to take the link
    return g


def the_changeover_is_placed_on_the_peers_transmission() -> None:
    """Where the packet that takes the link keys, once the grid has turned twice.

    `reverse` moves the transmit anchor by `data_n - cs_n` on the way to IRS and
    moves only `d_ref_n` on the way back, which is correct -- each rotation is
    one instant seen from one end, and `rx_due` comes out unmoved. The BOUNDARY
    COMB does not: it is 840 ms along, which is -410 ms on a 1.25 s cycle, and
    the packet ending walked back from `rx_due` is that far inside the packet it
    names. The changeover keyed there, seven times, and KB5LZK answered none of
    them -- 656-667 ms into its own transmission on the tape, which is this
    arithmetic plus the tracker's pulls.
    """
    print("\nThe changeover packet against KB5LZK's raster, 2026-09-03")
    g = _kb5lzk_grid()
    peer = lambda k: KB5LZK_PEER_AT + k * SLOT_N      # noqa: E731 -- one line
    slot = 23                                         # hold 9, where TX[17] keyed

    check("the arm's grid, before it turned, put the peer's break-in where the "
          "log read it",
          onair._MasterGrid(KB5LZK_ANCHOR, SLOT_N, 0, packet_n=P1_PACKET_N,
                            cs_n=P1_CS_N, d_max_n=0).boundary(18)
          + P1_PACKET_N + KB5LZK_D_N == KB5LZK_PEER_AT)
    check("hold 8's control signal is due where the log printed it",
          g.rx_due(22) == 1412433, f"{g.rx_due(22)} vs 1412432 in the log")
    check("...and the transmit anchor is 840 ms along, which the way back does "
          "not undo", g.anchor - KB5LZK_ANCHOR == P1_PACKET_N - P1_CS_N,
          f"{(g.anchor - KB5LZK_ANCHOR) / FS * 1e3:+.0f} ms")

    read = g.peer_packet_end(slot)
    end = read.at_slot
    stale = g.rx_due(slot) - g.cycle_n + g.data_n      # what the placement read
    check("the reading places the peer's packet ending where its raster has it",
          end == peer(4) + P1_PACKET_N, f"{end} vs {peer(4) + P1_PACKET_N}")
    check("...and it carries its own origin, so a tape can be walked back to it",
          (read.at, read.end, read.cycles)
          == (KB5LZK_PEER_AT, KB5LZK_PEER_AT + P1_PACKET_N, 4),
          f"onset {read.at}, {read.cycles} cycle(s) on")
    check("NEGATIVE CONTROL: walked back from `rx_due` it is 840 ms past the "
          "packet before, which is 550 ms INSIDE the one it names",
          stale - (peer(4) + P1_PACKET_N) == P1_PACKET_N - P1_CS_N
          and peer(5) < stale < peer(5) + P1_PACKET_N,
          f"{(stale - peer(5)) / FS * 1e3:.0f} ms into a "
          f"{P1_PACKET_N / FS * 1e3:.0f} ms packet")

    tx = onair.RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
    tx.aim(g, slot)
    tx.breakin_due = True
    key = tx._breakin_key(g, slot)
    lead_n = round(onair.BREAKIN_LEAD_S * FS)
    check("the changeover keys where the peer reads -- 170 ms less the "
          "turnaround past its packet, whatever the reversals did to our comb",
          key == end + g.peer_read_gap,
          f"{(key - end) / FS * 1e3:.1f} ms past it")
    check("...so its first bit is in no packet of the peer's",
          all(not peer(k) <= key < peer(k) + P1_PACKET_N for k in range(27)),
          f"{(key - peer(4) - P1_PACKET_N) / FS * 1e3:+.1f} ms past k=4")
    check("...and its CS3 head is whole before the peer's next transmission, "
          "which is the one it cancels", key + P1_CS_N <= peer(5),
          f"{(peer(5) - key - P1_CS_N) / FS * 1e3:.1f} ms to spare")
    check("the window in front of it closes at the key, and on a comb this "
          "reading agrees with the key IS the boundary -- reached without "
          "reading it", tx.key_instant(g, slot) == key
          and key == g.boundary(slot),
          f"{(key - g.boundary(slot)) / FS * 1e3:+.1f} ms off the boundary")

    flown = stale + lead_n
    check("NEGATIVE CONTROL: the instant the arm keyed is 622 ms into the "
          "peer's packet", peer(5) < flown < peer(5) + P1_PACKET_N,
          f"{(flown - peer(5)) / FS * 1e3:.0f} ms in")
    tx.breakin_at_boundary = False
    check("NEGATIVE CONTROL: --breakin-lead-72 keys clear of the packet and "
          "still 27 ms in front of the reader",
          tx._breakin_key(g, slot) == end + lead_n
          and g.peer_read_gap - lead_n > round(0.025 * FS),
          f"{(g.peer_read_gap - lead_n) / FS * 1e3:.1f} ms short")
    tx.breakin_at_boundary = True

    # ...and an instant already gone is GIVEN UP rather than carried a cycle.
    # The step read as prudence and cost the cycle it was meant to save: it runs
    # at render time, behind a window that has already closed against the
    # instant being abandoned, so the burst went out 1.25 s later against a
    # packet nothing had read. 2026-09-04's two mail arms took it 18 times of 18
    # and lost nine of the peer's packets an arm to it.
    class _Standing:
        pos = key + 1
        keyed_at = None

        def clamp_late(self, at: int) -> int:
            return max(0, self.pos - at)

    tx.live = _Standing()
    late = tx._breakin_key(g, slot)
    tx.live = None
    check("a placed instant already gone is not carried a cycle: the placement "
          "stands and the cycle is what yields", late == key,
          f"{(late - key) / FS * 1e3:+.0f} ms on")

    # THE GUARD, BOTH WAYS, on the same two instants.
    settle_n = round(0.040 * FS)
    air_n = settle_n + P1_PACKET_N
    check("the guard passes the placed changeover, which by design lies across "
          "the slot the peer would have transmitted in",
          g.key_refusal(key - settle_n, air_n, changeover=True) is None)
    ordinary = g.key_refusal(key - settle_n, air_n)
    check("...and refuses the same instant asked as an ordinary burst, which is "
          "what dropped it on the arm",
          ordinary is not None and "before our" in ordinary, ordinary or "")
    refused = g.key_refusal(flown - settle_n, air_n, changeover=True)
    check("the guard refuses the instant the arm keyed, because the packet it "
          "answers is still on the air",
          refused is not None and "still on the air" in refused, refused or "")


# KB5LZK's two mail arms of 2026-09-04, off their own logs
# (working/onair-0904-1343 and -1354) and the session captures beside them.
# Each row is (anchor, tracked d, the CS3 head of the break-in packet the
# gateway took the link with, the boundary M the arm's own `rot` line implies).
MAIL0904 = (("arm 1", 2725, round(0.0716 * FS), round(39.84 * FS), 96.8),
            ("arm 3", 2623, round(0.0724 * FS), round(41.09 * FS), 94.6))
#: 2026-08-22, KB5LZK, 100 Bd: the one changeover packet a real modem has
#: accepted from us. `d` was 78.9 ms over ten tracked cycles and TX[28]'s audio
#: began 94.9 ms after the peer's packet ended.
NIGHT17_D_N = round(0.0789 * FS)
#: Where our audio began on each of arm 3's nine changeover keyings, in stream
#: samples: the end of the deep T/R dropout the rig cuts into `stream.wav` at
#: every PTT, which the arm's own `RF started ... its audio +X` arithmetic
#: agrees with to +/-3 ms across all 36 of its keyings.
MAIL0904_A3_AUDIO = tuple(round(t * FS) for t in (
    44.618, 48.370, 50.868, 53.370, 55.868, 58.370, 60.869, 63.370, 65.869))


def _mail0904_grid(anchor: int, d_n: int, peer_at: int) -> onair._MasterGrid:
    """One arm's grid at its first changeover: reversed once, by the peer."""
    g = onair._MasterGrid(anchor, SLOT_N, 0, packet_n=P1_PACKET_N,
                          cs_n=P1_CS_N, d_max_n=round(0.130 * FS))
    g.d_n, g.d_ref_n, g.sending = float(d_n), P1_PACKET_N, True
    g.note_peer_codeword(peer_at, P1_CS_N, "CS3/break-in", "KB5LZK")
    g.reverse(to_iss=False)              # the peer broke in and holds the link
    return g


def the_changeover_keys_where_the_peer_reads() -> None:
    """`170 ms - d` past the peer's packet, not `BREAKIN_LEAD_S`.

    Eighteen changeover packets over the two mail arms, every one logged
    `+1322 ms past the peer's packet` -- 72 + 1250 -- and every one keyed, on
    the tape, at +68 to +70 ms past the peer's packet end. KB5LZK acknowledges a
    codeword of ours at 88.5-98.0 ms (twelve of them, four arms, two bands, both
    speeds; median 94.9), and the one changeover packet it has ever accepted sat
    at +94.9. The lead put all eighteen 21-30 ms in front of the earliest instant
    that reader has been shown to take, and none was answered -- in arms whose
    own codewords were read at +96.0 one cycle earlier.

    The gap is the cycle closing on itself, so it is computed and not carried:
    the peer's packet, our turnaround, our 120 ms, the turnaround back.
    """
    print("\nWhere the changeover keys, against where KB5LZK reads")
    lead_n = round(onair.BREAKIN_LEAD_S * FS)
    for name, anchor, d_n, peer_at, want in MAIL0904:
        g = _mail0904_grid(anchor, d_n, peer_at)
        slot = 34                              # where each arm's TX[23] keyed
        read = g.peer_packet_end(slot)
        tx = onair.RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
        tx.aim(g, slot)
        tx.breakin_due = True
        key = tx._breakin_key(g, slot)
        gap = (key - read.at_slot) / FS * 1e3

        check(f"{name}: the changeover's first bit lands 170 ms less the "
              f"tracked turnaround past the peer's packet end, and never later "
              f"than our own boundary",
              key == min(g.boundary(slot), read.at_slot + g.peer_read_gap)
              and g.peer_read_gap == SLOT_N - P1_PACKET_N - P1_CS_N - d_n,
              f"{gap:.1f} ms at d = {d_n / FS * 1e3:.1f}, rule "
              f"{g.peer_read_gap / FS * 1e3:.1f}")
        check(f"{name}: ...which is inside the window KB5LZK's packet counter "
              f"has advanced across, where the rule alone spends the tracker's "
              f"1.6-3.0 ms of prediction error on the far edge of it",
              91.0 <= gap <= 97.0 and abs(gap - want) < 0.1,
              f"{gap:.1f} ms against {want:.1f} off the arm's own rot line")
        check(f"{name}: ...and the CS3 head is whole a turnaround before the "
              f"peer's next packet, at every d",
              read.at_slot + SLOT_N - P1_PACKET_N - (key + P1_CS_N) >= d_n,
              f"{(read.at_slot + SLOT_N - P1_PACKET_N - key - P1_CS_N) / FS * 1e3:.1f}"
              f" ms to spare against a {d_n / FS * 1e3:.1f} ms turnaround")
        check(f"{name}: no cycle is stepped -- the placement names the cycle "
              f"the aimed slot names and stops there",
              read.at == peer_at and read.end == peer_at + P1_PACKET_N
              and read.at_slot == read.end + read.cycles * SLOT_N
              and 0 <= g.boundary(slot) - read.at_slot < SLOT_N,
              f"onset {read.at}, {read.cycles} cycle(s) on")

        # WHAT FLEW, on the same grid: the lead, and then a whole peer cycle on
        # top of it because the instant had gone by the time the burst was
        # rendered. The log named the first and the tape recorded the second.
        tx.breakin_at_boundary = False
        control = tx._breakin_key(g, slot)
        check(f"{name}: NEGATIVE CONTROL: --breakin-lead-72 is the placement "
              f"that flew, 22-25 ms in front of the reader and below every "
              f"instant it has acknowledged",
              control == read.at_slot + lead_n
              and gap - onair.BREAKIN_LEAD_S * 1e3 >= 22.0,
              f"+{onair.BREAKIN_LEAD_S * 1e3:.0f} ms, "
              f"{gap - onair.BREAKIN_LEAD_S * 1e3:.1f} ms short")
        if name == "arm 3":
            check(f"{name}: NEGATIVE CONTROL: and one of the peer's cycles on "
                  f"top of it is the carrier the tape recorded -- the step "
                  f"that hid the whole of it",
                  abs(control + SLOT_N - MAIL0904_A3_AUDIO[0])
                  <= round(0.005 * FS),
                  f"{(control + SLOT_N - MAIL0904_A3_AUDIO[0]) / FS * 1e3:+.1f}"
                  f" ms off the tape")

    # THE ONE ACCEPTED CHANGEOVER, 2026-08-22, at the same gateway on 40 m and
    # 100 Bd. Its `d` is 7 ms longer, so the rule moves with it -- which is why
    # it is a rule and not the 0.095 the same measurement would support.
    night17 = _mail0904_grid(2623, NIGHT17_D_N, round(36.09 * FS))
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
    tx.aim(night17, 34)
    tx.breakin_due = True
    accepted = ((tx._breakin_key(night17, 34)
                 - night17.peer_packet_end(34).at_slot) / FS * 1e3)
    check("2026-08-22's geometry puts the one accepted changeover's own "
          "instant in the same window -- it was read at +94.9",
          91.0 <= accepted <= 95.0,
          f"{accepted:.1f} ms at d = {NIGHT17_D_N / FS * 1e3:.1f}")
    moved = accepted - MAIL0904[1][4]
    check("...and a constant could not have served both: 6.5 ms of turnaround "
          "is 6.5 ms of placement, which is why this is a rule",
          -4.0 <= moved <= -3.0, f"{moved:.1f} ms")


# KB5LZK's three mail arms of the 2026-09-04 EVENING slot, off their own logs
# (working/onair-0904-1657, -1706 and -1715). Each row is (arm, the tracked `d`
# the log's last `[grid] d ...` line carried into the changeover run, the sample
# that line named, and the `+M ms past the peer's packet end` the arm printed
# for every placement made while that turnaround was still acquired).
EVE0904 = (("arm 1", round(0.0748 * FS), 1352665, 95.0),
           ("arm 3", round(0.0730 * FS), 4152947, 97.0),
           ("arm 4", round(0.0717 * FS), 4032796, 98.0))
#: What every placement in all three arms came out at once the turnaround had
#: been released: `settle + PREKEY_RESERVE_S`, the transmitter's own floor, and
#: 21-30 ms in front of the earliest instant KB5LZK has been shown to read at.
EVE0904_FLOOR_MS = 70.0


def _eve0904_grid(d_n: int, peer_at: int, *,
                  slack_ms: float = 105.0) -> onair._MasterGrid:
    """One evening arm's grid at its changeover run, holding the link.

    `slack_ms` is `boundary - at_slot`, which the arms' own placements fix from
    the other side: all three printed the rule's own gap and never the clamp, so
    the comb sat further past the peer's packet than `170 ms - d` does.
    """
    end = peer_at + P1_PACKET_N
    anchor = end + round(slack_ms * FS / 1e3) - 34 * SLOT_N
    g = onair._MasterGrid(anchor, SLOT_N, 0, packet_n=P1_PACKET_N,
                          cs_n=P1_CS_N, d_max_n=round(0.130 * FS))
    g.d_n, g.d_ref_n, g.sending = float(d_n), P1_PACKET_N, True
    g.note_peer_codeword(peer_at, P1_CS_N, "CS1/ack", "KB5LZK")
    return g


def _eve0904_tx(g: onair._MasterGrid, slot: int) -> onair.RadioTx:
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
    tx.aim(g, slot)
    tx.breakin_due = True
    return tx


def the_changeover_cliff_is_the_turnaround_and_not_the_onsets_age() -> None:
    """What moved 17 of 35 changeover packets onto +70 ms, and what did not.

    The evening slot logged a hard cliff: arm 4 placed at +98 ms on onsets aged
    1, 3, 4, 7 and 8 cycles and at +70 on the same onset aged 5 to 10, and the
    two other arms did the same at 95 and 97. Read as an age it is not one --
    age 8 placed at +98 and age 5 at +70, in the same arm, off the same reading.

    The line in between is `receive window released after 3 cycles (nothing
    heard)`, and it is in all three logs immediately before that arm's first
    +70. `peer_read_gap` is `cycle - packet - CS - d`; `d` falls back to
    `D_NOMINAL_S` when the window is released, and 1250 - 960 - 120 - 105 is
    65 ms, which is under the settle and reserve the transmitter owes. So the
    floor takes the placement and prints +70 -- the instant eighteen keyings
    refuted the same day -- with every other line still reading healthy.
    """
    print("\nThe +70 ms cliff, against the age it was read as")
    for name, d_n, peer_at, want in EVE0904:
        g = _eve0904_grid(d_n, peer_at)
        rule = g.peer_read_gap / FS * 1e3
        placed = []
        for age in range(0, 11):
            slot = 34 + age
            read = g.peer_packet_end(slot)
            tx = _eve0904_tx(g, slot)
            placed.append(((tx._breakin_key(g, slot) - read.at_slot)
                           / FS * 1e3, read.cycles))
        check(f"{name}: the placement is the same instant at every age out to "
              f"ten cycles -- the projection is not what moved it",
              {round(m, 1) for m, _ in placed} == {round(rule, 1)}
              and [c for _, c in placed] == list(range(11)),
              f"{sorted({round(m, 1) for m, _ in placed})} ms over ages "
              f"{placed[0][1]}-{placed[-1][1]}")
        check(f"{name}: ...and it is the +{want:.0f} ms the log printed, which "
              f"is `170 ms - d` at the turnaround that arm was tracking",
              abs(rule - want) < 0.6,
              f"{rule:.1f} ms at d = {d_n / FS * 1e3:.1f}")

        # THE RELEASE, which is the line the arms printed before their first +70.
        for _ in range(g.MAX_MISSES):
            line = g.update([])
        check(f"{name}: three cycles with nothing where a codeword is due "
              f"release the turnaround",
              "receive window released" in line and not g.locked, line)
        check(f"{name}: ...and the gap then falls back to a nominal 105 ms `d` "
              f"and 65 ms, under the transmitter's floor",
              round(g.peer_read_gap / FS * 1e3, 1) == 65.0
              and g.peer_read_gap < round((0.040 + onair.PREKEY_RESERVE_S) * FS),
              f"{g.peer_read_gap / FS * 1e3:.1f} ms")

        slot = 39
        read = g.peer_packet_end(slot)
        tx = _eve0904_tx(g, slot)
        flown = (tx._breakin_key(g, slot) - read.at_slot) / FS * 1e3
        check(f"{name}: NEGATIVE CONTROL: the arithmetic that flew puts the "
              f"packet on the floor at +70 ms, 25-28 ms in front of the reader",
              round(flown, 1) == EVE0904_FLOOR_MS and want - flown >= 24.0,
              f"+{flown:.1f} ms against +{want:.0f}")
        refusal = g.breakin_refusal(read)
        check(f"{name}: ...and it is refused instead, on the turnaround and by "
              f"name",
              refusal is not None and "turnaround was released" in refusal,
              refusal or "placed anyway")


def a_placement_that_cannot_be_made_spends_no_retry() -> None:
    """The refusal reaches the FSM as `arq.REFUSED`, and nothing goes out.

    A changeover has one instant and it is the peer's. Where that instant
    cannot be computed there is no second-best placement -- only our own comb,
    which is what the evening slot keyed 17 times. The budget counts air, so a
    cycle that keyed nothing spends none of it and the packet is placed again
    on the next reading.
    """
    print("\nA changeover that cannot be placed")
    _, d_n, peer_at, _ = EVE0904[2]
    g = _eve0904_grid(d_n, peer_at)
    for _ in range(g.MAX_MISSES):
        g.update([])
    tx = _eve0904_tx(g, 39)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rendered = tx.send_p1_breakin(b"LZK >\r", 100, 0)
    said = out.getvalue()

    check("the packet is not keyed", tx.n == 0 and not tx.keyings,
          f"{tx.n} transmission(s)")
    check("...the FSM is handed REFUSED, so the retry budget spends nothing",
          rendered == arq.REFUSED and tx.refused, f"{rendered}")
    check("...and the line names the refusal and what it stood on",
          "CHANGEOVER NOT PLACED" in said and "no retry spent" in said
          and "turnaround was released" in said, said.strip() or "(silent)")

    # THE AGE BOUND, on a grid whose turnaround never left.
    g = _eve0904_grid(d_n, peer_at)
    slot = 34 + onair.ONSET_MAX_CYCLES
    tx = _eve0904_tx(g, slot)
    check(f"the oldest reading with on-air evidence behind it still places: "
          f"arm 4's TX[55] was {onair.ONSET_MAX_CYCLES} cycles on, landed at "
          f"+98 and drew CS1/at anchor at zero bit errors",
          g.breakin_refusal(g.peer_packet_end(slot)) is None
          and g.peer_packet_end(slot).cycles == onair.ONSET_MAX_CYCLES)
    tx = _eve0904_tx(g, slot + 1)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rendered = tx.send_p1_breakin(b"LZK >\r", 100, 0)
    said = out.getvalue()
    check("one cycle past it is refused, and the line names the age",
          rendered == arq.REFUSED and tx.n == 0
          and f"projected {onair.ONSET_MAX_CYCLES + 1} cycles" in said
          and f"sample {peer_at}" in said, said.strip() or "(silent)")


def a_burst_nothing_decoded_re_origins_the_peers_raster() -> None:
    """What refreshes the reading while our own carrier is over the peer.

    Holding the link we key 960 ms of every 1.25 s cycle, so the peer's
    transmissions fall under our carrier: `note_peer_codeword` never fires and
    `_track` has no turnaround to measure against. The burst timer does fire --
    the evening arms printed 118-130 ms runs in the gaps and the reply test
    called them `p1reply` -- and those are what re-origin the raster.

    The acceptance rule carries the whole of it, because nothing here reads a
    bit: on the peer's OWN comb inside `MAX_PULL_S`, and at least
    `ONSET_MIN_MS` long. The first bounds how far one acceptance can move the
    origin, against the projection the last one left, so it cannot walk.
    """
    print("\nA burst nothing decoded, against the raster it re-origins")
    _, d_n, peer_at, want = EVE0904[2]
    g = _eve0904_grid(d_n, peer_at)
    slot = 34 + onair.ONSET_MAX_CYCLES + 1
    check("nine cycles on, the placement is refused on the reading's age",
          g.breakin_refusal(g.peer_packet_end(slot)) is not None)

    heard = peer_at + 8 * SLOT_N
    line = g.note_peer_bursts([(heard, round(0.125 * FS))])
    check("a 125 ms burst on the peer's comb re-origins the raster",
          line is not None and g.peer_at == heard and "125 ms" in line,
          line or "not taken")
    read = g.peer_packet_end(slot)
    tx = _eve0904_tx(g, slot)
    key = tx._breakin_key(g, slot)
    check("...so the reading is one cycle old again and the placement stands",
          read.cycles == 1 and g.breakin_refusal(read) is None,
          f"{read.cycles} cycle(s) on")
    true_m = (key - (peer_at + 9 * SLOT_N + P1_PACKET_N)) / FS * 1e3
    check("...and the burst lands where the arm's own decoded placements did, "
          "measured against the peer's UNMOVED raster",
          88.5 <= true_m <= 98.5 and abs(true_m - want) < 0.6,
          f"+{true_m:.1f} ms against the arm's +{want:.0f}")

    # AND THE READING'S OWN SLIP IS THE PLACEMENT'S, 1:1. `MAX_PULL_S` bounds
    # how far one acceptance can move the origin and says nothing about how
    # accurately: the detector places a burst on a 5 ms grid, and 5 ms of that
    # is 5 ms at the far end's reader, whose anchor `p1rx.CS_ANCHOR_S` records
    # as a cliff inside a few. It is what the codeword read buys where it can
    # be had, and it is why this path is the fallback and not the reference.
    slipped = []
    for slip_ms in (-5.0, 0.0, +5.0):
        g2 = _eve0904_grid(d_n, peer_at)
        g2.note_peer_bursts([(heard + round(slip_ms * FS / 1e3),
                              round(0.125 * FS))])
        tx2 = _eve0904_tx(g2, slot)
        slipped.append((tx2._breakin_key(g2, slot)
                        - (peer_at + 9 * SLOT_N + P1_PACKET_N)) / FS * 1e3)
    check("the slip in the reading is the slip in the placement, one for one -- "
          "the acceptance band bounds walking, not accuracy",
          all(abs((m - true_m) - s) < 0.1
              for m, s in zip(slipped, (-5.0, 0.0, +5.0))),
          f"{[round(m, 1) for m in slipped]} ms at -5/0/+5")

    # NEGATIVE CONTROLS, on the same grid and the same cycle.
    for label, at, width, why in (
            ("300 ms off the peer's comb", heard + round(0.300 * FS),
             round(0.125 * FS), "off-raster"),
            ("60 ms long, which no PACTOR-1 transmission is", heard,
             round(0.060 * FS), "too short")):
        g2 = _eve0904_grid(d_n, peer_at)
        before = g2.peer_at
        check(f"NEGATIVE CONTROL: a burst {label} does not move it ({why})",
              g2.note_peer_bursts([(at, width)]) is None
              and g2.peer_at == before,
              f"{g2.peer_at} vs {before}")


def the_changeover_gives_up_a_cycle_rather_than_stepping_one() -> None:
    """An instant already gone is refused, not carried onto the next peer cycle.

    The step read as prudence and cost the cycle it was meant to save. It runs
    at RENDER time, behind a window already closed against the instant being
    abandoned, so the burst went out 1.25 s later and the cycle in between was
    spent off the air and deaf: arm 3's windows run 47.096 -> 49.330 s with the
    peer's whole 49.831 -> 50.812 packet -- the one the next placement would
    have been read from -- in no capture the arm holds. Nine of those an arm,
    eighteen keyings of eighteen.

    What made the instant unreachable is the settle. A changeover keys `early`
    in front of the boundary, and everything that reserved the settle reserved
    it in front of the BOUNDARY, so the tick and the render were left
    `settle - early`: 17.4 ms of the 40 on arm 3, and the arm's own
    `off-grid +3.2 ms` puts 14.2 ms in hand for work its ordinary cycles
    measure 9-13 ms at.
    """
    print("\nThe changeover cycle's own settle")
    name, anchor, d_n, peer_at, _ = MAIL0904[1]
    g = _mail0904_grid(anchor, d_n, peer_at)
    slot, settle_n = 34, round(0.040 * FS)
    tx = onair.RadioTx(None, transmit=False, outdir=Path("."), settle=0.040)
    tx.aim(g, slot)
    tx.breakin_due = True
    key = tx._breakin_key(g, slot)
    early = g.boundary(slot) - key

    check("the window in front of the key closes at whichever of the key and "
          "the boundary comes first",
          tx.key_instant(g, slot) == min(g.boundary(slot), key),
          f"{early / FS * 1e3:+.1f} ms off the boundary")
    tx.breakin_at_boundary = False
    control = tx._breakin_key(g, slot)
    check("NEGATIVE CONTROL: --breakin-lead-72 keys 22.6 ms IN FRONT of the "
          "boundary, and that is what came off the settle: 17.4 ms of the 40 "
          "for a tick and a render this arm's ordinary cycles measure 9-13 at",
          round((g.boundary(slot) - control) / FS * 1e3, 1) == 22.6
          and round((settle_n - (g.boundary(slot) - control)) / FS * 1e3, 1)
          == 17.4,
          f"{(g.boundary(slot) - control) / FS * 1e3:.1f} ms in front")
    tx.breakin_at_boundary = True

    class _Standing:
        keyed_at = None

        def __init__(self, pos: int) -> None:
            self.pos = pos

        def clamp_late(self, at: int) -> int:
            return max(0, self.pos - at)

    tx.live = _Standing(key - settle_n)
    check("a cycle standing where the collect leaves it has the whole settle "
          "in hand for the tick and the render",
          tx.live.clamp_late(key) == 0 and tx._breakin_key(g, slot) == key)
    tx.live = _Standing(key + 1)
    check("NEGATIVE CONTROL: an instant already gone is NOT carried a peer "
          "cycle -- the placement stands and `_tx` gives the cycle up",
          tx._breakin_key(g, slot) == key,
          f"{(tx._breakin_key(g, slot) - key) / FS * 1e3:+.0f} ms on")
    tx.live = None

    # THE LINE. It named +1322 ms because it measured the achieved instant
    # against the reading rather than against the ending that reading projects.
    tx.aim(g, slot)
    tx.breakin_due = True
    said = tx._place_breakin()
    check("the log line names the ACHIEVED instant against the packet end it "
          "was placed on",
          f"+{(tx.boundary - g.peer_packet_end(slot).at_slot) / FS * 1e3:.0f}"
          f" ms past the peer's packet end" in said, said.strip())
    check("...and carries the projection's origin, so a tape can be walked "
          "back to it",
          f"sample {peer_at}" in said and "1 cycle(s) on" in said, said.strip())
    stepped = round(onair.BREAKIN_LEAD_S * FS) + SLOT_N
    check("NEGATIVE CONTROL: the arm's own +1322 ms is the lead plus one of "
          "the peer's cycles and says nothing about where the burst went",
          round(stepped / FS * 1e3) == 1322, f"{stepped / FS * 1e3:.0f} ms")


def the_clear_narrator_times_the_burst_it_heard() -> None:
    """`[clear]` said 1352 ms about a burst that ended 1290 ms before our carrier.

    It timed every onset as a 120 ms codeword. What it was timing was the
    193-200 ms TAIL of a 960 ms packet whose head was under our own carrier --
    the only part of KB5LZK's transmission that reached us in the changeover
    phase -- and the length was measured all along: `rxfront.p1_bursts` returns
    it and `p1_burst_onsets` threw it away.
    """
    print("\nWhat [clear] times")
    # TX[25] of onair-0904-1354, off the tape: our carrier up at the T/R
    # dropout, down 1.007 s later, and the one burst the 262 ms window in front
    # of it held.
    key_up = 2440368
    # The onset the arm's own detector reported -- the [clear] line names it at
    # 1352 ms with a 120 ms codeword assumed, which is this sample.
    tail_at, tail_n = 2369712, round(0.200 * FS)

    class _Tx:
        tx_key_up, tx_end = key_up, key_up + round(1.007 * FS)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        onair._report_collision([(tail_at, tail_n)], _Tx())
    said = out.getvalue().strip()
    gap = (key_up - tail_at - tail_n) / FS * 1e3
    check("the burst is timed to its own end, which the detector measured",
          f"{gap:.0f} ms" in said and "200 ms of it" in said, said)
    check("NEGATIVE CONTROL: called a codeword it is the arm's 1352 ms",
          round((key_up - tail_at - P1_CS_N) / FS * 1e3) == 1352,
          f"{(key_up - tail_at - P1_CS_N) / FS * 1e3:.0f} ms")


def the_entry_answer_reads_start_at_the_entry() -> None:
    """The first `ENTRY ANSWER` of an arm named a codeword the entry never saw.

    The window in front of a key is read after it, so the cycle the entry is
    keyed in reports an onset that arrived BEFORE the entry went out --
    2026-09-03's arms both did: the entry was keyed for slot 8, boundary 482625,
    and the first reading is sample 473649, 187 ms in front of it. `19 of 19 at
    the PACTOR-1 position` therefore carried one cycle that could not have read
    an entry packet. The population is decided on the sample now, not on the
    fold.
    """
    print("\nThe entry answer instrument, against the entry's own slot")
    g = onair._MasterGrid(2625, SLOT_N, 0, packet_n=P1_PACKET_N, cs_n=P1_CS_N,
                          d_max_n=round(0.130 * FS))
    g.d_n, g.d_ref_n, g.sending = float(round(0.0924 * FS)), P1_PACKET_N, True
    g.corroborated = True
    for k in (5, 6, 7):
        g.update([g.rx_due(k)])
    check("the peer's answers before the entry are the baseline and nothing else",
          g.answer_unread_ms is not None and not g.entry_answers,
          f"{g.answer_unread_ms:.1f} ms")

    g.keyed_slot = 8
    g.keying(Protocol.PACTOR3)
    check("the entry's own boundary is what the instrument remembers",
          g.entry_key_n == g.boundary(8) == 482625, f"{g.entry_key_n}")

    was = g.answer_unread_ms
    g.update([473649])
    check("a burst that arrived 187 ms BEFORE the entry keyed is not an answer "
          "to it", not g.entry_answers and g.answer_position() is None)
    check("...it is a reading of where the peer answered before the entry",
          g.answer_unread_ms != was, f"{g.answer_unread_ms:.1f} ms")
    check("...and the remembered instant is a sample, so the tracker's pull "
          "does not move it", g.entry_key_n == 482625)

    g.update([g.rx_due(9)])
    check("the first reading after the entry's own slot is the first answer to "
          "it", len(g.entry_answers) == 1
          and g.entry_answers[0][1] >= g.entry_key_n,
          f"{len(g.entry_answers)} reading(s)")
    check("...and the verdict counts that one and no other",
          "over 1 cycle(s)" in g.entry_verdict())


def the_entry_position_is_the_packet_that_was_keyed() -> None:
    """An `SL3 pkt` was judged against the SL1 template's extent.

    `ENTRY_END_N` is the template entry's, and the uninvited upgrade keys a
    level-3 data packet instead -- 25.8 ms longer on the generic raised cosine,
    which is past `MAX_PULL_S`, so a peer that genuinely read one lands in
    `OFF BOTH POSITIONS` and the instrument hides a positive. The extent the
    transmitter keyed is what the position is placed from; the constant stays as
    the template's prediction.

    `placement.PROTOCOL_RISE` has since brought the data packet onto the
    protocol's own pulse, 1.2 ms SHORTER than the template rather than 25.8 ms
    longer, so today's render no longer needs the rescue. The render that does
    is the one five arms flew, and it stays here as the negative control.
    """
    print("\nThe entry position, against the packet that went out")
    sl1 = onair._trim_silence(placement.link_packet(
        1, b"", 0x1a, swapped=False, flush=placement.ENTRY_FLUSH))
    sl3 = onair._trim_silence(placement.link_packet(3, b"", 0x01, swapped=False))
    path = placement.SPEED_PATHS[3]
    flown = onair._trim_silence(placement.data_packet(
        placement.field_info(b"", path.crc_bytes - 3, 0x01), path,
        cfg=modem.ModConfig()))
    check("the template entry is what the banked constant predicts",
          abs(sl1.size - onair.ENTRY_END_N) < FS // 200,
          f"keyed {sl1.size}, constant {onair.ENTRY_END_N}, "
          f"{(sl1.size - onair.ENTRY_END_N) / FS * 1e3:+.1f} ms")

    def armed(extent_n):
        g = onair._MasterGrid(2625, SLOT_N, 0, packet_n=P1_PACKET_N,
                              cs_n=P1_CS_N, d_max_n=round(0.130 * FS))
        g.d_n, g.d_ref_n, g.sending = (float(round(0.0924 * FS)), P1_PACKET_N,
                                       True)
        g.corroborated = True
        for k in (5, 6, 7):
            g.update([g.rx_due(k)])
        g.keyed_slot = 8
        g.keying(Protocol.PACTOR3, extent_n=extent_n)
        return g

    banked, keyed = armed(None), armed(sl3.size)
    check("a burst that says nothing about its extent leaves the prediction "
          "standing", banked.entry_end_n == onair.ENTRY_END_N)
    check("...and the packet that was keyed replaces it",
          keyed.entry_end_n == sl3.size != onair.ENTRY_END_N,
          f"{keyed.entry_end_n} keyed against {onair.ENTRY_END_N} banked")
    moved = keyed.entry_read_ms - banked.entry_read_ms
    # Pin the new sequence-parity header waveform extent, not the former
    # 21–26 ms envelope range whose upper edge changed with that header.
    check("...which moves the read position by the two packets' difference and "
          "nothing else",
          abs(moved - (sl3.size - onair.ENTRY_END_N) / FS * 1e3) < 0.05
          and sl3.size == 40207 and onair.ENTRY_END_N == 40266,
          f"{moved:+.1f} ms")

    at = keyed.boundary(9) + round(keyed.entry_read_ms / 1e3 * FS)
    keyed.update([at])
    banked.update([at])
    check("a peer answering where reading the SL3 packet puts it is read as "
          "having read it",
          "AT THE ENTRY POSITION" in keyed.answer_position(),
          keyed.answer_position())
    check("...and the protocol pulse puts that inside the template's own "
          "tolerance, so the banked constant reaches it as well",
          "AT THE ENTRY POSITION" in banked.answer_position(),
          banked.answer_position())

    flew, blind = armed(flown.size), armed(None)
    at = flew.boundary(9) + round(flew.entry_read_ms / 1e3 * FS)
    flew.update([at])
    blind.update([at])
    check("the raised-cosine packet five arms flew ends 25.8 ms past the "
          "template and is read as having been read from its own extent",
          "AT THE ENTRY POSITION" in flew.answer_position()
          and flown.size == 41517, flew.answer_position())
    check("NEGATIVE CONTROL: the same burst against the banked template is the "
          "reading that hid it",
          "OFF BOTH POSITIONS" in blind.answer_position(),
          blind.answer_position())
    check("...and the verdict reports the extent that went out",
          f"ends {sl3.size / FS * 1e3:.1f} ms" in keyed.entry_verdict(),
          keyed.entry_verdict())


def _run(*scenes) -> bool:
    """One scene or all of them, from a clean verdict."""
    global ok
    ok = True
    for scene in scenes:
        scene()
    return ok


SCENES_ALL = (the_changeover_is_placed_on_the_peers_transmission,
              the_entry_answer_reads_start_at_the_entry,
              the_entry_position_is_the_packet_that_was_keyed,
              the_emission_is_on_the_grid,
              the_carrier_keeps_the_shift_it_was_rendered_in,
              a_second_burst_in_a_cycle_belongs_to_the_next_one,
              the_recovered_slot_is_listened_through,
              the_reversed_link_answers_out_of_channel_time,
              the_changeover_cycle_keys_the_slot_the_rotation_bought,
              the_regrid_window_keeps_its_origin_across_its_own_key,
              the_decode_does_not_choose_the_window,
              the_window_is_fed_only_where_an_answer_can_still_be_used,
              the_slot_search_leaves_room_for_the_settle,
              the_cadence_instrument_reads_the_carriers,
              a_linked_session_holds_its_cadence,
              a_hushed_cycle_stands_on_its_own_boundary)


def main() -> int:
    passed = _run(*SCENES_ALL)
    print("\nALL PASS" if passed else "\nFAILED")
    return 0 if passed else 1


def test_grid() -> None:
    assert main() == 0


def test_the_decode_does_not_choose_the_window() -> None:
    assert _run(the_decode_does_not_choose_the_window)


def test_the_regrid_window_keeps_its_origin_across_its_own_key() -> None:
    assert _run(the_regrid_window_keeps_its_origin_across_its_own_key)


def test_the_window_is_fed_only_where_an_answer_can_still_be_used() -> None:
    assert _run(the_window_is_fed_only_where_an_answer_can_still_be_used)


def test_a_second_burst_in_a_cycle_belongs_to_the_next_one() -> None:
    assert _run(a_second_burst_in_a_cycle_belongs_to_the_next_one)


def test_a_hushed_cycle_stands_on_its_own_boundary() -> None:
    assert _run(a_hushed_cycle_stands_on_its_own_boundary)


def test_the_reversed_link_answers_out_of_channel_time() -> None:
    assert _run(the_reversed_link_answers_out_of_channel_time)


def test_the_changeover_cycle_keys_the_slot_the_rotation_bought() -> None:
    assert _run(the_changeover_cycle_keys_the_slot_the_rotation_bought)


def test_the_changeover_is_placed_on_the_peers_transmission() -> None:
    assert _run(the_changeover_is_placed_on_the_peers_transmission)


def test_the_changeover_keys_where_the_peer_reads() -> None:
    assert _run(the_changeover_keys_where_the_peer_reads)


def test_the_changeover_gives_up_a_cycle_rather_than_stepping_one() -> None:
    assert _run(the_changeover_gives_up_a_cycle_rather_than_stepping_one)


def test_the_changeover_cliff_is_the_turnaround_and_not_the_onsets_age() -> None:
    assert _run(the_changeover_cliff_is_the_turnaround_and_not_the_onsets_age)


def the_boundary_is_not_a_placement_with_the_turnaround_held() -> None:
    """KB5LZK 2026-09-07 21:57 CDT, TX[130] and TX[131]: `d` held at a 74 ms
    candidate, the read at +96, and the comb only 70 ms past the peer's
    projected ending -- so the floor bound and both went out at +70, unread.
    `breakin_refusal` sees a locked turnaround and passes it; the clamp rule
    has to refuse it by name.
    """
    print("\nThe boundary clamp with the turnaround held")
    _, d_n, peer_at, want = EVE0904[0]

    def packet_grid(slack_ms=105.0):
        g = _eve0904_grid(d_n, peer_at, slack_ms=slack_ms)
        # This negative control tests placement behind a full packet. The
        # shared fixture's CS1 is a bare word and now correctly supplies a
        # 120 ms duration, so use an unattributed packet onset for this case.
        g.peer_cs = None
        g.peer_onset = peer_at
        return g

    g = packet_grid(slack_ms=70.0)
    slot = 36
    read = g.peer_packet_end(slot)
    tx = _eve0904_tx(g, slot)
    check("the turnaround is held, so the released-turnaround rule passes it",
          g.locked and g.breakin_refusal(read) is None)
    flown = (tx._breakin_key(g, slot) - read.at_slot) / FS * 1e3
    check("NEGATIVE CONTROL: the placement arithmetic alone puts it on the "
          "floor, in front of the reader",
          round(flown, 1) == EVE0904_FLOOR_MS and want - flown >= 24.0,
          f"+{flown:.1f} ms against +{want:.0f}")
    why = tx._clamp_refusal(read)
    check("...and it is refused by name", why is not None
          and "boundary is not a placement" in why, why or "placed anyway")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rendered = tx.send_p1_breakin(b"LZK >\r", 100, 0)
    check("nothing is keyed and no retry is spent",
          tx.n == 0 and rendered == arq.REFUSED
          and "CHANGEOVER NOT PLACED" in out.getvalue(), out.getvalue().strip())
    g = packet_grid()
    tx = _eve0904_tx(g, slot)
    check("with the comb where the arms had it the placement stands",
          tx._clamp_refusal(g.peer_packet_end(slot)) is None)
    g = packet_grid(slack_ms=want - 3.0)
    tx = _eve0904_tx(g, slot)
    check("...and so does a comb 3 ms in front of the reader, which is what the "
          "acknowledged arms measured",
          tx._clamp_refusal(g.peer_packet_end(slot)) is None)


def test_a_placement_that_cannot_be_made_spends_no_retry() -> None:
    assert _run(a_placement_that_cannot_be_made_spends_no_retry)


def test_the_boundary_is_not_a_placement_with_the_turnaround_held() -> None:
    assert _run(the_boundary_is_not_a_placement_with_the_turnaround_held)


def test_a_burst_nothing_decoded_re_origins_the_peers_raster() -> None:
    assert _run(a_burst_nothing_decoded_re_origins_the_peers_raster)


def test_the_clear_narrator_times_the_burst_it_heard() -> None:
    assert _run(the_clear_narrator_times_the_burst_it_heard)


def test_the_entry_answer_reads_start_at_the_entry() -> None:
    assert _run(the_entry_answer_reads_start_at_the_entry)


def test_the_entry_position_is_the_packet_that_was_keyed() -> None:
    assert _run(the_entry_position_is_the_packet_that_was_keyed)


if __name__ == "__main__":
    sys.exit(main())
