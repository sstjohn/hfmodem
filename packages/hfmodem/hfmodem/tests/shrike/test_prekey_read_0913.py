# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The read in front of the key, over the arm whose every frame arrived late.

`captures/onair-0913-2320` keyed 182 IRS cycles at WS8EOC and its pre-key read
delivered one ordinary frame in all of them. The other 81 came from the NEXT
cycle's top-of-cycle sweep, so every codeword this station keyed answered the
packet before the one on the air -- 6.07 IRS cycles a delivery against the 2.56
of the arm that was working.

What put them there is `rxfront._frame_span`, whose last two rows are the
decoder's own margin rather than the packet. The window was sized on the CAPTURE
clock and the scan reads what the codec has DELIVERED, so it closed a holdback
inside that margin -- which the decode never missed, `UNREAD_TAIL_N` being past
every instant the grid touches, and which the ANCHOR did: `_p3_packet` floored
its projection against the same span, dropped a whole cycle and aimed behind the
start of the buffer. The packet itself ended a median 16.8 ms inside every one of
those windows. The negative control below is that floor and nothing else.

The audio is the arm's own PCM at the sample bounds its `hold_NN.json` sidecars
record, with the gaps between them -- our own transmitter, which the live stream
never contains -- zeroed as `test_slot_deadline` zeroes them. The clock is
`_Bench` under that file's charging, so the decode's wall time is charged to the
sample counter and an overrun is a number rather than a race.
"""
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, p3acquire, p3frame, rxfront
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench
from hfmodem.tests.shrike.test_slot_deadline import charging

FS, SPS = onair.FS, rxfront.SPS
FIXTURES = Path(__file__).with_name("fixtures") / "prekey-read-0913"
METADATA = FIXTURES / "metadata.json"
META = json.loads(METADATA.read_text()) if METADATA.exists() else None
pytestmark = pytest.mark.skipif(
    META is None or not (FIXTURES / META["file"]).exists(),
    reason=f"WS8EOC 80 m IRS crop is not installed: {METADATA}")

# The arm's own clocks line -- `block 128, ADC->DAC 1268 samples, DAC notice
# 1652, holdback 660`. The bench borrows `_LiveInput`'s notice and clamp
# arithmetic, so these put its codec where that night's was: 532 samples of
# input latency under a 128-frame block, which is how far behind the instant it
# waited for a read returns.
BLK, HOLDBACK, LAT_OUT = 128, 660, 1268
SETTLE_N = round(.04 * FS)
# Where the peer's data ended in front of our key, over the arm's 182 cycles:
# 76.4 ms median, 73.9 to 78.9 across the whole run.
D_N = round(.0739 * FS)
# A cycle of silence in front of the crop, so the clock the scene is seeded with
# -- row 0 of the cycle before the first one it reads -- is a positive index.
PRE = 60000


def pcm():
    return recorded_pcm({"file": f"prekey-read-0913/{META['file']}",
                         "sha256": META["pcm_sha256"]})


@contextlib.contextmanager
def _patched(owner, name, value):
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


class Prekey:
    """The hold loop's receive seam over those cycles, on the sample clock.

    Everything between one key and the next that has the peer's audio in front
    of it: the listen, the two collects the loop takes, the free sweep of last
    cycle's window, the pre-key tracked read and the host tick that chooses the
    codeword. What is not here is what no longer has any audio in front of it --
    the render, the drain and the carrier.

    `flown` puts back the term the arm flew with and nothing else: the anchor
    floored against the whole frame span rather than against the packet inside
    it. The window's own sizing is the same either way on this rig -- see
    `test_the_window_is_closed_on_what_the_codec_has_delivered`.
    """

    def __init__(self, *, flown: bool = False):
        audio = pcm()
        self.flown = flown
        self.bench = _Bench(seconds=(PRE + audio.size) / FS + 2,
                            blk=BLK, holdback=HOLDBACK, lat_in=HOLDBACK - BLK)
        self.bench._lat = LAT_OUT
        self.bench.audio[PRE:PRE + audio.size] = audio
        for lo, hi in META["own_carrier_spans"]:
            self.bench.audio[PRE + lo:PRE + hi] = 0.0
        self.bench.now = self.bench.pos = PRE
        self.session = _Session(role=arq.IRS)
        self.rx = self.session.rx
        self.rx.p3_receive_offset_hz = META["offset_hz"]
        # The clock a linked IRS stands on: the last frame it read, that frame's
        # geometry, and the speed level its CRC validated.
        self.rx._p3_row0 = PRE + META["seed_row0"]
        self.rx._p3_cycle_n = META["cycle_n"]
        self.rx._p3_span = META["frame_span"]
        self.rx.sync.packet_level = 1
        # Mid-stream, so the counter this end is waiting for is the one the peer
        # had on the air as the crop opens.
        self.session.host.arq._rx_seen = True
        self.session.host.arq._expected_seq = META["cycles"][0]["seq"]
        self.cycles, self.cost = [], {}

    def run(self):
        margin = (lambda span: span) if self.flown else rxfront._packet_span
        with charging(self.bench, self.cost), \
                _patched(rxfront, "_packet_span", margin):
            prev = None
            for row, nxt in zip(META["cycles"], META["cycles"][1:] + [None]):
                prev = self._cycle(row, nxt, prev)
        return self

    def _cycle(self, row, nxt, prev):
        bench, rx, host = self.bench, self.rx, self.session.host
        key_at = PRE + row["key"]
        flowing = rx.frame_seen
        before_sweep = len(self.session.packets)
        rx.new_cycle()
        # Last cycle's window, swept a whole cycle in front of this key: on
        # PACTOR-3 `_window_swept` never stands this one down.
        if prev is not None and not onair._window_swept(host, flowing):
            onair._scan_frame(rx, *prev)
            flowing = flowing or rx.frame_seen
        swept = [ev.packet[:3] for ev in self.session.packets[before_sweep:]]
        before_read = len(self.session.packets)
        whole = onair._listen_until_answer(
            bench, key_at - SETTLE_N - bench.holdback
            - round(onair.PREKEY_RESERVE_S * FS) - bench.pos, host, rx, 0)
        seg_start = bench.pos - whole.size
        whole, seg_start, _ = onair._collect(bench, rx, whole, seg_start,
                                             key_at - max(D_N, SETTLE_N))
        until = key_at - onair._prekey_lead(bench, SETTLE_N)
        if flowing:
            until = onair._p3_frame_ready(
                rx, onair._p3_decode_deadline(bench, key_at, SETTLE_N),
                bench.holdback)
        whole, seg_start, _ = onair._collect(bench, rx, whole, seg_start, until)
        if whole.size and flowing:
            onair._scan_frame(rx, whole, seg_start, tracked_only=True)
        got = self.session.packets[before_read:]
        # The codeword the FSM has for this key, taken where the loop takes it.
        host.tick()
        keyed = [i for kind, i in host.peer.sent if kind == "cs"]
        host.peer.sent.clear()
        self.cycles.append(dict(
            hold=row["hold"], flowing=flowing, swept=swept,
            prekey=[ev.packet[:3] for ev in got],
            rx_seq=host.arq.rx_seq, keyed=keyed,
            late=bench.clamp_late(key_at),
            room_ms=(key_at - bench.key_notice - bench.samples) / FS * 1e3))
        # Our own carrier, and the read cursor where `flush_to` leaves it.
        if nxt is not None:
            bench.flush_to(PRE + nxt["flown_window"][0])
        return whole, seg_start

    @property
    def delivered(self):
        """Cycles whose own packet reached the FSM in front of their own key."""
        return [c for c in self.cycles if c["prekey"]]


@pytest.fixture(scope="module")
def flown():
    return Prekey(flown=True).run()


@pytest.fixture(scope="module")
def fixed():
    return Prekey().run()


def test_the_flown_schedule_delivers_nothing_in_front_of_a_key(flown):
    """The negative control: the anchor back as it flew and nothing else moved.

    Floored against the whole frame span the projection is admissible on two of
    these eight cycles -- holds 113 and 118, which are the two the arm's own
    transcript records an aim for -- and on the other six `at` falls behind the
    start of the buffer and no read runs. Not one packet reaches the FSM in
    front of its own key. What the session does read arrives in the next cycle's
    sweep, a cycle behind the codeword that should have answered it, which is
    where all 81 of the arm's frames arrived.

    And the two gates compound, as they did on the air: a cycle that delivers
    nothing leaves the next one un-`flowing`, so the pre-key scan it would have
    had is stood down as well.
    """
    assert [r["hold"] for r in META["cycles"] if r["flown_aim_taken"]] == [113, 118]
    assert flown.delivered == []
    assert sum(bool(c["swept"]) for c in flown.cycles) == 3
    assert sum(c["flowing"] for c in flown.cycles) == 4


def test_the_pre_key_read_delivers_the_packet_on_the_air(fixed):
    """...and the same cycles, closed on the delivered frame and aimed at the
    packet's own end: every cycle that reaches the read takes the peer's packet
    before its own key, and takes the field the witness heard that cycle."""
    want = [(r["speed_level"], r["status"], bytes.fromhex(r["field"]))
            for r in META["cycles"][1:]]
    assert [c["hold"] for c in fixed.delivered] == [r["hold"]
                                                    for r in META["cycles"][1:]]
    assert [c["prekey"][0] for c in fixed.delivered] == want
    # Every one of them in its own cycle, so the sweep behind it finds nothing
    # left to take. The one exception is the first cycle's own packet: it opens
    # the scene with no delivery behind it, so it is not `flowing` and its window
    # reaches only the sweep -- which is the position all 81 of the arm's frames
    # were read from.
    assert [bool(c["swept"]) for c in fixed.cycles] == [False, True] + [False] * 6
    assert sum(c["flowing"] for c in fixed.cycles) == 7


def test_every_cycle_keys_the_codeword_the_counter_on_the_air_asks_for(fixed):
    """`_counter_cs_for` answers the last counter ACCEPTED, so a read that lands
    a cycle late acknowledges the packet before the one being sent. Read in the
    cycle it arrived in, the codeword answers the packet on the air -- across the
    peer's renumbering from 3 to 0 at hold 115 included."""
    on_air = {r["hold"]: r["seq"] for r in META["cycles"]}
    for c in fixed.cycles[1:]:
        want = arq.CS_REQUEST if c["rx_seq"] & 1 else arq.CS_ACK
        assert c["rx_seq"] == on_air[c["hold"]], c
        assert c["keyed"] and set(c["keyed"]) == {want}, c
    assert {c["rx_seq"] for c in fixed.cycles[1:]} == {3, 0}


def test_no_cycle_spends_past_its_own_key(fixed):
    """Charged through the harness `test_slot_deadline` measures slots with.

    The read this buys is the cheap one -- a tracked decode at the projected
    anchor, bounded to the last CRC-validated level with no occupancy ladder
    behind it -- and the window it reads closes where it always did, so a cycle
    that now delivers spends no more of its own key than one that did not.
    `clamp_late` is `_LiveInput`'s own guard, taken on the delivered count.
    """
    assert [c["late"] for c in fixed.cycles] == [0] * len(fixed.cycles)
    assert min(c["room_ms"] for c in fixed.cycles) > 0


def test_the_window_is_closed_on_what_the_codec_has_delivered():
    """The first change, and the rig that does not reach its floor.

    `_collect` waits for a capture instant and a read returns a `holdback`
    later, so a codec whose holdback is deeper than the alignment slack and the
    frame span's unread tail together hands the scan a window short of what the
    tracked grid reads. This station's 660 samples is not, and waits exactly as
    long as it did; a deeper one is held to the delivered end instead.
    """
    row0, span, cycle_n = 100_000, META["frame_span"], META["cycle_n"]
    rx = SimpleNamespace(_p3_row0=row0, _p3_span=span, _p3_cycle_n=cycle_n)
    deadline = row0 + cycle_n + span + 5_000
    flat = onair._p3_frame_ready(rx, deadline)
    assert flat == row0 + cycle_n + span + SPS // 2
    assert onair._p3_frame_ready(rx, deadline, HOLDBACK) == flat
    deep = rxfront.UNREAD_TAIL_N + SPS
    ready = onair._p3_frame_ready(rx, deadline, deep)
    assert ready == row0 + cycle_n + span - rxfront.UNREAD_TAIL_N + deep > flat
    # ...which is the delivered end reaching every instant the grid reads.
    assert ready - deep == row0 + cycle_n + span - rxfront.UNREAD_TAIL_N


def test_the_bounded_read_returns_what_the_occupancy_ladder_returns(fixed):
    """The third change, on the windows the second one reaches.

    `_packet_at_lock` walks the occupancy-ranked levels once the preferred one
    misses, and that ladder is up to 80 ms here against a 7.6 ms pre-key budget.
    Bounded to the level the last CRC validated, it returns the same events.
    """
    audio = pcm()
    # A cycle's worth of window around the anchor: the packet's header block sits
    # `DATA_OFFSET` rows AHEAD of grid row 0, so a buffer that opens on the row
    # holds no admissible alignment at all.
    lead = p3frame.DATA_OFFSET * SPS + 4 * SPS
    for row in META["cycles"]:
        lo = row["row0"] - lead
        seg = p3acquire.compensate(
            audio[lo:lo + lead + META["frame_span"] + 4 * SPS], META["offset_hz"])
        both = []
        for bounded in (True, False):
            sync = rxfront.SyncedRx()
            sync.packet_at, sync.packet_level = lead, row["speed_level"]
            both.append(sync.packet(seg, preferred_only=bounded))
        assert both[0] is not None and both[1] is not None, row["hold"]
        assert both[0].packet == both[1].packet, row["hold"]
        assert both[0].start == both[1].start, row["hold"]
