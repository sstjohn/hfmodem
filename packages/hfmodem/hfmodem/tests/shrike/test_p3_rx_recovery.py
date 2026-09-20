# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The cycle after a delivered P3 frame, read the way the live loop reads it.

`onair-0913-0014` delivered eleven of the thirty-four changeover packets WS8EOC
put on the air and ten more sit CRC-valid in the arm's own receive windows. All
ten are the cycle immediately following a delivered frame -- the cycle the hold
loop hands to `_scan_frame(..., tracked_only=True)`, which aimed at the
projected head and, when that missed, spent the cycle. Two of those windows also
need a 5 Hz frequency step the changeover search did not have.

The audio here is the arm's own PCM, cut at the sample bounds its `hold_NN.json`
sidecars record, so what these read is what the receiver had in hand that night.
"""
import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench

FS = onair.FS
FIXTURES = Path(__file__).with_name("fixtures")
META_PATH = FIXTURES / "ws8eoc-0913-rx-recovery.json"
META = json.loads(META_PATH.read_text()) if META_PATH.exists() else None
pytestmark = pytest.mark.skipif(
    META is None or not all((FIXTURES / row["file"]).exists()
                            for row in META["crops"]),
    reason="recorded WS8EOC 0913 receive-window crops are not installed")

# The changeover this peer repeated on every one of those cycles.
FIELD = (1, 0, b"RMS")

# WHERE A PRE-KEY READ PUTS CYCLE 6'S HEAD, which is not where the sweep puts it.
# The repeat used to reach the FSM a cycle late, out of the free top-of-cycle
# sweep, and a swept read is delivered by `_SessionRx._p3_changeover_packet` --
# `p3acquire.changeover`'s coherent argmax at the offset it acquires, -1.4 Hz on
# this window. That is the instant `cs3_head_sample` records and it is still what
# the no-room scene below asserts.
#
# `_p3_packet`'s changeover pre-key projection reads the same head BEFORE the key,
# which is the delivery this file was opened to get. The acquisition does not run
# there -- `_read_p3_packet` reaches `_p3_changeover_packet` only on a read that is
# not `tracked_only` -- so the estimator that fits in front of a key is
# `SyncedRx.control_signal_at`, read at the primary receive offset. On cycle 6 it
# lands a sixteenth of a symbol later; on cycle 5 the two agree to the sample.
#
# MEASURED, AND NOT THE PROJECTION'S AIM: `control_signal_at` returns the same
# start from every anchor across +/-1200 samples of either head, so the step is
# that window's residual carrier rotation across the twenty-symbol word and not
# the instant the clock projected. The projection aimed at +60 on cycle 6 -- the
# peer's cycle against our nominal 60000 -- and the read did not follow it there.
PREKEY_STEP_N = 30

# WHERE CYCLE 13'S CARRIERS ACTUALLY WERE, which is not the `offset_hz` its row
# records: that field is the HYPOTHESIS the 5 Hz list read the head at, and the
# 25 Hz list holds nothing within 8 Hz of it, which is the finding this file was
# opened for. The carriers themselves sit at -2.4 Hz -- the residual rotation of
# the head's own twenty symbol pairs (`p3acquire._refined`), a template fit
# against nominal 1080/1920 and a 1 Hz quality sweep all agree, the fit to
# 0.01 Hz and the sweep peaking at -2. Cycles 5 and 6, whose rows say +0.0, read
# -1.7 and -1.4 the same three ways.
CYCLE_13_TRUE_HZ = -2.4


def crop(name: str) -> tuple[np.ndarray, int]:
    """One recorded crop and the capture sample its first frame came from."""
    row = next(c for c in META["crops"] if c["file"] == name)
    with wave.open(str(FIXTURES / name)) as wav:
        assert (wav.getframerate(), wav.getsampwidth(),
                wav.getnchannels()) == (FS, 2, 1)
        raw = wav.readframes(wav.getnframes())
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768, row["first_sample"]


def window(cycle: str) -> tuple[np.ndarray, int]:
    """The receive window the arm actually collected for that peer cycle."""
    row = META["cycles"][cycle]
    pcm, first = crop(row["crop"])
    lo, hi = row["receive_window"]
    return pcm[lo - first:hi - first].copy(), lo


def scene(tmp_path, *, seeded_at: int, slot: int = 62):
    """A linked IRS receiver holding the raster its last delivery gave it."""
    session = _Session(role=arq.IRS)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(session.host)
    session.host.peer = tx
    tx.defer_p3_cs = True
    grid = onair._MasterGrid(2629, 60000, round(.185 * FS),
                             packet_n=46080, cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    grid.d_n = .0921 * FS
    grid.d_ref_n = 46080
    tx.raster = grid
    tx.aim(grid, slot)
    # What `_note_p3_frame` leaves behind after a CS3-headed delivery: the
    # peer may repeat it, so there is no variable header to track.
    session.rx._seed_p3_changeover_clock(seeded_at)
    return session, tx, grid


def read(session, cycle: str, *, tracked_only: bool = True) -> None:
    session.rx.new_cycle()
    audio, origin = window(cycle)
    onair._scan_frame(session.rx, audio, origin, tracked_only=tracked_only)


def clocked(tx, grid, slot: int, room_s: float) -> None:
    """Put the loop's clock `room_s` in front of this slot's key notice."""
    tx.aim(grid, slot)
    tx.live.now = (tx.key_instant(grid, slot) - tx.live.key_notice
                   - round(room_s * FS))
    assert not tx.live.clamp_late(tx.key_instant(grid, slot))


def benched(tmp_path, *, seeded_at: int, slot: int):
    session, tx, grid = scene(tmp_path, seeded_at=seeded_at, slot=slot)
    live = _Bench(seconds=1)
    live.audio = np.zeros(0, np.float32)
    live.limit = 1 << 40
    tx.live = live
    return session, tx, grid


def hold_seam(session, tx, grid, cycles, *, slot: int, prekey_room_s: float):
    """The hold loop's two frame-scan seams, in the loop's own order.

    Each iteration reads `flowing` before `new_cycle` clears it, gives the
    previous window the free top-of-cycle sweep when `_window_swept` says it is
    still owed one, collects this cycle's window, and then runs the pre-key scan
    under `tracked_only=tx.defer_p3_cs`. One iteration past the last window, so
    every window reaches the same seam the others did. What is under test is the
    predicate: everything else here is the loop's own sequence.
    """
    prev = None
    for step, cycle in enumerate([*cycles, None]):
        flowing = (session.rx.frame_seen
                   or session.host.arq.cycle_command_emitted)
        session.rx.new_cycle()
        # The top of the cycle, a whole listening window in front of the key.
        clocked(tx, grid, slot + step, 1.2)
        if prev is not None and not onair._window_swept(session.host, flowing):
            onair._scan_frame(session.rx, *prev)
            flowing = flowing or session.rx.frame_seen
        if cycle is None:
            return
        prev = window(cycle)
        clocked(tx, grid, slot + step, prekey_room_s)
        if flowing:
            onair._scan_frame(session.rx, *prev, tracked_only=tx.defer_p3_cs)


def test_the_cycle_after_a_delivery_still_reads_the_repeat(tmp_path):
    """Cycle 5 and then cycle 6, both through the tracked-only seam.

    Cycle 5 is one of the eleven the arm delivered; cycle 6 is the repeat it
    lost. Both windows hold the same CRC-valid field, and the tracked read
    alone used to find neither of them. It finds both now, and cycle 6 arrives
    at the pre-key estimator's instant -- `PREKEY_STEP_N` past the sweep's.
    """
    session, _, grid = scene(tmp_path, seeded_at=3408867)
    read(session, "5")
    assert [ev.packet[:3] for ev in session.packets] == [FIELD]
    assert session.rx._p3_delivered_at == META["cycles"]["5"]["cs3_head_sample"]

    read(session, "6")
    assert [ev.packet[:3] for ev in session.packets] == [FIELD, FIELD]
    assert all(ev.breakin for ev in session.packets)
    assert (session.rx._p3_delivered_at
            == META["cycles"]["6"]["cs3_head_sample"] + PREKEY_STEP_N)
    # A repeat of the same field on the next tick of the same comb, which is
    # what the grid has to see for the raster to corroborate at all.
    assert grid._p3_peer_identity == (True, *FIELD)
    assert grid._p3_raster_run == 2


def test_two_consecutive_cycles_deliver_through_the_hold_loop_seam(tmp_path):
    """The same two cycles, driven the way the hold loop drives them.

    Not `_scan_frame` twice: the loop chooses between the free sweep of the last
    window and a tracked read of this one, and that choice is what lost the
    repeat. Cycle 5 arrives in the sweep at the top of the second iteration --
    the position every one of the arm's eleven deliveries came from -- and cycle
    6 in the same iteration's pre-key read, which is why its instant is the
    pre-key estimator's.
    """
    session, tx, grid = benched(tmp_path, seeded_at=3408867, slot=62)
    hold_seam(session, tx, grid, ["5", "6"], slot=62, prekey_room_s=.060)
    assert [ev.packet[:3] for ev in session.packets] == [FIELD, FIELD]
    assert (session.rx._p3_delivered_at
            == META["cycles"]["6"]["cs3_head_sample"] + PREKEY_STEP_N)
    assert grid._p3_raster_run == 2


def test_a_cycle_its_key_refused_is_swept_at_the_next_top_of_cycle(tmp_path):
    """...and the window a reply deadline turned down is still owed a sweep.

    With 4 ms in front of the key the pre-key read declines the acquisition
    outright (`test_a_cycle_with_no_room_before_its_key_declines_the_acquisition`).
    Before this the cycle was then spent: the sweep that delivered cycle 5 set
    `frame_seen`, so cycle 6's own window never reached a sweep of its own. It
    reaches one now, one cycle late, where the loop has a whole cycle of room --
    and the peer keying every cycle is read every cycle.
    """
    session, tx, grid = benched(tmp_path, seeded_at=3408867, slot=62)
    hold_seam(session, tx, grid, ["5", "6"], slot=62, prekey_room_s=.004)
    assert [ev.packet[:3] for ev in session.packets] == [FIELD, FIELD]
    assert session.rx._p3_delivered_at == META["cycles"]["6"]["cs3_head_sample"]
    assert not tx.live.clamp_late(tx.key_instant(grid, 64))


def test_a_pactor3_window_is_never_swept_by_its_own_pre_key_read():
    """The predicate itself, which is where the repeat was being dropped.

    A PACTOR-3 pre-key scan is `tracked_only` on a window `_p3_frame_ready`
    closed early, so it never stands in for the sweep however much reached the
    FSM. A PACTOR-1 one reads its window out, and has no absolute-position
    watermark behind it, so there the sweep stays suppressed.
    """
    p3 = SimpleNamespace(protocol=spec.Protocol.PACTOR3)
    p1 = SimpleNamespace(protocol=spec.Protocol.PACTOR1)
    assert not onair._window_swept(p3, True)
    assert not onair._window_swept(p3, False)
    assert onair._window_swept(p1, True)
    assert not onair._window_swept(p1, False)


def test_one_physical_frame_across_a_read_boundary_is_delivered_once(tmp_path):
    """The control the fall-through must not break.

    Two overlapping tracked reads of the same window -- the shape a frame
    straddling the pre-key read and a recovered slot arrives in -- carry one
    physical changeover, and it reaches the host once.
    """
    session, _, _ = scene(tmp_path, seeded_at=3408867)
    audio, origin = window("5")
    head = META["cycles"]["5"]["cs3_head_sample"]
    for lead in (0, 2400):
        session.rx.new_cycle()
        onair._scan_frame(session.rx, audio[lead:], origin + lead,
                          tracked_only=True)
    assert [ev.packet[:3] for ev in session.packets] == [FIELD]
    assert session.rx._p3_delivered_at == head


def test_a_five_hertz_cycle_reads_through_the_tracked_seam(tmp_path):
    """Cycle 13, whose head the coarse 25 Hz grid has no hypothesis for."""
    session, _, _ = scene(tmp_path, seeded_at=4128807, slot=70)
    read(session, "13")
    assert [ev.packet[:3] for ev in session.packets] == [FIELD]
    assert session.rx.p3_receive_offset_hz == CYCLE_13_TRUE_HZ
    assert session.rx._p3_delivered_at == META["cycles"]["13"]["cs3_head_sample"]


def test_the_coarse_grid_alone_cannot_read_that_cycle(tmp_path):
    """...and the same window through each offset list, with nothing else moved.

    `p3acquire.changeover` is handed the two lists the receiver builds. The
    coarse one is the list the arm swept; it reads a head and no body.
    """
    session, _, _ = scene(tmp_path, seeded_at=4128807, slot=70)
    audio, _ = window("13")
    coarse = p3acquire.changeover(audio, offsets=session.rx._p3_coarse_offsets())
    assert coarse is None or coarse.event.packet is None
    fine = p3acquire.changeover(audio, offsets=session.rx._p3_fine_offsets())
    assert fine is not None and fine.event.packet is not None
    assert fine.event.packet[:3] == FIELD
    assert fine.coarse_hz == META["cycles"]["13"]["offset_hz"]
    assert fine.offset_hz == CYCLE_13_TRUE_HZ


def test_fine_offsets_bracket_both_the_acquired_correction_and_zero():
    steps = onair._SessionRx.P3_FINE_STEPS_HZ
    rx = onair._SessionRx.__new__(onair._SessionRx)
    rx.p3_receive_offset_hz = 0.0
    assert rx._p3_fine_offsets() == steps
    rx.p3_receive_offset_hz = -25.0
    assert rx._p3_fine_offsets() == (-20.0, -30.0, *steps)
    # Nothing the coarse sweep already tried, and nothing off the ends of
    # `p3acquire.CONTROL_OFFSETS_HZ`.
    rx.p3_receive_offset_hz = 75.0
    assert rx._p3_fine_offsets() == (70.0, *steps)


def test_a_cycle_with_no_room_before_its_key_declines_the_acquisition(tmp_path):
    """The deadline, on the loop's own admission arithmetic.

    A tracked read that falls through has to leave the carrier its slot. With
    the reserve in hand the frame is delivered; with the key one block away the
    read declines, and the slot it would have spent is still placeable.

    THE DELIVERED ONE IS THE PRE-KEY INSTANT, which is the whole difference
    between the two rows: the row that is delivered is delivered by the
    projection, and the row that declines never reads at all. The sweep's own
    instant is `test_a_cycle_its_key_refused_is_swept_at_the_next_top_of_cycle`.
    """
    head = META["cycles"]["6"]["cs3_head_sample"] + PREKEY_STEP_N
    for room_s, delivered in ((.060, True), (.004, False)):
        session, tx, grid = scene(tmp_path, seeded_at=3708867)
        live = _Bench(seconds=1)
        live.audio = np.zeros(0, np.float32)
        live.limit = 1 << 40
        tx.live = live
        slot = 63
        tx.aim(grid, slot)
        live.now = (tx.key_instant(grid, slot) - live.key_notice
                    - round(room_s * FS))
        assert not live.clamp_late(tx.key_instant(grid, slot))
        read(session, "6")
        assert bool(session.packets) is delivered
        assert not live.clamp_late(tx.key_instant(grid, slot))
        if delivered:
            assert session.rx._p3_delivered_at == head
