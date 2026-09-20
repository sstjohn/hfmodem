# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The cold half of the anchored codeword read, on a peer that sends no codeword.

`arm-v22-A-40-ws8eoc` held an IRS stint under a gateway that keys data packets
and nothing else, so `_p3_answer_at` was never written and `_SessionRx._p3_cs`
fell through to `_acquire_answer` on every cycle of the link. That sweep is
5.86 ms of `p3acquire.control_signal` against 1.8 ms of headroom between the
frame-end close and `_regrid`'s admission check: 23 of 56 slots went to
`SLOT N IS GONE`, the cadence went to two, and the gateway was answered every
2.5 s on a 1.25 s raster.

The other half of the same cycle is the transform. `_read_p3_packet` and
`_p3_cs` are handed the same window at the same receive offset and each ran
`p3acquire.compensate` on it -- 0.89 ms of a 0.94 s window, twice.
"""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, placement, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench

FS = onair.FS
CYCLE = 60000
SL = 1

# `captures/onair-0914-0850`'s own: the receive window a held short cycle
# collected, the offset the session latched off its changeover acquisition, and
# the room the frame-end close left in front of the key.
WINDOW_N = 45314
RECEIVE_OFFSET_HZ = -18.4
ARM_ROOM_S = .0018

# More than the answer sweep's reserve asks and less than the changeover
# acquisition's, so one cycle admits the one read under test and nothing else.
ROOM_S = .020

CYCLES = 6


def peer_window(seed: int) -> np.ndarray:
    """One cycle of a gateway that is the ISS: a data packet, and no codeword."""
    payload = bytes((3 * i + SL) & 0x5F | 0x20
                    for i in range(spec.SPEED_LEVELS[SL].payload_short))
    packet = placement.link_packet(SL, payload, 0)
    window = np.random.default_rng(seed).normal(0, .02, WINDOW_N)
    window[1500:1500 + packet.size] += packet
    return window.astype(np.float32)


@pytest.fixture
def stint(tmp_path):
    """A linked IRS receiver with a transmitter aimed at its next slot."""
    session = _Session(role=arq.IRS)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(session.host)
    session.host.peer = tx
    tx.defer_p3_cs = True
    grid = onair._MasterGrid(0, CYCLE, round(.185 * FS), packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    grid.d_n, grid.d_ref_n = round(.093 * FS), 46080
    tx.raster = grid
    live = _Bench(seconds=1)
    live.audio = np.zeros(0, np.float32)
    live.limit = 1 << 40
    tx.live = live
    session.rx.p3_receive_offset_hz = RECEIVE_OFFSET_HZ
    return session, tx, grid


def held_cycle(session, tx, grid, n: int, *, room_s: float) -> np.ndarray:
    """One held cycle, at the two seams the hold loop reads a P3 answer at."""
    slot = 4 + n
    tx.aim(grid, slot)
    tx.live.now = tx.live.pos = (tx.key_instant(grid, slot)
                                 - tx.live.key_notice - round(room_s * FS))
    audio, origin = peer_window(n), n * CYCLE
    session.rx.new_cycle()
    onair._scan_frame(session.rx, audio, origin, tracked_only=True)
    session.rx.control_signal(audio, origin,
                              origin + audio.size - round(.24 * FS))
    return audio


@pytest.fixture
def sweeps(monkeypatch):
    """Every `_acquire_answer` sweep this stint asks for."""
    asked: list[int] = []

    def counted(seg, *a, **kw):
        asked.append(seg.size)
        return None

    monkeypatch.setattr(p3acquire, "control_signal", counted)
    return asked


def test_a_data_only_peer_leaves_no_answer_instant_to_read_at(stint, sweeps):
    """The premise: nothing in this stint ever writes `_p3_answer_at`.

    Which is what puts the cold half in front of every cycle rather than in
    front of the first one.
    """
    session, tx, grid = stint
    for n in range(CYCLES):
        held_cycle(session, tx, grid, n, room_s=ROOM_S)
    assert session.rx._p3_answer_at is None


def test_the_answer_sweep_declines_a_cycle_that_cannot_afford_it(stint, sweeps):
    """1.8 ms of headroom against a 5.9 ms search, six cycles running."""
    session, tx, grid = stint
    for n in range(CYCLES):
        held_cycle(session, tx, grid, n, room_s=ARM_ROOM_S)
    assert sweeps == []


def test_the_answer_sweep_still_runs_where_the_cycle_has_room(stint, sweeps):
    """And the gate is a budget rather than a switch: given room, it runs."""
    session, tx, grid = stint
    for n in range(CYCLES):
        held_cycle(session, tx, grid, n, room_s=ROOM_S)
    assert len(sweeps) == CYCLES


def test_the_reserve_covers_the_search_it_guards():
    """5.86 ms median over the receive windows of `captures/onair-0914-0850`."""
    assert onair._SessionRx.P3_ANSWER_ACQUIRE_RESERVE_S * 1e3 >= 5.9
    assert (onair._SessionRx.P3_ANSWER_ACQUIRE_RESERVE_S
            < onair._SessionRx.CHANGEOVER_BODY_RESERVE_S)


def test_one_transform_a_window_across_both_reads(stint, sweeps, monkeypatch):
    """The packet read and the codeword read share the cycle's own transform."""
    session, tx, grid = stint
    transformed: list[tuple[int, int]] = []
    real = p3acquire.compensate

    def counted(audio, hz):
        transformed.append((audio.ctypes.data, audio.size))
        return real(audio, hz)

    monkeypatch.setattr(p3acquire, "compensate", counted)
    windows = [held_cycle(session, tx, grid, n, room_s=ARM_ROOM_S)
               for n in range(CYCLES)]
    assert transformed == [(w.ctypes.data, w.size) for w in windows]
    # ...and what the readers are handed is the transform they would have run.
    assert np.array_equal(session.rx._corrected(windows[-1]),
                          real(windows[-1], RECEIVE_OFFSET_HZ))


def test_two_windows_of_one_length_are_not_one_window(stint, monkeypatch):
    """The key is the buffer, not its shape: a second window pays for itself."""
    session, tx, grid = stint
    runs: list[float] = []
    real = p3acquire.compensate
    monkeypatch.setattr(p3acquire, "compensate",
                        lambda audio, hz: runs.append(hz) or real(audio, hz))
    first, second = peer_window(0), peer_window(1)
    assert not np.array_equal(first, second)
    for window in (first, second, second, first):
        session.rx._corrected(window)
    assert len(runs) == 3
