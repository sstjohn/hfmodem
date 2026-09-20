# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One field, one memory: the tracked read and the sweep combine the same copies.

The peer sends ONE run of copies and two readers see it a cycle at a time
between them -- `rxfront.SyncedRx.packet` on the cycles the lock answers,
`rxfront.decode_expected_packet` on the ones it misses. Round 33 gave the
tracked read a `p3rx.FieldMemory` of its own, so each reader held a fraction of
one field and the sum was in neither. `onair._SessionRx` now hands the tracked
reader the session's own memory, which is what these cases are about: a
delivery that needs a copy from each reader, and nothing delivered twice.

The audio is WS8EOC's 80 m arm of 2026-09-13 at the geometry
`test_sl2_rx_tracked.py` holds. Its own live session never fed the memory from
both readers, and the measurement says why: at the session's correction not one
of the 35 level 2 blocks reaches an anchor gate, so the sweep has no copy to
give. The one window that anchors does so at a correction 6.4 Hz off the peer
-- the one the 23:20 arm acquired -- and that is the window the combining case
reads the sweep's copy from. The order the copies are presented in is this
file's: `FieldMemory` groups by consecutive failure and knows nothing of cycle
numbers.
"""

import pytest

from hfmodem.shrike import onair, p3acquire, p3frame, p3rx, rx, rxfront, spec
from hfmodem.tests.shrike.test_sl2_rx_tracked import (
    BURST_N, CYCLE_N, GREETING, HELD, OFFSET_HZ, PEER0, PRE_N, ROW0_SL2,
    STREAM)

FS = onair.FS
MISCORRECTED_HZ = -12.9
"""The 23:20 arm's acquired correction, 6.4 Hz off this peer's carriers."""


pytestmark = pytest.mark.skipif(
    not STREAM.exists(), reason=f"80 m arm recording absent: {STREAM}")


@pytest.fixture(scope="module")
def stream():
    return rxfront.load_wav(str(STREAM))


def _window(stream, k, hz=OFFSET_HZ):
    s0 = PEER0 + CYCLE_N * k
    return p3acquire.compensate(stream[s0 - PRE_N:s0 + BURST_N + PRE_N], hz)


def _row0(k):
    return ROW0_SL2 + (k - 20) * CYCLE_N - (PEER0 + CYCLE_N * k - PRE_N)


def _tracked(sync, stream, k, hz=OFFSET_HZ):
    sync.packet_at, sync.packet_level = _row0(k), 2
    return sync.packet(_window(stream, k, hz))


def _sweep(stream, k, memory, hz=OFFSET_HZ):
    return rxfront.decode_expected_packet(_window(stream, k, hz), memory)


def _combining_run(stream, memory, sync):
    """Cycle 31's copy through the sweep, then three the tracked read misses.

    The lead read is what gives the session its angle, which is what puts every
    copy behind it on one set of axes."""
    assert _tracked(sync, stream, 28, MISCORRECTED_HZ) is not None
    assert _sweep(stream, 31, memory, MISCORRECTED_HZ) is None
    return [_tracked(sync, stream, k, MISCORRECTED_HZ) for k in (41, 42, 49)]


def test_the_shared_memory_delivers_a_field_neither_reader_reached(stream):
    memory = p3rx.FieldMemory()
    got = _combining_run(stream, memory, rxfront.SyncedRx(memory=memory))
    assert [ev is None for ev in got] == [True, True, False]
    assert got[-1].packet[0] == 2
    assert got[-1].packet[2] == GREETING
    assert "combined" in got[-1].text


def test_the_two_memories_of_round_33_reach_none_of_it(stream):
    """The same four windows, each reader holding its own copies."""
    memory = p3rx.FieldMemory()
    got = _combining_run(stream, memory, rxfront.SyncedRx())
    assert got == [None, None, None]


def test_the_combined_field_is_delivered_once(stream):
    memory = p3rx.FieldMemory()
    sync = rxfront.SyncedRx(memory=memory)
    assert _combining_run(stream, memory, sync)[-1] is not None
    # Delivered, so the run of copies is over -- for both readers, which is what
    # one memory between them means.
    assert not memory._copies
    assert _tracked(sync, stream, 49, MISCORRECTED_HZ) is None
    assert _sweep(stream, 31, memory, MISCORRECTED_HZ) is None


def test_the_session_reads_into_its_own_memory():
    rxs = onair._SessionRx(_StubHost(), tag="TEST")
    assert rxs.sync.memory is rxs.p3_memory


def test_sharing_the_memory_reads_the_arm_exactly_as_it_read_before(stream):
    """The 35 level 2 cycles, tracked read first and the sweep behind it."""
    memory = p3rx.FieldMemory()
    sync = rxfront.SyncedRx(memory=memory)
    reads = {}
    for k in range(20, 55):
        ev = _tracked(sync, stream, k)
        if ev is None:
            ev = _sweep(stream, k, memory)
            if ev is not None:
                sync.observe(ev)
        else:
            memory.clear()
        if ev is not None:
            reads[k] = ev.packet
    assert set(reads) == {20, 22, 26, 28, 34, 48, 52}, reads
    assert len(set(reads) & HELD) == 6
    assert all(reads[k][2] == GREETING for k in (22, 26, 28, 34, 48, 52))


def test_the_sweep_has_no_copy_to_give_at_the_arms_own_correction(stream):
    """Why the reads above are unchanged: nothing anchors, so nothing is fed.

    An anchor is what puts a copy in the memory from the sweep's side, and at
    level 2 the block is under both gates on every cycle of this arm. At a
    correction 6.4 Hz off the peer exactly one of the 35 clears `p3rx.VH_FIT`,
    a tenth of a symbol from the frame's own row 0."""
    def anchors(k, hz):
        audio = _window(stream, k, hz)
        pulse = rx._pulse(FS // 100)
        Z = {cn: rx._baseband(audio, cn, FS, pulse)
             for cn in range(spec.N_CHANNELS)}
        return (p3rx.header_anchors(Z, len(audio), fs=FS)
                + p3rx.vh_anchors(Z, len(audio), fs=FS))

    assert [k for k in range(20, 55) if anchors(k, OFFSET_HZ)] == []
    off = {k: anchors(k, MISCORRECTED_HZ) for k in range(20, 55)}
    assert [k for k, h in off.items() if h] == [31]
    head, = off[31]
    assert head.fit > p3rx.VH_FIT
    assert 2 in head.levels
    assert abs(head.at + p3frame.DATA_OFFSET * rxfront.SPS
               - _row0(31)) < rxfront.SPS / 4


class _StubArq:
    state = onair.State.CONNECTING
    role = "irs"


class _StubHost:
    protocol = onair.Protocol.PACTOR3
    sent_total = 0
    rcvd_total = 0
    peer = None

    def __init__(self) -> None:
        self.arq = _StubArq()

    def on_rx_event(self, ev) -> None:
        pass
