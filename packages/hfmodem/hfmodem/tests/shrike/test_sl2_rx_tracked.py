# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Speed level 2 at a held lock, on the 80 m arm that stopped reading it.

WS8EOC answered our CS4 with `SL2 DATA 23B` and then repeated one packet --
`b'minutes remaining with '` -- for thirty-three consecutive cycles while a
witness 160 miles from the gateway decoded every one of them and this station
read one. The offset was not the reason and neither was the band: the peer's
carriers measure -6.5 +- 0.6 Hz off nominal across both speed levels, the
session tracked -6.1, and the readers apply it before anything else runs.

What stopped is the ANCHOR. Level 2 lights six of the eighteen channels, so it
carries four of the sixteen constant header words and can never reach
`p3rx.HEADER_FIT`, which is scored over all sixteen; and it puts a sixth of the
burst's power on the two variable-header carriers where level 1 puts a half, so
the fit read there lands on `p3rx.VH_FIT` rather than over it -- the one frame
the arm read anchored at 0.900 against a gate of 0.90, and the 33 repeats behind
it scored 0.72-0.89. On the tracked path, where the block is read on the level's
own six channels, 29 of the 35 sit under `HEADER_FIT` and the level 1 frames
before them clear their own gate. A tracked read does not need one --
the position is the session's own clock and the block is being asked only for
the swap, the cycle length and the angle -- and taking the below-gate reading
there instead of assuming home order at zero rotation is worth two more reads of
that greeting block at no measured cost.

The swap and the angle then stopped coming from the block at all. Both are read
off the FIELD by `p3rx.field_rotation` -- 432 cells against the block's 48 --
which is what `test_sl2_rotation_field.py` measures, and it is what makes the
repeats combinable: the counts here go 3 to 5 single-shot and to 7 with the
memory carried, the two extra being sums of the copies behind them.

The second half of the file is the pre-key ledger's: a CS3 read at the tracked
anchor demodulates the 0.84 s packet behind it, on both carrier arrangements,
in front of a key that has 7.6 ms of notice -- and it ran twice on the same
audio and the same anchor, once from each reader that aims there.
"""

import numpy as np
import pytest

from hfmodem.shrike import onair, p3acquire, p3frame, p3rx, placement, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_answer_band import _restore, _stems, D_MAX_N

ARM = evidence.CAPTURES / "onair-0913-2135"
STREAM = ARM / "stream.wav"
FS = onair.FS

# The arm's own geometry, off `working/pactor3-header-0913/ledger-80m`: the
# peer's burst opens at `PEER0 + 60000k` and its packet's grid row 0 sits where
# the session decoded one. Cycles 3-19 carry speed level 1, 20-54 level 2.
PEER0, CYCLE_N, BURST_N, PRE_N = 588766, 60000, 38880, 2400
ROW0_SL1, ROW0_SL2 = 892895, 1792917
OFFSET_HZ = -6.1
GREETING = b"minutes remaining with "

# Cycles the arm kept a listen window for; the rest it was keying in. A read on
# one of these is a read the live session could have had.
HELD = {20, 21, 22, 23, 26, 27, 30, 31, *range(34, 45), *range(46, 55)}

pytestmark = pytest.mark.skipif(
    not STREAM.exists(), reason=f"80 m arm recording absent: {STREAM}")


def _window(stream, k):
    """One cycle of the peer's raster, compensated as the session did."""
    s0 = PEER0 + CYCLE_N * k
    return p3acquire.compensate(stream[s0 - PRE_N:s0 + BURST_N + PRE_N], OFFSET_HZ)


def _row0(k, anchor, first):
    """Where the session's own projection puts this cycle's grid row 0."""
    return anchor + (k - first) * CYCLE_N - (PEER0 + CYCLE_N * k - PRE_N)


def _read(stream, ks, level, anchor, first, sync=None):
    """The tracked reader over those cycles, aimed the way `_p3_packet` aims it."""
    sync = sync or rxfront.SyncedRx()
    out = {}
    for k in ks:
        sync.packet_at, sync.packet_level = _row0(k, anchor, first), level
        ev = sync.packet(_window(stream, k))
        if ev is not None:
            out[k] = (ev.packet, ev.text)
    return {k: v[0] for k, v in out.items()}, {k: v[1] for k, v in out.items()}


@pytest.fixture(scope="module")
def stream():
    return rxfront.load_wav(str(STREAM))


def test_the_tracked_reader_takes_level_2_frames_the_arm_read_nothing_in(stream):
    got, how = _read(stream, range(20, 55), 2, ROW0_SL2, 20)
    assert set(got) == {20, 22, 26, 28, 34, 48, 52}, got
    assert all(p[0] == 2 for p in got.values())
    assert got[20][2].startswith(b"0\r\nW9SSJ has")
    # The block the peer then repeated for thirty-three cycles, and the reads
    # the live session never made: five of these six are cycles it held a
    # window for, so they are reads it could have had.
    assert all(got[k][2] == GREETING for k in (22, 26, 28, 34, 48, 52))
    assert {22, 26, 34, 48, 52} <= HELD
    # Two of them are the soft sum of the copies behind them; the rest stand
    # alone at the lock.
    assert {k for k, t in how.items() if "combined" in t} == {34, 48}


def test_the_level_1_phase_reads_exactly_what_it_read_before(stream):
    got, _ = _read(stream, range(3, 20), 1, ROW0_SL1, 5)
    assert set(got) == {3, 4, 5, 6, 13, 14, 15, 16, 17, 19}, got
    assert all(p[0] == 1 for p in got.values())
    assert [got[k][2] for k in sorted(got)] == [b" Trim"] * 4 \
        + [b"ode 1"] * 3 + [b".4.3."] * 3


def test_the_recovered_frames_are_the_ones_whose_block_is_under_the_gate(stream):
    """The measurement the substitution rests on, at each level's own comb."""
    def fit(k, sl, anchor, first):
        path = placement.SPEED_PATHS[sl]
        audio = _window(stream, k)
        row0 = _row0(k, anchor, first)
        delay = (rxfront._matched_filter().size - 1) // 2
        starts = range(row0 - 2 * rxfront.SPS, row0 + 2 * rxfront.SPS + 1,
                       rxfront.SPS // 4)
        idx = rxfront.SyncedRx()._index(
            range(starts.start, starts.stop), path, delay,
            np.arange(-p3frame.DATA_OFFSET * rxfront.SPS,
                      4 * rxfront.SPS, rxfront.SPS // 4))
        Z = rxfront._sampled_baseband(audio, path.tones, idx)
        head = p3rx.header_of(Z, starts, path, fs=FS)
        return 0.0 if head is None else head.fit

    gate = p3rx.anchor_gate(placement.SPEED_PATHS[2])
    sl2 = {k: fit(k, 2, ROW0_SL2, 20) for k in range(20, 55)}
    # The two reads the substitution buys are the two whose block is under the
    # gate; the one the arm could already have had is the one over it.
    assert sl2[20] < gate and sl2[26] < gate
    assert sl2[28] >= gate
    assert sum(f < gate for f in sl2.values()) >= 29
    # ...against the level 1 phase, scored on the two channels it lights whole.
    sl1 = [fit(k, 1, ROW0_SL1, 5) for k in (3, 5, 6, 14, 17, 19)]
    assert min(sl1) >= p3rx.anchor_gate(placement.SPEED_PATHS[1])


@pytest.mark.parametrize("seed", range(6))
def test_a_below_gate_block_does_not_make_a_level_2_packet_out_of_noise(seed):
    """The trial the substitution adds, offered audio with no PACTOR in it.

    Same reader, same aim, same level preference -- 240 windows of white noise
    at the level the arm's own channel sat at. The header block below its gate
    describes a frame; the CRC is still what admits one.
    """
    rng = np.random.default_rng(seed)
    sync = rxfront.SyncedRx()
    at = p3frame.DATA_OFFSET * rxfront.SPS + rxfront.SPS
    for _ in range(40):
        sync.packet_at, sync.packet_level = at, 2
        assert sync.packet(rng.normal(0, .05, BURST_N + 2 * PRE_N)) is None


def test_the_changeover_body_is_not_decoded_when_the_key_is_close(stream, monkeypatch):
    """`details=False` yields the link on the codeword and spends nothing else."""
    seg = p3acquire.compensate(stream[PEER0 - PRE_N:PEER0 + BURST_N + PRE_N],
                               OFFSET_HZ)
    calls = []
    inner = p3rx.decode_changeover_details
    monkeypatch.setattr(p3rx, "decode_changeover_details",
                        lambda *a, **kw: (calls.append(1), inner(*a, **kw))[1])

    sync = rxfront.SyncedRx()
    full = sync.control_signal_at(seg, PRE_N)
    assert full is not None and full.cs == placement.BREAKIN_CS
    assert full.packet is not None and full.packet[2] == b"RMS"
    assert len(calls) == 1

    bare = sync.control_signal_at(seg, PRE_N, details=False)
    assert bare is not None and bare.cs == placement.BREAKIN_CS
    assert bare.kind == "cs" and bare.packet is None
    assert len(calls) == 1


def test_one_anchor_is_decoded_once_a_cycle(stream, monkeypatch):
    """The two readers aimed at the same instant, charged once.

    `deep_scan`'s tracked-changeover branch and `_p3_cs`'s anchored read are
    both aimed at the codeword the grid predicts, in the same cycle and over the
    same audio. 111 calls in 93 cycles of `captures/onair-0913-2152`; the second
    cannot learn what the first did not.
    """
    calls = []
    inner = p3rx.decode_changeover_details
    monkeypatch.setattr(p3rx, "decode_changeover_details",
                        lambda *a, **kw: (calls.append(1), inner(*a, **kw))[1])
    seg = p3acquire.compensate(stream[PEER0 - PRE_N:PEER0 + BURST_N + PRE_N],
                               OFFSET_HZ)
    sessrx = onair._SessionRx.__new__(onair._SessionRx)
    sessrx._p3_word_read = None
    ev = rxfront.SyncedRx().control_signal_at(seg, PRE_N)
    assert ev is not None and len(calls) == 1

    sessrx._p3_word_read = (PEER0, PEER0 - PRE_N + ev.start, ev)
    again = sessrx._p3_read_again(PEER0 - PRE_N, PRE_N)
    assert again is not None and again.cs == ev.cs
    # The instant the codeword was FOUND on, not the one it was aimed at: it is
    # what `_p3_answer_at` tracks the peer's answer clock by.
    assert again.packet == ev.packet and again.start == ev.start
    assert len(calls) == 1
    # ...and only at the instant it was read at.
    assert sessrx._p3_read_again(PEER0 - PRE_N, PRE_N + 4 * rxfront.SPS) is None


def test_the_peers_own_packet_is_not_reported_as_an_unattributed_occupant():
    """The seven prints of the arm, with the link's role in the question.

    Every one of them fell on a cycle the witness proves WS8EOC was transmitting
    its own `SL2 DATA 23B` in. An IRS owes a codeword and is owed a packet, so
    the window measured here is the peer's burst by construction -- the reading
    still holds the cycle, and it stops claiming the emission is unidentified.
    """
    stems = [s for s in _stems(ARM) if s.startswith("hold_")]
    if not stems:
        pytest.skip(f"80 m arm listen windows absent: {ARM}")
    said = {}
    for attributed in (False, True):
        band = onair._AnswerBand()
        lines = []
        for stem in stems:
            seen = band.sight(_restore(ARM, stem), 0, D_MAX_N, read=False,
                              answered=True, peer_owes_a_packet=attributed)
            if seen is not None and seen.occupied:
                lines.append(seen.line)
        said[attributed] = lines

    assert len(said[True]) == len(said[False]) >= 7
    assert all("ANSWER SLOT OCCUPIED" in line for line in said[False])
    assert all(line.startswith("PEER'S PACKET UNREAD") for line in said[True])
    assert all("not attributed" not in line for line in said[True])
    assert all("owes us its data packet" in line for line in said[True])
