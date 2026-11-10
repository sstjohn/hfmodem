# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The constellation angle read off the field instead of off the header block.

At speed level 2 the six carriers stand +0.3 to +1.8 dB over the median channel
of WS8EOC's 80 m arm of 2026-09-13, and the eight-symbol block in front of the
field is scored on four constant words out of sixteen. What comes back is a
magnitude that clears nothing and an angle that is mostly noise: over the arm's
35 level 2 cycles the block reports +24, -135, +25, -129, +9 deg and lands
within 20 deg of the session's own 43 on eight of them.

The field is the better instrument and it was there all along. Raising each
differential cell to as many powers as it has states folds the data out of it --
`ledger-80m`'s own estimator, turned on the angle rather than the frequency --
which puts 432 cells behind the reading instead of 48, each weighing what its
amplitude is worth. It lands within 20 deg on 32 of the 35, and the magnitude of
the same sum is what says which way round the carrier swap fell.

That is what makes the repeats combinable: copies in one cell order and on one
set of axes add, and copies that disagree about either cancel.
"""

import numpy as np
import pytest

from hfmodem.shrike import p3acquire, p3frame, p3rx, placement, rx, rxfront, spec
from hfmodem.tests import evidence

ARM = evidence.CAPTURES / "onair-0913-2135"
STREAM = ARM / "stream.wav"
FS, SPS = rxfront.FS, rxfront.SPS

PEER0, CYCLE_N, BURST_N, PRE_N = 588766, 60000, 38880, 2400
ROW0_SL2 = 1792917
OFFSET_HZ = -6.1
SESSION_ROT = np.deg2rad(43.0)
"""Where this arm's constellation sat, to the degree the decodes agree on."""

GREETING = b"minutes remaining with "

pytestmark = pytest.mark.skipif(
    not STREAM.exists(), reason=f"80 m arm recording absent: {STREAM}")


@pytest.fixture(scope="module")
def stream():
    return rxfront.load_wav(str(STREAM))


def _window(stream, k):
    s0 = PEER0 + CYCLE_N * k
    return p3acquire.compensate(stream[s0 - PRE_N:s0 + BURST_N + PRE_N], OFFSET_HZ)


def _row0(k):
    return ROW0_SL2 + (k - 20) * CYCLE_N - (PEER0 + CYCLE_N * k - PRE_N)


def _baseband(audio, at, path):
    """The field's sparse baseband, as `_level_at_lock` gathers it."""
    sync = rxfront.SyncedRx()
    delay = (rxfront._matched_filter().size - 1) // 2
    lead = p3frame.DATA_OFFSET * SPS
    cand = sync._candidates(audio, at,
                            rxfront._packet_span(rxfront._frame_span(path)),
                            SPS, lead)
    grid = np.arange(-1, path.n_symbols + len(cand) - 1) * SPS
    return rxfront._sampled_baseband(audio, path.tones,
                                     sync._index(cand, path, delay, grid))


def _both_arrangements(stream, k):
    """(rotation, weight) for each carrier ordering at the lock."""
    audio, at = _window(stream, k), _row0(k)
    Z = _baseband(audio, at, placement.SPEED_PATHS[2])
    out = {}
    for swapped in (False, True):
        path = p3rx.path_for(2, p3frame.PacketHeader(0, 2, 1.0, 0.0, swapped))
        out[swapped] = p3rx.field_rotation(Z, at, path, 0.0, fs=FS)
    return out


def _block_rot(stream, k):
    """The angle the header block reports, read as `_level_at_lock` reads it."""
    audio, at = _window(stream, k), _row0(k)
    home = placement.SPEED_PATHS[2]
    delay = (rxfront._matched_filter().size - 1) // 2
    lead = p3frame.DATA_OFFSET * SPS
    sync = rxfront.SyncedRx()
    cand = sync._candidates(audio, at,
                            rxfront._packet_span(rxfront._frame_span(home)),
                            SPS, lead)
    step = SPS // 4
    head = rxfront._sampled_baseband(
        audio, home.tones,
        sync._index(cand, home, delay,
                    np.arange(-lead, (len(cand) - 1) * SPS, step)))
    return p3rx.header_of(head, range(cand.start, cand[-1] + 1, step),
                          home, fs=FS).rot


def _off(rot, by=SESSION_ROT):
    """Degrees between two angles, inside a half turn."""
    return abs(np.degrees(np.angle(np.exp(1j * (rot - by)))))


def test_the_field_agrees_on_one_angle_where_the_block_does_not(stream):
    """The measurement the round rests on, over all 35 level 2 cycles."""
    field = [max(_both_arrangements(stream, k).values(), key=lambda t: t[1])[0]
             for k in range(20, 55)]
    block = [_block_rot(stream, k) for k in range(20, 55)]

    assert sum(_off(r) <= 20 for r in field) >= 32
    assert sum(_off(r) <= 20 for r in block) <= 10
    # ...and on the cycles the packet is readable in it is tighter than that.
    readable = [field[k - 20] for k in (20, 22, 23, 26, 28, 49, 52, 54)]
    assert max(_off(r) for r in readable) <= 5


def test_the_heavier_arrangement_is_the_one_the_frame_decodes_on(stream):
    """The swap, which a 0.42-to-0.55 variable-header fit cannot name.

    Every one of the eight cycles whose field passes a CRC at the lock does so
    on one carrier ordering and not the other; the field's own sum is heavier
    on that one in all eight, by 1.6x to 10x.
    """
    sent = {20: False, 22: False, 23: True, 26: False,
            28: False, 49: True, 52: False, 54: False}
    for k, swapped in sent.items():
        both = _both_arrangements(stream, k)
        assert both[swapped][1] > both[not swapped][1], k
        assert both[swapped][1] / both[not swapped][1] >= 1.5, k


def test_the_turn_is_the_one_thing_the_field_cannot_say(stream):
    """The fold costs a half circle, and `near` is what buys it back."""
    path = placement.SPEED_PATHS[2]
    audio, at = _window(stream, 28), _row0(28)
    Z = _baseband(audio, at, path)
    near_right, _ = p3rx.field_rotation(Z, at, path, SESSION_ROT, fs=FS)
    near_wrong, _ = p3rx.field_rotation(Z, at, path, SESSION_ROT + np.pi, fs=FS)
    assert _off(near_right) <= 5
    assert _off(near_wrong - np.pi) <= 5
    # DBPSK folds on the half turn; the DQPSK levels fold on the quarter.
    assert placement.SPEED_PATHS[2].bits_per_cell == 1
    assert placement.SPEED_PATHS[6].bits_per_cell == 2


def test_noise_carries_no_weight(stream):
    """The magnitude is a reading of the frame, not of the buffer's loudness."""
    rng = np.random.default_rng(0)
    path = placement.SPEED_PATHS[2]
    at = p3frame.DATA_OFFSET * SPS + SPS
    worst = 0.0
    for _ in range(40):
        audio = rng.normal(0, 0.05, BURST_N + 2 * PRE_N)
        _, weight = p3rx.field_rotation(_baseband(audio, at, path), at, path,
                                        0.0, fs=FS)
        worst = max(worst, weight)
    arm = min(max(v[1] for v in _both_arrangements(stream, k).values())
              for k in range(20, 55))
    assert worst < arm / 10


@pytest.mark.parametrize("sl", sorted(placement.SPEED_PATHS)[1:])
def test_a_packet_on_the_published_axes_reads_as_no_rotation_at_all(sl):
    """The fold's reference, at both cell widths.

    Rendered by the transmit path and read back where it was keyed, every level
    above 1 returns under a degree -- which is the constant the fold divides
    out being right for the two-state levels and the four-state ones alike.
    """
    path = placement.SPEED_PATHS[sl]
    info = b"T" * (path.crc_bytes - 3) + bytes([spec.status_byte(1)])
    audio = np.concatenate([np.zeros(FS), placement.data_packet(info, path),
                            np.zeros(FS)])
    row0 = (FS + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * SPS)
    rot, weight = p3rx.field_rotation(_baseband(audio, row0, path), row0, path,
                                      0.0, fs=FS)
    assert abs(np.degrees(rot)) < 1.0
    assert weight > 0.5


def test_a_combined_field_may_answer_by_losing_its_oldest_copy(stream):
    """`FieldMemory`'s second corroboration, on the arm's own repeats.

    Four consecutive copies of the peer's greeting sum to a CRC-valid field
    that the clock shift cannot follow -- an eighth of a symbol is more margin
    than the sum has. Dropping the oldest copy is a different set of air and
    the same field comes back.
    """
    copies, path = [], None
    for k in range(30, 34):
        audio, at = _window(stream, k), _row0(k)
        Z0 = _baseband(audio, at, placement.SPEED_PATHS[2])
        (rot, _), header = max(
            ((p3rx.field_rotation(Z0, at, p3rx.path_for(2, h), SESSION_ROT,
                                  fs=FS), h)
             for h in (p3frame.PacketHeader(0, 2, 1.0, 0.0, False),
                       p3frame.PacketHeader(0, 2, 1.0, 0.0, True))),
            key=lambda t: t[0][1])
        path = p3rx.path_for(2, header)
        Z = _baseband(audio, at, path)
        copies.append({off: p3rx._cells(audio, at + off, path, rot, fs=FS, Z=Z)
                       for off in (-p3rx.CONFIRM_STEP, 0, p3rx.CONFIRM_STEP)})

    def summed(off, of):
        field, ok = rx.decode_frame_softs(
            placement.deinterleave(np.sum([c[off] for c in of], axis=0), path),
            path)
        return field if ok else None

    memory = p3rx.FieldMemory()
    fields = [memory.add(c, path) for c in copies]
    # Two copies never answer this way: the sum minus its oldest is the single
    # frame that already failed its own CRC.
    assert fields[0] is None and fields[1] is None
    delivered = [f for f in fields if f is not None]
    assert delivered
    assert all(p3rx.packet_of(f, path, 0).payload == GREETING
               for f in delivered)
    # ...and the clock could not have said so. The sum that was admitted is
    # CRC-valid at its own instant and nowhere an eighth of a symbol away.
    n = fields.index(delivered[0]) + 1
    assert summed(0, copies[:n]) == delivered[0]
    assert all(summed(off, copies[:n]) is None
               for off in (-p3rx.CONFIRM_STEP, p3rx.CONFIRM_STEP))
    assert summed(0, copies[1:n]) == delivered[0]
