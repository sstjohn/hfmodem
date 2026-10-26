# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A listener joining mid-QSO cannot assume the peer's pair history."""
import numpy as np
import pytest

from hfmodem.kestrel.rx import bw500_locator as loc, varahf500 as rx
from hfmodem.kestrel.tx import varahf500_tx as tx


def body(level, value):
    payload = bytes([value]) * rx._PAYLOAD[level]
    return tx.build_frame(payload, 0x95) if level == 4 else tx.build_l3_frame(payload, 0x95, level)


def read(x, layout=(24000,)):
    receiver = loc.Receiver(observe_pairs=True)
    overs = []
    at = i = 0
    while at < len(x):
        stop = at + layout[i % len(layout)]
        overs += receiver.feed(x[at:stop])
        at = stop
        i += 1
    overs += receiver.feed(np.empty(0), eof=True)
    return receiver, overs


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('layout', [(24000,), (73, 1024, 5000)])
def test_observed_genuine_pair_is_published_once(level, layout):
    bodies = [body(level, n) for n in (31, 73)]
    x = np.pad(tx.synth_burst(bodies, onset=24000, level=level), (0, 48000))
    receiver, overs = read(x, layout)
    assert [f.frame_bytes for o in overs for f in o.frames] == bodies
    assert any(o.result.complete for o in overs)
    assert receiver.overflows == receiver.expired == 0


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_single_does_not_consume_following_different_level(level):
    other = level % 4 + 1
    bodies = [body(level, 31), body(other, 73)]
    x = np.concatenate([tx.synth_burst([bodies[0]], onset=24000, level=level),
                        np.zeros(24000),
                        tx.synth_burst([bodies[1]], onset=24000, level=other),
                        np.zeros(48000)])
    receiver, overs = read(x)
    assert [f.frame_bytes for o in overs for f in o.frames] == bodies
    assert not overs[0].result.complete  # An unseen partner remains uncertain.
    assert receiver.overflows == 0


def test_same_payload_in_later_over_is_not_deduplicated():
    frame = body(2, 31)
    one = tx.synth_burst([frame], onset=24000, level=2)
    _, overs = read(np.concatenate([one, np.zeros(24000), one, np.zeros(48000)]))
    assert [f.frame_bytes for o in overs for f in o.frames] == [frame, frame]


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_erased_partner_does_not_hide_later_over(level):
    frames = [body(level, value) for value in (31, 73)]
    pair = tx.synth_burst(frames, onset=24000, level=level)
    rec = rx.INDEX_RECORDS.get(level)
    boundary = 24000 + ((rec.lead + rec.ncols) * rec.dw if rec else (9 + rx.NSYM) * rx.H)
    pair[boundary:] = 0
    later = body(level, 91)
    x = np.concatenate([pair, np.zeros(24000),
                        tx.synth_burst([later], onset=24000, level=level), np.zeros(48000)])
    _, overs = read(x)
    assert [f.frame_bytes for o in overs for f in o.frames] == [frames[0], later]
    assert not overs[0].result.complete


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_eof_after_first_frame_keeps_text_without_claiming_complete(level):
    frame = body(level, 31)
    _, overs = read(tx.synth_burst([frame], onset=24000, level=level))
    assert [f.frame_bytes for o in overs for f in o.frames] == [frame]
    assert not any(o.result.complete for o in overs)


def test_expiry_preserves_valid_first_frame_without_owning_speculative_tail():
    receiver = loc.Receiver(observe_pairs=True)
    receiver.raw.feed(np.zeros(receiver.raw.capacity))
    frame = rx.FrameResult(0, (0,), 4, 1., True, body(2, 31), level=2, frames_in_burst=2)
    receiver.pending = [loc.Over(loc.Candidate(2, 0, 24., 0, 0),
                                 frames=[frame], first_end=10000, end=20000, pair=True)]
    overs = receiver.feed(np.zeros(loc.BLOCK), batches=[])
    assert [f.frame_bytes for o in overs for f in o.frames] == [frame.frame_bytes]
    assert not overs[0].result.complete
    assert receiver.expired == 1
    assert receiver._owned(9000) and not receiver._owned(15000)


def test_candidate_limit_keeps_validated_text_and_four_candidate_bound(monkeypatch):
    receiver = loc.Receiver(observe_pairs=True)
    frame = rx.FrameResult(0, (0,), 4, 1., True, body(2, 31), level=2, frames_in_burst=2)
    receiver.pending = [loc.Over(loc.Candidate(2, i * 20000, 24., 0, 0),
                                 frames=[frame] if i == 0 else [],
                                 first_end=10000 if i == 0 else 0,
                                 next_check=1000000) for i in range(4)]
    monkeypatch.setattr(receiver.grouper, 'feed',
                        lambda *a, **k: [loc.Candidate(3, 90000, 24., 0, 0)])
    overs = receiver.feed(np.zeros(loc.BLOCK), batches=[])
    assert [f.frame_bytes for o in overs for f in o.frames] == [frame.frame_bytes]
    assert not overs[0].result.complete
    assert len(receiver.pending) == 4 and receiver.overflows == 1
