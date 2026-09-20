# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ: reading a frame out of what its repeats agree on.

ARDOP resends an unacknowledged data frame under one frame type until it is
acked, so its repeats are the same bytes arriving again in different noise. The
reference keeps the carriers that decoded and angle-averages the differential
phases of those that did not (`SavePSKSamples`, `WeightedAngleAvg`,
`SaveFSKSamples`, SoundInput.c); besra keeps the same two things in
`besra.phy.memory`.

Everything memory ARQ produces still passes RS and the frame-type-bound CRC, so
these tests are as much about what it must *not* do — hold carriers across a rate
shift, a turnover, or another station's session — as about the frames it lifts.
The one measurement of what it is worth in decibels lives in
`working/besra-audit/memory_arq_gain.py`; the off-air demonstration is the
KE8LVA greeting in `test_offair_ke8lva_greeting.py`, unreadable on the
whole-capture pass under the old acquisition walk and read whole under this one.

What it must not do has a recording of its own at the foot of this file. Memory
that outlives the block it was read from does not fail an RS check on its way
out — a validated carrier put back is a carrier that already passed one — so
nothing downstream can catch it, and on 2026-08-29 it cost this station a
Winlink message that had been offered and accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora
from hfmodem.besra import radio as R
from hfmodem.besra.arq import session as S
from hfmodem.besra.host import protocol as P
from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy import memory
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.phy.demodulator import SAMPLE_RATE, Demodulator, frame_span

_SESSION = 0xAC
_TYPE = 0x51                     # 4PSK.500.100.O — two carriers, KN4LQN's own
_FSK_TYPE = 0x4B                 # 4FSK.500.100.O — one block of tone magnitudes


def _payload(ftype: int, seed: int = 0) -> bytes:
    fd = F.FRAMES[ftype]
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(32, 127, fd.k * fd.carriers, dtype=np.uint8))


def _arrival(ftype: int, payload: bytes, gain: float, seed: int) -> np.ndarray:
    """One transmission of ``payload`` in white noise ``gain`` times its own RMS."""
    frame = M.render_frame(ftype, payload, _SESSION).astype(np.float64)
    x = np.concatenate([np.zeros(2400), frame, np.zeros(4800)])
    rng = np.random.default_rng(seed)
    return x + gain * rng.normal(0.0, np.sqrt((frame ** 2).mean()), x.size)


def _read(demod: Demodulator, x: np.ndarray, ftype: int, want: bytes) -> bool:
    got = [f for f in demod.decode(x) if f.type == ftype]
    return bool(got) and got[0].ok and bytes(got[0].payload) == want


#: Noise at which no single arrival of a 4PSK.500.100 frame reads — pinned by
#: `test_the_arrivals_are_each_unreadable_alone` so the recoveries below cannot
#: quietly become fresh decodes if the demodulator gets more sensitive.
_UNREADABLE = 2.0


def test_the_arrivals_are_each_unreadable_alone():
    want = _payload(_TYPE)
    for seed in range(300, 306):
        assert not _read(Demodulator(), _arrival(_TYPE, want, _UNREADABLE, seed),
                         _TYPE, want)


def test_three_repeats_read_a_frame_no_one_of_them_carries():
    """The whole point: three arrivals, none readable, one frame out."""
    want = _payload(_TYPE)
    demod = Demodulator()
    read = [_read(demod, _arrival(_TYPE, want, _UNREADABLE, 300 + i), _TYPE, want)
            for i in range(3)]
    assert read[0] is False, "the first arrival has nothing to average against"
    assert any(read), f"three repeats and no decode: {read}"


def test_a_4fsk_frame_comes_back_the_same_way():
    """4FSK keeps per-symbol relative tone magnitudes rather than phases, and the
    normalisation is what makes the repeats comparable — see `memory._relative`."""
    want = _payload(_FSK_TYPE)
    demod = Demodulator()
    read = [_read(demod, _arrival(_FSK_TYPE, want, 2.6, 400 + i), _FSK_TYPE, want)
            for i in range(4)]
    assert read[0] is False and any(read), f"4FSK memory ARQ did nothing: {read}"


#: Two arrivals of one 4PSK.500.100 frame in noise light enough that a carrier
#: decodes and heavy enough that the other does not — seeds picked because they
#: fail on opposite carriers, which is the case `CarrierOk` exists for.
_SPLIT = (610, 627)


def test_a_frame_is_assembled_from_carriers_that_arrived_on_different_repeats():
    """The reference's `CarrierOk`: a carrier that passed RS and its CRC is
    finished, and a frame whose halves arrive on different transmissions is put
    together out of both. Neither arrival here carries the whole frame."""
    want = _payload(_TYPE)
    caps = [_arrival(_TYPE, want, 1.7, seed) for seed in _SPLIT]
    failed = []
    for cap in caps:
        got = [f for f in Demodulator().decode(cap) if f.type == _TYPE]
        assert got and not got[0].ok, "an arrival read whole"
        failed.append({c for c, r in got[0].soft.items() if r[0] != "good"})
    assert failed[0] and failed[1] and not (failed[0] & failed[1]), \
        f"the two arrivals lost the same carrier: {failed}"
    demod = Demodulator()
    assert not _read(demod, caps[0], _TYPE, want)
    assert _read(demod, caps[1], _TYPE, want), "the two halves were not assembled"


def test_nothing_is_kept_across_a_rate_shift():
    """A different frame type is a different block. Holding the old one's carriers
    across the shift would hand a fresh frame the phases of the frame before it."""
    want = _payload(_TYPE)
    demod = Demodulator()
    _read(demod, _arrival(_TYPE, want, _UNREADABLE, 300), _TYPE, want)
    assert demod.memory.repeats(0) == 1
    other = _payload(0x41)
    assert _read(demod, _arrival(0x41, other, 0.0, 310), 0x41, other)
    assert demod.memory.repeats(0) == 0, "the ladder rung did not clear the memory"
    demod.decode(_arrival(_TYPE, want, _UNREADABLE, 301))
    assert demod.memory.repeats(0) == 1, "carriers survived a round trip through 0x41"


def test_nothing_is_kept_across_a_turnover():
    """`ArqSession.rx_epoch` is the only reset a receiver cannot see for itself: the
    frame type and session are unchanged across a link turnover, and the block
    behind them is not."""
    want = _payload(_TYPE)
    epoch = [0]
    demod = Demodulator(rx_epoch=lambda: epoch[0])
    demod.decode(_arrival(_TYPE, want, _UNREADABLE, 300))
    assert demod.memory.repeats(0) == 1
    epoch[0] += 1
    demod.decode(_arrival(_TYPE, want, _UNREADABLE, 301))
    assert demod.memory.repeats(0) == 1, "the turnover did not clear the memory"


def test_an_arrival_re_read_is_not_a_second_repeat():
    """The live path decodes overlapping windows, so it reads each arrival around
    two dozen times. Counting those as repeats would give one transmission two
    dozen votes against the one the next transmission gets."""
    want = _payload(_TYPE)
    demod = Demodulator()
    once = _arrival(_TYPE, want, _UNREADABLE, 300)
    demod.decode(once, at=0)
    demod.decode(once, at=0)
    assert demod.memory.repeats(0) == 1


def test_an_arrival_behind_a_newer_one_is_not_folded_in():
    """A window that re-reads an old frame after a newer one has been read walks
    the frame type backwards, and a memory that followed the decode order rather
    than the stream would take the old frame for a fresh arrival of its type.

    That is the whole of the `20260819T024752Z` corruption: the handshake's second
    frame was re-acquired behind its third, its memory came back under the second's
    type, and the fourth frame — cut by the window edge, so nothing of it read —
    was answered whole out of it and reported as a clean decode of bytes that were
    never sent under that frame."""
    want = _payload(_TYPE)
    demod = Demodulator()
    first = _arrival(_TYPE, want, _UNREADABLE, 300)
    demod.decode(first, at=0)
    assert demod.memory.repeats(0) == 1
    other = _payload(0x41)
    demod.decode(_arrival(0x41, other, 0.0, 310), at=first.size)
    assert demod.memory.repeats(0) == 0
    demod.decode(first, at=0)
    assert demod.memory.repeats(0) == 0, "the older arrival came back behind a newer one"


def test_a_reading_older_than_the_isss_own_budget_is_not_a_repeat():
    """An ISS gives one block up after `ARQTimeout` of repeating it (ardopcf
    ARQ.c:462, 120 s), so a same-type frame further back than that is a different
    block wearing the same type, and its carriers must not be put back."""
    want = _payload(_TYPE)
    demod = Demodulator()
    assert not _read(demod, _arrival(_TYPE, want, _UNREADABLE, 300), _TYPE, want)
    got = [f for f in demod.decode(_arrival(_TYPE, want, _UNREADABLE, 301),
                                   at=121 * SAMPLE_RATE) if f.type == _TYPE]
    assert got and not got[0].ok, "a reading outlived the repeats it could belong to"


def test_a_repeat_still_inside_the_budget_is_folded_in():
    """The other side of the same bound: a gateway repeating into a long fade is
    still repeating one block, and dropping its earlier readings would cost the
    gain the memory exists for."""
    want = _payload(_TYPE)
    demod = Demodulator()
    assert not _read(demod, _arrival(_TYPE, want, _UNREADABLE, 300), _TYPE, want)
    got = [f for f in demod.decode(_arrival(_TYPE, want, _UNREADABLE, 301),
                                   at=100 * SAMPLE_RATE) if f.type == _TYPE]
    assert got and got[0].ok and bytes(got[0].payload) == want


def test_a_bare_control_frame_between_repeats_keeps_the_memory():
    """DATAACK, DATANAK, IDLE and the rest are short control frames, and the
    reference's `LastDataFrameType` never sees them — they arrive between a data
    frame and its repeat as a matter of course."""
    want = _payload(_TYPE)
    demod = Demodulator(expect_session=lambda: _SESSION)
    demod.decode(_arrival(_TYPE, want, _UNREADABLE, 300))
    assert demod.memory.repeats(0) == 1
    idle = M.render_frame(S.IDLE, b"", _SESSION).astype(np.float64)
    demod.decode(np.concatenate([np.zeros(2400), idle, np.zeros(4800)]))
    assert demod.memory.repeats(0) == 1
    assert _read(demod, _arrival(_TYPE, want, _UNREADABLE, 301), _TYPE, want), \
        "the IDLE cleared what it should not have"


def test_the_speculative_decodes_of_acquisition_are_not_repeats():
    """Acquisition decodes bodies for several header candidates and tone shifts and
    keeps one. Folding all of them in would average a frame against six readings of
    itself and call the result three repeats."""
    want = _payload(_TYPE)
    demod = Demodulator()
    demod.decode(_arrival(_TYPE, want, _UNREADABLE, 300))
    assert demod.memory.repeats(0) == 1 and demod.memory.repeats(1) == 1


def test_a_frame_from_another_session_is_not_a_repeat():
    good, other = _payload(_TYPE), _payload(_TYPE, seed=7)
    demod = Demodulator()
    demod.decode(_arrival(_TYPE, good, _UNREADABLE, 300))
    frame = M.render_frame(_TYPE, other, 0x5E).astype(np.float64)
    demod.decode(np.concatenate([np.zeros(2400), frame, np.zeros(4800)]))
    assert demod.memory.repeats(0) == 0, "a stranger's frame did not clear the memory"
    demod.decode(_arrival(_TYPE, good, _UNREADABLE, 301))
    assert demod.memory.repeats(0) == 1, "a stranger's frame counted as a repeat"


# -- the averages themselves -------------------------------------------------

def test_phases_are_averaged_as_angles_not_as_numbers():
    """Two readings either side of the wrap average to the wrap, not to zero."""
    got = memory._angle_avg(np.array([3000.0]), np.array([-3000.0]))
    assert abs(abs(float(got[0])) - 3141.6) < 1.0


def test_a_loud_repeat_does_not_outvote_a_quiet_one():
    """`WeightedAngleAvg` sums unit vectors: magnitude is absent despite the name,
    so a reading that arrived under a static crash carries one vote like any
    other. An average of the complex samples would let the crash decide."""
    frame = Demodulator().decode(
        M.render_frame(_TYPE, _payload(_TYPE), _SESSION).astype(np.float64))[0]
    frame.ok = False                      # a soft reading is only ever taken off one
    mem = memory.BlockMemory()
    quiet, loud = np.array([100.0]), np.array([1400.0])
    mem.psk(frame, 0, quiet, np.array([1.0, 1.0]))
    mem.keep(frame)
    frame.soft = {}
    # The repeat, on the grid an ISS actually repeats on: the frame itself and
    # then `ComputeInterFrameInterval` on top of it. Anything closer than the
    # frame's own length is this arrival read again, not another transmission.
    frame.offset += frame_span(_TYPE) + 2 * SAMPLE_RATE
    got, _ = mem.psk(frame, 0, loud, np.array([1e6, 1e6]))
    assert float(got[0]) == pytest.approx(750.0, abs=1.0)


def test_tone_magnitudes_are_scaled_before_they_are_averaged():
    """A repeat that arrived louder must not outvote a quieter one that read the
    tones more clearly, so each symbol's four magnitudes sum to one first."""
    tones = np.array([[4.0], [2.0], [1.0], [1.0]])
    rel = memory._relative(tones)
    assert rel.sum() == pytest.approx(1.0)
    assert np.allclose(memory._relative(tones * 1000.0), rel)


# -- the epoch the session publishes -----------------------------------------

def _session():
    class _Silent:
        def __getattr__(self, _):
            return lambda *a, **k: None

    class _Tx:
        def send(self, frame_type, payload, session_id):
            return 0.0

    return S.ArqSession("W9SSJ", transport=_Tx(), observer=_Silent())


def test_the_epoch_moves_where_the_reference_resets_memory():
    """Every one of the reference's five `ResetMemoryARQ` calls in ARQ.c sits under
    a `SetARDOPProtocolState(IRS)` — this end taking the receiving role, on a
    connect or a turnover. The teardown to DISC is the session ending."""
    s = _session()
    start = s.rx_epoch
    s._set_state(P.ArdopState.ISS)
    assert s.rx_epoch == start, "going to transmit is not a reset"
    s._set_state(P.ArdopState.IRS)
    assert s.rx_epoch == start + 1
    s._set_state(P.ArdopState.IDLE)
    s._set_state(P.ArdopState.IRS)
    assert s.rx_epoch == start + 2, "the turnover back to IRS did not move it"
    s._set_state(P.ArdopState.DISC)
    assert s.rx_epoch == start + 3


# -- through the decoder the station runs ------------------------------------

#: Headroom for `RollingDecoder` at the native rate, which hands the demodulator
#: its buffer as int16 (`_at_ardop_rate`). An arrival at `_UNREADABLE` peaks at
#: 112631 on the modulator's own scale and wraps there, which reads as a dead
#: receiver: nothing decodes in any window at all. The live path takes the card's
#: floats through `from_card`, which clips.
_ROOM = 0.2


class _CountsFolds(memory.BlockMemory):
    """A memory that says how many averages it handed back."""

    folded = 0

    def psk(self, frame, carrier, dphase, mag):
        got = super().psk(frame, carrier, dphase, mag)
        self.folded += got is not None
        return got


def _live_path(arrivals, gap_s: float = 2.0):
    """Push arrivals at `RollingDecoder` the way the capture thread does — the
    sound card's own 0.1 s blocks, a `pump` behind each, a `flush` at the end.

    The unit tests above hand the demodulator one buffer holding the repeats in
    order. This is the other regime and the only one the station has ever run in:
    overlapping windows that re-read every arrival around two dozen times, out of
    stream order, with a hold-back between decoding a frame and reporting it."""
    stream = [np.zeros(SAMPLE_RATE)]
    for arrival in arrivals:
        stream += [arrival, np.zeros(int(gap_s * SAMPLE_RATE))]
    au = np.concatenate(stream) * _ROOM
    demod = Demodulator(expect_session=lambda: _SESSION)
    demod.memory = _CountsFolds()
    got: list = []
    rx = R.RollingDecoder(demod, lambda pos, f: got.append(f), resample=False)
    block = SAMPLE_RATE // 10
    for i in range(0, au.size, block):
        rx.push(au[i:i + block].astype(np.float32))
        rx.pump()
    rx.flush()
    return demod, got


def test_one_transmission_is_never_averaged_against_itself():
    """The windows pin one arrival's leader edge a few samples differently as they
    slide over it, so the same transmission comes back at several positions. Held
    by exact position, each of those is an arrival in its own right and the memory
    votes with a second copy of the one reading it has.

    That is not a hazard the design anticipated and missed — it is invisible to a
    caller that decodes a buffer once, which is every test above. Replaying
    `20260826T133925Z` as the session heard it, 52 of the 81 averages it produced
    were an arrival voting with a second copy of its own reading. One arrival
    through the live path here produced 32."""
    demod, _ = _live_path([_arrival(_TYPE, _payload(_TYPE), _UNREADABLE, 300)])
    assert demod.memory.repeats(0) == 1, "one transmission, more than one reading"
    assert demod.memory.folded == 0, \
        f"{demod.memory.folded} averages against a single transmission"


def test_the_repeats_are_folded_through_the_rolling_windows():
    """The one thing the 2026-08-26 W6IDS link needed of memory ARQ happened here
    and nowhere else: three arrivals of one block, each unreadable alone, arriving
    through overlapping windows rather than down one buffer."""
    want = _payload(_TYPE)
    demod, got = _live_path([_arrival(_TYPE, want, _UNREADABLE, 300 + i)
                             for i in range(3)])
    read = [f for f in got if f.ok and bytes(f.payload) == want]
    assert demod.memory.folded, "no repeat was ever averaged"
    assert read, f"three repeats and nothing read: {[(f.name, f.ok) for f in got]}"


#: The 2026-08-29 18:54Z W6IDS link, at the stint where the CMS offered
#: `JWKY65C2OZES`: the gateway's BREAK to its first IDLE. Everything this station
#: keyed inside it is a bodyless control, which the memory takes no key from, so
#: the slice reproduces what the whole 302 s replay does at a ninth of the cost.
_W6IDS = evidence.LOGS / "onair" / "20260829T185417Z-besra-7060000.wav"
_W6IDS_LOG = evidence.WORKING / "onair-0829-1354" / "ardop-day-03-w6ids-third.log"
_STINT = (140.0, 212.0)


def _muted_stint() -> list[bytes]:
    """The 16-byte blocks W6IDS's proposal stint delivered, live path, in order.

    The mute is not optional and is why this reads the log: `RadioLink._capture`
    records every block the card delivers and pushes only the unmuted ones, so a
    replay that pushes the whole file hands the decoder 0.7 s of audio per keying
    that the session never saw, and its window grid — which is what decides the
    order arrivals are acquired in — is not the session's. Pushed whole, this
    stint does not splice at all.
    """
    rehear = corpora.harness("rehear.besra")
    sessionlog = corpora.harness("rehear.sessionlog")
    card = rehear._capture(_W6IDS)
    edges, _, level = rehear._key_edges(rehear.from_card(card, SAMPLE_RATE))
    log = sessionlog.read(_W6IDS_LOG)
    where = sessionlog.align(log, [a for a, _ in edges],
                             rehear._quieter(level, log.keyed))
    keyed = rehear._merge(sorted(rehear._on_this_recording(
        log.keyed, where.offset, card.size / R.FS_RADIO)))

    got: list = []
    rx = R.RollingDecoder(Demodulator(expect_session=lambda: 0x0D),
                          lambda _pos, f: got.append(f), resample=True)
    block = int(0.1 * R.FS_RADIO)
    for i in range(int(_STINT[0] * R.FS_RADIO), int(_STINT[1] * R.FS_RADIO), block):
        lo, hi = i / R.FS_RADIO, (i + block) / R.FS_RADIO
        if not any(a < hi and lo < b for a, b in keyed):
            rx.push(card[i:i + block])
        rx.pump()
    rx.flush()

    out, last = [], -1
    for f in got:                            # `ArqSession._receive_data`'s dedupe
        if f.ok and not f.header_only and len(f.payload or b"") == 16 \
                and f.type != last:
            last = f.type
            out.append(bytes(f.payload))
    return out


def test_a_block_the_peer_moved_on_from_is_not_put_back_into_the_next_one():
    """W6IDS, 2026-08-29 18:56Z: the proposal block, read as the station read it.

    The gateway sent `;PM: W9SSJ JWKY65C2OZES 524 saul.stjohn@gmail.com testing
    8/26 this is a test` in five `4PSK.200.100S` blocks, alternating `.E` and
    `.O`. The station delivered the first three and then a fourth that was the
    *second* over again: sixteen bytes already received took the place of the
    block carrying the end of the address and the start of the subject, and the
    `F>` that should have closed the proposal never arrived. W6IDS went to IDLE
    waiting for an `FS` there was nothing to answer with, and 524 bytes of mail
    stayed at the CMS.

    Nothing was wrong with the receiver while it happened — ten bodies decoded
    `ok=True` in that stint at quality 59 climbing to 84. The `.O` block the peer
    had moved on from was still in `BlockMemory` when its successor arrived,
    because the `.E` between the two was acquired out of order and reached
    `keep()` behind a frame that had already been reported.

    The run this recording can show ends at the address: the station acknowledged
    the invented block, so W6IDS never resent the real one and the subject is not
    in the file under any decode.
    """
    if not (_W6IDS.is_file() and _W6IDS_LOG.is_file()):
        pytest.skip(f"{_W6IDS.name} or its transcript is not in this checkout")
    blocks = _muted_stint()
    assert len(set(blocks)) == len(blocks), f"a block was delivered twice: {blocks}"
    assert b"".join(blocks).startswith(
        b";PM: W9SSJ JWKY65C2OZES 524 saul.stjohn@gmail.co"), blocks
