# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two overs in one transmission, and the answer that waits for the gap.

A stock VARA HF 4.9.0 does not key one over per transmission for long. Measured
on the cables, 2026-09-09: through a four-message fetch its keyings run 4.38 s
until the seventh over of the delivery, where it keys for 8.60 s and puts two
overs back to back inside one PTT window. Both defects that follows are in this
file.

**A station that is transmitting is not listening.** This station named the first
over as soon as it was whole and answered 0.16-0.24 s later — four seconds inside
the peer's own key-down. The acknowledgement was lost to the peer's transmitter
rather than to the channel: the same over, in both arms, with nothing injected,
and the fetch stopped there with 1587 bytes still in the sender's queue. So an
answer found on the raw stream waits for the channel to go quiet, and the reading
that says it has is the transport's own gate — the same one a turn-request is
keyed on  [see ``VaraIO.receiving``].

**And the second over is 89 bytes of somebody's mail.** Read off the b2a capture
of that arm, the window's two frames are column-contiguous, 24 of 24 reference
columns and CRC-clean each, 89 payload bytes each. A station that reads only the
first delivers half a window and acknowledges all of it.

Nothing here opens a device or keys a radio.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_data_over_gate import _connected

MYCALL, CALLED = "W9SSJ", "KC9GHZ"


_R = rx.RECORDS[rx.BASE_LEVELS["2300"]]
#: The frame's own span, which is what a sender's emission ends on: ours opens on
#: a lead-in and a stock one's two frames are column-contiguous inside a single
#: window, so a second over is joined here at its frame rather than at its burst.
_FRAME = _R.ncols * _R.dw50


def _over(payload: bytes, index: int) -> np.ndarray:
    return tx.synth_burst(_phy.vara_body(payload, MYCALL), over=index)


def _window(*payloads: bytes) -> np.ndarray:
    """One transmission carrying every payload, framed the way the 2026-09-09
    capture holds them: back to back with no gap between the frames."""
    outs = [_over(p, i) for i, p in enumerate(payloads)]
    return np.concatenate([outs[0]] + [o[-_FRAME:] for o in outs[1:]])


def _lead(x: np.ndarray, cols: int = 8) -> np.ndarray:
    """The head of the over behind this one, which is all of it this end holds
    when it reads the one in front  [see `_window_state`]."""
    return x[-_FRAME:][:cols * _R.dw50]


def _stream(hs, *bursts: np.ndarray, trail: float = 0.5) -> None:
    """Feed a window to the raw-stream route in the transport's own blocks.

    A trail, because the over search declines an alignment that ends at the
    buffer's own last column — on a live stream more audio always follows, and
    what the buffer stops on says nothing about the frame  [see _stream_over].
    """
    x = np.concatenate([*bursts, np.zeros(int(trail * MK.FS))])
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])


# --------------------------------------------------------------------------- #
def test_the_answer_waits_while_the_peer_is_still_keying():
    """The defect, at its own instant: the over is whole and the peer is not
    done, which the columns behind the frame say  [see `_window_state`]."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    assert io.host == [b"A" * 89], "the over went undelivered while it was held"
    assert io.keys == 0, [m for m in io.msgs if m.startswith("tx ")]
    assert hs._held_answer == (False, False), io.msgs
    assert any("still keying — holding the answer" in m for m in io.msgs), io.msgs


def test_the_over_that_ends_the_window_takes_the_answer():
    """One window, one acknowledgement, keyed where the peer is listening: the
    held answer is replaced by the last over's, and only that one goes out."""
    hs, io = _connected()
    _stream(hs, _window(b"A" * 89, b"B" * 17))
    assert io.host == [b"A" * 89, b"B" * 17], io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]
    assert hs._held_answer is None
    assert any("11-symbol control burst" in m for m in io.msgs), io.msgs


def test_a_window_of_two_full_overs_is_answered_keep_sending():
    """Which frame it draws is still the last over's own property."""
    hs, io = _connected()
    _stream(hs, _window(b"A" * 89, b"B" * 89))
    assert io.host == [b"A" * 89, b"B" * 89], io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]
    assert not any("11-symbol control burst" in m for m in io.msgs), io.msgs


def test_an_emission_that_never_ends_is_not_waited_out():
    """A timed-out receive window needs retransmission. A positive ACK could
    retire the unread block at the sender, so nothing keys at this timeout."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    assert io.keys == 0
    _stream(hs, *[_over(b"C" * 89, 2)[::-1]] * 3, trail=0.0)
    assert io.keys == 0, io.msgs
    assert any("no positive acknowledgement sent" in m for m in io.msgs), io.msgs
    assert hs._owed_block, io.msgs


def test_a_block_keyed_across_is_asked_for_again():
    """The unread remainder must be requested in the peer's next gap, without
    an intervening positive ACK. The debt persists in case this NAK is lost."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    _stream(hs, *[_over(b"C" * 89, 2)[::-1]] * 3, trail=0.0)
    assert hs._owed_block
    at = len(io.msgs)
    _stream(hs, MK.synth_burst(CALLED, VF.SESSION_RESPONDER_IDLE), trail=0.3)
    assert hs._owed_block
    assert any("asking KC9GHZ for the unread window again" in m
               for m in io.msgs[at:]), io.msgs[at:]


def test_a_peer_that_unkeys_is_answered_at_once():
    """The ordinary case, and what the hold costs it: the answer waits for the
    six columns behind the frame — 64 ms, which is what says the peer has
    finished — and then goes out in the same call."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), trail=1.0)
    assert io.keys == 1, io.msgs
    assert hs._held_answer is None
    assert not any("holding the answer" in m for m in io.msgs), io.msgs


def test_the_window_is_read_whole_on_the_transports_own_poll():
    """The live path feeds 20 ms at a time, not 100. The decode then completes
    before the columns behind the frame have arrived, and a short tail read as
    "the peer stopped" is what acknowledged the third two-block window of the
    2026-09-09 after arm 4.6 s into an 8.56 s transmission and lost its second
    block."""
    hs, io = _connected()
    x = np.concatenate([_window(b"A" * 89, b"B" * 89), np.zeros(MK.FS // 2)])
    for i in range(0, len(x), MK.FS // 50):
        hs.on_rx_stream(x[i:i + MK.FS // 50])
    assert io.host == [b"A" * 89, b"B" * 89], io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]


def test_a_repeated_window_on_the_stream_is_not_delivered_twice():
    """The NAK rung invites a repeat and a stock sender obliges. On the stream
    the two overs of a window are two deliveries, so a gate that remembers only
    the last one passes A B A B to the host and the B2F stream is 178 bytes of
    mail with 178 more spliced into it."""
    hs, io = _connected()
    window = np.concatenate([_window(b"A" * 89, b"B" * 89), np.zeros(MK.FS // 2)])
    for _ in range(2):
        for i in range(0, len(window), MK.FS // 50):
            hs.on_rx_stream(window[i:i + MK.FS // 50])
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89, io.host
    assert any("repeats one already delivered" in m for m in io.msgs), io.msgs


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize("second", [("A", "C"), ("B", "A"), ("C", "B"),
                                   ("A", "A"), ("A",)])
def test_a_new_window_preserves_recurring_content(streamed, second):
    """Content shared with the last window does not make this one a replay.

    Exercise order, multiplicity, a shared prefix and a shorter window on both
    receive routes. Advancing preamble indices do not change the decoded bodies.
    """
    hs, io = _connected()
    expected = []
    index = 0
    for names in [("A", "B"), second]:
        payloads = [name.encode() * 89 for name in names]
        bursts = [_over(p, index + i) for i, p in enumerate(payloads)]
        window = np.concatenate([bursts[0]] + [b[-_FRAME:] for b in bursts[1:]])
        if streamed:
            x = np.concatenate([window, np.zeros(MK.FS // 2)])
            for i in range(0, len(x), MK.FS // 50):
                hs.on_rx_stream(x[i:i + MK.FS // 50])
        else:
            hs.on_rx_audio(window)
        expected.extend(payloads)
        index += len(payloads)
        assert io.host == expected, io.msgs


def test_identical_blocks_in_one_window_keep_their_multiplicity():
    hs, io = _connected()
    window = _window(b"A" * 89, b"A" * 89)
    for _ in range(2):
        _stream(hs, window)
        assert io.host == [b"A" * 89, b"A" * 89], io.msgs


def test_a_refused_missing_block_nak_is_retained_for_the_next_gap():
    """A transport refusal must neither forget the block nor fall through to
    an acknowledgement of data that was never received."""
    from hfmodem.tests.kestrel.test_reack_over import keyed

    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    _stream(hs, *[_over(b"C" * 89, 2)[::-1]] * 3, trail=0.0)
    assert hs._owed_block
    io.tx_went_out = lambda: False
    for _ in range(2):
        before = len(io.sent)
        assert not hs._reack()
        assert hs._owed_block
        assert hs._reacks == 0
        assert len(io.sent) == before + 1
        assert keyed(io) == "nak"
    io.tx_went_out = lambda: True
    before = len(io.sent)
    assert hs._reack()
    # Transmitting the NAK requests the missing data; only its arrival can
    # settle the debt. A lost NAK must remain retryable in the next peer gap.
    assert hs._owed_block
    assert hs._reacks == 1
    assert len(io.sent) == before + 1
    assert keyed(io) == "nak"
    # The requested retry must fill the hole without repeating the prefix that
    # reached the host before the timeout, including on a second full replay.
    for _ in range(2):
        _stream(hs, _window(b"A" * 89, b"B" * 89))
        assert io.host == [b"A" * 89, b"B" * 89], io.msgs


@corpora.requires_fetch_two_block
def test_the_two_block_window_off_the_bench_tape_is_read_whole():
    """The gate, on the responder's own cable rather than on a fixture.

    A stock VARA HF 4.9.0 keys 81.330-89.880 s of the 2026-09-09 fetch — one PTT
    window, two 89-byte blocks, blob offsets 534 and 623 of the 2117 bytes its
    gateway handed it. Fed 20 ms at a time, which is what the live transport
    polls at, this station read one block and acknowledged 4.5 s inside the
    window; the second never reached the session and the B2F parser met the hole
    at the next block boundary. Both blocks, and one answer, keyed after the
    responder unkeys.
    """
    path, spec = corpora.FETCH_TWO_BLOCK, corpora.FETCH_TWO_BLOCK_AT
    unkey = spec[2]
    hs, io = _connected()
    hs.called = "W1AW"
    at = _replay(hs, io, path, spec)

    assert [len(b) for b in io.host] == [89, 89], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(at) == 1, (f"{len(at)} bursts keyed for one window", io.msgs)
    assert at[0] > unkey, (
        f"answered at {at[0]:.3f} s, {unkey - at[0]:.3f} s inside the window",
        io.msgs)
    assert at[0] < unkey + 0.4, (
        f"answered {at[0] - unkey:.3f} s past the unkey, outside the turnaround "
        "a VARA peer answers in", io.msgs)


def _replay(hs, io, path, at):
    """Feed one window of a bench capture to the stream route, 20 ms at a time,
    and return the instants this station keyed at."""
    t0, t1, _ = at
    x = corpora.wav_mono(path)
    keyed = []
    io.key = lambda on: keyed.append(t0 + hs._at) if on else None
    hs._at = 0.0
    window = np.asarray(x[int(t0 * MK.FS):int(t1 * MK.FS)], float)
    step = MK.FS // 50
    for i in range(0, len(window), step):
        hs._at = i / MK.FS
        hs.on_rx_stream(window[i:i + step])
    return keyed


@corpora.requires_fetch_idle_in_window
def test_no_burst_goes_out_inside_a_window_a_session_frame_was_named_in():
    """The other way a block is lost, off the tape that lost it.

    At 132.170 s of the 2026-09-09 fetch the responder keys two blocks in one
    8.55 s window. This station read the first, held its answer — and the
    32-symbol recogniser then took a `session-responder-over-idle` out of the
    second block at +0.257 s, which the ladder read as the peer's turnaround and
    keyed a rung into. The block at blob offset 1513 was never delivered and the
    parser refused the hole at the next boundary.

    A named frame is not a gap: a window this station is holding an answer for
    is a peer mid-transmission, whatever a recogniser makes of the audio.

    THIS REPLAY DOES NOT ITSELF NAME THE IDLE, and the case it holds is the
    window read whole. What decided the false name was the answer search's own
    phase, which a live station carries across its transmissions — the receive
    cursor jumps past every one of them — and a replay of the tape alone does
    not. The mechanism is locked below  [see
    test_a_frame_named_inside_a_held_window_keys_nothing].
    """
    path, at = corpora.FETCH_IDLE_IN_WINDOW, corpora.FETCH_IDLE_IN_WINDOW_AT
    hs, io = _connected()
    hs.called = "W1AW"
    keyed = _replay(hs, io, path, at)

    assert [len(b) for b in io.host] == [89, 89], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(keyed) == 1, (f"{len(keyed)} bursts keyed for one window", io.msgs)
    assert keyed[0] > at[2], (
        f"keyed at {keyed[0]:.3f} s, inside a window that ends at {at[2]}",
        io.msgs)


@corpora.requires_fetch_level_misses
def test_the_window_the_level_test_read_as_over_is_held():
    """The third way a block is lost, off the tape that lost it.

    At 135.295 s of the 2026-09-09 fetch the responder keys two blocks in one
    8.555 s window, and the six columns behind the first block's frame read
    0.304 of that frame — under the threshold, over a block that was there. The
    station answered 4.5 s into the window and blob offset 1691 was never read.
    The two populations overlap at their ends: a peer still keying has been
    measured at 0.28 and a quiet band at 0.41, so the level alone misses about
    one window in eight.

    What decides it here is the block's own structure: every reference column of
    the frame behind the first hits, where a band the peer has stopped keying
    into hits none  [see `_next_frame_here`].
    """
    path, spec = corpora.FETCH_LEVEL_MISSES, corpora.FETCH_LEVEL_MISSES_AT
    hs, io = _connected()
    hs.called = "W1AW"
    at = _replay(hs, io, path, spec)

    assert [len(b) for b in io.host] == [89, 89], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(at) == 1, (f"{len(at)} bursts keyed for one window", io.msgs)
    assert at[0] > spec[2], (
        f"answered at {at[0]:.3f} s, inside a window that ends at {spec[2]}",
        io.msgs)
    assert any("still keying" in m for m in io.msgs), io.msgs


def test_a_frame_named_inside_a_held_window_keys_nothing():
    """The mechanism, without the tape: the idle is named, said to be inside the
    peer's own emission, and nothing goes out  [see `_a_gap`]."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    assert hs._held_answer is not None
    hs._answer_owed, hs._reack_frame = VA._OWED_OVER, VA.OVER_CONTINUE_GENERATED
    at = io.keys
    _stream(hs, MK.synth_burst(CALLED, VF.SESSION_RESPONDER_IDLE), trail=0.3)
    assert io.keys == at, [m for m in io.msgs if m.startswith("tx ")]
    assert any("not a gap" in m for m in io.msgs), io.msgs


# --------------------------------------------------------------------------- #
# The bracket route, where the window has already closed.
def test_a_bracket_holding_two_overs_delivers_both_and_answers_once():
    hs, io = _connected()
    hs.on_rx_audio(_window(b"A" * 89, b"B" * 89))
    assert io.host == [b"A" * 89, b"B" * 89], io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]
    assert any("2 in this window" in m for m in io.msgs), io.msgs


def test_a_bracket_holding_two_overs_is_answered_by_its_last_one():
    """Which frame the window draws is the LAST over's property: a window that
    ends on a short over ends the delivery."""
    hs, io = _connected()
    hs.on_rx_audio(_window(b"A" * 89, b"tail"))
    assert io.host == [b"A" * 89, b"tail"], io.msgs
    assert any("11-symbol control burst" in m for m in io.msgs), io.msgs


def test_the_repeat_of_a_two_over_window_is_not_delivered_twice():
    """The duplicate gate is the whole window, so a sender that did not read our
    answer and keys both overs again does not hand the host the pair twice."""
    hs, io = _connected()
    window = _window(b"A" * 89, b"B" * 89)
    hs.on_rx_audio(window)
    hs.on_rx_audio(window)
    assert io.host == [b"A" * 89, b"B" * 89], io.msgs
    assert any("repeats one already delivered" in m for m in io.msgs), io.msgs


def test_one_over_still_reads_as_one():
    """The bound on the search behind a frame: a window with a single over in it
    must not report a second off the audio behind it."""
    hs, io = _connected()
    hs.on_rx_audio(np.concatenate([_over(b"A" * 89, 0), np.zeros(5 * MK.FS)]))
    assert io.host == [b"A" * 89], io.msgs
    assert not any("in this window" in m for m in io.msgs), io.msgs


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_an_ambiguous_window_is_decided_on_its_own_records_reference_bins(bw):
    """The structural probe belongs to the record the session is running.

    BW2750's base record is BW2300 record 3's index law on a wider comb, and the
    two share their 24 reference COLUMNS exactly — but not the bins those columns
    light: 20 bins at 6..25 against 16 at 9..24 puts every reference bin
    elsewhere. Scored against BW2300's table a valid BW2750 continuation hits
    none of them, so the window the level could not call reads quiet and the
    acknowledgement goes out across the peer's second block.

    The tail here is attenuated to land the level reading between
    `_QUIET_FRAC` and `_KEYING_FRAC` — the same overlap the 2026-09-09 fetch read
    0.304 in — which is the only regime the probe runs in.
    """
    lv = rx.BASE_LEVELS[bw]
    r = rx.RECORDS[lv]
    span = r.ncols * r.dw50
    first = tx.synth_burst(_phy.vara_body(b"A" * 89, MYCALL), lv, 0)[-span:]
    second = tx.synth_burst(_phy.vara_body(b"B" * 89, MYCALL), lv, 1)[-span:]
    x = np.concatenate([first, 0.5 * second[:VA._PROBE_COLS * r.dw50]])
    x += np.random.default_rng(7).normal(0, 0.02 * np.abs(first).max(), len(x))

    mag = rx._band_mag(x, lv, r.ncols + VA._PROBE_COLS)
    tail = mag[r.ncols:]
    level = VA._lit(tail[:VA._KEYING_COLS] ** 2) / VA._lit(mag[:r.ncols] ** 2)
    assert VA._QUIET_FRAC < level < VA._KEYING_FRAC, (
        f"the level reads {level:.3f}, outside the band the probe decides in")
    assert VA._next_frame_here(tail, r) == VA._PROBE_REFS
    assert VA._window_state(mag, r) == VA._KEYING
