# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two overs in one transmission, and the answer that waits for the gap.

A VARA HF sender does not key one over per transmission for long. Measured
2026-09-09: through a four-message fetch its keyings run 4.38 s
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
#: a lead-in and a VARA HF sender's two frames are column-contiguous inside a single
#: window, so a second over is joined here at its frame rather than at its burst.
_FRAME = _R.ncols * _R.dw50


def _over(payload: bytes, index: int) -> np.ndarray:
    from .receive_window_helpers import full_body
    body = full_body(payload, MYCALL)
    assert _phy.vara_payload(body, caller=MYCALL) == payload
    return tx.synth_burst(body, over=index)


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
    assert not io.host
    assert io.keys == 0, [m for m in io.msgs if m.startswith("tx ")]
    assert hs._held_answer == (False, False), io.msgs
    assert any("still keying — holding the answer" in m for m in io.msgs), io.msgs


def test_the_over_that_ends_the_window_takes_the_answer():
    """One window, one acknowledgement, keyed where the peer is listening: the
    held answer is replaced by the last over's, and only that one goes out."""
    hs, io = _connected()
    _stream(hs, _window(b"A" * 89, b"B" * 17))
    assert b"".join(io.host) == b"A" * 89 + b"B" * 17, io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]
    assert hs._held_answer is None
    assert any("11-symbol control burst" in m for m in io.msgs), io.msgs


def test_a_window_of_two_full_overs_is_answered_keep_sending():
    """Which frame it draws is still the last over's own property."""
    hs, io = _connected()
    _stream(hs, _window(b"A" * 89, b"B" * 89))
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89, io.msgs
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
    assert b"".join(io.host) == b''


def test_a_block_keyed_across_is_asked_for_again():
    """The unread remainder must be requested in the peer's next gap, without
    an intervening positive ACK. The debt persists in case this NAK is lost."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    _stream(hs, *[_over(b"C" * 89, 2)[::-1]] * 3, trail=0.0)
    assert hs._owed_block
    _stream(hs, MK.synth_burst(CALLED, VF.SESSION_RESPONDER_IDLE), trail=0.3)
    assert hs._owed_block
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert b"".join(io.host) == b''


def test_a_peer_that_unkeys_is_answered_at_once():
    """The ordinary case, and what the hold costs it: the answer waits for the
    six columns behind the frame — 64 ms, which is what says the peer has
    finished — and then goes out in the same call."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), trail=1.0)
    assert io.keys == 1, io.msgs
    assert hs._held_answer is None
    assert not any("holding the answer" in m for m in io.msgs), io.msgs
    assert b"".join(io.host) == b'A' * 89


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
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89, io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]


def test_a_repeated_window_on_the_stream_is_not_delivered_twice():
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
    hs, io = _connected()
    window = np.concatenate([_window(b"A" * 89, b"B" * 89), np.zeros(MK.FS // 2)])
    for _ in range(2):
        for i in range(0, len(window), MK.FS // 50):
            hs.on_rx_stream(window[i:i + MK.FS // 50])
    assert b"".join(io.host) == (b"A" * 89 + b"B" * 89) * 2, io.host
    assert hs._peer_over == 2


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize("second", [("A", "C"), ("B", "A"), ("C", "B"),
                                   ("A", "A"), ("A",)])
def test_a_new_window_preserves_recurring_content_existing_controls(streamed, second):
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
        assert b"".join(io.host) == b"".join(expected), io.msgs


def test_identical_blocks_in_one_window_keep_their_multiplicity():
    hs, io = _connected()
    window = _window(b"A" * 89, b"A" * 89)
    for occurrence in range(1, 3):
        _stream(hs, window)
        assert b"".join(io.host) == b"A" * (178 * occurrence), io.msgs


def test_a_refused_missing_block_nak_is_retained_for_the_next_gap():
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
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


    assert hs._owed_block
    assert hs._reacks == 1
    assert len(io.sent) == before + 1
    assert keyed(io) == "nak"


    for occurrence in range(1, 3):
        _stream(hs, _window(b"A" * 89, b"B" * 89))
        assert b"".join(io.host) == (b"A" * 89 + b"B" * 89) * occurrence, io.msgs


@corpora.requires_fetch_two_block
def test_the_two_block_window_off_the_recording_is_read_whole():
    """The gate, on the responder's own recording rather than on a fixture.

    The VARA HF sender keys 81.330-89.880 s of the 2026-09-09 fetch — one PTT
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

    assert [len(b) for b in io.host] == [178], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(at) == 1, (f"{len(at)} bursts keyed for one window", io.msgs)
    assert at[0] > unkey, (
        f"answered at {at[0]:.3f} s, {unkey - at[0]:.3f} s inside the window",
        io.msgs)
    assert at[0] < unkey + 0.4, (
        f"answered {at[0] - unkey:.3f} s past the unkey, outside the turnaround "
        "a VARA peer answers in", io.msgs)
    assert b"".join(io.host) == TAPE121


def _replay(hs, io, path, at):
    """Feed one window of a recording to the stream route, 20 ms at a time,
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

    assert [len(b) for b in io.host] == [178], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(keyed) == 1, (f"{len(keyed)} bursts keyed for one window", io.msgs)
    assert keyed[0] > at[2], (
        f"keyed at {keyed[0]:.3f} s, inside a window that ends at {at[2]}",
        io.msgs)
    assert b"".join(io.host) == TAPE122


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

    assert [len(b) for b in io.host] == [178], (
        f"{len(io.host)} of the window's two blocks reached the host", io.msgs)
    assert len(at) == 1, (f"{len(at)} bursts keyed for one window", io.msgs)
    assert at[0] > spec[2], (
        f"answered at {at[0]:.3f} s, inside a window that ends at {spec[2]}",
        io.msgs)
    assert any("still keying" in m for m in io.msgs), io.msgs
    assert b"".join(io.host) == TAPE123


def test_a_frame_named_inside_a_held_window_keys_nothing():
    """The mechanism, without the tape: the idle is named, said to be inside the
    peer's own emission, and nothing goes out  [see `_a_gap`]."""
    hs, io = _connected()
    _stream(hs, _over(b"A" * 89, 0), _lead(_over(b"B" * 89, 1)), trail=0.0)
    assert hs._held_answer is not None
    hs._answer_owed, hs._reack_frame = VA._OWED_OVER, VA.OVER_CONTINUE_GENERATED
    at = io.keys
    _stream(hs, MK.synth_burst(CALLED, VF.SESSION_RESPONDER_IDLE), trail=0.3)
    assert io.keys == at + 1, io.msgs
    assert b"".join(io.host) == b"A" * 89
    assert hs._rx_window.W is None



# --------------------------------------------------------------------------- #
# The bracket route, where the window has already closed.
def test_a_bracket_holding_two_overs_delivers_both_and_answers_once():
    hs, io = _connected()
    hs.on_rx_audio(_window(b"A" * 89, b"B" * 89))
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89, io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]
    assert len(hs._rx_window.K.frames) == 2


@pytest.mark.parametrize("hz", [-0.142, -0.1, 0.1, 29.0])
def test_a_bracket_holding_two_overs_delivers_both_off_our_grid(hz):
    """Two clean frames tie on reference columns and the search takes either:
    a peer 0.14 Hz off our grid read only the second of each pair,
    and the session's mail stopped a block short."""
    hs, io = _connected()
    hs._peer_offset = hz * MK.NFFT / MK.FS
    hs.on_rx_audio(_window(b"A" * 89, b"B" * 89))
    assert b"".join(io.host) == b"A" * 89 + b"B" * 89, io.msgs
    assert io.keys == 1, [m for m in io.msgs if m.startswith("tx ")]


def test_a_bracket_holding_two_overs_is_answered_by_its_last_one():
    """Which frame the window draws is the LAST over's property: a window that
    ends on a short over ends the delivery."""
    hs, io = _connected()
    hs.on_rx_audio(_window(b"A" * 89, b"tail"))
    assert b"".join(io.host) == b"A" * 89 + b"tail", io.msgs
    assert any("11-symbol control burst" in m for m in io.msgs), io.msgs


def test_the_repeat_of_a_two_over_window_is_not_delivered_twice():
    """DESIGN §§3–5: complete equal DATA without an aired NAK is new."""
    hs, io = _connected()
    window = _window(b"A" * 89, b"B" * 89)
    hs.on_rx_audio(window)
    hs.on_rx_audio(window)
    assert b"".join(io.host) == (b"A" * 89 + b"B" * 89) * 2, io.msgs
    assert hs._peer_over == 2


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
    from .receive_window_helpers import full_body
    lv = rx.BASE_LEVELS[bw]
    r = rx.RECORDS[lv]
    span = r.ncols * r.dw50
    first = tx.synth_burst(full_body(b"A" * 89, MYCALL), lv, 0)[-span:]
    second = tx.synth_burst(full_body(b"B" * 89, MYCALL), lv, 1)[-span:]
    x = np.concatenate([first, 0.5 * second[:VA._PROBE_COLS * r.dw50]])
    x += np.random.default_rng(7).normal(0, 0.02 * np.abs(first).max(), len(x))

    mag = rx._band_mag(x, lv, r.ncols + VA._PROBE_COLS)
    tail = mag[r.ncols:]
    level = VA._lit(tail[:VA._KEYING_COLS] ** 2) / VA._lit(mag[:r.ncols] ** 2)
    assert VA._QUIET_FRAC < level < VA._KEYING_FRAC, (
        f"the level reads {level:.3f}, outside the band the probe decides in")
    assert VA._next_frame_here(tail, r) == VA._PROBE_REFS
    assert VA._window_state(mag, r) == VA._KEYING


@pytest.mark.parametrize("streamed,held", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("second", [("A", "C"), ("B", "A"), ("C", "B"),
                                   ("A", "A"), ("A",), ("A", "B")])
def test_a_new_window_preserves_recurring_content(monkeypatch, streamed, second, held, peer_clock):
    _recurring_window(monkeypatch, streamed, second, held, False)


@pytest.mark.parametrize("second,streamed,held", [(("A",), True, False), (("A",), False, False),
                                                (("A", "B"), True, False), (("A", "B"), False, False),
                                                (("A", "B"), True, True)])
def test_a_new_window_preserves_recurring_content_authorized_resend(monkeypatch, streamed, second, held, peer_clock):
    _recurring_window(monkeypatch, streamed, second, held, True)


def _recurring_window(monkeypatch, streamed, second, held, authorized):
    from .receive_window_helpers import station, receive, query, arm_replay
    hs, io = station('2300', 'initiator')
    a, b = b'A' * 89, b'B' * 89
    receive(hs, _window(a, b), 'stream' if streamed else 'bracket')
    assert b''.join(io.host) == a + b
    if authorized:
        arm_replay(hs, io)
    payloads = [name.encode() * 89 for name in second]
    with monkeypatch.context() as patch:
        if held:
            patch.setattr(VA, '_window_state', lambda *args: VA._KEYING)
        receive(hs, _window(*payloads), 'stream' if streamed else 'bracket')
    if held:
        assert hs._held_answer is not None and b''.join(io.host) == a + b
        query(hs, delay=.2)
        assert hs._held_answer is None
    if authorized and second == ('A',):
        assert hs._rx_window.T.cursor == 89
        receive(hs, _window(b), 'stream' if streamed else 'bracket')
    expected = a + b + (b'' if authorized else b''.join(payloads))
    assert b''.join(io.host) == expected


@pytest.mark.parametrize('scenario', ['stream', 'bracket', 'equal', 'refused'])
def test_full_window_repeat_after_aired_nak(scenario, peer_clock):
    from .receive_window_helpers import station, receive, arm_replay
    hs, io = station('2300', 'initiator')
    a = b'A' * 89
    b = a if scenario == 'equal' else b'B' * 89
    route = 'bracket' if scenario == 'bracket' else 'stream'
    if scenario == 'refused':
        _stream(hs, _over(a, 0), _lead(_over(b, 1)), trail=0.)
        _stream(hs, *[_over(b'C' * 89, 2)[::-1]] * 3, trail=0.)
        assert hs._owed_block and not io.host
        io.tx_went_out = lambda: False
        for _ in range(2):
            assert not hs._reack()
            assert not hs._rx_window.A.sent and hs._rx_window.T is None
        io.tx_went_out = lambda: True
        assert hs._reack()
        assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    receive(hs, _window(a, b), route)
    assert b''.join(io.host) == a + b
    arm_replay(hs, io)
    receive(hs, _window(a, b), route)
    assert b''.join(io.host) == a + b
    assert hs._rx_window.T is None


from .receive_window_helpers import rx_evidence  # noqa: F401

TAPE121 = bytes.fromhex('6d61727900300002fa908b96020000ecfd7e1c6d679bc5e1df75bb1bdbac5e0f4fbb99cb743da01f1c2c1bdde53e16eec38b1802a9478142e685b5af8017f95417ee7fb100a508254ae94e7eff34aa77e3d04ebfffb301fc6ea3acbed27f4fd733e724fdcc645fbdd379a3080f33e3ab8f6179325bb8df74558beb0445ea0160a110c9d56f1b50b9f990984f9c7485b9c8a68f35f103f9c9e1a261c3f4d3aeb2a4044ba0b7fa0c03f24da9475437fb04857e')

TAPE122 = bytes.fromhex('0eed7554e9f9e3090481ed17072ec4667b27d340edc2707c603cd900c9ee6d96ff0960acb71c5139330b06cbd71dabb66500e7a9a039a43cf7b2daaaa8c2b6b9483912e4b2609d8c286797812dbd6b5baaca600f4d2c2d04551c49455aea7644e72887aa43dd2544b4ccbead0ec771a8d6509a797bcc753934e7832c525044460b8787a8474c6e6a9d8c57adb50004a4012d5765656b656e642056484620616374697669747920616e642074686520726570')

TAPE123 = bytes.fromhex('706561746572206d6f766500300002fad598bb020000ecfd7e1c6d6783b9c1e2f0fabd1beef76eeee33396e87b403e38583bede58f0b77617b1802a942be852e25bdb6002ff2942f7ffec4029020992fa543fbfcd4addfc5951e0ffda00fe3752367f698fa7eb99f3977ade3e3bdee1acf18417d9f1b59657d69d35f487ba36ca11d200f56e7ee7c5e1b3e2ea4cb71153c874051746e18a040f8c4c68d7a26992f168e0107592854bae0c80def35ec30f9c5')


def test_named_idle_in_held_window_retains_unanswered_bytes(monkeypatch):
    hs, io = _connected()
    _stream(hs, _over(b'A' * 89, 0), _lead(_over(b'B' * 89, 1)), trail=0.)
    assert hs._held_answer is not None
    hs._answer_owed, hs._reack_frame = VA._OWED_OVER, VA.OVER_CONTINUE_GENERATED
    before = io.keys
    monkeypatch.setattr(hs, '_peer_frame', lambda *args: (None, 0, True))
    monkeypatch.setattr(hs, '_complete_query_fit', lambda *args: None)
    assert hs._peer_responder_idle(np.zeros(VA._span(VF.SESSION_RESPONDER_IDLE)))
    assert not hs._idle_complete_exact and not hs._a_gap()
    # LEDGER #row-124: the held window has no positive credit here.
    assert io.keys == before
    assert b''.join(io.host) == b''


@pytest.fixture
def peer_clock(monkeypatch):
    from .test_idle_poll import _Clock
    _Clock.t = 1000.
    monkeypatch.setattr(VA, 'time', _Clock)
