# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The answer to an intermediate over of OURS, and the delivery it continues.

Every outbound message past the login is longer than one over: a proposal block
and a compressed body are several 89-byte overs, and each of them but the last is
an intermediate one. A station answers an intermediate over with a continue-class
frame and only the last one with the 11-symbol control burst, so a build that
locks the control burst alone reads the end of its own message and nothing before
it. The delivery stalls after the first over, in reverse, exactly the way seven
gateways' inbound ones did — and it has never bitten on the air only because the
B2F login block is 65 bytes.

TWO FRAMES ARRIVE IN THAT TURNAROUND AND BOTH ARE READ HERE.

* The eight-symbol continue burst, off the responder's cable of two two-cable
  bench sessions of 2026-08-30 — the only audio held anywhere in which a stock
  responder answers a stock CALLER's intermediate overs. Eight symbols,
  0.340-0.345 s at the key, 0.170-0.175 s behind the caller's last transmitted
  sample. It is on no callsign's lattice: every generated frame in this family is
  one tone per symbol over 15 or 31 payload symbols, and this is eight two-tone
  ones. Fourteen copies share one symbol, the first, which is the control burst's
  own opening pair; the seven behind it move from over to over.
* The generated 32-symbol `session-responder-over-answer`, which KE8LVA keyed
  0.17 s behind the first payload over this station ever put on the air. That one
  was already read and already logged; what it did not do was key the next over.

WHAT SEPARATES THE CONTINUE BURST FROM THE CONTROL BURST is the control burst's
own four-pair preamble, and the separation is total on the tapes: every 11-symbol
burst holds it for 59-60 alignments and every eight-symbol one for none. So the
control burst is decided first and the two answers never compete.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_turn_law import (_CALLED, _MYCALL, _connected,
                                                 _peer_control)

#: Seconds of audio kept either side of a burst cut out of a bench cable. The
#: lead is what the plateau is located from; without it the burst has nowhere to
#: begin.
_LEAD, _TRAIL = 0.30, 0.25


def _cut(x: np.ndarray, at: float, n_sym: int) -> np.ndarray:
    span = n_sym * MK.HOP / MK.FS
    return x[int((at - _LEAD) * MK.FS):int((at + span + _TRAIL) * MK.FS)]


def _stream(hs, x: np.ndarray) -> None:
    for i in range(0, len(x), MK.FS // 10):
        hs.on_rx_stream(x[i:i + MK.FS // 10])


def _over_answer() -> np.ndarray:
    """The generated 32-symbol answer, keyed to the station we dialled."""
    return MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_OVER_ANSWER)


def _drain(hs) -> dict:
    """Key a whole delivery, with the peer answering each over, and return the
    per-frame field of every body that went out.

    The answer is not decoration: an over nobody acknowledges is repeated to
    `_OVER_RETRY_MAX` and then closes the link, so a loop that only calls
    `_tx_data_over` never reaches the second block  [see `_retry_data_over`].
    And it clears the echo window, so each body is read off before the next
    answer rather than at the end  [see `_close_echo_window`].
    """
    hs.turn = VA._TURN_OURS
    fields: dict[bytes, int] = {}
    hs._tx_data_over()
    while True:
        fields.update({body[0:1]: body[-1] for body in hs._keyed_bodies})
        if not hs._txq:
            return fields
        hs._took_control_burst()


def _queued(blocks: int = 2):
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs._txq = [b"A" * 89] * (blocks - 1) + [b"tail"]
    return hs, io


# --------------------------------------------------------------------------- #
# The generated frame, synthetic.
def test_the_generated_answer_keys_the_next_over_when_one_is_queued():
    """KE8LVA's answer, read as the thing it is: our over was heard, and the rest
    of the message follows it."""
    hs, io = _queued()
    _stream(hs, np.concatenate([np.zeros(MK.FS // 2), _over_answer(),
                                np.zeros(2 * MK.FS)]))
    assert any("session-responder-over-answer" in m for m in io.msgs), io.msgs
    assert any("tx DATA over" in m for m in io.msgs), io.msgs
    assert len(hs._txq) == 1, io.msgs


def test_the_generated_answer_still_keys_nothing_on_an_empty_queue():
    """The bound the 2026-08-26 session set and this does not move: with nothing
    left to send, no recording says what answers this frame."""
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    _stream(hs, np.concatenate([np.zeros(MK.FS // 2), _over_answer(),
                                np.zeros(2 * MK.FS)]))
    assert any("session-responder-over-answer" in m for m in io.msgs), io.msgs
    assert io.keys == 0, io.msgs


# --------------------------------------------------------------------------- #
# The continue burst, off the bench cables.
@corpora.requires_over_continue_answers
@pytest.mark.parametrize("path,at", [(p, t) for p, ts in
                                     corpora.OVER_CONTINUE_ANSWERS for t in ts])
def test_a_responders_continue_burst_is_read_off_the_cable(path, at):
    hs, _ = _connected()
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    assert hs._peer_over_continue(_cut(x, at, VF.OVER_CONTINUE_NSYM))


@corpora.requires_over_continue_answers
@pytest.mark.parametrize("at", corpora.OVER_CONTINUE_NEGATIVES)
def test_the_control_bursts_on_the_same_cable_are_not_read_as_continues(at):
    """The last-over answer keeps its own meaning. Reading one as a continue
    would key an over into a turn the peer has finished with."""
    hs, _ = _connected()
    path = corpora.OVER_CONTINUE_ANSWERS[0][0]
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    seg = _cut(x, at, VF.CONNECTED_ACK_NSYM)
    assert hs._peer_control_burst(seg)
    assert not hs._peer_over_continue(seg)


@corpora.requires_over_continue_answers
def test_a_continue_burst_off_the_cable_keys_the_next_over():
    """The whole point, driven the way the transport drives it."""
    path, times = corpora.OVER_CONTINUE_ANSWERS[0]
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    hs, io = _queued()
    _stream(hs, _cut(x, times[0], VF.OVER_CONTINUE_NSYM))
    assert any("over-continue" in m for m in io.msgs), io.msgs
    assert any("tx DATA over" in m for m in io.msgs), io.msgs
    assert len(hs._txq) == 1, io.msgs


# --------------------------------------------------------------------------- #
# The floor.
@corpora.requires_regress_fixtures
def test_the_continue_shape_stands_nowhere_in_the_regression_corpus():
    """The false-accept measurement, at every alignment of real HF.

    Two of the 31 recordings reach an alignment at all. `qso_vara_ns0a.wav` holds
    for 17 and is a real VARA session's own control burst, which the reader
    refuses on the preamble before the shape is asked for; `edge_clipped_9pct.wav`
    reaches three. Genuine copies hold for 54-60, and still for 53 under additive
    noise at 0 dB SNR, so `_CONT_PLATEAU` sits four times over the floor and under
    a quarter of the narrowest copy.
    """
    hs, _ = _connected()
    held, taken = {}, {}
    for path in sorted(corpora.REGRESS_FIXTURES.glob("*.wav")):
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        wide = VA._cont_plateau(x)
        if wide:
            held[path.name] = wide
        if hs._peer_over_continue(x[:VA._DATA_OVER_MIN - 1]):
            taken[path.name] = wide
    assert held == {"edge_clipped_9pct.wav": 3, "qso_vara_ns0a.wav": 17}, held
    assert not taken, taken
    assert max(w for n, w in held.items() if n != "qso_vara_ns0a.wav") < \
        VA._CONT_PLATEAU


@pytest.mark.parametrize("path,t0,t1", [(p, a, b) for p, a, b, _
                                        in corpora.QUIET_STRETCHES])
def test_the_continue_shape_stands_nowhere_in_the_quiet_stretches(path, t0, t1):
    """407 s of real off-air HF holding no burst of ours: no alignment at all."""
    if not path.exists():
        pytest.skip(f"{path} not present")
    x = corpora.wav_mono(path)[int(t0 * MK.FS):int(t1 * MK.FS)]
    x = x / (np.abs(x).max() or 1.0)
    assert VA._cont_plateau(x) == 0


def test_our_own_session_frames_are_not_continue_bursts():
    """Every generated frame here is a single tone per symbol on the same grid,
    and the shape asks for two clear carriers in all eight."""
    hs, _ = _connected()
    for kind in (VF.SESSION_OVER_RESPONSE, VF.SESSION_TURN_REQUEST,
                 VF.SESSION_TURN_IDLE, VF.SESSION_KEEPALIVE_A,
                 VF.SESSION_RESPONDER_OVER_ANSWER):
        burst = np.concatenate([np.zeros(MK.FS // 2),
                                MK.synth_burst(_CALLED, kind)])
        assert not hs._peer_over_continue(burst), kind.name


# --------------------------------------------------------------------------- #
# The over the delivery ends on.
@pytest.mark.parametrize("n,blocks", [(89, 2), (178, 3), (356, 5)])
def test_a_payload_that_is_an_exact_multiple_queues_a_closing_over(n, blocks):
    """The outbound mirror of the rule the receive side already asserts: a full
    over carries no trailer and says another one follows it, so a delivery cannot
    end on one. The split stopped at the last full block and the peer was never
    told the message had ended — 356 bytes went out as four full overs and the
    turn never came back, with the reader for its answer already in place."""
    hs, _ = _connected()
    hs.send(b"A" * n)
    assert len(hs._txq) == blocks
    assert hs._txq[-1] == b""
    assert _phy.over_is_last(_phy.vara_body(hs._txq[-1], _MYCALL), _MYCALL)
    assert not _phy.over_is_last(_phy.vara_body(hs._txq[-2], _MYCALL), _MYCALL)


@pytest.mark.parametrize("n", [1, 88, 90, 300])
def test_a_payload_that_ends_short_owes_no_closing_over(n):
    """The short block IS the close, and a second empty over behind it would be a
    delivery the peer has already been told the end of."""
    hs, _ = _connected()
    hs.send(b"A" * n)
    assert hs._txq[-1] != b""
    assert _phy.over_is_last(_phy.vara_body(hs._txq[-1], _MYCALL), _MYCALL)


# --------------------------------------------------------------------------- #
# The field a full over carries.
@corpora.requires_over_frame_fields
@pytest.mark.parametrize("path,want", corpora.OVER_FRAME_FIELDS,
                         ids=lambda v: getattr(v, "parent", None)
                         and v.parent.name)
def test_a_stock_stations_own_bodies_carry_the_field(path, want):
    """Off the transmitter, not off our own encoder: every DATA body one station
    keyed, in order, read back through the receiver."""
    got = tuple(f.payload[-1] for f in rx.decode_overs(corpora.wav_mono(path)))
    assert got == want


@pytest.mark.parametrize("after,field", [(0, 0x81), (1, 0x8D), (2, 0x91),
                                         (3, 0x95)])
def test_the_field_counts_the_overs_still_to_come(after, field):
    """`0x81` closes the delivery's payload and the ones in front of it count up
    by four. The four together are a stock caller's own 200-byte BW500 delivery,
    and keying them at BW2300 delivered 356 of 356 bytes."""
    assert VA._frame_field(after) == field


def test_the_count_stops_at_the_block_that_ends_the_delivery():
    """A `send` behind this one is its own delivery and its own first over counts
    from the top, so the walk stops at the short block rather than running on."""
    hs, _ = _connected()
    hs._txq = [b"A" * 89, b"B" * 89, b"", b"C" * 89, b"D" * 4]
    assert hs._full_overs_after() == 1


def test_the_overs_of_one_delivery_carry_the_whole_sequence():
    """What goes on the air for a 356-byte message: four full overs counting down
    to `0x81`, then the empty one that closes it."""
    hs, io = _connected()
    hs.send(b"A" * 89 + b"B" * 89 + b"C" * 89 + b"D" * 89)
    assert _drain(hs) == {b"A": 0x95, b"B": 0x91, b"C": 0x8D, b"D": 0x81,
                          b"\x14": 0x82}, io.msgs


@pytest.mark.parametrize("close,field,record", [(60, 0x89, "the base level"),
                                                (40, 0x81, "record 2")])
def test_the_last_full_over_says_which_record_the_close_comes_at(close, field,
                                                                 record):
    """The send leg's own stall, and the byte that ends it.

    A stock 4.9.0 responder will not take the over that closes a multi-over
    delivery when it is keyed at the base record behind a full over announcing
    ``0x81``: on the cables of 2026-09-09, same tree and same cables, that drew
    nothing 0 of 4 while an identical close at record 2 was byte-exact 3 of 3.
    The sender's own convention is the pairing eight deliveries on tape already
    show — ``0x81`` before a record-2 close and ``0x89`` before a base-record
    one — and a close too long for record 2's 48-byte body has nowhere else to
    go  [see `arq.phy.close_level`, `_over_level`].
    """
    hs, _ = _connected()
    hs.send(b"A" * 89 + b"B" * 89 + b"C" * close)
    assert _drain(hs)[b"B"] == field, f"a close at {record} was announced wrong"
    assert _phy.overs_after(field) == 0, "the reader does not place it either"


def test_a_long_delivery_counts_down_to_the_close_it_will_key():
    """The whole sequence with a close that cannot drop a record: the countdown
    is the same and only its last step moves."""
    hs, _ = _connected()
    hs.send(b"A" * 89 + b"B" * 89 + b"C" * 89 + b"D" * 89 + b"E" * 89 + b"F" * 60)
    assert _drain(hs) == {b"A": 0x99, b"B": 0x95, b"C": 0x91, b"D": 0x8D,
                          b"E": 0x89, b"F": 0x82}


def test_the_control_burst_still_ends_a_delivery():
    """The reader that was there before this one, unmoved: an 11-symbol answer
    with an empty queue hands the turn back rather than continuing."""
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    _stream(hs, np.concatenate([np.zeros(MK.FS // 2), _peer_control(),
                                np.zeros(2 * MK.FS)]))
    assert hs.turn == VA._TURN_PEER, io.msgs
    assert any("turn released" in m for m in io.msgs), io.msgs
