# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW500 greeting -> login handover, when stock says it with the drained frame.

Both arms of the stock500-chain2 rerun of 2026-09-11: our final 11-symbol ACK of
the greeting ends at 42.909 s on the clean arm, and stock VARA HF 4.9.0 answers
with `session-drained-responder` 32/32 — not the 17-symbol
`session-turn-release-responder` the 06:37z run keyed there — repeating it at
42.885, 54.181 and 66.501 s. It is the frame stock keys to its own stock caller
after a delivery, at 55.586-56.963 s of the two-stock tape, where that caller
polls 0.242 s behind its last sample.

Holding "release owed", this station let the frame reach nothing: the bracket
route declines the 17-symbol release on length and the grant branch on the turn,
so the release ladder spent both rungs — the control burst again at +11.3 s, the
turn-request at +22.7 s — and only the third copy, taken as a grant, opened the
turn. 25.9 s per arm for a handover that was on the air at the first copy
[analysis/stock500-chain2, analysis/stock500-recovery].

The clip is that first copy off the responder cable. Nothing here opens a device
or keys a radio.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_bw500_recovery import _IO, _station, _stream

FIXTURE = Path(__file__).with_name("fixtures") / "bw500-drained-handover"
CALLED = "KC9GHZ"
DRAINED = VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "500")
REPLY = b"FC EM ABCDEF 100 90 0\rF> 3b\r"
#: What the segmenter handed the live run at this frame: the grant it drew was
#: named +0.157 s from its last symbol.
TAIL = int(0.157 * MK.FS)


@cache
def metadata():
    path = FIXTURE / "provenance.json"
    if not path.exists():
        pytest.skip(f'stock BW500 drained-handover metadata absent: {path}')
    meta = json.loads(path.read_text())
    assert meta["link"]["called"] == CALLED
    return meta


def _clip() -> np.ndarray:
    path = FIXTURE / "kc9ghz-drained-handover.wav"
    if not path.exists():
        pytest.skip(f'stock BW500 drained-handover recording absent: {path}')
    rate, x = wavfile.read(path)
    assert rate == MK.FS
    return x.astype(float)


def _bracket(tail: int = TAIL) -> np.ndarray:
    a, b = metadata()["frame_in_clip"]
    return _clip()[a - MK.HOP:b + tail]


def _wide_burst(kind: VF.BurstKind) -> np.ndarray:
    burst = MK.synth_burst(CALLED, VF.for_bw(kind, "2300"))
    return np.concatenate([np.zeros(MK.FS // 2), burst, np.zeros(MK.FS)])


def _narrow():
    return _station(CALLED)


def _wide():
    """The same link at BW2300, where the stream route owns the short frames an
    energy gate cannot see  [see _stream_answer]."""
    io = _IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", CALLED
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    return hs, io


def _awaiting_release(reply: bytes = REPLY, station=_narrow):
    """The end of the peer's greeting: the host's answer is queued inside the
    delivery, our final ACK goes out, and the peer owes us the turn."""
    hs, io = station()
    hs._answering = True
    hs.send(reply)
    hs._answering = False
    hs._key_over_answer(True, bool(reply))
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    assert hs.turn == VA._TURN_PEER and len(io.sent) == 1
    return hs, io


def _rungs(io) -> int:
    return sum("re-keying the 11-symbol control burst" in m
               or "asking for the turn" in m for m in io.msgs)


def test_the_clip_holds_the_frame_its_provenance_names():
    meta = metadata()
    a, b = meta["frame_in_clip"]
    heard = MK.demod_burst(_clip()[a:b], DRAINED, MK.band_for("500"))
    assert heard == VF.handshake_tones(CALLED, DRAINED)
    assert heard != VF.handshake_tones(meta["link"]["caller"], DRAINED)
    assert (DRAINED.seed_off, DRAINED.preadv, DRAINED.keyed_by) == (288, 807, "called")


def test_the_drained_frame_hands_the_turn_over_and_the_queue_goes_out():
    hs, io = _awaiting_release()
    hs.on_rx_audio(_bracket())
    assert len(io.sent) == 2, io.msgs
    # burst_spans needs a falling edge; the live path brackets _OVER500_TAIL.
    over = np.concatenate([io.sent[1], np.zeros(MK.HOP)])
    assert RX.decode_stream(over).complete
    assert hs.turn == VA._TURN_OURS and hs._txq == []
    assert hs._answer_owed is None and hs._reacks == 0
    assert hs._asked == 0, "asked for a turn the peer had already handed over"
    assert _rungs(io) == 0, io.msgs
    assert any("has nothing more to send" in m for m in io.msgs), io.msgs


def test_the_over_goes_out_inside_the_gap_the_frame_opened():
    """Named off the audio held past its last symbol, which is the whole of the
    delay this file owns: the stock caller answers this frame at 0.242 s and the
    peer repeats it every 11.4 s."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(_bracket())
    assert any("s from its last symbol" in m and "has nothing more to send" in m
               for m in io.msgs), io.msgs
    # The fit's own plateau is ~1400 samples wide, so the age is the bracket's
    # tail to within a symbol.
    assert abs(hs._grant_held - TAIL / MK.FS) <= MK.HOP / MK.FS
    assert hs._grant_held <= VA._GRANT_FRESH_S


def test_a_bracket_a_block_past_the_frame_is_left_to_the_ladder():
    """The peer's gap has gone by then and the over would land on its next
    burst, which is `_stream_grant`'s measurement  [see _GRANT_FRESH_S]."""
    hs, io = _awaiting_release()
    hs.on_rx_audio(_bracket(tail=VA._STREAM_BLOCK + MK.HOP))
    assert len(io.sent) == 1 and hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_RELEASE


def test_an_empty_queue_keys_nothing_at_the_frame():
    """The release is still owed with nothing to put on the air, and an empty
    queue owes the peer no transmission  [see _took_responder_release]."""
    hs, io = _awaiting_release(reply=b"")
    hs.on_rx_audio(_bracket())
    assert len(io.sent) == 1 and hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_RELEASE


def test_a_release_that_is_not_owed_leaves_the_frame_alone():
    """The peer's own cadence between deliveries is not a handover: no over of
    its was acknowledged into this one, and the turn stays where it is."""
    hs, io = _narrow()
    hs._answering = True
    hs.send(REPLY)
    hs._answering = False
    assert hs._answer_owed is None and hs.turn == VA._TURN_PEER and hs._txq
    keyed, asked = len(io.sent), hs._asked
    hs.on_rx_audio(_bracket())
    assert len(io.sent) == keyed and hs.turn == VA._TURN_PEER
    assert hs._asked == asked


def test_the_grant_route_still_takes_it_after_a_real_turn_request():
    hs, io = _awaiting_release()
    hs.turn, hs._asked = VA._TURN_ASKED, 1
    hs.on_rx_audio(_bracket())
    assert hs.turn == VA._TURN_OURS and len(io.sent) == 2
    assert any("turn granted" in m for m in io.msgs), io.msgs


def test_the_narrow_stream_route_keys_nothing_at_it():
    """At BW500 the stream owns the over and the recovery cue; this frame is the
    bracket route's, and both routes answering it would key twice."""
    hs, io = _awaiting_release()
    _stream(hs, io, _clip())
    assert len(io.sent) == 1 and hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_RELEASE


def test_the_release_ladder_is_unchanged_where_the_peer_only_idles():
    hs, io = _awaiting_release()
    hs.on_rx_audio(MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE,
                                                    "500")))
    assert hs.turn == VA._TURN_PEER and hs._reacks == 1
    assert any("re-keying the 11-symbol control burst" in m for m in io.msgs), io.msgs


def test_the_wide_stream_keeps_the_release_and_declines_an_unowed_drained_frame():
    """Two frames hand the turn over at 2300/2750, and the second one is a state.

    Every handover on the stock tapes is `session-turn-release-responder` at
    17/17 — three BW2300 and ten BW2750 runs of
    working/vara-outbound-0910/stock/runs — and the stream still takes it.

    K0SI (RMS Trimode) handed over with the 32-symbol drained frame instead, at
    67.334 and 76.734 s, 31/31 tones, 0.12 s behind our ACK
    [docs/protocols/vara/20-peer-turn-measurements.md], so the stream takes that one too —
    where it is a handover  [test_drained_handover_2300]. Owed no release, or
    with nothing queued for the gap it opens, the turn stays the peer's.
    """
    hs, io = _awaiting_release(station=_wide)
    _stream(hs, io, _wide_burst(VF.SESSION_TURN_RELEASE_RESPONDER))
    assert hs.turn == VA._TURN_OURS and len(io.sent) == 2
    assert hs._answer_owed is None and hs._txq == []

    hs, io = _awaiting_release(reply=b"", station=_wide)
    _stream(hs, io, _wide_burst(VF.SESSION_DRAINED_RESPONDER))
    assert hs.turn == VA._TURN_PEER and len(io.sent) == 1
    assert hs._answer_owed == VA._OWED_RELEASE

    hs, io = _wide()
    hs._answering = True
    hs.send(REPLY)
    hs._answering = False
    assert hs._answer_owed is None and hs._txq
    _stream(hs, io, _wide_burst(VF.SESSION_DRAINED_RESPONDER))
    assert hs.turn == VA._TURN_PEER and not io.sent and hs._asked == 0
