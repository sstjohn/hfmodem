# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A gateway hands the channel over without being asked, and the greeting that
looks cut off is a greeting broken mid-word to hand it.

Three arms of the 2026-08-29 day slot, two gateways, each took one over of 89
bytes and then went quiet for 100-200 s without disconnecting. What the gateway
keyed 0.13-0.15 s behind this station's control burst is
``SESSION_TURN_RELEASE_RESPONDER``, the responder's release — the same handover
two stock 4.9.0s pass over a cable, arriving here with nothing having asked for
it.

The recogniser was already right. `_peer_responder_release` takes all three at
every alignment a live buffer can present them at; what it was never given was a
turn state to run in, because `_stream_grant` owns the frame only between a
turn-request of ours and its answer.

Given one, it was still a second late. The buffer these frames are found on waited
for the longest of them plus a scan block before it looked at anything, so a
0.73 s release lying 0.03 s past the cursor sat there whole for 1.11-1.18 s —
1.31-1.36 s on the air once the key-up is counted, against the 0.075-0.126 s two
stock 4.9.0s answer each other in. Both gateways had stopped listening.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _load_48k
from hfmodem.tests.kestrel.test_rx_segmenter import _record, _transport
from hfmodem.tests.kestrel.test_turn_law import _IO

kc = corpora.harness("kestrel_connect")

RELEASE = VF.SESSION_TURN_RELEASE_RESPONDER
MYCALL = "W9SSJ"
FS = MK.FS

#: Audio taken either side of the burst, enough that the stream route sees it whole
#: at more than one buffer boundary.
_LEAD, _TRAIL = 1.5, 1.5

#: The turnaround each arm actually ran, read off its envelope: our own control
#: burst's last sample, and where the capture buffer had reached when
#: `play_drained` came back — 0.58-0.62 s later, all of it channel. The audio in
#: between is the gateway keying the release, and the cursor was set at the second
#: number rather than the first.
_TURNAROUND = {"20260829T141915Z-W9SSJ-KB5LZK": (72.580, 73.200),
               "20260829T143002Z-W9SSJ-N5TW": (110.640, 111.220),
               "20260829T144437Z-W9SSJ-KB5LZK": (98.540, 99.160)}

#: The control burst's audio, first sample to last  [vara_frames, 11 symbols].
_CONTROL_S = 0.470


def _window(path, at: float) -> np.ndarray:
    x = _load_48k(path)
    a = max(0, int((at - _LEAD) * FS))
    return np.array(x[a:int((at + _TRAIL) * FS)])


def _connected(called: str):
    io = _IO()
    hs = VA.VaraStationHandshake([MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", called, MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    return hs, io


def _feed(hs, x, block: int = 4800) -> None:
    for i in range(0, len(x), block):
        hs.on_rx_stream(x[i:i + block])


# --------------------------------------------------------------------------- #
# What the frame is.
def test_the_release_sits_where_its_pre_advance_says_it_does():
    step = VF.lattice_step(RELEASE)
    assert (RELEASE.preadv - 1) % step == 0
    assert (RELEASE.preadv - 1) // step == corpora.UNASKED_HANDOVER_POSITION


@pytest.mark.parametrize("path,called,at,tones,state",
                         corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_burst_is_the_responders_release_at_its_own_lattice_position(
        path, called, at, tones, state):
    x = _window(path, at)
    heard = [int(t) for t in VA._payload_alignment(
        x, RELEASE, np.array(VF.handshake_tones(called, RELEASE), np.int32))]
    expected = VF.handshake_tones(called, RELEASE)
    assert sum(1 for a, b in zip(heard[2:], expected[2:]) if a == b) == tones
    assert VF.recognize(heard, called, RELEASE)
    if state is None:                    # tones misread, not missing: no state
        assert VF.payload_states(heard[2:], RELEASE) == ()
        return
    assert VF.payload_states(heard[2:], RELEASE) == (state,)
    assert (VF.payload_position(heard[2:], called, RELEASE, span=4096)
            == corpora.UNASKED_HANDOVER_POSITION)


@corpora.requires_onair_unasked_handover
def test_the_second_callsign_is_what_separates_the_state_from_the_identity():
    """Both gateways key one state at one position; neither reproduces the other's
    tones. A pair derived from a single recording would prove neither."""
    calls = {called for _, called, *_ in corpora.ONAIR_UNASKED_HANDOVER}
    assert len(calls) > 1
    for path, called, at, _tones, _state in corpora.ONAIR_UNASKED_HANDOVER:
        x = _window(path, at)
        heard = [int(t) for t in VA._payload_alignment(
            x, RELEASE, np.array(VF.handshake_tones(called, RELEASE), np.int32))]
        for other in calls - {called}:
            wrong = VF.handshake_tones(other, RELEASE)
            assert sum(1 for a, b in zip(heard, wrong) if a == b) <= 3
            assert not VF.recognize(heard, other, RELEASE)


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_no_other_frame_in_the_family_answers_to_these_tones(
        path, called, at, _t, _s):
    x = _window(path, at)
    heard = [int(t) for t in VA._payload_alignment(
        x, RELEASE, np.array(VF.handshake_tones(called, RELEASE), np.int32))]
    for kind in VF.BURSTS.values():
        if kind is RELEASE:
            continue
        wrong = VF.handshake_tones(called, kind)
        assert sum(1 for a, b in zip(heard, wrong) if a == b) <= 3, kind.name


# --------------------------------------------------------------------------- #
# What the cursor leaves in front of it.
@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_live_cursor_leaves_the_release_in_front_of_it(
        monkeypatch, path, called, at, _t, _s):
    """The frames above are read from a window cut around them. This one is read
    from where the running station's cursor actually stood.

    `tx` set it from the capture buffer's length when `play_drained` came back,
    and on this turnaround that is 0.58-0.62 s past our own last sample — the
    whole of the gateway's release but its last tenth. Every one of the three arms
    heard the handover, none of them saw it, and each answered with keepalives
    until the gateway timed out.
    """
    x = _load_48k(path)
    end, back = _TURNAROUND[path.stem]
    io = _transport(monkeypatch)

    def _slow_drain(block, fs, device):
        # Our own burst, the blackout, and the release the drain kept running through.
        _record(io, x[int((end - _CONTROL_S - 0.1) * FS):int(back * FS)])

    monkeypatch.setattr(kc, "play_drained", _slow_drain)
    _record(io, x[int((end - _CONTROL_S - 1.1) * FS):int((end - _CONTROL_S - 0.1) * FS)])
    io.tx(np.zeros(int(_CONTROL_S * FS)))
    _record(io, x[int(back * FS):int((back + _TRAIL) * FS)])

    hs, stub = _connected(called)
    io.next_rx_burst(timeout=0.0, hs=hs)
    assert any("session-turn-release-responder" in m for m in stub.msgs), stub.msgs

    stale, was = _connected(called)          # from the drain's return, guard and all
    _feed(stale, np.array(x[int(back * FS):int((back + _TRAIL) * FS)]))
    assert not any("session-turn-release-responder" in m for m in was.msgs), (
        "the release survives the old cursor too, so this arm proves nothing")


# --------------------------------------------------------------------------- #
# What the station owes it.
@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_release_is_read_while_the_turn_is_the_peers(path, called, at, _t, _s):
    hs, io = _connected(called)
    _feed(hs, _window(path, at))
    assert any("session-turn-release-responder" in m for m in io.msgs), io.msgs
    assert hs.turn == VA._TURN_PEER, "the channel was taken and never given back"


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_a_handover_with_nothing_queued_puts_the_channel_straight_back(
        path, called, at, _t, _s):
    """The whole of the failure: a release that goes no further than a variable
    leaves the gateway listening to its own inactivity timeout."""
    hs, io = _connected(called)
    _feed(hs, _window(path, at))
    keyed = [MK.demod_tones(s, 17) for s in io.sent
             if len(s) >= 16 * MK.HOP + MK.STRIDE]
    assert any(VF.recognize(t, called, VF.SESSION_TURN_RELEASE) for t in keyed), (
        "nothing went on the air, so the gateway heard no handover back")


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_a_handover_with_a_payload_queued_keys_the_over(path, called, at, _t, _s):
    hs, io = _connected(called)
    hs._txq = [b"x" * 89]
    _feed(hs, _window(path, at))
    assert hs.turn == VA._TURN_OURS and not hs._txq
    assert max(len(s) for s in io.sent) > 4 * FS, "no DATA over reached the air"


# --------------------------------------------------------------------------- #
# The negative population is the recordings themselves.
@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_nothing_else_in_these_recordings_is_this_frame(path, called, at, _t, _s):
    """1.87 s windows every 0.5 s across the whole session, clear of the burst."""
    x = _load_48k(path)
    need = VA._SESSION_NEED + VA._STREAM_BLOCK
    tones = np.array(VF.handshake_tones(called, RELEASE), np.int32)
    windows = accepts = 0
    worst = 0
    for a in range(0, len(x) - need, FS // 2):
        if a / FS - 0.8 <= at <= (a + need) / FS:
            continue
        heard = [int(t) for t in VA._payload_alignment(x[a:a + need], RELEASE, tones)]
        worst = max(worst, sum(1 for u, v in zip(heard[2:], tones[2:]) if u == v))
        accepts += bool(VF.recognize(heard, called, RELEASE))
        windows += 1
    assert windows > 300
    assert accepts == 0 and worst <= 5, f"{accepts} of {windows}, worst {worst}/15"


# --------------------------------------------------------------------------- #
# How soon the station owes it.
#
# The frames were right and the schedule was not. Every one of these numbers is
# read off the recordings by replaying the live receive stream from where the
# cursor actually stood, so what they time is this file's own scan grid and not
# the rig's key-up behind it.

#: The gateway's greeting over, located at 24 of 24 reference columns by
#: `_rec3_alignment` swept across the eight seconds in front of our control burst.
#: It ends 1.03-1.11 s before that burst's last sample on all three arms.
_OVER_AT = {"20260829T141915Z-W9SSJ-KB5LZK": 67.340,
            "20260829T143002Z-W9SSJ-N5TW": 105.340,
            "20260829T144437Z-W9SSJ-KB5LZK": 93.220}

#: `kestrel_connect.TX_IDLE_HOLD_S + RX_ECHO_GUARD_S` — where `tx` leaves the
#: receive cursor past our own last transmitted sample, and so where the buffer
#: these searches run on begins.
_CURSOR = 0.10

#: Two stock 4.9.0s answer each other's bursts this long after the last sample
#: [vara_frames, SESSION_TURN_RELEASE_RESPONDER]. It is a cable measurement and
#: carries no PTT, so it is the bound on the scan grid rather than on the air.
_BENCH_TURNAROUND = 0.126

_BLOCK = int(0.02 * FS)             # what `next_rx_burst` hands over per poll
#: The longest an acknowledgement waits, on top of the grid, for the columns
#: behind the frame to say whether the peer is keying another block into the same
#: window: six of them where the level is decisive, and where it is not, the
#: structural probe's own thirty-two  [see `vara_arq._window_state`].
_WINDOW_WAIT = VA._PROBE_COLS * 512


def _answered_at(x, cursor: float, called: str, queued: bytes | None = None):
    """Replay the receive stream from ``cursor`` and return where our answer was
    keyed, as an instant in the recording, with the log beside it."""
    hs, io = _connected(called)
    if queued is not None:
        hs._txq = [queued]
    a = int(cursor * FS)
    fed = 0
    while a + fed + _BLOCK < len(x) and not io.sent and fed < int(8 * FS):
        hs.on_rx_stream(x[a + fed:a + fed + _BLOCK])
        fed += _BLOCK
    return ((a + fed) / FS if io.sent else None), hs, io


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_release_is_answered_inside_the_turnaround_it_opened(
        path, called, at, _t, _s):
    """1.11-1.18 s before this, on all three arms, and 1.31-1.36 s on the air once
    the key-up behind it is counted. The gateway had stopped listening by then and
    six greetings never resumed.

    What that second was spent on: the buffer these frames are found on waited for
    the LONGEST of them plus a whole scan block — 1.87 s — before its first scan,
    while the release is 0.73 s and lands 0.03 s past the cursor. It was whole in
    the buffer for a second before anything looked at it.
    """
    x = _load_48k(path)
    keyed, _hs, io = _answered_at(x, _TURNAROUND[path.stem][0] + _CURSOR, called)
    assert keyed is not None, io.msgs
    late = keyed - (at + VA._span(RELEASE) / FS)
    assert late < _BENCH_TURNAROUND + _BLOCK / FS, (
        f"answered {late:.3f} s after the release ended")
    assert late > -0.10, "the answer cannot precede the frame it answers"
    tones = MK.demod_tones(io.sent[0], 17)
    assert VF.recognize(tones, called, VF.SESSION_TURN_RELEASE), (
        "the handover was answered with something that is not the release")


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_a_caller_with_payload_answers_the_release_just_as_promptly(
        path, called, at, _t, _s):
    """The case that already worked has to keep the schedule too: a queue turns the
    answer into a DATA over and must not turn it into a late one."""
    x = _load_48k(path)
    keyed, hs, io = _answered_at(x, _TURNAROUND[path.stem][0] + _CURSOR, called,
                                 queued=b"x" * 89)
    assert keyed is not None, io.msgs
    assert keyed - (at + VA._span(RELEASE) / FS) < _BENCH_TURNAROUND + _BLOCK / FS
    assert hs.turn == VA._TURN_OURS and not hs._txq
    assert len(io.sent[0]) > 4 * FS, "no DATA over reached the air"


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_over_is_answered_wherever_the_cursor_fell_in_front_of_it(
        path, called, at, _t, _s):
    """The per-over control burst, and the same defect one buffer over.

    Where the cursor lands in front of the gateway's over is the gateway's own
    turnaround and not ours, so the wait has to be bounded across the whole block
    it can fall in — swept here rather than taken at the one offset these arms
    happened to run. On a half-second grid it ran 0.09-0.54 s decided by nothing
    but that offset, which is what put our acknowledgement 0.51-0.63 s behind an
    over the bench answers in 0.155-0.169. On a quarter block it still ran
    0.037-0.137 by the same lottery; on the over route's own step it is
    -0.003-0.037 across all eighteen, and the offset decides 0.04 s of it.

    On top of that sits one cost, the same for every offset: the answer waits for
    the columns behind the frame, which are what say whether the peer is keying a
    second block into the same window  [see `_window_state`]. Sixty-four
    milliseconds where the band is decisive and 341 where the structural probe
    has to run, which on a real off-air over's own turnaround is the usual case.
    It is the offset's own spread this measures, and that is unmoved.
    """
    x = _load_48k(path)
    on = _OVER_AT[path.stem]
    for ahead in (0.00, 0.10, 0.15, 0.20, 0.30, 0.45):
        keyed, _hs, io = _answered_at(x, on - ahead, called)
        assert keyed is not None, (ahead, io.msgs)
        late = keyed - (on + VA._OVER_NEED / FS)
        assert late < (VA._OVER_TURNAROUND_STEP + _BLOCK + _WINDOW_WAIT) / FS, (
            f"cursor {ahead:.2f} s ahead of the over: answered {late:.3f} s late")
        assert any("rx DATA over" in m for m in io.msgs), (ahead, io.msgs)


def test_the_fine_grid_is_the_turnaround_and_the_steady_state_is_the_block():
    """A quarter-block step across the one block an answer to our own burst can be
    in, and the half-second everywhere else. The cost of this route was measured
    at that half-second and a grid that stayed fine would carry four times it.

    The over route asks for a finer one again: an over's last sample is the peer's
    unkey, so every sample of grid behind it stands in front of our answer.
    """
    for first in (VA._SESSION_MIN, VA._OVER_NEED):
        step = [VA._next_scan(n, first) - n
                for n in range(first, first + 4 * VA._STREAM_BLOCK,
                               VA._TURNAROUND_STEP)]
        assert step[:4] == [VA._TURNAROUND_STEP] * 4
        assert set(step[4:]) == {VA._STREAM_BLOCK}
        assert VA._next_scan(first - 1, first) - (first - 1) == VA._TURNAROUND_STEP

    over = [VA._next_scan(n, VA._OVER_NEED, VA._OVER_TURNAROUND_STEP) - n
            for n in range(VA._OVER_NEED,
                           VA._OVER_NEED + 4 * VA._STREAM_BLOCK,
                           VA._OVER_TURNAROUND_STEP)]
    assert over[:16] == [VA._OVER_TURNAROUND_STEP] * 16
    assert set(over[16:]) == {VA._STREAM_BLOCK}
    assert VA._OVER_TURNAROUND_STEP < VA._TURNAROUND_STEP


@pytest.mark.parametrize("path,called,at,_t,_s", corpora.ONAIR_UNASKED_HANDOVER,
                         ids=lambda v: getattr(v, "stem", None))
@corpora.requires_onair_unasked_handover
def test_the_shorter_buffers_take_nothing_the_longer_ones_did_not(
        path, called, at, _t, _s):
    """The first scans now run on 0.73-1.23 s of audio where they used to run on
    1.87, and a symbol past the end of a buffer is not comparable rather than
    wrong — so the shorter windows are a weaker test and the population they are
    weaker on is these recordings themselves.

    The buffer is emptied every 1.8 s so the fine grid runs the whole session
    instead of once per keying. Across all three arms that is ~1300 scans on a
    buffer shorter than the old threshold, and the only frame any of them takes is
    the handover.
    """
    x = _load_48k(path)
    hs, io = _connected(called)
    hs._stream_over = lambda s: None            # the wideband route is not on trial
    short = 0
    since = 0
    for i in range(0, len(x) - _BLOCK, _BLOCK):
        if since >= int(1.8 * FS):
            hs._reset_answer_search()
            since = 0
        hs.turn = VA._TURN_PEER                 # every scan asked the same question
        n, due = len(hs._ans_buf), hs._ans_due
        hs._stream_answer(x[i:i + _BLOCK])
        short += hs._ans_due != due and n + _BLOCK < VA._SESSION_NEED + VA._STREAM_BLOCK
        since += _BLOCK
    assert short > 300, f"only {short} scans ran on a short buffer"
    took = [m for m in io.msgs if m.startswith("rx ")]
    assert len(took) <= 1, took
    for m in took:
        assert "session-turn-release-responder" in m
