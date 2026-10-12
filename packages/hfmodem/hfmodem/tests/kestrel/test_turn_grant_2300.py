# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A stock Trimode hands the turn over with its 11-symbol control burst.

KB8AY answered our acknowledgement of its last over with answer-stream frame 13
keyed to its own callsign, twice on 2026-09-26 — 0.14-0.16 s behind our last
sample, 0.50 s long, scoring 2.65-2.67 where the next frame of any stream
scores 1.12  [working/vara-levels-air-0927/arm2.console, tape 59.2 and 100.1 s].
It is the burst the two-stock BW500 bench labelled TURN behind the caller's
turn-request  [vara_frames, CONTROL_TAIL_TURN], and the same handover the
17- and 32-symbol responder frames carry. The peer's-turn branch read those two
and nothing else, so the run logged "no release in KB8AY's cadence" and
re-keyed the control burst 10 s later across a channel it had been handed.

The burst is generated, not taped: the same generator that keys our own answers
renders the peer's  [vara_frames, turn_grant].
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

CALLER, CALLED, BW = "W9SSJ", "KB8AY", "2300"
FS = MK.FS

#: Our last sample to the grant's first, as KB8AY keyed it.
_TURNAROUND = 0.15


class IO:
    def __init__(self):
        self.log_lines, self.sent, self.at = [], [], []
        self.now = 0.0

    def key(self, on): ...

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))
        self.at.append(self.now)

    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, p): ...
    def log(self, m): self.log_lines.append(m)


def _after_our_ack(queued: int):
    """The station as the arm left it: the peer's last over answered with our
    control burst, the turn still the peer's and its release owed."""
    io = IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw=BW)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    hs._txq = [b"x" * 89] * queued
    hs.tx_level = 3
    hs._key_over_answer(last=True, owes_release=False)
    assert hs._answer_owed == VA._OWED_RELEASE and len(io.sent) == 1
    return hs, io


def _grant(called: str = CALLED, trail: float = 1.5) -> np.ndarray:
    return np.concatenate([np.zeros(int(_TURNAROUND * FS)),
                           MK.synth_tone_pairs(VF.turn_grant(called, BW)),
                           np.zeros(int(trail * FS))])


def _drive(hs, io, x, block: float = 0.125) -> None:
    n = int(block * FS)
    for a in range(0, len(x), n):
        io.now = (a + n) / FS
        hs.on_rx_stream(x[a:a + n])


def _last_symbol() -> float:
    return _TURNAROUND + VA._POLL_MIN / FS


@pytest.mark.parametrize("queued", [0, 27])
def test_the_grant_behind_our_final_ack_is_taken_on_the_stream(queued):
    hs, io = _after_our_ack(queued)
    _drive(hs, io, _grant())

    assert any(m.startswith("rx turn grant") for m in io.log_lines), io.log_lines
    assert hs.turn == VA._TURN_OURS, io.log_lines
    assert 0 <= hs._grant_held <= VA._GRANT_FRESH_S
    assert hs._answer_owed is None
    assert not any("no release in" in m for m in io.log_lines), io.log_lines

    assert len(io.sent) == 2, f"{len(io.sent) - 1} bursts answered one grant"
    assert io.at[1] >= _last_symbol(), (
        f"keyed {_last_symbol() - io.at[1]:.3f} s inside the grant")
    if queued:
        assert len(io.sent[1]) > 4 * FS and len(hs._txq) == queued - 1
        assert any("tx DATA over" in m for m in io.log_lines), io.log_lines
    else:
        # Handed the turn empty, stock polls with its control burst, frame 1, and
        # does not hand it back: two stock 4.9s at BW2300, 2026-09-28. The release
        # this asserted until then is what drew nothing from KD7UHR the same day
        # [see test_idle_poll].
        poll = VF.control_bursts(CALLER, BW, frame=1)[0]
        assert np.array_equal(io.sent[1], MK.synth_tone_pairs(poll)), (
            "an empty queue answers the handover with the idle poll")


def test_the_grant_is_read_whole_and_not_by_its_preamble():
    hs, _ = _after_our_ack(1)
    assert hs._peer_turn_grant(_grant(trail=0.3))
    assert not hs._peer_turn_grant(_grant("N0XYZ", trail=0.3))
    ours = np.concatenate([np.zeros(int(_TURNAROUND * FS)),
                           MK.synth_tone_pairs(VF.control_bursts(CALLER, BW)[0]),
                           np.zeros(int(0.3 * FS))])
    assert not hs._peer_turn_grant(ours)


def test_a_foreign_grant_leaves_the_turn_where_it_was():
    hs, io = _after_our_ack(1)
    _drive(hs, io, _grant("N0XYZ"))
    assert hs.turn == VA._TURN_PEER and len(io.sent) == 1, io.log_lines


def test_the_bench_turn_tail_is_this_frame():
    assert VF.turn_grant("BBBB2", "500")[len(VF.CONNECTED_ACK_PREAMBLE):] == \
        VF.CONTROL_TAIL_TURN
