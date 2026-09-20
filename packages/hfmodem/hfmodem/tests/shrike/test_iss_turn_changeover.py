# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Standing role requests keep the P3 changeover unacknowledged.

The failed 0913-2143 run repeated empty status-0x43 packets on the peer's old
raster. These are requests to take the channel, not evidence that our CS3
payload was accepted. The 0919-2325 repeat of status 0x40 exposed the cost of
that old interpretation: it discarded ;FW and advanced to counter 1 without a
control acknowledgement. The geometry fixtures still describe that failed
run; successful login progression below requires an actual CS1/CS2 answer.
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_iss_login_advance import (
    CHANGEOVER_FIELD, GREETING, ScriptedPeer)
from hfmodem.tests.shrike.test_p3_breakin_timing import transitioned, tx_at
from hfmodem.winlink import B2FSession, MailClient

# `ptc._answer_link_setup` under `--announce-lower`, requeued by the grant path.
ANNOUNCE = b"1w9ssj\r"
CEDE = 0x43                     # Standing request, not confirmation of our CS3.

CYCLE_N = 60000                 # 1.25 s
CS_N = 10080                    # the 210 ms PACTOR-3 control
WIDTH = 39120                   # the peer's 815 ms packet
REVERSE_GAP = 7225              # 150.5 ms, the arm's own
AFTER_CONTROL = REVERSE_GAP + CS_N
# The peer's last three CRC frames while we were still the IRS; the third is the
# onset hold 102 read (`+815 ms`) and the two before it lie on its comb.
IRS_FRAMES = (6708720, 6768720, 6828650)
# Every peer packet the offline rescan recovered during the ISS stint, holds
# 103-120, as RF onsets. Nine readings spanning 27 of the peer's cycles.
CEDE_ONSETS = (7008720, 7548480, 7668600, 7788480, 7908600,
               8268240, 8388600, 8508120, 8628240)


# --------------------------------------------------------------------------- #
# The grid: the reply-timing pair through the reversal.

def iss_grid(frames=IRS_FRAMES):
    """Corroborated as the IRS, then reversed by the changeover of hold 102."""
    g = transitioned()
    for at in frames:
        g.cycles += 1
        g.note_p3_control(at - AFTER_CONTROL)
        g.note_p3_packet(at, WIDTH, CYCLE_N)
    g.reverse(to_iss=True)
    return g


def hear(g, at, *, cycles=None):
    """One CRC-valid peer packet, `cycles` of ours after the last."""
    g.cycles += cycles if cycles is not None else 1
    g.note_p3_packet(at, WIDTH, CYCLE_N)
    return g._p3_reply_timing()


def test_iss_turn_the_arms_geometry_is_the_one_that_was_refused():
    g = iss_grid()
    assert g.cs_n == CS_N and g.cycle_n == CYCLE_N
    assert g._p3_raster_corroborated
    assert g.sending, "hold 102 keyed the changeover: this end is the ISS"
    timing = g._p3_reply_timing()
    assert timing.reverse_gap == REVERSE_GAP
    assert timing.packet_phase == IRS_FRAMES[-1]


def test_iss_turn_the_peer_held_one_raster_for_the_whole_stint():
    """PREMISE. The nine offline readings are one comb, 27 of its cycles long."""
    g = iss_grid()
    for at in CEDE_ONSETS:
        assert g._peer_raster_position(at, CYCLE_N) is not None, at
        g.note_p3_packet(at, WIDTH, CYCLE_N)
    assert round((CEDE_ONSETS[-1] - CEDE_ONSETS[0]) / CYCLE_N) == 27


def test_iss_turn_the_pair_is_re_anchored_on_every_frame_the_arm_decoded():
    g = iss_grid()
    previous = IRS_FRAMES[-1]
    for at in CEDE_ONSETS:
        timing = hear(g, at, cycles=round((at - previous) / CYCLE_N))
        assert timing is not None, at
        assert timing.packet_phase == at, "the age moves with the frame"
        # The link's geometry is not re-measured; only when it was read.
        assert timing.reverse_gap == REVERSE_GAP
        assert timing.control_phase == IRS_FRAMES[-1] - AFTER_CONTROL
        assert timing.packet_width == WIDTH and timing.cycle_n == CYCLE_N
        previous = at


def test_iss_turn_the_changeover_stays_placeable_while_the_peer_sends(tmp_path):
    """Twenty consecutive cycles, which is the stint the arm gave up in.

    Every one of them places at `BREAKIN_LEAD_S` past the peer's projected
    packet end, and the ISS refusal budget is never asked for.
    """
    g = iss_grid()
    for n in range(1, 21):
        at = CEDE_ONSETS[0] + n * CYCLE_N
        assert hear(g, at) is not None, n
        slot = (at - g.anchor) // CYCLE_N + 1
        end = g.peer_packet_end(slot)
        assert g.breakin_refusal(end) is None, n
        tx = tx_at(g, slot, tmp_path)
        assert tx._place_breakin() and tx.placed and tx.unplaceable is None
        past = tx.boundary - end.at_slot
        assert 0 < past <= round(onair.BREAKIN_LEAD_S * onair.FS) + g.d_max_n


def test_iss_turn_a_peer_that_stops_sending_ages_the_pair_out(tmp_path):
    """NEGATIVE CONTROL, and the arm's own eight keyings.

    Nothing re-anchors a pair nothing corroborates. The reading survives
    `ONSET_MAX_CYCLES` cycles past the reversal -- holds 102 and 104-110, the
    eight changeover packets that are on disk -- and the ninth is hold 111.
    """
    g = iss_grid()
    for age in range(1, onair.ONSET_MAX_CYCLES + 1):
        g.cycles += 1
        assert g._p3_reply_timing() is not None, age
    g.cycles += 1
    assert g._p3_reply_timing() is None
    slot = (g.anchor + 9 * CYCLE_N - g.anchor) // CYCLE_N + 1
    tx = tx_at(g, slot, tmp_path)
    assert tx._place_breakin() == ""
    assert "no fresh corroborated" in tx.unplaceable


@pytest.mark.parametrize("reason", ["reach", "comb"])
def test_iss_turn_a_frame_that_is_not_of_this_raster_re_anchors_nothing(reason):
    """NEGATIVE CONTROLS. The pair does not move, and then it ages out.

    `RASTER_PROJECT_CYCLES` bounds the ISS side as it bounds the IRS side, and a
    reading off the peer's comb by more than the tracker calls the same
    transmission is a reading of something else.
    """
    g = iss_grid()
    if reason == "reach":
        at, cycles = (IRS_FRAMES[-1] + (onair.RASTER_PROJECT_CYCLES + 1) * CYCLE_N,
                      onair.RASTER_PROJECT_CYCLES + 1)
    else:
        at, cycles = CEDE_ONSETS[0] + round(2 * onair.MAX_PULL_S * onair.FS), 3
    timing = hear(g, at, cycles=cycles)
    assert timing is None or timing.packet_phase == IRS_FRAMES[-1]
    g.cycles += onair.ONSET_MAX_CYCLES
    assert g._p3_reply_timing() is None


# --------------------------------------------------------------------------- #
# The link layer: what the changeover carries, and what answers it.

def at_the_prompt(*, announce: bytes = ANNOUNCE, allow_breakin: bool = True):
    """An IRS holding the complete greeting, the login queued behind `announce`."""
    peer = ScriptedPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    host.arq._sl = 3
    if announce:
        host._setup_bytes = announce
        host.arq.on_host_data(announce)
    mail = MailClient(B2FSession("W9SSJ", target="WS8EOC"), host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    mail.on_link_data(GREETING)
    assert mail.session.sent_text, "the client must have written its reply"
    host.app_turns(allow_breakin=allow_breakin)
    return host, peer


def takes_the_channel(**kw):
    host, peer = at_the_prompt(**kw)
    login = bytes(host.arq._outbuf)
    host.arq._breakin_armed = True
    host.tick()
    assert host.arq.role == arq.ISS
    return host, peer, login


def cede(host, *, status: int = CEDE, payload: bytes = b"", sl: int = 1):
    """The peer's own packet, as `rxfront` hands one to `PtcHost`."""
    host.on_rx_event(rxfront.Event(
        0.0, "packet", "scripted peer", protocol=spec.Protocol.PACTOR3,
        packet=(sl, status, payload, True)))


def test_iss_turn_the_changeover_carries_the_reply_and_not_the_announcement():
    host, peer, login = takes_the_channel()
    assert login.startswith(b";FW"), "the B2F login, and nothing ahead of it"
    sl, field, status, breakin = peer.last
    assert breakin and status & 3 == 0
    assert field == login[:CHANGEOVER_FIELD] == b";FW"
    assert field != ANNOUNCE[:CHANGEOVER_FIELD]
    assert ANNOUNCE not in bytes(host.arq._outbuf)
    assert host._txbuf == len(login)


def test_iss_turn_the_announcement_survives_until_the_application_answers():
    """NEGATIVE CONTROL. The grant path's requeue is untouched before the turn.

    Round 23's gate withholds the automatic break-in until the greeting is
    complete, and for as long as it does the announcement is still the only
    thing this link has to say.
    """
    host, peer = at_the_prompt(allow_breakin=False)
    assert not host.arq._breakin_pending
    assert bytes(host.arq._outbuf).startswith(ANNOUNCE)
    assert host._setup_bytes == ANNOUNCE


def test_iss_turn_an_announcement_with_nothing_behind_it_is_not_dropped():
    """NEGATIVE CONTROL. Dropping the whole queue keys an empty changeover."""
    peer = ScriptedPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    host._setup_bytes = ANNOUNCE
    host.arq.on_host_data(ANNOUNCE)
    host.app = MailClient(B2FSession("W9SSJ", target="WS8EOC"),
                          host.arq.on_host_data)
    host.app.link_up()
    host.app_turns()
    assert bytes(host.arq._outbuf) == ANNOUNCE


# --------------------------------------------------------------------------- #

def test_iss_turn_control_acknowledges_the_changeover_and_the_login_follows():
    host, peer, login = takes_the_channel()
    keyed = len(peer.bursts)
    peer.answer()

    assert host.arq.role == arq.ISS
    sl, field, status, breakin = peer.last
    assert len(peer.bursts) == keyed + 1, "one packet, in the cycle it arrived in"
    assert not breakin, "the changeover is spent; its head must not ride again"
    assert field == login[CHANGEOVER_FIELD:], "the next chunk of the login"
    assert status & 3 == 1, "the counter continues from the changeover's zero"
    assert sl == 3, "the traffic level, not the level the peer answered in"


def test_iss_turn_control_ack_is_followed_by_a_packet_and_never_a_codeword():
    """The former ISS acknowledges with a control on its new IRS raster."""
    class Watched(ScriptedPeer):
        codewords = 0

        def send_cs(self, cs_index):
            self.codewords += 1

    host, peer, login = takes_the_channel()
    peer.__class__ = Watched
    peer.codewords = 0
    peer.answer()
    host.tick()
    assert peer.codewords == 0
    assert not peer.last[3] and peer.last[1]


def test_iss_turn_the_requests_empty_field_never_reaches_the_host():
    host, peer, login = takes_the_channel()
    cede(host)
    assert host.rcvd_total == 0
    assert not any("rx data" in line for line in host.log_lines)


def test_iss_turn_the_login_reaches_the_air_with_control_acknowledgements():
    host, peer, login = takes_the_channel()
    sent = bytearray(peer.last[1])
    counters = [peer.last[2] & 3]
    peer.answer()
    for _ in range(6):
        sent += peer.last[1]
        counters.append(peer.last[2] & 3)
        if peer.last[2] & spec.STATUS_CHANGEOVER:
            break
        peer.answer()
        host.app_turns()
        host.tick()
    assert bytes(sent) == login, "every login byte, once, in order"
    assert counters == [n % 4 for n in range(len(counters))]
    assert not host.arq._outbuf


def test_iss_turn_the_stint_tail_is_not_the_cede():
    """NEGATIVE CONTROL. The peer's last packet carries the counter we await.

    `_take_link` flips the role a cycle before the peer can read the CS3, so a
    packet already cut arrives behind it. That is the closing of the stint we
    just ended, it IS delivered, and it settles nothing of ours.
    """
    host, peer, login = takes_the_channel()
    assert host.arq._stint_tail
    cede(host, status=0x40 | host.arq._expected_seq, payload=b"tail", sl=3)
    assert host.arq.unconfirmed_breakin, "the changeover is still in flight"
    assert peer.last[3], "and it is still the burst in hand"
    assert host.rcvd_total == len(b"tail")


def test_iss_turn_a_goodbye_beside_the_request_is_not_a_cede():
    """NEGATIVE CONTROL. Bit 7 outranks bit 6, as it does on the IRS side."""
    host, peer, login = takes_the_channel()
    host.arq._stint_tail = False
    cede(host, status=CEDE | spec.STATUS_QRT)
    assert host.arq.unconfirmed_breakin, "a peer signing off has ceded nothing"
    assert peer.last[3] and peer.last[1] == login[:CHANGEOVER_FIELD]


def test_iss_turn_a_peer_still_keying_does_not_spend_the_refusal_budget():
    """The seam cannot place the changeover; the peer is transmitting anyway.

    Round 23's rule from the sending side: a CRC-valid frame in the cycle is
    the refutation of a seam nothing can get past, and the reply position the
    placement wants is re-anchored on exactly those frames.
    """
    host, peer, login = takes_the_channel()
    peer.refuse = True
    for _ in range(4 * host.arq.cfg.max_retries):
        cede(host, status=0x0b, payload=b"x", sl=3)     # no bit 6: not the cede
        host.tick()
    assert host.arq.state is arq.State.CONNECTED
    assert not host.arq._qrt_pending
    assert host.arq._refused_cycles == 0
    assert host.arq.unconfirmed_breakin, "the changeover is still what we owe"
    assert bytes(host.arq._outbuf) == login[CHANGEOVER_FIELD:]


def test_iss_turn_a_refused_changeover_still_ends_on_the_peers_silence():
    """...and the bound is unchanged the moment the frames stop."""
    host, peer, login = takes_the_channel()
    peer.refuse = True
    cede(host, status=0x0b, payload=b"x", sl=3)
    host.tick()
    assert host.arq._refused_cycles == 0
    for n in range(1, host.arq.cfg.max_retries + 2):
        host.tick()
        if host.arq._qrt_pending:
            break
    assert n == host.arq.cfg.max_retries + 1
    assert any("transmit seam refused every cycle" in line
               for line in host.log_lines)


@pytest.mark.parametrize("status", [0x43, 0x40, 0x41, 0x42])
def test_iss_turn_standing_request_does_not_acknowledge_changeover(status):
    host, peer, login = takes_the_channel()
    host.arq._stint_tail = False
    pending = host.arq._inflight
    buffered = host.arq._buffer_raw
    keyed = len(peer.bursts)
    for _ in range(3):
        cede(host, status=status)
        assert host.arq._inflight is pending
        assert host.arq.unconfirmed_breakin
        assert host.arq._buffer_raw == buffered
        assert len(peer.bursts) == keyed
    assert pending.payload == b";FW" and pending.status & 3 == 0
    peer.answer()
    assert not host.arq.unconfirmed_breakin
    assert host.arq._buffer_raw == buffered - CHANGEOVER_FIELD
    assert peer.last[1] == login[CHANGEOVER_FIELD:]
    assert peer.last[2] & 3 == 1


def test_iss_turn_request_then_control_ack_keeps_every_login_byte():
    host, peer, login = takes_the_channel()
    cede(host)
    assert host.arq.unconfirmed_breakin
    peer.answer()
    for _ in range(6):
        if not host.arq._outbuf:
            break
        peer.answer()
        host.app_turns()
        host.tick()
    assert b"".join(burst[1] for burst in peer.bursts) == login
    assert not host.arq._qrt_pending


# --------------------------------------------------------------------------- #
# The window the peer's answer has to arrive in.

# What an ISS keying 810 ms of a 1250 ms cycle leaves, and what a cycle given to
# the receiver has instead. The arm's own two numbers: every keyed cycle of the
# 2320 stint collected 0.356-0.367 s.
KEYED_WINDOW_S = 0.365
LISTEN_WINDOW_S = 1.250
# Where the peer's burst opens in the window. Its packet is 0.86 s wide, so it
# fits one of the two and is cut by our own carrier in the other.
LEAD_N = round(0.020 * onair.FS)
FEED_N = onair.FEED_MAX_SLOTS * CYCLE_N


class _Live:
    """The capture stream the listen loop reads, over one window of audio."""

    def __init__(self, audio):
        self.audio, self.pos = audio, 0

    def read(self, n):
        got = self.audio[self.pos:self.pos + n]
        self.pos += got.size
        return np.ascontiguousarray(got)


@cache
def cede_burst():
    """The peer's own transmission: 0 bytes at SL1, status 0x43, as rendered."""
    return np.asarray(placement.link_packet(1, b"", CEDE), np.float32)


def cede_window(seconds: float) -> np.ndarray:
    """One cycle's window with the peer's packet at `LEAD_N`, cut to length."""
    win = np.zeros(round(seconds * onair.FS), np.float32)
    burst = cede_burst()[:max(0, win.size - LEAD_N)]
    win[LEAD_N:LEAD_N + burst.size] += burst
    return win


def stint(cycles: int, *, hush: bool):
    """The 2320 stint: our changeover keyed, the peer answering every cycle.

    The whole cycle, through the seams that decide it -- the production listen
    loop into the production `_SessionRx`, the grid the decode re-anchors, and
    the ARQ deciding which kind of cycle this is. `hush=False` reverts that last
    decision to what flew: every cycle keyed, every window 0.365 s.

    The peer's side is its own: one packet on its own raster, every cycle, for
    as long as the stint runs, which is what WS8EOC did 25 times.
    """
    host, peer, login = takes_the_channel()
    grid = iss_grid()
    peer.raster = grid
    sessrx = onair._SessionRx(host, tag="ISS")
    # Put the session's clock where the arm's was, so the frames this reads land
    # on the comb `iss_grid` corroborated rather than at the origin.
    sessrx.skip((CEDE_ONSETS[0] - LEAD_N) / onair.FS)
    read = []
    for _ in range(cycles):
        listening = hush and host.arq.breakin_listen_due
        grid.cycles += 1
        sessrx.new_cycle()
        win = cede_window(LISTEN_WINDOW_S if listening else KEYED_WINDOW_S)
        before = sessrx.count
        onair._listen_until_answer(_Live(win), win.size, host, sessrx, FEED_N)
        sessrx.flush()
        sessrx.skip((CYCLE_N - win.size) / onair.FS)   # our own carrier
        # `onair.RadioTx.listening`: a cycle given to the receiver renders,
        # refuses at the seam and keys nothing.
        peer.refuse = listening
        host.tick()
        peer.refuse = False
        read.append(sessrx.count - before)
    return host, peer, grid, login, read


def test_iss_listen_a_keyed_cycle_cannot_read_the_peers_answer():
    """The window an ISS leaves is under every reader's floor, by construction.

    0.365 s against `live.MIN_DECODE_S`'s 0.5 and `RollingRx.flush`'s half
    second, with the peer's packet 0.86 s wide. Nothing decodes, so nothing can
    -- which is the arm: 27 hold windows read back, 0 events from the rolling
    decoder, the deep scan and the flush alike on every keyed one of them.
    """
    host, peer, grid, login, read = stint(1, hush=False)
    assert KEYED_WINDOW_S < onair.MIN_DECODE_S
    assert read == [0]
    assert host.arq.unconfirmed_breakin, "the changeover is still unanswered"
    assert host.arq.breakin_listen_due, "...so the next cycle is the receiver's"


def test_iss_listen_the_hushed_cycle_reads_request_without_settling_login():
    host, peer, grid, login, read = stint(2, hush=True)
    assert read[0] == 0, "the keyed cycle reads nothing"
    assert read[1] > 0, "the hushed one reads the peer's standing request"
    assert host.arq.unconfirmed_breakin
    assert host.arq._inflight.payload == login[:CHANGEOVER_FIELD]
    assert host.arq._inflight.status & 3 == 0
    assert host.arq._buffer_raw == len(login)
    assert all(b[3] for b in peer.bursts)
    assert not host.arq._qrt_pending and host.arq.state is arq.State.CONNECTED


def test_iss_listen_the_hushed_cycle_re_anchors_the_reply_timing_pair():
    """...and the pair the placement stands on is refreshed by that same frame.

    `_SessionRx` hands a CRC-valid frame to `_MasterGrid.note_p3_packet`, which
    is the path round 26's `_reanchor_p3_reply_timing` was written for and was
    never once reached on the arm.
    """
    host, peer, grid, login, read = stint(2, hush=True)
    timing = grid._p3_reply_timing()
    assert timing is not None
    assert timing.packet_phase > IRS_FRAMES[-1], "a fresh frame, not the old one"
    assert grid._peer_raster_position(timing.packet_phase, CYCLE_N) is not None
    assert timing.reverse_gap == REVERSE_GAP, "the geometry is not re-measured"


def test_iss_listen_reverted_the_stint_starves_exactly_as_it_flew():
    """NEGATIVE CONTROL, and it is the arm. Every cycle keyed, nothing read.

    Nine cycles of a peer transmitting on its own comb, none of them readable,
    and the reply-timing pair ages out at `ONSET_MAX_CYCLES` with the peer still
    keying -- "projected 9 cycles", then None, then a changeover with nowhere to
    be placed. This is what the hush is measured against.
    """
    keyed = onair.ONSET_MAX_CYCLES + 1
    host, peer, grid, login, read = stint(keyed, hush=False)
    assert read == [0] * keyed, "nine cycles, and not one of them readable"
    assert grid._p3_reply_timing() is None, "the pair aged out under a keying peer"
    assert [b[3] for b in peer.bursts] == [True] * keyed, "nine changeovers"
    assert host.arq._qrt_pending
    assert any("max retries" in line for line in host.log_lines)
    assert {b[1] for b in peer.bursts} == {login[:CHANGEOVER_FIELD]}, (
        "the same three bytes nine times; the rest of the login never went out")
    assert not host.arq._outbuf, "...and the give-up discarded it"


@pytest.mark.parametrize("owed", [True, False])
def test_iss_listen_the_occupied_slot_is_the_station_we_are_answering(owed,
                                                                      monkeypatch):
    """Between keying a changeover and reading its answer, the peer holds the air.

    `_report_answer_band` asked this by ROLE, so at the ISS end -- exactly where
    the 2320 arm was -- hold 210 printed `ANSWER SLOT OCCUPIED` against the
    station it was answering. An unconfirmed changeover is the same position the
    IRS branch describes: the peer is still keying its own packet on its own
    raster, which is the whole reason the changeover went out.
    """
    host, peer, login = takes_the_channel()
    if not owed:
        peer.answer()                       # actual control confirms the turn
        assert not host.arq.unconfirmed_breakin
    asked = {}
    monkeypatch.setattr(onair._AnswerBand, "sight",
                        lambda self, *a, **kw: asked.update(kw))
    onair._report_answer_band(onair._AnswerBand(), host, iss_grid(),
                              np.zeros(CYCLE_N, np.float32),
                              since_tx=CYCLE_N, d_max_n=6240, read=True)
    assert asked["peer_owes_a_packet"] is owed
