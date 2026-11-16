# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Three ways our PACTOR-3 changeover went unread on 2026-11-14, each pinned.

K4MSU 02:01Z read the changeover keyed at +85 ms past its packet at once, then
read none of about 45 keyed on the held grid after the comb drifted 8 ms in
front of that instant. WS8EOC 01:58Z repeated its long packet over our CS6,
the changeover went out on the held grid for want of a fresh packet/control
pair, and fifteen were keyed unread. In both sessions the turn was never
handed back.
"""
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.p3_reply_support import duplex, host_stub
from hfmodem.tests.shrike.test_p3_breakin_timing import (
    PAYLOAD, STATUS, changeover_lead, transitioned)

FS = onair.FS
CYCLE = 60000

# K4MSU 2026-11-14 02:01Z, log line 1382: "emitted control phase 9450087, CRC
# packet phase 9466872; reverse gap 139.69 ms, forward gap 85.31 ms", on the
# 815 ms short packet its changeover was read behind at +85 (TX[52]).
CONTROL, PACKET, WIDTH = 9450087, 9466872, 39120


def k4msu(early_n=round(.008 * FS)):
    """The recorded pair, with our comb `early_n` in front of the read (TX[77])."""
    g = onair._MasterGrid(0, CYCLE, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 38880
    g.note_p3_packet(PACKET - CYCLE, WIDTH, CYCLE)
    g.note_p3_control(CONTROL)
    g.note_p3_packet(PACKET, WIDTH, CYCLE)
    read = PACKET + WIDTH + g.peer_read_gap
    g.anchor = read - early_n - CYCLE
    return g, read


def test_the_recorded_pair_puts_the_read_at_85_ms():
    g, read = k4msu()
    timing = g._p3_reply_timing()
    assert timing is not None
    assert round(timing.reverse_gap / FS * 1e3, 2) == 139.69
    assert round(g.peer_read_gap / FS * 1e3, 2) == 85.31


def test_a_comb_8_ms_early_keys_the_changeover_at_the_read(tmp_path):
    g, read = k4msu()
    tx = onair.RadioTx(None, transmit=False, settle=.040, outdir=tmp_path)
    tx.aim(g, 1)
    end = g.peer_packet_end(1)
    assert "8 ms" in tx._clamp_refusal(end)  # The disagreement is still named.
    placed = tx._place_breakin()
    assert tx.unplaceable is None and tx.placed
    assert placed.startswith(" +85 ms past the peer's packet end")
    assert tx.boundary == read == end.at_slot + g.peer_read_gap
    tx.breakin_due = True
    assert tx.key_instant(g, 1) <= tx._breakin_reference(g, 1)


def test_the_boundary_knob_keeps_the_old_clamp(tmp_path):
    g, read = k4msu()
    tx = onair.RadioTx(None, transmit=False, settle=.040, outdir=tmp_path)
    tx.p3_breakin_place = "boundary"
    tx.aim(g, 1)
    assert tx._place_breakin() == " on the held grid"
    assert "8 ms" in tx.unplaceable
    assert tx.boundary == tx._breakin_reference(g, 1) == read - round(.008 * FS)


def test_the_settle_floor_still_binds(tmp_path):
    g, read = k4msu()
    tx = onair.RadioTx(None, transmit=False, settle=.070, outdir=tmp_path)
    tx.aim(g, 1)
    tx._place_breakin()
    end = g.peer_packet_end(1)
    assert tx.boundary == end.at_slot + round((.070 + onair.PREKEY_RESERVE_S) * FS)


def _stale():
    g = transitioned()
    g.note_p3_packet(2568704, 38880, CYCLE)
    assert g.p3_pair_stale and g._p3_reply_timing() is None
    return g


def test_the_irs_changeover_with_no_fresh_pair_gives_way_to_its_answer(tmp_path, capsys):
    """WS8EOC 01:58Z TX[12]: keyed on the held grid, unread, fourteen more after it."""
    g = _stale()
    tx = duplex(g, tmp_path)
    tx._flip = lambda: False
    tx.aim(g, 44)
    tx.breakin_due = True
    tx.live.now = tx.live.pos = tx._breakin_reference(g, 44) - changeover_lead() - 2000 \
        - tx.live.key_notice
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    said = capsys.readouterr().out
    assert "CHANGEOVER NOT PLACED" in said and "no fresh corroborated" in said
    assert "ON THE HELD GRID" not in said
    assert not tx.live.emissions and tx.refused and not tx.breakin_due
    assert tx.unplaceable is None and not tx.placed


def test_the_changeover_refused_for_want_of_a_pair_keys_at_the_read_behind_the_next_packet(
        tmp_path, capsys):
    """The default path past the refusal: the displaced answer and the peer's
    next packet are the fresh pair, and the changeover keys at the read."""
    g = _stale()
    tx = duplex(g, tmp_path)
    tx._flip = lambda: False
    tx.aim(g, 44)
    tx.breakin_due = True
    tx.live.now = tx.live.pos = tx._breakin_reference(g, 44) - changeover_lead() - 2000 \
        - tx.live.key_notice
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    # Our answer on slot 44's comb, then the peer's next packet a cycle on.
    g.cycles += 1
    g.note_p3_control(tx._breakin_reference(g, 44) - CYCLE)
    g.note_p3_packet(2628704, 38880, CYCLE)
    assert not g.p3_pair_stale
    capsys.readouterr()
    tx.refused = False
    tx.aim(g, 45)
    tx.breakin_due = True
    end = g.peer_packet_end(45)
    read = end.at_slot + g.peer_read_gap
    tx.live.now = tx.live.pos = read - changeover_lead() - 8000 - tx.live.key_notice
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) != arq.REFUSED
    said = capsys.readouterr().out
    assert "NOT PLACED" not in said and "HELD GRID" not in said
    assert f"+{round(g.peer_read_gap / FS * 1e3)} ms past the peer's packet end" in said
    assert len(tx.live.emissions) == 1 and not tx.refused


@pytest.mark.parametrize("why", ["held_knob", "iss_retry", "goodbye"])
def test_retries_goodbyes_and_the_held_knob_still_key_on_the_held_grid(why, tmp_path, capsys):
    g = _stale()
    tx = duplex(g, tmp_path)
    status = STATUS
    if why == "held_knob":
        tx.p3_breakin_unplaced = "held"
    elif why == "iss_retry":
        tx.host = host_stub(arq.ISS)
    else:
        status |= spec.STATUS_QRT
    tx._flip = lambda: False
    tx.aim(g, 44)
    tx.breakin_due = True
    held = tx._breakin_reference(g, 44)
    tx.live.now = tx.live.pos = held - changeover_lead() - 2000 - tx.live.key_notice
    assert tx.send_packet(3, PAYLOAD, status, breakin=True) != arq.REFUSED
    said = capsys.readouterr().out
    assert "CHANGEOVER ON THE HELD GRID" in said and "NOT PLACED" not in said
    assert len(tx.live.emissions) == 1 and not tx.refused


class Seam(arq.ArqIO):
    protocol = spec.Protocol.PACTOR3

    def __init__(self):
        self.events = []
        self.delivered = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.events.append(("packet", bytes(payload), breakin))
        return len(payload)

    def send_cs(self, index):
        self.events.append(("cs", index))

    def deliver(self, data):
        self.delivered.append(data)


def taken(tries=None):
    io = Seam()
    a = arq.PactorArq(io)
    if tries is not None:
        a.cfg.p3_breakin_tries = tries
    a.role = arq.IRS
    a._enter_connected()
    a.breakin_bytes_override = 3
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True, protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    a.on_rx_packet(1, b" Trim", 1, True, protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    a.on_host_data(b"ABCDEF")
    a.on_host_breakin()
    a.on_rx_packet(1, b"ode 1", 2, True, protocol=spec.Protocol.PACTOR3)
    io.events.clear()
    a.on_cycle()
    assert io.events == [("packet", b"ABC", True)]
    assert a.role == arq.ISS and a.unconfirmed_breakin
    return a, io


def test_three_unread_changeovers_hand_the_turn_back_and_the_slot_keys_the_ack():
    a, io = taken()
    a.on_cycle()
    a.on_cycle()
    assert io.events == [("packet", b"ABC", True)] * 3
    a.on_cycle()
    # THE FOURTH SLOT STILL KEYS: the ACK of the packet the peer is repeating.
    assert io.events[-1] == ("cs", arq.CS_ACK)
    assert a.role == arq.IRS and not a.unconfirmed_breakin
    assert a._inflight is None and bytes(a._outbuf) == b"ABCDEF"
    assert a.state == arq.State.CONNECTED and a._breakin_pending
    assert io.delivered == [b"RMS", b" Trim", b"ode 1"]
    # The peer's duplicate is ACKed, not broken into...
    a.on_rx_packet(1, b"ode 1", 2, True, protocol=spec.Protocol.PACTOR3)
    assert io.events[-1] == ("cs", arq.CS_ACK) and not a._breakin_armed
    keyed = len(io.events)
    a.on_cycle()
    assert len(io.events) == keyed      # that ACK was this cycle's answer
    # ...and its next packet carries the changeover again, on fresh timing.
    a.on_rx_packet(1, b"x", 3, True, protocol=spec.Protocol.PACTOR3)
    assert a._breakin_armed
    a.on_cycle()
    assert io.events[-1] == ("packet", b"ABC", True) and a.role == arq.ISS
    assert io.delivered == [b"RMS", b" Trim", b"ode 1", b"x"]


def test_an_answered_changeover_is_not_handed_back():
    a, io = taken()
    a.on_cycle()
    a.on_cycle()
    a.on_rx_cs(arq.CS_REQUEST)          # the peer has the turn and asks again
    a.on_cycle()
    a.on_cycle()
    a.on_cycle()
    # Answered once, so three more unanswered keyings before the hand-back.
    assert a.role == arq.ISS and a.unconfirmed_breakin
    assert all(e == ("packet", b"ABC", True) for e in io.events)


def test_a_peer_asking_us_to_send_is_not_handed_the_turn_back():
    """WS8EOC 0919 22:27 (captures/onair-0919-2227) sent 0x40 behind our third
    changeover and ACKed the fourth."""
    a, io = taken()
    a.on_cycle()
    a.on_cycle()
    a.on_rx_packet(1, b"", 0x40, True, protocol=spec.Protocol.PACTOR3)
    assert a._peer_wants_us_sending
    a.on_cycle()
    assert io.events == [("packet", b"ABC", True)] * 4
    assert a.role == arq.ISS and a.unconfirmed_breakin


def test_zero_tries_keeps_the_old_unbounded_changeover():
    a, io = taken(tries=0)
    for _ in range(6):
        a.on_cycle()
    assert io.events == [("packet", b"ABC", True)] * 7
    assert a.role == arq.ISS and a.unconfirmed_breakin


def test_the_tries_flag_reaches_the_arq():
    args = onair.build_parser().parse_args(["--p3-breakin-tries", "0"])
    assert args.p3_breakin_tries == 0
    args = onair.build_parser().parse_args([])
    assert args.p3_breakin_tries == arq.ArqConfig.p3_breakin_tries == 3
    assert args.p3_breakin_place == "read" and args.p3_breakin_unplaced == "answer"
    with pytest.raises(SystemExit):
        onair.build_parser().parse_args(["--p3-breakin-tries", "-1"])

