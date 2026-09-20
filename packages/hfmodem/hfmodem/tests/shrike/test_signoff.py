# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What reaches the air when this station, rather than its host, ends the link.

A LINK ONE END GIVES UP ON IS STILL UP AT THE OTHER END. PACTOR has no
inactivity timeout for the far station to recover with -- the only figure in the
corpus is the 30 s of grid extrapolation a station will still answer a late QRT
across (pactor1-timing.md §5) -- so the goodbye is the whole of what resolves a
channel, and a station that simply stops keying leaves the peer owing its side
of a cycle to somebody who left.

MEASURED, 2026-08-22, four sessions in one evening (~/act-01, -04, -06, -07).
Each connected in PACTOR-1, upgraded on the peer's grant, and read nothing back:

    [host] nothing decoded in 4 cycles since the upgrade -> the peer did not follow
    [host] the peer never answered the upgrade -> link falls back to PACTOR-1
    [host] max retries -> yield the link and listen
    [host] changeover -> IRS
    [host] no decodable traffic from the peer -> abort
    ** LINK DOWN ** ... no goodbye was sent

The yield is deliberate and it is not what failed: it exists to break a lost
turnaround, and a peer that never took the channel leaves this end holding the
RECEIVING role when the budget behind it runs out. Ending there put both
stations in the same role, which is what the operator heard as "it ended with
both of us acting as though we were receiving" and, the cycle before, as a peer
still keying control tones at a station that had already gone.

So the give-up is asked the one question that separates the two findings an
operator's ear cannot: did anything go out. QRT rides a data packet, so a
station receiving when its budget expires must take the link back with a
CS3-headed break-in before it can say anything at all.

Run: python -m pytest hfmodem/tests/shrike/test_signoff.py
"""
from __future__ import annotations

from hfmodem.shrike import spec
from hfmodem.shrike.arq import (CS_ACK, CS_BREAKIN, CS_REQUEST,
                                GOODBYE_CYCLES, IRS, ISS, RECLAIM_CODEWORDS,
                                ArqConfig, ArqIO, PactorArq, State)

CFG = ArqConfig(max_retries=4)


class _Air(ArqIO):
    """Every burst this station keys, in the order it keys them."""

    def __init__(self):
        self.packets: list[tuple[int, bool]] = []      # (status byte, break-in)
        self.cs: list[int] = []
        self.lines: list[str] = []

    def connect_burst(self, mycall, dxcall):
        pass

    def send_packet(self, sl, payload, status, breakin=False):
        self.packets.append((status, breakin))

    def send_cs(self, cs_index):
        self.cs.append(cs_index)

    def send_p1_cs(self, index):
        self.cs.append(index)

    def upgrade(self, payload_waiting):
        return False

    def log(self, msg):
        self.lines.append(msg)


def _qrts(air: _Air) -> list[tuple[int, bool]]:
    return [p for p in air.packets if p[0] & spec.STATUS_QRT]


def _stranded() -> tuple[_Air, PactorArq]:
    """The 2026-08-22 shape, up to and including the yield.

    A link that comes up, is handed packets it discards on its own role -- the
    stranded-ISS signature the yield exists for -- and hands the channel to a
    peer that never asked for it.
    """
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_host_data(b"hello")
    for _ in range(CFG.max_retries + 4):
        if a.role == IRS:
            break
        a.on_rx_packet(1, b"", 0, True)
        a.on_cycle()
    assert a.role == IRS, "the deafness yield never fired"
    return air, a


def test_the_receiving_end_takes_the_link_back_to_say_goodbye():
    air, a = _stranded()
    for _ in range(CFG.max_retries + GOODBYE_CYCLES + 4):
        a.on_cycle()
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    goodbye = _qrts(air)
    assert goodbye, f"nothing said goodbye; log {air.lines[-3:]}"
    assert goodbye[0][1], "the goodbye did not break in, so nobody heard it"
    assert a.said_goodbye


def test_the_sending_end_says_goodbye_too():
    """The other give-up door: retries spent with the channel already ours."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_host_data(b"hello")
    for _ in range(CFG.max_retries + GOODBYE_CYCLES + 6):
        a.on_cycle()
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    assert _qrts(air), f"nothing said goodbye; log {air.lines[-3:]}"
    assert a.said_goodbye


def test_the_goodbye_is_bounded():
    """A peer that answers nothing must not be talked at for the whole budget."""
    air, a = _stranded()
    for _ in range(60):
        a.on_cycle()
    assert len(_qrts(air)) <= GOODBYE_CYCLES + 1, air.packets


def test_a_goodbye_nobody_answers_still_ends_the_link():
    air, a = _stranded()
    for _ in range(60):
        a.on_cycle()
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    assert any("the goodbye went unanswered" in m for m in air.lines), air.lines[-3:]


def test_an_answered_goodbye_closes_on_the_spot():
    """The acknowledgement is the clean close, and it must still be the fast one."""
    air, a = _stranded()
    for _ in range(CFG.max_retries + 4):
        if a.state == State.DISCONNECTING:
            break
        a.on_cycle()
    a.on_rx_cs(CS_ACK)
    assert a.state in (State.DISCONNECTED, State.LISTENING), a.state
    assert len(_qrts(air)) == 1, air.packets


# --------------------------------------------------------------------------- #
# The strand the sign-off was standing in for -- KB5LZK, 2026-08-28,
# captures/onair-0828-1838. That arm is void for timing (27 x SLOT IS GONE, and
# the run's own SESSION INVALID at cycle 6); its ROLE sequence is not, and it is
# what these pin. This station read the peer's CS3, changed over to IRS, and
# then spent sixteen consecutive cycles from TX[19] sending nothing but 120 ms
# acknowledgements while the far end sent no data at all and answered with bare
# codewords -- `rx CS REQ` standing in the host log through the whole of it.
# --------------------------------------------------------------------------- #
def _receiving() -> tuple[_Air, PactorArq]:
    """The arm's own route into the receiving role: the peer broke in."""
    air = _Air()
    a = PactorArq(air, CFG)
    a.on_host_connect("W9SSJ", "KI0BK")
    a.on_rx_cs(CS_ACK)
    a.on_host_data(b"hello")
    a.on_cycle()
    a.on_rx_cs(CS_BREAKIN)
    assert a.role == IRS, "the peer's break-in did not hand us the receiving role"
    return air, a


def test_a_stranded_irs_takes_the_link_back_rather_than_signing_off():
    air, a = _receiving()
    for _ in range(RECLAIM_CODEWORDS):
        a.on_rx_cs(CS_REQUEST)          # a bare codeword: the peer is an IRS too
        a.on_cycle()
    assert a.role == ISS, f"still receiving; log {air.lines[-3:]}"
    assert any(p[1] for p in air.packets), \
        f"the link changed hands with nothing on the air: {air.packets}"
    assert not _qrts(air), f"it signed off instead of reclaiming: {air.packets}"


def test_the_reclaim_needs_the_peer_and_not_our_own_counter():
    """The `_on_nak` correction read from the receiving side.

    A changeover is a statement about the far end, and the live role-desync
    issue is one that fired on this station's own retry count with nothing from
    the peer asking for it. So a budget running out is not a reclaim: an IRS
    that hears nothing at all keeps the receiving role and says goodbye.
    """
    air, a = _receiving()
    for _ in range(CFG.max_retries + GOODBYE_CYCLES + 4):
        a.on_cycle()
    assert a.said_goodbye, f"no sign-off; log {air.lines[-3:]}"
    assert not any("take the link back" in m for m in air.lines), air.lines[-3:]


def test_one_codeword_short_of_the_finding_does_not_reclaim():
    air, a = _receiving()
    for _ in range(RECLAIM_CODEWORDS - 1):
        a.on_rx_cs(CS_REQUEST)
        a.on_cycle()
    assert a.role == IRS, f"reclaimed on {RECLAIM_CODEWORDS - 1} codewords"


def test_a_changeover_head_withdraws_the_finding():
    """CS3 is the peer holding the SENDING role, so there is no strand to end.

    It is also the case the reclaim must not steal: the WS8EOC 2026-08-09
    sessions read the peer's changeover packet head every cycle, and a station
    that took the channel off one would be keying over a packet in flight.
    """
    air, a = _receiving()
    for _ in range(RECLAIM_CODEWORDS * 3):
        a.on_rx_cs(CS_REQUEST)
        a.on_rx_cs(CS_BREAKIN)      # ...and the head of a packet we cannot read
        a.on_cycle()
        assert a.role == IRS, f"reclaimed against a sending peer; {air.lines[-3:]}"
    assert not _qrts(air), f"and the budget must not end it either: {air.lines[-3:]}"


# --------------------------------------------------------------------------- #
# What the link-dead budget spends itself on. Presence identifies nobody
# (`note_peer_heard` says so in its own first line) and forgave ten cycles of
# the same arm on a third station's energy -- two with the grid reporting
# nothing heard at all, the rest putting the burst at -17, -79, -92, -107, -156
# or -216 ms, where an answer has to fall between 40 and 130. Position is what
# makes a burst evidence about the far end.
# --------------------------------------------------------------------------- #
def test_presence_no_longer_forgives_the_link_dead_budget():
    air, a = _receiving()
    for _ in range(CFG.max_retries + GOODBYE_CYCLES + 4):
        a.note_peer_heard()                       # energy, and no position
        a.note_burst(-79.0, at_anchor=False)      # ...measured out of the band
        a.on_cycle()
    assert a.said_goodbye, f"a dead link held open on presence; {air.lines[-3:]}"


def test_a_burst_where_the_answer_is_due_holds_the_link_open():
    air, a = _receiving()
    for _ in range(CFG.max_retries * 3):
        a.note_burst(95.0, at_anchor=True)
        a.on_cycle()
    assert not a.said_goodbye, f"signed off on an answered slot; {air.lines[-3:]}"
    assert a.state == State.CONNECTED, a.state
