# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VE3KPG-80 0924-0033: our proposal leg, the counters under it, and the turn after.

`working/onair-0923-1933/pactor-current-ve3kpg-80-pounce-20260924T003320Z.log`,
--p3-mail --long-cycle --speed-policy stock. The gateway answered TX35 (c1) and
TX37 (c2) with CS4; the stock gear hold read both as repeat requests, re-cut each
counter to 5 bytes and requeued 18, and VE3KPG received
`[Pat-1.0.0-B2FHSJ\\r[Pat-1.0.0-B2FHM$]\\r;at-1.0.0-...`. When it did miss c3 it
answered CS1 on the odd counter.
"""
import pytest

from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_p3_speed_policy import stock
from hfmodem.tests.shrike.test_tx_gear_hold import control

PROPOSAL = (b': W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 09088167\r; VE3KPG DE W9SSJ\r'
            b'FC EM ZXK0M0UFE0PD 674 500 0\rF> 84\r')
SID = b': W9SSJ\r[Pat-1.0.0-B2FHM$]\r'


class Peer:
    """An IRS that keeps what it acknowledges once per counter, as VE3KPG does."""

    def __init__(self, seam):
        self.seam, self.got, self.last = seam, b'', None

    def answer(self, host, cs, *, takes=True):
        sl, status, payload = self.seam.frames[-1]
        if takes and status & 3 != self.last:
            self.got += payload
            self.last = status & 3
        control(host, cs)
        host.tick()


def ack(host):
    """The codeword that acknowledges the counter in flight."""
    control(host, 1 if host.arq.tx_seq % 2 == 0 else 2)


def proposal_leg():
    """TX34's changeover in flight, the proposal queued behind it."""
    host, seam = stock(PROPOSAL, inflight=arq._Packet(0, b';FW', 1, breakin=True))
    host.p3_mail_active = True
    seam.frames.append((1, 0, b';FW'))
    return host, seam, Peer(seam)


@pytest.mark.xfail(strict=True, reason='Operator decision 2026-10-03 never lose bytes: CS4 alone no longer credits a field')
def test_cs4_on_c1_and_c2_acknowledges_and_nothing_is_requeued():
    host, seam, peer = proposal_leg()
    peer.answer(host, 4, takes=False)          # the changeover's CS4
    for _ in range(2):                          # TX35 and TX37's CS4
        peer.answer(host, 4)
    counters = [status & 3 for _, status, _ in seam.frames[1:]]
    assert counters == [1, 2, 3]
    assert not any('GEAR HOLD' in line or 'transmit seam rendered' in line
                   for line in host.log_lines)
    assert peer.got == PROPOSAL[:len(peer.got)]
    assert peer.got.startswith(SID)


@pytest.mark.xfail(strict=True, reason='Operator decision 2026-10-03 never lose bytes: CS4 alone no longer credits a field')
def test_the_arm_answers_deliver_the_proposal_once_with_a_whole_sid():
    host, seam, peer = proposal_leg()
    peer.answer(host, 4, takes=False)
    peer.answer(host, 4)                        # c1
    peer.answer(host, 4)                        # c2
    peer.answer(host, 1, takes=False)           # CS1 on c3: it missed it
    c3 = seam.frames[-1]
    assert c3[1] & 3 == 3 and c3[2] == seam.frames[-2][2]
    peer.answer(host, 6)                        # the repeat read; CS6 acks and toggles
    for _ in range(12):
        if not host.arq._outbuf and not seam.frames[-1][2]:
            break
        seq = seam.frames[-1][1] & 3
        peer.answer(host, 1 if seq % 2 == 0 else 2)
    assert peer.got == PROPOSAL
    assert peer.got.count(b'[Pat-') == 1


def test_a_trial_falls_back_re_cut_under_its_counter():
    host, seam = stock(PROPOSAL, inflight=arq._Packet(0, b';FW', 1, breakin=True))
    a = host.arq
    control(host, 4)                            # climb: SL2 trial on c1
    trial = seam.frames[-1]
    assert (trial[0], trial[1] & 3, a._inflight.trial_from) == (2, 1, 1)
    control(host, 1)                            # CS1 on the odd counter: a request
    again = seam.frames[-1]
    assert a.speed_level == 1
    assert again == (1, trial[1], PROPOSAL[:5])
    assert a._inflight.payload + bytes(a._outbuf) == PROPOSAL
    ack(host)                                   # c1 acknowledged
    nxt = seam.frames[-1]
    assert (nxt[0], nxt[1] & 3, nxt[2]) == (1, 2, PROPOSAL[5:10])


def test_a_fitting_payload_returns_at_the_lower_level_with_the_same_bytes():
    host, seam = stock(PROPOSAL, inflight=arq._Packet(0, b';FW', 1, breakin=True))
    a = host.arq
    control(host, 4)
    a._inflight.payload, a._outbuf = PROPOSAL[:4], bytearray(PROPOSAL[4:])
    control(host, 1)
    assert seam.frames[-1][0] == 1 and seam.frames[-1][2] == PROPOSAL[:4]
    assert a._inflight.seq == 1


def test_a_cs5_still_re_cuts_because_the_peer_holds_nothing_under_the_counter():
    host, seam = stock(PROPOSAL, inflight=arq._Packet(0, b';FW', 1, breakin=True))
    a = host.arq
    control(host, 4)
    control(host, 5)
    assert a.speed_level == 1 and seam.frames[-1][0] == 1
    assert seam.frames[-1][2] == PROPOSAL[:5] and a._inflight.seq == 1


# -- idle behaviour ---------------------------------------------------------

def test_our_over_raises_bit6_on_the_packet_that_drains_it_and_not_after():
    host, seam = stock(b'84\r', inflight=arq._Packet(3, b'x', 1))
    a = host.arq
    a.on_host_over()
    statuses = []
    for _ in range(4):
        ack(host)
        statuses.append(seam.frames[-1][1])
    assert [s & spec.STATUS_CHANGEOVER for s in statuses] == [spec.STATUS_CHANGEOVER, 0, 0, 0]
    assert statuses == [0x41, 0x1a, 0x1b, 0x18]
    a.on_host_data(b'more')
    ack(host)
    assert seam.frames[-1][2] == b'more' and seam.frames[-1][1] & spec.STATUS_CHANGEOVER
    ack(host)
    assert not seam.frames[-1][1] & spec.STATUS_CHANGEOVER


# -- the turn after the long cycle -------------------------------------------

def test_the_peers_cs3_turns_us_irs_on_the_short_cycle():
    host, seam = stock(b'', inflight=arq._Packet(0x58, b'', 1))
    a = host.arq
    a.cycle_long = True
    control(host, 3)
    assert a.role == arq.IRS and not a.cycle_long
