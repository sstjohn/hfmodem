# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A CS6 in the acknowledgement slot leaves the CS1/CS2 alternation on the
counter's parity, and a skipped counter is never delivered.

WS8EOC-80 0924-0249 (`working/pactor-level-air-0916/arms/
pactor-current-ws8eoc-80-pounce-20260924T024946Z`): our CS6 on counter 2 was not
taken, the peer repeated 2, and a phase moved by that CS6 answered it CS2
thirty-six times until the gateway hung up. Counter parity is every one of the
corpus's 488 advances: VE3KPG-80 0924-0010 and VE1YZ-40 0921-0539 took CS2 for 3
behind our CS6 on 2, and WS8EOC-40 0920-1537 and KB5LZK-80 0921-0348 took CS1
for a 2 or 0 they repeated behind it.
"""
from hfmodem.shrike import arq, spec
from hfmodem.tests.shrike.test_p3_speed_policy import stock
from hfmodem.tests.shrike.test_tx_gear_hold import control

P3 = spec.Protocol.PACTOR3


def irs():
    host, seam = stock()
    a = host.arq
    a._inflight = None
    a._give_link()
    sent = []
    host.peer.send_cs = lambda cs, **kw: sent.append(cs)
    return host, a, sent


def feed(a, seq, payload=b'abc'):
    a.on_rx_packet(1, payload, spec.status_byte(seq), True, protocol=P3)


def word(host):
    """The physical codeword the IRS keys for an ACK now: 1 or 2."""
    return host._counter_cs_for(arq.CS_ACK) + 1


def run(host, a, answers):
    """Feed counters and record our word for each; 6 marks a CS6 answer."""
    words = []
    for seq, ours in answers:
        feed(a, seq)
        if ours == 6:
            a.on_cs_emitted(arq.CS_CYCLE_TOG)
            words.append(6)
        else:
            words.append(word(host))
            a.on_cs_emitted(arq.CS_ACK)
    return words


def test_a_taken_cs6_on_an_even_counter_leaves_parity_for_the_next():
    host, a, _ = irs()
    words = run(host, a, [(0, 1), (1, 2), (2, 6), (3, 2), (0, 1), (1, 2)])
    assert words == [1, 2, 6, 2, 1, 2]      # VE3KPG-80 0924-0010, VE1YZ-40 0921-0539


def test_an_unread_slot_after_our_cs6_holds_on_the_counter_we_took():
    host, a, _ = irs()
    run(host, a, [(0, 1), (1, 2), (2, 6)])
    assert word(host) == 1


def test_ve3kpg_0921_the_cs6_on_an_odd_counter_leaves_parity_alone():
    host, a, _ = irs()
    words = run(host, a, [(0, 1), (1, 6), (2, 1), (3, 2), (0, 1), (1, 2)])
    assert words == [1, 6, 1, 2, 1, 2]


def test_the_iss_reads_the_peers_acknowledgement_on_parity_after_its_cs6():
    host, seam = stock(bytes(range(65, 91)) * 4, inflight=arq._Packet(2, b'ab', 1))
    a = host.arq
    a._next_seq = 3
    control(host, 6)                # the CS6 acknowledges counter 2
    assert a.tx_seq == 3
    control(host, 1)                # CS1 on 3: a repeat request
    assert a.tx_seq == 3
    control(host, 2)
    assert a.tx_seq == 0
    control(host, 1)
    assert a.tx_seq == 1


def test_a_skipped_counter_is_not_delivered():
    host, a, sent = irs()
    feed(a, 1, b'.4.3')
    feed(a, 2, b'ode 1')
    before = bytes(host.channel(host.ptchn).rx)
    feed(a, 0, b'remaining')        # counter 3 was never read
    assert bytes(host.channel(host.ptchn).rx) == before
    assert a.rx_seq == 2 and sent[-1] == arq.CS_ACK   # CS1: still on counter 2
    assert any('3 was never read' in line for line in host.log_lines)


def test_a_renumbered_train_after_the_entry_packet_is_still_delivered():
    host, a, _ = irs()
    feed(a, 2, b'entry')            # DL6MAA: entry at 2, the train opens at 1
    feed(a, 1, b'Maildrop')
    assert b'Maildrop' in bytes(host.channel(host.ptchn).rx)


def test_ws8eoc_0249_a_repeat_behind_our_cs6_draws_cs1():
    """The arm's own answers: CS1, CS6 and CS2 on 1, CS6 on 2, then 2 again."""
    host, a, sent = irs()
    a.cfg.long_cycle = True

    def answer(seq, payload, long_asked=0x20):
        n = len(sent)
        a.on_rx_packet(1, payload, spec.status_byte(seq) | long_asked, True, protocol=P3)
        for cs in sent[n:]:
            a.on_cs_emitted(arq.CS_ACK if cs in (arq.CS_ACK, arq.CS_REQUEST) else cs)
        return [cs + 1 for cs in sent[n:]]

    assert answer(0, b'RMS', 0) == [1]
    assert [answer(1, b' Trim') for _ in range(3)] == [[6], [2], [2]]
    assert [answer(2, b'ode 1') for _ in range(3)] == [[6], [1], [1]]
    assert answer(3, b'next', 0) == [2]
