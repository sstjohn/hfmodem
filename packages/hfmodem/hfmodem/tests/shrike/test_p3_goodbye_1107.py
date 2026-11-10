# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The QRT we still owe goes out while the peer that broke in on it is sending.

WS8EOC 2026-11-07 17:44Z answered our QRT with its CS3 and an empty changeover
0x40. We ACKed that. The break-in carrying the QRT was not placed in the next
hold, and the peer's empty 0x01 and its 0x42 invitation each drew another ACK.
The goodbye clock then ran out under the peer's CRC-valid packets, so the link
closed with the QRT unconfirmed and slot 109 keyed nothing. A stock IRS with
something to send ACKs a turn once or twice and then takes the link, and it
answers a bit-6 packet with CS3 on the next slot.
"""
from hfmodem.shrike import arq, rxfront, spec
from hfmodem.tests.shrike.test_p3_clean_shutdown_1002 import keyed, link


def heard(host, status, *, breakin=False, payload=b''):
    host.on_rx_event(rxfront.Event(0, 'packet', 'WS8EOC 17:44Z',
                                   protocol=spec.Protocol.PACTOR3,
                                   packet=(1, status, payload, True), breakin=breakin))


def words(peer):
    return len(peer.controls) + len(peer.keyed_bursts)


def turned(host, peer):
    """Our QRT keyed, the peer's CS3 changeover read and ACKed in its hold."""
    first = keyed(host)
    heard(host, 0x40, breakin=True)
    assert peer.controls == [0]
    host.tick()
    assert host.arq.role == arq.IRS and host.arq._qrt_keyed is first
    return first


def test_the_owed_qrt_goes_out_on_the_first_placeable_cycle():
    host, peer = link()
    turned(host, peer)
    keyed_before = words(peer)
    peer.refuse = True                      # hold 43: CHANGEOVER NOT PLACED
    host.tick()
    assert host.arq.role == arq.IRS and words(peer) == keyed_before + 1
    peer.refuse = False
    heard(host, 0x40, breakin=True)         # its changeover again
    heard(host, 0x01)                       # and its empty 0x01
    host.tick()                             # hold 44
    assert host.arq.role == arq.ISS
    assert peer.last == (1, b'', 0x98, True) and words(peer) == keyed_before + 2


def test_the_peers_invitation_is_answered_with_the_qrt():
    host, peer = link()
    turned(host, peer)
    heard(host, 0x41)                       # counter 1, bit 6 standing
    host.tick()
    assert host.arq.role == arq.ISS and peer.last == (1, b'', 0x98, True)


def test_the_goodbye_clock_waits_while_the_peer_sends():
    """No placement for eight holds: the peer repeats each packet our repeat
    request asks for, the link stays up and every hold keys its answer."""
    host, peer = link()
    turned(host, peer)
    peer.refuse = True
    for status in (0x01,) * 4 + (0x42,) * 4:
        keyed_before = words(peer)
        heard(host, status)
        host.tick()
        assert host.arq.state == arq.State.DISCONNECTING
        assert words(peer) == keyed_before + 1, 'every hold keys its answer'
    # Silence still ends it.
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    assert host.arq.state == arq.State.DISCONNECTED and not host.arq.goodbye_acked


def test_the_qrt_changeover_still_waits_for_the_turns_ack():
    host, peer = link()
    keyed(host)
    peer.defer_p3_cs = True
    heard(host, 0x40, breakin=True)
    host.tick()
    assert host.arq.role == arq.IRS and host.arq._turn_ack_owed
    heard(host, 0x01)
    host.tick()
    assert host.arq.role == arq.IRS, 'no CS3 before the turn has had its ACK'


def test_the_qrt_never_acknowledges_a_packet_it_did_not_deliver():
    """A review of 88ee3e1ad: counter 2 behind a missed 1 is held, not
    delivered, and the QRT's CS3 in that hold would settle it at the peer."""
    host, peer = link()
    keyed(host)
    heard(host, 0x00, breakin=True, payload=b'A')
    host.tick()
    bursts = len(peer.keyed_bursts)
    heard(host, 0x02, payload=b'C')
    host.tick()
    assert len(peer.keyed_bursts) == bursts, 'no CS3 behind a packet held back'
    assert bytes(host.channel(host.ptchn).rx) == b'A'
    heard(host, 0x01, payload=b'B')         # the repeat it asked for
    host.tick()
    assert host.arq.role == arq.ISS and peer.last == (1, b'', 0x98, True)
    assert bytes(host.channel(host.ptchn).rx) == b'AB'


def test_a_peer_inviting_us_while_our_qrt_goes_unread_keeps_the_link_and_the_key():
    """A review of 88ee3e1ad: the peer missed our QRT's changeover and kept
    inviting us with CRC-valid 0x41; the retry budget ended the link under them
    with the closing slot unkeyed."""
    host, peer = link()
    turned(host, peer)
    heard(host, 0x41)
    host.tick()
    assert host.arq.role == arq.ISS and peer.last == (1, b'', 0x98, True)
    for _ in range(2 * host.arq.cfg.max_retries):
        keyed_before = words(peer)
        heard(host, 0x41)
        host.tick()
        assert host.arq.state == arq.State.DISCONNECTING
        assert words(peer) == keyed_before + 1 and peer.last[2] == 0x98
    host.tick(elapsed_ticks=arq.GOODBYE_CYCLES + 1)
    assert host.arq.state == arq.State.DISCONNECTED
