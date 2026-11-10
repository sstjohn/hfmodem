# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC-80 0924-0344: the body's last field, answered CS4 at SL6.

`working/onair-0923-2244/pactor-current-ws8eoc-80-pounce-20260924T034419Z.log`,
TX[48]-[53]: the break-in carrying `\\x01 0 W`, then 23, 59, 122, 212 and 124
bytes at SL2 to SL6, every one answered CS4. The sixth CS4 answers a packet
already at the top level, where there is no next level to demand; the ISS took
it as an acknowledgement, keyed idle packets behind it, and the 124 bytes
carrying the EOT and its checksum never went out again. The gateway asked for
that counter (CS1 on the odd counter, then CS5 eight times), broke in, and the
CMS never received the message.

With --p3-cs4-parity the loss is caught where it began. `PIII_Complete_1`'s IRS
answers every even counter CS1 and puts CS4 only on odd ones. Read that way,
the CS4 on counter 2 at SL3 asked for that field again.
"""
from hfmodem.shrike import arq
from hfmodem.shrike.compress import transparent
from hfmodem.tests.shrike.test_p3_speed_policy import stock
from hfmodem.tests.shrike.test_p3_ve3kpg_proposal_0924 import Peer
from hfmodem.winlink import compose
from hfmodem.winlink.session import B2FSession, _Outbound

SUBJECT = "W9SSJ PACTOR test via WS8EOC 2026-09-24 0344z"


def wire() -> bytes:
    """A 477-byte block's worth of body, framed and escaped as the host path
    queues it."""
    msg = compose("W9SSJ", "SMTP:op@example.net", SUBJECT,
                  bytes(range(256)) * 2, mid="829XTENK3736")
    out = bytearray()
    B2FSession._send_body(B2FSession.__new__(B2FSession), _Outbound(msg), 0, out)
    return transparent(bytes(out))


def the_climb():
    body = wire()
    host, seam = stock(body[3:], inflight=arq._Packet(0, body[:3], 1, breakin=True),
                       policy="current")
    seam.frames.append((1, 0, body[:3]))
    peer = Peer(seam)
    peer.answer(host, 4)                        # the break-in's CS4
    for _ in range(4):                          # SL2 .. SL5, each CS4
        peer.answer(host, 4)
    return body, host, seam, peer


def test_the_arm_climbs_to_sl6_on_cs4():
    body, host, seam, peer = the_climb()
    assert [f[0] for f in seam.frames] == [1, 2, 3, 4, 5, 6]
    assert body.startswith(peer.got)


def test_a_cs4_at_the_top_repeats_the_packet_it_answers():
    body, host, seam, peer = the_climb()
    sl6 = seam.frames[-1]
    peer.answer(host, 4, takes=False)           # the SL6 field did not arrive
    again = seam.frames[-1]
    assert again[1] & 3 == sl6[1] & 3 and again[2] == sl6[2], seam.frames[-2:]
    assert any("is no acknowledgement" in line for line in host.log_lines)


def test_the_body_arrives_whole_once_the_tail_is_read():
    body, host, seam, peer = the_climb()
    peer.answer(host, 4, takes=False)
    for _ in range(12):
        if not host.arq._outbuf and not seam.frames[-1][2]:
            break
        seq = seam.frames[-1][1] & 3
        peer.answer(host, 1 if seq % 2 == 0 else 2)
    assert peer.got == body


# --p3-cs4-parity: the same climb, with a CS4 on an even counter held as a repeat.

def after_the_break_in(parity=True):
    body = wire()
    host, seam = stock(body[3:], inflight=arq._Packet(0, body[:3], 1, breakin=True),
                       policy="current")
    host.arq.cfg.p3_cs4_parity = parity
    seam.frames.append((1, 0, body[:3]))
    peer = Peer(seam)
    peer.answer(host, 4)                        # the break-in's CS4: exempt
    peer.answer(host, 4)                        # counter 1 at SL2: the odd slot
    return body, host, seam, peer


def test_parity_an_odd_counters_cs4_acknowledges_and_climbs():
    body, host, seam, peer = after_the_break_in()
    assert [(f[0], f[1] & 3) for f in seam.frames] == [(1, 0), (2, 1), (3, 2)]
    assert len(peer.got) == 3 + 23


def test_parity_an_even_counters_cs4_repeats_its_field():
    body, host, seam, peer = after_the_break_in()
    sl3 = seam.frames[-1]
    peer.answer(host, 4, takes=False)           # counter 2 at SL3, as flown
    assert seam.frames[-1][1] & 3 == 2 and seam.frames[-1][2] == sl3[2]


def test_without_parity_the_even_counters_cs4_still_acknowledges():
    body, host, seam, peer = after_the_break_in(parity=False)
    peer.answer(host, 4, takes=False)
    assert seam.frames[-1][1] & 3 == 3


def test_parity_the_body_arrives_whole_when_the_gateway_answers_as_the_reference():
    body, host, seam, peer = after_the_break_in()
    peer.answer(host, 4, takes=False)           # the SL3 trial missed, as flown
    for _ in range(40):
        if not host.arq._outbuf and not seam.frames[-1][2]:
            break
        seq = seam.frames[-1][1] & 3
        peer.answer(host, 1 if seq % 2 == 0 else 2)
    assert peer.got == body
