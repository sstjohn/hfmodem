# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Our caller against a callee whose ceiling is PACTOR-2.

The 2x2 crossing of our announcement's status bits 4-5 at two Winlink gateways
(docs/protocols/pactor/pactor-capability.md section 5) is the evidence: `0x01`
is acknowledged, `0x11` draws `0x6A9` every cycle with the counter frozen, and
`0x31` draws `0x59A`. The two-bit-ceiling reading predicts that a PACTOR-2-only
callee answers our `0x31` with `0x6A9`. No stock PACTOR-1-to-PACTOR-2 entry is
on tape, so taking `0x6A9` as a grant stays behind `PtcHost.p2_grant_6a9`; not
losing the link to it is the default.
"""
import pytest

from hfmodem.shrike import arq, onair, pactor1, ptc, rxfront
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p2_policy import P2Keyed
from hfmodem.tests.shrike.test_p3_offer import cs_event

MAIL = b'mail waiting ' * 4


class Announcing(P2Keyed):
    """Records the status bits 4-5 each PACTOR-1 packet goes out with, as
    `onair.RadioTx.send_p1_packet` computes them for a default arm."""

    def __init__(self):
        super().__init__()
        self.bits: list[int] = []

    def send_p1_packet(self, payload, baud, seq, **kw):
        self.bits.append(0 if kw.get('qrt') else self.host.p1_announcement_bits(3))
        super().send_p1_packet(payload, baud, seq, **kw)


def word(spare):
    return rxfront.Event(0.2, 'unassigned', hex(spare), protocol=Protocol.PACTOR1, spare=spare)


def caller(**flags):
    seam = Announcing()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    for name, value in flags.items():
        setattr(host, name, value)
    host.arq.on_host_connect('W9SSJ', 'N0P2')
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.arq.on_host_data(MAIL)
    return host, seam


def p2_only_callee(host, seam, cycles=40, *, silent=False):
    """Answer an announcing packet with 0x6A9 (or nothing), acknowledge a plain one."""
    ack = pactor1.CS_ACK_A
    for _ in range(cycles):
        if host.arq.state is not State.CONNECTED or host.protocol is not Protocol.PACTOR1:
            return
        if seam.bits[-1]:
            if not silent:
                host.on_rx_event(word(pactor1.CS_6A9))
        else:
            host.on_rx_event(cs_event(ack))
            ack ^= 1
        host.tick()


def test_a_p2_callee_answering_6a9_keeps_the_link_in_p1():
    host, seam = caller()
    p2_only_callee(host, seam)
    assert host.arq.state is State.CONNECTED, host.log_lines[-4:]
    assert host.protocol is Protocol.PACTOR1
    assert seam.bits[0] == 3 and seam.bits[-1] == 0
    assert not host.arq._outbuf
    assert MAIL in b''.join(dict.fromkeys(p for _, p in seam.packets))
    assert any('0x01' in line for line in host.log_lines)


def test_silence_keeps_the_announcement_to_the_retry_limit():
    # Silence is the 0x21 cell, which we never send, and a late 0x59A can still
    # arrive in those cycles (WS8EOC 2026-09-11): the 0x31 stands until QRT.
    host, seam = caller()
    p2_only_callee(host, seam, silent=True)
    assert host.arq.state is not State.CONNECTED
    sent = [b for b in seam.bits if b]
    assert len(sent) > 5 and set(sent) == {3}
    assert not any('withdrawn' in line for line in host.log_lines)


def test_a_fade_after_a_p3_fallback_does_not_withdraw():
    host, seam = caller()
    host.on_rx_event(word(pactor1.CS_59A))
    host.tick()
    assert host.protocol is Protocol.PACTOR3
    host.fall_back('test fade', rule_out=False)
    assert host.protocol is Protocol.PACTOR1
    for _ in range(ptc.ANNOUNCE_6A9_CYCLES + 1):
        host.on_rx_event(word(pactor1.CS_6A9))
        host.tick()
    assert host.p1_announcement_bits(3) == 3


def test_only_consecutive_6a9_answers_withdraw():
    host, seam = caller()
    for spare in (pactor1.CS_6A9, pactor1.CS_6A9, None, pactor1.CS_6A9, pactor1.CS_6A9):
        host.on_rx_event(cs_event(pactor1.CS_SPEED) if spare is None else word(spare))
        host.tick()
    assert host.p1_announcement_bits(3) == 3
    host.on_rx_event(word(pactor1.CS_6A9))
    assert host.p1_announcement_bits(3) == 0


def test_a_granting_callee_is_unaffected():
    host, seam = caller()
    host.on_rx_event(word(pactor1.CS_59A))
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.state is State.CONNECTED


def test_the_withdrawal_ends_with_the_contact():
    host, seam = caller()
    p2_only_callee(host, seam, cycles=8)
    assert host.p1_announcement_bits(3) == 0
    host.arq.on_host_abort()
    host.arq.on_host_connect('W9SSJ', 'N0P2')
    assert host.p1_announcement_bits(3) == 3


def test_6a9_grant_option_enters_p2_on_the_announcement():
    host, seam = caller(p2_grant_6a9=True)
    host.on_rx_event(word(pactor1.CS_6A9))
    assert host.protocol is Protocol.PACTOR2
    assert host.arq.speed_level == 1 and host.arq.ladder is arq.P2_LADDER
    assert seam.packets[-1][0] is Protocol.PACTOR2
    assert not seam.p3
    # The peer has named its ceiling: nothing above PACTOR-2 for this contact.
    host.on_rx_event(word(pactor1.CS_59A))
    assert host.protocol is Protocol.PACTOR2


def test_6a9_grant_option_still_takes_a_p3_grant():
    host, seam = caller(p2_grant_6a9=True)
    host.on_rx_event(word(pactor1.CS_59A))
    assert host.protocol is Protocol.PACTOR3


def test_6a9_grant_option_respects_mylevel_1():
    host, seam = caller(p2_grant_6a9=True)
    host.settings['MYLEVEL'] = '1'
    host.on_rx_event(word(pactor1.CS_6A9))
    assert host.protocol is Protocol.PACTOR1


def test_6a9_grant_ceiling_ends_with_the_contact():
    host, seam = caller(p2_grant_6a9=True)
    host.on_rx_event(word(pactor1.CS_6A9))
    assert host.pactor2_only
    host.arq.on_host_abort()
    assert not host.pactor2_only


def test_6a9_grant_then_follows_into_p2_over_audio():
    from hfmodem.tests.shrike.test_p2qso import _linked, _drain, rx_of

    link = _linked()
    link.a.p2_grant_6a9 = True
    link.b.pactor2_only = True
    link.a.on_rx_event(word(pactor1.CS_6A9))
    payload = b'PACTOR TWO ON A 0x6A9 GRANT\r' * 3
    link.a.arq.on_host_data(payload)
    _drain(link, 40)
    assert link.a.protocol == link.b.protocol == Protocol.PACTOR2
    assert rx_of(link.b).endswith(payload)


def test_withdrawn_announcement_renders_clear_bits(monkeypatch, tmp_path):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    host, _ = caller()
    tx.attach(host)
    tx.p1_status_bits45 = 3
    monkeypatch.setattr(tx, '_tx', lambda *a, **kw: None)
    seen = []
    render = pactor1.packet_signal

    def record(*args, **kwargs):
        seen.append(kwargs['bits45'])
        return render(*args, **kwargs)

    monkeypatch.setattr(pactor1, 'packet_signal', record)
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    for _ in range(ptc.ANNOUNCE_6A9_CYCLES):
        host.on_rx_event(word(pactor1.CS_6A9))
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    assert seen == [3, 0]


def test_the_option_is_on_the_command_line():
    assert onair.build_parser().parse_args(['--mycall', 'W9SSJ', '--p2-6a9-grant']).p2_grant_6a9
    assert not onair.build_parser().parse_args(['--mycall', 'W9SSJ']).p2_grant_6a9
