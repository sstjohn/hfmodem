# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A station run as a PACTOR-2-capped modem for trials against stock peers.

`onair --pactor2-only` and the station's `MYLevel 2` are one policy: the setup
packet announces bits45=1 (never 0x31), PACTOR-2 is entered only on 0x6A9 or a
peer leading into it, and nothing above PACTOR-2 is keyed, granted or followed.
"""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.shrike.station_runtime import StationPtcRuntime, _apply_level
from hfmodem.tests.shrike.test_p2_callee import Announcing, word
from hfmodem.tests.shrike.test_p3_offer import cs_event

MAIL = b'capped mail ' * 8


class CappedSeam(Announcing):
    """`RadioTx.send_p1_packet`'s pactor2_only rule: bits only while setup is owed."""

    def send_p1_packet(self, payload, baud, seq, **kw):
        super().send_p1_packet(payload, baud, seq, **kw)
        if not self.host.p2_announcement_pending:
            self.bits[-1] = 0


def capped(*, answering=False):
    seam = CappedSeam()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    host._store('MYLEVEL', '2')
    _apply_level(host, SimpleNamespace(p1_status_bits45=None))
    if answering:
        host.arq.on_host_listen(True)
        host.arq.role, host.arq.dxcall = arq.IRS, 'N0P3'
        host.arq._enter_connected()
    else:
        host.arq.on_host_connect('W9SSJ', 'N0P3')
        host.on_rx_event(cs_event(pactor1.CS_SPEED))
        host.tick()
    host.arq.on_host_data(MAIL)
    return host, seam


def assert_never_above_p2(host, seam):
    assert host.protocol in (Protocol.PACTOR1, Protocol.PACTOR2)
    assert not seam.p3
    assert all(protocol in (Protocol.PACTOR1, Protocol.PACTOR2) for protocol, _ in seam.cs)
    assert all(bits in (0, 1) for bits in seam.bits)


def test_mylevel_2_is_accepted_and_is_the_pactor2_only_policy():
    host = PtcHost()
    StationPtcRuntime._validate_setting(SimpleNamespace(host=host), 'MYLEVEL', '2')
    tx = SimpleNamespace(p1_status_bits45=None)
    host._store('MYLEVEL', '2')
    _apply_level(host, tx)
    assert host.pactor2_only and tx.p1_status_bits45 == 1
    host._store('MYLEVEL', '3')
    _apply_level(host, tx)
    assert not host.pactor2_only and tx.p1_status_bits45 == 3


def test_the_announcement_is_never_0x31_on_the_renderer(monkeypatch, tmp_path):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    host = PtcHost(tx, mycall='W9SSJ')
    host._store('MYLEVEL', '2')
    _apply_level(host, tx)
    host.arq.on_host_connect('W9SSJ', 'N0P3')
    host.arq.state, host.arq.role = State.CONNECTED, arq.ISS
    monkeypatch.setattr(tx, '_tx', lambda *a, **kw: None)
    seen = []
    render = pactor1.packet_signal
    monkeypatch.setattr(pactor1, 'packet_signal',
                        lambda *a, **kw: seen.append(kw['bits45']) or render(*a, **kw))
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    tx.p1_status_bits45 = 3              # an arm asking for 0x31 is still capped
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    assert seen == [1, 1]


@pytest.mark.parametrize('answer', ['59a', 'ack', 'silence'])
def test_a_p3_gateway_cannot_draw_p3_from_the_capped_caller(answer):
    host, seam = capped()
    ack = pactor1.CS_ACK_A
    for _ in range(30):
        if host.arq.state is not State.CONNECTED:
            break
        if answer == '59a':
            host.on_rx_event(word(pactor1.CS_59A))
        elif answer == 'ack':
            host.on_rx_event(cs_event(ack))
            ack ^= 1
        host.tick()
    assert_never_above_p2(host, seam)
    assert host.protocol is Protocol.PACTOR1


def test_0x6a9_enters_p2_and_host_mode_reports_level_2():
    host, seam = capped()
    host.on_rx_event(word(pactor1.CS_6A9))
    host.tick()
    assert host.protocol is Protocol.PACTOR2
    assert host._status_bytes()[1] == 2
    for _ in range(4):
        host.on_rx_event(word(pactor1.CS_59A))
        host.tick()
    assert_never_above_p2(host, seam)


P3_EVENTS = {
    'p3 cs': cs_event(pactor1.CS_SPEED, Protocol.PACTOR3),
    'p3 changeover': cs_event(pactor1.CS_CHANGEOVER, Protocol.PACTOR3),
    'p3 packet': rxfront.Event(0.2, 'packet', 'P3', protocol=Protocol.PACTOR3,
                               packet=(3, 0, b'upgrade', True)),
    'p3 breakin': rxfront.Event(0.2, 'packet', 'P3', protocol=Protocol.PACTOR3,
                                packet=(1, 0, b'upgrade', True), breakin=True),
}


# A plain P3 packet heard by our ISS is stale news uncapped too, so it is only
# a case for the callee; every case here moves an uncapped host off PACTOR-1.
CASES = [(name, answering) for name in P3_EVENTS for answering in (False, True)
         if (name, answering) != ('p3 packet', False)]


@pytest.mark.parametrize('name,answering', CASES)
def test_an_unasked_p3_upgrade_is_not_followed(name, answering):
    host, seam = capped(answering=answering)
    for _ in range(3):
        host.on_rx_event(P3_EVENTS[name])
        host.tick()
    assert_never_above_p2(host, seam)
    assert host.protocol is Protocol.PACTOR1
    assert host._status_bytes()[1] == 1
