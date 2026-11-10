# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""MYLevel-2-style policy, including grants and peer-initiated transitions.

These tests establish policy/ARQ wiring, not SCS handshake interoperability.
"""
import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p3_offer import Keyed, cs_event


class P2Keyed(Keyed):
    def send_p2_packet(self, sl, payload, status):
        self._packet(payload)
        return arq.P2_LADDER.payload(sl, False)

    def send_p2_cs(self, index):
        self.cs.append((Protocol.PACTOR2, index))


def linked():
    seam = P2Keyed()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    host.pactor2_only = True
    host.arq.on_host_connect(host.mycall, 'KB5LZK')
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    return host, seam


def grant(word):
    return rxfront.Event(0.2, 'unassigned', hex(word),
                         protocol=Protocol.PACTOR1, spare=word)


def test_only_6a9_can_enter_p2_and_it_keys_sl1():
    host, seam = linked()
    host.on_rx_event(grant(pactor1.CS_59A))
    assert host.protocol == Protocol.PACTOR1
    assert not host._grant_taken
    host.on_rx_event(grant(pactor1.CS_6A9))
    assert host.protocol == Protocol.PACTOR2
    assert host.arq.speed_level == 1
    assert host.arq.ladder == arq.P2_LADDER
    assert seam.packets[-1][0] == Protocol.PACTOR2
    assert not seam.p3
    assert host._grant_taken


def test_ordinary_p1_ack_keeps_p1_even_with_mail_queued():
    host, seam = linked()
    assert host.asking_for_a_grant and host.p2_announcement_pending
    host.arq.on_host_data(b'mail payload' * 8)
    host.on_rx_event(cs_event(pactor1.CS_ACK_A))
    assert host.protocol == Protocol.PACTOR1
    assert not host.asking_for_a_grant and not host.p2_announcement_pending
    assert not host.upgrade(payload_waiting=True)
    assert not seam.p3


@pytest.mark.parametrize('kind', ['cs', 'packet'])
@pytest.mark.parametrize('role', [arq.ISS, arq.IRS])
def test_p3_events_cannot_change_protocol_role_or_ack_bytes(kind, role):
    host, seam = linked()
    host.arq.role = role
    before = (host.arq.role, host.arq.speed_level, list(seam.packets), list(seam.cs))
    event = (cs_event(pactor1.CS_CHANGEOVER, Protocol.PACTOR3) if kind == 'cs'
             else rxfront.Event(0.2, 'packet', 'P3', protocol=Protocol.PACTOR3,
                                packet=(3, 0, b'not ours', True), breakin=True))
    host.on_rx_event(event)
    assert host.protocol == Protocol.PACTOR1
    assert (host.arq.role, host.arq.speed_level, seam.packets, seam.cs) == before


def test_follow_p2_fallback_and_reconnect_preserve_ceiling():
    host, seam = linked()
    host.arq.role = arq.IRS
    host._follow_peer(Protocol.PACTOR2)
    assert host.protocol == Protocol.PACTOR2
    host._follow_peer(Protocol.PACTOR3)
    assert host.protocol == Protocol.PACTOR2
    host.fall_back('test timeout')
    assert host.protocol == Protocol.PACTOR1
    host.arq.on_host_abort()
    host.arq.on_host_connect(host.mycall, 'KB5LZK')
    host.on_rx_event(cs_event(pactor1.CS_SPEED))
    host.tick()
    host.on_rx_event(grant(pactor1.CS_6A9))
    assert host.protocol == Protocol.PACTOR2
    assert not seam.p3


def test_qrt_cannot_be_reopened_by_6a9():
    host, seam = linked()
    host.arq.on_host_disconnect()
    before = list(seam.packets)
    host.on_rx_event(grant(pactor1.CS_6A9))
    assert host.protocol == Protocol.PACTOR1
    assert not host._grant_taken
    assert seam.packets == before


def test_p1_fallback_clears_announcement_on_the_actual_renderer(monkeypatch, tmp_path):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    host, _ = linked()
    tx.attach(host)
    tx.p1_status_bits45 = 1
    monkeypatch.setattr(tx, '_tx', lambda *a, **kw: None)
    seen = []
    render = pactor1.packet_signal

    def record(*args, **kwargs):
        seen.append(kwargs['bits45'])
        return render(*args, **kwargs)

    monkeypatch.setattr(pactor1, 'packet_signal', record)
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    host.on_rx_event(cs_event(pactor1.CS_ACK_A))
    tx.send_p1_packet(b'mail', 100, 2)
    assert seen == [1, 0]


def test_p2_uses_the_psk_power_phase_before_entry():
    from hfmodem.tests.shrike.test_phase_power_cat import _FakeCat

    host, _ = linked()
    cat = _FakeCat()
    power = onair._PhasePower(cat, p1=15, p3=60)
    host.phase_power = power
    power.select(Protocol.PACTOR1, 'setup')
    cat.note = lambda: host.protocol
    host.on_rx_event(grant(pactor1.CS_6A9))
    assert cat.writes[-1] == (0.60, Protocol.PACTOR1)
    assert host.protocol == Protocol.PACTOR2
    host.fall_back('test timeout')
    assert cat.writes[-1][0] == 0.15


def test_granted_p2_carries_payload_through_the_audio_path():
    from hfmodem.tests.shrike.test_p2qso import _linked, _drain, rx_of

    link = _linked()
    link.a.pactor2_only = link.b.pactor2_only = True
    link.a.on_rx_event(grant(pactor1.CS_6A9))
    payload = b'PACTOR TWO MAIL PATH\r' * 3
    link.a.arq.on_host_data(payload)
    _drain(link, 40)
    assert link.a.protocol == link.b.protocol == Protocol.PACTOR2
    assert rx_of(link.b).endswith(payload)


def test_summary_names_p2_controls(capsys):
    onair._summary([cs_event(0, Protocol.PACTOR2)], [], 1, [], 1.25,
                   [1, 2], onair._ConnectEvidence(), ended='test')
    text = capsys.readouterr().out
    assert '1 PACTOR-2' in text
    assert 'verdict: the link ran in PACTOR-2 -- 1 control signals' in text
