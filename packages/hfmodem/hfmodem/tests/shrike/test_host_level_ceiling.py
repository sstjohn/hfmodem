# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A host level ceiling applies before RF events earn any ARQ side effect."""
from types import SimpleNamespace

import pytest
from hfhost import ptc

from hfmodem.shrike import arq, onair, p1rx, pactor1, rxfront
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_ptc_compatibility import Client


@pytest.mark.parametrize('maximum,heard', [(1, Protocol.PACTOR2),
                                         (1, Protocol.PACTOR3),
                                         (2, Protocol.PACTOR3)])
@pytest.mark.parametrize('role', [arq.ISS, arq.IRS])
@pytest.mark.parametrize('kind', ['ack', 'changeover', 'packet', 'breakin'])
def test_over_ceiling_events_cannot_settle_deliver_or_change_roles(monkeypatch, maximum, heard, role, kind):
    host = PtcHost()
    client = Client(host)
    assert client.cmd(f'#MYL {maximum}').code == ptc.OK
    assert client.cmd('%M1').code == ptc.OK
    host.arq.state, host.arq.role = arq.State.CONNECTED, arq.ISS
    host.arq.dxcall = 'N0DX'
    assert client.exchange(bytes(range(256)), False).code == ptc.OK
    keyed = []
    monkeypatch.setattr(host, 'send_packet', lambda sl, payload, status, **kw:
                        keyed.append(bytes(payload)) or len(payload))
    host.arq._start_next_packet()
    host.arq.role = role

    def snapshot():
        packet = host.arq._inflight
        return (host.protocol, host.arq.state, host.arq.role, host.arq.speed_level,
                host.arq.tx_seq, host.sent_total, host.arq.payload_ledger.pending,
                host.arq.payload_ledger.unconfirmed, bytes(host.arq._outbuf),
                None if packet is None else (packet.seq, packet.payload, packet.retries),
                bytes(host.channel(31).rx), bytes(host.channel(31).echo), tuple(keyed))

    before = snapshot()
    event = (rxfront.Event(.2, 'cs', kind, protocol=heard,
                          cs=pactor1.CS_ACK_A if kind == 'ack' else pactor1.CS_CHANGEOVER)
             if kind in ('ack', 'changeover') else
             rxfront.Event(.2, 'packet', kind, protocol=heard,
                           packet=(3, 0, b'not our level', True), breakin=kind == 'breakin'))
    host.on_rx_event(event)
    assert snapshot() == before


@pytest.mark.parametrize('bypass', ['mail_wait', 'p4'])
def test_ceiling_precedes_special_receive_paths(monkeypatch, bypass):
    host = PtcHost()
    host._store('MYLEVEL', '1')
    seen = []
    if bypass == 'mail_wait':
        host.arq._mail_cs4_wait = object()
        monkeypatch.setattr(host.arq, 'on_rx_cs', seen.append)
        event = rxfront.Event(.2, 'cs', 'higher-level answer', protocol=Protocol.PACTOR3, cs=0)
    else:
        monkeypatch.setattr(host.arq, 'on_rx_p4', seen.append)
        event = rxfront.Event(.2, 'p4', 'higher-level observation', p4=SimpleNamespace())
    host.on_rx_event(event)
    assert not seen


@pytest.mark.parametrize('bits', [1, 2, 3])
@pytest.mark.parametrize('cached', [False, True])
def test_level_one_announcement_is_clear_in_emitted_audio(tmp_path, monkeypatch, bits, cached):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    host = PtcHost(tx, mycall='N0CALL')
    tx.p1_status_bits45 = bits
    if cached:
        # A host may lower its idle ceiling after setup waveforms were prepared.
        host._store('MYLEVEL', '3')
        tx.prepare_p1_setup()
    host._store('MYLEVEL', '1')
    keyed = []
    monkeypatch.setattr(tx, '_tx', lambda audio, *a, **kw: keyed.append(audio))
    tx.send_p1_packet(host.p1_setup_payload()[:8], 100, 1)
    packets = p1rx.decode_p1_packets(keyed[0])
    assert len(packets) == 1
    assert packets[0].status & 0x30 == 0


def test_level_one_precomputed_setup_has_no_higher_level_announcement(tmp_path):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    host = PtcHost(tx, mycall='N0CALL')
    host._store('MYLEVEL', '1')
    tx.p1_status_bits45 = 3
    tx.prepare_p1_setup()
    assert tx._p1_setup_waveforms
    assert all(key[4] == 0 for key in tx._p1_setup_waveforms)


@pytest.mark.parametrize('maximum,p2,expected', [(2, False, 0), (2, True, 1),
                                               (3, False, 3), (3, True, 1)])
def test_announcement_intersects_the_selected_ceiling_with_armed_entry(maximum, p2, expected):
    host = PtcHost()
    host._store('MYLEVEL', str(maximum))
    host.pactor2_only = p2
    assert host.p1_announcement_bits(3) == expected


def test_allowed_p1_ack_still_confirms_originals(monkeypatch):
    host = PtcHost()
    client = Client(host)
    assert client.cmd('#MYL 1').code == ptc.OK
    host.arq.state, host.arq.role = arq.State.CONNECTED, arq.ISS
    assert client.exchange(b'original', False).code == ptc.OK
    monkeypatch.setattr(host, 'send_packet', lambda sl, payload, status, **kw: len(payload))
    host.arq._start_next_packet()
    host.on_rx_event(rxfront.Event(.2, 'cs', 'P1 ACK', protocol=Protocol.PACTOR1,
                                  cs=pactor1.CS_ACK_A))
    assert host.sent_total == len(b'original')
