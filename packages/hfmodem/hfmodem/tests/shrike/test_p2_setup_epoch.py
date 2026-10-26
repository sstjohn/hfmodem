# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC 19:42: delayed CS1 at 33.61 s predates the first announcement."""
from dataclasses import replace
import pytest
from hfmodem.shrike import onair, pactor1
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p2_policy import linked, grant
from hfmodem.tests.shrike.test_p3_offer import cs_event


@pytest.mark.parametrize('kind', ['ack', 'grant'])
def test_old_connect_response_cannot_accept_p2_setup(kind):
    host, _ = linked()
    host.p2_announcement_end = round(34.787 * onair.FS)
    rx = onair._SessionRx(host)
    packet = host.arq._inflight
    ev = cs_event(pactor1.CS_ACK_A) if kind == 'ack' else grant(pactor1.CS_6A9)
    rx._on(replace(ev, t=33.61))
    assert host.arq._inflight is packet
    assert host.p2_announcement_pending
    assert host.protocol == Protocol.PACTOR1
    assert not rx.cs_seen
    # A real post-announcement response still works.
    rx._on(replace(ev, t=34.86))
    assert not host.p2_announcement_pending
    assert host.protocol == (Protocol.PACTOR1 if kind == 'ack' else Protocol.PACTOR2)


def test_guard_does_not_change_ordinary_p1_or_p3_policy():
    host, _ = linked()
    host.pactor2_only = False
    host.p2_announcement_end = round(34.787 * onair.FS)
    rx = onair._SessionRx(host)
    events = []
    host.on_rx_event = events.append
    ev = replace(cs_event(pactor1.CS_ACK_A), t=33.61)
    rx._on(ev)
    assert events == [ev]


def test_disconnect_resets_announcement_epoch():
    host, _ = linked()
    host.p2_announcement_end = 123456
    host.arq.on_host_abort()
    assert host.p2_announcement_end is None


@pytest.mark.parametrize('refused', [False, True])
def test_announcement_epoch_requires_emission_and_stays_fixed(tmp_path, refused):
    host, _ = linked()
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    tx.host = host
    tx.p1_status_bits45 = 1
    tx.p1_status_from = 1
    def emit(*args, **kwargs):
        tx.refused = refused
        tx.tx_end = 1669774
    tx._tx = emit
    tx.send_p1_packet(b'1w9ssj\r', 100, 1)
    assert host.p2_announcement_end == (None if refused else 1669774)
    if not refused:
        def repeat(*args, **kwargs):
            tx.tx_end = 1729774
        tx._tx = repeat
        tx.send_p1_packet(b'1w9ssj\r', 100, 1)
        assert host.p2_announcement_end == 1669774
