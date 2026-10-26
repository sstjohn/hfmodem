# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Expired mail retains physical RX attribution, never its retired TX field."""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair
from hfmodem.shrike.p3progress import MailProgress
from hfmodem.winlink import B2FSession, MailClient
from hfmodem.winlink.session import _WAIT_FS
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import rendered, presence_session
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
from hfmodem.tests.shrike.p3_receive_only_support import DeferredReplay


@pytest.mark.parametrize('case,route,evidence,initial', [
    pytest.param(case, route, evidence, initial,
                 id=f'{case}-{route}-{evidence}'+('-active' if initial == 'active' else ''))
    for case in ('current', 'masked', 'no_form', 'stale_form', 'new_generation', 'confirmed_iss')
    for route in ('direct', 'next_cycle') for evidence in ('crc', 'header')
    for initial in ('quiet', 'active')])
def test_expired_unconfirmed_iss_return_uses_current_driver_attribution(evidence, case, route, initial):
    audio, row = rendered(2)
    session = presence_session(row)
    a, rx = session.host.arq, session.rx
    a.cfg.long_cycle = False
    rx.sync.packet_level = 2
    rx._p3_presence_form = (a._link_generation, 2)
    rx._p3_clock_role = arq.IRS
    seam = session.host.peer = DeferredReplay(session.host)
    seam.slot, seam.raster = 1, irs_grid()
    seam.raster.boundary = lambda _: row+37000
    seam.live = SimpleNamespace(lost=0)
    app = B2FSession('W9SSJ', target='WS8EOC')
    app._state = _WAIT_FS
    client = MailClient(app, a.on_host_data)
    clock = [0.]
    timer = client.progress = MailProgress(client, a, now=lambda: clock[0])
    # No outstanding application watermark: this timer is legitimately armed.
    assert not timer.poll(qualified=False) and timer.deadline == 120.
    a.role = arq.ISS
    rx._p1_role = arq.ISS  # The quiet interval already spans local ISS cycles.
    a._inflight = arq._Packet(status=0, payload=b'retired turn', sl=1,
                              breakin=case != 'confirmed_iss')
    a._outbuf.extend(b'retired suffix')
    if initial == 'quiet':
        a._enter_receive_only()
    else:
        assert not a.p3_receive_only and not a._dark_windows
    clock[0] = 120.
    assert timer.poll(qualified=False)
    # Exercise ordinary maintenance before attributing a returning packet.
    # Sticky application failure cannot restart even an empty DATA packet.
    a.on_cycle()
    assert not seam.sent and seam.pending is None and a._inflight is None
    assert timer.expired and not timer.close_requested and a.p3_receive_only
    assert a._inflight is None and not a._outbuf and not a.unconfirmed_breakin
    retained = dict(a.transfer_unconfirmed)
    assert retained['field'] == b'retired turn' and retained['suffix'] == b'retired suffix'
    assert not seam.sent and seam.pending is None
    if case == 'new_generation':
        a._enter_connected()
        a.role = arq.ISS
        # Supply a current form to isolate stale physical ownership retirement.
        rx._p3_presence_form = (a._link_generation, 2)
    elif case == 'no_form':
        rx._p3_presence_form = None
    elif case == 'stale_form':
        rx._p3_presence_form = (a._link_generation-1, 2)
    elif case == 'masked':
        seam.keyings = [(row-100, row+100)]
    token = a.begin_receive_opportunity()
    a._receive_opportunity.capture_lost = 0
    credit = a.acked, a.tx_acked_bytes
    if route == 'next_cycle':
        rx.new_cycle()
        if case not in ('new_generation', 'confirmed_iss'):
            assert rx._p3_row0 == row and rx._p3_clock_role == arq.IRS
        else:
            assert rx._p3_row0 is None
    if evidence == 'crc':
        if route == 'next_cycle':
            read = onair._scan_previous_window(rx, audio, 0, sending=True, flowing=False)
            assert read == (case not in ('new_generation', 'confirmed_iss'))
        else:
            onair._scan_frame(rx, audio, 0, tracked_only=True)
            read = True
        if read:
            assert session.packets[-1].packet[3]
        # A new CRC can refresh the accepted form; exercise the explicitly
        # stale/missing-form consumer negatives after that normal publication.
        if case == 'no_form':
            rx._p3_presence_form = None
        elif case == 'stale_form':
            rx._p3_presence_form = (a._link_generation-1, 2)
    onair._p3_qualify_presence(rx, seam.raster, seam, audio, 0)
    qualified = a.peer_qualified
    assert qualified == (case == 'current')
    assert (a.acked, a.tx_acked_bytes) == credit and a.role == arq.ISS
    a.settle_peer_presence()
    if case == 'new_generation':
        assert not a.peer_qualified
        assert not timer.poll(qualified=a.peer_qualified)  # Old failure is not a new-link timer.
    else:
        assert timer.poll(qualified=a.peer_qualified)
        assert timer.close_requested == (case == 'current')
        assert a._qrt_pending == (case == 'current')
        assert a.transfer_unconfirmed == retained and client.session.failure
        assert not a._outbuf and a._inflight is None
        if case == 'current':
            assert not a.p3_receive_only
            assert a._receive_opportunity.crc == (evidence == 'crc')
            _, outgoing, _, _, _ = a._next_packet(False)
            assert outgoing == b''  # Returning peer cannot resurrect failed mail.
        else:
            assert a.p3_receive_only and not seam.sent and seam.pending is None
    a.finish_receive_opportunity(token)


def _offline_key_cycle(session, outdir):
    """Use the normal renderer/admission with the existing offline key stub."""
    from hfmodem.tests.shrike.test_p3_reply_placement import driver
    outdir.mkdir(parents=True, exist_ok=True)
    grid = irs_grid()
    if session.host.arq.role == arq.ISS:
        grid.reverse(to_iss=True)
    radio = driver(grid, outdir)
    radio.host = session.host
    previous, session.host.peer = session.host.peer, radio
    radio.aim(grid, 30)
    radio.live.now = radio.live.pos = radio.key_instant(grid, 30)-6000
    attempts = []
    send = radio.send_packet

    def record(sl, payload, status, breakin=False):
        attempts.append((len(payload), bool(status & arq.spec.STATUS_QRT), breakin))
        return send(sl, payload, status, breakin=breakin)

    radio.send_packet = record
    try:
        session.host.arq.on_cycle()
    finally:
        session.host.peer = previous
    return attempts, radio.n, radio.refused


@pytest.mark.parametrize('initial', ['quiet', 'active'])
@pytest.mark.parametrize('role', ['irs', 'unconfirmed_iss', 'confirmed_iss'])
@pytest.mark.parametrize('at_expiry', ['unsupported', 'qualified'])
@pytest.mark.parametrize('evidence', ['crc', 'header'])
def test_terminal_expiry_actual_cycle_keys_only_qualified_close(initial, at_expiry, evidence, role, tmp_path):
    audio, row = rendered(2)
    session = presence_session(row)
    a, rx = session.host.arq, session.rx
    a.cfg.long_cycle = False
    a.role = arq.IRS if role == 'irs' else arq.ISS
    a._inflight = (None if role == 'irs' else arq._Packet(
        status=0, payload=b'unconfirmed field', sl=1, breakin=role == 'unconfirmed_iss'))
    a._outbuf.extend(b'unconfirmed suffix')
    rx.sync.packet_level = 2
    rx._p3_presence_form = (a._link_generation, 2)
    rx._p3_clock_role, rx._p1_role = arq.IRS, arq.ISS
    seam = session.host.peer = DeferredReplay(session.host)
    seam.slot, seam.raster = 1, irs_grid()
    seam.raster.boundary = lambda _: row+37000
    seam.live = SimpleNamespace(lost=0)
    app = B2FSession('W9SSJ', target='WS8EOC')
    app._state = _WAIT_FS
    client = MailClient(app, a.on_host_data)
    clock = [0.]
    timer = client.progress = MailProgress(client, a, now=lambda: clock[0])
    assert not timer.poll(qualified=False) and timer.deadline == 120.
    if initial == 'quiet':
        a._enter_receive_only()
    credit = a.acked, a.tx_acked_bytes
    token = a.begin_receive_opportunity()
    a._receive_opportunity.capture_lost = 0
    original_owner = a.p3_tx_owner
    seam.send_cs(arq.CS_REQUEST)
    if at_expiry == 'unsupported' or role == 'confirmed_iss':
        clock[0] = 120.
        assert timer.poll(qualified=a.peer_qualified)
        assert a.p3_receive_only
        assert a.p3_peer_receive_owned == (role != 'confirmed_iss')
        assert seam.pending is None and not a.p3_tx_allowed(original_owner)
        assert _offline_key_cycle(session, tmp_path/'unsupported') == ([], 0, False)
        assert not timer.close_requested and not a._qrt_pending
    rx.new_cycle()
    if role == 'confirmed_iss':
        assert rx._p3_row0 is None
        onair._p3_qualify_presence(rx, seam.raster, seam, audio, 0)
        assert not a.peer_qualified and not timer.close_requested
        assert not a._qrt_pending and not a.p3_peer_receive_owned
        assert _offline_key_cycle(session, tmp_path/'unowned') == ([], 0, False)
        a.finish_receive_opportunity(token)
        return
    assert rx._p3_row0 == row
    if evidence == 'crc':
        if role == 'irs':
            onair._scan_frame(rx, audio, 0, tracked_only=True)
        else:
            assert onair._scan_previous_window(rx, audio, 0, sending=True, flowing=False)
        assert session.packets[-1].packet[3]
    onair._p3_qualify_presence(rx, seam.raster, seam, audio, 0)
    assert a.peer_qualified
    a.settle_peer_presence()
    clock[0] = 120.
    assert timer.poll(qualified=a.peer_qualified)
    assert timer.expired and timer.close_requested and a._qrt_pending
    assert not a.p3_receive_only and not a._outbuf and a._inflight is None
    assert (a.acked, a.tx_acked_bytes) == credit
    retained = dict(a.transfer_unconfirmed)
    assert retained['field'] == (b'' if role == 'irs' else b'unconfirmed field')
    assert retained['suffix'] == b'unconfirmed suffix'
    a.finish_receive_opportunity(token)
    attempts, keys, refused = _offline_key_cycle(session, tmp_path/'qualified')
    assert attempts == [(0, True, role == 'irs')] and keys == 1 and not refused
    assert seam.pending is None and not a.p3_tx_allowed(original_owner)
    assert a.transfer_unconfirmed == retained
    assert (a.acked, a.tx_acked_bytes) == credit


@pytest.mark.parametrize('initial', ['quiet', 'active'])
@pytest.mark.parametrize('role', ['irs', 'unconfirmed_iss', 'confirmed_iss'])
def test_application_expiry_cannot_extend_physical_lifetime(initial, role, tmp_path):
    from hfmodem.tests.shrike.test_p3_receive_only_1003 import _application_expirer
    session = presence_session(1000)
    a = session.host.arq
    a.role = arq.IRS if role == 'irs' else arq.ISS
    expire = _application_expirer(a)
    a._outbuf.extend(b'retained suffix')
    if role != 'irs':
        a._inflight = arq._Packet(status=0, payload=b'retained field', sl=1,
                                  breakin=role == 'unconfirmed_iss')
    if initial == 'quiet':
        a._enter_receive_only()
    a._p3_crc_wait_ticks = 95
    expire()
    retained = dict(a.transfer_unconfirmed)
    assert a.p3_receive_only
    token = a.begin_receive_opportunity()
    assert a.qualify_peer(token, a._link_generation, None)
    assert not a.settle_peer_presence(elapsed_ticks=1)
    # The real PtcHost callback rearms listening after the local abort.
    assert a.state == arq.State.LISTENING and a._p3_crc_wait_ticks == 96
    a.finish_receive_opportunity(token)
    assert _offline_key_cycle(session, tmp_path) == ([], 0, False)
    assert a.transfer_unconfirmed == retained | dict(
        host_accepted=0, host_confirmed=0, host_unconfirmed=b'', contact_generation=1)
    assert not a.p3_peer_receive_owned
    assert not a.p3_tx_allowed() and not a._qrt_pending
