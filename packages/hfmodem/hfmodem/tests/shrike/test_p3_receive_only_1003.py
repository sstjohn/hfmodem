# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-only lifetime, proposal cancellation and current return ownership."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import modem, opportunity, mail, acknowledge_all
from hfmodem.tests.shrike.test_tx_latency_correction import _Stream


@pytest.mark.parametrize('ticks', [1, 3, 7, 96])
def test_elapsed_physical_lifetime_is_settled_once_before_key(ticks):
    a, io = modem()
    a.on_host_data(b'owned suffix')
    keys = 0
    while a.state == arq.State.CONNECTED:
        token = a.begin_receive_opportunity()
        a.qualify_peer(token, a._link_generation, None)
        before = len(io.events)
        active = a.settle_peer_presence(elapsed_ticks=ticks)
        age = a._p3_crc_wait_ticks
        a.settle_peer_presence(elapsed_ticks=ticks)
        assert a._p3_crc_wait_ticks == age
        if active:
            a.on_cycle(elapsed_ticks=ticks, cycle_ticks=3 if ticks == 3 else 1)
        a.finish_receive_opportunity(token)
        keys += len(io.events)-before
        if age >= 96:
            assert len(io.events) == before
    assert keys <= 5
    assert 96 <= a._p3_crc_wait_ticks < 96+ticks
    assert a.transfer_unconfirmed['suffix'] == b'owned suffix'


def test_header_return_keeps_crc_ceiling_and_does_not_replay_canceled_answer(tmp_path):
    s = _Session(role=arq.IRS)
    a = s.host.arq
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.host = s.host
    s.host.peer = tx
    tx.defer_p3_cs = True
    a.on_host_data(b'keep me')
    tx.send_cs(arq.CS_REQUEST)
    original = a.p3_tx_owner
    for _ in range(6):
        opportunity(a, tick=False)
    assert a.p3_receive_only and tx._pending_p3_cs is None
    assert not a.p3_tx_allowed(original)
    assert opportunity(a, source='tracked SL2 short header', tick=False)
    assert not a.p3_receive_only and a._p3_crc_wait_ticks == 7
    assert tx._pending_p3_cs is None and not a.p3_tx_allowed(original)
    assert bytes(a._outbuf) == b'keep me' and a.acked == 0


def test_old_physical_crc_keeps_delivery_but_cannot_reset_current_lifetime():
    a, io = modem()
    for _ in range(6):
        opportunity(a)
    a._p3_crc_wait_ticks = 95
    token = a.begin_receive_opportunity()
    a.on_rx_packet(1, b'older valid bytes', 0x21, crc_ok=True)
    assert io.received == [b'older valid bytes']
    a.qualify_peer(token, a._link_generation, None)
    assert not a.settle_peer_presence()
    a.finish_receive_opportunity(token)
    assert a.state == arq.State.DISCONNECTED and a._p3_crc_wait_ticks == 96


def test_current_crc_at_lifetime_boundary_resets_before_settlement():
    a, _ = modem()
    a._p3_crc_wait_ticks = 95
    a._enter_receive_only()
    token = a.begin_receive_opportunity()
    a.on_rx_packet(1, b'', 0x21, crc_ok=True)
    a.qualify_peer(token, a._link_generation, 'current CRC')
    assert a.settle_peer_presence()
    assert not a.p3_receive_only and a._p3_crc_wait_ticks == 0
    a.finish_receive_opportunity(token)


def test_supported_header_at_crc_ceiling_uses_existing_close_only():
    a, _ = modem()
    a.on_host_data(b'unconfirmed')
    a._p3_crc_wait_ticks = 95
    assert opportunity(a, source='tracked SL2 short header', tick=False)
    assert a._qrt_pending and not a.p3_receive_only
    assert a.transfer_unconfirmed['suffix'] == b'unconfirmed'
    assert not a._outbuf


@pytest.mark.parametrize('request_before', [False, True])
def test_explicit_local_disconnect_wins_quiet_entry_and_cannot_revive(request_before):
    a, io = modem()
    a.on_host_data(b'pending source')
    for _ in range(5):
        opportunity(a)
    if request_before:
        a.on_host_disconnect()
    opportunity(a, tick=False)
    if not request_before:
        assert a.p3_receive_only
        a.on_host_disconnect()
    count = len(io.events)
    assert a.state == arq.State.DISCONNECTED
    assert a.transfer_unconfirmed['suffix'] == b'pending source'
    assert not opportunity(a, source='current CRC')
    assert len(io.events) == count


def test_legacy_absence_budget_enters_quiet_without_discard_or_qrt():
    a, io = modem()
    a.cfg.max_retries = 2
    a.on_host_data(b'still ours')
    opportunity(a)
    opportunity(a)
    count = len(io.events)
    opportunity(a)
    assert a.p3_receive_only and len(io.events) == count
    assert not a._qrt_pending and bytes(a._outbuf) == b'still ours'
    silence = a._silent_cycles
    for _ in range(4):
        opportunity(a)
    assert a._silent_cycles == silence and a._p3_crc_wait_ticks == 7


def _application_expirer(a):
    from hfmodem.shrike.p3progress import MailProgress
    from hfmodem.winlink import B2FSession, MailClient
    from hfmodem.winlink.session import _WAIT_FS
    app = B2FSession('W9SSJ', target='WS8EOC')
    app._state = _WAIT_FS
    client = MailClient(app, a.on_host_data)
    now = [0.]
    timer = client.progress = MailProgress(client, a, now=lambda: now[0])
    assert not timer.poll(qualified=False) and timer.deadline == 120.
    def expire():
        now[0] = 120.
        assert timer.poll(qualified=False) and timer.expired
    return expire


@pytest.mark.parametrize('cause,stage',
    [pytest.param('presence', stage, id=stage) for stage in ('initial', 'drain', 'wait', 'enqueue', 'key',
                                      'epoch', 'counter', 'tx_counter', 'role')]
    + [pytest.param('application', stage, id='application-'+stage)
       for stage in ('initial', 'drain', 'wait', 'enqueue', 'key')])
def test_actual_radio_admission_cancels_before_enqueue_or_key(tmp_path, stage, cause):
    s = _Session(role=arq.IRS)
    a = s.host.arq
    expire = _application_expirer(a) if cause == 'application' else None
    a.on_host_data(b'retained')
    stream = _Stream()
    keys = []
    rig = SimpleNamespace(ptt=lambda down: keys.append(down), key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    s.host.peer = tx
    tx.host, tx.live = s.host, stream
    tx.aim(onair._MasterGrid(100000, 60000, 0, packet_n=36000,
                            cs_n=12000, d_max_n=4800), 0)
    cancel = expire if cause == 'application' else a._enter_receive_only
    if stage == 'initial':
        cancel()
    elif stage == 'drain':
        old = stream.take_until
        def drain(end):
            result = old(end)
            cancel()
            return result
        stream.take_until = drain
    elif stage == 'wait':
        stream.wait_until = lambda _: cancel()
    elif stage == 'enqueue':
        stream._tx_done.clear = cancel
    elif stage == 'key':
        stream._sleep_until = lambda _: cancel()
    else:
        if stage == 'tx_counter':
            a._inflight = arq._Packet(status=0, payload=b'', sl=1)
        tx._p3_render_owner = a.p3_tx_owner
        if stage == 'epoch':
            cancel()
            a._p3_receive_only = False  # A qualified return cannot revive an old render.
        elif stage == 'counter':
            a._expected_seq = (a._expected_seq+1) & 3
        elif stage == 'tx_counter':
            a._inflight.status = 1
        else:
            a.role = arq.ISS
    tx._tx(np.ones(10080, np.float32), 'offline P3 cancellation')
    assert tx.refused and not tx.slots_used and not tx.keyed
    assert True not in keys and stream._tx is None
    if cause == 'application':
        assert not a._outbuf and a.transfer_unconfirmed['suffix'] == b'retained'
    else:
        assert bytes(a._outbuf) == b'retained'
    assert a.acked == 0


def test_failed_application_stays_failed_on_return_and_new_client_bytes():
    a, io, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll(qualified=True)
    for _ in range(6):
        opportunity(a)
    now[0] = 120.
    timer.poll(qualified=False)
    assert a.p3_receive_only and not a._qrt_pending
    a.on_host_data(b'later client bytes')
    assert not a._outbuf and a.transfer_unconfirmed['suffix'].endswith(b'later client bytes')
    assert opportunity(a, source='tracked SL2 short header', tick=False)
    assert timer.poll(qualified=True) and timer.close_requested and a._qrt_pending
    assert client.session.failure and not a._outbuf and a._inflight is None


@pytest.mark.parametrize('cause,form', [
    pytest.param(cause, form, id=form if cause == 'presence' else form+'-application')
    for cause in ('presence', 'application')
    for form in ('control', 'short', 'long', 'terminal')])
def test_cancel_and_return_during_render_cannot_revive_prepared_waveform(form, tmp_path, monkeypatch, cause):
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    s = _Session(role=arq.IRS)
    a = s.host.arq
    g = irs_grid()
    g.note_p3_packet(1500000, 39120, 60000, identity=(False, 1, 1, b''))
    tx = driver(g, tmp_path)
    tx.host, s.host.peer = s.host, tx
    tx.aim(g, 30)
    tx.live.now = tx.live.pos = tx.key_instant(g, 30)-1920
    expire = _application_expirer(a) if cause == 'application' else a._enter_receive_only
    offset = tx._p3_offset
    def cancel(*args, **kwargs):
        result = offset(*args, **kwargs)
        expire()
        a._p3_receive_only = False  # A returning peer is a new epoch.
        return result
    monkeypatch.setattr(tx, '_p3_offset', cancel)
    if form == 'control':
        result = tx._send_p3_control(arq.CS_ACK)
    elif form == 'terminal':
        result = tx.send_p3_terminal()
    elif form == 'long':
        result = tx.send_long_packet(2, b'owned', 0x21)
    else:
        result = tx.send_packet(1, b'owned', 0x21)
    assert result == arq.REFUSED
    assert tx.refused and not tx.live.emissions and not tx.slots_used


@pytest.mark.parametrize('mask', ['clear', 'ptt_lead', 'prior_key', 'context', 'lost'])
@pytest.mark.parametrize('unconfirmed', [False, True])
def test_current_crc_presence_requires_unmasked_recorded_support(mask, unconfirmed):
    from hfmodem.tests.shrike.test_p3_presence_progress_1003 import rendered, presence_session
    from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
    from hfmodem.tests.shrike.p3_receive_only_support import DeferredReplay
    audio, row = rendered(2)
    s = presence_session(row)
    a, rx = s.host.arq, s.rx
    a.cfg.long_cycle = False
    if unconfirmed:
        a.role = arq.ISS
        a._inflight = arq._Packet(status=0, payload=b'owned turn', sl=1, breakin=True)
    rx.sync.packet_level = 2
    rx._p3_presence_form = (a._link_generation, 2)
    seam = s.host.peer = DeferredReplay(s.host)
    seam.slot, seam.raster = 1, irs_grid()
    seam.raster.boundary = lambda _: row+37000
    seam.live = SimpleNamespace(lost=int(mask == 'lost'))
    token = a.begin_receive_opportunity()
    a._receive_opportunity.capture_lost = 0
    onair._scan_frame(rx, audio, 0, tracked_only=True)
    assert s.packets[-1].packet[3]
    delivered = s.host.rcvd_total
    if mask == 'ptt_lead':
        seam.tx_key_up, seam.tx_audio_start, seam.tx_end = row-100, row+2000, row+3000
    elif mask == 'prior_key':
        seam.keyings = [(row-100, row+100), (row+50000, row+51000)]
    elif mask == 'context':
        rx._p3_presence_gaps = [(row-4320-100, row-4320-10)]
    onair._p3_qualify_presence(rx, seam.raster, seam, audio, 0)
    assert a.peer_qualified == (mask == 'clear')
    assert a._receive_opportunity.crc and s.host.rcvd_total == delivered
    a._p3_crc_wait_ticks = 90
    a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == (0 if mask == 'clear' else 91)


def test_early_render_refusal_releases_scope_before_new_protocol(tmp_path, monkeypatch):
    s = _Session(role=arq.IRS)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.host = s.host
    monkeypatch.setattr(tx, '_render_p3_control', lambda *args, **kwargs: arq.REFUSED)
    assert tx._send_p3_control(arq.CS_ACK) == arq.REFUSED
    assert tx._p3_render_owner is None
    s.host.protocol = spec.Protocol.PACTOR1
    s.host.arq.state = arq.State.CONNECTING
    observed = []
    monkeypatch.setattr(tx, '_tx_admitted', lambda *args, **kwargs: observed.append(tx._p3_tx_admit))
    tx._tx(np.zeros(1), 'new P1 call')
    assert observed == [None]


@pytest.mark.parametrize('crc', [False, True])
def test_generation_change_after_qualification_retires_all_old_consumers(crc):
    a, _ = modem()
    token = a.begin_receive_opportunity()
    if crc:
        a.on_rx_packet(1, b'', 0x21, crc_ok=True)
    a.qualify_peer(token, a._link_generation, 'current CRC' if crc else 'tracked SL1 short header')
    assert a.peer_qualified
    a._enter_connected()
    a._silent_cycles, a._p3_rx_errors = 3, 2
    a._dark_windows, a._p3_crc_wait_ticks = 4, 17
    assert not a.peer_qualified
    assert not a.qualify_peer(token, a._link_generation, 'current CRC')
    a.settle_peer_presence(elapsed_ticks=7)
    a.finish_receive_opportunity(token)
    assert (a._silent_cycles, a._p3_rx_errors, a._dark_windows, a._p3_crc_wait_ticks) == (3, 2, 4, 17)
    assert a._receive_opportunity is None


def test_bounded_retry_commits_only_fresh_queued_counter_word(tmp_path, monkeypatch):
    from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
    s = _Session(role=arq.IRS)
    a = s.host.arq
    a.cfg.long_cycle = False
    a._expected_seq = 1
    g = irs_grid()
    g.note_p3_packet(1500000, 39120, 60000, identity=(False, 1, 0, b''))
    tx = driver(g, tmp_path)
    tx.host, s.host.peer = s.host, tx
    tx.defer_p3_cs = True
    tx.aim(g, 30)
    tx.live.now = tx.live.pos = tx.key_instant(g, 30)-1920
    tx.send_cs(arq.CS_ACK)
    calls, callbacks = [], []
    render, take = tx._render_p3_control, tx.live.take_until
    def first_late(cs, **kwargs):
        calls.append(cs)
        if len(calls) == 1:
            tx._rerender_late = tx.refused = True
            return arq.REFUSED
        return render(cs, **kwargs)
    def fresh(end):
        result = take(end)
        if a.rx_seq == 0:
            a.on_rx_packet(1, b'', 0x21, crc_ok=True)
        return result
    monkeypatch.setattr(tx, '_render_p3_control', first_late)
    monkeypatch.setattr(tx.live, 'take_until', fresh)
    callback = s.host.on_cs_emitted
    monkeypatch.setattr(s.host, 'on_cs_emitted', lambda cs: (callbacks.append(cs), callback(cs)))
    tx.emit_pending_cs()
    assert calls == [arq.CS_ACK, arq.CS_REQUEST]
    assert callbacks == [arq.CS_REQUEST]
    assert tx.n == 1 and not tx.refused and tx._pending_p3_cs is None


def test_old_crc_cannot_extend_lower_legacy_absence_allowance():
    a, io = modem()
    a.cfg.max_retries = 2
    for expected in (1, 2, 2):
        token = a.begin_receive_opportunity()
        a.on_rx_packet(1, b'older independently valid bytes', 0x21, crc_ok=True)
        a.qualify_peer(token, a._link_generation, None)
        a.settle_peer_presence()
        quiet = a.p3_receive_only
        a.on_cycle()
        a.finish_receive_opportunity(token)
        assert a._silent_cycles == expected
    assert quiet and a._dark_windows == a._p3_crc_wait_ticks == 3
    assert io.received == [b'older independently valid bytes']
    assert opportunity(a, source='tracked SL1 short header', tick=False)
    assert not a.p3_receive_only and a._silent_cycles == 0
    assert a._p3_crc_wait_ticks == 4
