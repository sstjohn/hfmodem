# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native first-call/raster handoff through an independent CRC client; no RF."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hfhost import ptc
from hfmodem.shrike import onair, p1rx
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_station_io import port, PollingCapture, FeedingCapture


def read_available(fd):
    out = bytearray()
    while True:
        try:
            out += os.read(fd, 4096)
        except BlockingIOError:
            return bytes(out)


def serial(tmp_path, *, during_key=lambda: None, capture_factory=PollingCapture):
    replies = []
    decoder = ptc.Deframer()
    def service():
        if station.rig.keyed:
            during_key()
        assert endpoint.service()
        replies.extend(decoder.feed(read_available(fd)))
    live, station = port(capture_factory=capture_factory, owner_poll=service)
    rig = SimpleNamespace(ptt=lambda on: pytest.fail('independent PTT'),
                          key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, drive=.6, settle=.04)
    tx.live = live
    host = PtcHost(tx, mycall='N0CALL')
    owner = host.enable_owner(defer_connect=True)
    host.tick = lambda **kw: pytest.fail('serial servicing ticked ARQ')
    endpoint = PtcSerialEndpoint(host)
    fd = os.open(endpoint.open(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    os.write(fd, b'PTC 31\rJHOST4\r')
    for _ in range(3):
        endpoint.service()
    read_available(fd)
    return host, owner, endpoint, fd, tx, live, station, replies


def test_first_serial_call_is_written_before_native_transmit_and_retry_is_cached(tmp_path):
    host, owner, ep, fd, tx, live, station, replies = serial(tmp_path)
    frame = ptc.Packet(31, True, b'C %N0DX', counter=1).encode()
    try:
        os.write(fd, frame)
        assert ep.service()
        assert host.arq.state == State.CONNECTING and owner.connect_pending
        assert host.arq.connect_variant == 'robust'
        assert not station.rig.edges and ep.pending_bytes
        assert not owner.start_connect(ep)
        assert ep.service() and not ep.pending_bytes
        result = ptc.Deframer().feed(read_available(fd))
        assert len(result) == 1 and result[0].code == ptc.OK
        # Lose/retry that response while the native producer is suspended in
        # its actual station wait. There must still be exactly one first call.
        os.write(fd, frame)
        assert owner.start_connect(ep)
        assert not owner.start_connect(ep) and not owner.connect_pending
        assert len(replies) == 1 and replies[0].code == ptc.OK
        assert host.arq._host_sync_packets == 1
        assert station.rig.edges == [True, False] and len(tx.keyed) == 1
        assert tx.tx_end == live.pos and station.arbiter.sent == 1
        emitted = station.audio.sent[0][1]
        calls = p1rx.search_calls(np.pad(emitted, (12000, 12000)), 'N0DX')
        assert len(calls) == 1 and calls[0][1].variant == 'Robust'
    finally:
        ep.close()
        os.close(fd)
        live.close()


def test_disconnect_during_first_call_waits_for_emission_to_return(tmp_path):
    requested = []
    def disconnect():
        if not requested:
            requested.append(True)
            os.write(fd, ptc.Packet(31, True, b'D', counter=0).encode())
        assert host.arq.state == State.CONNECTING
    host, owner, ep, fd, tx, live, station, replies = serial(tmp_path, during_key=disconnect)
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode())
        ep.service()
        ep.service()
        read_available(fd)
        assert owner.start_connect(ep)
        assert requested and host.arq.state == State.LISTENING
        assert len(tx.keyed) == 1 and not owner.connect_pending
        assert not owner.in_operation and not owner.pending_count
        assert any(reply.code == ptc.OK for reply in replies)
        assert station.rig.edges == [True, False]
    finally:
        ep.close()
        os.close(fd)
        live.close()


@pytest.mark.parametrize('cancel', [b'D', b'DD', b'#JHOST4'])
def test_cancel_or_reentry_before_output_drain_cannot_emit_stale_call(tmp_path, cancel):
    host, owner, ep, fd, tx, live, station, _ = serial(tmp_path)
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode()
                 + ptc.Packet(31, True, cancel, counter=0).encode())
        ep.service()
        ep.service()
        assert not owner.connect_pending and not owner.start_connect(ep)
        assert not tx.keyed and not station.rig.edges
        assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    finally:
        ep.close()
        os.close(fd)
        live.close()


def test_failed_initial_output_aborts_before_a_caller_can_resume(tmp_path):
    host, owner, ep, fd, tx, live, station, _ = serial(tmp_path)
    failure = OSError('fixture output refused')
    def fail(*args):
        raise failure
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode())
        ep.service()
        ep.service()
        tx.connect_burst = fail
        with pytest.raises(OSError) as caught:
            owner.start_connect(ep)
        assert caught.value is failure
        assert host.arq.state == State.LISTENING
        assert not owner.connect_pending and not owner.in_operation
        assert not owner.start_connect(ep) and not station.rig.edges
    finally:
        ep.close()
        os.close(fd)
        live.close()


def test_native_driver_continues_the_emitted_call_without_a_second_host_connect(monkeypatch, tmp_path):
    clock = _Bench(seconds=20)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live = clock
    tx.p4_probe = None
    host = PtcHost(tx, mycall='N0CALL')
    host.stay_in_pactor1 = True
    owner = host.enable_owner(defer_connect=True)
    owner.request('connect', 'N0CALL', 'N0DX')
    endpoint = SimpleNamespace(host=host, closed=False, pending_bytes=0)
    assert owner.start_connect(endpoint)
    initial = tx.tx_end
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL', '--max-cycles', '1'])
    # The driver is responsible for native listening and retry slots. Prevent
    # a duplicate C from being disguised as a legitimate later raster retry.
    host.arq.on_host_connect = lambda *a, **kw: pytest.fail('second initial C')
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    identity = SimpleNamespace(cw_call='N0CALL', closing_required=lambda *a, **kw: False)
    code = onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=clock,
        rig=None, dx='N0DX', reply=None, outdir=tmp_path, identity=identity, started=True)
    assert code == 0 and tx.raster is not None
    assert tx.raster.anchor == initial - round(tx.last_dur * onair.FS)
    assert clock.pos > initial and host.arq._host_sync_packets == 2
    assert len(tx.keyed) == 2  # initial call and the one allowed raster retry
    assert clock.emissions[1][0] - clock.emissions[0][0] == 60000


@pytest.mark.parametrize('when', ['before', 'capture', 'finding'])
def test_native_acquisition_rechecks_admission_before_accepting_a_call(monkeypatch, when):
    clock = _Bench(seconds=3)
    eligible = [when != 'before']
    real_read = clock.read
    def read(count):
        result = real_read(count)
        if when == 'capture':
            eligible[0] = False
        return result
    clock.read = read
    searches = []
    def search(*args, **kwargs):
        searches.append(True)
        if when == 'finding':
            eligible[0] = False
        return [(p1rx.CALL_N / onair.FS, p1rx.Connect('Normal', 'N0CALL', False))]
    monkeypatch.setattr(p1rx, 'search_calls', search)
    assert onair._await_call(clock, 'N0CALL', 3, eligible=lambda: eligible[0]) is None
    assert bool(searches) == (when == 'finding')
    assert not clock.emissions


@pytest.mark.parametrize('csdelay', [None, 10])
def test_native_acquired_call_places_answer_grid_before_the_first_ack(monkeypatch, tmp_path, csdelay):
    from hfmodem.shrike import pactor1
    clock = _Bench(seconds=8)
    signal = pactor1.connect_signal('N0CALL', invert=True, lead_s=0, tail_s=0)
    at = 12000
    clock.audio[at:at + len(signal)] = signal
    finding = onair._await_call(clock, 'N0CALL', 4, eligible=lambda: True)
    assert finding and abs(finding[0] - at) <= 48
    assert finding[1].inverted and finding[1].callsign == 'N0CALL'
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live, tx.p4_probe = clock, None
    host = PtcHost(tx, mycall='N0CALL')
    if csdelay is not None:
        host._store('TXDELAY', '4')
        host._store('CSDELAY', str(csdelay))
        host._store('MAXERROR', '70')
    host.enable_owner(defer_connect=True)
    host.set_listen(True)
    host.stay_in_pactor1 = True
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL', '--max-cycles', '1'])
    delay = onair.ANSWER_D_S if csdelay is None else csdelay * .005
    expected = onair._called_grid(*finding, tx.settle,
        offset_n=round(args.tx_offset*onair.FS), response_delay_s=delay)
    class FirstAnswerComplete(Exception):
        pass
    original = host.on_rx_event
    def accept(event):
        assert tx.raster.called and tx.raster.anchor == expected.anchor
        original(event)
        assert host.arq.answering and len(tx.keyed) == 1
        raise FirstAnswerComplete()
    host.on_rx_event = accept
    identity = SimpleNamespace(cw_call='N0CALL', closing_required=lambda *a, **kw: False)
    with pytest.raises(FirstAnswerComplete):
        onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=clock,
            rig=None, dx='', reply=None, outdir=tmp_path, identity=identity, called=finding,
            host_controlled=csdelay is not None)
    assert len(clock.emissions) == 1
    assert clock.emissions[0][0] == expected.boundary(0)
    assert clock.emissions[0][0] - finding[0] == round((.960 + delay) * onair.FS)
    assert not host._owner.connect_pending


@pytest.mark.parametrize('invalid', ['inactive', 'scan', 'output', 'wrong_call', 'future', 'clock'])
def test_preacquired_handoff_refuses_ineligible_or_mismatched_contact_before_transmit(tmp_path, invalid):
    clock = _Bench(seconds=3)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path)
    tx.live = clock
    host = PtcHost(tx, mycall='N0CALL')
    host.set_listen(True)
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL'])
    start, call = 0, p1rx.Connect('Normal', 'N0CALL', False)
    if invalid == 'inactive':
        host.set_listen(False)
        host.arq.on_host_listen(False)
    elif invalid == 'scan':
        host.arq._scan_hold = True
    elif invalid == 'output':
        host.channel(host.ptchn).rx += b'old contact'
    elif invalid == 'wrong_call':
        call.callsign = 'N0OTHER'
    elif invalid == 'future':
        start = 1
    elif invalid == 'clock':
        tx.live = None
    with pytest.raises(ValueError, match='current eligible acquisition'):
        onair._drive_session(args, tx=tx, host=host, sessrx=None, live=clock,
            rig=None, dx='', reply=None, outdir=tmp_path, identity=None, called=(start, call))
    assert not clock.emissions and not tx.keyed


@pytest.mark.parametrize('invalid', ['closed', 'other_host', 'inside_operation'])
def test_initial_call_cannot_bypass_endpoint_or_operation_boundaries(tmp_path, invalid):
    host, owner, ep, fd, tx, live, station, _ = serial(tmp_path)
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode())
        ep.service()
        ep.service()
        if invalid == 'inside_operation':
            with owner.operation('on_cycle'):
                with pytest.raises(RuntimeError, match='outer owner boundary'):
                    owner.start_connect(ep)
        else:
            endpoint = SimpleNamespace(host=host, closed=invalid == 'closed', pending_bytes=0)
            if invalid == 'other_host':
                endpoint.host = PtcHost()
            with pytest.raises((ConnectionAbortedError, ValueError)):
                owner.start_connect(endpoint)
        assert owner.connect_pending and not tx.keyed and not station.rig.edges
    finally:
        ep.close()
        os.close(fd)
        live.close()


def test_first_call_cancellation_closes_through_native_station_cw_under_one_lease(tmp_path):
    from hfmodem.core.identification import CallIdentity
    from hfmodem.station.arbiter import Identity
    sent_d, polled_close, refused_c = [], [], []
    def service():
        assert owner.contact_active
        if live._identifying is None and not sent_d:
            os.write(fd, ptc.Packet(31, True, b'D', counter=0).encode())
            sent_d.append(True)
        elif live._identifying is not None:
            if not polled_close:
                assert host.arq.state == State.LISTENING
                os.write(fd, ptc.Packet(31, True, b'G', counter=1).encode())
                polled_close.append(True)
            elif any(reply.code == ptc.LINK for reply in replies) and not refused_c:
                assert host.contact_ready()
                os.write(fd, ptc.Packet(31, True, b'C N0NEXT', counter=0).encode())
                refused_c.append(True)
    host, owner, ep, fd, tx, live, station, replies = serial(
        tmp_path, during_key=service, capture_factory=FeedingCapture)
    station.audio.feed = live.feed
    read = live.read
    def card_read(count):
        station.audio.wait_until(live.pos + count)
        return read(count)
    live.read = card_read
    tx.p4_probe = None
    station.arbiter.identity = Identity('K1ABC')
    host._store('CWSPEED', '70')
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL', '--max-cycles', '1'])
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode())
        ep.service()
        ep.service()
        assert not ep.pending_bytes
        accepted = ptc.Deframer().feed(read_available(fd))
        assert len(accepted) == 1 and accepted[0].code == ptc.OK
        assert not tx.keyed and owner.connect_pending
        code = onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=live,
            rig=None, dx='N0DX', reply=None, outdir=tmp_path,
            identity=CallIdentity('N0CALL', 'K1ABC', 'test station'),
            start_admitted=lambda: owner.start_connect(ep))
        assert code == 0 and sent_d and polled_close and refused_c
        assert any(reply.code == ptc.FAIL and 'OWNER BUSY' in reply.text for reply in replies)
        assert len(tx.keyed) == station.arbiter.sent == 2  # initial carrier and CW
        assert station.arbiter.last_sent['ident'] == tx.tx_end
        assert not station.arbiter.identity.due
        assert host.arq.state == State.LISTENING and host.arq._host_sync_packets == 1
        assert not host.arq.payload_ledger.accepted and not owner.connect_pending
        assert not owner.contact_active and not owner._driver_active
        assert live.closed and not station.audio.closed and not station.rig.closed
        assert station.rig.edges == [True, False, True, False]
    finally:
        ep.close()
        os.close(fd)
        live.close()


def test_driver_cannot_leave_an_unemitted_call_after_a_premature_start(tmp_path):
    from hfmodem.core.identification import CallIdentity
    host, owner, ep, fd, tx, live, station, _ = serial(tmp_path)
    tx.p4_probe = None
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL'])
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    try:
        os.write(fd, ptc.Packet(31, True, b'C N0DX', counter=1).encode())
        ep.service()
        assert ep.pending_bytes and owner.connect_pending
        # The adapter must wait for serial output before entering. An early
        # handoff fails closed instead of preserving a resumable dead stream.
        with pytest.raises(ValueError, match='did not complete'):
            onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=live,
                rig=None, dx='N0DX', reply=None, outdir=tmp_path,
                identity=CallIdentity('N0CALL', 'K1ABC', 'test station'),
                start_admitted=lambda: owner.start_connect(ep))
        assert host.arq.state == State.LISTENING
        assert not owner.connect_pending and not owner.contact_active
        assert live.closed and not tx.keyed and not station.rig.edges
        assert not owner.start_connect(ep)
    finally:
        ep.close()
        os.close(fd)
        live.close()
