# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PTC permission holds the real station arbiter, never a physical radio."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfhost import ptc
from hfmodem.shrike import onair
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.station.arbiter import LIVE, Identity, Refused, TxRequest
from hfmodem.tests.shrike.test_ptc_scan_hold import Wire
from hfmodem.tests.shrike.test_station_io import port, PollingCapture


def setup(tmp_path, *, enabled=True, owner_poll=None):
    live, station = port(capture_factory=PollingCapture, owner_poll=owner_poll)
    rig = SimpleNamespace(ptt=lambda on: pytest.fail('independent PTT'), key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    tx.live = live
    host = PtcHost(tx, mycall='N0CALL')
    owner = host.enable_owner(defer_connect=True)
    live.scannable = lambda: enabled and not owner.contact_active and not owner.in_operation
    host.set_listen(True)
    return live, station, host, owner, Wire(host)


def request(live, protocol):
    return TxRequest(LIVE, 0, protocol=protocol, audio=np.full(100, .1, np.float32),
                     emission=live.link.waveform.emission(7100000))


def test_ptc_scanner_permission_blocks_every_transmit_entry_and_retune_owner(tmp_path):
    live, station, host, owner, wire = setup(tmp_path)
    station.arbiter.identity = Identity('K1ABC')
    station.arbiter.identity.end()
    identity_time = station.arbiter.identity._last
    try:
        assert wire.command('%W0').text == '1'
        token = live._scan_token
        assert token is not None and station.arbiter.scan_held
        assert host.arq.scan_held and live.scan_epoch == 1
        assert host.arq.state == State.DISCONNECTED
        for protocol in ('pactor', 'vara', 'ardop', 'rpr', 'ident'):
            with pytest.raises(Refused, match='scanner lease'):
                station.arbiter.submit(request(live, protocol))
            result = station.arbiter.submit_result(request(live, protocol))
            assert result.outcome == 'refused' and 'scanner lease' in result.reason
        with pytest.raises(Refused, match='scanner lease'):
            station.arbiter.identify(request(live, 'ident').emission)
        assert station.arbiter.identity._last == identity_time
        with station.arbiter.try_hold() as held:
            assert not held
        with station.arbiter.try_hold(scan_token=object()) as held:
            assert not held
        with station.arbiter.try_hold(scan_token=token) as held:
            assert held
        with pytest.raises(ValueError, match='token'):
            station.arbiter.release_scan_hold(object())
        assert station.arbiter.scan_held
        assert not station.rig.edges and station.audio.pending is None
        assert station.arbiter.sent == station.arbiter.transmit_history().sequence == 0
        assert wire.command('C N0DX').code == ptc.FAIL
        host.arq.on_rx_connect('N0DX', 'N0CALL')
        assert not owner.connect_pending and not station.rig.edges
        wire.command('%W1')
        assert not station.arbiter.scan_held and live._scan_token is None
        assert live.scan_epoch == 2 and host.arq.state == State.LISTENING
        station.arbiter.identity.note_sent()
        with pytest.raises(ConnectionAbortedError, match='capture retired'):
            live.transmit(np.full(100, .1, np.float32))
        from hfmodem.shrike.station_io import StationInput
        fresh = StationInput(live.link)
        fresh.transmit(np.full(100, .1, np.float32))
        fresh.close()
        assert station.rig.edges == [True, False]
    finally:
        host.attachment_lost()
        live.close()


def test_native_default_profile_cannot_grant_without_explicit_owner_admission(tmp_path):
    live, _, host, _, wire = setup(tmp_path, enabled=False)
    try:
        assert wire.command('%W0').text == '0'
        assert not host.arq.scan_held and live._scan_token is None
    finally:
        host.attachment_lost()
        live.close()


def test_active_other_lane_or_retained_contact_prevents_scan_permission(tmp_path):
    live, station, host, owner, wire = setup(tmp_path)
    try:
        station.arbiter.active_protocols = lambda: ('vara',)
        assert wire.command('%W0').text == '0'
        station.arbiter.active_protocols = lambda: ()
        with owner.contact():
            assert wire.command('%W0').text == '0'
        assert not station.arbiter.scan_held
        assert wire.command('%W0').text == '1'
        assert not station.rig.edges
    finally:
        host.attachment_lost()
        live.close()


def test_permission_is_not_granted_inside_an_actual_transmit_wait(tmp_path):
    answers = []
    def poll():
        if station.audio.pending is not None and not answers:
            answers.append(wire.command('%W0').text)
    live, station, host, _, wire = setup(tmp_path, owner_poll=poll)
    try:
        live.transmit(np.full(4800, .1, np.float32), at=4000)
        assert answers == ['0'] and not station.arbiter.scan_held
        assert not host.arq.scan_held and station.rig.edges == [True, False]
        assert wire.command('%W0').text == '1'
    finally:
        host.attachment_lost()
        live.close()


@pytest.mark.parametrize('how', ['expire', 'attachment', 'port'])
def test_scanner_lease_is_released_on_owner_retirement_and_expiry(tmp_path, how):
    live, station, host, _, wire = setup(tmp_path)
    assert wire.command('%W0').text == '1'
    try:
        if how == 'expire':
            host.wait_until = 0
            host.service_host_output()
            assert not host.arq.scan_held and host.wait_until is None
            # An expired CRC permission is no longer a reusable success.
            assert host.feed(ptc.REQUEST) == b''
            station.arbiter.active_protocols = lambda: ('vara',)
            assert wire.retry().text == '0'
        elif how == 'attachment':
            host.attachment_lost()
            assert not host.arq.scan_held and not host.hostmode
        else:
            live.cancel()
            live.close()
        assert not station.arbiter.scan_held and live._scan_token is None
        assert live.scan_epoch == 2 and not station.rig.edges
    finally:
        host.attachment_lost()
        live.close()


def test_scanner_eligibility_cannot_key_recursively(tmp_path):
    live, station, host, _, _ = setup(tmp_path)
    try:
        def unsafe():
            station.arbiter.submit(request(live, 'pactor'))
            pytest.fail('recursive key admitted')
        with pytest.raises(Refused, match='owner I/O poll'):
            station.arbiter.acquire_scan_hold('pactor', eligible=unsafe)
        assert not station.arbiter.scan_held and not station.rig.edges
        assert station.arbiter.acquire_scan_hold('pactor', eligible=lambda: False) is None
        assert not station.arbiter.scan_held
    finally:
        host.attachment_lost()
        live.close()


def test_lane_admission_is_rechecked_after_the_scanner_eligibility_callback(tmp_path):
    live, station, host, _, _ = setup(tmp_path)
    try:
        def changed():
            station.arbiter.active_protocols = lambda: ('ardop',)
            return True
        assert station.arbiter.acquire_scan_hold('pactor', eligible=changed) is None
        assert not station.arbiter.scan_held and not station.rig.edges
    finally:
        host.attachment_lost()
        live.close()


def test_retries_renewal_and_expiry_do_not_reuse_a_released_physical_token(tmp_path):
    live, station, host, _, wire = setup(tmp_path)
    try:
        assert wire.command('%W0').text == '1'
        first, expiry = live._scan_token, host.wait_until
        assert wire.retry().text == '1'
        assert live._scan_token is first and host.wait_until == expiry and live.scan_epoch == 1
        assert wire.command('%W0').text == '1'
        assert live._scan_token is first and live.scan_epoch == 1
        host.wait_until = 0
        host.service_host_output()
        assert not station.arbiter.scan_held and live.scan_epoch == 2
        assert wire.retry().text == '1'
        assert live._scan_token is not first and live.scan_epoch == 3
        with pytest.raises(ValueError, match='token'):
            station.arbiter.release_scan_hold(first)
        assert station.arbiter.scan_held
    finally:
        host.attachment_lost()
        live.close()


def test_real_serial_permission_is_serviced_inside_idle_capture_polling(tmp_path):
    import os
    from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
    from hfmodem.tests.shrike.test_station_handoff import read_available
    from hfmodem.tests.shrike.test_station_io import chunk
    decoder = ptc.Deframer()
    replies = []
    def poll():
        assert endpoint.service()
        replies.extend(decoder.feed(read_available(fd)))
    live, station, host, _, _ = setup(tmp_path)
    host.tick = lambda **kw: pytest.fail('scanner service advanced the ARQ clock')
    endpoint = PtcSerialEndpoint(host)
    fd = os.open(endpoint.open(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        os.write(fd, b'JHOST4\r')
        for _ in range(3):
            endpoint.service()
        read_available(fd)
        live.owner_poll = poll
        live.feed(chunk(0, [1, 2]))
        os.write(fd, ptc.Packet(31, True, b'%W0', counter=1).encode())
        with pytest.raises(ConnectionAbortedError, match='capture retired'):
            live.read(1)
        poll()
        assert replies[-1].text == '1' and station.arbiter.scan_held
        os.write(fd, ptc.Packet(31, True, b'%W1', counter=0).encode())
        poll()
        # The remaining bounded output flush is owner I/O, never a radio tick.
        poll()
        assert replies[-1].code == ptc.MSG and replies[-1].text == ''
        assert not station.arbiter.scan_held
        assert not station.rig.edges and live.scan_epoch == 2
        assert live.pos == 0 and live.capture_retired
        with pytest.raises(ConnectionAbortedError, match='capture retired'):
            live.read_ready()
    finally:
        endpoint.close()
        os.close(fd)
        live.close()


def test_native_acquisition_retires_even_a_hold_released_within_one_poll(tmp_path):
    import os
    from hfmodem.shrike import pactor1
    from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
    from hfmodem.tests.shrike.test_station_handoff import read_available
    from hfmodem.tests.shrike.test_station_io import chunk
    live, station, host, _, _ = setup(tmp_path)
    endpoint = PtcSerialEndpoint(host)
    fd = os.open(endpoint.open(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        os.write(fd, b'JHOST4\r')
        for _ in range(3):
            endpoint.service()
        read_available(fd)
        waveform = pactor1.connect_signal('N0CALL', lead_s=0, tail_s=0)
        live.feed(chunk(0, np.pad(waveform, (0, 12000))))
        os.write(fd, ptc.Packet(31, True, b'%W0', counter=1).encode()
                 + ptc.Packet(31, True, b'%W1', counter=0).encode())
        live.owner_poll = lambda: endpoint.service()
        # Final state is listening again; only the changed epoch prevents
        # pre-retune PCM from becoming the next contact's acquisition evidence.
        finding = onair._await_call(live, 'N0CALL', 1, eligible=lambda: not host.arq.scan_held)
        assert finding is None and live.scan_epoch == 2
        assert host.arq.state == State.LISTENING and not station.arbiter.scan_held
        assert not station.rig.edges and live.pos == 0
    finally:
        endpoint.close()
        os.close(fd)
        live.close()


def test_retune_capture_is_discarded_without_overflow_or_premature_lease_release(tmp_path):
    from hfmodem.shrike.station_io import StationInput
    from hfmodem.tests.shrike.test_station_io import chunk
    live, station, host, _, wire = setup(tmp_path)
    fresh = None
    try:
        assert wire.command('%W0').text == '1'
        token = live._scan_token
        for block in range(600):
            live.feed(chunk(block*4800, np.zeros(4800, np.float32)))
        station.audio.samples = 600*4800
        assert live._frames.empty() and not live._overflow.is_set()
        with pytest.raises(ConnectionAbortedError, match='capture retired'):
            live.read_ready()
        assert not live.closed and station.arbiter.scan_held and live._scan_token is token
        assert wire.command('%W0').text == '1' and live._scan_token is token
        host.wait_until = 0
        host.service_host_output()
        assert not station.arbiter.scan_held and host.arq.state == State.LISTENING
        assert live.capture_retired and live.pos == 0
        fresh = StationInput(live.link)
        fresh.feed(chunk(station.audio.samples, [7, 8]))
        assert fresh.read(2).tolist() == [7, 8]
        assert fresh.pos == station.audio.samples + 2
        assert not station.rig.edges
    finally:
        host.attachment_lost()
        live.close()
        if fresh is not None:
            fresh.close()


def test_native_acquisition_does_not_swallow_an_unrelated_capture_failure():
    from hfmodem.tests.shrike.test_grid import _Bench
    live = _Bench(seconds=1)
    failure = ConnectionAbortedError('capture actually failed')
    def failed(count):
        raise failure
    live.read = failed
    with pytest.raises(ConnectionAbortedError) as caught:
        onair._await_call(live, 'N0CALL', 1, eligible=lambda: True)
    assert caught.value is failure
