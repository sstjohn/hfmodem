# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Independent PTC frames received inside real ARQ operations; no devices."""
import threading

import pytest

from hfhost import ptc
from hfmodem.shrike.arq import ISS, State
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.shrike.test_ptc_compatibility import Client


def connected(*, limit=64, crc=True):
    host = PtcHost(mycall='N0CALL')
    owner = host.enable_owner(limit=limit)
    client = Client(host) if crc else None
    host._store('MYLEVEL', '1')
    host.arq.state, host.arq.role = State.CONNECTED, ISS
    host.arq.mycall, host.arq.dxcall = 'N0CALL', 'N0DX'
    host._link_established('N0DX')
    if client:
        assert client.cmd('G').code == ptc.LINK
    return host, owner, client


@pytest.mark.parametrize('forwarded', [False, True])
def test_disconnect_waits_for_outer_operation_and_does_not_age_prior_slots(forwarded):
    host, owner, client = connected()
    assert client.exchange(b'original bytes still in flight', False).code == ptc.OK
    offered = []
    def packet(*args, **kwargs):
        inflight = host.arq._inflight
        offered.append(inflight)
        assert client.cmd('#D' if forwarded else 'D').code == ptc.OK
        assert client.retry().code == ptc.OK
        assert owner.pending_count == 1 and not owner.admitting_data
        assert not host.arq._qrt_pending and host.arq._inflight is inflight
        # A nested RF notification must not drain the outer operation's queue.
        host.arq.on_rx_ack_emitted()
        assert not host.arq._qrt_pending and owner.pending_count == 1
        assert client.cmd('L').data.split()[-1] == b'4'
        accepted = host.arq.payload_ledger.accepted
        assert client.exchange(b'late', False).code == ptc.FAIL
        assert host.arq.payload_ledger.accepted == accepted
    host.send_packet = packet
    host.arq.on_cycle(elapsed_ticks=7)
    assert len(offered) == 1 and host.arq._inflight is offered[0]
    assert host.arq._qrt_pending and host.arq._disconnect_ticks == 0
    assert not owner.in_operation and owner.pending_count == 0


def test_second_disconnect_after_polls_supersedes_graceful_work_without_credit():
    host, owner, client = connected()
    original = b'a\x1c\x1eb' * 8
    assert client.exchange(original, False).code == ptc.OK
    def packet(*args, **kwargs):
        inflight = host.arq._inflight
        assert client.cmd('D').code == ptc.OK
        assert client.cmd('L').data.split()[-1] == b'4'
        assert client.cmd('D').code == ptc.OK
        assert client.retry().code == ptc.OK
        assert owner.abort_requested and owner.pending_count == 1
        assert host.arq.state == State.CONNECTED and host.arq._inflight is inflight
        assert not host.arq._qrt_pending
    host.send_packet = packet
    host.arq.on_cycle()
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    retained = host.arq.transfer_unconfirmed
    assert retained['host_accepted'] == len(original)
    assert retained['host_unconfirmed'] == original
    assert retained['host_confirmed'] == 0 and host.sent_total == 0
    assert not owner.pending_count and not owner.in_operation


def test_bounded_control_admission_keeps_abort_available():
    host, owner, client = connected(limit=2)
    assert client.exchange(b'payload', False).code == ptc.OK
    def packet(*args, **kwargs):
        assert client.cmd('%Q').code == ptc.OK
        assert client.cmd('%I').code == ptc.OK
        assert client.cmd('%O').code == ptc.FAIL
        assert client.retry().code == ptc.FAIL
        assert owner.pending_count == 2
        assert not host.arq._over_pending and not host.arq._breakin_pending
        assert client.cmd('DD').code == ptc.OK
        assert owner.pending_count == 1 and owner.abort_requested
    host.send_packet = packet
    host.arq.on_cycle()
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)


def test_terminal_disconnect_uses_the_same_boundary():
    host, owner, _ = connected(crc=False)
    assert host.arq.on_host_data(b'terminal contact')
    def packet(*args, **kwargs):
        assert b'ERROR' not in host.feed(b'D\r')
        assert owner.pending_count == 1 and not host.arq._qrt_pending
    host.send_packet = packet
    host.arq.on_cycle()
    assert host.arq._qrt_pending and not owner.pending_count


def test_attachment_loss_is_retired_after_the_engine_stack_returns():
    host, owner, client = connected()
    original = b'not yet confirmed'
    client.exchange(original, False)
    def packet(*args, **kwargs):
        inflight = host.arq._inflight
        host.attachment_lost()
        assert owner.retired and owner.abort_requested
        assert host.arq.state == State.CONNECTED and host.arq._inflight is inflight
    host.send_packet = packet
    host.arq.on_cycle()
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    assert not host._channels and not host.hostmode
    assert host.arq.transfer_unconfirmed['host_unconfirmed'] == original
    host.attachment_lost()  # retirement is idempotent
    with pytest.raises(ConnectionAbortedError):
        host.feed(b'C N0NEXT\r')
    with pytest.raises(ConnectionAbortedError):
        host.arq.on_cycle()


def test_failed_operation_aborts_deferred_close_without_masking_the_failure():
    host, owner, client = connected()
    client.exchange(b'unconfirmed', False)
    failure = RuntimeError('synthetic radio failure')
    def packet(*args, **kwargs):
        assert client.cmd('D').code == ptc.OK
        raise failure
    host.send_packet = packet
    with pytest.raises(RuntimeError) as caught:
        host.arq.on_cycle()
    assert caught.value is failure and owner.cleanup_error is None
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    assert not owner.pending_count and not owner.in_operation
    assert host.arq.transfer_unconfirmed['host_confirmed'] == 0


def test_refused_connect_does_not_abort_the_current_contact():
    host, owner, client = connected()
    client.exchange(b'held', False)
    before = (host.arq.state, host.arq.mycall, host.arq.dxcall, host.arq.last_connect)
    with pytest.raises(ValueError):
        host.arq.on_host_connect('N0CALL', 'N0NEXT')
    assert before == (host.arq.state, host.arq.mycall, host.arq.dxcall, host.arq.last_connect)
    assert host.arq.payload_ledger.pending == 4 and not owner.retired


def test_busy_hostmode_reentry_refuses_until_safe_then_resets_without_leaks():
    host, owner, client = connected()
    client.exchange(b'old attachment', False)
    def packet(*args, **kwargs):
        result = client.cmd('#JHOST4')
        assert result.code == ptc.FAIL and 'ARQ OPERATION' in result.text
        assert host.hostmode and owner.abort_requested
        assert host.arq.state == State.CONNECTED
    host.send_packet = packet
    host.arq.on_cycle()
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    assert client.cmd('#JHOST4').code == ptc.OK
    assert client.cmd('G').code == ptc.OK
    assert not owner.retired and not owner.pending_count
    assert client.cmd('C N0NEXT').code == ptc.OK


def test_owner_is_confined_to_the_actual_thread_object():
    host, owner, client = connected()
    before = (host.arq.state, host.arq.dxcall, host.arq._buffer_raw)
    failures = []
    def intruder():
        for action in [lambda: host.feed(b'D\r'), host.tick, host.open, host.attachment_lost]:
            try:
                action()
            except RuntimeError:
                failures.append(True)
    thread = threading.Thread(target=intruder)
    thread.start()
    thread.join(1)
    assert not thread.is_alive() and len(failures) == 4
    assert before == (host.arq.state, host.arq.dxcall, host.arq._buffer_raw)
    assert not owner.retired


@pytest.mark.parametrize('limit', [0, 65, True])
def test_invalid_owner_limit_does_not_install_partial_hooks(limit):
    host = PtcHost()
    with pytest.raises(ValueError):
        host.enable_owner(limit=limit)
    assert host._owner is None and host.rf_scope is None


def test_owner_cannot_be_installed_on_an_active_contact():
    host = PtcHost()
    host.arq.state = State.CONNECTED
    with pytest.raises(ValueError):
        host.enable_owner()
    assert host._owner is None and host.rf_scope is None


def test_reentrant_data_is_fifo_and_only_peer_settlement_confirms_it():
    class Peer(SimPeer):
        callback = None
        def send_packet(self, *args, **kwargs):
            if self.callback is not None:
                action, self.callback = self.callback, None
                action()
            return super().send_packet(*args, **kwargs)
    peer = Peer()
    host = PtcHost(peer, mycall='N0CALL')
    owner = host.enable_owner()
    client = Client(host)
    assert client.cmd('#MYL 1').code == ptc.OK
    assert client.cmd('C N0DX').code == ptc.OK
    for _ in range(12):
        host.tick()
        if host.arq.state == State.CONNECTED:
            break
    assert host.arq.state == State.CONNECTED
    first, second = b'first\x1c\x1e'*8, b'second\xaa\x00'*8
    assert client.exchange(first, False).code == ptc.OK
    def append():
        inflight = host.arq._inflight
        assert client.exchange(second, False).code == ptc.OK
        assert client.retry().code == ptc.OK
        assert host.arq._inflight is inflight
        assert host.arq.payload_ledger.confirmed == 0
    peer.callback = append
    for _ in range(100):
        host.tick()
        if host.arq.payload_ledger.confirmed == len(first + second):
            break
    assert peer.received.endswith(first + second)
    assert host.arq.payload_ledger.accepted == host.arq.payload_ledger.confirmed == len(first + second)
    assert not owner.pending_count


def test_logical_disconnect_does_not_release_the_native_contact_lease():
    host, owner, client = connected()
    with owner.contact():
        assert owner.contact_active
        host.arq.on_host_abort()
        assert client.cmd('G').code == ptc.LINK
        assert host.contact_ready()
        assert client.cmd('C N0NEXT').code == ptc.FAIL
        assert 'OWNER BUSY' in client.retry().text
        assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
        assert not owner.connect_pending
    assert not owner.contact_active
    assert client.cmd('C N0NEXT').code == ptc.OK


def test_native_driver_scope_is_not_reentrant_and_retires_after_failure():
    host, owner, _ = connected()
    with pytest.raises(OSError, match='contact failure'):
        with owner.contact():
            with owner.driver():
                with pytest.raises(RuntimeError, match='already active'):
                    with owner.driver():
                        pytest.fail('recursive driver was admitted')
                assert owner.contact_active
                raise OSError('contact failure')
    assert not owner.contact_active and not owner._driver_active


def test_driver_retirement_preserves_original_payload_and_terminal_output():
    host, owner, client = connected()
    original = b'contact exited before peer confirmation\x00\x1c\xaa'
    client.exchange(original, False)
    with owner.driver():
        assert host.arq.state == State.CONNECTED
    assert host.arq.state in (State.DISCONNECTED, State.LISTENING)
    assert host.arq.transfer_unconfirmed['host_unconfirmed'] == original
    assert host.arq.transfer_unconfirmed['host_confirmed'] == 0
    assert not owner.retired and not owner.contact_active
    terminal = client.cmd('G')
    assert terminal.code == ptc.LINK and 'transfer unconfirmed' in terminal.text
    assert not owner.pending_count


def test_driver_cleanup_failure_cannot_replace_the_original_failure():
    host, owner, _ = connected()
    failure = RuntimeError('original native stream failure')
    def failed_abort():
        raise OSError('failed retirement')
    host.arq.on_host_abort = failed_abort
    with pytest.raises(RuntimeError) as caught:
        with owner.driver():
            raise failure
    assert caught.value is failure
    assert owner.retired and owner.cleanup_error == 'OSError'
    assert not owner.contact_active and not owner._driver_active
    with pytest.raises(ConnectionAbortedError):
        host.feed(b'C N0NEXT\r')


def test_native_wrapper_reports_payload_left_unconfirmed_at_final_retirement():
    from hfmodem.shrike import onair
    host, owner, client = connected()
    original = b'final native return still owns these bytes'
    client.exchange(original, False)
    @onair._contact_driver
    def finite_contact(args, *, host):
        assert owner.contact_active and host.arq.state == State.CONNECTED
        return 0
    assert finite_contact(None, host=host) == 1
    assert host.arq.transfer_unconfirmed['host_unconfirmed'] == original
    assert not owner.contact_active


def test_serial_reopen_preserves_controller_settings_and_history_but_not_crc_success():
    host, previous, client = connected(limit=2)
    assert client.cmd('%M1').code == ptc.OK
    original = b'old attachment unconfirmed\x00\x1c\xaa'
    assert client.exchange(original, False).code == ptc.OK
    previous_frame = client.frame
    settings = dict(host.settings)
    host.attachment_lost()
    retained = host.arq.transfer_unconfirmed
    assert previous.retired and previous._retirement_done
    current = host.reopen_owner()
    assert current is not previous and current.limit == previous.limit == 2
    assert previous.retired and host.rf_scope.__self__ is current
    assert host.arq.transfer_unconfirmed is retained
    assert retained['host_unconfirmed'] == original and retained['host_confirmed'] == 0
    assert host.settings == dict(settings, **{'%M': '0'})
    assert not host.hostmode and not any(c.pending() for c in host._channels.values())
    assert host.open()
    # The old data transaction earned OK in the old attachment; the new
    # disconnected attachment must execute/refuse it, never replay that OK.
    fresh = Client(host)
    result = ptc.Deframer().feed(host.feed(previous_frame))
    assert len(result) == 1 and result[0].code == ptc.FAIL
    assert host.sent_total == 0 and not current.pending_count
    with pytest.raises(ConnectionAbortedError):
        previous.request('connect', 'N0CALL', 'N0OLD')
    assert fresh.cmd('C N0NEXT').code == ptc.OK


def test_reopen_cannot_replace_an_owner_inside_a_retiring_radio_stack():
    host, previous, client = connected()
    client.exchange(b'waiting', False)
    def packet(*args, **kwargs):
        host.attachment_lost()
        assert previous.retired and not previous._retirement_done
        with pytest.raises(ValueError, match='fully retired'):
            host.reopen_owner()
        assert host._owner is previous
    host.send_packet = packet
    host.arq.on_cycle()
    assert previous._retirement_done
    assert host.reopen_owner() is not previous


def test_reopen_waits_for_the_native_contact_lease_and_stays_on_its_thread():
    host, previous, _ = connected()
    with previous.contact():
        host.attachment_lost()
        assert previous._retirement_done
        with pytest.raises(ValueError, match='fully retired'):
            host.reopen_owner()
    failures = []
    def wrong_thread():
        try:
            host.reopen_owner()
        except RuntimeError:
            failures.append(True)
    other = threading.Thread(target=wrong_thread)
    other.start()
    other.join(1)
    assert not other.is_alive() and failures == [True] and host._owner is previous
    assert host.reopen_owner() is not previous


def test_reopen_requires_a_retired_owner_without_changing_live_hooks():
    host, owner, _ = connected()
    with pytest.raises(ValueError, match='fully retired'):
        host.reopen_owner()
    assert host._owner is owner and host.rf_scope.__self__ is owner
    with pytest.raises(ValueError, match='no owner'):
        PtcHost().reopen_owner()


def test_reopen_requires_positive_retirement_of_the_previous_capture_port():
    from types import SimpleNamespace
    from hfmodem.tests.shrike.test_station_io import port
    host, previous, _ = connected()
    host.attachment_lost()
    live, _ = port()
    host.peer = SimpleNamespace(live=live)
    with pytest.raises(ValueError, match='capture port has not retired'):
        host.reopen_owner()
    assert host._owner is previous
    live.close()
    assert host.reopen_owner() is not previous


def test_real_pty_hangup_and_fresh_endpoint_cannot_replay_an_old_data_ack(tmp_path):
    import os
    from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
    from hfmodem.tests.shrike.test_station_handoff import read_available
    host = PtcHost(mycall='N0CALL')
    previous = host.enable_owner()
    alias = tmp_path / 'ptc'
    first, second, fd1, fd2 = PtcSerialEndpoint(host, endpoint=alias), None, None, None
    original = b'old PTY bytes\x00\x1c\xaa'
    frame = ptc.Packet(31, False, original, counter=1).encode()
    try:
        fd1 = os.open(first.open(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        os.write(fd1, b'PTC 31\rMYCALL N0PERSIST\rJHOST4\r')
        for _ in range(3):
            first.service()
        read_available(fd1)
        host.arq.state, host.arq.role = State.CONNECTED, ISS
        host.arq.mycall, host.arq.dxcall = host.mycall, 'N0DX'
        host._link_established('N0DX')
        os.write(fd1, frame)
        first.service()
        first.service()
        assert ptc.Deframer().feed(read_available(fd1))[0].code == ptc.OK
        os.close(fd1)
        fd1 = None
        assert not first.service() and first.closed and previous.retired
        assert not alias.is_symlink()
        assert host.arq.transfer_unconfirmed['host_unconfirmed'] == original
        current = host.reopen_owner()
        second = PtcSerialEndpoint(host, endpoint=alias)
        fd2 = os.open(second.open(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        os.write(fd2, b'JHOST4\r')
        for _ in range(3):
            second.service()
        read_available(fd2)
        assert host.mycall == 'N0PERSIST' and host.ptchn == 31
        # Even an idempotent late close of the old endpoint cannot retire this
        # attachment or reset its newly initialized parser.
        first.close()
        assert not current.retired and host.hostmode and alias.is_symlink()
        assert first.name == second.name == str(alias)
        os.write(fd2, frame)
        second.service()
        second.service()
        replies = ptc.Deframer().feed(read_available(fd2))
        assert len(replies) == 1 and replies[0].code == ptc.FAIL
        assert host.arq.transfer_unconfirmed['host_unconfirmed'] == original
        assert host.sent_total == 0 and len(host.arq.unconfirmed_transfers) == 1
    finally:
        first.close()
        if second is not None:
            second.close()
        if fd1 is not None:
            os.close(fd1)
        if fd2 is not None:
            os.close(fd2)
