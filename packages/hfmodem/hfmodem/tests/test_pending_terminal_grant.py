# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Grant/terminal ordering through production host events; fake TX seam only."""
import pytest

from hfmodem.shrike.arq import ISS, IRS, State
from hfmodem.shrike.ptc import PtcHost, Protocol, pactor1
from hfmodem.shrike.rxfront import Event


class _RecordedSeam:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return record


def _connected(baud=100):
    seam = _RecordedSeam()
    host = PtcHost(peer=seam, mycall='W9SSJ')
    host.protocol = Protocol.PACTOR1
    host.p1_baud = baud
    host.p1_grant_only = True
    host.no_p3_fallback = True
    arq = host.arq
    arq.state, arq.role = State.CONNECTED, ISS
    arq.mycall, arq.dxcall = 'W9SSJ', 'WS8EOC'
    arq.cfg.long_cycle = False
    arq.cfg.repeat_gear = 3
    arq.cfg.traffic_sl = 1
    arq.cfg.max_retries = 8
    arq._next_seq = 1
    arq._outbuf.extend(b'1w9ssj\r')
    arq._buffer_raw = 7
    arq._start_next_packet()
    return host, seam


def _grant(host):
    # spare is a table index, not the literal twelve-bit wire word.
    host.on_rx_event(Event(t=46.11, kind='unassigned', text='0x59A fixture',
                          protocol=Protocol.PACTOR1,
                          spare=pactor1.CS_59A, sense=1))


def _terminate(host, origin):
    if origin == 'operator':
        host.arq.on_host_disconnect()
    else:
        for _ in range(19):
            host.tick()
            if host.arq._qrt_pending:
                break
        assert any('max retries' in line for line in host.log_lines)
    assert host.arq._qrt_pending
    assert host.arq.state == State.CONNECTED


@pytest.mark.parametrize('baud', [100, 200])
def test_active_grant_still_starts_exact_entry(baud):
    host, _ = _connected(baud)
    assert not host.arq.terminal_pending
    _grant(host)
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.entry_pending
    assert host.arq._inflight.status == 0x1a
    assert host.arq._inflight.payload == b''
    assert host.arq._inflight.sl == 1
    assert host.arq._next_seq == 3


@pytest.mark.parametrize('origin', ['operator', 'automatic'])
def test_pending_stop_refuses_grant_before_all_upgrade_effects(origin, monkeypatch):
    host, seam = _connected()
    _terminate(host, origin)
    host._ruled_out.add(Protocol.PACTOR3)
    phase = []
    monkeypatch.setattr(host, '_phase_power', lambda *args: phase.append(args))
    packet = host.arq._inflight
    before = (host.arq._disconnect_ticks, host.arq._next_seq,
              bytes(host.arq._outbuf), len(seam.calls))
    _grant(host)
    assert host.protocol is Protocol.PACTOR1
    assert not host.arq.entry_pending
    assert host.arq._inflight is packet
    assert host.arq._qrt_pending and host.arq.terminal_pending
    assert not host._grant_taken and not host._grant_pending
    assert host._ruled_out == {Protocol.PACTOR3}
    assert phase == []
    assert before == (host.arq._disconnect_ticks, host.arq._next_seq,
                      bytes(host.arq._outbuf), len(seam.calls))
    assert any('termination already pending' in line for line in host.log_lines)


@pytest.mark.parametrize('origin', ['operator', 'automatic'])
def test_repeated_late_grants_do_not_reset_terminal_clock(origin):
    host, _ = _connected()
    _terminate(host, origin)
    for _ in range(3):
        before = host.arq._disconnect_ticks
        _grant(host)
        assert host.arq._disconnect_ticks == before
        host.tick()
        assert host.protocol is Protocol.PACTOR1
        assert not host.arq.entry_pending and not host._grant_pending
        if host.arq._disconnect_ticks is not None:
            assert host.arq._disconnect_ticks > before


def test_ack_after_rejected_grant_sends_goodbye_not_upgrade():
    host, _ = _connected()
    host.arq.on_host_disconnect()
    _grant(host)
    host.arq._on_ack()
    assert host.protocol is Protocol.PACTOR1
    assert not host._grant_pending and not host.arq.entry_pending
    assert host.arq.state == State.DISCONNECTING
    assert host.arq.said_goodbye
    assert host.arq._inflight.status & 0x80


def test_prearmed_grant_cannot_survive_operator_stop_into_final_ack():
    host, _ = _connected()
    host._grant_pending = True
    host.arq.on_host_disconnect()
    host.arq._on_ack()
    assert not host._grant_pending
    assert host.protocol is Protocol.PACTOR1
    assert host.arq._qrt_pending and not host.arq.entry_pending
    assert host.arq.state == State.DISCONNECTING
    assert host.arq.said_goodbye


def test_submitted_goodbye_still_refuses_grant():
    host, seam = _connected()
    host.arq.on_host_disconnect()
    host.arq._on_ack()
    before = len(seam.calls)
    packet = host.arq._inflight
    _grant(host)
    assert host.arq.state == State.DISCONNECTING
    assert host.protocol is Protocol.PACTOR1
    assert host.arq._inflight is packet and len(seam.calls) == before
    assert not host._grant_taken and not host._grant_pending


@pytest.mark.parametrize('flag', ['_qrt_pending', '_rx_close_pending', '_disconnect_ticks'])
def test_terminal_predicate_tracks_only_real_pending_terminal_state(flag):
    host, _ = _connected()
    setattr(host.arq, flag, 0 if flag == '_disconnect_ticks' else True)
    assert host.arq.terminal_pending


def test_healthy_irs_changeover_is_not_terminal():
    host, _ = _connected()
    host.arq.role = IRS
    host.arq._breakin_pending = True
    host.arq._turn_head_wait = True
    host.arq._over_pending = True
    assert not host.arq.terminal_pending


def test_fresh_session_can_upgrade_after_prior_terminal_refusal():
    host, _ = _connected()
    host.arq.on_host_disconnect()
    _grant(host)
    host.arq._finish_disconnected()
    assert not host.arq.terminal_pending
    assert not host._grant_taken and not host._grant_pending
    host.arq.on_host_connect('W9SSJ', 'WS8EOC')
    # This fixture supplies the already established state; connect waveform
    # acquisition is outside this regression's event/terminal ordering scope.
    host.arq.state = State.CONNECTED
    host.arq._next_seq = 1
    host.arq._outbuf.extend(b'1w9ssj\r')
    host.arq._start_next_packet()
    _grant(host)
    assert host.protocol is Protocol.PACTOR3
    assert host.arq.entry_pending and host.arq._inflight.status == 0x1a
