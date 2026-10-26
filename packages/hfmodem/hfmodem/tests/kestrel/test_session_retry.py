# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SESSION selection reaches native requests and the actual owner TX clock."""
import socket

import pytest

from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.host.server import HostSession
from hfmodem.tests.kestrel.test_compression import IO
from hfmodem.tests.kestrel.test_vara_host_modem import Observer, command_line


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Audio(IO):
    station_ptt = True
    receiving = False

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.hs = None
        self.edges = []
        self.bursts = []
        self.attempts = 0
        self.refused = False

    def bind_station(self, hs):
        self.hs = hs

    def key(self, on):
        if not on:
            # Owner bookkeeping after the actual unkey must not extend a
            # timer whose anchor is the arbiter's physical-key notification.
            self.clock.now += .2

    def tx(self, samples):
        self.attempts += 1
        if self.refused:
            return
        self.edges.append((True, self.clock.now))
        self.hs.io.report_ptt(True)
        self.clock.now += .04 + len(samples) / 48000
        self.edges.append((False, self.clock.now))
        self.hs.io.report_ptt(False)
        self.bursts.append(samples)

    def tx_went_out(self):
        return not self.refused

    def next_rx_burst(self, timeout, hs=None):
        return None


def runtime(*, mode=None, bandwidth='500'):
    clock = Clock()
    audio = Audio(clock)
    owner = AudioHostRuntime(audio, clock=clock)
    modem = owner.modem_factory()
    modem.start(Observer())
    modem.set_bandwidth(bandwidth)
    if mode:
        modem.set_session_mode(mode)
    modem.connect('W9SSJ', 'WI9YZA')
    return owner, modem, audio, clock


@pytest.mark.parametrize('mode,quiet', [('WINLINK', 1.30), ('P2P', 1.85)])
@pytest.mark.parametrize('bandwidth', ['500', '2300', '2750'])
def test_selected_profile_waits_from_actual_unkey_then_uses_native_short_retry(mode, quiet, bandwidth):
    owner, modem, audio, clock = runtime(mode=mode, bandwidth=bandwidth)
    owner.step()
    assert len(audio.bursts) == 1
    actual_end = audio.edges[-1][1]
    assert clock.now > actual_end
    assert modem.last_unkey_at == actual_end
    assert owner._retry_at == pytest.approx(actual_end + quiet)
    modem.set_session_mode(mode)
    owner.step()
    assert owner._retry_at == pytest.approx(actual_end + quiet)
    clock.now = actual_end + quiet - .001
    owner.step()
    assert len(audio.bursts) == 1
    clock.now = actual_end + quiet
    owner.step()
    assert len(audio.bursts) == 2
    assert audio.edges[2][1] - audio.edges[1][1] == pytest.approx(quiet)
    assert len(audio.bursts[1]) < len(audio.bursts[0])
    assert owner._retry_at == pytest.approx(audio.edges[-1][1] + quiet)


def test_unselected_bench_keeps_its_existing_retry_policy():
    owner, modem, audio, clock = runtime()
    owner.step()
    assert modem.connect_retry_quiet_seconds is None
    assert owner._retry_at == pytest.approx(clock.now + 8)
    clock.now = owner._retry_at
    owner.step()
    assert len(audio.bursts) == 2
    assert len(audio.bursts[1]) == len(audio.bursts[0])
    assert owner._retry_at == pytest.approx(clock.now + 8)


def test_receiving_and_missed_deadlines_do_not_create_a_retry_burst_train():
    owner, _, audio, clock = runtime(mode='P2P')
    owner.step()
    clock.now = owner._retry_at + 5
    audio.receiving = True
    owner.step()
    assert len(audio.bursts) == 1
    audio.receiving = False
    owner.step()
    assert len(audio.bursts) == 2
    owner.step()
    assert len(audio.bursts) == 2
    assert owner._retry_at > clock.now


def test_refused_output_retains_a_retry_delay_without_fake_ptt():
    owner, modem, audio, clock = runtime(mode='WINLINK')
    audio.refused = True
    owner.step()
    assert audio.attempts == 1 and not audio.edges
    assert modem.last_unkey_at is None
    owner.step()
    assert audio.attempts == 1
    clock.now = owner._retry_at
    owner.step()
    assert audio.attempts == 2 and not audio.edges
    assert owner._retry_at > clock.now


def test_host_acknowledges_only_admitted_selection_and_preserves_it_on_refusal():
    clock = Clock()
    owner = AudioHostRuntime(Audio(clock), clock=clock)
    modem = owner.modem_factory()
    command, client = socket.socketpair()
    data, appdata = socket.socketpair()
    client.settimeout(1)
    try:
        session = HostSession(command, data, modem)
        modem.start(session)
        session._dispatch('WINLINK SESSION')
        assert client.recv(32) == b'OK\r'
        assert session.session_mode == 'WINLINK'
        assert modem.connect_retry_quiet_seconds is None  # owner has not applied it
        modem.poll()
        assert command_line(client) == b'REGISTERED HFMODEM\r'
        assert modem.connect_retry_quiet_seconds == 1.30
        modem.connect('W9SSJ', 'WI9YZA')
        session._dispatch('P2P SESSION')
        assert client.recv(32) == b'WRONG\r'
        assert session.session_mode == 'WINLINK'
        assert modem.connect_retry_quiet_seconds == 1.30
    finally:
        session.close()
        modem.poll()
        session.ensure_retired()
        client.close()
        appdata.close()


def test_invalid_or_full_owner_queue_cannot_commit_timing_selection():
    clock = Clock()
    owner = AudioHostRuntime(Audio(clock), clock=clock)
    modem = owner.modem_factory()
    with pytest.raises(ValueError):
        modem.set_session_mode('invalid')
    for _ in range(64):
        modem.set_mycall(['W9SSJ'])
    with pytest.raises(ConnectionAbortedError, match='queue full'):
        modem.set_session_mode('P2P')
    assert modem._wanted_session_mode is None
    assert modem.connect_retry_quiet_seconds is None


@pytest.mark.parametrize('mode', [None, 'WINLINK', 'P2P'])
def test_long_setup_from_receive_gets_full_post_unkey_retry_window(mode):
    from hfmodem.kestrel.vara import vara_frames as F
    owner, modem, audio, clock = runtime(mode=mode)
    owner.step()
    old_deadline = owner._retry_at
    # The setup is keyed synchronously while processing a received response.
    def receive_response(timeout, hs=None):
        audio.next_rx_burst = lambda **kw: None
        clock.now = old_deadline - 1
        kind = F.connect_response('500', 1)
        hs.on_rx_tones(F.handshake_tones('WI9YZA', kind), F.connect_response('500'))
        return None
    audio.next_rx_burst = receive_response
    owner.step()
    assert modem.station.connect_setup_pending
    assert len(audio.bursts) == 2  # initial request and one long setup
    first_setup_end = audio.edges[-1][1]
    assert first_setup_end > old_deadline
    assert owner._retry_at == pytest.approx(first_setup_end + 8)
    clock.now = first_setup_end + 7.999
    owner.step()
    assert len(audio.bursts) == 2
    clock.now = first_setup_end + 8
    owner.step()
    assert len(audio.bursts) == 3
    assert audio.edges[-2][1] == pytest.approx(first_setup_end + 8)
    assert owner._retry_at == pytest.approx(audio.edges[-1][1] + 8)
