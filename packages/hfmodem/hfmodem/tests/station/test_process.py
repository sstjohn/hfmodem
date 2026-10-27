# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core import claim, config, hostio
from hfmodem.core.audio import AudioError, StationAudio
from hfmodem.core.operations import LeaseTimeout
from hfmodem.station import hosts, process


def configuration(*, transmit=True, cat='none', protocols=None):
    return config.StationConfig(
        station=config.Station(mycall='K1ABC', transmit=transmit, id_mode='voice',
                               control='local', regulatory='unregulated',
                               regulatory_settings={'because': 'fake devices'}),
        rig=config.Rig(cat=cat), audio=config.Audio(input='fake-in', output='fake-out'),
        protocols=protocols or {})


@pytest.fixture
def fake_claim(monkeypatch):
    monkeypatch.setattr(claim, 'process_started', lambda pid: 'fake-start')
    monkeypatch.setattr(claim, '_command', lambda pid: 'fake-station')
    monkeypatch.setattr(claim, 'check_devices', lambda *a, **kw: [])


class FakeLine:
    released_low = None

    def release(self):
        self.released_low = True


class FakeRig:
    retired = False
    retired_why = ''

    def __init__(self, calls, receive=False):
        self.calls = calls
        self.ptt = None if receive else FakeLine()
        self.released_low = None

    def close(self):
        self.calls.append('rig-close')
        if self.ptt:
            self.ptt.release()
            self.released_low = self.ptt.released_low
        return self.released_low


class FakeAudio:
    def __init__(self, calls):
        self.calls = calls

    def breathe(self):
        pass

    def close(self):
        self.calls.append('audio-close')


@pytest.fixture
def station(tmp_path, monkeypatch, fake_claim):
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))
    path = tmp_path / 'station.toml'
    path.write_text('[station]\ntransmit = false\n')
    st = process.Station(configuration(), station_file=path)
    st.calls = []

    def audio():
        st.calls.append('audio')
        st.audio = FakeAudio(st.calls)

    def rig(**kw):
        st.calls.append(('rig', kw['prove_ptt']))
        st.rig = FakeRig(st.calls, st.receive)

    def lanes():
        st.calls.append('lanes')
        st.lanes['fake'] = NS(stop=lambda: st.calls.append('lane-stop'))

    monkeypatch.setattr(st, 'preflight_reference', lambda: st.calls.append('reference') or 1)
    monkeypatch.setattr(st, 'open_audio', audio)
    monkeypatch.setattr(st, 'open_rig', rig)
    monkeypatch.setattr(st, 'preflight_measure', lambda ref: st.calls.append('measure'))
    monkeypatch.setattr(st, 'build_arbiter', lambda: None)
    monkeypatch.setattr(st, 'build_lanes', lanes)
    monkeypatch.setattr(st, '_tick', lambda: st._poll_stop.wait())
    monkeypatch.setattr(st, 'fatal_shutdown', lambda why: st.calls.append(('fatal', why)))
    yield st
    st._poll_stop.set()
    if st._poll_thread:
        st._poll_thread.join(2)
    if st.claim is not None:
        claim.close(st.claim)
        st.claim = None


@pytest.mark.parametrize('receive', [False, True])
def test_own_release_reown(station, receive):
    st = station
    expected = 'owning-receive' if receive else 'owning'
    assert st.own(receive=receive)['state'] == expected
    audio, rig, poll = st.audio, st.rig, st._poll_thread
    assert claim.claim_path().read_text()
    assert ('reference' in st.calls) is not receive
    assert ('measure' in st.calls) is not receive
    assert st.release()['state'] == 'observing'
    assert not poll.is_alive()
    assert not claim.claim_path().read_text()
    assert st.calls[-3:] == ['lane-stop', 'rig-close', 'audio-close']
    st.own(receive=receive)
    assert st.audio is not audio and st.rig is not rig
    st.release()


@pytest.mark.parametrize('step', ['preflight_reference', 'open_audio', 'open_rig',
                                  'preflight_measure', 'build_arbiter', 'build_lanes'])
def test_own_rolls_back_every_failed_step(station, monkeypatch, step):
    def fail(*a, **kw):
        raise RuntimeError(step)
    monkeypatch.setattr(station, step, fail)
    with pytest.raises(RuntimeError, match=step):
        station.own()
    assert station.state == 'observing'
    assert station.claim is None
    assert claim.claim_path().read_text() == ''
    acquired = claim.acquire('station', 'successor')
    claim.release(acquired)


def test_receive_rollback(station, monkeypatch):
    monkeypatch.setattr(station, 'build_lanes', lambda: (_ for _ in ()).throw(RuntimeError('lane')))
    with pytest.raises(RuntimeError, match='lane'):
        station.own(receive=True)
    assert 'reference' not in station.calls and 'measure' not in station.calls
    assert station.state == 'observing' and station.claim is None


def test_external_down_during_own(station, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    errors = []

    def reference():
        entered.set()
        assert finish.wait(3)
        raise RuntimeError('failed reference')

    def own():
        try:
            station.own()
        except RuntimeError as exc:
            errors.append(str(exc))

    monkeypatch.setattr(station, 'preflight_reference', reference)
    thread = threading.Thread(target=own)
    thread.start()
    try:
        assert entered.wait(2)
        cleanup = claim.mutex(timeout=0)
        with pytest.raises(RuntimeError, match='transitioning'):
            station.release()
        finish.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors == ['failed reference']
        assert station.release() == {'state': 'observing'}
        acquired = claim.acquire('station', 'external-down', cleanup=cleanup)
        claim.release(acquired)
        claim.mutex_release(cleanup)
    finally:
        finish.set()
        thread.join(3)


@pytest.mark.parametrize('receive', [False, True])
def test_inode_recheck_releases_with_reason(station, receive):
    station.own(receive=receive)
    # Use an exact synthetic epoch: (real_monotonic + 60) - real_monotonic
    # can round below 60 and leave the periodic check not yet due.
    station._last_inode_check = 0.0
    old = claim.claim_path().with_name('old-claim')
    claim.claim_path().rename(old)
    claim.claim_path().touch()
    station.check_claim(now=station._last_inode_check + 60)
    assert station.state == 'observing'
    assert 'descriptor does not name' in station.reason
    assert station.claim is None
    assert 'rig-close' in station.calls


def test_observing_waits_for_threads_and_line_low(station, monkeypatch):
    station.own()
    entered, finish = threading.Event(), threading.Event()
    line = station.rig.ptt

    def stop():
        entered.set()
        assert finish.wait(3)

    station.lanes['fake'] = NS(stop=stop)
    thread = threading.Thread(target=station.release)
    thread.start()
    try:
        assert entered.wait(2)
        assert station.state == 'releasing' and station.claim is not None
        assert line.released_low is None
        finish.set()
        thread.join(3)
        assert not thread.is_alive()
        assert line.released_low is True
        assert station.state == 'observing' and station.claim is None
    finally:
        finish.set()
        thread.join(3)


def test_unconfirmed_line_keeps_claim(station, monkeypatch):
    station.own()
    line = station.rig.ptt
    monkeypatch.setattr(line, 'release', lambda: None)
    with pytest.raises(process.RigError, match='unconfirmed'):
        station.release()
    assert station.state == 'failed'
    assert station.claim is not None
    assert 'unconfirmed' in station.reason


@pytest.mark.parametrize('part', ['host', 'lane', 'audio'])
def test_failed_fence_never_releases_claim(station, monkeypatch, part):
    station.own()
    if part == 'host':
        station.hosts['blocked'] = hosts.Served('blocked', NS(stop=lambda timeout: False), (), False)
    else:
        def fail():
            raise LeaseTimeout(part)
        if part == 'lane':
            station.lanes['fake'] = NS(stop=fail)
        else:
            monkeypatch.setattr(station.audio, 'close', fail)
    with pytest.raises(LeaseTimeout):
        station.release()
    assert station.state == 'failed' and station.claim is not None
    assert any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)


def test_stop_all_bounds_a_host_that_never_returns():
    entered, finish = threading.Event(), threading.Event()
    exited = threading.Event()

    def stop(timeout):
        entered.set()
        finish.wait()
        exited.set()
        return True

    served = {'stuck': hosts.Served('stuck', NS(stop=stop), (), False)}
    try:
        with pytest.raises(LeaseTimeout, match='stuck'):
            hosts.stop_all(served, timeout=0.05)
        assert entered.is_set() and not exited.is_set()
    finally:
        finish.set()
        assert exited.wait(2)


def test_stop_all_starts_each_host_independently():
    first, second = threading.Event(), threading.Event()

    def stop_first(timeout):
        first.set()
        return second.wait(timeout)

    def stop_second(timeout):
        second.set()
        return first.wait(timeout)

    hosts.stop_all({name: hosts.Served(name, NS(stop=fn), (), False)
                    for name, fn in [('first', stop_first), ('second', stop_second)]})


class StalledSocket:
    def __init__(self):
        self.writing = threading.Event()
        self.cancelled = threading.Event()

    def settimeout(self, value):
        pass

    def setsockopt(self, *args):
        pass

    def send(self, data):
        self.writing.set()
        self.cancelled.wait()
        raise OSError('closed')

    def shutdown(self, how):
        self.cancelled.set()

    def close(self):
        self.cancelled.set()


def test_stop_all_joins_connection_of_nonreading_client():
    sock = StalledSocket()
    conn = hostio.Connection(sock)
    conn.sendall(b'a' * hostio.MAX_OUTPUT)
    assert sock.writing.wait(2)

    def stop(timeout):
        conn.cancel()
        return conn.join(timeout)

    try:
        hosts.stop_all({'client': hosts.Served('client', NS(stop=stop), (), False)})
        assert not conn._writer.is_alive() and not conn._delivery_thread.is_alive()
    finally:
        conn.cancel()
        assert conn.join(2)


def test_send_later_has_no_item_limit(monkeypatch):
    sock = StalledSocket()
    conn = hostio.Connection(sock)
    try:
        conn.sendall(b'a' * hostio.MAX_OUTPUT)
        assert sock.writing.wait(2)
        for _ in range(4096):
            conn.send_later(b'x')
        assert not conn.closed
        assert conn._delivery_bytes > hostio.MAX_PENDING
    finally:
        conn.cancel()
        assert conn.join(2)


def test_send_later_caps_large_payload():
    conn = hostio.Connection(StalledSocket())
    try:
        conn.send_later(b'x' * (hostio.MAX_DELIVERY + 1))
        assert conn.closed
    finally:
        conn.cancel()
        assert conn.join(2)


def test_sendall_stall_deadline(monkeypatch):
    conn = hostio.Connection(StalledSocket())
    conn.sendall(b'x' * hostio.MAX_OUTPUT)
    assert conn.sock.writing.wait(2)
    monkeypatch.setattr(hostio, 'STALL_S', 0)
    try:
        conn.sendall(b'blocked')
        assert conn.closed
    finally:
        conn.cancel()
        assert conn.join(2)


@pytest.mark.parametrize('capture', [False, True])
def test_station_audio_stream_selection(monkeypatch, capture):
    import sounddevice
    opened = []

    class Stream:
        def __init__(self, **kw):
            self.kw = kw
            self.latency = 0.01 if capture else (0.01, 0.02)
            self.active = True
            opened.append(self)

        def start(self):
            for i in range(64):
                data = np.zeros((128, 1), dtype=np.float32)
                stamp = NS(inputBufferAdcTime=1 + i * 128 / 48000)
                if capture:
                    self.kw['callback'](data, 128, stamp, None)
                else:
                    stamp.outputBufferDacTime = stamp.inputBufferAdcTime + 0.03
                    self.kw['callback'](data, data.copy(), 128, stamp, None)

        def stop(self):
            self.active = False

        def close(self):
            self.closed = True

    def forbidden(**kw):
        pytest.fail('wrong stream opened')

    monkeypatch.setattr(sounddevice, 'InputStream', Stream if capture else forbidden)
    monkeypatch.setattr(sounddevice, 'Stream', forbidden if capture else Stream)
    audio = StationAudio(input_device='fake-in', output_device=None if capture else 'fake-out')
    audio.open()
    assert audio.failure is None and audio.samples == 8192
    assert opened[0].kw['device'] == ('fake-in' if capture else ('fake-in', 'fake-out'))
    if capture:
        assert audio.lat is None and audio.loop_lat is None and audio._dac is None
        for fn, args in [(audio.arm_burst, (np.zeros(8), 0)),
                         (audio.reserve_burst, (np.zeros(8), 0, 99)),
                         (audio.arm_stream, (0, 8))]:
            with pytest.raises(AudioError, match='capture-only'):
                fn(*args)
    else:
        assert audio.lat == 1440 and audio.loop_lat == 1440
        audio.arm_burst(np.zeros(8), 0)
    audio.close()
    assert opened[0].closed


@pytest.mark.parametrize('cat', ['none', 'yaesu', 'rigctld'])
def test_receive_open_rig_never_opens_line_or_arms(monkeypatch, cat):
    calls = []

    class Driver:
        def __init__(self, *a, **kw):
            calls.append('driver')
        def open(self):
            calls.append('open')
        def close(self):
            calls.append('close')

    def rig(**kw):
        assert kw['ptt'] is None and kw['transmit'] is False
        return NS(fatal=None)

    monkeypatch.setattr(process, 'YaesuCat', Driver)
    monkeypatch.setattr(process, 'RigctldCat', Driver)
    monkeypatch.setattr(process, 'Rig', rig)
    monkeypatch.setattr(process, 'RtsPtt', lambda *a: pytest.fail('opened PTT'))
    station = process.Station(configuration(transmit=False, cat=cat))
    station.open_rig()
    assert calls == ([] if cat == 'none' else ['driver', 'open'])
    assert (station.rig is None) == (cat == 'none')


@pytest.mark.parametrize('cat,driver', [('yaesu', 'YaesuCat'), ('civ', 'RigctldCat'),
                                        ('rigctld', 'RigctldCat'), ('none', None)])
def test_open_rig_builds_the_driver_its_cat_names(monkeypatch, cat, driver):
    built = []

    def fake(name):
        class Driver:
            def __init__(self, *a, **kw):
                built.append((name, a, kw))
            def open(self):
                pass
            def close(self):
                pass
        return Driver

    monkeypatch.setattr(process, 'YaesuCat', fake('YaesuCat'))
    monkeypatch.setattr(process, 'RigctldCat', fake('RigctldCat'))
    monkeypatch.setattr(process, 'Rig', lambda **kw: NS(fatal=None))
    station = process.Station(configuration(transmit=False, cat=cat))
    station.open_rig()
    assert [name for name, *_ in built] == ([driver] if driver else [])
    if driver == 'RigctldCat':
        assert built[0][1] == ('127.0.0.1', 4532)


def test_open_rig_refuses_a_cat_that_is_not_a_driver(monkeypatch):
    monkeypatch.setattr(process, 'YaesuCat', lambda *a, **kw: pytest.fail('fell through to Yaesu'))
    monkeypatch.setattr(process, 'RigctldCat', lambda *a, **kw: pytest.fail('built rigctld'))
    station = process.Station(configuration(transmit=False, cat='kenwood'))
    with pytest.raises(process.ConfigError, match="cat = 'kenwood' is not a CAT driver"):
        station.open_rig()


@pytest.mark.parametrize('cat,backend', [('yaesu', 'serial'), ('civ', 'rigctld'),
                                         ('rigctld', 'rigctld'), ('none', 'none')])
def test_cat_backend_names_how_each_driver_reaches_the_radio(cat, backend):
    assert process.cat_backend(cat) == backend


def test_cat_backend_refuses_an_unknown_driver():
    with pytest.raises(process.ConfigError, match='choose one of yaesu, civ, rigctld, none'):
        process.cat_backend('icom')



def test_lane_fence_uses_burst_and_item_bound():
    from hfmodem.station.link import _ThreadedModemLink
    link = object.__new__(_ThreadedModemLink)
    limits = []
    link.station = NS(cfg=NS(rig=NS(max_key_s=30, ptt=NS(settle_s=.4))), arbiter=None)
    link.modem = NS(stop=lambda timeout: limits.append(timeout) or True,
                    _audio_limit=96000, card_rate=48000)
    link.stop()
    assert limits == [32.4]


def test_audio_close_fences_a_stuck_stream():
    entered, finish = threading.Event(), threading.Event()
    closed = threading.Event()
    audio = StationAudio(input_device='fake', output_device=None)
    audio.close_budget_s = .02

    def stop():
        entered.set()
        finish.wait(3)

    audio._stream = NS(stop=stop, close=closed.set)
    try:
        with pytest.raises(LeaseTimeout, match='audio'):
            audio.close()
        assert entered.is_set() and not closed.is_set()
        assert audio._stream is not None and audio._close_thread.is_alive()
    finally:
        finish.set()
        audio._close_thread.join(2)
        audio.close()
    assert closed.is_set() and audio._stream is None


def test_audio_close_error_is_not_a_success():
    audio = StationAudio(input_device='fake', output_device=None)
    def fail():
        raise OSError('fake stream failed')
    audio._stream = NS(stop=fail)
    with pytest.raises(AudioError, match='fake stream failed'):
        audio.close()
    assert audio._stream is not None


def test_partial_host_start_is_retained_until_checked_stop(station, monkeypatch):
    station.cfg.protocols['ale'] = NS(enabled=True)
    server = NS(stop=lambda timeout: False)
    monkeypatch.setattr(hosts, '_build_one', lambda *a: hosts.Served('ale', server, (), False))
    monkeypatch.setattr(hosts, '_start', lambda server: (_ for _ in ()).throw(OSError('bind')))
    with pytest.raises(LeaseTimeout, match='ale'):
        station.own(receive=True)
    assert station.state == 'failed' and station.claim is not None
    assert station.hosts['ale'].server is server
    server.stop = lambda timeout: True


def test_failed_host_bind_reports_unready(station, monkeypatch):
    station.cfg.protocols['ale'] = NS(enabled=True)
    server = NS(stop=lambda timeout: True)
    monkeypatch.setattr(hosts, '_build_one', lambda *a: hosts.Served('ale', server, (), False))
    monkeypatch.setattr(hosts, '_start', lambda server: (_ for _ in ()).throw(OSError('port busy')))
    result = station.own(receive=True)
    assert result == {'state': 'owning-receive', 'ready': False}
    assert station.failed == [{'name': 'ale', 'why': 'port busy'}]
    station.release()


def test_integrated_protocols_open_their_hosts(station, monkeypatch):
    names = ['ardop', 'vara', 'sabir']
    station.cfg.protocols.update({name: NS(enabled=True, host='test') for name in names})
    started, stopped = [], []
    def build(name, protocol, owner):
        assert owner is station
        server = NS(start=lambda: started.append(name),
                    stop=lambda timeout: stopped.append(name) or True)
        return hosts.Served(name, server, (), True)
    monkeypatch.setattr(hosts, '_build_one', build)
    station.own(receive=True)
    supported = {'ardop', 'vara'}
    assert set(started) == supported
    assert set(station.hosts) == supported
    assert station.failed == [{'name': 'sabir', 'why': 'served from cutover 6.6'}]
    assert 'sabir' not in station.enabled
    station.release()
    assert set(stopped) == supported


def test_real_host_live_worker_blocks_release(station, monkeypatch):
    from hfmodem.tests.core.test_station_hosts import socketpair_network, _server
    from hfmodem.tests.core.test_host_teardown import Pin
    listeners = socketpair_network(monkeypatch)
    server = _server('ale')
    pin = Pin()
    monkeypatch.setattr(server, 'command', pin)
    server.start_background()
    client = listeners[0].connect()
    original = hosts.stop_all
    monkeypatch.setattr(hosts, 'stop_all', lambda served: original(served, timeout=.03))
    try:
        client.sendall(b'WAIT\r')
        assert pin.entered.wait(2)
        station.own(receive=True)
        station.hosts['ale'] = hosts.Served('ale', server, (), False)
        with pytest.raises(LeaseTimeout, match='ale'):
            station.release()
        assert station.state == 'failed' and station.claim is not None
        assert pin.thread.is_alive()
        assert all(conn.closed for conn in server._connections)
    finally:
        pin.release.set()
        client.close()
        assert server.stop(2)


def test_replay_owns_no_devices_or_claim(tmp_path, monkeypatch):
    station = process.Station(configuration(transmit=False), replay=np.zeros(4800))
    monkeypatch.setattr(claim, 'acquire', lambda *a, **kw: pytest.fail('replay claim'))
    monkeypatch.setattr(process.devices, 'find_device', lambda *a, **kw: pytest.fail('replay device'))
    station.own(receive=True)
    from hfmodem.core.virtualrig import VirtualRig
    assert station.claim is None and isinstance(station.rig, VirtualRig)
    assert station.rig.cat is station.rig.ptt is None
    assert not station.rig.transmit
    station.release()
    assert station.state == 'observing'


def test_ale_directory_stalled_supervisor_has_short_fence():
    import time
    from hfmodem.station.link import AleLink
    link = object.__new__(AleLink)
    link.station = NS(cfg=NS(rig=NS(max_key_s=30, ptt=NS(settle_s=.2))), arbiter=None, rig=None)
    link.modem = NS(stop=lambda timeout: True, _audio_limit=48000, card_rate=48000)
    finish = threading.Event()
    link._directory_stop = threading.Event()
    link._directory_thread = threading.Thread(target=finish.wait, daemon=True)
    link._directory_thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(LeaseTimeout, match='ale-directory'):
            link.stop()
        assert time.monotonic() - started < 1
        assert link._directory_stop.is_set()
    finally:
        finish.set()
        link._directory_thread.join(2)


def test_own_admission_deadline_is_cancelled_and_joined(station, monkeypatch):
    timers = []
    class Timer:
        def __init__(self, seconds, fn, args):
            self.seconds, self.fn, self.args = seconds, fn, args
            self.cancelled = self.joined = False
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            self.cancelled = True
        def join(self):
            self.joined = True
    monkeypatch.setattr(process.threading, 'Timer', Timer)
    station.own(receive=True)
    timer = timers[0]
    assert timer.seconds == 40 and timer.fn == station.fatal_shutdown
    assert timer.cancelled and timer.joined
    station.release()


def test_receive_open_audio_never_resolves_output(monkeypatch):
    found = []
    opened = []
    class Audio:
        def __init__(self, **kw):
            opened.append(kw)
        def open(self):
            pass
    monkeypatch.setattr(process, 'StationAudio', Audio)
    monkeypatch.setattr(process.devices, 'find_device',
                        lambda name, direction, **kw: found.append((name, direction)) or name)
    station = process.Station(configuration(transmit=False))
    station.open_audio()
    assert found == [('fake-in', 'in')]
    assert opened[0]['output_device'] is None


def test_capture_callback_failure_prevents_ready(monkeypatch):
    import sounddevice
    class Stream:
        latency = .01
        def __init__(self, **kw):
            self.callback = kw['callback']
        def start(self):
            self.callback(np.zeros((128, 1)), 128, NS(), None)
        def stop(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(sounddevice, 'InputStream', Stream)
    audio = StationAudio(input_device='fake', output_device=None)
    with pytest.raises(AudioError, match='callback failed'):
        audio.open()
    assert audio._stream is None


@pytest.mark.parametrize('part', ['lane-error', 'lane-exit', 'lane-false', 'arbiter', 'rig'])
def test_teardown_failure_still_closes_every_device(station, part):
    station.own()
    line, audio = station.rig.ptt, station.audio
    driver = NS(close=lambda: station.calls.append('CAT-close'))
    station._driver = driver

    def fail():
        if part == 'lane-exit':
            raise SystemExit('failed close')
        raise OSError('failed close')

    if part.startswith('lane'):
        station.lanes['bad'] = NS(stop=fail if part != 'lane-false' else lambda: False)
        station.lanes['last'] = NS(stop=lambda: station.calls.append('last-lane'))
    elif part == 'arbiter':
        station.arbiter = NS(stop=lambda owner: None, close=lambda **kw: fail())
    else:
        station.rig.close = fail
    with pytest.raises(LeaseTimeout):
        station.release()
    assert station.state == 'failed' and station.claim is not None
    assert line.released_low is True
    assert 'CAT-close' in station.calls and 'audio-close' in audio.calls
    if part.startswith('lane'):
        assert 'last-lane' in station.calls
    assert station.calls[-1][0] == 'fatal'


def test_inode_release_with_unconfirmed_line_stays_failed(station, monkeypatch):
    station.own()
    station._last_inode_check = 0.0
    monkeypatch.setattr(station.rig.ptt, 'release', lambda: None)
    monkeypatch.setattr(claim, 'check_inode', lambda held: (_ for _ in ()).throw(claim.ClaimError('changed inode')))
    station.check_claim(now=station._last_inode_check + 60)
    assert station.state == 'failed' and station.claim is not None
    assert 'unconfirmed' in station.snapshot()['service']['reason']
    assert not station._stop.is_set()


def test_inode_release_transition_race_is_retried(station, monkeypatch):
    station.own()
    station._last_inode_check = 0.0
    monkeypatch.setattr(claim, 'check_inode', lambda held: (_ for _ in ()).throw(claim.ClaimError('changed inode')))
    release = station.release
    calls = []

    def race(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError('transitioning')
        return release(**kw)

    monkeypatch.setattr(station, 'release', race)
    now = station._last_inode_check + 60
    station.check_claim(now=now)
    assert station.state == 'owning' and not station._stop.is_set()
    station.check_claim(now=now)
    assert station.state == 'observing' and len(calls) == 2
    assert station.reason == 'changed inode'


def test_stop_cancels_own_before_any_device_open(station, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    errors = []

    def reference():
        entered.set()
        assert finish.wait(3)

    monkeypatch.setattr(station, 'preflight_reference', reference)
    def own():
        try:
            station.own()
        except process.RigError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=own)
    thread.start()
    try:
        assert entered.wait(2)
        station.stop_transmit()
        finish.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors and 'cancelled' in errors[0]
        assert station.state == 'observing' and station.claim is None
        assert 'audio' not in station.calls
    finally:
        finish.set()
        thread.join(3)



def test_signal_handler_does_not_take_interrupted_locks(station, monkeypatch):
    import signal
    station.arbiter = NS(stop=lambda owner: pytest.fail('handler took admission lock'))
    station._stop = NS(set=lambda: pytest.fail('handler took Event lock'))
    station.rig = NS(panic=lambda: pytest.fail('handler ran panic'))
    with station._lifecycle:
        station._signal(signal.SIGTERM, None)
    assert station._signals == 1


def test_main_loop_runs_panic_and_escalates_second_signal(station, monkeypatch):
    import signal
    panic, finish = threading.Event(), threading.Event()
    monkeypatch.setattr(process, 'arm_deadman', lambda *a, **kw: None)
    class Exited(BaseException):
        pass
    monkeypatch.setattr(process.os, '_exit', lambda code: (_ for _ in ()).throw(Exited(code)))
    station.rig = NS(panic=lambda: (panic.set(), finish.wait(3)))
    station._signal(signal.SIGTERM, None)
    station._handle_signals()
    try:
        assert panic.wait(2)
        station._signal(signal.SIGINT, None)
        with pytest.raises(Exited) as exc:
            station._handle_signals()
        assert exc.value.args == (1,)
    finally:
        finish.set()
        station._panic_thread.join(2)


def test_listener_remains_available_through_release(station, monkeypatch):
    from hfmodem.station.service import Service
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    monkeypatch.setattr(station, '_install_signals', lambda: None)
    monkeypatch.setattr(service, 'close', lambda: station.calls.append('listener-close'))
    station.own()
    def stop():
        assert service.snapshot()['service']['state'] == 'releasing'
        assert 'listener-close' not in station.calls
        station.calls.append('last-lane')
    station.lanes['fake'].stop = stop
    station._stop.set()
    assert station.run(detach=True) == 0
    assert station.calls.index('last-lane') < station.calls.index('audio-close') < station.calls.index('listener-close')


def test_shutdown_failed_release_goes_fatal(station, monkeypatch):
    station.own()
    monkeypatch.setattr(station.rig.ptt, 'release', lambda: None)
    station.shutdown()
    assert station.state == 'failed' and station.claim is not None
    assert station.calls[-1][0] == 'fatal'
    assert 'shutdown release failed' in station.calls[-1][1]


def test_audio_reopen_closes_current_stream(monkeypatch):
    import sounddevice
    streams = []
    audio = StationAudio(input_device='fake', output_device=None)
    class Stream:
        latency = .01
        def __init__(self, **kw):
            self.stopped = self.closed = False
            streams.append(self)
        def start(self):
            audio.samples += 128
        def stop(self):
            self.stopped = True
        def close(self):
            self.closed = True
    monkeypatch.setattr(sounddevice, 'InputStream', Stream)
    for _ in range(2):
        audio.open()
        audio.close()
    assert len(streams) == 2 and all(s.stopped and s.closed for s in streams)


def test_duplex_open_waits_for_dac_offset(monkeypatch):
    import sounddevice
    audio = StationAudio(input_device='fake', output_device='fake')
    waits = []
    class Stream:
        latency = (.01, .02)
        def __init__(self, **kw):
            pass
        def start(self):
            audio.samples = 128
        def stop(self):
            pass
        def close(self):
            pass
    def settle(duration):
        waits.append(duration)
        audio.lat = 1440
    monkeypatch.setattr(sounddevice, 'Stream', Stream)
    monkeypatch.setattr('hfmodem.core.audio.time.sleep', settle)
    audio.open()
    try:
        assert waits and audio.lat == 1440
        assert audio.holdback == audio.blocksize + 480
    finally:
        audio.close()


def test_duplex_settled_callback_failure_keeps_base_open_behavior(monkeypatch):
    import sounddevice
    audio = StationAudio(input_device='fake', output_device='fake')
    class Stream:
        latency = (.01, .02)
        def __init__(self, **kw):
            pass
        def start(self):
            audio.samples, audio.lat = 128, 1440
            audio.failure = ValueError('late callback')
        def stop(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(sounddevice, 'Stream', Stream)
    audio.open()
    audio.close()


def test_send_later_disconnects_nonreader_above_eight_mib(monkeypatch):
    delivery = threading.Event()
    original = hostio.Connection._deliver
    def gated(conn):
        assert delivery.wait(3)
        original(conn)
    monkeypatch.setattr(hostio.Connection, '_deliver', gated)
    conn = hostio.Connection(StalledSocket())
    try:
        conn.sendall(b'x' * hostio.MAX_OUTPUT)
        assert conn.sock.writing.wait(2)
        for _ in range(8192):
            conn.send_later(b'x' * 1024)
        assert not conn.closed and conn._delivery_bytes == 8 * 1024 * 1024
        conn.send_later(b'x')
        assert conn.closed and conn.sock.cancelled.is_set()
    finally:
        delivery.set()
        conn.cancel()
        assert conn.join(2)


def test_sendall_two_second_deadline_does_not_need_socket_progress():
    import time
    conn = hostio.Connection(StalledSocket())
    sent = threading.Event()
    conn.sendall(b'x' * hostio.MAX_OUTPUT)
    assert conn.sock.writing.wait(2)
    begin = time.monotonic()
    thread = threading.Thread(target=lambda: (conn.sendall(b'blocked'), sent.set()))
    thread.start()
    try:
        assert sent.wait(2.8), 'sendall depended on the fake socket finishing'
        assert 1.8 <= time.monotonic() - begin < 2.8
        assert conn.closed
    finally:
        conn.cancel()
        thread.join(2)
        assert conn.join(2)


def test_host_outer_bound_allows_its_own_deadline(monkeypatch):
    clock = [10.0]
    joins, limits = [], []
    class Thread:
        def __init__(self, target, args, **kw):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)
        def join(self, timeout):
            joins.append(timeout)
        def is_alive(self):
            return False
    monkeypatch.setattr(hosts.threading, 'Thread', Thread)
    monkeypatch.setattr(hosts.time, 'monotonic', lambda: clock[0])
    def stop(timeout):
        limits.append(timeout)
        clock[0] += timeout
        return True
    hosts.stop_all({'host': hosts.Served('host', NS(stop=stop), (), False)})
    assert limits == [5.0] and joins[0] >= .2


@pytest.mark.parametrize('sources', [1, 20])
def test_ale_stalled_load_cannot_hold_release(station, monkeypatch, sources):
    from hfmodem.station.link import AleLink
    from hfmodem.bateleur.ale import directory
    entered, finish, done = threading.Event(), threading.Event(), threading.Event()
    def load(srcs, *, timeout):
        assert len(srcs) == sources and timeout == 30
        entered.set()
        finish.wait(10)
        done.set()
        return NS(summary=lambda: 'late directory')
    monkeypatch.setattr(directory, 'load', load)
    station.own(receive=True)
    station.rig.tune = lambda hz: None
    station.cfg.protocols['ale'] = NS(directory=['https://example.invalid/directory'] * sources)
    link = object.__new__(AleLink)
    link.station = station
    original = object()
    link.modem = NS(directory=original, start=lambda: None, stop=lambda timeout: True,
                    _audio_limit=48000, card_rate=48000)
    link.start()
    assert entered.wait(2)
    station.lanes['ale'] = link
    released = threading.Event()
    thread = threading.Thread(target=lambda: (station.release(), released.set()))
    thread.start()
    try:
        assert released.wait(1), 'directory I/O held the radio during release'
        assert station.state == 'observing'
        assert not link._directory_thread.is_alive()
        finish.set()
        assert done.wait(1)
        assert link.modem.directory is original
        assert not any(isinstance(call, tuple) and call[0] == 'fatal' for call in station.calls)
    finally:
        finish.set()
        thread.join(3)



@pytest.mark.parametrize('verdict', [0, 1, 2, 'signal'])
def test_preflight_close_error_preserves_verdict_or_signal(monkeypatch, capsys, verdict):
    from hfmodem.station.preflight import Preflight
    from contextlib import nullcontext
    check = Preflight(configuration())
    audio = NS(close=lambda: (_ for _ in ()).throw(AudioError('failed audio close')))
    monkeypatch.setattr('hfmodem.station.preflight.unkey_on_signal', lambda fn: nullcontext())
    monkeypatch.setattr(check, '_reference', lambda: None)
    monkeypatch.setattr(check, '_card', lambda ref: audio)
    monkeypatch.setattr(check, '_rig', lambda: None)
    monkeypatch.setattr(check, '_clock', lambda card: None)
    def keying(*args):
        if verdict == 'signal':
            raise SystemExit(143)
    monkeypatch.setattr(check, '_keying', keying)
    monkeypatch.setattr(check, 'verdict', lambda: verdict)
    if verdict == 'signal':
        with pytest.raises(SystemExit) as exc:
            check.run()
        assert exc.value.code == 143
    else:
        assert check.run() == verdict
    assert 'audio close: failed audio close' in capsys.readouterr().err


def test_hflink_start_uses_inherited_claim_without_proofs(monkeypatch):
    from hfmodem.tests.evidence import TREE
    monkeypatch.syspath_prepend(str(TREE / "tools"))
    from consolelib.hflink import Station
    calls = []
    st = Station(configuration())
    monkeypatch.setattr(claim, 'acquire', lambda *a, **kw: pytest.fail('second claim'))
    monkeypatch.setattr(st, 'preflight_reference', lambda: pytest.fail('capture proof'))
    monkeypatch.setattr(st, 'preflight_measure', lambda ref: pytest.fail('preflight keyings'))
    monkeypatch.setattr(st, 'open_audio', lambda: calls.append('audio'))
    monkeypatch.setattr(st, 'open_rig', lambda: calls.append('rig-without-proof'))
    monkeypatch.setattr(st, 'build_arbiter', lambda: calls.append('arbiter'))
    monkeypatch.setattr(st, 'build_lanes', lambda: calls.append('lanes'))
    monkeypatch.setattr(hosts, 'build', lambda *a, **kw: calls.append('hosts') or {})
    monkeypatch.setattr(st, '_poll', lambda: None)
    st._start()
    st._poll_thread.join(2)
    assert calls == ['audio', 'rig-without-proof', 'arbiter', 'lanes', 'hosts']
    assert st.claim is None


@pytest.mark.parametrize('during', ['own', 'release'])
def test_real_signals_escape_lifecycle_lock_and_escalate(during):
    import selectors
    import signal
    import subprocess
    import sys
    script = r'''
import os, threading, sys
from types import SimpleNamespace as NS
from hfmodem.station.process import Station
from hfmodem.tests.station.test_process import configuration
st = Station(configuration(protocols={'ale': NS(enabled=True)}))
class Rig:
    def panic(self):
        os.write(1, b'panic\n')
        threading.Event().wait()
st.rig = Rig()
def pinned():
    with st._lifecycle:
        os.write(1, b'ready\n')
        threading.Event().wait()
if sys.argv[1] == 'own':
    st._start = pinned
else:
    st.shutdown = pinned
    st._stop.set()
st.run(detach=sys.argv[1] == 'release')
'''
    with subprocess.Popen([sys.executable, '-c', script, during], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as child:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(5), 'child never reached lifecycle lock'
                assert child.stdout.readline() == b'ready\n'
                child.send_signal(signal.SIGTERM)
                assert selector.select(1), 'main loop did not run panic'
                assert child.stdout.readline() == b'panic\n'
                child.send_signal(signal.SIGINT)
                _, errors = child.communicate(timeout=1)
                assert child.returncode == 1 and b'asked again' in errors
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


def test_signal_over_admitted_operation_takes_fatal_unkey_path():
    import selectors
    import signal
    import subprocess
    import sys
    script = r'''
import os, threading
from types import SimpleNamespace as NS
from hfmodem.station.process import Station
from hfmodem.core.rig import Rig
from hfmodem.tests.station.test_process import configuration
st = Station(configuration(protocols={'ale': NS(enabled=True)}))
class Line:
    released_low = None
    def release(self):
        self.released_low = True
        os.write(1, b'down\n')
st.rig = Rig(model='fake', cat=NS(emergency_unkey=lambda: False), ptt=Line(), transmit=True,
             profile=None, control=None)
st.rig.fatal = st.fatal_shutdown
def start():
    with st.rig.operations.lease():
        os.write(1, b'ready\n')
        threading.Event().wait()
st._start = start
st.run()
'''
    with subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as child:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(5)
                assert child.stdout.readline() == b'ready\n'
            child.send_signal(signal.SIGTERM)
            out, errors = child.communicate(timeout=3)
            assert child.returncode == 1 and b'down\n' in out
            assert b'fatal_shutdown confirmed=true' in errors
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


def test_poll_claims_breathing_on_lane_thread(station, monkeypatch):
    seen = []
    station.audio = NS(breathe=lambda: seen.append(('breathe', threading.get_ident())))
    def tick():
        seen.append(('tick', threading.get_ident()))
        station._poll_stop.set()
    monkeypatch.setattr(station, '_tick', tick)
    worker = threading.Thread(target=station._poll)
    worker.start()
    worker.join(2)
    assert seen == [('breathe', worker.ident), ('tick', worker.ident)]
    assert worker.ident != threading.get_ident()


def test_coalesced_signals_start_panic_before_escalating(station, monkeypatch):
    import signal
    entered = threading.Event()
    monkeypatch.setattr(process, 'arm_deadman', lambda *a: None)
    monkeypatch.setattr(process.os, '_exit', lambda code: pytest.fail('skipped panic'))
    station.rig = NS(panic=entered.set)
    station._signal(signal.SIGHUP, None)
    station._signal(signal.SIGTERM, None)
    station._handle_signals()
    assert entered.wait(1)
    station._panic_thread.join(1)


def test_stop_at_release_tail_does_not_cancel_reown(station, monkeypatch):
    station.own()
    original = station._state
    def state(name, reason=''):
        original(name, reason)
        if name == 'observing':
            station.stop_transmit()
    monkeypatch.setattr(station, '_state', state)
    station.release()
    assert not station._own_cancel.is_set()
    assert station.own()['state'] == 'owning'
    station.release()


def test_failed_claim_is_not_torn_down_again(station, monkeypatch):
    station.own()
    station._last_inode_check = 0.0
    station.rig.ptt.release = lambda: None
    monkeypatch.setattr(claim, 'check_inode', lambda held: (_ for _ in ()).throw(claim.ClaimError('changed')))
    now = station._last_inode_check + 60
    station.check_claim(now=now)
    assert station.state == 'failed' and station.claim is not None
    before = list(station.calls)
    station.check_claim(now=now + 60)
    station.check_claim(now=now + 120)
    assert station.calls == before


def test_run_does_not_spawn_workers_during_transition(station, monkeypatch):
    monkeypatch.setattr(station, '_install_signals', lambda: None)
    station.claim = object()
    station._last_inode_check = 0
    station._lifecycle.acquire()
    checked = []
    monkeypatch.setattr(station, 'check_claim', lambda: checked.append(True))
    def finish():
        station._stop.set()
        station._lifecycle.release()
    timer = threading.Timer(.15, finish)
    timer.start()
    monkeypatch.setattr(station, 'shutdown', lambda: None)
    try:
        assert station.run(detach=True) == 0
        assert not checked
    finally:
        timer.join()
        station.claim = None


def test_rollback_cancels_admission_timer_before_teardown(station, monkeypatch):
    timers = []
    class Timer:
        def __init__(self, seconds, fn, args):
            self.cancelled = self.joined = False
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            self.cancelled = True
        def join(self):
            self.joined = True
    monkeypatch.setattr(process.threading, 'Timer', Timer)
    monkeypatch.setattr(station, 'open_audio', lambda: (_ for _ in ()).throw(process.RigError('rollback')))
    original = station._teardown
    def teardown(**kw):
        assert timers[0].cancelled and timers[0].joined
        original(**kw)
    monkeypatch.setattr(station, '_teardown', teardown)
    with pytest.raises(process.RigError, match='rollback'):
        station.own()
    assert station.state == 'observing'


def test_signal_during_http_own_before_rig_rolls_back(station, monkeypatch):
    import signal
    from hfmodem.station.service import Service
    from hfmodem.tests.station.test_service import command
    entered = threading.Event()
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    def reference():
        entered.set()
        assert station._own_cancel.wait(2)
    monkeypatch.setattr(station, 'preflight_reference', reference)
    monkeypatch.setattr(station, '_install_signals', lambda: None)
    monkeypatch.setattr(process, 'arm_deadman', lambda *a: None)
    dispatcher = threading.Thread(target=service._dispatch)
    dispatcher.start()
    service.submit(command(station, 'own'))
    assert entered.wait(2)
    station._signal(signal.SIGTERM, None)
    try:
        assert station.run(detach=True) == 128 + signal.SIGTERM
        assert station.state == 'observing' and station.claim is None
        assert 'audio' not in station.calls
        assert not any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)
    finally:
        service._closed.set()
        dispatcher.join(2)


class HookLock:
    def __init__(self, *, entering=None, acquired=None, released=None):
        self._lock = threading.Lock()
        self.hooks = {'entering': entering, 'acquired': acquired, 'released': released}

    def _fire(self, name):
        hook, self.hooks[name] = self.hooks[name], None
        if hook is not None:
            hook()

    def acquire(self, blocking=True, timeout=-1):
        got = self._lock.acquire(blocking, timeout)
        if got:
            self._fire('acquired')
        return got

    def __enter__(self):
        self._fire('entering')
        return self.acquire()

    def release(self):
        self._lock.release()
        self._fire('released')

    def __exit__(self, *exc):
        self.release()

    def locked(self):
        return self._lock.locked()


def test_queued_own_cannot_enter_shutdowns_cancel_gap(station, monkeypatch):
    entered, shut = threading.Event(), threading.Event()
    errors = []

    def reference():
        entered.set()
        assert shut.wait(3)
        return 1

    def own():
        try:
            station.own()
        except (RuntimeError, process.RigError) as exc:
            errors.append(str(exc))

    def gap():
        thread.start()
        while thread.is_alive() and not entered.wait(.01):
            pass

    thread = threading.Thread(target=own)
    monkeypatch.setattr(station, 'preflight_reference', reference)
    station._lifecycle = HookLock(released=gap)
    try:
        station.shutdown()
    finally:
        shut.set()
        thread.join(3)
    assert not any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)
    assert not entered.is_set() and errors == ['shutting-down']
    assert station.state == 'observing' and station.claim is None


def test_queued_release_cannot_enter_shutdowns_release(station, monkeypatch):
    from hfmodem.station.service import Service
    from hfmodem.tests.station.test_service import command
    station.own()
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    service.commands.put_nowait(command(station, 'release', 'queued'))
    dispatcher = threading.Thread(target=service._dispatch)
    held, done = threading.Event(), threading.Event()
    original = station._teardown

    def teardown(**kw):
        if threading.current_thread() is dispatcher:
            held.set()
            assert done.wait(3)
        original(**kw)

    def outcome():
        return next((e['data'] for e in service.events.after(0) if e['type'] == 'command'
                     and e['data']['state'] in ('refused', 'failed', 'done')), None)

    def gap():
        dispatcher.start()
        with service.events.condition:
            assert service.events.condition.wait_for(lambda: held.is_set() or outcome(),
                                                     timeout=2)

    monkeypatch.setattr(station, '_teardown', teardown)
    station._lifecycle = HookLock(released=gap)
    try:
        station.shutdown()
    finally:
        done.set()
        service._closed.set()
        dispatcher.join(3)
    assert not any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)
    assert not held.is_set() and outcome()['state'] == 'done'
    assert station.state == 'observing' and station.claim is None


def test_shutdown_after_owns_first_check_refuses_before_the_claim(station):
    def shutdown_begins():
        station._stop.set()
        station._own_cancel.set()
    station._lifecycle = HookLock(acquired=shutdown_begins)
    with pytest.raises(RuntimeError, match='^shutting-down$'):
        station.own()
    assert station.calls == [] and station.claim is None


def test_stop_after_own_takes_the_lifecycle_is_kept(station):
    station._lifecycle = HookLock(acquired=station.stop_transmit)
    with pytest.raises(process.RigError, match='STOP'):
        station.own()
    assert station.calls == ['reference']
    assert station.state == 'observing' and station.claim is None


def test_repeated_signals_without_a_rig_keep_the_first_deadman(station, monkeypatch):
    import signal
    armed = []
    monkeypatch.setattr(process, 'arm_deadman', lambda *a: armed.append(a))
    monkeypatch.setattr(process.os, '_exit', lambda code: pytest.fail('exit with no panic'))
    for _ in range(3):
        station._signal(signal.SIGINT, None)
        station._handle_signals()
    assert armed == [(5.0,)] and station._stop.is_set()


def test_own_after_shutdown_begins_is_refused(station):
    from hfmodem.station.service import Service
    from hfmodem.tests.station.test_service import command
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    service.commands.put_nowait(command(station, 'own', 'queued'))
    station._stop.set()
    with pytest.raises(RuntimeError, match='^shutting-down$'):
        station.own()
    with station._lifecycle, pytest.raises(RuntimeError, match='^shutting-down$'):
        station.own()
    late = service.submit(command(station, 'own', 'late'))
    dispatcher = threading.Thread(target=service._dispatch)
    dispatcher.start()

    def outcome(id):
        return next((e['data'] for e in service.events.after(0) if e['type'] == 'command'
                     and e['data']['id'] == id and e['data']['state'] in ('refused', 'failed', 'done')),
                    None)
    try:
        with service.events.condition:
            assert service.events.condition.wait_for(lambda: outcome('queued'), timeout=2)
    finally:
        service._closed.set()
        dispatcher.join(2)
    assert outcome('queued')['state'] == 'refused' and outcome('queued')['reason'] == 'shutting-down'
    assert service.events.after(late['seq'] - 1)[0]['data']['reason'] == 'shutting-down'
    assert station.claim is None and station.calls == []


@pytest.mark.parametrize('via', ['stop', 'shutdown'])
@pytest.mark.parametrize('step', ['proof-line', 'proof-cat', 'preflight', 'preflight-floor'])
def test_stop_during_keyed_own_unkeys_within_one_operation(station, monkeypatch, step, via):
    import time
    from contextlib import nullcontext
    from dataclasses import replace
    from hfmodem.station import preflight
    from hfmodem.station.service import Service
    from hfmodem.tests.station.test_service import command

    entered, finish = threading.Event(), threading.Event()
    edges, callers, rigs = [], [], []

    def block():
        entered.set()
        assert finish.wait(2)

    class Line:
        high = False
        released_low = None
        def open(self):
            pass
        def assert_(self, on):
            callers.append(threading.get_ident())
            self.high = on
            edges.append((on, time.monotonic()))
            rising = sum(1 for up, _ in edges if up)
            if on and (step, rising) in (('proof-line', 1), ('preflight', 2)):
                block()
        def sense(self):
            return self.high
        def release(self):
            self.high = False
            self.released_low = True
    line = Line()
    class Driver:
        cache_free = reads_ptt = True
        def open(self):
            pass
        def close(self):
            pass
        def identify(self):
            return 'fake'
        def freq(self):
            return 7100000
        def ptt_state(self, **kw):
            callers.append(threading.get_ident())
            if line.high and step == 'proof-cat':
                block()
            return line.high
        def sole_ptt_owner(self, notes):
            return True
        def hold(self):
            return nullcontext(True)
    class Audio:
        _lanes = []
        def subscribe(self, lane):
            self._lanes.append(lane)
            return lane
        def alive(self):
            pass
        def sample_now(self):
            if step == 'preflight-floor':
                block()
            return 0
        def close(self):
            station.calls.append('audio-close')
    monkeypatch.setattr(process, 'YaesuCat', lambda *a, **kw: Driver())
    monkeypatch.setattr(process, 'RigctldCat', lambda *a, **kw: pytest.fail('real driver'))
    monkeypatch.setattr(process, 'RtsPtt', lambda *a: line)
    monkeypatch.setattr(station, 'open_audio', lambda: setattr(station, 'audio', Audio()))
    monkeypatch.delattr(station, 'preflight_reference')
    monkeypatch.delattr(station, 'preflight_measure')
    monkeypatch.setattr(preflight.Preflight, '_reference', lambda self: object())
    for measurement in ('_duplex', '_xruns', '_offset', '_clock'):
        monkeypatch.setattr(preflight.Preflight, measurement, lambda self, *a: None)
    for wait in ('DWELL_S', 'FLOOR_S', 'RECOVER_S'):
        monkeypatch.setattr(preflight, wait, 0)
    station.cfg = replace(station.cfg, rig=replace(
        station.cfg.rig, model='generic', cat='yaesu', host='invalid.test', port=0,
        ptt=config.Ptt(port='fake-line')))
    def open_rig(**kw):
        try:
            return process.Station.open_rig(station, **kw)
        finally:
            rigs.append(station.rig)
    monkeypatch.setattr(station, 'open_rig', open_rig)
    shutting = threading.Event()
    station._lifecycle = HookLock(entering=shutting.set)
    shutdown = threading.Thread(target=station.shutdown)
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    dispatcher = threading.Thread(target=service._dispatch)
    dispatcher.start()
    service.submit(command(station, 'own'))

    def own_outcome():
        return next((e['data'] for e in service.events.after(0) if e['type'] == 'command'
                     and e['data']['id'] == 'one' and e['data']['state'] in ('failed', 'done')),
                    None)
    try:
        assert entered.wait(2)
        count, keyed = len(callers), len(edges)
        if via == 'stop':
            service.submit(command(station, 'stop', id='stop'))
        else:
            shutdown.start()
            assert shutting.wait(2)
        assert len(callers) == count, 'the request touched the rig during an operation'
        assert station._own_cancel.is_set()
        released = time.monotonic()
        finish.set()
        with service.events.condition:
            assert service.events.condition.wait_for(own_outcome, timeout=2)
        if via == 'shutdown':
            shutdown.join(2)
        after = edges[keyed:]
        if step == 'preflight-floor':
            assert not any(up for up, _ in after), 'keyed after the STOP'
        else:
            assert after and after[0][0] is False
            assert after[0][1] - released < .05, 'unkey waited out the hold'
        assert own_outcome()['state'] == 'failed'
        assert 'cancelled by STOP' in own_outcome()['reason']
        assert edges[-1][0] is False and not line.high
        assert station.state == 'observing' and station.claim is None
        assert set(callers) == {dispatcher.ident}
        assert station._stop.is_set() is (via == 'shutdown')
        assert not rigs[0].retired
        assert not any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)
    finally:
        finish.set()
        if shutdown.is_alive():
            shutdown.join(3)
        service._closed.set()
        dispatcher.join(3)


def test_ale_load_deadline_includes_stalled_dns_and_all_sources(monkeypatch):
    from hfmodem.station.link import AleLink
    from hfmodem.bateleur.ale import directory
    entered, finish, done = threading.Event(), threading.Event(), threading.Event()
    def load(sources, **kw):
        entered.set()
        finish.wait(3)
        done.set()
        return NS(summary=lambda: 'too late')
    monkeypatch.setattr(directory, 'load', load)
    link = object.__new__(AleLink)
    original = object()
    link.modem = NS(directory=original)
    link._directory_budget = .1
    link._directory_stop = threading.Event()
    worker = threading.Thread(target=link._load_directory, args=(['default'] * 20,))
    worker.start()
    try:
        assert entered.wait(1)
        worker.join(.5)
        assert not worker.is_alive(), 'whole-load deadline did not abandon DNS/I/O'
        assert not done.is_set() and link.modem.directory is original
    finally:
        finish.set()
        worker.join(2)
        assert done.wait(1)
    assert link.modem.directory is original


def test_records_root_expands_home_and_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HFMODEM_DATA_DIR', '~/records/../data')
    st = process.Station(configuration())
    assert st.sessions.records.directory.is_relative_to(tmp_path / 'data')
    assert st.sessions.records.directory == st.sessions.records.directory.resolve()


@pytest.mark.parametrize('line_state, reason', [
    ('unconfirmed', 'duplex preflight failed: the keying line is unconfirmed'),
    ('low', 'duplex preflight failed'),
])
def test_a_failed_preflight_names_an_unconfirmed_line(tmp_path, monkeypatch, line_state, reason):
    from hfmodem.station import preflight
    monkeypatch.setattr(preflight, 'DWELL_S', 0)
    path = tmp_path / 'station.toml'
    path.write_text('[station]\ntransmit = true\n')
    st = process.Station(configuration(), station_file=path)
    st.audio = NS(_lanes=[], subscribe=lambda lane: st.audio._lanes.append(lane) or lane)
    st.rig = NS(line_state='low')
    st._preflight = NS(_read=lambda: (np.zeros(0, np.float32), 0), _duplex=lambda *a: None,
                       _xruns=lambda a: None, _offset=lambda a: None, _clock=lambda a: None,
                       _keying=lambda audio, rig: setattr(rig, 'line_state', line_state),
                       verdict=lambda: 1)
    with pytest.raises(process.RigError) as exc:
        st.preflight_measure(0.1)
    assert str(exc.value) == reason
    assert st.audio._lanes == []
