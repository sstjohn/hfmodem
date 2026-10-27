# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual worker + independent serial client + native sample-clock producer."""
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfhost import ptc
from hfmodem.core.audio import Block, StreamLane, BurstInvalidated
from hfmodem.shrike.arq import State
from hfmodem.shrike.station_runtime import StationPtcRuntime
from hfmodem.station.arbiter import Identity, TxArbiter
from hfmodem.station.link import ShrikeLink
from hfmodem.station import hosts
from hfmodem.tests.shrike.test_station_io import Capture, Keyer


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError('condition did not become true')


def confirmation_sample_budget(original, capacity, *, cw_seconds=0., settle_seconds=.04):
    """A fixture bound, allowing one missed keyable boundary per DATA attempt.

    The native scheduler postpones a late key instead of shortening its listen
    floor (_keyable_slot). This integration test admits one such postponement
    per attempt, not arbitrary missed slots. Acquisition has already completed.
    """
    import math
    from hfmodem.shrike import compress, spec
    cycle = spec.CYCLE_SHORT_S
    # Announcement and host writes can occupy separate partly filled packets.
    packets = math.ceil(len(b'1n0call\r') / capacity)
    packets += math.ceil(len(compress.transparent(original)) / capacity)
    attempts = packets + 1  # exactly one deliberately lost DATA acknowledgement
    slots = attempts * 2 + 2  # each attempt plus postponement; final ACK/poll drain
    if cw_seconds:
        # CW consumes complete raster slots; allow a boundary on either side.
        slots += math.ceil((cw_seconds + settle_seconds) / cycle) + 2
    return round(slots * cycle * 48000)


def retirement_sample_budget(cw_seconds, *, forced=False):
    import math
    from hfmodem.shrike import arq, onair, spec
    cw_slots = math.ceil((cw_seconds + .04) / spec.CYCLE_SHORT_S)
    if forced:
        # DD owes no QRT exchange. Finish the current cycle and the bounded
        # regrid attempts; age the peer projection before CW, then drain.
        slots = (1 + onair.REGRID_TRIES) + (onair.ISS_GUARD_CYCLES + 1) + cw_slots + 1
    else:
        # Bound first placement by ARQ's independent terminal-placement clock.
        # h > goodbye + QRT_CYCLES permits initial and terminal opportunities.
        # Admit one postponed boundary per attempt, then final drain and CW.
        slots = arq.GOODBYE_PLACE_TICKS + 2 * (onair.QRT_CYCLES + 2) + 2
        slots += cw_slots + 2
    return round(slots * spec.CYCLE_SHORT_S * 48000)


def await_confirmation(poll, progress, samples, budget, *, stall_seconds=15,
                       clock=time.monotonic, sleep=time.sleep, start_samples=None):
    """Separate bounded protocol time from wall-clock harness liveness.

    Transfer progress is decoded peer bytes or confirmed host echo. Retirement
    progress is the terminal notification or a completed CW identification.
    Repeated packets, callback activity and advancing capture alone do not.
    """
    start = samples() if start_samples is None else start_samples
    last_progress, changed = progress(), clock()
    began = changed
    while True:
        done = poll()
        assert samples() - start <= budget, 'confirmation exceeded sample-clock budget'
        current, now = progress(), clock()
        if current != last_progress:
            assert all(a >= b for a, b in zip(current, last_progress))
            last_progress, changed = current, now
        if done:
            return samples() - start, now - began
        assert now - changed < stall_seconds, 'confirmation made no byte progress'
        sleep(.005)


def test_confirmation_budget_counts_wire_packets_loss_and_cw():
    from hfmodem.core import cwid
    original = bytes(range(256)) + b'the quick brown fox jumps over the lazy dog\r' * 3
    # 391 transparent bytes + separate announcement: 21 packets, one retry.
    assert confirmation_sample_budget(original, 20) == 46 * 60000
    assert confirmation_sample_budget(original, 20, cw_seconds=cwid.duration('N0CALL', 14)) == 54 * 60000
    assert retirement_sample_budget(cwid.duration('N0CALL', 14)) == 34 * 60000
    assert retirement_sample_budget(cwid.duration('K1ABC'), forced=True) == 13 * 60000


@pytest.mark.parametrize('failure', ['stall', 'sample-budget'])
def test_confirmation_watchdogs_reject_activity_without_byte_progress(failure):
    state = {'wall': 0., 'samples': 0}
    def tick(seconds):
        state['wall'] += 1
        state['samples'] += 100
    with pytest.raises(AssertionError, match=('no byte progress' if failure == 'stall'
                                             else 'sample-clock budget')):
        await_confirmation(lambda: False, lambda: (0, 0), lambda: state['samples'],
                           1000 if failure == 'stall' else 150,
                           stall_seconds=3, clock=lambda: state['wall'], sleep=tick)


def test_confirmation_progress_renews_liveness_without_extending_sample_budget():
    state = {'wall': 0., 'samples': 0, 'bytes': 0}
    def tick(seconds):
        state['wall'] += 2
        state['samples'] += 100
        state['bytes'] += 1
    await_confirmation(lambda: state['bytes'] == 4, lambda: (state['bytes'], 0),
                       lambda: state['samples'], 400, stall_seconds=3,
                       clock=lambda: state['wall'], sleep=tick)
    state.update(wall=0., samples=0, bytes=0)
    with pytest.raises(AssertionError, match='sample-clock budget'):
        await_confirmation(lambda: state['bytes'] == 4, lambda: (state['bytes'], 0),
                           lambda: state['samples'], 300, stall_seconds=3,
                           clock=lambda: state['wall'], sleep=tick)


def test_retirement_watchdog_allows_terminal_silence_but_rejects_a_stall():
    from hfmodem.core import cwid
    budget = retirement_sample_budget(cwid.duration('N0CALL', 14))
    watchdog = budget / 48000
    state = {'wall': 0.}
    def tick(seconds):
        state['wall'] += 1
    # No byte progress is possible during placement/CW. A completed terminal
    # milestone can arrive after the old 15-second transfer watchdog.
    await_confirmation(lambda: state['wall'] == 20,
                       lambda: (int(state['wall'] >= 20),), lambda: 0, budget,
                       stall_seconds=watchdog, clock=lambda: state['wall'], sleep=tick)
    state['wall'] = 0.
    with pytest.raises(AssertionError, match='no byte progress'):
        await_confirmation(lambda: False, lambda: (0,), lambda: 0, budget,
                           stall_seconds=watchdog, clock=lambda: state['wall'], sleep=tick)


@pytest.mark.parametrize('failure', ['stall', 'sample-budget'])
def test_forced_retirement_rejects_stall_and_sample_overrun(failure):
    from hfmodem.core import cwid
    budget = retirement_sample_budget(cwid.duration('K1ABC'), forced=True)
    state = {'wall': 0., 'samples': 0}
    def tick(seconds):
        state['wall'] += budget / 48000 + 1 if failure == 'stall' else 1
        state['samples'] += budget + 1 if failure == 'sample-budget' else 0
    with pytest.raises(AssertionError, match=('no byte progress' if failure == 'stall'
                                             else 'sample-clock budget')):
        await_confirmation(lambda: False, lambda: (0, 0), lambda: state['samples'],
                           budget, stall_seconds=budget / 48000,
                           clock=lambda: state['wall'], sleep=tick)


@pytest.mark.parametrize('overrun', [-128, 0, 128])
def test_retirement_completion_latch_ignores_late_poll_but_not_late_completion(overrun):
    from hfmodem.core import cwid
    budget = retirement_sample_budget(cwid.duration('K1ABC'), forced=True)
    completed_at = budget + overrun
    # The feeder observed completion, then idle capture advanced before polling.
    current = completed_at + 1000000
    assert current > budget
    def wait():
        return await_confirmation(lambda: True, lambda: (1, 1), lambda: completed_at,
                                  budget, start_samples=0, clock=lambda: 20.)
    if overrun > 0:
        with pytest.raises(AssertionError, match='sample-clock budget'):
            wait()
    else:
        assert wait()[0] == completed_at


def test_retirement_milestones_cannot_renew_the_wall_watchdog_forever():
    state = {'wall': 0.}
    def tick(seconds):
        state['wall'] += 2
    with pytest.raises(AssertionError, match='no byte progress'):
        await_confirmation(lambda: False, lambda: (min(int(state['wall'] / 2), 2),),
                           lambda: 0, 780000, stall_seconds=3,
                           clock=lambda: state['wall'], sleep=tick)
    assert state['wall'] <= 3 * 3  # two finite milestones plus one stalled phase


class Card(Capture):
    """Independent waveform feeder with bounded virtual capture lookahead.

    Decoder CPU time is not simulated radio time. Keep at most one block ahead
    of the native reader, except for an explicit settle/DAC wait. The feeder
    remains the only clock producer; neither path invents samples or replies.
    """
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.halt = threading.Event()
        self.failure = None
        self.source = lambda at, count: np.zeros(count, np.float32)
        self._wait_targets = {}

    def wait_until(self, index, **options):
        def check():
            if options.get('cancelled') is not None and options['cancelled']():
                raise BurstInvalidated('cancelled')
            if options.get('deadline') is not None and time.monotonic() >= options['deadline']:
                raise BurstInvalidated('deadline')
        if self.samples < index:
            check()
        token = object()
        with self.condition:
            self._wait_targets[token] = index
        try:
            while self.samples < index:
                check()
                with self.condition:
                    self.condition.wait(.002)
        finally:
            with self.condition:
                del self._wait_targets[token]

    def advance(self, link, runtime):
        with self.condition:
            port = runtime._port
            if port is not None and not port.closed:
                limit = max([port.pos + 128, *self._wait_targets.values()])
                if self.samples >= limit:
                    return False
            at = self.samples
            block = self.source(at, 128)
            self.samples += 128
            link.lane.push(Block(0., at, block))
            self.condition.notify_all()
        runtime.pump()
        return True

    def run(self, link, runtime):
        try:
            while not self.halt.wait(.002):
                self.advance(link, runtime)
        except BaseException as exc:
            self.failure = exc


def test_virtual_card_pauses_at_one_queued_block_until_reader_advances():
    card = Card()
    port = SimpleNamespace(pos=0, closed=False)
    frames = []
    runtime = SimpleNamespace(_port=port, pump=lambda: None)
    link = SimpleNamespace(lane=SimpleNamespace(push=frames.append))
    assert card.advance(link, runtime)
    for _ in range(10):
        assert not card.advance(link, runtime)
    assert card.samples == 128 and len(frames) == 1
    port.pos = 128
    assert card.advance(link, runtime)
    assert card.samples == 256 and len(frames) == 2
    assert [frame.n0 for frame in frames] == [0, 128]


def test_virtual_card_stalled_owner_still_expires_wall_watchdog():
    card = Card()
    runtime = SimpleNamespace(_port=SimpleNamespace(pos=0, closed=False), pump=lambda: None)
    link = SimpleNamespace(lane=SimpleNamespace(push=lambda block: None))
    assert card.advance(link, runtime)
    wall = [0.]
    def tick(seconds):
        wall[0] += 5
        assert not card.advance(link, runtime)
    with pytest.raises(AssertionError, match='no byte progress'):
        await_confirmation(lambda: False, lambda: (0, 0), lambda: card.samples,
                           54 * 60000, clock=lambda: wall[0], sleep=tick)
    assert wall[0] == 15 and card.samples == 128


@pytest.mark.parametrize('failure', [None, 'cancelled', 'deadline'])
def test_virtual_card_explicit_wait_and_stalled_owner(failure):
    card = Card()
    port = SimpleNamespace(pos=0, closed=False)
    frames = []
    runtime = SimpleNamespace(_port=port, pump=lambda: None)
    link = SimpleNamespace(lane=SimpleNamespace(push=frames.append))
    # Exercise the real independent feeder, including its bounded stopped-owner
    # behavior. A wait requests capture; it never advances the clock itself.
    feeder = threading.Thread(target=card.run, args=(link, runtime), daemon=True)
    feeder.start()
    try:
        eventually(lambda: card.samples == 128)
        if failure is None:
            card.wait_until(384, deadline=time.monotonic() + 1)
            assert card.samples == 384
        else:
            options = ({'cancelled': lambda: True} if failure == 'cancelled'
                       else {'deadline': time.monotonic() - 1})
            with pytest.raises(BurstInvalidated, match=failure):
                card.wait_until(384, **options)
            assert card.samples == 128
        assert not card._wait_targets
        assert not card.advance(link, runtime)
        assert len(frames) * 128 == card.samples and card.failure is None
    finally:
        card.halt.set()
        feeder.join(1)
        assert not feeder.is_alive()


class Client:
    def __init__(self, endpoint):
        self.fd = os.open(endpoint, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self.decoder, self.counter = ptc.Deframer(), 0
        self.read()
        os.write(self.fd, b'PTC 31\rJHOST4\r')
        time.sleep(.05)
        self.read()

    def read(self):
        data = bytearray()
        while True:
            try:
                part = os.read(self.fd, 4096)
                if not part:
                    break
                data += part
            except BlockingIOError:
                break
        return bytes(data)

    def command(self, text):
        return self.exchange(text.encode(), command=True)

    def exchange(self, data, *, command=False):
        self.counter ^= 1
        os.write(self.fd, ptc.Packet(31, command, data, counter=self.counter).encode())
        replies = []
        def receive():
            replies.extend(self.decoder.feed(self.read()))
            return replies
        return eventually(receive)[0]

    def close(self):
        os.close(self.fd)


@pytest.fixture
def running(tmp_path, monkeypatch, request):
    from hfmodem.shrike import telemetry
    bind = telemetry.bind
    # Force wraparound during actual binary contacts instead of waiting hours.
    monkeypatch.setattr(telemetry, 'bind', lambda tx, rx: bind(tx, rx, capacity=4))
    card, rig = Card(), Keyer()
    config = SimpleNamespace(enabled=True, host='ptc', options={}, pty=str(tmp_path / 'ptc'))
    managed = getattr(request, 'param', None) == 'managed'
    if managed:
        token = tmp_path/'token'
        token.write_text(bytes(range(32)).hex())
        config.host, config.pty = 'ptc-managed', ''
        config.auth_token_file, config.port = str(token), 0
    station = SimpleNamespace(audio=card, rig=rig, lanes={},
        cfg=SimpleNamespace(station=SimpleNamespace(mycall='N0CALL'),
                            rig=SimpleNamespace(power_w=10), protocols={'pactor': config}))
    station.arbiter = TxArbiter(card, rig, drive=.6, settle_s=.04,
                               identity=Identity('K1ABC'))
    link = ShrikeLink(station)
    runtime = link.runtime
    assert isinstance(link.lane, StreamLane)
    station.lanes['pactor'] = link
    station.arbiter.active_protocols = lambda: tuple(
        name for name, lane in station.lanes.items() if lane.active)
    feeder = threading.Thread(target=card.run, args=(link, runtime), daemon=True)
    feeder.start()
    link.start()
    served = hosts._build_one('pactor', config, station)
    assert served.keys
    assert len(served.ports) == (1 if managed else 0)
    server = served.server
    server.start()
    assert served.endpoints == ((str(server.port),) if managed else (config.pty,))
    try:
        yield runtime, station, server
    finally:
        server.stop()
        link.stop()
        card.halt.set()
        feeder.join(2)
        assert not feeder.is_alive()
        assert not runtime._thread.is_alive()
        assert not rig.keyed and not card.closed and not rig.closed


def test_worker_serves_real_serial_and_reopens_a_fresh_attachment(running):
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        assert client.command('#MODE 0').code == ptc.OK
        assert client.command('#MYL 2').code == ptc.FAIL
        assert client.command('C %N0DX').code == ptc.FAIL
        assert runtime.host.arq.state == State.LISTENING
        assert not station.rig.edges and runtime.host.settings['MYLEVEL'] == '3'
        old_owner, old_ep, old_port = runtime.host._owner, runtime._endpoint, runtime._port
    finally:
        client.close()
    eventually(lambda: runtime._endpoint is not old_ep and not runtime._endpoint.closed)
    assert old_owner.retired and old_port.closed and old_ep.hangup
    assert runtime.host.settings['MODE'] == '0'
    again = Client(server.endpoint)
    try:
        assert again.command('L').code == ptc.MSG
        assert runtime.host._owner is not old_owner
        assert runtime.failure is None and station.audio.failure is None
    finally:
        again.close()


@pytest.mark.parametrize('running', ['managed'], indirect=True)
def test_managed_stream_runs_on_the_native_owner_and_retires_before_reopen(running):
    from hfmodem.tests.core.test_managed_host import Client as BrokerClient, read
    runtime, station, broker = running
    client = BrokerClient(broker)
    try:
        for attempt in range(2):
            control, lease = client.register()
            stream = client.stream(3, lease)
            banner = bytearray()
            while b'cmd:' not in banner:
                banner += stream.recv(4096)
            stream.sendall(b'PTC 31\rJHOST4\r')
            time.sleep(.02)
            stream.setblocking(False)
            while True:
                try:
                    if not stream.recv(4096):
                        break
                except BlockingIOError:
                    break
            stream.settimeout(2)
            stream.sendall(ptc.Packet(31, True, b'L', counter=1).encode())
            decoder = ptc.Deframer()
            replies = []
            while not replies:
                replies.extend(decoder.feed(stream.recv(4096)))
            assert replies[0].code == ptc.MSG
            owner, endpoint, port = runtime.host._owner, runtime._endpoint, runtime._port
            if attempt == 0:
                stream.sendall(ptc.Packet(31, True, b'C N0DX', counter=0).encode())
                eventually(lambda: station.rig.keyed)
            control.close()
            eventually(lambda: broker._lease is None)
            read(stream, 4096)       # Drain the preceding C reply through EOF.
            assert stream.recv(1) == b''
            assert endpoint.closed and owner.retired
            assert port is None or port.closed
            assert not runtime.active and not station.rig.keyed
            assert runtime._port is None
            assert runtime.failure is None and not broker.failure
    finally:
        client.close()


def test_serial_disconnect_during_native_carrier_retires_after_station_cw(running):
    from hfmodem.shrike import p1rx
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        assert client.command('C N0DX').code == ptc.OK
        eventually(lambda: station.rig.keyed)
        assert runtime.active and runtime.host._owner.contact_active
        assert client.command('D').code == ptc.OK
        eventually(lambda: len(station.audio.sent) == 2 and not runtime.active, timeout=10)
        assert runtime.host.arq.state == State.LISTENING
        assert runtime.host.arq._host_sync_packets == 1
        calls = p1rx.search_calls(np.pad(station.audio.sent[0][1], (12000, 12000)), 'N0DX')
        assert calls and calls[0][1].variant == 'Normal'
        assert station.arbiter.last_sent['ident'] == station.arbiter.transmit_history().end
        assert station.rig.edges == [True, False, True, False]
        assert not station.arbiter.identity.due
        assert runtime.failure is None and station.audio.failure is None
    finally:
        client.close()


def test_scanner_hold_retires_capture_until_explicit_release(running):
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        old = runtime._port
        assert client.command('%W0').text == '1'
        assert station.arbiter.scan_held and old.capture_retired
        time.sleep(.05)
        assert runtime._port is old and not old.closed
        assert client.command('C N0DX').code == ptc.FAIL
        assert not station.rig.edges
        assert client.command('%W1').code == ptc.MSG
        eventually(lambda: runtime._port is not None and runtime._port is not old)
        assert not station.arbiter.scan_held and old.closed
        assert runtime.failure is None and station.audio.failure is None
    finally:
        client.close()


def test_native_incoming_call_is_answered_with_monitoring_disabled(running):
    from hfmodem.shrike import pactor1, p1rx
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        assert client.command('%L0').code == ptc.OK
        signal = pactor1.connect_signal('N0CALL', invert=True, lead_s=0, tail_s=0)
        at = station.audio.samples + 24000
        def source(first, count):
            block = np.zeros(count, np.float32)
            lo, hi = max(first, at), min(first + count, at + len(signal))
            if hi > lo:
                block[lo-first:hi-first] = signal[lo-at:hi-at]
            return block
        station.audio.source = source
        eventually(lambda: station.rig.keyed, timeout=10)
        assert runtime.active and runtime.host.arq.answering
        assert runtime.host._owner.contact_active and not runtime.host.listening
        assert client.command('DD').code == ptc.OK
        from hfmodem.core import cwid
        budget = retirement_sample_budget(cwid.duration('K1ABC'), forced=True)
        def retired():
            assert runtime.failure is None and station.audio.failure is None
            return len(station.audio.sent) >= 2 and not runtime.active
        used, elapsed = await_confirmation(
            retired, lambda: (int(runtime.host.arq.state == State.LISTENING),
                              min(len(station.audio.sent), 2)),
            lambda: station.audio.samples, budget, stall_seconds=budget / 48000)
        print(f'forced retirement: samples={used}/{budget}, wall={elapsed:.3f}s')
        assert not station.rig.keyed
        emitted_at, samples = station.audio.sent[0]
        assert abs(emitted_at - (at + len(signal) + round(.09 * 48000))) <= 128
        assert len(samples) < p1rx.CALL_N  # actual native CS answer, not a call
        decoded = p1rx.decode_control_signal(np.pad(samples, (4800, 4800)),
                                            .1, len(samples) / 48000)
        assert decoded is not None and decoded.errors == 0
        assert decoded.index == pactor1.CS_ACK_A
        assert runtime.host.arq.state == State.LISTENING
        assert runtime.failure is None and station.audio.failure is None
    finally:
        client.close()


def test_contype_rejection_does_not_enter_a_native_contact_or_send_cw(running):
    from hfmodem.shrike import pactor1
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        assert client.command('#CONT 2').code == ptc.OK
        signal = pactor1.connect_signal('N0CALL', lead_s=0, tail_s=0)
        at = station.audio.samples + 12000
        def source(first, count):
            block = np.zeros(count, np.float32)
            lo, hi = max(first, at), min(first + count, at + len(signal))
            if hi > lo:
                block[lo-first:hi-first] = signal[lo-at:hi-at]
            return block
        station.audio.source = source
        eventually(lambda: station.audio.samples > at + len(signal) + 24000)
        assert not station.rig.edges and not runtime.active
        assert runtime.host.arq.state == State.LISTENING
        assert client.command('L').code == ptc.MSG
        assert runtime.failure is None and station.audio.failure is None
    finally:
        client.close()


def test_stop_wakes_a_worker_waiting_before_host_publication(tmp_path):
    card, rig = Card(), Keyer()
    station = SimpleNamespace(audio=card, rig=rig)
    station.arbiter = TxArbiter(card, rig, identity=Identity('K1ABC'))
    runtime = StationPtcRuntime(SimpleNamespace(station=station))
    runtime.start()
    runtime.stop()
    runtime.stop()
    assert not runtime._thread.is_alive() and runtime.host is None
    assert not rig.edges


@pytest.mark.parametrize('retirement', ['hangup', 'stop'])
def test_attachment_or_station_loss_during_key_unkeys_without_another_carrier(running, retirement):
    runtime, station, server = running
    client = Client(server.endpoint)
    assert client.command('C N0DX').code == ptc.OK
    eventually(lambda: station.rig.keyed)
    old_endpoint = runtime._endpoint
    old_port = runtime._port
    if retirement == 'hangup':
        client.close()
        eventually(lambda: runtime._endpoint is not old_endpoint
                   and not runtime._endpoint.closed)
        assert old_endpoint.hangup and runtime.failure is None
        # The station main thread has not discharged the owed ID in this bench.
        # A fresh attachment may configure/query, but cannot start another call.
        again = Client(server.endpoint)
        try:
            response = again.command('C N0NEXT')
            assert response.code == ptc.FAIL and 'IDENTIFICATION PENDING' in response.text
            assert again.command('%W0').text == '0'
        finally:
            again.close()
    else:
        runtime.stop()
        client.close()
        assert not runtime._thread.is_alive()
    assert old_port.closed and station.rig.edges == [True, False]
    assert not station.rig.keyed and station.arbiter.identity.due
    assert not station.audio.closed and not station.rig.closed


def test_factory_preserves_an_existing_operator_endpoint(tmp_path):
    card, rig = Card(), Keyer()
    path = tmp_path / 'operator'
    path.write_text('keep me')
    station = SimpleNamespace(audio=card, rig=rig, lanes={},
        cfg=SimpleNamespace(station=SimpleNamespace(mycall='N0CALL')))
    station.arbiter = TxArbiter(card, rig, identity=Identity('K1ABC'), settle_s=.04)
    link = SimpleNamespace(station=station, waveform=SimpleNamespace(rate=48000))
    runtime = StationPtcRuntime(link)
    with pytest.raises(RuntimeError, match='endpoint failed'):
        runtime.start_host(path)
    assert path.read_text() == 'keep me'
    assert not runtime._thread.is_alive() and runtime._port is None
    assert runtime._endpoint is None and not rig.edges


def test_all_byte_values_cross_native_incoming_packets_and_real_pty_once(running):
    from hfmodem.shrike import compress, pactor1
    runtime, station, server = running
    client = Client(server.endpoint)
    try:
        assert client.command('#MYL 1').code == ptc.OK
        assert client.command('#MODE 0').code == ptc.OK
        original = bytes(range(256))
        wire = b'1n0dx\r' + compress.transparent(original)
        capacity = pactor1.DATA_FIELD[200]
        fields = [wire[n:n+capacity] for n in range(0, len(wire), capacity)]
        # A waveform-only peer: the first frame identifies the called station;
        # data identifies the caller. Repeated packets model a missed ACK.
        # No modem event, logical connection or delivery callback is injected.
        bursts = [(0, pactor1.connect_signal('N0CALL', invert=True, lead_s=0, tail_s=0))]
        slot = 1
        for number, field in enumerate(fields, 1):
            for _ in range(2):
                bursts.append((slot * 60000, pactor1.packet_signal(field, 200,
                    packet_count=number % 4, invert=not slot % 2,
                    lead_s=0, tail_s=0)))
                slot += 1
        at = station.audio.samples + 24000
        retirement = {'armed': False, 'at': None}
        def source(first, count):
            if retirement['armed'] and retirement['at'] is None and not runtime.active:
                # The independent feeder latches completion; a delayed test
                # poll must not charge later idle capture to this contact.
                retirement['at'] = first + count
            block = np.zeros(count, np.float32)
            for offset, samples in bursts:
                start = at + offset
                lo, hi = max(first, start), min(first + count, start + len(samples))
                if hi > lo:
                    block[lo-first:hi-first] += samples[lo-start:hi-start]
            return block
        station.audio.source = source
        received, links = bytearray(), []
        def poll():
            reply = client.command('G')
            if reply.code == ptc.DATA:
                received.extend(reply.data)
            elif reply.code == ptc.LINK:
                links.append(reply.text)
            assert runtime.failure is None and station.audio.failure is None
            return len(received) >= len(original)
        eventually(poll, timeout=60)
        assert bytes(received) == original
        assert links == ['(31) CONNECTED to N0DX']
        assert runtime.host.arq.dxcall == 'N0DX' and runtime.host._link_up
        # Let the final duplicate reach the native decoder before final poll.
        eventually(lambda: station.audio.samples >= at + slot * 60000, timeout=5)
        assert client.command('G').code == ptc.OK
        start_samples = station.audio.samples
        retirement['armed'] = True
        assert client.command('DD').code == ptc.OK
        from hfmodem.core import cwid
        budget = retirement_sample_budget(cwid.duration('K1ABC'), forced=True)
        def retired():
            assert runtime.failure is None and station.audio.failure is None
            return retirement['at'] is not None and not runtime.active
        used, elapsed = await_confirmation(
            retired, lambda: (int(runtime.host.arq.state == State.LISTENING),
                              int(retirement['at'] is not None)),
            lambda: station.audio.samples if retirement['at'] is None else retirement['at'],
            budget, start_samples=start_samples, stall_seconds=budget / 48000)
        print(f'incoming forced retirement: samples={used}/{budget}, wall={elapsed:.3f}s')
        assert runtime.host.arq.state == State.LISTENING
        assert not runtime.host.arq.transfer_unconfirmed
        assert runtime.failure is None and not station.rig.keyed
    finally:
        client.close()


@pytest.mark.parametrize('mode', [0, 2, 'retire', 'graceful', 'cwid'])
def test_binary_outgoing_peer_confirmation_and_echo_over_native_audio(running, mode):
    from hfmodem.shrike import compress, pactor1, p1rx, spec
    runtime, station, server = running
    client = Client(server.endpoint)
    received, packets, answers, cw_bursts = bytearray(), [], [], []
    decoder, supervisor = compress.Decoder(), compress.Supervisor()
    previous = None
    dropped = False
    finish = station.audio.finish_burst
    def emitted():
        nonlocal previous, dropped
        end = finish()
        at, samples = station.audio.sent[-1]
        audio = np.pad(samples, (4800, 4800))
        decoded = p1rx.decode_p1_packets(audio)
        if decoded:
            packet = decoded[0]
            packets.append(packet)
            key = packet.packet_count
            new = key != previous
            text = b''
            if new and not packet.status & spec.STATUS_QRT:
                text = supervisor.feed(decoder.feed(packet.payload, packet.data_type,
                                                    protocol='PACTOR-1'))
                received.extend(text)
                previous = key
            if packet.status & spec.STATUS_QRT:
                assert pactor1.qrt_address(packet.payload) == 'N0DX'
            # The peer received this packet but its first ACK is lost. Only
            # its repeated, decoded packet gets the answer; no logical event
            # reaches the replacement outside captured audio.
            if new and b'\x00' in text and not dropped:
                dropped = True
                return end
            word, inverted = key & 1, packet.inverted
        else:
            call = p1rx.decode_connect(audio)
            if call is None:
                if mode == 'cwid':
                    cw_bursts.append(samples.copy())
                return end             # closing CW gets no fabricated answer
            assert call.callsign == 'N0DX' and call.variant == 'Normal'
            word, inverted = pactor1.CS_ACK_A, call.inverted
        answer = np.trim_zeros(pactor1.control_signal(word, invert=inverted))
        nonzero = np.flatnonzero(samples)
        first = at + (int(nonzero[-1]) + 1 if nonzero.size else len(samples)) + 5040
        answers.append((first, answer))
        return end
    station.audio.finish_burst = emitted
    def source(first, count):
        block = np.zeros(count, np.float32)
        for at, samples in tuple(answers):
            lo, hi = max(first, at), min(first + count, at + len(samples))
            if hi > lo:
                block[lo-first:hi-first] += samples[lo-at:hi-at]
        return block
    station.audio.source = source
    original = bytes(range(256)) + b'the quick brown fox jumps over the lazy dog\r' * 3
    try:
        if mode == 'cwid':
            station.arbiter.identity.mycall = 'N0CALL'
            assert client.command('#CWS 70').code == ptc.OK
            assert client.command('#FSKA 750').code == ptc.OK
            assert client.command('#CWID 4 2').code == ptc.OK
        assert client.command('#MYL 1').code == ptc.OK
        assert client.command(f'#MODE {mode if isinstance(mode, int) else 0}').code == ptc.OK
        assert client.command('%M1').code == ptc.OK
        assert client.command('C N0DX').code == ptc.OK
        links = []
        def connected():
            reply = client.command('G')
            if reply.code == ptc.LINK:
                links.append(reply.text)
            assert runtime.failure is None and station.audio.failure is None
            return bool(links)
        eventually(connected, timeout=15)
        assert links == ['(31) CONNECTED to N0DX']
        if mode == 'cwid':
            schedule, cw_tx = runtime.host._cw_schedule, runtime.tx
            # Inject expiry on the actual indexed clock. The schedule unit
            # test checks 420 s; this case exercises interruption/retry through
            # the real serial owner, native waveform and audio-only peer.
            schedule.deadline = station.audio.samples
        for offset in range(0, len(original), 128):
            assert client.exchange(original[offset:offset+128]).code == ptc.OK
        if mode == 'retire':
            import hashlib
            import json
            eventually(lambda: dropped, timeout=15)
            assert client.command('%T').text == '0'
            old_endpoint = runtime._endpoint
            client.close()
            client = None
            eventually(lambda: runtime._endpoint is not old_endpoint
                       and not runtime._endpoint.closed)
            reports = list(runtime.recovery_root.glob('contact-*/p3-transfer-unconfirmed.json'))
            assert len(reports) == 1
            report = json.loads(reports[0].read_text())
            originals = report['host_originals']
            assert (reports[0].parent / originals['file']).read_bytes() == original
            assert originals['sha256'] == hashlib.sha256(original).hexdigest()
            assert originals['host_accepted'] == len(original)
            assert originals['host_confirmed'] == 0
            assert runtime.host.arq.unconfirmed_transfers[-1]['host_unconfirmed'] == original
            assert runtime.host.sent_total == 0 and runtime.host.settings['%M'] == '0'
            assert not station.rig.keyed and runtime.failure is None
            return
        echoes = bytearray()
        def confirmed():
            reply = client.command('G')
            if reply.code == ptc.SENT_ECHO:
                echoes.extend(reply.data)
            assert runtime.failure is None and station.audio.failure is None
            return len(echoes) >= len(original)
        from hfmodem.core import cwid
        budget = confirmation_sample_budget(
            original, runtime.host.arq._payload_bytes(),
            cw_seconds=cwid.duration('N0CALL', 14) if mode == 'cwid' else 0.)
        used, elapsed = await_confirmation(
            confirmed, lambda: (len(received), len(echoes)),
            lambda: station.audio.samples, budget)
        print(f'confirmation mode={mode}: samples={used}/{budget}, wall={elapsed:.3f}s; '
              'wall no-byte-progress watchdog=15s')
        assert bytes(echoes) == original
        assert bytes(received) == b'1n0call\r' + original
        assert dropped and any(a.packet_count == b.packet_count
                               for a, b in zip(packets, packets[1:]))
        assert client.command('%T').text == str(len(original))
        assert client.command('G').code == ptc.OK
        assert any(packet.data_type != spec.DataType.ASCII_8BIT for packet in packets) == (mode == 2)
        graceful = mode in ('graceful', 'cwid')
        assert client.command('D' if graceful else 'DD').code == ptc.OK
        if graceful:
            terminal = []
            def disconnected():
                reply = client.command('G')
                if reply.code == ptc.LINK:
                    terminal.append(reply.text)
                return bool(terminal)
            if mode == 'cwid':
                # Account for the closing waveform before requiring worker retirement.
                def retired():
                    if not terminal:
                        disconnected()
                    assert runtime.failure is None and station.audio.failure is None
                    return bool(terminal) and not runtime.active
                close_budget = retirement_sample_budget(cwid.duration('N0CALL', 14))
                # A terminal phase may legitimately be silent for its entire
                # placement/retry/CW allowance. Its nominal duration is also a
                # wall liveness watchdog, not a measured scheduling/latency bound.
                close_watchdog = close_budget / 48000
                used, elapsed = await_confirmation(
                    retired, lambda: (len(terminal), schedule.completed),
                    lambda: station.audio.samples, close_budget,
                    stall_seconds=close_watchdog)
                print(f'retirement mode={mode}: samples={used}/{close_budget}, '
                      f'wall={elapsed:.3f}s; wall milestone watchdog={close_watchdog}s')
            else:
                eventually(disconnected, timeout=15)
            assert terminal == ['(31) DISCONNECTED fm N0DX']
            assert any(packet.status & spec.STATUS_QRT for packet in packets)
        eventually(lambda: not runtime.active, timeout=10)
        assert bytes(received) == b'1n0call\r' + original
        assert not runtime.host.arq.transfer_unconfirmed and runtime.failure is None
        if mode == 'cwid':
            from hfmodem.core import cwid, levels
            from hfmodem.shrike import onair
            expected = onair._trim_silence(levels.at_drive(
                cwid.audio('N0CALL', wpm=14, tone_hz=1400),
                cw_tx._fsk_drive * 10**(-3/20)))
            assert len(cw_bursts) == schedule.completed == 2
            for audio in cw_bursts:
                np.testing.assert_allclose(audio, expected, atol=2e-8, rtol=1e-6)
            assert runtime.host._cw_schedule is None and cw_tx._host_cw_schedule is None
    finally:
        if client is not None:
            client.close()
