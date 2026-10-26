# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from hfmodem.ax25.frame import ui_frame
from hfmodem.ax25.kiss import Parser, encode
from hfmodem.brubru import tx
from hfmodem.brubru.host.server import KissServer
from hfmodem.core.audio import Block, BurstInvalidated, StationAudio
from hfmodem.core.config import Station as StationSettings
from hfmodem.core.operations import LeaseTimeout, OperationGate, ResourceClosed
from hfmodem.core.rig import KeyToken
from hfmodem.core.tx_events import read_line
from hfmodem.station.arbiter import LIVE, RequestState, TxArbiter, TxRequest
from hfmodem.station.link import LINKS, RprLink
from hfmodem.tests.brubru.test_host import Socket


class Audio(StationAudio):
    def __init__(self):
        super().__init__(input_device=None, output_device="fake-output", blocksize=256)
        self.lat = 0
        self.output = []
        self.on_output = None

    def sample_now(self):
        return self.samples

    def step(self):
        out = np.zeros((self.blocksize, 1), np.float32)
        n0 = self.samples
        self._fill(out, self.blocksize, n0)
        self.samples += self.blocksize
        self.output.append(out[:, 0].copy())
        if self.on_output is not None:
            self.on_output(n0, out[:, 0].copy())

    def wait_until(self, index, **kwargs):
        while self.samples < index:
            if kwargs.get('cancelled', lambda: False)():
                raise BurstInvalidated('cancelled')
            self.step()

    def dac_time(self, output_frame):
        return time.monotonic() + (output_frame - self.samples) / 48000

    def dac_frame_at(self, timestamp):
        return self.samples + (timestamp - time.monotonic()) * 48000

    def wait_until_dac(self, output_frame, **kwargs):
        self.wait_until(output_frame, **kwargs)


class Rig:
    def __init__(self, audio, *, key_delay=0):
        self.audio = audio
        self.max_key_s = 40
        self.retired = False
        self.keyed = False
        self.line_state = 'low'
        self.key_delay = key_delay
        self.assertions = []
        self.panics = []

    def dial(self):
        return 10_147_600

    def key(self, emission, *, sample_now, **kwargs):
        self.audio.wait_until(self.audio.samples + round(self.key_delay * 48000))
        self.keyed, self.line_state = True, 'high'
        at = int(sample_now())
        self.assertions.append(at)
        return KeyToken(len(self.assertions), 0, self.max_key_s, 'test', at,
                        at + round(self.max_key_s * 48000))

    def unkey(self, token):
        self.keyed, self.line_state = False, 'low'

    def panic_unkey(self, why):
        self.panics.append(why)
        self.retired = True
        self.unkey(None)


class Cat:
    cache_free = reads_ptt = True

    def __init__(self):
        self.clock = lambda: 0

    def open(self):
        pass

    def close(self):
        pass

    def hold(self, deadline_index=None):
        return nullcontext(True)

    def protect(self, start_index, end_index):
        return object()

    def unprotect(self, token):
        pass

    def identify(self):
        return ''

    def freq(self):
        return 10_147_600

    def ptt_state(self, *, _arming=False):
        return False


def armed_rig(ptt):
    from hfmodem.core.rig import Rig as StationRig
    rig = StationRig(model='test', identity='', cat=Cat(), ptt=ptt, transmit=True,
                     profile=SimpleNamespace(check=lambda *args, **kwargs: None),
                     control=None)
    rig.arm(prove_ptt=False)
    return rig


def station(options=None, *, transmitter=True):
    audio = Audio()
    rig = Rig(audio) if transmitter else None
    cfg = SimpleNamespace(protocols={'rpr': SimpleNamespace(options=options or {})},
                          rig=SimpleNamespace(max_key_s=40, power_w=1,
                                              ptt=SimpleNamespace(settle_s=0.04)))
    st = SimpleNamespace(cfg=cfg, audio=audio, rig=rig,
                         arbiter=TxArbiter(audio, rig) if rig else None)
    st.link = RprLink(st)
    audio.subscribe(st.link.lane)
    return st


def feed(link, samples, chunk=4096, start=0):
    result = []
    for at in range(0, len(samples), chunk):
        link.lane.push(Block(0, start + at, samples[at:at + chunk]))
        result.extend(link.poll())
    return result


@pytest.mark.parametrize('chunk', [256, 4096, 48000])
def test_card_stream_positions_chunk_invariant(chunk):
    st = station(transmitter=False)
    raw = ui_frame('N0CALL', 'CQ', b'stream')
    burst, = tx.transmit([raw])
    audio = np.concatenate([np.zeros(48000), burst.card_audio(), np.zeros(48000)])
    got = feed(st.link, audio, chunk)
    assert [d.frames for _, d in got] == [(raw,)]
    assert all(type(position) is int for position, _ in got)
    assert got[0][0] == 5 * got[0][1].ref + st.link.decimator.DELAY
    assert st.link.receiver.buffered <= st.link.receiver.MAX_SAMPLES


@pytest.mark.parametrize('cause', ['drop', 'resync', 'timestamp', 'flush', 'gap'])
def test_fault_discards_through_next_boundary_and_next_burst_decodes(cause):
    st = station(transmitter=False)
    link = st.link
    raw = ui_frame('N0CALL', 'CQ', b'after fault')
    burst, = tx.transmit([raw])
    feed(link, burst.card_audio()[:48000])
    assert link.receiver.held
    if cause == 'drop':
        from collections import deque
        link.lane._q = deque(maxlen=2)
        for i in range(3):
            link.lane.push(Block(0, 48000 + i * 256, np.zeros(256)))
        start = 48768
    elif cause == 'gap':
        start = 50000
        link.lane.push(Block(0, start, np.zeros(256)))
        start += 256
    else:
        if cause == 'resync':
            link.lane.resync(48000)
        elif cause == 'flush':
            link.lane.flush_to(48000)
        else:
            link.lane.fault(48000, 'timestamp loss')
        link.lane.push(Block(0, 48000, np.zeros(256)))
        start = 48256
    link.poll()
    assert not link.receiver.held and link.rx_events
    got = feed(link, np.concatenate([np.zeros(48000), burst.card_audio(), np.zeros(48000)]),
               start=start)
    assert [d.frames for _, d in got] == [(raw,)]


def test_timestamp_loss_is_published_without_overflow():
    st = station(transmitter=False)
    audio = st.audio
    for timestamp in (1.0, 1 + 256 / 48000 + 0.1):
        timing = SimpleNamespace(inputBufferAdcTime=timestamp,
                                 outputBufferDacTime=timestamp)
        audio._body(np.zeros((256, 1)), np.zeros((256, 1)), 256, timing, None)
    assert audio.lost > 0 and audio.overflows == 0
    assert st.link.lane.faults[-1].cause == 'timestamp loss'
    assert not st.link.access.eligible(st.link.lane.fault_serial)


def request(audio, state=None, **kw):
    return TxRequest(LIVE, 1, 'rpr', np.ones(4800), emission=object(),
                     state=state or RequestState(), lead_s=0.1, **kw)


@pytest.mark.parametrize('gain', [0., .1, .25, 1.])
def test_host_gain_reaches_reserved_card_output(gain):
    audio = Audio()
    rig = Rig(audio)
    arbiter = TxArbiter(audio, rig, drive=.6)
    result = arbiter.submit_result(request(audio, gain=gain))
    assert result.outcome == 'sent' and not rig.keyed
    assert np.max(np.abs(np.concatenate(audio.output))) == pytest.approx(.6*gain)


def test_tracker_xr_reaches_the_rpr_worker_and_card_output(monkeypatch):
    from hfmodem.brubru.host.runtime import PacketRuntime
    st = station()
    runtime = PacketRuntime(st.link)
    # The real packet worker still renders, reserves and completes each burst;
    # deterministic access removes the random slot and receive-clock dependency.
    monkeypatch.setattr(st.link.access, 'contend', lambda *args: True)
    monkeypatch.setattr(st.link, '_fresh', lambda: None)
    monkeypatch.setattr(st.link, '_eligible', lambda: True)
    raw = ui_frame('W9SSJ', 'CQ', b'original')
    from hfmodem.station.link import RprRequest
    try:
        for millivolts in (3000, 300):
            runtime._configure('%XR', millivolts)
            st.audio.output.clear()
            request = RprRequest(millivolts, raw, RequestState(), time.monotonic())
            st.link._serve(request)
            assert request.completion[0] == 'sent'
            assert np.max(np.abs(np.concatenate(st.audio.output))) == pytest.approx(
                .6*millivolts/3000, rel=1e-5)
    finally:
        runtime.stop()


@pytest.mark.parametrize('limiter', [False, True])
@pytest.mark.parametrize('key_delay', [0, 0.05])
def test_ptt_to_first_card_sample_includes_keying_delay(limiter, key_delay):
    audio = Audio()
    rig = Rig(audio, key_delay=key_delay)
    arbiter = TxArbiter(audio, rig)
    raw = ui_frame('N0CALL', 'CQ', b'delay')
    burst, = tx.transmit([raw], limiter=limiter)
    req = TxRequest(LIVE, 1, 'rpr', burst.audio, 9600, object(),
                    state=RequestState(), lead_s=burst.lead_s(), settle_s=0.04)
    result = arbiter.submit_result(req)
    assert (result.outcome, result.state) == ('sent', 'committed')
    output = np.concatenate(audio.output)
    onset = np.flatnonzero(np.abs(output) >= 0.5 / 32767)[0]
    assert (onset - rig.assertions[0]) / 48000 == pytest.approx(0.1, abs=0.001)
    assert not rig.keyed


def test_reservation_rejects_before_publication_even_with_callback_contending():
    audio = Audio()
    comparing, release, callback_started = threading.Event(), threading.Event(), threading.Event()
    emitted = []
    errors = []
    class Deadline:
        def __lt__(self, end):
            comparing.set()
            assert release.wait(1)
            return True
    def reserve():
        try:
            audio.reserve_burst(np.ones(4800), 0, Deadline())
        except BurstInvalidated as exc:
            errors.append(exc)
    def callback():
        callback_started.set()
        out = np.empty((256, 1))
        audio._fill(out, 256, 768)
        emitted.extend(out[:, 0])
    reservation = threading.Thread(target=reserve)
    reservation.start()
    assert comparing.wait(1)
    worker = threading.Thread(target=callback)
    worker.start()
    assert callback_started.wait(1)
    time.sleep(0.01)
    assert not emitted
    release.set()
    reservation.join(1)
    worker.join(1)
    assert len(errors) == 1 and errors[0].displacement == 768
    assert audio._burst is None and not any(emitted)


def test_displaced_burst_ends_committed_error_without_output():
    audio = Audio()
    rig = Rig(audio)
    rig.max_key_s = 2.5
    original = rig.key
    def delayed(*args, **kwargs):
        token = original(*args, **kwargs)
        audio.samples += 3 * 48000
        return token
    rig.key = delayed
    arbiter = TxArbiter(audio, rig)
    result = arbiter.submit_result(request(audio))
    assert result.outcome == 'invalidated' and result.state == 'committed'
    assert result.displacement > 0 and audio._burst is None and not rig.keyed
    assert not audio.output


def test_cancel_wins_lock_wait_and_commit_wins_later_cancel():
    audio = Audio()
    arbiter = TxArbiter(audio, Rig(audio))
    state = RequestState()
    result = []
    with arbiter.try_hold() as held:
        assert held
        worker = threading.Thread(target=lambda: result.append(arbiter.submit_result(request(audio, state))))
        worker.start()
        assert state.cancel()
        worker.join(1)
        assert not worker.is_alive()
    assert result[0].reason == 'cancelled'
    state = RequestState()
    assert state.commit() and not state.cancel() and state.value == 'committed'
    state = RequestState()
    assert state.cancel() and not state.commit() and state.value == 'cancelled'


def test_eligibility_runs_under_ownership_before_keying():
    audio = Audio()
    rig = Rig(audio)
    arbiter = TxArbiter(audio, rig)
    result = arbiter.submit_result(request(audio, eligible=lambda: False))
    assert result.outcome == 'deferred' and result.state == 'waiting'
    assert not rig.assertions and not audio.output


def test_leases_drain_reject_new_calls_and_time_out():
    gate = OperationGate()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    def operation():
        with gate.lease():
            entered.set()
            release.wait(1)
    worker = threading.Thread(target=operation)
    worker.start()
    assert entered.wait(1)
    closer = threading.Thread(target=lambda: (gate.close(1), closed.set()))
    closer.start()
    time.sleep(0.02)
    assert not closed.is_set()
    with pytest.raises(ResourceClosed):
        with gate.lease():
            pass
    release.set()
    worker.join(1)
    closer.join(1)
    assert closed.is_set()
    with pytest.raises(ResourceClosed):
        with gate.lease():
            pass
    gate = OperationGate()
    with gate.lease():
        with pytest.raises(LeaseTimeout):
            gate.close(0.01)


def test_journal_queue_limit_cancel_and_no_default_file(tmp_path):
    journal = tmp_path / 'tx.log'
    st = station({'tx_events': str(journal)})
    for _ in range(17):
        st.link.enqueue(ui_frame('N0CALL', 'CQ', b'queued'))
    assert len(st.link._queue) == 16
    st.link.drop_pending()
    st.link.stop()
    events = [read_line(line) for line in journal.read_text().splitlines()]
    assert len(events) == 17 and len({e['tx_id'] for e in events}) == 17
    assert events[0]['outcome'] == 'refused'
    assert all(e['reason'] == 'cancelled' for e in events[1:])
    assert not st.link.events.pending
    assert station().link._journal_path is None


def test_options_validate_before_audio_and_rpr_is_registered():
    from hfmodem.core.config import ProtocolConfig
    from hfmodem.station.process import Station
    from hfhost.config import ConfigError
    assert LINKS['rpr'] is RprLink
    cfg = SimpleNamespace(protocols={'rpr': ProtocolConfig(enabled=True, host='kiss',
                                                          options={'level': 2})})
    st = Station(cfg)
    with pytest.raises(ConfigError, match='level'):
        st.open_audio()
    assert st.audio is None and st.rig is None
    for options in ({'unknown': 0}, {'limiter': 1}, {'level': True}, {'txdelay': -1}):
        with pytest.raises(ConfigError):
            RprLink.validate_options(options)
    assert replace(ProtocolConfig(), options={}).options == {}


def test_two_kiss_endpoints_move_ui_both_directions():
    a, b = station({'persistence': 255, 'slot_ms': 1}), station({'persistence': 255, 'slot_ms': 1})
    for source, target in ((a, b), (b, a)):
        noise = np.random.default_rng(5)
        source.audio.on_output = lambda n, values, target=target, noise=noise: (
            target.link.lane.push(Block(0, n, values + noise.normal(0, 0.001, len(values)))),
            target.link.poll())
        source.host = KissServer(source.link, log=lambda _: None)
        source.host._accept(Socket())
        source.link.host = source.host
    for source, target, payload in ((a, b, b'outbound'), (b, a, b'return')):
        begin = max(source.audio.samples, target.audio.samples, source.link._expected or 0)
        quiet = np.random.default_rng(9).normal(0, 0.001, 3 * 48000)
        feed(source.link, quiet, start=begin)
        source.audio.samples = begin + len(quiet)
        raw = ui_frame('N0CALL', 'CQ', payload)
        source.host._receive(encode(raw))
        source.link.start()
        deadline = time.monotonic() + 5
        while not source.link.outcomes and time.monotonic() < deadline:
            time.sleep(0.005)
        assert source.link.outcomes[-1]['outcome'] == 'sent'
        # Supply the receive history needed to release the complete final frame.
        source.audio.wait_until(source.audio.samples + 48000)
        target.host._send(0)
        assert Parser().feed(target.host._client.wire) == [(0, 0, raw)]
    a.link.stop()
    b.link.stop()


@pytest.mark.parametrize('stop', [None, 'lead', 'body', 'tail'])
def test_callback_health_covers_lead_body_and_playback_tail(monkeypatch, stop):
    import hfmodem.core.audio as module
    clock = [0.0]
    audio = StationAudio(input_device=None, output_device="fake-output", blocksize=256)
    audio.lat = 4800
    audio._stream = SimpleNamespace(active=True, latency=(0.01, 0.01))
    audio._callback_at = 0
    audio.sample_now = lambda: round(clock[0] * 48000)
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    at, length = 9600, 4800
    audio.reserve_burst(np.ones(length), at, 20000)
    next_callback = [0.0]
    def sleep(duration):
        clock[0] += duration
        while next_callback[0] <= clock[0]:
            n = round(next_callback[0] * 48000)
            stopped = ((stop == 'lead') or (stop == 'body' and n >= 5500) or
                       (stop == 'tail' and audio._consumed == length))
            if not stopped:
                audio._fill(np.zeros((256, 1)), 256, n)
                audio.samples = n + 256
                audio._dac = ((n + audio.loop_lat) / 48000, n)
            next_callback[0] += 256 / 48000
    monkeypatch.setattr(module.time, 'sleep', sleep)
    if stop is None:
        audio.wait_until(at + length, healthy=True, deadline=2)
        assert audio.finish_burst() == at + length
    else:
        with pytest.raises(BurstInvalidated, match='callbacks stopped'):
            audio.wait_until(at + length, healthy=True, deadline=2)
        audio.cancel_burst()


def test_consumption_not_position_controls_completion():
    audio = Audio()
    at = audio.reserve_burst(np.ones(256), 1000, 10000)
    audio._emitted = 256
    audio.samples = at + 256
    with pytest.raises(BurstInvalidated):
        audio.finish_burst()
    assert audio._burst is None


@pytest.mark.parametrize('results,attempts,outcomes', [
    ([('refused', 'waiting')], 1, ['refused']),
    ([('invalidated', 'waiting'), ('sent', 'committed')], 2, ['error', 'sent']),
    ([('invalidated', 'committed')], 1, ['error']),
    ([('invalidated', 'cancelled')], 1, ['error']),
    ([('deferred', 'waiting'), ('sent', 'committed')], 1, ['sent']),
])
def test_worker_outcomes_retries_and_deferrals(results, attempts, outcomes):
    from hfmodem.station.arbiter import TxResult
    st = station()
    st.link.access.contend = lambda *args: True
    transitions = iter(results)
    def submit(req):
        outcome, state = next(transitions)
        return TxResult(outcome, state, reason='test')
    st.arbiter.submit_result = submit
    queued = st.link.enqueue(ui_frame('N0CALL', 'CQ', b'outcome'))
    st.link._queue.clear()
    st.link._serve(queued)
    assert queued.attempts == attempts
    assert [e['outcome'] for e in st.link.outcomes] == outcomes
    assert queued.deferrals == sum(outcome == 'deferred' for outcome, _ in results)
    assert not st.link.events.pending


def test_a_burst_is_sized_to_what_the_arbiter_admits():
    from hfmodem.station.arbiter import TxResult
    frame = ui_frame('N0CALL', 'CQ', bytes(200))
    burst, = tx.transmit([frame])
    lead = burst.lead_s(100, 0.04)
    duration = len(burst.audio) / tx.waveform.RATE + 0.1
    for slack, offered in ((max(2, duration / 4) + 0.01, True), (0.5, False)):
        st = station()
        st.cfg.rig.max_key_s = st.rig.max_key_s = lead + duration + slack
        st.link.access.contend = lambda *args: True
        submitted = []
        st.arbiter.submit_result = lambda req: submitted.append(req) or TxResult('sent', 'committed')
        queued = st.link.enqueue(frame)
        st.link._queue.clear()
        st.link._serve(queued)
        assert bool(submitted) is offered
        assert st.link.outcomes[-1]['outcome'] == ('sent' if offered else 'refused')


def test_a_keyed_interval_too_short_for_any_burst_is_refused_as_such():
    st = station()
    st.cfg.rig.max_key_s = st.rig.max_key_s = 2
    st.link.access.contend = lambda *args: True
    st.arbiter.submit_result = lambda req: pytest.fail('offered a burst that cannot fit')
    queued = st.link.enqueue(ui_frame('N0CALL', 'CQ', b'x'))
    st.link._queue.clear()
    st.link._serve(queued)
    outcome = st.link.outcomes[-1]
    assert outcome['outcome'] == 'refused'
    assert outcome['reason'].startswith("burst cannot fit the rig's keyed interval")


def test_unknown_commit_state_never_retries():
    st = station()
    st.link.access.contend = lambda *args: True
    def submit(req):
        raise BurstInvalidated('state unavailable')
    st.arbiter.submit_result = submit
    queued = st.link.enqueue(ui_frame('N0CALL', 'CQ', b'unknown'))
    st.link._queue.clear()
    st.link._serve(queued)
    assert queued.attempts == 1 and queued.state.value == 'committed'
    assert st.link.outcomes[-1]['arbiter_state'] is None


def test_fifo_expiry_disconnect_and_stop_during_dcd_wait():
    st = station()
    frames = [ui_frame('N0CALL', 'CQ', bytes([n])) for n in range(3)]
    requests = [st.link.enqueue(raw) for raw in frames]
    assert [r.frame for r in st.link._queue] == frames
    requests[0].queued -= 61
    st.link.start()
    deadline = time.monotonic() + 2
    while (st.link._active is None or st.link._active.identifier != requests[1].identifier):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    assert st.link.outcomes[0]['reason'] == 'request expired'
    st.link.drop_pending()
    assert requests[1].state.value == 'waiting'
    assert requests[2].state.value == 'cancelled'
    st.link.stop()
    assert requests[1].state.value == 'cancelled'
    assert not st.link._worker.is_alive() and 'rpr' in st.arbiter.disabled


def test_stop_between_eligibility_and_commit_cancels_without_keying():
    audio = Audio()
    arbiter = TxArbiter(audio, Rig(audio))
    reached, release = threading.Event(), threading.Event()
    state = RequestState()
    def eligible():
        reached.set()
        release.wait(1)
        return True
    result = []
    worker = threading.Thread(target=lambda: result.append(arbiter.submit_result(
        request(audio, state, eligible=eligible))))
    worker.start()
    assert reached.wait(1)
    state.cancel()
    release.set()
    worker.join(1)
    assert result[0].reason == 'cancelled' and not arbiter.rig.assertions


def test_audio_calls_after_close_are_rejected():
    audio = StationAudio(input_device=None, output_device="fake-output")
    audio.close()
    for call in (lambda: audio.wait_until(0),
                 lambda: audio.reserve_burst(np.zeros(1), 0, 10000),
                 audio.finish_burst):
        with pytest.raises(ResourceClosed):
            call()


def test_rig_close_waits_for_admitted_assertion_and_rejects_later_calls():
    from hfmodem.core.regulatory import Emission
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    edges = []
    class Ptt:
        released_low = True
        def assert_(self, value):
            if value:
                entered.set()
                assert release.wait(1)
            edges.append(value)
        def sense(self):
            return False
        def release(self):
            edges.append(False)
    rig = armed_rig(Ptt())
    rig._arm_watchdog = lambda *args: None
    burst = Emission(10147600, 1200, 1800)
    worker = threading.Thread(target=lambda: rig.key(burst, why='test', duration_s=0.1))
    worker.start()
    assert entered.wait(1)
    closer = threading.Thread(target=lambda: (rig.close(), closed.set()))
    closer.start()
    time.sleep(0.02)
    assert not closed.is_set() and not edges
    release.set()
    worker.join(1)
    closer.join(1)
    assert edges == [True, False] and closed.is_set()
    with pytest.raises(ResourceClosed):
        rig.key(burst, why='after close', duration_s=0.1)
    with pytest.raises(ResourceClosed):
        rig.dial()


def test_rpr_receive_only_host_reports_no_transmitter():
    from hfmodem.core.config import ProtocolConfig
    from hfmodem.station.hosts import _build_one
    st = station(transmitter=False)
    st.lanes = {'rpr': st.link}
    served = _build_one('rpr', ProtocolConfig(enabled=True, host='kiss'), st)
    assert not served.keys and served.server.host == '127.0.0.1'
    st.rig = object()
    st.arbiter = object()
    served = _build_one('rpr', ProtocolConfig(enabled=True, host='kiss'), st)
    assert served.keys


def test_assessment_binds_rpr_sources_and_occupied_span():
    from hfmodem.core.rxreadiness import assessment_sources
    rpr = assessment_sources('rpr')
    assert 'brubru/rx.py' in rpr and 'brubru/access.py' in rpr
    assert not any(name.startswith('shrike/') for name in rpr)
    assert all(len(digest) == 64 for digest in rpr.values())


def test_stop_during_committed_burst_waits_for_completion():
    st = station({'persistence': 255, 'slot_ms': 1})
    feed(st.link, np.random.default_rng(11).normal(0, 0.001, 144000))
    st.audio.samples = 144000
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = st.audio.wait_until
    def wait(index, **kwargs):
        if kwargs.get('healthy'):
            entered.set()
            assert release.wait(1)
        return original(index, **kwargs)
    st.audio.wait_until = wait
    queued = st.link.enqueue(ui_frame('N0CALL', 'CQ', b'active'))
    st.link.start()
    assert entered.wait(1)
    closer = threading.Thread(target=lambda: (st.link.stop(), stopped.set()))
    closer.start()
    time.sleep(0.02)
    assert not stopped.is_set() and queued.state.value == 'committed'
    release.set()
    closer.join(2)
    assert stopped.is_set() and not st.rig.keyed
    assert st.link.outcomes[-1]['outcome'] == 'sent'


def test_fault_published_during_arbiter_contention_defers():
    st = station({'persistence': 255, 'slot_ms': 1})
    feed(st.link, np.random.default_rng(11).normal(0, 0.001, 144000))
    st.audio.samples = 144000
    result = []
    req = request(st.audio, eligible=st.link._eligible)
    with st.arbiter.try_hold() as held:
        assert held
        worker = threading.Thread(target=lambda: result.append(st.arbiter.submit_result(req)))
        worker.start()
        st.link.lane.fault(144000, 'timestamp loss')
    worker.join(1)
    assert result[0].outcome == 'deferred' and not st.rig.assertions
    assert not st.link.receiver.events


def test_ale_retune_invalidates_rpr_assessment_before_new_dial_can_key():
    from hfmodem.core import band
    from hfmodem.station.link import AleLink

    st = station({'persistence': 255, 'slot_ms': 1})
    old_dial, new_dial = 7_100_000, 7_102_000
    dial = [old_dial]
    st.rig.dial = lambda: dial[0]
    st.rig.tune = lambda value: dial.__setitem__(0, value - band.DIAL_OFFSET_HZ)
    st.audio.samples = 3 * 48000
    st.link.access.feed(0, np.random.default_rng(11).normal(0, 0.001, 3 * 48000))
    assert st.link.access.clear(st.link.lane.fault_serial)

    ale = object.__new__(AleLink)
    ale.station = st
    ale.dials = [new_dial]
    st.lanes = {'rpr': st.link, 'ale': ale}
    assert ale._tune(0) == st.audio.samples
    assert dial[0] == new_dial
    assert st.link.access.floor is None
    assert not st.link._eligible()

    req = request(st.audio, eligible=st.link._eligible)
    result = st.arbiter.submit_result(req)
    assert result.outcome == 'deferred'
    assert st.rig.assertions == []

    # An already queued A chunk may reach RPR after the dial moved.
    st.link.access.feed(2 * 48000,
                        np.random.default_rng(13).normal(0, 0.001, 48000))
    assert st.link.access.floor is None

    # Only new-dial samples can rebuild the floor and clear the retune barrier.
    st.link.access.feed(st.audio.samples,
                        np.random.default_rng(12).normal(0, 0.001, 2 * 48000))
    assert st.link.access.clear(st.link.lane.fault_serial)


def test_journal_producers_record_and_drain_under_one_lock(tmp_path):
    journal = tmp_path / 'events.log'
    st = station({'tx_events': str(journal)})
    requests = [st.link.enqueue(ui_frame('N0CALL', 'CQ', b'x')) for _ in range(16)]
    def record(subset):
        for req in subset:
            st.link._record(req, 'refused', 'concurrent producer')
    left = threading.Thread(target=record, args=(requests[:8],))
    right = threading.Thread(target=record, args=(requests[8:],))
    left.start()
    right.start()
    left.join(1)
    right.join(1)
    rows = [read_line(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 16 and {r['tx_id'] for r in rows} == set(range(1, 17))
    assert not st.link.events.pending


@pytest.mark.parametrize('duration,depth', [(1, 0.1), (0.3, 0)])
def test_dcd_holds_through_actual_frame_fades(duration, depth):
    from hfmodem.core.resample import for_card
    from hfmodem.tests.brubru.bench import impair
    from hfmodem.tests.brubru.vectors import frames_for
    st = station(transmitter=False)
    frames = frames_for(825)
    burst, = tx.transmit(frames)
    audio = for_card(impair(burst.audio, snr=10, seed=9, notch_s=duration, depth=depth), 9600)
    mid = len(audio) // 2
    got = []
    for at in range(0, len(audio), 4096):
        st.link.lane.push(Block(0, at, audio[at:at + 4096]))
        got.extend(st.link.poll())
        if mid - duration * 24000 < at < mid + duration * 24000:
            assert st.link.receiver.held and not st.link.access.clear()
    got.extend(feed(st.link, np.zeros(48000), start=len(audio)))
    assert [list(d.frames) for _, d in got] == [frames]


def test_completion_rechecks_callback_health_after_playback_wait():
    audio = Audio()
    at = audio.reserve_burst(np.ones(256), 1000, 10000)
    audio.wait_until(at + 256)
    audio._callback_at -= 1
    with pytest.raises(BurstInvalidated, match='callbacks stopped'):
        audio.finish_burst()
    audio.cancel_burst()
    assert audio._burst is None


def test_station_builds_and_polls_rpr_lane_from_software_replay():
    from hfmodem.core.config import ProtocolConfig, Station as StationOptions
    from hfmodem.station.process import Station

    raw = ui_frame('N0AAA', 'N0BBB', b'station replay')
    burst, = tx.transmit([raw])
    cfg = SimpleNamespace(protocols={'rpr': ProtocolConfig(enabled=True, host='kiss')},
                          station=StationOptions(mycall='N0CALL', control='local',
                                                 regulatory='part97',
                                                 regulatory_settings={'licence': 'technician'},
                                                 transmit=False),
                          selected_rig='ft891',
                          rig=SimpleNamespace(max_key_s=40,
                                              ptt=SimpleNamespace(settle_s=0.04)))
    st = Station(cfg, replay=np.concatenate((burst.card_audio(), np.zeros(48000))))
    try:
        st.open_audio()
        st.build_lanes()
        link = st.lanes['rpr']
        assert isinstance(link, RprLink) and link._worker.is_alive()
        while st.audio.pump(32):
            link.poll()
        link.poll()
        assert [frame for _, decoded in link.received for frame in decoded.frames] == [raw]
        assert st.rig is None and st.arbiter is None
    finally:
        st.shutdown()
    assert not link._worker.is_alive()


@pytest.mark.parametrize('failure', ['construct', 'arm'])
def test_station_releases_ptt_when_rig_startup_fails(monkeypatch, failure):
    from hfmodem.core.rig import RigError
    from hfmodem.station import process

    lines = []

    class Line:
        def __init__(self, port):
            self.released = False
            lines.append(self)

        def open(self):
            pass

        def release(self):
            self.released = True

    class Radio:
        def __init__(self, *, ptt, **kw):
            if failure == 'construct':
                raise RigError('construction failed')
            self.ptt = ptt

        def arm(self, **kw):
            raise RigError('arming failed')

        def close(self):
            self.ptt.release()

    monkeypatch.setattr(process, 'RtsPtt', Line)
    monkeypatch.setattr(process, 'RigctldCat', lambda *a, **kw: SimpleNamespace(
        open=lambda: None, close=lambda: None))
    monkeypatch.setattr(process, 'Rig', Radio)
    cfg = SimpleNamespace(
        selected_rig='ft891', protocols={}, resolved_profile={},
        station=StationSettings(transmit=True, mycall='N0CALL'),
        rig=SimpleNamespace(ptt=SimpleNamespace(port='/dev/test'), model='ft891',
                            host='localhost', port=4532, max_key_s=40, cat='rigctld',
                            max_watts=100, identity='', dial_max_age_s=5),
        profile=None, control=None)
    with pytest.raises(RigError):
        process.Station(cfg).open_rig()
    assert len(lines) == 1 and lines[0].released


def test_card_deadline_does_not_add_watchdog_setup_time(monkeypatch):
    from hfmodem.core.regulatory import Emission

    clock = [0.0]
    def advance(*args):
        clock[0] += 0.05
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    ptt = SimpleNamespace(assert_=advance, sense=lambda: False, release=lambda: None,
                          released_low=True)
    rig = armed_rig(ptt)
    rig._arm_watchdog = advance
    try:
        token = rig.key(Emission(10147600, 1200, 1800), why='test', duration_s=1,
                        sample_now=lambda: round(clock[0] * 48000))
        assert token.card_index == 4800
        assert abs(token.card_deadline - round(token.deadline * 48000)) <= 1
    finally:
        rig.close()


@pytest.mark.parametrize('stall', [False, True])
def test_callback_health_with_three_ms_rms_delivery_jitter(monkeypatch, stall):
    import hfmodem.core.audio as module

    clock = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    audio = StationAudio(input_device=None, output_device="fake-output", blocksize=128)
    audio.lat = 1152
    audio._stream = SimpleNamespace(active=True, latency=(0.01, 0.01))
    audio.sample_now = lambda: round(clock[0] * 48000)
    # Reproducible delivery offsets at the measured 3 ms RMS, preserving order.
    nominal = np.arange(400) * 128 / 48000
    offsets = np.random.default_rng(71).normal(0, 0.003, len(nominal))
    delivered = np.maximum.accumulate(nominal + 0.010 + offsets)
    assert np.max(np.diff(delivered)) > 2 * 128 / 48000
    pending = iter(enumerate(delivered))
    next_delivery = [next(pending)]
    def sleep(duration):
        clock[0] += duration
        while next_delivery[0][1] <= clock[0]:
            index, when = next_delivery[0]
            if not stall or when < 0.3:
                audio._fill(np.zeros((128, 1), np.float32), 128, index * 128)
                audio.samples = (index + 1) * 128
                audio._dac = ((index * 128 + audio.loop_lat) / 48000, index * 128)
            next_delivery[0] = next(pending)
    monkeypatch.setattr(module.time, 'sleep', sleep)
    for _ in range(50):
        sleep(0.001)
    at = audio.reserve_burst(np.ones(24000), 4800, 48000)
    if stall:
        with pytest.raises(BurstInvalidated, match='callbacks stopped'):
            audio.wait_until_dac(at - audio.loop_lat + 24000, healthy=True, deadline=1)
        assert 0.3 < clock[0] < 0.35
        audio.cancel_burst()
    else:
        audio.wait_until_dac(at - audio.loop_lat + 24000, healthy=True, deadline=1)
        assert audio.finish_burst() == at + 24000


@pytest.mark.parametrize('latency', [0.01, 0.08])
def test_callback_allowance_tracks_reported_output_latency(monkeypatch, latency):
    import hfmodem.core.audio as module

    audio = StationAudio(input_device=None, output_device="fake-output", blocksize=128)
    audio._stream = SimpleNamespace(active=True, latency=(0.001, latency))
    audio._callback_at = 0
    timeout = latency + audio.blocksize / module.CARD_RATE_HZ + module.CALLBACK_JITTER_ALLOWANCE_S
    monkeypatch.setattr(module.time, 'monotonic', lambda: timeout - 0.001)
    audio.check_callbacks()
    monkeypatch.setattr(module.time, 'monotonic', lambda: timeout + 0.001)
    with pytest.raises(BurstInvalidated, match='callbacks stopped'):
        audio.check_callbacks()


def test_nonstation_rig_close_releases_ptt_after_drain_timeout(monkeypatch):
    from hfmodem.core.rig import Rig as StationRig

    calls = []
    rig = StationRig(model='test', cat=SimpleNamespace(close=lambda: calls.append('cat')),
                     ptt=SimpleNamespace(release=lambda: calls.append('ptt')),
                     transmit=False, profile=None, control=None)
    assert rig.fatal is None
    rig._timer = SimpleNamespace(cancel=lambda: calls.append('watchdog'))
    # An actual admitted operation, with a zero close budget for the test.
    drain = rig.operations.close
    monkeypatch.setattr(rig.operations, 'close', lambda timeout: drain(0))
    with rig.operations.lease():
        with pytest.raises(LeaseTimeout, match='admitted resource operation did not drain'):
            rig.close()
    assert calls == ['ptt']
    assert rig._timer is not None
    assert rig.retired


def test_nonstation_drain_timeout_bounds_blocked_ptt_release(monkeypatch):
    import hfmodem.core.rig as rig_module
    from hfmodem.core.rig import Rig as StationRig

    entered, unblock = threading.Event(), threading.Event()
    calls = []

    def blocked_release():
        entered.set()
        unblock.wait(2)

    rig = StationRig(model='test', cat=SimpleNamespace(close=lambda: calls.append('cat')),
                     ptt=SimpleNamespace(release=blocked_release),
                     transmit=False, profile=None, control=None)
    assert rig.fatal is None
    rig._timer = SimpleNamespace(cancel=lambda: calls.append('watchdog'))
    drain = rig.operations.close
    monkeypatch.setattr(rig.operations, 'close', lambda timeout: drain(0))
    monkeypatch.setattr(rig_module, 'RELEASE_BUDGET_S', 0.01)
    try:
        with rig.operations.lease():
            with pytest.raises(LeaseTimeout, match='keying line did not release'):
                rig.close()
        assert entered.is_set()
        assert calls == []
        assert rig._timer is not None
        assert rig.retired
    finally:
        unblock.set()


def test_blocked_release_reaches_station_fatal_path_with_watchdog_armed():
    from hfmodem.core.rig import Rig as StationRig
    from hfmodem.station.process import Station

    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked_release():
        entered.set()
        release.wait(2)

    rig = StationRig(model='test', cat=SimpleNamespace(close=lambda: calls.append('cat')),
                     ptt=SimpleNamespace(release=blocked_release),
                     transmit=False, profile=None, control=None)
    rig._timer = SimpleNamespace(cancel=lambda: calls.append('watchdog'))
    st = Station(SimpleNamespace(protocols={}))
    st.hosts, st.lanes, st.rig, st.audio = {}, {}, rig, None

    class FatalReached(BaseException):
        pass

    def fatal(reason):
        calls.append(('fatal', reason, rig._timer is not None))
        raise FatalReached

    st.fatal_shutdown = fatal
    try:
        with pytest.raises(FatalReached):
            st.shutdown()
        assert entered.is_set()
        assert calls[0] == 'cat'
        assert calls[-1][0] == 'fatal' and calls[-1][2] is True
        assert 'keying line did not release during close' in calls[-1][1]
        assert 'watchdog' not in calls
    finally:
        release.set()


def test_arbiter_public_results_and_immutable_receive_barrier():
    from dataclasses import FrozenInstanceError
    from hfmodem.station.arbiter import TxResult

    audio = Audio()
    arbiter = TxArbiter(audio, Rig(audio))
    before = arbiter.transmit_history()
    with pytest.raises(FrozenInstanceError):
        before.end = 1
    result = arbiter.submit_result(request(audio))
    assert type(result) is TxResult and result.outcome == 'sent'
    after = arbiter.transmit_history()
    assert after.sequence == before.sequence + 1 and after.end >= result.end
    assert before.end == 0
    arbiter.disable('rpr', 'test')
    result = arbiter.submit_result(request(audio))
    assert type(result) is TxResult and (result.outcome, result.state) == ('refused', 'waiting')
    assert 'disabled' in result.reason
    assert arbiter.transmit_history() == after
    assert not hasattr(arbiter, 'retired')


def test_protocol_options_are_validated_once_before_resources(monkeypatch):
    from hfmodem.core.config import ProtocolConfig, Station as StationOptions
    from hfmodem.station.process import Station

    calls = []
    monkeypatch.setattr(RprLink, 'validate_options', lambda options: calls.append(options))
    cfg = SimpleNamespace(protocols={'rpr': ProtocolConfig(enabled=True, host='kiss')},
                          station=StationOptions(mycall='N0CALL', control='local',
                                                 regulatory='part97',
                                                 regulatory_settings={'licence': 'technician'},
                                                 transmit=False), selected_rig='ft891',
                          rig=SimpleNamespace(max_key_s=40, centre_hz=7_102_000,
                                              ptt=SimpleNamespace(settle_s=0.04)))
    st = Station(cfg, replay=np.zeros(48000))
    st.open_audio()
    st.open_rig()
    st.build_lanes()
    st.shutdown()
    assert calls == [{}]


@pytest.mark.parametrize('name', list(LINKS))
def test_all_protocol_option_errors_use_config_error(name):
    from hfhost.config import ConfigError

    with pytest.raises(ConfigError):
        LINKS[name].validate_options({'unsupported': True})


def test_station_command_runs_rpr_configuration(monkeypatch, tmp_path):
    from hfmodem import cli
    from hfmodem.station.process import Station

    config = tmp_path / 'station.toml'
    config.write_text('schema = 1\n[station]\ntransmit = false\n'
                      'control = "local"\nregulatory = "unregulated"\nbecause = "software test"\n'
                      '[audio]\noutput = "fake output"\n'
                      '[protocols.rpr]\nenabled = true\nhost = "kiss"\nport = 8001\n')
    seen = []
    from hfmodem.station.service import Service
    monkeypatch.setattr(Service, 'bind', lambda self: None)
    monkeypatch.setattr(Station, 'run', lambda self, **kw: seen.append(self.cfg) or 0)
    assert cli.cmd_station(SimpleNamespace(file=str(config), target=None, detach=False, preflight=False)) == 0
    assert seen[0].protocols['rpr'].host == 'kiss'


@pytest.mark.parametrize('protocol', ['pactor', 'vara', 'ardop', 'sabir'])
def test_existing_lanes_keep_synchronous_end_index_and_refusal(protocol):
    from hfmodem.station.arbiter import Refused

    audio = Audio()
    arbiter = TxArbiter(audio, Rig(audio))
    req = TxRequest(LIVE, 1, protocol, np.ones(4800), emission=object())
    end = arbiter.submit(req)
    assert type(end) is int and end == 3 * audio.blocksize + 4800
    arbiter.disable(protocol, 'test')
    with pytest.raises(Refused, match='disabled'):
        arbiter.submit(req)


@pytest.mark.parametrize('reserved', [False, True])
def test_failed_key_up_immediately_runs_panic_release(reserved):
    from hfmodem.core.rig import RigError

    audio = Audio()

    class FailingRig:
        max_key_s = 40
        retired = False
        keyed = False
        line_state = 'low'

        def __init__(self):
            self.released = False

        def key(self, *args, **kwargs):
            self.retired, self.line_state = True, 'unconfirmed'
            raise RigError('PTT assertion failed')

        def unkey(self, token):
            pass

        def panic_unkey(self, why):
            self.released, self.line_state = True, 'low'

    rig = FailingRig()
    arbiter = TxArbiter(audio, rig)
    req = TxRequest(LIVE, 1, 'rpr', np.ones(4800), emission=object(),
                    state=RequestState() if reserved else None,
                    lead_s=0.1 if reserved else None)
    if reserved:
        result = arbiter.submit_result(req)
        assert result.outcome == 'invalidated' and 'retired' in result.reason
    else:
        with pytest.raises(BurstInvalidated, match='retired'):
            arbiter.submit(req)
    assert rig.released
