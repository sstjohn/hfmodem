# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR's existing producer uses station capture and the real arbiter policy."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core.audio import BurstInvalidated, StreamChunk, StreamFault
from hfmodem.shrike import onair, pactor1
from hfmodem.shrike.station_io import StationInput
from hfmodem.station.arbiter import LIVE, Refused, TxArbiter, TxRequest
from hfmodem.station.link import Waveform


class Capture:
    def __init__(self):
        self.samples = 0
        self.holdback, self.loop_lat, self.blocksize = 660, 1152, 128
        self.lat, self.tx_latency_n = 1152, 0
        self.closed = False
        self.pending = None
        self.sent, self.cancelled = [], 0
        self.before_wait = lambda: None

    def sample_now(self):
        return float(self.samples)

    def alive(self):
        pass

    def wait_until_dac(self, frame, **options):
        self.wait_until(frame + self.loop_lat, **options)

    def wait_until(self, index, **options):
        self.before_wait()
        self.samples = max(self.samples, index)

    def arm_burst(self, audio, at):
        self.pending = (at, np.array(audio, copy=True))
        return at

    def cancel_burst(self):
        self.pending = None
        self.cancelled += 1

    def finish_burst(self):
        if self.pending is None:
            return self.samples
        at, samples = self.pending
        self.sent.append(self.pending)
        self.pending = None
        return at + len(samples)


class Keyer:
    max_key_s = 40
    retired = keyed = closed = False
    line_state = 'low'

    def __init__(self):
        self.edges = []

    def dial(self):
        return 7100000

    def key(self, *args, **kwargs):
        self.keyed = True
        self.line_state = 'high'
        self.edges.append(True)
        return object()

    def unkey(self, token):
        self.keyed = False
        self.line_state = 'low'
        self.edges.append(False)


class PollingCapture(Capture):
    """Indexed card with the actual wait callback/deadline contract."""
    def __init__(self):
        super().__init__()
        self.wait_options = []

    def wait_until(self, index, **options):
        self.wait_options.append(options)
        self.before_wait()
        while self.samples < index:
            if options.get('cancelled') is not None and options['cancelled']():
                raise BurstInvalidated('cancelled')
            if options.get('deadline') is not None and time.monotonic() >= options['deadline']:
                raise BurstInvalidated('audio wait deadline expired')
            self.samples = min(index, self.samples + 480)


def port(*, capture_factory=Capture, **options):
    audio, rig = capture_factory(), Keyer()
    station = SimpleNamespace(audio=audio, rig=rig,
        cfg=SimpleNamespace(declared_power_w=10, rig=SimpleNamespace(power_w=10)))
    station.arbiter = TxArbiter(audio, rig, drive=.6)
    link = SimpleNamespace(station=station, waveform=Waveform('pactor', 48000))
    return StationInput(link, **options), station


def chunk(at, samples, *faults):
    return StreamChunk(at, np.asarray(samples, np.float32), faults, ())


def test_indexed_reads_preserve_partials_and_copy_capture():
    live, st = port()
    first = np.arange(9, dtype=np.float32)
    live.feed(chunk(0, first))
    live.feed(chunk(9, [9, 10, 11]))
    first[:] = -1
    np.testing.assert_array_equal(live.read(4), np.arange(4))
    np.testing.assert_array_equal(live.take_until(10), np.arange(4, 10))
    np.testing.assert_array_equal(live.read(2), [10, 11])
    assert live.pos == 12
    # Reads never advance the station's clock or tick an ARQ engine.
    assert st.audio.samples == 0 and not st.rig.edges
    assert live.key_notice == 1536 and live.holdback == 660


def test_capture_handoff_may_cross_threads_but_consumer_state_cannot():
    live, _ = port()
    producer = threading.Thread(target=lambda: live.feed(chunk(0, [1, 2])))
    producer.start()
    producer.join()
    assert live.read(1).tolist() == [1]
    failures = []

    def intruder():
        for operation in (lambda: live.read(1), lambda: live.flush_to(1), live.close):
            try:
                operation()
            except RuntimeError as exc:
                failures.append(str(exc))

    other = threading.Thread(target=intruder)
    other.start()
    other.join()
    assert len(failures) == 3 and not live.closed
    assert live.read(1).tolist() == [2]


@pytest.mark.parametrize('cause', ['index gap', 'deque drop', 'resync', 'flush'])
def test_capture_faults_retire_instead_of_creating_a_new_raster(cause):
    live, st = port()
    live.feed(chunk(0, [1, 2], StreamFault(0, cause, 1)))
    with pytest.raises(ConnectionAbortedError, match=cause):
        live.read(1)
    assert live.closed and live._frames.empty()
    assert not st.rig.edges and not st.audio.closed and not st.rig.closed


def test_unreported_index_gap_also_fails_and_preserves_shared_devices():
    live, st = port()
    live.feed(chunk(4, [4, 5]))
    with pytest.raises(ConnectionAbortedError, match='index gap: 0..4'):
        live.read(2)
    assert live.closed and not st.audio.closed and not st.rig.closed


def test_duplicate_capture_cannot_replay_audio_or_silently_rebase_the_clock():
    live, _ = port()
    live.feed(chunk(0, [0, 1, 2]))
    assert live.read(3).tolist() == [0, 1, 2]
    live.feed(chunk(1, [1, 2, 3]))
    with pytest.raises(ConnectionAbortedError, match='overlap'):
        live.read(1)


def test_ready_reads_do_not_wait_for_unhanded_capture_or_discard_a_suffix():
    live, st = port()
    st.audio.samples = 10000
    live.feed(chunk(0, [0, 1, 2]))
    assert live.read(1).tolist() == [0]
    live.feed(chunk(3, [3, 4]))
    assert live.read_ready().tolist() == [1, 2, 3, 4]
    assert live.pos == 5 and not live.read_ready().size
    live.feed(chunk(5, [5]))
    assert live.read(1).tolist() == [5]


def test_capture_floor_requires_own_completed_transmit():
    live, _ = port()
    live.feed(chunk(0, [0, 1, 2]))
    assert live.read(1).tolist() == [0]
    with pytest.raises(ValueError, match='completed transmission'):
        live.flush_to(2)
    assert live.read(2).tolist() == [1, 2]


def test_capture_overflow_and_stall_are_bounded_failures():
    live, _ = port(depth=1)
    live.feed(chunk(0, np.zeros(4801)))
    assert live._frames.qsize() == 1
    with pytest.raises(ConnectionAbortedError, match='queue overflow'):
        live.read(1)
    assert live.closed and live._frames.empty()
    stalled, _ = port(stall_timeout=.01)
    with pytest.raises(ConnectionAbortedError, match='capture stopped'):
        stalled.read(1)
    assert stalled.closed


def test_external_cancellation_wakes_and_retires_the_capture_on_its_owner():
    live, st = port(stall_timeout=10)
    began, errors = threading.Event(), []

    def owner():
        began.set()
        try:
            live.read(1)
        except ConnectionAbortedError as exc:
            errors.append(str(exc))
        finally:
            live.close()

    worker = threading.Thread(target=owner)
    worker.start()
    assert began.wait(1)
    live.cancel()
    worker.join(1)
    assert not worker.is_alive() and errors == ['PACTOR station input closed']
    assert live.closed and live._frames.empty()
    assert not st.rig.edges and not st.audio.closed and not st.rig.closed


def test_own_transmit_flush_keeps_queued_peer_suffix_and_actual_ptt():
    live, st = port()
    end = 1000 + 8
    live.feed(chunk(0, np.arange(end + 5)))
    forbidden = lambda on: pytest.fail('the protocol must not key the shared rig')
    assert live.transmit(np.full(8, .15, np.float32), at=1000, settle=.01,
                         key=forbidden) == (1000, end)
    assert st.rig.edges == [True, False]
    assert live.keyed_at == 520 and live.keyed_s == (end - 520) / 48000
    assert live.corrected_emission_start == 1000 and live.corrected_emission_end == end
    np.testing.assert_allclose(st.audio.sent[0][1], .15, rtol=0, atol=1e-8)
    live.flush_to(end)
    np.testing.assert_array_equal(live.read(5), np.arange(end, end + 5))
    live.feed(chunk(end + 5, [42], StreamFault(end, 'flush', 1)))
    assert live.read(1).tolist() == [42]
    assert not live.closed
    # An identical later flush cannot reuse a consumed permission.
    live.feed(chunk(end + 6, [43], StreamFault(end, 'flush', 2)))
    with pytest.raises(ConnectionAbortedError, match='flush'):
        live.read(1)


def test_obsolete_proposal_is_rechecked_after_wait_before_physical_key():
    live, st = port()
    current = [True]
    st.audio.before_wait = lambda: current.__setitem__(0, False)

    def admit():
        if not current[0]:
            raise onair._P3TxCancelled('obsolete P3 proposal')

    with pytest.raises(onair._P3TxCancelled):
        live.transmit(np.ones(100, np.float32) * .2, at=2000, admit=admit)
    assert st.audio.cancelled == 1 and st.audio.pending is None
    assert not st.rig.edges and not st.audio.sent
    assert live.keyed_at is None and live.corrected_emission_end is None


def test_no_owner_or_unsafe_waveform_cannot_key_or_claim_a_transmission():
    live, st = port()
    for samples in (np.ones(10), np.array([np.nan]), np.empty(0), np.zeros((2, 2))):
        with pytest.raises(ValueError):
            live.transmit(samples)
    st.arbiter.disable('pactor', 'offline test')
    with pytest.raises(Refused, match='disabled'):
        live.transmit(np.ones(10, np.float32) * .2)
    assert not st.rig.edges and not st.audio.sent
    assert live.corrected_emission_end is None


def test_existing_native_radio_producer_reaches_only_station_arbiter(tmp_path):
    live, st = port()
    producer_rig = SimpleNamespace(ptt=lambda on: pytest.fail('independent keyer used'),
                                   key_failure=lambda: None)
    tx = onair.RadioTx(producer_rig, transmit=True, outdir=tmp_path, drive=.6, settle=.04)
    tx.live = live
    tx.set_host_control('FSKAMPL', 750)
    tx._tx(pactor1.control_signal(pactor1.CS_SPEED), 'P1 station fixture', tx._fsk_drive)
    assert len(tx.keyed) == 1 and tx.tx_end == live.pos
    assert st.arbiter.sent == 1 and st.rig.edges == [True, False]
    assert np.max(np.abs(st.audio.sent[0][1])) == pytest.approx(.15)
    assert tx.tx_audio_start - tx.tx_key_up == 1920
    live.close()
    assert not st.audio.closed and not st.rig.closed
    live.feed(chunk(live.pos, [0]))
    assert live._frames.empty()


def test_owner_io_is_serviced_on_the_same_thread_during_keyed_audio():
    polls = []
    def service():
        polls.append((threading.current_thread(), st.audio.samples, st.rig.keyed))
    live, st = port(capture_factory=PollingCapture, owner_poll=service)
    assert live.transmit(np.full(4800, .15, np.float32), at=4000, settle=.02) == (4000, 8800)
    assert len(polls) > 10 and any(keyed for _, _, keyed in polls)
    assert all(owner is threading.current_thread() for owner, _, _ in polls)
    assert all(o['healthy'] and o['deadline'] > 0 for o in st.audio.wait_options)
    assert not any(at == live.keyed_at and not keyed for _, at, keyed in polls)
    assert st.rig.edges == [True, False] and st.arbiter.sent == 1


@pytest.mark.parametrize('keyed', [False, True])
def test_owner_io_failure_cancels_or_unkeys_and_retires_the_port(keyed):
    def service():
        if st.audio.pending is not None and st.rig.keyed == keyed:
            raise RuntimeError('owner serial failed')
    live, st = port(capture_factory=PollingCapture, owner_poll=service)
    with pytest.raises(RuntimeError, match='owner serial failed'):
        live.transmit(np.full(4800, .15, np.float32), at=4000, settle=.02)
    assert live.closed and not st.rig.keyed and not st.arbiter.sent
    assert st.rig.edges == ([True, False] if keyed else [])
    assert st.arbiter.transmit_history().sequence == int(keyed)
    assert st.arbiter.transmit_history().capture_floor == (8800 if keyed else 0)
    assert live.corrected_emission_end is None and not st.audio.pending
    with pytest.raises(ConnectionAbortedError, match='owner I/O failed'):
        live.read_ready()
    live.close()
    assert not st.audio.closed and not st.rig.closed


@pytest.mark.parametrize('operation', ['read', 'close'])
def test_owner_poll_cannot_consume_or_close_capture_reentrantly(operation):
    def service():
        (live.read_ready if operation == 'read' else live.close)()
    live, st = port(owner_poll=service)
    live.feed(chunk(0, [1, 2]))
    with pytest.raises(RuntimeError, match='owner I/O poll'):
        live.read(1)
    assert live.closed and live._frames.qsize() == 1
    live.close()
    assert live._frames.empty() and not st.rig.edges


def test_owner_poll_cannot_key_recursively_through_any_arbiter_entry():
    inside, refused = [False], []
    def request():
        return TxRequest(LIVE, 0, protocol='pactor', audio=np.full(10, .1),
                         emission=live.link.waveform.emission(st.rig.dial()))
    def service():
        inside[0] = True
        try:
            with pytest.raises(Refused, match='owner I/O poll'):
                st.arbiter.submit(request())
            result = st.arbiter.submit_result(request())
            assert result.outcome == 'refused' and 'owner I/O poll' in result.reason
            with pytest.raises(Refused, match='owner I/O poll'):
                st.arbiter.identify(request().emission)
            refused.append(True)
        finally:
            inside[0] = False
    live, st = port(capture_factory=PollingCapture, owner_poll=service)
    lock = threading.Lock()
    class GuardedLock:
        def acquire(self, *args, **kwargs):
            assert not inside[0], 'recursive owner tried to wait for the transmit lock'
            return lock.acquire(*args, **kwargs)
        def release(self):
            lock.release()
        def __enter__(self):
            self.acquire()
            return self
        def __exit__(self, *args):
            self.release()
    st.arbiter._lock = GuardedLock()
    live.transmit(np.full(4800, .15, np.float32), at=4000, settle=.02)
    assert len(refused) > 10 and st.rig.edges == [True, False]
    assert st.arbiter.sent == 1


def test_real_ptc_serial_poll_is_answered_while_the_station_is_keyed():
    import os
    from hfhost import ptc as client
    from hfmodem.shrike.ptc import PtcHost
    from hfmodem.shrike.serial_endpoint import PtcSerialEndpoint
    decoder, replies, answered_at = client.Deframer(), [], []
    def service():
        if not st.rig.keyed:
            return
        assert endpoint.service()
        try:
            data = os.read(fd, 4096)
        except BlockingIOError:
            return
        replies.extend(decoder.feed(data))
        answered_at.append(st.audio.samples)
    live, st = port(capture_factory=PollingCapture, owner_poll=service)
    host = PtcHost()
    host.tick = lambda *a, **kw: pytest.fail('serial service advanced the ARQ clock')
    endpoint = PtcSerialEndpoint(host)
    name = endpoint.open()
    fd = os.open(name, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        os.write(fd, b'JHOST4\r')
        for _ in range(3):
            assert endpoint.service()
        while True:
            try:
                if not os.read(fd, 4096):
                    break
            except BlockingIOError:
                break
        assert host.hostmode
        os.write(fd, client.Packet(host.ptchn, True, b'L', counter=1).encode())
        live.transmit(np.full(24000, .15, np.float32), at=4000, settle=.02)
        assert len(replies) == 1 and replies[0].code == client.MSG
        assert len(replies[0].data.split()) == 6
        assert answered_at and live.keyed_at <= answered_at[0] < live.corrected_emission_end
        assert st.rig.edges == [True, False]
    finally:
        endpoint.close()
        os.close(fd)
        live.close()


class FeedingCapture(PollingCapture):
    """Card callbacks continue while a native producer is waiting or keyed."""
    feed = lambda self, frame: None
    def wait_until(self, index, **options):
        self.wait_options.append(options)
        self.before_wait()
        while self.samples < index:
            if options.get('cancelled') is not None and options['cancelled']():
                raise BurstInvalidated('cancelled')
            previous = self.samples
            self.samples = min(index, self.samples + 128)
            self.feed(chunk(previous, np.zeros(self.samples - previous, np.float32)))


def test_short_card_callbacks_remain_bounded_during_a_maximum_length_burst():
    from hfmodem.station.arbiter import longest_burst_s
    live, st = port(capture_factory=FeedingCapture)
    st.audio.feed = live.feed
    # The longest watchdog-admitted burst of indexed callbacks exhausts the old queue
    # of 128 items even with 50 ms station chunks. No own capture is discarded
    # until this transmission completes and the producer authorizes its floor.
    count = int(longest_burst_s(st.rig.max_key_s) * 48000)
    _, end = live.transmit(np.full(count, .1, np.float32))
    assert not live._overflow.is_set() and live._frames.qsize() <= live._frames.maxsize
    assert sum(len(frame.samples) for frame in live._frames.queue) == end
    assert live.pos == 0 and not live._rest.size
    live.flush_to(end)
    assert not live.read_ready().size and live.pos == end
    assert st.rig.edges == [True, False] and st.arbiter.sent == 1
    live.close()


def test_queued_coalescing_keeps_faults_and_never_waits_for_a_full_chunk():
    live, _ = port(depth=1)
    live.feed(chunk(0, [1]))
    assert live.read(1).tolist() == [1]  # no batching latency
    live.feed(chunk(1, [2]))
    live.feed(chunk(2, [3], StreamFault(2, 'index gap', 1)))
    assert live._frames.qsize() == 1
    with pytest.raises(ConnectionAbortedError, match='index gap'):
        live.read(1)
    assert live.closed


@pytest.mark.parametrize('offset', [0, 120])
def test_native_closing_cw_disposes_of_the_station_identity_interlock(tmp_path, offset):
    from hfmodem.core import cwid, levels
    from hfmodem.core.occupied import occupied_hz
    from hfmodem.station.arbiter import Identity
    from hfhost import ptc
    from hfmodem.shrike.ptc import PtcHost
    from hfmodem.tests.shrike.test_ptc_compatibility import Client
    live, st = port(capture_factory=FeedingCapture)
    st.audio.feed = live.feed
    read = live.read
    def card_read(count):
        st.audio.wait_until(live.pos + count)
        return read(count)
    live.read = card_read
    st.arbiter.identity = Identity('K1ABC')
    rig = SimpleNamespace(ptt=lambda on: pytest.fail('independent PTT'), key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, drive=.6, settle=.04)
    tx.live = onair._Retuned(live, offset) if offset else live
    tx.set_host_control('FSKAMPL', 750)
    tx.set_host_control('CWSPEED', 70)
    host = PtcHost(tx, mycall='N0CALL')
    owner = host.enable_owner()
    client = Client(host)
    refused, requests = [], []
    submit = st.arbiter.submit
    def record(request):
        requests.append(request)
        return submit(request)
    st.arbiter.submit = record
    def service():
        if live._identifying is not None and st.rig.keyed and not refused:
            assert client.cmd('C N0NEXT').code == ptc.FAIL
            assert 'OWNER BUSY' in client.retry().text
            refused.append(True)
    live.owner_poll = service
    with owner.contact():
        tx._tx(pactor1.control_signal(pactor1.CS_SPEED), 'P1 data', tx._fsk_drive)
        g = onair._MasterGrid(tx.tx_audio_start, 60000, 0, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
        tx.aim(g, 1)
        st.arbiter.identity.end()
        assert st.arbiter.identity.due
        tx.identify('K1ABC')
    assert not st.arbiter.identity.due and st.arbiter.sent == 2
    assert st.arbiter.last_sent['ident'] == tx.tx_end == live.pos
    assert st.rig.edges == [True, False, True, False]
    centre = sum(occupied_hz('pactor')) / 2
    audio = cwid.audio('K1ABC', wpm=14, tone_hz=centre)
    if offset:
        audio = onair._shift_tones(audio, offset)[0]
    expected = onair._trim_silence(levels.at_drive(audio, .6 * .25 * 10**(-3/20)))
    np.testing.assert_allclose(st.audio.sent[-1][1], expected, atol=2e-8, rtol=1e-6)
    assert live._identifying is None and not live._overflow.is_set()
    assert refused and not owner.contact_active and not host.arq.payload_ledger.confirmed
    assert [request.protocol for request in requests] == ['pactor', 'ident']
    assert requests[-1].emission.audio_lo_hz == centre + offset - 75
    assert requests[-1].emission.audio_hi_hz == centre + offset + 75
    live.close()
    assert not st.audio.closed and not st.rig.closed


@pytest.mark.parametrize('invalid', ['missing', 'disabled', 'wrong_call', 'changed'])
def test_station_identification_cannot_credit_an_unconfigured_or_changed_identity(invalid):
    from hfmodem.station.arbiter import Identity
    live, st = port(capture_factory=PollingCapture)
    identity = st.arbiter.identity = Identity('K1ABC')
    identity.end()
    call = 'K1ABC'
    if invalid == 'missing':
        st.arbiter.identity = None
    elif invalid == 'disabled':
        identity.mode = 'none'
    elif invalid == 'wrong_call':
        call = 'N0OTHER'
    elif invalid == 'changed':
        st.audio.before_wait = lambda: setattr(identity, 'mycall', 'N0OTHER')
    with pytest.raises(ValueError, match='station|identity changed'):
        with live.identification_scope(call, centre_hz=1500):
            live.transmit(np.full(4800, .1, np.float32), at=4000)
    assert not st.rig.edges and not st.arbiter.sent and live._identifying is None
    assert identity._last < time.monotonic() - identity.interval_s
    live.close()


def test_failed_station_identification_unkeys_without_crediting_the_id():
    from hfmodem.station.arbiter import Identity
    def failed():
        if st.rig.keyed:
            raise OSError('serial lost during CW')
    live, st = port(capture_factory=PollingCapture, owner_poll=failed)
    st.arbiter.identity = Identity('K1ABC')
    st.arbiter.identity.end()
    with pytest.raises(OSError, match='during CW'):
        with live.identification_scope('K1ABC', centre_hz=1500):
            live.transmit(np.full(4800, .1, np.float32), at=4000)
    assert st.arbiter.identity.due and not st.arbiter.sent
    assert st.rig.edges == [True, False] and live._identifying is None and live.closed
    assert st.arbiter.transmit_history().sequence == 1
    live.close()


def test_new_contact_ignores_queued_pre_epoch_pcm_and_old_faults():
    old, st = port()
    old.feed(chunk(0, [1, 2, 3]))
    assert old.read(1).tolist() == [1]
    old.close()
    st.audio.samples = 10000
    fresh = StationInput(old.link, depth=1)
    # Repeated old handoffs neither fill the new queue nor revive the old
    # contact's flush/error. The new PCM is still strictly indexed at birth.
    for _ in range(5):
        fresh.feed(chunk(0, np.ones(10000), StreamFault(9999, 'flush', 3)))
    assert fresh._frames.empty() and not fresh._overflow.is_set()
    fresh.feed(chunk(9998, [8, 9, 10, 11], StreamFault(9999, 'index gap', 4)))
    assert fresh.read(2).tolist() == [10, 11]
    assert fresh.pos == 10002 and old.pos == 1
    assert not st.audio.closed and not st.rig.edges
    fresh.close()


@pytest.mark.parametrize('cause', ['flush', 'index gap', 'resync', 'deque drop'])
def test_new_contact_cannot_ignore_a_fault_at_its_current_epoch_boundary(cause):
    old, st = port()
    old.close()
    st.audio.samples = 10000
    fresh = StationInput(old.link)
    fresh.feed(chunk(9998, [8, 9, 10, 11], StreamFault(10000, cause, 4)))
    with pytest.raises(ConnectionAbortedError, match=cause):
        fresh.read(1)
    assert fresh.closed and fresh.pos == 10000
    assert not st.rig.edges and not st.audio.closed


def test_new_contact_birth_does_not_authorize_an_unowned_capture_flush():
    old, st = port()
    old.close()
    st.audio.samples = 10000
    fresh = StationInput(old.link)
    with pytest.raises(ValueError, match='completed transmission'):
        fresh.flush_to(10001)
    fresh.feed(chunk(10001, [1]))
    with pytest.raises(ConnectionAbortedError, match='index gap: 10000..10001'):
        fresh.read(1)
    assert fresh.pos == 10000 and not st.rig.edges


def test_next_contact_waits_for_a_failed_keyings_conservative_retirement_floor():
    def lost():
        if st.rig.keyed:
            raise OSError('contact lost while output pending')
    old, st = port(capture_factory=PollingCapture, owner_poll=lost)
    with pytest.raises(OSError):
        old.transmit(np.full(4800, .1, np.float32), at=4000, settle=.02)
    history = st.arbiter.transmit_history()
    assert history.sequence == 1 and st.audio.samples < history.capture_floor == 8800
    with pytest.raises(ValueError, match='wait past retired station output'):
        StationInput(old.link)
    assert st.rig.edges == [True, False]
    # The new contact begins only when real capture has reached that floor.
    st.audio.samples = history.capture_floor
    fresh = StationInput(old.link)
    fresh.feed(chunk(8798, [8, 9, 10, 11]))
    assert fresh.read(2).tolist() == [10, 11]
    assert fresh.pos == history.capture_floor + 2
    fresh.close()
    old.close()


def test_new_contact_recognizes_only_a_published_pre_birth_transmit_flush():
    from hfmodem.station.arbiter import TransmitHistory
    old, st = port()
    old.close()
    st.audio.samples = 10000
    st.arbiter._history = TransmitHistory(1, 10000)
    old.link.lane = SimpleNamespace(fault_serial=3)
    fresh = StationInput(old.link)
    fresh.feed(chunk(9998, [8, 9, 10, 11], StreamFault(10000, 'flush', 3)))
    assert fresh.read(2).tolist() == [10, 11]
    fresh.feed(chunk(10002, [12], StreamFault(10000, 'flush', 4)))
    with pytest.raises(ConnectionAbortedError, match='flush'):
        fresh.read(1)
    assert fresh.closed and not st.rig.edges
