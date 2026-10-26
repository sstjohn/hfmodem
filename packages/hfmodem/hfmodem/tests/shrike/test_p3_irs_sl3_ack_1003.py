# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K4MSU 1003-0033: current-cycle SL3 ACKs on the retained PCM and budget."""
import contextlib
from functools import lru_cache
import gc
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core import audio as cardaudio, wav
from hfmodem.shrike import arq, onair, p3acquire, placement, rxfront
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid

PINS = json.loads(evidence.require_input(
    Path(__file__).with_name('fixtures') / 'p3-irs-sl3-ack-1003.json',
    allow_module_level=True).read_text())
CROPS = {p['hold']: p for p in PINS['crops']}


@contextlib.contextmanager
def receive_gc():
    with onair._receive_gc() as enabled:
        yield
    if enabled:
        gc.collect(2)


@lru_cache(maxsize=3)
def stream(arm):
    path = evidence.require_input(evidence.CAPTURES / f'onair-{arm}/stream.wav')
    fs, raw = wavfile.read(wav.source(path))
    assert fs == onair.FS and raw.dtype == np.int16 and raw.ndim == 1
    return raw


def crop(hold):
    p = CROPS[hold]
    raw = stream('1003-0033')[p['origin']:p['end']]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == p['pcm_sha256']
    return raw.astype(float) / 32768, p


def receiver(row, *, level=2, hz=-13., requested=True):
    s = _Session(role=arq.IRS)
    s.host.arq.cfg.long_cycle = False
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = 3
    s.rx._p3_row0 = s.rx._p3_delivered_at = row - 60000
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[level])
    s.rx._p3_cycle_n = 60000
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = level
    s.rx.p3_wideband_prekey = True
    s.rx.p3_receive_offset_hz = hz
    s.rx._p3_offset_fixed = True
    if requested:
        s.rx.note_p3_speedup_emitted(2, 3)
    return s


def budget(s, monkeypatch, allowance):
    spent, trace = [0.], []
    peer = s.host.peer
    peer.live = SimpleNamespace(transmit=True,
        clamp_late=lambda deadline: round(spent[0] * 48) > deadline)
    phase = s.rx._p3_row0 + 60000 - 4320
    peer.raster = irs_grid(anchor=phase)
    peer.slot = 204
    peer.key_instant = lambda *args: round(allowance * 48)
    for method, cost in [('sl2_packet_at', 7.51), ('sl1_packet_at', 3.30),
                         ('wideband_packet_at', 12.52)]:
        fn = getattr(s.rx.sync, method)
        def read(*args, _fn=fn, _method=method, _cost=cost, **kwargs):
            trace.append(_method)
            got = _fn(*args, **kwargs)
            spent[0] += (3. if _method == 'wideband_packet_at'
                         and kwargs.get('prepared') is not None else _cost)
            return got
        monkeypatch.setattr(s.rx.sync, method, read)
    return spent, trace


def stage_sl3(s, audio, origin):
    for fraction in (.65, .85, .97):
        s.rx.prepare_p3_sl3(audio[:int(len(audio) * fraction)], origin)


class SliceLive:
    """Production listener input; deadline starts before final-slice preparation."""
    transmit = True

    def __init__(self, audio, origin):
        self.audio, self.pos, self.offset = audio, origin, 0
        self.started = None

    def read(self, n):
        piece = self.audio[self.offset:self.offset+n]
        self.offset += len(piece)
        self.pos += len(piece)
        if self.offset == len(self.audio):
            self.started = time.perf_counter()
        return piece

    def clamp_late(self, deadline):
        return (self.started is not None and
                (time.perf_counter()-self.started)*onair.FS > deadline)


def listen_current(s, audio, origin, allowance=18.):
    live = SliceLive(audio, origin)
    peer = s.host.peer
    peer.live = live
    peer.raster = irs_grid(anchor=s.rx._p3_row0+60000-4320)
    peer.slot = 204
    peer.key_instant = lambda *args: round(allowance*48)
    with receive_gc():
        held = onair._listen_until_answer(live, len(audio), s.host, s.rx, 0)
        onair._scan_frame(s.rx, held, origin, tracked_only=True)
        elapsed = (time.perf_counter()-live.started)*1000
    return elapsed


@pytest.fixture(scope='module')
def audio_callback_load():
    station = cardaudio.StationAudio(input_device=None, output_device=None)
    station.subscribe(cardaudio._BaseLane(cardaudio.CARD_RATE_HZ))
    inputs = np.zeros((128, 1), np.float32)
    outputs = inputs.copy()
    stop, ready = threading.Event(), threading.Event()
    def callbacks():
        due = time.perf_counter()
        while not stop.is_set():
            t = station.samples / cardaudio.CARD_RATE_HZ
            stamps = SimpleNamespace(inputBufferAdcTime=t, outputBufferDacTime=t + .02)
            station._callback(inputs, outputs, 128, stamps, False)
            ready.set()
            due += 128 / cardaudio.CARD_RATE_HZ
            stop.wait(max(0, due - time.perf_counter()))
    thread = threading.Thread(target=callbacks)
    thread.start()
    try:
        assert ready.wait(1)
        yield station
    finally:
        stop.set()
        thread.join()
    assert station.failure is None


@pytest.mark.parametrize('hold,tx,digest', [(201, 202, '22ff7be5333b'),
                                         (206, 207, 'eabd137832c3')])
@pytest.mark.parametrize('allowance', [18., 40., 44.])
def test_requested_sl3_reaches_current_ack(tmp_path, monkeypatch, hold, tx, digest, allowance):
    audio, p = crop(hold)
    s = receiver(p['row'])
    target = PINS['reply_targets'][str(tx)]
    slot = 204 if hold == 201 else 209
    g = irs_grid(anchor=target['audio_start'] - slot * 60000 - 28800)
    radio = driver(g, tmp_path)
    radio.host, radio.sessrx = s.host, s.rx
    radio.defer_p3_cs = True
    radio.send_cs(arq.CS_ACK)
    sent = s.host.peer.send_cs
    def queue(ci):
        sent(ci)
        return radio.send_cs(ci)
    monkeypatch.setattr(s.host.peer, 'send_cs', queue)
    spent, trace = budget(s, monkeypatch, allowance)
    if allowance == 18.:
        stage_sl3(s, audio, p['origin'])
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert trace == ['wideband_packet_at']
    assert (3. if allowance == 18. else spent[0]) + 6 <= allowance
    ev, = s.packets
    sl, status, payload, crc = ev.packet
    assert (sl, status, len(payload), crc) == (3, 0x23, 59, True)
    assert hashlib.sha256(payload).hexdigest()[:12] == digest
    assert s.rx._p3_delivered_at == p['row']
    assert s.host.arq.rx_seq == 3
    assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
    assert radio._pending_p3_cs == arq.CS_REQUEST
    pulse = target['audio_start'] + min(target['pulse_offsets'])
    phase = p['row'] - 4320
    assert 880 <= (pulse - phase) / 48 <= 910
    assert s.rx._p3_acquisition_fits(0)
    g.note_p3_packet(phase, 38880, 60000, identity=(False, 3, 0x23, payload))
    radio.live.limit = 500 * onair.FS
    radio.reply_clock = ReplyClock(entry_phase=pulse - 28950,
                                   pulse_epoch=pulse, gap=pulse - phase)
    radio.aim(g, slot)
    radio.live.now = radio.live.pos = radio.key_instant(g, slot) - 1920
    radio.emit_pending_cs()
    assert radio._pending_p3_cs is None
    assert radio.tx_audio_start == target['audio_start']
    assert list(radio.tx_pulse_offsets) == target['pulse_offsets']
    assert radio.n == 1 and not radio.refused


def test_crc_does_not_authorize_an_overdue_key(tmp_path):
    audio, p = crop(201)
    s = receiver(p['row'])
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
    target = PINS['reply_targets']['202']
    pulse = target['audio_start'] + min(target['pulse_offsets'])
    phase = p['row'] - 4320
    g = irs_grid(anchor=target['audio_start'] - 204 * 60000 - 28800)
    g.note_p3_packet(phase, 38880, 60000, identity=(False, *s.packets[0].packet[:3]))
    radio = driver(g, tmp_path)
    radio.host, radio.sessrx = s.host, s.rx
    radio.live.limit = 500 * onair.FS
    radio.reply_clock = ReplyClock(entry_phase=pulse - 28950,
                                   pulse_epoch=pulse, gap=pulse - phase)
    radio.aim(g, 204)
    radio.live.now = radio.live.pos = radio.key_instant(g, 204) + 960
    assert radio._send_p3_control(arq.CS_REQUEST) is None
    assert radio.slot == 205
    assert radio.tx_audio_start == target['audio_start'] + 60000
    assert radio.n == 1



@pytest.mark.parametrize('hold', [201, 206])
@pytest.mark.parametrize('allowance', [0., 14., 18., 22.])
def test_unaffordable_sl3_does_not_decode_or_ack(monkeypatch, hold, allowance):
    audio, p = crop(hold)
    s = receiver(p['row'])
    _, trace = budget(s, monkeypatch, allowance)
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert trace == {0.: [], 14.: ['sl2_packet_at'],
                     18.: ['sl2_packet_at', 'sl1_packet_at'],
                     22.: ['sl2_packet_at', 'sl2_packet_at']}[allowance]
    assert not s.packets and not s.host.peer.sent
    assert s.host.arq.rx_seq != 3


@pytest.mark.parametrize('hold,digest', [(202, '65e85f13c2b1'), (207, '6aa3fa0af6fc')])
@pytest.mark.parametrize('level', [None, 2])
@pytest.mark.parametrize('allowance', [14., 18., 22., 40.])
def test_declined_sl3_still_reads_sl2(monkeypatch, hold, digest, level, allowance):
    audio, p = crop(hold)
    s = receiver(p['row'])
    s.rx.sync.packet_level = level
    _, trace = budget(s, monkeypatch, allowance)
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert trace == (['wideband_packet_at', 'sl2_packet_at']
                     if allowance == 40. else ['sl2_packet_at'])
    ev, = s.packets
    assert ev.packet[:2] == (2, 0x23) and len(ev.packet[2]) == 23
    assert hashlib.sha256(ev.packet[2]).hexdigest()[:12] == digest
    assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
    assert s.rx._p3_expected() == (2, 3)


def test_requested_sl3_window_uses_wideband_filter_support():
    s = receiver(CROPS[201]['row'])
    deadline = CROPS[201]['row'] + 42000
    ready = onair._p3_frame_ready(s.rx, deadline, 660)
    s.rx._p3_expected_levels = ()
    narrow = onair._p3_frame_ready(s.rx, deadline, 660)
    assert ready > narrow
    assert ready <= deadline


def test_requested_sl3_miss_does_not_open_another_body_or_ladder(monkeypatch):
    p = CROPS[201]
    s = receiver(p['row'])
    def forbid(*args, **kwargs):
        pytest.fail('bounded requested-level miss opened another reader')
    monkeypatch.setattr(s.rx, '_read_p3_packet', forbid)
    spent, trace = budget(s, monkeypatch, 44.)
    for ahead in (0, 1):
        spent[0], trace[:] = 0., []
        onair._scan_frame(s.rx, np.zeros(p['end'] - p['origin']),
                          p['origin'] + ahead * 60000, tracked_only=True)
        assert trace == ['wideband_packet_at', 'sl2_packet_at', 'sl2_packet_at',
                         'sl1_packet_at']
    assert s.rx.sync.packet_level is None
    assert s.rx._p3_expected() == (2, 3)
    assert not s.packets and not s.host.peer.sent


@pytest.mark.parametrize('hold,digest,status', [(60, 'd85c654326e5', 0x02),
                                              (245, '30dd75b9a4c9', 0x22)])
def test_nominal_crc_preserves_measured_rx_and_tx(hold, digest, status):
    audio, p = crop(hold)
    s = receiver(p['row'], requested=False)
    at = p['row'] - p['origin']
    read = lambda a: s.rx.sync.sl2_packet_at(a, at)
    assert read(p3acquire.compensate(audio, -13.)) is None
    ev = s.rx._p3_sl2_at_nominal(read, audio)
    assert ev.packet[:2] == (2, status)
    assert hashlib.sha256(ev.packet[2]).hexdigest()[:12] == digest
    assert s.rx.p3_receive_offset_hz == s.rx.p3_transmit_offset_hz == -13.
    assert s.rx._p3_offset_candidate is None


def test_nominal_fallback_requalifies_a_real_carrier_change():
    packet = placement.link_packet(2, b'fresh', 0x23)
    audio = np.pad(packet, (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    audio += np.random.default_rng(5).normal(0, .7, audio.size)
    assert rxfront.SyncedRx().sl2_packet_at(p3acquire.compensate(audio, -13.), row) is None
    s = receiver(row, requested=False)
    for cycle in range(4):
        s.rx.new_cycle()
        onair._scan_frame(s.rx, audio, cycle * 60000, tracked_only=True)
        assert len(s.packets) == cycle + 1
        if cycle < 2:
            assert s.rx.p3_receive_offset_hz == s.rx.p3_transmit_offset_hz == -13.
        else:
            assert s.rx.p3_receive_offset_hz == pytest.approx(0., abs=.2)
            assert s.rx.p3_transmit_offset_hz == pytest.approx(0., abs=.2)


@pytest.mark.parametrize('sl', [1, 2])
@pytest.mark.parametrize('swapped', [False, True])
def test_pending_sl3_preserves_narrow_fallback(sl, swapped):
    audio = np.pad(placement.link_packet(sl, b'fresh', 0x23, swapped=swapped),
                   (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    s = receiver(row, hz=0.)
    onair._scan_frame(s.rx, audio, 0, tracked_only=True)
    ev, = s.packets
    assert ev.packet == (sl, 0x23, b'fresh', True)
    assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]


@pytest.mark.parametrize('hold,allowance', [(201, 18.), (206, 18.),
                                          (202, 18.), (207, 18.),
                                          (202, 22.), (207, 22.),
                                          (202, 14.), (207, 14.)])
@pytest.mark.wall_clock
def test_recorded_current_ack_with_real_wall_budget(audio_callback_load, hold, allowance):
    audio, p = crop(hold)
    costs = []
    for _ in range(5):
        s = receiver(p['row'])
        peer = s.host.peer
        peer.raster = irs_grid(anchor=s.rx._p3_row0 + 60000 - 4320)
        peer.slot = 204
        peer.key_instant = lambda *args: round(allowance * 48)
        with receive_gc():
            started = time.perf_counter()
            peer.live = SimpleNamespace(transmit=True,
                clamp_late=lambda deadline: (time.perf_counter() - started) * onair.FS > deadline)
            if hold in (201, 206):
                elapsed = listen_current(s, audio, p['origin'], allowance)/1000
            else:
                onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
                elapsed = time.perf_counter() - started
            costs.append(elapsed * 1000)
            ev, = s.packets
            assert ev.packet[:2] == (3 if hold in (201, 206) else 2, 0x23)
            assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
            assert elapsed + onair.P3_DECODE_RESERVE_S < allowance / 1000
            assert s.rx._p3_acquisition_fits(0)
    print(f'ACK_METRIC hold={hold} allowance={allowance:g} scan={max(costs):.3f}')


@pytest.mark.wall_clock
def test_gen2_cannot_steal_current_sl2_ack(audio_callback_load, monkeypatch):
    audio, p = crop(207)
    events = []
    collection_started = None

    def collection(phase, info):
        nonlocal collection_started
        if info['generation'] != 2:
            return
        if phase == 'start':
            collection_started = time.perf_counter()
        else:
            time.sleep(max(0., .013 - (time.perf_counter() - collection_started)))
        events.append((phase, time.perf_counter()))

    enabled = gc.isenabled()
    gc.enable()
    try:
        gc.callbacks.append(collection)
        try:
            for protected in (False, True):
                s = receiver(p['row'])
                peer = s.host.peer
                peer.raster = irs_grid(anchor=s.rx._p3_row0 + 60000 - 4320)
                peer.slot = 204
                peer.key_instant = lambda *args: round(18. * 48)
                fits = s.rx._p3_acquisition_fits
                interval = []

                def admission(reserve):
                    if not interval:
                        interval.append(time.perf_counter())
                        if gc.isenabled():
                            gc.collect(2)
                        interval.append(time.perf_counter())
                    return fits(reserve)

                monkeypatch.setattr(s.rx, '_p3_acquisition_fits', admission)
                scope = onair._receive_gc() if protected else contextlib.nullcontext()
                with scope:
                    started = time.perf_counter()
                    peer.live = SimpleNamespace(transmit=True,
                        clamp_late=lambda deadline:
                            (time.perf_counter() - started) * onair.FS > deadline)
                    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
                    elapsed = time.perf_counter() - started
                    assert len(interval) == 2
                    during = [(phase, at) for phase, at in events
                              if interval[0] <= at <= interval[1]]
                    if protected:
                        assert not during
                        assert not [at for _, at in events if started <= at <= started + elapsed]
                        ev, = s.packets
                        assert ev.packet[:2] == (2, 0x23)
                        assert len(ev.packet[2]) == 23 and ev.packet[3]
                        assert hashlib.sha256(ev.packet[2]).hexdigest()[:12] == '6aa3fa0af6fc'
                        assert peer.sent == [('cs', arq.CS_REQUEST)]
                        assert elapsed + onair.P3_DECODE_RESERVE_S < .018
                        assert s.rx._p3_acquisition_fits(0)
                    else:
                        assert [phase for phase, _ in during] == ['start', 'stop']
                        assert during[1][1] - during[0][1] >= .013
                        assert not s.packets and not peer.sent
            after = time.perf_counter()
            gc.collect(2)
            assert [phase for phase, at in events if at >= after] == ['start', 'stop']
        finally:
            gc.callbacks.remove(collection)
    finally:
        if not enabled:
            gc.disable()


@pytest.mark.parametrize('swapped', [False, True])
@pytest.mark.wall_clock
def test_unaffordable_sl3_reads_sl1_with_real_wall_budget(audio_callback_load, monkeypatch,
                                                       swapped):
    audio = np.pad(placement.link_packet(1, b'fresh', 0x23, swapped=swapped),
                   (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    for _ in range(5):
        s = receiver(row, hz=0.)
        peer = s.host.peer
        peer.raster = irs_grid(anchor=row - 4320)
        peer.slot = 204
        peer.key_instant = lambda *args: round(12. * 48)
        def forbid(*args, **kwargs):
            pytest.fail('unaffordable SL3 reader started')
        monkeypatch.setattr(s.rx.sync, 'wideband_packet_at', forbid)
        with receive_gc():
            started = time.perf_counter()
            peer.live = SimpleNamespace(transmit=True,
                clamp_late=lambda deadline: (time.perf_counter() - started) * onair.FS > deadline)
            onair._scan_frame(s.rx, audio, 0, tracked_only=True)
            elapsed = time.perf_counter() - started
            ev, = s.packets
            assert ev.packet == (1, 0x23, b'fresh', True)
            assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
            assert elapsed + onair.P3_DECODE_RESERVE_S < .012
            assert s.rx._p3_acquisition_fits(0)


@pytest.mark.parametrize('hold', [201, 206])
@pytest.mark.wall_clock
def test_staged_sl3_loaded_post_capture_budget(audio_callback_load, hold):
    audio, p = crop(hold)
    preparation, finish, scan = [], [], []
    for _ in range(5):
        s = receiver(p['row'])
        prepare = s.rx.prepare_p3_sl3
        def preparing(*args, **kwargs):
            start = time.perf_counter()
            result = prepare(*args, **kwargs)
            preparation.append((time.perf_counter() - start) * 1000)
            return result
        s.rx.prepare_p3_sl3 = preparing
        original = s.rx.sync.wideband_packet_at
        def read(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            finish.append((time.perf_counter() - start) * 1000)
            return result
        s.rx.sync.wideband_packet_at = read
        scan.append(listen_current(s, audio, p['origin']))
        assert len(s.packets) == 1 and s.packets[0].packet[3]
        assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
    assert max(scan) <= 12.
    print(f'SL3_METRIC hold={hold} prep={max(preparation):.3f} '
          f'finish={max(finish):.3f} scan={max(scan):.3f}')


@pytest.mark.parametrize('damage', ['tail', 'body', 'noise', 'clock', 'carrier'])
def test_staged_sl3_requires_current_crc(damage):
    audio, p = crop(201)
    audio = audio.copy()
    s = receiver(p['row'])
    if damage == 'tail':
        audio = audio[:-2500]
    elif damage == 'body':
        audio[17000:28000] = 0
    elif damage == 'noise':
        audio = np.random.default_rng(1003).normal(0, .3, audio.size)
    elif damage == 'clock':
        s.rx._p3_row0 += 2400
    else:
        s.rx.p3_receive_offset_hz = 80.
    stage_sl3(s, audio, p['origin'])
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert not s.packets and not s.host.peer.sent


def test_staged_sl3_overlap_and_loss_invalidation(monkeypatch):
    audio, p = crop(201)
    s = receiver(p['row'])
    stage_sl3(s, audio, p['origin'])
    assert s.rx._p3_early_sl3 is not None
    monkeypatch.setattr(s.host.peer, '_observe_input_loss', lambda name: True, raising=False)
    s.rx.prepare_p3_sl3(audio, p['origin'])
    assert s.rx._p3_early_sl3 is None
    monkeypatch.setattr(s.host.peer, '_observe_input_loss', lambda name: False)
    stage_sl3(s, audio, p['origin'])
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert len(s.packets) == 1 and len(s.host.peer.sent) == 1


@pytest.mark.parametrize('hold', [202, 207])
def test_sl2_header_does_not_nominate_staged_sl3(hold):
    audio, p = crop(hold)
    s = receiver(p['row'])
    stage_sl3(s, audio, p['origin'])
    assert s.rx._p3_early_sl3 is None
    onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
    assert len(s.packets) == 1 and s.packets[0].packet[0] == 2


def fingerprint(ev):
    sl, status, payload, crc = ev.packet
    assert crc
    return dict(sl=sl, status=status, bytes=len(payload),
                hash=hashlib.sha256(payload).hexdigest()[:12])


def clean_replay(arm):
    raw = stream(arm)
    result = []
    for p in PINS['clean'][arm]:
        origin, end = p['row'] - 6500, p['row'] + 42000
        s = receiver(p['row'], level=p['sl'], hz=p['hz'], requested=False)
        s.rx._scan_origin, s.rx._tracked_only = origin, True
        got = s.rx._p3_packet(raw[origin:end].astype(float) / 32768)
        bounded = got is not None
        if not bounded:
            s.rx._tracked_only = False
            s.rx.sync.packet_at = p['row'] - origin
            got = s.rx._read_p3_packet(raw[origin:end].astype(float) / 32768, origin)
        assert got is not None
        ev, how = got
        assert fingerprint(ev) == {k: p[k] for k in ('sl', 'status', 'bytes', 'hash')}
        result.append(dict(**fingerprint(ev), row=origin + ev.start, how=how, bounded=bounded,
                           hz=s.rx.p3_transmit_offset_hz))
    return result


@pytest.mark.parametrize('arm', ['1003-0000', '1002-2349'])
def test_clean_recorded_receives_unchanged(arm):
    with contextlib.redirect_stdout(io.StringIO()):
        result = clean_replay(arm)
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    assert digest == PINS['clean_fingerprints'][arm]


@pytest.mark.parametrize('seq,copies', [(0, 30), (1, 35)])
def test_correct_ack_trains_deliver_once(seq, copies):
    rows = PINS['trains'][str(seq)]
    assert len(rows) == copies
    s = receiver(rows[0]['row'], requested=False)
    s.host.arq._expected_seq = seq
    deliveries = []
    s.host.arq.io.deliver = deliveries.append
    raw = stream('1003-0033')
    for p in rows:
        s.rx.new_cycle()
        origin, end = p['row'] - 6500, p['row'] + 42000
        onair._scan_frame(s.rx, raw[origin:end].astype(float) / 32768,
                          origin, tracked_only=True)
        assert fingerprint(s.packets[-1]) == {k: p[k] for k in ('sl', 'status', 'bytes', 'hash')}
        assert s.host.peer.sent[-1] == ('cs', arq.CS_REQUEST if seq & 1 else arq.CS_ACK)
        dac = wav.read(evidence.require_input(
            evidence.CAPTURES / f"onair-1003-0033/tx_{p['tx']:02d}.wav"), onair.FS)
        assert np.max(np.abs(dac)) == pytest.approx(.97, abs=1 / 32768)
        assert p['pulse_offsets'] == [852, 852]
        pulse = p['audio_start'] + min(p['pulse_offsets'])
        assert 38880 < pulse - (p['row'] - 4320) < 60000
        sidecar = json.loads((evidence.CAPTURES /
                             f"onair-1003-0033/tx_{p['tx']:02d}.json").read_text())
        assert sidecar['audio_start'] == p['audio_start']
        assert sidecar['pulse_offsets'] == p['pulse_offsets']
        corrected = np.pad(p3acquire.compensate(dac, p['tx_hz']), (4800, 4800))
        control = rxfront.SyncedRx().control_signal_tracked(corrected, 5652, details=False)
        assert control.cs == (arq.CS_REQUEST if seq & 1 else arq.CS_ACK)
    assert len(s.host.peer.sent) == copies and len(deliveries) == 1
    assert len(s.packets) == copies
    assert s.rx.p3_transmit_offset_hz == -13.


@pytest.mark.parametrize('p', PINS['live'], ids=lambda p: str(p['hold']))
def test_live_sl1_repeats_pay_both_pending_sl2_hypotheses(monkeypatch, p):
    raw = stream('1003-0045')[p['origin']:p['end']]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == p['pcm_sha256']
    s = receiver(p['row'], level=1, hz=-12.9, requested=False)
    s.rx.note_p3_speedup_emitted(1, 2)
    _, trace = budget(s, monkeypatch, 25.)
    onair._scan_frame(s.rx, raw.astype(float) / 32768, p['origin'], tracked_only=True)
    assert trace == ['sl2_packet_at', 'sl2_packet_at', 'sl1_packet_at']
    assert s.packets[0].packet[0] == 1
    assert not s.rx.sl3_prekey_expected()


def test_carrier_requalification_waits_for_room_and_expires_on_turn_change(monkeypatch):
    audio = np.pad(placement.link_packet(2, b'fresh', 0x23), (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    s = receiver(row, requested=False)
    read = lambda a: s.rx.sync.sl2_packet_at(a, row)
    assert s.rx._p3_sl2_at_nominal(read, audio) is not None
    pending = s.rx._p3_carrier_pending
    s.host.peer.live = SimpleNamespace(clamp_late=lambda deadline: True)
    s.rx.nominate_p3_offset_after_control(48000)
    assert s.rx._p3_carrier_pending is pending
    assert s.rx._p3_carrier_candidate is None
    assert s.rx.p3_transmit_offset_hz == -13.
    s.host.arq.role = arq.ISS
    s.rx.nominate_p3_offset_after_control()
    assert s.rx._p3_carrier_pending is s.rx._p3_carrier_candidate is None
    assert s.rx.p3_transmit_offset_hz == -13.


@pytest.mark.parametrize('allowance', [14., 18., 22.])
@pytest.mark.wall_clock
def test_faded_sl2_keeps_fallback_budget_under_callback_load(audio_callback_load, allowance):
    audio = np.pad(placement.link_packet(2, b'fresh', 0x23), (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    audio[:row - 480] = 0.
    audio += np.random.default_rng(1003).normal(0., .05, audio.size)
    for _ in range(50):
        s = receiver(row, hz=0.)
        peer = s.host.peer
        peer.raster = irs_grid(anchor=row - 4320)
        peer.slot = 204
        peer.key_instant = lambda *args: round(allowance * 48)
        with receive_gc():
            started = time.perf_counter()
            peer.live = SimpleNamespace(transmit=True,
                clamp_late=lambda deadline: (time.perf_counter() - started) * onair.FS > deadline)
            with contextlib.redirect_stdout(io.StringIO()):
                onair._scan_frame(s.rx, audio, 0, tracked_only=True)
            ev, = s.packets
            assert ev.packet == (2, 0x23, b'fresh', True)
            assert peer.sent == [('cs', arq.CS_REQUEST)]
            assert s.rx._p3_acquisition_fits(0)


@pytest.mark.parametrize('level,hz,role,long_cycle,command,expected_ms', [
    (None, 0., arq.IRS, False, False, 23.33),
    (1, -13., arq.IRS, False, False, 30.84),
    (2, 0., arq.IRS, False, False, 23.33),
    (2, -13., arq.IRS, False, False, 30.84),
    (3, -13., arq.IRS, False, False, 12.52),
    (2, -13., arq.ISS, False, False, 15.82),
    (2, -13., arq.IRS, True, False, 15.82),
    (2, -13., arq.IRS, False, True, 12.52),
])
def test_speculative_sl3_reserves_every_reachable_fallback(
        level, hz, role, long_cycle, command, expected_ms):
    s = receiver(6500, hz=hz)
    s.rx.sync.packet_level = level
    s.host.arq.role = role
    s.rx._p3_cycle_n = 144000 if long_cycle else 60000
    s.host.arq._cycle_command_emitted = command
    assert s.rx._p3_speculative_sl3_reserve() * 1000 == pytest.approx(expected_ms)


@pytest.mark.parametrize('hold', [201, 206])
@pytest.mark.wall_clock
def test_recorded_sl3_at_18_ms_is_declined_under_callback_load(audio_callback_load, hold):
    audio, p = crop(hold)
    for _ in range(5):
        s = receiver(p['row'])
        peer = s.host.peer
        peer.raster = irs_grid(anchor=p['row'] - 4320)
        peer.slot = 204
        peer.key_instant = lambda *args: round(18. * 48)
        with receive_gc():
            started = time.perf_counter()
            peer.live = SimpleNamespace(transmit=True,
                clamp_late=lambda deadline: (time.perf_counter() - started) * onair.FS > deadline)
            with contextlib.redirect_stdout(io.StringIO()):
                onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
            assert not s.packets and not peer.sent


@pytest.mark.parametrize('hz,allowance,admitted', [
    (0., 29.32, False), (0., 29.35, True),
    (-13., 36.82, False), (-13., 36.86, True),
])
def test_speculative_sl3_admission_boundary_preserves_sl1(monkeypatch, hz, allowance, admitted):
    audio = np.pad(placement.link_packet(1, b'fresh', 0x23), (4800, 4800))
    row = 4800 + (placement.protocol_config().pulse().size - 1) // 2 + 4320
    audio = p3acquire.compensate(audio, -hz)
    s = receiver(row, hz=hz)
    spent, trace = budget(s, monkeypatch, allowance)
    onair._scan_frame(s.rx, audio, 0, tracked_only=True)
    assert ('wideband_packet_at' in trace) == admitted
    assert trace[-1] == 'sl1_packet_at'
    assert trace.count('sl2_packet_at') == 1 + bool(hz)
    ev, = s.packets
    assert ev.packet == (1, 0x23, b'fresh', True)
    assert s.host.peer.sent == [('cs', arq.CS_REQUEST)]
    assert spent[0] + 6 <= allowance


@pytest.mark.parametrize('p', PINS['live'], ids=lambda p: str(p['hold']))
@pytest.mark.wall_clock
def test_live_sl1_repeats_keep_22_ms_wall_budget(audio_callback_load, p):
    raw = stream('1003-0045')[p['origin']:p['end']]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == p['pcm_sha256']
    audio = raw.astype(float) / 32768
    for _ in range(5):
        s = receiver(p['row'], level=1, hz=-12.9, requested=False)
        s.rx.note_p3_speedup_emitted(1, 2)
        peer = s.host.peer
        peer.raster = irs_grid(anchor=p['row'] - 4320)
        peer.slot = 204
        peer.key_instant = lambda *args: round(22. * 48)
        with receive_gc():
            started = time.perf_counter()
            peer.live = SimpleNamespace(transmit=True,
                clamp_late=lambda deadline: (time.perf_counter() - started) * onair.FS > deadline)
            with contextlib.redirect_stdout(io.StringIO()):
                onair._scan_frame(s.rx, audio, p['origin'], tracked_only=True)
            assert s.packets[0].packet[0] == 1
            assert peer.sent
            assert s.rx._p3_acquisition_fits(0)
