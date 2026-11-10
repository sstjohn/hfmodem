# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.wav import source
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.rx import broadcast as UI
from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.rx.cq_stream import CQDetector, EARLY, FS
from hfmodem.kestrel.rx.observation import CaptureContext, FrameOrigin
from hfmodem.kestrel.rx.ui_spans import (
    CLEAR, CONTESTED, MAX_HORIZON, SpanPublished, SpanResolved, UISpanLedger,
)
from hfmodem.kestrel.tx.beacon import synth_beacon
from hfmodem.kestrel.tx.broadcast import synth_broadcast
from hfmodem.kestrel.tx import varahf500_tx as TX
from hfmodem.kestrel.tx.ui import compact, packetize, packets
from hfmodem.kestrel.ui import UIAddress, UIFrame
from hfmodem.tests.evidence import CORPUS, require_input

FIXTURES = Path(__file__).parent / 'fixtures/kiss-ui-1031'
PEER_FILE = FIXTURES / 'peer-packets.json'
# Recovered from VARA HF emissions, so it stays out of the distribution.
PEER = json.loads(PEER_FILE.read_text())['captures'] if PEER_FILE.exists() else []
peer_captures = pytest.mark.skipif(not PEER, reason=f'recording unavailable: {PEER_FILE}')
NATIVE = json.loads((FIXTURES / 'native-replay.json').read_text())['cases']


def frame(info=b'hello', *, command=0, destination='ALL'):
    return UIFrame(command, (UIAddress(destination), UIAddress('W9SSJ', flags=0x61)),
                   3, 0xF0, info)


def normalized(message):
    return replace(message, addresses=tuple(
        replace(a, flags=0x60 | (a.flags & 0x81 if i > 1 else a.flags & 1))
        for i, a in enumerate(message.addresses)))


def expected(case, key):
    delivered = case[key]
    message = UIFrame.from_ax25(bytes.fromhex(delivered[0])[1:], case['command']) if delivered else normalized(
        UIFrame.from_ax25(bytes.fromhex(case['sent']), case['command']))
    message = replace(message, addresses=tuple(
        replace(a, ssid=a.ssid - 8) if i > 1 and a.base in ('WIDE1', 'WIDE2') and a.ssid >= 10 else a
        for i, a in enumerate(message.addresses)))
    return message.ax25()


@peer_captures
@pytest.mark.parametrize('case', PEER, ids=lambda r: f"{r['session']}/{r['case']}")
def test_all_peer_packets_deliver_received_bytes(case):
    result = UI.parse_packets([bytes.fromhex(p) for p in case['packets']], level=case['level'])
    if (case['session'], case['case']) == ('peer-main-1', 'src-blank'):
        # The recorded blank-source emission is garbled; Kestrel rejects its malformed group.
        assert result is None
        return
    assert result is not None
    assert result.command == case['command']
    assert result.ax25() == expected(case, 'delivered')


@pytest.mark.parametrize('case', NATIVE, ids=lambda r: f"{r['source']}/{r['case']}")
def test_all_native_alternatives_deliver_received_bytes(case):
    result = UI.parse_packets([bytes.fromhex(p) for p in case['packets']], level=case['level'])
    assert result is not None
    assert result.ax25() == expected(case, 'peer_delivered')


def crc(raw):
    return bytes(raw) + crc16_genibus(raw).to_bytes(2, 'big')


@peer_captures
@pytest.mark.parametrize('case', [r for r in PEER if len(r['packets']) > 1],
                         ids=lambda r: f"{r['session']}/{r['case']}")
def test_no_partial_or_corrupt_group_delivers(case):
    level, encoded = case['level'], [bytes.fromhex(p) for p in case['packets']]
    for count in range(len(encoded)):
        assert UI.parse_packets(encoded[:count], level=level) is None
    bad = encoded.copy()
    middle = len(bad) // 2
    bad[middle] = bad[middle][:-1] + bytes([bad[middle][-1] ^ 1])
    assert UI.parse_packets(bad, level=level) is None
    bad = encoded.copy()
    raw = bytearray(bad[0][:-2])
    raw[1] ^= 0x20
    bad[0] = crc(raw)
    assert UI.parse_packets(bad, level=level) is None
    assert UI.parse_packets(encoded[-1:] + encoded[:-1], level=level) is None
    bad = encoded.copy()
    raw = bytearray(bad[-1][:-2])
    raw[0] |= 128
    bad[-1] = crc(raw)
    assert UI.parse_packets(bad, level=level) is None


def test_table_code_five_and_invalid_padding_are_malformed():
    message = frame(b'!4303.14N/08924.53W-symbol', destination='APRS')
    core, mode = compact(message)
    fields = UI._Compact(core)
    fields.header()
    bits = ''.join(f'{b:08b}' for b in core)
    table = fields.at + 21
    assert bits[table:table + 4] == '1000'
    bits = bits[:table] + '1101' + bits[table + 4:]
    bad = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    level, encoded = packets(bad, mode)
    assert UI.parse_packets(encoded, level=level) is None
    for message in (frame(b'hello'), frame(b'\x80binary'), message):
        core, mode = compact(message)
        level, encoded = packets(core[:-1] + bytes([core[-1] ^ 1]), mode)
        assert UI.parse_packets(encoded, level=level) is None


@pytest.mark.parametrize('command', [0, 1])
def test_empty_information_and_normalized_flags(command):
    message = frame(b'', command=command)
    message = replace(message, addresses=tuple(replace(a, flags=a.flags | 0x80) for a in message.addresses))
    level, encoded = packetize(message)
    assert UI.parse_packets(encoded, level=level).ax25() == normalized(message).ax25()


def feed(detector, samples):
    results = []
    for at in range(0, len(samples), 4800):
        results.extend(detector.feed(samples[at:at + 4800]))
        assert len(detector._candidates) + len(detector._groups) <= 4
        assert sum(g.extended for g in detector._groups) <= 1
        if detector._groups and not detector._candidates:
            level = detector._groups[0].level
            assert len(detector._buffer) <= (UI.PERIOD[level] + 2048) // 4
    return results


AUDIO_CASES = [('peer-main-1', 'len400'), ('peer-aprs3-1', 'q-core23'),
               ('peer-aprs3-1', 'q-core50'), ('peer-main-1', 'via1-used'),
               ('peer-tail-1', 'aprs-pos'), ('peer-aprs-1', 'i-object-kill'),
               ('peer-limit-1', 'len800')]


@peer_captures
@pytest.mark.parametrize('session,name', AUDIO_CASES)
def test_peer_stream_in_tenth_second_chunks(session, name):
    path = require_input(CORPUS / 'kestrel-kiss-1031' / session / f'{name}.wav')
    rate, audio = wavfile.read(path)
    assert rate == 48000
    detector = CQDetector()
    results = feed(detector, np.r_[audio, np.zeros(48000)])
    case = next(r for r in PEER if (r['session'], r['case']) == (session, name))
    assert [r.ax25() for r in results] == [expected(case, 'delivered')]


@pytest.mark.parametrize('message,level,count', [
    (frame(b'!4303.14N/08924.53W-' + b'x' * 3, destination='APRS'), 1, 3),
    (frame(b'!4303.14N/08924.53W-' + b'x' * 38, destination='APRS'), 2, 3),
    (frame(b'\xff' * 730), 4, 18),
])
def test_native_stream_and_successive_long_groups(message, level, count):
    assert packetize(message)[0] == level
    assert len(packetize(message)[1]) == count
    wave = synth_broadcast(message, amplitude=0.2)
    detector = CQDetector()
    results = feed(detector, np.r_[np.zeros(24000), wave, np.zeros(48000), wave, np.zeros(48000)])
    assert [r.ax25() for r in results] == [message.ax25()] * 2


@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('lost', [False, True])
def test_deadline_and_slot_retirement_at_each_level(monkeypatch, level, lost):
    message = frame(b'x' * (20 if level == 1 else 60 if level == 2 else 100))
    _, encoded = packetize(message, level=level)
    count = len(encoded)
    detector = CQDetector()
    candidate = (32., 0, 1, 0, EARLY, f'ui{level}')
    group = detector._start_ui(candidate, level, None if lost else encoded[0], 0.)
    expected_count = 18 if lost else count
    assert group.deadline * 4 == UI.deadline(0, level, expected_count)
    seen = []

    def slot(candidate, actual_level, index, hz=None, offset=0):
        assert actual_level == level
        seen.append(index)
        assert len(seen) == len(set(seen))
        return encoded[index], 0., 0., offset

    monkeypatch.setattr(detector, '_ui_slot', slot)
    monkeypatch.setattr(detector, '_search', lambda: setattr(detector, '_scan', detector._output_count - 512))
    if lost:
        end = group.deadline * 4
        assert feed(detector, np.zeros(end - 4800)) == []
        assert detector._groups == [group]
        assert feed(detector, np.zeros(9600)) == []
        assert detector._groups == [] and seen == []
    else:
        final_ready = 67072 + (count - 1) * UI.PERIOD[level] + UI.BODY_SAMPLES[level] + 2048
        assert final_ready > count * UI.PERIOD[level] + 48000
        assert feed(detector, np.zeros(final_ready - 4800)) == []
        assert detector._groups == [group]
        assert group.deadline == UI.deadline(0, level, count) // 4
        results = feed(detector, np.zeros(9600))
        assert [r.ax25() for r in results] == [message.ax25()]
        assert seen == list(range(1, count))
        assert not detector._groups


@pytest.mark.parametrize('meta,count', [(0x60, 2), (0x50, 18), (0x00, 2), (0x10, 18),
                                      (0x01, 3), (0xAF, 17)])
def test_full_metadata_byte_seeds_deadline(meta, count):
    assert UI.packet_count(meta) == count


def test_false_headers_evict_oldest_provisional_and_preserve_extended(monkeypatch):
    detector = CQDetector()
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    _, encoded = packetize(frame(b'x' * 400))
    group = detector._start_ui(candidate, 4, encoded[0], 0.)
    monkeypatch.setattr(detector, '_ui_slot', lambda *args: (None, None, None, 0))
    monkeypatch.setattr(detector, '_search', lambda: setattr(detector, '_scan', detector._output_count - 512))
    for index in range(1, 100):
        detector._add_candidate((20., index * FS // 10, 1, 0, EARLY, 'ui4'))
        assert detector.feed(np.zeros(4800)) == []
        assert len(detector._candidates) + len(detector._groups) <= 4
        assert detector._groups == [group]
    assert [c[1] for c in detector._candidates] == [i * FS // 10 for i in (97, 98, 99)]
    detector.reset()
    assert not detector._groups and not detector._candidates


def test_overlapping_extended_groups_drop_second_and_recover_next(monkeypatch, caplog):
    import logging

    detector = CQDetector()
    message = frame(b'x' * 100)
    _, encoded = packetize(message)
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    first = detector._start_ui(candidate, 4, encoded[0], 0.)
    second = (32., FS, 1, 0, EARLY, 'ui4')
    with caplog.at_level(logging.DEBUG):
        assert detector._start_ui(second, 4, encoded[0], 0.) is None
    assert 'Overlapping extended UI group dropped' in caplog.text
    seen = []

    def slot(candidate, level, index, hz=None, offset=0):
        seen.append((candidate[1], index))
        assert len(seen) == len(set(seen))
        return encoded[index], 0., 0., offset

    monkeypatch.setattr(detector, '_ui_slot', slot)
    monkeypatch.setattr(detector, '_search', lambda: setattr(detector, '_scan', detector._output_count - 512))
    results = feed(detector, np.zeros(first.deadline * 4))
    assert [r.ax25() for r in results] == [message.ax25()]
    later = (32., detector._output_count, 1, 0, EARLY, 'ui4')
    next_group = detector._start_ui(later, 4, encoded[0], 0.)
    results = feed(detector, np.zeros((next_group.deadline - detector._output_count) * 4))
    assert [r.ax25() for r in results] == [message.ax25()]
    assert not detector._groups


@pytest.mark.parametrize('expected_level,extra', [(1, 3), (2, 38)])
@pytest.mark.parametrize('change', ['truncate1', 'truncate2', 'middle'])
def test_native_stream_truncation_and_corruption_never_deliver(change, expected_level, extra):
    message = frame(b'!4303.14N/08924.53W-' + b'x' * extra, destination='APRS')
    level, encoded = packetize(message)
    assert level == expected_level and len(encoded) == 3
    audio = synth_broadcast(message, amplitude=0.2)
    if change.startswith('truncate'):
        count = int(change[-1])
        audio = audio[:67072 + count * UI.PERIOD[level] - 1536]
    else:
        at = 67072 + UI.PERIOD[level]
        audio[at:at + UI.BODY_SAMPLES[level]] = 0
    assert feed(CQDetector(), np.r_[audio, np.zeros(8 * 48000)]) == []


@pytest.mark.parametrize('standing', [False, True])
def test_received_general_path_frame_fans_out_without_connected_data(standing):
    import threading
    from types import SimpleNamespace
    from hfmodem.ax25 import kiss
    from hfmodem.kestrel.host.audio_runtime import KissObserver
    from hfmodem.kestrel.host.kiss import KissPort, _Client
    from hfmodem.kestrel.host.server import HostSession
    from hfmodem.kestrel.host.vara_modem import _HostIO

    message = frame(b'\xffbinary', command=1, destination='APRS')
    message = replace(message, addresses=(message.addresses[0], UIAddress('W9SSJ'),
                                        UIAddress('WIDE2', 2, 0xE1)), control=0x13, pid=0xCF)
    level, encoded = packetize(message)
    received = UI.parse_packets(encoded, level=level)
    port = object.__new__(KissPort)
    port._lock = threading.Lock()
    port._clients = {object(): _Client(), object(): _Client()}
    port._clock = lambda: 0
    if standing:
        observer = KissObserver(SimpleNamespace(_kiss=port))
    else:
        observer = object.__new__(HostSession)
        observer._kiss = port
    modem = SimpleNamespace(observer=observer)
    _HostIO(modem, SimpleNamespace()).cq(received)
    expected_wire = kiss.encode(message.ax25(), message.command)
    assert [list(c.output) for c in port._clients.values()] == [[expected_wire], [expected_wire]]


def test_connected_station_feeds_shared_ui_receiver_and_delivers_kiss_once(monkeypatch):
    from hfmodem.kestrel.vara.vara_arq import VaraState, _TURN_PEER
    from .test_receive_context_runtime import owner

    runtime, transport, delivered = owner()
    runtime.step(0)
    station = runtime.modem.station
    station.role, station.caller, station.called = 'initiator', 'W9SSJ', 'K1ABC'
    station.state, station.turn = VaraState.CONNECTED, _TURN_PEER
    context = runtime.receive_context
    assert station.cq_detector is context.detector
    message = frame(b'inside link')
    calls = []
    original = context.detector.feed

    def feed(samples, *, sample_start, ui=True):
        calls.append((sample_start, len(samples), ui))
        return original(samples, sample_start=sample_start, ui=ui) + [message]

    monkeypatch.setattr(context.detector, 'feed', feed)
    monkeypatch.setattr(station, '_feed_shared_front', lambda samples: None)
    for _ in range(2):
        transport.chunks.append((0, np.zeros(4800)))
        runtime.step(0)
    assert calls == [(0, 4800, True)]
    assert delivered == [(message.command, message.encode())]
    assert station._rx_window.W is None and station._rx_window.K is None
    assert station._rx_window.R == b''


def test_disabling_ui_retires_groups_and_preserves_cq_search():
    detector = CQDetector()
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    _, encoded = packetize(frame(b'x' * 400))
    detector._start_ui(candidate, 4, encoded[0], 0.)
    detector._add_candidate(candidate)
    cq = candidate[:5] + ('cq',)
    detector._add_candidate(cq)
    assert detector.feed(np.zeros(4800), ui=False) == []
    assert not detector._groups
    assert detector._candidates == [cq]


@pytest.mark.parametrize('scenario', ['false_headers', 'overlapping_long_groups'])
def test_audio_candidate_bounds_and_recovery_after_eviction(scenario):
    first = synth_broadcast(frame(b'x' * 400), amplitude=0.2)
    if scenario == 'false_headers':
        audio = np.tile(np.r_[first[:32 * 2048], np.zeros(16000)], 8)
    else:
        second = synth_broadcast(frame(b'y' * 400), amplitude=0.2)
        shift = 67072 + 2 * UI.PERIOD[4]
        audio = np.zeros(max(len(first), shift + len(second)))
        audio[:len(first)] += first
        audio[shift:shift + len(second)] += second
    detector = CQDetector()
    results = feed(detector, audio)
    if scenario == 'false_headers':
        assert results == []
    assert feed(detector, np.zeros(UI.deadline(0, 4) + 48000)) == []
    assert not detector._groups and not detector._candidates
    following = frame(b'recovered after eviction')
    results = feed(detector, np.r_[synth_broadcast(following, amplitude=0.2), np.zeros(48000)])
    assert [r.ax25() for r in results] == [following.ax25()]


def apply_span_events(detector, ledger):
    events = detector.take_span_events()
    for event in events:
        ledger.apply(event)
    return events


def slot_origin(span, index):
    start = span.first_training_origin + index * span.period
    return FrameOrigin(span.tune_epoch, 0, start, start + UI.BODY_SAMPLES[span.level],
                       span.level, 500)


@pytest.mark.parametrize('message,level', [
    (frame(b'!4303.14N/08924.53W-xxx', destination='APRS'), 1),
    (frame(b'!4303.14N/08924.53W-' + b'x' * 38, destination='APRS'), 2),
    (frame(b'x' * 100), 4),
])
def test_span_publication_precedes_first_body_decode(message, level):
    assert packetize(message)[0] == level
    wave = synth_broadcast(message, amplitude=.2)
    detector = CQDetector(sample_start=900000, tune_epoch=7)
    # The scanner batches FFT work; header acceptance needs its next search.
    header_ready = 67072 + 9600
    assert feed(detector, wave[:header_ready]) == []
    assert detector.turbo_attempts == 0
    events = detector.take_span_events()
    published = [e.span for e in events if isinstance(e, SpanPublished)]
    assert len(published) == 1
    span, = published
    assert (span.tune_epoch, span.level, span.possible_packet_count) == (7, level, 18)
    assert abs(span.first_training_origin - (900000 + 67072)) <= span.tolerance
    assert not any(isinstance(e, SpanResolved) for e in events)
    assert detector.take_span_events() == []
    results = feed(detector, np.r_[wave[header_ready:], np.zeros(48000)])
    assert [r.ax25() for r in results] == [normalized(message).ax25()]


@peer_captures
@pytest.mark.parametrize('damage', ['erased_first', 'failed_reference'])
def test_q_core50_unread_shared_header_body_keeps_eighteen_slots(monkeypatch, damage):
    path = require_input(CORPUS / 'kestrel-kiss-1031' / 'peer-aprs3-1' / 'q-core50.wav')
    rate, audio = wavfile.read(source(path))
    assert rate == 48000
    probe = CQDetector()
    first = None
    for at in range(0, len(audio), 4800):
        probe.feed(audio[at:at + 4800])
        publications = [e.span for e in probe.take_span_events() if isinstance(e, SpanPublished)]
        if publications:
            first = publications[0]
            break
    assert first is not None and first.level == 2
    audio = audio.copy()
    if damage == 'erased_first':
        start = max(0, first.first_training_origin - first.tolerance)
        stop = first.first_training_origin + UI.BODY_SAMPLES[2] + first.tolerance
        audio[start:stop] = 0
    else:
        monkeypatch.setattr(RX, '_l3_align', lambda *args: (0, 0, 0, None))
    detector, ledger = CQDetector(), UISpanLedger()
    assert feed(detector, np.r_[audio, np.zeros(48000)]) == []
    events = apply_span_events(detector, ledger)
    span = next(s for s in ledger.spans if s.level == 2)
    assert span.possible_packet_count == 18
    assert ledger.classify(slot_origin(span, 17)) == CONTESTED
    assert not any(isinstance(e, SpanResolved) and e.outcome in ('beacon', 'singleton', 'group')
                   for e in events)
    if damage == 'failed_reference':
        assert detector.turbo_attempts == 0
        assert any(isinstance(e, SpanResolved) and e.outcome == 'failed' for e in events)


def test_validated_beacon_closes_future_slots_before_following_data():
    detector, ledger = CQDetector(), UISpanLedger()
    beacon = synth_beacon('20OEN63BCW9SSJ', amplitude=.2)
    raw = TX.build_l3_frame(b'D' * RX.INDEX_RECORDS[2].payload, 0x99, 2)
    wave = np.pad(TX.synth_l3_burst(raw, onset=12000, level=2), (0, 12000))
    training = 67072 + UI.PERIOD[2]
    audio = np.r_[beacon, np.zeros(training - len(beacon)), wave[12000:], np.zeros(48000)]
    results = feed(detector, audio)
    assert [r.text for r in results] == ['20OEN63BCW9SSJ']
    events = apply_span_events(detector, ledger)
    span, = ledger.spans
    assert span.resolution == 'beacon' and span.possible_packet_count == 1
    resolved = [e for e in events if isinstance(e, SpanResolved) and e.outcome == 'beacon']
    assert len(resolved) == 1 and resolved[0].validated_end == span.classification_end
    assert ledger.classify(slot_origin(span, 0)) == CONTESTED
    assert ledger.classify(slot_origin(span, 1)) == CLEAR
    capture = training - 12000
    decoded = RX.decode_burst_frames(wave, 12000, len(wave) - 12000, level=2,
                                     context=CaptureContext(0, 1, capture, 48000))[0]
    assert decoded.crc_ok and decoded.frame_bytes == raw
    assert abs(decoded.origin.training_origin - slot_origin(span, 1).training_origin) <= span.tolerance
    assert ledger.classify(decoded.origin) == CLEAR


@pytest.mark.parametrize('case', [r for r in PEER if r['case'] == 'len400']
                         + [r for r in NATIVE if r['case'] == 'n-len400'],
                         ids=lambda r: r.get('session', r.get('source')))
@pytest.mark.parametrize('damage', ['middle', 'invalid_early_final'])
def test_failed_len400_and_invalid_early_final_never_acquit(monkeypatch, case, damage):
    encoded = [bytes.fromhex(p) for p in case['packets']]
    assert len(encoded) > 2
    detector, ledger = CQDetector(), UISpanLedger()
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    group = detector._start_ui(candidate, 4, encoded[0], 0.)
    early = crc(bytes([encoded[1][0] & 0x7F]) + encoded[1][1:-2])
    assert UI.parse_packets([encoded[0], early], level=4) is None

    def slot(candidate, level, index, hz=None, offset=0):
        raw = (None if index == 2 else encoded[index]) if damage == 'middle' else early
        return raw, 0., 0., offset

    monkeypatch.setattr(detector, '_ui_slot', slot)
    assert detector._advance_ui(group, group.deadline) is None
    events = apply_span_events(detector, ledger)
    span, = ledger.spans
    assert span.possible_packet_count == UI.packet_count(encoded[0][1])
    assert ledger.classify(slot_origin(span, len(encoded) - 1)) == CONTESTED
    assert not any(isinstance(e, SpanResolved) and e.outcome == 'group' for e in events)


def test_ambiguous_two_packet_collision_delivers_but_keeps_later_slots(monkeypatch):
    # Binary information keeps the shared prefix independent of message length;
    # the suffixes make both compact-core checksum high nibbles 0x50.
    message = frame(b'\x80' + b'x' * 58 + b'\x00')
    long_message = frame(b'\x80' + b'x' * 738 + b'\x08')
    _, encoded = packetize(message, level=4)
    _, long_encoded = packetize(long_message, level=4)
    assert len(encoded) == 2 and len(long_encoded) == 18
    first = encoded[0]
    assert first == long_encoded[0]
    assert first[1] == 0x50
    for source_message, group_packets in ((message, encoded), (long_message, long_encoded)):
        assert all(crc(raw[:-2]) == raw for raw in group_packets)
        assert UI.parse_packets(group_packets, level=4).ax25() == normalized(source_message).ax25()
    detector, ledger = CQDetector(), UISpanLedger()
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    group = detector._start_ui(candidate, 4, first, 0.)
    monkeypatch.setattr(detector, '_ui_slot', lambda *args: (encoded[1], 0., 0., 0))
    result = detector._advance_ui(group, group.deadline)
    assert result.ax25() == normalized(message).ax25()
    apply_span_events(detector, ledger)
    span, = ledger.spans
    assert span.resolution == 'group' and span.ambiguous_2_or_18
    assert span.possible_packet_count == 18
    assert span.deadline == UI.deadline(span.header_origin, 4, 18)
    assert ledger.classify(slot_origin(span, 17)) == CONTESTED


def test_eviction_refinement_and_decoder_reset_preserve_header_coverage():
    detector, ledger = CQDetector(), UISpanLedger()
    candidate = (32., 0, 1, 0, EARLY, 'ui4')
    detector._add_candidate(candidate)
    detector._add_candidate((33., 128, 1, 0, EARLY, 'ui4'))
    for index in range(1, 20):
        detector._add_candidate((32., index * 1200, 1, 0, EARLY, 'ui4'))
    assert len(detector._candidates) == 4
    assert candidate not in detector._candidates
    apply_span_events(detector, ledger)
    assert len(ledger.spans) == 20
    first = ledger.spans[0]
    assert ledger.classify(slot_origin(first, 17)) == CONTESTED
    detector.reset()
    detector.gap(500000)
    apply_span_events(detector, ledger)
    assert ledger.classify(slot_origin(first, 17)) == CONTESTED
    assert all(s.resolution == 'truncated' for s in ledger.spans)


def test_rejected_overlapping_long_group_retains_provisional_extent():
    detector, ledger = CQDetector(), UISpanLedger()
    _, encoded = packetize(frame(b'x' * 400))
    detector._start_ui((32., 0, 1, 0, EARLY, 'ui4'), 4, encoded[0], 0.)
    assert detector._start_ui((32., FS, 1, 0, EARLY, 'ui4'), 4, encoded[0], 0.) is None
    apply_span_events(detector, ledger)
    assert len(ledger.spans) == 2
    rejected = max(ledger.spans, key=lambda s: s.header_origin)
    assert rejected.possible_packet_count == 18
    assert ledger.classify(slot_origin(rejected, 17)) == CONTESTED


@pytest.mark.parametrize('level', [1, 2, 4])
@pytest.mark.parametrize('gap', [False, True])
def test_ui_slots_agree_numerically_with_data_origins_across_tx_gap(level, gap):
    message = frame(b'!4303.14N/08924.53W-' + b'x' * (3 if level == 1 else 38),
                    destination='APRS') if level != 4 else frame(b'x' * 100)
    actual, encoded = packetize(message)
    assert actual == level
    wave = synth_broadcast(message, amplitude=.2)
    anchor = 900000
    detector, ledger = CQDetector(sample_start=anchor, tune_epoch=7), UISpanLedger(7)
    assert feed(detector, wave[:67072 + 9600]) == []
    apply_span_events(detector, ledger)
    span, = ledger.spans
    if gap:
        detector.gap(anchor + 67072 + UI.PERIOD[level])
        apply_span_events(detector, ledger)
        assert not detector._candidates and not detector._groups
    for index in (0, 1):
        start = 67072 + index * UI.PERIOD[level]
        if gap and index == 0:
            continue
        lo = start - 4096
        hi = start + UI.BODY_SAMPLES[level] + 4096
        context = CaptureContext(7, int(gap), anchor + lo, 48000)
        decoded = RX.decode_burst_frames(wave[lo:hi], start - lo,
                                         start - lo + UI.BODY_SAMPLES[level], level,
                                         context=context)[0]
        assert decoded.crc_ok and decoded.frame_bytes == encoded[index]
        predicted = span.first_training_origin + index * span.period
        assert abs(predicted - decoded.origin.training_origin) <= span.tolerance
        assert ledger.classify(decoded.origin) == CONTESTED
    if gap:
        assert feed(detector, np.r_[wave[67072 + UI.PERIOD[level]:], np.zeros(48000)]) == []
        assert detector.input_clock == anchor + len(wave) + 48000


@pytest.mark.parametrize('level', [1, 2, 4])
def test_packet_alignment_callback_uses_the_data_origin_convention(level):
    _, encoded = packetize(frame(b'x' * 60), level=level)
    raw = encoded[0]
    body = (TX.synth_burst(raw, onset=12000, ncol=403) if level == 4 else
            TX.synth_l3_burst(raw, onset=12000, level=level))
    body = np.pad(body, (0, 12000))
    aligned = []
    assert UI.decode_packet(body, 12000, level=level, on_alignment=aligned.append) == raw
    assert len(aligned) == 1
    stop = 12000 + UI.BODY_SAMPLES[level]
    data = RX.decode_burst_frames(body, 12000, stop, level=level)[0]
    assert data.crc_ok and data.frame_bytes == raw
    assert abs(aligned[0] - data.origin.training_origin) <= 64


@pytest.mark.parametrize('rate', [12000, 24000, 44100, 48000])
def test_header_frontier_and_watermark_include_filter_anchor_and_rounding(rate):
    anchor = 912345
    detector = CQDetector(rate, sample_start=anchor)
    delay = 24 if rate > 24000 else 0
    assert detector.header_frontier == anchor - delay
    previous = detector.classification_watermark
    for _ in range(40):
        assert detector.feed(np.zeros(4800)) == []
        history = 0 if detector._spectra is None else len(detector._spectra)
        local = max(0, detector._scan - history * 128)
        frontier = anchor + round(local * rate / FS) - delay
        assert detector.header_frontier == frontier
        assert detector.classification_watermark == (frontier + round(67072 * rate / 48000)
                                                       - round(16384 * rate / 48000))
        assert detector.classification_watermark >= previous
        previous = detector.classification_watermark
    clock = detector.input_clock
    detector.gap(clock + 12345)
    assert detector.header_frontier == clock + 12345 - delay
    assert detector.classification_watermark >= previous
    assert detector.feed(np.zeros(4800)) == []


@pytest.mark.parametrize('phase', [0, 1, 4799])
def test_watermark_never_passes_a_header_still_awaiting_search(phase):
    detector = CQDetector()
    wave = np.r_[np.zeros(phase), synth_broadcast(frame(b'x' * 100), amplitude=.2)[:67072],
                 np.zeros(9600)]
    previous = detector.classification_watermark
    published = []
    for at in range(0, len(wave), 4800):
        assert detector.feed(wave[at:at + 4800]) == []
        for event in detector.take_span_events():
            if isinstance(event, SpanPublished):
                published.append(event.span)
                assert previous <= event.span.first_training_origin - event.span.tolerance
        previous = detector.classification_watermark
    assert len(published) == 1


def test_detector_retune_discards_old_events_without_reusing_span_ids():
    detector = CQDetector(tune_epoch=7)
    detector._add_candidate((32., 0, 1, 0, EARLY, 'ui4'))
    first, = detector.take_span_events()
    detector.reset(sample_start=500000, tune_epoch=8)
    detector._add_candidate((32., 0, 1, 0, EARLY, 'ui4'))
    events = detector.take_span_events()
    assert len(events) == 1 and events[0].span.tune_epoch == 8
    assert events[0].span.id > first.span.id


def test_undrained_detector_events_and_false_headers_remain_bounded(monkeypatch):
    detector = CQDetector()
    monkeypatch.setattr(detector, '_search',
                        lambda: setattr(detector, '_scan', detector._output_count - 512))
    monkeypatch.setattr(detector, '_decode_ui', lambda candidate: None)
    limit = 2 * MAX_HORIZON // 48000 + 3
    for _ in range(400):
        detector._add_candidate((32., detector._output_count, 1, 0, EARLY, 'ui4'))
        assert detector.feed(np.zeros(48000)) == []
        assert len(detector._candidates) <= 4
        assert len(detector._span_ledger.spans) <= limit
        assert len(detector._span_events) <= limit
    span = detector._span_ledger.spans[-1]
    for _ in range(100):
        detector._resolve_span(span, 'failed')
        detector._resolve_span(span, 'truncated')
    assert len(detector._span_events[span.id]) <= 3
    events = detector.take_span_events()
    ledger = UISpanLedger()
    for event in events:
        ledger.apply(event)
    assert ledger.classify(slot_origin(span, 17)) == CONTESTED
