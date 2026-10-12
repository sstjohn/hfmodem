import hashlib
import json
import os
import wave
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert, resample_poly

from hfmodem.core.wav import read
from hfmodem.shrike import p4decode, p4frame, p4rx, p4short, p4shortlevels
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.shrike.test_p4turn import machine


ROOT = Path(__file__).resolve().parents[5]
FS = 48000
DIGEST = 'ae9ea9706f28fbffffa0d9d3e75c4299d17c5c54a02bf3d15ffbd0fda5a51c81'


@pytest.fixture(scope='module')
def recording():
    capture = ROOT / 'captures/onair-1002-2037'
    if not (capture / 'stream.flac').exists():
        pytest.skip('local recording absent')
    probe = json.loads((capture / 'p4-probe.json').read_text())
    return read(capture / 'stream.flac'), [r for r in probe['heard'] if r['kind'] == 'normal'], probe


def receiver(row):
    at, cfo = row['onset_sample'], row['cfo_hz']
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    rx.peer_onset = at - 60000
    rx._peer_heads = [at-120000, at-60000]
    rx.acked_counter = 0
    return rx


def read_row(audio, row, rx, end=.810):
    at = row['onset_sample']
    origin = at-2400
    return [r for r in rx.read_window(audio[origin:at+round(end*FS)], origin, whole=True)
            if r.observation.frame is not None]


def ideal(value=65, width=4):
    frame = p4frame.decode_field(p4frame.encode_field(bytes([value])*(44*width-4),
                                                  p4frame.Status(1)), data_bytes=44*width-4)
    return frame, 4*(1.-2*p4shortlevels.codeword(frame, width))


def decoder(ranked, now=monotonic, status_ok=lambda *a, **k: True):
    return p4shortlevels.QamShortDecode(ranked, costs={}, events=[], at=0,
                                      status_ok=status_ok, now=now)


def test_complete_eight_iterations_before_unique_success():
    frame, fields = ideal()
    candidate = decoder([(4, fields)])
    assert candidate.decode() == frame
    assert len([e for e in candidate.events if e.get('reason') == 'candidate']) == 8
    assert candidate.events[-1]['stage'] == 'complete'
    assert candidate.width == 4


def test_later_iteration_ambiguity_is_sticky(monkeypatch):
    first, fields = ideal(65)
    second, other = ideal(66)
    fields *= np.sign(fields) == np.sign(other)
    monkeypatch.setattr(p4shortlevels, 'agreement', lambda f: .6)

    def results(*a, **k):
        for iteration, frame in enumerate([first]*6 + [second, first], 1):
            wire = p4frame.encode_field(frame.data, frame.status)
            bits = np.r_[np.unpackbits(np.frombuffer(wire, dtype=np.uint8),
                                       bitorder='little'), np.zeros(8, dtype=np.uint8)]
            yield SimpleNamespace(bits=bits, diagnostics=[SimpleNamespace(iteration=iteration)])

    monkeypatch.setattr(p4decode, 'turbo_iterations', results)
    candidate = decoder([(4, fields)])
    assert candidate.decode() is None and candidate.done
    assert len(candidate.valid) == 2
    assert not any(e.get('reason') == 'accepted' for e in candidate.events)


def test_later_credible_mode_ambiguity_is_sticky():
    qam, qam_fields = ideal()
    psk, psk_fields = ideal(66, 2)
    candidate = decoder([(4, qam_fields), (2, psk_fields)])
    assert candidate.decode() is None and candidate.done
    assert {frame for frame, width in candidate.valid.values()} == {qam, psk}
    assert len([e for e in candidate.events if e.get('iterations')]) == 16


def test_saved_iterations_cannot_extend_original_answer_deadline(monkeypatch):
    frame, fields = ideal()
    now, calls = [0.], []
    original = p4decode.turbo_iterations

    def charged(*args, **kwargs):
        for iteration, result in enumerate(original(*args, **kwargs), 1):
            now[0] += .006
            calls.append(iteration)
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', charged)
    candidate = decoder([(4, fields)], now=lambda: now[0])
    assert candidate.decode(.013) is None and calls == [1]
    assert candidate.decode(.100) is None and calls == [1]
    assert candidate.deadline == .013
    assert candidate.valid and candidate.frame is None
    now[0] = .013
    assert candidate.decode(.100) is None and candidate.done
    assert not candidate.valid


@pytest.mark.parametrize('step', [.020, .010])
def test_overrun_never_publishes_even_with_early_crc(monkeypatch, step):
    frame, fields = ideal()
    now = [0.]
    original = p4decode.turbo_iterations

    def charged(*a, **k):
        for result in original(*a, **k):
            now[0] += step
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', charged)
    candidate = decoder([(4, fields)], now=lambda: now[0])
    assert candidate.decode(.010) is None
    assert candidate.done and candidate.frame is None
    assert any(e.get('reason') == 'late' for e in candidate.events)
    assert candidate.decode(1.) is None


def test_status_refusal_cannot_become_success():
    frame, fields = ideal()
    candidate = decoder([(4, fields)], status_ok=lambda *a, **k: False)
    assert candidate.decode() is None and candidate.done
    assert len([e for e in candidate.events if e.get('reason') == 'status']) == 8


def test_failed_qam_fallback_overrun_cannot_report_late_success(monkeypatch):
    _, qam = ideal()
    frame, bpsk = ideal(width=1)
    now = [0.]
    monkeypatch.setattr(p4shortlevels, 'agreement', lambda f: .6 if len(f) == 4224 else .5)
    original = p4decode.turbo_iterations
    decoded = iter([None]*8+[frame])
    monkeypatch.setattr(p4shortlevels, '_frame', lambda *a: next(decoded))

    def charged(sys, *a, **k):
        for result in original(sys, *a, **k):
            now[0] += .001 if len(sys) == 1408 else .050
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', charged)
    candidate = decoder([(4,qam),(1,bpsk)], now=lambda: now[0])
    assert candidate.decode(.030) is None and candidate.done
    assert candidate.frame is None
    assert candidate.events[-1]['reason'] == 'late'
    assert not any(e.get('reason') == 'accepted' for e in candidate.events)


@pytest.mark.parametrize('index', [2, 8])
@pytest.mark.realtime
@pytest.mark.skipif(os.environ.get('HFMODEM_REALTIME') != '1',
                    reason='explicit host performance run; charged semantic pins run normally')
def test_recorded_complete_domain_timely_positive(recording, index):
    audio, rows, probe = recording
    row = rows[index-1]
    decision = min((e for e in probe['link_events'] if e.get('event') == 'Decision'
                    and e.get('decision_at')), key=lambda e: abs(e['at']-row['onset_sample']))
    rx = receiver(row)
    gate = monotonic()+(decision['decision_at']-row['capture_end_sample'])/FS
    rx.normal_deadline = lambda at: gate
    records = read_row(audio, row, rx)
    finished = monotonic()
    assert len(records) == 1, dict(margin_ms=1000*(gate-finished),
        steps=[(e.get('iterations'), e.get('elapsed_ms'), e.get('allowance_ms'),
                e.get('budget_ms'), e.get('reason')) for e in rx.hold.events
               if e.get('event') == 'normal-decode'])
    assert finished < gate
    assert records[0].observation.speedlevel == 'SL9/10-short'
    assert hashlib.sha256(records[0].observation.frame.data).hexdigest() == DIGEST
    candidate = next(e['decoder'] for e in rx._normal_cache.values() if e.get('decoder'))
    assert candidate.done and not candidate.pending
    assert len([e for e in candidate.events if e.get('reason') == 'candidate']) == 8


@pytest.mark.parametrize('index', [2, 8])
def test_recorded_complete_domain_with_declared_costs(recording, monkeypatch, index):
    audio, rows, probe = recording
    row = rows[index-1]
    rx = receiver(row)
    now = [.010]
    rx.hold.now = lambda: now[0]
    rx.normal_deadline = lambda at: .0614166666666667
    original = p4decode.turbo_iterations

    def charged(*a, **k):
        for result in original(*a, **k):
            now[0] += .006
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', charged)
    records = read_row(audio, row, rx)
    assert len(records) == 1 and now[0] == pytest.approx(.058)
    assert records[0].observation.speedlevel == 'SL9/10-short'
    assert hashlib.sha256(records[0].observation.frame.data).hexdigest() == DIGEST
    candidate = next(e['decoder'] for e in rx._normal_cache.values() if e.get('decoder'))
    assert candidate.done and not candidate.pending


def test_recorded_copy1_incomplete_domain_has_no_success(recording, monkeypatch):
    audio, rows, probe = recording
    row = rows[0]
    rx = receiver(row)
    now = [0.]
    rx.hold.now = lambda: now[0]
    rx.normal_deadline = lambda at: .0614166666666667
    original = p4decode.turbo_iterations

    def charged(sys, *a, **k):
        for result in original(sys, *a, **k):
            now[0] += .006 if len(sys) == 1408 else .003
            yield result

    monkeypatch.setattr(p4decode, 'turbo_iterations', charged)
    assert read_row(audio, row, rx) == []
    candidate = next(e['decoder'] for e in rx._normal_cache.values() if e.get('decoder'))
    assert len(candidate.valid) == 1 and candidate.pending and not candidate.done
    assert candidate.pending[0][0] == 2
    now[0] = .062
    assert rx.refine(deadline=1.) == []
    assert candidate.done and candidate.frame is None


@pytest.mark.parametrize('change', ['none', 'counter', 'generation', 'next-cycle',
                                   'word-clock', 'peer-evidence', 'gap', 'unadmitted',
                                   'generation-during'])
def test_saved_refinement_requires_current_owner_counter_and_input(recording, monkeypatch, change):
    audio, rows, _ = recording
    row = rows[1]
    rx = receiver(row)
    rx.hold.now = lambda: 0.
    rx.normal_deadline = lambda at: .100
    rx._normal_costs[4] = [1.]
    assert not read_row(audio, row, rx)
    candidate = next(e['decoder'] for e in rx._normal_cache.values() if e.get('decoder'))
    assert candidate.pending and not candidate.done
    rx._normal_costs[4].clear()
    if change == 'counter':
        rx.acked_counter = 2
    elif change == 'generation':
        rx._train_generation += 1
    elif change == 'next-cycle':
        rx.new_cycle()
    elif change == 'word-clock':
        rx.clock.expects = Expect.WORD
    elif change == 'peer-evidence':
        rx._peer_heads.clear()
    elif change == 'gap':
        rx.note_gap(row['onset_sample']+100, row['onset_sample']+200)
    elif change == 'unadmitted':
        rx.admitted = False
    elif change == 'generation-during':
        # The saved generator already exists; revoke ownership in its CRC reader.
        unpack = p4shortlevels._frame
        def revoke(*args):
            rx._train_generation += 1
            return unpack(*args)
        monkeypatch.setattr(p4shortlevels, '_frame', revoke)
    records = rx.refine(deadline=.100)
    assert len(records) == (1 if change == 'none' else 0)
    assert candidate.done
    assert (candidate.frame is not None) == (change == 'none')
    if change == 'counter':
        assert rx.hold.acked_counter == 2


def test_recorded_repeats_deliver_once(recording):
    audio, rows, probe = recording
    rx = receiver(rows[1])
    turn, _ = machine(role='irs')
    delivered = []
    turn.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    for row in (rows[1], rows[7]):
        rx.clock = Clock(Expect.PACKET, row['onset_sample'], .330, 1.25, row['cfo_hz'])
        records = read_row(audio, row, rx)
        assert len(records) == 1
        for registered in records:
            at = registered.observation.stream_sample
            turn.observe(registered, now=at+round(.810*FS))
            proposal = turn.decide(at+round(.920*FS)-turn.lead)
            assert proposal and proposal.word == 'V27'
            turn.keyed(proposal.stream_sample)
        rx.new_cycle()
    assert len(delivered) == 1
    assert receiver(rows[1])._normal_level is None


def test_refused_qam_answer_keeps_key_history_and_deduplicates_source(recording):
    audio, rows, probe = recording
    turn, arq = machine(role='irs')
    arq.on_host_data(b'queued outbound bytes')
    before = arq._buffer_raw
    delivered = []
    turn.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    for index in (2, 8):
        row = rows[index-1]
        records = read_row(audio, row, receiver(row))
        assert len(records) == 1
        at = records[0].observation.stream_sample
        turn.observe(records[0], now=at+round(.810*FS))
        proposal = turn.decide(at+round(.920*FS)-turn.lead)
        assert proposal is not None and proposal.word == 'V27'
        previous = turn.ack_word, turn._last_ack
        turn.refused('deadline')
        assert (turn.ack_word, turn._last_ack) == previous
        assert arq._buffer_raw == before
    assert len(delivered) == 1


def direct_search(z, start, cfo):
    equalized = []
    scored = p4short.short_scores(z, first_training_start=start, frequency_offset=cfo,
        refine=0, on_equalized=lambda s, n: equalized.extend((s, n)))
    if scored is None:
        return None
    ranked = p4shortlevels.candidates(*equalized, scored[2], qam=True)
    if not any(w == 4 and p4shortlevels.agreement(f) >= .55 for w,f in ranked):
        return None
    return decoder(ranked).decode()


@pytest.mark.parametrize('control', ['timing', 'carrier', 'noise'])
def test_recorded_qam_wrong_placement_and_noise_rejected(recording, control):
    audio, rows, probe = recording
    row = rows[1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], FS)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM, row['cfo_hz'],
                                      offsets=range(-60, 61))
    if control == 'timing':
        start += 64
    elif control == 'carrier':
        cfo += 30
    else:
        z = np.random.default_rng(10032026).normal(size=len(z)).astype(complex)
    assert direct_search(z, start, cfo) is None


@pytest.mark.parametrize('name,seconds,center', [('kiwi', 1433., 2410.),
                                               ('monitor', 342.298, 5496.)])
def test_unrelated_active_recording_crops_rejected(name, seconds, center):
    path = ROOT/'captures/p4-qam16-negatives'/(name+'.wav')
    if not path.exists():
        pytest.skip('local unrelated recording absent')
    with wave.open(str(path)) as stream:
        fs, channels = stream.getframerate(), stream.getnchannels()
        assert stream.getsampwidth() == 2
        stream.setpos(round((seconds-.05)*fs))
        audio = np.frombuffer(stream.readframes(round(.95*fs)), dtype='<i2')/32768.
    if channels != 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = np.real(hilbert(audio)*np.exp(-2j*np.pi*(center-1500)*np.arange(len(audio))/fs))
    if fs != FS:
        import math
        gcd = math.gcd(fs, FS)
        audio = resample_poly(audio, FS//gcd, fs//gcd)
    z = p4rx.audio_to_baseband(audio, FS)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM, 0., offsets=range(-60,61))
    assert direct_search(z, start, cfo) is None


@pytest.mark.parametrize('arm,at,status', [('onair-1002-1013', 2148674, 0x61),
                                         ('onair-1002-1049', 2088660, 1)])
def test_recorded_lower_mode_unchanged(arm, at, status):
    capture = ROOT/'captures'/arm
    if not (capture/'stream.flac').exists():
        pytest.skip('local recording absent')
    probe = json.loads((capture/'p4-probe.json').read_text())
    row = next(r for r in probe['heard'] if r['kind'] == 'normal'
               and abs(r['onset_sample']-at) <= 48 and r.get('status') == status)
    rx = receiver(row)
    records = read_row(read(capture/'stream.flac'), row, rx, end=.846)
    assert len(records) == 1 and records[0].observation.frame.status.byte == status
    assert rx._normal_level == 1 and records[0].observation.speedlevel is None
