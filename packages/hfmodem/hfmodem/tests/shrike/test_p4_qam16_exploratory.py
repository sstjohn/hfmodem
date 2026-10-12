import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hfmodem.bateleur.s4539 import QAM16
from hfmodem.core.wav import read
from hfmodem.shrike import p4coding, p4decode, p4frame, p4qam16, p4rx, p4short, p4shortlevels
from hfmodem.tests.evidence import CAPTURES


_CAPTURE = CAPTURES / 'onair-1002-2037'
_PASSED = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18,
           19, 20, 25, 26, 27, 28, 29, 30)
_DIGEST = 'ae9ea9706f28fbffffa0d9d3e75c4299d17c5c54a02bf3d15ffbd0fda5a51c81'


@pytest.fixture(scope='module')
def recording():
    if not (_CAPTURE / 'stream.flac').exists():
        pytest.skip('recording unavailable')
    sidecar = json.loads((_CAPTURE / 'stream.json').read_text())
    assert (sidecar['samplerate'], sidecar['samples']) == (48000, 4194944)
    heard = json.loads((_CAPTURE / 'p4-probe.json').read_text())['heard']
    normal = [row for row in heard if row['kind'] == 'normal']
    assert len(normal) == 36
    return read(_CAPTURE / 'stream.flac'), normal


def _fields(recording, index):
    audio, normal = recording
    row = normal[index - 1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    equalized = []
    p4short.short_scores(z, first_training_start=start, frequency_offset=cfo,
                         refine=0, on_equalized=lambda s, n: equalized.extend((s, n)))
    return p4qam16.demap(*equalized), equalized


def test_p4_qam16_table_provenance():
    order = p4qam16.component_permutation()
    assert np.array_equal(np.sort(order), np.arange(1408))
    assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == (
        'fa95eb73f365e41be7e8d82bc7d45d2b81d659af39f0247fd440d67f667b2e23')
    assert np.array_equal(p4qam16.QAM16, QAM16)
    assert len(set(QAM16)) == 16
    assert np.count_nonzero(abs(QAM16) < .5) == 4


def test_p4_qam16_optional_original_archive():
    path=Path(__file__).resolve().parents[5]/'working/pactor/spec/interleaver_tables.npz'
    if not path.is_file():
        pytest.skip('private original interleaver archive not present')
    with np.load(path,allow_pickle=False) as original:
        assert np.array_equal(p4qam16.component_permutation(),original['0x935b0_1408'])


def test_p4_qam16_exact_likelihood_and_block_noise():
    symbols = np.full(1056, QAM16[0], dtype=complex)
    noise = np.r_[np.full(176, .1), np.full(880, .8)]
    exact = p4qam16.demap(symbols, noise, exact=True)
    maxlog = p4qam16.demap(symbols, noise)
    assert exact.shape == maxlog.shape == (4224,)
    assert np.all(np.isfinite(exact))
    assert not np.allclose(exact, maxlog)
    assert np.mean(abs(p4qam16.demap(symbols, .1, exact=True))) > np.mean(
        abs(p4qam16.demap(symbols, .8, exact=True)))
    with pytest.raises(ValueError):
        p4qam16.demap(symbols, np.r_[noise[:-1], 0], exact=True)


def test_p4_qam16_demapper_removes_own_prior():
    symbols = np.full(1056, QAM16[3], dtype=complex)
    prior = np.zeros((1056, 4))
    prior[:, 0] = 1.7
    base = p4qam16.demap(symbols, .3, exact=True)
    updated = p4qam16.demap(symbols, .3, exact=True,
                             prior=prior[p4qam16._SYMBOL_ORDER].ravel())
    np.testing.assert_allclose(updated.reshape(1056, 4)[:, 0],
                               base.reshape(1056, 4)[:, 0], atol=1e-12)
    assert np.max(abs(updated-base)) > .01


def test_p4_qam16_rejects_crc_valid_psk8_body():
    rng = np.random.default_rng(10032027)
    for width in (1, 2, 3):
        data = rng.bytes(44*width-4)
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status(1)),
                                 data_bytes=len(data))
    fields = p4shortlevels.codeword(frame, 3)
    labels = fields.reshape(1056, 3)[np.argsort(p4qam16._SYMBOL_ORDER)]
    gray = np.arange(8) ^ (np.arange(8) >> 1)
    symbols = np.exp(2j*np.pi*np.argsort(gray)[labels @ np.array([4, 2, 1])]/8)
    assert p4shortlevels.ShortDecode(
        [(3, p4shortlevels.demap(symbols, 1e-4, 3))], costs={3:[]}, events=[],
        at=0, status_ok=lambda decoded, combined: True).decode() == frame
    stream = np.zeros(1280, complex)
    for block in range(7):
        stream[208*block:208*block+32] = p4short.TRAIN[block]
    for block in range(6):
        stream[208*block+32:208*block+208] = symbols[176*block:176*(block+1)]
    impulses = np.zeros(16*1279+1, complex)
    impulses[::16] = stream
    z = np.pad(np.convolve(impulses, p4rx.RRC_TAPS), (0, 200))
    equalized = []
    p4short.short_scores(z, first_training_start=0, refine=0,
                         on_equalized=lambda s, n: equalized.extend((s, n)))
    assert p4qam16._lower_order_body(equalized[0])
    assert p4qam16.decode_body(z, first_training_start=0) is None


def test_p4_qam16_crc_requires_close_codeword():
    wire = p4frame.encode_field(bytes(172), p4frame.Status(1))
    frame = p4frame.decode_field(wire, data_bytes=172)
    decoded = SimpleNamespace(bits=np.r_[np.unpackbits(np.frombuffer(
        wire, dtype=np.uint8), bitorder='little'), np.zeros(8, dtype=np.uint8)],
        diagnostics=[SimpleNamespace(iteration=1)])
    scores = 1. - 2.*p4qam16.codeword(frame)
    with patch.object(p4decode, 'turbo_iterations', return_value=[decoded]):
        near = scores.copy()
        near[:600] *= -1
        assert p4qam16.decode_fields(near) is not None
        far = scores.copy()
        far[:700] *= -1
        assert p4qam16.decode_fields(far) is None


def test_p4_qam16_requires_unique_body_candidate():
    def frame(value):
        wire = p4frame.encode_field(bytes([value])*172, p4frame.Status(1))
        return p4frame.decode_field(wire, data_bytes=172)

    first, second = frame(65), frame(66)
    assert first != second
    results = iter(((first, 1, .01), (second, 1, .01)))
    with patch.object(p4short, 'short_scores', return_value=(0., 0., None)), \
         patch.object(p4qam16, 'decode_equalized', side_effect=lambda *a, **k: next(results, None)), \
         patch.object(p4qam16, '_wide_equalized', return_value=None), \
         patch.object(p4qam16, '_body_em_candidate', return_value=None), \
         patch.object(p4qam16, '_wide_iterative_candidate', return_value=None), \
         patch.object(p4qam16, '_iterative_candidate', return_value=None):
        assert p4qam16.decode_body(None, first_training_start=0) is None


def test_p4_qam16_private_zero_state_component_matches_general_encoder():
    rng = np.random.default_rng(10032028)
    cases = [bits for size in range(11) for bits in itertools.product((0,1), repeat=size)]
    cases += [rng.integers(0,2,size=size) for size in (11,13,14,15,31,32,33,1400,1405,1408,16560)]
    for bits in cases:
        parity, state = p4qam16._zero_state_component(bits)
        expected = p4coding.rsc_encode(bits, initial_state=(0,0,0))
        assert np.array_equal(parity, expected.parity)
        assert state == expected.final_state
        tail = p4coding.rsc_terminate(state)
        closed, final = p4qam16._zero_state_component(list(bits)+list(tail.systematic))
        assert np.array_equal(closed, expected.parity+tail.parity)
        assert final == (0,0,0)


@pytest.mark.parametrize('size', [0,1,171,173,512])
def test_p4_qam16_codeword_rejects_wrong_field_length(size):
    with pytest.raises(ValueError):
        p4qam16.codeword(p4frame.Frame(bytes(size), p4frame.Status(1), 0))


def test_p4_qam16_codeword_rejects_invalid_field_types():
    for frame in (p4frame.Frame([0]*172, p4frame.Status(1), 0),
                  p4frame.Frame(bytes(172), 1, 0)):
        with pytest.raises(TypeError):
            p4qam16.codeword(frame)


def test_p4_qam16_codeword_matches_general_component_encoder():
    rng = np.random.default_rng(10032026)
    for status in range(256):
        frame = p4frame.decode_field(p4frame.encode_field(rng.bytes(172),
            p4frame.Status.from_byte(status)), data_bytes=172)
        systematic = np.unpackbits(np.frombuffer(p4frame.encode_field(frame.data, frame.status),
                                                 dtype=np.uint8), bitorder='little').tolist()+[0]*5
        state = p4coding.rsc_encode(systematic, initial_state=(0,0,0)).final_state
        systematic += list(p4coding.rsc_terminate(state).systematic)
        c1, c2 = p4coding.encode_components(systematic, permutation=p4qam16.component_permutation(),
                                           initial_states=((0,0,0),(0,0,0)))
        assert np.array_equal(p4qam16.codeword(frame), np.r_[systematic,c1.parity,c2.parity])


def test_p4_qam16_reference_cache_keeps_all_identities_and_rechecks_each_field():
    frames = [p4frame.decode_field(p4frame.encode_field(bytes([value])*172,
                  p4frame.Status(counter)), data_bytes=172)
              for value, counter in ((65, 1), (66, 1), (65, 2))]
    references = [p4qam16.codeword(frame) for frame in frames]
    common = (1. - 2.*references[0])*np.all(np.array(references) == references[0], axis=0)

    def iterations(*args, **kwargs):
        for iteration, frame in enumerate(frames + frames, 1):
            wire = p4frame.encode_field(frame.data, frame.status)
            bits = np.r_[np.unpackbits(np.frombuffer(wire, dtype=np.uint8),
                                       bitorder='little'), np.zeros(8, dtype=np.uint8)]
            yield SimpleNamespace(bits=bits, diagnostics=[SimpleNamespace(iteration=iteration)])

    admitted, checked = [], []
    with patch.object(p4decode, 'turbo_iterations', side_effect=iterations), \
         patch.object(p4qam16, 'codeword', wraps=p4qam16.codeword) as encode:
        assert p4qam16.decode_fields(common, collect=admitted.append,
            status_ok=lambda frame: checked.append(frame) is None) is None
        assert [result[0] for result in admitted] == frames
        assert [result[1] for result in admitted] == [4, 5, 6]
        assert checked == frames + frames
        assert encode.call_count == 3
        result = p4qam16.decode_fields(1. - 2.*references[0])
        assert result == (frames[0], 4, 0.)
        assert encode.call_count == 6


def test_p4_qam16_ambiguous_candidate_poisons_later_unique_candidate():
    def frame(value):
        wire = p4frame.encode_field(bytes([value])*172, p4frame.Status(1))
        return p4frame.decode_field(wire, data_bytes=172)

    first, second = frame(65), frame(66)
    code_a, code_b = p4qam16.codeword(first), p4qam16.codeword(second)
    common = (1. - 2.*code_a)*(code_a == code_b)
    fields = iter([common, 1. - 2.*code_a, common, common, common])
    calls = []

    def decoded(candidate):
        wire = p4frame.encode_field(candidate.data, candidate.status)
        bits = np.r_[np.unpackbits(np.frombuffer(wire, dtype=np.uint8),
                                   bitorder='little'), np.zeros(8, dtype=np.uint8)]
        return SimpleNamespace(bits=bits,
                               diagnostics=[SimpleNamespace(iteration=1)])

    def turbo(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return iter((decoded(first), decoded(second)))
        return iter((decoded(first),)) if len(calls) == 2 else iter(())

    def short_scores(*args, **kwargs):
        kwargs['on_equalized'](np.zeros(1056, complex), 1.)
        return 0., 0., None

    admitted = []
    with patch.object(p4decode, 'turbo_iterations', side_effect=turbo):
        assert p4qam16.decode_fields(common, collect=admitted.append) is None
    assert {result[0] for result in admitted} == {first, second}
    calls.clear()
    with patch.object(p4short, 'short_scores', side_effect=short_scores), \
         patch.object(p4qam16, '_lower_order_body', return_value=False), \
         patch.object(p4qam16, 'demap', side_effect=lambda *args, **kwargs: next(fields)), \
         patch.object(p4decode, 'turbo_iterations', side_effect=turbo), \
         patch.object(p4qam16, '_wide_equalized', return_value=None), \
         patch.object(p4qam16, '_body_em_candidate', return_value=None), \
         patch.object(p4qam16, '_wide_iterative_candidate', return_value=None), \
         patch.object(p4qam16, '_iterative_candidate', return_value=None):
        assert p4qam16.decode_body(None, first_training_start=0) is None
    assert len(calls) == 5


def test_p4_qam16_decision_tracking_handles_block_drift():
    rng = np.random.default_rng(42)
    stream = np.zeros(1280, complex)
    for block in range(7):
        stream[208*block:208*block+32] = p4short.TRAIN[block]
    truth = QAM16[rng.integers(0, 16, 1056)]
    for block in range(6):
        stream[208*block+32:208*block+208] = truth[176*block:176*(block+1)]
    padded = np.pad(stream, 2)
    taps = np.array([.07+.02j, .13-.06j, 1, .09+.05j, .04-.03j])
    received = np.stack([padded[i:i+1280] for i in range(5)], axis=1) @ taps
    time = np.arange(1280)
    received *= np.where(time % 208 >= 32, np.exp(.08j*(time//208+1)), 1)
    received += .03*(rng.normal(size=1280)+1j*rng.normal(size=1280))
    z = np.zeros(21000, complex)
    z[128+16*time] = received
    with patch.object(p4rx, 'matched_filter', side_effect=lambda values: values):
        pilot = p4qam16._block_equalized(z, 0, 0, 0)
        tracked = p4qam16._block_equalized(
            z, 0, 0, 0, (truth, np.ones(1056), np.zeros(1056)))
    pilot_error = np.mean(abs(pilot[0]-truth)**2)
    tracked_error = np.mean(abs(tracked[0]-truth)**2)
    assert tracked_error < pilot_error/3


@pytest.mark.parametrize('index,iterations', [(index, 8) for index in _PASSED] + [(10, 32)])
def test_p4_qam16_recorded_independent_crc(recording, index, iterations):
    fields, _ = _fields(recording, index)
    frame, _, distance = p4qam16.decode_fields(fields, iterations=iterations)
    assert frame.status.byte == frame.status.counter == 1
    assert hashlib.sha256(frame.data).hexdigest() == _DIGEST
    assert distance < .10


def test_p4_qam16_recorded_negatives(recording):
    fields, (symbols, noise) = _fields(recording, 2)
    assert p4qam16.decode_fields(fields) is not None
    assert p4qam16.decode_fields(-fields) is None
    swapped = fields.reshape(1056, 4)[:, ::-1].ravel()
    assert p4qam16.decode_fields(swapped) is None
    symbol_inverse = fields.reshape(1056, 4)[np.argsort(
        p4qam16._SYMBOL_ORDER)].ravel()
    assert p4qam16.decode_fields(symbol_inverse) is None
    axes = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    square = (axes[:, None] + 1j*axes[None, :]).ravel()
    distance = abs(symbols[:, None]-square)**2
    labels = p4qam16._LABELS
    wrong = np.stack([distance[:, labels[:, bit] == 1].min(axis=1)
                      - distance[:, labels[:, bit] == 0].min(axis=1)
                      for bit in range(4)], axis=1)
    assert p4qam16.decode_fields(wrong[p4qam16._SYMBOL_ORDER].ravel()/noise) is None
    systematic, c1, c2 = np.split(fields, 3)
    c1, c2 = c1.copy(), c2.copy()
    c1[-8:] = c2[-8:] = 0
    inverse = np.argsort(p4qam16.component_permutation())
    for decoded in p4decode.turbo_iterations(systematic, c1, c2, inverse):
        wire = np.packbits(decoded.bits[:1400], bitorder='little').tobytes()
        assert p4frame.decode_field(wire, data_bytes=172) is None
    frame, _, _ = p4qam16.decode_fields(fields)
    wire = bytearray(p4frame.encode_field(frame.data, frame.status))
    wire[0] ^= 1
    assert p4frame.decode_field(bytes(wire), data_bytes=172) is None


def test_p4_qam16_standalone_body(recording):
    audio, normal = recording
    row = normal[1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    frame, _, _ = p4qam16.decode_body(z, first_training_start=start,
                                      frequency_offset=cfo)
    assert frame.status.byte == 1
    assert hashlib.sha256(frame.data).hexdigest() == _DIGEST


@pytest.mark.parametrize('index', [11, 21, 31])
def test_p4_qam16_refined_independent_crc(recording, index):
    audio, normal = recording
    row = normal[index - 1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    frame, _, distance = p4qam16.decode_body(
        z, first_training_start=start, frequency_offset=cfo)
    assert frame.status.byte == frame.status.counter == 1
    assert hashlib.sha256(frame.data).hexdigest() == _DIGEST
    assert distance < .10


@pytest.mark.parametrize('index', [22, 23, 24, 32, 33, 34, 35, 36])
def test_p4_qam16_wide_isi_independent_crc(recording, index):
    audio, normal = recording
    row = normal[index - 1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    frame, _, distance = p4qam16.decode_body(
        z, first_training_start=start, frequency_offset=cfo)
    assert frame.status.byte == frame.status.counter == 1
    assert hashlib.sha256(frame.data).hexdigest() == _DIGEST
    assert distance < .10


@pytest.mark.parametrize('index,shift', [
    (22, 6), (23, 4), (24, 6), (32, 6), (33, 4), (34, 2), (35, 4), (36, 8),
])
def test_p4_qam16_genie_channel_only(recording, index, shift):
    fields, _ = _fields(recording, 2)
    reference = p4qam16.decode_fields(fields)[0]
    encoded = p4qam16.codeword(reference)
    labels = encoded.reshape(1056, 4)[np.argsort(p4qam16._SYMBOL_ORDER)]
    transmitted = QAM16[labels @ np.array([8, 4, 2, 1])]
    source = np.zeros(1280, complex)
    for block in range(7):
        source[208*block:208*block+32] = p4short.TRAIN[block]
    for block in range(6):
        source[208*block+32:208*block+208] = transmitted[176*block:176*(block+1)]
    audio, normal = recording
    row = normal[index-1]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    z = z/np.max(abs(z))*np.exp(-2j*np.pi*cfo*np.arange(len(z))/28800)
    matched = p4rx.matched_filter(z)
    observed = matched[start+shift+128+16*np.arange(1280)]
    design = np.pad(source, 15)[np.arange(1280)[:, None]+15+np.arange(-7, 8)]
    equalized = np.empty(1056, complex)
    variance = np.empty(1056)
    for block in range(6):
        fit = np.arange(208*block+15, 208*block+193)
        channel = np.linalg.lstsq(design[fit], observed[fit], rcond=None)[0]
        residual = observed[fit]-design[fit]@channel
        noise = np.mean(abs(residual)**2)
        convolution = np.column_stack([np.convolve(channel, row)
                                       for row in np.eye(31)])
        target = np.zeros(45)
        target[22] = 1
        inverse = np.linalg.solve(convolution.conj().T@convolution
                                  + noise*np.eye(31), convolution.conj().T@target)
        response = np.convolve(channel, inverse)
        gain = response[22]
        response[22] = 0
        body = 208*block+32+np.arange(176)
        windows = body[:, None]+np.arange(-15, 16)
        symbols = (observed[windows]@inverse)/gain
        part = slice(176*block, 176*(block+1))
        equalized[part] = symbols
        variance[part] = (noise*np.vdot(inverse, inverse).real
                          + np.vdot(response, response).real)/abs(gain)**2
    scores = p4qam16.demap(equalized, variance, exact=True)
    signed = (1-2*encoded)*scores
    information = 4-4*np.mean(np.logaddexp(0, -signed))/np.log(2)
    assert information > 1400/1056
    decoded = p4qam16.decode_fields(scores, iterations=32)
    assert decoded is not None
    assert hashlib.sha256(decoded[0].data).hexdigest() == _DIGEST


def test_p4_qam16_repeated_analog_consensus(recording):
    audio, normal = recording
    repeats = []
    target = None
    for index, row in enumerate(normal, 1):
        at = row['onset_sample']
        z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
        _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                           row['cfo_hz'], offsets=range(-60, 61))
        if index == 24:
            target = z, start, cfo
        else:
            repeats.append((z, start, cfo))
    assert target is not None
    independent = p4qam16.decode_body(target[0], first_training_start=target[1],
                                       frequency_offset=target[2])
    assert independent is not None
    assert hashlib.sha256(independent[0].data).hexdigest() == _DIGEST
    result = p4qam16.decode_repeated_body(
        target[0], first_training_start=target[1], frequency_offset=target[2],
        repeats=repeats)
    assert result is not None
    assert hashlib.sha256(result[0].data).hexdigest() == _DIGEST


def test_p4_qam16_body_refit_rejects_wrong_acquisition(recording):
    audio, normal = recording
    row = normal[23]
    at = row['onset_sample']
    z = p4rx.audio_to_baseband(audio[at-2400:at+43200], 48000)
    _, start, cfo = p4short.place_short(z, 1440+p4short.GEOM,
                                       row['cfo_hz'], offsets=range(-60, 61))
    wrong_carrier = p4qam16._body_em_candidate(z, start+6, cfo+15, 32, None)
    wrong_origin = p4qam16._body_em_candidate(z, start+406, cfo, 32, None)
    assert not bool(wrong_carrier)
    assert not bool(wrong_origin)
