# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import pytest

import hfmodem
from hfmodem.shrike import (p4coding, p4decode, p4frame, p4long, p4longtx,
                            p4longrates, p4robustc2, p4robustrx, p4rx,
                            p4sf8rx, session, p4longdecode as engine)
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.test_p4_robust_c2_target_1004 import FIXTURES as ROBUST_PANEL

FX = Path(__file__).parent / 'fixtures'
ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def python_reference(monkeypatch):
    from hfmodem.shrike import p4longnative
    monkeypatch.setattr(p4longnative, 'backend', None)


def required(path):
    return require_input(path)


def encoded(sl, seed=1111):
    g = engine.geometry(sl)
    rng = np.random.default_rng(seed)
    data = rng.bytes(g.data_bytes)
    status = p4frame.Status.from_byte(int(rng.integers(256)))
    wire = p4frame.encode_field(data, status)
    prefix = np.r_[np.unpackbits(np.frombuffer(wire, np.uint8), bitorder='little'), np.zeros(g.padding, np.uint8)]
    tail = p4coding.rsc_terminate(p4coding.rsc_encode(prefix, initial_state=(0, 0, 0)).final_state)
    bits = np.r_[prefix, tail.systematic].astype(np.uint8)
    first = p4coding.rsc_encode(bits, initial_state=(0, 0, 0)).parity
    second = p4coding.rsc_encode(bits[g.order], initial_state=(0, 0, 0)).parity
    coded = np.r_[bits, np.asarray(first)[g.c1_slice], np.asarray(second)[g.c2_slice]]
    return g, bits, coded, p4frame.decode_field(wire, data_bytes=g.data_bytes)


def search(sl, fields, **kwargs):
    return engine.LongSearch([engine.LongMember(engine.geometry(sl), fields)], **kwargs)


@pytest.mark.parametrize('sl', [2, 3, 4])
@pytest.mark.parametrize('end_state', [0, None])
def test_batched_components_match_independent_exact_recurrences(sl, end_state):
    g = engine.geometry(sl)
    rng = np.random.default_rng(1113)
    systematic = rng.normal(0, 12, (6, g.systematic))
    parity = rng.normal(0, 12, systematic.shape)
    parity[:, ::3] = 0
    systematic[0] *= 1000
    posterior, extrinsic = engine._components(systematic, parity, end_state=end_state)
    for i in range(len(systematic)):
        expected = p4decode.component_decode(np.clip(systematic[i], -40, 40),
            np.clip(parity[i], -40, 40), initial_state=0, end_state=end_state)
        np.testing.assert_allclose(posterior[i], expected.posterior, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(extrinsic[i], expected.extrinsic, atol=2e-10, rtol=2e-10)
        decisive = abs(expected.posterior) > 1e-6
        np.testing.assert_array_equal(posterior[i, decisive] < 0, expected.posterior[decisive] < 0)


@pytest.mark.parametrize('sl,remaining', [(6, .012), (10, .060)])
def test_python_probability_component_reserve_defers_without_work(monkeypatch, sl, remaining):
    g = engine.geometry(sl)
    job = search(sl, np.random.default_rng(1113).normal(size=g.serialized_size), component_budget=0)
    assert job.advance().completed_tier == 0
    job.component_budget = None
    monkeypatch.setattr(engine.time, 'monotonic', lambda: 0.)

    def forbidden(*args, **kwargs):
        pytest.fail('component started without its reserve')

    monkeypatch.setattr(engine, '_components', forbidden)
    result = job.advance(remaining)
    assert result.state == 'incomplete' and result.completed_tier == 0
    assert result.component_calls == 0


def test_batched_search_retains_complete_tier_uniqueness_and_resume(monkeypatch):
    g, _, coded, frame = encoded(3)
    # Force C1 recovery; the valid member is followed by unrelated hypotheses.
    fields = 8*(1-2.0*coded)
    fields[7] *= -1
    members = [engine.LongMember(g, fields)]
    rng = np.random.default_rng(1114)
    members += [engine.LongMember(g, rng.normal(size=g.serialized_size)) for _ in range(15)]
    expected_job = engine.LongSearch(members, component_budget=16)
    expected = expected_job.advance()
    assert expected.frame == frame and expected.completed_tier == 1
    clock = [0.]
    batch = engine._batch_step
    def timed(*args):
        result = batch(*args)
        clock[0] += .065
        return result
    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(engine, '_batch_step', timed)
    job = engine.LongSearch(members)
    paused = job.advance(deadline=.065)
    assert paused.frame is None and paused.state == 'incomplete'
    assert paused.completed_tier == 0 and job.pending
    assert clock[0] == .065
    actual = job.advance(deadline=.066)
    assert actual.frame == expected.frame and actual.provenance == expected.provenance
    assert actual.component_calls == expected.component_calls
    assert not job.pending and clock[0] == .065
    np.testing.assert_array_equal(job.first, expected_job.first)


def test_batched_mixed_geometry_turbo_matches_scalar_search():
    rng = np.random.default_rng(1115)
    members = []
    for sl in (2, 3)*8:
        g = engine.geometry(sl)
        members.append(engine.LongMember(g, rng.normal(size=g.serialized_size)))
    scalar = engine.LongSearch(members, iterations=2, policy='exhaustive', component_budget=64)
    expected = scalar.advance()
    batched = engine.LongSearch(members, iterations=2, policy='exhaustive')
    actual = batched.advance()
    assert actual.provenance == expected.provenance
    assert actual.completed_tier == expected.completed_tier == 3
    assert actual.component_calls == expected.component_calls == 64
    np.testing.assert_array_equal(batched.priors, scalar.priors)
    np.testing.assert_array_equal(batched.first, scalar.first)


def test_batched_search_keeps_conflicting_identity_after_later_batch():
    g, _, coded, frame = encoded(3)
    _, _, other, other_frame = encoded(3, seed=1116)
    fields, alternate = 8*(1-2.0*coded), 8*(1-2.0*other)
    fields[7] *= -1
    alternate[7] *= -1
    rng = np.random.default_rng(1117)
    members = [engine.LongMember(g, fields)]
    members += [engine.LongMember(g, rng.normal(size=g.serialized_size))
                for _ in range(46)]
    members.append(engine.LongMember(g, alternate))
    job = engine.LongSearch(members)
    result = job.advance()
    assert result.state == 'ambiguous' and result.frame is None
    assert result.completed_tier == 1
    assert result.identities == (frame, other_frame)
    assert result.component_calls == 48 and not job.pending


@pytest.mark.parametrize('sl', range(2, 11))
def test_all_forms_round_trip_and_geometry(sl):
    assert Path(hfmodem.__file__).resolve().parents[1] == Path(__file__).resolve().parents[3]
    g, bits, coded, frame = encoded(sl)
    assert g.systematic == 8 * (g.data_bytes + 3) + g.padding + 3
    assert g.order is g.order
    with pytest.raises(ValueError):
        g.order.setflags(write=True)
    fields = 12 * (1 - 2.0 * coded) + np.random.default_rng(1112).normal(0, .2, len(coded))
    result = search(sl, fields).advance()
    assert result.state == 'unique' and result.frame == frame
    assert result.completed_tier == 0 and result.component_calls == 0
    assert all(r.agreement_fraction == 1 for r in result.selected_provenance.agreements)
    if sl <= 4:
        assert g.order.astype('<u2').tobytes() == required(
            ROBUST_PANEL / f'c2-sl{sl}.u16le').read_bytes()
    if sl == 5:
        assert np.array_equal(coded, p4longtx.coded_bits(p4frame.encode_field(frame.data, frame.status)))
        assert np.array_equal(g.order, p4long.component_permutation())
    if sl >= 6:
        old = p4longrates.geometry(sl)
        assert (g.systematic, g.data_bytes, g.padding) == (old.systematic, old.data_bytes, old.padding)
        assert np.array_equal(g.order, p4longrates.component_permutation(sl))
        assert p4longrates._check_frame(bits, old) == (frame, True, True)


def test_policy_ambiguity_zero_and_immutable_inputs(monkeypatch):
    g, _, coded, frame = encoded(2)
    fields = 8 * (1 - 2.0 * coded)
    member = engine.LongMember(g, fields)
    fields[:] = 0
    with pytest.raises(ValueError):
        member.fields.setflags(write=True)
    live = engine.LongSearch([member]).advance()
    full = engine.LongSearch([member], policy='exhaustive').advance()
    assert live.frame == full.frame == frame
    assert live.completed_tier == 0 and full.completed_tier == 9
    assert full.component_calls == 16
    other = encoded(2, 1112)[2]
    ambiguous = engine.LongSearch([member, engine.LongMember(g, 8 * (1 - 2.0 * other))]).advance()
    assert ambiguous.state == 'ambiguous' and ambiguous.frame is None
    assert len(ambiguous.identities) == 2 and ambiguous.completed_tier == 0
    def forbidden(*args, **kwargs):
        raise AssertionError('component called on zero input')
    monkeypatch.setattr(p4decode, 'component_decode', forbidden)
    zero = search(2, np.zeros(g.serialized_size)).advance()
    assert zero.state == 'unread' and zero.component_calls == 0


def test_resume_at_every_tier(monkeypatch):
    g, _, coded, _ = encoded(2)
    fields = 2 * (1 - 2.0 * coded) + np.random.default_rng(22).normal(0, 2, len(coded))
    uninterrupted = search(2, fields, policy='exhaustive')
    expected = uninterrupted.advance()
    resumed = search(2, fields, policy='exhaustive')
    assert resumed.advance(deadline=time.monotonic() - 1).state == 'incomplete'
    original = engine._check
    clock = [0.0]
    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])
    def step(*args):
        result = original(*args)
        clock[0] += 1
        return result
    # Pause at a completed tier boundary; recurrence and validation now have
    # separate admission checkpoints, covered by the pending-validation pin.
    monkeypatch.setattr(engine, '_check', step)
    for tier in range(10):
        result = resumed.advance(deadline=clock[0] + .5)
        assert result.completed_tier == tier
        assert result.state == ('incomplete' if tier < 9 else expected.state)
        if tier < 9:
            assert result.frame is None
    assert result.identities == expected.identities
    assert result.provenance == expected.provenance
    assert result.component_calls == expected.component_calls
    assert np.array_equal(resumed.priors, uninterrupted.priors)


@pytest.mark.parametrize('sl', [2, 4])
def test_independent_robust_panel(sl):
    base = ROBUST_PANEL
    panel = json.loads(required(base / 'FIXED-PANEL.json').read_bytes())
    rows = [r for r in panel['cases'] if r['speedlevel'] == sl and r['es_n0_db'] == 24]
    for row in rows:
        path = required(base / row['path'])
        assert sha256(path.read_bytes()).hexdigest() == row['file_sha256']
        with np.load(path, allow_pickle=False) as data:
            raw, wire = data['clean_symbols'], data['wire'].tobytes()
        g = p4robustc2._geometry(sl)
        members = [engine.LongMember(engine.geometry(sl), m.fields, m.quarter_turn, m.weighting)
                   for m in p4robustc2._eligible_members(raw, g)]
        result = engine.LongSearch(members).advance()
        assert result.state == 'unique' and result.completed_tier == 0
        actual = result.frame.data + bytes([result.frame.status.byte]) + result.frame.crc.to_bytes(2, 'little')
        assert sha256(actual).digest() == sha256(wire).digest()


@pytest.mark.parametrize('name', ['n9vc_0', 'n9vc_1', 've3kpg_0', 've3kpg_1'])
def test_recorded_sf16(name):
    with np.load(required(FX / 'p4-long-0929/symbols.npz'), allow_pickle=False) as fixture:
        symbols = fixture[name]
    old = p4robustrx.decode_symbols(symbols)
    assert old is not None
    delta = symbols[1:] * symbols[:-1].conjugate()
    unit = delta / np.maximum(abs(delta), 1e-30)
    unit *= np.exp(-1j * np.angle(np.mean(unit**4)) / 4)
    data = unit[old.field_start:old.field_start + 348][p4robustrx._SYMBOL_ORDER]
    members = []
    for turn in range(4):
        z = data * (1j ** turn)
        fields = 4 * np.stack((z.real + z.imag, z.real - z.imag), axis=-1).ravel()
        members.append(engine.LongMember(engine.geometry(old.speedlevel), fields, turn))
    result = engine.LongSearch(members).advance()
    assert result.completed_tier == (3 if name == 've3kpg_1' else 0)
    assert result.frame == old.frame
    assert sha256(result.frame.data).digest() == sha256(old.frame.data).digest()
    exhaustive = engine.LongSearch(members, policy='exhaustive').advance()
    assert exhaustive.state == 'unique' and exhaustive.frame == result.frame


def test_recorded_sf8_copies():
    stream = required(CAPTURES / 'onair-1002-2154/stream.flac')
    audio = session.load_wav(str(stream), 48000)
    for n in range(12):
        onset = 3528586 + 180000 * n
        chunk = audio[onset - 4800:onset - 4800 + round(3.48 * 48000)]
        baseband = p4rx.audio_to_baseband(chunk, 48000)
        observed = p4rx.recover_robust(baseband, reference_start=8000, spread_factor=8,
            data_symbols=696, timing_offsets=range(-48, 49), cfo_hz=(.69,))
        old = p4sf8rx.decode_symbols(observed.symbols)
        members = [engine.LongMember(engine.geometry(4), m.fields, m.quarter_turn, m.weighting)
                   for m in p4robustc2._eligible_members(observed.symbols, p4robustc2._geometry(4))]
        result = engine.LongSearch(members).advance()
        assert result.completed_tier == (2 if n == 1 else 1 if n in (2, 3) else 0)
        # The SF8 adapter and engine recover the same weak-copy identity.
        assert old is not None
        assert result.frame is not None
        assert sha256(result.frame.data).hexdigest() == 'e8acf31f32a128b6d89a3bb2780cf3fa513014358934a4ba3f8966a922704a92'
        assert result.frame.status.byte == 3
        if old is not None:
            assert result.frame == old.frame
        exhaustive = engine.LongSearch(members, policy='exhaustive').advance()
        assert exhaustive.state == 'unique' and exhaustive.frame == result.frame


@pytest.mark.parametrize('sl,sigma', [(2, .7), (3, .5), (4, .45), (5, .7),
                                       (6, .4), (7, .4), (8, .4), (9, .4), (10, .4)])
def test_turbo_requires_correct_c2(sl, sigma, monkeypatch):
    g, _, coded, frame = encoded(sl)
    fields = 2 / sigma**2 * ((1 - 2.0 * coded) + np.random.default_rng(1113).normal(0, sigma, len(coded)))
    result = search(sl, fields).advance()
    assert result.state == 'unique' and result.frame == frame
    assert result.completed_tier == 2 and result.selected_provenance.iteration == 1
    assert not any(p.identity_index is not None for p in result.provenance[:2])
    erased = fields.copy()
    erased[g.systematic + g.parity:] = 0
    assert search(sl, erased).advance().state == 'unread'
    wrong = np.argsort(g.order)
    monkeypatch.setattr(engine, '_order', lambda level: wrong)
    assert search(sl, fields).advance().state == 'unread'


@pytest.mark.parametrize('sl', range(2, 11))
@pytest.mark.parametrize('region', ['padding', 'flush', 'systematic', 'retained-c1', 'retained-c2'])
def test_crc_alone_does_not_accept_bad_tails_or_parity(sl, region):
    g, bits, coded, _ = encoded(sl)
    fields = 8 * (1 - 2.0 * coded)
    if region in ('padding', 'flush'):
        bits[-4 if region == 'padding' else -1] ^= 1
    else:
        _, lo, hi = next(r for r in g.regions if r[0] == region)
        fields[lo:hi] *= -1
    found = {}
    outcome = engine._check(bits, engine.LongMember(g, fields), 0, 0, found)
    assert outcome.crc_valid and outcome.identity_index is None and not found
    if region == 'padding':
        assert not outcome.padding_valid
    if region == 'flush':
        assert not outcome.flush_valid
    if region.startswith('retained'):
        assert next(r for r in outcome.agreements if r.region == region).weighted_support == -1


def test_exhaustive_preserves_later_conflicting_identity(monkeypatch):
    g, a, code_a, frame_a = encoded(2)
    _, b, code_b, _ = encoded(2, 1112)
    fields = 8 * (1 - code_a.astype(float) - code_b)
    member = engine.LongMember(g, fields)
    def step(g, fields, prior, first, tier):
        return a if tier == 0 else b, prior, first, 0
    monkeypatch.setattr(engine, '_member_step', step)
    live = engine.LongSearch([member]).advance()
    full = engine.LongSearch([member], policy='exhaustive').advance()
    assert live.state == 'unique' and live.frame == frame_a and live.completed_tier == 0
    assert full.state == 'ambiguous' and full.frame is None and len(full.identities) == 2
    assert full.completed_tier == 9


def test_component_boundaries_extrinsics_and_observed_suffix(monkeypatch):
    g = engine.geometry(4)
    fields = np.linspace(-2, 3, g.serialized_size)
    calls = []
    def component(systematic, parity, *, end_state):
        calls.append((np.clip(systematic[0], -40, 40), parity[0].copy(), end_state))
        return np.ones_like(systematic), (np.arange(g.systematic, dtype=float) + len(calls))[None]
    monkeypatch.setattr(engine, '_components', component)
    prior, first = np.zeros(g.systematic), np.zeros(g.systematic)
    for tier in (1, 2, 3):
        _, prior, first, _ = engine._member_step(g, fields, prior, first, tier)
    assert [c[2] for c in calls] == [0, None, 0, None]
    c1, c2 = np.zeros(g.systematic), np.zeros(g.systematic)
    c1[::10], c2[4::10] = fields[1160:1276], fields[1276:]
    assert np.array_equal(calls[0][1], c1) and np.array_equal(calls[1][1], c2)
    assert c1[1150] != 0 and c2[1154] != 0
    assert np.array_equal(calls[1][0], np.clip((fields[:1160] + np.arange(1160) + 1)[g.order], -40, 40))
    exchanged = np.empty(1160)
    exchanged[g.order] = np.arange(1160) + 2
    assert np.array_equal(calls[2][0], np.clip(fields[:1160] + exchanged, -40, 40))
    assert np.array_equal(calls[3][0], np.clip((fields[:1160] + np.arange(1160) + 3)[g.order], -40, 40))


@pytest.mark.parametrize('value', [True, 1, 11, 2.0, '2'])
def test_invalid_geometry(value):
    with pytest.raises(ValueError):
        engine.geometry(value)


@pytest.mark.parametrize('kind', ['short', 'rank', 'complex', 'nan', 'inf'])
def test_invalid_members(kind):
    g = engine.geometry(2)
    fields = np.ones(g.serialized_size)
    if kind == 'short': fields = fields[:-1]
    if kind == 'rank': fields = fields[None, :]
    if kind == 'complex': fields = fields.astype(complex)
    if kind == 'nan': fields[0] = np.nan
    if kind == 'inf': fields[0] = np.inf
    with pytest.raises(ValueError):
        engine.LongMember(g, fields)


@pytest.mark.parametrize('sl', range(6, 11))
def test_independent_normal_codewords(long_encoder, sl):
    g = engine.geometry(sl)
    data = np.random.default_rng(1114 + sl).bytes(g.data_bytes)
    fixture = long_encoder.generate(sl, data, 0xa3)
    fields = 8 * (1 - 2.0 * fixture.fields)
    fields[0] *= -.05
    result = search(sl, fields).advance()
    assert result.state == 'unique' and result.completed_tier == 1
    assert result.frame.data == data and result.frame.status.byte == 0xa3
    assert np.array_equal(g.order, long_encoder.constants(sl).c2_order)


@pytest.mark.parametrize('index,crc', [(0, 0x5a2c), (1, 0x1bf8)])
def test_recorded_sl5(index, crc):
    path = required(FX / 'sigidwiki/p4-long/audio.npz')
    assert sha256(path.read_bytes()).hexdigest() == 'c641415075f315e839691230e186136601c06c74b64d95539cabc2a243858fd7'
    with np.load(path, allow_pickle=False) as source:
        baseband = p4rx.audio_to_baseband(source['audio'][index].astype(float) / 32768, 28800)
    fields = p4long.body_scores(baseband, first_training_start=2880)
    old = p4long.decode_scores(fields)
    members = [engine.LongMember(engine.geometry(5), polarity * fields, polarity=polarity)
               for polarity in (1, -1)]
    result = engine.LongSearch(members).advance()
    assert result.completed_tier == 2
    assert result.state == 'unique' and result.frame == old and result.frame.crc == crc
    exhaustive = engine.LongSearch(members, policy='exhaustive').advance()
    assert exhaustive.state == 'unique' and exhaustive.frame == result.frame


def test_deadline_never_publishes_half_a_tier(monkeypatch):
    g, _, a, _ = encoded(2)
    _, _, b, _ = encoded(2, 1112)
    members = [engine.LongMember(g, 8 * (1 - 2.0 * bits)) for bits in (a, b)]
    clock = [0.0]
    original = engine._member_step
    def step(*args):
        result = original(*args)
        clock[0] += 1
        return result
    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(engine, '_member_step', step)
    job = engine.LongSearch(members)
    result = job.advance(deadline=.5)
    assert result.state == 'incomplete' and result.frame is None
    assert result.completed_tier == -1 and len(result.provenance) == 1
    result = job.advance(deadline=clock[0]+.5)
    assert result.state == 'ambiguous' and result.frame is None
    assert len(result.provenance) == 2


@pytest.mark.parametrize('kwargs', [dict(quarter_turn=True), dict(quarter_turn=4),
    dict(polarity=0), dict(polarity=True), dict(weighting=[]), dict(label={})])
def test_invalid_provenance(kwargs):
    with pytest.raises(ValueError):
        engine.LongMember(engine.geometry(2), np.ones(696), **kwargs)


@pytest.mark.parametrize('deadline', [True, 'soon', 1j, np.nan, np.inf])
def test_invalid_deadline(deadline):
    with pytest.raises(ValueError):
        search(2, np.ones(696)).advance(deadline=deadline)
