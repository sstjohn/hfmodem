# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Build first: cargo build --offline --release --manifest-path packages/hfmodem/native/p4long/Cargo.toml."""
import ctypes as ct
from dataclasses import replace
from pathlib import Path
import runpy
import subprocess
import time

import hfmodem
import numpy as np
import pytest

from hfmodem.shrike import p4longdecode as engine, p4longnative as native
from hfmodem.tests.shrike import test_p4_long_decode_1111 as reference


@pytest.fixture
def backend():
    assert Path(hfmodem.__file__).resolve().parents[1] == Path(__file__).resolve().parents[3]
    path = native._library_path()
    if not path.is_file():
        pytest.skip('P4 LONG native library is not built; build it with the command in the module docstring')
    # A present but broken library must fail, never silently skip.
    result = native._load(path)
    assert result is not None, 'Built P4 LONG library failed ABI validation or self-test'
    return result


def comparable(result):
    return replace(result, stage_seconds=(), elapsed_seconds=0.)


def run_python(job):
    engine._python_advance(job, None)
    return job.advance() if job.done else None


@pytest.mark.parametrize('offset', [-1e-10, 0., 1e-10])
def test_c1_negative_tie_boundary(backend, offset):
    g = engine.geometry(2)
    rng = np.random.default_rng(1113)
    systematic = rng.normal(0, 3, g.systematic)
    c1 = rng.normal(0, 3, g.parity)
    systematic[236] = -1.7665516003528008 + offset
    fields = np.r_[systematic, c1, np.zeros(g.parity)]
    bits, _, first, _ = engine._member_step(g, fields, np.zeros(g.systematic),
                                           np.zeros(g.systematic), 1)
    job = engine.LongSearch([engine.LongMember(g, fields)], component_budget=1)
    backend(job, None)
    np.testing.assert_array_equal(job._native.arrays['bits'][1, 0, :g.systematic], bits)
    np.testing.assert_array_equal(job.first[0], first)
    batch = engine.LongSearch([engine.LongMember(g, fields)] * 2)
    batch_bits, _, batch_first = engine._batch_step(batch, [0, 1], 1)
    np.testing.assert_array_equal(batch_bits, np.tile(bits, (2, 1)))
    np.testing.assert_array_equal(batch_first, np.tile(first, (2, 1)))


@pytest.mark.parametrize('sl', range(2, 11))
@pytest.mark.parametrize('mode', ['gaussian', 'punctured', 'erased', 'strong', 'near-tie'])
def test_seeded_every_tier(backend, monkeypatch, sl, mode):
    g = engine.geometry(sl)
    rng = np.random.default_rng(1111 + sl)
    fields = rng.normal(0, 3, g.serialized_size)
    if mode == 'punctured':
        fields[g.systematic::2] = 0
    if mode == 'erased':
        fields[:] = 0
    if mode == 'strong':
        fields = np.where(fields < 0, -1000., 1000.)
    if mode == 'near-tie':
        fields *= 1e-7
    members = [engine.LongMember(g, fields)]
    if mode == 'erased':
        members.append(engine.LongMember(g, rng.normal(size=g.serialized_size)))
    expected = engine.LongSearch(members, policy='exhaustive')
    captured = {}
    check = engine._check
    def capture(bits, member, index, tier, found):
        captured[tier, index] = np.asarray(bits).copy()
        return check(bits, member, index, tier, found)
    with monkeypatch.context() as patch:
        patch.setattr(engine, '_check', capture)
        expected_result = run_python(expected)
    actual = engine.LongSearch(members, policy='exhaustive')
    monkeypatch.setattr(native, 'backend', backend)
    result = actual.advance()
    assert comparable(result) == comparable(expected_result)
    assert actual._native.crossings == 1
    np.testing.assert_allclose(actual.priors, expected.priors, atol=2e-10, rtol=2e-10)
    # Check live stopping independently of the per-tier hard-decision comparison.
    live_python = engine.LongSearch(members)
    expected_live = run_python(live_python)
    live = engine.LongSearch(members)
    assert comparable(live.advance()) == comparable(expected_live)
    for (tier, i), bits in captured.items():
        np.testing.assert_array_equal(actual._native.arrays['bits'][tier, i, :g.systematic],
                                      bits, err_msg=f'SL{sl} {mode} tier {tier}')


@pytest.mark.parametrize('sl', range(2, 11))
def test_resume_every_tier(backend, monkeypatch, sl):
    g, _, coded, _ = reference.encoded(sl)
    fields = 2 * (1 - 2. * coded) + np.random.default_rng(sl).normal(0, 2, len(coded))
    members = [engine.LongMember(g, fields), engine.LongMember(g, -fields)]
    monkeypatch.setattr(native, 'backend', backend)
    uninterrupted = engine.LongSearch(members, policy='exhaustive')
    expected = uninterrupted.advance()
    resumed = engine.LongSearch(members, policy='exhaustive', component_budget=0)
    assert resumed.advance(time.monotonic() - 1).completed_tier == -1
    for tier, per_member_calls in enumerate([0, 1, 2, 4, 6, 8, 10, 12, 14, 16]):
        resumed.component_budget = len(members) * per_member_calls
        result = resumed.advance()
        assert result.completed_tier == tier
        assert result.state == ('incomplete' if tier < 9 else expected.state)
    assert comparable(result) == comparable(expected)
    assert resumed._native.crossings == 11
    for key in ('priors', 'first', 'bits', 'flags'):
        np.testing.assert_array_equal(resumed._native.arrays[key], uninterrupted._native.arrays[key])


@pytest.mark.parametrize('sl,sigma', [(2, .7), (3, .5), (4, .45), (5, .7),
                                       (6, .4), (7, .4), (8, .4), (9, .4), (10, .4)])
def test_valid_frames_stop_tier_and_identity(backend, monkeypatch, sl, sigma):
    g, _, coded, frame = reference.encoded(sl)
    fields = 2 / sigma**2 * ((1 - 2. * coded) + np.random.default_rng(1113).normal(0, sigma, len(coded)))
    members = [engine.LongMember(g, fields), engine.LongMember(g, fields, label='duplicate')]
    expected = run_python(engine.LongSearch(members))
    monkeypatch.setattr(native, 'backend', backend)
    job = engine.LongSearch(members)
    result = job.advance()
    assert comparable(result) == comparable(expected)
    assert result.frame == frame and result.completed_tier == 2 and len(result.identities) == 1
    assert job._native.crossings == 1


@pytest.mark.parametrize('case', ['n9vc_0', 'n9vc_1', 've3kpg_0', 've3kpg_1', 'sf8', 'sl5_0', 'sl5_1'])
def test_recordings_both_backends(backend, monkeypatch, case):
    def checked(job, deadline):
        assert deadline is None
        shadow = engine.LongSearch(job.members, policy=job.policy, iterations=job.last_tier - 1)
        expected = run_python(shadow)
        backend(job, deadline)
        assert job.tier == shadow.tier
        assert job.found == shadow.found
        assert tuple(job.outcomes) == expected.provenance
        assert job.calls == shadow.calls and job.positions == shadow.positions
    monkeypatch.setattr(native, 'backend', checked)
    if case == 'sf8':
        reference.test_recorded_sf8_copies()
    elif case.startswith('sl5'):
        index = int(case[-1])
        reference.test_recorded_sl5(index, [0x5a2c, 0x1bf8][index])
    else:
        reference.test_recorded_sf16(case)


def test_mixed_geometry_ambiguity(backend, monkeypatch):
    members = [engine.LongMember(reference.encoded(sl)[0], 10 * (1 - 2. * reference.encoded(sl)[2]))
               for sl in range(2, 11)]
    expected = run_python(engine.LongSearch(members))
    monkeypatch.setattr(native, 'backend', backend)
    job = engine.LongSearch(members)
    assert comparable(job.advance()) == comparable(expected)
    assert len(job.found) == 9 and job._native.crossings == 1


@pytest.mark.parametrize('delta', [-1e-12, 0., 1e-12])
@pytest.mark.parametrize('paired', [False, True])
def test_support_cancellation_has_identical_acceptance(backend, monkeypatch, delta, paired):
    g, _, coded, _ = reference.encoded(2)
    contributions = np.array([1.]*50 + [-1.]*50 + [2.**-52]*37 + [-2.**-52]*37)
    rng = np.random.default_rng(5678)
    rng.permutation(contributions)
    contributions = rng.permutation(contributions)
    if paired:
        contributions = np.r_[np.tile([1., -1.], 50), np.tile([2.**-52, -2.**-52], 37)]
    contributions[0] += delta
    fields = 8*(1-2.0*coded)
    fields[g.systematic:g.systematic+g.parity] = contributions * (1-2.0*coded[g.systematic:g.systematic+g.parity])
    members = [engine.LongMember(g, fields)]
    expected = run_python(engine.LongSearch(members))
    monkeypatch.setattr(native, 'backend', backend)
    actual = engine.LongSearch(members).advance()
    accepted = delta > 0 or delta == 0 and not paired
    assert actual.state == expected.state == ('unique' if accepted else 'unread')
    assert comparable(actual) == comparable(expected)


def test_subnormal_support_has_identical_acceptance(backend, monkeypatch):
    g, _, coded, _ = reference.encoded(2)
    fields = 8*(1-2.0*coded)
    contributions = np.zeros(g.parity)
    contributions[:3] = 1., -1., np.nextafter(0., 1.)
    fields[g.systematic:g.systematic+g.parity] = contributions * (1-2.0*coded[g.systematic:g.systematic+g.parity])
    members = [engine.LongMember(g, fields)]
    expected = run_python(engine.LongSearch(members))
    monkeypatch.setattr(native, 'backend', backend)
    actual = engine.LongSearch(members).advance()
    assert actual.state == expected.state == 'unread'
    assert comparable(actual) == comparable(expected)


@pytest.mark.parametrize('budget', [1, 7, 9, 18])
def test_native_partial_member_grant_resumes_identically(backend, monkeypatch, budget):
    g = engine.geometry(5)
    members = [engine.LongMember(g, np.random.default_rng(i).normal(size=g.serialized_size))
               for i in range(8)]
    expected_job = engine.LongSearch(members, policy='exhaustive')
    expected = run_python(expected_job)
    monkeypatch.setattr(native, 'backend', backend)
    job = engine.LongSearch(members, policy='exhaustive', component_budget=budget)
    partial = job.advance()
    assert partial.component_calls == budget
    assert partial.completed_tier == (0 if budget < 8 else 1 if budget < 16 else 2)
    assert partial.state == 'incomplete'
    before = job.priors.copy(), job.first.copy()
    job.advance(time.monotonic()-1)
    np.testing.assert_array_equal(job.priors, before[0])
    np.testing.assert_array_equal(job.first, before[1])
    job.component_budget = None
    assert comparable(job.advance()) == comparable(expected)
    np.testing.assert_allclose(job.priors, expected_job.priors, atol=2e-10, rtol=2e-10)


def test_native_partial_crc_waits_for_remaining_identities(backend, monkeypatch):
    members = []
    for seed in (1111, 1112):
        g, _, coded, _ = reference.encoded(2, seed)
        fields = 8*(1-2.0*coded)
        fields[:g.systematic] *= .25
        fields[0] *= -1
        members.append(engine.LongMember(g, fields))
    expected = run_python(engine.LongSearch(members))
    assert expected.state == 'ambiguous' and expected.completed_tier == 1
    monkeypatch.setattr(native, 'backend', backend)
    job = engine.LongSearch(members, component_budget=1)
    partial = job.advance()
    assert partial.state == 'incomplete' and partial.frame is None
    assert len(partial.identities) == 1 and partial.component_calls == 1
    job.component_budget = 2
    assert comparable(job.advance()) == comparable(expected)


def test_runtime_backend_disagreement_immediately_runs_fresh_python_retry(monkeypatch):
    g, _, coded, frame = reference.encoded(2)
    job = engine.LongSearch([engine.LongMember(g, 8*(1-2.0*coded))])
    def broken(job, deadline):
        job.priors[:] = 99
        job.found['invalid'] = frame
        job.done = True
        raise RuntimeError('P4 LONG native stop failed Python acceptance')
    monkeypatch.setattr(native, 'backend', broken)
    result = job.advance()
    assert result.state == 'unique' and result.frame == frame
    assert result.identities == (frame,) and job.backend_failed
    assert not np.any(job.priors)


@pytest.mark.parametrize('failure', ['absent', 'abi', 'self-test'])
def test_fallback_still_decodes(monkeypatch, tmp_path, failure):
    path = tmp_path / 'library'
    if failure == 'absent':
        assert native._load(path) is None
    else:
        path.touch()
        class Entry:
            def __call__(self, *args):
                return 77 if failure == 'abi' else native._ABI
        class Library:
            p4long_advance_v2 = Entry()
        monkeypatch.setattr(native.ct, 'CDLL', lambda path: Library())
        if failure == 'self-test':
            def fail(self):
                raise ValueError('self-test failed')
            monkeypatch.setattr(native.NativeBackend, 'self_test', fail)
        assert native._load(path) is None
    monkeypatch.setattr(native, 'backend', None)
    g, _, coded, frame = reference.encoded(2)
    assert engine.LongSearch([engine.LongMember(g, 8 * (1 - 2. * coded))]).advance().frame == frame


@pytest.mark.parametrize('damage', ['length', 'alias', 'geometry', 'order', 'nan', 'tier', 'abi', 'size', 'alignment'])
def test_abi_rejects_bad_buffers_before_mutation(backend, damage):
    g = engine.geometry(2)
    job = engine.LongSearch([engine.LongMember(g, np.ones(g.serialized_size))])
    owned = native._State(job)
    request = owned.request
    request.last_tier, request.budget = 9, 100
    abi, size = native._ABI, ct.sizeof(request)
    if damage == 'length': request.bits.length -= 1
    if damage == 'alias': request.first = request.priors
    if damage == 'geometry': owned.arrays['geometries'][0] = 11
    if damage == 'order':
        orders = owned.arrays['orders'].copy()
        orders[0, 0] = g.systematic
        request.orders = native._Buffer(orders.ctypes.data, orders.size)
    if damage == 'nan':
        llrs = job.llrs.copy()
        llrs[0, 0] = np.nan
        request.llrs = native._Buffer(llrs.ctypes.data, llrs.size)
    if damage == 'tier': owned.arrays['state'][0] = 11
    if damage == 'abi': abi += 1
    if damage == 'size': size -= 1
    if damage == 'alignment': request.priors.data += 1
    state = owned.arrays['state'].copy()
    assert backend.entry(abi, ct.byref(request), size) < 0
    np.testing.assert_array_equal(owned.arrays['state'], state)
    assert not np.any(job.priors) and not np.any(job.first)


@pytest.mark.parametrize('available', ['missing', 'failed', 'built'])
def test_optional_packaging(monkeypatch, tmp_path, available):
    setuptools = pytest.importorskip('setuptools', reason='setup.py needs setuptools in the test interpreter')
    captured = {}
    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: captured.update(kwargs))
    setup = Path(hfmodem.__file__).resolve().parents[1] / 'setup.py'
    definitions = runpy.run_path(str(setup))
    command_type = definitions['NativeBuild']
    monkeypatch.setitem(command_type.build_extension.__globals__, 'HERE', tmp_path)
    command = command_type(setuptools.Distribution({'ext_modules': captured['ext_modules']}))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / 'output')
    extension = captured['ext_modules'][0]
    assert extension.optional
    monkeypatch.delenv('CARGO_BUILD_TARGET', raising=False)
    monkeypatch.setattr(definitions['shutil'], 'which', lambda name: None if available == 'missing' else 'cargo')
    warnings = []
    monkeypatch.setattr(command, 'warn', warnings.append)
    def build(args, **kwargs):
        assert '--offline' in args and '--release' in args and '--locked' in args
        assert 'RUSTFLAGS' not in kwargs['env'] and 'CARGO_ENCODED_RUSTFLAGS' not in kwargs['env']
        if available == 'failed':
            raise subprocess.CalledProcessError(1, args)
        suffix = Path(command.get_ext_fullpath(extension.name)).suffix
        source = tmp_path / 'native/p4long/target/release' / (
            ('hfmodem_p4long' if suffix == '.dll' else 'libhfmodem_p4long') + suffix)
        source.parent.mkdir(parents=True)
        source.write_bytes(b'packaged native library')
    monkeypatch.setattr(definitions['subprocess'], 'run', build)
    command.build_extension(extension)
    destination = Path(command.get_ext_fullpath(extension.name))
    if available == 'built':
        assert destination.read_bytes() == b'packaged native library' and not warnings
    else:
        assert not destination.exists() and warnings


@pytest.mark.wall_clock
@pytest.mark.parametrize('sl', range(2, 11))
def test_whole_frame_cost(backend, monkeypatch, record_property, sl):
    g = engine.geometry(sl)
    fields = np.random.default_rng(1111 + sl).normal(size=g.serialized_size)
    member = engine.LongMember(g, fields)
    monkeypatch.setattr(native, 'backend', backend)
    samples = []
    for _ in range(12):
        job = engine.LongSearch([member], policy='exhaustive')
        started = time.perf_counter_ns()
        result = job.advance()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        assert result.completed_tier == 9 and job._native.crossings == 1
    record_property('form', f'SL{sl}')
    record_property('native_median_ms', float(np.median(samples[1:])))
    record_property('native_p95_ms', float(np.percentile(samples[1:], 95)))
    record_property('crossings_per_frame', 1)

