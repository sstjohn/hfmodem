# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Focused checks for the scheduling and IPC seams used by the interop test."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest

import run as h
import paths as p
import prepare as setup_tool


def source_tree(root, version='0.3.2'):
    for name in ('hfmodem', 'hfhost', 'creance'):
        project = root / 'packages' / name
        package = project / name
        package.mkdir(parents=True)
        (project / 'pyproject.toml').write_text(f'[project]\nversion = "{version}"\n')
        (package / '__init__.py').write_text(f'__version__ = "{version}"\n')
        (package / 'runtime.py').write_text('VALUE = 1\n')
    return root


@pytest.mark.parametrize('xdg', ['unset', 'empty', 'relative', 'absolute'])
def test_default_cache_is_external_and_namespaced_by_source(monkeypatch, tmp_path, xdg):
    home = tmp_path / 'home'
    monkeypatch.setattr(Path, 'home', lambda: home)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    base = home / '.cache'
    if xdg == 'absolute':
        base = tmp_path / 'explicit-cache'
        monkeypatch.setenv('XDG_CACHE_HOME', str(base))
    elif xdg == 'empty':
        monkeypatch.setenv('XDG_CACHE_HOME', '')
    elif xdg == 'relative':
        monkeypatch.setenv('XDG_CACHE_HOME', 'relative-cache')
    root = tmp_path / 'archive'
    got = p.default_cache(root)
    assert got.is_relative_to(base / 'hfmodem/xara-interop')
    assert not got.is_relative_to(root)
    assert got != p.default_cache(tmp_path / 'another-archive')


def test_explicit_cache_override_survives_archive_path_selection(tmp_path):
    wanted = tmp_path / 'user-cache'
    env = dict(os.environ, HFMODEM_INTEROP_CACHE=str(wanted), HFMODEM_INTEROP_ROOT=str(tmp_path / 'kit'))
    got = subprocess.check_output([sys.executable, '-c', 'import paths; print(paths.CACHE)'],
                                  cwd=p.HERE, env=env, text=True).strip()
    assert got == str(wanted.resolve())


def archive_entry(root):
    tools = root / 'tools/interop/xara'
    tools.mkdir(parents=True)
    for name in ('check.sh', 'cli.py'):
        shutil.copyfile(p.HERE / name, tools / name)
    return tools / 'check.sh'


def test_archive_shell_entry_works_without_git_metadata(tmp_path):
    entry = archive_entry(tmp_path / 'kit')
    env = dict(os.environ, HFMODEM_INTEROP_PYTHON=sys.executable)
    result = subprocess.run(['bash', str(entry), '--help'], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Explicit offline Xara interop' in result.stdout


def test_source_identity_is_relocatable_and_binds_runtime_assets(tmp_path):
    first = source_tree(tmp_path / 'first')
    asset = first / 'packages/hfmodem/hfmodem/assets/table.npz'
    asset.parent.mkdir()
    asset.write_bytes(b'runtime asset')
    second = tmp_path / 'second'
    shutil.copytree(first, second)
    want = p.source_identity(first)
    assert want == p.source_identity(second)
    assert want['package_versions'] == dict.fromkeys(('hfmodem', 'hfhost', 'creance'), '0.3.2')
    assert 'packages/hfmodem/hfmodem/assets/table.npz' in want['files']
    (second / 'packages/hfmodem/hfmodem/assets/table.npz').write_bytes(b'changed asset')
    assert p.source_identity(second)['sha256'] != want['sha256']


def test_source_identity_excludes_test_and_generated_cache_bulk(tmp_path):
    root = source_tree(tmp_path / 'kit')
    want = p.source_identity(root)
    for name in ('tests/fixture.wav', '__pycache__/runtime.pyc', '.pytest_cache/state'):
        path = root / 'packages/hfmodem/hfmodem' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not runtime identity')
    (root / '.venv').mkdir()
    (root / '.venv/generated').write_bytes(b'not source')
    assert p.source_identity(root) == want


@pytest.mark.parametrize('conflict', ['metadata', 'runtime', 'package'])
def test_source_identity_rejects_mixed_versions(tmp_path, conflict):
    root = source_tree(tmp_path / 'kit')
    if conflict in ('metadata', 'package'):
        (root / 'packages/creance/pyproject.toml').write_text('[project]\nversion = "different"\n')
    if conflict in ('runtime', 'package'):
        (root / 'packages/creance/creance/__init__.py').write_text('__version__ = "different"\n')
    with pytest.raises(RuntimeError, match='version'):
        p.source_identity(root)


def test_source_override_imports_selected_checkout_and_records_provenance(tmp_path):
    # Reuse packages, but select a distinct root so an ignored override is visible.
    (tmp_path / 'packages').symlink_to(h.ROOT / 'packages', target_is_directory=True)
    env = dict(os.environ, HFMODEM_INTEROP_ROOT=str(tmp_path))
    script = '''
import json
import run as h
h.PROBE = h.Path(h.__file__)
h.subprocess.check_output = lambda argv, **kw: str(h.ROOT) if argv[2] == str(h.ROOT) else 'xara'
print(json.dumps({'root': str(h.ROOT), 'phy': h.phy.__file__, 'source': h.provenance()['kestrel_source_root']}))
'''
    got = json.loads(subprocess.check_output(
        [sys.executable, '-c', script], cwd=Path(h.__file__).parent, env=env, text=True))
    assert got['root'] == got['source'] == str(tmp_path.resolve())
    assert Path(got['phy']).is_relative_to(tmp_path / 'packages')


@pytest.mark.parametrize('scenario', ['missing', 'preloaded'])
def test_wrong_source_cannot_fall_back_to_an_editable_install(tmp_path, scenario):
    env = dict(os.environ, HFMODEM_INTEROP_ROOT=str(tmp_path if scenario == 'missing' else h.ROOT))
    script = 'import run'
    if scenario == 'preloaded':
        script = "import sys, types; sys.modules['hfmodem'] = types.SimpleNamespace(__file__='/elsewhere/__init__.py'); import run"
    result = subprocess.run([sys.executable, '-c', script], cwd=p.HERE, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert ('refusing an editable-install fallback' if scenario == 'missing'
            else 'already imported outside selected Kestrel root') in result.stderr


def test_manifest_paths_do_not_depend_on_the_current_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_tool, 'CACHE', tmp_path / 'cache with spaces')
    monkeypatch.setattr(setup_tool, 'XARA', tmp_path / 'peer with spaces')
    manifest = setup_tool.prepare_manifest()
    import tomllib
    got = tomllib.loads(manifest.read_text())
    assert got['package']['build'] == str(p.HERE / 'build.rs')
    assert got['bin'][0]['path'] == str(p.HERE / 'src/main.rs')
    assert got['dependencies']['xara-phy']['path'] == str(tmp_path / 'peer with spaces/crates/xara-phy')
    assert (manifest.parent / 'Cargo.lock').read_bytes() == (p.HERE / 'Cargo.lock').read_bytes()


@pytest.mark.parametrize('revision,dirty', [(p.PIN, ' M file.rs'),
    (p.PIN, '?? .cargo/config.toml'), ('wrong', '')])
def test_unreviewed_xara_source_is_refused(monkeypatch, revision, dirty):
    monkeypatch.setattr(p, 'git', lambda path, *args: revision if args[0] == 'rev-parse' else dirty)
    with pytest.raises(RuntimeError, match='clean at selected commit'):
        p.validate_xara()


@pytest.mark.parametrize('changed', ['source', 'binary', 'location'])
def test_stale_adapter_receipt_is_refused(monkeypatch, tmp_path, changed):
    binary, receipt = tmp_path / 'probe', tmp_path / 'build.json'
    binary.write_bytes(b'probe')
    inputs = {'src/main.rs': 'original'}
    monkeypatch.setattr(p, 'PROBE', binary)
    monkeypatch.setattr(p, 'RECEIPT', receipt)
    monkeypatch.setattr(p, 'validate_xara', lambda: p.PIN)
    monkeypatch.setattr(p, 'build_inputs', lambda: dict(inputs))
    receipt.write_text(json.dumps({'harness_root': str(p.HERE),
        'xara_root': str(p.XARA), 'xara_commit': p.PIN,
        'requested_ref': p.PIN, 'source_url': p.XARA_URL, 'probe_path': str(binary),
        'rust_version': p.RUST_VERSION, 'build_inputs': dict(inputs),
        'probe_sha256': p.sha256(binary)}))
    p.validate_probe()
    if changed == 'source':
        inputs['src/main.rs'] = 'edited while staged'
    elif changed == 'binary':
        binary.write_bytes(b'different')
    else:
        monkeypatch.setattr(p, 'XARA', tmp_path / 'other-peer')
    with pytest.raises(RuntimeError, match='absent or stale'):
        p.validate_probe()


def test_setup_does_not_overwrite_an_existing_non_git_directory(monkeypatch, tmp_path):
    peer = tmp_path / 'peer'
    peer.mkdir()
    (peer / 'keep').write_text('untouched')
    monkeypatch.setenv('HFMODEM_INTEROP_XARA', str(peer))
    with pytest.raises(RuntimeError, match='refusing to replace'):
        setup_tool.fetch_xara('unused')
    assert (peer / 'keep').read_text() == 'untouched'


def test_provenance_reports_staged_edits(monkeypatch, tmp_path):
    # `git diff` alone misses index-only edits, the normal state of this tool's
    # initial staged review. Use a real tiny repository to exercise that seam.
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True)
    source_tree(tmp_path)
    git('init', '-q')
    tracked = tmp_path / 'source.py'
    tracked.write_text('before\n')
    git('add', 'source.py', 'packages')
    git('-c', 'user.name=Harness Test', '-c', 'user.email=harness@example.invalid',
        'commit', '-qm', 'fixture')
    tracked.write_text('after\n')
    git('add', 'source.py')
    assert git('diff', '--name-only') == ''
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    monkeypatch.setattr(h, 'XARA', tmp_path)
    monkeypatch.setattr(h, 'PROBE', Path(h.__file__))
    got = h.provenance()
    assert got['kestrel_tracked_changes'] == 'source.py'
    assert 'M  source.py' in got['kestrel_status']


@pytest.fixture
def local_peer(tmp_path):
    upstream = tmp_path / 'upstream'
    subprocess.run(['git', 'init', '-q', str(upstream)], check=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(upstream), *args], text=True).strip()
    git('config', 'user.name', 'Harness Test')
    git('config', 'user.email', 'harness@example.invalid')
    def commit(text):
        (upstream / 'source').write_text(text)
        git('add', 'source')
        git('commit', '-qm', text)
        return git('rev-parse', 'HEAD')
    first = commit('first version')
    git('tag', '-am', 'annotated fixture', 'fixture-v1')
    return upstream, first, commit


def test_archive_does_not_inherit_git_identity_or_environment(monkeypatch, tmp_path, local_peer):
    upstream, first, _ = local_peer
    root = source_tree(upstream / 'extracted')
    assert p.git_metadata(upstream)['commit'] == first
    assert p.git_metadata(root) is None
    entry = archive_entry(root)
    borrowed = upstream / '.venv/bin/python'
    borrowed.parent.mkdir(parents=True)
    borrowed.write_text('#!/bin/sh\nexit 79\n')
    borrowed.chmod(0o755)
    env = dict(os.environ)
    env.pop('HFMODEM_INTEROP_PYTHON', None)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (binaries / 'python3').symlink_to(sys.executable)
    env['PATH'] = str(binaries) + os.pathsep + env.get('PATH', '')
    result = subprocess.run(['bash', str(entry), '--help'], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_archive_provenance_has_content_identity_without_a_fake_commit(monkeypatch, tmp_path, local_peer):
    upstream, first, _ = local_peer
    root = source_tree(tmp_path / 'archive')
    monkeypatch.setattr(h, 'ROOT', root)
    monkeypatch.setattr(h, 'XARA', upstream)
    monkeypatch.setattr(h, 'PROBE', Path(h.__file__))
    monkeypatch.setattr(h, 'selection', lambda: {'source_url': str(upstream), 'requested_ref': first})
    got = h.provenance()
    assert got['kestrel_source_kind'] == 'archive'
    assert got['kestrel_commit'] is got['kestrel_tracked_changes'] is got['kestrel_status'] is None
    assert got['kestrel_source_identity'] == p.source_identity(root)
    assert got['xara_resolved_commit'] == first


@pytest.mark.parametrize('style', ['tag', 'hash', 'HEAD'])
def test_setup_resolves_refs_to_separate_immutable_checkouts(monkeypatch, tmp_path, local_peer, style):
    upstream, first, commit = local_peer
    monkeypatch.setattr(setup_tool, 'CACHE', tmp_path / 'cache')
    monkeypatch.delenv('HFMODEM_INTEROP_XARA', raising=False)
    ref = {'tag': 'fixture-v1', 'hash': first, 'HEAD': 'HEAD'}[style]
    chosen = setup_tool.fetch_xara(str(upstream), ref)
    assert chosen['requested_ref'] == ref and chosen['xara_commit'] == first
    assert chosen['source_url'] == str(upstream)
    source = Path(chosen['xara_root'])
    p.validate_checkout(source, first)
    second = commit('second version')
    newer = setup_tool.fetch_xara(str(upstream), 'HEAD')
    assert newer['xara_commit'] == second and newer['xara_root'] != str(source)
    assert (source / 'source').read_text() == 'first version'
    p.validate_checkout(source, first)


def test_dirty_override_is_preserved_instead_of_reset(monkeypatch, tmp_path, local_peer):
    upstream, first, _ = local_peer
    monkeypatch.setattr(setup_tool, 'CACHE', tmp_path / 'cache')
    monkeypatch.delenv('HFMODEM_INTEROP_XARA', raising=False)
    chosen = setup_tool.fetch_xara(str(upstream), first)
    source = Path(chosen['xara_root'])
    (source / 'source').write_text('user edit')
    monkeypatch.setenv('HFMODEM_INTEROP_XARA', str(source))
    with pytest.raises(RuntimeError, match='refusing to reset'):
        setup_tool.fetch_xara(str(upstream), 'fixture-v1')
    assert (source / 'source').read_text() == 'user edit'


def test_failed_alternative_build_keeps_ordinary_run_on_previous_selection(monkeypatch, tmp_path, local_peer):
    upstream, first, commit = local_peer
    cache = tmp_path / 'cache'
    monkeypatch.setattr(setup_tool, 'CACHE', cache)
    monkeypatch.delenv('HFMODEM_INTEROP_XARA', raising=False)
    old = setup_tool.fetch_xara(str(upstream), 'fixture-v1')
    probe = cache / 'bin' / first / 'kestrel-xara-probe'
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b'previous working adapter')
    receipt = cache / 'build.json'
    before = json.dumps(dict(old, harness_root=str(p.HERE), rust_version=p.RUST_VERSION,
                             build_inputs=p.build_inputs(), probe_path=str(probe),
                             probe_sha256=p.sha256(probe)))
    receipt.write_text(before)
    commit('an incompatible later peer')
    monkeypatch.setattr(setup_tool, 'RECEIPT', receipt)
    monkeypatch.setattr(setup_tool, 'prepare_toolchain', lambda env: 'fixture-cargo')
    real_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == 'fixture-cargo':
            raise subprocess.CalledProcessError(1, argv)
        return real_run(argv, **kwargs)
    monkeypatch.setattr(setup_tool.subprocess, 'run', run)
    assert setup_tool.main(['--xara-url', str(upstream), '--xara-ref', 'HEAD']) == 1
    assert receipt.read_text() == before and probe.read_bytes() == b'previous working adapter'
    # An ordinary fresh runtime must validate the old selection even though
    # the generated build manifest now contains the failed alternative's path.
    env = dict(os.environ, HFMODEM_INTEROP_CACHE=str(cache))
    result = real_run([sys.executable, '-c',
        'import paths; paths.validate_probe(); print(paths.selection()["requested_ref"])'],
        cwd=p.HERE, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'fixture-v1'


def test_setup_default_still_requests_the_tested_pin(monkeypatch):
    requested = []
    def fetch(url, ref):
        requested.append(ref)
        raise RuntimeError('stop before building')
    monkeypatch.setattr(setup_tool, 'fetch_xara', fetch)
    assert setup_tool.main([]) == 1
    assert requested == [p.PIN]


def test_head_provenance_stays_on_resolved_commit_until_setup(monkeypatch, tmp_path, local_peer):
    upstream, first, commit = local_peer
    monkeypatch.setattr(setup_tool, 'CACHE', tmp_path / 'cache')
    monkeypatch.delenv('HFMODEM_INTEROP_XARA', raising=False)
    chosen = setup_tool.fetch_xara(str(upstream), 'HEAD')
    commit('remote HEAD moved after setup')
    monkeypatch.setattr(h, 'XARA', Path(chosen['xara_root']))
    monkeypatch.setattr(h, 'PROBE', Path(h.__file__))
    monkeypatch.setattr(h, 'selection', lambda: dict(chosen))
    got = h.provenance()
    assert got['xara_requested_ref'] == 'HEAD'
    assert got['xara_commit'] == got['xara_resolved_commit'] == first
    assert got['xara_source_url'] == str(upstream)


def test_timeline_transmissions_are_serial_and_partial_blocks_keep_their_place():
    t = h.Timeline()
    t.n = 3
    t.emit(np.arange(1, 6))
    t.emit(np.arange(6, 9))
    assert t.blocked_until == 11
    assert t.clock() == 1000 + 11/48000
    np.testing.assert_array_equal(t.output(0, 7), [0, 0, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(t.output(7, 7), [5, 6, 7, 8, 0, 0, 0])
    assert not t.tx


def test_adapter_short_response_is_an_error():
    reader, writer = os.pipe()
    with os.fdopen(reader, 'rb', buffering=0) as pipe:
        os.write(writer, b'ab')
        os.close(writer)
        with pytest.raises(EOFError):
            h.read_exact(pipe, 4)


def test_adapter_silence_has_a_deadline():
    reader, writer = os.pipe()
    try:
        with os.fdopen(reader, 'rb', buffering=0) as pipe:
            with pytest.raises(TimeoutError):
                h.read_exact(pipe, 4, timeout=.01)
    finally:
        os.close(writer)


@pytest.mark.usefixtures('real_peer')
def test_bw500_low_renderers_accept_host_levels_and_cross_decode(tmp_path):
    # This catches confusing BW500's host-level API with wideband record keys.
    for level, size in [(1, 12), (2, 25), (3, 37)]:
        body = bytes(range(size-3)) + b'\x82'
        frame = body + h.crc16_genibus(body).to_bytes(2, 'big')
        path = tmp_path / f'l{level}.f64'
        h.channel(h.render([frame], 500, level)).tofile(path)
        assert h.invoke('rx', 500, level, path, 1).strip() == frame.hex()


@pytest.mark.usefixtures('real_peer')
def test_crc_damage_is_rejected_by_both_receivers(tmp_path):
    size = 92  # BW2300 L4
    body = bytes(range(size-3)) + b'\x82'
    bad = body + (h.crc16_genibus(body) ^ 1).to_bytes(2, 'big')
    p = tmp_path / 'bad.f64'
    h.channel(h.render([bad], 2300, 4)).tofile(p)
    assert h.invoke('rx', 2300, 4, p, 1).strip() == ''
    h.invoke('tx', 2300, 4, bad.hex(), p)
    read = h.phy.decode_level(h.channel(np.fromfile(p, dtype='<f8')), '2300', 4)
    assert read is None or not read.frames


@pytest.mark.usefixtures('real_peer')
def test_real_station_ipc_connect_emits_audio_and_retires():
    events = []
    x = h.Xara(2300, False, 'OFF', lambda side, event: events.append(event))
    try:
        x.command('CONNECT')
        audio = [x.step(np.zeros(480)) for _ in range(100)]
        assert np.max(np.abs(np.concatenate(audio))) > .01
        assert any(e.startswith('Trace("tx c26') for e in events)
    finally:
        x.close()
    assert x.process.returncode == 0


@pytest.fixture
def peer_free(monkeypatch):
    def forbidden():
        pytest.fail('peer-free selection tried to validate Xara')
    monkeypatch.setattr(h, 'validate_probe', forbidden)
    monkeypatch.setattr(p, 'validate_probe', forbidden)


@pytest.fixture
def real_peer():
    p.validate_probe()


@pytest.mark.usefixtures('peer_free')
def test_peer_free_chunking_seeds_and_silence():
    from channel import Channel, Condition
    x = np.random.default_rng(9).normal(0, .03, 48000*2+113)
    condition = Condition('-5', 'moderate', 50, .1, 50, 'standard')
    def run(chunks, seed=32):
        c = Channel(condition, seed=seed, horizon=10)
        pieces, at = [], 0
        for size in chunks:
            if at >= len(x): break
            pieces.append(c.process(x[at:at+size], at))
            at += len(pieces[-1])
        if at < len(x): pieces.append(c.process(x[at:], at))
        return np.concatenate(pieces), c
    whole, a = run([len(x)])
    blocks, b = run([480]*300)
    odd, _ = run([1, 777, 5, 1231, 999]*100)
    np.testing.assert_allclose(whole, blocks, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(whole, odd)
    np.testing.assert_array_equal(whole, run([len(x)])[0])
    assert not np.array_equal(whole, run([len(x)], 33)[0])
    assert a.evidence()['seeds'] == b.evidence()['seeds']


@pytest.mark.usefixtures('peer_free')
def test_peer_free_snr2500_and_ssb():
    from channel import Channel, Condition, RATE, filters
    from scipy.signal import periodogram
    n = np.arange(RATE*20)
    condition = Condition('-15')
    x = np.sqrt(2)*condition.rms*np.sin(2*np.pi*1500*n/RATE)
    noise = Channel(condition, horizon=21)
    signal = Channel(Condition('inf'), horizon=21)
    y = signal.process(x, 0)[RATE*2:-RATE]
    z = noise.process(np.zeros(len(x)), 0)[RATE*2:-RATE]
    f, pxx = periodogram(z, RATE)
    power = float(np.sum(pxx[(f >= 250) & (f < 2750)])*(f[1]-f[0]))
    snr = 10*np.log10(np.mean(y*y)/power)
    assert abs(snr+15) < .25
    assert abs(10*np.log10(power/.05**2)) < .25
    assert abs(10*np.log10(noise.evidence()['measured_reference_noise_power']/.05**2)) < .25
    for meta in filters()[2].values():
        assert meta['ripple_db'] <= .1 and meta['rejection_db'] >= 60
    assert np.sum(pxx[f >= 3500]) < np.sum(pxx[(f >= 60) & (f <= 3150)])*1e-5


@pytest.mark.usefixtures('peer_free')
def test_peer_free_delayed_fading_tap_keeps_boundary_energy():
    from channel import Fading, PART
    f = Fading('poor', 11)
    for tap in f.model.taps: tap[:] = 1
    x = np.zeros(PART, complex)
    x[-1] = 1
    first = f.process(x, 0)
    second = f.process(np.zeros(PART, complex), PART)
    assert first[-1] == pytest.approx(1/np.sqrt(2))
    assert second[95] == pytest.approx(1/np.sqrt(2))
    assert np.count_nonzero(second) == 1


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('ppm', [-50, 50])
def test_peer_free_clock_slope_and_horizon(ppm):
    from channel import Clock, PART
    c = Clock(ppm, 3600)
    out = []
    for at in range(0, 48000, PART):
        out.append(c.process(np.arange(at, at+PART, dtype=float), at))
    out = np.concatenate(out)
    n = np.arange(len(out))
    valid = n > c.delay+64
    delta = n[valid]-out[valid]
    slope = np.polyfit(n[valid], delta, 1)[0]
    assert slope == pytest.approx(-ppm*1e-6, abs=1e-8)
    assert np.mean(delta+ppm*1e-6*n[valid]) == pytest.approx(c.delay, abs=.001)
    with pytest.raises(ValueError): Clock(ppm, 3601)
    with pytest.raises(ValueError): Clock(51, 1)


@pytest.mark.usefixtures('peer_free')
def test_peer_free_phase_and_latency():
    from channel import Channel, Condition, RATE
    from scipy.signal import hilbert
    n = np.arange(RATE*4)
    x = np.sin(2*np.pi*1000*n/RATE)
    c = Channel(Condition('inf', offset=50, drift=.1), horizon=5)
    y = c.process(x, 0)
    angle = np.unwrap(np.angle(hilbert(y)))
    lo, hi = RATE, RATE*3
    frequency = np.diff(angle[lo:hi])*RATE/(2*np.pi)
    fit = np.polyfit(n[lo:hi-1]/RATE, frequency, 1)
    assert fit[0] == pytest.approx(.1, abs=.003)
    assert fit[1] == pytest.approx(1050, abs=.05)
    baseline = Channel(Condition('inf'), horizon=5)
    fade = Channel(Condition('inf', 'poor'), horizon=5)
    assert c.delay == baseline.delay == fade.delay
    for at in range(0, RATE, 480):
        baseline.process(np.zeros(480), at)
    assert baseline.delay == c.delay


@pytest.mark.usefixtures('peer_free')
def test_peer_free_clean_bypass_and_calibration_extent():
    from channel import Channel, Condition, calibrate, subseed
    x = np.array([0., .1, -.2, .3, 0.])
    out = Channel(horizon=1).process(np.pad(x, (0, 3000)), 0)
    np.testing.assert_array_equal(out[2400:2405], x)
    assert not out[:2400].any()
    for condition in (Condition(), Condition('3')):
        wire = Channel(condition, horizon=1)
        for at in range(0, 4800, 480):
            preview = wire.next_block()
            actual = wire.process(np.ones(480), at)
            np.testing.assert_array_equal(preview, actual)
    a, table = calibrate(x, .12)
    b, padded = calibrate(np.pad(x*100, (100, 200)), .12)
    np.testing.assert_allclose(a, b[100:105], atol=1e-16)
    assert table['samples'] == padded['samples'] == 3
    assert padded['gain'] == pytest.approx(table['gain']/100)
    assert subseed(1, 500, 'kestrel', 'c', 0, 'A->B', 'fade') != subseed(1, 500, 'kestrel', 'c', 0, 'B->A', 'fade')
    for kwargs in ({'offset': 1}, {'fading': 'good'}, {'qrn': 'standard'}):
        with pytest.raises(ValueError): Condition(**kwargs)


@pytest.mark.usefixtures('peer_free')
def test_peer_free_qrn_tails_rate_and_power():
    from channel import QRN, PART, subseed
    seeds = {name: subseed(27, 2300, 'xara', 'qrn', 0, 'A->B', name)
             for name in ('qrn-arrival', 'qrn-duration', 'qrn-amplitude')}
    q = QRN(seeds, 2)
    duration = q.duration
    from types import SimpleNamespace
    q.duration = SimpleNamespace(uniform=lambda *args: np.log(.0003))
    q.next = PART-2
    first = q.process(0)
    q.duration = duration
    second = q.process(PART)
    assert np.any(first[-2:]) and np.any(second[:30])
    for at in range(2*PART, 48000*60, PART): q.process(at)
    assert abs(q.events/60-20) < 2
    expected_samples = q.events*(.0003-.00005)/np.log(.0003/.00005)*48000
    assert q.raw_energy/expected_samples == pytest.approx(2, rel=.2)


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('extra', [
    ['--noise', '.01'], ['--noise=.01'], ['--snr', 'nan'], ['--snr', '-inf'],
    ['--snr', 'clean', '--offset', '50'], ['--snr', '3', '--clock-ppm', '51'],
    ['--ladder', 'full', '--snr', '3'], ['--bytes', '0']])
def test_peer_free_parser_rejects_invalid_conditions(extra, capsys):
    with pytest.raises(SystemExit): h.parse_args(['ladder', '--out', '/unused', *extra])
    if extra[0].startswith('--noise'): assert 'migrate to --snr' in capsys.readouterr().err


@pytest.mark.usefixtures('peer_free')
def test_peer_free_parser_inventory_and_pinned_events():
    args = h.parse_args(['ladder', '--out', '/unused', '--snr', '3', '--fading', 'poor'])
    assert args.custom and args.condition.snr == '3'
    assert len(h.conditions()) == 11
    assert len({c.identity for c in h.conditions(True)}) == len(h.conditions(True))
    assert h.structured('Bitrate { level: 4, bps: 123, tx: false }')['tx'] is False
    assert h.structured('Trace("tx L12x2 Data")')['frames'] == 2
    assert h.structured('Trace("tx c26 f0+ Connect")')['extra'] == 9
    assert h.structured('Trace("tx c22 f1+ Connect")')['extra'] == 7
    assert h.structured('Trace("tx c12 f0 Ack")')['extra'] == 0
    for size in (1, 256, 9999): assert all(len(p) == size for p in h.payloads(size))


@pytest.mark.usefixtures('peer_free')
def test_peer_free_prefix_classification_and_thresholds():
    h.check_prefix(b'abc', b'abcdef')
    for got in (b'abx', b'abcdefg', b'abcdefabcdef'):
        with pytest.raises(h.DeliveryError): h.check_prefix(got, b'abcdef')
    assert h.classify(h.DeliveryError('deadline'), 'connect', False, None) == 'no-connect'
    assert h.classify(h.DeliveryError('deadline'), 'a-to-b', True, None) == 'transfer-timeout'
    assert h.classify(EOFError(), 'connect', False, None) == 'invalid'
    assert h.classify(None, 'post-close', True, 'a') == 'corruption'
    base = dict(ok=True, outcome='pass')
    session = dict(ok=False, outcome='transfer-timeout', failed_phase='b-to-a',
                   **{'ptt-replay': {'outcome': 'recovered', 'recovered_phases': ['b-to-a']}})
    assert h.paired(session, base) == 'rx-gate-shortfall'
    session['corruption_receiver'] = 'a'
    assert h.paired(session, dict(ok=False, outcome='corruption')) == 'kestrel-corruption'
    cases = []
    for snr in (10, 3, 0, -5):
        for caller in ('kestrel', 'xara'):
            for seed in range(3):
                row = dict(bw=500, registered=False, condition=dict(snr=str(snr), fading='none', offset=0,
                    drift=0, clock_ppm=0, qrn='off'), caller=caller, seed=42, replicate=seed, ok=snr >= 10)
                cases.append(dict(session=row, control=dict(row, ok=snr >= 0)))
    assert h.thresholds(cases, 3)[0]['verdict'] == 'shortfall'
    assert h.thresholds([p for p in cases if p['session']['replicate'] == 0], 1)[0]['verdict'] == 'smoke'
    duplicated = [p for p in cases if p['session']['caller'] == 'kestrel']*2
    assert h.thresholds(duplicated, 3)[0]['verdict'] == 'incomplete'
    from copy import deepcopy
    wrong = deepcopy(cases)
    wrong[0]['control']['seed'] = 99
    assert h.thresholds(wrong, 3)[0]['verdict'] == 'incomplete'
    assert h.thresholds(cases, 3, 99)[0]['verdict'] == 'incomplete'


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('transmit_in_stream', [False, True])
def test_peer_free_receive_path_matches_production(transmit_in_stream):
    from hfmodem.kestrel.host.station_runtime import StationTransport
    from types import SimpleNamespace
    samples = np.random.default_rng(91).normal(0, .001, 48000*4).astype(np.float32)
    samples[48000:96000] += .2*np.sin(2*np.pi*1500*np.arange(48000)/48000)
    def run(production):
        rx = h.ReceivePath()
        stream, bursts, states = [], [], []
        def feed(x):
            assert x.dtype == np.float32
            stream.append(x.copy())
            if transmit_in_stream and len(stream) == 11: rx._reset_rx(53000)
        hs = SimpleNamespace(on_rx_stream=feed)
        rx.hs = hs
        rx._skip_before = 137
        if production:
            for at in range(0, len(samples), 4800):
                rx.feed(at, samples[at:at+4800])
                burst = StationTransport.next_rx_burst(rx, 1, hs)
                if burst is not None: bursts.append(burst)
                states.append(rx.receiving)
        else:
            endpoint = h.Kestrel.__new__(h.Kestrel)
            endpoint.t, endpoint.clean, endpoint.replay = h.Timeline(), False, False
            endpoint.pieces, endpoint.rx = [], rx
            endpoint.trace = lambda *args: None
            def step(timeout):
                burst = rx.next_rx_burst(timeout, hs)
                if burst is not None: bursts.append(burst)
                states.append(rx.receiving)
            endpoint.runtime = SimpleNamespace(step=step)
            for at in range(0, len(samples), 480):
                endpoint.step(samples[at:at+480].astype(float), at)
        assert len(stream[0]) == 4800-137
        return stream, bursts, states
    actual, expected = run(False), run(True)
    assert actual[2] == expected[2]
    for left, right in zip(actual[:2], expected[:2]):
        assert len(left) == len(right)
        for a, b in zip(left, right): np.testing.assert_array_equal(a, b)
    if not transmit_in_stream: assert actual[1]


@pytest.mark.usefixtures('peer_free')
def test_peer_free_calibration_trace_precedes_samples(monkeypatch, tmp_path):
    from channel import Condition
    requests = []
    native = np.pad(.1*np.sin(2*np.pi*1500*np.arange(900)/48000), (7, 53))
    def invoke(*args):
        requests.append(args[:-1])
        native.astype('<f8').tofile(args[-1])
        return ''
    monkeypatch.setattr(h, 'invoke', invoke)
    table = h.Calibration(500, Condition('3'), tmp_path)
    table.observe('Trace("tx c26 f0+ Connect")', 0)
    output = table.process(native, 0)
    assert requests == [('tx-control', 500, 0, 26, 9)]
    assert table.bursts[0]['realized_calibrated_rms'] == pytest.approx(Condition('3').rms)
    table.observe('Trace("tx c26 f0+ Connect")', 960)
    np.testing.assert_array_equal(output, table.process(native, 960))
    assert len(requests) == 1
    with pytest.raises(ValueError, match='uncalibrated'):
        table.process(native, 1920)


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('snr', ['clean', 'inf'])
def test_peer_free_runtime_epoch_and_startup(snr):
    from contextlib import ExitStack
    from types import SimpleNamespace
    from channel import Condition
    with ExitStack() as resources:
        endpoint = h.Kestrel(SimpleNamespace(condition=Condition(snr), registered=False, compression='OFF'),
                             500, lambda *args: None, resources)
        rx = endpoint.rx
        before = rx.receive_epoch
        assert before is not None
        rx._reset_rx(100)
        assert rx.receive_epoch != before
        before = rx.receive_epoch
        rx.contact.receive_boundary = (1, 200)
        rx.before_step()
        assert rx.receive_epoch != before and rx._skip_before == 200
        endpoint.step(np.zeros(4800), 0)
        assert endpoint.runtime.modem is endpoint.modem


@pytest.mark.usefixtures('peer_free')
def test_peer_free_runtime_consumes_response_at_retry_deadline():
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch
    from channel import Condition
    from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK
    with ExitStack() as resources:
        endpoint = h.Kestrel(SimpleNamespace(condition=Condition('inf'), registered=False, compression='OFF'),
                             500, lambda *args: None, resources)
        with patch.object(endpoint.arq, 'time', SimpleNamespace(monotonic=endpoint.t.clock)):
            endpoint.command('CONNECT')
            endpoint.runtime.step(0)
            answer = MK.synth_burst('W1AW', VF.for_bw(VF.connect_response('500'), '500'), .27)
            audio = np.pad(answer, (6720, 14400))
            at = endpoint.t.blocked_until+h.ECHO_GUARD
            endpoint.runtime._retry_at = 1000+(at+len(audio))/h.RATE
            endpoint.step(audio, at)
        kinds = [b['metadata']['packet_kind'] for b in endpoint.calibration]
        assert kinds == ['connect-request-500', 'link-setup']
        assert endpoint.modem.station.connect_setup_pending
        assert all(generation != endpoint.rx._rx_generation for generation, _, _ in endpoint.rx.frames.queue)


@pytest.mark.usefixtures('peer_free')
def test_peer_free_ptt_replay_recovers_only_failed_receive_phase():
    primary = dict(connected=True, disconnected=False, received=True, confirmed=256, close_received=False)
    replay = dict(primary)
    assert h.recovered_phases(primary, replay, 256) == []
    row = dict(ok=False, outcome='missing-orderly-close', failed_phase='disconnect',
               **{'ptt-replay': dict(outcome='pass', recovered_phases=['b-to-a'])})
    assert h.paired(row, dict(ok=True, outcome='pass')) == 'cell-shortfall'
    replay['disconnected'] = True
    assert h.recovered_phases(primary, replay, 256) == []
    replay['close_received'] = True
    recovered = h.recovered_phases(primary, replay, 256)
    assert recovered == ['disconnect']
    row['ptt-replay']['recovered_phases'] = recovered
    assert h.paired(row, dict(ok=True, outcome='pass')) == 'rx-gate-shortfall'
    primary = dict(connected=False, disconnected=False, received=False, confirmed=0)
    assert h.recovered_phases(primary, replay, 256) == ['connect', 'a-to-b', 'b-to-a', 'disconnect']


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('receiver', ['a', 'b'])
def test_peer_free_control_corruption_is_a_peer_finding(receiver):
    control = dict(ok=False, outcome='corruption', corruption_receiver=receiver)
    for ok in (False, True):
        session = dict(ok=ok, outcome='pass' if ok else 'no-connect')
        assert h.paired(session, control) == 'peer-finding'
    session = dict(ok=False, outcome='corruption', corruption_receiver='b')
    assert h.paired(session, control) == 'peer-finding'
    assert h.paired(session, dict(ok=True, outcome='pass')) == 'interop-corruption'
    session['corruption_receiver'] = 'a'
    assert h.paired(session, control) == 'kestrel-corruption'


@pytest.mark.usefixtures('peer_free')
def test_peer_free_cli_deduplicates_selections():
    args = h.parse_args(['ladder', '--out', '/unused', '--bw', '500', '--bw', '500',
                         '--caller', 'kestrel', '--caller', 'kestrel', '--caller', 'xara',
                         '--level', '1', '--level', '1'])
    assert args.bw == [500] and args.level == [1] and args.caller == ['kestrel', 'xara']


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('kind', ['cw', 'unknown'])
def test_peer_free_fallback_extent_completion_and_channel_registration(monkeypatch, tmp_path, kind):
    from channel import Channel, Condition
    condition = Condition('inf')
    table = h.Calibration(500, condition, tmp_path)
    monkeypatch.setattr(h, 'inventory', lambda: [dict(bw=500, level=1)])
    monkeypatch.setattr(table, 'entry', lambda *args: dict(native_rms=.1))
    table.observe(f'Trace("tx {kind}")', 0)
    native = np.r_[np.zeros(7), np.full(200, .1), np.zeros(73), np.full(200, .1)]
    wire = Channel(condition, horizon=1)
    out = table.process(native, 0)
    entry = table.bursts[0]
    burst = wire.register(entry['start_sample'], None, 'a')
    wire.process(out, 0)
    table.observe('Trace("tx cw")', 480)
    wire.process(table.process(np.zeros(480), 480), 480)
    assert table.active is entry and len(table.pending) == 1
    table.observe('Ptt(false)', 960)
    wire.process(table.process(np.zeros(480), 960), 960)
    assert table.active is None and entry['samples'] == 473
    assert entry['native_count'] == 473 and entry['consumed'] == 1433
    assert entry['realized_native_rms'] == pytest.approx(np.sqrt(400*.1**2/473))
    wire.finish(burst, entry['samples'])
    wire.process(np.zeros(4800), 1440)
    evidence = wire.evidence()['bursts'][0]
    assert evidence['count'] == evidence['samples'] == 473 and evidence['energy'] > 0
    assert 'pending' not in evidence
    json.dumps(wire.evidence())


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('ppm', [-50, 50])
def test_peer_free_late_burst_snr_tracks_clock_warp(ppm):
    from channel import Channel, Condition, RATE
    wire = Channel(Condition('inf', clock_ppm=ppm), horizon=3600)
    at = 3000*RATE
    wire.at = wire.cursor = at
    count = 2400
    tone = .12*np.sqrt(2)*np.sin(2*np.pi*1500*np.arange(count)/RATE+.37)
    burst = wire.register(at, count, 'late')
    out = wire.process(np.pad(tone, (0, RATE)), at)
    lo, hi = wire.window(at, count, output=True)
    image = wire.filter_info['image']['group_delay_samples']
    receive = wire.filter_info['receive']['group_delay_samples']
    expected = int(np.ceil((at+image+wire.clock.delay)/(1+ppm*1e-6)+receive+wire.propagation+480))
    assert lo == expected
    assert abs(lo-(at+wire.delay)) > 7000
    assert burst['samples'] == hi-lo
    assert burst['energy'] == pytest.approx(float(out[lo-at:hi-at]@out[lo-at:hi-at]), abs=1e-12)
    assert 10*np.log10(burst['energy']/burst['samples']/.12**2) == pytest.approx(0, abs=.25)


@pytest.mark.usefixtures('peer_free')
def test_peer_free_nominal_delay_filter_metadata_and_horizon(monkeypatch):
    from types import SimpleNamespace
    from channel import Channel, Condition, filters
    import hashlib
    image, receive, meta = filters()
    assert meta['image']['taps'] == len(image) == 1055
    assert meta['receive']['kind'] == 'elliptic-sos' and meta['receive']['order'] == 18
    for name, coefficients in (('image', image), ('receive', receive)):
        assert 'coefficients' not in meta[name]
        assert meta[name]['sha256'] == hashlib.sha256(coefficients.astype('<f8').tobytes()).hexdigest()
    for fading in ('none', 'poor'):
        wire = Channel(Condition('inf', fading), horizon=945)
        assert wire.delay == 2400 and wire.clock.headroom == 0
        assert wire.propagation+480+wire.clock.delay+sum(m['group_delay_samples'] for m in meta.values()) == 2400
        for detail in wire.evidence()['filters'].values():
            assert set(detail) == {'taps', 'group_delay_samples', 'enbw_hz', 'sha256'}
        json.dumps(wire.evidence())
    for ppm in (-50, 50):
        wire = Channel(Condition('inf', clock_ppm=ppm), horizon=1)
        assert wire.delay == wire.window(0, 480, output=True)[0] == 2400
    wire = Channel(Condition('inf', clock_ppm=50), horizon=945)
    assert wire.propagation == 0 and wire.delay > 2400
    assert wire.evidence()['delay_samples'] == wire.delay
    monkeypatch.setattr(h, 'inventory', lambda: [dict(bw=500, level=2, bps=41.1), dict(bw=500, level=1, bps=18.1)])
    transfer, horizon = h.case_horizon(SimpleNamespace(bytes=256, application_gap=0), 500)
    assert transfer == pytest.approx(60+3*8*256/18.1)
    assert horizon == 944.9


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('kind', ['session', 'control', 'ladder'])
def test_peer_free_readable_summary_for_all_session_kinds(tmp_path, kind):
    from channel import Condition
    row = dict(condition=h.asdict(Condition('3')), bw=500, caller='kestrel', seed=42, replicate=0,
               registered=False, outcome='pass', control=kind == 'control', trace='trace.jsonl',
               timing={'a-to-b': dict(seconds=12.5, goodput_bps=163.84)},
               metrics={'a': dict(levels={'tx': dict(highest=4, final=3)}, data_retries=2)})
    cases = [dict(session=row, control=dict(row, control=True), comparison='both-pass')] if kind == 'ladder' else [row]
    thresholds = [dict(bw=500, fading='none', registered=False, rungs=[10, 3],
                       kestrel_threshold=3, xara_threshold=3, verdict='smoke')]
    h.summary(dict(cases=cases, thresholds=thresholds), tmp_path)
    text = (tmp_path/'summary.txt').read_text()
    assert '\t' not in text and '{' not in text
    assert '4/3' in text and '12.50' in text
    if kind != 'control':
        assert '163.84' in text
    assert text.count('a-to-b') == text.count('b-to-a') == 1
    assert 'Xara/Xara outcome' in text and 'Threshold: bw=500 fading=none mode=limited' in text


@pytest.mark.usefixtures('peer_free')
@pytest.mark.parametrize('events,disconnected,recovers_close', [
    ([], True, False),
    (['rx control burst — W1AW answered our poll'], True, False),
    (['rx turn grant — W1AW handed the channel over'], True, False),
    (['rx turn grant — W1AW handed the channel over', 'close keyed to W1AW — session closed'], False, False),
    (['rx turn grant — W1AW handed the channel over', 'close keyed to W1AW — session closed'], True, True),
    (['rx session-disconnect-request-responder — W1AW closed the link'], True, True),
    (["rx W1AW's CW ident — it has closed the link"], True, True),
])
def test_peer_free_ptt_replay_records_close_recovery(monkeypatch, tmp_path, events, disconnected, recovers_close):
    from types import SimpleNamespace
    from channel import Channel
    expected = b'pb'
    np.zeros(960, '<f4').tofile(tmp_path/'replay.f32')
    primary = SimpleNamespace(connected=True, disconnected=False, got=expected,
                              modem=SimpleNamespace(confirmed_total=2))
    class Replay:
        connected, disconnected, got = True, False, expected
        modem = SimpleNamespace(confirmed_total=2)
        arq = SimpleNamespace(time=None)
        t = SimpleNamespace(clock=lambda: 0)
        requested = False
        def __init__(self, args, bw, trace, resources, **kwargs):
            self.trace = trace
        def command(self, command):
            self.requested = command == 'DISCONNECT'
        def step(self, samples, at, **kwargs):
            if self.requested and at >= 480:
                for event in events:
                    self.trace('a', event)
                self.disconnected = disconnected
    monkeypatch.setattr(h, 'Kestrel', Replay)
    result = h.ptt_replay(SimpleNamespace(bytes=2), 500, tmp_path, [(0, 'DISCONNECT')], [], 960,
                             Channel(horizon=1), expected, dict(failed_phase='disconnect'), primary)
    assert result['received_length'] == 2
    assert result['recovered_phases'] == (['disconnect'] if recovers_close else [])
    assert result['outcome'] == ('recovered' if recovers_close else 'incomplete')
    if recovers_close:
        assert result['phases']['disconnect'] == .02


@pytest.mark.usefixtures('peer_free')
def test_peer_free_ptt_replay_disconnect_without_connection(tmp_path):
    from types import SimpleNamespace
    from channel import Channel, Condition
    expected = b'pb'
    np.zeros(960, '<f4').tofile(tmp_path/'replay.f32')
    primary = SimpleNamespace(connected=True, disconnected=False, got=expected,
                              modem=SimpleNamespace(confirmed_total=2))
    args = SimpleNamespace(bytes=2, condition=Condition('inf'), registered=False, compression='OFF')
    row = dict(ok=False, outcome='missing-orderly-close', failed_phase='disconnect')
    row['ptt-replay'] = h.ptt_replay(args, 500, tmp_path, [(0, 'CONNECT'), (480, 'DISCONNECT')],
                                       [], 960, Channel(horizon=1), expected, row, primary)
    replay = row['ptt-replay']['replay']
    assert not replay['connected'] and replay['disconnected']
    assert not replay['received'] and replay['confirmed'] == 0
    assert not replay['close_received']
    assert row['ptt-replay']['recovered_phases'] == []
    assert row['ptt-replay']['outcome'] == 'incomplete'
    assert h.paired(row, dict(ok=True, outcome='pass')) == 'cell-shortfall'
