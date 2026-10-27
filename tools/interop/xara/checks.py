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


def test_bw500_low_renderers_accept_host_levels_and_cross_decode(tmp_path):
    # This catches confusing BW500's host-level API with wideband record keys.
    for level, size in [(1, 12), (2, 25), (3, 37)]:
        body = bytes(range(size-3)) + b'\x82'
        frame = body + h.crc16_genibus(body).to_bytes(2, 'big')
        path = tmp_path / f'l{level}.f64'
        h.channel(h.render([frame], 500, level)).tofile(path)
        assert h.invoke('rx', 500, level, path, 1).strip() == frame.hex()


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
