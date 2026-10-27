# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Resolve the requested Xara ref and build its adapter; default to the tested commit."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess

from pathlib import Path

from paths import (CACHE, HERE, PIN, RECEIPT, RUST_VERSION, TOOLCHAIN,
                   XARA, XARA_URL, build_inputs, git, sha256, validate_checkout)


def fetch_xara(url, requested_ref=PIN):
    """Resolve only on explicit setup; never reset an existing source checkout."""
    override = os.environ.get('HFMODEM_INTEROP_XARA')
    if override:
        supplied = Path(override).expanduser().resolve()
        if supplied.exists() and not (supplied / '.git').exists():
            raise RuntimeError(f'refusing to replace nonempty Xara directory {supplied}')
    # A tiny bare fetch cache can move its FETCH_HEAD without moving any endpoint.
    remote_key = hashlib.sha256(url.encode()).hexdigest()[:16]
    remote = CACHE / 'remotes' / (remote_key + '.git')
    if not remote.exists():
        remote.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '--bare', '--quiet', str(remote)], check=True, timeout=30)
    subprocess.run(['git', '-C', str(remote), 'fetch', '--depth=1', '--no-tags',
                    '--', url, requested_ref], check=True, timeout=600)
    resolved = git(remote, 'rev-parse', 'FETCH_HEAD^{commit}')
    source = (Path(override).expanduser().resolve() if override
              else CACHE / 'checkouts' / resolved)
    if source.exists():
        validate_checkout(source, resolved)
    else:
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-C', str(remote), 'worktree', 'add', '--detach',
                        str(source), resolved], check=True, timeout=120)
        validate_checkout(source, resolved)
    return {'xara_root': str(source), 'xara_commit': resolved,
            'requested_ref': requested_ref, 'source_url': url}


def toolchain_environment():
    cargo_home, rustup_home = TOOLCHAIN / 'cargo', TOOLCHAIN / 'rustup'
    return dict(os.environ, CARGO_HOME=str(cargo_home), RUSTUP_HOME=str(rustup_home),
                PATH=str(cargo_home / 'bin') + os.pathsep + os.environ.get('PATH', ''),
                CARGO_BUILD_JOBS='1', CARGO_TARGET_DIR=str(CACHE / 'target'),
                HFMODEM_INTEROP_XARA=str(XARA))


def prepare_toolchain(env):
    cargo = TOOLCHAIN / 'cargo/bin/cargo'
    if not cargo.is_file():
        hosts = {('Darwin', 'arm64'): 'aarch64-apple-darwin',
                 ('Darwin', 'x86_64'): 'x86_64-apple-darwin',
                 ('Linux', 'x86_64'): 'x86_64-unknown-linux-gnu',
                 ('Linux', 'aarch64'): 'aarch64-unknown-linux-gnu'}
        host = hosts.get((platform.system(), platform.machine()))
        if not host:
            raise RuntimeError('unsupported local Rust toolchain host')
        bootstrap = TOOLCHAIN / 'bootstrap/rustup-init'
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['curl', '-L', '--fail', '--connect-timeout', '30',
                        '--max-time', '300', '--output', str(bootstrap),
                        f'https://static.rust-lang.org/rustup/dist/{host}/rustup-init'],
                       check=True, timeout=330)
        bootstrap.chmod(0o755)
        subprocess.run([str(bootstrap), '-y', '--no-modify-path', '--profile',
                        'minimal', '--default-toolchain', RUST_VERSION],
                       env=env, check=True, timeout=900)
    available = subprocess.run([str(cargo), f'+{RUST_VERSION}', '--version'], env=env,
                               capture_output=True, text=True, timeout=60)
    if available.returncode:
        subprocess.run([str(TOOLCHAIN / 'cargo/bin/rustup'), 'toolchain', 'install',
                        RUST_VERSION, '--profile', 'minimal'], env=env, check=True, timeout=900)
    return cargo


def prepare_manifest():
    adapter = CACHE / 'adapter'
    adapter.mkdir(parents=True, exist_ok=True)
    manifest = (HERE / 'Cargo.toml.in').read_text()
    # JSON's quoted string escaping is also valid inside these TOML basic strings.
    for token, path in (('@HARNESS_ROOT@', HERE), ('@XARA_ROOT@', XARA)):
        manifest = manifest.replace(token, json.dumps(str(path), ensure_ascii=False)[1:-1])
    (adapter / 'Cargo.toml').write_text(manifest)
    shutil.copyfile(HERE / 'Cargo.lock', adapter / 'Cargo.lock')
    return adapter / 'Cargo.toml'


def main(argv=None):
    global XARA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xara-url', default=XARA_URL,
                        help='Git URL or local mirror used to resolve the explicitly selected ref')
    parser.add_argument('--xara-ref', default=PIN,
                        help='tag, full commit hash or remote HEAD (default: the tested commit)')
    args = parser.parse_args(argv)
    try:
        chosen = fetch_xara(args.xara_url, args.xara_ref)
        XARA = Path(chosen['xara_root'])
        env = toolchain_environment()
        cargo = prepare_toolchain(env)
        manifest = prepare_manifest()
        inputs = build_inputs()
        subprocess.run([str(cargo), f'+{RUST_VERSION}', 'build', '--locked', '--release',
                        '--manifest-path', str(manifest)], env=env, check=True, timeout=1800)
        validate_checkout(XARA, chosen['xara_commit'])
        if inputs != build_inputs():
            raise RuntimeError('adapter sources changed during compilation; run setup again')
        probe = CACHE / 'bin' / chosen['xara_commit'] / 'kestrel-xara-probe'
        probe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CACHE / 'target/release/kestrel-xara-probe', probe)
        receipt = dict(chosen, harness_root=str(HERE), rust_version=RUST_VERSION,
                       build_inputs=inputs, probe_path=str(probe), probe_sha256=sha256(probe))
        temporary = RECEIPT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(RECEIPT)
        print(f"Ready: {probe} (requested {args.xara_ref}, resolved {chosen['xara_commit']})")
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'setup failed: {error}')
        return 1
