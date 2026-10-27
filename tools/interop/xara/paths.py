# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Paths and endpoint identity shared by the explicitly invoked interop tools."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PIN = '2e96fcd1563c765a0f5731dae9cc9465c4b86ac6'
XARA_URL = 'https://github.com/x0rro/xara'
RUST_VERSION = '1.90.0'


def configured_path(variable, default):
    return Path(os.environ.get(variable, default)).expanduser().resolve()


def default_cache(root):
    requested = os.environ.get('XDG_CACHE_HOME')
    base = Path(requested).expanduser() if requested else Path.home() / '.cache'
    if not base.is_absolute():
        base = Path.home() / '.cache'
    namespace = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    return base / 'hfmodem' / 'xara-interop' / namespace


ROOT = configured_path('HFMODEM_INTEROP_ROOT', REPOSITORY)
CACHE = configured_path('HFMODEM_INTEROP_CACHE', default_cache(ROOT))
TOOLCHAIN = configured_path('HFMODEM_INTEROP_TOOLCHAIN', CACHE / 'toolchain')
RECEIPT = CACHE / 'build.json'


def selection():
    try:
        value = json.loads(RECEIPT.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


XARA = configured_path('HFMODEM_INTEROP_XARA', selection().get('xara_root', CACHE / 'xara'))
PROBE = Path(selection().get('probe_path', CACHE / 'target/release/kestrel-xara-probe')).resolve()


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_metadata(root):
    """Git identity belongs to this exact tree, never an enclosing checkout."""
    try:
        top = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if Path(top).resolve() != Path(root).resolve():
        return None
    return {'commit': git(root, 'rev-parse', 'HEAD'),
            'tracked_changes': git(root, 'diff', 'HEAD', '--name-only'),
            'status': git(root, 'status', '--porcelain')}


def source_identity(root):
    """Hash runtime packages and their metadata, not a whole release archive."""
    root = Path(root)
    versions, files = {}, {}
    for name in ('hfmodem', 'hfhost', 'creance'):
        project = root / 'packages' / name
        metadata = project / 'pyproject.toml'
        try:
            versions[name] = tomllib.loads(metadata.read_text())['project']['version']
            if not isinstance(versions[name], str):
                raise ValueError('package version must be a string')
            module = ast.parse((project / name / '__init__.py').read_text())
            declared = [ast.literal_eval(node.value) for node in module.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == '__version__'
                            for target in node.targets)]
        except (OSError, ValueError, KeyError, SyntaxError) as error:
            raise RuntimeError(f'missing or invalid source package metadata: {metadata}') from error
        if declared != [versions[name]]:
            raise RuntimeError(f'{name} runtime version disagrees with source package metadata')
        files[metadata.relative_to(root).as_posix()] = sha256(metadata)
        for directory, subdirs, names in os.walk(project / name):
            subdirs[:] = sorted(d for d in subdirs if d not in ('tests', '__pycache__', '.pytest_cache'))
            for filename in sorted(names):
                path = Path(directory) / filename
                if path.is_file() and path.suffix not in ('.pyc', '.pyo'):
                    files[path.relative_to(root).as_posix()] = sha256(path)
    if len(set(versions.values())) != 1:
        raise RuntimeError(f'source package versions disagree: {versions}')
    content = {'package_versions': versions, 'files': files}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(content, schema_version=1, sha256=digest,
                scope='three runtime package trees and their pyproject.toml files; '
                      'excludes tests, __pycache__, .pytest_cache, pyc and pyo; '
                      'harness sources are recorded separately in harness_sha256')


def build_inputs():
    return {name: sha256(HERE / name) for name in
            ('Cargo.toml.in', 'Cargo.lock', 'build.rs', 'src/main.rs')}


def validate_kestrel_imports():
    packages = ('hfmodem', 'hfhost', 'creance')
    for name in packages:
        source = ROOT / 'packages' / name
        if not (source / name / '__init__.py').is_file():
            raise RuntimeError(f'selected Kestrel root {ROOT} is missing packages/{name}; '
                               'refusing an editable-install fallback')
        sys.path.insert(0, str(source))
    for name in packages:
        module = importlib.import_module(name)
        expected = (ROOT / 'packages' / name).resolve()
        if not Path(module.__file__).resolve().is_relative_to(expected):
            raise RuntimeError(f'{name} was already imported outside selected Kestrel root {ROOT}; '
                               'start a fresh interpreter with HFMODEM_INTEROP_ROOT set')


def validate_checkout(path, expected):
    try:
        revision = git(path, 'rev-parse', 'HEAD')
        dirty = git(path, 'status', '--porcelain')
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Xara is missing; run the explicit setup command first') from error
    if revision != expected or dirty:
        raise RuntimeError(f'Xara must be clean at selected commit {expected}; refusing to reset {path}')
    return revision


def validate_xara():
    return validate_checkout(XARA, selection().get('xara_commit', PIN))


def validate_probe():
    resolved = validate_xara()
    try:
        receipt = json.loads(RECEIPT.read_text())
        valid = (receipt['harness_root'] == str(HERE)
                 and receipt['xara_root'] == str(XARA)
                 and receipt['xara_commit'] == resolved
                 and receipt['requested_ref'] and receipt['source_url']
                 and receipt['probe_path'] == str(PROBE)
                 and receipt['rust_version'] == RUST_VERSION
                 and receipt['build_inputs'] == build_inputs()
                 and receipt['probe_sha256'] == sha256(PROBE))
    except (OSError, ValueError, KeyError):
        valid = False
    if not valid:
        raise RuntimeError('Xara adapter is absent or stale; run the explicit setup command first')
