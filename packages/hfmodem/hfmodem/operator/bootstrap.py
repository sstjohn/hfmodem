# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Run the same reviewed console from a checkout or an installed wheel."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile


def tools_root():
    package = Path(__file__).resolve().parents[1]
    bundled = package / '_operator'
    checkout = package.parents[2]
    if not bundled.is_dir():
        if (checkout / 'tools/console.py').is_file():
            return checkout, False
        raise RuntimeError('Operator resources are missing; reinstall hfmodem.')
    files = sorted(p for p in bundled.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(bundled).as_posix().encode())
        digest.update(path.read_bytes())
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'hfmodem'
    root = cache / ('operator-' + digest.hexdigest()[:16])
    if not (root / '.complete').is_file():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='operator-', dir=cache))
        try:
            for source in files:
                target = temporary / source.relative_to(bundled)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if source.suffix == '.sh' or source.name == 'console':
                    target.chmod(0o700)
            (temporary / '.complete').write_text(digest.hexdigest())
            try:
                temporary.rename(root)
            except OSError:
                if not (root / '.complete').is_file():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return root, True


def installed_data():
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'hfmodem'


def environment(root, installed):
    os.environ['HFMODEM_PYTHON'] = sys.executable
    if installed:
        os.environ['HFMODEM_OPERATOR_INSTALLED'] = '1'
        config = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'hfmodem'
        os.environ.setdefault('HFMODEM_DATA_DIR', str(installed_data()))
        os.environ.setdefault('HFMODEM_STATION', str(config / 'station.toml'))
    elif (root / 'station.toml').is_file():
        os.environ.setdefault('HFMODEM_STATION', str(root / 'station.toml'))
    tools = str(root / 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    os.environ['PYTHONPATH'] = os.pathsep.join([tools, *filter(None, [os.environ.get('PYTHONPATH')])])


def console(argv=None):
    root, installed = tools_root()
    environment(root, installed)
    try:
        spec = importlib.util.spec_from_file_location('hfmodem_operator_console', root / 'tools/console.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main(argv)
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def winlink(argv=None):
    root, installed = tools_root()
    environment(root, installed)
    from consolelib.onboarding import winlink as guided
    try:
        return guided(argv)
    except KeyboardInterrupt:
        print('Stopped.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def face():
    try:
        root, installed = tools_root()
    except RuntimeError:
        return None
    spec = importlib.util.spec_from_file_location('desk.face', root / 'tools/desk/face.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = installed_data() if installed else root / 'operator-data'
    module.data_root = Path(os.environ.get('HFMODEM_DATA_DIR', data)).expanduser().resolve()
    return module
