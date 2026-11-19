# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Offline installed-wheel smoke for chat assets and optional Textual imports."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

import pytest

from hfmodem.tests import hermetic

REPO = Path(__file__).resolve().parents[5]
pytestmark = pytest.mark.wheel


def test_installed_chat_wheel_without_textual(tmp_path):
    if subprocess.run([sys._base_executable, '-c', 'import setuptools'], capture_output=True).returncode:
        pytest.skip(f'{sys._base_executable} has no setuptools to build the wheel with')
    source = tmp_path / 'source'
    package = source / 'packages/hfmodem'
    shutil.copytree(REPO / 'packages/hfmodem', package,
                    ignore=shutil.ignore_patterns('tests', 'build', '*.egg-info', '__pycache__'))
    inventory = REPO / 'publish/operator-tools.toml'
    (source / 'publish').mkdir()
    shutil.copy2(inventory, source / 'publish/operator-tools.toml')
    for name in tomllib.loads(inventory.read_text())['operators']['paths']:
        if Path(name).name.startswith('test_') or name.endswith('.md'):
            continue
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    output = tmp_path / 'wheels'
    # Use locally installed build tools, with network and build isolation
    # disabled. The source copy keeps build artefacts out of the worktree.
    # The chat assets are the subject, so the build takes setup.py's Python fallback
    # for the P4 LONG kernel: no cargo on the child's PATH.
    path = os.pathsep.join(d for d in os.environ.get('PATH', '').split(os.pathsep)
                           if not shutil.which('cargo', path=d))
    env = dict(os.environ, PATH=path, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    # A native run's HFMODEM_NATIVE=1 would pull in setuptools-rust, which the base
    # interpreter need not have.
    env.pop('HFMODEM_NATIVE', None)
    # setuptools-rust, loaded by every setuptools build, probes for a Rust toolchain.
    hermetic.guard.admit(['rustc'], env)
    result = subprocess.run([sys._base_executable, '-m', 'pip', 'wheel', '--no-index',
                             '--no-deps', '--no-build-isolation', '-w', str(output), str(package)],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    wheels = list(output.glob('hfmodem-*.whl'))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert wheel.read('hfmodem/ui/assets/station_chat.mjs') == (
            REPO / 'packages/hfmodem/hfmodem/ui/assets/station_chat.mjs').read_bytes()
        assert wheel.read('hfmodem/ui/assets/hfdesk_chat.mjs')
        assert wheel.read('hfmodem/_operator/tools/logbook_view/chat_mock.mjs')
        metadata = wheel.read(next(name for name in wheel.namelist() if name.endswith('.dist-info/METADATA'))).decode()
        assert 'Provides-Extra: tui' in metadata
        assert any('textual' in line and 'extra == "tui"' in line for line in metadata.splitlines())
    installed = tmp_path / 'installed'
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                             '--no-compile', '--target', str(installed), str(wheels[0])],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    script = r'''
import importlib
import importlib.abc
import importlib.resources
from pathlib import Path
import socket
import sys
sys.path.insert(0, sys.argv[1])
class NoTextual(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'textual' or fullname.startswith('textual.'):
            raise AssertionError('line console imported optional Textual')
sys.meta_path.insert(0, NoTextual())
try:
    importlib.import_module('textual')
except AssertionError:
    pass
else:
    raise SystemExit('the Textual block does not hold')
def forbidden(*args, **kwargs):
    raise AssertionError('network factory invoked')
class NoSocket(socket.socket):
    def __new__(cls, *args, **kwargs):
        return forbidden(*args, **kwargs)
socket.socket = NoSocket
socket.create_connection = forbidden
import hfmodem
assert Path(hfmodem.__file__).is_relative_to(Path(sys.argv[1]))
from hfmodem.contact.chat_session import SessionAdapter
from hfmodem.station.client import Client
from hfmodem.station.service import Service
for member in (SessionAdapter, Client, Service):
    assert Path(sys.modules[member.__module__].__file__).is_relative_to(Path(sys.argv[1]))
sys.path.insert(0, str(Path(sys.argv[1]) / 'hfmodem/_operator/tools'))
import varac_chat
assert Path(varac_chat.__file__).is_relative_to(Path(sys.argv[1]))
assert varac_chat.parser() is not None
assert importlib.resources.files('hfmodem.ui').joinpath('assets/station_chat.mjs').read_bytes()
assert not any(name == 'textual' or name.startswith('textual.') for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script, str(installed)],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
