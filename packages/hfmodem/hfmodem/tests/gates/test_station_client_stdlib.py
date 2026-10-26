# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
from pathlib import Path
import subprocess
import sys

import pytest

STATION = Path(__file__).resolve().parents[2] / 'station'


def import_offences(path):
    offences = []
    tree = ast.parse(path.read_text(), filename=str(path))
    dynamic_names = {'__import__'}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ('importlib', 'builtins'):
            dynamic_names.update(alias.asname or alias.name for alias in node.names
                                 if alias.name in ('import_module', '__import__'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if ((isinstance(function, ast.Name) and function.id in dynamic_names)
                    or (isinstance(function, ast.Attribute)
                        and function.attr in ('import_module', '__import__'))):
                offences.append(f'{path.name}:{node.lineno}: dynamic import')
            continue
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if (path.name == 'client.py' and node.level == 1
                    and node.module is None
                    and all(alias.name == 'wire' for alias in node.names)):
                continue
            if node.level:
                offences.append(f'{path.name}:{node.lineno}: relative import')
                continue
            names = [node.module or '']
            if node.module == 'hfmodem.station':
                names = [node.module + '.' + alias.name for alias in node.names]
        else:
            continue
        for name in names:
            if name.split('.')[0] in sys.stdlib_module_names:
                continue
            if path.name == 'client.py' and name == 'hfmodem.station.wire':
                continue
            offences.append(f'{path.name}:{node.lineno}: {name}')
    return offences


@pytest.mark.parametrize('name', ['client.py', 'wire.py'])
def test_station_client_imports_only_stdlib_and_wire(name):
    assert not import_offences(STATION / name)


@pytest.mark.parametrize('source', [
    'import numpy', 'from hfmodem import station', 'from ..core import config',
    'from . import process', 'import hfmodem.station.process',
    '__import__("numpy")',
    'import importlib; importlib.import_module("numpy")',
    'import importlib as il; il.import_module("numpy")',
    'from importlib import import_module as load; load("numpy")',
    'import builtins; builtins.__import__("numpy")',
    'from builtins import __import__ as load; load("numpy")',
])
def test_static_gate_rejects_nonportable_imports(tmp_path, source):
    path = tmp_path / 'client.py'
    path.write_text(source)
    assert import_offences(path)


def test_client_and_wire_run_by_file_path_without_hfmodem():
    script = '''
import importlib.util
import pathlib
import sys
import tempfile
import types
root = pathlib.Path(sys.argv[1])
package = types.ModuleType('portable_station')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
for name in ('wire', 'client'):
    spec = importlib.util.spec_from_file_location('portable_station.' + name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
wire = sys.modules['portable_station.wire']
client = sys.modules['portable_station.client']
assert wire.DEFAULT_PORT == 8737
with tempfile.TemporaryDirectory() as directory:
    station = pathlib.Path(directory) / 'station.toml'
    station.write_text('[station]' + chr(10))
    assert client.Client(station, port=8737).url == 'http://127.0.0.1:8737'
assert client.Claimed.exit_code == 5
assert not any(name.split('.')[0] in {'hfmodem', 'numpy', 'scipy', 'sounddevice'} for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(STATION)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
