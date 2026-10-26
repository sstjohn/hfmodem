# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Test-only import audit, lifecycle serialization and retained closure capture."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import FunctionType

EVENTS = []

def record(label, **values):
    EVENTS.append(dict(label=label, **values))

def freeze_guard(fn):
    values = tuple(c.cell_contents for c in fn.__closure__ or ())
    def cell(value):
        return (lambda: value).__closure__[0]
    frozen = FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__,
                          tuple(cell(v) for v in values) or None)
    frozen.__kwdefaults__ = fn.__kwdefaults__
    assert frozen.__code__ is fn.__code__ and frozen.__globals__ is fn.__globals__
    assert frozen.__defaults__ is fn.__defaults__
    assert all(a is b.cell_contents for a, b in zip(values, frozen.__closure__ or ()))
    return frozen, dict(zip(fn.__code__.co_freevars, values))

def imports():
    tree = Path(os.environ['CONDUCTOR_TEST_TREE']).resolve()
    for module in ('p4receiver', 'p4observation', 'p4long', 'p4chirprx', 'p4robustrx', 'p4sf8rx'):
        importlib.import_module('hfmodem.shrike.' + module)
    paths = {}
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, '__file__', None)
        if path and name.split('.')[0] in ('hfmodem','hfhost','creance'):
            p = Path(path).resolve()
            if os.environ.get('SECOND_DELTA_MAIN_REPLAY') and name.endswith('test_p4_bare_y27_0929'):
                continue
            assert p.is_relative_to(tree), (name, p, tree)
            paths[name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return paths

def pytest_sessionstart(session):
    record('imports_before', paths=imports(), priority=os.getpriority(os.PRIO_PROCESS, 0))

def pytest_sessionfinish(session, exitstatus):
    record('imports_after', paths=imports(), exit=int(exitstatus))
    path = Path(os.environ['SECOND_DELTA_EVIDENCE'])
    assert not path.exists()
    path.write_text(json.dumps(EVENTS, indent=2, default=str)+'\n')
