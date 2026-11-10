# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
from types import SimpleNamespace as NS

from hfmodem.tests.coexist.test_rows_cleanup import _console_source


def test_console_preparation_probe_checks_the_forwarded_descriptor(tmp_path):
    source = _console_source(NS(root=tmp_path), 'preparation')
    preparation, = [node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'preparation']
    events, descriptor = [], object()
    namespace = dict(claim=NS(inherited=lambda fd: events.append(fd)),
                     coexist=NS(checkpoint=events.append), point='preparation')
    exec(compile(ast.Module(body=[preparation], type_ignores=[]), '<preparation probe>', 'exec'), namespace)
    namespace['preparation']({}, descriptor=descriptor)
    assert events == [descriptor, 'console_point']


def test_console_spawn_probe_keeps_the_pinned_child_entry(tmp_path):
    source = _console_source(NS(root=tmp_path), 'spawn')
    remembered, = [node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'remembered']
    process, entry, result = NS(args=['fake child']), object(), object()
    calls = []
    def remember(child, *, entry):
        calls.append((child, entry))
        return result
    namespace = dict(remember=remember, point='readback')
    exec(compile(ast.Module(body=[remembered], type_ignores=[]), '<spawn probe>', 'exec'), namespace)
    assert namespace['remembered'](process, entry=entry) is result
    assert calls == [(process, entry)]


def test_console_readback_probe_accepts_the_lease_descriptor(tmp_path):
    source = _console_source(NS(root=tmp_path), 'preparation')
    readback, = [node.value for node in ast.parse(source).body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
                     and target.attr == 'check_rig' for target in node.targets)]
    check = eval(compile(ast.Expression(body=readback), '<readback probe>', 'eval'))
    assert check({}, descriptor=object()) == {}
