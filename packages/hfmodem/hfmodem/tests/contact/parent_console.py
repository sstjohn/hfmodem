# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import ast
from hfmodem.tests.evidence import require_input
from functools import cache
from pathlib import Path


@cache
def source():
    return require_input(Path(__file__).parent / 'fixtures/main_varac_chat.py.txt').read_text(encoding='utf-8')


def symbols(namespace):
    tree = ast.parse(source())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ('Terminal', 'WindowsTerminal', 'terminal', 'logging_say')]
    result = dict(namespace)
    exec(compile(tree, 'main_varac_chat.py', 'exec'), result)
    return result
