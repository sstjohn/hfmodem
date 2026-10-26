# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Keep the explicit qualification phase through the row shell's reset."""
import os
import subprocess

import pytest

from hfmodem.tests.coexist import test_rows_admission as rows


@pytest.mark.parametrize('phase', ['coexist', 'inventory'])
def test_row_shell_restores_its_explicit_phase(tmp_path, monkeypatch, phase):
    tools = tmp_path / 'tools/tests'
    tools.mkdir(parents=True)
    (tools / 'hermetic.sh').write_text(
        'unset HFMODEM_QUALIFICATION_PHASE HFMODEM_QUALIFICATION_PHASE_SECRET ROW_SECRET\n')
    monkeypatch.setattr(rows, 'TREE', tmp_path)
    monkeypatch.setenv('HFMODEM_QUALIFICATION_PHASE', phase)
    monkeypatch.setenv('HFMODEM_QUALIFICATION_PHASE_SECRET', 'phase-prefix-secret')
    monkeypatch.setenv('ROW_SECRET', 'inherited-secret')
    env = dict(os.environ, HAMLIB_BIN=str(tmp_path / 'bin'))
    command = rows.shell_argv(['/bin/bash', '-c', 'env -0'], env)
    result = subprocess.run(command, env=env, capture_output=True, check=True)
    actual = dict(item.split(b'=', 1) for item in result.stdout.split(b'\0') if item)
    assert actual[b'HFMODEM_QUALIFICATION_PHASE'] == phase.encode()
    assert b'HFMODEM_QUALIFICATION_PHASE_SECRET' not in actual
    assert b'ROW_SECRET' not in actual
