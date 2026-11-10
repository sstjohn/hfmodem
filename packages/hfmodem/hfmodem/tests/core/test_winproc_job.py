# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hfmodem.core import winproc


def test_job_contains_children_and_kills_them_on_close(monkeypatch):
    api = SimpleNamespace(CreateJobObjectW=Mock(return_value=42),
        SetInformationJobObject=Mock(return_value=True),
        AssignProcessToJobObject=Mock(return_value=True),
        TerminateJobObject=Mock(return_value=True))
    monkeypatch.setattr(winproc, '_api', lambda: api)
    close = Mock()
    monkeypatch.setattr(winproc.winclaim, 'close_handle', close)
    job = winproc.Job()
    args = api.SetInformationJobObject.call_args.args
    assert args[:2] == (42, 9)
    assert args[2]._obj.basic.flags == 0x2000
    job.assign(SimpleNamespace(_handle=17))
    api.AssignProcessToJobObject.assert_called_once_with(42, 17)
    job.terminate()
    api.TerminateJobObject.assert_called_once_with(42, 1)
    job.close()
    job.close()
    close.assert_called_once_with(42)


@pytest.mark.parametrize('boundary', ['create', 'limits', 'assign', 'terminate'])
def test_job_reports_native_failures_and_closes_owned_handle(monkeypatch, boundary):
    api = SimpleNamespace(CreateJobObjectW=Mock(return_value=42 if boundary != 'create' else 0),
        SetInformationJobObject=Mock(return_value=boundary != 'limits'),
        AssignProcessToJobObject=Mock(return_value=boundary != 'assign'),
        TerminateJobObject=Mock(return_value=boundary != 'terminate'))
    monkeypatch.setattr(winproc, '_api', lambda: api)
    monkeypatch.setattr(winproc.winclaim, '_error', lambda message: OSError(message))
    close = Mock()
    monkeypatch.setattr(winproc.winclaim, 'close_handle', close)
    job = None
    try:
        with pytest.raises(OSError):
            job = winproc.Job()
            job.assign(SimpleNamespace(_handle=17))
            job.terminate()
    finally:
        if job is not None:
            job.close()
    assert close.call_count == int(boundary != 'create')
