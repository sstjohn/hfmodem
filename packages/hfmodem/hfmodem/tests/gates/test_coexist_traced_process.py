# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The Linux tracing adapter preserves the requested child working directory."""
from types import SimpleNamespace

import pytest

from hfmodem.tests.coexist import observer


@pytest.mark.parametrize('explicit', [False, True])
def test_traced_process_forwards_the_child_working_directory(tmp_path, monkeypatch, explicit):
    calls = []
    monkeypatch.setattr(observer.sys, 'platform', 'linux')

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(observer.subprocess, 'Popen', spawn)
    cwd = tmp_path / 'another-cwd' if explicit else None
    traced = observer.TracedProcess(['program', '../station.toml'], directory=tmp_path / 'trace',
                                    env={'FIXTURE': '1'}, **({'cwd': cwd} if explicit else {}))
    try:
        argv, options = calls[0]
        assert len(calls) == 1
        assert options['cwd'] == cwd
        assert options['env'] == {'FIXTURE': '1'}
        assert '-yy' in argv and argv[-2:] == ['program', '../station.toml']
    finally:
        traced._out.close()
        traced._err.close()


def test_trace_decoder_uses_authoritative_path_after_relative_open(tmp_path):
    target = tmp_path / 'station.toml'
    trace = observer.parse_trace(
        f'11 1.0 openat(AT_FDCWD<{tmp_path}/another-cwd>, "../station.toml", O_RDONLY) '
        f'= 3<{target}>\n11 1.1 +++ exited with 0 +++\n')
    assert trace.records[0]['path'] == str(target)
    with pytest.raises(observer.ObservationError, match='relative open requires'):
        observer.parse_trace('11 1.0 openat(AT_FDCWD, "../station.toml", O_RDONLY) = 3\n')
