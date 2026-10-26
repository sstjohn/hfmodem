# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Kernel pipe targets remain distinct from observed filesystem device opens."""
import pytest

from hfmodem.tests.coexist.observer import ObservationError, parse_trace
from hfmodem.tests.coexist.oracle import Evidence, OracleFailure, validate


# Exact failing line retained from inherited-launch case 6, trace-0/trace.
# Trace SHA-256: 318108f2eb192de19f3c58885edf2253075324b49f913d0c1cecf575f866cb83
PIPE_OPEN = ('234629 1793036507.475859 '
             'openat(AT_FDCWD</tmp/hfmodem-observer-focused-45ab/source>, '
             '"/dev/fd/63", O_RDONLY) = 3<pipe:[688034]>\n')


@pytest.mark.parametrize('call', [
    'open("/dev/fd/63", O_RDONLY)',
    'openat(AT_FDCWD</tmp/source>, "/dev/fd/63", O_RDONLY)',
    'openat2(AT_FDCWD</tmp/source>, "/dev/fd/63", {flags=O_RDONLY, resolve=0}, 24)',
    'creat("/dev/fd/63", 0600)',
])
def test_pipe_annotations_retain_the_successful_syscall_without_inventing_a_path(call):
    trace = parse_trace(f'11 1.0 {call} = 3<pipe:[688034]>\n11 1.1 +++ exited with 0 +++\n')
    assert trace.pids == trace.exited == {11}
    assert len(trace.records) == 1
    record = trace.records[0]
    assert record['pid'] == 11 and record['value'] == 3
    assert record['syscall'] == call.split('(', 1)[0]
    assert record['args'] == call.split('(', 1)[1][:-1]
    assert record['result'] == '3<pipe:[688034]>'
    assert record['fd_target'] == 'pipe:[688034]'
    assert 'path' not in record


def test_exact_retained_pipe_line_is_not_an_unresolved_relative_open():
    trace = parse_trace(PIPE_OPEN + '234629 1793036507.5 +++ exited with 0 +++\n')
    assert trace.records[0]['fd_target'] == 'pipe:[688034]'
    assert trace.pids == trace.exited == {234629}


def test_resumed_pipe_open_keeps_its_original_start_and_process_identity():
    trace = parse_trace('11 1.0 openat(AT_FDCWD, "/dev/fd/63", O_RDONLY <unfinished ...>\n'
                        '12 1.1 close(4) = 0\n'
                        '11 1.2 <... openat resumed>) = 3<pipe:[688034]>\n'
                        '11 1.3 +++ exited with 0 +++\n12 1.4 +++ exited with 0 +++\n')
    pipe = next(record for record in trace.records if record['syscall'] == 'openat')
    assert pipe['pid'] == 11 and pipe['time_ns'] == 1_000_000_000
    assert pipe['fd_target'] == 'pipe:[688034]' and 'path' not in pipe
    assert trace.pids == trace.exited == {11, 12}


@pytest.mark.parametrize('target', ['relative/file', 'pipe:[]', 'pipe:[abc]',
                                    'pipe:[123]/file', 'pipe:[123]extra',
                                    'pipe:[１２３]', 'socket:[123]', 'anon_inode:[eventpoll]'])
def test_unknown_or_malformed_nonabsolute_targets_stay_fail_closed(target):
    with pytest.raises(ObservationError, match='relative open requires'):
        parse_trace(f'11 1.0 openat(AT_FDCWD, "/dev/fd/63", O_RDONLY) = 3<{target}>\n')


def test_unannotated_relative_filesystem_open_stays_fail_closed():
    with pytest.raises(ObservationError, match='relative open requires'):
        parse_trace('11 1.0 openat(AT_FDCWD, "relative/file", O_RDONLY) = 3\n')


def test_filesystem_named_like_a_pipe_keeps_its_authoritative_absolute_path():
    path = '/tmp/pipe:[688034]'
    trace = parse_trace(f'11 1.0 openat(AT_FDCWD, "pipe:[688034]", O_RDONLY) = 3<{path}>\n')
    assert trace.records[0]['path'] == path
    assert 'fd_target' not in trace.records[0]


def test_fd_alias_to_watched_device_remains_an_oracle_checked_filesystem_open(tmp_path):
    watched = tmp_path / 'fake-audio'
    trace = parse_trace(PIPE_OPEN +
                        f'12 1793036507.6 openat(AT_FDCWD, "/dev/fd/62", O_RDONLY) = 4<{watched}>\n'
                        '234629 1793036507.7 +++ exited with 0 +++\n'
                        '12 1793036507.8 +++ exited with 0 +++\n')
    assert trace.records[1]['path'] == str(watched)
    assert len(trace.records) == 2
    proof = Evidence('run', {str(watched)}, 'linux', trace=trace,
                     tracer_returncode=0, tree_finished=True)
    with pytest.raises(OracleFailure, match='unlogged traced opener: 12'):
        validate(proof)
    proof.events.append(dict(event='claim_acquired', run='run', pid=12))
    assert validate(proof) == []


def test_a_pipe_record_never_masks_an_incomplete_process_tree(tmp_path):
    trace = parse_trace(PIPE_OPEN)
    proof = Evidence('run', {str(tmp_path / 'fake-audio')}, 'linux', trace=trace,
                     tracer_returncode=0, tree_finished=True)
    with pytest.raises(OracleFailure, match='trace ends before its tree'):
        validate(proof)
