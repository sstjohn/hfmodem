# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Witness teardown distinguishes policy absence from unavailable selected audio."""
import os
import shlex
import subprocess
import sys

import pytest

from hfmodem.tests import hermetic
from hfmodem.tests.station.recording_stage5_support import (
    ROOT, STOP, entry, save_selection, wave_file, write_json,
)

pytestmark = pytest.mark.posix_only(
    "witness_stop is tools/onair.sh bash; Windows stops the witness in consolelib.launch_capture")


def stop(root, policy, *, child=False, destination=''):
    source = (ROOT / 'tools/onair.sh').read_text()
    functions = source[source.index('recording_available()'):
                       source.index('\n}', source.index('witness_stop()')) + 2]
    script = ['set -eu', 'REPO=' + shlex.quote(str(ROOT)), 'SESSION=' + shlex.quote(str(root)),
        'RECORDING_DESTINATION=' + shlex.quote(str(destination)),
        'VENV=/unused', 'WITNESS=1', 'WITNESS_PID=', 'WITNESS_WAV="$SESSION/witness.wav"',
        'claim_processes() { :; }',
        'timeout() { printf "WITNESS ALIGN"; printf " <%s>" "$@"; printf "\\n"; }', functions]
    if child:
        script += ['/bin/sh -c "exit 0" &', 'WITNESS_PID=$!', 'wait "$WITNESS_PID"']
    script += ['witness_stop']
    env = os.environ | {'HFMODEM_PYTHON': sys.executable, 'HFMODEM_RECORDING': policy}
    hermetic.guard.admit(['bash', '-c'], env)
    return subprocess.run(['bash', '-c', '\n'.join(script)], capture_output=True, text=True,
        timeout=10, env=env)


@pytest.mark.parametrize('policy,completion,expected', [
    ('off', '', 'intentional absence'),
    ('contacts', 'WITNESS COVERAGE: contact-only; retained 0 samples\n', 'no retained contact'),
    ('contacts', 'WITNESS NONE: capture failed\n', 'unavailable'),
    ('everything', 'WITNESS COVERAGE: everything; retained 16 samples\n', 'possibly moved'),
])
def test_witness_stop_reports_state_without_child_pid(tmp_path, policy, completion, expected):
    (tmp_path / 'witness.log').write_text(completion)
    write_json(tmp_path / 'recording.json', {'policy': policy, 'lanes': {}, 'contacts': {}, 'segments': []})
    result = stop(tmp_path, policy)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize('encoded', [False, True])
def test_contact_only_witness_aligns_only_overlapping_absolute_intervals(tmp_path, encoded):
    selected = entry(tmp_path, protocol='pactor', source='working/run/stream.wav')
    local = save_selection(tmp_path, selected)
    witness = wave_file(tmp_path / 'witness.wav', count=32)
    if encoded:
        from hfmodem.core import wav
        assert 'kept' not in wav.store(witness)
    write_json(tmp_path / 'recording.json', {'policy': 'contacts', 'lanes': {},
        'contacts': {'c2': selected['recording']}, 'segments': []})
    (tmp_path / 'witness.log').write_text('WITNESS COVERAGE: contact-only; retained 32 samples\n')
    write_json(tmp_path / 'witness.json', {'recording': selected['recording'],
        'source_id': 'source-witness', 'clock': 'kiwi', 'rate': 8,
        'actual': [8016, 8048], 'source_anchor': {'clock': 'card', 'sample': 8016, 'position': 8016}})
    result = stop(tmp_path, 'contacts', child=True)
    assert result.returncode == 0, result.stderr
    assert 'ALIGN' in result.stdout, 'contact-only references must align without a session base file'
    assert str(local) in result.stdout
    assert '<--selection>' in result.stdout
    assert str(8016) in result.stdout and str(STOP) in result.stdout


@pytest.mark.parametrize('attempt', ['', 'cap-1-attempt-2'])
def test_contact_only_witness_reads_the_index_at_the_recording_destination(tmp_path, attempt):
    session, destination = tmp_path / 'session', tmp_path / 'dest' / attempt
    selected = entry(destination, protocol='pactor', source='working/run/stream.wav')
    local = save_selection(destination, selected)
    wave_file(session / 'witness.wav', count=32)
    write_json(destination / 'recording.json', {'policy': 'contacts', 'lanes': {},
        'contacts': {'c2': selected['recording']}, 'segments': []})
    (session / 'witness.log').write_text('WITNESS COVERAGE: contact-only; retained 32 samples\n')
    write_json(session / 'witness.json', {'recording': selected['recording'],
        'source_id': 'source-witness', 'clock': 'kiwi', 'rate': 8,
        'actual': [8016, 8048], 'source_anchor': {'clock': 'card', 'sample': 8016, 'position': 8016}})
    result = stop(session, 'contacts', child=True, destination=destination)
    assert result.returncode == 0, result.stderr
    assert '<--selection>' in result.stdout, result.stdout
    assert str(local) in result.stdout and str(session / 'witness.wav') in result.stdout


def test_moved_contact_only_witness_is_unavailable_instead_of_retained_success(tmp_path):
    (tmp_path / 'witness.log').write_text('WITNESS COVERAGE: contact-only; retained 32 samples\n')
    write_json(tmp_path / 'recording.json', {'policy': 'contacts', 'lanes': {
        'witness': {'source': 'missing.wav', 'state': 'finalized'}},
        'contacts': {'c2': {'lanes': {'witness': {'source': 'missing.wav', 'actual': [0, 32]}}}},
        'segments': [{'source': 'missing.wav', 'actual': [0, 32], 'state': 'finalized'}]})
    result = stop(tmp_path, 'contacts', child=True)
    assert result.returncode == 0, result.stderr
    assert 'unavailable here (possibly moved)' in result.stdout


@pytest.mark.parametrize('line', ['WITNESS NONE: candidate inventory absent', 'WITNESS SHORT: receiver unavailable'])
def test_prepared_capture_witness_failure_reaches_the_arm_transcript(tmp_path, monkeypatch, line):
    from contextlib import redirect_stdout
    import io
    from hfmodem.tests.station.recording_stage5_support import readers
    readers()
    import capture_identity as helper
    data, session, handed = tmp_path / 'data', tmp_path / 'session', {}

    def prepared(script, words, environment):
        with monkeypatch.context() as child:
            for name in ('HFMODEM_CAPTURE_FD', 'HFMODEM_CAPTURE_METADATA_FD', 'HFMODEM_CAPTURE_WITNESS_LOG_FD'):
                child.setenv(name, environment[name])
            child.setattr(sys, 'argv', ['capture_identity.py', 'prepared', '--repo', str(data), '--', *words[2:]])
            output = io.StringIO()
            with redirect_stdout(output):
                helper._main()
        handed.update(word.split('=', 1) for row in output.getvalue().splitlines() for word in shlex.split(row))
        os.write(int(environment['HFMODEM_CAPTURE_WITNESS_LOG_FD']), (line + '\n').encode())

    helper.launch(data, ROOT / 'tools/onair.sh',
                  ['pactor', 'K9ZZT', '7105000', '--outdir', str(session), '--run-dir', str(session)],
                  exec_fn=prepared)
    assert handed['SESSION'] == str(session) and not (session / 'witness.log').exists()
    if 'WITNESS_LOG' in handed:
        monkeypatch.setenv('WITNESS_LOG', handed['WITNESS_LOG'])
    result = stop(session, 'everything', child=True)
    assert result.returncode == 0, result.stderr
    assert '  ' + line in result.stdout, result.stdout
    assert '!! NO WITNESS for this arm' in result.stdout, result.stdout
    assert f"{handed['WITNESS_LOG']} says why" in result.stdout
