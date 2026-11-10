# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from .oracle import OracleFailure, pulse_upper_bound


pytestmark = pytest.mark.hardware
OPEN_PATHS = ('acquisition', 'drop_rts', 'line_keyer_arm', 'cat_one_shot',
              'console_readback', 'down_cat', 'daemon_start', 'yaesu_open',
              'interactive_open', 'interactive_reopen')
PULSE_CASES = ('measured', 'near_limit', 'subsample_phase_0', 'subsample_phase_1',
               'subsample_phase_2', 'delayed', 'filtered', 'lowered_bound', 'slow_witness')
NODE_CASES = ('failed_close', 'unplug_replug', 'cu_tty', 'by_id', 'second_opens', 'permission')


@pytest.fixture
def attended():
    name = os.environ.get('HFMODEM_COEXIST_HW_RECEIPT')
    if not name:
        pytest.skip('attended hardware receipt required; no radio or adapter is opened by these tests')
    path = Path(name)
    receipt = json.loads(path.read_text())
    assert receipt['attended'] is True and receipt['schema_version'] == 1
    for name, record in receipt['artifacts'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        artifact = path.parent / relative
        assert not artifact.is_symlink() and artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == record['sha256']
    runs = [run for run in receipt['runs'] if run['platform'] == sys.platform]
    assert runs, f'no attended measurements for {sys.platform}'
    return runs


@pytest.mark.parametrize('path', OPEN_PATHS)
@pytest.mark.parametrize('case', PULSE_CASES)
def test_initial_open_flash(attended, path, case):
    for run in attended:
        record = run['pulse_fixtures'][path][case]
        assert record['baseline_low'] is True and record['wiring_high'] is True
        options = {key: record.get(key) for key in (
            'limit_ns', 'asymmetry_ns', 'blindness_ns', 'reporting_delay_ns', 'shortest_pulse_ns')}
        if record['status'] == 'unresolved':
            assert case != 'measured'
            with pytest.raises(OracleFailure):
                pulse_upper_bound(record['samples'], **options)
        else:
            assert record['status'] == 'passed'
            assert pulse_upper_bound(record['samples'], **options) == record['upper_bound_ns']


@pytest.mark.parametrize('case', NODE_CASES)
def test_real_node_lifetime(attended, case):
    for run in attended:
        record = run['node_fixtures'][case]
        if case == 'cu_tty' and sys.platform != 'darwin' or case == 'by_id' and sys.platform != 'linux':
            assert record == {'status': 'not-applicable', 'platform': sys.platform}
            continue
        assert record['status'] == 'passed'
        assert record['attempts']
        if case == 'failed_close':
            assert record['close'] == 'failed'
            assert record['node_free'] is True and record['line'] is False
        elif case == 'unplug_replug':
            assert record['before'] == record['after']
        elif case in ('cu_tty', 'by_id', 'permission'):
            assert record['reported'] is True and record['opened_over'] is False
        else:
            assert all(attempt['errno'] == 'EBUSY' for attempt in record['attempts'])


def test_real_adapter_subset(attended):
    for run in attended:
        assert all(run[key] for key in ('host', 'adapter', 'rig', 'date', 'rigctl_version',
                                       'configuration', 'edge_reference'))
        measurements = run['measurements']
        assert {'acquire', 'exclusive', 'release', 'replug', 'discovery_permission'} <= measurements.keys()
        assert all(item['status'] == 'passed' and item['evidence'] for item in measurements.values())
        assert measurements['acquire']['line'] is False
        assert measurements['exclusive']['errno'] == 'EBUSY'
        assert measurements['release']['line'] is False and measurements['release']['node_free'] is True
