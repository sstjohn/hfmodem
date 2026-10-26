# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pure console planning only: never invoke fly, a launcher or a device hook."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.core import capture_identity as ci

pytestmark = pytest.mark.usefixtures('tools')


@pytest.fixture(autouse=True)
def explicit_fake_station(tmp_path, monkeypatch):
    config = tmp_path / 'station.toml'
    config.write_text('''schema=1
[station]
mycall="N0CALL"
grid="EM12jt"
control="local"
regulatory="unregulated"
because="offline console lifecycle test with fake launchers"
transmit=true
[rig]
model="ft891"
''')
    monkeypatch.setenv('HFMODEM_STATION', str(config))
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path / 'data'))


def plan(tmp_path):
    directory = tmp_path / 'arms' / 'pactor-planned'
    mail_path = str(directory / 'outgoing.b2f')
    return {'arm': 'pactor-planned', 'directory': str(directory), 'pounce_s': 0,
            'command': ['not-a-real-launcher', 'pactor', '--log', 'pactor-planned',
                        '--mail-send', mail_path],
            'delivery': {'path': mail_path, 'sha256': 'fixed-synthetic-hash'},
            'source': {'synthetic': True}, 'git': {'head': 'unknown', 'dirty': False}}


def test_console_plan_binds_three_roots_and_rebinds_only_delivery_path(tmp_path, monkeypatch):
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    old = plan(tmp_path)
    preserved = deepcopy(old)
    new, capture = arm.capture_plan(SimpleNamespace(name='pactor'), old)
    assert old == preserved
    assert new['requested_arm'] == old['arm']
    assert new['arm'].endswith('--' + capture.capture_id)
    assert new['directory'] == str(capture.root('native'))
    assert new['capture_root'] == str(capture.root('capture'))
    assert str(capture.root('transcript')) in new['transcript']
    assert new['delivery']['sha256'] == old['delivery']['sha256']
    assert Path(new['delivery']['path']).parent == capture.root('native')
    assert new['delivery']['path'] in new['command']
    assert old['delivery']['path'] not in new['command']
    assert new['command'][-4:] == ['--capture-id', capture.capture_id,
                                  '--capture-root', str(capture.root('capture'))]
    for role in capture.record['roots']:
        assert ci.load(capture.root(role)).capture_id == capture.capture_id


@pytest.mark.parametrize('extra', [['--outdir', 'foreign'],
                                  ['--outdir=foreign'],
                                  ['--outdir=earlier', '--outdir=foreign']])
def test_console_conflicting_output_refuses_before_claim_or_producer(tmp_path, monkeypatch, extra):
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    old = plan(tmp_path)
    old['command'] += extra
    with pytest.raises(ci.IdentityError):
        arm.capture_plan(SimpleNamespace(name='pactor'), old)
    assert not (tmp_path / 'captures').exists()
    assert not (tmp_path / 'working').exists()
    assert not (tmp_path / 'arms').exists()


def test_other_modem_plan_unchanged_and_no_identity_allocated(tmp_path):
    from consolelib import arm
    old = plan(tmp_path)
    new, capture = arm.capture_plan(SimpleNamespace(name='vara'), old)
    assert new is old and capture is None
    assert not Path(old['directory']).exists()


def test_pounce_plan_is_group_and_not_an_arm(tmp_path, monkeypatch):
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    old = plan(tmp_path)
    old['pounce_s'] = 10
    _, capture = arm.capture_plan(SimpleNamespace(name='pactor'), old)
    assert capture.kind == 'launch-group'
    assert 'modem' not in capture.record['writers']
    assert ci.provenance(capture.root('capture'))['kind'] == 'launch-group'


def _fake_console_environment(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    prepared = plan(tmp_path)
    prepared.update(delivery=None, env={}, band_m=40, call='synthetic',
                    codec_gain_working_point={},
                    adjustments={'kiwi': False, 'tap_tail_s': 0})
    resided = {'modules': 'synthetic-ready', 'refusal': ''}
    monkeypatch.setattr(arm, 'qualified', lambda mode, plan, qualify: (plan, resided))
    monkeypatch.setattr(arm.claim, 'lease', lambda mode, *, env=None: nullcontext())
    monkeypatch.setattr(arm, 'ready', lambda mode, plan: nullcontext())
    monkeypatch.setattr(arm.claim, 'spacing_note', lambda *a: None)
    monkeypatch.setattr(arm.station, 'check_rig', lambda *a: {'codec_gain_working_point': {}})
    monkeypatch.setattr(arm.station, 'selection_for_plan', lambda *a: None)
    monkeypatch.setattr(arm.station, 'spell_environment', lambda *a, **k: [])
    monkeypatch.setattr(arm.station, 'read_environment', lambda **k: {})
    monkeypatch.setattr(arm.residency, 'evidence', lambda *a: resided)
    monkeypatch.setattr(arm.residency, 'label', lambda *a: 'synthetic-ready')
    monkeypatch.setattr(arm, 'passthrough_note', lambda *a: '')
    monkeypatch.setattr(arm, 'spelled', lambda *a: 'synthetic')
    monkeypatch.setattr(arm, 'verdict', lambda *a: {'reached': '', 'adjustments': [], 'modules': '', 'line': 'synthetic'})
    monkeypatch.setattr(arm.station, 'start_tap', lambda *a, **k: None)
    monkeypatch.setattr(arm.claim.hardware, 'inherited',
                        lambda fd, *, launch=False: SimpleNamespace(fd=None, failed=lambda: None))
    monkeypatch.setattr(arm.claim, 'remember', lambda process: process)
    monkeypatch.setattr(arm.claim, 'end_child', lambda process, **kw: None)
    monkeypatch.setattr(arm.claim, 'end_children', lambda **kw: None)
    monkeypatch.setattr(arm, '_down_once', lambda env, timeout: (0, 'PTT LINE low\nCLEANUP COMPLETE\n'))
    return arm, prepared


def test_actual_console_qualification_failure_records_end_before_producers(tmp_path, monkeypatch):
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    original = RuntimeError('synthetic qualification failure')
    def fail(*args):
        raise original
    monkeypatch.setattr(arm, 'qualified', fail)
    with pytest.raises(RuntimeError) as caught:
        arm.fly(SimpleNamespace(name='pactor'), plan(tmp_path))
    assert caught.value is original
    # Locate the declared primary root without relying on wallclock basename.
    manifests = list(tmp_path.rglob(ci.MANIFEST))
    assert manifests
    capture = ci.load(manifests[0].parent)
    assert capture.ended() and not capture.lifecycle()['capture_complete']
    assert not capture.lifecycle()['writers']


def test_actual_console_producer_failure_keeps_partial_ownership(tmp_path, monkeypatch):
    arm, prepared = _fake_console_environment(tmp_path, monkeypatch)
    original = OSError('synthetic producer failure')
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(arm.station, 'start_tap', fail)
    with pytest.raises(OSError) as caught:
        arm.fly(SimpleNamespace(name='pactor'), prepared)
    assert caught.value is original
    capture = ci.load(next(tmp_path.rglob(ci.MANIFEST)).parent)
    row = capture.lifecycle()
    assert row['arm_ended'] and not row['capture_complete']
    assert set(row['writers']) == {'native-metadata', 'native-transcript'}
    assert all(r['state'] == 'partial' for r in row['writers'].values())


def test_actual_console_fake_complete_run_ends_and_finishes_native_writers(tmp_path, monkeypatch):
    arm, prepared = _fake_console_environment(tmp_path, monkeypatch)
    class FakeLauncher:
        pass
    monkeypatch.setattr(arm.subprocess, 'Popen', lambda *a, **k: FakeLauncher())
    def pump(launcher, sink, screen):
        sink.write('synthetic-console-line\n')
        return 0
    monkeypatch.setattr(arm, 'pump', pump)
    assert arm.fly(SimpleNamespace(name='pactor'), prepared) == 0
    capture = ci.load(next(tmp_path.rglob(ci.MANIFEST)).parent)
    row = capture.lifecycle()
    assert row['arm_ended'] and row['claimed_writers_complete']
    assert not row['capture_complete']  # Unlaunched delegated producers remain explicit.
    assert all(r['state'] == 'finalized' for r in row['writers'].values())


def test_actual_console_primary_failure_survives_cleanup_failure(tmp_path, monkeypatch):
    arm, prepared = _fake_console_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(arm.subprocess, 'Popen', lambda *a, **k: SimpleNamespace())
    original = OSError('synthetic pump failure')
    def fail_pump(*args):
        raise original
    class FailingView:
        def close(self):
            raise RuntimeError('synthetic cleanup failure')
    monkeypatch.setattr(arm, 'pump', fail_pump)
    monkeypatch.setattr(arm.view, 'View', FailingView)
    with pytest.raises(OSError) as caught:
        arm.fly(SimpleNamespace(name='pactor'), prepared)
    assert caught.value is original
    capture = ci.load(next(tmp_path.rglob(ci.MANIFEST)).parent)
    assert capture.ended() and not capture.lifecycle()['capture_complete']
    assert all(r['state'] == 'partial' for r in capture.lifecycle()['writers'].values())


def test_actual_console_primary_failure_survives_ending_journal_failure(tmp_path, monkeypatch):
    from consolelib import arm
    monkeypatch.setattr(arm.station, 'ROOT', tmp_path)
    original = OSError('synthetic qualification failure')
    def fail_qualification(*args):
        raise original
    def fail_ending(self):
        raise RuntimeError('synthetic ending journal failure')
    monkeypatch.setattr(arm, 'qualified', fail_qualification)
    monkeypatch.setattr(ci.Capture, 'end', fail_ending)
    with pytest.raises(OSError) as caught:
        arm.fly(SimpleNamespace(name='pactor'), plan(tmp_path))
    assert caught.value is original
    capture = ci.load(next(tmp_path.rglob(ci.MANIFEST)).parent)
    assert not capture.ended() and not capture.lifecycle()['capture_complete']
