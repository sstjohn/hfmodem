# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Resolved policies carry their source and one inherited environment value."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from hfmodem.tests.station.recording_support import module

WARNING = 'Recording off: session and contact audio will not be saved.'


def resolve(*, explicit=None, inherited=None, station=None, no_record=False, record=None):
    env = {} if inherited is None else {'HFMODEM_RECORDING': inherited}
    reports = []
    cfg = None if station is None else SimpleNamespace(recording=SimpleNamespace(policy=station))
    result = module().resolve_recording(explicit, config=cfg, no_record=no_record,
                                        record=record, environ=env, report=reports.append)
    return result, env, reports


def test_policy_names_and_exact_warning():
    rec = module()
    assert rec.POLICIES == ('off', 'contacts', 'everything')
    result, env, reports = resolve(explicit='off')
    assert result.policy == 'off'
    assert env == {'HFMODEM_RECORDING': 'off'}
    assert reports == [WARNING]


@pytest.mark.parametrize('argv,expected', [([], None), (['--recording', 'off'], 'off'),
                                        (['--recording', 'contacts'], 'contacts'),
                                        (['--recording', 'everything'], 'everything')])
def test_shared_argparse_helper(argv, expected):
    parser = argparse.ArgumentParser()
    module().add_recording_argument(parser)
    assert parser.parse_args(argv).recording == expected
    assert '{off,contacts,everything}' in parser.format_help()


def test_parser_rejects_unknown_policy():
    parser = argparse.ArgumentParser()
    module().add_recording_argument(parser)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--recording', 'all'])
    assert exc.value.code == 2


@pytest.mark.parametrize('explicit,inherited,station,expected,source', [
    (None, None, None, 'everything', 'default'),
    (None, None, 'off', 'off', 'station'),
    (None, 'contacts', 'off', 'contacts', 'inherited'),
    ('everything', 'contacts', 'off', 'everything', 'explicit'),
    ('contacts', 'off', 'everything', 'contacts', 'explicit'),
    ('off', None, 'everything', 'off', 'explicit'),
])
def test_resolution_precedence(explicit, inherited, station, expected, source):
    result, env, reports = resolve(explicit=explicit, inherited=inherited, station=station)
    assert (result.policy, result.source) == (expected, source)
    assert env['HFMODEM_RECORDING'] == expected
    assert reports == ([WARNING] if expected == 'off' and inherited is None else [])


@pytest.mark.parametrize('kwargs,expected', [({'no_record': True}, 'off'),
                                          ({'record': 'chosen.wav'}, 'everything'),
                                          ({'explicit': 'off', 'no_record': True}, 'off'),
                                          ({'explicit': 'everything', 'record': 'chosen.wav'}, 'everything')])
def test_explicit_legacy_aliases(kwargs, expected):
    result, _, _ = resolve(inherited='contacts', station='contacts', **kwargs)
    assert (result.policy, result.source) == (expected, 'explicit')


@pytest.mark.parametrize('kwargs', [
    {'explicit': 'contacts', 'no_record': True},
    {'explicit': 'everything', 'no_record': True},
    {'explicit': 'off', 'record': 'chosen.wav'},
    {'explicit': 'contacts', 'record': 'chosen.wav'},
    {'no_record': True, 'record': 'chosen.wav'},
])
def test_contradictory_explicit_selections_are_errors(kwargs):
    recording = module()
    with pytest.raises(ValueError, match=r'(?i)(contradict|conflict)'):
        recording.resolve_recording(kwargs.pop('explicit', None), environ={}, **kwargs)


def test_invalid_inherited_policy_is_rejected():
    recording = module()
    with pytest.raises(ValueError, match='HFMODEM_RECORDING'):
        recording.resolve_recording(None, environ={'HFMODEM_RECORDING': 'unknown'})


def test_off_warning_occurs_only_at_top_level():
    rec = module()
    env, reports = {}, []
    parent = rec.resolve_recording('off', environ=env, report=reports.append)
    for _ in range(3):
        child = rec.resolve_recording(None, environ=dict(env), report=reports.append)
        assert (child.policy, child.source) == ('off', 'inherited')
    assert parent.policy == 'off'
    assert reports == [WARNING]


@pytest.mark.parametrize('explicit,expected,source', [
    (None, 'contacts', 'station'), ('off', 'off', 'explicit'),
])
def test_station_environment_selects_config_file(tmp_path, explicit, expected, source):
    rec = module()
    path = tmp_path / 'station.toml'
    example = Path(__file__).resolve().parents[5] / 'examples/station.toml'
    path.write_text(example.read_text() + '\n[recording]\npolicy = "contacts"\n')
    env = {'HFMODEM_STATION': str(path)}
    result = rec.resolve_recording(explicit, environ=env, report=lambda message: None)
    assert (result.policy, result.source) == (expected, source)
    assert env['HFMODEM_RECORDING'] == expected


def test_invalid_explicit_value_is_rejected():
    rec = module()
    with pytest.raises(ValueError, match=r'(?i)(policy|recording)'):
        rec.resolve_recording('all', environ={})
