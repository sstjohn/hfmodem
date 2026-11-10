# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import asdict

import pytest

from hfmodem.core import config
from hfmodem.core.rigs import RIGS, resolve_tx_timing


def _audio_table(rig, *, input='', output='', latency=None):
    chain = RIGS[rig].get('audio_chain')
    header = f'[audio.chains.{chain}]\n' if chain else '[audio]\n'
    return (header + (f'input="{input}"\n' if input else '')
            + (f'output="{output}"\n' if output else '')
            + (f'tx_latency_ms={latency}\n' if latency is not None else ''))


def test_unset_rig_keeps_ft891_generic_flight_coordinate(monkeypatch):
    monkeypatch.delenv('HFMODEM_RIG', raising=False)
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    plan = resolve_tx_timing()
    assert plan.rig == 'ft891'
    assert plan.latency_samples == plan.geometry_samples == 960


@pytest.mark.parametrize('rig', RIGS)
@pytest.mark.parametrize('path', ['generic', 'flown', 'p2-mail', 'p4-bench'])
def test_profile_resolution_and_geometry(rig, path):
    plan = resolve_tx_timing(rig=rig, command_path=path, environ={})
    assert plan.tx_latency_ms == (20 if rig == 'ft891' else 7 if rig == 'ftdx10' else 0)
    assert plan.geometry_samples == (816 if (rig, path) == ('ft891', 'p4-bench')
                                     else 960 if (rig, path) == ('ft891', 'generic') else 0)
    assert plan.calibration_status == ('INFERRED' if rig == 'ft891' else 'MEASURED' if rig == 'ftdx10' else 'UNMEASURED')
    assert asdict(plan)['latency_source'] == 'rig-profile'


@pytest.mark.parametrize('rig', RIGS)
@pytest.mark.parametrize('path', ['generic', 'flown', 'p2-mail', 'p4-bench'])
@pytest.mark.parametrize('override', [0, 7, 60])
def test_operator_wins_over_present_chain_value(tmp_path, rig, path, override):
    station = tmp_path / 'chain.toml'
    station.write_text('schema = 1\n[station]\ncontrol="local"\nregulatory="unregulated"\nbecause="offline timing fixture"\n'
                       + _audio_table(rig, input='selected ADC', output='selected DAC', latency=11))
    plan = resolve_tx_timing(override, rig=rig, command_path=path,
                             environ={config.STATION_ENV: str(station)})
    assert plan.tx_latency_ms == override
    assert plan.latency_source == 'operator'
    assert plan.geometry_samples == 0 and not plan.compatibility_identity
    assert 'selected ADC -> selected DAC' in plan.chain


@pytest.mark.parametrize('rig', RIGS)
@pytest.mark.parametrize('present', [False, True])
def test_absent_key_and_explicit_toml_zero_are_distinct(tmp_path, rig, present):
    station = tmp_path / 'chain.toml'
    station.write_text('schema = 1\n[station]\ncontrol="local"\nregulatory="unregulated"\nbecause="offline timing fixture"\n'
                       + _audio_table(rig, latency=0 if present else None))
    assert config.load(station, rig_name=rig, environ={}).audio.tx_latency_ms == (0 if present else None)
    plan = resolve_tx_timing(rig=rig, station_path=station,
                             command_path='p4-bench', environ={})
    assert plan.latency_source == ('station-toml' if present else 'rig-profile')
    assert plan.geometry_samples == (816 if rig == 'ft891' and not present else 0)
    assert plan.tx_latency_ms == (20 if rig == 'ft891' and not present else
                                  7 if rig == 'ftdx10' and not present else 0)


@pytest.mark.parametrize('value', [-1, 61, float('nan'), float('inf'), '20'])
def test_invalid_operator_and_chain_values_are_refused(tmp_path, value):
    with pytest.raises(config.ConfigError, match='tx_latency_ms'):
        resolve_tx_timing(value, environ={})
    station = tmp_path / 'chain.toml'
    literal = repr(value) if isinstance(value, str) else str(value)
    station.write_text(f'schema=1\n[station]\ncontrol="local"\nregulatory="unregulated"\nbecause="offline timing fixture"\n[audio]\ntx_latency_ms={literal}\n')
    with pytest.raises(config.ConfigError, match='tx_latency_ms'):
        resolve_tx_timing(station_path=station, environ={})


def test_selected_file_errors_are_not_profile_fallback(tmp_path):
    with pytest.raises(config.ConfigError, match='HFMODEM_STATION'):
        resolve_tx_timing(0, environ={config.STATION_ENV: str(tmp_path / 'missing')})


def test_numeric_config_accessor_uses_the_shared_resolver(monkeypatch):
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    monkeypatch.setenv('HFMODEM_RIG', 'ft891')
    assert config.tx_latency_ms() == 20
    assert config.tx_latency_ms(0) == 0
