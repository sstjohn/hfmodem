"""Identity resolution and policy, including the boundaries before hardware."""
from dataclasses import replace
from pathlib import Path

import pytest

from hfmodem.core import config, identification
from hfmodem.tests.core.test_config import base


def station_file(tmp_path, call='W1AW'):
    path = tmp_path / 'station.toml'
    source = Path(__file__).resolve().parents[5] / 'examples/station.toml'
    path.write_text(source.read_text().replace('[station]', f'[station]\nidcall = "{call}"'))
    return str(path)


@pytest.mark.parametrize('override,env_call,expected,source', [
    ('K1ABC', 'K2ABC', 'K1ABC', '--idcall'),
    (None, 'K2ABC', 'K2ABC', 'IDCALL'),
    (None, None, 'W1AW', '[station] idcall'),
])
def test_precedence(tmp_path, override, env_call, expected, source):
    env = {config.STATION_ENV: station_file(tmp_path)}
    if env_call is not None:
        env['IDCALL'] = env_call
    identity = identification.resolve('TACTIC-1', override, environ=env)
    assert (identity.digital_call, identity.cw_call, identity.source) == ('TACTIC-1', expected, source)


def test_fallback_and_normalization(tmp_path):
    env = {config.STATION_ENV: station_file(tmp_path, '')}
    identity = identification.resolve(' w1aw/p ', environ=env)
    assert identity.cw_call == 'W1AW/P' and not identity.separate
    assert identity.digital_call == ' w1aw/p '
    assert identification.resolve('W1AW', environ={}).cw_call == 'W1AW'


@pytest.mark.parametrize('value', ['', ' ', 'W1AW-1', 'W1 AW', 'ß', 'Ｋ1ABC', '/', None, 1])
def test_explicit_env_and_config_invalid_values_are_not_fallbacks(tmp_path, value):
    with pytest.raises(ValueError):
        identification.resolve('W1AW', environ={'IDCALL': value})
    raw = base()
    raw['station']['idcall'] = value
    if value == '':
        assert config.parse(raw).station.idcall == ''
    else:
        with pytest.raises(ValueError, match='idcall'):
            config.parse(raw)
    if value is not None:
        with pytest.raises(ValueError):
            identification.resolve('W1AW', value, environ={})


def test_broken_selected_config_never_silently_changes_identity(tmp_path):
    path = tmp_path / 'missing.toml'
    env = {config.STATION_ENV: str(path)}
    with pytest.raises(config.ConfigError, match='missing.toml'):
        identification.resolve('W1AW', environ=env)
    path.write_text('schema = 999')
    with pytest.raises(config.ConfigError, match='schema'):
        identification.resolve('W1AW', environ=env)
    # A deliberate per-run override does not need the file for identity.
    assert identification.resolve('TACTIC', 'K1ABC', environ=env).cw_call == 'K1ABC'


@pytest.mark.parametrize('mode', ['pactor', 'vara', 'ardop'])
@pytest.mark.parametrize('optional', [False, True])
@pytest.mark.parametrize('linked', [False, True])
@pytest.mark.parametrize('separate', [False, True])
def test_closing_policy(mode, optional, linked, separate):
    identity = identification.resolve('TACTIC' if separate else ' w1aw ', 'W1AW', environ={})
    expected = separate or not optional or (mode == 'pactor' and not linked)
    assert identity.closing_required(mode, optional=optional, linked=linked) is expected


def test_final_options_and_end_of_options():
    args = identification.options(['--timeout', '5', '--mycall=A1', '--idcall', 'W1AW',
                                   '--mycall', 'TACTIC', '--idcall=K1ABC'], mycall='BEFORE')
    assert (args.mycall, args.idcall) == ('TACTIC', 'K1ABC')
    args = identification.options(['--', '--idcall=AFTER'], mycall='W1AW')
    assert args.idcall is None


def test_station_resolves_before_devices_and_uses_only_config(monkeypatch):
    from hfmodem.station import process
    cfg = config.parse(base())
    monkeypatch.setenv('IDCALL', 'K9ENV')
    monkeypatch.setenv('MYCALL', 'ENV')
    cfg = replace(cfg, station=replace(cfg.station, transmit=True, mycall='TACTIC', idcall='W1AW'))
    st = process.Station(cfg)
    assert st.cw_identity.cw_call == 'W1AW' and st.cw_identity.digital_call == 'TACTIC'
    cfg = replace(cfg, station=replace(cfg.station, idcall='', mycall='TACTIC-1'))
    monkeypatch.setattr(process.devices, 'find_device', lambda *a, **kw: pytest.fail('device opened'))
    monkeypatch.setattr(process, 'RtsPtt', lambda *a, **kw: pytest.fail('PTT opened'))
    with pytest.raises(ValueError, match='mycall'):
        process.Station(cfg)
    assert process.Station(replace(cfg, station=replace(cfg.station, transmit=False))).cw_identity is None
    assert process.Station(replace(cfg, station=replace(cfg.station, id_mode='none'))).cw_identity is None


def test_station_interlock_receives_cw_identity(monkeypatch):
    from unittest.mock import Mock
    from hfmodem.station import process
    from hfmodem.tests.kestrel.test_station_id import _decode
    from hfmodem.core import cwid
    cfg = config.parse(base())
    cfg = replace(cfg, station=replace(cfg.station, transmit=True, mycall='TACTIC', idcall='W1AW'),
                  rig=replace(cfg.rig, ptt=replace(cfg.rig.ptt, port='/dev/mock')))
    rig, line = Mock(), Mock()
    monkeypatch.setattr(process, 'RtsPtt', lambda *a: line)
    monkeypatch.setattr(process, 'Rig', lambda **kw: rig)
    seen = []
    monkeypatch.setattr(process, 'TxArbiter', lambda *a, **kw: seen.append(kw['identity']) or Mock())
    st = process.Station(cfg)
    st.open_rig()
    assert _decode(seen[0].burst(), cwid.WPM) == 'W1AW'
    assert seen[0].interval_s == cfg.station.id_interval_s
    assert st.cw_identity.digital_call == 'TACTIC'
