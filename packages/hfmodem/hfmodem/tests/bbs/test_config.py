# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.bbs.config import BBSConfig, Limits, load, normalize_address


def test_defaults_overrides_ssids_portable_and_aliases():
    config = BBSConfig(callsigns={"K1ABC": {"door_seconds_per_day": 0}, "W9SSJ": {"connection_seconds": 7200}}, aliases={"12345": "K1ABC"})
    assert config.effective_limits("k1abc-2").door_seconds_per_day == 0
    assert config.effective_limits("12345").door_seconds_per_day == 0
    assert config.effective_limits("W9SSJ").connection_seconds == 7200
    assert config.effective_limits("W9SSJ").idle_seconds == 900
    assert config.resolve("EA/K1ABC/P-1") == "EA/K1ABC/P"
    assert normalize_address("K1ABC-1", False) == "K1ABC-1"
    assert config.resolve("K1ABC/P") != config.resolve("K1ABC")


@pytest.mark.parametrize("value", ["K1ABC-16", "K1ABC-01", "K1ABC--1", "K1ABC-X", "../K1ABC", "K1ABC\nX", "", "A" * 33])
def test_bad_addresses(value):
    with pytest.raises(ValueError):
        normalize_address(value)


@pytest.mark.parametrize("kwargs", [{"idle_seconds": 0}, {"connection_seconds": 86401}, {"door_seconds_per_day": -1}, {"door_seconds_per_day": True}, {"idle_seconds": 100, "connection_seconds": 99}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        Limits(**kwargs)


def test_alias_cycles_and_conflicting_overrides():
    with pytest.raises(ValueError):
        BBSConfig(aliases={"K1ABC": "W9SSJ", "W9SSJ": "K1ABC"})
    with pytest.raises(ValueError):
        BBSConfig(aliases={"K1ABC": "W9SSJ"}, callsigns={"K1ABC": {"door_seconds_per_day": 0}, "W9SSJ": {"door_seconds_per_day": 1}})


def test_config_paths_are_relative_to_file(tmp_path):
    path = tmp_path / "bbs.toml"
    path.write_text('[bbs]\nid="test"\nstate_root="state"\nexport_root="files"\n[bbs.callsigns.K1ABC]\ndoor_seconds_per_day=0\n[games.zork]\nstory="assets/zork.z3"\n')
    config = load(path)
    assert config.state_root == tmp_path / "state"
    assert config.export_root == tmp_path / "files"
    assert config.games["zork"]["story"] == str(tmp_path / "assets/zork.z3")
    assert config.effective_limits("K1ABC-15").door_seconds_per_day == 0


def test_message_area_defaults_and_config(tmp_path):
    assert BBSConfig().message_areas == {'general': 'General'}
    assert BBSConfig().message_posts_per_day == 20
    path = tmp_path / 'messages.toml'
    path.write_text('[bbs]\nmessage_posts_per_day=0\n[bbs.message_areas]\nradio="Radio talk"\nnews="News"\n')
    config = load(path)
    assert config.message_areas == {'radio': 'Radio talk', 'news': 'News'}
    assert config.message_posts_per_day == 0


@pytest.mark.parametrize('areas', [{'SYSOP': 'Wrong'}, {'sysop': 'Reserved'}, {'../a': 'Path'}, {'a': ''}, {'a': 'x\ny'}, {'a': 'é' * 37}, {str(i): 'Area' for i in range(33)}, []])
def test_invalid_message_areas(areas):
    with pytest.raises(ValueError):
        BBSConfig(message_areas=areas)


@pytest.mark.parametrize('limit', [-1, 1001, True, 1.5])
def test_invalid_message_post_cap(limit):
    with pytest.raises(ValueError):
        BBSConfig(message_posts_per_day=limit)
