# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Station configuration — mostly the refusals, since that is the point.

Every one of these corresponds to a way this station has actually been got
wrong, or to a way the merge makes it newly possible to get wrong.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from hfhost.config import ConfigError

from hfmodem.core import config

REPO = Path(__file__).resolve().parents[5]
EXAMPLE = REPO / "examples" / "station.toml"


def base() -> dict:
    return tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_the_shipped_example_loads():
    cfg = config.load(EXAMPLE)
    assert cfg.station.mycall == "N0CALL"   # a template, not a station
    assert cfg.rig.dial_hz == cfg.rig.centre_hz - 1500
    assert cfg.profile.name == "part97"


def test_transmit_defaults_off_in_the_example():
    """The shipped example must not be a config that keys a radio if run."""
    assert config.load(EXAMPLE).station.transmit is False


def test_the_example_names_no_real_station():
    """It ships in the distribution. A callsign, grid or licence class in it is
    a claim about a person, and the least permissive class is the safe default
    for a file someone will edit rather than read."""
    cfg = config.load(EXAMPLE)
    assert cfg.station.mycall in ("N0CALL", "")
    assert cfg.station.grid in ("AA00", "")
    assert cfg.profile.licence.value == "technician"


def test_the_receive_filter_defaults_to_an_ssb_passband_and_can_be_named():
    assert config.load(EXAMPLE).rig.filter_hz == (300, 2900)
    raw = base(); raw["rig"]["filter_hz"] = [200, 3100]
    assert config.parse(raw).rig.filter_hz == (200, 3100)
    raw["rig"]["filter_hz"] = [3100, 200]
    with pytest.raises(ConfigError, match="filter_hz"):
        config.parse(raw)


# --- typos are errors ---------------------------------------------------------

def test_an_unknown_key_is_an_error_not_a_shrug():
    raw = base(); raw["audio"]["input_gian"] = 0.04
    with pytest.raises(ConfigError, match="input_gian"):
        config.parse(raw)


def test_an_unknown_table_is_an_error():
    raw = base(); raw["raido"] = {}
    with pytest.raises(ConfigError, match="raido"):
        config.parse(raw)


def test_an_unknown_protocol_is_an_error():
    raw = base(); raw["protocols"]["winmor"] = {"enabled": True}
    with pytest.raises(ConfigError, match="winmor"):
        config.parse(raw)


def test_the_schema_version_is_checked():
    raw = base(); raw["schema"] = 2
    with pytest.raises(ConfigError, match="schema"):
        config.parse(raw)


# --- the facts that have gone wrong before ------------------------------------

def test_dial_hz_is_a_named_rejected_key():
    """Winlink publishes the centre. Every modem here has had this backwards,
    and the failure is silent in both directions."""
    raw = base(); raw["rig"]["dial_hz"] = 7_100_000
    with pytest.raises(ConfigError, match="centre_hz"):
        config.parse(raw)


@pytest.mark.parametrize("gain", [0.51, 0.50, 0.18, 0.118])
def test_the_four_gains_this_station_has_been_wrong_at_are_refused(gain):
    raw = base(); raw["audio"]["input_gain"] = gain
    with pytest.raises(ConfigError, match="0.040"):
        config.parse(raw)


def test_the_gain_bound_is_coarse_and_says_so():
    """It refuses the values this station has demonstrably decoded nothing at,
    and it cannot refuse a plausible wrong value for someone else's chain. That
    is what preflight measurement is for, not this."""
    raw = base(); raw["audio"]["input_gain"] = 0.06
    assert config.parse(raw).audio.input_gain == 0.06


def test_a_port_collision_across_protocols_is_an_error():
    """Even different host dialects cannot share one TCP port."""
    raw = base()
    raw["protocols"]["sabir"] = {"enabled": True, "host": "hostapi",
                                 "port": 8300}
    with pytest.raises(ConfigError, match="8300"):
        config.parse(raw)


def test_an_unset_output_device_with_a_protocol_enabled_is_an_error():
    """An unset device is the system default, which is the built-in speakers,
    and every symptom of that looks like a radio fault."""
    raw = base(); raw["audio"]["output"] = ""
    with pytest.raises(ConfigError, match="output"):
        config.parse(raw)


# --- the emission profile must be declared ----------------------------------------------

def test_control_mode_is_required():
    raw = base(); del raw["station"]["control"]
    with pytest.raises(ConfigError, match="control is required"):
        config.parse(raw)


def test_the_regulatory_profile_is_required():
    raw = base(); del raw["station"]["regulatory"]
    with pytest.raises(ConfigError, match="regulatory is required"):
        config.parse(raw)


def test_an_unknown_regime_is_refused_rather_than_defaulted():
    raw = base(); raw["station"]["regulatory"] = "ofcom"
    with pytest.raises(ValueError, match="unknown regulatory profile"):
        config.parse(raw)


def test_the_unregulated_profile_is_selectable_and_costs_a_reason():
    """The choice for any limits hfmodem does not encode — and it is the
    override flag, so it is not free."""
    raw = base()
    raw["station"]["regulatory"] = "unregulated"
    del raw["station"]["licence"]
    with pytest.raises((ValueError, TypeError)):
        config.parse(raw)
    raw["station"]["because"] = "dummy load, no antenna connected"
    p = config.parse(raw).profile
    assert p.name == "unregulated" and "dummy load" in p.because


def test_part97_without_a_licence_class_is_refused():
    raw = base(); del raw["station"]["licence"]
    with pytest.raises((ConfigError, TypeError)):
        config.parse(raw)


def test_listens_reports_whether_the_station_answers_unattended():
    raw = base()
    assert config.parse(raw).listens is True
    for p in raw["protocols"].values():
        p["listen"] = False
    assert config.parse(raw).listens is False


def test_an_unimplemented_ptt_backend_is_refused_rather_than_substituted():
    """`backend = "rigctld"` validated and then keyed RTS anyway — a transmitter
    keyed by a route other than the configured one, which is the failure the
    single-owner rule exists to prevent."""
    raw = base()
    raw["rig"]["ptt"]["backend"] = "rigctld"
    with pytest.raises(ConfigError, match="not implemented"):
        config.parse(raw)


def test_a_guard_that_cannot_fire_is_not_left_looking_like_protection():
    """The PTT-is-the-CAT-port check compared against a helper that returned an
    empty string unconditionally, so no configuration could ever trip it. What
    protects that boundary is the preflight naming the device it is about to key,
    and the arm gate refusing a daemon that owns PTT itself."""
    import inspect
    from hfmodem.core import config as mod
    assert "_cat_port" not in inspect.getsource(mod)


# --- `hfmodem config` and the keying line -------------------------------------
#
# The one verb an operator reaches for to check a station file must be able to
# say the PTT port is wrong. A stat and never an open: opening an RTS keying
# line asserts RTS, which is a key-down. Every path named here is under
# tmp_path or is /dev/null, so nothing on this bench can be keyed.

def _station_naming_port(tmp_path, port) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        'settle_s = 0.04', f'settle_s = 0.04\nport = "{port}"')
    f = tmp_path / "station.toml"
    f.write_text(text, encoding="utf-8")
    return f


def _config_output(tmp_path, capsys, port) -> str:
    from hfmodem import cli
    assert cli.main(["config", str(_station_naming_port(tmp_path, port))]) == 0
    return capsys.readouterr().out


def test_config_names_a_ptt_port_that_does_not_exist(tmp_path, capsys):
    out = _config_output(tmp_path, capsys, tmp_path / "no-such-tty")
    assert "No such file" in out
    assert "a character device" not in out


def test_config_names_a_ptt_port_that_is_no_device(tmp_path, capsys):
    plain = tmp_path / "plain-file"
    plain.write_text("")
    out = _config_output(tmp_path, capsys, plain)
    assert "is not a character device" in out


def test_config_confirms_a_real_character_device(tmp_path, capsys):
    out = _config_output(tmp_path, capsys, "/dev/null")
    assert "(a character device)" in out


def test_config_still_reports_an_unset_port_as_unset(tmp_path, capsys):
    from hfmodem import cli
    f = tmp_path / "station.toml"
    f.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    assert cli.main(["config", str(f)]) == 0
    assert "(unset)" in capsys.readouterr().out


@pytest.mark.parametrize("name,backend,identity", [
    ("ft891", "yaesu", "ft891"), ("ftdx10", "yaesu", "ftdx10"),
    ("generic", "rigctld", ""),
])
def test_station_cat_profile_defaults(name, backend, identity):
    raw = base()
    raw["rig"].pop("cat", None)
    cfg = config.parse(raw, rig_name=name, environ={})
    assert cfg.station.port == 8737
    assert cfg.rig.cat == backend
    assert cfg.rig.model == name
    assert cfg.rig.identity == identity
    assert cfg.rig.dial_max_age_s == 30


@pytest.mark.parametrize("backend", ["yaesu", "civ", "rigctld", "none"])
def test_explicit_cat_selection(backend):
    raw = base()
    raw["rig"]["cat"] = backend
    assert config.parse(raw, environ={}).rig.cat == backend


def test_generic_keeps_operator_mode_ptt_and_audio():
    raw = base()
    raw["rig"] = {"model": "generic", "mode": "LSB", "identity": "My radio",
                  "max_watts": 75, "ptt": {"port": "fake-line"}}
    raw["audio"]["chains"] = {"generic": {"input": "fake-input", "output": "fake-output"}}
    cfg = config.parse(raw, environ={})
    assert cfg.rig.mode == "LSB" and cfg.rig.identity == "My radio"
    assert cfg.rig.ptt.port == "fake-line" and cfg.rig.ptt.settle_s == 0.1
    assert cfg.audio.input == "fake-input" and cfg.audio.output == "fake-output"
    assert cfg.rig.max_watts == 75


@pytest.mark.parametrize("section,key,value", [
    ("station", "port", 0), ("station", "port", 65536), ("station", "port", True),
    ("station", "port", "8737"), ("station", "port", 8737.0),
    ("rig", "cat", "typo"), ("rig", "identity", 123), ("rig", "identity", "bad\nname"),
    ("rig", "dial_max_age_s", 0), ("rig", "dial_max_age_s", -1),
    ("rig", "dial_max_age_s", float("nan")), ("rig", "dial_max_age_s", float("inf")),
    ("rig", "dial_max_age_s", True), ("rig", "dial_max_age_s", "30"),
    ("rig", "max_watts", -1), ("rig", "max_watts", True),
    ("rig", "power_clamp_w", 0), ("rig", "power_clamp_w", -25), ("rig", "power_clamp_w", True),
    ("rig", "power_clamp_w", "25"), ("rig", "power_clamp_w", float("inf")),
])
def test_invalid_station_rig_fields(section, key, value):
    raw = base()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=key):
        config.parse(raw, environ={})


def test_power_clamp_is_read_from_the_rig_table():
    raw = base()
    assert config.parse(raw, environ={}).rig.power_clamp_w is None
    raw["rig"]["power_clamp_w"] = 25
    assert config.parse(raw, environ={}).rig.power_clamp_w == 25


@pytest.mark.parametrize("watts, clamp, held, warned", [
    (60, 25, 25, "warning: /power asks 60 W, above the 25 W station clamp ([rig] power_clamp_w); "
                 "setting 25 W"),
    (25, 25, 25, ""), (20, 50, 20, ""), (100, None, 100, ""),
])
def test_clamp_power_holds_and_names_both_figures(watts, clamp, held, warned):
    assert config.clamp_power(watts, clamp, "/power") == (held, warned)


def test_power_limits_follow_the_named_station_file(tmp_path, monkeypatch):
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    monkeypatch.delenv("HFMODEM_RIG", raising=False)
    assert config.power_limits() == (100, None, 10, True)
    monkeypatch.setenv("HFMODEM_RIG", "ftdx10")
    assert config.power_limits().tune_carrier is False
    monkeypatch.delenv("HFMODEM_RIG")
    path = tmp_path / "station.toml"
    path.write_text(EXAMPLE.read_text(encoding="utf-8").replace(
        "[rig]\n", "[rig]\npower_clamp_w = 50\nmax_watts = 200\ntune_power_w = 20\n", 1))
    monkeypatch.setenv(config.STATION_ENV, str(path))
    assert config.power_limits() == (200, 50, 20, True)


def test_tune_power_defaults_to_ten_watts_and_must_be_positive():
    raw = base()
    assert config.parse(raw, environ={}).rig.tune_power_w == 10
    raw["rig"]["tune_power_w"] = 0
    with pytest.raises(ConfigError, match="tune_power_w"):
        config.parse(raw, environ={})


def test_every_rig_profile_says_whether_a_tune_carrier_helps():
    from hfmodem.core.rigs import RIGS
    assert {name: profile["tune_carrier"] for name, profile in RIGS.items()} == dict(
        generic=True, ft891=True, ftdx10=False, g90=True, x6100=True)
    raw = base()
    raw["rig"]["profiles"] = {"ft891": {"tune_carrier": False}}
    assert config.parse(raw, environ={}).resolved_profile["tune_carrier"] is False


def test_the_full_scale_is_the_station_files_then_the_rigs():
    raw = base()
    assert config.parse(raw, environ={}).full_scale_w == 100
    raw["rig"]["max_watts"] = 200
    assert config.parse(raw, environ={}).full_scale_w == 200


def test_a_clamp_below_the_rigs_minimum_is_held_at_the_minimum(capsys):
    raw = base()
    raw["rig"]["power_clamp_w"] = 2
    assert config.parse(raw, environ={}).rig.power_clamp_w == 5
    assert "power_clamp_w 2 W is below the FT891's 5 W minimum; clamping at 5 W" in capsys.readouterr().err


@pytest.mark.parametrize("power, clamp, declared", [(100, 50, 50), (20, 50, 20), (100, None, 100),
                                                     (None, 50, None)])
def test_the_declared_power_is_held_to_the_clamp(power, clamp, declared):
    raw = base()
    for key, value in (("power_w", power), ("power_clamp_w", clamp)):
        if value is not None:
            raw["rig"][key] = value
    cfg = config.parse(raw, environ={})
    assert cfg.declared_power_w == declared
    from hfmodem.contact.facts import facts_from
    assert facts_from(cfg)["power"] == ("" if declared is None else f"{declared:g} watts")


def test_no_cat_requires_receive_only():
    raw = base()
    raw["rig"]["cat"] = "none"
    raw["station"]["transmit"] = True
    with pytest.raises(ConfigError, match="transmit = false"):
        config.parse(raw, environ={})


def test_station_port_cannot_collide_with_enabled_host():
    raw = base()
    raw["station"]["port"] = 8300
    raw["protocols"]["sabir"] = {"enabled": True, "host": "hostapi", "port": 8300}
    with pytest.raises(ConfigError, match=r"already used by \[station\]"):
        config.parse(raw, environ={})



def test_default_generic_mode_and_ft891_profile_override():
    raw = base()
    raw["rig"] = {"model": "generic"}
    assert config.parse(raw, environ={}).rig.mode == "USB"
    cfg = config.parse(raw, rig_name="ft891", environ={})
    assert cfg.rig.model == "ft891" and cfg.rig.cat == "yaesu"


def test_desk_adif_directory_is_optional_and_explicit():
    assert config.parse(base()).desk.adif_dir == ''
    assert config.parse(base() | {'desk': {'adif_dir': 'imports'}}).desk.adif_dir == 'imports'


@pytest.mark.parametrize('desk', [{'adif_dir': 3}, {'adif_dir': 'bad\x00path'}, {'adiff_dir': 'imports'}])
def test_invalid_desk_settings_are_rejected(desk):
    with pytest.raises(ConfigError, match=r'\[desk\]'):
        config.parse(base() | {'desk': desk})
