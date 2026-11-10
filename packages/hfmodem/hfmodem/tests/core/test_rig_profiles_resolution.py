# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline selection and chain isolation for the station profiles."""
from __future__ import annotations

import copy

import pytest

from hfmodem.core import config, rigs


def station():
    return {"schema": 1,
            "station": {"control": "local", "regulatory": "unregulated",
                        "because": "offline fixture"},
            "rig": {"model": "ft891", "ptt": {"port": "mobile-line"}},
            "audio": {"input": "mobile ADC", "output": "mobile DAC",
                      "input_gain": .04, "tx_drive": .6}}


def test_name_precedence_and_unknown_names():
    assert rigs.resolve_rig(environ={}) == "ft891"
    assert rigs.resolve_rig(environ={"HFMODEM_RIG": ""}, legacy_model="g90") == "g90"
    assert rigs.resolve_rig(environ={"HFMODEM_RIG": "ftdx10"}, legacy_model="ft891") == "ftdx10"
    assert rigs.resolve_rig("x6100", environ={"HFMODEM_RIG": "ftdx10"}) == "x6100"
    with pytest.raises(ValueError, match="unknown rig"):
        rigs.resolve_rig(environ={"HFMODEM_RIG": "wrong"})


def test_ftdx10_ignores_losing_mobile_chain_and_ptt(capsys):
    raw = station()
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert cfg.selected_rig == "ftdx10" and cfg.rig_source == "HFMODEM_RIG"
    assert cfg.audio_chain == "ftdx10_usb"
    assert cfg.audio.input == cfg.audio.output == "USB AUDIO  CODEC"
    assert cfg.audio.input_gain is cfg.audio.tx_drive is cfg.audio.rx_floor_dbfs is None
    assert cfg.audio.tx_latency_ms is None
    assert cfg.rig.ptt.port == "/dev/cu.usbserial-01D13C961" and cfg.rig.model == "ftdx10"
    assert cfg.resolved_profile["cat_port"].endswith("960")
    assert cfg.resolved_profile["endpoint_status"] == "CONFIRMED"
    assert "2026-10-02" in cfg.resolved_profile["endpoint_evidence"]
    assert cfg.resolved_profile["tx_latency_ms"] == 7
    assert cfg.resolved_profile["tx_latency_status"] == "MEASURED"
    cfg.require_audio("transmit", input=True, output=True, drive=True)
    cfg.require_audio("preflight", floor=True)
    assert cfg.audio.drive_for() == .6
    assert "incomplete FTDX10 audio configuration" in capsys.readouterr().err


def test_selected_chain_and_endpoint_sources():
    raw = station()
    raw["rig"]["profiles"] = {"ftdx10": {"cat_port": "configured-cat",
                                         "ptt_port": "verified-line",
                                         "settle": .06, "tr_switch_s": .08,
                                         "tr_recovery_slack_s": .02,
                                         "tx_latency_ms": 4}}
    raw["audio"]["rate"] = 44100
    raw["audio"]["chains"] = {"ftdx10_usb": {
        "input": "station ADC", "output": "station DAC", "input_gain": .4,
        "gain_range": [.2, .8], "tx_drive": 0, "p1_drive": .2,
        "rx_floor_dbfs": -43, "tx_latency_ms": 0,
        "calibration_status": "PROVISIONAL", "calibration_evidence": "bench note"}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10", "CAT_PORT": "exported-cat"})
    assert cfg.resolved_profile["cat_port"] == "exported-cat"
    assert cfg.resolution_sources["cat_port"] == "CAT_PORT"
    assert cfg.resolution_sources["ptt_port"] == "station-toml"
    assert cfg.rig.ptt.port == "verified-line" and cfg.rig.ptt.settle_s == .06
    assert cfg.audio.rate == 44100 and cfg.audio.gain_range == (.2, .8)
    assert cfg.audio.drive_for() == 0 and cfg.audio.drive_for("p1") == .2
    assert cfg.audio.drive_for("p1", 0) == 0
    cfg.require_audio("transmit", input=True, output=True, drive=True, floor=True)
    name, values, sources = rigs.resolve_profile("ftdx10", environ={"PTT_PORT": "exported-line"},
        profiles=raw["rig"]["profiles"], cat_port="flag-cat")
    assert (name, values["cat_port"], values["ptt_port"]) == ("ftdx10", "flag-cat", "exported-line")
    assert sources["cat_port"] == "operator-flag" and sources["ptt_port"] == "PTT_PORT"


@pytest.mark.parametrize("field,value", [
    ("input_gain", -.1), ("input_gain", 1.1), ("tx_drive", -1),
    ("tx_drive", 1.01), ("rx_floor_dbfs", 0), ("rx_floor_dbfs", "quiet"),
    ("tx_latency_ms", 61), ("gain_range", [0.5, .2]),
    ("p1_drive", float("nan")), ("p3_drive", True), ("p4_drive", 2),
    ("input_gain", float("inf")),
])
def test_invalid_chain_values_warn_without_mobile_fallback(field, value, capsys):
    raw = station()
    raw["audio"]["chains"] = {"ftdx10_usb": {field: value}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert getattr(cfg.audio, field) is None
    cfg.require_audio("transmit", input=True, output=True, drive=True, floor=True)
    assert field in capsys.readouterr().err


def test_unknown_keys_and_bad_overlay_warn(capsys):
    raw = station()
    raw["audio"]["chains"] = {"ftdx10_usb": {"input_gian": .4}}
    config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert "input_gian" in capsys.readouterr().err
    raw = station()
    raw["rig"]["profiles"] = {"ftdx10": {"settle": float("nan")}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert cfg.rig.ptt.settle_s == .04
    assert "settle" in capsys.readouterr().err
    raw = station()
    raw["audio"]["chains"] = {"ftdx10_usb": {"calibration_status": "MEASURED"}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert cfg.audio.calibration_status == "PROVISIONAL"
    assert "calibration_evidence" in capsys.readouterr().err


def test_xiegu_rows_use_same_unconfigured_chain_resolution():
    for name in ("x6100", "g90"):
        cfg = config.parse(station(), environ={"HFMODEM_RIG": name})
        assert cfg.audio_chain == rigs.RIGS[name]["audio_chain"]
        assert cfg.audio.input == cfg.audio.output == ""
        assert cfg.rig.ptt.port == ""
        assert cfg.resolved_profile["tx_latency_status"] == "UNMEASURED"


def test_resolution_does_not_mutate_profiles():
    original = copy.deepcopy(rigs.RIGS)
    rigs.resolve_profile("ftdx10", profiles={"ftdx10": {"settle": .09}})
    assert rigs.RIGS == original


def test_no_station_file_warns_and_supplies_provisional_drive(monkeypatch, capsys):
    monkeypatch.setenv("HFMODEM_RIG", "ftdx10")
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    assert config.tx_drive() == .6
    assert "provisional" in capsys.readouterr().err
    assert config.tx_drive(0) == 0


def test_optional_mobile_calibration_keeps_audio_defaults_shape():
    raw = station()
    raw["audio"].update(gain_range=[.02, .08], rx_floor_dbfs=-34.9,
                        p1_drive=.75, calibration_status="MEASURED",
                        calibration_evidence="mobile bench log")
    cfg = config.parse(raw, environ={})
    assert cfg.audio.input_gain == .04 and cfg.audio.tx_drive == .6
    assert cfg.audio_calibration["gain_range"] == (.02, .08)
    assert cfg.audio_calibration["p1_drive"] == .75
    raw["audio"]["input_gain"] = .01
    cfg = config.parse(raw, environ={})
    assert cfg.audio.input_gain == .01


def test_mobile_endpoint_overlay_is_optional_and_environment_wins():
    raw = station()
    raw["rig"]["profiles"] = {"ft891": {"ptt_port": "profile-line", "settle": .05}}
    cfg = config.parse(raw, environ={})
    assert cfg.rig.ptt.port == "profile-line" and cfg.rig.ptt.settle_s == .05
    cfg = config.parse(raw, environ={"PTT_PORT": "exported-line"})
    assert cfg.rig.ptt.port == "exported-line"
    assert cfg.resolution_sources["ptt_port"] == "PTT_PORT"


def test_device_name_is_required_only_when_no_default_exists(capsys):
    cfg = config.parse(station(), environ={"HFMODEM_RIG": "x6100"})
    cfg.require_audio("inspect", drive=True, floor=True)
    with pytest.raises(config.ConfigError, match="device cannot be named: input"):
        cfg.require_audio("capture", input=True)
    assert "warning:" in capsys.readouterr().err


def test_flight_defaults_use_confirmed_devices_without_station_entries(tmp_path, monkeypatch):
    from argparse import Namespace
    path = tmp_path / "station.toml"
    path.write_text('schema = 1\n[station]\ncontrol = "local"\nregulatory = "unregulated"\nbecause = "offline fixture"\n[rig]\nmodel = "ft891"\n')
    monkeypatch.setenv("HFMODEM_RIG", "ftdx10")
    monkeypatch.delenv("CAT_PORT", raising=False)
    monkeypatch.delenv("PTT_PORT", raising=False)
    # What is pinned here is the station's own ports as resolved, which every
    # other test run refuses; nothing below opens them.
    monkeypatch.setattr(rigs, "refuse_device_in_test", lambda *ports: None)
    for station_path in (None, str(path)):
        if station_path:
            monkeypatch.setenv("HFMODEM_STATION", station_path)
        else:
            monkeypatch.delenv("HFMODEM_STATION", raising=False)
        args = rigs.flight_defaults(Namespace(rig=None, serial=None, ptt_port=None,
                                              audio_in=None, audio_out=None))
        assert args.serial == "/dev/cu.usbserial-01D13C960"
        assert args.ptt_port == "/dev/cu.usbserial-01D13C961"
        assert args.audio_in == args.audio_out == "USB AUDIO  CODEC"


@pytest.mark.parametrize("field,value", [
    ("input", 42), ("rate", 0), ("blocksize", True), ("latency", "invalid"),
    ("calibration_status", "invalid"), ("calibration_evidence", 1),
    ("rx_hot_rms", float("nan")), ("rx_silent_rms", .9),
    ("peak_limit_dbfs", -7), ("peak_target_dbfs", 0),
    ("input_gain", .9),
])
def test_other_chain_checks_warn_and_preflight_continues(field, value, monkeypatch, capsys):
    from hfmodem.station import preflight
    raw = station()
    raw["audio"]["chains"] = {"ftdx10_usb": {"gain_range": [.1, .2], field: value}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    monkeypatch.setattr(preflight.Preflight, "run", lambda self: 0)
    assert preflight.run(cfg) == 0
    assert cfg.audio.input == cfg.audio.output == "USB AUDIO  CODEC"
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("settle", 0), ("tr_switch_s", float("nan")), ("tr_recovery_slack_s", -1),
    ("tx_latency_ms", 61), ("cat_port", 42), ("ptt_port", True),
    ("calibration_status", "invalid"), ("calibration_evidence", 42),
    ("unknown", 1),
])
def test_other_profile_checks_warn(field, value, capsys):
    name, profile, sources = rigs.resolve_profile("ftdx10", environ={},
        profiles={"ftdx10": {field: value}})
    assert name == "ftdx10"
    assert field in capsys.readouterr().err
    if field in rigs.RIGS[name]:
        assert profile[field] == rigs.RIGS[name][field]
        assert sources[field] == "rig-profile"


def test_protocol_tables_do_not_replace_selected_rig_name():
    raw = station()
    raw["protocols"] = {"pactor": {"enabled": False}}
    cfg = config.parse(raw, environ={"HFMODEM_RIG": "ftdx10"})
    assert cfg.selected_rig == "ftdx10"
