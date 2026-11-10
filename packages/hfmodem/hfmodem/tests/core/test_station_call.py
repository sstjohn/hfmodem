# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The callsign a tool signs with comes from the operator, never from the source.

`$MYCALL` first, as the launchers have it, then `[station] mycall`; with neither
the tool warns and carries on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hfmodem.core import config
from hfmodem.tests.evidence import require_input

_REPO = Path(__file__).resolve().parents[5]


@pytest.fixture
def station(tmp_path, monkeypatch):
    monkeypatch.delenv("MYCALL", raising=False)
    path = tmp_path / "station.toml"
    path.write_text((_REPO / "examples" / "station.toml").read_text().replace("N0CALL", "K1ABC"))
    monkeypatch.setenv(config.STATION_ENV, str(path))
    return path


def test_the_station_file_names_the_call(station, capsys):
    assert config.mycall() == "K1ABC"
    assert capsys.readouterr().err == ""


def test_the_operators_mycall_comes_first(station, monkeypatch):
    monkeypatch.setenv("MYCALL", "N0OVR")
    assert config.mycall() == "N0OVR"


def test_no_station_file_warns_and_goes_on(monkeypatch, capsys):
    monkeypatch.delenv("MYCALL", raising=False)
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    assert config.mycall() == ""
    assert "no callsign configured" in capsys.readouterr().err


def test_an_unreadable_station_file_warns_and_goes_on(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MYCALL", raising=False)
    monkeypatch.setenv(config.STATION_ENV, str(tmp_path / "missing.toml"))
    assert config.mycall() == ""
    err = capsys.readouterr().err
    assert "missing.toml" in err and "no callsign configured" in err


@pytest.mark.parametrize("tool, flag", [("ardop_call", "--mycall"), ("alc_knee", "--call")])
def test_tools_carry_no_callsign_of_their_own(tool, flag, monkeypatch):
    require_input(_REPO / "tools" / f"{tool}.py")
    monkeypatch.syspath_prepend(str(_REPO / "tools"))
    module = __import__(tool)
    argv = ["K0SI"] if tool == "ardop_call" else []
    assert getattr(module.parser().parse_args(argv), flag.lstrip("-").replace("-", "_")) is None


def test_the_station_file_names_the_grid(station, capsys):
    assert config.grid() == "AA00"
    assert capsys.readouterr().err == ""


def test_no_grid_warns_and_goes_on(monkeypatch, capsys):
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    assert config.grid() == ""
    assert "no grid configured" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["[station\nmycall =", "schema = 1\n[station]\nmycall = 5\n"])
def test_a_station_file_that_will_not_load_warns_and_goes_on(tmp_path, monkeypatch, capsys, text):
    """TOML syntax errors and validation errors alike: an advisory lookup never raises."""
    monkeypatch.delenv("MYCALL", raising=False)
    bad = tmp_path / "bad.toml"
    bad.write_text(text)
    monkeypatch.setenv(config.STATION_ENV, str(bad))
    assert config.mycall() == "" and config.grid() == ""
    assert "bad.toml" in capsys.readouterr().err
