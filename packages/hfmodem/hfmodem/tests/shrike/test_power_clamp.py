# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The launcher holds its own RFPOWER writes to the station's `[rig] power_clamp_w`."""
from pathlib import Path

import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_armdefaults import _armed

EXAMPLE = Path(__file__).resolve().parents[5] / "examples" / "station.toml"


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_PORT", "/dev/null")
    monkeypatch.setenv("PTT_PORT", "/dev/null")


@pytest.fixture
def clamped(tmp_path, monkeypatch):
    path = tmp_path / "station.toml"
    path.write_text(EXAMPLE.read_text().replace("[rig]\n", "[rig]\npower_clamp_w = 50\n", 1))
    monkeypatch.setenv("HFMODEM_STATION", str(path))


def test_a_phase_level_above_the_clamp_is_held_to_it(clamped, capsys):
    args = _armed("--p3-watts", "100")
    with pytest.raises(SystemExit, match="--transmit"):
        onair.run(args)
    assert args.p3_watts == 50
    assert ("warning: --p3-watts asks 100 W, above the 50 W station clamp ([rig] power_clamp_w); "
            "setting 50 W") in capsys.readouterr().out


def test_without_a_clamp_the_phase_levels_are_the_operators(monkeypatch, capsys):
    monkeypatch.setenv("HFMODEM_STATION", str(EXAMPLE))
    args = _armed("--p3-watts", "100")
    with pytest.raises(SystemExit, match="--transmit"):
        onair.run(args)
    assert args.p3_watts == 100 and "clamp" not in capsys.readouterr().out


def station(tmp_path, monkeypatch, rig_lines):
    path = tmp_path / "station.toml"
    path.write_text(EXAMPLE.read_text().replace("[rig]\n", "[rig]\n" + rig_lines, 1))
    monkeypatch.setenv("HFMODEM_STATION", str(path))


class Rig:
    def __init__(self, baseline="1.0"):
        self.levels, self.oneshots, self.baseline = [], [], baseline

    def set_power(self, level):
        self.levels.append(level)
        return True

    def set_power_once(self, level):
        self.oneshots.append(level)
        return True

    def get_power(self):
        return self.baseline

    def ptt(self, on):
        pass


@pytest.mark.parametrize("rig_lines, levels", [
    ("power_clamp_w = 25\n", [0.10, 0.25]),
    ("", [0.10, 0.50]),
    ("power_clamp_w = 5\n", [0.05, 0.05]),
    ("max_watts = 200\npower_clamp_w = 50\n", [0.05, 0.25]),
    ("tune_power_w = 20\n", [0.20, 0.50]),
    ("tune_power_w = 40\npower_clamp_w = 25\n", [0.25, 0.25]),
])
def test_the_tune_carrier_and_its_restore_are_held_to_the_clamp(rig_lines, levels, tmp_path, monkeypatch):
    station(tmp_path, monkeypatch, rig_lines)
    monkeypatch.setattr(onair.ota, "_play", lambda *a, **k: None)
    rig = Rig()
    onair.tune_carrier(rig, None, 0.01, 0.1)
    assert rig.levels == pytest.approx(levels)


def test_the_arm_start_level_is_restored_no_higher_than_the_clamp(tmp_path, monkeypatch, capsys):
    station(tmp_path, monkeypatch, "power_clamp_w = 50\n")
    rig = Rig(baseline="1.0")
    power = onair._PhasePower(rig, p1=20.0, p3=None)
    power.select(onair.Protocol.PACTOR1, "the call")
    power.restore()
    assert rig.levels == [0.20] and rig.oneshots == [0.50]
    assert ("warning: the arm-start level restored asks 100 W, above the 50 W station clamp"
            in capsys.readouterr().out)


def test_phase_levels_follow_the_station_full_scale(tmp_path, monkeypatch):
    station(tmp_path, monkeypatch, "max_watts = 200\n")
    rig = Rig(baseline="0.5")
    power = onair._PhasePower(rig, p1=50.0, p3=None)
    power.select(onair.Protocol.PACTOR1, "the call")
    power.restore()
    assert rig.levels == [0.25] and rig.oneshots == [0.5]


def test_a_rig_whose_tuner_ignores_a_carrier_keys_none(tmp_path, monkeypatch, capsys):
    station(tmp_path, monkeypatch, "")
    monkeypatch.setenv("HFMODEM_RIG", "ftdx10")
    monkeypatch.setattr(onair.ota, "_play", lambda *a, **k: pytest.fail("keyed a TUNE carrier"))
    rig = Rig()
    onair.tune_carrier(rig, None, 0.01, 0.1)
    assert rig.levels == []
    assert "TUNE: nothing keyed -- this rig's tuner does not search on a carrier" in capsys.readouterr().out


def test_a_tune_power_above_the_clamp_is_said(tmp_path, monkeypatch, capsys):
    station(tmp_path, monkeypatch, "tune_power_w = 40\npower_clamp_w = 25\n")
    monkeypatch.setattr(onair.ota, "_play", lambda *a, **k: None)
    onair.tune_carrier(Rig(), None, 0.01, 0.1)
    assert "warning: the TUNE carrier asks 40 W, above the 25 W station clamp" in capsys.readouterr().out
