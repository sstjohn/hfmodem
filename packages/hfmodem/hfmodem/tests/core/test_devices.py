# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Device resolution, and the refusal that keeps a transmit run off the laptop's
own speakers. No sound card is opened here."""
from __future__ import annotations

import sys
import types

import pytest

from hfmodem.core.devices import find_device


def _fake_sd(monkeypatch, devices):
    sd = types.ModuleType("sounddevice")
    sd.query_devices = lambda: devices
    monkeypatch.setitem(sys.modules, "sounddevice", sd)


@pytest.mark.parametrize("kind,word", [("out", "speakers"), ("in", "microphone")])
def test_a_transmit_path_refuses_the_system_default(kind, word):
    """The failure this exists for is silent: the default device is the laptop's
    own, so a missing flag records the room and modulates the speakers, and every
    symptom of that reads as a radio fault."""
    with pytest.raises(SystemExit) as exc:
        find_device(None, kind, required=True)
    assert word in str(exc.value) and f"--audio-{kind}" in str(exc.value)


def test_unspecified_is_the_default_when_it_is_not_required():
    assert find_device(None, "out") is None
    assert find_device(None, "in") is None


def test_an_index_resolves_without_asking_the_host():
    """No `sounddevice` is injected: an index and the refusal above must work on a
    machine with no audio libraries at all."""
    assert find_device("3", "out") == 3
    assert find_device(2, "in") == 2


def test_a_name_substring_matches_a_device_with_channels_of_that_kind(monkeypatch):
    _fake_sd(monkeypatch, [
        {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
        {"name": "USB Audio Device", "max_input_channels": 2, "max_output_channels": 2},
    ])
    assert find_device("usb audio", "out") == 1
    assert find_device("built-in", "in") == 0


def test_a_device_with_no_channels_of_that_kind_is_not_a_match(monkeypatch):
    _fake_sd(monkeypatch, [
        {"name": "USB Audio Device", "max_input_channels": 2, "max_output_channels": 0},
    ])
    with pytest.raises(SystemExit, match="is not an out device -- it is in-only"):
        find_device("usb audio", "out")


def test_an_output_device_is_matched_on_output_channels():
    """`kind` is "in" or "out". Passing "output" reads as correct and matched on
    *input* channels instead, so a transmit-only interface never resolved and the
    refusal told the operator `--audio-in is required` for the output device.

    The rig's codec is both, which is why this survived: it resolved by accident
    on the only hardware anyone tried.
    """
    sd = pytest.importorskip(
        "sounddevice",
        reason="install the `radio` extra (pip install 'hfmodem[radio]') — the "
               "sound card is optional and everything else runs on recordings")
    out_only = [i for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] > 0 and d["max_input_channels"] == 0]
    if not out_only:
        pytest.skip("no output-only device on this machine")
    name = sd.query_devices()[out_only[0]]["name"]
    assert find_device(name, "out") == out_only[0]
    with pytest.raises(ValueError, match="'in' or 'out'"):
        find_device(name, "output")


def test_the_station_asks_for_devices_the_way_find_device_answers():
    """The two are a matched pair and were not matched."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "station" / "process.py").read_text()
    kinds = [n.args[1].value for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "find_device"]
    assert kinds and all(k in ("in", "out") for k in kinds), kinds
