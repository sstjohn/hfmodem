# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Protocol checks run at config load, before constructing a station."""
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from hfmodem.core import config

EXAMPLE = Path(__file__).resolve().parents[5] / "examples" / "station.toml"


def _raw(**settings):
    raw = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["protocols"] = {"bateleur": dict(
        enabled=True, host="s5066", addresses=["0.0.0.1"], **settings)}
    return raw


@pytest.mark.parametrize("mode", ["", "110 1200L", "4285 1200L", "4539 3200S",
                                  "39 600/33/A10.8", "16 1200", " 110 1200l ",
                                  "WBHF 600S", "4529 600L", "FSK 75W"])
def test_s5066_accepts_modem_modes_without_changing_the_config(mode):
    p = config.parse(_raw(mode=mode)).protocols["bateleur"]
    assert p.mode == mode
    assert p.addresses == ["0.0.0.1"]


@pytest.mark.parametrize("address", ["0.0.0.0", "15.255.255.255", "1", "2.3.4"])
def test_s5066_accepts_the_existing_address_forms(address):
    raw = _raw()
    raw["protocols"]["bateleur"]["addresses"] = [address]
    assert config.parse(raw).protocols["bateleur"].addresses == [address]


@pytest.mark.parametrize(("settings", "message"), [
    ({"addresses": []}, 'host = "s5066" takes one address in addresses, '
     'the node\'s own ("0.0.0.1" and the like)'),
    ({"addresses": ["1", "2"]}, 'host = "s5066" takes one address in addresses, '
     'the node\'s own ("0.0.0.1" and the like)'),
    ({"addresses": ["16.0.0.0"]}, "addresses: an address is w.x.y.z with w 0-15 "
     "and the rest 0-255: '16.0.0.0'"),
    ({"addresses": ["0.0.0.256"]}, "addresses: an address is w.x.y.z with w 0-15 "
     "and the rest 0-255: '0.0.0.256'"),
    ({"addresses": ["1.2.3.4.5"]}, "addresses: an address is w.x.y.z, not '1.2.3.4.5'"),
    ({"addresses": ["bad"]}, "addresses: invalid literal for int() with base 10: 'bad'"),
    ({"mode": "1200L"}, "mode: a mode is a standard and a name, not '1200L'"),
    ({"mode": "bogus 1200L"}, "mode: no standard 'BOGUS': 110, 4285, 4529, 4539, WBHF, 39, 16 or FSK"),
])
def test_s5066_rejects_invalid_config_at_parse_with_the_same_error(settings, message):
    raw = _raw()
    raw["protocols"]["bateleur"].update(settings)
    with pytest.raises(config.ConfigError) as exc:
        config.parse(raw)
    assert str(exc.value) == f"[protocols.bateleur]: {message}"


@pytest.mark.parametrize("settings", [{"enabled": False}, {"host": "bateleur"}])
def test_s5066_checks_only_apply_to_an_enabled_s5066_host(settings):
    raw = _raw(mode="bad")
    raw["protocols"]["bateleur"].update(addresses=[], **settings)
    p = config.parse(raw).protocols["bateleur"]
    assert p.mode == "bad" and p.addresses == []


def test_load_checks_s5066_in_a_fresh_process_without_a_station_import(tmp_path):
    path = tmp_path / "station.toml"
    # Preserve the station and device declarations, replacing only the protocols.
    text = EXAMPLE.read_text(encoding="utf-8").split("[protocols.", 1)[0]
    text += '\n[protocols.bateleur]\nenabled = true\nhost = "s5066"\n'
    path.write_text(text + 'addresses = ["0.0.0.1"]\n', encoding="utf-8")
    script = '''
import sys
from hfmodem.core.config import ConfigError, load
assert not any(n.startswith("hfmodem.station") for n in sys.modules)
path = sys.argv[1]
assert load(path).protocols["bateleur"].mode == ""
from pathlib import Path
with Path(path).open("a") as f:
    f.write('mode = "bad"\\n')
try:
    load(path)
except ConfigError as exc:
    assert str(exc) == "[protocols.bateleur]: mode: a mode is a standard and a name, not 'bad'"
else:
    raise AssertionError("invalid mode accepted")
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
