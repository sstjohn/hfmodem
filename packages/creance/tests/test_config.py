# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Config loading: the shipped example parses; every validation error fires."""

from pathlib import Path

import pytest

from creance.config import Config, ConfigError, load

EXAMPLE = Path(__file__).parent.parent / "examples" / "site-home.toml"

MINIMAL = """
[site]
name = "test"
mycall = "N0CAL"

[[modem]]
name = "kestrel"
cmd_port = 8300
data_port = 8301
"""


def load_text(tmp_path, text) -> Config:
    p = tmp_path / "site.toml"
    p.write_text(text)
    return load(p)


def test_example_config(tmp_path):
    cfg = load(EXAMPLE)
    assert cfg.site.mycall == "N0CAL"
    assert [m.name for m in cfg.modems] == ["kestrel", "sabir-A", "sabir-B"]
    # Sabir uses one framed-CBOR socket per station.
    assert [m.dialect for m in cfg.modems[-2:]] == ["hostapi", "hostapi"]
    assert all(m.data_port == 0 for m in cfg.modems[-2:])

    kestrel = cfg.modem("kestrel")
    assert kestrel.spawn is not None and kestrel.spawn[0] == "python3"
    assert kestrel.spawn[2] == "hfmodem.kestrel.host.run_server"
    assert kestrel.quirks.preconnect_queue is True       # LoopbackModem buffers pre-connect writes (spec §7.4)
    assert kestrel.quirks.version_reply is True          # default

    a = cfg.modem("sabir-A")
    assert a.spawn_group == "sabir-pair"
    assert (a.cmd_port, a.data_port) == (8320, 0)
    group = cfg.spawn_groups[0]
    assert group.name == "sabir-pair"
    assert "--ports" in group.spawn

    assert cfg.responder.pending_timeout_s == 30
    assert cfg.responder.hello_timeout_s == 10
    assert cfg.responder.max_session_s == 900
    assert cfg.initiator.connect_timeout_s == 90
    assert cfg.initiator.max_session_s == 900
    assert cfg.rig is None


def test_minimal_defaults(tmp_path):
    cfg = load_text(tmp_path, MINIMAL)
    assert cfg.responder.on_plain_peer == "sink"
    assert cfg.initiator.on_plain_peer == "disconnect"
    assert cfg.site.results_dir == "results"
    m = cfg.modem("kestrel")
    assert (m.bandwidth, m.compression) == ("2300", "OFF")
    assert m.quirks.iamalive_s == 60.0
    assert m.quirks.hello_timeout_s is None
    with pytest.raises(KeyError):
        cfg.modem("shrike")


def test_retention_caps_come_from_config(tmp_path):
    # they used to be __init__ kwargs nothing set, so a remote disk filled
    assert load_text(tmp_path, MINIMAL).responder.retention_max_age_s == 30 * 86400
    cfg = load_text(tmp_path, MINIMAL + "[responder]\n"
                    "retention_max_age_s = 3600\n"
                    "retention_max_bytes = 1048576\n"
                    "retention_interval_s = 60\n")
    assert (cfg.responder.retention_max_age_s,
            cfg.responder.retention_max_bytes,
            cfg.responder.retention_interval_s) == (3600, 1048576, 60)


def test_sim_time_flag(tmp_path):
    cfg = load(EXAMPLE)
    assert cfg.modem("kestrel").sim_time is False       # default
    assert cfg.modem("sabir-A").sim_time is True       # tagged in the example
    assert cfg.modem("sabir-B").sim_time is True
    parsed = load_text(tmp_path, MINIMAL.replace(
        "data_port = 8301", "data_port = 8301\nsim_time = true"))
    assert parsed.modem("kestrel").sim_time is True


def test_rig_section(tmp_path):
    cfg = load_text(tmp_path, MINIMAL + '\n[rig]\nport = 4533\n')
    assert cfg.rig.host == "127.0.0.1"
    assert cfg.rig.port == 4533


@pytest.mark.parametrize("text,match", [
    ("[site]\nname = 't'\nmycall = 'X'\n", "no \\[\\[modem\\]\\] entries"),
    ("[[modem]]\nname = 'm'\ncmd_port = 1\ndata_port = 2\n", "missing \\[site\\]"),
    (MINIMAL.replace('mycall = "N0CAL"\n', ""), "mycall"),
    (MINIMAL + "[[modem]]\nname = 'kestrel'\ncmd_port = 1\ndata_port = 2\n",
     "duplicate modem name"),
    (MINIMAL + "[[modem]]\nname = 'other'\ncmd_port = 8300\ndata_port = 9\n",
     "port 8300 used by both"),
    (MINIMAL.replace("data_port = 8301", "data_port = 8300"),
     "port 8300 used by both"),
    (MINIMAL + "[[modem]]\ncmd_port = 1\ndata_port = 2\n", "missing name"),
    (MINIMAL + "[[modem]]\nname = 'h'\ncmd_port = 1\ndata_port = 2\nspawn_group = 'ghost'\n",
     "spawn_group 'ghost' is not defined"),
    (MINIMAL.replace("cmd_port = 8300",
                     "cmd_port = 8300\nspawn = ['x']\nspawn_group = 'g'")
     + "[spawn_group.g]\nspawn = ['y']\n", "mutually exclusive"),
    (MINIMAL.replace("cmd_port = 8300", "cmd_port = 8300\nbandwidth = '9999'"),
     "bandwidth must be one of"),
    (MINIMAL.replace("cmd_port = 8300", "cmd_port = 8300\ncompression = 'ZIP'"),
     "compression must be one of"),
    (MINIMAL + "[responder]\non_plain_peer = 'panic'\n", "on_plain_peer must be one of"),
    (MINIMAL + "[initiator]\non_plain_peer = 'sink'\n", "on_plain_peer must be one of"),
    (MINIMAL.replace("cmd_port = 8300", "cmd_port = 8300\nspawn = []"),
     "non-empty list of strings"),
    (MINIMAL + "[spawn_group.g]\ncwd = '/tmp'\n", "missing spawn"),
    (MINIMAL + "[haunted]\nx = 1\n", "unknown section"),
    (MINIMAL.replace("cmd_port = 8300", "cmd_port = 8300\ncolour = 'red'"),
     "unknown key"),
    (MINIMAL + "[modem.quirks]\nchatty = true\n", "unknown key"),
    ("not toml [", "site.toml"),
])
def test_validation_errors(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_text(tmp_path, text)
