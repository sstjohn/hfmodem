# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The rig table. What is asserted here is that it is complete, because the way
it went wrong was a copy that silently lacked a field its readers defaulted."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hfmodem.core.rigs import RIGS


@pytest.mark.parametrize("name", sorted(RIGS))
def test_every_rig_carries_what_a_keying_path_reads(name):
    """A reader that has to `.get("settle", ...)` substitutes a value measured on
    another radio. The FT-891 spent a while keying on the X6100's 0.40 that way."""
    r = RIGS[name]
    assert {"model", "mode", "settle"} <= set(r)
    assert isinstance(r["model"], int) and r["settle"] > 0
    # The generic row delegates CAT to an operator-managed daemon and supplies
    # no keying line (§2.2). Its serial settings are not facts in this table.
    if name == "generic" and not r.get("ptt_port"):
        assert r["cat"] == "rigctld"
        return
    assert {"baud", "note"} <= set(r)
    assert r["baud"] > 0


def test_the_settle_values_are_the_measured_ones():
    assert RIGS["ft891"]["settle"] == 0.04       # external dongle, no codec re-init
    assert RIGS["x6100"]["settle"] == 0.40       # its USB codec re-initialises on key


def test_only_a_rig_with_a_second_serial_interface_declares_one():
    for name, r in RIGS.items():
        if r.get("ptt_type") == "RTS":
            assert "ptt_iface" in r, name
        else:
            assert "ptt_iface" not in r, name


def test_the_shell_rig_lookup_runs_silently(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k not in {"HFMODEM_STATION", "HFMODEM_RIG", "CAT_PORT", "PTT_PORT"}}
    env.update(CAT_PORT=str(tmp_path / "cat"), PTT_PORT=str(tmp_path / "ptt"))
    run = subprocess.run([sys.executable, "-m", "hfmodem.core.rigs"], env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0 and run.stderr == "", run
