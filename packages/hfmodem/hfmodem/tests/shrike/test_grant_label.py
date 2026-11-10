# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The transmit line names the two unassigned words, not a CS5/CS6 that PACTOR-1
does not have."""
import pytest

from hfmodem.shrike import onair, pactor1


@pytest.mark.parametrize("index,label", [
    (pactor1.CS_59A, "0x59A"), (pactor1.CS_6A9, "0x6A9"), (0, "P1 CS1")])
def test_a_keyed_grant_is_named(tmp_path, capsys, index, label):
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    tx.send_p1_cs(index)
    line = next(l for l in capsys.readouterr().out.splitlines() if "TX[1]" in l)
    assert f"TX[1] {label} " in line, line
