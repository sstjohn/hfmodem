# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""What a host program reads when it opens the PTC port.

Pat discards the start banner. RMS Trimode waits for `cmd:`. Winlink Express
waits for the `cmd: ` prompt, reacts to a few fixed replies (`pac: `, `INVALID`,
`ERROR`, `NOT OK`, `* `), and identifies the modem by %V, whose major version
must be at least 4. The banner keeps the manual's layout, model and version lines; its
last line names this program.
"""
from hfmodem.core import hostmode
from hfmodem.shrike import ptc
from hfmodem.shrike.ptc import PtcHost, SimPeer


def test_open_shows_the_banner_then_the_prompt_clients_wait_for():
    opened = PtcHost(SimPeer(), mycall="W9SSJ").open().decode("latin-1")
    assert opened == ptc.BANNER + ptc.PROMPT and opened.endswith("\r\ncmd: ")
    lines = ptc.BANNER.split("\r\n")[1:-1]
    assert len(lines) == 6 and lines[0] == lines[-1] == "*" * 40
    assert all(line.startswith("*") and line.endswith("*") for line in lines)
    assert "PTC-IIIusb" in lines[1] and "Version V.4.1" in lines[2]
    assert "BIOS Version 2.90" in lines[3]


def test_banner_carries_no_reply_a_client_acts_on_and_no_copyright_line():
    for reply in ("cmd:", "pac:", "INVALID", "ERROR", "NOT OK", "COMPATIBILITY MODE"):
        assert reply not in ptc.BANNER
    assert "(C)" not in ptc.BANNER and "GmbH" not in ptc.BANNER


def test_percent_v_reports_a_version_winlink_express_accepts():
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.open()
    host.feed(b"PTCH 31\r")
    host.feed(b"JHOST4\r")
    (resp,) = hostmode.Decoder("master").feed(host.feed(hostmode.command(31, "%V", counter=1)))
    text = resp.data.decode("ascii")
    assert resp.code == hostmode.MSG and text == "4.1 2.90"
    assert int(text.split(".")[0]) >= 4
