# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Transport-level conformance for the PTC emulation: framing and mode re-entry.

Covers the two creance acceptance items that the session test does not reach:
a payload spanning many 256-byte chunks and containing every byte value (so the
0xAA header byte must survive stuffing), and JHOST0/JHOST4 round-tripping so a
client that reconnects does not need the process restarted.

    python3 tests/shrike/test_ptc_transport.py
"""
from __future__ import annotations

import sys

from hfmodem.shrike import hostmode
from hfmodem.shrike.ptc import PtcHost, SimPeer

# Sibling test module, imported as part of the `tests` package. Inserting this
# file's own directory on sys.path worked but made the import depend on how the
# module was launched; `tests` is a package, so name it as one.
from hfmodem.tests.shrike.test_ptc import PACTOR_CH, Master, check


def open_hostmode(m: Master, host: PtcHost) -> None:
    host.open()
    for line in ("MYcall N0CALL", f"PTCH {PACTOR_CH}", "CONType 3", "MODE 0"):
        m.terminal(line)
    m.terminal("JHOST4")


def main() -> int:
    print("PTC-IIIusb transport conformance\n")

    # -- framing: every byte value, many chunks ---------------------------
    # 0xAA is the hostmode header byte, so a payload carrying it exercises the
    # stuff/unstuff path that ASCII traffic never touches.
    peer = SimPeer()
    host = PtcHost(peer, mycall="N0CALL")
    m = Master(host)
    open_hostmode(m, host)
    m.cmd(PACTOR_CH, "C N0DX")
    for _ in range(40):
        host.tick()
        if m.link_status()[5] == 4:
            break
    m.drain()

    payload = bytes(range(256)) * 5
    chunks = [payload[i:i + hostmode.MAX_DATA]
              for i in range(0, len(payload), hostmode.MAX_DATA)]
    check("payload spans many chunks", len(chunks) >= 5, f"{len(chunks)} chunks")
    check("payload carries the header byte", payload.count(0xAA) >= 5,
          f"{payload.count(0xAA)} occurrences of 0xAA")

    for i, chunk in enumerate(chunks):
        check(f"chunk {i} accepted", m.write(PACTOR_CH, chunk).code == hostmode.OK)
    for _ in range(400):
        host.tick()
        if m.link_status()[2:4] == [0, 0]:
            break
    check("all frames transmitted and acknowledged", m.link_status()[2:4] == [0, 0],
          str(m.link_status()))
    check("peer received the payload byte-exact", bytes(peer.received) == payload,
          f"{len(peer.received)}/{len(payload)} bytes")

    # -- mode re-entry ----------------------------------------------------
    m.cmd(0, "JHOST0")
    check("JHOST0 leaves hostmode", not host.hostmode)
    check("terminal mode answers again", "PTC-IIIusb" in m.terminal("VERsion"))
    m.terminal("JHOST4")
    check("JHOST4 re-enters hostmode", host.hostmode)
    check("hostmode still serves the extended poll", isinstance(m.poll_channels(), list))

    print("\nALL PASS")
    return 0


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
