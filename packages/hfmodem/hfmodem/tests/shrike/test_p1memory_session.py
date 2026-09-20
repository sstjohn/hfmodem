# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Memory ARQ where a link actually runs: through `onair._SessionRx`.

`p1rx.PacketMemory` was built, tested and never constructed outside its own test
file, so the live receiver read every copy of the peer's packet alone and threw
the failures away -- with no FEC in PACTOR-1 at all, that is the mode's whole
error-correction budget spent on nothing. `test_p1memory.py` proves the
combining; this proves the WIRING, which is a separate claim and the one that
was false.

The entry point is the production one. `deep_scan` is what the cycle calls
before its key, `_scan` picks the reader for the link's protocol, and
`_p1_packet` is the only place the session's memory can be handed over -- so a
test that calls `rxfront.decode_expected_p1_packet` directly would pass over a
receiver that never passes a memory at all, which is the state this file was
written against.

Two copies at a noise sigma past the single-shot cliff, one per shift sense: the
first must deliver nothing, the second must deliver the packet to the host. And
the memory is the LINK's, not the session's -- a disconnect and a changeover
both end the run of copies it is holding, so the same second copy must stand
alone across either.

Run:  python -m hfmodem.tests.shrike.test_p1memory_session
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import onair, p1rx, pactor1
from hfmodem.shrike.arq import IRS, ISS
from hfmodem.shrike.ptc import PtcHost

PAYLOAD = b"MEMARQ12"
SIGMA = 0.55
"""`test_p1memory`'s sigma against `packet_signal`'s 0.11-amplitude render: past
the single-shot cliff and inside combining's reach. Pinned, with the seeds."""
SEED = 5

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _copy(k: int) -> np.ndarray:
    """One ARQ cycle's receive window: copy `k` of the peer's packet, in noise.

    The shift inverts with every transmission, so consecutive copies arrive with
    MARK and SPACE exchanged -- which is the thing the memory has to undo from
    the data before a sum means anything."""
    burst = pactor1.packet_signal(PAYLOAD, 100, packet_count=1,
                                  invert=bool(k & 1), lead_s=0.1, tail_s=0.15)
    rng = np.random.default_rng(SEED * 1000 + k)
    return (burst + rng.normal(0, SIGMA, burst.size)).astype(np.float32)


class _Seam:
    """A transmit seam that records instead of keying."""

    def __init__(self) -> None:
        self.sent: list = []

    def attach(self, host) -> None:
        pass

    def connect_burst(self, mycall, dxcall) -> None:
        self.sent.append(("connect", dxcall))

    def send_cs(self, i) -> None:
        self.sent.append(("cs", i))

    def send_p1_cs(self, i) -> None:
        self.sent.append(("p1cs", i))

    def send_p1_packet(self, payload, baud, packet_count, **kw) -> None:
        self.sent.append(("p1pkt", payload))

    def send_packet(self, sl, payload, status, breakin=False) -> None:
        self.sent.append(("pkt", payload))

    def pump(self) -> None:
        pass

    def cycle(self) -> None:
        pass


def _linked() -> PtcHost:
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    host.peer.sent.clear()
    return host


def _received(host: PtcHost) -> bytes:
    return bytes(host.channel(host.ptchn).rx)


def the_copies_fail_alone() -> None:
    """The premise, asserted rather than presumed."""
    print("\ntwo copies of one packet, each past the single-shot cliff")
    alone = [bool(p1rx.decode_p1_packets(_copy(k))) for k in (0, 1)]
    check("neither copy decodes on its own", not any(alone), str(alone))


def the_session_combines_the_repeat() -> None:
    print("\nthe second copy, through the receiver the cycle actually calls")
    host = _linked()
    rx = onair._SessionRx(host, tag="TEST")

    rx.new_cycle()
    rx.deep_scan(_copy(0))
    check("the first copy delivers nothing", rx.count == 0 and not _received(host),
          repr(_received(host)[:24]))

    rx.new_cycle()
    rx.deep_scan(_copy(1))
    check("the second copy delivers the packet", _received(host) == PAYLOAD,
          repr(_received(host)[:24]))
    check("...and it was answered with a PACTOR-1 control signal",
          any(k == "p1cs" for k, _ in host.peer.sent), str(host.peer.sent))
    check("the deliverer was the session's own memory, not a second look",
          rx.count == 1, f"{rx.count} frames through the FSM")


def the_memory_does_not_outlive_the_link() -> None:
    """A disconnect ends the run of copies: the packet is not repeated any more,
    and the next link's first copy stands alone."""
    print("\nacross a disconnect")
    host = _linked()
    rx = onair._SessionRx(host, tag="TEST")
    rx.new_cycle()
    rx.deep_scan(_copy(0))

    host.arq._finish_disconnected()
    rx.new_cycle()
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    rx.new_cycle()
    rx.deep_scan(_copy(1))
    check("the copy held before the link went down is gone",
          rx.count == 0 and not _received(host), repr(_received(host)[:24]))


def the_memory_does_not_cross_a_changeover() -> None:
    """The same rule from the other direction: once we are the sending station
    the copies stop arriving, so the run the memory is holding has ended."""
    print("\nacross a changeover")
    host = _linked()
    rx = onair._SessionRx(host, tag="TEST")
    rx.new_cycle()
    rx.deep_scan(_copy(0))

    host.arq.role = ISS
    rx.new_cycle()
    host.arq.role = IRS
    rx.new_cycle()
    rx.deep_scan(_copy(1))
    check("the copy held before the turn is gone",
          rx.count == 0 and not _received(host), repr(_received(host)[:24]))


def main() -> int:
    global ok
    ok = True
    print("PACTOR-1 memory ARQ, through the session receiver that runs on the air")
    the_copies_fail_alone()
    the_session_combines_the_repeat()
    the_memory_does_not_outlive_the_link()
    the_memory_does_not_cross_a_changeover()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
