# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The arq module's own loopback self-test, collected where the suite looks.

``python -m hfmodem.shrike.arq`` has always carried a full two-endpoint session
-- both changeover directions, a break-in pre-empting a live packet, a lost
break-in stranding both ends as ISS, and a QRT from the receiving end --
asserting byte-exact delivery in both directions. It ran only when someone
typed the module name, so the 2026-08-03 duplicate-delivery regression (a
delivered-but-unacked packet requeued across a counter-resetting changeover)
went through 102 green pytest runs on the very cycle-and-loss paths the block
exercises. Collecting it here is the fix for that class of miss; the module
name still runs it directly.
"""
from hfmodem.shrike import arq


def test_loopback_selftest():
    arq._selftest()


class _Seam(arq.ArqIO):
    """A transmit seam that keys nothing and refuses on command."""

    def __init__(self) -> None:
        self.refuse = False
        self.offered = 0

    def send_packet(self, sl, payload, status, breakin=False):
        self.offered += 1
        return arq.REFUSED if self.refuse else None


def _iss(seam: _Seam) -> arq.PactorArq:
    a = arq.PactorArq(seam, arq.ArqConfig())
    a.role, a.state = arq.ISS, arq.State.CONNECTED
    a.on_host_data(b"the budget counts air")
    return a


def test_a_refused_burst_spends_no_retry():
    """A guard that drops a burst takes no air, so the cycle is not a loss.

    `onair.RadioTx._refused` is the only thing that reaches this, and the
    changeover packet is what reaches it: on 2026-09-03 the ACK guard and the
    QRM guard dropped one apiece, both were charged to `_Packet.retries`, and
    the link signed off at `max retries` with TX numbering that jumps 14 to 17.
    The packet is still re-offered every cycle -- a refusal is not a pause, and
    it is not forever either: `_refused_cycles` bounds the run below.
    """
    seam = _Seam()
    seam.refuse = True
    a = _iss(seam)
    cycles = a.cfg.max_retries
    for _ in range(cycles):
        a.on_cycle()
    assert a._inflight is not None and a._inflight.retries == 0
    assert a.state is arq.State.CONNECTED
    assert seam.offered == cycles, seam.offered

    # ...and the budget is untouched rather than forgiven: the first cycle that
    # actually keys is spent at the next accounting, like any other.
    seam.refuse = False
    for _ in range(3):
        a.on_cycle()
    assert a._inflight.retries == 2, a._inflight.retries


def test_an_iss_that_cannot_key_at_all_ends_the_link_on_the_same_budget():
    """The other side of it: 0913-1837 spent 88 cycles placing nothing.

    A refused cycle costs the peer no retry, so an ISS whose every burst the
    seam drops used to hold the link open until the driver's hold ran out --
    with the login still in `_outbuf` and no sign-off on the air. This is the
    IRS control seam's rule (`test_arq_refusal_budget`) read from the sending
    side, and it is the only thing that bounds a station nothing can key.
    """
    seam = _Seam()
    seam.refuse = True
    a = _iss(seam)
    for cycles in range(1, 4 * a.cfg.max_retries):
        a.on_cycle()
        if a._qrt_pending:
            break
    assert cycles == a.cfg.max_retries + 2   # the first cycle builds the packet
    assert a._inflight is None and not a._outbuf
    assert a._qrt_pending and a.state is arq.State.CONNECTED


def test_the_budget_still_ends_a_strand_that_keys():
    """NEGATIVE CONTROL: the same cycles with nothing refused sign off."""
    seam = _Seam()
    a = _iss(seam)
    for _ in range(3 * a.cfg.max_retries + 1):
        a.on_cycle()
    assert a.state is not arq.State.CONNECTED


def test_refused_breakin_keeps_receiving_until_a_packet_really_keys():
    seam = _Seam()
    a = _iss(seam)
    a.role = arq.IRS
    a._breakin_pending = a._breakin_armed = True
    queued = bytes(a._outbuf)
    seq = a._next_seq
    seam.refuse = True
    a.on_cycle()
    assert a.role == arq.IRS
    assert a._inflight is None
    assert bytes(a._outbuf) == queued and a._next_seq == seq
    assert a._breakin_pending

    # A newly decoded packet authorizes the next attempt. The successful
    # changeover must carry the same bytes, once, starting at counter zero.
    a._breakin_armed = True
    seam.refuse = False
    a.on_cycle()
    assert a.role == arq.ISS
    assert a._inflight.breakin and a._inflight.status & 3 == 0
    assert a._inflight.payload + bytes(a._outbuf) == queued


def test_refused_receiving_goodbye_does_not_rotate_or_start_teardown():
    seam = _Seam()
    a = _iss(seam)
    a.role = arq.IRS
    a._outbuf.clear()
    a.on_host_disconnect()
    seam.refuse = True
    for _ in range(arq.GOODBYE_CYCLES + 2):
        a.on_cycle()
        assert a.role == arq.IRS
        assert not a.said_goodbye
        assert a.state == arq.State.CONNECTED
