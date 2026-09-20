# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A waiting IRS ends the link on its own refusals, not only on the peer's silence."""
from hfmodem.shrike import arq
import pytest


class Seam(arq.ArqIO):
    def __init__(self, refuse):
        self.refuse = refuse
        self.sent = []
        self.logs = []

    def send_cs(self, cs_index):
        self.sent.append(cs_index)
        return arq.REFUSED if self.refuse else None

    def log(self, msg):
        self.logs.append(msg)


def waiting_irs(refuse):
    io = Seam(refuse)
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    return a, io


def cycles_to_give_up(a, limit=60):
    for n in range(1, limit + 1):
        a.on_cycle()
        if a._qrt_pending or a.state is not arq.State.CONNECTED:
            return n    # an IRS says goodbye through a packet, one cycle later
    return None


def test_a_seam_refusing_every_cycle_ends_the_link_at_the_budget():
    a, io = waiting_irs(refuse=True)
    assert cycles_to_give_up(a) == a.cfg.max_retries + 1
    assert a._silent_cycles == 0        # the peer was never charged for this
    assert io.sent == [arq.CS_REQUEST] * (a.cfg.max_retries + 1)
    assert any("transmit seam refused" in line for line in io.logs)


def test_a_cycle_that_keys_resets_the_refusal_count():
    a, io = waiting_irs(refuse=True)
    for _ in range(a.cfg.max_retries):
        a.on_cycle()
    assert a.state is arq.State.CONNECTED and a._refused_cycles == a.cfg.max_retries
    io.refuse = False
    a.on_cycle()
    assert a._refused_cycles == 0 and a._silent_cycles == 1
    io.refuse = True
    assert cycles_to_give_up(a) == a.cfg.max_retries + 1


@pytest.mark.parametrize('budget', [8, 64])
def test_a_silent_peer_still_ends_the_link_at_the_budget(budget):
    a, io = waiting_irs(refuse=False)
    a.cfg.max_retries = budget
    assert cycles_to_give_up(a, limit=budget+2) == a.cfg.max_retries + 1
    assert a._silent_cycles == a.cfg.max_retries + 1
    assert any("no decodable traffic" in line for line in io.logs)


def test_a_codeword_the_driver_reports_keyed_clears_the_count():
    a, io = waiting_irs(refuse=True)
    a.on_cycle()
    assert a._refused_cycles == 1
    a.on_cs_emitted(arq.CS_ACK)
    assert a._refused_cycles == 0


class PacketSeam(Seam):
    """...and the sending side's, which refuses a packet rather than a codeword."""

    def __init__(self, refuse=False):
        super().__init__(refuse)
        self.packets = []

    def send_packet(self, sl, payload, status, breakin=False, **kw):
        self.packets.append((sl, bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse else None


def iss_owing_a_changeover():
    """An ISS that has keyed a changeover, into a seam that then refuses.

    The 0913-2320 geometry: the break-in went out, the peer answers on its own
    raster every cycle, and nothing this end keys can be placed afterwards.
    """
    io = PacketSeam()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    a.on_host_data(b";FW: W9SSJ\r")
    a._breakin_pending = a._breakin_armed = True
    a.on_cycle()
    assert a.role is arq.ISS and a.unconfirmed_breakin
    io.refuse = True
    return a, io


def test_a_burst_on_the_peers_raster_holds_the_iss_refusal_budget_open():
    """The frame the budget wants is the one our own transmitter makes unreadable.

    Keying 810 ms of a 1250 ms cycle leaves 365 ms, and the peer's packet is
    815 -- so on 0913-2320 the budget fired one line before a peer frame reached
    the ARQ. A burst on the peer's corroborated raster is the same refutation on
    the evidence the grid already trusts for it (`_MasterGrid.note_peer_bursts`,
    which `p3_control_refusal` reads as "the peer is still transmitting").
    """
    a, io = iss_owing_a_changeover()
    for _ in range(4 * a.cfg.max_retries):
        a.note_peer_raster_burst()
        a.on_cycle()
    assert a.state is arq.State.CONNECTED and not a._qrt_pending
    assert a._refused_cycles == 0
    assert a.unconfirmed_breakin, "the changeover is still what we owe"


def test_the_bound_is_unchanged_the_moment_the_bursts_stop():
    """NEGATIVE CONTROL. Silence still ends the link at the budget."""
    a, io = iss_owing_a_changeover()
    for n in range(1, 4 * a.cfg.max_retries):
        a.on_cycle()
        if a._qrt_pending or a.state is not arq.State.CONNECTED:
            break
    assert n <= a.cfg.max_retries + 2, n
    assert a._qrt_pending or a.state is not arq.State.CONNECTED


def test_the_cycle_after_an_unanswered_changeover_is_the_receivers():
    """The alternation, and that it alternates rather than persisting.

    A cycle that keyed a changeover and read nothing is the cycle that could not
    have read anything -- 365 ms of window against an 815 ms packet -- so the
    next one is given to the receiver whole (`onair`'s listening cycle, which
    refuses at the seam) and the changeover is re-keyed behind it.
    """
    a, io = iss_owing_a_changeover()
    io.refuse = False
    a.on_cycle()
    assert a.breakin_listen_due, "the changeover was keyed and nothing came back"
    io.refuse = True                     # the listening cycle keys nothing
    a.on_cycle()
    assert not a.breakin_listen_due, "...and the cycle after it keys again"
    io.refuse = False
    a.on_cycle()
    assert a.breakin_listen_due


def test_nothing_is_owed_to_the_receiver_without_a_changeover_in_flight():
    """NEGATIVE CONTROL. A waiting IRS is not the station this is about."""
    a, io = waiting_irs(refuse=False)
    for _ in range(3):
        a.on_cycle()
        assert not a.breakin_listen_due
