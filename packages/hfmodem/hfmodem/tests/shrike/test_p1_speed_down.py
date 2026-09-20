# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-1 IRS at 200 Bd asking the sender for 100 Bd.

Level-1 §4.3 gives the receiving station one codeword for a path that will not
carry the rate: CS4, which a sender at 200 Bd reads as REJECT -- discard the
unacknowledged packet and send its information again at 100 Bd. This station
read that clause on receive (`ptc.PtcHost._logical_cs` -> `_fall_back_to_100`)
and had no transmit path that could key it: `arq._gear_cs` returns CS_ACK on the
PACTOR-1 seam before any gear decision is reached, and `send_cs(CS_NAK)` -- the
one event meaning a decode failed -- rendered as the held acknowledgement, the
repeat request, "send that again at the same rate", until the silence budget
signed the link off.

The asymmetry was the finding. The SENDING half of this link comes down at 200 Bd
two ways -- on the peer's CS4, and on its own evidence after `P1_HISPEED_RETRIES`
unanswered requests -- and K0NTS 40 m 2026-09-17 14:04 flew the first of those:
the gateway answered the call CS4, the link came up at 100 and every packet after
it keyed at 100. The RECEIVING half holds the same evidence about the peer's
transmissions, and `ptc._p1_cs_for` is where it now goes: a packet that failed
its CRC at 200 Bd, and a run of `P1_HISPEED_RETRIES` cycles that decoded nothing
at all on a link that has read this peer.

KB5LZK 40 m, 2026-09-17 13:54: 200 Bd, thirteen peer data packets, two decoded,
eleven CS1 keyed and a QRT on the eighth silent cycle. That arm is what the two
asking tests below are, and it would have qualified for both.
"""
import numpy as np
import pytest

from hfmodem.shrike import p1rx, pactor1, ptc, rxfront
from hfmodem.shrike.arq import IRS, State


class Peer:
    """Records the codewords and packets the link layer hands the renderer."""

    def __init__(self):
        self.cs, self.frames = [], []

    def send_p1_cs(self, index):
        self.cs.append(index)

    def send_p1_packet(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, payload))

    def send_p1_breakin(self, payload, baud, count, **kwargs):
        self.frames.append((count, baud, "BK", payload))

    def __getattr__(self, name):
        return lambda *a, **kw: None


def _cs(index):
    return rxfront.Event(0, "cs", "", protocol="PACTOR-1", cs=index)


def irs_at_200():
    """The arm's own geometry: we called, CS1 brought the link up at 200 Bd,
    the peer took it with a changeover packet and we are the IRS."""
    peer = Peer()
    host = ptc.PtcHost(peer=peer, mycall="W9SSJ")
    host.stay_in_pactor1 = True
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.arq.on_cycle()
    host.on_rx_event(_cs(pactor1.CS_ACK_A))
    host.arq.on_cycle()
    assert host.p1_baud == 200
    host.on_rx_event(_cs(pactor1.CS_CHANGEOVER))
    host.on_rx_event(rxfront.Event(0, "packet", "", protocol="PACTOR-1",
                                   packet=(0, 0, b"", True)))
    host.arq.on_cycle()
    assert host.arq.role is IRS
    peer.cs.clear()
    return host, peer


def test_irs_that_reads_nothing_at_200_asks_for_100():
    """A run of cycles nothing decoded in, on a link that HAS read this peer:
    `ptc._p1_cs_for` keys CS4 on the fifth, the count `_request_at_speed` spends
    in the other direction. The codeword goes out in the cycle that would have
    carried the held acknowledgement, so it costs no extra air."""
    host, peer = irs_at_200()
    for _ in range(host.arq.cfg.max_retries):
        host.arq.on_cycle()
    assert pactor1.CS_SPEED in peer.cs


def test_packets_that_will_not_decode_at_200_draw_a_speed_down():
    """The specification's plain case: "kann immer nach einem fehlerhaften Paket
    ein CS4 senden". A packet that reached the ARQ and failed its CRC is the one
    event PACTOR-1 has a NAK for, and at 200 Bd `ptc._p1_cs_for` now renders that
    NAK as the codeword the clause names instead of the held acknowledgement."""
    host, peer = irs_at_200()
    for _ in range(2):
        host.arq.on_rx_packet(0, b"", 0x01, False, protocol="PACTOR-1")
        host.arq.on_cycle()
    assert pactor1.CS_SPEED in peer.cs


def test_a_link_that_has_never_read_its_peer_commands_nothing():
    """The condition on the run above. A speed-down is addressed to a station,
    and "nothing decoded" cannot tell a fading one from a departed one -- so the
    codeword is keyed on evidence that there is a peer to read it and not on the
    silence alone. `arq.peer_reads` is that evidence: frames of the peer's read
    on THIS link, and a run of dead cycles that resets on any decode keeps the
    reading it stands on within the run's own length."""
    host, peer = irs_at_200()
    host.arq.peer_reads = 0
    for _ in range(host.arq.cfg.max_retries):
        host.arq.on_cycle()
    assert pactor1.CS_SPEED not in peer.cs


def test_an_unreadable_200_bd_channel_holds_the_acknowledgement_then_asks():
    """The run in full, and the first four of it are unchanged: the codeword is
    a function of the counter last accepted (`ptc._counter_cs_for`), so a cycle
    that accepts nothing leaves it where it stands. Eleven of those went out on
    2026-09-17 against a peer transmitting every cycle, and the fifth is now the
    speed-down instead -- every cycle after it, because a channel this bad loses
    transmissions and a CS4 the peer never reads commands nothing.

    THIS STATION'S OWN RATE DOES NOT MOVE. CS4 asks the peer to re-chunk what it
    is sending; what we transmit is codewords, and their 200 Bd is the peer's to
    change when it answers."""
    host, peer = irs_at_200()
    for _ in range(host.arq.cfg.max_retries):
        host.arq.on_cycle()
    asks = host.arq.cfg.max_retries - ptc.P1_HISPEED_RETRIES
    assert peer.cs == ([pactor1.CS_ACK_A] * ptc.P1_HISPEED_RETRIES
                       + [pactor1.CS_SPEED] * asks)
    assert host.p1_baud == 200


def test_the_silence_budget_ends_a_link_whose_control_words_arrive_clean():
    """`_charge_irs_silence` counts cycles nothing decoded in. A codeword read
    at zero bit errors resets it; a 960 ms data packet that will not decode on
    the same path does not reach the ARQ at all, so the budget cannot tell a
    peer that has gone from one it can no longer read."""
    host, peer = irs_at_200()
    for _ in range(host.arq.cfg.max_retries + 1):
        host.arq.on_cycle()
    assert host.arq.state is State.CONNECTED
    host.arq.on_cycle()
    assert host.arq.state is State.DISCONNECTING


@pytest.mark.parametrize("baud, field", [(100, 8), (200, 20)])
def test_the_reader_names_the_rate_it_found(baud, field):
    """Asking for the drop is the only half that is missing: `decode_p1_packets`
    already CRC-gates both rates in one pass, so a peer that geared down on its
    own decodes and is reported at 100 Bd. It cannot come back as doubled 200 Bd
    bytes -- the 200 Bd lane reads 24 bytes and its CRC would be doubled too."""
    payload = bytes(range(0x41, 0x41 + field))
    pad = np.zeros(int(0.15 * pactor1.FS))
    signal = pactor1.packet_signal(payload, baud, packet_count=1)
    packets = p1rx.decode_p1_packets(np.concatenate([pad, signal, pad]))
    assert [(p.baud, p.payload) for p in packets] == [(baud, payload)]


def iss_at_200():
    peer = Peer()
    host = ptc.PtcHost(peer=peer, mycall="W9SSJ")
    host.stay_in_pactor1 = True
    host.arq.on_host_connect("W9SSJ", "KB5LZK")
    host.arq.on_cycle()
    host.arq.on_host_data(bytes(range(65, 125)))
    host.on_rx_event(_cs(pactor1.CS_ACK_A))
    host.arq.on_cycle()
    assert host.p1_baud == 200
    return host, peer


def test_a_peers_cs4_at_200_does_drop_this_station_to_100():
    """The other half exists. An ISS at 200 Bd reading CS4 re-chunks the
    unacknowledged packet at 100, which is what we would be asking a peer for."""
    host, peer = iss_at_200()
    host.on_rx_event(_cs(pactor1.CS_SPEED))
    host.arq.on_cycle()
    assert host.p1_baud == 100
    assert peer.frames[-1][1] == 100


def test_an_iss_gears_itself_down_after_four_unanswered_requests():
    """And the asymmetry, in one place. The sending half of this link drops to
    100 Bd on its own evidence, with no codeword from anybody -- a run of
    repeat requests at 200 Bd IS the 200 failing. The receiving half has the
    same evidence about the peer's transmissions and no way to act on it."""
    host, peer = iss_at_200()
    for _ in range(ptc.P1_HISPEED_RETRIES):
        host.on_rx_event(_cs(pactor1.CS_ACK_A))
        host.arq.on_cycle()
    assert host.p1_baud == 100
    assert peer.frames[-1][1] == 100
