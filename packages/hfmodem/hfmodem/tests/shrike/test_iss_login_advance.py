# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The login an acknowledged changeover packet is supposed to carry.

0913-1837: the greeting completed, the mail client wrote its ~65-byte reply into
`PactorArq._outbuf`, the break-in took the channel with the first three of those
bytes -- and then the same three bytes went out eight more times and the other
sixty-odd never left the buffer. Round 21 gave the ISS a reader for the peer's
PACTOR-3 answer, so the acknowledgement of that changeover packet can now arrive;
this file is what has to happen when it does, and what has to happen when it does
not.

The peer is scripted rather than simulated: it records what the seam hands it and
answers with the WIRE codeword its own counter rule would choose -- CS1 against an
even packet counter, CS2 against an odd one -- so the reply travels the real
`PtcHost._counter_logical_cs` path into the FSM. Everything else is the station's
own: `PtcHost.tick`, `app_turns`, the renderer seam and `PactorArq`.
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import arq, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.winlink import B2FSession, MailClient

GREETING = (b"RMS Trimode 1.3.46.0\r[RMS-1.0-B2FHM$]\r"
            b";PQ: 12345678\rCMS via WS8EOC >\r")
CHANGEOVER_FIELD = 3            # placement.CHANGEOVER.crc_bytes - 3


class ScriptedPeer:
    """A far end that keys nothing and remembers everything."""

    def __init__(self):
        self.bursts: list[tuple[int, bytes, int, bool]] = []
        self.refuse = False

    def attach(self, host):
        self.host = host

    def send_packet(self, sl, payload, status, breakin=False):
        self.bursts.append((sl, bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse else None

    def send_cs(self, cs_index):
        pass

    def pump(self):
        pass

    def cycle(self):
        pass

    @property
    def last(self):
        return self.bursts[-1]

    def answer(self, *, ack: bool = True) -> None:
        """The codeword an IRS owes the packet it has just read."""
        counter = self.last[2] & 3
        wire = (counter & 1) if ack else 1 - (counter & 1)
        self.host.on_rx_event(rxfront.Event(
            0.0, "cs", "scripted peer", protocol=spec.Protocol.PACTOR3, cs=wire))


def logged_in(*, sl: int = 3):
    """An IRS holding a complete greeting, its login queued, about to break in."""
    peer = ScriptedPeer()
    host = PtcHost(peer, mycall="W9SSJ")
    host.protocol = spec.Protocol.PACTOR3
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    host.arq._sl = sl
    mail = MailClient(B2FSession("W9SSJ", target="WS8EOC"), host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    mail.on_link_data(GREETING)
    host.app_turns()
    assert mail.session.sent_text, "the client must have written its reply"
    assert host.arq._breakin_pending
    return host, peer, bytes(host.arq._outbuf)


def cycles_to_sign_off(host, limit: int = 60):
    """Cycles until the link ends itself -- `_give_up` queues a QRT, it does not
    tear down, so the goodbye is what says the budget ran out."""
    for n in range(1, limit + 1):
        host.tick()
        if host.arq._qrt_pending:
            return n
    return None


def breaks_in(**kw):
    """...and takes the channel, so the changeover packet is the burst in hand."""
    host, peer, login = logged_in(**kw)
    host.arq._breakin_armed = True
    host.tick()
    assert host.arq.role == arq.ISS
    sl, field, status, breakin = peer.last
    assert breakin and field == login[:CHANGEOVER_FIELD] and status & 3 == 0
    return host, peer, login


# -- the acknowledged changeover -------------------------------------------
def test_an_acknowledged_changeover_is_followed_by_an_ordinary_data_packet():
    host, peer, login = breaks_in()
    peer.answer()
    host.tick()

    sl, field, status, breakin = peer.last
    assert not breakin, "the changeover is spent; its head must not ride again"
    assert field == login[CHANGEOVER_FIELD:], "the next chunk of the login"
    assert status & 3 == 1, "the counter continues from the changeover's zero"
    assert sl == host.arq._sl == 3, "the traffic level, not a level of its own"
    assert CHANGEOVER_FIELD < len(field) <= host.arq._payload_bytes(False)


def test_the_login_reaches_the_air_once_and_in_order_and_the_turn_follows_it():
    host, peer, login = breaks_in()
    counters, sent = [peer.last[2] & 3], bytearray(peer.last[1])
    for _ in range(6):
        peer.answer()
        host.app_turns()
        host.tick()
        counters.append(peer.last[2] & 3)
        sent += peer.last[1]
        if peer.last[2] & spec.STATUS_CHANGEOVER:
            break

    assert bytes(sent) == login, "every login byte, once, in order"
    assert counters == [n % 4 for n in range(len(counters))]
    assert not host.arq._outbuf and host._txbuf == 0
    assert not any(burst[3] for burst in peer.bursts[1:])
    assert peer.last[2] & spec.STATUS_CHANGEOVER, "bit 6 offers the peer the turn"


@pytest.mark.parametrize("sl", [1, 3, 6])
def test_an_acknowledged_changeover_advances_at_every_traffic_level(sl):
    host, peer, login = breaks_in(sl=sl)
    assert len(peer.last[1]) == CHANGEOVER_FIELD, "the head's field is fixed"
    peer.answer()
    host.tick()
    sl_sent, field, status, breakin = peer.last
    assert not breakin and sl_sent == sl and status & 3 == 1
    assert field == login[CHANGEOVER_FIELD:CHANGEOVER_FIELD + len(field)]


# -- the unacknowledged changeover ------------------------------------------
def test_a_repeat_request_re_keys_the_same_changeover_packet():
    host, peer, login = breaks_in()
    for repeat in range(1, 4):
        peer.answer(ack=False)
        host.tick()
        sl, field, status, breakin = peer.last
        assert breakin, "the peer is still the ISS until it reads the head"
        assert field == login[:CHANGEOVER_FIELD]
        assert status & 3 == 0, "a repeat is not a new packet"
        assert host.arq._inflight.repeats == repeat
        assert host.arq._inflight.retries == 0    # the reverse channel works
    assert bytes(host.arq._outbuf) == login[CHANGEOVER_FIELD:]
    assert host.arq.state is arq.State.CONNECTED


def test_a_silent_peer_spends_the_retry_budget_and_signs_off():
    host, peer, login = breaks_in()
    assert cycles_to_sign_off(host) == host.arq.cfg.max_retries + 1
    assert peer.last[3] and peer.last[1] == login[:CHANGEOVER_FIELD]
    assert any("max retries" in line for line in host.log_lines)


# -- the refused ISS --------------------------------------------------------
def test_a_seam_refusing_every_iss_cycle_ends_the_link_at_the_budget():
    host, peer, login = breaks_in()
    peer.refuse = True
    keyed = len(peer.bursts)
    # The keyed changeover spends the first cycle; the refusals start behind it.
    assert cycles_to_sign_off(host) == host.arq.cfg.max_retries + 2
    assert host.arq._inflight is None and not host.arq._outbuf
    assert any("transmit seam refused every cycle" in line
               for line in host.log_lines)
    assert len(peer.bursts) == keyed + host.arq.cfg.max_retries + 1, \
        "the packet is re-offered every cycle -- a refusal is not a pause"
    peer.refuse = False
    host.tick()
    assert peer.last[2] & spec.STATUS_QRT, "the link ends by saying so"
    assert host.arq.state is arq.State.DISCONNECTING


def test_a_refused_cycle_does_not_charge_the_peer_for_our_own_silence():
    host, peer, login = breaks_in()
    peer.refuse = True
    for _ in range(host.arq.cfg.max_retries):
        host.tick()
    assert host.arq.state is arq.State.CONNECTED
    assert host.arq._inflight.retries == 1, "only the keyed cycle asked the peer"
    assert host.arq._refused_cycles == host.arq.cfg.max_retries - 1


def test_a_cycle_that_reaches_the_air_clears_the_refusal_count():
    host, peer, login = breaks_in()
    peer.refuse = True
    for _ in range(host.arq.cfg.max_retries):
        host.tick()
    assert host.arq._refused_cycles
    peer.refuse = False
    host.tick()
    assert host.arq._refused_cycles == 0
    peer.refuse = True
    assert cycles_to_sign_off(host) == host.arq.cfg.max_retries + 2
