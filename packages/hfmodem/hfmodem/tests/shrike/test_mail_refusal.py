# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Refusals close the link; delayed proposal answers survive idle hand-backs."""
import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.winlink.test_session import WS8EOC_REFUSAL_PARTS
from hfmodem.winlink import B2FSession, MailClient, compose


def mail_link(*, proposing=True):
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    host.protocol = spec.Protocol.PACTOR3
    msg = compose("W9SSJ", "WS8EOC", "mail", "test\r\n", mid="MAIL1132")
    session = B2FSession("W9SSJ", target="WS8EOC",
                         outbox=[msg] if proposing else [])
    client = getattr(onair, "_PactorMailClient", MailClient)
    mail = client(session, host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    mail.on_link_data(b"[WL2K-5.0-B2FWIHJM$]\rCMS via WS8EOC >\r")
    host.arq._outbuf.clear()
    host.arq._buffer_raw = 0
    host.buffer(0)
    return host, mail


def packet(host, status, payload=b"", *, breakin=False, crc_ok=True):
    host.arq.on_rx_packet(1, payload, status, crc_ok,
                          breakin=breakin, protocol=spec.Protocol.PACTOR3)


def invitations(host, first, count):
    for seq in range(first, first + count):
        packet(host, 0x40 | seq % 4)


def close_exchange(host, mail):
    budget = onair._HoldBudget(200)
    stopped = onair._mail_close(host, mail, budget, 10)
    assert stopped == f"the mail exchange failed: {mail.session.failure}"
    assert budget.closed and budget.deadline == 10
    host.arq.on_host_disconnect()
    assert host.arq._qrt_pending and host.arq._breakin_pending
    for _ in range(20):
        host.arq.on_cycle()
        if host.arq.said_goodbye:
            break
        if host.arq.role == arq.ISS and host.arq._inflight is not None:
            host.on_rx_event(rxfront.Event(
                0., "cs", "ACK", protocol=spec.Protocol.PACTOR3,
                cs=host.arq._inflight.seq & 1))
    assert host.arq.said_goodbye
    assert host.arq._inflight.status & spec.STATUS_QRT
    for _ in range(arq.GOODBYE_CYCLES + 1):
        host.arq.on_cycle()
    assert host.arq.state not in onair.LINKED
    return stopped


@pytest.mark.parametrize("count", [6, 7, 76, 100])
@pytest.mark.parametrize("chunked", [False, True], ids=["whole", "partial"])
@pytest.mark.parametrize("prelude", [b"", b"CMS via WS8EOC >\r"],
                         ids=["silent", "free-text"])
def test_delayed_fs_completes_after_empty_proposal_handbacks(count, chunked, prelude):
    host, mail = mail_link()
    session = mail.session
    sent = session.sent_text[:]
    assert session.stage == "awaiting proposal answer"
    if prelude:
        mail.on_link_data(prelude)
    if chunked:
        packet(host, 0x01, b"FS ")
    invitations(host, 2 if chunked else 1, count)
    assert not mail.done and not session.failure
    assert not session.sent_mids and session.sent_text == sent
    packet(host, (count + (2 if chunked else 1)) % 4,
           b"Y\r" if chunked else b"FS Y\r")
    assert session.sent_mids == ["MAIL1132"]
    assert not mail.done and not session.failure
    mail.on_link_data(b"FF\r")
    assert mail.done and not session.failure
    assert session.our_fq and session.remote_ff and host._txbuf


def test_empty_breakin_packet_is_not_a_handback_invitation():
    host, mail = mail_link()
    for _ in range(76):
        packet(host, 0x40, breakin=True)
    invitations(host, 1, 76)
    assert not mail.done and not mail.session.failure
    mail.on_link_data(b"FS Y\rFF\r")
    assert mail.done and not mail.session.failure
    assert mail.session.sent_mids == ["MAIL1132"]


@pytest.mark.parametrize("chunked", [False, True])
def test_1132_received_refusal_closes_even_with_outgoing_bytes(chunked):
    host, mail = mail_link()
    host.arq.on_host_data(b"queued proposal\r")
    parts = WS8EOC_REFUSAL_PARTS if chunked else [b"".join(WS8EOC_REFUSAL_PARTS)]
    for part in parts:
        mail.on_link_data(part)
    assert mail.done and mail.session.remote_refusal.startswith("Invalid login")
    assert not mail.session.our_fq and not mail.session.sent_mids
    sent = mail.session.sent_text[:]
    assert host._txbuf
    close_exchange(host, mail)
    invitations(host, 1, 76)
    mail.on_link_data(b"FS Y\rFF\r")
    assert mail.session.sent_text == sent
    assert not mail.session.sent_mids
    fresh_host, fresh = mail_link()
    fresh.on_link_data(b"FS Y\rFF\r")
    assert fresh.done and not fresh.session.failure
    assert fresh.session.sent_mids == ["MAIL1132"]
    assert fresh.session.our_fq and fresh_host._txbuf


def test_1137_fs_body_ff_and_fq_keep_the_successful_close_path():
    host, mail = mail_link()
    invitations(host, 1, 6)
    packet(host, 0x03, b"FS ")
    invitations(host, 8, 6)
    packet(host, 0x02, b"Y\r")
    invitations(host, 15, 76)
    assert not mail.done and not mail.session.failure
    mail.on_link_data(b"FF\r")
    assert mail.done and mail.session.our_fq and mail.session.remote_ff
    assert mail.session.sent_mids == ["MAIL1132"]
    budget = onair._HoldBudget(200)
    assert onair._mail_close(host, mail, budget, 10) is None
    assert not budget.closed
    host.arq._outbuf.clear()
    host.arq._buffer_raw = 0
    host.buffer(0)
    assert onair._mail_close(host, mail, budget, 11) == "the mail exchange completed"
    assert budget.closed and budget.deadline == 11


def test_empty_handbacks_after_ff_still_close_with_fq():
    host, mail = mail_link(proposing=False)
    invitations(host, 1, 6)
    assert mail.done and mail.session.our_fq and not mail.session.failure
    assert bytes(host.arq._outbuf) == b"FQ\r"
