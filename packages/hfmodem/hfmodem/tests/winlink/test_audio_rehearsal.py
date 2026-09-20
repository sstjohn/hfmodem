# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Identical authenticated B2F mail through three modem audio transports.

The fixtures and assertions are protocol-independent. Only audio transport,
link setup and turn discipline differ; both mailboxes must receive exact bytes.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from hfmodem.winlink import B2FSession, MailExchange, compose
from .audio_rehearsal import transport_for


def _message(sender, recipient, mid, seed):
    # All byte values, followed by incompressible data to force multiple modem
    # frames and exercise PACTOR's long cycle, escaping and B2F compression.
    body = bytes(range(256)) + np.random.default_rng(seed).bytes(644)
    return compose(sender, recipient, "shared audio proof", body, mid=mid,
                   date=datetime(2026, 9, 8, tzinfo=timezone.utc))


def _require_login(rms):
    """Use a published test vector; reject proposals before its exact response.

The normal simulated RMS accepts PR without checking it, so merely seeing a
completed loopback cannot prove authentication. This guard verifies that byte
sequence before releasing any proposal processing to the answering session.
"""
    start, feed = rms.start, rms.feed
    rms.start = lambda: start().replace(
        b"; W9SSJ DE", b";PQ: 23753528\r; W9SSJ DE")
    pending = bytearray()
    authenticated = []

    def guarded(data):
        if authenticated:
            return feed(data)
        pending.extend(data)
        reply = bytearray()
        while b"\r" in pending:
            line, _, rest = pending.partition(b"\r")
            pending[:] = rest
            assert not line.startswith((b"FC ", b"F>", b"FF")), "mail before login"
            if line.startswith(b";PR:"):
                assert line == b";PR: 72768415", "incorrect secure-login response"
                authenticated.append(True)
                reply.extend(feed(bytes(line) + b"\r" + bytes(pending)))
                pending.clear()
                break
            reply.extend(feed(bytes(line) + b"\r"))
        return bytes(reply)

    rms.feed = guarded
    return authenticated


def _trace_bytes(session):
    """Observe the application seam, independently of modem frame boundaries."""
    start, feed = session.start, session.feed
    sent, received = bytearray(), bytearray()

    def traced_start():
        reply = start()
        sent.extend(reply)
        return reply

    def traced_feed(data):
        received.extend(data)
        reply = feed(data)
        sent.extend(reply)
        return reply

    session.start, session.feed = traced_start, traced_feed
    return sent, received


@pytest.mark.parametrize("protocol", ("pactor3", "ardop", "vara"))
@pytest.mark.parametrize("mailboxes", ("both", "fetch", "empty"))
def test_authenticated_mail_crosses_audio_byte_exact(protocol, mailboxes):
    out = _message("W9SSJ", "SMTP:op@example.net", "AUDIOTOPEER1", 1798)
    back = _message("K7ABC", "W9SSJ", "AUDIOFROMGW1", 1823)
    outgoing = [out] if mailboxes == "both" else []
    incoming = [back] if mailboxes != "empty" else []
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         password="FOOBAR", client_sid="Pat-1.0.0", outbox=outgoing)
    transport = transport_for(protocol, "W9SSJ", "K7ABC", incoming)
    authenticated = _require_login(transport.rms)
    caller_tx, caller_rx = _trace_bytes(session)
    gateway_tx, gateway_rx = _trace_bytes(transport.rms)
    report = MailExchange(session, transport, max_steps=300).run()
    assert authenticated, report
    assert session.done and not session.failure, report
    assert transport.rms.done and not transport.rms.failure, report
    assert caller_tx == gateway_rx, report
    assert gateway_tx == caller_rx, report
    assert [m.render() for m in session.inbox] == [m.render() for m in incoming], report
    assert [m.render() for m in transport.rms.inbox] == [m.render() for m in outgoing], report
    assert session.sent_mids == [m.mid for m in outgoing]
    assert transport.rms.sent_mids == [m.mid for m in incoming]
    assert not transport.connected
    if protocol == "pactor3":
        assert transport.entry_decoded
        if incoming:
            assert len(transport.link.p3_rx) > 10
            assert any("LONG" in s for io in (transport.link.a_io, transport.link.b_io)
                       for s in io.sent)
