# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Mail over shrike's own byte path: a B2F exchange across the loopback link.

The far end is a real answering session behind `SimPeer`'s honest path, so
every byte both ways rides the ARQ — chunked into PACTOR packets, acknowledged
on the cycle grid, the channel handed back and forth by changeover. What this
cannot prove is an SCS modem's opinion or the RF; it proves everything nearer
than those.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from hfmodem.shrike.arq import IRS, ISS, State
from hfmodem.shrike.compress import SB
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.shrike.spec import IDLE
from hfmodem.station.mail import PactorLoopback
from hfmodem.winlink import B2FSession, MailClient, MailExchange, compose, compress


def _exchange(outbox=None, rms_outbox=None, max_steps=200):
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         outbox=outbox or [])
    transport = PactorLoopback("W9SSJ", "K7ABC", rms_outbox=rms_outbox or [])
    report = MailExchange(session, transport, max_steps=max_steps).run()
    return session, transport, report


def test_a_message_crosses_the_link_byte_exact_both_ways():
    out = compose("W9SSJ", "SMTP:op@example.net", "loopback proof",
                  b"a message carried over shrike's own byte path.\r\n")
    back = compose("K7ABC", "W9SSJ", "return mail",
                   b"and one the other way.\r\n")
    session, t, report = _exchange(outbox=[out], rms_outbox=[back])

    assert session.done and not session.failure, report
    assert t.rms.done and not t.rms.failure
    assert [m.render() for m in t.rms.inbox] == [out.render()]
    assert [m.render() for m in session.inbox] == [back.render()]
    assert session.sent_mids == [out.mid]
    # The far inbox filled through the link layer and nowhere else: everything
    # our side said — handshake, proposals, the compressed body — arrived as
    # delivered link payload, so the link carried at least the body's size.
    assert t.carried() >= len(compress(out.render()))
    assert "closed" in report


def _binary_message():
    """A message whose lzhuf body carries both values the character stream keeps
    for itself. Fixed mid and date, so the compressed bytes are the same every
    run and the count below is a property of them rather than of the day."""
    body = np.random.default_rng(1798).integers(0, 256, 900,
                                                dtype=np.uint8).tobytes()
    return compose("W9SSJ", "SMTP:op@example.net", "binary body", body,
                   mid="TRANSPARNT01",
                   date=datetime(2026, 9, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize("held", (False, True))
@pytest.mark.parametrize("refusals", (0, 1, 3))
def test_mail_recovers_after_a_breakin_is_refused(held, refusals):
    from hfmodem.shrike.arq import REFUSED

    out = _binary_message()
    back = compose("K7ABC", "W9SSJ", "return mail", b"recovered\r\n")
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         password="FOOBAR", outbox=[out])
    transport = PactorLoopback("W9SSJ", "K7ABC", rms_outbox=[back])
    transport.host.stay_in_pactor1 = held
    # Published test vector, not an account credential. Put the challenge on
    # the ARQ byte path and require its response before accepting proposals.
    start = transport.rms.start
    transport.rms.start = lambda: start().replace(
        b"; W9SSJ DE", b";PQ: 23753528\r; W9SSJ DE")
    feed = transport.rms.feed
    pending = bytearray()
    authenticated = []

    def require_login(data):
        if authenticated:
            return feed(data)
        pending.extend(data)
        while b"\r" in pending:
            line, _, rest = pending.partition(b"\r")
            pending[:] = rest
            assert not line.startswith((b"FC ", b"F>")), "proposal before login"
            if line.startswith(b";PR:"):
                assert line == b";PR: 72768415"
                authenticated.append(True)
                return feed(bytes(pending))
            feed(bytes(line) + b"\r")
        return b""

    transport.rms.feed = require_login
    send = transport.host.send_packet
    refused = []

    def guarded_send(sl, payload, status, breakin=False):
        if breakin and len(refused) < refusals:
            refused.append(payload)
            return REFUSED
        return send(sl, payload, status, breakin=breakin)

    transport.host.send_packet = guarded_send
    report = MailExchange(session, transport, max_steps=1000).run()
    assert len(refused) == refusals
    assert authenticated
    assert session.done and not session.failure, report
    assert transport.rms.done and not transport.rms.failure
    assert [m.render() for m in transport.rms.inbox] == [out.render()]
    assert [m.render() for m in session.inbox] == [back.render()]


@pytest.mark.parametrize("held", (False, True))
def test_a_binary_body_crosses_the_link_byte_exact(held: bool):
    """0x1C and 0x1E in an lzhuf body, over the live host path at both levels.

    A B2F body is compressed, so this is not a corner: seven 0x1C and six 0x1E
    in 1078 bytes here, and the link layer under it reserves both -- 0x1E is
    IDLE wherever it sits and 0x1C opens a supervisor block. Unescaped, the
    first is deleted and the second eats everything to the next space, at every
    level: PACTOR-1 by the 1990 description and MEASURED against SCS's own
    monitor, PACTOR-3 because a part-filled field is padded with IDLE and
    stripped of it at the far end.

    Held in PACTOR-1 the fields are 8 and 20 bytes and the status byte's data
    type is two bits wide; on the default path the link upgrades mid-message and
    the same stream finishes over 59-byte fields and a three-bit field. The
    escape is above both, which is the point of putting it in `on_host_data`.
    """
    out = _binary_message()
    wire = compress(out.render())
    assert wire.count(SB) and wire.count(IDLE), wire.hex()

    session = B2FSession("W9SSJ", role="calling", target="K7ABC", outbox=[out])
    t = PactorLoopback("W9SSJ", "K7ABC")
    t.host.stay_in_pactor1 = held
    report = MailExchange(session, t).run()

    assert session.done and not session.failure, report
    assert [m.render() for m in t.rms.inbox] == [out.render()], report
    assert t.carried() >= len(wire)
    assert any("link upgraded" in line for line in t.host.log_lines) != held


def test_an_empty_exchange_still_closes_cleanly():
    session, t, report = _exchange()
    assert session.done and t.rms.done
    assert not session.failure and not t.rms.failure
    assert not session.inbox and not t.rms.inbox
    assert "closed" in report


def test_onair_reports_fragmented_login_and_proposal_before_teardown(capsys):
    from hfmodem.winlink import progress_to_stdout

    message = compose("W9SSJ", "SMTP:op@example.net", "progress", b"test\r\n")
    session = B2FSession("W9SSJ", role="calling", target="KB5LZK",
                         password="test-only", outbox=[message])
    sent = []
    greeting = (b"RMS Trimode\r[WL2K-5.0-B2FWIHJM$]\r"
                b";PQ: 28055883\rCMS via KB5LZK >\r")
    with progress_to_stdout():
        client = MailClient(session, sent.append)
        client.link_up()
        for start in range(0, len(greeting), 8):
            client.on_link_data(greeting[start:start + 8])
        output = capsys.readouterr().out
        assert "mail: stage awaiting greeting" in output
        assert "mail: stage awaiting proposal answer" in output
        assert "test-only" not in output
        assert b";PR: " in b"".join(sent)
        assert b"FC EM " in b"".join(sent)
        assert not session.done


def test_onair_mail_logging_is_restored_after_an_exception(capsys):
    import logging
    from hfmodem.winlink import progress_to_stdout

    logger = logging.getLogger("hfmodem.winlink.client")
    before = logger.level, list(logger.handlers)
    with pytest.raises(RuntimeError), progress_to_stdout():
        logger.info("mail: progress before failure")
        raise RuntimeError("stopped")
    assert "mail: progress before failure" in capsys.readouterr().out
    assert (logger.level, logger.handlers) == before


def test_the_link_is_down_after_the_exchange():
    session, t, _ = _exchange()
    assert t.host.arq.state != State.CONNECTED


def test_deliver_routes_to_the_app_and_only_to_the_app():
    """The seam itself: with an app attached, delivered payload feeds it and
    stays out of the hostmode channel buffer. Breaking this wire is silent
    mail loss, which is why it has a test of its own."""
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    fed = []
    host.app = type("App", (), {
        "link_up": staticmethod(lambda: None),
        "on_link_data": staticmethod(fed.append),
        "done": False})()
    host.deliver(b"payload bytes")
    assert fed == [b"payload bytes"]
    assert not host.channel(host.ptchn).rx

    host.app = None
    host.deliver(b"for the hostmode client")
    assert bytes(host.channel(host.ptchn).rx) == b"for the hostmode client"
    assert fed == [b"payload bytes"]


def test_app_turns_runs_the_changeover_discipline():
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    session = B2FSession("W9SSJ", role="calling", target="K7ABC")
    host.app = MailClient(session, host.arq.on_host_data)

    # Not connected: nothing pends.
    host.app_turns()
    assert not host.arq._over_pending and not host.arq._breakin_pending

    host.arq.state = State.CONNECTED
    host.arq.role = ISS
    host._txbuf = 0
    host.app_turns()
    assert host.arq._over_pending, "a drained ISS owes the peer its turn"

    host.arq._over_pending = False
    host.arq.role = IRS
    host._txbuf = 12
    host.app_turns()
    assert host.arq._breakin_pending, "an IRS holding an answer takes the channel"

    # A finished exchange still drains queued output, then stops driving.
    from hfmodem.winlink.session import _DONE
    host.arq._breakin_pending = False
    session._state = _DONE
    host.app_turns()
    assert host.arq._breakin_pending
    host.arq._breakin_pending = False
    host._txbuf = 0
    host.app_turns()
    assert not host.arq._breakin_pending


def test_finished_b2f_drains_fq_without_waiting_for_disconnect():
    message = compose("W9SSJ", "test@example.net", "FQ drain", "hello\n")
    session = B2FSession("W9SSJ", target="K7ABC", outbox=[message])
    transport = PactorLoopback("W9SSJ", "K7ABC")
    transport.attach(MailClient(session, transport.send))
    assert transport.connect()
    closing_with_data = False
    for _ in range(200):
        transport.step()
        if session.done:
            assert not session.failure
            if transport.host._txbuf:
                closing_with_data = True
            else:
                break
    else:
        pytest.fail("FQ remained queued after B2F completed")
    assert closing_with_data
    assert session.our_fq and transport.rms.done and not transport.rms.failure
    assert (False, "FQ", True) in transport.rms.exchange
    assert transport.host._txbuf == transport.host.arq._buffer_raw == 0
    assert not transport.host.arq._qrt_pending


def test_cli_mail_verb_runs_the_pactor_rehearsal(tmp_path, capsys):
    """The entry point end to end: compose from a body file, exchange over the
    loopback, fetch the waiting message, write it out, report the stages.

    The outbound summary distinguishes an accepted proposal from a refusal,
    without claiming that preparing a body proves transport delivery.
    """
    from hfmodem import cli
    body = tmp_path / "note.txt"
    body.write_bytes(b"cli-driven mail body\r\n")
    rc = cli.main(["mail", "--gateway", "K7ABC", "--protocol", "pactor",
                   "--mycall", "W9SSJ", "--send", str(body),
                   "--to", "SMTP:op@example.net", "--subject", "cli test",
                   "--fetch", "--out", str(tmp_path / "in"),
                   "--freq", "3598500"])
    outtext = capsys.readouterr().out
    assert rc == 0, outtext
    assert "stage closed" in outtext
    assert "mail: prepared " in outtext and "accepted FS +" in outtext
    written = list((tmp_path / "in").glob("*.b2f"))
    assert len(written) == 1, "the fetched message lands on disk"
    assert "hfmodem.shrike.onair" in outtext, "the on-air command is spelled out"


def test_onair_dry_run_attaches_mail_and_reports_its_stage(tmp_path):
    """The keying tool's mail flags, end to end on its radioless dry path: the
    session attaches, the run completes without a rig, and the summary names
    the stage the exchange died in — here 'awaiting greeting', the verdict a
    session that never connected should carry."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "hfmodem.shrike.onair",
         "--mycall", "W9SSJ", "--dxcall", "K7ABC", "--center", "3598500",
         "--mail-fetch", "--max-cycles", "1",
         "--mail-out", str(tmp_path / "mail"), "--outdir", str(tmp_path)],
        capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mail: stage awaiting greeting" in r.stdout
    assert "mail: nothing moved" in r.stdout
