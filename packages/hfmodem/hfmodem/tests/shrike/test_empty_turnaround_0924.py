# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""After our body, a link with nothing to say in either direction.

WS8EOC, 80 m, 2026-09-24 03:44Z
(`working/pactor-level-air-0916/arms/pactor-current-ws8eoc-80-pounce-20260924T034419Z`,
transcript `working/onair-0923-2244/`): greeting, `;PM` twice, `FS Y`, and our
477-byte block out behind the break-in. The CMS never answered it. From hold
cycle 54 the link turned round with empty fields -- the gateway's bit 6 taken
by our empty break-in, our bit 6 taken by its -- thirty-five times, and every
one of those changeovers bought the hold another `--hold 96`, so a session
nothing was moving on ran to cycle 251 of 281.
"""
from __future__ import annotations

from hfmodem.shrike import onair
from hfmodem.shrike.arq import IRS, ISS
from hfmodem.tests.shrike.test_b2f_loopback_send_ws8eoc import (
    ACCEPT, gateway_read, outbound)
from hfmodem.tests.shrike.test_b2f_loopback_ws8eoc import Ws8eoc, linked, run
from hfmodem.winlink import compress

#: The arm's greeting and proposal answer, `mail: peer said:` rejoined, with
#: the `;PM` recipient replaced.
GREETING = (b"RMS Trimode 1.4.4.0\r\n"
            b"W9SSJ has 113 daily minutes remaining with WS8EOC (EN72QQ)\r"
            b"[WL2K-5.0-B2FWIHJM$]\r"
            b";PQ: 39917013\r"
            b"CMS via WS8EOC >\r")
PENDING = (b";PM: W9SSJ 6C19LLAEHDZL 528 op@example.net //WL2K vara500 "
           b"receive test 0923 B\r"
           b";PM: W9SSJ 1KUDM5V76988 547 op@example.net //WL2K p3 "
           b"receive test 0923 A\r")

#: Hold cycles off the transcript's `[grid] hold N` lines: the last cycle a
#: byte moved (the SL6 field's acknowledgement), and each `changeover -> IRS`
#: after it.
LAST_BYTE = 44
TO_IRS = (54, 58, 61, 64, 67, 70, 74, 77, 87, 91, 95, 98, 104, 107, 110, 113,
          116, 119, 122, 125, 129, 135, 138, 141, 145, 148, 153, 156, 159, 163,
          166, 169, 173, 180, 185)
TO_ISS = (56, 59, 62, 65, 68, 72, 75, 85, 89, 93, 96, 102, 105, 108, 111, 114,
          117, 120, 123, 127, 133, 136, 139, 143, 146, 151, 154, 157, 161, 164,
          167, 171, 178, 181, 187)
HOLD = 96


def silent_cms():
    """The arm's far end: it takes our proposal and our body, and says nothing
    after either."""
    msg = outbound(subject="W9SSJ PACTOR test via WS8EOC 2026-09-24 0344z",
                   mid="829XTENK3736")
    gw = Ws8eoc([(b"F> ", PENDING + ACCEPT)], out=GREETING)
    host, session = linked(gw, outbox=[msg])
    run(host, gw, cycles=400)
    return msg, host, session, gw


def test_the_body_crosses_and_the_session_waits_for_the_cms():
    msg, host, session, gw = silent_cms()
    assert gateway_read(bytes(gw.rx))[2] == compress(msg.render())
    assert session.sent_mids == [msg.mid]
    assert session.stage == "their turn"


def test_nothing_to_say_is_not_a_break_in():
    """An empty break-in of ours after the body hands the gateway a turn it
    has already said it has no use for; the ARQ acknowledges its standing
    invitation instead and waits for the CMS on the receiving side."""
    msg, host, session, gw = silent_cms()
    last = max(i for i, b in enumerate(gw.bursts) if b[1])
    empty = [b for b in gw.bursts[last + 1:] if b[3]]
    assert not empty, "\n".join(gw.tape[-40:])
    assert host.arq.role == IRS
    assert not host.arq._breakin_pending


def test_a_reply_written_later_still_takes_the_channel():
    """The invitation stands on every packet the gateway keys, so bytes the
    host writes after it are carried on the next one."""
    msg, host, session, gw = silent_cms()
    host.arq.on_host_data(b"FQ\r")
    for _ in range(6):
        gw.key()
        host.tick()
    assert bytes(gw.rx).endswith(b"FQ\r"), "\n".join(gw.tape[-20:])


def _held(budget: onair._HoldBudget) -> int:
    role = ISS
    sent = 0
    for h in range(1, 400):
        was = onair._Link(sent, 0, role)
        if h <= LAST_BYTE:
            sent += 1
        if h in TO_IRS:
            role = IRS
        elif h in TO_ISS:
            role = ISS
        budget.cycle(h, was, onair._Link(sent, 0, role))
        if h >= budget.deadline:
            return h
    return 0


def test_an_empty_turnaround_loop_ends_inside_the_hold():
    """The first changeover after the last byte buys the peer its packet; the
    thirty-four behind it, none carrying a byte, buy nothing."""
    assert _held(onair._HoldBudget(HOLD)) == TO_IRS[0] + HOLD


def test_a_changeover_with_bytes_behind_it_still_buys_the_hold():
    budget = onair._HoldBudget(HOLD)
    budget.cycle(10, onair._Link(0, 0, ISS), onair._Link(0, 0, IRS))
    budget.cycle(12, onair._Link(0, 0, IRS), onair._Link(0, 5, IRS))
    budget.cycle(14, onair._Link(0, 5, IRS), onair._Link(0, 5, ISS))
    budget.cycle(16, onair._Link(0, 5, ISS), onair._Link(0, 5, IRS))
    assert budget.deadline == 16 + HOLD
