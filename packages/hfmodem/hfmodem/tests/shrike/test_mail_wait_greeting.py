# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keep ACKing a recorded greeting despite requeued setup announcement bytes."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement
from hfmodem.shrike.ptc import PtcHost, SimPeer
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_control_profiles import _received_reply
from hfmodem.tests.shrike.test_entry_candidate_body import META
from hfmodem.winlink import B2FSession, MailClient


def attach_mail(host):
    mail = MailClient(B2FSession("W9SSJ", target="WS8EOC"),
                      host.arq.on_host_data)
    host.app = mail
    mail.link_up()
    host.arq.on_host_data(b"1w9ssj")
    return mail


def test_recorded_rms_repeat_emits_b_ack_without_taking_the_channel(
        tmp_path, monkeypatch):
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, "historical", "pulse-center")
    tx.emit_pending_cs()
    mail = attach_mail(session.host)
    mail.on_link_data(b"RMS")
    onair._mail_app_turns(session.host, mail, wait_greeting=True)
    assert not session.host.arq._breakin_pending

    shift = 2 * grid.cycle_n
    origin = META["first_sample"] + shift
    crop = recorded_pcm(META)
    live.audio[origin:origin+len(crop)] = crop
    head = session.rx._p3_delivered_at + shift
    end = head + round(.83 * onair.FS)
    live.pos = live.now = end
    grid.cycles += 2
    session.rx.new_cycle()
    tx.aim(grid, slot + 2)
    onair._scan_frame(session.rx, live.audio[origin:end].copy(), origin)
    assert len(session.packets) == 2
    assert session.packets[-1].packet[2] == b"RMS"
    assert grid._p3_reply_timing() is not None
    assert not session.host.arq._breakin_armed
    session.host.tick()
    tx.emit_pending_cs()
    onair._mail_app_turns(session.host, mail, wait_greeting=True)

    assert len(emitted) == 2 and not tx.refused
    expected = onair._trim_silence(placement.historical_control_signal(arq.CS_ACK))
    expected = (expected * (tx.drive / max(abs(expected)))).astype(np.float32)
    np.testing.assert_allclose(emitted[-1], expected, rtol=0, atol=2e-12)
    lead = placement.historical_control_pulse_lead(arq.CS_ACK)
    assert tx.tx_audio_start + lead == grid.boundary(slot + 2)
    assert tx.rig.edges == [True, False, True, False]
    assert session.host.arq.role == arq.IRS and not grid.sending
    assert not session.host.arq._breakin_pending
    assert bytes(session.host.arq._outbuf) == b"1w9ssj"
    assert session.host.arq._buffer_raw == session.host._txbuf == 6
    assert mail.session.sent_text == []


def test_prompt_releases_gate_but_fragments_and_sid_do_not():
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    mail = attach_mail(host)
    for fragment in (b"RMS", b" Trim", b"ode 1\r\n", b"[RMS-1.0-B2FHM$]\r\n",
                     b"WS8EOC>"):
        mail.on_link_data(fragment)
        onair._mail_app_turns(host, mail, wait_greeting=True)
        assert not host.arq._breakin_pending
        assert bytes(host.arq._outbuf) == b"1w9ssj"
    mail.on_link_data(b"\r\n")
    assert mail.session.sent_text
    onair._mail_app_turns(host, mail, wait_greeting=True)
    assert host.arq._breakin_pending
    assert bytes(host.arq._outbuf).startswith(b"1w9ssj;FW: W9SSJ")


@pytest.mark.parametrize("wait", [False, True])
def test_automatic_breakin_is_opt_in_gated_and_teardown_remains_available(wait):
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    mail = attach_mail(host)
    onair._mail_app_turns(host, mail, wait_greeting=wait)
    assert host.arq._breakin_pending is (not wait)
    host.arq.on_host_disconnect()
    assert host.arq._qrt_pending
    assert host.arq._breakin_pending


def test_waiting_app_still_hands_over_a_drained_sending_turn():
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.arq.role = arq.ISS
    host.arq._enter_connected()
    mail = MailClient(B2FSession("W9SSJ"), host.arq.on_host_data)
    host.app = mail
    assert host._txbuf == 0
    onair._mail_app_turns(host, mail, wait_greeting=True)
    assert host.arq._over_pending


@pytest.mark.xfail(strict=True, reason=(
    "Unfixed, and recorded rather than left red: an arm that yields to a gateway "
    "which then says nothing cannot take the channel back for the rest of the hold, "
    "because the release is gated on having sent text and a banner, a SID or a bare "
    "prompt does not count. Which remedy is right -- a bounded timeout on a silent "
    "receiving stint, or releasing on the peer's own empty changeover packet, which "
    "both stranded arms recorded just before going quiet -- is a protocol decision. "
    "Strict, so the day it is fixed this goes red for passing and the marker comes "
    "off deliberately."))
def test_a_gateway_that_hands_over_and_says_nothing_does_not_strand_the_link():
    """0917-0007, KB5LZK 40 m: one turn, then the rest of the hold acking silence.

    The gateway asked for the channel (CS3 at 36.09 s), took it with an empty
    break-in packet, and then said nothing at all. `--mail-wait-greeting` holds
    the automatic break-in until B2F has answered a greeting prompt, and with no
    prompt to answer the only packet the arm keyed after that changeover was its
    QRT. Every arm since 2026-09-14 reads the same way -- one
    `GRID REVERSED -> IRS` and no second turn -- against two to six before it,
    and 0910-2300, which broke in nine times, is the arm that moved mail.

    So the gate has one release the link survives, and hanging up is not it. It
    exists to keep requeued setup bytes off a grant; a peer that has stopped
    speaking with bytes it has never been offered is the case where taking the
    channel is the only move left.
    """
    host = PtcHost(SimPeer(), mycall="W9SSJ")
    host.arq.role = arq.IRS
    host.arq._enter_connected()
    mail = attach_mail(host)
    mail.on_link_data(b"RMS Trimode 1\r\n")

    asked = gave_up = None
    for cycle in range(onair.MAIL_HOLD_CYCLES):
        onair._mail_app_turns(host, mail, wait_greeting=True)
        if host.arq._qrt_pending:
            gave_up = cycle
            break
        if host.arq._breakin_pending:
            asked = cycle
            break
        assert host._txbuf and not mail.session.sent_text
        host.tick()

    assert asked is not None, (
        f"stranded: the IRS of a silent peer with bytes queued, and the channel "
        f"asked for only by the teardown in cycle {gave_up}")
