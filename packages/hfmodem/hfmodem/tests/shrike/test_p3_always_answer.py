# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""PACTOR-3 keys every slot, on grid, in either role, whatever the presence verdict.

WS8EOC 40 m 2026-11-08 19:35Z: we keyed the `;FW` changeover on grid for six
cycles the gateway did not answer, went "P3 peer unsupported -> receive only;
retained bytes, no response key" on the sixth, and then left seventeen
zero-error REQs for that packet unanswered until the link ended. VE3KPG went
the same way that afternoon. The peer answers an ISS in control words alone,
so those words are its presence, and every cycle keys our packet. As IRS the
same verdict silenced K4MSU on 11-07. Budgets end a link, on a goodbye; a
verdict never skips a slot.
"""
import pytest

from hfmodem.shrike import arq
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import (
    IO, acknowledge_all, mail, opportunity)


class _Air(IO):
    def send_packet(self, sl, payload, status, breakin=False):
        self.events.append(('packet', bytes(payload), breakin))
        return len(payload)


def _changed_over(max_retries=200, *, entry_keyed=False):
    io = _Air()
    a = arq.PactorArq(io, arq.ArqConfig(max_retries=max_retries))
    a._enter_connected()
    a._entry_keyed = entry_keyed
    a.role = arq.IRS
    a.on_host_data(b';FW')
    a.on_host_breakin()
    # The peer's CRC-valid packet whose acknowledgement our CS3 carries.
    a._breakin_armed = True
    changeover = _cycle(a, io, source='current CRC')
    assert len(changeover) == 1 and changeover[0][2]
    assert a.role == arq.ISS and a.unconfirmed_breakin
    return a, io, changeover


def _cycle(a, io, *, source=None, word=None):
    token = a.begin_receive_opportunity()
    before = len(io.events)
    if word is not None:
        a.on_rx_cs(word)
    a.qualify_peer(token, a._link_generation, source)
    if a.settle_peer_presence():
        a.on_cycle()
    a.finish_receive_opportunity(token)
    return [e for e in io.events[before:] if e[0] == 'packet']


def test_the_changeover_keys_every_cycle_through_silence_and_the_peers_requests():
    a, io, changeover = _changed_over()
    for k in range(1, 7):
        assert _cycle(a, io) == changeover, f'silent cycle {k}'
        assert a._dark_windows == k
    assert a.p3_tx_allowed()
    for k in range(17):
        assert _cycle(a, io, word=arq.CS_REQUEST) == changeover, f'REQ {k}'
        assert a._dark_windows == 0 and a._p3_crc_wait_ticks == 0
        assert a.state == arq.State.CONNECTED and a.unconfirmed_breakin
    assert not any('receive only' in line for line in io.lines)
    assert any('source=peer control' in line for line in io.lines)


def test_after_an_entry_every_request_up_to_the_unread_rung_bound_is_answered():
    """The 11-08 link had keyed its entry packet, so seventeen REQs reach the
    unread-rung bound, which is that rule's to end. Each one before it keys."""
    a, io, changeover = _changed_over(entry_keyed=True)
    for k in range(1, 7):
        assert _cycle(a, io) == changeover, f'silent cycle {k}'
    for k in range(arq.UNREAD_RUNG_REPEATS):
        assert _cycle(a, io, word=arq.CS_REQUEST) == changeover, f'REQ {k}'
    assert a.state == arq.State.CONNECTED and a.unconfirmed_breakin


def test_a_silent_changeover_keys_past_its_retry_budget_and_ends_on_the_lifetime():
    """0913-2143's gateway ceded twelve cycles on, past an 8-retry budget."""
    a, io, changeover = _changed_over(max_retries=8)
    for k in range(1, arq.P3_CRC_WAIT_TICKS + 2):
        keyed = _cycle(a, io)
        if a._qrt_pending:
            break
        assert keyed == changeover, f'silent cycle {k}'
    assert a._qrt_pending and a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS
    assert a.transfer_unconfirmed['field'] == changeover[0][1]


@pytest.mark.parametrize('role,word,counts', [
    (arq.ISS, arq.CS_REQUEST, True),
    (arq.ISS, arq.CS_ACK, True),
    (arq.ISS, arq.CS_NAK, True),
    (arq.ISS, arq.CS_BREAKIN, False),   # the head of our own changeover
    (arq.IRS, arq.CS_BREAKIN, False),
    (arq.IRS, arq.CS_REQUEST, False),   # a word we key as IRS ourselves
])
def test_an_answer_word_is_the_iss_peer_present_and_cs3_never_is(role, word, counts):
    io = _Air()
    a = arq.PactorArq(io, arq.ArqConfig(max_retries=200))
    a._enter_connected()
    a.role = role
    if role == arq.ISS:
        a._inflight = arq._Packet(status=0, payload=b';FW', sl=1, breakin=True)
    a._dark_windows = 5
    token = a.begin_receive_opportunity()
    a._note_peer_control(word)
    a.qualify_peer(token, a._link_generation, None)
    assert a.settle_peer_presence()
    assert (a._dark_windows == 0) == counts
    # One verdict for every consumer after settlement, mail progress included.
    assert a.peer_qualified == counts
    # Not packet-raster evidence: the burst reread stays on the peer's frames.
    assert not a.peer_frame_qualified
    a.finish_receive_opportunity(token)


def test_the_irs_keys_every_slot_through_unsupported_windows_and_its_silence_budget():
    io = _Air()
    a = arq.PactorArq(io, arq.ArqConfig(max_retries=2))
    a._enter_connected()
    a.role = arq.IRS
    for k in range(20):
        count = len(io.events)
        assert opportunity(a)
        assert len(io.events) == count + 1, f'window {k} keyed nothing'
    assert a._dark_windows == 20 and a._silent_cycles > a.cfg.max_retries
    assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert not any('receive only' in line for line in io.lines)


def test_a_bare_cs3_head_during_an_unconfirmed_changeover_keys_the_changeover():
    a, io, changeover = _changed_over()
    held = a._inflight, bytes(a._outbuf)
    a.on_unconfirmed_breakin_head()
    assert _cycle(a, io) == changeover
    assert (a._inflight, bytes(a._outbuf)) == held and a.unconfirmed_breakin


def test_mail_expiry_with_the_peer_unsupported_keys_the_goodbye():
    a, io, client, timer, now = mail()
    acknowledge_all(a)
    timer.poll()
    for _ in range(6):
        opportunity(a)
    now[0] = 120.
    assert timer.poll() and timer.close_requested and a._qrt_pending
    count = len(io.events)
    assert opportunity(a)
    assert len(io.events) == count + 1 and io.events[-1][0] == 'packet'


def test_a_stale_reply_clock_keys_the_answer_on_the_held_grid(tmp_path, capsys):
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_p3_reply_placement import (
        A5_PEER, CYCLE, IDENTITY, PACKET_N, REPLY_N, driver, irs_grid, key_ack)
    g = irs_grid()
    g.note_p3_packet(A5_PEER, PACKET_N, CYCLE, identity=IDENTITY)
    slot = (A5_PEER + REPLY_N - g.anchor) // CYCLE + onair.ONSET_MAX_CYCLES
    assert "no fresh packet clock" in g.p3_control_refusal(slot)
    tx = driver(g, tmp_path)
    assert key_ack(tx, g, slot) is None
    assert tx.n == 1 and not tx.refused
    assert "ON THE HELD GRID" in capsys.readouterr().out
