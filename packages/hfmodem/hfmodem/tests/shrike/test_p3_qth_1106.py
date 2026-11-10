# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The 2026-11-06 QTH arms: a keyed CS4 wait, a steady changeover, a cheap scan."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, pactor1, rxfront
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_p3_breakin_timing import transitioned, tx_at
from hfmodem.tests.shrike.test_p3_timing_trial import receive, setup
from hfmodem.tests.shrike.test_tx_gear_hold import control, sender


def waiting():
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    control(host, 4)
    control(host, 4)
    assert host.arq._mail_cs4_wait == 0
    return host, seam, host.arq._inflight


def test_k4msu_0458_wait_keys_the_exact_field_every_cycle_uncredited():
    host, seam, packet = waiting()
    field, credit = seam.frames[-1], host.arq.tx_acked_bytes
    host.tick()                        # the cycle the wait opened in: already keyed
    for _ in range(5):
        host.tick()                    # quiet: the reply interval needs a key
    control(host, 4)                   # a repeated CS4 is answered by the field
    host.tick()
    assert seam.frames[2:] == [field] * 6
    assert host.arq._inflight is packet and host.arq.tx_acked_bytes == credit
    control(host, 2)                   # the validating word credits it once
    assert host.arq._mail_cs4_wait is None
    assert host.arq.tx_acked_bytes == credit + len(packet.payload)


def peer_packet(host, payload, status, *, breakin=False):
    host.on_rx_event(rxfront.Event(0, 'packet', 'peer', protocol=arq.spec.Protocol.PACTOR3,
                                   packet=(1, status, payload, True), breakin=breakin))


def test_wait_held_stays_off_the_air_once_the_peer_breaks_in():
    host, seam, _ = waiting()
    host.arq.cfg.p3_mail_cs4_breakin = 'hold'
    keyed = len(seam.frames)
    control(host, 3)
    for _ in range(3):
        host.tick()
    assert len(seam.frames) == keyed and host.arq._mail_cs4_wait is not None


def test_a_cs3_in_the_wait_hands_the_peer_the_link_and_settles_the_field():
    host, seam, packet = waiting()
    credit, keyed = host.arq.tx_acked_bytes, len(seam.frames)
    control(host, 3)
    assert host.arq._mail_cs4_wait is None and host.arq.role == arq.IRS
    assert host.arq.tx_acked_bytes == credit + len(packet.payload)
    assert host.arq.transfer_unconfirmed is None
    host.tick()
    assert len(seam.frames) == keyed


def test_wait_still_ends_unconfirmed_on_its_bound():
    host, seam, packet = waiting()
    remaining = packet.payload + bytes(host.arq._outbuf)
    host.tick(elapsed_ticks=96)
    assert host.arq.state == arq.State.DISCONNECTED
    retained = host.arq.transfer_unconfirmed
    assert retained['field'] + retained['suffix'] == remaining


def test_ws8eoc_0501_a_stepped_frame_reading_does_not_refuse_the_changeover(tmp_path):
    g = transitioned()
    g.note_p3_control(2551434)
    g.note_p3_control(2611434)
    g.note_p3_packet(2568704, 38880, 60000)
    for k in (1, 2):
        g.note_p3_control(2611434 + k * 60000)
        g.note_p3_packet(2568704 + k * 60000, 38880, 60000)
    g.reverse(to_iss=True)
    # The tracker re-locks 6 ms late, then steps back 2.5 ms: one transmission.
    for k, at in enumerate((2748992, 2808992, 2868872)):
        g.cycles += 1
        g.note_p3_packet(at, 38880, 60000)
        tx = tx_at(g, 48 + k, tmp_path)
        assert tx._clamp_refusal(g.peer_packet_end(48 + k)) is None
        assert tx.boundary == g.boundary(48 + k)


def test_kf7rsf_0443_a_held_link_reads_branch_b_at_its_own_pair_only():
    burst = pactor1.build_call_b('KF7RSF', 'robust')
    audio = np.random.default_rng(1).normal(0, .01, int(3.7 * 48000))
    for k in range(3):
        start = 9600 + k * 60000
        audio[start:start + burst.size] += burst[:audio.size - start]
    kinds = [e.kind for e in rxfront.decode_events(audio, p3_envelope=False, linked=True)]
    assert kinds.count('callb') == 3


@pytest.mark.parametrize('late', [0, 1296, 1944])
def test_ve3kpg_a_reply_sits_on_the_answer_slot_wherever_the_packet_lands(tmp_path, late, capsys):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock()
    tx.reply_clock.entry(entry)
    receive(s, tx, live, grid, entry + 44400 + late)   # +1944: +58 ms, VE3KPG's
    tx.aim(grid, 1)
    tx._pending_p3_cs = 0
    onair._p3_place_reply(grid, tx, 1)
    tx.emit_pending_cs()
    assert not tx.refused
    # The answer to the changeover stays on our own raster wherever its head
    # lands, unless that would bring the PTT inside the head's packet: +58 ms.
    pulse = tx.tx_audio_start + min(tx.tx_pulse_offsets)
    at = grid._p3_peer[0]
    late_by = (pulse - entry - 28800) % 60000
    if late == 1944:
        assert 0 < late_by < 480 and tx.tx_key_up >= at + 38880
    else:
        assert not late_by


def test_a_held_wait_ignores_a_peer_packet_and_stops_keying():
    host, seam, packet = waiting()
    host.arq.cfg.p3_mail_cs4_breakin = 'hold'
    keyed = len(seam.frames)
    peer_packet(host, b'RMS', 0x00, breakin=True)
    for _ in range(3):
        host.tick()
    control(host, 4)
    assert len(seam.frames) == keyed and host.arq._inflight is packet


def test_a_held_wait_still_ends_on_the_peers_qrt():
    host, seam, packet = waiting()
    host.arq.cfg.p3_mail_cs4_breakin = 'hold'
    peer_packet(host, b'', 0x80)
    assert host.arq.state == arq.State.DISCONNECTED
    assert host.arq.transfer_unconfirmed['field'] == packet.payload


def test_a_field_never_keyed_is_requeued_not_settled_by_the_break_in():
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    seam.refuse = True
    control(host, 4)
    packet = host.arq._inflight
    control(host, 4)
    assert host.arq._mail_cs4_wait == 0 and packet.sent_at_level == 0
    credit, queued = host.arq.tx_acked_bytes, packet.payload + bytes(host.arq._outbuf)
    control(host, 3)
    assert host.arq.role == arq.IRS and host.arq.tx_acked_bytes == credit
    assert host.arq._inflight is None and bytes(host.arq._outbuf) == queued


def test_the_changeover_payload_is_delivered_once():
    host, seam, _ = waiting()
    for _ in range(2):
        peer_packet(host, b';PM', 0x00, breakin=True)
        host.tick()
    peer_packet(host, b': W9S', 0x21)
    host.tick()
    assert bytes(host.channel(host.ptchn).rx).endswith(b';PM: W9S')
    assert bytes(host.channel(host.ptchn).rx).count(b';PM') == 1


@pytest.mark.parametrize('breakin', [True, False])
def test_the_peers_turn_during_the_wait_is_read_not_swallowed(breakin):
    host, seam, packet = waiting()
    credit = host.arq.tx_acked_bytes
    if breakin:
        peer_packet(host, b'', 0x00, breakin=True)     # its changeover, VE3KPG 0538
    peer_packet(host, b'\x01\rsho', 0x61)
    assert host.arq._mail_cs4_wait is None and host.arq.role == arq.IRS
    assert host.arq.tx_acked_bytes == credit + len(packet.payload)
    assert host.channel(host.ptchn).rx.endswith(b'\x01\rsho')


def test_mail_receive_k4msu_cs4_on_our_fs_then_the_body():
    """Our FS break-in drew CS4, its tail CS4 again (K4MSU's answer to every
    packet it takes), then the gateway breaks in with the message."""
    host, seam = sender(enabled=False, payload=b'FS +\r')
    host.p3_mail_active = True
    control(host, 4)
    control(host, 4)
    a = host.arq
    assert a._mail_cs4_wait == 0 and a._inflight.payload == b'+\r'
    control(host, 3)
    peer_packet(host, b'', 0x00, breakin=True)
    for chunk, status in ((b'\x01\rsho', 0x61), (b'rt te', 0x22), (b'st\x000\x00', 0x23)):
        peer_packet(host, chunk, status)
    assert host.channel(host.ptchn).rx.endswith(b'\x01\rshort test\x000\x00')
    assert a.state == arq.State.CONNECTED and a.transfer_unconfirmed is None


def test_a_peer_that_only_answers_cs4_ends_the_wait_after_twelve():
    host, seam, packet = waiting()
    for _ in range(arq.P3_CS6_HOLD_REPEATS - 2):
        control(host, 4)
    assert host.arq._mail_cs4_wait is not None
    control(host, 4)
    assert host.arq.state == arq.State.DISCONNECTED
    assert host.arq.transfer_unconfirmed['field'] == packet.payload


def test_a_peer_qrt_during_the_wait_ends_it_unconfirmed():
    host, seam, packet = waiting()
    host.arq.on_rx_packet(1, b'', 0x80, True, protocol=arq.spec.Protocol.PACTOR3)
    assert host.arq.state == arq.State.DISCONNECTED
    assert host.arq.transfer_unconfirmed['field'] == packet.payload


def test_no_resend_once_the_wait_has_no_cycle_left():
    host, seam, _ = waiting()
    host.tick(elapsed_ticks=95)
    keyed = len(seam.frames)
    control(host, 4)
    assert len(seam.frames) == keyed


def test_a_slot_given_up_does_not_read_presence_past_the_captured_audio():
    """WS8EOC replays 11-06: SLOT n IS GONE moved the reply a cycle on, and the
    presence read aimed at the peer's next copy, which had not arrived yet."""
    from types import SimpleNamespace
    from hfmodem.tests.shrike.test_p3_presence_progress_1003 import presence_session, rendered
    audio, row = rendered(1)
    s = presence_session(row, cfo=-25.2)     # the replays' tracked offset
    token = s.host.arq.begin_receive_opportunity()
    tx = SimpleNamespace(slot=76)
    raster = SimpleNamespace(boundary=lambda _: row + 60000 + 37000)
    onair._p3_qualify_presence(s.rx, raster, tx, audio, 0)
    assert not s.host.arq.peer_qualified
    s.host.arq.finish_receive_opportunity(token)


def test_ws8eoc_1107_a_long_cycle_grant_validates_the_waiting_field():
    """WS8EOC 40 m 11-07: CS4 on our changeover, three more CS4s on SL2 seq=1
    (status 0x21, bit 5 up), then CS6 on every long cycle while the wait
    re-keyed the field short. The CS1 it sent between them now ends the wait
    itself (test_p3_cs4_wait_1004, WS8EOC 11-08)."""
    host, seam = sender(enabled=False)
    host.p3_mail_active = True
    a = host.arq
    a.cfg.long_cycle = True
    control(host, 4)
    field = seam.frames[-1]
    assert field[:2] == (2, 0x21) and len(field[2]) == 23
    for cs in (4, 4, 4):
        control(host, cs)
    assert a._mail_cs4_wait is not None and seam.frames[-1] == field
    credit, longs = a.tx_acked_bytes, []
    seam.send_long_packet = lambda *packet: longs.append(packet) or len(packet[1])
    control(host, 6)
    assert a._mail_cs4_wait is None and a.tx_acked_bytes == credit + 23
    assert a.cycle_long and seam.frames[-1] == field
    (sl, payload, status), = longs
    assert (sl, status & 3) == (2, 2) and len(payload) > 23
