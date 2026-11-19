# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K4MSU 40 m 2026-11-16 22:42Z: the SL2 trial a CS4 answered and the peer never read.

`operator-data/working/onair-20261116T224243197453Z-162cab4a-…`: TX[37] SL1 long
counter 3 drew CS4 at 71.63 s; TX[38], SL2 long counter 0 with the proposal's
last 25 bytes, drew CS4 at 75.38. The first SL3 packet (idle, counter 1) missed
slot 60 and was never keyed, and the tape holds K4MSU's CS2 at 79.13 and 82.88 s
at 0 bit errors: on cycles we keyed nothing, the even-awaiting word, counter 0.
The ARQ had taken the second CS4 as acknowledgement and climb, so counter 0
was never keyed again and the gateway went silent.
"""
from hfmodem.shrike import arq, placement
from hfmodem.tests.shrike.p3_arq_support import control, sender

TAIL = b'4N7K4FT3 610 451 0\rF> C2\r'
HEAD = b'FC EM E4NI' + b'x' * 26


def long_frames(seam):
    def send_long_packet(sl, payload, status):
        if seam.refuse:
            return arq.REFUSED
        n = placement.LONG_PATHS[sl].crc_bytes - 3
        seam.frames.append((sl, status, payload[:n]))
        return n
    seam.send_long_packet = send_long_packet


def to_the_trial(answer='parity'):
    host, seam = sender(enabled=False, payload=HEAD + TAIL)
    long_frames(seam)
    a = host.arq
    a.cfg.p3_unkeyed_answer = answer
    a._cycle_long = True
    a._outbuf = bytearray(TAIL)
    a._buffer_raw = len(HEAD) + len(TAIL)
    a._inflight = arq._Packet(0x23, HEAD, 1, sent_sl=1, sent_at_level=1, offered=True)
    a._next_seq = 4
    seam.frames.append((1, 0x23, HEAD))
    control(host, 4)                         # 71.63: counter 3 read, climb
    assert seam.frames[-1] == (2, 0x00, TAIL) and a._inflight.trial_from == 1
    control(host, 4)                         # 75.38: the word repeated over the trial
    seam.refuse = True                       # slot 60: LATE TO THE KEY
    for _ in range(6):
        host.tick()
        if a._inflight is not None and a._inflight.offered:
            break
    assert (a._inflight.seq, a._inflight.sl, a._inflight.sent_sl) == (1, 3, None)
    seam.refuse = False
    return host, seam


def test_the_failed_sl2_trial_goes_out_again_at_sl1_with_its_counter():
    host, seam = to_the_trial()
    control(host, 2)                         # 79.13, on a cycle we did not key
    assert seam.frames[-1] == (1, 0x00, TAIL)
    assert host.arq.speed_level == 1
    control(host, 1)                         # read at SL1: counter 1 is next
    assert host.arq._inflight is None or host.arq._inflight.seq == 1
    assert host.arq.tx_acked_bytes == len(HEAD) + len(TAIL)


def test_no_acknowledgement_is_credited_from_the_mismatched_cs2():
    host, seam = to_the_trial()
    acked, settled = host.arq.acked, host.arq.tx_acked_bytes
    assert settled == len(HEAD)              # the trial's bytes wait on its answer
    control(host, 2)
    assert (host.arq.acked, host.arq.tx_acked_bytes) == (acked, settled)
    assert any('never keyed' in line for line in host.log_lines)


def test_without_a_held_trial_the_unkeyed_cycle_still_credits_nothing():
    host, seam = to_the_trial()
    host.arq._p3_unsettled = None
    acked = host.arq.acked
    control(host, 2)
    assert host.arq.acked == acked and host.arq._inflight.seq == 1
    assert len(seam.frames) == 2


def test_ack_restores_the_flown_reading():
    host, seam = to_the_trial('ack')
    assert host.arq.tx_acked_bytes == len(HEAD) + len(TAIL)
    control(host, 2)
    assert host.arq._inflight is None or host.arq._inflight.seq == 2
    assert not any(f[2] == TAIL and f[0] == 1 for f in seam.frames)


def test_the_trial_settles_once_the_packet_behind_it_is_answered():
    host, seam = to_the_trial()
    for _ in range(8):
        if host.arq._inflight.sent_sl is not None:
            break
        host.tick()
    assert seam.frames[-1] == (3, 0x19, b'')
    control(host, 2)                         # counter 1 keyed and acknowledged
    assert host.arq.tx_acked_bytes == len(HEAD) + len(TAIL)


def test_the_first_sl3_long_packet_after_a_cs4_is_a_selection(tmp_path, monkeypatch, capsys):
    """TX[38]'s answer at 75.38 climbed to SL3, whose long field nothing had
    rendered: slot 60 went 6.4 ms late and the grid fell behind."""
    import time
    from hfmodem.shrike import onair
    from hfmodem.tests.shrike.test_p3_missed_slots_0922 import station
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    a = s.host.arq
    a._inflight, a._sl, a._next_seq = None, 2, 0     # TX[38]: SL2, counter 0
    live = tx.live
    slot = 57
    tx.aim(g, slot)
    live.now = live.pos = tx.key_instant(g, slot) - onair._prekey_lead(live, 1920)
    a._start_next_packet()
    assert tx.slots_used == [slot] and not tx.refused
    render, keyed, renders = placement.link_packet, [], []
    monkeypatch.setattr(placement, 'link_packet',
                        lambda *args, **kw: renders.append((args[0], bool(keyed)))
                        or render(*args, **kw))
    nxt = g.next_slot(slot)
    tx.aim(g, nxt)
    live.now = live.pos = tx.key_instant(g, nxt) - onair._prekey_lead(live, 1920)
    started, emit = time.perf_counter(), tx._tx

    def charged_tx(audio, what, *args, **kw):
        live.spend(round((time.perf_counter() - started) * FS))
        keyed.append(what)
        return emit(audio, what, *args, **kw)
    monkeypatch.setattr(tx, '_tx', charged_tx)
    a.on_rx_cs(arq.CS_SPEED_UP)               # the field drained: the tick keys an idle
    for _ in range(4):
        if keyed:
            break
        s.host.tick()
    assert tx.slots_used == [slot, nxt] and not tx.refused
    assert keyed == ['SL3 LONG pkt 0B status=0x19'] and renders, renders
    assert all(behind for _, behind in renders), renders
    assert onair.LATE_KEY not in capsys.readouterr().out


FS = 48000
