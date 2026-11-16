# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import arq, onair, placement, spec
from hfmodem.shrike.p3trial import CYCLE, FS, ReplyClock, TimingTrial
from hfmodem.tests.shrike.onair_support import _Bench, _Rig, _Session
from hfmodem.tests.shrike.p3_arq_support import granted, repeat


WIDTH, FORM, READ_LAG = 39120, (False, 1), 90

UNBOUNDED_TRIES = 0
"""`--p3-breakin-tries 0`, the old unbounded changeover, for the pins of a
stint the default hands back after three unanswered keyings
(`test_p3_breakin_d5`)."""


def old_clamp(tx):
    """`--p3-breakin-place boundary`, the old clamp, for the pins of a
    held-grid changeover; the default keys at the read (`test_p3_breakin_d5`)."""
    tx.p3_breakin_place = "boundary"


def setup(tmp_path, arm, delay=0):
    s = _Session(role=arq.ISS, entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    live = _Bench(seconds=60)
    g = onair._MasterGrid(10*FS, CYCLE, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.raster, tx.sessrx = live, g, s.rx
    tx.timing_trial = TimingTrial(arm)
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'none'
    tx.entry_delay_n = delay
    tx.aim(g, 0)
    live.now = live.pos = 9*FS
    tx.send_entry_packet(1, b'', 0x1a)
    assert not tx.refused and tx.timing_trial.entry_phase is not None
    return s, tx, live, g


def receive(s, tx, live, g, head):
    raw = placement.changeover_packet(b'RMS', 0, swapped=False)
    center = int(np.argmax(placement.protocol_config().pulse()))
    pad = 9600-center
    x = np.pad(raw, (pad, FS//10))
    origin = head-9600
    live.audio[origin:origin+len(x)] = x
    live.now = live.pos = head+round(.84*FS)
    tx.aim(g, 2)
    s.rx.control_signal(x, origin, head)
    assert s.packets and s.packets[-1].packet[2] == b'RMS'
    assert s.host.arq.role == arq.IRS
    g.reverse(to_iss=False)


def mail_link(tmp_path, frames=4):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    tx.p3_control_instant = 'peer'
    receive(s, tx, live, grid, entry + 44400)
    origin = grid._p3_peer[0]
    for index in range(2, 2 + frames):
        grid.cycles = index
        frame(grid, origin + (index - 1) * CYCLE, index)
        tx.aim(grid, index)
        onair._p3_place_reply(grid, tx, index)
    return s, tx, live, grid, origin, index


def frame(grid, at, index):
    grid.note_p3_packet(at, WIDTH, CYCLE, swapped=None, identity=(*FORM, index % 4, b'test'))


def top_of_cycle(tx, live, grid, slot):
    """What the hold loop does before the window: place the comb, arm."""
    tx.aim(grid, slot)
    slot = onair._p3_place_reply(grid, tx, slot)
    live.now = live.pos = grid.boundary(slot) - CYCLE // 2
    tx.arm_p3_reply()
    return slot


def owe(s, tx, cs):
    tx.send_cs(cs)


def turned(tmp_path, gap, waveform):
    s, tx, live, grid = setup(tmp_path, 'B', delay=234)
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry)
    tx.p3_control_waveform = waveform
    receive(s, tx, live, grid, entry + gap - READ_LAG)
    return s, tx, live, grid, entry


def host_stub(role=arq.IRS):
    # Render-only host uses the current ARQ owner, not invented permission.
    io = arq.ArqIO()
    io.protocol = spec.Protocol.PACTOR3
    a = arq.PactorArq(io, arq.ArqConfig())
    a.state = arq.State.CONNECTED
    a.role = role
    a.entry_pending = False
    a._rx_seen = True
    return SimpleNamespace(protocol=spec.Protocol.PACTOR3, arq=a)


def duplex(g, tmp_path, seconds=200.0):
    """A real transmitter on the bench stream, with PTT going nowhere."""
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                       settle=.04)
    tx.live = _Bench(seconds=seconds)
    tx.raster = g
    tx.host = host_stub()
    return tx


def pending(tmp_path):
    g = onair._MasterGrid(32581,CYCLE,0,packet_n=38880,cs_n=10080,d_max_n=6240)
    g.protocol,g.sending = spec.Protocol.PACTOR3,False
    g.acquired=g.corroborated=True
    g.d_n,g.d_ref_n=4800,38880
    g.cycles=63
    clock=ReplyClock(entry_phase=3964483,pulse_epoch=3993433)
    g.reply_clock=clock
    g.note_p3_packet(7128585,39120,CYCLE,swapped=False)
    g.note_p3_control(7173433)
    g.note_p3_packet(7188585,39120,CYCLE,swapped=True)
    sess=_Session(role=arq.ISS)
    tx=duplex(g,tmp_path)
    tx.reply_clock=clock
    sess.host.peer=tx
    tx.attach(sess.host)
    packet=arq._Packet(status=0,payload=b';FW',sl=1,breakin=True)
    sess.host.arq._inflight=packet
    sess.host.arq._next_seq=1
    sess.host.arq._buffer_raw=3
    tx.aim(g,121)
    tx.breakin_due=True
    tx.live.now=tx.live.pos=7280000
    tx.send_packet(1,b';FW',0,breakin=True)
    assert not tx.refused
    assert tx.tx_audio_start+min(tx.tx_pulse_offsets)==7293433
    assert tx._p3_emitted_turn is not None
    g.reverse(to_iss=True)
    g.turn_accepted=False
    g.cycles=65
    tx.aim(g,124)
    tx.breakin_due=True
    tx.live.now=tx.live.pos=7467800
    return sess,tx,g,packet


def fallback():
    host, _ = granted()
    for _ in range(arq.ENTRY_GRANT_CYCLES):
        repeat(host)
    assert host.protocol == spec.Protocol.PACTOR1 and host.arq.role == arq.ISS
    return host


def harness(tmp_path, *, slot=48):
    host = fallback()
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    host.peer = tx
    tx.attach(host)
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    bench = _Bench(seconds=100)
    tx.live = bench
    # M06: slot44's stale IRS boundary2682952 minus44*60000 minus40320.
    grid = onair._MasterGrid(2632, 60000, 8880,
                            packet_n=46080, cs_n=5760, d_max_n=6240)
    grid.d_n, grid.d_ref_n = .094 * onair.FS, 46080
    grid.keyed_slot = slot - 1
    tx.aim(grid, slot)
    # These cases measure the staggered SCS control's own geometry, which is
    # the experiment now that the defaults are historical/audio-start.
    tx.p3_control_waveform, tx.p3_control_placement = "current", "pulse-center"
    # The recorded peer carries its own -50 Hz error and the geometry below is
    # the nominal staggered control's. Keying onto the peer's raster is
    # `test_control_offset_follow`'s subject, not this file's.
    tx.p3_follow_offset = "none"
    tx.defer_p3_cs = True
    return host, rx, tx, grid, bench


def receiving(tmp_path):
    s, tx, live, grid = setup(tmp_path, 'B')
    entry = tx.timing_trial.entry_phase
    tx.timing_trial = None
    tx.reply_clock = ReplyClock(entry_phase=entry, pulse_epoch=entry)
    s.host.arq._yield_link(acked=True)
    s.host.arq.entry_pending = False
    grid.reverse(to_iss=False)
    arm_reply(tx, live, grid, 1)
    receive_packet(s, 0, breakin=True)
    assert emit_reply(tx, live) == arq.CS_ACK
    return s, tx, live, grid


def arm_reply(tx, live, grid, slot):
    grid.cycles = slot
    tx.aim(grid, slot)
    live.now = live.pos = tx.boundary - CYCLE // 2
    tx.arm_p3_reply()
    assert tx._p3_armed is not None
    return tx._p3_armed.cs


def receive_packet(s, seq, *, breakin=False, cycle_long=None, wants_long=False):
    status = seq | (spec.STATUS_LONG_CYCLE if wants_long else 0)
    s.host.arq.on_rx_packet(1, b'x' * arq.P3_LADDER.payload(1, False), status, True,
        breakin=breakin, protocol=spec.Protocol.PACTOR3, cycle_long=cycle_long)


def emit_reply(tx, live, *, late=False):
    if tx._pending_p3_cs is None:
        tx.host.arq._rx_this_cycle = False
        tx.host.tick(cycle_ticks=tx.raster.ticks, elapsed_ticks=tx.raster.ticks)
    live.now = live.pos = tx.boundary - (0 if late else 5000)
    tx.emit_pending_cs()
    assert not tx.refused
    assert tx.slots_used[-1] == tx.slot
    assert live.emissions[-1][0] == tx.boundary
    return tx._p3_emitted_cs


def disarm_reply(tx, live):
    assert live.disarm()
    tx._p3_armed = None
