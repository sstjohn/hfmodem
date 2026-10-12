# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every long cycle is answered once a decoded window is judged for free.

0926-2353 WS8EOC-80 (`working/onair-0926-1854`, PACTOR-3 mail, `--long-cycle`):
from the switch to the 3.75 s cycle our CS2 keyed on every SECOND long slot --
30, 36, 42, 48 -- and WS8EOC repeated seq=3 seven times. Every lost slot reads
the same in the log: `SLOT 33 IS GONE -- ... +6.0 ms (+0.0 ms on the bare
clamp)`, then +22.1, +21.4, +13.3 ... all at +0.0 on the bare clamp. The grid
was never late; the admission carried `prekey_cost_n` forward and that stood
at 36.6 ms -- `_PeerSpectrum.hear` on the whole window between the admission
and the key, 8.1 ms on an ordinary 3.4 s window and 34.4 on the 7.1 s window
behind a handed-back slot, so one recovered window put every following
ordinary slot past the notice. The recovered slot itself read to the packet's
delivery and was admitted, which is what made it every second one.

The scene is the arm's own geometry on the station's clocks: the peer's
36-byte long SL1 packet every 3.75 s, our reply 3.390 s past its phase
reference, the DAC notice and holdback of log:46, and each stage charged at
what the arm's `[p3 timing]` lines and its own hold windows measured. One
slot is handed back on purpose, as slot 27 was, and the question is whether
the cycles after the recovered one keep their slots.
"""
from dataclasses import replace

import numpy as np

from hfmodem.shrike import arq, onair, p3frame, placement, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FS = 48000
SETTLE_N = 1920
LONG_N = round(spec.CYCLE_LONG_S * FS)
# The station's clocks (log:46): ADC->DAC 1268 + 960 of transmit latency, DAC
# notice 2612, holdback 660.
LAT, TX_LATENCY_N, HOLDBACK = 1268, 960, 660
# A long SL1 pre-key read on the arm: 24.5 ms on hold 7, 40.3 on hold 8; the
# tick and the render behind the admission, 2 ms.
SCAN_MS, RENDER_MS = 25.0, 2.0
# `_PeerSpectrum.hear` on the arm's own hold windows, (seconds, ms).
HEARD = ((3.405, 8.1), (7.137, 34.4), (14.662, 90.6))
# The first packet's data row, one cycle in; its phase reference is the header
# in front of that, and our reply slot sits `P3_LONG_REPLY_S` past it.
ROW0 = 7200 + LONG_N
FIRST_SLOT = 6
ANCHOR = (ROW0 - p3frame.DATA_OFFSET * rxfront.SPS
          + round(onair.P3_LONG_REPLY_S * FS) - FIRST_SLOT * 60000)
GREETING = b'utes remaining with WS8EOC (EN72QQ)\r'


def heard_n(size: int) -> int:
    """What `hear` costs a window of `size`, on the arm's measurements."""
    s = size / FS
    (s0, ms0), (s1, ms1), (s2, ms2) = HEARD
    if s <= s0:
        ms = ms0 * s / s0
    elif s <= s1:
        ms = ms0 + (ms1 - ms0) * (s - s0) / (s1 - s0)
    else:
        ms = ms1 + (ms2 - ms1) * (s - s1) / (s2 - s1)
    return round(ms / 1e3 * FS)


def tape(cycles: int) -> np.ndarray:
    head = ((placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * rxfront.SPS)
    audio = np.zeros(ROW0 + (cycles + 1) * LONG_N, np.float32)
    for k in range(cycles):
        pcm = placement.link_packet(
            1, GREETING, spec.status_byte(k & 3, long_cycle_request=True),
            long_cycle=True)
        at = ROW0 + k * LONG_N - head
        audio[at:at + pcm.size] = pcm
    return audio


def station(tmp_path, monkeypatch, cycles=10):
    s = _Session(role=arq.IRS)
    host = s.host
    host.arq.cycle_long = True
    host.arq.speed_level = 1
    host.arq._expected_seq = 0
    host.arq._ladder = replace(
        arq.P3_LADDER,
        payloads=((arq.P3_LADDER.payloads[0][0], 36),) + arq.P3_LADDER.payloads[1:])
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    host.peer = tx
    tx.attach(host)
    tx.sessrx = s.rx
    tx.defer_p3_cs = True
    bench = _Bench(seconds=(cycles + 4) * spec.CYCLE_LONG_S, holdback=HOLDBACK)
    bench._lat, bench.tx_latency_n = LAT, TX_LATENCY_N
    assert bench.key_notice == 2612
    pcm = tape(cycles)
    bench.audio[:pcm.size] = pcm
    tx.live = bench
    g = onair._MasterGrid(ANCHOR, 60000, SETTLE_N, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(True)
    s.rx._p3_row0 = ROW0 - LONG_N
    s.rx._p3_cycle_n = LONG_N
    s.rx._p3_span = rxfront._frame_span(placement.LONG_PATHS[1])
    s.rx.sync.packet_level = 1
    # Where the last cycle left the read cursor: behind our own codeword.
    bench.now = bench.pos = g.boundary(FIRST_SLOT) - LONG_N + 11325
    # Each stage charged at what the arm measured for it.
    scan = onair._scan_frame

    def scan_charged(*a, **k):
        bench.spend(round(SCAN_MS / 1e3 * FS))
        return scan(*a, **k)
    monkeypatch.setattr(onair, '_scan_frame', scan_charged)
    hear = onair._PeerSpectrum.hear

    def hear_charged(self, seg):
        bench.spend(heard_n(seg.size))
        return hear(self, seg)
    monkeypatch.setattr(onair._PeerSpectrum, 'hear', hear_charged)
    render = tx._p3_offset

    def render_charged(*a, **k):
        bench.spend(round(RENDER_MS / 1e3 * FS))
        return render(*a, **k)
    tx._p3_offset = render_charged
    return s, tx, g, bench, onair._PeerSpectrum()


def cycle(s, tx, g, bench, judge, slot, *, flown: bool, stall_n: int = 0):
    """One held-link cycle as `onair.run` spells it for a PACTOR-3 IRS.

    `flown` spells the pre-key path as 72c00129 flew it: the listen closes on
    the fixed geometry and the whole window is heard between the admission
    and the key. Otherwise the shipping helpers are used where the tree has
    them, and the flown spelling where it does not -- so on the tree that
    flew this arm the shipping scene below is the flown one.
    """
    host, sessrx = s.host, s.rx
    sessrx.new_cycle()
    tx.defer_p3_cs = True
    tx.listening = tx.cede_listening = tx.breakin_due = False
    slot, _ = onair._regear_next_slot(g, slot, host.arq.cycle_long)
    tx.aim(g, slot)
    slot = onair._p3_place_reply(g, tx, slot)
    early = g.boundary(slot) - tx.key_instant(g, slot)
    slot = onair._keyable_slot(bench, g, slot, SETTLE_N + early)
    tx.aim(g, slot)
    key_at = tx.key_instant(g, slot)
    before = sessrx.count
    win = (key_at - SETTLE_N - bench.holdback
           - round(onair.PREKEY_RESERVE_S * FS) - bench.pos)
    close = None if flown else getattr(onair, '_p3_window_close', None)
    closed = (close(bench, sessrx, host, tx, g, key_at, SETTLE_N, True)
              if close else None)
    if closed is not None:
        win = min(win, closed - bench.pos)
    whole = onair._listen_until_answer(
        bench, win, host, sessrx, 0,
        ahead=lambda audio, origin: onair._read_ahead(sessrx, g, audio, origin))
    seg_start = bench.pos - whole.size
    now = max(bench.pos, int(bench.sample_now()))
    whole, seg_start, _ = onair._collect(
        bench, sessrx, whole, seg_start,
        sessrx.control_bridge_until(key_at - max(g.d, SETTLE_N),
                                    key_at - onair._prekey_lead(bench, SETTLE_N),
                                    now, g))
    until = onair._p3_frame_ready(
        sessrx, onair._p3_decode_deadline(bench, key_at, SETTLE_N), bench.holdback)
    whole, seg_start, _ = onair._collect(bench, sessrx, whole, seg_start, until)
    if whole.size:
        onair._scan_frame(sessrx, whole, seg_start, tracked_only=True)
    slot, whole, seg_start = onair._p3_transition_window(
        bench, g, tx, host, sessrx, slot, whole, seg_start, SETTLE_N)
    _, _, whole, seg_start = onair._read_held_window(
        bench, g, tx, sessrx, slot, whole, seg_start, tx.boundary)
    if stall_n:
        bench.spend(stall_n)
    slot, whole, seg_start = onair._regrid(
        bench, g, tx, host, sessrx, slot, whole, seg_start, SETTLE_N)
    decoded = sessrx.count != before
    if flown or not hasattr(judge, 'verdict'):
        judge.hear(whole)
        carried = decoded or judge.carries_peer(whole)
    else:
        carried = judge.verdict(whole, g.cycle_n, decoded=decoded)
    if carried:
        host.arq.note_peer_carrier()
    host.tick(elapsed_ticks=1, cycle_ticks=g.ticks)
    tx.emit_pending_cs()
    if not flown:
        judge.hear(whole[-g.cycle_n:])
    if decoded:
        judge.learn(whole)
    return g.next_slot(slot), whole.size / FS


def run(s, tx, g, bench, judge, *, flown: bool, cycles: int = 5):
    """A slot handed back on the first cycle, then the cycles behind it."""
    slot, windows = FIRST_SLOT, []
    for n in range(cycles):
        slot, seconds = cycle(s, tx, g, bench, judge, slot, flown=flown,
                              stall_n=round(.1 * FS) if n == 0 else 0)
        windows.append(round(seconds, 2))
    return windows


def test_the_cycles_behind_a_recovered_window_keep_their_slots(tmp_path, monkeypatch, capsys):
    s, tx, g, bench, judge = station(tmp_path, monkeypatch)
    windows = run(s, tx, g, bench, judge, flown=False)
    out = capsys.readouterr().out
    # The planted hand-back and its recovered window, as on the arm; then
    # every long slot keyed, one per peer packet.
    assert out.count(onair.SLOT_GONE) == 1, out
    assert windows[0] > 6.5 and all(w < 4 for w in windows[1:]), windows
    assert tx.slots_used == [FIRST_SLOT + 3 * k for k in (1, 2, 3, 4, 5)], tx.slots_used
    assert onair.SLOT_UNRENDERED not in out
    assert len(s.packets) == 6
    assert tx.prekey_cost_n <= round(2 * RENDER_MS / 1e3 * FS), tx.prekey_cost_n


def test_as_flown_the_recovered_window_costs_every_second_slot(tmp_path, monkeypatch, capsys):
    """The arm's own lines: the recovered window's key `GIVEN UP BEFORE THE
    RENDER` on the bare clamp (slots 54, 69, 78), and the ordinary slot behind
    it `IS GONE ... (+0.0 ms on the bare clamp)` (33, 39, 45, ...)."""
    s, tx, g, bench, judge = station(tmp_path, monkeypatch)
    def as_flown(cs_index, *, p1_codeword=False):
        # Historical single render, with truthful current emission bookkeeping.
        tx._p3_emitted_cs = None
        before = tx.tx_end
        result = tx._render_p3_control(cs_index, p1_codeword=p1_codeword)
        if result != onair.REFUSED and tx.tx_end != before:
            tx._p3_emitted_cs = cs_index
        return result

    monkeypatch.setattr(tx, "_send_p3_control", as_flown)
    run(s, tx, g, bench, judge, flown=True)
    out = capsys.readouterr().out
    assert onair.SLOT_UNRENDERED in out
    assert '(+0.0 ms on the bare clamp)' in out
    assert len(tx.slots_used) <= 4 and tx.slots_used[-1] > FIRST_SLOT + 15, tx.slots_used


def test_a_decoded_window_is_judged_for_nothing_and_a_dark_one_on_its_cycle(monkeypatch):
    judge = onair._PeerSpectrum()
    heard = []
    monkeypatch.setattr(onair._PeerSpectrum, 'hear',
                        lambda self, seg: heard.append(seg.size))
    whole = np.zeros(2 * LONG_N + 12000, np.float32)
    assert judge.verdict(whole, LONG_N, decoded=True)
    assert heard == []
    assert not judge.verdict(whole, LONG_N, decoded=False)
    assert heard == [LONG_N]
