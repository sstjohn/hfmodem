"""The handed-back PACTOR-3 slots of 2026-09-19..21, replayed on this station's clocks.

Six mechanisms rank by seconds lost, worst first: the recovery loop's
unbounded rescan, the ISS long-cycle answer keyed from the post-tick flush,
two tracked-decode overrun paths, a clamp mismatch between two gates, and a
stale changeover instant. Each scene here stands one of them on the bench
with the station's own DAC notice and holdback (0920-1946 log:43) and asks
whether the key instant is still met.
"""
import json
from pathlib import Path
import time

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_ws8eoc_outbound_cycle_0920 import sender

FS = 48000
NOTICE, HOLDBACK = 2612, 660


def station(tmp_path, *, long=True):
    s, tx, g = sender(tmp_path)
    live = tx.live
    live._lat, live.holdback = NOTICE - 3 * live._blk, HOLDBACK
    if long:
        g.regear(True)
        s.host.arq._cycle_long = True
    return s, tx, g


def stand_late(live, g, tx, slot, ms, into=0):
    """Stand the grid so the admission check is exactly `ms` past the key.

    The clamp is taken on the delivered count, a multiple of the block, so
    the anchor is moved to put the key `ms` in front of a block edge. `into`
    is how far the converter stands into that block: 0 leaves a whole block
    before the next lands, `blk - 1` lands it under any work at all.
    """
    late = round(ms / 1e3 * FS)
    g.anchor -= (tx.key_instant(g, slot) - live.key_notice + late) % live._blk
    tx.aim(g, slot)
    key = tx.key_instant(g, slot)
    live.now = live.pos = key - live.key_notice + late + into
    assert live.clamp_late(key) == late
    return late


def owing(live, tx):
    """The render spends what the cycle was measured to owe, on the bench clock.

    Charged behind `_flip`, so `_regrid`, `_advance_aim` and `_tx` each meet
    the clock where the station's own cycle puts it.
    """
    render = tx._p3_offset

    def run(*a, **k):
        audio = render(*a, **k)
        live.spend(tx.prekey_cost_n)
        return audio
    tx._p3_offset = run


COST_MS = 1.5


@pytest.mark.parametrize('ms,lands', [(2.5, False), (4.0, False), (2.0, True)])
def test_a_sub_symbol_overrun_is_the_emission_paths_to_key(tmp_path, capsys, ms, lands):
    """0921-1053 holds 10, 13, 16: +4.8 ms, handed back as 3.75 s apiece.

    `_clamp_forgives` deducted last cycle's tick-and-render cost from the
    tolerance, so the grid refused at +2.2..3.5 what `_tx` keys at +5. The
    cost moves the clock, not the delivered count: `_tx` measures what the
    grid measured, plus a block where one lands inside the cost -- and the
    grid keeps exactly what that comes to inside the tolerance.
    """
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    live = tx.live
    slot = 45
    tx.prekey_cost_n = round(COST_MS / 1e3 * FS)
    late = stand_late(live, g, tx, slot, ms, into=live._blk - 1 if lands else 0)
    at_key = late + (live._blk if lands else 0)
    assert at_key <= round(onair.KEY_CLAMP_TOL_S * FS)
    owing(live, tx)
    got, _, _ = onair._regrid(live, g, tx, s.host, s.rx, slot,
                              np.zeros(0, np.float32), live.pos, 1920)
    out = capsys.readouterr().out
    assert got == slot and f"{at_key / FS * 1e3:+.1f} ms" in out
    assert onair.SLOT_KEPT in out
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert tx.slots_used[-1] == slot and not tx.refused
    out = capsys.readouterr().out
    assert onair.KEYED_LATE in out and onair.SLOT_UNRENDERED not in out
    assert live.emissions[-1][0] == g.boundary(slot) + at_key
    assert tx.prekey_cost_n == round(COST_MS / 1e3 * FS)


@pytest.mark.parametrize('ms', [2.5, 4.0])
def test_an_overrun_the_next_block_carries_past_the_tolerance_is_handed_back(
        tmp_path, capsys, ms):
    """The overrun inside the tolerance that the owed cost takes past it.

    The grid measures +2.5 or +4.0 with a block due inside the 1.5 ms the
    cycle still owes, so `_tx` will measure a block more. Forgiven on the
    bare clamp, the grid keeps a slot `_tx` then refuses: the packet is
    retained and the recovered window is never read. Predicted, the grid
    hands the slot back itself -- one cycle, listened to, and the next key
    met.
    """
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    live = tx.live
    slot = 45
    tx.prekey_cost_n = round(COST_MS / 1e3 * FS)
    late = stand_late(live, g, tx, slot, ms, into=live._blk - 1)
    assert late <= round(onair.KEY_CLAMP_TOL_S * FS) < late + live._blk
    owing(live, tx)
    costs = charged(live, s.rx, 'control_signal_in', 'deep_scan')
    got, _, _ = onair._regrid(live, g, tx, s.host, s.rx, slot,
                              np.zeros(0, np.float32), live.pos, 1920)
    out = capsys.readouterr().out
    assert got == slot + 3 and onair.SLOT_GONE in out and onair.SLOT_KEPT not in out
    assert f"{(late + live._blk) / FS * 1e3:+.1f} ms" in out
    assert 'control_signal_in' in costs
    assert not live.clamp_late(tx.key_instant(g, got))
    tx.send_long_packet(1, b'X' * 36, 0x22)
    out = capsys.readouterr().out
    assert tx.slots_used == [got] and not tx.refused
    assert onair.LATE_KEY not in out and onair.SLOT_UNRENDERED not in out


CAPTURES = Path(__file__).parents[5] / 'captures'
recorded = pytest.mark.skipif(not CAPTURES.is_dir(), reason='needs captures/')


def hold(arm, n):
    """A recorded hold window and the stream sample it starts on."""
    meta = json.loads((CAPTURES / f'onair-{arm}' / f'hold_{n:02d}.json').read_text())
    pcm = rxfront.load_wav(str(CAPTURES / f'onair-{arm}' / f'hold_{n:02d}.wav'))
    return pcm, meta['end_stream_sample'] - len(pcm)


def charged(live, obj, *names):
    """Charge each decoder's own wall time to the bench clock: the replay clock."""
    costs = {}
    for name in names:
        fn = getattr(obj, name)

        def run(*a, _fn=fn, _name=name, **k):
            started = time.perf_counter()
            try:
                return _fn(*a, **k)
            finally:
                dt = time.perf_counter() - started
                costs[_name] = costs.get(_name, 0.0) + dt
                live.spend(round(dt * FS))
        setattr(obj, name, run)
    return costs


def long_sl1(rx, row0, delivered=None):
    rx._p3_row0, rx._p3_delivered_at = row0, row0 if delivered is None else delivered
    rx._p3_cycle_n = 180000
    rx._p3_span = rxfront._frame_span(placement.LONG_PATHS[1])
    rx.sync.packet_level = 1


@recorded
def test_the_sending_grids_tracked_read_waits_for_the_next_cycle(tmp_path):
    """0920-2142 hold 23: +15.9 ms with control 5 ms and tracked 12 ms.

    The read stopped 64 ms before the key and the admission check is 10 ms
    behind it; `_read_held_window` then ran the tracked frame read anyway.
    Deferred, the key is met and the peer's repeated changeover is read at
    the top of the next cycle, in front of a whole cycle of listening.
    """
    pcm, origin = hold('0920-2142', 23)
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    a = s.host.arq
    a._inflight = arq._Packet(status=0, payload=b';FW', sl=1, breakin=True)
    assert a.unconfirmed_breakin
    s.rx.p3_receive_offset_hz = -14.0
    long_sl1(s.rx, 3653019)
    live = tx.live
    end = origin + len(pcm)
    live.audio[origin:end] = pcm
    slot = 69
    g.anchor = end + onair._prekey_lead(live, 1920) - slot * 60000
    tx.aim(g, slot)
    live.now = live.pos = end
    costs = charged(live, s.rx, 'control_signal_in', 'deep_scan')
    heard, _, audio, at = onair._read_held_window(
        live, g, tx, s.rx, slot, pcm, origin, g.boundary(slot))
    assert heard is None and 'deep_scan' not in costs and s.rx._p3_scan_deferred
    assert not live.clamp_late(tx.key_instant(g, slot)), costs
    # ...and the key, taken where the loop takes it.
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert tx.slots_used[-1] == slot and not tx.refused
    tx.aim(g, g.next_slot(slot))
    assert onair._scan_previous_window(s.rx, audio, at, sending=True, flowing=False)
    assert [p.packet[2] for p in s.packets] == [b'MS via VE3KPG >\r']
    assert not live.clamp_late(tx.key_instant(g, tx.slot))


@recorded
def test_the_deferred_read_is_the_next_sweeps_tracked_read(tmp_path):
    """0920-2142 hold 23 again, with our own turn confirmed.

    An unconfirmed break-in sweeps the previous window whole, which is what
    the scene above reads the changeover through. With nothing pending, a
    sending cycle's sweep reads only what `_read_held_window` deferred, and
    reads it tracked: the flag is spent on that one read, the changeover
    decodes from it, and the sweep after it scans nothing.
    """
    pcm, origin = hold('0920-2142', 23)
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    s.host.arq._inflight = arq._Packet(status=0x22, payload=b'X' * 36, sl=1)
    assert not s.host.arq.unconfirmed_breakin
    s.rx.p3_receive_offset_hz = -14.0
    long_sl1(s.rx, 3653019)
    live = tx.live
    end = origin + len(pcm)
    live.audio[origin:end] = pcm
    slot = 69
    g.anchor = end + onair._prekey_lead(live, 1920) - slot * 60000
    tx.aim(g, slot)
    live.now = live.pos = end
    scans = []
    scan = onair._scan_frame
    onair._scan_frame = lambda *a, **k: scans.append(k) or scan(*a, **k)
    try:
        heard, _, audio, at = onair._read_held_window(
            live, g, tx, s.rx, slot, pcm, origin, g.boundary(slot))
        assert heard is None and not scans and s.rx._p3_scan_deferred
        tx.send_long_packet(1, b'X' * 36, 0x22)
        assert tx.slots_used[-1] == slot and not tx.refused
        tx.aim(g, g.next_slot(slot))
        assert onair._scan_previous_window(s.rx, audio, at, sending=True,
                                           flowing=False)
        assert scans == [{'tracked_only': True}] and not s.rx._p3_scan_deferred
        assert [p.packet[2] for p in s.packets] == [b'MS via VE3KPG >\r']
        assert not onair._scan_previous_window(s.rx, audio, at, sending=True,
                                               flowing=False)
        assert len(scans) == 1
    finally:
        onair._scan_frame = scan


@recorded
def test_a_bounded_sl1_miss_opens_no_ladder_on_the_short_cycle(tmp_path):
    """0919-2059 hold 10: 51.6 ms tracked on a window holding no frame.

    The frame the arm's reader missed decodes from this window now (the
    bounded SL1 reader of 9e778eeb), so its last rows are cut to stand the
    arm's window: the bounded read misses, and the short cycle used to fall
    through to the 18-trial CRC ladder the long cycle already declined.
    """
    pcm, origin = hold('0919-2059', 10)
    pcm = pcm[:2918069 - origin + round(.25 * FS)]
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -11.8
    s.rx._p3_row0 = s.rx._p3_delivered_at = 2858069
    s.rx._p3_cycle_n = 60000
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
    s.rx.sync.packet_level = 1
    ladder = []
    packet = s.rx.sync.packet
    s.rx.sync.packet = lambda *a, **k: ladder.append(k) or packet(*a, **k)
    onair._scan_frame(s.rx, pcm, origin, tracked_only=True)
    assert not ladder and not s.packets
    assert [round(ms) for ms in s.rx._p3_stage_ms['tracked']] < [40]


@recorded
@pytest.mark.parametrize('arm,n,slot,late_ms,anchor,row0,hz', [
    ('0920-1946', 19, 46, 195.5, 32590, None, 0.0),
    ('0920-1056', 15, 32, 1453.4, 32687, 1613082, -13.0),
    ('0920-1115', 18, 35, 1463.5, 32685, None, -13.0),
    ('0920-2217', 46, 137, 25.8, None, 7726364, -13.3),
    ('0920-2151', 27, 66, 931.5, None, 5273398, -12.6),
])
def test_a_recovered_long_slot_costs_one_cycle(tmp_path, arm, n, slot, late_ms,
                                               anchor, row0, hz):
    """The sending grid's recovered window is read once, bounded, and keyed.

    These holds lost three slots apiece and then gave up before the render,
    or lost one at +900 ms: the recovered window is a whole 3.75 s cycle and
    the read of it was O(window) -- a blind sweep on the 09-20 morning tree,
    the changeover acquisition over the whole window after it. Replayed with
    each decoder's own wall time charged to the clock, one cycle is handed
    back and the next key is met.
    """
    pcm, origin = hold(arm, n)
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    a = s.host.arq
    a._inflight = (arq._Packet(status=0, payload=b';FW', sl=1, breakin=True)
                   if arm == '0920-2217' else
                   arq._Packet(status=0x22, payload=b'X' * 36, sl=1))
    s.rx.p3_receive_offset_hz = hz
    if row0 is not None:
        long_sl1(s.rx, row0)
    live = tx.live
    end = origin + len(pcm)
    live.audio[origin:end] = pcm
    lead = onair._prekey_lead(live, 1920)
    g.anchor = end + lead - slot * 60000 if anchor is None else anchor
    tx.aim(g, slot)
    key = tx.key_instant(g, slot)
    seg_end = min(end, key - lead)
    live.pos = seg_end
    live.now = max(seg_end, key + round(late_ms / 1e3 * FS) - live.key_notice)
    costs = charged(live, s.rx, 'control_signal_in', 'deep_scan', 'flush')
    got, _, _ = onair._regrid(live, g, tx, s.host, s.rx, slot,
                              live.audio[origin:seg_end].copy(), origin, 1920)
    assert got == slot + 3, costs
    assert not live.clamp_late(tx.key_instant(g, got)), costs
    assert sum(costs.values()) < onair.REGRID_RESERVE_S


@pytest.mark.parametrize('word', [arq.CS_ACK, arq.CS_REQUEST])
def test_the_long_cycle_answer_is_a_selection_not_a_render(tmp_path, word, monkeypatch, capsys):
    """0921-0039 holds 31-40: the packet behind the peer's word keyed late.

    The ISS reads the word at its deadline and has 10 ms to the admission
    check; a long SL1 render is 4 ms and its carrier correction 3.7. Both
    packets the word can ask for are rendered behind our own carrier, so the
    path from the word to the key renders nothing.
    """
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    a = s.host.arq
    a._inflight = None
    s.rx.p3_receive_offset_hz = -11.3
    live = tx.live
    slot = 45
    tx.aim(g, slot)
    live.now = live.pos = tx.key_instant(g, slot) - onair._prekey_lead(live, 1920)
    a._start_next_packet()
    assert tx.slots_used == [slot] and not tx.refused
    first = a._inflight
    assert len(tx._p3_rendered) == 3 and len(tx._p3_shifted) == 3
    # Renders in front of the key are the ones on the critical path; the
    # candidates rendered behind the carrier that follows are not.
    render, keyed = placement.link_packet, []
    renders = []
    monkeypatch.setattr(placement, 'link_packet',
                        lambda *args, **kw: renders.append(bool(keyed)) or render(*args, **kw))
    nxt = g.next_slot(slot)
    tx.aim(g, nxt)
    live.now = live.pos = tx.key_instant(g, nxt) - onair._prekey_lead(live, 1920)
    started = time.perf_counter()
    emit = tx._tx

    def charged_tx(audio, what, *args, **kw):
        live.spend(round((time.perf_counter() - started) * FS))
        keyed.append(what)
        return emit(audio, what, *args, **kw)
    monkeypatch.setattr(tx, '_tx', charged_tx)
    a.on_rx_cs(word)
    assert tx.slots_used == [slot, nxt] and not tx.refused
    assert renders and all(renders), renders
    assert (a._inflight is first) == (word == arq.CS_REQUEST)
    assert onair.LATE_KEY not in capsys.readouterr().out
