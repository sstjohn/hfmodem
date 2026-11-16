# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A re-keyed break-in's window holds the whole of the gateway's answer.

WS8EOC 40 m, 2026-11-08 19:35Z, TX[42]-TX[47]: six 833 ms changeover packets
on the 1.25 s cycle, each answered with a zero-error REQ starting 144-148 ms
past our audio end on the capture clock, so ending at +355 to +358 ms. The
session read each window to +347 to +349 ms and printed NO CONTROL SIGNAL and
ANSWER SLOT OCCUPIED. `control_collect_until` had taken the 12 ms acquisition
reserve off the read deadline, and the trimmed window is short of the word it
was trimmed to acquire. The flown read deadline is the logged RX end plus that
reserve.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.core.rigs import RIGS, resolve_tx_timing
from hfmodem.shrike import arq, onair, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _Bench, _CallbackBench, _FakeCodec, _Rig, _Session

CAP = (evidence.RECORD / 'operator-data' / 'captures'
       / 'onair-20261108T193518175243Z-849e1e66-2259-41a9-8251-1c7629cced6b'
       / 'stream.wav')
CYCLE = 60000
FIRST, LAST = 2491202, 2850000
RESERVE = round(onair._SessionRx.P3_ANSWER_ACQUIRE_RESERVE_S * onair.FS)

# slot: (audio start, audio end, logged CS due, logged RX end of the next hold)
BREAKINS = {
    41: (2491202, 2531195, 2542346, 2547968),
    42: (2551199, 2591195, 2602346, 2607872),
    43: (2611202, 2651195, 2662346, 2667904),
    44: (2671199, 2711195, 2722312, 2727936),
    45: (2731202, 2771195, 2782312, 2787968),
    46: (2791199, 2831195, 2842312, 2847872),
}


@pytest.fixture(scope='module')
def recording():
    evidence.require_input(CAP.with_name(
        'stream-l7278-20261108T1935Z-s00000000000000000000.flac'))
    return wav.read(CAP, start=FIRST, stop=LAST).astype(np.float32)


def unconfirmed_breakin(slot, start, end, due):
    s = _Session(role=arq.IRS)
    a = s.host.arq
    a.cfg.speed_up = 'hold'
    a.cfg.traffic_sl = 1
    a.on_host_data(b';FW')
    a.on_host_breakin()
    a.on_rx_packet(1, b'', 0, True)
    a.on_cycle()
    assert a.unconfirmed_breakin and a.role == arq.ISS
    g = onair._MasterGrid(32690, CYCLE, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.keyed_slot = slot
    g._p3_keyed_reply = (slot, start, end, CYCLE)
    g.rx_due = lambda _slot: due
    s.host.peer.raster = g
    s.rx.p3_receive_offset_hz = -12.2
    s.rx.new_cycle()
    return s, g


@pytest.mark.parametrize('slot', sorted(BREAKINS))
def test_rekeyed_breakin_reads_the_gateways_whole_answer(recording, slot):
    start, end, due, flown_end = BREAKINS[slot]
    s, g = unconfirmed_breakin(slot, start, end, due)
    until = flown_end + RESERVE
    hi = s.rx.control_collect_until(until, end, g)
    heard, _ = s.rx.control_signal_in(recording[end - FIRST:hi - FIRST], end, g)
    assert heard == arq.CS_REQUEST
    assert 143 <= (s.rx._p3_answer_at - end) / onair.FS * 1e3 <= 149


def ft891_cycle(recording, slot, tmp_path):
    start, end, due, _ = BREAKINS[slot]
    s, g = unconfirmed_breakin(slot, start, end, due)
    timing = resolve_tx_timing(rig='ft891', command_path='generic', environ={})
    live = _Bench(seconds=0, holdback=660, lat_in=532)
    live.LAT = live._lat = 1268
    live.tx_latency_n = timing.latency_samples
    live.geometry_n = timing.geometry_samples
    live.audio = np.pad(recording, (FIRST, 0))
    live.limit = LAST + 2 * CYCLE
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path,
                       settle=RIGS['ft891']['settle'], record_audio=False)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.sessrx = live, s.rx
    tx.defer_p3_cs = True
    tx.breakin_due = s.host.breakin_due
    tx.aim(g, slot + 1)
    return s, g, tx, live


@pytest.mark.parametrize('slot', sorted(BREAKINS))
def test_ft891_breakin_deadline_keeps_the_word_without_a_sweep(
        recording, slot, tmp_path, monkeypatch):
    s, g, tx, live = ft891_cycle(recording, slot, tmp_path)
    end = BREAKINS[slot][1]
    final = tx.key_instant(g, slot + 1) - onair._prekey_lead(live, 1920)
    assert live.key_notice == 1652
    assert onair._prekey_lead(live, 1920) == 2016
    assert s.rx.control_collect_until(final, end, g) == final
    live.now = final

    def sweep(*args, **kwargs):
        pytest.fail('the anchored break-in answer needed a sweep')

    monkeypatch.setattr(s.rx, '_acquire_answer', sweep)
    hi = live.samples
    heard, _ = s.rx.control_signal_in(recording[end - FIRST:hi - FIRST], end, g)
    assert heard == arq.CS_REQUEST


@pytest.mark.parametrize('realtime', [False, pytest.param(True, marks=pytest.mark.wall_clock)],
                         ids=['arithmetic', 'live'])
def test_ft891_breakin_answer_keys_every_slot(recording, realtime, tmp_path, monkeypatch,
                                             request):
    # Restarting ARQ between replies would hide the transition from a cold
    # answer read to the clock learned from the first REQ.
    origin = (32690 + 40 * CYCLE - onair.FS) // 128 * 128
    s, g = unconfirmed_breakin(0, BREAKINS[41][0] - CYCLE - origin,
                              BREAKINS[41][1] - CYCLE - origin,
                              BREAKINS[41][2] - CYCLE - origin)
    # The mail reply pulse epoch differs from the original P1 grid's anchor.
    g.anchor = BREAKINS[41][0] + 666 - CYCLE - origin
    del g.rx_due
    g.d_ref_n = 46080
    g.d_n = BREAKINS[41][2] - origin - g.boundary(1) - g.d_ref_n
    audio = np.pad(recording, (FIRST - origin, 2 * CYCLE))
    timing = resolve_tx_timing(rig='ft891', command_path='generic', environ={})
    # The tape's input holdback was 916 samples. The shorter 660-sample
    # bench opens the parent's window enough to learn an answer clock.
    started = time.monotonic()
    codec = (_FakeCodec(in_lat=788 / onair.FS, out_lat=480 / onair.FS, audio=audio)
             if realtime else None)
    live = codec.live if codec else _CallbackBench(seconds=0, holdback=916, lat_in=788)
    if codec:
        request.addfinalizer(codec.close)
        monkeypatch.setattr(live, 'close', codec.close)
        read_samples = live.read

        def bounded_read(count):
            assert live.pos + count <= audio.size, 'replay exceeded its sample budget'
            return read_samples(count)

        monkeypatch.setattr(live, 'read', bounded_read)
    else:
        live.LAT = live._lat = 1268
        live.audio, live.limit = audio, audio.size
    live.tx_latency_n, live.geometry_n = timing.latency_samples, timing.geometry_samples
    assert live.holdback == 916 and live.key_notice == 1652
    assert onair._prekey_lead(live, 1920) == 2016
    rig = _Rig()
    tx = onair.RadioTx(rig=rig, transmit=True, outdir=tmp_path,
                       settle=RIGS['ft891']['settle'], record_audio=False)
    tx.attach(s.host)
    s.host.peer = tx
    s.host.arq.dxcall = 'WS8EOC'
    tx.live, tx.sessrx, tx.p4_probe = live, s.rx, None
    tx.breakin_due = s.host.breakin_due
    tx.tx_end = g._p3_keyed_reply[2]
    tx.last_dur = (g._p3_keyed_reply[2] - g._p3_keyed_reply[1]) / onair.FS
    # The padded prefix has no peer answer or emitted CS3. Crediting a
    # phantom emission makes the setup loop spend its key budget reading it.
    g._p3_keyed_reply = None
    # The held-link loop prerenders before listening; setup has no such pass.
    tx.prepare_p3_breakin()
    live.flush_to(tx.tx_end)
    monkeypatch.setattr(s.host.arq, 'on_host_connect', lambda *args: None)
    monkeypatch.setattr(onair, '_MasterGrid', lambda *args, **kwargs: g)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *args, **kwargs: None)
    heard = []
    windows = []
    emissions = []
    sweeps = []
    read = s.rx.control_signal_in
    acquire = s.rx._acquire_answer
    transmit = live.transmit

    def observed(*args, **kwargs):
        answered_slot = g.keyed_slot
        if g._p3_keyed_reply is not None and 1 <= answered_slot <= 6:
            windows.append((answered_slot, args[1] + args[0].size, s.rx._p3_answer_at))
        result = read(*args, **kwargs)
        if result[0] is not None:
            heard.append((answered_slot, result[0], s.rx._p3_answer_at))
        return result

    def sweep(*args, **kwargs):
        sweeps.append(tx.slot)
        return acquire(*args, **kwargs)

    def emitted(*args, **kwargs):
        result = transmit(*args, **kwargs)
        assert s.host.protocol == spec.Protocol.PACTOR3 and s.host.arq.unconfirmed_breakin
        emissions.append((tx.slot, result, getattr(live, 'emitted', None)))
        return result

    new_cycle = s.rx.new_cycle

    def next_cycle():
        if tx.slot >= 7:
            s.host.arq.on_host_abort()
        new_cycle()

    monkeypatch.setattr(s.rx, 'control_signal_in', observed)
    monkeypatch.setattr(s.rx, '_acquire_answer', sweep)
    monkeypatch.setattr(live, 'transmit', emitted)
    monkeypatch.setattr(s.rx, 'new_cycle', next_cycle)
    args = onair.build_parser().parse_args(['--mycall', 'W9SSJ', '--hold', '12'])
    identity = SimpleNamespace(cw_call='W9SSJ', closing_required=lambda *args, **kwargs: False)
    code = onair._drive_session(args, tx=tx, host=s.host, sessrx=s.rx, live=live,
        rig=None, dx='WS8EOC', reply=None, outdir=tmp_path, identity=identity)
    if codec:
        assert 7.5 < time.monotonic() - started < 20
        assert not codec._thread.is_alive()
    assert code == 1 and s.host.arq.transfer_unconfirmed is not None
    assert [(slot, word) for slot, word, _ in heard] == [
        (slot, arq.CS_REQUEST) for slot in range(1, 7)]
    assert [slot for slot, _, _ in windows] == list(range(1, 7))
    assert windows[0][2] is None
    assert all(clock is not None for _, _, clock in windows[1:])
    for slot, end, _ in windows:
        expected = BREAKINS[slot + 40][3] + RESERVE - origin
        assert abs(end - expected) <= 2 * live._blk
    assert not sweeps, 'the cold sweep spent the break-in key budget'
    assert tx.slots_used == list(range(1, 8))
    assert [slot for slot, _, _ in emissions] == tx.slots_used
    for slot, (start, end), actual in emissions:
        expected = BREAKINS[41][0] - origin + (slot - 1) * CYCLE
        assert abs(start - expected) <= round(.005 * onair.FS)
        assert end - start > round(.8 * onair.FS)
        if codec:
            assert actual == ('sent', start, end)
            lo = start - live._lat - live.tx_latency_n + live.geometry_n
            blocks = [(at, block) for at, block in codec.output if lo <= at < lo + end - start]
            assert sum(block.size for _, block in blocks) == end - start
            assert any(np.any(block) for _, block in blocks)
    assert rig.edges.count(True) == 7
    assert not tx.refused
