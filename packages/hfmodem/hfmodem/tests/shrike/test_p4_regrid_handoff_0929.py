# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p4probe import EntryProbe
from hfmodem.shrike.p4observation import Kind, Observation
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_grid import _Bench, _Rig
from hfmodem.tests.shrike.test_p4_peer_breakin_0927 import peer_wave, run_arm, FS


def recovery_scene(tmp_path, monkeypatch, kind):
    live = _Bench(seconds=6.)
    y = round(2.02*FS)
    if kind is not None:
        wave, first = peer_wave(kind)
        live.audio[y-first:y-first+len(wave)] = wave
    raster = onair._MasterGrid(anchor=0, slot_n=60000, offset_n=8880,
        packet_n=46080, cs_n=5760, d_max_n=6240)
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.040)
    host = SimpleNamespace(protocol=onair.Protocol.PACTOR1,
        arq=SimpleNamespace(state=onair.State.CONNECTED), p4_after_first_ack=True)
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    rx = SimpleNamespace(p4=receiver, bridge=lambda *a: None, flush=lambda: None)
    tx.live, tx.sessrx = live, rx
    tx.p4_probe = EntryProbe()
    tx.p4_link_options = SimpleNamespace()
    tx.p4_link_host = host
    tx.tx_end = round(.96*FS)
    tx.aim(raster, 1)
    monkeypatch.setattr(onair, '_listen_until_answer',
        lambda stream, want, *a, **kw: stream.read(want))
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **kw: None)
    seg = live.read(round(1.35*FS))
    return live, raster, tx, host, rx, seg, y


@pytest.mark.parametrize('kind', ['bare', 'good', 'bad'])
def test_recovered_p1_window_hands_y27_to_link_before_next_key(tmp_path, monkeypatch, kind):
    live, raster, tx, host, rx, seg, y = recovery_scene(tmp_path, monkeypatch, kind)
    receiver = rx.p4
    slot, audio, origin = onair._regrid(live, raster, tx, host, rx, 1, seg, 0, round(.04*FS))
    assert tx.p4_probe.requested
    assert tx.p4_probe.receiver is receiver
    assert tx.p4_probe.peer_breakin
    assert live.pos < y+round(.846*FS)
    assert not live.emissions
    start, handoff = tx.p4_probe.handoff_audio
    assert start == origin and np.array_equal(handoff, audio)
    assert any(abs(o.stream_sample-y) < round(.002*FS) and o.word == 'Y27'
               for o in receiver.observations)


@pytest.mark.parametrize('deferred', [False, True])
def test_no_p4_read_preserves_recovered_p1_key_budget(tmp_path, monkeypatch, deferred):
    live, raster, tx, host, rx, seg, y = recovery_scene(tmp_path, monkeypatch, None)
    reads = []
    listen = onair._listen_until_answer
    def collect(*args, **kwargs):
        audio = listen(*args, **kwargs)
        if deferred:
            live.spend(round(.060*FS))
        return audio
    def read(*args):
        reads.append(live.sample_now())
        live.spend(round(.063*FS))
        return None
    monkeypatch.setattr(onair, '_listen_until_answer', collect)
    monkeypatch.setattr(onair, '_p4_hold_word', read)
    monkeypatch.setattr(onair, '_scan_frame', lambda *a, **kw: live.spend(round(.021*FS)))
    rx.flush = lambda: live.spend(round(.027*FS))
    slot, _, _ = onair._regrid(live, raster, tx, host, rx, 1, seg, 0, round(.04*FS))
    assert slot == 2
    assert not reads
    assert not onair._late_at_key(live, tx.key_instant(raster, slot), tx.cycle_cost_n())
    assert not tx.p4_probe.requested and not live.emissions


def test_main_transfers_recovered_word_before_legacy_tick(tmp_path, monkeypatch):
    regrid, tick, drive = onair._regrid, PtcHost.tick, onair._run_p4_link
    recovered, handed = [], []

    def recover(live, raster, tx, host, rx, slot, seg, origin, settle):
        if not recovered and host.p4_after_first_ack and 4.5*FS < live.pos < 5.2*FS:
            start = live.pos
            more = live.take_until(round(6.235*FS))
            if origin+len(seg) == start:
                seg = np.concatenate([seg, more])
            else:
                seg, origin = more, start
            found = onair._p4_peer_breakin(tx, live, raster, seg, origin, tx.tx_end)
            assert found is not None
            recovered.append(live.pos)
            return slot, seg, origin
        return regrid(live, raster, tx, host, rx, slot, seg, origin, settle)

    def p1_tick(self, **kwargs):
        assert not recovered or handed, 'legacy tick after recovered Y27'
        return tick(self, **kwargs)

    def handoff(tx, live, raster):
        handed.append(live.pos)
        return drive(tx, live, raster)

    monkeypatch.setattr(onair, '_regrid', recover)
    monkeypatch.setattr(PtcHost, 'tick', p1_tick)
    monkeypatch.setattr(onair, '_run_p4_link', handoff)
    y, keyed, machines, receivers, log = run_arm(tmp_path, monkeypatch, 'hold', 'good')
    assert len(recovered) == len(handed) == 1, log
    assert all(at < y for at, what in keyed if what.startswith('P1 ')), keyed
    first = next(at for at, what in keyed if what.startswith('P4 LINK'))
    assert abs(first-(y+round(.920*FS))) <= round(.002*FS), log
    assert receivers[0][2] < y+round(.846*FS)


@pytest.mark.parametrize('cancelled,read', [(False, False), (True, True)])
def test_breakin_handoff_ignores_old_or_cancelled_word(cancelled, read):
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.register(Observation(Kind.WORD, FS, 0, FS, FS, .240, 0., .8,
                                  word='Y27', score=.99, provisional=cancelled))
    probe = EntryProbe()
    tx = SimpleNamespace(p4_probe=probe, sessrx=SimpleNamespace(p4=receiver),
        p4_link_host=SimpleNamespace(arq=object()), tx_end=0)
    origin = FS if cancelled else 2*FS
    audio = np.zeros(round(.4*FS), np.float32)
    found = onair._p4_peer_breakin(tx, SimpleNamespace(sample_now=lambda: origin+len(audio)),
        SimpleNamespace(anchor=0, slot_n=60000), audio, origin, 0, read=read)
    assert found is None and not probe.requested
    if cancelled and read:
        assert not receiver.observations


@pytest.mark.parametrize('provisional', [False, True])
def test_breakin_handoff_keeps_live_word_overlapping_window_tail(provisional):
    receiver = P4Receiver(FS, admitted=True, breakin=True)
    receiver.register(Observation(Kind.WORD, FS, 0, FS, FS, .240, 0., .8,
                                  word='Y27', score=.99, provisional=provisional))
    probe = EntryProbe()
    tx = SimpleNamespace(p4_probe=probe, sessrx=SimpleNamespace(p4=receiver),
        p4_link_host=SimpleNamespace(arq=object()), tx_end=0)
    origin = round(1.1*FS)
    found = onair._p4_peer_breakin(tx, SimpleNamespace(sample_now=lambda: round(1.4*FS)),
        SimpleNamespace(anchor=0, slot_n=60000), np.zeros(round(.3*FS)), origin, origin, read=False)
    assert found == 0 and probe.requested
    assert probe.receiver is receiver


@pytest.mark.parametrize('position,now,remaining', [(1, 7, 13), (9, 7, 11), (1, 25, 0)])
def test_silent_listen_reports_remaining_entry_ceiling(monkeypatch, capsys, position, now, remaining):
    probe = EntryProbe(attempts=0, timeout=20., packet='robust')
    tx = SimpleNamespace(p4_probe=probe, slot=0, sessrx=None)
    live = SimpleNamespace(pos=position*FS, sample_now=lambda: now*FS)
    raster = SimpleNamespace(next_slot=lambda slot: slot+1)
    monkeypatch.setattr(onair, '_listen_for_p4_word', lambda *a, **kw: None)
    onair._run_p4_probe(tx, live, raster, stalled=12)
    text = capsys.readouterr().out
    assert 'path: silent listen -- our first P1 packet drew 12' in text
    assert f'entry ceiling in {remaining:.1f}s' in text
