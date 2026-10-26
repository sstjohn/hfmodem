# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual offline driver sinks and captured production guards; no clock repair."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.core import wav
from hfmodem.shrike import onair, p4turn, p4robusttx
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, FS
from hfmodem.tests.shrike.test_p4_part_a_driver_1011 import AbsoluteStream
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE
from hfmodem.tests.shrike.second_delta_support_1011 import freeze_guard, record

# Required recording crop, pinned together with its metadata.
CLIP_SHA256 = 'd592597ec6569cd2d8489a07a67b1e644d28bd46c5a66088f10e1c121f7279ba'
SIDECAR_SHA256 = 'bba178f78904555d7304c3c529c1ffbfac8479af14167d95f8d57d2527018941'
CLIP_BITS = 16
FIXTURE = Path(__file__).with_name('fixtures') / 'p4-reader-owner-c-1020.flac'
SOURCE_SHA256 = '880a6ab6541fd0c22e0f747fa5a3f3270b2c82301c59eabe98b789d93056f47e'


def required_recording_c():
    """Required public PCM, never an optional private-input skip."""
    assert CLIP_BITS in (16, 24), 'unsupported recording-crop bit depth'
    side = FIXTURE.with_suffix('.json')
    assert FIXTURE.is_file() and side.is_file(), 'required reader-owner fixture absent'
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == CLIP_SHA256
    assert hashlib.sha256(side.read_bytes()).hexdigest() == SIDECAR_SHA256
    meta = json.loads(side.read_text())
    assert meta['source']['container_sha256'] == SOURCE_SHA256
    assert meta['source']['rate'] == 48000 and meta['source']['samples'] == 9524096
    assert meta['source']['bits'] == CLIP_BITS and meta['source']['channel'] == 0
    assert meta['crop'] == [8016000, 9504000] and meta['origin'] == 8016000
    assert meta['clip']['container_sha256'] == CLIP_SHA256
    rate, raw = wavfile.read(wav.source(FIXTURE))
    dtype = '<i2' if CLIP_BITS == 16 else '<i4'
    assert rate == FS == 48000 and raw.shape == (1488000,)
    assert raw.dtype.kind == 'i' and raw.dtype.itemsize == np.dtype(dtype).itemsize
    pcm = np.asarray(raw, dtype=dtype)
    if CLIP_BITS == 24:
        assert np.all((pcm & 255) == 0), '24-bit PCM must remain int32 left-aligned'
    assert meta['pcm'] == dict(dtype=dtype, shape=[1488000], endian='little',
                               sha256=hashlib.sha256(pcm.tobytes()).hexdigest())
    audio = pcm.astype(np.float64) / -np.iinfo(pcm.dtype).min
    assert meta['float64'] == dict(dtype='<f8', shape=[1488000], endian='little',
                                   sha256=hashlib.sha256(audio.astype('<f8').tobytes()).hexdigest())
    return audio, meta

GRID = onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800)

def lifecycle(managed, recovery):
    source, meta = required_recording_c()
    start, stop = 8016000, (10040000 if recovery == 'exhaustion' else 9504000)
    if recovery == 'exhaustion':
        # Explicit constructed input: genuine initial packet, then synthetic misses.
        audio = np.zeros(stop-start,np.float32)
        audio[:KEY-start] = source[:KEY-start]
    else:
        audio = source
    assert meta['inputs'][recovery] == dict(
        dtype='<f4' if recovery == 'exhaustion' else '<f8', shape=[stop-start],
        endian='little', sha256=hashlib.sha256(
            audio.astype('<f4' if recovery == 'exhaustion' else '<f8').tobytes()).hexdigest())
    live = AbsoluteStream(audio,start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none','body'
    tx.p4_link_options.p4_link_timeout = (stop-start)/FS
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET,AT,.330,3.75,-.19)
    rx.cfo_hz, rx.expected = -.19, True
    cls = p4turn.P4Turn
    machines, captures, keys, counts, feeds, reads, replays = [], [], [], [], [], [], []
    first_machine_watch = []
    for name in ('read_partial','read_window','read_chirp'):
        method = getattr(rx,name)
        def read(audio, origin, _name=name, _method=method, **kw):
            owner, due = rx._watch_read_generation, rx.clock.next_due
            result = _method(audio,origin,**kw)
            reads.append((_name,owner,due,origin,origin+len(audio),live.sample_now()))
            return result
        setattr(rx,name,read)
    def factory(*args,**kw):
        kw.update(role='irs',managed_application=managed)
        m = cls(*args,**kw)
        machines.append(m)
        observe = m.observe
        def feed(reg,**kw):
            before = m._next
            prior = m.state
            count = m.peer_watch.attempts if m.peer_watch else None
            result = observe(reg,**kw)
            if m.peer_watch is not None and not first_machine_watch:
                first_machine_watch.append(m.peer_watch)
            feeds.append((reg.observation.stream_sample,reg.peer_watch_generation,
                          before,m._next,count,m.peer_watch.attempts if m.peer_watch else None,prior,m.state))
            return result
        m.observe = feed
        return m
    first_owner = []  # Retain real objects without changing production state.
    emit = tx._tx
    def sink(audio,what):
        m = machines[0]
        guard = tx.p4_tx_admit
        assert guard is not None
        frozen, cells = freeze_guard(guard)
        captured = cells['proposal']
        assert cells['waveform'] is audio and cells['machine'] is m and cells['aimed'] is tx.raster
        if captured.word == 'Y27':
            if not first_owner:
                assert not keys and m.peer_watch is not None and rx.peer_watch is not None
                assert m.peer_watch is first_machine_watch[0]
                assert captured is m._next if managed else captured == m._next
                assert captured.peer_watch_generation is not None
                assert captured.peer_watch_generation == m.peer_watch.generation == rx.peer_watch.generation
                first_owner.append(dict(proposal=captured, machine=m, watch=m.peer_watch,
                                        receiver=rx, receiver_watch=rx.peer_watch,
                                        generation=captured.peer_watch_generation))
            assert m.can_key(captured)
            captures.append((captured,frozen,cells,audio))
        guard()
        current = m._next  # Must be sampled AFTER final admission.
        assert current is captured if managed else current == captured
        assert np.array_equal(audio,p4robusttx.robust_word(current.word))
        if current.word == 'X27' and captures and not replays:
            # Replay during the LIVE recovered answer's sink lifecycle, before
            # its genuine key. Terminal cleanup cannot be the rejection cause.
            assert m.peer_watch is not None and m.state != p4turn.State.DOWN
            stale,retained,oldcells,oldwave = captures[0]
            before=(len(tx.slots_used),m._next,m.peer_watch.attempts)
            assert oldcells['proposal'] is stale and oldcells['machine'] is m
            assert all(c.cell_contents is oldcells[n] for n,c in zip(retained.__code__.co_freevars,retained.__closure__))
            entered=[]
            def initial():
                if not m.can_key(stale): raise onair._MissedTxSlot('proposal no longer keyable')
            for boundary,admit in [('initial',initial),('final',retained)]:
                with pytest.raises(onair._MissedTxSlot,match='proposal no longer keyable'):
                    admit()  # Before physical expiry or sink entry.
                    entered.append(oldwave)
                assert not entered and before==(len(tx.slots_used),m._next,m.peer_watch.attempts)
                replays.append(boundary)
                record('captured_guard_replay',managed=managed,recovery=recovery,boundary=boundary,
                    captured=id(stale),prior_authorized_keys=sum(p is stale for p in keys),additional_keys=0,
                    freevars={n:id(v) for n,v in oldcells.items()},
                    code_sha256=hashlib.sha256(retained.__code__.co_code).hexdigest(),
                    waveform_sha256=hashlib.sha256(oldwave.tobytes()).hexdigest(),current=id(current),
                    count=m.peer_watch.attempts,state=m.state)
        emit(audio,what)
        keys.append(current)
        counts.append(m.peer_watch.attempts if m.peer_watch else None) # Before driver keyed.
    tx._tx = sink
    with patch.object(p4turn,'P4Turn',factory):
        onair._run_p4_link(tx,live,GRID)
    m = machines[0]
    record('driver_lifecycle',managed=managed,recovery=recovery,
           input_label='constructed initial valid PCM plus synthetic misses' if recovery=='exhaustion' else 'unchanged recording C',
           input_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
           keys=[asdict(p) for p in keys], counts_before_key=counts, reads=reads,
           first_owner=first_owner[0],
           feeds=[(a,b,asdict(c) if c else None,asdict(d) if d else None,e,f,g,h) for a,b,c,d,e,f,g,h in feeds],
           events=tx.p4_probe.link_events,emissions=tx.p4_probe.emissions)
    return m,rx,tx,live,captures,keys,counts,feeds,replays

def test_d1_actual_driver_empty_cooldown_slot():
    m,rx,tx,live,captures,keys,counts,feeds,replays = lifecycle(True,'exhaustion')
    assert [(p.stream_sample,p.word) for p in keys[:4]] == [(KEY+i*CYCLE,'Y27') for i in range(4)]
    assert (keys[4].stream_sample,keys[4].word) == (8911870,'X27')
    assert (keys[5].stream_sample,keys[5].word) == (9091870,'X27'), 'empty cooldown slot must actually emit held answer'
    assert not any(p.word=='Y27' and 8731870<p.stream_sample<9811870 for p in keys)
    renewed = next(p for p in keys if p.word=='Y27' and p.stream_sample>8731870)
    assert renewed.stream_sample == 9811870
    assert counts[:5] == [0,1,2,3,4]
    assert tx.p4_link_host.arq.tx_acked_bytes == 0 and not m.outbox.inflight
    assert bytes(tx.p4_link_host.arq._outbuf) == b'TEST\r'

@pytest.mark.parametrize('managed',[False,True])
@pytest.mark.parametrize('recovery',['continuation','exhaustion'])
def test_c3_real_recovery_rejects_frozen_initial_and_final_guard(managed,recovery):
    m,rx,tx,live,captures,keys,counts,feeds,replays = lifecycle(managed,recovery)
    assert captures and keys
    # Retain an actually authorized request. Exhaustion has four prior keys;
    # this replay is explicitly additional stale work, not a concurrent fifth.
    captured,guard,cells,waveform = captures[0]
    assert sum(p is captured for p in keys) == 1
    assert cells['proposal'] is captured and guard.__code__.co_freevars == tuple(cells)
    assert all(c.cell_contents is cells[n] for n,c in zip(guard.__code__.co_freevars,guard.__closure__))
    assert hashlib.sha256(guard.__code__.co_code).hexdigest()
    assert any(p.word=='X27' and p.state==p4turn.State.IRS_ANSWER for p in keys)
    if recovery=='exhaustion':
        assert counts[:5] == [0,1,2,3,4]
        assert all(p.word=='Y27' for p in keys[:4]) and keys[4].word=='X27'
        assert not any(p.word=='Y27' for p in keys[4:] if p.stream_sample<keys[3].stream_sample+960000)
    else:
        assert any(g is not None and prior==p4turn.State.IRS_BREAKIN and state==p4turn.State.IRS_ANSWER
                   for at,g,before,mstate,a,b,prior,state in feeds)
    assert replays==['initial','final']
    emitted=[e for e in tx.p4_probe.link_events if e['event']=='Keyed']
    assert len(emitted)==len(keys)==len(tx.p4_probe.emissions)
    for p,e,em in zip(keys,emitted,tx.p4_probe.emissions):
        assert e['at']==p.stream_sample==em['audio_start'] and e['word']==p.word==em['word']
        assert e['accepted_target']==p.accepted_target and e['current_slot']==p.current_slot
