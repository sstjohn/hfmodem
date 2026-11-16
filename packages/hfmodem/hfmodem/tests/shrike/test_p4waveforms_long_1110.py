# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import onair, p4robustlongtx, p4waveforms
from hfmodem.shrike.p4turn import Kind
from hfmodem.tests.shrike.test_p4turn_long_1110 import caller, ladder
from hfmodem.tests.shrike.test_p4robustlongtx import recover_audio_body


def key(p):
    return p4waveforms.cache_key(p.kind, p.word, p.value, p.header, p.payload,
                               p.status, p.repeat, p.chirp_slot, p.level)


@pytest.mark.parametrize('level,size', [(2,40),(3,69),(4,141)])
def test_reachable_long_keys_and_audio_loopback(level, size):
    m, _ = caller(b'B'*(size+1), long_level=level)
    p = ladder(m)
    cache = p4waveforms.Waveforms('off')
    p4waveforms.prepare(cache, m)
    assert key(p) in cache
    assert all(k[-2] == level and len(k[4]) <= size for k in cache if k[0] == Kind.LONG)
    for repeat in (False, True):
        q = replace(p, repeat=repeat)
        wave = cache[key(q)]
        r, _ = recover_audio_body(wave, level)
        assert r.frame.status.byte == 0x61 and r.frame.status.counter == 1
        assert r.frame.data == b'B'*size
        np.testing.assert_array_equal(wave, p4robustlongtx.robust_long(q.payload, q.status, level, repeat=repeat))
    assert not np.array_equal(cache[key(p)], cache[key(replace(p,repeat=True))])
    np.testing.assert_array_equal(cache[key(p)][20000:], cache[key(replace(p,repeat=True))][20000:])
    m.outbox.commit()
    m._data_packet = m._next = None
    p4waveforms.prepare(cache, m)
    assert key(p) not in cache


def test_long_keys_distinguish_level_and_repeat_but_chirp_key_unchanged():
    keys = {p4waveforms.cache_key(Kind.LONG, payload=b'x', level=level, repeat=repeat)
            for level in (2,3,4) for repeat in (False,True)}
    assert len(keys) == 6
    assert p4waveforms.cache_key(Kind.CHIRP, payload=b'x', repeat=True, chirp_slot=3) == (
        Kind.CHIRP,None,None,'A',b'x',0,True,1)


def test_modem_long_flags_and_ladder_validation():
    args = onair.build_parser().parse_args(['--p4-data','long'])
    assert (args.p4_long_level,args.p4_ladder_admit,args.p4_spug,args.p4_iss_u27) == (3,'u27,x27','queued','repeat')
    args = onair.build_parser().parse_args(['--p4-long-level','4','--p4-ladder-values','8,59',
                                      '--p4-spug','off','--p4-iss-u27','ack'])
    assert args.p4_ladder_values == (8,59)
    for value in ('', '64', '-1', '8,,59', 'x'):
        with pytest.raises(SystemExit):
            onair.build_parser().parse_args(['--p4-ladder-values',value])
    with pytest.raises(SystemExit):
        onair.build_parser().parse_args(['--p4-ladder','long'])


def test_driver_keys_prepared_longs_on_long_grid_and_records_probe_fields(monkeypatch):
    from hashlib import sha256
    from hfmodem.shrike import p4robusttx
    from hfmodem.shrike.p4turn import P4Turn, State
    from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

    audio = np.zeros(13*FS, np.float32)
    for at, name in [(0.93,'X27'), (2.18,'X27'), (3.43,'U27'),
                     (7.15,'U27'), (10.90,'X27')]:
        wave = p4robusttx.robust_word(name)
        start = round(at*FS)
        audio[start:start+len(wave)] = wave
    stream = Stream(audio)
    tx = Tx(stream)
    tx.settle = .037
    tx.p4_link_options.p4_data = 'long'
    tx.p4_link_options.p4_normal_ack = 'toggle'
    tx.p4_link_options.p4_link_timeout = 12
    tx.p4_link_host.arq.on_host_data(b'A'*31+b'B'*69)
    machines = []

    def seed(m, *args, **kwargs):
        machines.append(m)
        m._begin_data_turn(0, State.ISS_DATA_ACK)
        m.keyed()
    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    grids, keyed = [], []
    original_aim = tx.aim
    original_tx = tx._tx
    def aim(grid, slot):
        grids.append(grid)
        original_aim(grid, slot)
    def emit(wave, label):
        keyed.append((tx.raster.boundary(tx.slot), sha256(wave.tobytes()).hexdigest()))
        original_tx(wave, label)
    tx.aim, tx._tx = aim, emit
    onair._run_p4_link(tx, stream, onair._MasterGrid(0,60000,0,
        packet_n=36000,cs_n=12000,d_max_n=4800))
    rows = [r for r in tx.p4_probe.link_events if r['event']=='Keyed' and r['kind']==Kind.LONG]
    assert len(rows) == 2
    assert [r['at'] for r in rows] == [round(3.75*FS),round(7.5*FS)]
    assert [r['header'] for r in rows] == ['new','rq']
    assert [r['repeat_index'] for r in rows] == [0,1]
    assert {r['status'] for r in rows} == {0x41}
    for r in rows:
        assert r['level'] == 3 and r['counter'] == 1 and r['final']
        assert r['recut_from'] is None and r['ladder_index'] == 1
        assert (r['at'],r['waveform_sha256']) in keyed
        assert next(g for g in grids if g.anchor==r['at']).slot_n == 180000
    assert not [r for r in tx.p4_probe.link_events if r['event']=='Refused' and r['kind']==Kind.LONG]
    decisions = [r for r in tx.p4_probe.link_events if r['event']=='IssAck']
    assert [(r['word'],r['expected'],r['verdict']) for r in decisions][-2:] == [
        ('U27','X27','repeat'), ('X27','X27','ack')]
    assert any(e['event'] == 'data-turn-ended' for e in machines[0].link_events)
    assert tx.p4_link_host.arq._buffer_raw == 0
