# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest
import json
from pathlib import Path

from hfmodem.shrike import onair
from hfmodem.shrike.p4turn import State
from hfmodem.shrike.p4observation import Registered
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx


@pytest.mark.parametrize('tau', [.9194, .9240, .9379, .9432, .947])
@pytest.mark.parametrize('late', [False, True])
def test_step_after_an_ack_is_on_our_raster(tau, late):
    m, _ = machine(prekey_lead=2131/FS)
    m.ladder = (7, 38)
    m._short(0, 0)
    m.keyed()
    onset = round(tau*FS)
    now = 60000-m.lead+1 if late else onset+round(.215*FS)
    m.observe(word('X27', onset, responds_to=0), now=now)
    assert m.decide(now).stream_sample == (120000 if late else 60000)


@pytest.mark.parametrize('step', ['ladder', 'body', 'chirp', 'breakin', 'grant'])
def test_every_word_triggered_packet_is_on_our_raster(step):
    m, a = machine(data='chirp' if step == 'chirp' else 'body', prekey_lead=2131/FS)
    a.on_host_data(b'A'*90)
    if step == 'ladder':
        m._short(0, 1)
        name = 'U27'
    elif step in ('breakin', 'grant'):
        m._body(0, 0x40, State.IRS_BREAKIN if step == 'breakin' else State.ISS_GRANT)
        name = 'V27'
    else:
        m._data(0)
        name = 'X27' if step == 'chirp' else 'V27'
    m.keyed()
    onset = round((3.4232 if step == 'chirp' else .9432)*FS)
    now = onset+round(.250*FS)
    m.observe(word(name, onset, responds_to=0), now=now)
    assert m.decide(now).stream_sample == (180000 if step == 'chirp' else 60000)


def test_deferred_confirmation_skips_only_the_missed_slot():
    m, a = machine(data='body', prekey_lead=2131/FS)
    a.on_host_data(b'A'*31)
    m._short(0, 1)
    m.keyed()
    onset = round(.9432*FS)
    early = word('U27', onset, partial=True, responds_to=0)
    m.observe(early, now=onset+round(.215*FS))
    m.silence(60000-m.lead)
    m.observe(word('U27', onset, outcome=Registered.CONFIRMED,
                   replaced=early.observation, responds_to=0), now=58000)
    assert m.decide(58000).stream_sample == 120000


@pytest.mark.parametrize('cost_ms,at', [(0, 60000), (90, 120000)])
def test_driver_keys_the_step_on_the_raster(monkeypatch, cost_ms, at):
    from hfmodem.shrike import p4robusttx

    audio = np.zeros(4*FS)
    onset = round(.9432*FS)
    wave = p4robusttx.robust_word('X27')
    audio[onset:onset+len(wave)] = wave
    stream = Stream(audio)
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.key_notice = 1652
    stream.transmit = True
    stream.clamp_late = lambda t: max(0, stream.sample_now()+stream.key_notice-t)
    tx = Tx(stream)
    tx.p4_probe.emissions[0].update(value=7, header='A')
    tx.p4_probe.values = (7, 38)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    delayed = []

    def timed(read):
        def run(*args, **kwargs):
            result = read(*args, **kwargs)
            if not delayed and any(r.observation.word == 'X27' for r in result):
                stream.wall = stream.pos+round(cost_ms/1000*FS)
                delayed.append(True)
            return result
        return run

    for name in ('read_partial', 'read_window'):
        monkeypatch.setattr(tx.sessrx.p4, name, timed(getattr(tx.sessrx.p4, name)))
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert delayed
    emitted = tx.p4_probe.emissions[1:]
    assert emitted[0]['value'] == 38
    assert emitted[0]['audio_start'] == at
    assert all(e.get('refused') is None for e in emitted)


@pytest.mark.parametrize('arm,first_v38', [('1749', 1862733), ('1044', 2162733),
                                         ('0943', 1142738)])
def test_recorded_ack_advances_on_the_original_raster(arm, first_v38):
    capture = Path.cwd() / 'captures' / f'onair-0928-{arm}'
    if not (capture / 'stream.flac').exists():
        pytest.skip('local capture absent')
    audio = onair.session.load_wav(str(capture / 'stream.flac'), FS)
    entry = json.loads((capture / 'p4-probe.json').read_text())['emissions'][0]
    stream = Stream(audio)
    stream.pos = entry['audio_end']
    tx = Tx(stream)
    tx.p4_probe.emissions = [entry]
    key = tx._tx

    def flushing_key(wave, what):
        key(wave, what)
        stream.pos = tx.tx_end

    tx._tx = flushing_key
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    emissions = tx.p4_probe.emissions
    assert next(e['audio_start'] for e in emissions if e['value'] == 38) == first_v38
    assert all((e['audio_start']-entry['audio_start']) % 60000 == 0 for e in emissions)
    assert not any(e['event'] == 'off-turnaround' for e in tx.p4_probe.link_events)
