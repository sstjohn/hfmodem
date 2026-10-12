# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.shrike import onair, p4robusttx
from hfmodem.shrike.p4observation import Registered
from hfmodem.tests.shrike.test_p4turn import FS, machine, word
from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx


@pytest.mark.parametrize('now,expected', [(50000, 89120), (5*FS, 269120)])
def test_expired_breakin_resumes_on_an_eligible_raster_slot(now, expected):
    m, _ = machine(prekey_lead=2131/FS)
    m.start(0)
    m.keyed()
    early = word('Y27', 44000, partial=True, responds_to=0)
    m.observe(early, now=45000)
    m.observe(word('Y27', 44000, outcome=Registered.EXPIRED,
                   replaced=early.observation, responds_to=0), now=now)
    m.silence(now)
    proposal = m.decide(now)
    # S5(b), 2031: an unresolved head anchors fallback at onset + .940,
    # advancing by whole 1.250 s cycles when the first slot is already late.
    assert proposal.stream_sample == expected
    assert proposal.stream_sample >= now+m.lead
    assert proposal.value == m.ladder[0]


@pytest.mark.parametrize('cost_ms,expected', [(0, 60000), (90, 120000)])
def test_driver_explains_a_slot_lost_during_read(monkeypatch, capsys, cost_ms, expected):
    audio = np.zeros(3*FS)
    onset = round(.9432*FS)
    wave = p4robusttx.robust_word('X27')
    audio[onset:onset+len(wave)] = wave
    stream = Stream(audio)
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    tx = Tx(stream)
    tx.sessrx.p4.packet_read = dict(at=-5*FS, value=53, reason='decoded')
    tx.p4_probe.values = (7, 38)
    tx.p4_probe.emissions[0].update(value=7, header='A')
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    read = tx.sessrx.p4.read_partial
    delayed = []

    def delayed_read(*args, **kwargs):
        records = read(*args, **kwargs)
        if not delayed and any(r.observation.word == 'X27' for r in records):
            stream.wall = stream.pos+round(cost_ms/1000*FS)
            delayed.append(True)
        return records

    monkeypatch.setattr(tx.sessrx.p4, 'read_partial', delayed_read)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert tx.p4_probe.emissions[1]['audio_start'] == expected
    rows = [r for r in tx.p4_probe.link_events if r['event'] == 'Decision']
    assert rows and rows[0]['scheduled_at'] == 60000
    assert rows[0]['proposed_at'] == expected
    assert rows[0]['missed_slots'] == int(cost_ms > 0)
    assert rows[0]['work_ms']['read_partial'] >= 0
    assert rows[0]['postdecision_work_ms']['read_window'] >= 0
    assert rows[0]['packet_read'] == {}
    heard = next(r for r in tx.p4_probe.heard if r['word'] == 'X27')
    assert heard['provisional'] and heard['score'] > .88
    assert heard['received_at_sample'] >= onset
    output = capsys.readouterr().out
    assert 'provisional' in output and '[p4 decision]' in output
    if cost_ms:
        assert 'missed_slots=1' in output
