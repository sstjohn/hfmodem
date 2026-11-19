# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""2026-11-16 22:08-22:17Z mail arms replayed under the declared sample clock:
a K4MSU turn taken on another station's V27s, the WS8EOC answer chain,
unkeyed slots and a VE3KPG close keyed over the peer's turn."""

import json

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.p4observation import PEER_WORD_CFO_HZ
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.tests.shrike import test_p4_irs_slot_fixes_1114 as slots
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx

slots.CAPTURES.update({
    'ws8eoc-1116': 'onair-20261116T220806820479Z-13ef68ca-eb46-468b-81c3-bd1c32d02229',
    'k4msu-1116': 'onair-20261116T221102981815Z-7f4574c0-1e24-4f9b-bd51-7a09c45b7646',
    've3kpg-1116': 'onair-20261116T221506901608Z-7c30f553-975e-40a6-8abb-340b191cc223',
})
# The live mail client queued its proposal here (arm.log 22:13:33Z, 22:16:36Z).
PROPOSAL = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;'
K4MSU = dict(queued=(117.4, PROPOSAL))
VE3KPG = dict(queued=(57.4, PROPOSAL))
ROBUST = 'onair-20261116T221745091177Z-e6ab9e8a-b882-4eff-87cb-de69544cddda'
CARRIER = {'X27': 3.65, 'V27': -1.45, 'Y27': 1.40, 'U27': -3.30, 'T27': 3.80}


def keyed(probe, start=0., end=1e9):
    return slots.keyed(probe, start, end)


def every_slot(probe, start, end):
    decisions = [e for e in probe.link_events if e['event'] == 'Decision'
                 and start <= e['at'] / FS < end]
    assert decisions
    assert all(d['proposed_at'] is not None for d in decisions), [
        (d['at'] / FS, d['reason']) for d in decisions if d['proposed_at'] is None]
    keys = {e['at'] for e in keyed(probe)}
    assert all(d['proposed_at'] in keys for d in decisions)
    return decisions


def test_k4msu_turn_is_taken_on_the_peers_carrier():
    probe = slots.replay('k4msu-1116', **K4MSU)
    lock = np.median([h['cfo_hz'] for h in probe.heard if h['kind'] in ('body', 'normal', 'chirp')
                      and not h['provisional'] and h.get('cfo_hz') is not None])
    words = [h for h in probe.heard if h['kind'] == 'word' and h.get('word') in CARRIER
             and not h['provisional'] and 125 < h['onset_sample'] / FS < 175]
    off = [h for h in words if abs(h['cfo_hz'] - CARRIER[h['word']] - lock) > PEER_WORD_CFO_HZ]
    # The other carrier's V27s are still heard; they neither take the lock nor grant.
    assert off and all(110 < abs(h['cfo_hz'] - CARRIER[h['word']] - lock) < 115 for h in off)
    turns = [e for e in probe.link_events if e['event'] == 'turn']
    assert turns
    for turn in turns:
        rows = [h for h in words if h['onset_sample'] == turn['at']]
        assert rows and all(abs(h['cfo_hz'] - CARRIER['V27'] - lock) < 2 for h in rows)


def test_k4msu_weak_head_slot_repeats_the_held_answer():
    probe = slots.replay('k4msu-1116', **K4MSU)
    every_slot(probe, 39.5, 53.5)
    assert any(abs(e['at'] / FS - 41.926) < .01 for e in keyed(probe))


def test_ws8eoc_chain_answers_each_packet_with_the_word_the_gateway_took():
    # On the tape V27 at 28.18 drew c2 (header A at 28.51), V27 at 29.43 drew a
    # c2 repeat and X27 at 30.68 drew c3 (31.01): c1 went on V27, c2 on X27.
    # The c0 word (our deferred X27 after the 14.43 repeat request, where the
    # gateway took that V27) stays an on-air question for --p4-first-ack.
    probe = slots.replay('ws8eoc-1116')
    words = [(round(e['at'] / FS, 2), e['word'], e['ack_identity'][0] & 3 if e['ack_identity'] else None)
             for e in keyed(probe, 14.4, 32.) if e['kind'] == 'word']
    assert words[:3] == [(14.43, 'V27', None), (15.68, 'X27', 0), (16.93, 'V27', 1)]
    chain = {c: {w for _, w, n in words if n == c} for c in (1, 2, 3)}
    assert chain == {1: {'V27'}, 2: {'X27'}, 3: {'V27'}}
    assert [w for t, w, _ in words if t in (28.18, 29.43, 30.68)] == ['V27', 'X27', 'X27']
    assert not any(e['event'] == 'deferred-answer-cancelled' and 14. < e['at'] / FS < 16.
                   for e in probe.link_events)


@pytest.mark.xfail(strict=True, reason=(
    'the 42.260 decision skips the 43.180 slot (no new registration) and keys 44.430, '
    'as the arm did: margin +1224 ms, missed_slots=0'))
def test_ws8eoc_slow_reads_key_43_18():
    probe = slots.replay('ws8eoc-1116', read_partial=.055, read_window=.018)
    every_slot(probe, 31., 82.)


def test_ve3kpg_word_grant_close_asks_for_the_turn_back_in_word_slots():
    probe = slots.replay('ve3kpg-1116', **VE3KPG)
    grant = [e for e in keyed(probe) if e.get('purpose') == 'close-grant']
    assert grant and grant[0]['kind'] == 'word'
    after = keyed(probe, grant[0]['at'] / FS + .1)
    assert after and all(e['kind'] == 'word' for e in after)
    assert after[0]['word'] == 'Y27' and after[0]['purpose'] == 'close-request'


def test_ve3kpg_robust_probe_hands_the_link_its_first_slot(tmp_path, monkeypatch):
    capture = RECORD / 'operator-data/captures' / ROBUST
    recorded = json.loads(require_input(capture / 'p4-probe.json').read_text())
    audio = slots.tape(capture)
    for e in recorded['emissions']:
        audio[e['audio_start']:e['audio_end']] = 0
    monkeypatch.setattr(onair, '_load', lambda _: audio.copy())
    live = onair._ReplayInput('ve3kpg-1116-robust')
    first = recorded['emissions'][0]['audio_start']
    live.pos = first - FS // 2
    live.xruns, live.lost, live.lost_at = 0, 0, []
    seed = Tx(live)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.live, tx.sessrx = live, seed.sessrx
    tx.p4_probe = probe = seed.p4_probe
    probe.emissions.clear()
    probe.grant_sample = recorded['grant_sample']
    probe.timeout = 20.
    probe.packet, probe.cadence = 'robust', 'consecutive'
    probe.values, probe.headers, probe.attempts = (6, 9, 8), ('A', 'A', 'A'), 3
    tx.p4_link_options = seed.p4_link_options
    tx.sessrx.host = seed.p4_link_host
    tx.qrm_guard = False
    grid = onair._MasterGrid(first % 60000, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)
    slot = next(s for s in range(64) if grid.boundary(s) >= first)
    assert grid.boundary(slot) == first
    tx.aim(grid, slot)
    handoffs = []
    monkeypatch.setattr(onair, '_run_p4_link',
                        lambda tx, live, raster: handoffs.append(live.pos) or 'handed off')
    assert onair._run_p4_probe(tx, live, grid, in_slot=True) == 'handed off', probe.reason
    last = probe.emissions[-1]['audio_start']
    assert [e['audio_start'] for e in probe.emissions] == [
        e['audio_start'] for e in recorded['emissions'][:3]]
    # The link reads the last short's answer window at the instant it reads
    # every later word slot, ahead of its decision for the 6.367 key.
    assert handoffs == [last + round(1.138 * FS)]
    assert grid.boundary(grid.next_slot(probe.emissions[-1]['slot'])) == last + round(1.25 * FS)
