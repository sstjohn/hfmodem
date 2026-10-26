# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Narrow existing-fixture controls; archived CRC state is not fresh RF proof.

The genuine decoded repeat proof is reused from semantic-final-r2 (13 passes).
These controls exercise selection, real turn lifecycle, cutoff and detachment.
"""
from dataclasses import replace
from pathlib import Path
import json
from hfmodem.tests.shrike import p4_replay_history_support_1012 as helper
from types import SimpleNamespace

import pytest
from hfmodem.shrike import p4frame, p4robusttx
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.shrike.test_p4turn import machine

ROOT = Path(__file__).resolve().parents[5]
H = vars(helper)
FS, AT = 48000, 1908010
DATA = '524d53205472696d6f646520312e342e342e30205765'


def row(at=AT, **changes):
    return dict(kind='chirp', onset_sample=at, received_at_sample=at+161307,
        capture_end_sample=at+159628, status=97, data=DATA, cfo_hz=.444286032449214,
        outcome='confirmed', provisional=False, weak=False, unexpected=False, **changes)


def turn():
    m, _ = machine(role='iss', ladder='none', data='body', chirp_ack='alternate')
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=lambda b: delivered.append(b.hex()), supervisor=lambda *a: None)
    return m, delivered


def observation(h, kind=Kind.CHIRP):
    frame = p4frame.decode_field(p4frame.encode_field(bytes.fromhex(h['data']),
        p4frame.Status.from_byte(h['status'])), data_bytes=22)
    at = h['onset_sample']
    return Observation(kind, FS, at, 0, at, 3.3075, h['cfo_hz'], 0., frame=frame)


def test_selection_is_chronological_physical_root_not_payload_identity():
    rows = [row(AT+180000), row(), row()]
    selected, excluded = H['completed_history'](rows, AT+500000, FS)
    assert [s['row_index'] for s in selected] == [1, 0]
    assert [s['root'][1] for s in selected] == [AT, AT+180000]
    assert excluded[0]['row_index'] == 2
    assert excluded[0]['reasons'] == ['same-physical-root-refinement']


@pytest.mark.parametrize('change,reason', [
    ({'received_at_sample':AT+500000}, 'receipt-not-before-start'),
    ({'capture_end_sample':AT+500000, 'received_at_sample':AT+500001}, 'capture-end-not-before-start'),
    ({'capture_end_sample':AT+158759}, 'receipt-support-does-not-prove-completion'),
    ({'received_at_sample':AT+159627}, 'receipt-support-does-not-prove-completion'),
    ({'provisional':True}, 'provisional'), ({'weak':True}, 'weak'),
    ({'unexpected':True}, 'unexpected'), ({'outcome':'expired'}, 'registration-not-accepted'),
    ({'outcome':'duplicate'}, 'registration-not-accepted'), ({'data':None}, 'no-completed-status-data')])
def test_selection_admission_and_recorded_coordinate_negatives(change, reason):
    h = row(); h.update(change)
    selected, excluded = H['completed_history']([h], AT+500000, FS)
    assert selected == [] and reason in excluded[0]['reasons']


@pytest.mark.parametrize('delta,reason', [(158760,'wave-end-not-before-start'),
                                         (163200,'onset-plus-3.4-not-before-start')])
def test_strict_wave_and_nominal_answer_cutoffs(delta, reason):
    selected, excluded = H['completed_history']([row()], AT+delta, FS)
    assert selected == [] and reason in excluded[0]['reasons']


@pytest.mark.parametrize('change', [{'data':'00'*22}, {'source_root':['chirp',AT-1]},
                                   {'replaced':{'kind':'word','onset_sample':AT}}])
def test_conflicting_or_unresolved_physical_identity_stops(change):
    h = row(); h.update(change)
    with pytest.raises(H['HistoryConflict']):
        H['completed_history']([row(),h], AT+500000, FS)


def test_actual_full_word_cutoff_stops_before_key_without_shortening():
    m, delivered = turn()
    length = len(p4robusttx.robust_word('X27', FS))
    audit = {}
    with pytest.raises(H['HistoryConflict'], match='full proposed answer'):
        H['seed_history'](m, [row()], AT+163200+length, FS, audit)
    assert m.last is None and audit['actions'][0]['word_samples'] == length
    assert m._next.stream_sample == AT+163200
    assert m.peer_watch.onset == AT and delivered == [DATA]


def test_machine_generated_root_held_ack_and_detached_snapshot(tmp_path):
    m, delivered = turn(); audit = {}
    H['seed_history'](m, [row(),row(AT+180000)], AT+500000, FS, audit)
    assert delivered == [DATA] and m.adapter.result.duplicate
    at = AT+180000
    assert m._held_answer.onset == m.peer_phase.onset == m.peer_watch.onset == at
    assert m._last_ack[1] is m.last and m.last.source_root == (Kind.CHIRP,at)
    assert m.last.receive_epoch == m._receive_epoch == 0 and not m._receive_active
    assert m.adapter.result.receipt is None and m.acked_counter == 1
    rx = P4Receiver(FS, admitted=True, chirp=True)
    rx.expected = True; rx.cfo_hz = row()['cfo_hz']
    rx.clock = Clock(Expect.PACKET, at+180000, .330, 3.75, rx.cfo_hz)
    record, encoded = H['boundary_snapshot'](m, rx, delivered, audit, row(at), 1,
                                            AT+500000, {'fixture':'explicit archived identity'})
    assert json.loads(encoded) == record
    assert record['derived_completed_turn_history']['delivery_journal'] == [DATA]
    assert rx.peer_watch is None and rx.observations == rx.registrations == rx.emissions == []
    old = rx.clock.next_due
    rx.new_cycle()
    delivered.clear(); audit['actions'].clear()
    assert rx.clock.next_due == old+180000
    assert H['serialized'](record) == encoded
    (tmp_path/'snapshot.json').write_text(encoded+'\n')


@pytest.mark.parametrize('bad', [object(), {'nested':[object()]}, float('nan')])
def test_snapshot_rejects_nonprimitive_or_nonfinite_values(bad):
    with pytest.raises(TypeError): H['primitive'](bad)


def test_actual_duplicate_root_stale_onset_and_provisional_do_not_gain_admission():
    m, delivered = turn(); audit = {}
    H['seed_history'](m, [row()], AT+500000, FS, audit)
    watch, last = m.peer_watch, m.adapter.last
    original = observation(row())
    for bad in [original, observation(row(AT-180000)),
                replace(observation(row(AT+180000)), provisional=True)]:
        m.observe(RegisteredObservation(bad, Registered.NEW), now=AT+400000)
        assert m.peer_watch is watch and m.adapter.last == last
    assert delivered == [DATA]


def test_prior_epoch_root_is_denied_after_actual_body_opens_next_epoch():
    m, delivered = turn(); audit = {}
    H['seed_history'](m, [row()], AT+500000, FS, audit)
    old = observation(row())
    h = row(AT+180000); h['status'] = 2
    body = observation(h, Kind.BODY)
    m.observe(RegisteredObservation(body, Registered.NEW), now=body.stream_sample+39000)
    assert m._receive_epoch == 1 and m._receive_active
    assert m._root_epoch[(Kind.CHIRP,AT)] == 0
    source, phase, watch = m.adapter.last, m.peer_phase, m.peer_watch
    before = list(delivered)
    m.observe(RegisteredObservation(old, Registered.NEW), now=AT+400000)
    assert m.adapter.last == source and m.peer_phase is phase and m.peer_watch is watch
    assert delivered == before
