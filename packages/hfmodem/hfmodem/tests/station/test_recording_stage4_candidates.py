# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Manual QSO decisions belong to one generation and precede text append."""
import json
from pathlib import Path

import pytest

from hfmodem.core import recording, wav
from hfmodem.tests.station.recording_stage4_support import (
    audio_files, one_session, finish, flush_all, bounded, PEER,
)
from hfmodem.tests.station.recording_live_support import OTHER
from hfmodem.tests.station.recording_support import restore, snapshot
from hfmodem.tests.station.test_recording_stage4_voice import (
    select, raw, heard, draft, send,
)


def rows(state):
    flush_all(state)
    rec = one_session(state, 'contacts')
    return rec, list(rec.index['contacts'].values())


def end(owner, action):
    if action == 'reset':
        owner.terminal.key('c')
        if owner.session.pending is not None:
            owner.session.pending.result(timeout=5)
            owner.session.cancel_draft()
    elif action == 'replacement':
        select(owner, OTHER)
    elif action == 'log':
        owner.session.log_contact()
        owner.session.log_contact()
    bounded(owner.session.close)


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('order', ('rx-tx', 'tx-rx'))
@pytest.mark.parametrize('ending', ('reset', 'replacement', 'teardown', 'log'))
def test_manual_two_sided_generation_kept_once_for_either_over_order(
        voice_owner, stage4, waveform, order, ending):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner)
    if order == 'rx-tx':
        heard(owner)
        send(owner)
    else:
        send(owner)
        heard(owner)
    end(owner, ending)
    finish(stage4)
    rec, contacts = rows(stage4)
    kept = [row for row in contacts if row.get('peer') == PEER and row.get('keep')]
    assert len(kept) == 1, 'one manual generation has exactly one frozen keep decision'
    assert audio_files(stage4.root)
    if ending != 'log':
        assert kept[0]['state'] == 'partial'


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('evidence', ('selection', 'preview', 'draft', 'cq', 'self-copy', 'unrelated-peer'))
def test_manual_unproven_candidate_discards(voice_owner, stage4, waveform, evidence):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner)
    if evidence == 'preview': send(owner, preview=True)
    elif evidence == 'draft': draft(owner)
    elif evidence == 'cq': send(owner, kind='cq')
    elif evidence == 'self-copy':
        heard(owner, self_copy=True)
        send(owner)
    elif evidence == 'unrelated-peer':
        heard(owner, OTHER)
        send(owner)
    bounded(owner.session.close)
    finish(stage4)
    assert audio_files(stage4.root) == [], 'selection, preview, CQ and unrelated audio do not prove participation'
    _, contacts = rows(stage4)
    assert not any(row.get('keep') for row in contacts)


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_operator_log_keeps_long_candidate_before_append(voice_owner, stage4, monkeypatch, waveform):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner, 125)
    rec, contacts = rows(stage4)
    assert len(contacts) == 1 and contacts[0]['provisional'] is True
    directory = rec.directory / contacts[0]['directory']
    assert json.loads((directory / 'candidate.json').read_text())['state'] == 'undecided'
    assert len(audio_files(directory)) > 60, 'active QSO must outlive the RAM ring and tiny segment limit'
    finalized = [d for d in rec.index['segments'] if d.get('state') == 'finalized']
    assert not finalized, 'candidate segments remain uncompressed until the keep decision'
    opened = Path.open
    decisions = []
    def append(path, *args, **kwargs):
        if path == owner.directory / 'contacts.jsonl' and args and 'a' in args[0]:
            assert any(row.get('keep') is True for row in rec.index['contacts'].values())
            decisions.append(True)
            raise OSError('synthetic contact append failure')
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', append)
    with pytest.raises(OSError, match='contact append failure'):
        owner.session.log_contact()
    assert decisions == [True]
    # Closing stores every one of the candidate's 125 segments, republishing the
    # index after each; the bound is for a hang, not for speed.
    bounded(owner.session.close, timeout=60)
    finish(stage4)
    kept = json.loads((directory / 'candidate.json').read_text())
    assert kept['state'] == 'keep'
    before = {p: p.read_bytes() for p in audio_files(directory)}
    assert before
    voice_owner('contacts', waveform, name='new-voice-directory')
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_partner_replacement_cannot_borrow_previous_generation_overs(voice_owner, stage4, waveform):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner)
    heard(owner)
    send(owner)
    select(owner, OTHER)
    raw(owner)
    bounded(owner.session.close)
    finish(stage4)
    _, contacts = rows(stage4)
    assert len([row for row in contacts if row.get('peer') == PEER and row.get('keep')]) == 1
    assert not any(row.get('peer') == OTHER and row.get('keep') for row in contacts)


@pytest.mark.parametrize('decision', ('undecided', 'keep'))
def test_interrupted_owner_candidate_cleanup_from_new_voice_directory(
        voice_owner, stage4, decision):
    owner = voice_owner()
    select(owner)
    raw(owner, 2)
    rec, contacts = rows(stage4)
    assert len(contacts) == 1
    directory = rec.directory / contacts[0]['directory']
    if decision == 'keep':
        owner.session.log_contact()
        flush_all(stage4)
    saved = snapshot(directory)
    assert json.loads((directory / 'candidate.json').read_text())['state'] == decision
    assert recording.main(['discard-candidate', str(directory)]) != 0, 'active owner lifetime lock must refuse cleanup'
    bounded(owner.session.close)
    finish(stage4)
    restore(directory, saved)
    voice_owner(name='new-voice-directory')
    foreign = directory / 'unowned.txt'
    foreign.write_text('unrelated record')
    before = snapshot(directory)
    rc = recording.main(['discard-candidate', str(directory)])
    if decision == 'keep':
        assert rc != 0 and snapshot(directory) == before
    else:
        assert rc == 0 and audio_files(directory) == []
        assert foreign.read_text() == 'unrelated record'


@pytest.mark.parametrize('refusal', ('identity', 'unknown', 'interrupted-cleanup'))
def test_owner_candidate_exact_path_refusals_and_retry(voice_owner, stage4, monkeypatch, refusal):
    owner = voice_owner()
    select(owner)
    raw(owner, 2)
    rec, contacts = rows(stage4)
    directory = rec.directory / contacts[0]['directory']
    saved = snapshot(directory)
    bounded(owner.session.close)
    finish(stage4)
    restore(directory, saved)
    voice_owner(name='new-voice-directory')
    manifest = directory / 'candidate.json'
    if refusal != 'interrupted-cleanup':
        facts = json.loads(manifest.read_text())
        if refusal == 'identity': facts['candidate_id'] = 'different-candidate'
        else: facts['state'] = 'unknown'
        manifest.write_text(json.dumps(facts))
        before = snapshot(directory)
        assert recording.main(['discard-candidate', str(directory)]) != 0
        assert snapshot(directory) == before
    else:
        unlink, removed = Path.unlink, []
        def interrupted(path, *a, **k):
            result = unlink(path, *a, **k)
            if path.suffix == '.wav' and not removed:
                removed.append(path)
                raise OSError('synthetic interrupted cleanup')
            return result
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', interrupted)
            assert recording.main(['discard-candidate', str(directory)]) != 0
        assert removed and manifest.exists()
        assert recording.main(['discard-candidate', str(directory)]) == 0
        assert audio_files(directory) == []


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_manual_operator_end_retains_exact_available_pre_and_post_roll(voice_owner, stage4, waveform):
    owner = voice_owner('contacts', waveform)
    raw(owner, 2)
    select(owner)
    raw(owner, 2)
    owner.session.log_contact()
    raw(owner, 5)
    bounded(owner.session.close)
    finish(stage4)
    rec, contacts = rows(stage4)
    assert len(contacts) == 1
    rx = next(lane for lane in rec.lanes.values() if 'rx' in lane.name)
    reference = contacts[0]['lanes'][rx.name]
    assert reference['actual'] == [rx.origin, rx.origin + 9 * 48000]
    assert reference['missing_pre_roll'] == 58 * 48000
    assert reference['missing_post_roll'] == 0
    assert len(reference['segments']) == 9


def test_owner_crash_after_keep_before_storage_remains_protected(voice_owner, stage4, monkeypatch):
    import threading
    owner = voice_owner()
    select(owner)
    raw(owner, 2)
    rec, contacts = rows(stage4)
    directory = rec.directory / contacts[0]['directory']
    entered, release = threading.Event(), threading.Event()
    store = wav.store
    def blocked(path, **kwargs):
        entered.set()
        assert release.wait(5), 'candidate storage was not released'
        return store(path, **kwargs)
    monkeypatch.setattr(wav, 'store', blocked)
    try:
        owner.session.log_contact()
        assert entered.wait(5), 'kept candidate did not reach deferred storage'
        assert json.loads((directory / 'candidate.json').read_text())['state'] == 'keep'
        saved = snapshot(directory)
    finally:
        release.set()
    bounded(owner.session.close)
    finish(stage4)
    restore(directory, saved)
    voice_owner(name='new-voice-directory')
    before = snapshot(directory)
    assert recording.main(['discard-candidate', str(directory)]) != 0
    assert snapshot(directory) == before


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('peer_over', (False, True))
def test_automatic_closing_exchange_requires_both_sides(voice_owner, stage4, waveform, peer_over):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner)
    if peer_over:
        heard(owner)
    send(owner, kind='close')
    bounded(owner.session.close)
    finish(stage4)
    rec, contacts = rows(stage4)
    assert bool(audio_files(stage4.root)) == peer_over
    assert len([row for row in contacts if row.get('keep')]) == int(peer_over)


def test_queued_receive_cannot_prove_a_reselected_generation(voice_owner, stage4, monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from hfmodem.contact import runtime
    from hfmodem.contact.audio import Segment
    owner = voice_owner()
    select(owner)
    raw(owner)
    session = owner.session
    session._capture(np.full((4800, 1), .125, np.float32), 4800, NS(), False)
    owner.terminal.key('c')
    select(owner)
    class Segments:
        def __init__(self, *a): pass
        def feed(self, audio, probability, own, at):
            return Segment(audio, own, at, at + .1, .98)
        def finish(self): return None
    monkeypatch.setattr(runtime, 'Segmenter', Segments)
    request = session.speech.request
    def once(op, **values):
        result = request(op, **values)
        if op == 'vad': session.stop.set()
        return result
    monkeypatch.setattr(session.speech, 'request', once)
    bounded(session._receive)
    event = session.events.get(timeout=5)
    assert event['kind'] == 'heard'
    owner.terminal.heard_event(event)
    session.stop.clear()
    send(owner)
    bounded(session.close)
    finish(stage4)
    _, contacts = rows(stage4)
    assert not any(row.get('keep') for row in contacts)
