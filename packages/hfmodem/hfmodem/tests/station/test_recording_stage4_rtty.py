# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""RTTY candidates resolve at QSO end, independently of ADIF retries."""
import io
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.core.audio import Block
from hfmodem.lanner import qso
from hfmodem.station.qsolog import QsoLog
from hfmodem.tests.lanner.test_qso import Loopback, Keyer, run
from hfmodem.tests.station.recording_stage4_support import (
    audio_files, finish, one_session, flush_all, LOCAL, PEER,
)
from hfmodem.tests.station.recording_live_support import OTHER


@pytest.fixture
def rtty_owner(stage4, monkeypatch):
    owners = []
    monkeypatch.setattr(qso.rtty, 'Stream', lambda **k: NS(feed=lambda *a, **k: None, events=[]))
    def make(policy='contacts', name='rtty', log=None):
        monkeypatch.setenv('HFMODEM_RECORDING', policy)
        directory = stage4.root / name
        directory.mkdir()
        loop = Loopback(15)
        monkeypatch.setattr(qso.time, 'monotonic', lambda: loop.t)
        log = log or QsoLog(stage4.root / 'contacts.adi', LOCAL)
        keyer = Keyer(loop.sa)
        session = qso.Session(loop.sa, keyer, log, mycall=LOCAL, directory=directory,
                              dial_hz=14080000, out=io.StringIO(), shout=lambda line: None)
        monkeypatch.setattr(session, '_track', lambda: None)
        owners.append(session)
        return NS(session=session, loop=loop, directory=directory, keyer=keyer, log=log)
    yield make
    for session in reversed(owners):
        session.close()
        session.finish(timeout=5)


def feed(owner, seconds, state):
    for _ in range(seconds * 2):
        first = owner.loop.sa.samples
        block = np.full(24000, .125, np.float32)
        for lane in owner.loop.sa._lanes:
            lane.push(Block(first / 48000, first, block))
        owner.loop.sa.samples += len(block)
        owner.loop.t += .5
        owner.session.step()
        assert owner.session._disk.flush(5)
        flush_all(state)


def heard(owner, call=PEER):
    owner.session._decoded([NS(accepted=True, text=f'{LOCAL} DE {call} K ', shift_known=True)])


def select(owner, call=PEER):
    heard(owner, call)
    assert owner.session.set_call() == call


def transmit(owner):
    for ch in 'DE ' + LOCAL + ' K':
        owner.session.type(ch)
    owner.session.turn_over()
    assert run(owner.loop, owner.session, 6, until=lambda: owner.session.state == 'rx')
    assert any(on for on, _ in owner.keyer.events), 'fake output must actually submit an over'


def close(owner):
    owner.session.close()
    owner.session.finish(timeout=5)


@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_rtty_idle_tape_obeys_policy_and_preserves_monitored_pcm(rtty_owner, stage4, policy):
    owner = rtty_owner(policy)
    feed(owner, 3, stage4)
    close(owner)
    finish(stage4)
    assert bool(audio_files(stage4.root)) == (policy == 'everything')
    rec = one_session(stage4, policy)
    assert not rec.index['contacts']
    assert sum(lane.position - (lane.origin or 0) for lane in rec.lanes.values()) >= 3 * 48000
    assert (owner.directory / 'transcript.txt').is_file()


@pytest.mark.parametrize('order', ('rx-tx', 'tx-rx'))
@pytest.mark.parametrize('ending', ('commit', 'replacement', 'teardown'))
def test_rtty_two_sided_candidate_keeps_once_at_real_end(rtty_owner, stage4, order, ending):
    owner = rtty_owner()
    select(owner)
    if order == 'rx-tx':
        heard(owner)
        transmit(owner)
    else:
        transmit(owner)
        heard(owner)
    flush_all(stage4)
    if ending == 'commit':
        owner.session.commit()
        owner.session.commit()
    elif ending == 'replacement':
        select(owner, OTHER)
    close(owner)
    finish(stage4)
    rec = one_session(stage4, 'contacts')
    rows = [row for row in rec.index['contacts'].values() if row.get('peer') == PEER and row.get('keep')]
    assert len(rows) == 1, 'proven two-sided RTTY QSO must resolve exactly once'
    assert audio_files(stage4.root)
    if ending != 'commit':
        assert rows[0]['state'] == 'partial'


@pytest.mark.parametrize('action', ('selection', 'draft', 'cq', 'unrelated-rx', 'refused-tx'))
def test_rtty_unproven_candidates_discard(rtty_owner, stage4, action):
    owner = rtty_owner()
    select(owner)
    feed(owner, 2, stage4)
    if action == 'draft':
        for ch in 'UNSENT': owner.session.type(ch)
    elif action == 'cq':
        owner.session.macro(0)
    elif action == 'unrelated-rx':
        heard(owner, OTHER)
    elif action == 'refused-tx':
        owner.keyer.key = lambda on, note='': not on
        owner.session.type('A')
        owner.session.turn_over()
    close(owner)
    finish(stage4)
    assert audio_files(stage4.root) == [], 'unproven RTTY candidate must discard owned temporary audio'
    rec = one_session(stage4, 'contacts')
    assert not any(row.get('keep') for row in rec.index['contacts'].values())


def test_rtty_operator_commit_without_detected_overs_keeps_long_candidate(rtty_owner, stage4, monkeypatch):
    owner = rtty_owner()
    stored = []
    from hfmodem.core import wav
    store = wav.store
    monkeypatch.setattr(wav, 'store', lambda path, **k: stored.append(path) or store(path, **k))
    select(owner)
    feed(owner, 125, stage4)
    rec = one_session(stage4, 'contacts')
    assert not stored
    assert audio_files(stage4.root), 'active candidate longer than the ring needs provisional segments'
    owner.session.commit()
    close(owner)
    finish(stage4)
    rows = list(rec.index['contacts'].values())
    assert len(rows) == 1 and rows[0]['keep'] is True
    assert stored
    assert sum(d['samples'] for d in rec.index['segments'] if d.get('contact_id')) >= 125 * 48000


def test_rtty_keep_is_frozen_before_failed_adif_and_retry_is_text_only(rtty_owner, stage4, monkeypatch):
    log = QsoLog(stage4.root / 'contacts.adi', LOCAL)
    owner = rtty_owner(log=log)
    select(owner)
    feed(owner, 2, stage4)
    decisions = []
    def failed(record):
        decisions.append(bool(stage4.sessions) and any(
            row.get('keep') is True for row in stage4.sessions[0].index['contacts'].values()))
        raise OSError('synthetic ADIF append failure')
    monkeypatch.setattr(log, 'write', failed)
    owner.session.commit()
    assert owner.session._contacts.flush(timeout=5)
    close(owner)
    finish(stage4)
    assert decisions and all(decisions), 'RTTY must freeze keep before every ADIF append attempt'
    kept = {p: p.read_bytes() for p in audio_files(stage4.root)}
    assert kept
    reopened = QsoLog(log.path, LOCAL)
    retry = qso._Contacts(reopened, lambda line: None)
    try:
        assert retry.flush(timeout=5)
    finally:
        retry.close(5)
    assert log.path.is_file()
    assert all(path.read_bytes() == content for path, content in kept.items())
    assert set(audio_files(stage4.root)) == set(kept)


def test_rtty_old_adif_retry_cannot_promote_new_selection(rtty_owner, stage4):
    log = QsoLog(stage4.root / 'contacts.adi', LOCAL)
    log.start(OTHER)
    _, record = log.record(mode='RTTY', freq_hz=14080000, rst_sent='599', rst_rcvd='599')
    assert log.keep([record])
    owner = rtty_owner(log=log)
    select(owner)
    feed(owner, 2, stage4)
    assert owner.session._contacts.flush(timeout=5)
    close(owner)
    finish(stage4)
    assert log.path.is_file() and OTHER in log.path.read_text()
    assert audio_files(stage4.root) == [], 'an ADIF retry must not promote another candidate'


@pytest.mark.parametrize('action', ('cq', 'idle-key'))
def test_rtty_peer_over_does_not_promote_cq_or_idle_output(rtty_owner, stage4, action):
    owner = rtty_owner()
    select(owner)
    heard(owner)
    if action == 'cq':
        owner.session.macro(0)
    else:
        owner.session.key()
    run(owner.loop, owner.session, 3)
    close(owner)
    finish(stage4)
    assert audio_files(stage4.root) == []
    assert not any(row.get('keep') for row in one_session(stage4, 'contacts').index['contacts'].values())


def test_rtty_fragmented_peer_identification_counts_for_selected_generation(rtty_owner, stage4):
    owner = rtty_owner()
    select(owner)
    for char in f'{LOCAL} DE {PEER} K ':
        owner.session._decoded([NS(accepted=True, text=char, shift_known=True)])
    transmit(owner)
    close(owner)
    finish(stage4)
    assert any(row.get('keep') for row in one_session(stage4, 'contacts').index['contacts'].values())


def test_rtty_run_policy_resolves_before_admission(stage4, monkeypatch, capsys):
    from hfmodem.core import claim, recording
    from hfmodem.tests.station.test_recording_launchers import station_file
    monkeypatch.setenv('HFMODEM_STATION', str(station_file(stage4.root, 'everything')))
    def refuse(*args, **kwargs):
        assert qso.os.environ['HFMODEM_RECORDING'] == 'off'
        raise claim.ClaimError('synthetic admission refused')
    monkeypatch.setattr(claim, 'inherited', refuse)
    assert qso.main(['--audio-in', 'synthetic', '--audio-out', 'synthetic',
                     '--ptt-device', 'synthetic', '--adif', str(stage4.root / 'contacts.adi'),
                     '--recording', 'off']) == 2
    assert capsys.readouterr().out.count(recording.OFF_WARNING) == 1
    assert not stage4.sessions


def test_rtty_cancel_before_card_output_has_no_submitted_over(rtty_owner, stage4):
    owner = rtty_owner()
    select(owner)
    heard(owner)
    for ch in 'DE ' + LOCAL + ' K':
        owner.session.type(ch)
    owner.session.turn_over()
    assert owner.session._tx_stream.written > 0
    assert owner.session._tx_stream.emitted == 0
    assert owner.session._record_tx.position == 0
    owner.session.abort()
    close(owner)
    finish(stage4)
    assert not any(row.get('keep') for row in one_session(stage4, 'contacts').index['contacts'].values())


@pytest.mark.parametrize('interrupted', (False, True))
def test_rtty_card_output_positions_survive_starvation(rtty_owner, stage4, interrupted):
    owner = rtty_owner('everything')
    select(owner)
    heard(owner)
    captured = []
    class Tap:
        def push(self, block): pass
        def flush_to(self, index): pass
        def resync(self, index): pass
        def push_tx(self, first, pcm): captured.append((first, pcm.copy()))
    owner.loop.sa.subscribe(Tap())
    owner.session.type('A')
    owner.session.key()
    run(owner.loop, owner.session, 1, step=False)
    if not interrupted:
        owner.session.type('B')
        owner.session._pump()
        run(owner.loop, owner.session, .5, step=False)
    owner.session.abort()
    assert owner.session._record_tx.position == captured[-1][0] + len(captured[-1][1])
    close(owner)
    finish(stage4)
    from hfmodem.tests.station.recording_stage4_support import lane_audio
    rec = one_session(stage4, 'everything')
    assert len(lane_audio(rec, owner.session._record_tx)) == sum(len(pcm) for _, pcm in captured)


@pytest.mark.parametrize('text', (f'CQ CQ DE {PEER} K ', f'{OTHER} DE {PEER} K '))
def test_rtty_peer_traffic_without_local_address_does_not_qualify(rtty_owner, stage4, text):
    owner = rtty_owner()
    select(owner)
    owner.session._decoded([NS(accepted=True, text=text, shift_known=True)])
    transmit(owner)
    close(owner)
    finish(stage4)
    assert not any(row.get('keep') for row in one_session(stage4, 'contacts').index['contacts'].values())


def test_rtty_replacement_uses_emitted_progress_before_session_step(rtty_owner, stage4):
    owner = rtty_owner()
    select(owner)
    heard(owner)
    for ch in 'AB':
        owner.session.type(ch)
    owner.session.key()
    session = owner.session
    for _ in range(20):
        run(owner.loop, session, .1, step=False)
        session._pump()
        if (session._flight and session._flight[0][1] is not None
                and session._tx_stream.progress.card_end(session._flight[0][1]) is not None):
            break
    assert session._flight and session._flight[0][1] is not None
    assert session._tx_stream.progress.card_end(session._flight[0][1]) is not None
    select(owner, OTHER)
    close(owner)
    finish(stage4)
    rows = one_session(stage4, 'contacts').index['contacts'].values()
    assert any(row.get('peer') == PEER and row.get('keep') for row in rows)


@pytest.mark.parametrize('case', ('unset-local', 'fragmented-prefix'))
def test_rtty_ambiguous_calls_discard_at_teardown(rtty_owner, stage4, case):
    owner = rtty_owner()
    select(owner)
    transmit(owner)
    if case == 'unset-local':
        owner.session.mycall = ''
        text = f'CQ DE {PEER} K '
    else:
        text = f'{LOCAL} DE {PEER}A K '
    for char in text:
        owner.session._decoded([NS(accepted=True, text=char, shift_known=True)])
    close(owner)
    finish(stage4)
    assert not any(row.get('keep') for row in one_session(stage4, 'contacts').index['contacts'].values())
    assert audio_files(stage4.root) == []
