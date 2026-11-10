# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Voice and CW owners retain raw input and submitted output on shared lanes."""
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.contact import runtime
from hfmodem.contact.audio import Segment
from hfmodem.merlin.modem import render
from hfmodem.tests.station.recording_stage4_support import (
    Output, audio_files, one_session, finish,
    flush_all, bounded, lane_audio, LOCAL, PEER,
)


def select(owner, call=PEER):
    owner.terminal.edit('call', call, 'Confirmed partner')
    owner.terminal.edited()
    assert owner.session.contact.partner == call


def raw(owner, seconds=1):
    session = owner.session
    request = session.speech.request
    def request_once(op, **values):
        result = request(op, **values)
        if op == 'vad' and session.blocks.empty():
            session.stop.set()
        return result
    session.speech.request = request_once
    try:
        for _ in range(seconds):
            session.stop.clear()
            owner.state.clock.now += 1
            session._capture(np.full((48000, 1), .125, np.float32), 48000, NS(), False)
            bounded(session._receive)
            assert session.receive_failure is None
            flush_all(owner.state)
    finally:
        session.speech.request = request
        session.stop.clear()


def heard(owner, call=PEER, *, self_copy=False):
    session = owner.session
    if session.waveform.name == 'cw':
        data = render(f'{LOCAL} DE {call} K')
    else:
        session.speech.text = f'{LOCAL} DE {call} K'
        data = np.full(4800, .125, np.float32)
    session._decode(Segment(data, self_copy, owner.state.clock.now,
                            owner.state.clock.now + len(data) / 48000, .98))
    event = session.events.get(timeout=5)
    assert event['kind'] == ('self' if self_copy else 'heard')
    owner.terminal.heard_event(event)
    return event


def draft(owner, kind='exchange'):
    owner.session.make_draft(kind, f'{PEER} DE {LOCAL} K')
    owner.session.pending.result(timeout=5)
    value = owner.session.poll_draft()
    assert value and value['audio'].size
    return value


def send(owner, *, preview=False, kind='exchange'):
    draft(owner, kind)
    owner.session.play(preview=preview)
    assert owner.session.output_done.wait(5)
    owner.session.finish_playback()
    flush_all(owner.state)


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_raw_receive_lane_keeps_idle_pcm_without_audio_working_files(voice_owner, stage4, policy, waveform):
    owner = voice_owner(policy, waveform)
    raw(owner, 3)
    bounded(owner.session.close)
    finish(stage4)
    assert bool(audio_files(stage4.root)) == (policy == 'everything')
    rec = one_session(stage4, policy)
    rx = [lane for lane in rec.lanes.values() if 'rx' in lane.name]
    assert len(rx) == 1 and rx[0].position - rx[0].origin == 3 * 48000
    assert not rec.index['contacts']
    if policy == 'everything':
        np.testing.assert_array_equal(lane_audio(rec, rx[0]), np.full(3 * 48000, 4096, '<i2'))


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('policy', ('off', 'contacts', 'everything'))
def test_decode_draft_preview_replay_and_transmit_use_memory(voice_owner, stage4, policy, waveform):
    owner = voice_owner(policy, waveform)
    select(owner)
    event = heard(owner)
    assert PEER in event['recognition']['text']
    assert (event['started'], event['ended']) == (1000., pytest.approx(1000. + (len(render(f'{LOCAL} DE {PEER} K')) / 48000 if waveform == 'cw' else .1)))
    if policy == 'off':
        assert audio_files(stage4.root) == [], 'ASR and over processing must not persist audio under off'
    draft(owner)
    if policy == 'off':
        assert audio_files(stage4.root) == [], 'synthesis and drafts must remain in memory under off'
    owner.session.play(preview=True)
    assert owner.session.output_done.wait(5)
    owner.session.finish_playback()
    owner.session.replay_heard()
    assert owner.session.output_done.wait(5)
    owner.session.finish_playback()
    send(owner)
    bounded(owner.session.close)
    finish(stage4)
    rec = one_session(stage4, policy)
    tx = next(lane for lane in rec.lanes.values() if 'tx' in lane.name)
    assert tx.position > 0
    if policy == 'off':
        assert audio_files(stage4.root) == []
    elif policy == 'everything':
        assert len(lane_audio(rec, tx)) == tx.position - tx.origin
    events = [json.loads(line) for line in (owner.directory / 'events.jsonl').read_text().splitlines()]
    assert any(row['kind'] == 'preview' for row in events)
    assert any(row['kind'] == 'sent' for row in events)


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_off_backend_never_opens_audio_files_for_decode_or_synthesis(voice_owner, monkeypatch, waveform):
    owner = voice_owner('off', waveform)
    def no_audio(*a, **k):
        raise AssertionError('off processing attempted file audio transport')
    monkeypatch.setattr(runtime, 'write_wav', no_audio)
    monkeypatch.setattr(runtime, 'read_wav', no_audio)
    from hfmodem.merlin import modem
    monkeypatch.setattr(modem.wavfile, 'write', no_audio)
    monkeypatch.setattr(modem.wavfile, 'read', no_audio)
    heard(owner)
    send(owner, preview=True)
    owner.session.replay_heard()
    owner.session.finish_playback()


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_everything_records_only_submitted_tx_not_preview_or_unsent_draft(voice_owner, stage4, waveform):
    owner = voice_owner('everything', waveform)
    raw(owner)
    draft(owner)
    owner.session.play(preview=True)
    owner.session.finish_playback()
    flush_all(stage4)
    rec = one_session(stage4, 'everything')
    tx = [lane for lane in rec.lanes.values() if 'tx' in lane.name]
    assert not tx or tx[0].position == 0, 'preview and synthesis are not submitted TX'
    draft(owner)
    owner.session.play()
    assert owner.session.output_done.wait(5)
    submitted = owner.session.output.blocks
    owner.session.finish_playback()
    bounded(owner.session.close)
    finish(stage4)
    tx = next(lane for lane in rec.lanes.values() if 'tx' in lane.name)
    expected = np.concatenate(submitted)
    retained = lane_audio(rec, tx)
    np.testing.assert_array_equal(retained, np.clip(np.round(expected * 32768), -32768, 32767).astype('<i2'))


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_partial_output_keeps_only_submitted_prefix_and_excludes_unsent_tail(voice_owner, stage4, monkeypatch, waveform):
    owner = voice_owner('everything', waveform)
    class Partial(Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, limit=1024)
    monkeypatch.setattr(sys.modules['sounddevice'], 'OutputStream', Partial)
    value = draft(owner)
    assert len(value['audio']) > 1024
    owner.session.play()
    owner.session.finish_playback(cancelled=True)
    bounded(owner.session.close)
    finish(stage4)
    rec = one_session(stage4, 'everything')
    tx = next(lane for lane in rec.lanes.values() if 'tx' in lane.name)
    assert tx.position - tx.origin == 1024
    retained = lane_audio(rec, tx)
    np.testing.assert_array_equal(retained, np.round(value['audio'][:1024] * 32768).astype('<i2'))


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
def test_key_refusal_creates_no_submitted_audio(voice_owner, stage4, waveform):
    owner = voice_owner('contacts', waveform)
    select(owner)
    raw(owner)
    draft(owner)
    def refuse(*a, **k): raise ValueError('synthetic key refusal')
    owner.session.rig.key = refuse
    with pytest.raises(ValueError, match='synthetic key refusal'):
        owner.session.play()
    bounded(owner.session.close)
    finish(stage4)
    assert audio_files(stage4.root) == [], 'unsent render must not survive refused transmission'
    rec = one_session(stage4, 'contacts')
    assert not any(lane.position for lane in rec.lanes.values() if 'tx' in lane.name)


@pytest.mark.parametrize('waveform', ('voice', 'cw'))
@pytest.mark.parametrize('operation', ('draft', 'preview', 'replay', 'transmit'))
def test_off_each_processing_path_operates_without_audio_files(voice_owner, stage4, monkeypatch, waveform, operation):
    owner = voice_owner('off', waveform)
    def no_audio(*a, **k):
        raise AssertionError(f'off {operation} attempted file audio transport')
    monkeypatch.setattr(runtime, 'write_wav', no_audio)
    monkeypatch.setattr(runtime, 'read_wav', no_audio)
    from hfmodem.merlin import modem
    monkeypatch.setattr(modem.wavfile, 'read', no_audio)
    monkeypatch.setattr(modem.wavfile, 'write', no_audio)
    if operation == 'replay':
        owner.session.latest = dict(audio=[.125] * 4800, rate=48000)
        try:
            owner.session.replay_heard()
        except KeyError as exc:
            pytest.fail(f'replay_heard still requires an audio pathname: {exc}')
        owner.session.finish_playback()
    else:
        draft(owner)
        if operation != 'draft':
            owner.session.play(preview=operation == 'preview')
            owner.session.finish_playback()
    assert audio_files(stage4.root) == []


@pytest.mark.parametrize('timestamped', (False, True))
def test_voice_adc_loss_preserves_absolute_contact_bounds(voice_owner, stage4, timestamped):
    owner = voice_owner('everything')
    session = owner.session
    rate = runtime.RATE
    n = rate // 4
    first = NS(inputBufferAdcTime=100., currentTime=100.) if timestamped else NS()
    second = NS(inputBufferAdcTime=101., currentTime=101.) if timestamped else NS()
    session._capture(np.full((n, 1), .125, np.float32), n, first, False)
    flush_all(stage4)
    select(owner)
    session._capture(np.full((n, 1), .25, np.float32), n, second, NS(input_overflow=True))
    flush_all(stage4)
    session._end_candidate('reset', operator=True)
    session.close()
    finish(stage4)
    lane = session._record_rx
    expected = rate + n if timestamped else 2*n + 1
    assert lane.position == expected
    assert lane.gaps == [[n, expected-n]]
    if not timestamped:
        assert lane.row.get('uncertain')
    row = next(iter(session.recording.index['contacts'].values()))
    assert row['end_positions']['input'] == expected
