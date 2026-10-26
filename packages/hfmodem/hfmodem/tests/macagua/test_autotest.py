# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Automatic computer QSOs and their isolation from the radio approval path."""
import numpy as np
import pytest

from hfmodem.contact.exchange import Contact
from hfmodem.contact.frame import Over
from hfmodem.contact.simulation import TestOptions as SimulationOptions, decision, says_goodbye
from hfmodem.macagua.runtime import Session
from hfmodem.macagua.terminal import Terminal
from hfmodem.tests.macagua.test_voice import cfg


@pytest.mark.parametrize('text,expected', [
    ('Thanks for the contact, 73!', True), ('Seventy three, good night.', True),
    ('Seven three from KD0DLK.', True), ('Goodbye.', True), ('Signing off now.', True),
    ('What does 73 mean?', False), ('I am not saying 73 yet.', False),
    ('My antenna is 73 feet long.', False), ('You are five seven.', False)])
def test_spoken_sign_off(text, expected):
    assert says_goodbye(text) is expected


def test_decisions_keep_talking_then_close():
    contact = Contact('W9SSJ')
    options = SimulationOptions(max_turns=4)
    over = Over(de='KD0DLK')
    assert decision(contact, over, 'This is KD0DLK', 0, options) == 'accept'
    contact.confirm('KD0DLK')
    assert decision(contact, Over(), 'What rig are you using?', 1, options) == 'talk'
    assert decision(contact, Over(), 'Thanks, 73.', 1, options) == 'close'
    assert decision(contact, Over(), 'Another topic', 4, options) == 'close'
    assert decision(contact, Over(de='K4MCK'), 'A different caller', 1, options) == 'repeat_message'
    assert decision(contact, Over(confidence={'de': .2}), 'noise', 1, options) == 'repeat_message'


def simulator(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=True, simulation=SimulationOptions())
    terminal = Terminal(session)
    calls = []
    monkeypatch.setattr(session, 'make_draft', lambda kind, text: calls.append((kind, text)))
    monkeypatch.setattr(session, 'make_free', lambda text, **options: calls.append(('talk', text, options)))
    return session, terminal, calls


def heard(over, text):
    return dict(kind='heard', over=over.record(), recognition={'text': text}, wav='test.wav')


def test_computer_auto_copies_call_without_assuming_report_and_uses_history(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    try:
        terminal.heard_event(heard(Over(de='KD0DLK'), 'This is Kilo Delta zero Delta Lima Kilo.'))
        assert session.contact.partner == 'KD0DLK'
        assert calls[-1][0] == 'exchange'
        assert 'You are' not in calls[-1][1] and session.contact.report_sent == ''
        terminal.auto.replies = 1
        terminal.heard_event(heard(Over(free='What rig are you using?'), 'What rig are you using?'))
        assert calls[-1][0] == 'talk' and calls[-1][2]['conversation'] is True
        assert calls[-1][2]['may_close'] is False
        assert len(session.history) == 2
        terminal.auto.replies = 3
        terminal.heard_event(heard(Over(), 'That is an interesting station.'))
        assert calls[-1][2]['may_close'] is True
        terminal.heard_event(heard(Over(), 'Thanks for the contact. Seventy three.'))
        assert calls[-1][0] == 'close'
    finally:
        session.close()


def test_uncertain_calls_get_spoken_fill_not_keyboard_prompt(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    try:
        terminal.heard_event(heard(Over(), 'I am calling you.'))
        assert calls[-1][0] == 'fill' and not terminal.editor
        assert not session.contact.partner
    finally:
        session.close()


def test_auto_speaker_playback_pauses_for_rx_and_escape_cancels_future_playback(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    played = []
    monkeypatch.setattr(session, 'play', lambda **kwargs: played.append('speaker'))
    try:
        session.draft = dict(kind='cq')
        terminal.auto.ready(now=0)
        session.receiving = True
        terminal.auto.tick(now=1)
        assert played == []
        session.receiving = False
        terminal.auto.tick(now=1)
        assert played == ['speaker']
        session.draft = dict(kind='free')
        terminal.auto.ready(now=0)
        terminal.key('\x1b')
        terminal.auto.tick(now=2)
        assert played == ['speaker'] and terminal.auto.paused and session.draft is None
    finally:
        session.close()


def test_receive_failure_prevents_automatic_playback(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    monkeypatch.setattr(session, 'play', lambda **kwargs: pytest.fail('played after receive failure'))
    try:
        session.draft = dict(kind='cq')
        terminal.auto.ready(now=0)
        session.events.put(dict(kind='receive_error', text='audio failed'))
        terminal.step()
        assert not terminal.receive_ok and terminal.auto.paused
    finally:
        session.close()


def test_close_logs_only_after_completed_playback_then_schedules_another_cq(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    try:
        session.contact.confirm('KD0DLK')
        session.contact.heard = Over(de='KD0DLK', report='57')
        draft = dict(kind='close', text='closing', wav='test.wav')
        session.playing = dict(draft=draft, preview=False, started='test', token=None, cancelled=False)
        played = session.finish_playback()
        terminal.auto.finished(played, now=0)
        assert len(session.contacts) == 1 and session.contacts[0]['test']
        terminal.auto.tick(now=6)
        assert calls[-1][0] == 'cq' and not session.contact.partner
        assert not session.history
    finally:
        session.close()


def test_quiet_partner_timeout_closes_and_unanswered_cq_repeats(tmp_path, monkeypatch):
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    try:
        terminal.auto.listen_at = 0
        terminal.auto.tick(now=31)
        assert calls[-1][0] == 'cq'
        session.contact.confirm('KD0DLK')
        terminal.auto.listen_at = 0
        terminal.auto.tick(now=31)
        assert calls[-1][0] == 'close'
    finally:
        session.close()


def test_auto_policy_rejects_a_rig_even_if_added_after_start(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='computer-audio'):
        Session(cfg(), tmp_path, test=False, simulation=SimulationOptions())
    session, terminal, calls = simulator(tmp_path, monkeypatch)
    try:
        session.rig = object()
        with pytest.raises(RuntimeError, match='rig'):
            terminal.auto.tick()
    finally:
        session.rig = None
        session.close()


def test_radio_still_waits_for_enter(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=False)
    terminal = Terminal(session)
    monkeypatch.setattr(session, 'play', lambda **kwargs: pytest.fail('radio played without Enter'))
    try:
        session.draft = dict(kind='cq', text='CQ', duration_s=1)
        terminal.step()
        assert terminal.auto is None and session.draft is not None
    finally:
        session.close()


def test_tiny_model_close_is_rendered_as_template_and_recorded(tmp_path):
    session = Session(cfg(), tmp_path, test=True, simulation=SimulationOptions())
    session.contact.confirm('KD0DLK')
    session.history = [dict(role='partner', text='Thanks for the exchange')]
    class Language:
        def converse(self, text, facts, history, may_close):
            assert history == session.history and may_close
            return dict(action='close', reply='Thanks for the chat.', claims={})
        def close(self): pass
    class Speech:
        def request(self, operation, text, path):
            assert 'Seventy three' in text and 'Kilo Delta zero Delta Lima Kilo' in text
            from hfmodem.macagua.runtime import write_wav
            write_wav(path, np.sin(2 * np.pi * 1000 * np.arange(24000) / 24000), 24000)
            return {}
        def close(self): pass
    session.language, session.speech = Language(), Speech()
    try:
        session.make_free('Thanks for the exchange', conversation=True, may_close=True)
        session.pending.result(timeout=3)
        draft = session.poll_draft()
        assert draft['kind'] == 'close' and draft['conversation_decision']['action'] == 'close'
    finally:
        session.close()


def test_language_history_has_roles_and_does_not_repeat_current_question():
    import threading
    from hfmodem.contact.language import Language
    host = Language.__new__(Language)
    host.lock = threading.Lock()
    requests = []
    def request(path, data):
        requests.append(data)
        return {'choices': [{'message': {'content': '{"reply":"I use an FTDX10.","claims":{"rig":"FTDX10"},"action":"continue"}'}}]}
    host.request = request
    result = host.converse('What radio do you use?', {'rig': 'FTDX10'},
        [dict(role='partner', text='Hello'), dict(role='station', text='Thanks for calling.'),
         dict(role='partner', text='What radio do you use?')])
    assert result['action'] == 'continue'
    messages = requests[0]['messages']
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[-1]['content'] == 'What radio do you use?'
    assert requests[0]['response_format']['schema']['properties']['action']['enum'] == ['continue']


def test_partner_location_cannot_be_spoken_as_our_station_fact():
    from hfmodem.contact.language import Language
    host = Language.__new__(Language)
    host.complete = lambda *args, **kwargs: dict(reply='Boston, MA', claims={'grid': 'EN63bb'}, action='continue')
    result = host.converse('And where are you located?', {'grid': 'EN63bb', 'qth': '', 'rig': 'FTDX10'}, [])
    assert result['reply'] == 'My grid is EN63bb.'
    assert result['claims'] == {'grid': 'EN63bb'}
    result = host.converse('What is your name?', {'name': '', 'rig': 'FTDX10'}, [])
    assert result['reply'] == 'My name is not configured.'


@pytest.mark.parametrize('reply', [
    "What is up?", 'Please say again your call.', 'I will wait for your call.',
    'Thanks for calling. How are conditions at your end?'])
def test_bad_model_replies_do_not_restart_the_identity_or_repeat_questions(reply):
    from hfmodem.contact.language import Language
    host = Language.__new__(Language)
    host.complete = lambda *args, **kwargs: dict(reply=reply, claims={}, action='continue')
    result = host.converse('What is up?', {}, [dict(role='station', text='How are conditions at your end?')])
    assert result['repair_reason'] and 'your call' not in result['reply']
    assert result['reply'] != 'What is up?' and 'How are conditions' not in result['reply']
