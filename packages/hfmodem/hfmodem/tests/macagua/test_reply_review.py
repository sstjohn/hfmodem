# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A refused model draft remains editable text and cannot reach synthesis/TX."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hfmodem.contact.language import Language, ReplyError
from hfmodem.contact.runtime import Session, write_wav
from hfmodem.contact.simulation import TestOptions as SimulationOptions
from hfmodem.contact.terminal import Terminal
from hfmodem.macagua import script
from hfmodem.macagua.phonetics import callsigns
from hfmodem.tests.macagua.test_voice import cfg


@pytest.fixture
def session(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('a review-only reply opened playback')
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=forbidden))
    session = Session(cfg(), tmp_path, test=True)
    session.contact.confirm('KD0DLK')
    session.contact.state = 'FREE'
    synthesized = []
    def synthesize(action, **request):
        assert action == 'synthesize'
        synthesized.append(request['text'])
        write_wav(request['path'], np.sin(2 * np.pi * 1000 * np.arange(24000) / 24000), 24000)
        return {}
    session.speech = SimpleNamespace(request=synthesize, close=lambda: None)
    yield session, synthesized
    session.close()


def model(session, reply):
    session.language = SimpleNamespace(draft=lambda *args, **kwargs: reply,
        converse=lambda *args, **kwargs: dict(action='continue', reply=reply, claims={}), close=lambda: None)


@pytest.mark.parametrize('reply', [
    'This is W9SSJ. Thanks for the antenna report. Over.',
    'KD0DLK, thanks for the antenna report. This is W9SSJ. Over.',
    'Whiskey niner Sugar Sugar Juliet, thanks for the antenna report. Over.',
])
def test_known_identifiers_are_removed_and_codec_identifies_once(session, reply):
    live, synthesized = session
    model(live, reply)
    live.make_free('Here is my antenna report.')
    live.pending.result(timeout=3)
    draft = live.poll_draft()
    assert 'review_error' not in draft
    assert len(synthesized) == 1
    assert callsigns(draft['text']) == ['W9SSJ']
    assert draft['text'].lower().count('over') == 1
    assert 'Thanks for the antenna report.' in draft['text'] or 'thanks for the antenna report.' in draft['text']
    assert 'This is .' not in draft['text']


def test_unknown_call_retains_candidate_and_edit_recovers_without_redrafting(session):
    live, synthesized = session
    reply = 'K1FAKE, your antenna sounds good.'
    model(live, reply)
    live.make_free('How is my antenna?')
    live.pending.result(timeout=3)
    ui = Terminal(live)
    ui.draft_text = 'Drafting a brief reply…'
    ui.step()
    assert ui.draft_text == reply and 'K1FAKE' in ui.notice and 'e edits' in ui.notice
    assert live.draft is None and live.pending is None and not synthesized
    ui.key('\n')
    assert not synthesized
    ui.key('e')
    assert ui.editor['value'] == reply
    ui.key('\x01')
    ui.key('\x0b')
    for char in 'Thanks for the antenna report. Over.': ui.key(char)
    ui.key('\n')
    live.pending.result(timeout=3)
    ui.step()
    assert live.draft is not None and len(synthesized) == 1
    assert 'K1FAKE' not in live.draft['text']
    assert callsigns(live.draft['text']) == ['W9SSJ']
    assert 'draft_review' in (live.directory / 'events.jsonl').read_text()


def test_unsupported_station_claim_preserves_the_actual_model_reply(session):
    live, synthesized = session
    language = Language.__new__(Language)
    language.complete = lambda *args, **kwargs: dict(reply='My antenna is a beam.', claims={'antenna': 'beam'})
    language.close = lambda: None
    live.language = language
    live.make_free('Thanks for calling.')
    live.pending.result(timeout=3)
    review = live.poll_draft()
    assert review['text'] == 'My antenna is a beam.'
    assert 'unsupported station claim' in review['review_error']
    assert live.draft is None and not synthesized


def test_identification_only_reply_is_reviewable_with_original_text(session):
    live, synthesized = session
    model(live, 'This is W9SSJ. Over.')
    live.make_free('Tell me about your antenna.')
    live.pending.result(timeout=3)
    review = live.poll_draft()
    assert review['text'] == 'This is W9SSJ. Over.'
    assert 'no conversational reply' in review['review_error'] and not synthesized


def test_model_failure_clears_placeholder_and_keeps_recovery_controls(session):
    live, synthesized = session
    def fail(*args, **kwargs): raise RuntimeError('local model disconnected')
    live.language = SimpleNamespace(draft=fail, close=lambda: None)
    live.make_free('Hello.')
    live.pending.result(timeout=3)
    ui = Terminal(live)
    ui.draft_text = 'Drafting a brief reply…'
    ui.step()
    assert ui.draft_text == '' and 'disconnected' in ui.notice and 'r redrafts' in ui.notice
    assert live.draft is None and not synthesized


def test_automatic_simulation_pauses_on_an_unconfirmed_model_call(session):
    live, synthesized = session
    live.simulation = SimulationOptions()
    model(live, 'K1FAKE, thanks for that.')
    ui = Terminal(live)
    ui.auto.ready_at = 0
    live.make_free('Hello.', conversation=True)
    live.pending.result(timeout=3)
    ui.step()
    assert ui.auto.paused and ui.auto.ready_at is None
    assert live.draft is None and not synthesized


def test_reply_error_carries_candidate_without_changing_error_type():
    error = ReplyError('needs review', 'Candidate reply')
    assert isinstance(error, ValueError) and error.reply == 'Candidate reply'


@pytest.mark.parametrize('grid', ['EN63bb', 'Echo November six three Bravo Bravo'])
def test_known_call_cleanup_does_not_change_trusted_model_station_facts(session, grid):
    live, synthesized = session
    model(live, 'I am using FTDX10. My grid is ' + grid + '.')
    live.make_free('What radio and grid?')
    live.pending.result(timeout=3)
    draft = live.poll_draft()
    assert 'FTDX10' in draft['text'] and script.spell_grid('EN63bb') in draft['text']
    assert len(synthesized) == 1 and callsigns(draft['text']).count('W9SSJ') == 1


def test_redraft_answers_partner_message_after_self_copy(session, monkeypatch):
    from hfmodem.contact.frame import Over
    live, synthesized = session
    ui = Terminal(live)
    heard = []
    monkeypatch.setattr(live, 'make_free', lambda text, **kwargs: heard.append(text))
    ui.heard_event(dict(kind='heard', recognition=dict(text='How are conditions?'),
        over=Over(free='How are conditions?').record()))
    ui.heard_event(dict(kind='self', recognition=dict(text='This is W9SSJ. Over.'), loopback_calls_ok=True))
    ui.key('r')
    assert heard[-1] == 'How are conditions?'
    assert ui.partner_text == 'How are conditions?'


def test_opening_conversation_does_not_feed_own_cq_to_model(session, monkeypatch):
    live, synthesized = session
    ui = Terminal(live)
    heard = []
    monkeypatch.setattr(live, 'make_free', lambda text, **kwargs: heard.append(text))
    ui.heard_event(dict(kind='self', recognition=dict(text=script.cq('W9SSJ')), loopback_calls_ok=True))
    ui.key('t')
    assert heard == ['The operator opens conversation.']
