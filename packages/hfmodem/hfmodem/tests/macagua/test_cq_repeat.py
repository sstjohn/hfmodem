# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CQ repetition reuses approved audio; pronunciation varies only at rendering."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.contact.runtime import Session, write_wav
from hfmodem.core import config
from hfmodem.macagua import script
from hfmodem.macagua.phonetics import callsigns


@pytest.fixture
def session(tmp_path):
    cfg = config.parse(dict(schema=1, station=dict(mycall='W9SSJ', grid='EN63bb',
        control='local', regulatory='unregulated', because='offline CQ caching test'),
        voice=dict(phonetics='varied')), environ={})
    session = Session(cfg, tmp_path, test=True)
    synthesized = []
    def synthesize(action, **request):
        assert action == 'synthesize'
        synthesized.append(request['text'])
        t = np.arange(24000) / 24000
        write_wav(request['path'], np.sin(2 * np.pi * 1000 * t) * .2, 24000)
        return dict(elapsed_s=.01)
    session.speech = SimpleNamespace(request=synthesize, close=lambda: None)
    yield session, synthesized
    session.close()


def render(session, text, kind='cq'):
    session.make_draft(kind, text)
    session.pending.result(timeout=5)
    return session.poll_draft()


def test_unchanged_cq_reuses_waveform_without_another_synthesis(session):
    live, synthesized = session
    text = script.cq('W9SSJ')
    first = render(live, text)
    live.cancel_draft()
    second = render(live, text)
    assert len(synthesized) == 1
    assert second['cache_hit'] and second['cached_from_revision'] == first['revision']
    assert second['synthesis_s'] == 0
    assert second['revision'] != first['revision']
    assert second['audio'] is first['audio']
    assert second['text'] == first['text'] and second['wav'] == first['wav']


def test_changed_cq_or_audio_settings_require_fresh_synthesis(session):
    live, synthesized = session
    text = script.cq('W9SSJ')
    render(live, text)
    render(live, 'Calling CQ again. This is W9SSJ. Over.')
    assert len(synthesized) == 2
    live.cfg = replace(live.cfg, voice=replace(live.cfg.voice, compression='medium'))
    render(live, 'Calling CQ again. This is W9SSJ. Over.')
    assert len(synthesized) == 3


def test_edited_reply_does_not_duplicate_own_call_at_rendering(session):
    live, synthesized = session
    live.cfg = replace(live.cfg, voice=replace(live.cfg.voice, phonetics='nato'))
    text = 'The second antenna sounds good. W9SSJ. Over.'
    result = render(live, text, kind='free')
    assert result['text'] == synthesized[-1] == text


def test_alternate_pronunciation_keeps_call_copy_and_conversation_history():
    text = script.identify('Your antenna sounds good.', 'W9SSJ')
    result = script.vary_calls(text, ('W9SSJ',), 'alternate')
    assert 'Whiskey niner Sugar Sugar Juliett' in result
    assert callsigns(result) == ['W9SSJ']
    assert script.dialogue_text(result, 'W9SSJ') == 'Your antenna sounds good.'
    assert script.identify(result, 'W9SSJ') == result


def test_varied_pronunciation_changes_rendered_calls_without_changing_payload():
    text = 'KE4QJW, your antenna sounds good. W9SSJ. Over.'
    results = {script.vary_calls(text, ('W9SSJ', 'KE4QJW'), 'varied', seed=i) for i in range(20)}
    assert len(results) > 1
    for result in results:
        assert callsigns(result) == ['KE4QJW', 'W9SSJ']
        assert 'your antenna sounds good.' in result and result.endswith('Over.')
    assert script.vary_calls(text, ('W9SSJ', 'KE4QJW'), 'nato') == text
