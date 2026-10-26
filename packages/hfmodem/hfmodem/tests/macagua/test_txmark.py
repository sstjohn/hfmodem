# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Sampler-only marks, accumulated evidence and normal voice/CW rendering."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.contact.txmark import Tag, LLMMark, LABELS
from hfmodem.contact.runtime import Session
from hfmodem.contact.terminal import Terminal
from hfmodem.contact.audio import Segment
from hfmodem.merlin.modem import CwIO, CwWaveform, render
from hfmodem.tests.macagua.test_voice import cfg

QUESTION = 'What do you enjoy about trying different antenna ideas and how do you find time for the radio hobby?'


class Language:
    def __init__(self):
        self.calls = []
        self.spellings = {}

    def request(self, path, data):
        assert path == '/tokenize'
        spelling = data['content']
        token = self.spellings.setdefault(spelling, len(self.spellings))
        return dict(tokens=[dict(id=token, piece=spelling)])

    def complete(self, instruction, content, schema, **kwargs):
        self.calls.append((instruction, content, schema, kwargs))
        return dict(reply=QUESTION)

    def close(self):
        pass


@pytest.fixture(scope='module')
def codec():
    return LLMMark(Language())


def test_payload_changes_only_actual_token_bias(codec):
    language = codec.language
    tags = [Tag(), Tag('hfmodem'), Tag('code', '0000'), Tag('code', '0123')]
    for tag in tags:
        codec.generate(tag, 'weather', seed=7)
    calls = language.calls[-4:]
    assert all(call[:3] == calls[0][:3] for call in calls)
    assert all(call[3]['sampling']['seed'] == 7 for call in calls)
    biases = [call[3]['sampling']['logit_bias'] for call in calls]
    assert biases[0] == {} and all(biases[1:])
    assert biases[1] != biases[2] != biases[3]
    assert all(set(b).issubset({str(t) for t in codec.tokens}) for b in biases)


@pytest.mark.parametrize('tag', [Tag('hfmodem'), Tag('code', '0000'), Tag('code', '0123'), Tag('code', '9999')])
def test_detector_recovers_profile_with_independent_receiver(codec, tag):
    # Detector unit test supplies ideal evidence; it does not pretend a fake
    # model validates biased generation. The local-model rehearsal covers that.
    codec.reset('unit')
    pattern = codec.patterns[LABELS.index(tag.label)]
    words = np.array(codec.words)[pattern]
    for offset in range(0, len(words), 16):
        evidence = codec.observe(' '.join(words[offset:offset+16]), stream='unit')
    assert evidence['mark'] == tag.record()


def test_all_profiles_are_distinct_and_unbiased_word_set_is_not_a_mark(codec):
    assert len({row.tobytes() for row in np.packbits(codec.patterns, axis=1)}) == 10001
    codec.reset('normal')
    assert not codec.observe(' '.join(codec.words), 'normal')['accepted']


def test_repeated_short_over_is_partial_not_added_confidence(codec):
    codec.reset('short')
    for _ in range(100):
        result = codec.observe(QUESTION, 'short')
    assert result['evidence'] < 40 and result['mark'] is None and result['overs'] == 64
    assert codec.observe(QUESTION, 'other')['overs'] == 1
    codec.reset('short')
    assert codec.observe(QUESTION, 'short')['overs'] == 1


def test_key_changes_sampler_and_wrong_key_does_not_recover(codec):
    keyed = LLMMark(Language(), key=b'x'*32)
    tag = Tag('code', '0123')
    assert keyed.bias(tag) != codec.bias(tag)
    words = np.array(keyed.words)[keyed.patterns[LABELS.index(tag.label)]]
    codec.reset('wrong-key')
    assert not codec.observe(' '.join(words), 'wrong-key')['accepted']
    with pytest.raises(ValueError):
        LLMMark(Language(), key=b'short')


@pytest.mark.parametrize('code', ['123', '12345', 'abcd', '12 3', '１２３４'])
def test_code_requires_exactly_four_ascii_digits(code):
    with pytest.raises(ValueError):
        Tag('code', code)


@pytest.mark.parametrize('wpm', [12., 20.])
@pytest.mark.parametrize('tone', [600., 1000.])
def test_cw_text_survives_original_keyer_and_receive_filter(tmp_path, wpm, tone):
    body = 'CQ CQ DE W9SSJ K'
    text = QUESTION.upper() + ' BT ' + body
    audio = render(text, wpm, tone)
    audio += np.random.default_rng(42).normal(0, .003, len(audio)).astype(np.float32)
    pcm = CwWaveform(wpm, tone).receive_chain().block(audio)
    path = tmp_path / 'received.wav'
    wavfile.write(path, 16000, pcm)
    copied = CwIO(wpm, tone).request('transcribe', path=str(path))
    assert copied['text'] == text
    assert LLMMark.split(copied['text'], 'cw')[1] == body


@pytest.mark.parametrize('waveform', ['voice', 'cw'])
@pytest.mark.parametrize('kind', ['cq', 'exchange', 'fill', 'free', 'close'])
def test_every_draft_kind_uses_llm_sampler_before_synthesis(tmp_path, codec, waveform, kind):
    session = Session(cfg(), tmp_path, test=True, waveform=waveform, tx_mark=Tag('code', '0123'))
    session.mark_codec = codec
    requested = []
    if waveform == 'cw':
        session.speech = CwIO(20, 700)
    else:
        def synth(operation, **values):
            requested.append(values['text'])
            wavfile.write(values['path'], 48000, (.1*np.sin(2*np.pi*700*np.arange(48000)/48000)).astype(np.float32))
            return dict(elapsed_s=.01)
        session.speech = SimpleNamespace(request=synth, close=lambda: None)
    try:
        original = session.codec.cq(session.contact.own_call) if kind == 'cq' else session.codec.identify('R', session.contact.own_call)
        session.make_draft(kind, original)
        session.pending.result(timeout=5)
        draft = session.poll_draft()
        assert draft['tx_mark'] == Tag('code', '0123').record()
        assert draft['source_text'] == original
        assert draft['mark_generation']['layer'] == 'llm-logit-bias'
        assert codec.language.calls[-1][3]['sampling']['logit_bias'] == codec.bias(Tag('code', '0123'))
        if waveform == 'voice':
            assert LLMMark.split(requested[0], 'voice')[1] == original
        else:
            copied = session.speech.request('transcribe', path=draft['wav'])
            assert LLMMark.split(copied['text'], 'cw')[1] == original
            assert np.allclose(draft['audio'], session.waveform.condition(render(draft['text']), 48000, session.cfg.voice.drive), atol=4e-5)
    finally:
        session.close()


def test_console_mark_controls_are_either_or_and_redraft_without_stacking(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=True)
    terminal = Terminal(session)
    drafted = []
    monkeypatch.setattr(session, 'make_draft', lambda kind, text: drafted.append((kind, text)))
    try:
        source = session.codec.cq(session.contact.own_call)
        session.draft = dict(kind='cq', text=source, source_text=source)
        assert terminal.key('m')
        assert session.tx_mark == Tag('hfmodem') and drafted[-1] == ('cq', source)
        terminal.key('d')
        terminal.editor['value'] = '0007'
        terminal.edited()
        assert session.tx_mark == Tag('code', '0007')
        terminal.key('m')
        assert session.tx_mark == Tag('hfmodem')
        terminal.key('z')
        assert session.tx_mark == Tag()
        session.playing = dict(active=True)
        terminal.key('m')
        assert session.tx_mark == Tag()
        session.playing = None
    finally:
        session.close()


def test_mark_invalidates_pending_draft_and_is_blocked_on_rig(tmp_path):
    session = Session(cfg(), tmp_path, test=True)
    try:
        old = session.revision
        session.set_tx_mark(Tag('code', '0123'))
        assert session.revision > old
        session.rig = object()
        with pytest.raises(ValueError, match='computer'):
            session.set_tx_mark(Tag())
        session.rig = None
    finally:
        session.close()


@pytest.mark.parametrize('known', [False, True])
def test_received_cw_cover_preserves_report_and_callsign(tmp_path, codec, known):
    session = Session(cfg(), tmp_path, test=True, waveform='cw')
    session.speech = CwIO(20, 700)
    session.mark_codec = codec
    try:
        codec.reset('partner:KD0DLK')
        if known:
            pattern = codec.patterns[LABELS.index('0123')]
            codec.observe(' '.join(np.array(codec.words)[pattern]), 'partner:KD0DLK')
        text = 'W9SSJ DE KD0DLK UR 579 K'
        audio = render(QUESTION.upper() + ' BT ' + text)
        session._decode(Segment(audio, False, 0, len(audio)/48000, .99))
        received = session.events.get_nowait()
        assert received['transmission_mark'] == (Tag('code', '0123').record() if known else None)
        if not known:
            assert received['mark_evidence']['evidence'] < 40
        assert received['over']['report'] == '579' and received['over']['de'] == 'KD0DLK'
        assert received['content_text'] == (text if known else QUESTION.upper() + ' BT ' + text)
    finally:
        session.close()
