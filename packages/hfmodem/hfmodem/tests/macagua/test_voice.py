# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Contact trust, audio gating and cancellation, without model weights or hardware."""
from concurrent.futures import Future
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hfmodem.contact.exchange import Contact
from hfmodem.contact.frame import Over, reconcile
from hfmodem.core import config
from hfmodem.core.regulatory import Emission
from hfmodem.core.rig import KeyToken
from hfmodem.macagua import script
from hfmodem.macagua.chain import BLOCK, Segmenter, transmit
from hfmodem.macagua.phonetics import callsigns
from hfmodem.contact.runtime import Session
from hfmodem.contact.terminal import Terminal


def cfg():
    rig = config.Rig(max_key_s=30)
    return SimpleNamespace(station=config.Station(mycall='W9SSJ', grid='EN63bb'),
        rig=rig, selected_rig='ftdx10', voice=config.Voice(),
        models=config.Models(), declared_power_w=rig.power_w)


def test_callsign_cannot_come_from_language_model_alone():
    over = reconcile(dict(de='K1FAKE'), ['K4MCK'], 'W9SSJ')
    assert over.de == 'K4MCK' and over.review
    assert reconcile(dict(de='K1FAKE'), [], 'W9SSJ').de == ''
    assert reconcile(dict(de='W9SSJ'), ['W9SSJ'], 'W9SSJ').de == ''


def test_explicit_fields_disagreeing_with_model_are_flagged():
    over = reconcile(dict(de='K4MCK', report='59', name='Bob', qth='Kilo', free=''),
        ['K4MCK'], 'W9SSJ', .95, 'You are 5-7. My name is Alex. My location is Boston. Over.')
    assert (over.report, over.name, over.qth) == ('57', 'Alex', 'Boston')
    assert len(over.review) == 3


def test_small_model_cannot_fill_absent_exchange_from_prompt_example():
    over = reconcile(dict(de='KD0DLK', report='57', name='Alex', qth='Boston'),
        ['KD0DLK'], 'W9SSJ', .95, 'This is Kilo Delta zero Delta Lima Kilo.')
    assert over.de == 'KD0DLK'
    assert not over.report and not over.name and not over.qth
    assert len(over.review) == 3


def test_empty_model_conversation_cannot_close_a_clearly_heard_question():
    over = reconcile(dict(free='', end='clear'), [], 'W9SSJ', .98, 'What radio are you using? Over.')
    assert over.free == 'What radio are you using?' and over.end == 'over'
    contact = Contact('W9SSJ')
    contact.confirm('KD0DLK')
    contact.sent('exchange')
    assert contact.receive(over) == 'FREE'


def test_spoken_signoff_closes_despite_model_free_text_or_wrong_end():
    over = reconcile(dict(free='Thanks for the contact', end='over'), [], 'W9SSJ', .98,
                     'Thanks for the contact. Seven three.')
    contact = Contact('W9SSJ')
    contact.confirm('KD0DLK')
    assert contact.receive(over) == 'CLOSE'


def test_supervised_standalone_over_preserves_a_ready_reply(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=True)
    session.contact.confirm('KD0DLK')
    session.contact.exchanged = True
    draft = session.draft = dict(kind='free', text='A useful reply')
    terminal = Terminal(session)
    terminal.heard_event(dict(kind='heard', recognition=dict(text='Over.'),
                             over=Over(end='over').record(), wav='test.wav'))
    assert session.draft is draft and session.pending is None
    session.close()


def test_supervised_new_cq_starts_fresh_history_and_operator_edits(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=True)
    session.contact.confirm('KD0DLK')
    session.history = [dict(role='station', text='Previous contact')]
    session.edits = [dict(kind='call', value='KD0DLK')]
    monkeypatch.setattr(session, 'make_draft', lambda *args: None)
    Terminal(session).key('c')
    assert not session.contact.partner and not session.history and not session.edits
    session.close()


def test_terminal_stops_for_first_call_confirmation(tmp_path):
    session = Session(cfg(), tmp_path, test=True)
    term = Terminal(session)
    heard = Over(de='K4MCK', report='57')
    term.heard_event(dict(kind='heard', recognition=dict(text='Kilo four Mike Charlie Kilo.'),
                         over=heard.record(), wav='test.wav'))
    assert session.contact.state == 'CONFIRM' and session.contact.partner == ''
    assert session.draft is None and session.pending is None
    assert 'v' in term.notice
    session.close()


def test_a_different_caller_requires_confirmation_even_with_an_existing_partner(tmp_path):
    session = Session(cfg(), tmp_path, test=True)
    session.contact.confirm('K4MCK')
    session.contact.report_sent = '57'
    terminal = Terminal(session)
    over = Over(de='KD0DLK')
    terminal.heard_event(dict(kind='heard', recognition=dict(text='This is KD0DLK'),
                             over=over.record(), wav='test.wav'))
    assert session.contact.state == 'CONFIRM' and session.contact.partner == 'K4MCK'
    assert session.pending is None and session.draft is None
    session.close()


def test_normalizer_repeated_calls_and_ordinary_words():
    assert callsigns('Whiskey 5 Echo, Whiskey 5 Echo. Special event.') == ['W5E', 'W5E']
    assert callsigns('kilo four mike charlie kilo') == ['K4MCK']
    assert callsigns('Roger, you are five nine in Missouri, thanks.') == []


def test_templates_share_station_facts_and_always_identify():
    facts = script.facts_from(cfg())
    assert facts['grid'] == 'EN63bb' and facts['rig'] == 'FTDX10'
    with pytest.raises(ValueError, match='report'):
        script.exchange('W9SSJ', 'K1ABC', '', facts)
    over = script.exchange('W9SSJ', 'K1ABC', '57', facts)
    assert 'five seven' in over and 'Echo November six three Bravo Bravo' in over
    assert 'name' not in over
    assert script.phonetic('W9SSJ') in script.identify('Edited text.', 'W9SSJ')
    assert script.identify(over, 'W9SSJ') == over


def test_contact_advances_only_after_completed_send_and_keeps_received_report():
    contact = Contact('W9SSJ')
    contact.sent('cq')
    assert contact.receive(Over(de='K1ABC', report='59')) == 'CONFIRM'
    assert not contact.partner
    contact.confirm('K1ABC')
    contact.report_sent = '57'
    contact.sent('exchange')
    assert contact.receive(Over(de='K1ABC', end='clear')) == 'CLOSE'
    assert contact.heard.report == '59'
    contact.sent('close')
    assert contact.state == 'LOGGED'
    contact.reset()
    assert not contact.partner and not contact.report_sent and not contact.exchanged
    contact.confirm('K1ABC')
    assert contact.receive(Over(de='K1ABC', free='What is your antenna?')) == 'FREE'


def test_silence_never_forms_an_over_and_preroll_marks_self_overlap():
    segmenter = Segmenter(silence_s=.064, preroll_s=.064)
    block = np.zeros(BLOCK, np.float32)
    for n in range(100):
        assert segmenter.feed(block, .1, False, n * .032) is None
    assert segmenter.feed(block, .1, True, 3.2) is None
    assert segmenter.feed(block, .9, False, 3.232) is None
    assert segmenter.feed(block, .1, False, 3.264) is None
    segment = segmenter.feed(block, .1, False, 3.296)
    assert segment.self_copy and len(segment.audio) == 5 * BLOCK


def test_transmit_chain_scales_peak_and_rejects_invalid_audio():
    speech = np.sin(2 * np.pi * 1000 * np.arange(24000) / 24000)
    pcm = transmit(speech, 24000, .35)
    assert len(pcm) == 48000 and pcm.dtype == np.float32
    assert np.max(np.abs(pcm)) == pytest.approx(.35)
    with pytest.raises(ValueError):
        transmit(np.full(100, np.nan), 24000, .35)
    with pytest.raises(ValueError, match='silence'):
        transmit(np.zeros(100), 24000, .35)


@pytest.mark.parametrize('mode', ['off', 'soft', 'medium', 'strong'])
def test_compression_preserves_duration_peak_ceiling_and_fades(mode):
    t = np.arange(48000) / 24000
    speech = (.1 + .9 * (t > 1)) * np.sin(2 * np.pi * 1000 * t)
    pcm = transmit(speech, 24000, .35, compression=mode)
    assert len(pcm) == 96000 and np.isfinite(pcm).all()
    assert np.max(np.abs(pcm)) == pytest.approx(.35)
    assert pcm[0] == pcm[-1] == 0
    assert not transmit(speech, 24000, 0, compression=mode).any()


def test_dynamic_compression_raises_quiet_speech_at_same_peak_and_filters_products():
    t = np.arange(96000) / 24000
    speech = np.where(t < 2, .1, 1.) * np.sin(2 * np.pi * 1000 * t)
    plain = transmit(speech, 24000, .35, compression='off')
    compressed = transmit(speech, 24000, .35, compression='medium')
    quiet = slice(24000, 72000)
    assert np.sqrt(np.mean(compressed[quiet] ** 2)) > 2 * np.sqrt(np.mean(plain[quiet] ** 2))
    spectrum = np.abs(np.fft.rfft(compressed)) ** 2
    frequencies = np.fft.rfftfreq(len(compressed), 1 / 48000)
    assert spectrum[frequencies > 5000].sum() / spectrum.sum() < 1e-5
    with pytest.raises(ValueError, match='compression'):
        transmit(speech, 24000, .35, compression='unknown')


def test_escape_invalidates_a_late_synthesis_result(tmp_path):
    session = Session(cfg(), tmp_path, test=True)
    future = Future()
    session.pending, session.revision = future, 1
    terminal = Terminal(session)
    assert terminal.key('\x1b')
    future.set_result(dict(revision=1)) if not future.cancelled() else None
    assert session.poll_draft() is None and session.draft is None
    session.close()


@pytest.mark.parametrize('test,preview,dial', [(True, False, 14285000), (False, True, 14285000),
                                             (False, False, 7208000)])
def test_output_prepared_before_key_and_ptt_drops_before_audio_teardown(tmp_path, monkeypatch, test, preview, dial):
    operations = []
    class Output:
        def __init__(self, **kwargs):
            operations.append('prepare')
        def start(self): operations.append('audio')
        def abort(self): operations.append('abort')
        def close(self): operations.append('close')
    class Rig:
        def key(self, emission, **kwargs):
            assert isinstance(emission, Emission)
            assert emission.span[1] < dial  # LSB below the dial
            operations.append('key')
            return KeyToken(1, 0, 3, 'test')
        def unkey(self, token): operations.append('unkey')
        def close(self): pass
    monkeypatch.setitem(sys.modules, 'sounddevice', SimpleNamespace(OutputStream=Output))
    session = Session(cfg(), tmp_path, test=test, dial_hz=dial, output_device=0, preview_device=1)
    if not test:
        session.rig = Rig()
    session.draft = dict(audio=np.ones(48000, np.float32), duration_s=1., kind='cq', text='test', wav='test.wav')
    assert not operations  # constructing a session cannot key
    session.play(preview=preview)
    session.finish_playback(cancelled=True)
    if test or preview:
        assert operations == ['prepare', 'audio', 'abort', 'close']
    else:
        assert operations == ['prepare', 'key', 'audio', 'unkey', 'abort', 'close']
    assert session.contact.state == 'CQ'  # cancelled playback does not advance
    session.close()


def test_new_station_facts_do_not_duplicate_voice_identity():
    raw = dict(schema=1, station=dict(mycall='W9SSJ', grid='EN63bb', qth='Chicago', name='Sam',
               control='local', regulatory='unregulated', because='offline test'))
    parsed = config.parse(raw, environ={})
    assert script.facts_from(parsed)['qth'] == 'Chicago'
    assert not hasattr(parsed.voice, 'operator')
    assert parsed.models.memory_gb == 48
    for overrides in (dict(drive=float('nan')), dict(silence_s=0), dict(voice_agc=0)):
        with pytest.raises(config.ConfigError):
            config.parse(dict(raw, voice=overrides), environ={})
