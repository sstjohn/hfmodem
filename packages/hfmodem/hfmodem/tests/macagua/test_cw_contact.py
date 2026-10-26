# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CW synthesis decoded by the existing independent narrow-track reader."""
import subprocess
import sys
import numpy as np
import pytest

from hfmodem.contact.frame import Over
from hfmodem.contact.runtime import Session
from hfmodem.contact.terminal import Terminal
from hfmodem.contact.simulation import TestOptions as SimulationOptions
from hfmodem.merlin import script
from hfmodem.merlin.modem import CwIO, CwWaveform, render
from hfmodem.tests.macagua.test_voice import cfg


@pytest.mark.parametrize('wpm', [12., 16., 20.])
@pytest.mark.parametrize('tone', [600., 700., 1000.])
def test_cw_contact_over_round_trip_with_noise(tmp_path, wpm, tone):
    from scipy.io import wavfile
    text = 'W9SSJ DE KD0DLK UR 5NN QTH BOSTON K'
    audio = render(text, wpm, tone)
    audio += np.random.default_rng(41).normal(0, .003, len(audio)).astype(np.float32)
    path = tmp_path / 'partner.wav'
    wavfile.write(path, 48000, audio)
    modem = CwIO(wpm, tone)
    result = modem.request('transcribe', path=str(path))
    assert result['text'] == text
    waveform = CwWaveform(wpm, tone)
    raw, over = waveform.parse(None, result, waveform.callsigns(result['text']), 'W9SSJ')
    assert (over.de, over.to, over.report, over.qth, over.end) == ('KD0DLK', 'W9SSJ', '599', 'BOSTON', 'over')


def test_cw_prosign_closes_the_same_contact(tmp_path):
    modem = CwIO(20, 700)
    text = script.close('W9SSJ', 'KD0DLK')
    path = str(tmp_path / 'close.wav')
    modem.request('synthesize', text=text, path=path)
    result = modem.request('transcribe', path=path)
    assert result['text'] == text and '<SK>' in result['text']
    over = CwWaveform().parse(None, result, CwWaveform().callsigns(result['text']), 'KD0DLK')[1]
    assert over.end == 'clear'


def test_cw_parser_does_not_treat_ur_call_as_a_signal_report():
    waveform = CwWaveform()
    result = dict(text='R UR CALL KD0DLK K', probability=.98)
    over = waveform.parse(None, result, waveform.callsigns(result['text']), 'W9SSJ')[1]
    assert over.report == ''


def test_cw_activity_gate_detects_tone_and_rejects_silence_and_noise():
    modem = CwIO(20, 700)
    t = np.arange(512) / 16000
    assert modem.request('vad', audio=(.1 * np.sin(2*np.pi*700*t)).tolist())['probability'] == 1
    assert modem.request('vad', audio=[0.]*512)['probability'] == 0
    noise = np.random.default_rng(21).normal(0, .01, 512)
    assert modem.request('vad', audio=noise.tolist())['probability'] == 0


def test_cw_session_uses_common_pilot_and_local_codec(tmp_path):
    session = Session(cfg(), tmp_path, test=True, simulation=SimulationOptions(), waveform='cw')
    session.speech = session.waveform.open_modem(session.cfg.voice, tmp_path)
    terminal = Terminal(session)
    try:
        terminal.heard_event(dict(kind='heard', recognition=dict(text='W9SSJ DE KD0DLK K'),
                                  over=Over(de='KD0DLK').record(), wav='test.wav'))
        session.pending.result(timeout=3)
        draft = session.poll_draft()
        assert draft['kind'] == 'exchange' and 'DE W9SSJ K' in draft['text']
        assert 'Whiskey' not in draft['text'] and session.contact.partner == 'KD0DLK'
        terminal.heard_event(dict(kind='heard', recognition=dict(text='TU 73 DE KD0DLK <SK>'),
                                  over=Over(de='KD0DLK', end='clear').record(), wav='test.wav'))
        session.pending.result(timeout=3)
        draft = session.poll_draft()
        assert draft['kind'] == 'close' and draft['text'].endswith('<SK>')
    finally:
        session.close()


def test_cw_does_not_import_or_load_speech_stack():
    result = subprocess.run([sys.executable, '-c',
        'import sys; from hfmodem.contact.runtime import Session; '
        'from hfmodem.contact.waveforms import select_waveform; '
        'w=select_waveform("cw"); w.open_modem(None,None); '
        'assert not any(n.startswith(("hfmodem.macagua", "mlx_whisper", "kokoro", "torch")) for n in sys.modules)'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_second_call_regression_keeps_identity_and_closes_on_digit_signoff(tmp_path, monkeypatch):
    session = Session(cfg(), tmp_path, test=True, simulation=SimulationOptions())
    terminal = Terminal(session)
    drafts = []
    monkeypatch.setattr(session, 'make_draft', lambda kind, text: drafts.append((kind, text)))
    try:
        def hear(text, over):
            terminal.heard_event(dict(kind='heard', recognition=dict(text=text), over=over.record(), wav='test.wav'))
        hear("This is KD0DLK. What's up?", Over(de='KD0DLK', confidence={'de': .92}))
        assert drafts[-1][0] == 'exchange' and session.pending is None
        hear('W9SSJ KD0DLK', Over(de='KD0DLK', confidence={'de': .01}))
        assert drafts[-1][0] == 'free' and 'I have your call' in drafts[-1][1]
        hear('Why do you keep asking me for my call?', Over(confidence={'de': .98}))
        assert 'I have your call' in drafts[-1][1]
        current = len(drafts)
        hear('Over.', Over())
        assert len(drafts) == current
        hear("W90SJ, I am going to let you go 7-3.", Over(de='W90SJ', confidence={'de': .88}))
        assert drafts[-1][0] == 'close' and session.contact.partner == 'KD0DLK'
    finally:
        session.close()


def test_automatic_cw_receive_stream_decodes_caller_without_speech_models(tmp_path, monkeypatch):
    from hfmodem.contact.audio import BLOCK, Segmenter
    session = Session(cfg(), tmp_path, test=True, simulation=SimulationOptions(), waveform='cw')
    session.speech = session.waveform.open_modem(None, tmp_path)
    terminal = Terminal(session)
    chain = session.waveform.receive_chain()
    segmenter = Segmenter(silence_s=session.waveform.silence_s(.8))
    audio = np.r_[render('W9SSJ DE KD0DLK K'), np.zeros(48000)]
    try:
        for offset in range(0, len(audio)-BLOCK, BLOCK):
            block = audio[offset:offset+BLOCK]
            probability = session.speech.request('vad', audio=chain.block(block).tolist())['probability']
            segment = segmenter.feed(block, probability, False, offset/48000)
            if segment:
                session._decode(segment)
        assert session.events.qsize() == 1
        terminal.step()
        session.pending.result(timeout=3)
        draft = session.poll_draft()
        assert session.contact.partner == 'KD0DLK' and draft['kind'] == 'exchange'
        # The outgoing CW over can be copied by the independent receive reader.
        copied = session.speech.request('transcribe', path=draft['wav'])
        assert copied['text'] == draft['text']
    finally:
        session.close()


def test_cw_rst_template_fits_the_default_key_watchdog(tmp_path):
    session = Session(cfg(), tmp_path, test=True, waveform='cw')
    session.speech = session.waveform.open_modem(None, tmp_path)
    try:
        session.make_draft('exchange', script.exchange('W9SSJ','KD0DLK','579',session.facts))
        session.pending.result(timeout=3)
        draft=session.poll_draft()
        assert draft['duration_s'] < 24 and '579' in draft['text']
    finally:
        session.close()
