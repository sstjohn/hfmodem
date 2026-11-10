# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Speech and the separate CW backend use PCM/rate JSON transport."""
import io
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem.contact.audio import RATE
from hfmodem.merlin import modem
from hfmodem.macagua import speech
from hfmodem.tests.station.recording_stage4_support import audio_files, LOCAL, PEER


@pytest.mark.parametrize('wpm,tone', [(12., 600.), (20., 1000.)])
def test_cw_io_synthesis_and_transcription_pcm_rate_without_file(stage4, monkeypatch, wpm, tone):
    monkeypatch.setenv('HFMODEM_RECORDING', 'off')
    def forbidden(*a, **k):
        raise AssertionError('CwIO attempted file audio transport under off')
    monkeypatch.setattr(modem.wavfile, 'read', forbidden)
    monkeypatch.setattr(modem.wavfile, 'write', forbidden)
    text = f'{LOCAL} DE {PEER} K'
    cw = modem.CwWaveform(wpm, tone).open_modem(None, stage4.root)
    try:
        try:
            rendered = cw.request('synthesize', text=text)
        except KeyError as exc:
            pytest.fail(f'CwIO synthesis still requires a file path: {exc}')
        assert rendered['rate'] == RATE
        audio = np.asarray(rendered['audio'], np.float32)
        np.testing.assert_array_equal(audio, modem.render(text, wpm, tone))
        assert rendered['duration_s'] == len(audio) / RATE
        result = cw.request('transcribe', audio=audio.tolist(), rate=RATE)
        assert result['text'] == text
        assert all(0 <= word['start'] < len(audio) / RATE for word in result['words'])
        assert result['elapsed_s'] >= 0 and result['probability'] > 0
        assert audio_files(stage4.root) == []
    finally:
        cw.close()


@pytest.mark.parametrize('rate', (16000, 48000))
def test_cw_reader_pcm_contract_preserves_input_rate_and_timing(stage4, monkeypatch, rate):
    monkeypatch.setattr(modem.wavfile, 'read', lambda *a: pytest.fail('CW transcription read an audio file'))
    cw = modem.CwIO(20, 700)
    audio = modem.render(f'{LOCAL} DE {PEER} K', rate=rate)
    try:
        result = cw.request('transcribe', audio=audio.tolist(), rate=rate)
    except KeyError as exc:
        pytest.fail(f'CwIO transcription still requires a file path: {exc}')
    assert result['text'] == f'{LOCAL} DE {PEER} K'
    assert result['words'][-1]['start'] < len(audio) / rate


@pytest.fixture
def speech_host(stage4, monkeypatch):
    recognition, synthesis = stage4.root / 'recognition', stage4.root / 'synthesis'
    recognition.mkdir()
    synthesis.mkdir()
    seen = []
    def transcribe(audio, **kwargs):
        seen.append((np.asarray(audio).copy(), kwargs))
        return dict(segments=[dict(text=PEER, words=[dict(word=PEER, start=0., probability=.98)],
                                  no_speech_prob=.01, compression_ratio=1.)])
    monkeypatch.setitem(sys.modules, 'mlx_whisper', NS(transcribe=transcribe))
    monkeypatch.setitem(sys.modules, 'torch', NS(Tensor=type('Tensor', (), {}), set_num_threads=lambda n: None,
        from_numpy=lambda a: a, load=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, 'silero_vad', NS(load_silero_vad=lambda:
        NS(__call__=lambda *a: NS(item=lambda: .9), reset_states=lambda: None)))
    monkeypatch.setitem(sys.modules, 'psutil', NS(Process=lambda: NS(memory_info=lambda: NS(rss=0))))
    class Pipeline:
        def __init__(self, **k): pass
        def __call__(self, *a, **k):
            yield None, None, NS(numpy=lambda: np.full(32, .125, np.float32))
            yield None, None, NS(numpy=lambda: np.full(32, .25, np.float32))
    monkeypatch.setitem(sys.modules, 'kokoro', NS(KModel=lambda **k: None, KPipeline=Pipeline))
    class Voice:
        @staticmethod
        def load(*a): return Voice()
        def synthesize(self, *a, **k):
            yield NS(audio_float_array=np.full(64, .125, np.float32), sample_rate=22050,
                     sample_channels=1, sample_width=2)
        def synthesize_wav(self, *a, **k):
            raise AssertionError('speech host attempted WAV synthesis under off')
    monkeypatch.setitem(sys.modules, 'piper', NS(PiperVoice=Voice, SynthesisConfig=lambda **k: NS(**k)))
    monkeypatch.setitem(sys.modules, 'soundfile', NS(
        read=lambda *a, **k: pytest.fail('speech host attempted file transcription'),
        write=lambda *a, **k: pytest.fail('speech host attempted file synthesis')))
    def run(request, backend='kokoro'):
        settings = dict(recognition_model=str(recognition), synthesis_model=str(synthesis),
            synthesis_backend=backend, synthesis_speaker=0, synthesis_voice='test', speed=1.)
        output = io.StringIO()
        monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(request) + '\n'))
        monkeypatch.setattr(sys, 'stdout', output)
        speech.host(settings)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert rows[0]['ready']
        return rows[-1], seen
    return run


@pytest.mark.parametrize('rate', (8000, 48000))
def test_macagua_host_transcribes_in_memory_json_pcm(speech_host, rate):
    result, seen = speech_host(dict(operation='transcribe', audio=[.125] * 80, rate=rate,
                                    own_call=LOCAL, partner=PEER))
    assert 'error' not in result, f'speech host has no PCM transcription contract: {result}'
    assert result['text'] == PEER and result['probability'] == .98
    audio, kwargs = seen[-1]
    assert len(audio) == round(80 * 16000 / rate)
    assert PEER in kwargs['initial_prompt'] and LOCAL in kwargs['initial_prompt']


@pytest.mark.parametrize('backend,rate', [('kokoro', 24000), ('piper', 22050)])
def test_macagua_host_synthesizes_pcm_without_temporary_audio(speech_host, backend, rate):
    result, _ = speech_host(dict(operation='synthesize', text=f'{LOCAL} calling'), backend)
    assert 'error' not in result, f'speech host has no PCM synthesis contract: {result}'
    assert result['rate'] == rate and len(result['audio']) == 64
    assert result['duration_s'] == 64 / rate
    assert 'path' not in result


def test_speech_client_pipe_preserves_pcm_rate_and_text(stage4, monkeypatch):
    from hfmodem.core.config import Voice
    requests = io.StringIO()
    output = io.StringIO(json.dumps(dict(ready=True, rss_gb=0.)) + '\n' +
                         json.dumps(dict(text=PEER, probability=.98, words=[])) + '\n')
    process = NS(stdin=requests, stdout=output, poll=lambda: None,
                 terminate=lambda: None, wait=lambda **k: 0)
    monkeypatch.setattr(speech.subprocess, 'Popen', lambda *a, **k: process)
    client = speech.Speech(Voice(), stage4.root)
    try:
        result = client.request('transcribe', audio=[.125, -.125], rate=16000,
                                own_call=LOCAL, partner=PEER)
        assert result['text'] == PEER
        sent = json.loads(requests.getvalue())
        assert sent == dict(operation='transcribe', audio=[.125, -.125], rate=16000,
                            own_call=LOCAL, partner=PEER)
        assert audio_files(stage4.root) == []
    finally:
        client.close()


def test_cw_synthesis_bounds_text_before_render(monkeypatch):
    monkeypatch.setattr(modem, 'render', lambda *a, **k: pytest.fail('unbounded CW render'))
    with pytest.raises(ValueError, match='text'):
        modem.CwIO(20, 700).request('synthesize', text='E' * 10001)


@pytest.mark.parametrize('audio', ([], [[[.125]]]))
def test_speech_host_rejects_empty_or_non_pcm_dimensions(speech_host, audio):
    result, _ = speech_host(dict(operation='transcribe', audio=audio, rate=16000))
    assert 'error' in result
