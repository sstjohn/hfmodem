# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Speech host protocol: bounded requests, local weights, no device access."""
from dataclasses import asdict
import json
import importlib.util
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from hfmodem.contact.language import stop_process


class Speech:
    def __init__(self, settings, directory):
        self.lock, self.responses = threading.Lock(), queue.Queue()
        self.log = (Path(directory) / 'speech-host.log').open('w')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        package_roots = [str(Path(importlib.util.find_spec(name).origin).resolve().parents[1])
                         for name in ('hfmodem', 'hfhost')]
        env['PYTHONPATH'] = os.pathsep.join([*package_roots, os.environ.get('PYTHONPATH', '')])
        self.process = None
        try:
            self.process = subprocess.Popen([settings.worker_python or sys.executable, '-u', '-m',
                'hfmodem.macagua.speech', json.dumps(asdict(settings))], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            try:
                ready = self.responses.get(timeout=120)
            except queue.Empty:
                raise RuntimeError(f'speech host did not load; see {self.log.name}') from None
            if not ready or 'error' in ready:
                raise RuntimeError(f'speech host failed: {ready}; see {self.log.name}')
            self.ready = ready
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as exc:
            self.responses.put(dict(error=str(exc)))
        finally:
            self.responses.put(None)

    def request(self, operation, **values):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f'speech host exited; see {self.log.name}')
            self.process.stdin.write(json.dumps(dict(operation=operation, **values)) + '\n')
            self.process.stdin.flush()
            try:
                answer = self.responses.get(timeout=60)
            except queue.Empty:
                stop_process(self.process)
                raise RuntimeError('speech host timed out; restart the session') from None
            if answer is None or 'error' in answer:
                raise RuntimeError(f'speech request failed: {answer}')
            return answer

    def close(self):
        stop_process(self.process)
        self.log.close()


def host(settings):
    # Imports and every model allocation stay out of the audio/terminal process.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import contextlib
    import numpy as np
    import soundfile as sf
    import torch
    import mlx_whisper
    import psutil
    from silero_vad import load_silero_vad
    from hfmodem.macagua.chain import resample

    output = sys.stdout
    def send(data):
        output.write(json.dumps(data) + '\n')
        output.flush()

    with contextlib.redirect_stdout(sys.stderr):
        torch.set_num_threads(2)
        synth = Path(settings['synthesis_model']).expanduser()
        recognition = Path(settings['recognition_model']).expanduser()
        if not recognition.is_dir() or not synth.is_dir():
            raise ValueError('recognition_model and synthesis_model must name local model directories')
        if settings['synthesis_backend'] == 'piper':
            import wave
            from piper import PiperVoice, SynthesisConfig
            voice = PiperVoice.load(str(synth / 'en_AU-librivox-medium.onnx'))
        else:
            from kokoro import KModel, KPipeline
            model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(synth / 'config.json'),
                           model=str(synth / 'kokoro-v1_0.pth'))
            pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=model, device='cpu')
            voice = torch.load(synth / 'voices' / (settings['synthesis_voice'] + '.pt'),
                               map_location='cpu', weights_only=True)
        vad = load_silero_vad()
        # Load Whisper once, then share the cached model across every over.
        mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=str(recognition),
            language='en', condition_on_previous_text=False, verbose=None)
    send(dict(ready=True, rss_gb=psutil.Process().memory_info().rss / 1e9))
    for line in sys.stdin:
        try:
            start = time.monotonic()
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                op = request['operation']
                if op == 'vad':
                    audio = np.asarray(request['audio'], dtype=np.float32)
                    probability = float(vad(torch.from_numpy(audio), 16000).item())
                    result = dict(probability=probability)
                elif op == 'vad_reset':
                    vad.reset_states()
                    result = dict(ok=True)
                elif op == 'synthesize':
                    if settings['synthesis_backend'] == 'piper':
                        with wave.open(request['path'], 'wb') as stream:
                            voice.synthesize_wav(request['text'], stream, SynthesisConfig(
                                speaker_id=settings['synthesis_speaker'], length_scale=1/settings['speed']))
                        audio, rate = sf.read(request['path'], dtype='float32')
                    else:
                        chunks = [audio.numpy() for _, _, audio in pipeline(request['text'], voice=voice,
                            speed=settings['speed'])]
                        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                        rate = 24000
                        sf.write(request['path'], audio, rate, subtype='FLOAT')
                    result = dict(path=request['path'], rate=rate, duration_s=len(audio) / rate)
                elif op == 'transcribe':
                    audio, rate = sf.read(request['path'], dtype='float32')
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio = resample(audio, rate, 16000)
                    # Light gain bounded at 6 dB, never raising a noise floor without VAD.
                    rms = float(np.sqrt(np.mean(audio * audio)))
                    audio = np.clip(audio * min(2., .1 / max(rms, 1e-6)), -1, 1)
                    hint = ' '.join([request.get('own_call', ''), request.get('partner', '')]).strip()
                    raw = mlx_whisper.transcribe(audio, path_or_hf_repo=str(recognition), language='en',
                        initial_prompt=f'Amateur radio. {hint}', condition_on_previous_text=False,
                        word_timestamps=True, no_speech_threshold=.6, compression_ratio_threshold=2.4,
                        verbose=None)
                    segments = [s for s in raw.get('segments', [])
                                if s.get('no_speech_prob', 0) < .6 and s.get('compression_ratio', 0) < 2.4]
                    words = [w for s in segments for w in s.get('words', [])]
                    result = dict(text=' '.join(s['text'].strip() for s in segments), words=words,
                                  probability=float(np.mean([w.get('probability', 0) for w in words])) if words else 0.)
                else:
                    raise ValueError(f'unknown speech operation: {op}')
            result['elapsed_s'] = time.monotonic() - start
            send(result)
        except Exception as exc:
            send(dict(error=f'{type(exc).__name__}: {exc}'))


if __name__ == '__main__':
    try:
        host(json.loads(sys.argv[1]))
    except Exception as exc:
        print(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')), flush=True)
