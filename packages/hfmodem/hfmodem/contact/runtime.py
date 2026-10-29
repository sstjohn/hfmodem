# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The same supervised loop on computer audio, recordings, and an RTS-keyed rig."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import threading
import time
import wave
import warnings

import numpy as np
from scipy.io import wavfile

from hfmodem.contact.exchange import Contact
from hfmodem.contact.language import Language
from hfmodem.core.devices import find_device
from hfmodem.core.regulatory import Emission
from .audio import BLOCK, RATE, Segmenter, resample
from .waveforms import select_waveform
from .facts import facts_from
from .txmark import Tag, LLMMark


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_wav(path, audio, rate=RATE):
    audio = np.asarray(audio)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())


def read_wav(path):
    with warnings.catch_warnings():
        # libsndfile FLOAT WAVs carry a PEAK/fact chunk; scipy reads the PCM but
        # warns about that harmless metadata, which would overwrite the terminal.
        warnings.simplefilter('ignore', wavfile.WavFileWarning)
        rate, audio = wavfile.read(path)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def pcm_device(name, direction):
    # Defaults are only allowed in computer test mode, whose callers pass None.
    return None if name is None or name == '' else find_device(name, direction, required=True)


class Session:
    def __init__(self, cfg, directory, *, test=False, replay=None, input_device=None,
                 output_device=None, preview_device=None, dial_hz=None, monitor=False, simulation=None, waveform="voice", cw_wpm=20., cw_tone=700., watermark=None, watermark_key=None, tx_mark=None):
        if simulation is not None and (not test or replay or monitor):
            raise ValueError('Automatic replies are available only in computer-audio test mode.')
        if watermark and (not test or replay or monitor or waveform != 'voice'):
            raise ValueError('Watermarking is available only in voice computer-audio test mode.')
        if watermark_key is not None and (not (watermark or (tx_mark and tx_mark.mode != 'off'))
                                         or not isinstance(watermark_key, bytes) or len(watermark_key) != 32):
            raise ValueError('A shared 32-byte selection key requires an enabled watermark')
        self.watermark_key = watermark_key
        self.tx_mark = tx_mark or Tag()
        if self.tx_mark.mode != 'off' and (not test or replay or monitor or watermark):
            raise ValueError('Transmission marks are available in computer tests; choose one marking method')
        self.watermark_name, self.watermark_encoder = watermark, None
        self.mark_codec = None
        self.cfg, self.directory = cfg, Path(directory)
        self.test, self.replay, self.monitor = test, replay, monitor
        self.simulation = simulation
        self.waveform = select_waveform(waveform, cw_wpm, cw_tone)
        self.codec = self.waveform.codec
        self.receive_max_over_s = max(cfg.voice.max_over_s, 120) if waveform == 'cw' else cfg.voice.max_over_s
        if waveform == 'cw' and test:
            self.receive_max_over_s = max(self.receive_max_over_s, 190)
        self.input_device, self.output_device = input_device, output_device
        self.preview_device = preview_device
        self.audio_ready = False
        self.radio_check = None
        self.tx_power_w = cfg.declared_power_w
        self.receive_failure = None
        self.dial_hz = dial_hz
        self.contact = Contact(cfg.station.mycall.upper())
        self.facts = facts_from(cfg)
        self.events, self.blocks = queue.Queue(), queue.Queue(maxsize=1800)
        self.stop, self.output_done = threading.Event(), threading.Event()
        self.capture = self.output = self.rig = self.speech = self.language = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='voice-draft')
        self.pending = None
        self.revision, self.tx_number, self.rx_number = 0, 0, 0
        self.draft = None
        self._cq_cache = None
        self._draft_cache_key = None
        self.playing = None
        self.self_until = 0.
        self.lost_blocks = 0
        self.sink_lock = threading.Lock()
        self.contacts, self.edits, self.tx_scripts = [], [], []
        self.history = []
        self.receiving = False
        self.latest = None
        self.started = utc()
        self.contact_started = self.started
        self._closed = False

    def event(self, kind, **record):
        data = dict(kind=kind, utc=utc(), **record)
        with self.sink_lock:
            with (self.directory / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(data) + '\n')
        return data

    def start_models(self):
        self.speech = self.waveform.open_modem(self.cfg.voice, self.directory)
        self.language = Language(self.cfg.voice.language_model, self.cfg.voice.llama_server, self.directory)
        if self.watermark_name:
            from .watermark import Encoder, Options
            self.watermark_encoder = Encoder(self.language, Options(codebook=self.watermark_name, symbol_bits=1,
                method='logit'), key=self.watermark_key)
            self.event('watermark_configuration', codebook=self.watermark_name,
                       vocabulary=self.watermark_encoder.codebook.words,
                       vocabulary_sha256=self.watermark_encoder.codebook.digest)
        elif self.test:
            self.mark_codec = LLMMark(self.language, key=self.watermark_key)
            self.event('tx_mark_configuration', layer='llm-logit-bias',
                       vocabulary=self.mark_codec.words, vocabulary_sha256=self.mark_codec.digest,
                       keyed=self.mark_codec.keyed)
        # llama.cpp writes its own memory allocation detail; RSS plus MLX pool is
        # sampled by the speech host and the language process has a separate sample.
        import subprocess
        rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(self.language.process.pid)],
                             capture_output=True, text=True, timeout=2)
        total = self.speech.ready['rss_gb'] + int(rss.stdout.strip() or 0) * 1024 / 1e9
        self.event('models_loaded', measured_rss_gb=total, memory_gb=self.cfg.models.memory_gb,
                   settings=asdict(self.cfg.voice))
        if total > self.cfg.models.memory_gb:
            self.events.put(dict(kind='notice', text=f'Model RSS {total:.1f} GB exceeds memory ceiling '
                                 f'{self.cfg.models.memory_gb:g} GB; continuing.'))

    def start_audio(self):
        import sounddevice as sd
        if self.replay:
            self.thread = threading.Thread(target=self._replay, daemon=True)
        else:
            if not self.audio_ready:
                self.prepare_audio()
            self.capture = sd.InputStream(device=pcm_device(self.input_device, 'in'),
                samplerate=RATE, channels=1, blocksize=BLOCK, dtype='float32', callback=self._capture)
            self.capture.start()
            self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def prepare_audio(self):
        """Validate concrete live endpoints before opening CAT or the RTS line."""
        import sounddevice as sd
        if not self.test and (self.input_device is None or self.input_device == ''):
            raise ValueError('Name the radio receive device before starting audio')
        self.input_device = pcm_device(self.input_device, 'in')
        sd.check_input_settings(device=self.input_device, channels=1, dtype='float32', samplerate=RATE)
        if not self.monitor and not self.replay:
            if not self.test and (self.output_device is None or self.output_device == ''):
                raise ValueError('Name the rig output device before starting audio')
            self.output_device = pcm_device(self.output_device, 'out')
            sd.check_output_settings(device=self.output_device, channels=1, dtype='float32', samplerate=RATE)
        if self.preview_device is not None:
            self.preview_device = self._playback_device(preview=True)
            sd.check_output_settings(device=self.preview_device, channels=1, dtype='float32', samplerate=RATE)
        self.audio_ready = True
        self.event('audio_ready', input_device=self.input_device, output_device=self.output_device,
                   preview_device=self.preview_device, sample_rate=RATE)

    def _playback_device(self, preview=False):
        import sounddevice as sd
        if not preview:
            if not self.test and (self.output_device is None or self.output_device == ''):
                raise ValueError('Name the rig output device before sending')
            return pcm_device(self.output_device, 'out')
        device = pcm_device(self.preview_device, 'out')
        if not self.test:
            # Resolve the default too: an OS default can be the rig codec.
            if device is None:
                device = int(sd.default.device[1])
            rig_device = pcm_device(self.output_device, 'out')
            if rig_device is None or device == rig_device:
                raise ValueError('Preview must use local speakers; set --preview-output to a device distinct from the rig codec')
        return device

    def _receive_failed(self, exc):
        if self.receive_failure:
            return
        self.receive_failure = str(exc)
        self.receiving = False
        self.events.put(dict(kind='receive_error', text=self.receive_failure))

    def check_receive(self):
        if self.capture is not None and not self.receive_failure:
            try:
                if not self.capture.active:
                    self._receive_failed(RuntimeError('Receive audio stream stopped; restart the session'))
            except Exception as exc:
                self._receive_failed(exc)

    def _capture(self, audio, frames, times, status):
        if status:
            self.lost_blocks += 1
        at = time.monotonic()
        item = (audio[:, 0].copy(), self.playing is not None or at < self.self_until, at)
        try:
            self.blocks.put_nowait(item)
        except queue.Full:
            self.lost_blocks += 1

    def _receive(self):
        segmenter = Segmenter(self.cfg.voice.vad_threshold, self.waveform.silence_s(self.cfg.voice.silence_s),
                              self.cfg.voice.preroll_s, self.receive_max_over_s)
        chain = self.waveform.receive_chain()
        lost = 0
        previous_own = None
        try:
            while not self.stop.is_set():
                try:
                    audio, own, at = self.blocks.get(timeout=.2)
                except queue.Empty:
                    continue
                if self.lost_blocks != lost:
                    lost = self.lost_blocks
                    segmenter = Segmenter(self.cfg.voice.vad_threshold, self.waveform.silence_s(self.cfg.voice.silence_s),
                                          self.cfg.voice.preroll_s, self.receive_max_over_s)
                    chain = self.waveform.receive_chain()
                    self.speech.request('vad_reset')
                    self.events.put(dict(kind='notice', text=f'Audio gap ({lost} blocks lost); over discarded.'))
                if previous_own is not None and own != previous_own:
                    # A quick answer after our audio is a new over, even before
                    # VAD has accumulated its normal end-of-over silence.
                    previous = segmenter.finish()
                    if previous is not None:
                        self._decode(previous)
                    segmenter = Segmenter(self.cfg.voice.vad_threshold, self.waveform.silence_s(self.cfg.voice.silence_s),
                                          self.cfg.voice.preroll_s, self.receive_max_over_s)
                    self.receiving = False
                previous_own = own
                pcm = chain.block(audio)
                probability = self.speech.request('vad', audio=pcm.tolist())['probability']
                if probability >= self.cfg.voice.vad_threshold and not own:
                    self.receiving = True
                segment = segmenter.feed(audio, probability, own, at)
                if segment is not None:
                    self._decode(segment)
                    self.receiving = False
        except Exception as exc:
            self._receive_failed(exc)

    def _replay(self):
        try:
            audio, rate = read_wav(self.replay)
            audio = resample(audio, rate, RATE)
            segmenter = Segmenter(self.cfg.voice.vad_threshold, self.waveform.silence_s(self.cfg.voice.silence_s),
                                  self.cfg.voice.preroll_s, self.receive_max_over_s)
            chain = self.waveform.receive_chain()
            for offset in range(0, len(audio), BLOCK):
                if self.stop.is_set():
                    break
                block = np.zeros(BLOCK, np.float32)
                kept = audio[offset:offset + BLOCK]
                block[:len(kept)] = kept
                probability = self.speech.request('vad', audio=chain.block(block).tolist())['probability']
                segment = segmenter.feed(block, probability, False, offset / RATE)
                if segment:
                    self._decode(segment)
            tail = segmenter.finish()
            if tail:
                self._decode(tail)
            self.events.put(dict(kind='replay_done', text='Recording finished.'))
        except Exception as exc:
            self._receive_failed(exc)

    def _decode(self, segment):
        self.rx_number += 1
        stem = f'over-{self.rx_number:03d}'
        path = self.directory / f'{stem}.wav'
        write_wav(path, segment.audio)
        # Conditioning for ASR does not replace the original recording.
        conditioned = self.waveform.receive_chain().block(segment.audio)
        asr_path = self.directory / f'{stem}-16k.wav'
        write_wav(asr_path, conditioned, 16000)
        partner = self.contact.partner
        result = self.speech.request('transcribe', path=str(asr_path), own_call=self.contact.own_call,
                                     partner=partner)
        content, received_mark, mark_evidence = result['text'], None, None
        carrier = None
        if self.mark_codec is not None:
            carrier, candidate_body = self.mark_codec.split(result['text'], self.waveform.name)
            if carrier:
                sender_calls = self.waveform.callsigns(candidate_body)
                sender = ','.join(call for call in sorted(sender_calls)
                                  if call != self.contact.own_call) or partner or 'unknown'
                stream = 'self' if segment.self_copy else 'partner:' + sender
                mark_evidence = self.mark_codec.observe(carrier, stream=stream)
                received_mark = mark_evidence['mark']
                # A discourse marker alone is not proof of a carrier. Keep
                # ordinary unmarked conversation intact while evidence is weak.
                if received_mark:
                    content = candidate_body
        candidates = self.waveform.callsigns(content)
        raw, over = {}, None
        if result['text'] and not segment.self_copy:
            raw, over = self.waveform.parse(self.language, dict(result, text=content), candidates, self.contact.own_call)
        watermark = None
        if self.watermark_encoder is not None:
            # The experimental prefix ends at the spoken discourse marker.
            prefix = re.split(r'\banyway\b', result['text'], maxsplit=1, flags=re.I)[0]
            watermark = self.watermark_encoder.codebook.score(prefix)
            watermark['signature_match'] = watermark['symbol'] == 0
        data = self.event('self' if segment.self_copy else 'heard', wav=str(path),
            started=segment.started, ended=segment.ended, speech_probability=segment.speech_probability,
            recognition=result, content_text=content, transmission_mark=received_mark,
            mark_evidence=mark_evidence,
            calls=candidates, extracted=raw, over=over.record() if over else None,
            partner_hint=partner, watermark=watermark)
        if segment.self_copy:
            data['loopback_calls_ok'] = self.contact.own_call in candidates
        (self.directory / f'{stem}.json').write_text(json.dumps(data, indent=2) + '\n')
        self.events.put(data)

    def make_draft(self, kind, text):
        if self.playing is not None:
            raise ValueError('Finish or cancel the playing over before changing the draft.')
        self.revision += 1
        revision = self.revision
        self.draft = None
        # Plain CQs are immutable audio for these settings. Marked experiments
        # still generate new cover when explicitly drafting a new CQ.
        key = (text, self.cfg.voice, self.waveform) if kind == 'cq' and self.tx_mark.mode == 'off' and not self.watermark_name else None
        self._draft_cache_key = key
        if key is not None and self._cq_cache is not None and self._cq_cache[0] == key:
            cached = self._cq_cache[1]
            self.pending = Future()
            self.pending.set_result(dict(cached, revision=revision, cache_hit=True, synthesis_s=0.,
                cached_from_revision=cached.get('cached_from_revision', cached['revision'])))
        else:
            self.pending = self.executor.submit(self._render_over, kind, text, revision)
        return revision

    def set_tx_mark(self, tag):
        if not self.test or self.rig is not None or self.monitor or self.replay:
            raise ValueError('Transmission marks are currently available in computer tests only')
        if self.playing is not None:
            raise ValueError('Finish or cancel playback before changing the mark')
        if self.watermark_name:
            raise ValueError('Restart without legacy --watermark to use the integrated mark control')
        self.cancel_draft()
        self.tx_mark = tag
        if self.mark_codec is not None:
            self.mark_codec.reset('self')
        self.event('tx_mark_changed', mark=tag.record())

    def _render_over(self, kind, text, revision, decision=None):
        source_text = self.codec.identify(text, self.contact.own_call)
        if self.waveform.name == 'voice':
            source_text = self.codec.vary_calls(source_text,
                (self.contact.own_call, self.contact.partner), self.cfg.voice.phonetics, seed=revision)
        spoken = source_text
        tag = self.tx_mark
        mark_generation = None
        if tag.mode != 'off':
            if not self.test or self.rig is not None:
                raise ValueError('Marked transmission experiment cannot use a rig')
            if self.mark_codec is None:
                raise ValueError('Start the language model before generating a marked transmission')
            _, spoken = self.mark_codec.split(spoken, self.waveform.name)
            source_text = spoken
            from .cover import TOPICS
            # Vary neutral cover topics so repeated overs do not merely repeat
            # the same small vocabulary. Neither topic nor prompt carries data.
            topic = tuple(TOPICS)[(revision - 1) % len(TOPICS)]
            mark_generation = self.mark_codec.generate(tag, topic=topic, seed=revision)
            carrier = mark_generation['text']
            spoken = (carrier + ' Anyway, ' + spoken if self.waveform.name == 'voice'
                      else self.codec.body(carrier) + ' BT ' + spoken)
        spoken, watermark = self.mark_text(spoken, seed=revision) if kind in ('exchange', 'free') else (spoken, None)
        source = self.directory / f'draft-{revision:03d}-raw.wav'
        timing = self.speech.request('synthesize', text=spoken, path=str(source))
        audio, rate = read_wav(source)
        options = {'compression': self.cfg.voice.compression} if self.waveform.name == 'voice' else {}
        audio = self.waveform.condition(audio, rate, self.cfg.voice.drive, **options)
        duration = len(audio) / RATE
        ceiling = min(self.cfg.rig.max_key_s / 1.25 - self.cfg.rig.ptt.settle_s,
                      self.cfg.rig.max_key_s - 2.1 - self.cfg.rig.ptt.settle_s)
        if self.test and self.waveform.name == 'cw' and tag.mode != 'off':
            # Wording carries the mark, so computer-only CW can take longer.
            # The radio render ceiling and keyer timing are unchanged.
            ceiling = 180.
        if duration > ceiling:
            raise ValueError(f'Over is {duration:.1f}s; shorten it below {ceiling:.1f}s for the key watchdog.')
        path = self.directory / f'draft-{revision:03d}.wav'
        write_wav(path, audio)
        return dict(kind=kind, text=spoken, source_text=source_text, tx_mark=tag.record(), watermark=watermark,
                    mark_generation=mark_generation,
                    duration_s=duration, wav=str(path), audio=audio, revision=revision,
                    synthesis_s=timing.get('elapsed_s'), conversation_decision=decision)

    def mark_text(self, text, seed=1):
        if self.watermark_encoder is None:
            return text, None
        if not self.test or self.rig is not None or self.waveform.name != 'voice':
            raise ValueError('Watermark experiment cannot run on a rig or CW session')
        from .cover import topic_from
        context = ' '.join(entry['text'] for entry in self.history[-2:])
        generated = self.watermark_encoder.generate(0, topic=topic_from(context), seed=seed)
        # Preserve every factual word and the final trusted ID verbatim.
        spoken = generated['text'] + ' Anyway, ' + text
        return spoken, dict(original_text=text, carrier=generated, signature=self.watermark_name)

    def make_free(self, text, conversation=False, may_close=False):
        if self.playing is not None:
            raise ValueError('Finish or cancel the playing over first.')
        self.revision += 1
        revision = self.revision
        self.draft = None
        self._draft_cache_key = None
        history = [dict(entry, text=self.codec.dialogue_text(entry['text'], self.contact.own_call,
                                                        self.contact.partner))
                   for entry in self.history]
        def render():
            kind, decision, reply, raw_reply = 'free', None, '', ''
            try:
                if conversation:
                    mode_args = {'mode': 'cw'} if self.waveform.name == 'cw' else {}
                    decision = self.language.converse(text, self.facts, history, may_close=may_close, **mode_args)
                    self.event('conversation_model', result=decision)
                    reply = decision['reply']
                    if decision['action'] == 'close':
                        kind = 'close'
                        reply = self.codec.close(self.contact.own_call, self.contact.partner)
                else:
                    reply = self.language.draft(text, self.facts, **({'mode': 'cw'} if self.waveform.name == 'cw' else {}))
                raw_reply = reply
                self.event('reply_model', reply=reply, revision=revision, over_kind=kind)
                if kind != 'close':
                    scan = reply
                    # A spelled Maidenhead grid can contain a substring that
                    # resembles a call. It is already a trusted station fact.
                    grid = self.facts.get('grid')
                    if grid:
                        for form in (grid, self.codec.spell_grid(grid)):
                            pattern = re.escape(form).replace(r'\ ', r'[\s,]+')
                            scan = re.sub(r'\b' + pattern + r'\b', '', scan, flags=re.I)
                    found = self.waveform.callsigns(scan)
                    unknown = [call for call in found if call not in (self.contact.own_call, self.contact.partner)]
                    if unknown:
                        raise ValueError('Model used an unconfirmed callsign: ' + ', '.join(dict.fromkeys(unknown)))
                    if found:
                        # The codec owns station identity. Known identifiers in
                        # a tiny model's response need not dead-end the contact.
                        reply = self.codec.dialogue_text(reply, self.contact.own_call, self.contact.partner)
                        if self.waveform.name == 'voice':
                            reply = re.sub(r'\b(?:over|QRZ|73)[.!?\s]*$', '', reply, flags=re.I).strip(' ,;')
                        if not reply.strip(' .,!?:;'):
                            raise ValueError('Model returned only callsigns, with no conversational reply')
                if self.facts.get('grid'):
                    reply = re.sub(r'\b' + re.escape(self.facts['grid']) + r'\b',
                        self.codec.spell_grid(self.facts['grid']), reply, flags=re.I)
                return self._render_over(kind, reply, revision, decision)
            except (ValueError, RuntimeError, OSError) as exc:
                # Keep the candidate as text, never as playable audio. Enter
                # cannot transmit it until an operator accepts an edit.
                return dict(kind=kind, text=getattr(exc, 'reply', raw_reply or reply), revision=revision,
                            review_error=str(exc), conversation_decision=decision)
        self.pending = self.executor.submit(render)

    def poll_draft(self):
        if self.pending is None or not self.pending.done():
            return None
        pending, self.pending = self.pending, None
        result = pending.result()
        if result['revision'] != self.revision:
            return None
        if result.get('review_error'):
            self.draft = None
            self.event('draft_review', draft=result)
            return result
        self.draft = result
        if result['kind'] == 'cq' and self._draft_cache_key is not None:
            self._cq_cache = (self._draft_cache_key, dict(result))
        self.event('draft', draft={k: v for k, v in result.items() if k != 'audio'})
        return result

    def cancel_draft(self):
        self.revision += 1
        if self.pending:
            self.pending.cancel()
            self.pending = None
        self.draft = None
        self._draft_cache_key = None

    def play(self, preview=False):
        """Main-thread Enter/preview, or the computer-only simulation pilot.

        Computer mode never instantiates Rig. Radio mode opens output before key,
        uses Rig.key's emission check/watchdog, and releases PTT before stream teardown.
        """
        import sounddevice as sd
        if self.playing is not None or self.draft is None:
            return
        if (self.monitor or self.replay) and not preview:
            raise ValueError('Monitor/replay sessions cannot transmit or play drafts.')
        if not preview:
            self.check_receive()
            if self.receive_failure:
                raise ValueError('Receive failed; restart before sending: ' + self.receive_failure)
            if not self.test and self.rig is None:
                raise ValueError('Radio playback requires an armed rig')
        device = self._playback_device(preview)
        if self.rig is not None and not preview and self.radio_check is not None:
            self.event('prekey_readback', readback=self.radio_check())
        draft = self.draft
        self.output_done.clear()
        playing = dict(draft=draft, preview=preview, started=utc(), token=None, cancelled=False,
                       audio_complete=False)
        self.playing = playing
        audio, index = draft['audio'], 0
        def callback(out, frames, times, status):
            nonlocal index
            out.fill(0)
            if status:
                playing['audio_status'] = str(status)
                raise sd.CallbackAbort
            count = min(frames, len(audio) - index)
            if count > 0:
                out[:count, 0] = audio[index:index + count]
                index += count
            if index >= len(audio):
                playing['audio_complete'] = True
                raise sd.CallbackStop
        def finished():
            if self.playing is playing:
                self.output_done.set()
        try:
            self.output = sd.OutputStream(device=device, samplerate=RATE,
                channels=1, blocksize=BLOCK, dtype='float32', callback=callback,
                finished_callback=finished)
            if self.rig is not None and not preview:
                # Sideband geometry matters: LSB audio occupies BELOW the dial.
                lo, hi = self.waveform.emission_edges(self.dial_hz)
                token = self.rig.key(Emission(self.dial_hz, lo, hi, power_w=self.tx_power_w,
                    designator=self.waveform.designator, technique=self.waveform.name), why='operator-approved voice over',
                    duration_s=draft['duration_s'], settle_s=self.cfg.rig.ptt.settle_s)
                self.playing['token'] = token
                time.sleep(self.cfg.rig.ptt.settle_s)
            self.output.start()
        except BaseException:
            self.finish_playback(cancelled=True)
            raise

    def finish_playback(self, cancelled=False):
        playing = self.playing
        if playing is None:
            return None
        cancelled = cancelled or not playing.get('audio_complete', True) or bool(playing.get('audio_status'))
        # The line goes down before any PortAudio stop, file or model request.
        failures = []
        if self.rig is not None and not playing['preview']:
            try:
                self.rig.unkey(playing['token'])
            except BaseException as exc:
                failures.append(exc)
                cancelled = True
        self.self_until = time.monotonic() + (.12 if self.test else .5)
        output, self.output = self.output, None
        self.playing = None
        if output is not None:
            try:
                output.abort() if cancelled else output.stop()
            except BaseException as exc:
                failures.append(exc)
                cancelled = True
            finally:
                try:
                    output.close()
                except BaseException as exc:
                    failures.append(exc)
                    cancelled = True
        playing['cancelled'] = cancelled
        draft = playing['draft']
        self.event('preview' if playing['preview'] else 'sent',
            started_utc=playing['started'], ended_utc=utc(), cancelled=cancelled,
            audio_status=playing.get('audio_status', ''), script=draft['text'],
            cleanup_errors=[str(exc) for exc in failures],
            transmission_mark=draft.get('tx_mark'),
            wav=draft['wav'], test=self.test, key_token=asdict(playing['token']) if playing['token'] else None)
        if not cancelled and not playing['preview'] and not playing.get('audio_status'):
            self.contact.sent(draft['kind'])
            self.tx_scripts.append(draft['text'])
            if draft['kind'] != 'cq':
                self.history.append(dict(role='station', text=self.codec.dialogue_text(
                    draft.get('source_text', draft['text']), self.contact.own_call, self.contact.partner)))
            if draft['kind'] == 'close':
                self.log_contact()
            # Keep the exact approved CQ ready for another Enter while listening.
            # A heard partner over or edit cancels it in the usual way. The
            # automatic computer pilot owns its own repeat scheduling.
            self.draft = draft if draft['kind'] == 'cq' and self.simulation is None else None
        if failures:
            for extra in failures[1:]:
                failures[0].add_note(str(extra))
            raise failures[0]
        return playing

    def replay_heard(self):
        if not self.latest or self.playing is not None:
            return
        audio, rate = read_wav(self.latest['wav'])
        # Preview through the local speakers; it never uses the rig codec by default.
        previous = self.draft
        self.draft = dict(kind='replay', text='replay heard over', wav=self.latest['wav'],
                          audio=resample(audio, rate, RATE), duration_s=len(audio) / rate)
        try:
            self.play(preview=True)
        finally:
            self.draft = previous

    def log_contact(self):
        heard = self.contact.heard
        record = dict(call=self.contact.partner, report_sent=self.contact.report_sent,
            report_received=heard.report if heard else '', name=heard.name if heard else '',
            qth=heard.qth if heard else '', started_utc=self.contact_started, ended_utc=utc(),
            frequency_hz=self.dial_hz, mode='CW' if self.waveform.name == 'cw' else 'SSB',
            submode=None if self.waveform.name == 'cw' else 'LSB' if self.dial_hz and self.dial_hz < 10_000_000 else 'USB',
            test=self.test, edits=self.edits.copy())
        self.contacts.append(record)
        with (self.directory / 'contacts.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.stop.set()
        failures = []
        def clean(action):
            try:
                action()
                return True
            except BaseException as exc:
                failures.append(exc)
                return False
        clean(lambda: self.finish_playback(cancelled=True))
        if self.rig is not None:
            if clean(self.rig.close):
                self.rig = None
            else:
                clean(lambda: self.rig.panic_unkey('voice rig close failed'))
        if self.capture is not None:
            clean(self.capture.abort)
            clean(self.capture.close)
        clean(self.cancel_draft)
        clean(lambda: self.executor.shutdown(wait=False, cancel_futures=True))
        if self.speech is not None:
            clean(self.speech.close)
        if self.language is not None:
            clean(self.language.close)
        if self.thread is not None:
            clean(lambda: self.thread.join(timeout=2))
        if failures:
            for extra in failures[1:]:
                failures[0].add_note(str(extra))
            raise failures[0]


def model_hashes(settings, waveform='voice'):
    files = [Path(settings.language_model).expanduser()]
    for directory in ((settings.recognition_model, settings.synthesis_model) if waveform == 'voice' else ()):
        files.extend(p for p in Path(directory).expanduser().rglob('*') if p.is_file())
    result = {}
    for path in files:
        if path.is_file():
            with path.open('rb') as stream:
                result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result
