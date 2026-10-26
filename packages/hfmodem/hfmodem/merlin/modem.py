# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CW audio codec using the existing Morse alphabet and narrow-track reader."""
import math
import re
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from hfmodem.core import cwid
from hfmodem.core.levels import at_drive
from hfmodem.monitor.cw import CwReader
from hfmodem.contact.audio import RATE, resample
from hfmodem.contact.calls import CALL
from hfmodem.contact.frame import Over
from . import script

PROSIGNS = {'<SK>': '...-.-', '<KN>': '-.--.', '<AR>': '.-.-.'}


def render(text, wpm=20., tone=700., rate=RATE):
    dit = cwid.dit_seconds(wpm)
    alphabet = dict(cwid.MORSE, **{'?': '..--..'})
    pieces = []
    for word_index, word in enumerate(text.upper().split()):
        if word_index:
            pieces.append(np.zeros(round(7 * dit * rate)))
        tokens = [word] if word in PROSIGNS else list(word)
        for index, token in enumerate(tokens):
            if index:
                pieces.append(np.zeros(round(3 * dit * rate)))
            code = PROSIGNS.get(token) or alphabet.get(token)
            if code is None:
                raise ValueError(f'No CW code for {token!r}')
            for element_index, element in enumerate(code):
                if element_index:
                    pieces.append(np.zeros(round(dit * rate)))
                count = round((3 if element == '-' else 1) * dit * rate)
                envelope = np.ones(count)
                ramp = max(1, round(min(cwid.RAMP_S, dit / 4) * rate))
                rise = .5 - .5 * np.cos(np.pi * np.arange(ramp) / ramp)
                envelope[:ramp], envelope[-ramp:] = rise, rise[::-1]
                pieces.append(envelope)
    if not pieces:
        raise ValueError('CW script is empty')
    envelope = np.concatenate([np.zeros(round(.6 * rate)), *pieces, np.zeros(round(.4 * rate))])
    return (.5 * envelope * np.sin(2 * np.pi * tone * np.arange(len(envelope)) / rate)).astype(np.float32)


class ReceiveChain:
    def __init__(self, tone):
        self.sos = butter(4, [tone - 250, tone + 250], 'bandpass', fs=RATE, output='sos')
        self.state = np.zeros((len(self.sos), 2))

    def block(self, audio):
        audio, self.state = sosfilt(self.sos, audio, zi=self.state)
        return resample(audio, RATE, 16000)


class CwIO:
    def __init__(self, wpm, tone):
        self.wpm, self.tone = wpm, tone
        self.ready = dict(rss_gb=0.)

    def request(self, operation, **values):
        start = time.monotonic()
        if operation == 'synthesize':
            audio = render(values['text'], self.wpm, self.tone)
            wavfile.write(values['path'], RATE, audio)
            result = dict(path=values['path'], rate=RATE, duration_s=len(audio) / RATE)
        elif operation == 'transcribe':
            rate, audio = wavfile.read(values['path'])
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(float) / 32768
            reader = CwReader(rate, self.tone)
            records = []
            for offset in range(0, len(audio), 4096):
                records.extend(reader.push(audio[offset:offset + 4096]))
            records.extend(reader.flush())
            words = [dict(word=r['text'], start=r['t'], probability=.98 if r['grade'] == 'confirmed' else .8,
                          detail=r['detail'], grade=r['grade']) for r in records]
            result = dict(text=' '.join(r['text'] for r in records), words=words,
                          probability=float(np.mean([w['probability'] for w in words])) if words else 0.)
        elif operation == 'vad':
            # Narrow carrier activity, not a speech VAD. Compare neighbouring
            # bins so the gate follows noise level rather than a fixed amplitude.
            audio = np.asarray(values['audio'])
            t = np.arange(len(audio)) / 16000
            powers = [abs(np.mean(audio * np.exp(-2j * np.pi * f * t))) ** 2
                      for f in (self.tone, self.tone - 180, self.tone + 180)]
            result = dict(probability=1. if powers[0] > 10 * max(powers[1], powers[2], 1e-12) else 0.)
        elif operation == 'vad_reset':
            result = dict(ok=True)
        else:
            raise ValueError(f'Unknown CW operation {operation}')
        result['elapsed_s'] = time.monotonic() - start
        return result

    def close(self):
        pass


class CwWaveform:
    name = 'cw'
    designator = '150HJ2A'
    codec = script

    def __init__(self, wpm=20., tone=700.):
        if not math.isfinite(wpm) or not 8 <= wpm <= 20:
            raise ValueError('CW speed must be 8–20 WPM for this initial keyer')
        if not math.isfinite(tone) or not 300 < tone < 2500:
            raise ValueError('CW tone must be between 300 and 2500 Hz')
        self.wpm, self.tone = wpm, tone

    def open_modem(self, settings, directory):
        return CwIO(self.wpm, self.tone)

    def receive_chain(self):
        return ReceiveChain(self.tone)

    def condition(self, audio, rate, drive):
        audio = resample(audio, rate, RATE)
        if not np.isfinite(audio).all() or not audio.size:
            raise ValueError('Invalid CW audio')
        return at_drive(audio, drive)

    def callsigns(self, text):
        return [word for word in text.upper().split() if CALL.fullmatch(word)]

    def parse(self, language, result, candidates, own_call):
        text = result['text'].upper()
        tokens = text.split()
        partner = ''
        if 'DE' in tokens:
            at = tokens.index('DE') + 1
            if at < len(tokens) and tokens[at] in candidates and tokens[at] != own_call:
                partner = tokens[at]
        others = list(dict.fromkeys(c for c in candidates if c != own_call))
        if not partner and len(others) == 1:
            partner = others[0]
        fields = {}
        for field, label in (('report', r'UR|RST|RPT'), ('name', 'NAME'), ('qth', r'QTH|GRID')):
            found = re.search(r'\b(?:' + label + r')\s+([A-Z0-9/]+)', text)
            if found:
                value = found[1].replace('N', '9') if field == 'report' else found[1]
                if field != 'report' or re.fullmatch(r'[1-5][1-9](?:[1-9])?', value):
                    fields[field] = value
        if 'qth' not in fields:
            grid = re.search(r'\b[A-R]{2}[0-9]{2}(?:[A-X]{2})?\b', text)
            if grid:
                fields['qth'] = grid[0]
        end = 'clear' if '<SK>' in tokens or 'SK' in tokens or '73' in tokens else 'over' if any(t in tokens for t in ('K', '<KN>', 'BK')) else 'none'
        over = Over(de=partner, to=own_call if own_call in tokens else '', **fields, end=end,
                    confidence=dict(de=result['probability']),
                    review=('uncertain CW copy',) if result['probability'] < .65 else ())
        return over.record(), over

    def silence_s(self, default):
        return max(default, 9 * 1.2 / self.wpm)

    def emission_edges(self, dial):
        lo, hi = self.tone - cwid.BANDWIDTH_HZ / 2, self.tone + cwid.BANDWIDTH_HZ / 2
        return (-hi, -lo) if dial < 10_000_000 else (lo, hi)
