# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Experimental marks carried by LLM token probabilities, before synthesis.

The payload changes logit_bias only. Prompts and grammars do not encode it.
Detection accumulates distinct words; repetition alone adds no confidence.
Scores are experimental evidence, not authentication or calibrated proof.
"""
from dataclasses import dataclass
import hashlib
import hmac
import re
import threading

import numpy as np
from scipy.stats import binom

from .watermark import Encoder, Options, normalized
from . import cover

VERSION = 'tx-logits-v1'
LABELS = ('hfmodem', *(f'{n:04d}' for n in range(10000)))


@dataclass(frozen=True)
class Tag:
    mode: str = 'off'
    code: str = ''

    def __post_init__(self):
        if self.mode not in ('off', 'hfmodem', 'code'):
            raise ValueError('mark must be off, hfmodem or code')
        if (self.mode == 'code' and not re.fullmatch('[0-9]{4}', self.code)) or (self.mode != 'code' and self.code):
            raise ValueError('Choose exactly four ASCII digits, including leading zeros, or hfmodem')

    @property
    def label(self):
        return self.code if self.mode == 'code' else self.mode

    def record(self):
        return dict(mode=self.mode, code=self.code)


class LLMMark:
    def __init__(self, language, key=None):
        # Reuse the lab adapter's verified whole-word tokenizer mapping, rather
        # than biasing fragments or guessing token IDs for another model.
        adapter = Encoder(language, Options(method='logit', symbol_bits=1), key=key)
        self.language, self.tokens = language, adapter.tokens
        self.words = tuple(sorted(set(self.tokens.values())))
        self.columns = {word: i for i, word in enumerate(self.words)}
        self.digest = adapter.codebook.digest
        self.keyed = key is not None
        self.patterns = np.zeros((len(LABELS), len(self.words)), dtype=bool)
        for index, label in enumerate(LABELS):
            raw = f'{VERSION}\0{self.digest}\0{label}'.encode()
            seed = hmac.digest(key, raw, 'sha256') if key is not None else hashlib.sha256(raw).digest()
            ranks = np.frombuffer(hashlib.shake_256(seed).digest(4 * len(self.words)), dtype='>u4')
            self.patterns[index, np.argsort(ranks, kind='stable')[:len(self.words)//2]] = True
        self.streams = {}
        self.lock = threading.Lock()

    def bias(self, tag):
        if tag.mode == 'off':
            return {}
        pattern = self.patterns[LABELS.index(tag.label)]
        return {str(token): 3. for token, word in self.tokens.items() if pattern[self.columns[word]]}

    def generate(self, tag, topic='antenna', seed=1):
        for attempt in range(4):
            try:
                return self._generate(tag, topic, seed + 1000 * attempt)
            except (ValueError, RuntimeError):
                if attempt == 3:
                    raise

    def _generate(self, tag, topic, seed):
        # The grammar and prompt are identical for every code and the unmarked
        # control. The model supplies the actual words; no phrase lookup exists.
        topic = topic if topic in cover.TOPICS else cover.topic_from(topic)
        subject = cover.TOPICS[topic][seed % len(cover.TOPICS[topic])]
        reply = self.language.complete(
            'You are an amateur radio operator chatting with another operator. '
            'Ask one friendly question about their hobby or daily life. '
            'Use one sentence of eight to twenty four words. '
            'Ask about the supplied subject. Do not invent facts or repeat yourself. '
            'No callsigns, numbers, greetings, sign offs, Over, Anyway or BT. '
            'Return JSON with only reply.', 'Ask about ' + subject + '.',
            dict(type='object', additionalProperties=False, required=['reply'], properties=dict(
                reply=dict(type='string', pattern=r'^(?:How|What|Would|Could|Do|Have|Are)[A-Za-z ,\x27]{12,100}[?]$'))),
            sampling=dict(temperature=.7, seed=seed, max_tokens=180, logit_bias=self.bias(tag)))['reply'].strip()
        if not 4 <= len(reply.split()) <= 28 or re.search(
                r'\d|\b(?:over|anyway|bt|qrz|goodbye)\b', reply, re.I):
            raise ValueError(f'LLM returned an invalid mark cover sentence: {reply!r}')
        return dict(text=reply, layer='llm-logit-bias', version=VERSION, keyed=self.keyed)

    @staticmethod
    def split(text, waveform):
        delimiter = r'\bany\s*way\b' if waveform == 'voice' else r'\bBT\b'
        parts = re.split(delimiter, text, maxsplit=1, flags=re.I)
        if len(parts) != 2 or not 4 <= len(normalized(parts[0]).split()) <= 32:
            return None, text
        return parts[0], parts[1].lstrip(' ,.!')

    def reset(self, stream):
        with self.lock:
            self.streams.pop(stream, None)

    def observe(self, text, stream='partner'):
        eligible = set(normalized(text).split()) & self.columns.keys()
        with self.lock:
            window = self.streams.setdefault(stream, [])
            window.append(eligible)
            del window[:-64]
            words = set().union(*window)
            overs = len(window)
        columns = [self.columns[word] for word in words]
        scores = self.patterns[:, columns].sum(axis=1)
        ranked = np.argsort(-scores, kind='stable')
        best, second = (int(n) for n in ranked[:2])
        evidence, votes = len(columns), int(scores[best])
        fraction = votes / evidence if evidence else 0.
        margin = votes - int(scores[second])
        # Correct for searching 10001 labels. This nominal binomial statistic
        # assumes independent unbiased words; natural language violates that
        # assumption, so it must not be shown as a calibrated false-positive rate.
        nominal_tail = min(1., float(binom.sf(votes - 1, evidence, .5)) * len(LABELS)) if evidence else 1.
        accepted = evidence >= 40 and fraction >= .75 and margin >= 6 and nominal_tail <= .001
        label = LABELS[best]
        tag = Tag('hfmodem') if label == 'hfmodem' else Tag('code', label)
        return dict(mark=tag.record() if accepted else None, accepted=accepted, evidence=evidence,
                    overs=overs, fraction=fraction, margin=margin, nominal_tail=nominal_tail,
                    version=VERSION, keyed=self.keyed, interpretation='experimental; not authentication')
