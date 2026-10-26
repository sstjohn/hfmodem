# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Experimental word-level logit watermark and binary packet codec.

No audio, model weights, network, device or transmitter interface lives here.
"""
from dataclasses import dataclass
import hashlib
import hmac
import math
import re
import struct
import binascii
from . import cover

# Versioned common-word carrier. Calls, figures, negation and operator facts
# are deliberately absent. The active subset is checked against the tokenizer.
WORDS = tuple(sorted(set('''
a about above across after afternoon again air all along also always am an and
another any anyone anything are around as at away back be beautiful because been
before being best better between big bit both bright bring busy by call calling
can care carry catch change chat chatting check clear close come comes coming
comfortable communication company complete contact conversation copy could course
day days dear different discuss doing done down during each early easy either
else end enjoy enjoyed enjoying enough even evening ever every everyone everything
excellent experience far feel feeling few fine first for forward free fresh friendly
from full fun get getting glad go goes going good great happy hard have having
hear heard hearing hello help here high hope hoping how i idea if in interesting
into is it its job join just keep kind know last late later learn learning leave
like listen listening little live long look looking lot lovely low make making
many may maybe me meet message might more morning most much my near need next
nice now of off often oh on once one only open opportunity or other our out over
own part people perhaps place pleasant please pleasure point possible practice
pretty probably question quick quite rather reach ready real really receive
receiving reply right room round run said same say saying see seems send sending
share sharing short should simple since small so some someone something sometimes
soon sound sounds speak speaking special start stay still such sure take taking
talk talking tell than thank thanks that the their them then there these they
thing things think this those through time times to today together too touch
try trying turn understand until up us use useful using very wait waiting want
warm was way we welcome well were what when where which while who will wish with
wonder wonderful work working works world would yes yet you your yours yourself
able accept actual almost answer anywhere appreciate attention available begin
believe bring brought calm chance choice choose common continue convenient curious
decide detail direct especially familiar favorite finish focus following future
gentle greeting heard helpful important information keep listen local moment
natural normal occasion ordinary patient perfect personal positive prepare
reason recent remember repeat response return sense several smooth standard
support thought truly usual voice watch whenever wherever whole worth
'''.split())))
WORDS = tuple(sorted(set(WORDS) | cover.VOCABULARY))
VERSION = 'loaded-words-v1'


class GenerationFailure(ValueError):
    def __init__(self, symbol, attempts):
        super().__init__(f'Could not generate a usable block for symbol {symbol} in {len(attempts)} attempts')
        self.attempts = attempts


def carrier_text(transcript):
    """Remove the appended spoken ID without requiring exact ASR spelling."""
    matches = list(re.finditer(r'\bthis is\b', transcript, re.I))
    if matches and matches[-1].start() > 0:
        return transcript[:matches[-1].start()].strip()
    return transcript.strip()


@dataclass(frozen=True)
class Options:
    codebook: str = 'hfmodem-v1'
    symbol_bits: int = 2
    bias: float = 3.
    temperature: float = .7
    min_words: int = 6
    min_fraction: float = .65
    min_margin: int = 3
    max_words: int = 24
    attempts: int = 12
    cover_mode: str = 'qso'
    method: str = 'sentence'

    def __post_init__(self):
        if not self.codebook or len(self.codebook) > 128:
            raise ValueError('codebook must contain 1–128 characters')
        if self.symbol_bits not in (1, 2, 4):
            raise ValueError('symbol_bits must be 1, 2 or 4')
        if self.cover_mode not in ('qso', 'free'):
            raise ValueError('cover_mode must be qso or free')
        if self.method not in ('sentence', 'logit') or (self.method == 'sentence' and self.cover_mode != 'qso'):
            raise ValueError('method must be sentence with qso cover, or logit')
        if not math.isfinite(self.bias) or not 0 <= self.bias <= 8:
            raise ValueError('bias must be finite and between 0 and 8')
        if not math.isfinite(self.temperature) or not .1 <= self.temperature <= 2:
            raise ValueError('temperature must be between .1 and 2')
        if not 0 < self.min_fraction <= 1 or not math.isfinite(self.min_fraction):
            raise ValueError('min_fraction must be between 0 and 1')
        if not 1 <= self.min_words <= self.max_words <= 40 or self.max_words < 8 or self.min_margin < 1 or not 1 <= self.attempts <= 100:
            raise ValueError('invalid evidence, word or attempt limit')


class Codebook:
    def __init__(self, options=Options(), vocabulary=WORDS, key=None):
        self.options = options
        self.words = tuple(sorted(set(vocabulary)))
        if len(self.words) < 32 or any(not re.fullmatch('[a-z]+', w) for w in self.words):
            raise ValueError('codebook needs at least 32 lowercase ASCII words')
        self.size = 1 << options.symbol_bits
        if key is not None and (not isinstance(key, bytes) or len(key) != 32):
            raise ValueError('watermark key must be exactly 32 raw bytes')
        self.keyed = key is not None
        # Overlapping balanced patterns retain half the word choices per symbol,
        # even for a 16-symbol alphabet. The receiver compares all patterns.
        self.patterns = []
        for symbol in range(self.size):
            def rank(word):
                raw = f'{VERSION}\0{options.codebook}\0{symbol}\0{word}'.encode()
                return hmac.digest(key, raw, 'sha256') if key is not None else hashlib.sha256(raw).digest()
            ordered = sorted(self.words, key=rank)
            self.patterns.append(frozenset(ordered[:len(ordered) // 2]))
        self.digest = hashlib.sha256('\n'.join(self.words).encode()).hexdigest()

    def score(self, text):
        words = sorted(set(re.findall(r'\b[a-z]+\b', text.lower())) & set(self.words))
        counts = [len(set(words) & pattern) for pattern in self.patterns]
        ranked = sorted(range(self.size), key=lambda n: (-counts[n], n))
        best, second = ranked[:2]
        votes = counts[best]
        fraction = votes / len(words) if words else 0.
        margin = votes - counts[second]
        accepted = (len(words) >= self.options.min_words and fraction >= self.options.min_fraction
                    and margin >= self.options.min_margin)
        return dict(symbol=best if accepted else None, best=best, counts=counts, evidence=len(words),
                    eligible_words=words, fraction=fraction, margin=margin, accepted=accepted)

    def combine(self, observations):
        counts = [sum(o['counts'][i] for o in observations) for i in range(self.size)]
        ranked = sorted(range(self.size), key=lambda n: (-counts[n], n))
        best, second = ranked[:2]
        total, votes = sum(o['evidence'] for o in observations), counts[best]
        fraction = votes / total if total else 0.
        margin = votes - counts[second]
        accepted = (total >= self.options.min_words and fraction >= self.options.min_fraction
                    and margin >= self.options.min_margin)
        return dict(symbol=best if accepted else None, best=best, counts=counts, fraction=fraction,
                    margin=margin, evidence=total, accepted=accepted, observations=len(observations))


def normalized(text):
    text = text.lower().replace('’', "'")
    for short, full in (("i'd", 'i would'), ("i'm", 'i am'), ("it's", 'it is'), ("you've", 'you have')):
        text = text.replace(short, full)
    return ' '.join(re.findall(r'[a-z]+', text))


class SentenceCodebook:
    """Implicit symbols in choices among plausible, meaning-preserving wording.

    This is an encoding, not an encryption or authorship scheme. Ordinary
    unmarked sentences can also have symbol values; packet validity is separate.
    """
    def __init__(self, options, vocabulary=WORDS, key=None):
        self.options, self.size = options, 1 << options.symbol_bits
        if key is not None and (not isinstance(key, bytes) or len(key) != 32):
            raise ValueError('watermark key must be exactly 32 raw bytes')
        self.keyed = key is not None
        self.words = tuple(sorted(set(vocabulary)))
        self.phrases = tuple(sorted({s for topic in cover.TOPICS for s in cover.candidates(topic, options.max_words)}))
        self.digest = hashlib.sha256('\n'.join(self.phrases).encode()).hexdigest()
        self.mapping = {}
        for phrase in self.phrases:
            text = normalized(phrase)
            raw = f'{VERSION}\0sentence\0{options.codebook}\0{text}'.encode()
            digest = hmac.digest(key, raw, 'sha256') if key is not None else hashlib.sha256(raw).digest()
            self.mapping[text] = digest[0] % self.size

    def score(self, text):
        symbol = self.mapping.get(normalized(text))
        counts = [int(symbol == i) for i in range(self.size)]
        return dict(symbol=symbol, best=symbol, counts=counts, evidence=int(symbol is not None),
                    accepted=symbol is not None, fraction=1. if symbol is not None else 0.,
                    margin=int(symbol is not None), eligible_words=[],
                    presence_detection=False)

    def combine(self, observations):
        counts = [sum(o['counts'][i] for o in observations) for i in range(self.size)]
        ranked = sorted(range(self.size), key=lambda n: (-counts[n], n))
        best, second = ranked[:2]
        total, votes = sum(counts), counts[best]
        fraction = votes / total if total else 0.
        accepted = votes > counts[second] and fraction >= self.options.min_fraction
        return dict(symbol=best if accepted else None, best=best, counts=counts, evidence=total,
                    fraction=fraction, margin=votes-counts[second], accepted=accepted,
                    observations=len(observations), presence_detection=False)


def detector(options, vocabulary=WORDS, key=None):
    cls = SentenceCodebook if options.method == 'sentence' else Codebook
    return cls(options, vocabulary, key=key)


def pack(payload, sequence=0):
    if not isinstance(payload, bytes) or len(payload) > 255 or not 0 <= sequence <= 255:
        raise ValueError('packet payload must be bytes, at most 255 bytes; sequence is 0–255')
    body = bytes([0xD1, sequence, len(payload)]) + payload
    return body + struct.pack('>H', binascii.crc_hqx(body, 0xffff))


def unpack(frame):
    if len(frame) < 5 or frame[0] != 0xD1 or len(frame) != frame[2] + 5:
        raise ValueError('invalid packet framing')
    if binascii.crc_hqx(frame[:-2], 0xffff) != int.from_bytes(frame[-2:], 'big'):
        raise ValueError('packet checksum mismatch')
    return frame[1], frame[3:-2]


def symbols(frame, width):
    if width not in (1, 2, 4):
        raise ValueError('symbol width must divide eight and be 1, 2 or 4')
    mask = (1 << width) - 1
    return [(byte >> shift) & mask for byte in frame for shift in range(8 - width, -1, -width)]


def frame_from_symbols(values, width):
    if width not in (1, 2, 4) or len(values) % (8 // width):
        raise ValueError('incomplete symbol frame')
    frame = bytearray()
    for offset in range(0, len(values), 8 // width):
        byte = 0
        for value in values[offset:offset + 8 // width]:
            if value is None or not 0 <= value < (1 << width):
                raise ValueError('erased or invalid symbol')
            byte = (byte << width) | value
        frame.append(byte)
    return bytes(frame)


class Encoder:
    """Model adapter: bias real token logits; detector needs only shared words."""
    def __init__(self, language, options=Options(), key=None):
        self.language, self.options = language, options
        # Two spellings, checked independently. Never bias subword fragments.
        token_words = {}
        for word in WORDS:
            for spelling in (' ' + word, ' ' + word.capitalize()):
                result = language.request('/tokenize', dict(content=spelling, add_special=False, with_pieces=True))
                tokens = result['tokens']
                if len(tokens) == 1 and isinstance(tokens[0], dict) and tokens[0]['piece'] == spelling:
                    token_words[tokens[0]['id']] = word
        self.tokens = token_words
        self.codebook = detector(options, token_words.values(), key=key)

    def generate(self, symbol, topic='A friendly conversation about enjoying amateur radio.', seed=1, marked=True):
        if not 0 <= symbol < self.codebook.size:
            raise ValueError('symbol outside codebook alphabet')
        bias = {str(token): self.options.bias for token, word in self.tokens.items()
                if word in self.codebook.patterns[symbol]} if marked and self.options.method == 'logit' else {}
        attempts = []
        choices = cover.candidates(topic, self.options.max_words) if self.options.cover_mode == 'qso' else None
        if choices and marked and self.options.method == 'sentence':
            # Payload constrains only the sampler's legal completions. It never
            # appears in the cover prompt or a separate ciphertext representation.
            choices = tuple(s for s in choices if self.codebook.score(s)['symbol'] == symbol)
            if not choices:
                raise GenerationFailure(symbol, [dict(error='No plausible cover for this symbol and topic')])
        reply_schema = dict(type='string', enum=list(choices)) if choices else dict(type='string', pattern=
            r"^[A-Za-z]+(?:'[A-Za-z]+)?(?: [A-Za-z]+(?:'[A-Za-z]+)?){7,"
            + str(self.options.max_words - 1) + r'}[.!]$')
        for attempt in range(self.options.attempts):
            try:
                reply = self.language.complete(
                'Choose a natural follow-up for a relaxed amateur radio conversation. '
                f'in ONE sentence under {self.options.max_words} words. No callsigns, names, numbers, '
                'station facts, signal reports, greetings or sign-offs. '
                'Follow the topic without inventing personal facts. '
                'Do not claim anything about actual reception, conditions or equipment. '
                'Use a varied vocabulary, avoid repeating words. Return JSON with only reply.', topic,
                dict(type='object', additionalProperties=False, required=['reply'],
                     properties=dict(reply=reply_schema)),
                sampling=dict(temperature=self.options.temperature, seed=seed + attempt,
                              logit_bias=bias, max_tokens=100))['reply'].strip()
            except RuntimeError as exc:
                attempts.append(dict(error=str(exc), seed=seed + attempt))
                continue
            score = self.codebook.score(reply)
            safe = (bool(reply) and 8 <= len(reply.split()) <= self.options.max_words
                    and not re.search(r'\d|\b(?:over|qrz|goodbye|anyway)\b', reply, re.I))
            attempts.append(dict(text=reply, score=score, safe=safe, seed=seed + attempt))
            if safe and (not marked or score['symbol'] == symbol):
                return dict(text=reply, score=score, attempts=attempts, marked=marked)
        raise GenerationFailure(symbol, attempts)
