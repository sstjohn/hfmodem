# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Computer-audio QSO decisions; this policy has no transmitter interface."""
from dataclasses import dataclass
import math
import re


def says_goodbye(text):
    # Distinguish a sign-off from discussing the expression or an antenna length.
    if re.search(r"\b(?:what (?:does|is)|don't say|do not say|not saying|haven't said)\b", text, re.I):
        return False
    return bool(re.search(r'\b(?:73s?|7[ -]3|seventy[ -]three|seven[ -]three|goodbye|signing off|going qrt|let you go)\b'
                          r'(?!\s*(?:feet|foot|meters|metres|watts|mhz|khz))', text, re.I))


def asks_question(text):
    return '?' in text or bool(re.search(
        r'\b(?:what|where|when|why|how|which|could you|can you|do you|are you)\b', text, re.I))


@dataclass(frozen=True)
class TestOptions:
    report: str = ''
    partner: str = ''
    max_turns: int = 8
    reply_timeout_s: float = 30.
    cq_interval_s: float = 30.
    next_cq_s: float = 5.

    def __post_init__(self):
        if self.report and not re.fullmatch('[1-5][1-9](?:[1-9])?', self.report):
            raise ValueError('--report must be an RS or RST report, e.g. 57 or 579')
        if self.max_turns < 1:
            raise ValueError('--max-turns must be positive')
        for name in ('reply_timeout_s', 'cq_interval_s', 'next_cq_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


def decision(contact, over, text, replies, options):
    """Do not depend on a tiny model's guesses about contact endings."""
    if says_goodbye(text):
        return 'close' if contact.partner else 'wait'
    if not contact.partner:
        if over.confidence.get('de', 1.) < .65 or not over.de or any('callsign' in reason for reason in over.review):
            return 'fill'
        return 'accept'
    if re.fullmatch(r'\s*(?:over|qrz|bk|kn|k)[.!?\s]*', text, re.I):
        return 'ignore'
    if replies >= options.max_turns:
        return 'close'
    remaining = text.upper().replace(contact.own_call, '').replace(contact.partner, '')
    remaining = re.sub(r'\b(?:THIS IS|FROM|DE|OVER|QRZ|BK|KN|K)\b|[^A-Z0-9]', '', remaining)
    if not remaining or re.search(r'\b(?:given you my call|already.*my call|keep asking.*(?:my|your) call)\b', text, re.I):
        return 'ack_call'
    if over.confidence.get('de', 1.) < .65:
        return 'repeat_message'
    if over.de and over.de != contact.partner:
        return 'repeat_message'
    return 'talk'
