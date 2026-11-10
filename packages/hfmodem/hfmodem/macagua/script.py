# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Templates own CQ, exchange, close and identification."""
import re
import random
from hfmodem.core.identification import validate_call
from hfmodem.contact.facts import facts_from as facts_from

ALPHABET = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    'Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo Lima Mike '
    'November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu'.split()))
DIGITS = dict(zip('0123456789', 'zero one two three four five six seven eight nine'.split()))


def phonetic(call, mode='nato', rng=None):
    call = validate_call(call, 'voice callsign')
    if mode not in ('nato', 'alternate', 'varied'):
        raise ValueError('phonetics must be nato, alternate or varied')
    rng = rng or random
    words = []
    for c in call:
        word = ALPHABET.get(c, DIGITS.get(c, 'stroke'))
        alternative = {'9': 'niner', 'S': 'Sugar'}.get(c)
        if alternative and (mode == 'alternate' or mode == 'varied' and rng.choice((False, True))):
            word = alternative
        words.append(word)
    return ' '.join(words)


def call_pattern(call):
    words = [r'(?:nine|niner|9)' if c == '9' else r'(?:Sierra|Sugar)' if c == 'S'
             else r'Juliet(?:t|te)?' if c == 'J' else re.escape(ALPHABET.get(c, DIGITS.get(c, 'stroke')))
             for c in validate_call(call, 'voice callsign')]
    return r'(?<!\w)(?:' + re.escape(call) + '|' + r'[\s,]+'.join(words) + r')(?!\w)'


def vary_calls(text, calls, mode='nato', seed=None):
    if mode == 'nato':
        return text
    rng = random.Random(seed)
    for call in dict.fromkeys(c for c in calls if c):
        text = re.sub(call_pattern(call), lambda _: phonetic(call, mode, rng), text, flags=re.I)
    return text


def identify(text, own_call):
    from .phonetics import callsigns
    own_call = validate_call(own_call, 'voice callsign')
    ending = f'This is {phonetic(own_call)}. Over.'
    text = text.strip()
    if own_call in callsigns(text):
        return text
    # An operator's terminal Over remains the final word. Put a missing ID
    # before it, rather than producing "Over. This is ... Over."
    over = re.search(r'\bover[.!?\s]*$', text, flags=re.I)
    if over:
        text = text[:over.start()].rstrip(' ,')
        if text and text[-1] not in '.!?':
            text += '.'
    return f'{text} {ending}'.strip()


def dialogue_text(text, own_call, partner=''):
    """Model history carries conversation; the codec adds IDs afresh on every over."""
    text = re.sub(r'This is\s+' + call_pattern(own_call) + r'\.\s*Over\.\s*$', '', text, flags=re.I).strip()
    for call in (own_call, partner):
        if call:
            text = re.sub(r'\b(?:This is|From|DE)\s+' + call_pattern(call) + r'[\s,.;:]*', '', text, flags=re.I)
            text = re.sub(call_pattern(call), '', text, flags=re.I)
    return text.strip(' ,')


def cq(own_call):
    spoken = phonetic(own_call)
    return f'CQ, CQ, CQ. This is {spoken}, {spoken}, calling CQ and standing by. This is {spoken}. Over.'


def exchange(own_call, partner, report, facts, heard=None, fields=('report', 'location')):
    if not re.fullmatch(r'[1-5][1-9]', report):
        raise ValueError('Choose a voice signal report, e.g. 57; no report is assumed.')
    parts = [f'{phonetic(partner)}, thanks for the call.']
    if 'report' in fields:
        parts.append(f'You are {DIGITS[report[0]]} {DIGITS[report[1]]}.')
    if 'location' in fields:
        if facts.get('qth'):
            parts.append(f'My location is {facts["qth"]}.')
        elif facts.get('grid'):
            parts.append(f'My grid is {spell_grid(facts["grid"])}.')
    if (heard and heard.name or 'name' in fields) and facts.get('name'):
        parts.append(f'My name is {facts["name"]}.')
    return identify(' '.join(parts), own_call)


def spell_grid(grid):
    return ' '.join(ALPHABET.get(c, DIGITS.get(c, c)) for c in grid.upper())


def close(own_call, partner):
    return identify(f'{phonetic(partner)}, thanks for the contact. Seventy three.', own_call)


def fill(own_call, field='call'):
    return identify(f'Please say again your {field}.', own_call)


def greeting(own_call, partner, facts, report=''):
    parts = [f'{phonetic(partner)}, thanks for calling.']
    if report:
        if not re.fullmatch('[1-5][1-9]', report):
            raise ValueError('Choose the simulated signal report explicitly.')
        parts.append(f'You are {DIGITS[report[0]]} {DIGITS[report[1]]}.')
    if facts.get('qth'):
        parts.append(f'My location is {facts["qth"]}.')
    elif facts.get('grid'):
        parts.append(f'My grid is {spell_grid(facts["grid"])}.')
    parts.append('How are conditions at your end?')
    return identify(' '.join(parts), own_call)

