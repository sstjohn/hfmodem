# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CW idiom for the shared contact overs."""
import re
from hfmodem.contact.calls import checked
from hfmodem.contact.facts import facts_from as facts_from


def phonetic(call):
    return checked(call)


def spell_grid(grid):
    return grid.upper()


def dialogue_text(text, own_call, partner=''):
    text = re.sub(r'\bDE\s+' + re.escape(own_call) + r'\s*(?:<SK>|<KN>|SK|KN|BK|K)?$', '', text, flags=re.I)
    for call in (own_call, partner):
        if call:
            text = re.sub(r'\b' + re.escape(call) + r'\b', '', text, flags=re.I)
    return text.strip(' ,')


def body(text):
    replacements = [
        (r'I have your call\.? Please go ahead with your question\.?', 'R UR CALL PSE GA'),
        (r'Please say again your last message\.?', 'PSE RPT MSG'),
        (r'Please say again your call\.?', 'PSE RPT CALL'),
        (r'Thanks for that\.? Is there anything else you would like to discuss\??', 'TNX ANYTHING ELSE'),
        (r'I am using ', 'RIG '), (r'My radio is ', 'RIG '),
        (r'My (?:location|grid) is ', 'QTH '), (r'My name is ', 'NAME '),
        (r'My antenna is ', 'ANT '), (r'My power is ', 'PWR '), (r'My weather is ', 'WX '),
        (r'How are conditions at your end\??', 'HW CPY'),
        (r'thanks for calling', 'TNX CALL'), (r'thanks for the contact', 'TU'),
        (r'seventy three', '73'), (r'not configured', 'NIL'),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    # Restrict to the Morse alphabet plus supported punctuation/prosigns.
    return re.sub(r'\s+', ' ', re.sub(r'[^A-Z0-9/ ?<>]', ' ', text.upper())).strip()


def identify(text, own_call):
    own_call = checked(own_call)
    ending = re.compile(r'\bDE\s+' + re.escape(own_call) + r'\s+(?:K|<KN>|<SK>)$')
    if ending.search(text.upper()):
        return text.upper()
    return f'{body(text)} DE {own_call} K'


def cq(own_call):
    return f'CQ CQ DE {checked(own_call)} K'


def greeting(own_call, partner, facts, report=''):
    parts = [checked(partner), 'R']
    if report:
        parts.extend(['UR', report])
    if facts.get('grid'):
        parts.append(facts['grid'].upper())
    parts.extend(['DE', checked(own_call), 'K'])
    return ' '.join(parts)


def exchange(own_call, partner, report, facts, heard=None, fields=('report', 'location')):
    if not re.fullmatch(r'[1-5][1-9](?:[1-9])?', report):
        raise ValueError('Choose a CW report explicitly, e.g. 579.')
    return greeting(own_call, partner, facts, report)


def close(own_call, partner):
    return f'{checked(partner)} TU 73 DE {checked(own_call)} <SK>'


def fill(own_call, field='call'):
    return identify('PSE RPT ' + ('CALL' if field == 'call' else 'MSG'), own_call)
