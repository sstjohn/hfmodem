# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One partner over, with disagreements carried to the operator."""
from dataclasses import asdict, dataclass, field
import re
from .simulation import asks_question, says_goodbye

_DIGITS = dict(zip('zero one two three four five six seven eight nine niner'.split(), '01234567899'))


def literal_fields(text):
    """Unambiguous exchange phrases supply an independent check on model fields."""
    found = {}
    number = r'(?:[0-9]+|one|two|three|four|five|six|seven|eight|nine|niner)'
    report = re.search(r"\b(?:you are|you're|your report is|signal report(?: is)?)\s+(" + number +
                       r')(?:[\s,-]+(' + number + r'))?', text, re.I)
    if report:
        parts = [_DIGITS.get(p.lower(), p) for p in report.groups() if p]
        value = ''.join(parts)
        if re.fullmatch('[1-5][1-9]', value):
            found['report'] = value
    for key, phrase in (('name', r'(?:my name is|the name here is|name here is)'),
                        ('qth', r'(?:my location is|my qth is|the qth is|located in)')):
        match = re.search(r'\b' + phrase + r'\s+([^.!?;]+)', text, re.I)
        if match:
            found[key] = re.split(r',|\b(?:and|over)\b', match.group(1), maxsplit=1, flags=re.I)[0].strip()
    return found


@dataclass(frozen=True)
class Over:
    to: str = ""
    de: str = ""
    report: str = ""
    name: str = ""
    qth: str = ""
    free: str = ""
    end: str = "none"
    confidence: dict = field(default_factory=dict)
    review: tuple[str, ...] = ()

    def record(self):
        return asdict(self)


def reconcile(extracted, candidates, own_call, word_probability=1.0, text=''):
    """The language model never supplies a callsign to log or speak.

    Even agreement is a candidate until the operator confirms the first copy.
    """
    candidates = list(dict.fromkeys(c for c in candidates if c != own_call))
    review = []
    model_call = str(extracted.get('de', '')).upper().strip()
    de = model_call if model_call in candidates else candidates[0] if len(candidates) == 1 else ''
    if model_call and model_call != de:
        review.append(f'callsign disagreement: normalizer {candidates}, model {model_call}')
    if len(candidates) > 1:
        review.append(f'multiple callsigns: {", ".join(candidates)}')
    if word_probability < 0.65:
        review.append('low recognition confidence; replay or edit')
    fields = {key: str(extracted.get(key, '') or '').strip()[:500]
              for key in ('report', 'name', 'qth', 'free')}
    literals = literal_fields(text)
    for key, literal in literals.items():
        if fields[key].casefold() != literal.casefold():
            review.append(f'{key} disagreement: transcript {literal!r}, model {fields[key]!r}')
        fields[key] = literal
    if text:
        # A very small model can fill absent fields from the prompt's example.
        # Those fields have no independent copy and must remain unconfirmed.
        for key in ('report', 'name', 'qth'):
            if fields[key] and key not in literals:
                review.append(f'{key} has no independent transcript copy; confirm or edit')
                fields[key] = ''
    end = extracted.get('end', 'none')
    if end not in ('over', 'clear', 'standing_by', 'none'):
        end = 'none'
    if text:
        if says_goodbye(text):
            end = 'clear'
        else:
            # A tiny model's omitted conversation must not turn a clear
            # question into a terse exchange and an unsolicited sign-off.
            if asks_question(text) and not fields['free']:
                fields['free'] = re.sub(r'\s*\bover\b[.!?\s]*$', '', text, flags=re.I).strip()
            if end == 'clear':
                end = 'over' if re.search(r'\bover\b', text, re.I) else 'none'
    return Over(to=own_call if own_call in str(extracted.get('to', '') or '').upper() else '',
                de=de, **fields, end=end, review=tuple(review),
                confidence={key: word_probability for key in ('de', *fields)})
