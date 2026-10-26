# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Plausible QSO cover, independent of the operators' message stream.

Each choice is a complete conversational follow-up, with no invented station
or family facts. Model sampling can choose wording within this finite grammar.
"""
import re

TOPICS = {
    'antenna': (
        'what you like about your antenna setup', 'how you settled on your antenna',
        'whether you have any antenna changes in mind', 'what you would change about your antenna setup',
        'how you found room for your antenna', 'what you think about trying a different antenna',
        'how your antenna plans have worked out', 'what antenna ideas you have been thinking about',
    ),
    'amplifier': (
        'how you get along with your amplifier', 'whether you think an amplifier is worth the trouble',
        'what you like about running an amplifier', 'what you would look for in an amplifier',
        'whether an amplifier is on your list of plans', 'how you think about power and station improvements',
        'what amplifier ideas you have been considering', 'whether you prefer keeping the station simple',
    ),
    'home': (
        'how you fit radio time around things at home', 'what the family thinks of the radio hobby',
        'how you balance the hobby with everything else', 'whether anyone at home shares your radio interests',
        'how you make room for the hobby at home', 'what you do when the radio has to wait',
        'whether the family enjoys hearing about your contacts', 'how radio fits into your usual day',
    ),
    'weather': (
        'what the weather has been like at your end', 'whether the weather has changed your plans today',
        'how you work around the weather when doing antenna jobs', 'whether you prefer warm weather for antenna work',
        'what kind of weather you have been getting lately', 'how the weather affects your plans for the station',
        'whether you have had time to get outside today', 'what you do with the hobby on a rainy day',
    ),
    'appointments': (
        'how you fit radio time around appointments', 'whether you have any appointments to work around this week',
        'what you do to keep busy while waiting for an appointment', 'how you keep the hobby going during a busy week',
        'whether your schedule leaves much time for the radio', 'how you find time for a chat between errands',
        'whether you get much time for the hobby these days', 'what you look forward to after a busy day',
    ),
    'solar': (
        'how you think solar activity has been affecting the bands', 'what you have noticed about the bands lately',
        'whether you follow solar activity before getting on the radio', 'what you make of the talk about solar flares',
        'how you choose a band when conditions are changing', 'whether you prefer checking conditions or just listening',
        'what you think the bands might do next', 'how you keep track of changing band conditions',
    ),
    'customs': (
        'what you think of discussions about operating rules', 'how you think the usual operating conventions have changed',
        'whether you enjoy comparing notes about operating habits', 'what makes a pleasant contact in your view',
        'what you think about the way people use the bands', 'how you learned the usual operating customs',
        'whether you prefer short contacts or a longer conversation', 'what you think makes a good radio conversation',
    ),
}
HEADS = (
    'I would be interested to hear', 'It would be nice to hear', 'I always enjoy hearing',
    'I would like to hear', 'It is good to hear', 'I would enjoy hearing',
    'I am curious to hear', 'I would be glad to hear', 'I would appreciate hearing',
    'I would really like to hear', 'Perhaps you could tell me', 'You could always tell me',
)
TAILS = ('', ' when you have a moment', ' if you feel like comparing notes',
         ' when we get a chance', ' while we have time to talk', ' sometime during our chat')


def topic_from(text):
    for topic, pattern in (
        ('antenna', r'antenna|aerial|dipole|vertical|beam'),
        ('amplifier', r'amplifier|\bamp\b|power'),
        ('home', r'\bxyl\b|wife|family|home'),
        ('weather', r'weather|rain|snow|temperature'),
        ('appointments', r'doctor|appointment|errand|schedule'),
        ('solar', r'solar|flare|sunspot|conditions|propagation'),
        ('customs', r'rule|custom|convention|operating habit'),
    ):
        if re.search(pattern, text, re.I):
            return topic
    return 'antenna'


def candidates(topic='antenna', max_words=24):
    topic = topic if topic in TOPICS else topic_from(topic)
    return tuple(sorted({head + ' ' + clause + tail + '.' for head in HEADS
                         for clause in TOPICS[topic] for tail in TAILS
                         if 8 <= len((head + ' ' + clause + tail).split()) <= max_words}))


VOCABULARY = set(re.findall(r'\b[a-z]+\b', ' '.join(
    text.lower() for topic in TOPICS for text in candidates(topic))))
