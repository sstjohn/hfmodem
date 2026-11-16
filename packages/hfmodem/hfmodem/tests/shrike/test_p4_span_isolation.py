# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Span evidence must not rewrite already answered word registrations."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hfmodem.shrike.p4receiver import Emission, Kind, P4Receiver, Registered, Role
from hfmodem.tests.evidence import CAPTURES, WORKING, require_input
from hfmodem.tests.shrike.p4_chirp_deadline_support import FS, replay


@pytest.fixture(scope='module')
def k4msu_confirmation():
    arm = 'onair-1001-0411'
    require_input(CAPTURES / arm / 'stream.flac')
    manifest = Path(__file__).with_name('fixtures') / 'chirp-deadline' / arm / 'external.json'
    for row in json.loads(manifest.read_text())['files']:
        root = WORKING / arm if row['role'] == 'log' else CAPTURES / arm
        require_input(root / row['file'])
    observations = {}
    register = P4Receiver.register

    def capture(self, obs):
        if obs.kind == Kind.WORD and obs.word == 'X27' and obs.stream_sample in (1187653, 1187648):
            observations.setdefault(obs.provisional, obs)
        return register(self, obs)

    with patch.object(P4Receiver, 'register', capture):
        result = replay(arm, stop=30 * FS)
    head, full = observations[True], observations[False]
    assert (head.stream_sample, full.stream_sample) == (1187653, 1187648)
    assert not head.word_span_supported and not full.word_span_supported
    return head, full, result


@pytest.mark.parametrize('answer', ['accepted', 'unaccepted', 'absent'])
def test_recorded_x27_confirmation_preserves_only_an_answered_train(k4msu_confirmation, answer):
    head, full, _ = k4msu_confirmation
    rx = P4Receiver(FS, admitted=True, breakin=True)
    request, keyed = 1142623, 1202623
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, request, .758, train=True))
    first = rx.register(head)
    assert first.outcome == Registered.NEW and first.responds_to == request
    if answer == 'absent':
        rx.revoke_train()
    else:
        rx.note_emission(Emission(Kind.SHORT, Role.ISS, keyed, .758,
                                 current_slot=head.stream_sample,
                                 current_slot_accepted=answer == 'accepted'))
    assert rx.train_lease is None
    confirmed = rx.register(full)
    assert confirmed.outcome == Registered.CONFIRMED
    assert confirmed.responds_to == (request if answer == 'accepted' else None)
    assert confirmed.answered == (keyed if answer == 'accepted' else None)


def test_recorded_x27_extent_jitter_keeps_registration_and_key_samples(k4msu_confirmation):
    _, full, result = k4msu_confirmation
    rx = P4Receiver(FS, admitted=True)
    assert rx.register(full).outcome == Registered.NEW
    shorter = replace(full, extent_s=full.extent_s - .001,
                      stream_sample=full.stream_sample + 7, first_chip=full.first_chip + 7)
    assert rx.register(shorter).outcome == Registered.DUPLICATE
    assert rx._classification(full.stream_sample) is full
    assert [(e['at'], e.get('word')) for e in result['events'] if e['event'] == 'Keyed'] == [
        (1202623, None), (1291813, 'X27'), (1351808, 'X27'), (1411828, 'X27')]
