# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KO4HJO 80 m 11-17 05:21Z: the gateway's greeting broke into our completed
opening at 26.011 and stayed unread. We keyed one V27 at 26.931, then nothing
for sixteen slots while reading the gateway's V27 on that word's raster ten
times, and the arm ended "peer silent"."""

from hfmodem.tests.shrike import test_p4_irs_slot_fixes_1114 as tapes
from hfmodem.tests.shrike.test_p4_irs_slot_fixes_1114 import FS, keyed

CAPTURE = 'onair-20261117T052150789829Z-73bd799e-dff1-4610-9022-cbf0355a03ab'
SLOT = round(1.25 * FS)


def replay(monkeypatch):
    monkeypatch.setitem(tapes.CAPTURES, 'ko4hjo-1117', CAPTURE)
    return tapes.replay('ko4hjo-1117')


def test_unread_greeting_keys_every_slot_and_the_v27_run_takes_the_turn(monkeypatch):
    probe = replay(monkeypatch)
    # The gateway's last V27 is at 47.862; its answer slot is 48.184.
    after = keyed(probe, 26.9, 48.5)
    assert round(after[0]['at'] / FS, 3) == 26.934 and round(after[-1]['at'] / FS, 3) == 48.184
    assert all(b['at'] - a['at'] == SLOT for a, b in zip(after, after[1:]))
    turn = next(e for e in probe.link_events if e['event'] == 'turn')
    asks = [e for e in after if e['kind'] == 'word']
    assert len(asks) >= 2 and all((e['word'], e['purpose']) == ('V27', 'repeat-request')
                                  for e in asks)
    # With no mail queued, the turn goes straight back to the gateway.
    answer = after[len(asks)]
    assert answer['at'] > turn['at']
    assert (answer['kind'], answer['state'], answer['status']) == ('body', 'ISS_HANDBACK', 0x40)
    assert probe.reason != 'peer silent'
