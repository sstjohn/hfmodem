# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A valid short retry must escape VE3KPG's repeated-CS6 negotiation loop."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scipy.io import wavfile

from hfmodem.shrike import arq, p3acquire, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURES = Path(__file__).parent / 'fixtures' / 've3kpg-cs6-0920'
if not (FIXTURES / "metadata.json").is_file():
    pytest.skip("VE3KPG off-air fixtures are not included in this distribution",
                allow_module_level=True)


def test_recorded_short_retry_gets_ack_and_delivers_once():
    s = _Session(role=arq.IRS)
    s.host.arq.cfg.long_cycle = True
    s.host.arq.cfg.repeat_gear = 0  # The flown mail profile.
    delivered = []
    s.host.deliver = delivered.append
    rows = json.loads((FIXTURES / 'metadata.json').read_text())
    events = []
    for row in rows:
        fs, raw = wavfile.read(FIXTURES / row['file'])
        assert fs == 48000
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['pcm_sha256']
        audio = p3acquire.compensate(raw.astype(float) / 32768, row['offset_hz'])
        ev = rxfront.SyncedRx().sl2_packet_at(audio, row['row0_local'])
        assert ev is not None and ev.cycle_long is False
        assert ev.packet == (2, 0x20, b'1\r\nW9SSJ has 107 daily ', True)
        events.append(ev)
        s.host.on_rx_event(ev)
    assert [ev.carrier_swapped for ev in events] == [False, True]
    assert s.host.peer.sent == [('cs', arq.CS_CYCLE_TOG), ('cs', arq.CS_ACK)]
    assert not s.host.arq.cycle_long
    assert s.host.arq.cycle_request is None
    assert not s.host.arq.cycle_command_emitted
    # Repeating as long as the observed contact must never redeliver or renew
    # the request. The physical carrier alternation is already covered above.
    for _ in range(26):
        s.host.on_rx_event(events[-1])
    assert s.host.peer.sent[1:] == [('cs', arq.CS_ACK)] * 27
    assert b''.join(delivered) == events[0].packet[2]
    assert s.host.arq.rx_progress == 1
    # A hypothetical next packet verifies forward progress, not a claim that
    # the real peer sent these bytes after the operator stopped the session.
    next_packet = replace(events[-1], packet=(2, 1, b'minutes', True))
    s.host.on_rx_event(next_packet)
    assert s.host.peer.sent[-1] == ('cs', arq.CS_REQUEST)  # odd-counter ACK
    assert b''.join(delivered) == events[0].packet[2] + b'minutes'
    assert s.host.arq.rx_progress == 2
