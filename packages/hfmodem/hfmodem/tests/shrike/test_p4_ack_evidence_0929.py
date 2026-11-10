# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path

import pytest

from hfmodem.shrike import session
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Kind

ROOT = Path(__file__).resolve().parents[5]
CAPTURE = ROOT/'captures/onair-0929-0923'
FS = 48000
C2 = b'(EN72QQ)\r{SFI = 95 On 2026-09-29 11:00 U'
C3 = b'TC}\r\n[WL2K-5.0-B2FWIHJM$]\r;PQ: 29277567\r'


def test_ws8eoc_0923_independent_counter_two_copies_advance_after_v27():
    path = CAPTURE/'stream.flac'
    report = CAPTURE/'p4-probe.json'
    if not path.exists() or not report.exists():
        pytest.skip('local WS8EOC 0923 recording absent')
    audio = session.load_wav(str(path), FS)
    cases = (
        (3947729, -3.191030398445935, 2, C2),
        (4007729, -3.091030398445932, 2, C2),
        (4067730, -3.2910303984459297, 3, C3),
    )
    for at, cfo, status, data in cases:
        receiver = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
        receiver.enable_hold()
        receiver.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
        receiver.peer_onset = at-60000
        receiver.acked_counter = status-1
        origin = at-round(.050*FS)
        obs = receiver._normal(audio[origin:at+round(.846*FS)], origin)
        assert obs is not None and obs.kind == Kind.NORMAL and obs.placed
        assert abs(obs.stream_sample-at) <= 48
        assert obs.frame is not None
        assert (obs.frame.status.byte, obs.frame.data) == (status, data)
        events = receiver.hold.events
        assert any(e.get('event') == 'normal-decode' and e.get('copies') == 1
                   and e.get('crc_valid') and e.get('reason') == 'accepted' for e in events)
        assert not any(e.get('event') == 'held-same' for e in events)
        assert len(receiver.hold.members) == 1

    recorded = json.loads(report.read_text())
    answers = [e for e in recorded['emissions'] if 3947729 < e['audio_start'] < 4067730]
    assert [(e['word'], e['audio_start']) for e in answers] == [('V27', 4051889)]
    assert answers[0]['keyed_at'] == 4051889 and answers[0]['refused'] is None
    assert 4051889-4007729 == round(.920*FS)
    assert abs(4067730-4051889-round(.330*FS)) <= 1
