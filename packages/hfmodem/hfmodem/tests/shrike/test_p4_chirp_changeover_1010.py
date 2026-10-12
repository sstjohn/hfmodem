# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace

from hfmodem.shrike import p4frame
from hfmodem.shrike.p4observation import Kind, Observation, Registered, RegisteredObservation
from hfmodem.tests.shrike.test_p4turn import FS, machine

B = [(0x61, 1128765, '524d53205472696d6f646520312e342e342e300d0a57'),
     (0x62, 1308787, '3953534a2068617320313035206461696c79206d696e'),
     (0xe3, 1488710, '010101757465732072656d61696e696e672077697468'),
     (0x60, 1668646, '20575338454f432028454e37325151290d5b574c324b'),
     (0x61, 1848678, '2d352e302d4232465749484a4d245d0d3b50513a2030'),
     (0x62, 2028706, '323038393639370d434d532076696120575338454f43'),
     (0x03, 2208752, '203e0d1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e')]


def chirp(at, status, data):
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                 data_bytes=len(data))
    assert frame is not None and frame.data == data
    return RegisteredObservation(Observation(Kind.CHIRP, FS, 0, at, at, 3.28, 0, .9,
                                             frame=frame), Registered.NEW)


def test_recorded_chirp_counter_words_ignore_protocol_history():
    m, _ = machine(role='irs', data='body', ladder='none', prekey_lead=.025)
    m.first_body = False
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    words = []
    for status, at, data in [*B, (0x03, 2388808, B[-1][2])]:
        m.observe(chirp(at, status, bytes.fromhex(data)), now=at+158400)
        answer = m.decide(at+163200-m.lead)
        assert answer.stream_sample == at+163200
        words.append(m.keyed().word)
        if at == 2208752:
            once = list(delivered)
    assert words == ['X27', 'V27', 'X27', 'V27', 'X27', 'V27', 'X27', 'X27']
    assert delivered == once
