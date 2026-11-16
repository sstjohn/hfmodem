# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A V27 turn starts on our short raster, independently of the peer's."""
import hashlib
import json

import pytest

from hfmodem.shrike import session
from hfmodem.shrike.p4receiver import Emission, Kind, P4Receiver, Registered, Role
from hfmodem.shrike.p4turn import State
from hfmodem.tests.evidence import RECORD, require_input
from hfmodem.tests.shrike.test_p4_chirp_changeover_1010 import chirp
from hfmodem.tests.shrike.test_p4_turn_recognition_0929 import idle_long
from hfmodem.tests.shrike.test_p4turn import FS, machine, word

CAPTURE = 'onair-20261027T050303283404Z-1192d336-ec33-4cad-86a7-b1ab245d203d'
REQUEST = 4591625
PAYLOAD = b'K4MSU regression\r'


@pytest.fixture(scope='module')
def recording():
    root = RECORD / 'operator-data' / 'captures' / CAPTURE
    for name, digest in (
        ('p4-probe.json', '5fc280ece955e2f71dbc5ab22a3b005fe5d14e8f866318887d74b40d1bb5f353'),
        ('stream.flac', '4f401e7be0e22af36444b50a4744afcdc71967cae171bdb3738388ee87b23ee7'),
    ):
        require_input(root / name)
        with (root / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    probe = json.loads((root / 'p4-probe.json').read_text())
    return session.load_wav(str(root / 'stream.flac'), FS), probe


def test_recorded_chirp_v27_first_body_uses_y27_short_raster(recording):
    audio, probe = recording
    last = next(r for r in reversed(probe['heard'])
                if r['kind'] == 'chirp' and r.get('status') is not None
                and r['onset_sample'] < REQUEST)
    assert (last['onset_sample'], last['status']) == (4428425, 0x42)
    assert bytes.fromhex(last['data']) == b'\x1e' * 22
    row, = [r for r in probe['heard'] if r.get('word') == 'V27'
            and r.get('responds_to') == REQUEST and r['outcome'] == 'confirmed']
    assert row['onset_sample'] == 4637205
    assert row['extent_s'] >= .240 and not row['weak'] and not row['unexpected']
    m, a = machine(role='irs', data='body', prekey_lead=.04,
                   normal_ack='parity', chirp_ack='parity')
    m.first_body = False
    a.on_host_data(PAYLOAD)
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=row['cfo_hz'])
    rx.enable_hold()
    reg = rx.register(chirp(last['onset_sample'], last['status'],
                            bytes.fromhex(last['data'])).observation)
    m.observe(reg, now=last['received_at_sample'])
    request = m.decide(REQUEST - m.lead)
    emitted, = [e for e in probe['emissions'] if e['audio_start'] == REQUEST]
    assert (request.kind, request.word, request.stream_sample) == (Kind.WORD, 'Y27', REQUEST)
    assert emitted['word'] == request.word and emitted['payload'] == ''
    m.keyed()
    assert m.peer_phase.kind == Kind.CHIRP and m.peer_watch is not None
    rx.note_emission(Emission(Kind.WORD, Role.ISS, REQUEST,
                             (emitted['audio_end'] - REQUEST) / FS))
    origin = row['onset_sample'] - round(.15 * FS)
    samples = audio[origin:row['onset_sample'] + round(.90 * FS)].copy()
    head = rx._word(samples, origin, provisional=True)
    assert head is not None and head.word == 'V27' and head.provisional
    early, = [r for r in probe['heard'] if r.get('word') == 'V27'
              and r.get('responds_to') == REQUEST and r['provisional']]
    m.observe(rx.register(head), now=early['received_at_sample'])
    assert not [e for e in m.link_events if e['event'] == 'turn']
    regs = rx.read_window(samples, origin, whole=True)
    full, = [r for r in regs if r.observation.word == 'V27' and not r.observation.provisional]
    assert full.outcome == Registered.CONFIRMED and full.responds_to == REQUEST
    assert abs(full.observation.stream_sample - row['onset_sample']) <= 48
    assert full.observation.extent_s >= .240 and not full.observation.weak
    m.observe(full, now=row['received_at_sample'])
    assert [e['event'] for e in m.link_events if e['event'] == 'turn'] == ['turn']
    p = m._next
    assert (p.kind, p.status, p.payload, p.state) == (Kind.BODY, 0x40, PAYLOAD, State.ISS_DATA)
    assert p.stream_sample == REQUEST + 2 * round(1.25 * FS) == 4711625
    assert p.stream_sample != 4771625
    assert m.peer_phase is None and m.peer_watch is None
    assert a.tx_acked_bytes == 0


@pytest.mark.parametrize('state', [State.IRS_BREAKIN, State.ISS_GRANT])
@pytest.mark.parametrize('delay,refused,slots', [(0, None, 1), (1, None, 2), (0, 60001, 2)])
def test_long_peer_v27_first_body_uses_short_raster(state, delay, refused, slots):
    m, a = machine(role='irs', data='body', prekey_lead=.04)
    m.first_body = False
    a.on_host_data(PAYLOAD)
    at = FS
    m.observe(idle_long(at), now=at + round(3.3 * FS))
    request = m.decide(at + round(3.4 * FS) - m.lead)
    assert request.kind == Kind.WORD and request.word == 'Y27'
    if state == State.ISS_GRANT:
        m._body(request.stream_sample, 0x40, state)
    m.keyed()
    assert m.peer_phase.kind == Kind.LONG and m.peer_watch is not None
    key = m.last.stream_sample
    now = key + round(1.25 * FS) - m.lead + delay
    m._refused_due = None if refused is None else key + refused
    m.observe(word('V27', key + round(.935 * FS), extent=.245,
                   responds_to=key, outcome=Registered.CONFIRMED), now=now)
    assert [e['event'] for e in m.link_events if e['event'] == 'turn'] == ['turn']
    p = m._next
    assert (p.kind, p.status, p.payload, p.state) == (Kind.BODY, 0x40, PAYLOAD, State.ISS_DATA)
    assert p.stream_sample == key + slots * round(1.25 * FS)
    assert m.peer_phase is None and m.peer_watch is None
