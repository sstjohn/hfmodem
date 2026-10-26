# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""KQ4QWH's away greeting ACK loses only its first fixed preamble pair."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_responder_control_stream import station, feed
from .test_idle_poll import clock  # noqa: F401


def recorded(name):
    root = Path(__file__).with_name('fixtures') / 'kq4qwh-away-1025'
    entry = next(x for x in json.loads((root/'provenance.json').read_text())['slices']
                 if x['file'] == name+'.wav')
    path = root/entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    rate, x = wavfile.read(path)
    assert rate == M.FS
    return x.astype(float)/32768


def sending(long=False):
    h, io = station('W9SSJ')
    h.caller = 'KQ4QWH'
    h.chat = True
    h.turn = A._TURN_OURS
    h._link_high_speed = True
    h._start_compression('OFF')
    payload = bytes(range(256))*2 if long else b'<AWAY>\r\nhttps://hfmodem.ssj.lol'
    h.send(payload)
    return h, io, payload


@pytest.mark.parametrize('block', [960, 4096, 24000])
def test_recorded_ack_settles_greeting_and_next_request_gets_turn(block):
    h, io, payload = sending()
    feed(h, recorded('ack'), block)
    assert h.payload_ledger.confirmed == len(payload), io.msgs
    assert h._tx_pending is None
    feed(h, recorded('request'), block)
    assert h.turn == A._TURN_PEER
    np.testing.assert_array_equal(io.sent[-1], M.synth_tone_pairs(F.turn_grant(h.called, '500')))


@pytest.mark.parametrize('case', ['wrong-call', 'wrong-role', 'payload-erasure', 'tail-erasure',
                                  'two-head-erasures', 'intermediate', 'request-only'])
def test_erasure_recovery_never_settles_unproven_bytes(case):
    h, io, payload = sending(long=case == 'intermediate')
    if case == 'request-only':
        x = recorded('request')
    elif case == 'intermediate':
        assert not A._phy.over_is_last(h._tx_pending[0], h.caller)
        x = recorded('ack')
    else:
        pairs = F.control_bursts('N0ZZZ' if case == 'wrong-call' else h.caller, '500')[case == 'wrong-role']
        x = M.synth_tone_pairs(pairs)
        x[:M.HOP] = 0
        if case == 'payload-erasure': x[5*M.HOP:6*M.HOP] = 0
        if case == 'tail-erasure': x[10*M.HOP:] = 0
        if case == 'two-head-erasures': x[:2*M.HOP] = 0
        x = np.r_[np.zeros(11000), x, np.zeros(24000)]
    feed(h, x)
    assert h.payload_ledger.confirmed == 0 and h._tx_pending is not None, io.msgs


@pytest.mark.parametrize('frame', [0, 1])
@pytest.mark.parametrize('delay', [4800, 11000, 24000])
def test_single_erased_fixed_pair_in_either_final_ack(frame, delay):
    h, io, payload = sending()
    x = M.synth_tone_pairs(F.control_bursts(h.caller, '500', frame)[0])
    x[:M.HOP] = 0
    feed(h, np.r_[np.zeros(delay), x, np.zeros(24000)])
    assert h.payload_ledger.confirmed == len(payload), io.msgs
    assert h._tx_pending is None


@pytest.mark.parametrize('case', ['disconnected', 'peer-turn', 'no-pending'])
def test_recorded_ack_needs_outstanding_final_over(case):
    h, io, payload = sending()
    if case == 'disconnected': h.state = A.VaraState.DISCONNECTED
    if case == 'peer-turn': h.turn = A._TURN_PEER
    if case == 'no-pending': h._tx_pending = None
    feed(h, recorded('ack'))
    assert h.payload_ledger.confirmed == 0
