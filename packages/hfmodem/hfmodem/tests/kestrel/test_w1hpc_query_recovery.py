# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A strongly identified solicited NAK can lose two trailing tones to erasure."""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_bw500_query_recovery import _IO, clock as clock  # noqa: F401
from .test_chat_query_budget import reply

PAYLOAD = b'25 <QSYF>000014104250</QSYF>'


def pending(clock, query=True):
    io = _IO()
    h = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True)
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'W1HPC'
    h.state, h.turn = A.VaraState.CONNECTED, A._TURN_OURS
    h._link_high_speed, h._rec500 = True, 2
    h._lead = A.VB6Rnd(F.lead_state(h.caller, True, '500'))
    h._start_compression('OFF')
    h.send(PAYLOAD)
    if query:
        clock.now += 2
        h._heard += 2 * M.FS
        assert h._probe_intermediate_answer()
    return h, io


def feed(h, clock, x, block=960):
    for at in range(0, len(x), block):
        z = x[at:at+block]
        clock.now += len(z) / M.FS
        h.on_rx_stream(z)


def retained(h):
    return h._tx_payload(h._tx_pending[0]) + b''.join(h._txq)


@pytest.mark.parametrize('block', [512, 960, 4096, 24000])
def test_recorded_w1hpc_query_answer_retries_same_bytes_lower_once(clock, block):
    h, io = pending(clock)
    before = len(io.sent)
    feed(h, clock, reply(), block)
    assert h._tx_retries == 1 and len(io.sent) == before + 1, io.msgs
    assert h._tx_pending[2] < 2 and retained(h) == PAYLOAD
    assert h.payload_ledger.confirmed == 0
    feed(h, clock, reply(), block)
    assert h._tx_retries == 1 and len(io.sent) == before + 1


def generated(erased=2, wrong=False):
    kind = F.for_bw(F.SESSION_DATA_NAK_QUERY, '500')
    tones = F.handshake_tones('W1HPC', kind)
    if wrong: tones[12] = 50 if tones[12] != 50 else 52
    x = M.synth_tones(tones)
    if erased: x[(len(tones)-erased)*M.HOP:] = 0
    return np.r_[np.zeros(4800), x, np.zeros(12000)]


def test_erased_tail_is_not_permission_to_transmit_before_full_interval(clock):
    h, io = pending(clock)
    x = generated()
    cut = 4800 + 30*M.HOP
    before = len(io.sent)
    feed(h, clock, x[:cut])
    assert len(io.sent) == before and h._tx_retries == 0
    feed(h, clock, x[cut:])
    assert h._tx_retries == 1


@pytest.mark.parametrize('case', ['short-prefix', 'wrong-tone', 'wrong-call', 'unsolicited',
                                  'stale', 'peer-turn', 'no-pending'])
def test_erasure_recovery_requires_full_identity_and_current_query(case, clock):
    h, io = pending(clock, query=case != 'unsolicited')
    if case == 'wrong-call': h.called = 'N0ZZZ'
    if case == 'stale': clock.now += 5
    if case == 'peer-turn': h.turn = A._TURN_PEER
    if case == 'no-pending': h._tx_pending = None
    before = len(io.sent)
    feed(h, clock, generated(erased=4 if case == 'short-prefix' else 2,
                             wrong=case == 'wrong-tone'))
    assert h._tx_retries == 0 and len(io.sent) == before, io.msgs
    assert h.payload_ledger.confirmed == 0


def test_recovery_finishes_the_qsy_delivery_without_duplicate_bytes(clock):
    from .test_responder_control_stream import station
    sender, tx = pending(clock)
    peer, rx = station('W1HPC')
    peer.caller, peer.chat, peer.turn = 'W9SSJ', True, A._TURN_PEER
    peer._link_high_speed = True
    peer._start_compression('OFF')
    peer._expect_level = 2
    peer._rx_resend, peer._rx_failed = (3, (False,)), True
    peer._answer_owed, peer._owed_block = A._OWED_OVER, True
    sent_at = len(tx.sent)
    feed(sender, clock, reply())
    replies = 0
    for _ in range(6):
        for wave in tx.sent[sent_at:]:
            feed(peer, clock, np.r_[np.zeros(4800), wave, np.zeros(24000)])
        sent_at = len(tx.sent)
        for wave in rx.sent[replies:]:
            feed(sender, clock, np.r_[np.zeros(4800), wave, np.zeros(24000)])
        replies = len(rx.sent)
        if sender.payload_ledger.pending == 0:
            break
    assert b''.join(rx.host) == PAYLOAD, (tx.msgs, rx.msgs)
    assert sender.payload_ledger.confirmed == len(PAYLOAD)
    assert sender.payload_ledger.pending == 0 and sender._tx_pending is None


def test_recorded_receive_gaps_recover_on_third_query_instead_of_looping(clock):
    import hashlib
    import json
    from pathlib import Path
    from scipy.io import wavfile
    p = Path(__file__).parent / 'fixtures/w1hpc-query-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    h, io = pending(clock, query=False)
    for entry in meta['receive_gaps']:
        path = p / entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        fs, x = wavfile.read(path)
        assert fs == M.FS
        feed(h, clock, x.astype(float) / 32768)
        assert h._tx_retries == 0 and h.payload_ledger.confirmed == 0
        assert h._probe_intermediate_answer()
    assert h._intermediate_query_attempts == 3
    feed(h, clock, reply())
    assert h._tx_retries == 1 and h._tx_pending[2] == 1
    assert retained(h) == PAYLOAD and h.payload_ledger.confirmed == 0
    assert any('session-data-nak-query' in m and '— NAK;' in m for m in io.msgs)
