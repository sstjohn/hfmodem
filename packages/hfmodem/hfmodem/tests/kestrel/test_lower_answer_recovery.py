# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Recovery owns the actual pending record through changes of rate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from .test_data_over_gate import _IO
from .test_final_ack_recovery import stream


def sender(level=1):
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='2750', tx_level=level)
    hs.state, hs.role, hs.turn = VA.VaraState.CONNECTED, 'initiator', VA._TURN_OURS
    hs.caller, hs.called = 'W9SSJ', 'KC9GHZ'
    return hs, io


@pytest.mark.parametrize('target', [1, 2, 3])
def test_each_announced_record_can_query_before_reaching_target(target):
    hs, io = sender(target)
    hs.send(bytes(range(196)))
    while hs._txq:
        original = hs._tx_pending
        assert hs._intermediate_answer_candidate()
        hs._retry_data_over()
        assert hs._intermediate_query_for == original
        assert hs._tx_pending == original and hs._tx_retries == 0
        hs._took_intermediate_query_answer()
        assert hs._tx_pending[1] == original[1] + 1
    assert hs._final_ack_candidate()


@pytest.mark.parametrize('bw,level', [('2300', 1), ('2300', 2), ('2300', 3),
                                     ('2750', 100), ('2750', 101), ('2750', 102), ('2750', 103)])
@pytest.mark.parametrize('short', [False, True])
def test_query_timeout_keeps_bytes_and_never_replays_data(bw, level, short, monkeypatch):
    hs, io = sender()
    hs.bw, hs._band = bw, VA.MK.band_for(bw)
    cap = phy.body_size(bw, level) - 1
    body = phy.vara_body(b'x' * (3 if short else cap), hs.caller,
                         tail=0x89, body_len=cap + 1)
    hs._tx_pending = original = body, 7, level
    hs._txq = [] if short else [b'last']
    hs._full_keyed = 2
    monkeypatch.setattr(VA.OF, 'data_over_tx', lambda *a, **k: pytest.fail('blind DATA repeat'))
    for _ in range(VA._FINAL_QUERY_MAX):
        hs._retry_data_over()
        hs._final_query_at -= 5
        hs._intermediate_query_at -= 5
    hs._retry_data_over()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert hs._tx_pending == original and hs._tx_retries == 0


@pytest.mark.parametrize('mutation', ['body_size', 'record', 'bandwidth', 'queued_final'])
def test_unsupported_or_inconsistent_geometry_remains_unconfirmed(mutation):
    hs, io = sender()
    body = phy.vara_body(b'x' * 22, hs.caller, tail=0x89, body_len=23)
    hs._tx_pending = body, 1, 101
    hs._txq = [b'last']
    if mutation == 'body_size': hs._tx_pending = body, 1, 102
    if mutation == 'record': hs._tx_pending = body, 1, 104
    if mutation == 'bandwidth': hs.bw = '500'
    if mutation == 'queued_final':
        hs._tx_pending = phy.vara_body(b'short', hs.caller, body_len=23), 1, 101
    original = hs._tx_pending
    assert not hs._intermediate_answer_candidate() and not hs._final_ack_candidate()
    hs._retry_data_over()
    assert hs._tx_pending == original and hs._tx_retries == 0
    assert hs.state == VA.VaraState.DISCONNECTED


def test_lower_record_does_not_accept_the_base_records_clipped_tail():
    from .test_head_cut_continue import recorded, pending as base_pending
    hs, io = sender(2)
    hs._tx_pending = phy.vara_body(b'x' * 22, hs.caller, tail=0x89, body_len=23), 2, 101
    hs._txq = [b'last']
    x = recorded('night-caller-over2')
    base, _ = base_pending()
    assert base._peer_head_cut_continue(x, VA._top3_track(x, band=base._band))
    assert not hs._peer_head_cut_continue(x, VA._top3_track(x, band=hs._band))


FIXTURES = Path(__file__).with_name('fixtures') / 'lower-answer-recovery'
# Keep synthetic checks runnable when the optional recording corpus is absent.
EVIDENCE = (json.loads((FIXTURES / 'manifest.json').read_text())
            if (FIXTURES / 'manifest.json').is_file() else [])
NATIVE = [r for r in EVIDENCE if not r.get('negative')]
NEGATIVES = [r for r in EVIDENCE if r.get('negative')]


def native_pending(row):
    p = row['pending']
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw=row['bw'],
        tx_level=p['level'] - 99 if row['bw'] == '2750' and not row.get('retry_count') else None)
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.role, hs.turn, hs.caller, hs.called = 'initiator', VA._TURN_OURS, 'W9SSJ', 'KC9GHZ'
    hs._tx_pending = bytes.fromhex(p['body']), p['over'], p['level']
    hs._txq = [bytes.fromhex(x) for x in row['queue']]
    hs._over, hs._full_keyed = p['over'], row['full_keyed']
    hs._tx_retries = row.get('retry_count', 0)
    hs._query_retry_phase = hs._tx_retries % 2
    hold_lower = row.get('retry_count') and len(b''.join(hs._txq)) < 89
    if hold_lower:
        remaining = b''.join(hs._txq)
        cap = phy.body_size(hs.bw, p['level']) - 1
        hs._txq = [remaining[i:i + cap] for i in range(0, len(remaining), cap)]
        if len(remaining) % cap == 0: hs._txq.append(b'')
    if hold_lower or (row['bw'] == '2300' and p['level'] == 1 and not row.get('retry_count')):
        hs._tx_recovery_levels = [p['level']] * len(hs._txq)
    fs, x = wavfile.read(FIXTURES / (row['name'] + '.wav'))
    assert fs == VA.MK.FS
    assert hashlib.sha256((FIXTURES / (row['name'] + '.wav')).read_bytes()).hexdigest() == row['sha256']
    return hs, io, x.astype(float)


@pytest.mark.parametrize('row', NATIVE, ids=lambda r: r['name'] if isinstance(r, dict) else 'recordings-unavailable')
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_lower_reply_advances_exactly_once_after_query(row, chunk):
    hs, io, x = native_pending(row)
    original = hs._tx_pending
    hs._retry_data_over()
    assert hs._tx_pending == original and len(io.sent) == 1
    kind = VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[int(row['query_preadv'] == 745)]
    np.testing.assert_array_equal(io.sent[0], VA.MK.synth_burst(hs.called, VF.for_bw(kind, hs.bw)))
    stream(hs, x, chunk)
    assert len(io.sent) == 2 and hs._tx_retries == 0
    if row['queue']:
        assert hs._tx_pending[1] == original[1] + 1
        body = hs._tx_pending[0]
        assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)).hex() == row['queue'][0]
    else:
        assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    hs.on_rx_audio(x)
    assert len(io.sent) == 2


@pytest.mark.parametrize('row', NATIVE, ids=lambda r: r['name'] if isinstance(r, dict) else 'recordings-unavailable')
@pytest.mark.parametrize('invalid', ['unsolicited', 'wall_expired', 'samples_expired', 'changed_pending'])
def test_native_reply_cannot_retire_an_unrelated_or_stale_pending_frame(row, invalid):
    hs, io, x = native_pending(row)
    if invalid != 'unsolicited': hs._retry_data_over()
    if invalid == 'wall_expired':
        hs._intermediate_query_at -= 5
        hs._final_query_at -= 5
    if invalid == 'samples_expired':
        hs._intermediate_query_samples = 5 * VA.MK.FS
        hs._final_query_samples = 5 * VA.MK.FS
    if invalid == 'changed_pending':
        body, over, level = hs._tx_pending
        hs._tx_pending = body, over + 1, level
    original = hs._tx_pending
    stream(hs, x)
    assert hs._tx_pending == original


def test_bw2300_full_record2_queries_with_payload_unconfirmed():
    from .test_data_over_gate import _connected
    hs, io = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs._tx_pending = original = phy.vara_body(b'x' * 47, hs.caller, tail=0x89, body_len=48), 1, 2
    hs._txq = [b'last']
    hs._retry_data_over()
    assert hs._tx_pending == original and hs._tx_retries == 0
    assert hs.state == VA.VaraState.CONNECTED and hs._intermediate_query_attempts == 1


@pytest.mark.parametrize('row', NEGATIVES, ids=lambda r: r['name'] if isinstance(r, dict) else 'recordings-unavailable')
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_lower_nak_repeats_exact_pending_without_retiring_bytes(row, chunk):
    hs, io, x = native_pending(row)
    old, queued, retries = hs._tx_pending, list(hs._txq), hs._tx_retries
    hs._retry_data_over()
    stream(hs, x, chunk)
    assert hs._tx_pending == old and hs._txq == queued
    assert hs._tx_retries == retries + 1
    assert len(io.sent) == 2
    np.testing.assert_array_equal(io.sent[-1], VA.OF.data_over_tx(old[0], over=old[1], level=old[2], bw=hs.bw))


@pytest.mark.parametrize('row', [r for r in NATIVE if r['answer']['preadv'] == 249], ids=lambda r: r['name'] if isinstance(r, dict) else 'recordings-unavailable')
@pytest.mark.parametrize('invalid', ['not_retry', 'explicit_ladder', 'missing_tail'])
def test_retry_base_answer_requires_retry_context_and_complete_tail(row, invalid):
    hs, io, x = native_pending(row)
    if invalid == 'not_retry': hs._tx_retries = 0
    if invalid == 'explicit_ladder': hs.tx_level = 2
    if invalid == 'missing_tail':
        kind = VF.for_bw(VF.SESSION_RETRY_QUERY_ANSWER, hs.bw)
        want = np.array(VF.handshake_tones(hs.called, kind))
        _, at = VA._payload_fit(x, kind, want, band=hs._band)
        x = x[:at + 29 * VA.MK.HOP]
    original = hs._tx_pending
    hs._retry_data_over()
    stream(hs, x)
    assert hs._tx_pending == original and len(io.sent) == 1


@pytest.mark.parametrize('refused', [False, True])
def test_full_retry_advances_query_phase_only_when_transmitted(refused):
    from .test_nak_repacketization import pending
    hs, io, _ = pending(n=94)
    original_ladder = hs._full_keyed
    if refused: io.tx_went_out = lambda: False
    hs._took_responder_nak()
    assert hs._full_keyed == original_ladder
    assert hs._query_retry_phase == (0 if refused else 1)
    io.tx_went_out = lambda: True
    hs._retry_data_over()
    expected = VF.for_bw(VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[0 if refused else 1], hs.bw)
    np.testing.assert_array_equal(io.sent[-1], VA.MK.synth_burst(hs.called, expected))
    # Querying again after silence retains the same phase.
    hs._intermediate_query_at -= 5
    hs._retry_data_over()
    np.testing.assert_array_equal(io.sent[-1], VA.MK.synth_burst(hs.called, expected))


@pytest.mark.skipif(not (FIXTURES / 'wrong-phase16.wav').is_file(),
                    reason='VARA wrong-phase reply recording is not included')
def test_wrong_phase_short_response_is_not_a_data_ack():
    from .test_nak_repacketization import pending
    hs, io, _ = pending(n=94)
    hs._took_responder_nak()
    hs._retry_data_over()
    original, queued, count = hs._tx_pending, list(hs._txq), len(io.sent)
    fs, x = wavfile.read(FIXTURES / 'wrong-phase16.wav')
    assert fs == VA.MK.FS
    stream(hs, x.astype(float))
    assert hs._tx_pending == original and hs._txq == queued and len(io.sent) == count


@pytest.mark.skipif(not NATIVE, reason='VARA lower-answer recordings are not included')
def test_native_base_reentry_preserves_later_writes_and_refused_next_tx():
    row = next(r for r in NATIVE if r['name'] == 'stock-2750-retry-base-answer')
    hs, io, x = native_pending(row)
    hs.send(b'later delivery')
    hs._retry_data_over()
    io.tx_went_out = lambda: False
    stream(hs, x)
    assert hs._tx_pending is None  # The solicited answer confirmed the small retry.
    assert hs._txq == [bytes.fromhex(row['queue'][0]), b'later delivery']
    assert not hs._tx_recovery_levels
    assert hs._over == row['pending']['over']
