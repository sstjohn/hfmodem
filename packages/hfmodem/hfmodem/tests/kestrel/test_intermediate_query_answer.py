# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stock intermediate-answer recovery, with independent positive and negative captures."""
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_intermediate_query_probe import pending
from hfmodem.tests.kestrel.test_final_ack_recovery import stream


def native(name):
    path = Path(__file__).parent / 'fixtures/intermediate-query-answer' / (name+'.wav')
    if not path.is_file():
        pytest.skip(f'native intermediate answer unavailable: {path}')
    fs, x = wavfile.read(path)
    assert fs == MK.FS and x.ndim == 1
    return x.astype(float) / 32768 if x.dtype == np.int16 else x.astype(float)


def timely(hs, x):
    kind = VF.for_bw(VF.SESSION_INTERMEDIATE_QUERY_ANSWER, '2750')
    tones = np.asarray(VF.handshake_tones(hs.called, kind))
    _, at = VA._payload_fit(x, kind, tones, band=hs._band)
    return x[:at + VA._span(kind) + int(.1 * MK.FS)]


def queried():
    hs, io = pending()
    assert hs._probe_intermediate_answer()
    return hs, io


@pytest.mark.parametrize('name', ['stock', 'onair', 'mail-query-answer'])
@pytest.mark.parametrize('raw', [True, False])
def test_native_solicited_answer_retires_exactly_once_and_keys_distinct_data(name, raw):
    hs, io = queried()
    before, queued = hs._tx_pending, list(hs._txq)
    x = native(name)
    bracket = timely(hs, x)
    assert hs._peer_intermediate_query_answer(bracket)
    if raw:
        stream(hs, x)
    else:
        hs.on_rx_audio(bracket)
    assert hs._tx_pending[1] == before[1] + 1 and hs._tx_pending[0] != before[0]
    assert len(hs._txq) == len(queued) - 1 and len(io.sent) == 3
    assert hs.turn == VA._TURN_OURS and not hs._released and not hs._handed_over
    assert hs._intermediate_query_for is None and not hs._intermediate_query_attempted
    after = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    hs.on_rx_audio(bracket)
    hs._took_intermediate_query_answer()
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == after


@pytest.mark.parametrize('field,value', [
    ('called', 'KB5LZK'), ('bw', '2300'), ('bw', '500'), ('role', 'responder'),
    ('turn', VA._TURN_PEER), ('turn', VA._TURN_ASKED),
    ('state', VA.VaraState.CONNECTING),
    ('probe_intermediate_query', False), ('_txq', []), ('_tx_pending', None),
    ('_intermediate_query_for', None), ('_intermediate_query_at', -100),
    ('_intermediate_query_samples', int(4.5 * MK.FS) + 1),
])
def test_answer_requires_same_fresh_solicited_boundary(field, value):
    hs, io = queried()
    x = timely(hs, native('stock'))
    setattr(hs, field, value)
    assert not hs._peer_intermediate_query_answer(x)


def test_pending_identity_must_match_query():
    hs, _ = queried()
    body, over, level = hs._tx_pending
    hs._intermediate_query_for = body, over + 1, level
    assert not hs._peer_intermediate_query_answer(timely(hs, native('stock')))


def test_refused_query_has_no_acceptance_marker():
    hs, io = pending()
    io.tx_went_out = lambda: False
    assert not hs._probe_intermediate_answer()
    assert hs._intermediate_query_attempted and hs._intermediate_query_for is None
    assert not hs._peer_intermediate_query_answer(timely(hs, native('stock')))


def test_partial_tail_and_too_few_clear_symbols_rejected():
    hs, _ = queried()
    x = native('stock')
    kind = VF.for_bw(VF.SESSION_INTERMEDIATE_QUERY_ANSWER, '2750')
    tones = np.asarray(VF.handshake_tones(hs.called, kind))
    _, at = VA._payload_fit(x, kind, tones, band=hs._band)
    assert not hs._peer_intermediate_query_answer(x[:at + 31 * MK.HOP])
    missing = x.copy()
    missing[:at + 14 * MK.HOP] = 0
    assert not hs._peer_intermediate_query_answer(missing)


def test_other_called_keyed_controls_cannot_substitute():
    hs, io = queried()
    before = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    for kind in [VF.SESSION_TURN_REQUEST_RESPONDER, VF.SESSION_RESPONDER_IDLE,
                 VF.SESSION_IDLE_RESPONSE, VF.SESSION_DRAINED_RESPONDER]:
        called = hs.caller if kind.keyed_by == 'caller' else hs.called
        x = np.pad(MK.synth_burst(called, VF.for_bw(kind, '2750')), (4800, 4800))
        assert not hs._peer_intermediate_query_answer(x)
    hs._took_turn_request()
    hs._took_control_burst()
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before


def test_driver_query_deadline_follows_successful_data_unkey(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    hs, io = pending()
    assert hs._pending_answer_at == 100
    assert hs.intermediate_query_due_in() == pytest.approx(1.8)
    assert not hs.query_intermediate_answer() and len(io.sent) == 1
    clock[0] = 101.79
    assert not hs.query_intermediate_answer()
    assert hs._pending_answer_at == 100
    clock[0] = 101.81
    assert hs.query_intermediate_answer() and len(io.sent) == 2
    assert hs._pending_answer_at == 100 and hs.intermediate_query_due_in() == pytest.approx(4.5)
    assert not hs.query_intermediate_answer()


def test_refused_data_does_not_start_clock():
    hs, io = pending()
    hs._tx_pending = None
    hs._pending_answer_at = None
    io.tx_went_out = lambda: False
    hs._tx_data_over()
    assert hs._pending_answer_at is None and hs._tx_pending is None
    assert hs.intermediate_query_due_in() is None


def test_originate_resets_probe_markers():
    hs, _ = queried()
    hs.originate('KC9GHZ', 'W9SSJ')
    assert not hs._intermediate_query_attempted and hs._intermediate_query_for is None
    assert hs._intermediate_query_samples == 0 and hs._pending_answer_at is None


@pytest.mark.parametrize('name', ['stock', 'onair', 'mail-query-answer'])
def test_late_bracket_is_not_a_fresh_answer(name):
    hs, _ = queried()
    x = timely(hs, native(name))
    late = np.pad(x, (0, int((VA._GRANT_FRESH_S + .1) * MK.FS)))
    assert not hs._peer_intermediate_query_answer(late)


def test_native_missing_data_reply_does_not_acknowledge_pending():
    hs, io = queried()
    x = native('negative')
    # Native negative has a measured onset, not a positive-template alignment.
    bracket = x[:12680 + VA._span(VF.SESSION_INTERMEDIATE_QUERY_ANSWER) + 4800]
    before = hs._tx_pending, list(hs._txq), hs._over, hs.progress
    assert not hs._peer_intermediate_query_answer(bracket)
    hs.on_rx_audio(bracket)
    assert (hs._tx_pending, hs._txq, hs._over, hs.progress) == before
    assert len(io.sent) == 2 and not hs._released and not hs._handed_over


def test_payload_alignment_does_not_require_its_unaligned_preamble():
    hs, _ = queried()
    x = timely(hs, native('mail-query-answer'))
    kind = VF.for_bw(VF.SESSION_INTERMEDIATE_QUERY_ANSWER, '2750')
    tones = np.asarray(VF.handshake_tones(hs.called, kind))
    heard, at = VA._payload_fit(x, kind, tones, band=hs._band)
    assert heard[0] == 73 and tones[0] == 74
    assert np.array_equal(heard[1:], tones[1:])
    assert hs._peer_intermediate_query_answer(x)
    # All32 are independently intact at a nearby offset; source alignment did
    # not optimize the preamble. Retain this proof alongside the payload test.
    bins, clear, _resid = VA._top3_track(x, band=hs._band)
    offset = at + 256
    j = np.rint((offset + np.arange(32) * MK.HOP + MK._WOFF) / VA._ACK_GRID).astype(int)
    assert np.all(clear[j, 0] >= VA._CLEAR_DB)
    assert np.array_equal(bins[j, 0], tones)


@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_failed_mail_native_recovers_through_public_stream_chunk_sizes(chunk):
    hs, io = queried()
    before = hs._tx_pending
    stream(hs, native('mail-query-answer'), chunk=chunk)
    assert hs._tx_pending[1] == before[1] + 1
    assert hs._tx_pending[0] != before[0] and len(io.sent) == 3
    assert hs.turn == VA._TURN_OURS and hs._intermediate_query_for is None


def test_intact_preamble_does_not_excuse_a_clear_wrong_payload_symbol():
    hs, io = queried()
    kind = VF.for_bw(VF.SESSION_INTERMEDIATE_QUERY_ANSWER, '2750')
    tones = VF.handshake_tones(hs.called, kind)
    assert tones[0] == 74
    tones[13] += 6
    x = np.pad(MK.synth_tones(tones), (4800, 4800))
    before = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    assert not hs._peer_intermediate_query_answer(x)
    stream(hs, x)
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before


@pytest.mark.parametrize('reply', [1, 2])
@pytest.mark.parametrize('damage', [None, 'missing-tail', 'wrong-call'])
def test_k0si_query_answer_rescans_at_tail_with_uneven_receive_chunks(reply, damage):
    from hfmodem.tests.kestrel.test_data_over_gate import _connected
    hs, io = _connected()
    hs.called = 'K0SI' if damage != 'wrong-call' else 'KB5LZK'
    hs._peer_offset = -9.3 / (MK.FS / MK.NFFT)
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 55)
    assert hs._probe_intermediate_answer()
    before = hs._tx_pending
    x = native(f'k0si-20260920-query-{reply}')
    if damage == 'missing-tail':
        x[int(1.32 * MK.FS):] = 0
    last = 0
    # At 1.37 s the complete payload is recognizable but the reply's tail
    # has not arrived. The old half-second stride next scanned at 1.89 s,
    # after the freshness limit; the 1.46 s callback must now resolve it.
    for seconds in (.35, .75, .90, 1.05, 1.2, 1.37, 1.46, 1.89, 2.1):
        end = round(seconds * MK.FS)
        hs.on_rx_stream(x[last:end])
        last = end
        if seconds <= 1.37 or damage:
            assert hs._tx_pending == before
        else:
            assert hs._tx_pending[1] == before[1] + 1
            assert hs._tx_pending[0].startswith(b'B' * 55)
            assert not hs._txq
    assert len(io.sent) == (2 if damage else 3)
