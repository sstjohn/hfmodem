# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stock's final-answer query/request/drained exchange preserves DATA ownership."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'final-ack-recovery'


def recorded(name):
    path = FIXTURES / (name + '.wav')
    if not path.exists():
        pytest.skip(f'stock final-ACK recovery recording absent: {path}')
    fs, x = wavfile.read(path)
    assert fs == MK.FS and x.dtype == np.float32
    return x.astype(float)


def pending(bw='2750', payload=79, extra=False):
    hs, io = _connected(bw=bw)
    hs.turn = VA._TURN_OURS
    hs._txq = [b'P' * payload] + ([b'next'] if extra else [])
    hs._tx_data_over()
    return hs, io


def stock_reply():
    # Native response to stock's successfully keyed query. Its fitted first
    # symbol is0.336667s into the crop; retain80ms leading audio, matching the
    # measured query-unkey -> responder-key gap, plus its complete tail.
    x = recorded('responder-turn-request')
    return x[12320:int(1.91 * MK.FS)]


def stream(hs, x, chunk=4096):
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])


def is_burst(x, kind, called='KC9GHZ'):
    kind = VF.for_bw(kind, '2750')
    want = np.asarray(VF.handshake_tones(called, kind))
    # Supply a bracket around locally generated zero-origin waveforms too;
    # otherwise the payload fitter can put the first symbol outside the array.
    heard, _ = VA._payload_fit(np.pad(x, (4800, 4800)), kind, want,
                              band=MK.band_for('2750'))
    return np.array_equal(heard, want)


@pytest.mark.parametrize('name,kind,call', [
    ('caller-final-answer-poll', VF.SESSION_FINAL_ANSWER_QUERY, 'KC9GHZ'),
    ('responder-turn-request', VF.SESSION_TURN_REQUEST_RESPONDER, 'W9SSJ'),
    ('caller-drained', VF.SESSION_DRAINED, 'KC9GHZ'),
])
def test_generated_control_is_the_stock_cables_own_tones(name, kind, call):
    assert is_burst(recorded(name), kind, call)


@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_query_then_stock_reply_retires_only_after_drained_transmission(chunk):
    hs, io = pending()
    original = hs._tx_pending
    hs._retry_data_over(final_query=True)
    assert hs._tx_pending == original and hs._final_query_attempts == 1
    assert is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)
    stream(hs, stock_reply(), chunk)
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert len(io.sent) == 3  # DATA, solicitation, measured32-symbol drained.
    assert is_burst(io.sent[-1], VF.SESSION_DRAINED)
    assert hs._final_query_for is None and hs._final_query_attempts == 0
    assert hs._released and hs._handed_over


def test_unsolicited_request_queries_but_does_not_acknowledge_final_data():
    from hfmodem.tests.kestrel.test_connected_turn_requests import audio
    hs, io = pending()
    original = hs._tx_pending
    # Stop at the query: the rest of the old KC9 tape is not a response to a
    # transmission our live implementation never made that morning.
    x = audio('first-pair')[:int(1.7 * MK.FS)]
    for at in range(0, len(x), 512):
        hs.on_rx_stream(x[at:at + 512])
        if len(io.sent) > 1:
            break
    assert len(io.sent) == 2 and hs._tx_pending == original
    assert is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)
    assert hs._final_ack_confirmed is None


@pytest.mark.parametrize('bw,payload,extra', [
    # A BW2300 final short over is now a measured shape — a responder acks before
    # it asks, so a turn-request behind it retires the over [see vara_arq
    # _took_turn_request; test_caller_in_peer_turn]. This case keeps the 2300
    # INTERMEDIATE shape, which still retains the turn and retries the data.
    ('2750', 89, False), ('2750', 79, True), ('2300', 79, True), ('500', 12, False),
])
def test_unmeasured_pending_shapes_close_without_blind_data_retry(bw, payload, extra):
    hs, io = pending(bw, payload, extra)
    original = hs._tx_pending
    assert not hs._took_turn_request(raw=True)
    assert len(io.sent) == 1
    hs._retry_data_over(final_query=True)
    assert hs._tx_pending == original and hs._tx_retries == 0
    assert hs.state == VA.VaraState.DISCONNECTED
    assert not np.array_equal(io.sent[0], io.sent[1])
    assert hs._final_query_attempts == 0


def test_corrupted_final_ack_does_not_retire_or_solicit_on_noise():
    hs, io = pending()
    original = hs._tx_pending
    stream(hs, recorded('corrupted-final-ack'))
    assert hs._tx_pending == original and len(io.sent) == 1


def test_query_budget_and_refused_query_preserve_pending(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(VA.time, 'monotonic', lambda: clock[0])
    hs, io = pending()
    original = hs._tx_pending
    io.tx_went_out = lambda: False
    hs._retry_data_over(final_query=True)
    assert hs._final_query_attempts == 0 and hs._final_query_for is None
    assert hs._since_progress == 0
    io.tx_went_out = lambda: True
    for n in range(VA._FINAL_QUERY_MAX):
        hs._retry_data_over(final_query=True)
        assert hs._final_query_attempts == n + 1
        assert hs._tx_pending == original
        # A second call in the same query window emits nothing.
        sent = len(io.sent)
        hs._retry_data_over(final_query=True)
        assert len(io.sent) == sent
        clock[0] += VA._FINAL_QUERY_REPLY_S + .1
    hs._retry_data_over(final_query=True)
    assert hs.state == VA.VaraState.DISCONNECTED
    assert hs._tx_pending == original


def test_refused_drained_retains_the_final_frame_and_confirmed_answer():
    hs, io = pending()
    original = hs._tx_pending
    hs._retry_data_over(final_query=True)
    io.tx_went_out = lambda: False
    stream(hs, stock_reply())
    assert hs._tx_pending == original and hs._final_ack_confirmed == original
    assert hs.turn == VA._TURN_OURS
    io.tx_went_out = lambda: True
    hs._retry_data_over(final_query=True)
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert is_burst(io.sent[-1], VF.SESSION_DRAINED)


def test_a_late_prequery_bracket_cannot_confirm_at_zero_receive_age():
    hs, io = pending()
    x = stock_reply()[:int(1.6 * MK.FS)]
    hs.on_rx_audio(x)  # Bracket-only request can solicit, never confirm.
    original = hs._tx_pending
    assert hs._final_query_attempts == 1 and hs._final_query_samples == 0
    hs.on_rx_audio(x)
    assert hs._tx_pending == original and len(io.sent) == 2
    assert hs._final_ack_confirmed is None


@pytest.mark.parametrize('age_source', ['wall', 'samples'])
def test_reply_after_query_window_opens_another_query_not_a_false_ack(age_source):
    hs, io = pending()
    original = hs._tx_pending
    hs._retry_data_over(final_query=True)
    if age_source == 'wall':
        hs._final_query_at -= VA._FINAL_QUERY_REPLY_S + .1
    else:
        hs._final_query_samples = int((VA._FINAL_QUERY_REPLY_S + .1) * MK.FS)
    stream(hs, stock_reply())
    assert hs._tx_pending == original and hs._final_query_attempts == 2
    assert is_burst(io.sent[-1], VF.SESSION_FINAL_ANSWER_QUERY)


def test_explicit_nak_retries_data_and_invalidates_query():
    hs, io = pending()
    original = hs._tx_pending
    hs._retry_data_over(final_query=True)
    hs._took_nak()
    assert np.array_equal(io.sent[0], io.sent[-1])
    assert hs._tx_pending == original and hs._final_query_for is None


def test_real_ack_and_new_session_reset_final_query_state():
    hs, io = pending()
    hs._retry_data_over(final_query=True)
    hs._took_control_burst()
    assert hs._tx_pending is None and hs._final_query_for is None
    assert hs._final_query_attempts == 0
    hs._final_query_attempts = 2
    hs.originate('KC9GHZ', 'W9SSJ')
    assert hs._final_query_attempts == 0 and hs._final_query_for is None
