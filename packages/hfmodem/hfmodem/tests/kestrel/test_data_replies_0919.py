# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Stock-confirmed BW2300 replies lost by the real receive cursor on September 18."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.kestrel.arq import phy
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'data-replies-0919'


def native(name):
    path = FIXTURES / (name + '.wav')
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == MK.FS
    return x.astype(float) / (32768 if x.dtype == np.int16 else 1)


@pytest.mark.parametrize('name', [
    '20260918T052953Z-over3', '20260918T231756Z-over2',
    '20260918T235020Z-over2',
])
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_clipped_continue_advances_once_from_actual_guard_cursor(name, chunk):
    hs, io = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'B' * 89 + b'last')
    pending = hs._tx_pending
    x = native(name)
    assert VA._cont_plateau(x, band=hs._band) < VA._CONT_PLATEAU
    assert hs._peer_over_continue(x)
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])
    assert hs._tx_pending[1] == pending[1] + 1
    assert hs._tx_pending[0] != pending[0]
    assert len(io.sent) == 2 and len(hs._txq) == 1
    hs.on_rx_audio(x)
    assert len(io.sent) == 2  # The bracket cannot acknowledge the next block.


@pytest.mark.parametrize('name', ['stock-2300-continue-1', 'stock-2300-continue-2'])
def test_independent_stock_replies_match_the_full_link_tail(name):
    x = native(name)
    band = MK.band_for('2300')
    track = VA._top3_track(x, band=band)
    at, width = VA._widest(VA._cont_held(track[0], track[1], VA._live_track(x)))
    assert width >= VA._CONT_PLATEAU
    pairs = MK.demod_tone_pairs(x[(at + width // 2) * VA._ACK_GRID:], 8, band=band)
    assert tuple(pairs) == VF.OVER_CONTINUE_RESPONDER_BY_LINK['W9SSJ', 'KC9GHZ', '2300']


def test_truncated_other_frames_and_missing_tail_do_not_retire_data():
    hs, io = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + b'last')
    pending = hs._tx_pending
    for pairs in (VF.NAK_RESPONDER_2300, VF.CONTROL_BURST_RESPONDER_2300,
                  VF.OVER_CONTINUE_CALLER_2300):
        x = np.pad(MK.synth_tone_pairs(pairs)[MK.HOP:], (0, 4800))
        assert not hs._peer_head_cut_continue(x, VA._top3_track(x, band=hs._band))
    x = native('20260918T052953Z-over3')[:5 * MK.HOP]
    hs.on_rx_audio(x)
    assert hs._tx_pending == pending and len(io.sent) == 1


def query_pending(*, last_full=False):
    hs, io = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 89 + (b'' if last_full else b'B' * 89) + b'last')
    hs._retry_data_over()
    assert hs._intermediate_query_for == hs._tx_pending
    return hs, io


def through_frame(hs, name, kind, call=None):
    x = native(name)
    kind = VF.for_bw(kind, hs.bw)
    tones = np.asarray(VF.handshake_tones(call or hs.called, kind))
    _, at = VA._payload_fit(x, kind, tones, band=hs._band)
    return x[:at + VA._span(kind) + int(.08 * MK.FS)]


@pytest.mark.parametrize('last_full', [False, True])
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_stock_solicited_answer_advances_one_distinct_block(last_full, chunk):
    hs, io = query_pending(last_full=last_full)
    old = hs._tx_pending
    name, kind = (('stock-2300-last-full-answer', VF.SESSION_RESPONDER_OVER_ANSWER)
                  if last_full else ('stock-2300-query-answer', VF.SESSION_INTERMEDIATE_QUERY_ANSWER))
    x = through_frame(hs, name, kind)
    assert hs._peer_intermediate_query_answer(x)
    x = native(name)  # Continuous capture includes the next scheduled scan.
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])
    assert hs._tx_pending[1] == old[1] + 1 and hs._tx_pending[0] != old[0]
    assert hs._tx_retries == 0 and len(io.sent) == 3
    hs.on_rx_audio(x)
    assert len(io.sent) == 3


def test_stock_missing_data_answer_retries_without_retiring_bytes():
    hs, io = query_pending()
    old, queue = hs._tx_pending, list(hs._txq)
    x = through_frame(hs, 'stock-2300-query-nak', VF.SESSION_OVER_NAK_RESPONDER)
    assert not hs._peer_intermediate_query_answer(x)
    x = native('stock-2300-query-nak')
    for at in range(0, len(x), 512):
        hs.on_rx_stream(x[at:at + 512])
    body = hs._tx_pending[0]
    assert hs._tx_pending[1:] == (old[1], 1)
    assert len(body) == 23
    assert (phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            + b"".join(hs._txq) == phy.vara_payload(old[0], caller=hs.caller)
            + b"".join(queue))
    assert hs._tx_retries == 1 and hs._intermediate_query_for is None


def test_stock_final_query_answer_requires_solicitation_before_drained_handover():
    hs, io = _connected(bw='2300')
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 79)
    old = hs._tx_pending
    x = through_frame(hs, 'stock-2300-final-query-answer', VF.SESSION_TURN_REQUEST_RESPONDER, hs.caller)
    for at in range(0, len(x), 512):
        hs.on_rx_stream(x[at:at + 512])
    assert hs._tx_pending == old and hs._final_query_attempts == 1
    for at in range(0, len(x), 512):
        hs.on_rx_stream(x[at:at + 512])
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._tx_retries == 0
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(hs.called, VF.SESSION_DRAINED))
