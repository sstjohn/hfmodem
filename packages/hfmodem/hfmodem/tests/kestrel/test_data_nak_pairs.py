# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native eight-symbol NACKs must never retire a CRC-rejected DATA frame."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

HERE = Path(__file__).with_name('fixtures') / 'data-nak-pairs-0919'
# Recording corpora are optional in the public distribution.
ROWS = (json.loads((HERE / 'manifest.json').read_text())
        if (HERE / 'manifest.json').is_file() else [])


def setup(row):
    hs, io = _connected(bw=row['bw'])
    hs.turn = VA._TURN_OURS
    capacity = phy.body_size(row['bw'], row['record']) - 1
    data = b'A' * capacity + b'end'
    hs._tx_pending = (phy.vara_body(data[:capacity], hs.caller, tail=0x89,
                                   body_len=capacity + 1), 1, row['record'])
    hs._txq = [b'end']
    hs._pending_answer_at = time.monotonic()
    fs, audio = wavfile.read(HERE / row['name'])
    assert fs == MK.FS
    audio = audio.astype(float) / (32768 if audio.dtype == np.int16 else 1)
    return hs, io, data, audio


def unconfirmed(hs):
    body = hs._tx_pending[0]
    return phy.vara_payload(body, caller=hs.caller, body_len=len(body)) + b''.join(hs._txq)


@pytest.mark.parametrize('row', ROWS, ids=[r['name'] for r in ROWS])
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_nak_preserves_all_bytes_and_retries_at_supported_lower_record(row, chunk):
    hs, io, data, audio = setup(row)
    # The generic common-lead detector sees this as a continue. The full-link
    # negative classification must win before any pending byte can be retired.
    assert VA._cont_plateau(audio, band=hs._band) >= VA._CONT_PLATEAU
    assert not hs._peer_over_continue(audio)
    if row['bw'] == '500':
        hs.on_rx_audio(audio)
    else:
        for start in range(0, len(audio), chunk):
            hs.on_rx_stream(audio[start:start + chunk])
    assert hs._tx_retries == 1 and len(io.sent) == 1
    assert hs._tx_pending[1:] == (1, phy.lower_level(row['bw'], row['record']))
    assert unconfirmed(hs) == data
    assert any('session-data-pair-nak' in msg for msg in io.msgs)
    if row['bw'] != '500':
        hs.on_rx_audio(audio)
        assert len(io.sent) == 1


@pytest.mark.parametrize('row', ROWS, ids=[r['name'] for r in ROWS])
def test_no_action_before_tail_or_after_fresh_window(row):
    hs, io, data, audio = setup(row)
    assert not hs._peer_data_nak(audio[:int(.42 * MK.FS)])
    hs._pending_answer_at -= 10
    assert not hs._peer_data_nak(audio)
    assert not hs._peer_over_continue(audio)
    assert unconfirmed(hs) == data and not io.sent


@pytest.mark.parametrize('row', ROWS, ids=[r['name'] for r in ROWS])
def test_refusal_and_exhausted_budget_keep_rejected_bytes(row):
    hs, io, data, audio = setup(row)
    # Use the bracket to isolate transaction behavior from callback scheduling.
    at = hs._pair_data_nak_at(audio)
    fresh = audio[:at + 8 * MK.HOP + int(.1 * MK.FS)]
    assert hs._peer_data_nak(fresh)
    old, queue = hs._tx_pending, list(hs._txq)
    io.tx_went_out = lambda: False
    hs._took_data_nak()
    assert hs._tx_pending == old and hs._txq == queue and hs._tx_retries == 0
    io.tx_went_out = lambda: True
    hs._tx_retries = VA._OVER_RETRY_MAX
    assert hs._peer_data_nak(fresh)
    hs._took_data_nak()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert unconfirmed(hs) == data


@pytest.mark.parametrize('bw,levels,sizes', [
    ('500', [4,2,1,0], [44,35,23,10]),
    ('2300', [3,2,1,0], [90,48,23,10]),
    ('2750', [103,102,101,100], [90,48,23,10]),
])
def test_recovery_ladder_uses_codec_geometry_and_stops_at_floor(bw, levels, sizes):
    assert [phy.body_size(bw, level) for level in levels] == sizes
    assert [phy.lower_level(bw, level) for level in levels] == levels[1:] + levels[-1:]


@pytest.mark.skipif(not (HERE / 'query-manifest.json').is_file(),
                    reason='VARA DATA NACK query recordings are not included')
def test_bw500_queried_missing_lowest_retry_uses_bracket_route():
    row = dict(json.loads((HERE / 'query-manifest.json').read_text()), record=0)
    hs, io, data, audio = setup(row)
    old = hs._tx_pending
    # Establish the solicited reply window using the production query sender.
    assert hs._probe_intermediate_answer()
    count = len(io.sent)
    hs.on_rx_audio(audio)
    assert hs._tx_retries == 1 and len(io.sent) == count + 1
    assert hs._tx_pending == old and unconfirmed(hs) == data
    assert any('rx responder NAK' in msg for msg in io.msgs)


@pytest.mark.skipif(not (HERE / 'query-manifest.json').is_file(),
                    reason='VARA DATA NACK query recordings are not included')
def test_bw500_unqueried_missing_reply_does_not_authorize_lower_retry():
    row = dict(json.loads((HERE / 'query-manifest.json').read_text()), record=0)
    hs, io, data, audio = setup(row)
    old = hs._tx_pending
    hs.on_rx_audio(audio)
    assert hs._tx_retries == 0 and not io.sent
    assert hs._tx_pending == old and unconfirmed(hs) == data


@pytest.mark.skipif(not ROWS, reason='VARA DATA NACK recordings are not included')
def test_bw500_record2_crc_rejection_uses_stock_qualified_floor():
    row = dict(next(r for r in ROWS if r['bw'] == '500'), record=2)
    hs, io, data, audio = setup(row)
    assert hs._tx_retries == 0  # Also applies to a new over after an ACK.
    hs.on_rx_audio(audio)
    assert hs._tx_pending[2] == 0 and hs._tx_retries == 1
    assert hs._tx_recovery_levels and set(hs._tx_recovery_levels) == {0}
    assert unconfirmed(hs) == data


@pytest.mark.skipif(not ROWS, reason='VARA DATA NACK recordings are not included')
def test_unmapped_eight_symbol_nak_is_retained_for_query(monkeypatch):
    row = next(r for r in ROWS if r['bw'] == '2300')
    hs, io, data, audio = setup(row)
    monkeypatch.delitem(VF.DATA_NAK_RESPONDER_BY_LINK, (hs.caller, hs.called, hs.bw))
    assert not hs._peer_data_nak(audio)
    assert not hs._peer_over_continue(audio)
    assert hs._unclassified_continue is not None
    assert hs._probe_intermediate_answer()
    assert unconfirmed(hs) == data and not hs._confirmed_continue_pairs
