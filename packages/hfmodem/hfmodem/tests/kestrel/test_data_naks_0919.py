# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CRC-rejection NACKs from stock VARA and the K0SI failed-mail recording."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_mfsk as MK
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'data-naks-0919'
if not (FIXTURES / "manifest.json").is_file():
    pytest.skip("VARA DATA NACK recordings are not included in this distribution",
                allow_module_level=True)
ROWS = json.loads((FIXTURES / 'manifest.json').read_text())


def native(row):
    fs, x = wavfile.read(FIXTURES / (row['name'] + '.wav'))
    assert fs == MK.FS
    return x.astype(float) / (32768 if x.dtype == np.int16 else 1)


def pending(row):
    hs, io = _connected()
    hs.called = row['call']
    hs._peer_offset = -11.2 / (48000 / 2048) if row['call'] == 'K0SI' else 0
    hs.turn = VA._TURN_OURS
    hs._over = 1
    data = bytes(range(144))
    hs.send(data)
    if row['seed'] == 288:
        assert hs._probe_intermediate_answer()
    return hs, io, data


def unconfirmed(hs):
    body = hs._tx_pending[0]
    return phy.vara_payload(body, caller=hs.caller, body_len=len(body)) + b''.join(hs._txq)


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['name'])
@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_nack_retries_smaller_without_retiring_any_bytes(row, chunk):
    hs, io, data = pending(row)
    old_over, sent = hs._tx_pending[1], len(io.sent)
    x = native(row)
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])
    assert hs._tx_pending[1:] == (old_over, 2)
    assert len(hs._tx_pending[0]) == 48 and unconfirmed(hs) == data
    assert hs._tx_retries == 1 and len(io.sent) == sent + 1
    assert hs._tx_recovery_levels == [2, 2, 2]
    assert any('— NAK;' in msg and f"({row['seed']}/{row['preadv']})" in msg for msg in io.msgs)
    hs.on_rx_audio(x)
    hs._took_data_nak()
    assert hs._tx_retries == 1 and len(io.sent) == sent + 1


@pytest.mark.parametrize('row', [ROWS[0], ROWS[1]], ids=lambda r: r['name'])
@pytest.mark.parametrize('state', ['no_pending', 'peer_turn', 'responder', 'connecting',
                                  'wrong_call', 'expired', 'short_frame', 'lower_record'])
def test_native_nack_cannot_escape_its_pending_data_window(row, state):
    hs, io, _ = pending(row)
    if state == 'no_pending': hs._tx_pending = None
    if state == 'peer_turn': hs.turn = VA._TURN_PEER
    if state == 'responder': hs.role = 'responder'
    if state == 'connecting': hs.state = VA.VaraState.CONNECTING
    if state == 'wrong_call': hs.called = 'N0CALL'
    if state == 'expired':
        hs._pending_answer_at -= 10
        hs._intermediate_query_at -= 10
    if state == 'short_frame':
        hs._tx_pending = (phy.vara_body(b'short', hs.caller), 2, 3)
    if state == 'lower_record':
        hs._tx_pending = (phy.vara_body(b'A' * 47, hs.caller, body_len=48), 2, 2)
    assert not hs._peer_data_nak(native(row))


def test_abandoned_query_does_not_authorize_a_delayed_negative_tail():
    row = ROWS[1]
    hs, io, _ = pending(row)
    x = native(row)
    assert not hs._peer_data_nak(x[:int(.65 * MK.FS)])
    hs._intermediate_query_for = None
    assert not hs._peer_data_nak(x)


def test_refused_retry_preserves_geometry_budgets_and_later_host_delivery():
    row = ROWS[0]
    hs, io, data = pending(row)
    hs.send(b'next host write')
    old, queue = hs._tx_pending, list(hs._txq)
    # Match a fresh complete bracket, not the .65 seconds retained for stream tests.
    x = native(row)[:int(1.0 * MK.FS)]
    assert hs._peer_data_nak(x)
    io.tx_went_out = lambda: False
    hs._took_data_nak()
    assert hs._tx_pending == old and hs._txq == queue
    assert hs._tx_retries == 0 and not hs._tx_recovery_levels
    assert unconfirmed(hs) == data + b'next host write'


def test_retry_and_close_preserve_separate_host_delivery():
    hs, io, data = pending(ROWS[0])
    hs.send(b'later')
    assert hs._peer_data_nak(native(ROWS[0])[:48000])
    hs._took_data_nak()
    decoded = bytearray()
    while True:
        body, over, level = hs._tx_pending
        assert level == 2
        decoded.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        if phy.over_is_last(body, hs.caller):
            break
        hs._took_over_continue()
    assert bytes(decoded) == data
    assert hs._txq == [b'later']
    hs._took_control_burst()
    assert hs._tx_pending[2] == 3 and unconfirmed(hs) == b'later'


def test_nack_budget_exhaustion_never_retires_data():
    hs, io, data = pending(ROWS[0])
    old = hs._tx_pending
    hs._tx_retries = VA._OVER_RETRY_MAX
    assert hs._peer_data_nak(native(ROWS[0])[:48000])
    hs._took_data_nak()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert hs._tx_pending == old and unconfirmed(hs) == data


def test_delayed_action_cannot_apply_to_a_new_reply_window():
    hs, io, data = pending(ROWS[0])
    assert hs._peer_data_nak(native(ROWS[0])[:48000])
    hs._pending_answer_at += .1
    hs._took_data_nak()
    assert hs._tx_retries == 0 and len(io.sent) == 1
