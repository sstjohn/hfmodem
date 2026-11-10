# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VA2OPS, 2026-11-08: a clean retry is not refused for missing our prediction.

Our NAK of its level-6 over predicted level 5. It keyed level 6 again, then 4,
then 1, every one CRC-clean, and each was refused for not being level 5.
"""
import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A
from .test_resend_ladders_1020 import receiver
from .test_responder_control_stream import feed
from .test_idle_poll import clock as clock

FIXTURE = Path(__file__).with_name('fixtures') / 'va2ops-retries-1108'
BODY = b'5 <AWQ>'


def recorded(level):
    meta = json.loads((FIXTURE / 'provenance.json').read_text())['files'][f'level{level}.wav']
    path = FIXTURE / f'level{level}.wav'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
    rate, x = wavfile.read(path)
    assert rate == 48000
    return x.astype(float) / 32768


def station(over=6):
    h, io = receiver('500', over)
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'VA2OPS'
    h.turn = A._TURN_PEER
    h._start_compression('OFF')
    h._peer_delivery_open = True
    h._irs_nak(A._phy.LevelRead(over, (), None, 1))
    return h, io


def test_the_nak_predicts_level_5():
    h, _ = station()
    assert h._expect_level == 5 and h._rx_resend is not None


@pytest.mark.parametrize('level', [6, 4, 1])
@pytest.mark.parametrize('block', [960, 4096])
def test_recorded_clean_retry_is_delivered(clock, level, block):
    h, io = station()
    x = recorded(level)
    feed(h, x, block)
    assert b''.join(io.host) == BODY, io.msgs
    assert h._rx_resend is None and not h._rx_failed
    assert not any('will not decode' in m for m in io.msgs)
    assert any(f'CRC-checked DATA received at level {level}' in m for m in io.msgs)


@pytest.mark.parametrize('level', [4, 1])
def test_base_reader_takes_a_base_retry_under_a_high_prediction(level):
    h, io = station()
    x = recorded(level)
    bodies = h._peer_data_over_500(x, len(x) / 48000)
    assert [b for b in bodies if b.startswith(BODY)], io.msgs


@pytest.mark.parametrize('first_over', [False, True])
@pytest.mark.parametrize('case', ['our-echo', 'damaged'])
def test_a_base_frame_still_has_to_be_a_clean_retry_of_the_peer(clock, monkeypatch, case, first_over):
    h, io = station()
    if first_over:
        monkeypatch.setattr(h, '_first_over_owed', lambda: True)
    x = recorded(4)
    if case == 'our-echo':
        d = A._RX500.decode_stream(x)
        h._keyed_bodies.update(bytes(f.payload) + bytes([f.marker]) for f in d.data_frames)
    if case == 'damaged':
        x = x.copy()
        x[48000:168000] = 0
    assert h._peer_data_over_500(x, len(x) / 48000) == []
    state, sent = (h._rx_resend, h._expect_level), len(io.sent)
    feed(h, x, 960)
    assert not io.host and (h._rx_resend, h._expect_level) == state
    assert len(io.sent) == sent, io.msgs


def test_previous_rate_rule_opens_to_every_base_level_only_above_four():
    h, _ = station()
    assert all(h._previous_base_retry_rate(level) for level in (1, 2, 3, 4))
    assert not h._previous_base_retry_rate(5)
    h._rx_resend, h._expect_level = (4, (False,)), 3
    h._rx_retry_rates = {4, 3}
    assert not h._previous_base_retry_rate(1)


def test_an_unpredicted_level_must_be_one_asked_for_and_read_whole():
    h, _ = station()
    read = A._phy.LevelRead
    assert h._rx_retry_rates == {6, 5}
    assert h._read_matches(read(6, (b'x',), None, 1))
    assert h._read_matches(read(6, (b'x', b'y'), None, 2))
    assert not h._read_matches(read(6, (b'x',), None, 2))
    assert not h._read_matches(read(6, (), None, 1))
    assert not h._read_matches(read(7, (b'x',), None, 1))
    assert not h._read_matches(read(8, (b'x', b'y'), None, 2))
