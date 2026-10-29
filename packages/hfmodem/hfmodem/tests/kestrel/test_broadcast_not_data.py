# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Observed application-port bytes were KISS broadcasts, not linked DATA.

Payloads are exact event-log bytes; CRC fields below are rebuilt, not claimed
as independently recovered RF CRCs. No audio decoding or transmission here.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.rx import beacon as B, varahf500 as rx
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from .test_bw500_locator_ownership import _station, _snapshot

OBSERVED = (
    ('2164c26c49addc4fa92e330a52e1a04081020f143100', '20MDM78OTKF0RKC'),
    ('2164c23c598e5b43873984aad410204081020f140000', '20GEL96CCN0JVP'),
)


def frame(payload, marker=0, level=2):
    body = bytes.fromhex(payload) if isinstance(payload, str) else payload
    body += bytes([marker])
    return rx.FrameResult(0, (), 0, 1., True,
                          body + crc16_genibus(body).to_bytes(2, 'big'), level=level)


@pytest.mark.parametrize('payload,text', OBSERVED)
@pytest.mark.parametrize('marker', [0, 0x40, 0x80])
def test_observed_bytes_are_canonical_broadcast_not_compressed_chat(payload, text, marker):
    f = frame(payload, marker)
    assert B.parse_payload(f.payload).text == text
    assert B.build_payload(text, f.payload[20]) == f.payload
    assert B.is_broadcast_frame(f)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('payload,text', OBSERVED)
@pytest.mark.parametrize('incomplete', [False, True])
def test_bracket_rejects_before_data_delivery_ack_or_ledger_changes(role, payload, text, incomplete):
    station = _station(role)
    f = frame(payload)
    result = rx.DecodeResult([f], [f], observation_complete=not incomplete)
    station._resend_decoded = result
    before = _snapshot(station)
    assert station._peer_data_over_500(np.empty(0), 5.1) == []
    assert _snapshot(station) == before
    assert all(event[0] == 'log' for event in station.io.events)


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('paired', [False, True])
def test_shared_locator_retires_broadcast_before_expected_or_pair_debt(role, paired):
    station = _station(role)
    f = frame(OBSERVED[0][0])
    over = SimpleNamespace(generation=station._front_generation, frames=[f], end=100,
                           decision=100, pair=paired)
    station._front_heard = 100
    station._base_ready = [over]
    station._base_rx = SimpleNamespace(history=[over])
    station._expect_500 = 2
    before = _snapshot(station)
    assert not station._take_located_500()
    assert station._expect_500 == 2 and not station._peer_turn_proven
    assert station._base_rx.history == [] and station._base_ready == []
    assert _snapshot(station) == before
    assert all(event[0] == 'log' for event in station.io.events)


def test_phy_base_veto_preserves_generic_receiver_and_blocks_mixed_partial(monkeypatch):
    beacon = frame(OBSERVED[0][0])
    bad = deepcopy(beacon)
    bad.crc_ok = False
    result = rx.DecodeResult([beacon, bad], [beacon, bad], observation_complete=False)
    monkeypatch.setattr(rx, 'decode_stream', lambda *args, **kwargs: result)
    read = phy._decode_base(np.empty(0), '500')
    assert read.broadcast and not read.frames
    assert phy.decode_level(np.empty(0), '500', 2) is None
    assert len(result.frames) == 2  # Not removed from CQ/passive consumers.


@pytest.mark.parametrize('mutation', ['bad_crc', 'level3', 'level4', 'pad', 'zero', 'marker', 'type'])
def test_noncanonical_or_longer_binary_data_is_not_excluded(monkeypatch, mutation):
    payload = bytearray.fromhex(OBSERVED[0][0])
    level, marker = 2, 0
    if mutation == 'pad':
        payload[18] ^= 1
    elif mutation == 'zero':
        payload[21] = 1
    elif mutation == 'type':
        payload[0] = 0x20
    elif mutation == 'marker':
        marker = 0x01
    elif mutation in ('level3', 'level4'):
        level = int(mutation[-1])
        payload.extend(bytes(rx._PAYLOAD[level] - len(payload)))
    f = frame(bytes(payload), marker, level)
    if mutation == 'bad_crc':
        f.crc_ok = False
    assert not B.is_broadcast_frame(f)
    if f.crc_ok:
        result = rx.DecodeResult([f], [f])
        monkeypatch.setattr(rx, 'decode_stream', lambda *args, **kwargs: result)
        read = phy._decode_base(np.empty(0), '500')
        assert not read.broadcast and read.frames == (f.frame_bytes[:-2],)


@pytest.mark.parametrize('level', [5, 6])
def test_broadcast_vetoes_false_high_speed_locator_not_just_base_frames(monkeypatch, level):
    f = frame(OBSERVED[0][0])
    result = rx.DecodeResult([f], [f])
    monkeypatch.setattr(rx, 'decode_stream', lambda *args, **kwargs: result)
    receiver = phy._family('500', level)[0]
    monkeypatch.setattr(receiver, 'decode_burst', lambda *args: SimpleNamespace(
        frames=(), level=level, quality=SimpleNamespace(snr_db=10.)))
    monkeypatch.setattr(phy, 'ident', lambda audio: None)
    monkeypatch.setattr(phy, '_tone_burst', lambda *args: True)
    monkeypatch.setattr(phy, '_retune', lambda audio, *args: audio)
    # No DSP: family decode, ident and tuning are replaced with fake observations.
    assert phy.decode_level(np.zeros(500000), '500', level) is None


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('complete', [False, True])
def test_mixed_real_data_and_beacon_cannot_credit_or_change_existing_debt(role, complete):
    station = _station(role)
    data = frame(b'5 <AWQ>'.ljust(22, b'\0'))
    beacon = frame(OBSERVED[0][0])
    station._resend_decoded = rx.DecodeResult([data, beacon], [data, beacon],
                                             observation_complete=complete)
    station._answer_owed = 'over'
    station._owed_block = b'existing owed block'
    station._held_answer = ('existing held answer',)
    station._txq = [b'pending outbound bytes']
    station._peer_delivery_open = True
    station._peer_over = 3
    before = _snapshot(station)
    assert not station._peer_data_over_500(np.empty(0), 10.)
    assert _snapshot(station) == before
    assert all(event[0] == 'log' for event in station.io.events)


# WB8SKP's two-packet UI broadcast to ALL, heard 2026-10-28 while linked: its
# last packet arrived without its first and was delivered as session DATA.
UI_TAIL = '50e9ce63d689143900'


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_lone_ui_tail_packet_is_not_delivered_or_answered(role):
    f = frame(UI_TAIL, 0x08, level=1)
    assert B.is_broadcast_frame(f)
    station = _station(role)
    station._resend_decoded = rx.DecodeResult([f], [f])
    before = _snapshot(station)
    assert station._peer_data_over_500(np.empty(0), 2.0) == []
    assert _snapshot(station) == before
    assert all(event[0] == 'log' for event in station.io.events)


@pytest.mark.parametrize('body', [
    phy.vara_body(b'PING', 'W9SSJ', body_len=10, level=1),
    phy.vara_body(b'P' + bytes(6), 'W9SSJ', body_len=10, level=1),
    phy.vara_body(bytes.fromhex(UI_TAIL), 'W9SSJ', 0x95, body_len=10),
    bytes.fromhex('50e9ce63d68900391408'),  # stock close, field = caller CRC
], ids=['short', 'short-zeros', 'full', 'crc-field'])
def test_session_data_led_by_a_ui_kind_byte_is_still_delivered(body):
    f = frame(body[:-1], body[-1], level=1)
    assert not B.is_broadcast_frame(f)
    station = _station('initiator')
    station._resend_decoded = rx.DecodeResult([f], [f])
    assert station._peer_data_over_500(np.empty(0), 2.0) == [body]
