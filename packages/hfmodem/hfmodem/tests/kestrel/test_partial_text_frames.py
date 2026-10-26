# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Generated DATA boundaries; no captured stock frames or waveform files."""
import pytest

from hfmodem.kestrel.arq import phy


@pytest.mark.parametrize('marker_kind', range(4))
@pytest.mark.parametrize('body_len', [10, 44, 90])
@pytest.mark.parametrize('spare', [1, 2, 3, 4])
def test_near_capacity_plain_body_roundtrip_for_every_caller_marker_kind(
        marker_kind, body_len, spare):
    caller = next(f'N0T{i}' for i in range(1000)
                  if phy.crc16_genibus(f'N0T{i}'.encode()) >> 8 & 3 == marker_kind)
    payload = b'X' * (body_len-spare)
    body = phy.vara_body(payload, caller, body_len=body_len, tail=0x89)
    assert phy.vara_payload(body, caller=caller, body_len=body_len) == payload
    assert phy.over_is_last(body, caller) == (spare != 1)


def test_kind_01_overlapping_crc_keeps_its_existing_interpretation():
    caller = 'W1AW'
    key = phy.crc16_genibus(caller.encode()) >> 8
    assert key & 3 == 1
    payload = b'X' * 42
    body = payload + bytes([0x14, key])
    assert phy.vara_payload(body, caller=caller, body_len=44) == payload
    assert not phy.over_is_last(body, caller)


@pytest.mark.parametrize('caller', ['W9SSJ-14', 'W9SSJ-12', 'W9SSJ-10'])
@pytest.mark.parametrize('field', [0xa0, 0xa4, 0xb4])
def test_single_byte_text_trailer_is_not_payload_or_a_terminal_over(caller, field):
    payload = bytes(range(42))
    body = payload + b'\x14' + bytes([field])
    assert phy.vara_payload(body, caller=caller, body_len=44) == payload
    assert not phy.over_is_last(body, caller)


@pytest.mark.parametrize('length', [3, 17, 40, 41])
def test_partial_text_with_complete_trailer_still_owns_its_turn(length):
    caller = 'W9SSJ-14'
    payload = bytes(range(length))
    key = phy.crc16_genibus(caller.encode()) >> 8
    body = (payload + bytes([0x14, key])).ljust(43, b'\0') + b'\xb4'
    assert phy.vara_payload(body, caller=caller, body_len=44) == payload
    assert not phy.over_is_last(body, caller)


@pytest.mark.parametrize('field', [0x81, 0x99, 0xa1, 0xb5])
def test_full_plain_or_coded_frame_keeps_a_literal_final_trailer_byte(field):
    payload = bytes(range(42)) + b'\x14'
    body = payload + bytes([field])
    assert phy.vara_payload(body, caller='W9SSJ-14', body_len=44) == payload
    assert not phy.over_is_last(body, 'W9SSJ-14')


def test_terminal_overlapping_crc_keeps_its_existing_close_interpretation():
    # A CRC that itself has the partial-field bit pattern is an existing
    # ambiguity. Do not reclassify that historical short close as nonfinal.
    caller = next(f'N0T{i}' for i in range(1000)
                  if phy.crc16_genibus(f'N0T{i}'.encode()) >> 8 & 0x23 == 0x20)
    key = phy.crc16_genibus(caller.encode()) >> 8
    payload = b'x' * 42
    body = payload + bytes([0x14, key])
    assert phy.vara_payload(body, caller=caller, body_len=44) == payload
    assert phy.over_is_last(body, caller)
