# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P4 framing checked against retained Dragon fields and bitwise invariants."""
import hashlib

import pytest

from hfmodem.shrike import coding, p4frame
from hfmodem.tests.shrike.test_p4chirp import REAL_FIELDS


def test_published_status_layout():
    # P4 0x60 is one two-bit SPUG value, not simultaneous P3 BK/long flags.
    status = p4frame.Status.from_byte(0x63)
    assert status.counter == 3
    assert status.protocol is p4frame.ProtocolStatus.SPEED_UP
    assert status.source_mode == 0 and not status.extended
    assert p4frame.Status.from_byte(0x40).protocol is p4frame.ProtocolStatus.CHANGEOVER
    assert p4frame.Status.from_byte(0x20).protocol is p4frame.ProtocolStatus.CLOSE
    for byte in range(256):
        assert p4frame.Status.from_byte(byte).byte == byte


def test_whitening_vector_and_reset():
    expected = bytes.fromhex('cd70b7506acf508d436b7dfe1563d9a0bb27616aa6819653c9')
    assert p4frame.whiten(bytes(25)) == expected
    assert p4frame.whiten(expected) == bytes(25)
    for length in (25, 43, 206):
        data = bytes(i % 256 for i in range(length))
        assert p4frame.whiten(p4frame.whiten(data)) == data
        assert p4frame.whiten(data)[:25] == p4frame.whiten(data[:25])


def test_four_dragon_chirp_fields():
    hashes = (
        'ea0cbf3be97058b325a498ca424dacac698dffbaa92310e8a4c495440ac6dbd0',
        '7ca1bde32a6dc29aa01a3d192b7bf332b1314d8556aba09c7bc8d07400873b95',
        'a79809eaa1d3cf67cce1e58392508caf1ac2acfa3b6aaffc1821353edd9bafa2',
        'c1f9242351695ce0ccd5ab4a0b5c6b1fa37a16abb0d28e0d891764791354ae53',
    )
    for wire, status_byte, digest in zip(REAL_FIELDS, (0x63, 0x60, 0x61, 0x63), hashes):
        packet = p4frame.whiten(wire)
        assert hashlib.sha256(packet).hexdigest() == digest
        # Standard CRC residue computed independently of coding.crc16.
        state = 0xffff
        for byte in packet:
            state ^= byte
            for _ in range(8):
                state = (state >> 1) ^ (0x8408 if state & 1 else 0)
        assert state == 0xf0b8
        frame = p4frame.decode_field(wire, data_bytes=22)
        assert frame is not None and frame.status.byte == status_byte
        assert p4frame.encode_field(frame.data, frame.status) == wire
        # Every single-bit error must fail; no delivery of damaged fields.
        for bit in range(200):
            bad = bytearray(wire)
            bad[bit // 8] ^= 1 << (bit % 8)
            assert p4frame.decode_field(bytes(bad), data_bytes=22) is None


def test_crc_offset_is_whitening_not_an_alternate_crc():
    for value in (0, 1, 128, 255):
        plain = bytes([value]) * 23
        wire = p4frame.whiten(plain + coding.crc16(plain).to_bytes(2, 'little'))
        assert coding.crc16(wire[:-2]) ^ int.from_bytes(wire[-2:], 'little') == 0x53e1
        # The old renderer passes its XOR check yet decodes to different data.
        old = plain + (coding.crc16(plain) ^ 0x53e1).to_bytes(2, 'little')
        decoded = p4frame.decode_field(old, data_bytes=22)
        assert decoded is not None and decoded.data != plain[:-1]


@pytest.mark.parametrize('size', [0, 22, 40, 203])
def test_roundtrip_lengths_and_reject_truncation(size):
    data = bytes(i % 256 for i in range(size))
    status = p4frame.Status(2, 0, p4frame.ProtocolStatus.NONE)
    wire = p4frame.encode_field(data, status)
    frame = p4frame.decode_field(wire, data_bytes=size)
    assert frame.data == data and frame.status == status
    assert p4frame.decode_field(wire[:-1], data_bytes=size) is None
    assert p4frame.decode_field(wire + b'\0', data_bytes=size) is None


@pytest.mark.parametrize('kwargs', [dict(counter=-1), dict(counter=4),
    dict(counter=0, source_mode=8), dict(counter=0, protocol=4),
    dict(counter=0, extended=1), dict(counter=0.5)])
def test_invalid_status(kwargs):
    with pytest.raises(ValueError):
        p4frame.Status(**kwargs)
