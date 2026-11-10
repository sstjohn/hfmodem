# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra's integrity primitives, validated byte-exact against the reference
vectors generated from ardopcf's verbatim functions."""

from __future__ import annotations


from hfmodem.besra import crc
from . import groundtruth as gt

pytestmark = gt.requires_reference


def test_crc16_matches_reference():
    vectors = gt.reference_vectors("crc16")
    assert vectors
    for data_hex, want in vectors:
        assert crc.crc16(bytes.fromhex(data_hex)) == int(want, 16), data_hex


def test_crc8_matches_reference():
    vectors = gt.reference_vectors("crc8")
    assert vectors
    for text, want in vectors:
        assert crc.crc8(text.encode("ascii")) == int(want, 16), text


def test_type_parity_all_256():
    vectors = gt.reference_vectors("parity")
    assert len(vectors) == 256
    for type_hex, want in vectors:
        assert crc.type_parity(int(type_hex, 16)) == int(want), type_hex


def test_crc16_frametype_roundtrip():
    for ftype in (0x40, 0x4A, 0x74):
        block = crc.append_crc16_frametype(b"besra ardop", ftype)
        assert crc.check_crc16_frametype(block, ftype)
        assert not crc.check_crc16_frametype(block, ftype ^ 0x01)
