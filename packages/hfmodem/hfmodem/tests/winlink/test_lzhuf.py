# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The FBB compression codec against material this project did not produce.

The arbiter is a real Winlink message and its compressed form from outside the
project (corpora.real_pair): the decoder must reproduce the message from the
compressed bytes, and the encoder — same algorithm, same match rules — turns
out to reproduce the compressed file byte for byte, which pins every detail
down to the choice among equal-length matches. Round trips and planted
corruptions carry the rest: each guard is shown a case where it must go red.
"""
from __future__ import annotations

import pytest

from hfmodem.tests.winlink import corpora
from hfmodem.winlink import lzhuf
from hfmodem.winlink.lzhuf import LzhufError, compress, crc16, decompress


def test_crc16_check_value():
    """CRC-16/XMODEM's published check value."""
    assert crc16(b"123456789") == 0x31C3


# --------------------------------------------------------------------------- #
# The external pair: decode direction first, because receive is what works today.
def test_real_compressed_message_decodes_byte_exact():
    plain, packed = corpora.real_pair()
    assert decompress(packed) == plain


def test_real_message_compresses_byte_exact():
    plain, packed = corpora.real_pair()
    assert compress(plain) == packed


def test_declared_size_matches_the_real_message():
    plain, packed = corpora.real_pair()
    assert decompress(packed, expected_size=len(plain)) == plain


def test_the_classic_window_size_writes_garbage(monkeypatch):
    """The classic LZHUF window is 4096; the Winlink codec's is 2048. The
    direction that bites is encode: back-references are relative, so a wider
    decoder happens to read a narrower stream, but a 4096-window encoder emits
    positions a 2048-window decoder wraps to the wrong bytes — and the CRC,
    which covers the compressed bytes rather than the output, passes anyway.
    This is the planted counterexample proving the window size is measured,
    not assumed."""
    plain, _ = corpora.real_pair()
    monkeypatch.setattr(lzhuf, "N", 4096)
    packed_classic = compress(plain)
    monkeypatch.undo()
    assert decompress(packed_classic) != plain


# --------------------------------------------------------------------------- #
# Round trips (self-consistency, kept honest by the external pair above).
@pytest.mark.parametrize("payload", [
    b"",
    b"a",
    b"Subject: hello\r\n\r\nshort message\r\n",
    b"ab" * 5000,                       # long matches, window wrap
    bytes(range(256)) * 40,             # every literal
])
def test_roundtrip(payload):
    assert decompress(compress(payload)) == payload


def test_roundtrip_survives_tree_rebuild():
    """MAX_FREQ is 0x8000: past ~32k symbols the adaptive tree halves its
    counts, and both coders must do it on the same symbol."""
    import random
    rng = random.Random(7)
    payload = bytes(rng.randrange(256) for _ in range(40000))
    assert decompress(compress(payload)) == payload


# --------------------------------------------------------------------------- #
# Planted corruptions: every one must go red.
def test_a_flipped_bit_fails_the_crc():
    blob = bytearray(compress(b"the quick brown fox jumps over the lazy dog"))
    blob[10] ^= 0x04
    with pytest.raises(LzhufError, match="CRC"):
        decompress(bytes(blob))


def test_a_truncated_stream_fails_the_crc():
    blob = compress(b"the quick brown fox jumps over the lazy dog" * 10)
    with pytest.raises(LzhufError, match="CRC"):
        decompress(blob[:-3])


def test_a_size_lie_is_refused_when_the_proposal_disagrees():
    blob = compress(b"hello world, this is mail")
    with pytest.raises(LzhufError, match="proposal"):
        decompress(blob, expected_size=10)


def test_too_short_to_be_a_stream():
    with pytest.raises(LzhufError):
        decompress(b"\x00\x01\x02")


def test_an_absurd_declared_size_is_refused_before_decoding():
    """The size field is 32 bits off the air; nothing may decode toward it
    unless a proposal vouches for it."""
    stream = bytes((0, 0, 0, 8))       # 128 MiB declared, no data
    crc = crc16(stream)
    blob = bytes((crc & 0xFF, crc >> 8)) + stream
    with pytest.raises(LzhufError, match="not a message"):
        decompress(blob)
