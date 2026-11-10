# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.ax25.kiss import Parser, encode


@pytest.mark.parametrize("chunk", [1, 2, 7, 4096])
def test_incremental_escape_and_commands(chunk):
    payload = bytes(range(256))
    wire = b'noise' + encode(payload) + encode(b'\x0a', 1) + encode(b'two', 0, 2)
    parser = Parser()
    got = []
    for i in range(0, len(wire), chunk):
        got.extend(parser.feed(wire[i:i + chunk]))
    assert got == [(0, 0, payload), (0, 1, b'\x0a'), (2, 0, b'two')]
    assert parser.dropped == 0
    assert encode(b'\xc0\xdb') == b'\xc0\x00\xdb\xdc\xdb\xdd\xc0'


@pytest.mark.parametrize("bad", [b'\xc0\0\xdbxgarbage', b'\xc0\0\xdb', b'\xc0\0' + bytes(2049)])
def test_malformed_frame_resynchronises(bad):
    parser = Parser()
    assert parser.feed(bad + encode(b'good')) == [(0, 0, b'good')]
    assert parser.dropped == 1


def test_size_boundary():
    assert Parser().feed(encode(bytes(2048))) == [(0, 0, bytes(2048))]
    with pytest.raises(ValueError):
        encode(bytes(2049))
