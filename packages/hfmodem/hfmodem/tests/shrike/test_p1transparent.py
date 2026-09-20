# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A Winlink body is binary, and PACTOR-1's data field is not a byte pipe.

0x1E is IDLE at every position of the field and 0x1C opens a supervisor block.
MEASURED against SCS's own monitor on 2026-09-01, over renders of our own
transmitter at both speeds:

  * a full 20-byte field ending in a data 0x1E read back 19 bytes, and so did
    the same field with the 0x1E in the middle;
  * 0x1E walked through all 8 positions of a 100 Bd field and all 20 of a
    200 Bd one -- 28 of 28 came back one byte short;
  * a field of nothing but 0x1E read `LEN: 0`, an idle packet;
  * a run of 0x00-0xFF came back byte for byte APART from `1c 1d 1e 1f 20`,
    where the supervisor block swallowed all five bytes;
  * `1c 7e` delivered a 0x1E and `1c 7c` a 0x1C, each on its own two bytes,
    and both worked with the SB in one packet and the SIC in the next.

So the field carries characters, `compress.transparent` is what turns bytes
into characters, and this file is the byte-exactness that buys -- through the
render/parse pair below, and through `arq.on_host_data` over a keyed link at the
end of it. PACTOR-3 makes the same bargain with its own padding, which is why
the escape sits above the level rather than inside PACTOR-1.

Run:  pytest -q packages/hfmodem/hfmodem/tests/shrike/test_p1transparent.py
"""
from __future__ import annotations

import numpy as np

from hfmodem.shrike import compress, p1rx, pactor1, placement, spec
from hfmodem.shrike.arq import State
from hfmodem.shrike.spec import DataType

BAUDS = (100, 200)


def _field(packet: bytes) -> bytes:
    """The field of a packet that passes the receiver's own CRC gate."""
    frame = np.frombuffer(packet, dtype=np.uint8)[None, :]
    assert p1rx._crc_pass(frame)[0], packet.hex()
    return packet[1:-3]


def _link(stream: bytes, baud: int) -> bytes:
    """`stream` through a whole direction of a link: chunked into fields, sent
    as packets, and read back the way `arq.PactorArq` reads a peer's -- with
    the supervisor layer the live receive path is still missing."""
    n = pactor1.DATA_FIELD[baud]
    decoder, si = compress.Decoder(), compress.Supervisor()
    out = bytearray()
    for i in range(0, len(stream), n):
        packet = pactor1.data_packet(stream[i:i + n], baud, (i // n) & 3)
        payload = pactor1.field_bytes(_field(packet), DataType.ASCII_8BIT)
        out += si.feed(decoder.feed(payload, DataType.ASCII_8BIT))
    return bytes(out)


def test_sixty_four_kilobytes_of_binary_survive_the_link():
    """The B2F case: an lzhuf body holds 0x1E about once every 256 bytes, and
    every field position has to be able to hold one."""
    blob = np.random.default_rng(1798).integers(0, 256, 65536,
                                                dtype=np.uint8).tobytes()
    assert blob.count(spec.IDLE) > 200 and blob.count(compress.SB) > 200
    stream = compress.transparent(blob)
    for baud in BAUDS:
        n = pactor1.DATA_FIELD[baud]
        assert {i % n for i, b in enumerate(blob) if b == spec.IDLE} == set(range(n))
        assert _link(stream, baud) == blob


def test_the_unescaped_stream_does_not():
    """The control, and it is what makes the test above mean anything: the same
    bytes without `transparent` come back short."""
    blob = np.random.default_rng(1798).integers(0, 256, 4096,
                                                dtype=np.uint8).tobytes()
    assert _link(blob, 100) != blob


def test_idle_packets_deliver_nothing():
    """What an ISS with an empty buffer sends stays invisible to the host."""
    si = compress.Supervisor()
    for baud in BAUDS:
        field = _field(pactor1.data_packet(b"", baud))
        assert field == bytes([spec.IDLE]) * pactor1.DATA_FIELD[baud]
        assert pactor1.field_bytes(field, DataType.ASCII_8BIT) == b""
        assert si.feed(pactor1.field_bytes(field, DataType.ASCII_8BIT)) == b""


def test_a_coded_field_keeps_its_idle_bytes():
    """DL6MAA's 200 Bd announcement, off tape: eight bytes of Huffman and then
    twelve of `spec.TEMPLATE` resumed at its tenth byte, `1e 1e` among them.

    A real station's PACTOR-1 fill is the walking-bit pattern, not 0x1E, and
    under a coded mode 0x1E is a byte of the bit stream like any other. Taken
    out as padding, the two desynchronise the tree walk and the callsign grows
    a tail."""
    field = bytes.fromhex("1438842e2d04f3f078783c3c1e1e0f8f87c7c3e3")
    assert field[8:] == (spec.TEMPLATE[9:] + spec.TEMPLATE)[:12]
    assert pactor1.field_bytes(field, DataType.HUFFMAN) == field
    assert compress.Decoder().feed(field, DataType.HUFFMAN) == b"1dl6maa\r"


def test_the_reserved_values_survive_the_air():
    """Off the audio and not just off the bytes, at both speeds."""
    blob = b"\x1eB\x1cD"
    for baud in BAUDS:
        stream = compress.transparent(blob)
        packets = p1rx.decode_p1_packets(
            pactor1.data_signal(stream, baud, repeats=2))
        assert len(packets) == 1, [p.payload for p in packets]
        assert compress.Supervisor().feed(packets[0].payload) == blob


def test_the_pactor3_field_is_the_same_bargain():
    """And so the escape belongs above the level, not inside PACTOR-1.

    A part-filled PACTOR-3 field is padded with IDLE (`placement.field_info`)
    and the far end drops it (`spec.field_payload`, all three receive paths), so
    a payload whose last byte is a data 0x1E arrives one short there too. Where
    PACTOR-3 differs is that only the TRAILING run goes -- an interior 0x1E
    survives the field and is taken out by `Supervisor` instead, which is why
    the escape has to be what wrote it.
    """
    path = placement.SPEED_PATHS[2]
    n = path.crc_bytes - 3
    body = b"ends in idle\x1e"
    status = spec.status_byte(1)

    bare = placement.field_info(body, n, status)
    assert spec.field_payload(bare[:-1]) == body.rstrip(b"\x1e")

    wire = compress.transparent(body)
    info = placement.field_info(wire, n, status)
    assert compress.Supervisor().feed(spec.field_payload(info[:-1])) == body


def test_a_host_stream_crosses_a_keyed_pactor1_link_byte_exact():
    """The live path, over the waveform: `arq.on_host_data` at one end and the
    hostmode receive buffer at the other, every burst rendered and demodulated.

    Held in PACTOR-1, so what carries these bytes is `pactor1.data_packet` and
    what reads them is `p1rx` -- the pair the measurements above were taken
    against. The blob walks 0x18-0x23 (both reserved values in a row), puts a
    `1c 1e` pair mid-field and closes on a run of 0x1E and one ordinary byte, so
    a padding rule that guesses from the end has nothing to guess with.
    """
    from hfmodem.tests.shrike.test_qso import AudioLink, rx_of

    link = AudioLink("W9SSJ", "K7ABC", verbose=False)
    link.a.stay_in_pactor1 = link.b.stay_in_pactor1 = True
    link.a.arq.on_host_connect("W9SSJ", "K7ABC")
    link.exchange(2)
    assert link.a.arq.state == State.CONNECTED == link.b.arq.state

    blob = bytes(range(0x18, 0x24)) + b"MID\x1c\x1eEND" + bytes([spec.IDLE]) * 3 + b"Z"
    announced = len(rx_of(link.b))
    link.a.arq.on_host_data(blob)
    for _ in range(8):
        link.exchange(1)
        if rx_of(link.b)[announced:] == blob:
            return
    assert rx_of(link.b)[announced:] == blob
