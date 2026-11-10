# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import socket

import pytest

from hfhost.ptc import (DATA, MSG, OK, Deframer, Packet, PtcClient, PtcError,
                        crc16, stuff, unstuff)


def test_crc_is_canonical_x25():
    """0x906e for '123456789' is the published check value for CRC-16/X-25.

    The SCS manual's worked example (04 01 01 71 71 -> lo 213, hi 153) is in
    decimal, as its #170 #170 header is; read as hex it misses by a mile, which
    is the whole trap. shrike's crc16 agrees on both vectors.
    """
    assert crc16(b"123456789") == 0x906E
    assert crc16(bytes([4, 1, 1, 71, 71])) == 0x99D5


@pytest.mark.parametrize("raw", [
    b"", b"\xaa", b"\xaa\xaa", b"\x01\xaa\x02", b"\xaa" * 8, bytes(range(256)),
])
def test_stuffing_round_trips(raw):
    assert unstuff(stuff(raw)) == raw


def test_stuffing_prevents_a_false_header():
    """The point of the stuffing: no AA AA can occur inside a frame body."""
    assert b"\xaa\xaa" not in stuff(b"\xaa\xaa\xaa")


def test_packet_encodes_with_header_and_crc():
    frame = Packet(4, True, b"L 4").encode()
    assert frame[:2] == b"\xaa\xaa"
    body = unstuff(frame[2:])
    assert body[0] == 4 and body[1] & 1 == 1        # channel, command flag
    assert body[2] == 2                             # len - 1
    assert body[3:6] == b"L 4"
    crc = body[6] | (body[7] << 8)
    assert crc == crc16(body[:6])


def test_oversize_packet_is_refused_before_the_wire():
    with pytest.raises(PtcError):
        Packet(4, False, b"\x00" * 257).encode()


def _frame(channel: int, code: int, body: bytes = b"") -> bytes:
    raw = bytes([channel, code]) + body
    crc = crc16(raw)
    return b"\xaa\xaa" + stuff(raw + bytes([crc & 0xFF, crc >> 8]))


def test_deframer_reads_each_response_shape():
    d = Deframer()
    out = d.feed(_frame(0, OK)
                 + _frame(4, MSG, b"hello\x00")
                 + _frame(4, DATA, bytes([3]) + b"abcd"))
    assert [r.code for r in out] == [OK, MSG, DATA]
    assert out[1].text == "hello"
    assert out[2].data == b"abcd"


def test_deframer_reassembles_across_chunk_boundaries():
    whole = _frame(4, MSG, b"split me\x00")
    d = Deframer()
    got = []
    for i in range(len(whole)):
        got += d.feed(whole[i:i + 1])
    assert len(got) == 1 and got[0].text == "split me"


def test_deframer_survives_leading_garbage():
    d = Deframer()
    out = d.feed(b"\x01\x02\x03" + _frame(4, MSG, b"ok\x00"))
    assert [r.text for r in out] == ["ok"]


def test_a_corrupt_crc_yields_nothing_rather_than_a_wrong_frame():
    good = bytearray(_frame(4, MSG, b"payload\x00"))
    good[-1] ^= 0xFF
    assert Deframer().feed(bytes(good)) == []


def test_data_containing_the_header_byte_survives():
    """AA in payload is what the stuffing exists for; a decoder that misses it
    resynchronises on a header that is not there."""
    payload = b"\xaa\xaa\xaa"
    out = Deframer().feed(_frame(4, DATA, bytes([len(payload) - 1]) + payload))
    assert out[0].data == payload


class _Pair:
    """A socketpair standing in for a serial port: read/write/close is all the
    client asks of a transport, so a pty is not needed to exercise it."""

    def __init__(self):
        self.a, self.b = socket.socketpair()
        self.a.setblocking(False)
        self.b.setblocking(False)

    def client_stream(self):
        pair = self

        class S:
            def read(self, n): 
                try:
                    return pair.a.recv(n)
                except BlockingIOError:
                    return b""
            def write(self, data): pair.a.sendall(data)
            def close(self): pair.a.close()
        return S()

    def modem_recv(self, n=4096):
        try:
            return self.b.recv(n)
        except BlockingIOError:
            return b""

    def modem_send(self, data):
        self.b.sendall(data)

    def close(self):
        self.a.close()
        self.b.close()


def test_client_sends_terminal_commands_and_reads_the_reply():
    pair = _Pair()
    c = PtcClient(pair.client_stream())
    try:
        import threading
        def answer():
            import time
            time.sleep(0.05)
            pair.modem_send(b"cmd: ")
        threading.Thread(target=answer, daemon=True).start()
        reply = c.command("MYcall N0CRE", timeout=1.0)
        assert pair.modem_recv().startswith(b"MYcall N0CRE\r")
        assert "cmd:" in reply
    finally:
        c.close()
        pair.close()


def test_client_round_trips_hostmode_frames():
    pair = _Pair()
    c = PtcClient(pair.client_stream(), channel=31)
    try:
        c.cmd("L 31", channel=0)
        sent = pair.modem_recv()
        assert sent[:2] == b"\xaa\xaa"
        pair.modem_send(_frame(0, MSG, b"0 0 0 0 0 0\x00"))
        got = c.poll(timeout=1.0)
        assert [r.text for r in got] == ["0 0 0 0 0 0"]
    finally:
        c.close()
        pair.close()


def test_write_data_splits_at_the_dialect_limit():
    pair = _Pair()
    c = PtcClient(pair.client_stream())
    try:
        c.write_data(b"x" * 600)
        raw = pair.modem_recv(65536)
        assert raw.count(b"\xaa\xaa") == 3      # 256 + 256 + 88
    finally:
        c.close()
        pair.close()


def test_one_corrupt_frame_does_not_deafen_the_link_forever():
    """A header whose body never checks out sits at offset 0 for the life of the
    link unless something bounds the wait: every later frame queues behind it,
    the buffer grows without bound, and the candidate scan is quadratic in what
    it holds. Corrupting any of the ten non-header byte positions does it."""
    good = bytearray(_frame(4, MSG, b"payload\x00"))
    for i in range(2, len(good)):
        broken = bytearray(good)
        broken[i] ^= 0xFF
        d = Deframer()
        d.feed(bytes(broken))
        after = d.feed(_frame(4, MSG, b"after\x00") + _frame(0, OK))
        assert [r.code for r in after] == [MSG, OK], (
            f"a corrupt byte at {i} deafened the deframer")
        assert d.resyncs >= 1, "a dropped frame was not counted"


def test_a_stream_that_never_frames_does_not_grow_without_bound():
    d = Deframer()
    for _ in range(400):
        d.feed(b"\xaa\xaa" + bytes(254))
    assert len(d._buf) <= d.MAX_FRAME + 2, f"buffer reached {len(d._buf)} bytes"
