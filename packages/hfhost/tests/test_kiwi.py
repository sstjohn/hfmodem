# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import socket
import struct
import threading
import time

import pytest

from hfhost import kiwi


def _frame(payload: bytes, opcode: int = 0x2, fin: bool = True) -> bytes:
    n = len(payload)
    head = bytes([(0x80 if fin else 0) | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 1 << 16:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    return head + payload


def _msg(text: str) -> bytes:
    return _frame(b"MSG" + b"\x00" + text.encode())


def _snd(samples: bytes, *, iq: bool = True, flags: int = 0x08) -> bytes:
    head = b"SND" + struct.pack("<BI", flags, 7) + struct.pack(">H", 1000)
    return _frame(head + (bytes(10) if iq else b"") + samples)


class _Socket:
    """A Kiwi's side of the conversation, scripted; nothing leaves the process."""

    def __init__(self, reply: bytes, chunk: int = 7):
        self.reply, self.chunk = reply, chunk
        self.sent = bytearray()

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        out, self.reply = self.reply[:min(n, self.chunk)], self.reply[self.chunk:]
        return out

    def close(self):
        pass

    def frames(self):
        """Every masked frame the client sent, as ``(opcode, payload)``."""
        raw, out = bytes(self.sent[self.sent.index(b"\r\n\r\n") + 4:]), []
        while raw:
            n = raw[1] & 0x7F
            at = 2
            if n == 126:
                n, at = struct.unpack(">H", raw[2:4])[0], 4
            mask, body = raw[at:at + 4], raw[at + 4:at + 4 + n]
            out.append((raw[0] & 0x0F, bytes(b ^ mask[i % 4] for i, b in enumerate(body))))
            raw = raw[at + 4 + n:]
        return out

    def commands(self):
        return [body.decode() for opcode, body in self.frames() if opcode == 0x1]


HELLO = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n"


def _open(reply: bytes, **kw):
    sock = _Socket(HELLO + reply)
    ws = kiwi.WebSocket.handshake(sock, "kiwi.example", 8073, "/0/SND")
    return sock, kiwi.KiwiStream(ws, 7104.0, name="kiwi.example", **kw)


def test_the_stream_tunes_iq_uncompressed_with_agc_off_and_yields_raw_samples():
    iq = struct.pack(">4h", 100, -100, 200, -200)
    sock, stream = _open(_msg("version_maj=1 version_min=812")
                         + _msg("audio_rate=12000 sample_rate=12001.135")
                         + _snd(iq) + _snd(iq[::-1]))
    stream.start()
    assert stream.sample_rate == 12001.135
    assert list(stream.blocks()) == [iq, iq[::-1]]
    sent = sock.commands()
    assert sent[0] == "SET auth t=kiwi p="
    assert "SET AR OK in=12000 out=44100" in sent
    assert "SET mod=iq low_cut=-6000 high_cut=6000 freq=7104.000" in sent
    assert "SET agc=0 hang=0 thresh=-100 slope=6 decay=1000 manGain=50" in sent
    assert "SET compression=0" in sent
    assert sock.sent.startswith(b"GET /0/SND HTTP/1.1\r\n")


def test_a_sideband_stream_has_no_gps_header_to_strip():
    audio = struct.pack(">3h", 1, 2, 3)
    _, stream = _open(_msg("sample_rate=12000") + _snd(audio, iq=False, flags=0),
                      mode="usb", lo_hz=0, hi_hz=3000)
    assert list(stream.start().blocks()) == [audio]


def test_a_kiwi_with_a_frequency_offset_is_tuned_at_baseband():
    sock, stream = _open(_msg("freq_offset=7000.000")
                         + _msg("sample_rate=12000") + _snd(b"\0\0\0\0"))
    stream.start()
    assert "SET mod=iq low_cut=-6000 high_cut=6000 freq=104.000" in sock.commands()


def test_continuation_frames_are_joined_and_pings_answered():
    part = b"SND" + struct.pack("<BI", 0x08, 1) + struct.pack(">H", 0) + bytes(10)
    reply = (_msg("sample_rate=12000") + _frame(part, fin=False)
             + _frame(b"", opcode=0x9) + _frame(b"\x00\x01\x00\x02", opcode=0x0))
    sock, stream = _open(reply)
    assert list(stream.start().blocks()) == [b"\x00\x01\x00\x02"]
    assert (0xA, b"") in sock.frames()


def test_a_full_kiwi_says_so():
    _, stream = _open(_msg("too_busy=4"))
    with pytest.raises(kiwi.KiwiError, match="all 4 channels are taken"):
        stream.start()


def test_compressed_audio_is_refused_rather_than_decoded_as_pcm():
    _, stream = _open(_msg("sample_rate=12000") + _snd(b"\0\0", iq=False, flags=0x10))
    with pytest.raises(kiwi.KiwiError, match="compressed"):
        stream.start()


def test_a_refused_upgrade_names_the_status():
    with pytest.raises(kiwi.KiwiError, match="404"):
        kiwi.WebSocket.handshake(_Socket(b"HTTP/1.1 404 Not Found\r\n\r\n"),
                                 "kiwi.example", 8073, "/wb/0/SND")


def test_a_kiwi_that_closes_before_audio_is_an_error_not_an_empty_stream():
    _, stream = _open(_msg("sample_rate=12000") + _frame(b"", opcode=0x8))
    with pytest.raises(kiwi.KiwiError, match="before sending any audio"):
        stream.start()


def _redirecting_server(replies):
    import socket
    import threading
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    hosts = []

    def serve():
        for reply in replies:
            conn, _ = server.accept()
            hosts.append(conn.recv(4096).split(b"Host: ")[1].split(b"\r\n")[0])
            conn.sendall(reply)
            conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], hosts


def test_a_redirect_is_followed_to_the_new_host_keeping_port_and_path(capsys):
    port, hosts = _redirecting_server([
        b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://localhost\r\n\r\n", HELLO])
    ws = kiwi.WebSocket.open("127.0.0.1", port, "/0/SND")
    assert ws.host == "localhost"
    assert hosts == [f"127.0.0.1:{port}".encode(), f"localhost:{port}".encode()]
    assert "localhost" in capsys.readouterr().out


def test_redirects_stop_after_a_couple_of_hops():
    again = b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://127.0.0.1\r\n\r\n"
    port, hosts = _redirecting_server([again] * 3)
    with pytest.raises(kiwi.KiwiError, match="too many redirects"):
        kiwi.WebSocket.open("127.0.0.1", port, "/0/SND")
    assert len(hosts) == 3


def test_closing_wakes_a_reader_when_the_kiwi_never_answers():
    server = socket.create_server(("127.0.0.1", 0))
    with server, socket.create_connection(server.getsockname()) as sock:
        silent, _ = server.accept()
        with silent:
            ws = kiwi.WebSocket(sock)
            reading = threading.Event()

            def read():
                reading.set()
                try:
                    list(ws.messages())
                except OSError:
                    pass

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            assert reading.wait(2.0)
            time.sleep(0.1)
            ws.close()
            reader.join(2.0)
            assert not reader.is_alive()


def test_a_frame_longer_than_any_kiwi_message_is_refused_before_it_is_read():
    _, stream = _open(b"\x82\x7f" + struct.pack(">Q", 1 << 40))
    with pytest.raises(kiwi.KiwiError, match="past any message"):
        stream.start()


def test_continuations_may_not_add_up_past_the_ceiling():
    half = bytes(kiwi.MAX_MESSAGE // 2 + 1)
    sock, stream = _open(_frame(half, fin=False) + _frame(half, opcode=0x0))
    sock.chunk = 1 << 16
    with pytest.raises(kiwi.KiwiError, match="past any message"):
        stream.start()


def test_empty_fragments_cost_nothing_to_keep():
    """A run of empty continuations is a slow stream, not a growing list."""
    part = b"SND" + struct.pack("<BI", 0x08, 1) + struct.pack(">H", 0) + bytes(10)
    reply = (_msg("sample_rate=12000") + _frame(part, fin=False)
             + _frame(b"", opcode=0x0, fin=False) * 20000 + _frame(b"\x00\x01", opcode=0x0))
    sock, stream = _open(reply)
    sock.chunk = 1 << 16
    assert list(stream.start().blocks()) == [b"\x00\x01"]


@pytest.mark.parametrize("frame", [_frame(b"x", opcode=0xB, fin=False),
                                   _frame(b"", opcode=0x9, fin=False)])
def test_reserved_opcodes_and_fragmented_control_frames_are_refused(frame):
    _, stream = _open(frame * 3)
    with pytest.raises(kiwi.KiwiError, match="opcode"):
        stream.start()


@pytest.mark.parametrize("text", ["sample_rate=x", "audio_rate=inf", "sample_rate=nan",
                                  "sample_rate=1e999", "sample_rate=0", "audio_rate=-12000",
                                  "freq_offset=inf"])
def test_an_unreadable_number_is_a_kiwi_error(text):
    _, stream = _open(_msg(text))
    with pytest.raises(kiwi.KiwiError, match="unreadable"):
        stream.start()


def test_a_snd_message_without_its_header_is_a_kiwi_error():
    _, stream = _open(_msg("sample_rate=12000") + _frame(b"SND"))
    with pytest.raises(kiwi.KiwiError, match="short of its header"):
        stream.start()


def test_an_unreadable_redirect_is_a_kiwi_error():
    reply = b"HTTP/1.1 302 Found\r\nLocation: http://[\r\n\r\n"
    with pytest.raises(kiwi.KiwiError, match="unreadable redirect"):
        kiwi.WebSocket.handshake(_Socket(reply), "kiwi.example", 8073, "/0/SND")


def test_a_control_frame_past_its_125_bytes_is_refused():
    _, stream = _open(_frame(bytes(126), opcode=0x9))
    with pytest.raises(kiwi.KiwiError, match="past any message"):
        stream.start()


@pytest.mark.parametrize("terminated", [False, True])
def test_an_upgrade_reply_past_the_header_limit_is_refused(terminated):
    reply = b"HTTP/1.1 101 \r\nX: " + bytes(kiwi.MAX_HEADER) + (b"\r\n\r\n" if terminated else bytes(1 << 20))
    with pytest.raises(kiwi.KiwiError, match="bytes"):
        kiwi.WebSocket.handshake(_Socket(reply, chunk=1 << 20), "kiwi.example", 8073, "/0/SND")


@pytest.mark.parametrize("split", range(5))
def test_an_upgrade_reply_at_the_header_limit_is_read_however_it_splits(split):
    head = b"HTTP/1.1 101 Switching\r\n"
    head += b"X: " + b"a" * (kiwi.MAX_HEADER - len(head) - 3)
    reply = head + b"\r\n\r\n" + _msg("next")
    sock = _Socket(b"", chunk=1 << 20)
    parts = [reply[:len(head) + split], reply[len(head) + split:]]
    sock.recv = lambda n: parts.pop(0) if parts else b""
    ws = kiwi.WebSocket.handshake(sock, "kiwi.example", 8073, "/0/SND")
    assert next(ws.messages()) == b"MSG\x00next"


def _at(address: str):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, (address, 8073))


def _redirecting_to(target: str, monkeypatch, **dns) -> list:
    addresses = {host.replace("_", "."): [_at(a) for a in found] for host, found in dns.items()}
    monkeypatch.setattr(kiwi, "_resolve", lambda host, port: addresses[host])
    connected = []

    def connect(found, timeout):
        connected.append([sockaddr[0] for _, sockaddr in found])
        return _Socket(HELLO if len(connected) > 1 else
                       f"HTTP/1.1 302 Found\r\nLocation: http://{target}:8073/\r\n\r\n".encode())

    monkeypatch.setattr(kiwi, "_connect", connect)
    return connected


@pytest.mark.parametrize("address", ["127.0.0.1", "192.168.8.1", "169.254.1.1",
                                     "::1", "fe80::1%en0", "10.0.0.5"])
def test_a_public_kiwi_cannot_redirect_into_the_station_network(address, monkeypatch):
    connected = _redirecting_to("inside.example", monkeypatch, kiwi_example=["8.8.8.8"],
                                inside_example=["1.1.1.1", address])
    with pytest.raises(kiwi.KiwiError, match="private address refused"):
        kiwi.WebSocket.open("kiwi.example", 8073, "/0/SND")
    assert connected == [["8.8.8.8"]]


def test_a_name_that_also_resolves_inside_is_still_a_public_kiwi(monkeypatch):
    connected = _redirecting_to("localhost", monkeypatch,
                                kiwi_example=["8.8.8.8", "10.0.0.1"], localhost=["127.0.0.1"])
    with pytest.raises(kiwi.KiwiError, match="private address refused"):
        kiwi.WebSocket.open("kiwi.example", 8073, "/0/SND")
    assert connected == [["8.8.8.8", "10.0.0.1"]]


def test_a_public_kiwi_may_redirect_to_another_public_host(monkeypatch):
    connected = _redirecting_to("other.example", monkeypatch,
                                kiwi_example=["8.8.8.8"], other_example=["1.1.1.1"])
    assert kiwi.WebSocket.open("kiwi.example", 8073, "/0/SND").host == "other.example"
    assert connected == [["8.8.8.8"], ["1.1.1.1"]]


def test_a_redirect_to_a_name_no_resolver_can_encode_is_a_kiwi_error():
    with pytest.raises(kiwi.KiwiError, match="not a host name"):
        kiwi._resolve("a" * 64 + ".example", 8073)


def test_a_fractional_sample_rate_is_kept():
    _, stream = _open(_msg("sample_rate=12000.049") + _snd(b"\0\0"))
    assert stream.start().sample_rate == 12000.049


def test_a_link_local_address_keeps_its_scope_through_resolution():
    index, name = socket.if_nameindex()[0]
    (family, sockaddr), = kiwi._resolve(f"fe80::1%{name}", 8073)
    assert family == socket.AF_INET6 and sockaddr[3] == index
