# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CWP framing: round-trips, chunked reassembly, and every failure mode."""

import json

import pytest

from creance import hproto
from creance.hproto import (DATA, Deframer, Desync, End, Frame, Hello,
                            HelloAck, NotCwp, Report, pack)


def feed_all(stream: bytes) -> list[Frame]:
    return Deframer().feed(stream)


def test_pack_parse_round_trip():
    frames = feed_all(pack(DATA, b"payload") + pack(DATA, b""))
    assert frames == [Frame(DATA, b"payload"), Frame(DATA, b"")]


def test_control_round_trips():
    hello = Hello(sid="s1", call="N0CAL", scenario="unidir",
                  params={"size": 10240, "payload": "prbs9"})
    ack = HelloAck(accept=True, caps={"scenarios": ["unidir", "echo"]})
    end = End(sha256="ab" * 32, bytes=10240, dur_s=12.5)
    report = Report(sid="s1", bytes=10240, sha256="ab" * 32, dur_s=13.0)
    stream = hello.pack() + ack.pack() + end.pack() + report.pack()
    f = feed_all(stream)
    assert Hello.parse(f[0]) == hello
    assert HelloAck.parse(f[1]) == ack
    assert End.parse(f[2]) == end
    assert Report.parse(f[3]) == report


def test_reassembly_at_every_split():
    stream = (Hello(sid="s", call="C", scenario="echo").pack()
              + pack(DATA, bytes(range(100)))
              + End(sha256="00" * 32, bytes=100, dur_s=1.0).pack())
    for cut in range(1, len(stream)):
        d = Deframer()
        frames = d.feed(stream[:cut]) + d.feed(stream[cut:])
        assert [f.type for f in frames] == [hproto.HELLO, DATA, hproto.END]
        assert frames[1].payload == bytes(range(100))


def test_incomplete_waits():
    stream = pack(DATA, b"abc")
    d = Deframer()
    assert d.feed(stream[:-3]) == []
    assert d.feed(stream[-3:]) == [Frame(DATA, b"abc")]


def test_truncated_final_frame():
    d = Deframer()
    frames = d.feed(pack(DATA, b"one") + pack(DATA, b"two")[:-5])
    assert frames == [Frame(DATA, b"one")]   # the tail just waits


def test_not_cwp_at_stream_start():
    d = Deframer()
    with pytest.raises(NotCwp) as exc:
        d.feed(b"hello plain winlink peer")
    assert exc.value.buffered == b"hello plain winlink peer"


def test_not_cwp_detected_on_partial_prefix():
    d = Deframer()
    assert d.feed(b"C") == []            # still a plausible magic prefix
    with pytest.raises(NotCwp) as exc:
        d.feed(b"X")
    assert exc.value.buffered == b"CX"


def test_crc_damage_is_desync():
    stream = bytearray(pack(DATA, b"one") + pack(DATA, b"two"))
    stream[-6] ^= 0xFF                   # corrupt second frame's payload
    d = Deframer()
    with pytest.raises(Desync) as exc:
        d.feed(bytes(stream))
    assert exc.value.reason == "crc mismatch"
    assert exc.value.evidence
    assert exc.value.frames == [Frame(DATA, b"one")]   # good frames not lost
    with pytest.raises(Desync):          # dead after desync, never resyncs
        d.feed(pack(DATA, b"three"))


def test_bad_magic_mid_stream_is_desync():
    d = Deframer()
    with pytest.raises(Desync):
        d.feed(pack(DATA, b"ok") + b"garbage here")


def test_oversize_length_rejected():
    header = hproto.MAGIC + bytes([DATA]) + (5000).to_bytes(2, "big")
    with pytest.raises(Desync) as exc:
        Deframer().feed(header)
    assert "5000" in exc.value.reason
    with pytest.raises(ValueError):
        pack(DATA, b"x" * (hproto.MAX_PAYLOAD + 1))


def test_unknown_json_keys_ignored():
    body = {"v": 1, "sid": "s", "call": "C", "scenario": "echo",
            "params": {}, "future_knob": 42}
    frame = feed_all(pack(hproto.HELLO, json.dumps(body).encode()))[0]
    assert Hello.parse(frame) == Hello(sid="s", call="C", scenario="echo")


def test_unknown_frame_type_surfaced_not_fatal():
    d = Deframer()
    frames = d.feed(pack(0x7F, b"mystery") + pack(DATA, b"still fine"))
    assert frames[0].name == "0x7f"
    assert frames[1] == Frame(DATA, b"still fine")


def test_bad_json_body_is_desync():
    with pytest.raises(Desync):
        Hello.parse(Frame(hproto.HELLO, b"{not json"))
    with pytest.raises(Desync):
        Hello.parse(Frame(hproto.HELLO, b"[1,2]"))
    with pytest.raises(ValueError):      # wrong frame type is a caller bug
        Hello.parse(Frame(DATA, b"{}"))
