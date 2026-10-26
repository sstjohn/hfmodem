# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Stock byte oracles and immutable recordings for unlinked UI broadcasts."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import hilbert
from hfmodem.contact.broadcast import Broadcast
from hfmodem.kestrel.rx import broadcast as B
from hfmodem.kestrel.rx.cq_stream import CQDetector
from hfmodem.kestrel.tx.broadcast import synth_broadcast
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.tests.evidence import CORPUS, require_input

FIXTURES = json.loads(
    (
        Path(__file__).parent / "fixtures/varac-broadcast-1024/provenance.json"
    ).read_text()
)
PACKETS = [
    (1, "W9SSJ", "ALL", b"A", 1, ["50a979134ca0010414088b78"]),
    (1, "W9SSJ", "ALL", b"AA", 1, ["50a979134ca001060809b327"]),
    (
        1,
        "W9SSJ",
        "ALL",
        b"UI1 TEST",
        1,
        ["d050a979134ca0015609c645", "504b141522d3a9140008bfff"],
    ),
    (
        0,
        "W9SSJ",
        "ALL",
        b"UI0 TEST",
        1,
        ["c090a979134ca0015609b2b8", "404b041522d3a914000816dd"],
    ),
    (
        1,
        "KC1ABCD",
        "ALL",
        b"SEVEN",
        1,
        ["d070a8b0dc0420c405095816", "5038b5a2ceff1400000847da"],
    ),
    (
        1,
        "W9SSJ-7",
        "ALL",
        b"SSID",
        1,
        ["d030a979134ca01d4e0982b8", "509c989f140000000008f4bf"],
    ),
    (
        1,
        "W9SSJ",
        "TEST",
        b"DEST",
        1,
        ["d0f05054d4001791340917cc", "50ca001122d3a91400081871"],
    ),
    (
        1,
        "W9SSJ",
        "ALL",
        "Café ☃".encode(),
        1,
        ["d040a979134ca0028609d9cc", "50c2cd875241c5310609b7d8"],
    ),
    (
        1,
        "W9SSJ",
        "ALL",
        (b"0123456789 ABCDEFGHIJ " * 4)[:77],
        4,
        [
            "d0a0a979134ca000c18b266d1ab66ee1ca0830a1c48b1a3c8932903062c99b46ad9bb872820c287122c68f01405c",
            "50224ca40c18b266d1ab66ee1ca0830a1c48b1a3c8932903062c99b46ad9bb87283f140000000000000000003a71",
        ],
    ),
]


@pytest.mark.parametrize("command,source,destination,info,level,hexes", PACKETS)
def test_exact_stock_packet_bytes(command, source, destination, info, level, hexes):
    message = Broadcast(command, source, destination, info)
    actual_level, frames = B.packetize(message)
    assert actual_level == level and [f.hex() for f in frames] == hexes
    assert B.parse_packets(frames, level=level) == message


def replay(detector, x):
    got = []
    for at in range(0, len(x), 4800):
        got += detector.feed(x[at : at + 4800])
    return got


@pytest.mark.parametrize("prov", FIXTURES, ids=lambda p: p["name"])
def test_stock_audio_exact_host_bytes(prov):
    path = require_input(CORPUS / prov["corpus_path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prov["sha256"]
    rate, x = wavfile.read(path)
    assert rate == 48000
    detector = CQDetector()
    got = replay(detector, np.r_[x, np.zeros(6 * 48000)])
    assert len(got) == 1
    assert (got[0].command, got[0].source, got[0].destination, got[0].info) == (
        prov["command"],
        prov["source"],
        prov["destination"],
        bytes.fromhex(prov["info_hex"]),
    )
    assert detector.turbo_attempts <= 4
    assert len(detector._buffer) <= 16 * 12000


@pytest.mark.parametrize(
    "change", ["crc", "checksum", "missing", "swapped", "third", "marker"]
)
def test_incomplete_or_corrupt_group_never_delivers(change):
    level, frames = B.packetize(Broadcast(1, "W9SSJ", "ALL", b"UI1 TEST"))
    if change == "crc":
        frames[1] = frames[1][:-1] + bytes([frames[1][-1] ^ 1])
    elif change == "missing":
        frames = frames[:1]
    elif change == "swapped":
        frames = frames[::-1]
    elif change == "third":
        frames += frames[-1:]
    else:
        b = bytearray(frames[0][:-2])
        b[1 if change == "checksum" else -1] ^= 16
        frames[0] = bytes(b) + crc16_genibus(b).to_bytes(2, "big")
    assert B.parse_packets(frames, level=level) is None


def test_identity_and_two_packet_limits_are_explicit():
    with pytest.raises(ValueError, match="truncates"):
        B.packetize(Broadcast(1, "X-W9SSJ", "ALL", b"HELLO"))
    with pytest.raises(ValueError, match="two physical"):
        B.packetize(Broadcast(1, "KC1ABCD", "ALL", b"\xc3" * 77))


@pytest.mark.parametrize(
    "info", [b"<Q:75>", b"\xff\x80binary", b"REPEAT", b"A\x7f", b"A\x00", b"\x00\x00"]
)
def test_received_info_is_opaque_and_tags_do_not_execute(info):
    message = Broadcast(1, "W9SSJ", "ALL", info)
    level, frames = B.packetize(message)
    assert B.parse_packets(frames, level=level).info == info


@pytest.mark.parametrize("iq", [False, True])
@pytest.mark.parametrize("info", ["Café ☃".encode(), b"x" * 25, b"x" * 77],
                         ids=["utf8", "long-ui1", "long-ui4"])
def test_generated_stream_and_duplicate_messages_remain_independent(iq, info):
    message = Broadcast(1, "KC1ABCD", "ALL", info)
    wave = synth_broadcast(message, amplitude=0.2)
    x = np.r_[np.zeros(24000), wave, np.zeros(6 * 48000), wave, np.zeros(6 * 48000)]
    if iq:
        x = hilbert(x) * np.exp(-2j * np.pi * 1500 * np.arange(len(x)) / 48000)
    got = replay(CQDetector(), x)
    assert [(v.source, v.info) for v in got] == [(message.source, message.info)] * 2


def test_headerless_noise_does_not_spend_turbo_or_grow_storage():
    d = CQDetector()
    x = np.random.default_rng(1024).normal(0, 0.1, 20 * 48000)
    assert replay(d, x) == [] and d.turbo_attempts == 0
    assert len(d._buffer) <= 16 * 12000


@pytest.mark.parametrize(
    "info", [b"A", b"UI1 TEST", b"ABCDEFGHIJKLMNOPQRSTUVWXY", b"x" * 77]
)
@pytest.mark.parametrize("role", ["initiator", "responder"])
def test_ui_is_never_delivered_as_connected_application_data(info, role):
    from hfmodem.kestrel.rx import varahf500 as R, beacon
    from .test_bw500_locator_ownership import _station, _snapshot

    level, packets = B.packetize(Broadcast(1, "W9SSJ", "ALL", info))
    f = R.FrameResult(0, (), 0, 1.0, True, packets[0], level=level)
    assert beacon.is_broadcast_frame(f)
    station = _station(role)
    station._resend_decoded = R.DecodeResult([f], [f], observation_complete=False)
    before = _snapshot(station)
    assert station._peer_data_over_500(np.empty(0), 5.1) == []
    assert _snapshot(station) == before
    assert all(event[0] == "log" for event in station.io.events)


@pytest.mark.parametrize(
    "mutation", ["checksum_form", "type", "marker", "address", "physical_crc"]
)
def test_ui_admission_requires_more_than_an_initial_type_byte(mutation):
    from hfmodem.kestrel.rx import varahf500 as R

    level, frames = B.packetize(Broadcast(1, "W9SSJ", "ALL", b"UI1 TEST"))
    raw = bytearray(frames[0][:-2])
    if mutation == "checksum_form":
        raw[1] |= 1
    elif mutation == "type":
        raw[0] = 0xD1
    elif mutation == "marker":
        raw[-1] = 0
    elif mutation == "address":
        raw[2] = 0xFF
    wire = bytes(raw) + crc16_genibus(raw).to_bytes(2, "big")
    if mutation == "physical_crc":
        wire = wire[:-1] + bytes([wire[-1] ^ 1])
    f = R.FrameResult(0, (), 0, 1.0, True, wire, level=level)
    assert not B.is_broadcast_frame(f)
