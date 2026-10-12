# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""D_PDUs, C_PDU link control and S_ primitives, against the annexes."""
from __future__ import annotations

import pytest

from hfmodem.bateleur.s5066 import dpdu as D
from hfmodem.bateleur.s5066 import primitives as P

HEAD = bytes([0xF0, 0x00, 0x00, 0x47, 0x05, 0x64, 0x02])


def test_the_annex_s_worked_header():
    """C.3.2.8: a WARNING from 0.0.0.100 to 0.0.0.5, reason 2, and the
    bytes the annex gives for it, sync to CRC."""
    w = D.DPDU(D.WARNING, dest=5, src=100, rx_type=0, reason=2)
    assert D.crc16(HEAD) == 0x1E5F
    assert D.encode(w) == bytes([0x90, 0xEB]) + HEAD + bytes([0x5F, 0x1E])


def test_the_annex_s_worked_segment_crc():
    """C.3.2.11: the 32-bit CRC over the same seven bytes."""
    assert D.crc32(HEAD) == 0xF4178F95


@pytest.mark.parametrize("kind", sorted(D.NAMES))
def test_every_type_round_trips(kind):
    p = D.DPDU(kind, dest=D.address("1.2.3.4"), src=D.address("0.0.9.9"), eow=0x4A5, eot=17,
               start=True, end=kind != D.NON_ARQ, in_order=True, seq=201, cpdu_id=9,
               rx_lwe=250, acks=D.ack_bitmap(250, 4, {251, 3, 4}), reset_win_cmnd=True,
               new_rx_lwe=12, reset_id=7, valid=True, frame_id=33, extended=b"xy",
               group=kind == D.NON_ARQ, cpdu_size=900, offset=400, window=77, rx_type=3,
               reason=1, data=b"segment" * 20 if kind in D.DATA_TYPES else b"")
    q = D.decode(D.encode(p))
    assert (q.type, q.dest, q.src, q.eow, q.eot) == (kind, p.dest, p.src, 0x4A5, 17)
    assert len(D.encode(p)) == D.size(p)
    if kind in D.DATA_TYPES:
        assert q.data == p.data and q.data_ok
    if kind in (D.ACK_ONLY, D.DATA_ACK, D.EXP_ACK_ONLY):
        assert sorted(D.acked(q.rx_lwe, q.acks)) == [3, 4, 251]
    if kind == D.NON_ARQ:
        assert (q.cpdu_id, q.group, q.in_order, q.cpdu_size, q.offset, q.window) == \
            (9, True, True, 900, 400, 77)


def test_addresses_take_as_few_half_bytes_as_they_need():
    """C.3.2.6: 0.0.0.5 and 0.0.0.100 in one byte each; 3.5 bytes at most."""
    assert D.address("15.255.255.255") == D.MAX_ADDRESS
    assert D.address_text(D.address("7.1.2.3")) == "7.1.2.3"
    small = D.encode(D.DPDU(D.ACK_ONLY, 1, 2))
    big = D.encode(D.DPDU(D.ACK_ONLY, D.MAX_ADDRESS, 2))
    assert len(big) - len(small) == 6
    with pytest.raises(ValueError):
        D.address("16.0.0.0")


def test_a_bad_segment_keeps_its_header():
    """C.3.2.8 (5): a good header is used whatever became of the data."""
    raw = bytearray(D.encode(D.DPDU(D.DATA_ONLY, 2, 1, seq=5, data=b"abcdef")))
    raw[-6] ^= 0x10
    p = D.decode(bytes(raw))
    assert p.seq == 5 and not p.data_ok


def test_the_parser_finds_frames_among_noise_and_across_chunks():
    frames = [D.encode(D.DPDU(D.DATA_ONLY, 2, 1, seq=i, data=bytes([i]) * 50))
              for i in range(5)]
    bad = bytearray(frames[2])
    bad[6] ^= 1                                  # a header CRC failure
    stream = b"\x90noise\x90\xeb" + frames[0] + frames[1] + bytes(bad) + frames[3] + frames[4]
    par = D.Parser()
    got = []
    for i in range(0, len(stream), 7):
        got += par.push(stream[i:i + 7])
    assert [p.seq for p in got] == [0, 1, 3, 4]
    assert par.bad_headers >= 1


def test_selective_acks_as_figure_c_11_maps_them():
    """The LWE itself is never sent; its successor is bit 0 of byte 1; the
    last byte holds the UWE, padded with zeros above it."""
    assert D.ack_bitmap(10, 10, set()) == b""
    assert D.ack_bitmap(10, 12, {12}) == bytes([0b10])
    # Across the wrap: 255, 0, 1, 2, 3 are bits 0-4; 255, 1 and 3 came.
    assert D.ack_bitmap(254, 3, {255, 1, 3}) == bytes([0b10101])
    assert D.acked(254, bytes([0b10101])) == [255, 1, 3]
    assert len(D.ack_bitmap(0, 127, set(range(1, 128)))) == 16


def test_eow_messages():
    assert D.eow_fields(D.eow_drc_request(1200, "long", 1)) == (1, (4 << 4) | (2 << 2) | 1)
    assert D.eow_fields(D.eow_drc_response(D.RESP_REFUSE, D.REASON_LOCAL)) == \
        (2, (1 << 5) | 5)


@pytest.mark.parametrize("prim", [
    P.prim(P.BIND_REQUEST, sap=6, rank=3,
           service=P.ServiceType(P.MODE_NON_ARQ, P.CONFIRM_NONE, False, retransmissions=4)),
    P.prim(P.BIND_ACCEPTED, sap=6, mtu=2048),
    P.prim(P.BIND_REJECTED, reason=P.BIND_SAP_TAKEN),
    P.prim(P.UNBIND_REQUEST),
    P.prim(P.HARD_LINK_ESTABLISH, link_type=2, link_priority=1, remote_sap=9, remote=0x1020304),
    P.prim(P.HARD_LINK_REJECTED, reason=P.HL_REMOTE_BUSY, link_type=1, link_priority=3,
           remote_sap=2, remote=77),
    P.prim(P.UNIDATA_REQUEST, priority=5, dest_sap=7, dest=0xABCDEF, group=True,
           mode=P.ServiceType(P.MODE_ARQ, P.CONFIRM_CLIENT, True), ttl=600, u_pdu=b"hello"),
    P.prim(P.EXPEDITED_UNIDATA_REQUEST, dest_sap=7, dest=3,
           mode=P.ServiceType(P.MODE_DEFAULT), ttl=0, u_pdu=b"!"),
    P.prim(P.UNIDATA_INDICATION, priority=2, dest_sap=7, dest=2, group=False,
           mode=P.MODE_NON_ARQ_ERRORS, src_sap=5, src=9, errored=[(10, 4)],
           missing=[(0, 3), (40, 2)], u_pdu=b"x" * 50),
    P.prim(P.UNIDATA_REQUEST_REJECTED, reason=P.REJECT_TTL, dest_sap=7, dest=2, u_pdu=b"ab"),
])
def test_primitives_round_trip(prim):
    raw = P.encode(prim)
    assert raw[:3] == b"\x90\xeb\x00"
    assert int.from_bytes(raw[3:5], "big") == len(raw) - 5
    got = P.Framer().push(b"\x00\x90" + raw[:4])
    assert got == []
    f = P.Framer()
    out = f.push(raw[:3]) + f.push(raw[3:])
    assert len(out) == 1 and out[0].type == prim.type
    for k, v in prim.args.items():
        assert out[0][k] == v, k
