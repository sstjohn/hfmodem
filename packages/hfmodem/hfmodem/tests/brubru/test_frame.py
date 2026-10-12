# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfmodem.ax25.frame import address, ui_frame
from hfmodem.brubru import frame as f
from hfmodem.core.crc import crc16
from hfmodem.tests.brubru.vectors import CAPACITIES, frames_for


HEADER_WORDS = (
    '3ac56a5ca35cc6ac',
    '06f956609f60fa90',
    '09f6596f906ff59f',
    '609f3006f9069cf6',
    '6f903f09f60993f9',
    'a35cf3c53ac55f35',
    'ac53fcca35ca503a',
    'ca359aac53ac365c',
    '3ac595a35ca3c6ac',
    '06f9a99f609ffa90',
    '09f6a6906f90f59f',
    '609fcff906f99cf6',
    '6f90c0f609f693f9',
    'a35c0c3ac53a5f35',
    'ac530335ca35503a',
    'ca356553ac53365c',
    '3ac595a3a35c3953',
    '06f9a99f9f60056f',
    '09f6a690906f0a60',
    '609fcff9f9066309',
    '6f90c0f6f6096c06',
    'a35c0c3a3ac5a0ca',
    'ac53033535caafc5',
    'ca35655353acc9a3',
    '3a3a6aa35c5c39ac',
    '0606569f60600590',
    '090959906f6f0a9f',
    '606030f9060663f6',
    '6f6f3ff609096cf9',
    'a3a3f33ac5c5a035',
    'acacfc35cacaaf3a',
    'caca9a53acacc95c',
    '3a3a6aa3a3a3c653',
    '0606569f9f9ffa6f',
    '090959909090f560',
    '606030f9f9f99c09',
    '6f6f3ff6f6f69306',
    'a3a3f33a3a3a5fca',
    'acacfc35353550c5',
    'caca9a53535336a3',
    'c5c56aa35c5cc653',
    'f9f9569f6060fa6f',
    'f6f659906f6ff560',
    '9f9f30f906069c09',
    '90903ff609099306',
    '5c5cf33ac5c55fca',
    '5353fc35caca50c5',
    '35359a53acac36a3',
    'c5c56aa3a3a339ac',
    'f9f9569f9f9f0590',
    'f6f6599090900a9f',
    '9f9f30f9f9f963f6',
    '90903ff6f6f66cf9',
    '5c5cf33a3a3aa035',
    '5353fc353535af3a',
    '35359a535353c95c',
    'c53a95a3a35cc6ac',
    'f906a99f9f60fa90',
    'f609a690906ff59f',
    '9f60cff9f9069cf6',
    '906fc0f6f60993f9',
    '5ca30c3a3ac55f35',
    '53ac033535ca503a',
    '35ca655353ac365c',
)


def test_spec_vectors():
    assert f.PREAMBLE.hex() == '420c39d5b0979d28'
    assert f.pn9(8) == f.PREAMBLE
    bits = np.unpackbits(np.array([1, 0], np.uint8), bitorder='little')
    encoded = f.CODE.encode(bits, terminate=False)
    assert ''.join(str(b) for b in encoded[:18]) == '111000100111111011'
    assert not encoded[18:].any()
    assert list(f.interleaver(1200)[1:8]) == [50, 40, 2, 42, 23, 123, 247]
    for n in (160, 240):
        assert list(f.interleaver(n)[1:8]) == [57, 37, 35, 16, 51, 73, 146]
    for value, call in ((0x44D1409D, 'N0CALL'), (0xE15B6C42, 'K1A')):
        wire = value.to_bytes(4, 'little') + b'\x60'
        assert f.expand_address(wire) == address(call)
        assert f.compress_address(address(call)) == wire
    assert crc16(b'123456789') == 0x906E


def test_header_codewords():
    book = f.header_codebook()
    assert np.array_equal(book @ book.T, 64 * np.eye(64))
    assert np.packbits(f.header_bits(0), bitorder='little').tobytes().hex() == '3ac56a5ca35cc6ac'
    for i in range(64):
        assert np.packbits(f.header_bits(i), bitorder='little').tobytes().hex() == HEADER_WORDS[i]
        assert f.decode_header(book[i]) == (i, 1.0)


@pytest.mark.parametrize('capacity', CAPACITIES)
@pytest.mark.parametrize('level', [0, 1])
def test_all_classes_bit_chain(capacity, level):
    frames = frames_for(capacity)
    packet = f.assemble(frames, suggestion=3)
    assert len(packet.data) == capacity
    encoded = f.encode_packet(packet.data, level)
    assert np.array_equal(np.sort(f.interleaver(len(encoded))), np.arange(len(encoded)))
    plain, agreement = f.decode_packet(1.0 - 2 * encoded, capacity, level)
    assert plain == packet.data[:-1] and agreement == 1
    assert f.parse(plain) == (3, frames)


@pytest.mark.parametrize('capacity', range(345, 826, 30))
@pytest.mark.parametrize('delta', [-1, 0, 1])
def test_exact_extended_capacity(capacity, delta):
    size = capacity + delta
    if size > 825:
        with pytest.raises(ValueError):
            f.length_class(size)
    else:
        cls, ext = f.length_class(size)
        assert f.packet_bytes(cls, ext) == (capacity if delta <= 0 else capacity + 30)


@pytest.mark.parametrize('addresses', range(2, 11))
def test_bare_length_transitions(addresses):
    prefix = b''.join(address('A', last=i == addresses - 1, flag=i == 0) for i in range(addresses))
    for capacity in CAPACITIES:
        for delta in (-1, 0, 1):
            size = capacity + delta
            raw_len = size + 2 * addresses - 4
            if raw_len < 7 * addresses + 2:
                continue
            raw = prefix + b'\x03\xf0' + bytes(raw_len - len(prefix) - 2)
            if raw_len > 399:
                with pytest.raises(ValueError, match='399'):
                    f.assemble([raw])
                continue
            cls, ext = f.length_class(size)
            if f.packet_bytes(cls, ext) - size > 31:
                with pytest.raises(ValueError, match='padding'):
                    f.assemble([raw])
            else:
                packet = f.assemble([raw])
                assert (packet.cls, packet.ext) == (cls, ext)
                assert f.parse(packet.data[:-1])[1] == [raw]


@pytest.mark.parametrize('dest_c,src_c', [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize('level', [0, 1])
def test_container_representability(dest_c, src_c, level):
    first = ui_frame('N0CALL', 'CQ', b'first')
    later = address('CQ', flag=bool(dest_c)) + address('N0CALL', flag=bool(src_c), last=True) + b'\x03\xf0later'
    packets = f.pack_frames([first, later])
    assert len(packets) == (1 if dest_c != src_c else 2)
    result = []
    for packet in packets:
        plain, _ = f.decode_packet(1.0 - 2 * f.encode_packet(packet.data, level), len(packet.data), level)
        result.extend(f.parse(plain)[1])
    assert result == [first, later]


def test_one_frame_container_receive_only():
    raw = frames_for(313)[0]
    with pytest.raises(ValueError, match='padding'):
        f.assemble([raw])
    body = f.compress_address(raw[:7]) + f.compress_address(raw[7:14]) + raw[14:]
    content = b'\x80' + len(body).to_bytes(2, 'little') + body
    packet = f.pack(content + crc16(content).to_bytes(2, 'little'), bare=False)
    assert (packet.cls, packet.ext) == (15, 0)
    assert f.parse(packet.data[:-1]) == (0, [raw])


def test_empty_and_invalid_headers():
    for operation in (lambda: f.assemble([]), lambda: f.pack_frames([]), lambda: f.parse(b''),
                      lambda: f.packet_bytes(16), lambda: f.packet_bytes(15, 17),
                      lambda: f.packet_bytes(3, 1), lambda: f.decode_header(np.zeros(64)),
                      lambda: f.header_bits(64), lambda: f.assemble(frames_for(21), 4)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize('container', [False, True])
def test_crc_corruption_and_truncation(container):
    frames = [ui_frame('N0CALL', 'CQ', b'test')]
    if container:
        frames *= 2
    packet = f.assemble(frames).data[:-1]
    for n in range(len(packet)):
        with pytest.raises(ValueError):
            f.parse(packet[:n])
    broken = bytearray(packet)
    broken[14] ^= 4
    with pytest.raises(ValueError):
        f.parse(bytes(broken))
    encoded = f.encode_packet(f.assemble(frames).data, 0)
    with pytest.raises(ValueError, match='truncated'):
        f.decode_packet(encoded[:-1], len(packet) + 1, 0)


@pytest.mark.parametrize('container', [False, True])
def test_padding_outside_the_crc_does_not_change_delivered_frames(container):
    frames = [ui_frame('N0CALL', 'CQ', b'x')] * (2 if container else 1)
    packet = bytearray(f.assemble(frames).data[:-1])
    packet[-1] = 0xA5
    assert f.parse(bytes(packet)) == (0, frames)


@pytest.mark.parametrize('kind', ['count', 'length', 'body', 'callsign'])
def test_container_valid_crc_does_not_hide_bad_structure(kind):
    raw = ui_frame('N0CALL', 'CQ', b'')
    packet = bytearray(f.assemble([raw, raw]).data[:-1])
    crc_at = 1 + 2 + 12 + 2 + 2
    if kind == 'count':
        packet[0] = 0x87
    elif kind == 'length':
        packet[1:3] = b'\xff\x01'
    elif kind == 'body':
        packet[15:17] = b'\x01\x00'
        packet[18:19] = b''
        crc_at -= 1
    else:
        packet[3:7] = b'\xff' * 4
    packet[crc_at:crc_at + 2] = crc16(packet[:crc_at]).to_bytes(2, 'little')
    with pytest.raises(ValueError):
        f.parse(bytes(packet))


def test_invalid_frame_is_rejected_before_packing():
    good = ui_frame('N0CALL', 'CQ', b'')
    for raw in (b'', good[:14], good[:15], bytes([good[0] | 1]) + good[1:], good[:13]):
        with pytest.raises(ValueError):
            f.pack_frames([good, raw])


@pytest.mark.parametrize('flags', [(False, False), (True, True)])
def test_unrepresentable_first_candidate_stays_bare(flags):
    raw = address('CQ', flag=flags[0]) + address('N0CALL', flag=flags[1], last=True) + b'\x03\xf0'
    other = ui_frame('N0CALL', 'CQ', b'other')
    packets = f.pack_frames([raw, other])
    assert len(packets) == 2 and all(not p.data[0] & 0x80 for p in packets)
    assert [r for p in packets for r in f.parse(p.data[:-1])[1]] == [raw, other]


def test_container_count_and_address_groups():
    one = ui_frame('N0CALL', 'CQ', b'x')
    other = ui_frame('N0CALL', 'APRS', b'y')
    packets = f.pack_frames([one] * 9 + [other])
    assert len(packets) == 3
    assert [len(f.parse(p.data[:-1])[1]) for p in packets] == [8, 1, 1]
    assert [r for p in packets for r in f.parse(p.data[:-1])[1]] == [one] * 9 + [other]
    with pytest.raises(ValueError):
        f.assemble([one] * 9)
    assert f.pack_frames([ui_frame('N0CALL', 'CQ', bytes(384))]) == []


def test_receive_accepts_a_bare_packet_beyond_sender_frame_policy():
    raw = ui_frame('N0CALL', 'CQ', bytes(484))
    body = f.compress_address(raw[:7]) + f.compress_address(raw[7:14]) + raw[14:]
    packet = f.pack(b'\0' + body + crc16(raw).to_bytes(2, 'little'), bare=True)
    assert f.parse(packet.data[:-1]) == (0, [raw])
    with pytest.raises(ValueError, match='399'):
        f.assemble([raw])


@pytest.mark.parametrize('capacity', range(345, 826, 30))
@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('level', [0, 1])
def test_extended_boundary_packets(capacity, delta, level):
    size = capacity + delta
    frames = frames_for(size)
    if size > 825:
        with pytest.raises(ValueError):
            f.assemble(frames)
        return
    packet = f.assemble(frames)
    assert len(packet.data) == (capacity if delta <= 0 else capacity + 30)
    plain, agreement = f.decode_packet(1.0 - 2 * f.encode_packet(packet.data, level),
                                       len(packet.data), level)
    assert agreement == 1
    assert f.parse(plain)[1] == frames


@pytest.mark.parametrize('offset,mask', [(0, 8), (0, 16), (2, 0x7e),
                                         (16, 0x7e), (2, 0x80)])
@pytest.mark.parametrize('first_c', [False, True])
def test_container_ignores_unassigned_bits(offset, mask, first_c):
    first = (address('CQ', flag=first_c) +
             address('N0CALL', flag=not first_c, last=True) + b'\x03\xf0')
    later = ui_frame('N0CALL', 'CQ', b'')
    packet = bytearray(f.assemble([first, later]).data[:-1])
    packet[offset] |= mask
    crc_at = 19
    packet[crc_at:crc_at + 2] = crc16(packet[:crc_at]).to_bytes(2, 'little')
    assert f.parse(bytes(packet)) == (0, [first, later])


@pytest.mark.parametrize('score', [0.499, 0.5, 0.501])
def test_header_correlation_boundary(score):
    # Sixteen dispersed errors keep word zero the unique best candidate.
    book = f.header_codebook().astype(float)
    soft = book[0].copy()
    errors = [47, 9, 59, 62, 56, 61, 42, 44, 10, 53, 29, 13, 25, 8, 28, 38]
    soft[errors] *= -1
    soft[errors[0]] *= (33 - 63 * score) / (1 + score)
    measured = book @ soft / np.abs(soft).sum()
    assert measured.argmax() == 0
    assert measured[0] == pytest.approx(score)
    if score < 0.5:
        with pytest.raises(ValueError, match='correlation'):
            f.decode_header(soft)
    else:
        index, correlation = f.decode_header(soft)
        assert index == 0 and correlation == pytest.approx(score)


@pytest.mark.parametrize('position', [0, 1, 2])
@pytest.mark.parametrize('reason', ['padding', 'capacity'])
def test_packing_skips_unsendable_frame_and_keeps_order(position, reason, caplog):
    good = [ui_frame('N0CALL', 'CQ', b'first'), ui_frame('N0CALL', 'APRS', b'last')]
    bad = frames_for(313 if reason == 'padding' else 105)[0]
    frames = good.copy()
    frames.insert(position, bad)
    packets = f.pack_frames(frames, max_capacity=825 if reason == 'padding' else 75)
    assert [raw for p in packets for raw in f.parse(p.data[:-1])[1]] == good
    assert f'frame {position}' in caplog.text
    assert ('padding' if reason == 'padding' else 'capacity') in caplog.text
