# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Field coding and original-byte credit across fixed RF field boundaries."""
import random

import pytest
from hfhost import ptc

from hfmodem.shrike import compress, pactor1, p1rx, placement, spec
from hfmodem.shrike.arq import State, ISS, REFUSED
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_ptc_compatibility import Client


@pytest.mark.parametrize('protocol', [spec.Protocol.PACTOR1, spec.Protocol.PACTOR2,
                                      spec.Protocol.PACTOR3])
@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('size', [7, 8, 20, 59])
def test_all_binary_values_and_runs_survive_arbitrary_field_boundaries(protocol, mode, size):
    original = (bytes(range(256)) + b' an English text message '*30 + b' '*287
                + b'Z'*31 + b'\x80'*96 + b'\xff'*200 + b']\x7f'*13
                + random.Random(1234).randbytes(193))
    pending = compress.transparent(original)
    decoder, supervisor = compress.Decoder(), compress.Supervisor()
    received = bytearray()
    while pending:
        field = compress.encode_field(pending, size, mode=mode, protocol=protocol)
        assert 0 < field.consumed <= len(pending)
        assert len(field.data) == size
        received += supervisor.feed(decoder.feed(field.data, field.data_type, protocol))
        pending = pending[field.consumed:]
    assert received == original


def test_published_p1_codes_and_automatic_ascii_fallback():
    # The ARRL PACTOR annex specifies LSB-first: space=10, e=011.
    field = compress.encode_field(b' e', 8, mode=1, protocol=spec.Protocol.PACTOR1)
    assert field.data_type == spec.DataType.HUFFMAN
    bits = [(byte >> b) & 1 for byte in field.data for b in range(8)]
    assert bits[:5] == [1, 0, 0, 1, 1]
    for byte in (93, 14, 127+1, 255):
        fallback = compress.encode_field(bytes([byte])*8, 8, mode=1,
                                         protocol=spec.Protocol.PACTOR1)
        assert fallback.data_type == spec.DataType.ASCII_8BIT
        assert fallback.consumed == 8


def test_full_policy_has_real_capacity_and_p1_mode2_uses_mode1():
    chars = b' '*400
    plain = compress.encode_field(chars, 8, mode=0)
    simple = compress.encode_field(chars, 8, mode=1)
    full = compress.encode_field(chars, 8, mode=2)
    assert plain.consumed == 8
    assert simple.consumed == 32
    assert full.consumed > simple.consumed
    assert compress.encode_field(chars, 8, mode=2, protocol=spec.Protocol.PACTOR1) == \
        compress.encode_field(chars, 8, mode=1, protocol=spec.Protocol.PACTOR1)


class P1Peer:
    def __init__(self, refuse=False):
        self.frames = []
        self.refuse = refuse

    def attach(self, host):
        self.host = host

    def send_p1_packet(self, payload, baud, count, **kwargs):
        if self.refuse:
            return REFUSED
        self.frames.append(pactor1.data_packet(payload, baud, count, **kwargs))
        return pactor1.DATA_FIELD[baud]


def test_ptc_effective_compression_ack_credit_echo_and_identical_retry():
    peer = P1Peer()
    host = PtcHost(peer=peer)
    client = Client(host)
    assert client.cmd('#MO 2').code == ptc.OK
    assert client.cmd('%M1').code == ptc.OK
    host.arq.state, host.arq.role = State.CONNECTED, ISS
    host.arq.dxcall = 'K7ABC'
    host._link_established('K7ABC')
    client.cmd('G')
    original = b' '*50
    assert client.exchange(original, False).code == ptc.OK
    host.arq._start_next_packet()
    packet = host.arq._inflight
    assert len(packet.payload) == 32
    frame = peer.frames[-1]
    assert (frame[-3] >> 2) & 3 == spec.DataType.HUFFMAN
    assert compress.decompress(frame[1:-3], spec.DataType.HUFFMAN,
                               spec.Protocol.PACTOR1) == original[:32]
    host.send_packet(packet.sl, packet.payload, packet.status)
    assert peer.frames[-1] == frame
    assert host.arq.payload_ledger.confirmed == 0
    host.arq._settle_inflight()
    assert host.arq.payload_ledger.confirmed == 32
    assert int(client.cmd('%T').text) == 32
    assert client.cmd('G').data == original[:32]
    assert client.cmd('#MO 0').code == ptc.FAIL
    assert host.settings['MODE'] == '2'


def test_transport_refusal_never_credits_a_coded_field():
    peer = P1Peer(refuse=True)
    host = PtcHost(peer=peer)
    host._store('MODE', '1')
    host.arq.state, host.arq.role = State.CONNECTED, ISS
    host.arq.on_host_data(b' '*40)
    assert host.arq._start_next_packet() == REFUSED
    assert host.arq.payload_ledger.confirmed == 0
    assert host.arq.payload_ledger.accepted == 40
    assert host.arq._inflight.payload == b' '*32


@pytest.mark.parametrize('breakin', [False, True])
def test_p1_signal_carries_the_coded_data_type_and_character_prefix(breakin):
    size = (pactor1.BREAKIN_FIELD if breakin else pactor1.DATA_FIELD)[200]
    field = compress.encode_field(b' an English message '*10, size, mode=2,
                                  protocol=spec.Protocol.PACTOR1)
    render = pactor1.breakin_signal if breakin else pactor1.packet_signal
    frames = p1rx.decode_p1_packets(render(field.data, 200, data_type=field.data_type),
                                   breakin=breakin)
    assert len(frames) == 1
    assert frames[0].data_type == field.data_type
    assert compress.decompress(frames[0].payload, frames[0].data_type,
                               spec.Protocol.PACTOR1) == (b' an English message '*10)[:field.consumed]


def test_p3_original_binary_credit_survives_coding_and_short_renderer_fallback():
    class Peer:
        def attach(self, host): self.host = host
        def send_packet(self, sl, payload, status, breakin=False):
            size = (placement.CHANGEOVER if breakin else placement.SPEED_PATHS[sl]).crc_bytes-3
            self.latest = payload[:size], (status >> 2) & 7
            return min(size, len(payload))
    peer = Peer()
    host = PtcHost(peer=peer)
    host._store('MODE', '2')
    host.protocol = spec.Protocol.PACTOR3
    host.arq.state, host.arq.role = State.CONNECTED, ISS
    host.arq._cycle_long = True  # peer has only a short renderer
    original = bytes(range(256)) + b' '*191 + b' an English message '*20
    assert host.arq.on_host_data(original)
    decoder, supervisor, received = compress.Decoder(), compress.Supervisor(), bytearray()
    while host.arq._outbuf:
        host.arq._start_next_packet()
        data, coding = peer.latest
        received += supervisor.feed(decoder.feed(data, coding, spec.Protocol.PACTOR3))
        host.arq._settle_inflight()
        assert host.arq.payload_ledger.confirmed == len(received)
    assert received == original
    assert host.arq.payload_ledger.confirmed == len(original)
    assert not host.host_output_failure
