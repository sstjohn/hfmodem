# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Generated frames only: no stock audio or extracted vectors."""
import random
import zlib

import pytest

from hfmodem.kestrel.vara import compression as C
from hfmodem.kestrel.vara import vara_frames as F
from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.kestrel.arq import phy


def packed(bits):
    bits += ('1' if bits[-1] == '0' else '0') * (8 - len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, 'big')


def deflated(data):
    d = zlib.compressobj(wbits=-15)
    return d.compress(data) + d.flush()


def container(body):
    return (0x8000 | len(body)).to_bytes(2, 'big') + body


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('mode,marker', [('OFF', 0x81), ('TEXT', 0x83), ('FILES', 0x85)])
def test_setup(mode, marker, bw, level):
    f = F.link_setup_frame('W9SSJ', bw, level=level, high_speed=True, compression=mode)
    assert f[7] == marker
    assert F.is_link_setup(f)
    assert F.link_setup_compression(f) == mode
    assert int.from_bytes(f[-2:], 'big') == F.crc16_genibus(f[:-2])
    assert F.link_setup_frame('W9SSJ', bw, level=level, compression=mode)[7] == 0x80


@pytest.mark.parametrize('size', [1, 32500, 32501, 40000])
@pytest.mark.parametrize('stored', [False, True])
def test_files_across_arbitrary_over_boundaries(size, stored):
    data = random.Random(7).randbytes(size) if stored else b'a' * size
    wire = C.encode_files(data)
    assert bool(wire[0] & 0x80) == (not stored and size > 1)
    rx = C.ReceiveCompression('FILES')
    got = bytearray()
    # A split header, blocks spanning many overs, and more than one block per over.
    cuts = [1, 2, 89, len(wire) - 1, len(wire)]
    start = 0
    for end in sorted(set(c for c in cuts if 0 < c <= len(wire))):
        out, warning = rx.feed(wire[start:end], 0x82, end == len(wire))
        assert warning is None
        got.extend(out)
        start = end
    assert got == data
    assert not rx.buffer


@pytest.mark.parametrize('bad', [
    container(deflated(b'a' * 130)[:-1]),
    container(deflated(b'a' * 130) + b'xx'),
    container(deflated(b'a' * 32501)),
    b'\x7f\xff',
    b'\x00',
    b'\x00\x02x',
    container(b'\xff'),
])
def test_files_damage_preserves_only_verified_blocks(bad):
    rx = C.ReceiveCompression('FILES')
    out, warning = rx.feed(C.encode_files(b'before') + bad, 0x82, True)
    assert out == b'before'
    assert warning.startswith('WRONG COMPRESSED DATA offset=')
    assert rx.damaged and not rx.buffer
    assert rx.feed(C.encode_files(b'after'), 0x82, True) == (b'', None)


@pytest.mark.parametrize('data', [b'a'*130, b'ab'*65, b'The prose has a VARA token.\r\nNext line!\r',
    b'\x80\x81\x82a', b'aa\r', b'aa\r\n', bytes(range(256)), b'\r\r\n\t\r'])
def test_text_roundtrip(data):
    r = C.decode_text(C.encode_text(data))
    assert r.error is None
    assert r.data == data


def test_generated_fixed_vector_and_table():
    assert C.encode_text(b'a' * 130) == b'\x15\xa8' + b'\x88'*64 + b'\x9f'
    codes = list(C.SYMBOLS) + [C.ESC]
    assert len(C.BYTE_CODES) == 87
    assert sum(2**-len(c) for c in codes) == 0.9765625
    assert not any(a != b and b.startswith(a) for a in codes for b in codes)


@pytest.mark.parametrize('wire,prefix', [
    (packed('101' + '0100'), b''),
    (packed('000' + f'{0xeb:08b}' + '0100'), b''),
    (packed('001' + C.BYTE_CODES[97] + C.BYTE_CODES[13] + f'{0xeb:08b}'), b'a\r'),
    (bytes.fromhex('29b8ff'), b'a'),
    (bytes.fromhex('2886c0'), b'aa\r'),
    (packed('001' + '0100' + '10111'), b'a'),
    (packed('001' + '0100' + C.ESC + '101'), b'a'),
    (packed('000' + '1010'), b''),
    (b'', b''),
])
def test_text_failure_and_direction_latch(wire, prefix):
    rx = C.ReceiveCompression('TEXT')
    out, warning = rx.feed(wire, 0xa2, True)
    assert out == prefix
    assert warning is not None
    assert rx.feed(C.encode_text(b'after', first=False), 0xa2, True) == (b'', None)
    assert rx.feed(b'plain after', 0x82, True) == (b'', None)


def test_self_contained_continuations_and_plain_fallback():
    for raw, data in [(False, b'next line'), (True, b'\x80\x81a tail')]:
        assert C.decode_text(C.encode_text(data, first=False, raw=raw)).data == data
    rx = C.ReceiveCompression('TEXT')
    assert rx.feed(b'\xff\x00plain', 0x82, True) == (b'\xff\x00plain', None)


class IO(A.VaraIO):
    def __init__(self):
        self.data_received = []
        self.warnings = []
    def log(self, message): pass
    def data(self, data): self.data_received.append(data)
    def compression_error(self, message): self.warnings.append(message)


def station(mode='OFF'):
    io = IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, compression=mode, high_speed=True)
    hs.caller = 'W9SSJ'
    return hs, io


def test_receive_after_duplicate_suppression_and_new_session_reset():
    hs, io = station()
    hs._start_compression('FILES')
    wire = C.encode_files(b'a' * 40000)
    cap = phy.body_size(hs.bw, hs._tx_level) - 1
    chunks = [wire[i:i+cap] for i in range(0, len(wire), cap)]
    bodies = [phy.vara_body(c, hs.caller, body_len=phy.body_size(hs.bw, hs._tx_level)) for c in chunks]
    hs._deliver(bodies)
    hs._deliver(bodies)
    assert b''.join(io.data_received) == b'a' * 40000
    assert not io.warnings
    hs._deliver_block(phy.vara_body(b'\x00', hs.caller, body_len=phy.body_size(hs.bw, hs._tx_level)))
    hs._deliver_block(phy.vara_body(C.encode_files(b'after'), hs.caller, body_len=phy.body_size(hs.bw, hs._tx_level)))
    assert len(io.warnings) == 1
    assert b''.join(io.data_received) == b'a' * 40000
    hs._start_compression('FILES')
    hs._deliver_block(phy.vara_body(C.encode_files(b'new'), hs.caller, body_len=phy.body_size(hs.bw, hs._tx_level)))
    assert io.data_received[-1] == b'new'


def test_files_tx_delivery_limit_and_preconnect_queue():
    hs, io = station('FILES')
    data = b'abcde' * 14000
    hs.send(data)  # provisional plain bytes before the caller mode is known
    hs._start_compression(C.caller_mode(hs.compression, hs.high_speed))
    cap = hs._tx_payload_size()
    rx = C.ReceiveCompression('FILES')
    totals = []
    total = 0
    for chunk in hs._txq:
        last = len(chunk) < cap
        out, warning = rx.feed(chunk, 0x82, last)
        assert warning is None
        total += len(out)
        if last:
            totals.append(total)
            total = 0
    assert totals == [50000, 20000]
    hs.set_compression('OFF')
    assert hs.session_compression == 'FILES'


def test_text_advertisement_and_transmission_remain_gated():
    assert C.caller_mode('TEXT', True) == 'OFF'
    assert C.caller_mode('FILES', False) == 'OFF'
    hs, io = station('OFF')
    hs._start_compression('TEXT')  # caller wins over our preference
    hs.state = A.VaraState.CONNECTING
    hs.send(b'a' * 130)
    assert b''.join(hs._txq) == b'a' * 130


@pytest.mark.parametrize('mode', ['OFF', 'TEXT', 'FILES'])
def test_peer_mode_wins_at_real_setup_seam(mode, monkeypatch):
    hs, io = station('OFF')
    hs.called = 'WI9YZA'
    hs.state = A.VaraState.CONNECTING
    f = F.link_setup_frame('W9SSJ', high_speed=True, compression=mode)
    monkeypatch.setattr(A.OF, 'link_setup_read', lambda *a, **kw: f)
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: True)
    assert hs._rx_link_setup([])
    assert hs.session_compression == mode


@pytest.mark.parametrize('mode,expected', [('OFF', 'OFF'), ('TEXT', 'OFF'), ('FILES', 'FILES')])
def test_outgoing_mode_forwarded_to_waveform(mode, expected, monkeypatch):
    hs, io = station(mode)
    monkeypatch.setattr(hs, '_send_burst', lambda *a, **kw: True)
    hs.originate('WI9YZA')
    seen = []
    monkeypatch.setattr(A.OF, 'link_setup_tx', lambda *a, **kw: seen.append(kw))
    hs._tx_link_setup()
    assert seen[0]['compression'] == expected


def test_reencoded_text_retry_does_not_repeat_host_bytes():
    hs, io = station()
    hs._start_compression('TEXT')
    text = b'hello world'
    def body(data, first=False, full=False):
        wire = C.encode_text(data, first=first)
        frame = phy.vara_body(wire, hs.caller, body_len=len(wire)+1 if full else 90)
        return frame[:-1] + bytes([frame[-1] | C.TEXT_FLAG])
    first = body(text, first=True)
    hs._deliver([first])
    # Same plaintext, changed over header; wire bytes differ.
    hs._deliver([body(text)])
    assert io.data_received == [text]
    # A known delivered prefix reappearing across two lower-level overs.
    hs._resent = text
    hs._deliver_block(body(text[:5], full=True))
    hs._deliver_block(body(text[5:] + b' new'))
    assert io.data_received == [text, b' new']


def test_host_command_and_fault_reach_tcp_sockets(monkeypatch):
    import socket
    from hfmodem.kestrel.host.server import HostSession
    from hfmodem.kestrel.host.vara_modem import VaraHostModem
    modem = VaraHostModem(IO(), high_speed=True)
    cmd, appcmd = socket.socketpair()
    data, appdata = socket.socketpair()
    appcmd.settimeout(1)
    appdata.settimeout(1)
    try:
        session = HostSession(cmd, data, modem)
        modem.start(session)
        session._dispatch('COMPRESSION FILES')
        assert appcmd.recv(4096) == b'OK\r'
        monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
        modem.connect('W9SSJ', 'WI9YZA')
        modem.poll()
        assert modem.station.session_compression == 'FILES'
        appcmd.recv(4096)  # initial BUFFER
        modem.station._deliver_block(phy.vara_body(C.encode_files(b'hello'), 'W9SSJ'))
        assert appdata.recv(4096) == b'hello'
        modem.station._deliver_block(phy.vara_body(b'\x00', 'W9SSJ'))
        assert appcmd.recv(4096).startswith(b'WRONG COMPRESSED DATA offset=')
        assert modem.station._rx_compression.damaged
    finally:
        for sock in (cmd, data, appcmd, appdata):
            sock.close()


@pytest.mark.parametrize('level', [4, 5, 6, 10])
@pytest.mark.parametrize('size', [84, 85, 86])
def test_files_close_without_full_trailer(level, size):
    # Stored container sizes 86/87/88 exercise all compact short endings.
    data = random.Random(91).randbytes(size)
    wire = C.encode_files(data)
    frame = phy.vara_body(wire, 'W9SSJ', level=level)
    assert phy.vara_payload(frame, caller='W9SSJ') == wire
    assert phy.over_is_last(frame, 'W9SSJ')
    assert C.ReceiveCompression('FILES').feed(wire, frame[-1], True) == (data, None)


def test_preconnect_and_live_preference_changes(monkeypatch):
    hs, io = station('FILES')
    monkeypatch.setattr(hs, '_send_burst', lambda *a, **kw: True)
    hs.send(b'first')
    hs.originate('WI9YZA')
    assert C.ReceiveCompression('FILES').feed(b''.join(hs._txq), 0x82, True) == (b'first', None)
    hs.set_compression('OFF')
    hs.originate('WI9YZA', retry=True)
    assert hs.session_compression == 'FILES'  # retries don't change the announced mode
    hs._txq.clear()
    hs.state = A.VaraState.DISCONNECTED
    hs.send(b'next session')
    hs.originate('WI9YZA')
    assert hs.session_compression == 'OFF'
    assert b''.join(hs._txq) == b'next session'


def test_escape_can_end_over_before_raw_continuation():
    first = C.decode_text(packed('001' + C.BYTE_CODES[97] + C.ESC))
    assert first.data == b'a' and first.error is None
    second = C.decode_text(C.encode_text(b'\x80a', first=False, raw=True))
    assert second.data == b'\x80a' and second.error is None


def test_off_ignores_text_bit_and_preserves_duplicate_handling():
    payload = b'\xffplain\x00\xfe'
    rx = C.ReceiveCompression('OFF')
    assert rx.feed(payload, 0xa2, True) == (payload, None)
    assert not rx.damaged
    hs, io = station()
    body = phy.vara_body(payload, hs.caller)
    body = body[:-1] + bytes([body[-1] | C.TEXT_FLAG])
    hs._deliver([body])
    hs._deliver([body])
    assert io.data_received == [payload]
    assert not io.warnings


def test_unknown_code_mid_over_delivers_no_following_symbols():
    bits = '001' + C.BYTE_CODES[97] + '110111000' + C.BYTE_CODES[98] * 12
    rx = C.ReceiveCompression('TEXT')
    out, warning = rx.feed(packed(bits), 0xa2, True)
    assert out == b'a'
    assert warning is not None and 'unknown code' in warning
    assert rx.feed(b'later', 0x82, True) == (b'', None)


@pytest.mark.parametrize('level', range(1, max(phy.TOP.values()) + 1))
@pytest.mark.parametrize('size', [86, 87, 88, 89])
def test_off_compact_close_and_full_over_on_high_speed_ladder(level, size):
    payload = random.Random(111).randbytes(size)
    body = phy.vara_body(payload, 'W9SSJ', level=level, tail=0x81)
    assert phy.vara_payload(body, caller='W9SSJ') == payload
    assert phy.over_is_last(body, 'W9SSJ') == (size < 89)
    rx = C.ReceiveCompression('OFF')
    assert rx.feed(phy.vara_payload(body, caller='W9SSJ'), body[-1], size < 89) == (payload, None)


def test_no_setup_responder_consumes_live_writes_once(monkeypatch):
    hs, io = station('FILES')
    hs.mfsk_only = True
    hs.listen()
    monkeypatch.setattr(hs, '_send_burst', lambda *a, **kw: True)
    monkeypatch.setattr(hs, '_tx_connected_ack', lambda: True)
    monkeypatch.setattr(hs, '_matches', lambda *a: True)
    monkeypatch.setattr(A.VF, 'best_match', lambda *a: ('W9SSJ', 10, 10))
    hs.on_rx_tones([], F.connect_request(hs.bw))
    assert hs.state == A.VaraState.CONNECTED
    assert hs.session_compression == 'OFF'
    hs._answering = True  # keep this test at the queue seam
    hs.send(b'live responder data')
    assert hs.preconnect_data == ()
    hs._txq.clear()  # peer has consumed it
    hs.state = A.VaraState.DISCONNECTED
    hs.originate('WI9YZA')
    assert hs.pending_bytes == 0


def test_default_warning_is_logged_once():
    class Logs(A.VaraIO):
        def __init__(self): self.lines = []
        def log(self, message): self.lines.append(message)
    io = Logs()
    hs = A.VaraStationHandshake(['W9SSJ'], io)
    hs.caller = 'W9SSJ'
    hs._start_compression('FILES')
    hs._deliver_block(phy.vara_body(b'\x00', hs.caller))
    assert sum('WRONG COMPRESSED DATA' in line for line in io.lines) == 1
