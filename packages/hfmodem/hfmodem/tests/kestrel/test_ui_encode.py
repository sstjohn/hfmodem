# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.broadcast import Broadcast
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.host.kiss_tx import KissTxQueue
from hfmodem.kestrel.rx.broadcast import parse_compact, parse_packets
from hfmodem.kestrel.ui import UIAddress, UIFrame
from hfmodem.kestrel.tx.ui import compact, packetize, packets

FIXTURES = Path(__file__).parent / 'fixtures/kiss-ui-1031'
PEER_FILE = FIXTURES / 'peer-packets.json'
# Recovered from VARA HF emissions, so it stays out of the distribution.
PEER = json.loads(PEER_FILE.read_text())['captures'] if PEER_FILE.exists() else []
peer_captures = pytest.mark.skipif(not PEER, reason=f'recording unavailable: {PEER_FILE}')
NATIVE = json.loads((FIXTURES / 'native-replay.json').read_text())['cases']


def frame(info=b'hello', *, destination='APRS', path=(), command=0, control=3, pid=0xF0):
    addresses = [UIAddress(destination), UIAddress('W9SSJ'), *path]
    addresses[-1] = replace(addresses[-1], flags=addresses[-1].flags | 1)
    return UIFrame(command, tuple(addresses), control, pid, info)


def assert_packets(actual, case):
    level, encoded = actual
    assert level == case['level']
    assert [p.hex() for p in encoded] == case['packets']


@peer_captures
@pytest.mark.parametrize('case', PEER, ids=lambda r: f"{r['session']}/{r['case']}")
def test_every_peer_capture(case):
    raw = bytes.fromhex(case['sent'])
    message = UIFrame.from_ax25(raw, case['command'])
    assert message.ax25() == raw
    assert_packets(packetize(message, as_captured=True), case)


@pytest.mark.parametrize('case', NATIVE, ids=lambda r: f"{r['source']}/{r['case']}")
def test_every_native_replay(case):
    message = UIFrame.from_ax25(bytes.fromhex(case['frame']), case['command'])
    as_captured = case['case'] in {'n-gen7-pid', 'a-wide2-2', 'l-eq-peer', 'l-wide2-2-peer'}
    if case['case'] == 'a-path-pid8':
        # This early encoded case added an entire padding byte, corrected in m-path-pid8.
        core, mode = compact(message)
        actual = packets(core + b'\x00', mode)
        corrected = next(r for r in NATIVE if r['case'] == 'm-path-pid8')
        assert_packets(packetize(message), corrected)
    else:
        # The early S/E case used level 2 for a 24-byte compressed core;
        # later VARA HF size captures establish level 1 at that boundary.
        level = 2 if case['case'] == 'a-pos-se' else None
        actual = packetize(message, as_captured=as_captured, level=level,
                           full_path=case['case'] == 'm-full-path')
    assert_packets(actual, case)


@pytest.mark.parametrize('command', [0, 1])
@pytest.mark.parametrize('flags', [0, 0x20, 0x40, 0x60, 0x80, 0xE0])
def test_host_flags_and_lowercase_roundtrip_but_do_not_change_rf(command, flags):
    message = frame(path=(UIAddress('Wide2', 1, flags | 0x80),), command=command)
    message = replace(message, addresses=tuple(
        replace(a, flags=flags | (a.flags & 0x81)) for a in message.addresses))
    raw = message.ax25()
    parsed = UIFrame.from_ax25(raw, command)
    assert parsed == message
    assert parsed.ax25() == raw
    canonical = replace(message.normalized(), addresses=tuple(
        replace(a, flags=0x60 | (a.flags & 0x81)) for a in message.normalized().addresses))
    assert packetize(parsed) == packetize(canonical)


@pytest.mark.parametrize('info,mode', [
    (b'!4303.14N/08924.53W-symbol', '01'),
    (b'!4303.14N/08924.53W-\x00', '00'),
    (b'!4303.14N/08924.53W-x\x00', '01'),
    (b'!4363.14N/08924.53W-symbol', '01'),
    (b'!4303.14N/99999.99W-symbol', '00'),
    (b'=4303.14N/08924.53W-symbol', '00'),
    (b'!4303.  N/08924.  W-symbol', '00'),
    (b'!    .  N/     .  W-symbol', '00'),
    (b'!4303.14N/08924.53W', '00'),
    (b':N0CALL-1 :ack12', '10'),
    (b':BLN1     :bulletin', '10'),
    (b':n0call   :hello', '00'),
    (b':EOC/NET  :hello', '00'),
    (b':N0CALL-0 :hello', '00'),
    (b':N0CALL-01:hello', '00'),
    (b': N0CALL  :hello', '00'),
    (b':N0CALL-16:hello', '00'),
    (b':ABCDEFG  :hello', '00'),
    (b':         :hello', '00'),
    (b':N0CALL   :\x05literal', '00'),
])
def test_compress_only_when_information_expands_identically(info, mode):
    assert compact(frame(info))[1] == mode


@pytest.mark.parametrize('prefix', [b'!4303.14N/08924.53W-', b':N0CALL   :'])
@pytest.mark.parametrize('length', range(8))
@pytest.mark.parametrize('tail', [b'', b'\x00', b'\x00\x00'])
def test_serialized_aprs_roundtrip_at_every_padding_boundary(prefix, length, tail):
    message = frame(prefix + b'x' * length + tail)
    core, _ = compact(message)
    decoded = parse_compact(core)
    assert decoded is not None and decoded.ax25() == message.ax25()
    level, encoded = packetize(message)
    received = parse_packets(encoded, level=level)
    assert received is not None and received.ax25() == message.ax25()


@pytest.mark.parametrize('hops', [4, 8])
@pytest.mark.parametrize('command', [0, 1])
def test_long_sequential_wide_path_roundtrip(hops, command):
    message = frame(b'!4303.14N/08924.53W-path', command=command,
                    path=tuple(UIAddress(f'WIDE{k}', 1) for k in range(1, hops + 1)))
    core, _ = compact(message)
    decoded = parse_compact(core)
    assert decoded is not None and decoded.ax25() == message.ax25()
    level, encoded = packetize(message)
    received = parse_packets(encoded, level=level)
    assert received is not None and received.ax25() == message.ax25()


@pytest.mark.parametrize('info', [b'!4303.14N/08924.53W-symbol', b':N0CALL   :ack12'])
@pytest.mark.parametrize('change', [dict(command=1), dict(destination='ALL'), dict(pid=0xCF)])
def test_compression_requires_ap_command_zero_and_ui_control_pid(info, change):
    assert compact(frame(info, **change))[1] == '00'
    assert compact(frame(info + b'\x80'))[1] == '00'


@pytest.mark.parametrize('name', ['WIDE1', 'WIDE2'])
@pytest.mark.parametrize('ssid', range(4))
def test_lossless_wide_hops(name, ssid):
    message = frame(path=(UIAddress(name, ssid, 0xE0),))
    encoded = packetize(message)
    assert (encoded == packetize(message, as_captured=True)) == (ssid < 2)
    if ssid >= 2:
        assert encoded == packetize(message, full_path=True)


@pytest.mark.parametrize('control,pid', [(3, 0xCF), (0x13, 0xF0), (0x83, 0x01), (0x13, 0x01)])
def test_general_width_preserves_control_and_pid(control, pid):
    message = frame(destination='ALL', control=control, pid=pid)
    encoded = compact(message)
    assert (encoded == compact(message, as_captured=True)) == (control < 128 and pid < 128)


@pytest.mark.parametrize('level,size', [(1, 9), (2, 22), (4, 43)])
@pytest.mark.parametrize('count', range(2, 19))
def test_packet_count_metadata_and_reassembly(level, size, count):
    core = b'\x40' + bytes((i * 71) % 256 for i in range(count * (size - 1) - 1))
    actual_level, encoded = packets(core, level=level)
    assert actual_level == level and len(encoded) == count
    assert encoded[0][1] == (crc16_genibus(core) >> 8 & 0xF0) | (count - 2)
    assert encoded[0][0] == 0xC0 and encoded[-1][0] == 0x40
    assert all(p[0] == 0xC0 for p in encoded[:-1])
    assert bytes([encoded[-1][0]]) + encoded[0][2:-3] + b''.join(
        p[1:-3] for p in encoded[1:]) == core
    for p in encoded:
        assert len(p) == size + 3
        assert p[-3] == (9 if level == 1 else 1)
        assert crc16_genibus(p[:-2]) == int.from_bytes(p[-2:], 'big')
    if count == 18:
        with pytest.raises(ValueError, match='18 physical packets'):
            packets(core + b'x', level=level)


@pytest.mark.parametrize('last', [0, 1])
def test_eighteen_packet_metadata_overlaps_checksum(last):
    core = b'\x40' + b'x' * 740 + bytes([last])
    _, encoded = packets(core)
    assert len(encoded) == 18
    assert encoded[0][1] == (crc16_genibus(core) >> 8 & 0xF0) | 16


def test_legacy_broadcast_has_no_77_byte_or_two_packet_tx_limit():
    message = Broadcast(1, 'ABCDEFG', 'ALL', b'\xFF' * 740)
    assert len(packetize(message)[1]) == 18
    with pytest.raises(ValueError, match='18 physical packets'):
        packetize(replace(message, info=b'\xff' * 750))


@pytest.mark.parametrize('command', [0, 1])
def test_empty_information_is_encodable(command):
    message = frame(b'', command=command)
    assert UIFrame.from_ax25(message.ax25(), command) == message
    assert packetize(message) == packetize(message, as_captured=True)


def test_synthesis_emits_all_eighteen_packets_in_order(monkeypatch):
    from hfmodem.kestrel.tx import broadcast as tx
    message = frame(b'\xFF' * 730, destination='ALL')
    level, encoded = packetize(message)
    assert level == 4 and len(encoded) == 18
    seen = []

    def body(raw, **kwargs):
        seen.append(raw)
        return np.ones(4)

    monkeypatch.setattr(tx, 'synth_tones', lambda *a, **kw: np.ones(8))
    monkeypatch.setattr(tx.T, 'synth_burst', body)
    wave = tx.synth_broadcast(message)
    assert seen == encoded
    assert len(wave) == 32 * 2048 + 1536 + 18 * 4 + 17 * 1536


@peer_captures
@pytest.mark.parametrize('case', [r for r in PEER if r['case'] in {'src-blank', 'c1-hyphen'}],
                         ids=lambda r: r['case'])
def test_historical_bad_identities_are_never_admitted(case):
    message = UIFrame.from_ax25(bytes.fromhex(case['sent']), case['command'])
    with pytest.raises(ValueError):
        packetize(message)
    queue = KissTxQueue()
    assert not queue.broadcast(message.command, message.ax25())
    assert not queue


@pytest.mark.parametrize('source', ['K1ABC', 'AAR5ZZ', '123456', 'X', 'n0call'])
def test_queue_accepts_every_encodable_identity_without_mycall(source):
    message = frame()
    message = replace(message, addresses=(message.addresses[0], UIAddress(source, 15, 1)))
    logs = []
    queue = KissTxQueue(clock=lambda: 0, log=lambda *args: logs.append(args))
    assert queue.broadcast(0, message.ax25())
    queue.clock = lambda: 2
    claimed = queue.claim().value
    assert claimed.source == source.upper() + '-15'
    assert bool(logs) == (source != source.upper())


@pytest.mark.parametrize('info,warning', [(b'x' * 78, False), (b'x' * 735, False),
                                       (b'x' * 737, True), (b'\x80' * 730, True)])
def test_queue_warns_at_peer_receive_edge_without_refusing(info, warning):
    message = frame(info, destination='ALL')
    logs = []
    queue = KissTxQueue(log=lambda *args: logs.append(args))
    assert queue.broadcast(0, message.ax25())
    assert any('receive edge' in text for _, text in logs) == warning


@pytest.mark.parametrize('message', [
    frame(b'', path=(UIAddress('WIDE2', 2, 0xE0),)),
    frame(b'path and PID', path=(UIAddress('WIDE1', 1),), pid=0xCF),
    frame(b'eight hops', path=tuple(UIAddress(f'K{i}ABC', i) for i in range(1, 9))),
    frame(b'poll', control=0x13),
    frame(command=1, path=(UIAddress('ABCDEFG', 15),)),
])
def test_queue_preserves_all_measured_fields(message):
    queue = KissTxQueue(clock=lambda: 0)
    assert queue.broadcast(message.command, message.ax25())
    queue.clock = lambda: 2
    claimed = queue.claim().value
    assert claimed == message
    assert packetize(claimed) == packetize(message)


@pytest.mark.parametrize('message,reason', [
    (frame(path=(UIAddress('WIDE1', 1),) * 9), 'eight digipeaters'),
    (frame(b'x' * 900), '18 physical packets'),
    (frame(destination='W9/SJ'), 'A-Z or 0-9'),
    (frame(path=(UIAddress('X-Y'),)), 'A-Z or 0-9'),
])
def test_queue_refuses_unencodable_frames_and_recovers(message, reason):
    logs, diagnostics = [], []
    queue = KissTxQueue(log=lambda *args: logs.append(args),
                        diagnostic=lambda *a, **kw: diagnostics.append((a, kw)))
    assert not queue.broadcast(message.command, message.ax25())
    assert reason in logs[-1][1] and reason in diagnostics[-1][1]['reason']
    assert queue.broadcast(0, frame().ax25())
    assert len(queue) == 1


@pytest.mark.parametrize('raw,reason', [
    (b'\x00', 'incomplete UI address'),
    (frame().ax25()[:7] + b'\x82\x84\x86\x88\x8a\x8c\x8e\x61\x03\xf0hello', 'shifted ASCII'),
    (b'A' + frame().ax25()[1:], 'shifted ASCII'),
    (frame().ax25()[:14], 'control and PID'),
])
def test_queue_rejects_malformed_host_envelopes(raw, reason):
    logs = []
    queue = KissTxQueue(log=lambda *args: logs.append(args))
    assert not queue.broadcast(0, raw)
    assert reason in logs[-1][1]
