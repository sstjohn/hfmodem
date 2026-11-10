# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""B8 Xara L2x2 head-only regression, at the wired decoder/ARQ boundary.

Source: b8/main-tree/runs/*/bw2300-kestrel-limited/trace.jsonl, main
696e77a28. The four corruption_implementation=Kestrel cases below supply
the payload offsets, head/query times and c12/c13 queries. Raw PHY frames
are not logged: reconstruct the full bodies from the logged application
pattern and Xara 2e96fcd's framing, then append the CRC. Its queue_hint
gives h5 (400 bytes) or h6 (1200 bytes); the two preceding control reads
select reply F1 (hf/frames.rs ReplyFormat), hence fields 0x55 and 0x59.
FRAMES freezes both frames, including the missing 22-byte companion.

Both observed answer orders are pinned: QUIET at a decoded head, and a
held head followed by its companion slot and query. Decode outcomes enter
through LevelRead/_irs_nak; answer selection, framing, commit and host IO
are production code. No window-machine actions or verdicts are mocked.
"""

import importlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as VF, vara_mfsk as MK
from .test_rxwindow_adapter import TraceIO


TAPES = [
    ('b2300-m10-none-s0', 400, 176, 618.05, 623.79, 630.89, 632.39, 0),
    ('b2300-m10-none-s1', 400, 264, 639.46, 645.19, 652.33, 653.79, 0),
    ('b2300-m10-none-s2', 400, 176, 640.66, 646.49, 653.47, 654.89, 0),
    ('p2300-m5-poor-s0', 1200, 132, 1014.46, 1020.29, 1027.32, 1028.79, 1),
]


FRAMES = {
    176: ('6465666768696a0d0a48656c6c6f204b65737472656c5554e9',
          '2066726f6d20586172612e206162636465666768696a552db8'),
    264: ('204b65737472656c2066726f6d20586172612e20616255ba1e',
          '636465666768696a0d0a48656c6c6f204b657374726555819f'),
    132: ('72612e206162636465666768696a0d0a48656c6c6f2059c4b1',
          '4b65737472656c2066726f6d20586172612e2061626359cb0b'),
}


class PacketIO(TraceIO):
    def __init__(self):
        super().__init__()
        self.packets = []
        self.metadata = {}

    def tx_metadata(self, **metadata):
        self.metadata = metadata

    def tx(self, samples):
        super().tx(samples)
        self.packets.append(self.metadata.get('packet_kind', ''))


def stream(size):
    return (b'Hello Kestrel from Xara. abcdefghij\r\n' * 100)[:size]


def raw_frame(payload):
    return TX.build_frame(payload + b'\x99', 1)


def station(module, monkeypatch, now):
    monkeypatch.setattr(module, 'time', type('Clock', (), {'monotonic': staticmethod(lambda: now[0])}))
    sink = PacketIO()
    hs = module.VaraStationHandshake(['W9SSJ'], sink, bw='2300')
    hs.role, hs.called, hs.caller = 'initiator', 'W1AW', 'W9SSJ'
    hs.state, hs.step = module.VaraState.CONNECTED, module._I_CONNECTED
    hs.turn = module._TURN_PEER
    hs._link_high_speed = False
    hs._rx_read, hs._expect_level = (2, None), 2
    hs._start_compression('OFF')
    return hs, sink


def decoded(hs, raw, seconds, now, *, hold=False, level=2):
    assert VF.crc16_genibus(raw[:-2]) == int.from_bytes(raw[-2:], 'big')
    now[0] = seconds
    audio = TX.synth_frame(raw, phy.level_key('2300', level), over=1)
    hs._level_read = phy.LevelRead(level, (raw[:-2],), None, 1, end=len(audio))
    try:
        assert hs._answer_data_over(audio, hold=hold)
    finally:
        hs._level_read = None


def prefix(hs, sink, tape, now):
    """Six singles, then complete doubles, establish the trace's hold history."""
    _, size, offset, start, *_ = tape
    data = stream(size)
    for at in range(0, offset, 22):
        held = at >= 132 and (at - 132) % 44 == 0
        decoded(hs, raw_frame(data[at:at + 22]), start - (offset - at) / 4, now, hold=held)
    assert b''.join(sink.host) == data[:offset]
    return data


def damaged_frames(data, offset):
    frames = tuple(bytes.fromhex(raw) for raw in FRAMES[offset])
    assert b''.join(raw[:-3] for raw in frames) == data[offset:offset + 44]
    for raw in frames:
        assert VF.crc16_genibus(raw[:-2]) == int.from_bytes(raw[-2:], 'big')
    return frames


def peer_query(hs, tape, now):
    *_, start, end, phase = tape
    now[0] = end
    kind = VF.SESSION_RESPONDER_OVER_IDLE if phase else VF.SESSION_RESPONDER_IDLE
    hs.on_rx_audio(MK.synth_burst(hs.called, kind))
    if hs._held_answer is not None:
        hs._release_held_answer(round((end - tape[4]) * MK.FS))


@pytest.mark.parametrize('tape', TAPES, ids=[t[0] for t in TAPES])
@pytest.mark.parametrize('held', [False, True], ids=['quiet-head', 'held-head'])
@pytest.mark.parametrize('tail', ['lost', 'failed-located'])
def test_incomplete_l2x2_never_credits_or_emits_head(monkeypatch, tape, held, tail):
    """§3 CERTAIN; §2 OPEN/Q → INCOMPLETE → drop W, NAK-OWED.

    A located failure also exercises OPEN/X (or held/X); its query takes
    the NAK-OWED debt-first row or repeats the transmitted NAK.
    """
    _, _, offset, start, head_end, query_start, _, _ = tape
    now = [0.0]
    hs, sink = station(A, monkeypatch, now)
    data = prefix(hs, sink, tape, now)
    head, companion = damaged_frames(data, offset)
    assert hs._peer_double_verdict() == 'CERTAIN'
    before = len(sink.sent)
    decoded(hs, head, head_end, now, hold=held)
    assert len(sink.sent) == before
    assert hs._rx_window.W is not None
    assert hs._rx_window.state == 'OPEN'
    assert b''.join(sink.host) == data[:offset]
    assert hs._rx_window.R == data[:offset]
    if hs._rx_window.W is not None:
        assert hs._rx_window.W.expected == frozenset((2,))
    if tail == 'failed-located':
        failed = companion[:-1] + bytes([companion[-1] ^ 1])
        assert VF.crc16_genibus(failed[:-2]) != int.from_bytes(failed[-2:], 'big')
        now[0] = start + phy.envelope_lengths('2300', 2)[1] / MK.FS
        hs._irs_nak(phy.LevelRead(2, (), None, 1))
        assert hs._rx_window.A.kind == 'nak'
        assert b''.join(sink.host) == data[:offset]
    peer_query(hs, tape, now)
    assert hs._rx_window.W is None
    assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
    assert len(sink.sent) > before
    assert all('nak' in kind for kind in sink.packets[before:])
    assert b''.join(sink.host) == data[:offset]
    assert hs._rx_window.R == data[:offset]
    assert hs._decoded_frame_end is None or query_start > hs._decoded_frame_end
    assert not hs._rx_window.closed

    for index, at in enumerate(range(offset, len(data), 9)):
        payload = data[at:at + 9]
        body = phy.vara_body(payload, hs.caller, body_len=10)
        if len(payload) == 9:
            body = payload + b'\x99'
        hold = index >= 6 and (index - 6) % 2 == 0
        if hold and len(payload) < 9:
            marker = bytes([VF.crc16_genibus(hs.caller.encode()) >> 8])
            body = (payload + b'\x14' + marker)[:9].ljust(9, b'\0') + b'\x98'
        raw = TX.build_frame(body, 0)
        decoded(hs, raw, now[0] + 6.5, now, level=1, hold=hold)
    if hold or len(data[offset:]) % 9 == 0:
        decoded(hs, TX.build_frame(phy.vara_body(b'', hs.caller, body_len=10), 0),
                now[0] + 6.5, now, level=1)
    assert hs._rx_window.A.kind == 'final'
    assert hs._rx_window.R == data
    assert b''.join(sink.host) == data
    assert hs.state == A.VaraState.CONNECTED


@pytest.fixture(scope='module')
def release043(tmp_path_factory):
    """Extract the immutable release adapter and its dependencies, never patch it."""
    root = Path(__file__).resolve().parents[5]
    base = tmp_path_factory.mktemp('l2x2-release043')
    package = '_l2x2_release043'
    prefix = 'packages/hfmodem/hfmodem/'
    archive = subprocess.check_output(['git', 'archive', 'e597c9aca', prefix], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        for member in files:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix):]
            if rel.startswith('tests/') or not rel.endswith(('.py', '.json', '.npz', '.npy', '.bin')):
                continue
            target = base / package / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files.extractfile(member).read())
    sys.path.insert(0, str(base))
    try:
        yield importlib.import_module(package + '.kestrel.vara.vara_arq')
    finally:
        sys.path.remove(str(base))
        for name in list(sys.modules):
            if name == package or name.startswith(package + '.'):
                del sys.modules[name]


@pytest.mark.parametrize('tape', TAPES, ids=[t[0] for t in TAPES])
@pytest.mark.parametrize('held', [False, True], ids=['quiet-head', 'held-head'])
def test_release043_head_only_positive_loses_22_bytes(monkeypatch, release043, tape, held):
    """Old path credits both frames; the held derivative needs two late queries."""
    _, _, offset, _, head_end, _, _, _ = tape
    now = [0.0]
    hs, sink = station(release043, monkeypatch, now)
    data = prefix(hs, sink, tape, now)
    head, _ = damaged_frames(data, offset)
    before = len(sink.sent)
    decoded(hs, head, head_end, now, hold=held)
    assert b''.join(sink.host) == data[:offset + 22]
    if held:
        assert len(sink.sent) == before
    else:
        assert len(sink.sent) == before + 1
        np.testing.assert_array_equal(sink.sent[-1],
            MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE_SHORT))
    peer_query(hs, tape, now)
    if held:
        assert len(sink.sent) == before
        later = (*tape[:5], tape[5] + 3.6, tape[6] + 3.6, tape[7])
        peer_query(hs, later, now)
    assert len(sink.sent) > before
    positive = MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE_SHORT)
    np.testing.assert_array_equal(sink.sent[-1], positive)
    for at in range(offset + 44, len(data), 22):
        payload = data[at:at + 22]
        body = payload + b'\x99' if len(payload) == 22 else phy.vara_body(payload, hs.caller, body_len=23)
        decoded(hs, TX.build_frame(body, 1), now[0] + 6.5, now)
    expected = data[:offset + 22] + data[offset + 44:]
    assert b''.join(sink.host) == expected
    assert len(data) - len(expected) == 22
    assert b''.join(sink.host) != data
