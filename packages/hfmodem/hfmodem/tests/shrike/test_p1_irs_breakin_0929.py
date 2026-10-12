# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import onair, pactor1
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.tests.shrike.test_p1_robust_0926 import FS, PACKET_N, SLOT_N

# onair-0929-1240: 789 ms for the 6.08 s IRS window of hold 3, 659 ms for 4.83 s.
READ_MS_PER_S = 130
BREAKIN = 2
PAYLOADS = [b'[RMS Tri', b'mode-1.4', b'.4.0-B2F', b'HM$]\r']


class ChargedReplay(onair._ReplayInput):
    """A file whose converter also moves for the work charged against it."""

    now = 0

    def spend(self, n):
        self.now = max(self.now, self.pos) + n

    def sample_now(self):
        return float(max(self.now, self.pos))

    def clamp_late(self, index):
        return max(0, int(self.sample_now()) - index)


def peer_wav(path):
    """A peer that takes the link and then sends its greeting, on our grid."""
    out = np.zeros((BREAKIN + len(PAYLOADS) + 14) * SLOT_N, np.float32)
    for k in range(BREAKIN + len(PAYLOADS) + 4):
        if k < BREAKIN:
            burst = pactor1.control_signal(pactor1.CS_SPEED, invert=k % 2)
        elif k == BREAKIN:
            burst = pactor1.breakin_signal(b'', 100, invert=k % 2, lead_s=0, tail_s=0)
        elif k - BREAKIN - 1 < len(PAYLOADS):
            n = k - BREAKIN
            burst = pactor1.packet_signal(PAYLOADS[n - 1], 100, packet_count=n % 4,
                                          invert=k % 2, lead_s=0, tail_s=0)
        else:
            burst = pactor1.packet_signal(b'', 100, packet_count=(k - BREAKIN) % 4,
                                          invert=k % 2, lead_s=0, tail_s=0)
        rf = onair._trim_silence(np.asarray(burst, np.float32))
        at = k * SLOT_N + PACKET_N + round(.100 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def run_irs(tmp_path, monkeypatch):
    wav = tmp_path / 'peer.wav'
    peer_wav(wav)
    keyed, streams, made = [], [], []
    key, read = onair.RadioTx._tx, P4Receiver.read_breakin

    def stream(*args, **kwargs):
        streams.append(ChargedReplay(*args, **kwargs))
        return streams[-1]

    def tx(self, audio, what, *args, **kwargs):
        made.append(self)
        before = len(self.slots_used)
        key(self, audio, what, *args, **kwargs)
        if len(self.slots_used) > before:
            if what.startswith('P1 CS'):
                wave = onair._trim_silence(np.asarray(audio))
                cs = onair.p1rx.decode_control_signal(wave, 0, .120)
                keyed.append((what, self.tx_audio_start, cs[0], cs.sense))

    def charged(self, audio, origin, *, carrier_end):
        streams[0].spend(round(len(audio) * READ_MS_PER_S / 1000))
        return read(self, audio, origin, carrier_end=carrier_end)

    monkeypatch.setattr(onair, '_ReplayInput', stream)
    monkeypatch.setattr(onair.RadioTx, '_tx', tx)
    monkeypatch.setattr(P4Receiver, 'read_breakin', charged)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', [
        'shrike.onair', '--replay', str(wav), '--max-cycles', '12', '--hold', '10',
        '--mycall', 'W9SSJ', '--dxcall', 'VE3KPG', '--dial', '7089500',
        '--outdir', str(tmp_path / 'out'), '--after-connect', 'p1-robust',
        '--p1-status-bits45', '2', '--p1-robust-stall', '12',
        '--p1-robust-stall-then', 'silent', '--p4-entry-timeout', '25',
        '--p4-link', '--p4-data', 'none', '--p4-link-timeout', '3'])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        onair.main()
    host = made[0].host
    return keyed, bytes(host.channel(host.ptchn).rx), output.getvalue()


def test_irs_keys_every_cs_inside_its_cycle_and_reads_the_greeting(tmp_path, monkeypatch):
    keyed, rx, log = run_irs(tmp_path, monkeypatch)
    assert len(keyed) >= len(PAYLOADS), log
    greeting_acks = keyed[1:1+len(PAYLOADS)]
    assert [(at, cs, sense) for _, at, cs, sense in greeting_acks] == [
        ((BREAKIN+1+i)*SLOT_N+PACKET_N+round(.100*FS)+PACKET_N+round(.070*FS),
         pactor1.CS_ACK_B if i % 2 == 0 else pactor1.CS_ACK_A, i % 2)
        for i in range(len(PAYLOADS))]
    assert 'IS GONE' not in log and 'OUTSIDE the 1.25 s cycle' not in log, log
    assert b''.join(PAYLOADS) in rx, log


def test_irs_hold_word_reads_the_peer_position_not_the_window(monkeypatch):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    capture = Path(__file__).resolve().parents[5] / 'captures/onair-0929-1240/hold_03.flac'
    require_input(capture)
    audio = onair.session.load_wav(str(capture), FS)
    end = json.loads(capture.with_suffix('.json').read_text())['end_stream_sample']
    start = end - len(audio)
    live = Stream(np.zeros(end, np.float32))
    live.pos = end
    tx = Tx(live)
    tx.p4_probe.requested = False
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    raster = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=5760, d_max_n=6240)
    raster.sending = False
    method = 'read_hold_word' if hasattr(P4Receiver, 'read_hold_word') else 'read_breakin'
    reads, read = [], getattr(P4Receiver, method)

    def spy(self, audio, origin, **kwargs):
        reads.append((origin, len(audio)))
        return read(self, audio, origin, **kwargs)

    monkeypatch.setattr(P4Receiver, method, spy)
    assert onair._p4_hold_word(tx, host, live, raster, audio, start, start) is None
    (origin, n), = reads
    assert n <= round(.645 * FS)
    # The peer's CS1 at 27.363 s, the last of its bursts the 6.08 s window holds whole.
    assert origin <= round(27.363 * FS) and round(27.49 * FS) <= origin + n


@pytest.mark.parametrize('number,expected', [
    (2, [(16.116, 1), (17.368, 0), (18.615, 1)]),
    (3, [(22.369, 0), (24.863, 0), (26.113, 1)]),
    (4, [(29.867, 0), (31.110, 1), (32.364, 0), (33.611, 1), (34.864, 0)]),
    (5, [(37.362, 0), (38.614, 1), (39.864, 0)]),
])
def test_1240_hold_windows_contain_recorded_controls(number, expected):
    from hfmodem.shrike import p1rx, rxfront
    path = Path(__file__).resolve().parents[5]/f'captures/onair-0929-1240/hold_{number:02}.flac'
    require_input(path)
    audio = onair.session.load_wav(str(path), FS)
    origin = json.loads(path.with_suffix('.json').read_text())['end_stream_sample']-len(audio)
    controls = []
    for at, duration in rxfront._p1_cs_bursts(audio):
        cs = p1rx.decode_control_signal(audio, at, duration)
        if cs is not None and cs[1] == 0:
            assert cs[0] == pactor1.CS_ACK_A
            controls.append((round(origin/FS+at, 3), cs.sense))
    assert controls == expected
    assert p1rx.decode_p1_packets(audio, FS) == []


@pytest.mark.parametrize('cycles', [0, 1, 4])
@pytest.mark.parametrize('gap', [.010, .070, .130])
def test_bounded_irs_read_keeps_y27_at_turnaround_and_skipped_cycles(cycles, gap):
    from hfmodem.shrike import p4robusttx
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    carrier_end = FS
    at = carrier_end+cycles*SLOT_N+round(gap*FS)
    end = at+round(.27*FS)
    audio = np.zeros(end)
    wave = p4robusttx.robust_word('Y27')
    audio[at:at+len(wave)] = wave
    live = Stream(audio)
    live.pos = end
    tx = Tx(live)
    tx.tx_end = carrier_end
    tx.p4_probe.requested = False
    tx.p4_link_host.p4_after_first_ack = True
    raster = onair._MasterGrid(0, SLOT_N, 0, packet_n=PACKET_N, cs_n=5760, d_max_n=6240)
    raster.sending = False
    result = onair._p4_hold_word(tx, tx.p4_link_host, live, raster, audio, 0, carrier_end)
    assert result is not None and tx.p4_probe.peer_breakin
    assert any(o.word == 'Y27' and abs(o.stream_sample-at) <= 2 for o in tx.sessrx.p4.observations)
