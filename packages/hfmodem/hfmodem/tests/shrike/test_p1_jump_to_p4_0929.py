# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`--p1-robust-stall-then shorts`: the P1 phase jump is the acceptance.

WS8EOC-80, onair-0929-2208: CS1 answered our bit-5 packet 1 at 200 Bd in
alternating shift twice, then in the shift of the cycle before (8.61 s, our
`SHIFT REALIGNED`), and in alternation again for the rest of the session.
A Trimode gateway answers that way when it has decoded the announcement, and
then waits for our P4 short; KY2D keyed a robust short on that CS1 and got P4.
The gateway's own bursts, cut from the stream, are laid on the replay grid."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_p1_robust_0926 import FS, PACKET_N, SLOT_N

STREAM = Path(__file__).resolve().parents[5] / 'captures/onair-0929-2208/stream.flac'
# The peer's CS1 onsets as the arm's grid read them, 4.86 s to 16.11 s.
BURSTS = (233034, 293178, 353108, 413082, 473130, 533226, 593130, 653034,
          713226, 773082)
JUMP = 3
PAD = round(.005 * FS)
_KEY = onair.RadioTx._tx


def _gateway(order):
    if not STREAM.exists():
        pytest.skip('local September 29 flight capture absent')
    stream = onair.session.load_wav(str(STREAM), FS)
    cut = [stream[at - PAD:at + round(.125 * FS)] for at in BURSTS]
    return [cut[k] for k in order]


def _wav(path, bursts):
    out = np.zeros((len(bursts) + 16) * SLOT_N, np.float32)
    for k, burst in enumerate(bursts):
        at = k * SLOT_N + PACKET_N + round(.090 * FS) - PAD
        out[at:at + burst.size] += burst
    onair.session.write_wav(str(path), out)


def _arm(tmp_path, monkeypatch, bursts, then):
    tmp_path.mkdir(exist_ok=True)
    wav = tmp_path / 'peer.wav'
    _wav(wav, bursts)
    keyed, key = [], _KEY

    def tx(self, audio, what, *a, **k):
        before = len(self.slots_used)
        key(self, audio, what, *a, **k)
        if len(self.slots_used) > before and not what.startswith('connect'):
            keyed.append((self.slot, what.split('  (')[0]))

    monkeypatch.setattr(onair.RadioTx, '_tx', tx)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    monkeypatch.setattr(sys, 'argv', [
        'shrike.onair', '--replay', str(wav), '--max-cycles', '3',
        '--mycall', 'W9SSJ', '--dxcall', 'WS8EOC', '--dial', '3595000',
        '--outdir', str(tmp_path / 'out'), '--after-connect', 'p1-robust',
        '--p1-status-bits45', '2', '--p4-probe-packet', 'robust',
        '--p4-probe-values', '6,9,8', '--p4-probe-headers', 'A,A,A',
        '--p4-probe-cadence', 'consecutive', '--p4-probe-listen', '2',
        '--p1-robust-stall', '12', '--p1-robust-stall-then', then,
        '--p4-entry-timeout', '20'])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    probe = tmp_path / 'out' / 'p4-probe.json'
    return keyed, log.getvalue(), json.loads(probe.read_text()) if probe.exists() else None


def test_the_phase_jump_keys_the_shorts_in_the_next_slot(tmp_path, monkeypatch):
    keyed, log, probe = _arm(tmp_path, monkeypatch, _gateway(range(len(BURSTS))), 'shorts')
    p1 = [(slot, what) for slot, what in keyed if what.startswith('P1 ')]
    assert [what for _, what in p1] == ['P1 pkt#1 200Bd 7B'] * (JUMP - 1), keyed
    shorts = keyed[len(p1):]
    assert [what for _, what in shorts] == [
        'P4 PROBE ROBUST 1/3 v=6 hdr=A', 'P4 PROBE ROBUST 2/3 v=9 hdr=A',
        'P4 PROBE ROBUST 3/3 v=8 hdr=A'], keyed
    assert shorts[0][0] == p1[-1][0] + 1, keyed
    assert 'the peer\'s acceptance of the announcement' in log, log
    assert 'falls back to 100 Bd' not in log, log
    assert probe['trigger_word'] == 'P1 phase-jump acceptance'
    assert probe['path'] == 'trigger'


def test_without_the_flag_the_jump_is_a_repeat_request(tmp_path, monkeypatch):
    keyed, log, _ = _arm(tmp_path, monkeypatch, _gateway(range(len(BURSTS))), 'silent')
    assert keyed and all(what.startswith('P1 pkt#1') for _, what in keyed), keyed
    assert keyed[JUMP - 1] == (keyed[JUMP - 2][0] + 1, 'P1 pkt#1 200Bd 7B'), keyed
    assert 'falls back to 100 Bd' not in log, log


def test_a_repeat_request_in_alternation_keys_nothing_new(tmp_path, monkeypatch):
    """The same bursts with the jump left out: every CS1 keeps alternating."""
    bursts = _gateway([k for k in range(len(BURSTS)) if k != JUMP])
    runs = [_arm(tmp_path / then, monkeypatch, bursts, then)[:2]
            for then in ('silent', 'shorts')]
    (silent, _), (shorts, log) = runs
    assert shorts == silent, (shorts, silent)
    assert all(what.startswith('P1 pkt#1') for _, what in shorts), shorts
    assert 'acceptance of the announcement' not in log, log


def _cs4_order():
    """0929-2204's order: the CS1 answer read twice, CS4 twice at 200 Bd, then
    the CS1 that acknowledges the 100 Bd packet in the shift of the cycle before."""
    words = [(pactor1.CS_ACK_A, 1), (pactor1.CS_ACK_A, 0), (pactor1.CS_SPEED, 1),
             (pactor1.CS_SPEED, 0), (pactor1.CS_ACK_A, 0)]
    return [onair._trim_silence(np.asarray(pactor1.control_signal(cs, invert=sense),
                                           np.float32)) for cs, sense in words]


def test_a_cs4_arm_is_unaffected(tmp_path, monkeypatch):
    (silent, _, before), (shorts, log, after) = [
        _arm(tmp_path / then, monkeypatch, _cs4_order(), then)
        for then in ('silent', 'shorts')]
    assert shorts == silent, (shorts, silent)
    assert 'acceptance of the announcement' not in log, log
    assert before == after and after['trigger_word'] == 'P1 ACK', after
