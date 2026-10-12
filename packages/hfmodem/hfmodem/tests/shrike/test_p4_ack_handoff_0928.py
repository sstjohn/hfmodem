# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import contextlib
import io
import sys

from hfmodem.shrike import onair
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_p1_robust_0926 import _peer_wav, FS


def test_main_hands_the_decoded_ack_to_p4_before_another_p1_cycle(tmp_path, monkeypatch):
    wav = tmp_path / 'peer.wav'
    _peer_wav(wav)
    ready, handed, keys = [], [], []
    word_read = onair._p4_hold_word
    def word(tx, host, live, raster, whole, origin, carrier_end):
        found = word_read(tx, host, live, raster, whole, origin, carrier_end)
        if found is None and tx.p4_probe.requested and not host.p1_stalled:
            assert host.arq.acked == 1
            assert not ready
            ready.append((tx.slot, live.sample_now()))
            actual_now = live.sample_now
            delayed = actual_now()+round(.560*FS)
            monkeypatch.setattr(live, 'sample_now', lambda: max(actual_now(), delayed))
        return found
    monkeypatch.setattr(onair, '_p4_hold_word', word)
    for name in ('_receive_changeover', '_regrid'):
        original = getattr(onair, name)
        def legacy(*args, _original=original, **kwargs):
            assert not ready, 'P1 receive/regrid ran after the held ACK requested P4'
            return _original(*args, **kwargs)
        monkeypatch.setattr(onair, name, legacy)
    tick = PtcHost.tick
    def p1_tick(self, **kwargs):
        assert not ready, 'P1 tick ran after the held ACK requested P4'
        return tick(self, **kwargs)
    monkeypatch.setattr(PtcHost, 'tick', p1_tick)
    probe = onair._run_p4_probe
    def handoff(tx, live, raster, **kwargs):
        assert ready and not kwargs.get('in_slot', False)
        handed.append(live.sample_now())
        return probe(tx, live, raster, **kwargs)
    monkeypatch.setattr(onair, '_run_p4_probe', handoff)
    monkeypatch.setattr(onair, '_run_p4_link', lambda *args: 'handoff pin complete')
    key = onair.RadioTx._tx
    def emitted(self, audio, what, *args, **kwargs):
        before = len(self.slots_used)
        key(self, audio, what, *args, **kwargs)
        if len(self.slots_used) > before and what.startswith('P4 PROBE'):
            keys.append((self.tx_audio_start, what))
    monkeypatch.setattr(onair.RadioTx, '_tx', emitted)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, 'argv', [
        'shrike.onair', '--replay', str(wav), '--max-cycles', '3',
        '--mycall', 'W9SSJ', '--dxcall', 'N9VC', '--dial', '7100910',
        '--outdir', str(tmp_path / 'out'), '--after-connect', 'p1-robust',
        '--p1-status-bits45', '2', '--p4-probe-values', '7,38',
        '--p4-probe-headers', 'A,A', '--p4-probe-cadence', 'consecutive', '--p4-link'])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        onair.main()
    assert len(ready) == len(handed) == 1
    assert len(keys) == 2, output.getvalue()
    assert keys[0][0] > handed[0]
    assert keys[1][0]-keys[0][0] == 60000
    assert 'v=7 hdr=A' in keys[0][1] and 'v=38 hdr=A' in keys[1][1]
