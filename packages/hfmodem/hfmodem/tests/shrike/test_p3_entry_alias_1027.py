# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC trial B: first body must prevent another entry on the peer's turn."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench

BASE = Path(__file__).with_name('fixtures')/'ws8eoc-1027'/'entry-first-body'


@pytest.mark.parametrize('damaged', [False, True])
def test_recorded_first_body_at_alias_before_entry_retry(tmp_path, damaged):
    pin = json.loads(BASE.with_suffix('.json').read_text())
    fs, raw = wavfile.read(BASE.with_suffix('.wav'))
    assert fs == onair.FS
    assert hashlib.sha256(raw.tobytes()).hexdigest() == pin['sha256']
    audio = raw.astype(float)/32768
    if damaged:
        # Preserve the head but destroy the body: no role change or ACK.
        start = pin['head']-pin['origin']+12000
        audio[start:] = np.random.default_rng(1027).normal(0, .01, len(audio)-start)
    s = _Session(entry_pending=True)
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    s.host.peer = tx
    tx.defer_p3_cs = True
    live = _Bench(seconds=15)
    live.pos = live.now = pin['end']
    grid = onair._MasterGrid(2765, 60000, 8880,
                            packet_n=46080, cs_n=5760, d_max_n=6240)
    grid.protocol = spec.Protocol.PACTOR3
    tx.raster = grid
    tx.aim(grid, 7)
    s.rx._p3_entry_body_candidate = (pin['head'], pin['alias_hz'])
    onair._finish_entry_candidate(live, grid, tx, s.host, s.rx, 7,
                                 audio, pin['origin'], 1920)
    assert tx.n == 0
    if damaged:
        assert not s.packets and s.host.arq.entry_pending
        assert s.host.arq.role == arq.ISS
        assert s.rx.p3_receive_offset_hz == 0
    else:
        assert len(s.packets) == 1 and len(s.packets[0].packet[2]) == 3
        assert s.host.arq.role == arq.IRS and not s.host.arq.entry_pending
        assert not grid.sending
        assert s.rx.p3_receive_offset_hz == -2
        s.host.tick()
        assert tx.n == 0 and tx._pending_p3_cs == arq.CS_ACK
        # A second scan of the same physical body cannot deliver it twice.
        onair._scan_frame(s.rx, audio, pin['origin'])
        assert len(s.packets) == 1


def test_nominal_body_retry_obeys_decode_budget(monkeypatch):
    pin = json.loads(BASE.with_suffix('.json').read_text())
    _, raw = wavfile.read(BASE.with_suffix('.wav'))
    s = _Session(entry_pending=True)
    s.rx._p3_entry_body_checked_at = pin['head']
    s.rx._scan_origin = pin['origin']
    s.rx.p3_receive_offset_hz = pin['alias_hz']
    monkeypatch.setattr(s.rx, '_p3_acquisition_fits', lambda reserve: False)
    # Isolate the added fixed-head reader from the existing acquisition path.
    monkeypatch.setattr(onair.p3acquire, 'changeover', lambda *a, **kw: None)
    monkeypatch.setattr(onair.rxfront, '_cs_event',
                        lambda *a, **kw: pytest.fail('body read after deadline'))
    assert s.rx._p3_changeover_packet(raw.astype(float)/32768) is None
    assert s.rx.p3_receive_offset_hz == pin['alias_hz']
    assert s.host.arq.entry_pending
