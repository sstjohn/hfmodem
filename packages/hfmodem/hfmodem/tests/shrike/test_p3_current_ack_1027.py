# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC: speculative SL2 must not spend the current SL1 ACK deadline."""
import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_entry_answer import _Session

FOLDER = Path(__file__).with_name('fixtures') / 'ws8eoc-1026' / 'acks'
PINS = json.loads((FOLDER / 'manifest.json').read_text())


@pytest.mark.parametrize('pin', PINS, ids=lambda p: f"tx{p['tx']}")
def test_current_crc_selects_ack_before_speculative_upgrade_spends_budget(pin, monkeypatch):
    fs, pcm = wavfile.read(FOLDER / f"tx{pin['tx']}.wav")
    assert fs == onair.FS
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == pin['pcm_sha256']
    s = _Session(role=arq.IRS)
    a, r = s.host.arq, s.rx
    a.cfg.long_cycle = False
    a._rx_seen = True
    a._expected_seq = pin['current_seq']
    r._p3_row0 = r._p3_delivered_at = pin['row']-60000
    r._p3_clock_role = arq.IRS
    r.sync.packet_level = 1
    r.p3_receive_offset_hz = pin['offset_hz']
    r.p3_wideband_prekey = True
    r.note_p3_speedup_emitted(1, 2)
    # Explicit virtual CPU budget, not a wall-clock or RF claim. One SL2
    # attempt fits, but spends the opportunity needed by the known SL1 body.
    remaining = [.015]
    monkeypatch.setattr(r, '_p3_acquisition_fits',
                        lambda reserve: remaining[0] >= reserve + onair.P3_DECODE_RESERVE_S)
    for name, charge in [('sl1_packet_at', .0033), ('sl2_packet_at', .00751)]:
        original = getattr(r.sync, name)
        def read(*args, _original=original, _charge=charge, **kwargs):
            remaining[0] -= _charge
            return _original(*args, **kwargs)
        monkeypatch.setattr(r.sync, name, read)
    onair._scan_frame(r, pcm.astype(float)/32768, pin['origin'], tracked_only=True)
    assert r._p3_delivered_at == pin['row']
    assert a.rx_seq == pin['current_seq']
    expected = arq.CS_ACK if pin['current_seq'] % 2 == 0 else arq.CS_REQUEST
    assert s.host.peer.sent[-1] == ('cs', expected)
