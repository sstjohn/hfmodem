# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC SL2 recovery must run after the next listening deadline is aimed."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid

FOLDER = Path(__file__).with_name('fixtures') / 'ws8eoc-1026' / 'carrier'
PINS = json.loads((FOLDER / 'manifest.json').read_text())


@pytest.mark.parametrize('room', [False, True])
def test_quiet_current_deadline_allows_crc_carrier_requalification(room, monkeypatch):
    s = _Session(role=arq.IRS)
    r, a, tx = s.rx, s.host.arq, s.host.peer
    a.cfg.long_cycle = False
    r.p3_receive_offset_hz = 87.8
    r._p3_offset_fixed = True
    r._p3_clock_role = arq.IRS
    r._p3_row0 = PINS[0]['row']-60000
    r._p3_span = rxfront._frame_span(placement.SPEED_PATHS[2])
    r.sync.packet_level = 2
    a._enter_receive_only()
    expired = [True]
    tx.live = SimpleNamespace(transmit=True, clamp_late=lambda _: int(expired[0]))
    tx.raster, tx.slot = irs_grid(), 1
    tx.key_instant = lambda *args: 10000000
    for pin in PINS:
        fs, pcm = wavfile.read(FOLDER / f"hold{pin['hold']}.wav")
        assert fs == 48000
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == pin['pcm_sha256']
        expired[0] = True  # new_cycle still sees the preceding expired aim.
        r.new_cycle()
        expired[0] = not room  # the driver now aims the upcoming opportunity.
        onair._scan_previous_window(r, pcm.astype(float)/32768, pin['origin'],
                                    sending=False, flowing=False)
    if room:
        assert r.p3_receive_offset_hz == pytest.approx(-.7, abs=.3)
    else:
        assert r.p3_receive_offset_hz == 87.8
    # Frequency qualification alone cannot grant transmit or byte ACK credit.
    assert a.p3_receive_only and a.acked == 0
    if room:
        pin = json.loads((FOLDER / 'next.json').read_text())
        fs, pcm = wavfile.read(FOLDER / 'hold26.wav')
        assert fs == 48000
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == pin['pcm_sha256']
        r.new_cycle()
        # One SL2 attempt fits; two frequency hypotheses would miss this
        # modeled deadline. Keep response deferred until presence settles.
        remaining = [.015]
        monkeypatch.setattr(r, '_p3_acquisition_fits', lambda reserve:
                            remaining[0] >= reserve + onair.P3_DECODE_RESERVE_S)
        original = r.sync.sl2_packet_at
        def read(*args):
            remaining[0] -= r.P3_SL2_RESERVE_S
            return original(*args)
        monkeypatch.setattr(r.sync, 'sl2_packet_at', read)
        a._rx_seen, a._expected_seq = True, 2
        onair._scan_frame(r, pcm.astype(float)/32768, pin['origin'], tracked_only=True)
        assert r._p3_delivered_at == pin['row']
        assert a.rx_seq == 2
        assert a.acked == 0
