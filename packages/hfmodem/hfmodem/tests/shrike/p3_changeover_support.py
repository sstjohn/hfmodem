# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.onair_support import _Rig, _Session
from hfmodem.tests.shrike.p3_cycle_support import ElapsedPCM


FIXTURES = Path(__file__).with_name('fixtures') / 've3kpg-changeover-0920'
if not (FIXTURES / "metadata.json").is_file():
    pytest.skip("VE3KPG off-air fixtures are not included in this distribution",
                allow_module_level=True)
ROWS = json.loads((FIXTURES / 'metadata.json').read_text())
FS = 48000


class Capture(ElapsedPCM):
    holdback = 660
    _blk = 128
    _callback_timing = ((0, 128, 0.0, 0.0), (128, 128, .00004, .002667),
                        (256, 128, .00008, .005333))

    def take_until(self, until):
        # The real capture read trails the converter. Model callback batching
        # as well as time spent decoding; neither may be rewound by a read.
        end = min(len(self.audio), max(self.pos, int(until)))
        if end > self.pos:
            self.samples = max(self.samples, end + 384)
        audio = self.audio[self.pos:end]
        self.pos = end
        return audio

    def sample_now(self):
        return self.samples + self.holdback


def arm(tmp_path, row, damage=False, fixtures=FIXTURES):
    fs, pcm = wavfile.read(fixtures / row['file'])
    assert fs == FS
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == row['pcm_sha256']
    audio = pcm.astype(float) / 32768
    head = row['head'] - row['origin']
    if damage:
        audio[head + 12000:] = 0
    s = _Session(role=arq.ISS)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    s.host.arq.cfg.long_cycle = False
    s.rx.p3_receive_offset_hz = row['offset_hz']
    ack = row['first_ack_audio'] - row['origin']
    g = onair._MasterGrid(ack - 28800, 60000, 0, packet_n=38880,
                         cs_n=10080, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.acquired = g.corroborated = g.turn_accepted = True
    g.d_n, g.d_ref_n = 4800, 38880
    # The already emitted mail phase rotates with this accepted ISS turn.
    clock = ReplyClock(entry_phase=0, pulse_epoch=ack + 852 - 28800)
    g.reply_clock = clock
    tx = onair.RadioTx(_Rig(), transmit=True, settle=.04, outdir=tmp_path)
    tx.reply_clock = clock
    tx.attach(s.host)
    s.host.peer = tx
    live = Capture(audio, 1536)
    tx.live, tx.raster, tx.sessrx = live, g, s.rx
    tx.defer_p3_cs = True
    tx.p3_follow_offset = 'all'
    tx.aim(g, 0)
    # Decode only the head, as the ISS control window does. Its end must not
    # contain the body the new IRS is about to collect.
    live.pos = head + 12000
    live.samples = live.pos
    seg = audio[:live.pos]
    ev = s.rx.sync.control_signal_tracked(s.rx._corrected(seg), head, details=False)
    assert ev is not None and ev.cs == arq.CS_BREAKIN and ev.packet is None
    s.rx._on(ev, anchored=True)
    assert s.host.arq.role == arq.IRS
    return s, tx, live, g, seg
