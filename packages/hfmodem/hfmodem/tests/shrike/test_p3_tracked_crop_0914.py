# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bounded tracked windows retain the header of the last complete frame."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3frame, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _Session


@pytest.mark.parametrize('cycles', [1, 4, 20])
def test_tracked_crop_keeps_projected_header_without_retaining_all_history(cycles):
    s = _Session()
    r = s.rx
    r._tracked_only = True
    r._scan_origin = 0
    r._p3_row0 = 7200
    r._p3_span = 35760
    r._p3_cycle_n = round(spec.CYCLE_SHORT_S * onair.FS)
    # Last complete row zero is followed by 1.96 s of samples. The next
    # frame's data end is still just outside the window.
    row0 = r._p3_row0 + cycles*r._p3_cycle_n
    end = row0 + round(1.96*onair.FS)
    got = {}

    def read(audio, origin=None):
        got.update(samples=len(audio), origin=origin, at=r.sync.packet_at)
        return None

    r._read_p3_packet = read
    r._p3_packet(np.zeros(end, np.float32))
    assert got['at'] is not None
    assert got['origin'] + got['at'] == row0
    assert got['at'] >= p3frame.DATA_OFFSET*rxfront.SPS + 2400
    assert got['samples'] <= (r._p3_cycle_n + rxfront._packet_span(r._p3_span)
                              + p3frame.DATA_OFFSET*rxfront.SPS + 2400)
    assert got['origin'] > 0


def test_1002_hold11_reads_the_requested_sl3_before_its_ack(tmp_path, monkeypatch):
    fixture = Path(__file__).with_name('fixtures') / 'ws8eoc-speed-1002' / 'hold11.json'
    meta = json.loads(fixture.read_text())
    recording = evidence.CAPTURES / 'onair-1002-1132' / 'stream.flac'
    if not recording.is_file():
        pytest.skip(f'local recording unavailable: {recording}')
    assert hashlib.sha256(recording.read_bytes()).hexdigest() == meta['source_sha256']
    audio = rxfront.load_wav(str(recording))[
        round(28.30 * onair.FS):round(29.357 * onair.FS)]
    assert audio.size == round(1.057 * onair.FS)
    old = rxfront.SyncedRx()
    old.packet_at, old.packet_level = 15360, 2
    assert old.packet(audio, preferred_only=True) is None
    top = rxfront.SyncedRx()
    top.packet_at, top.packet_level = 15360, 3
    assert top.packet(audio, preferred_only=True, requested_level=7).packet[0] == 3
    s = _Session(role=arq.IRS)
    r = s.rx
    r._p3_row0 = round(28.30 * onair.FS) + 15360 - 60000
    r._p3_delivered_at = r._p3_row0
    r._p3_clock_role = arq.IRS
    r.sync.packet_level = 2
    r.p3_receive_offset_hz = 0.0
    r._p3_offset_fixed = True
    s.host.arq._p3_rx_sl = 2
    s.host.arq.cfg.long_cycle = False
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = 3
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    tx.attach(s.host)
    tx.sessrx = r
    monkeypatch.setattr(tx, '_tx', lambda *a, **k: setattr(tx, 'n', tx.n + 1))
    tx._send_p3_control(arq.CS_SPEED_UP)
    assert r._p3_expected() == (2, 3) and r.sync.packet_level == 2
    read = r._read_p3_packet
    reads = []

    def tracked(*args, **kwargs):
        got = read(*args, **kwargs)
        if got is not None:
            reads.append(got[1])
        return got

    monkeypatch.setattr(r, '_read_p3_packet', tracked)
    r.new_cycle()
    onair._scan_frame(r, audio, round(28.30 * onair.FS), tracked_only=True)
    assert [(ev.packet[0], ev.packet[1], len(ev.packet[2])) for ev in s.packets] == [(3, 3, 45)]
    assert r.sync.packet_level == 3
    assert r._p3_delivered_at == round(28.30 * onair.FS) + 15360
    assert reads == ['tracked'] and not r._p3_expected()
    assert hashlib.sha256(s.packets[0].packet[2]).hexdigest() == meta['payload_sha256']
