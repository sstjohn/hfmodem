# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WS8EOC hold174: weak header, valid empty packet, no response admission."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, p3acquire, p3presence
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import (
    opportunity, presence_session,
)


@pytest.fixture
def recorded():
    folder = Path(__file__).with_name('fixtures') / 'ws8eoc-1026'
    meta = json.loads((folder / 'weak-header-empty.json').read_text())
    fs, pcm = wavfile.read(folder / 'weak-header-empty.wav')
    assert fs == 48000
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == meta['pcm_sha256']
    return meta, pcm.astype(float) / 32768


@pytest.mark.parametrize('role', [arq.IRS, arq.ISS])
def test_current_recorded_crc_reopens_admission_without_delivery_credit(recorded, role):
    meta, audio = recorded
    session = presence_session(meta['row'], cfo=meta['offset_hz'])
    a = session.host.arq
    a.on_host_data(b'pending bytes')
    for _ in range(6):
        opportunity(a, tick=False)
    assert a.p3_receive_only
    # An unanswered local turn can leave ISS on the retained receive raster.
    a.role = role
    before = bytes(a._outbuf), a.acked, a.rx_seq
    token = a.begin_receive_opportunity()
    tx = SimpleNamespace(slot=179)
    raster = SimpleNamespace(boundary=lambda _: meta['row']+37000)
    onair._p3_qualify_presence(session.rx, raster, tx, audio, meta['origin'])
    assert a._receive_opportunity.presence_source == 'tracked SL1 CRC'
    assert a.peer_qualified
    a.settle_peer_presence()
    assert not a.p3_receive_only
    assert a._dark_windows == a._p3_crc_wait_ticks == 0
    assert (bytes(a._outbuf), a.acked, a.rx_seq) == before
    a.finish_receive_opportunity(token)


@pytest.mark.parametrize('damage', ['loss', 'generation', 'deadline'])
def test_recorded_crc_cannot_bypass_attribution_or_read_budget(recorded, damage):
    meta, audio = recorded
    session = presence_session(meta['row'], cfo=meta['offset_hz'])
    a = session.host.arq
    token = a.begin_receive_opportunity()
    a._receive_opportunity.capture_lost = 0
    if damage == 'generation':
        session.rx._p3_presence_form = (a._link_generation-1, 1)
    elif damage == 'deadline':
        session.rx._p3_acquisition_fits = lambda reserve: False
    tx = SimpleNamespace(slot=179, live=SimpleNamespace(lost=int(damage == 'loss')))
    raster = SimpleNamespace(boundary=lambda _: meta['row']+37000)
    onair._p3_qualify_presence(session.rx, raster, tx, audio, meta['origin'])
    assert not a.peer_qualified
    a.finish_receive_opportunity(token)


@pytest.mark.parametrize('damage', ['previous', 'truncated', 'overlap', 'noise'])
def test_crc_presence_still_requires_current_complete_unmasked_packet(recorded, damage):
    meta, audio = recorded
    audio = p3acquire.compensate(audio, meta['offset_hz'])
    row, origin, gaps = meta['row'], meta['origin'], ()
    if damage == 'previous':
        row += 60000
    elif damage == 'truncated':
        audio = audio[:-2400]
    elif damage == 'overlap':
        gaps = [(row+10000, row+11000)]
    else:
        audio = np.random.default_rng(1027).normal(0, .1, len(audio))
    assert p3presence.tracked_structure(audio, origin, row, (1,), gaps=gaps) is None
