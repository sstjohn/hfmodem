# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from pathlib import Path

import pytest

from hfmodem.shrike import p4frame
from hfmodem.shrike.p4hold import BreakinHold

ROOT = Path(__file__).resolve().parents[5]
FS = 48000
PREFIX = b'\x01\x01\x01'


def frame(status, data=PREFIX+b'306\rCMS via WS8EOC >\r'):
    data = data.ljust(40, b'\x1e')
    return p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                data_bytes=40)


@pytest.mark.parametrize('status,acked', [(0x80, 3), (0x82, 1), (0x83, 2),
                                          (0xe1, 0), (0xe3, 2)])
def test_observed_prefix_passes_combined_status(status, acked):
    h = BreakinHold(FS)
    h._select('normal')
    h.acked_counter = acked
    assert h.status_ok(frame(status), combined=True)


@pytest.mark.parametrize('status,data', [
    (0x80, b'\x01\x02\x01text'),
    (0x80, b'text'),
    (0x84, PREFIX+b'text'),
    (0xa0, PREFIX+b'text'),
    (0xe5, PREFIX+b'text'),
])
def test_other_extended_frames_stay_single_copy_only(status, data):
    h = BreakinHold(FS)
    h._select('normal')
    h.acked_counter = 0
    assert not h.status_ok(frame(status, data), combined=True)


def test_recorded_0x80_copies_decode_combined_one_cycle_early():
    from hfmodem.shrike import onair, p4rx, p4short
    path = ROOT/'captures/onair-0929-0923/stream.flac'
    if not path.exists():
        pytest.skip('local WS8EOC 0923 recording absent')
    audio = onair.session.load_wav(str(path), FS)
    h = BreakinHold(FS)
    h.acked_counter = 3
    held = []
    for predicted in (4787712, 4847712):
        z = p4rx.audio_to_baseband(audio[predicted-2400:predicted+43200], FS)
        nominal = 1440+p4short.GEOM
        coherence, training, cfo = p4short.place_short(z, nominal, -3.29)
        scores = p4short.short_scores(z, first_training_start=training,
                                      frequency_offset=cfo, refine=0)[2]
        at = predicted+round((training-nominal)*FS/28800)
        held.append(h.normal(at, training-nominal, cfo, coherence, scores,
                             (z, nominal, -3.29, at)))
    assert held[0] is None
    assert held[1] is not None and len(h.members) == 2
    assert held[1].status.byte == 0x80 and held[1].data.startswith(PREFIX)
    assert not any(e['event'] == 'status-rejected' for e in h.events)
