# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Native captured cutoff and both physical carrier hints, no resampling."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.shrike import p3acquire, p3rx, placement, rxfront


@pytest.mark.parametrize('swapped', [False, True])
@pytest.mark.parametrize('missing', [0, 576])
def test_weak_sl3_preserves_physical_order_without_field_ranking(monkeypatch, swapped, missing):
    audio = np.pad(placement.link_packet(3, b'physical order', 0x21,
                                        swapped=swapped), (4800, 4800))
    at = 4800 + (placement.protocol_config().pulse().size-1)//2 + 4320
    span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[3]))
    audio = audio[:at+span-missing]
    sync = rxfront.SyncedRx()
    header = sync._wideband_header_candidate_at(audio, at)
    assert header.swapped == swapped
    monkeypatch.setattr(sync, '_wideband_header_candidate_at',
                        lambda *a: replace(header, fit=.1))
    monkeypatch.setattr(sync, '_field_shape',
                        lambda *a: pytest.fail('SL3 must not rank invariant permutation weights'))
    event, owned = sync.wideband_packet_at(audio, at, allow_short_fallback=True)
    assert owned and event is not None
    assert event.carrier_swapped == swapped
    assert event.start == at and event.cycle_long is False
    assert event.packet == (3, 0x21, b'physical order', True)


def test_native_hold56_driver_cutoff_one_crc_at_retained_offset(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    path = root / 'captures/onair-0914-1835/stream.wav'
    if not path.exists():
        pytest.skip('external native capture not present')
    fs, record = wavfile.read(path, mmap=True)
    assert fs == 48000
    row, end, prefix = 5273079, 5306880, 6720
    raw = record[row-prefix:end]
    if raw.ndim == 2:
        raw = raw[:, 0]
    assert raw.ndim == 1
    corrected = p3acquire.compensate(raw.astype(float)/32767, -66.8)
    decode = p3rx.decode_at
    calls = []
    def counted(*args, **kwargs):
        calls.append((args[1], args[2], kwargs['header'].long_cycle))
        return decode(*args, **kwargs)
    monkeypatch.setattr(p3rx, 'decode_at', counted)
    event, owned = rxfront.SyncedRx().wideband_packet_at(
        corrected, prefix, allow_short_fallback=True, can_decode=lambda: True)
    assert calls == [(prefix, 3, False)]
    assert owned and event is not None
    assert 'short fallback' in event.text
    assert event.start == prefix and event.carrier_swapped is False
    assert event.packet == (3, 0x21,
        b' Trimode 1.4.3.0\r\nW9SSJ has 71 daily minutes remaining with', True)
