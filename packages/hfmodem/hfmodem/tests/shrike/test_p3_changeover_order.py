# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Carry the CRC-confirmed CS3 arrangement without changing the legacy API."""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import modem, p3acquire, p3frame, p3rx, placement, rxfront
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm

FIXTURES = Path(__file__).with_name('fixtures')
INDEX = FIXTURES / 'ws8eoc-0912-first-changeover.json'
META = json.loads(INDEX.read_text()) if INDEX.exists() else None
requires_recording = pytest.mark.skipif(
    META is None or not (FIXTURES / 'ws8eoc-0912-first-changeover.wav').exists(),
    reason='September12 WS8EOC first-CS3 PCM fixture is absent')


@requires_recording
def test_real_first_rms_has_unique_carrier_order_and_legacy_tuple(monkeypatch):
    audio = p3acquire.compensate(recorded_pcm(META), -20)
    field, ok, swapped = p3rx.decode_changeover_details(audio, 3900)
    assert ok and field == bytes.fromhex('524d53002810') and swapped is False
    assert p3rx.decode_changeover(audio, 3900) == (field, True)
    original = p3rx.rx.case0_softs

    def opposite_only(*args, **kwargs):
        if tuple(kwargs['order']) == tuple(p3frame.VH_ORDER):
            return None
        return original(*args, **kwargs)

    monkeypatch.setattr(p3rx.rx, 'case0_softs', opposite_only)
    assert p3rx.decode_changeover_details(audio, 3900) == (b'', False, None)


@requires_recording
def test_real_cs3_event_retains_measured_arrangement():
    audio = p3acquire.compensate(recorded_pcm(META), -20)
    event = rxfront._cs_event(audio, placement.BREAKIN_CS, 0, 3900, 3900/48000, '')
    assert event.breakin and event.packet[2] == b'RMS'
    assert event.carrier_swapped is False


@pytest.mark.parametrize('swapped', [False, True])
def test_generated_cs3_metadata_and_existing_api_agree(swapped):
    audio = placement.changeover_packet(b'abc', 0x20, swapped=swapped)
    head = int(np.argmax(modem.matched_pulse(480)))
    field, ok, measured = p3rx.decode_changeover_details(audio, head)
    assert ok and measured == swapped
    assert p3rx.decode_changeover(audio, head) == (field, True)
    event = rxfront._cs_event(audio, placement.BREAKIN_CS, 0, head, head/48000, '')
    assert event.packet[2] == b'abc' and event.carrier_swapped == swapped


def test_absent_cs3_body_does_not_guess_carrier_order():
    assert p3rx.decode_changeover_details(np.zeros(50000), 3900) == (b'', False, None)
    assert p3rx.decode_changeover(np.zeros(50000), 3900) == (b'', False)
    event = rxfront._cs_event(np.zeros(50000), placement.BREAKIN_CS, 0, 3900, .08125, '')
    assert event.packet is None and event.carrier_swapped is None
