# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A locally emitted turn must not erase the still-sending peer's RX clock."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, placement, rxfront
from hfmodem.tests.shrike.test_entry_answer import _Session

SEED = 2033186
FIRST = 1013123
FIXTURE = Path(__file__).with_name('fixtures') / 'pending-turn-0914.wav'
HOLD31 = FIXTURE.with_name('pending-turn-hold31-0914.wav')


def pending_session():
    s = _Session(role=arq.IRS)
    s.rx.new_cycle()
    s.rx._p3_row0 = SEED
    s.rx._p3_delivered_at = SEED
    s.rx._p3_clock_role = arq.IRS
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[4])
    s.rx.sync.packet_level = 4
    s.rx.sync.packet_at = 7200
    s.rx.p3_receive_offset_hz = -21.4
    a = s.host.arq
    a.on_host_data(b'ABCDEF')
    a.on_host_breakin()
    a.on_rx_packet(4, b'', 0, True)
    a.on_cycle()
    assert a.unconfirmed_breakin
    return s


def test_pending_turn_preserves_receive_clock_and_watermark():
    s = pending_session()
    for _ in range(3):
        s.rx.new_cycle()
        assert s.rx._p3_row0 == SEED
        assert s.rx._p3_delivered_at == SEED
        assert s.rx._p3_clock_role == arq.IRS
        assert s.rx.sync.packet_at == 7200


@pytest.mark.parametrize('ending', ['ack', 'cancel', 'disconnect'])
def test_old_clock_expires_even_without_another_role_change(ending):
    s = pending_session()
    s.rx.new_cycle()
    a = s.host.arq
    if ending == 'ack':
        a._on_ack()
    elif ending == 'cancel':
        a._inflight = None
    else:
        a.on_host_abort()
    assert not a.unconfirmed_breakin
    s.rx.new_cycle()
    assert s.rx._p3_row0 is None
    assert s.rx.sync.packet_at is None


def test_fresh_clock_assigned_to_current_role_survives_ack():
    s = pending_session()
    s.rx.new_cycle()
    s.rx._p3_row0 = SEED + 60000
    s.rx._p3_clock_role = arq.ISS
    s.host.arq._on_ack()
    s.rx.new_cycle()
    assert s.rx._p3_row0 == SEED + 60000


def test_confirmed_direction_change_still_discards_old_clock():
    s = pending_session()
    s.host.arq._on_ack()
    s.rx.new_cycle()
    assert s.rx._p3_row0 is None


def test_retention_does_not_arm_the_irs_early_transmit_path():
    s = pending_session()
    s.rx.p3_wideband_prekey = True
    sent = list(s.host.peer.sent)
    s.rx.new_cycle()
    assert not s.rx.wideband_prekey_active()
    assert s.host.peer.sent == sent


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason=f'the recorded retry {FIXTURE.name} is not in this tree')
def test_native_recorded_retry_reaches_arq_without_settling_our_bytes():
    fs, raw = wavfile.read(FIXTURE)
    assert fs == onair.FS
    pcm = raw.astype(float) / 32768
    s = pending_session()
    before = s.host.arq._buffer_raw
    s.rx.new_cycle()
    origin = FIRST + 29 * 60000 - 7200
    onair._scan_frame(s.rx, pcm, origin)
    assert len(s.packets) == 1
    assert s.packets[0].packet == (4, 0, b'', True)
    assert s.host.arq._peer_frame_cycle
    assert s.host.arq._peer_asked_for_channel
    assert s.host.arq.unconfirmed_breakin
    assert s.host.arq._buffer_raw == before
    s.rx.new_cycle()
    onair._scan_frame(s.rx, pcm, origin)
    assert len(s.packets) == 1


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason=f'the recorded retry {FIXTURE.name} is not in this tree')
def test_erased_clock_loses_the_tracked_read_of_the_same_retry():
    """Without the clock the retry is no longer TRACKED -- acquisition finds it.

    The clock is what the tracked reader has and the blind sweep does not, and
    that is what this file is about. The sweep used to come back empty here and
    now returns the same field 63 samples from where the clock put it, because
    `p3rx.header_anchors` scores a level's header block on the channels the
    level lights instead of on all sixteen: speed level 4 lights twelve of them
    and was being averaged against four dark ones. Same bytes, two independent
    readers, so the clock's own claim is unchanged.
    """
    _, raw = wavfile.read(FIXTURE)
    s = pending_session()
    s.rx.new_cycle()
    s.rx._p3_row0 = None
    s.rx.sync.packet_at = None
    onair._scan_frame(s.rx, raw.astype(float) / 32768,
                      FIRST + 29 * 60000 - 7200)
    assert [p.packet for p in s.packets] == [(4, 0, b'', True)]
    assert 'tracked' not in s.packets[0].text
    assert abs(s.packets[0].start - 7263) <= rxfront.SPS // 4


@pytest.mark.skipif(not HOLD31.exists(),
                    reason=f'the saved hold {HOLD31.name} is not in this tree')
@pytest.mark.parametrize('flowing', [False, True])
def test_sender_reads_saved_hold_while_its_turn_is_pending(flowing):
    meta = json.loads(HOLD31.with_suffix('.json').read_text())
    fs, raw = wavfile.read(HOLD31)
    assert fs == onair.FS
    s = pending_session()
    s.rx.new_cycle()
    sent = list(s.host.peer.sent)
    assert onair._scan_previous_window(
        s.rx, raw.astype(float)/32768, meta['start'],
        sending=True, flowing=flowing)
    assert s.packets[-1].packet == (4, 0, b'', True)
    assert s.host.arq._peer_frame_cycle
    assert s.host.arq.unconfirmed_breakin
    assert s.host.peer.sent == sent


def test_confirmed_sender_does_not_start_an_early_packet_scan(monkeypatch):
    s = pending_session()
    s.host.arq._on_ack()
    monkeypatch.setattr(onair, '_scan_frame',
                        lambda *a, **kw: pytest.fail('unexpected sender scan'))
    assert not onair._scan_previous_window(
        s.rx, np.ones(48000), 0, sending=True, flowing=False)


def test_empty_previous_window_never_starts_a_scan(monkeypatch):
    s = pending_session()
    monkeypatch.setattr(onair, '_scan_frame',
                        lambda *a, **kw: pytest.fail('empty scan'))
    assert not onair._scan_previous_window(
        s.rx, np.zeros(0), 0, sending=True, flowing=False)
