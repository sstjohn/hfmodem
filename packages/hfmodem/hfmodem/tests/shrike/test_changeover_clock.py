# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A CRC-valid changeover supplies the clock for the first normal P3 packet."""
import json

import pytest

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.archive import P3_FIXTURES, requires_ws8eoc_p3
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_gateway_acquisition import audio


@requires_ws8eoc_p3
@pytest.mark.parametrize("reader", ["scan", "anchored"])
@pytest.mark.parametrize("first,next_frame", [
    (100000, 160000),       # Adjacent cycles in the composite bench.
    (1424880, 2444640),     # Actual Sept 9 stream origins, almost 17 cycles apart.
])
def test_first_recorded_data_can_use_the_changeover_clock_without_a_blind_scan(
        reader, first, next_frame):
    s = _Session(entry_pending=True)
    samples = audio("ws8eoc-0909-p3-rms.wav")
    if reader == "scan":
        onair._scan_frame(s.rx, samples, first)
    else:
        s.rx.control_signal(samples, first, first + 3840)
    assert s.host.arq.role == arq.IRS
    assert bytes(s.host.channel(s.host.ptchn).rx) == b"RMS"
    s.rx.new_cycle()
    onair._scan_frame(s.rx, audio("ws8eoc-0909-p3-trim.wav"), next_frame,
                      tracked_only=True)
    assert bytes(s.host.channel(s.host.ptchn).rx) == b"RMS Trim"
    assert s.rx.sync.tracked == 1


@requires_ws8eoc_p3
def test_changeover_clock_leaves_decode_time_before_the_first_data_reply():
    s = _Session(entry_pending=True)
    onair._scan_frame(s.rx, audio("ws8eoc-0909-p3-rms.wav"), 100000)
    s.rx.new_cycle()
    deadline = 208000
    ready = onair._p3_frame_ready(s.rx, deadline)
    assert 204000 < ready < deadline - round(.0162 * onair.FS)


@requires_ws8eoc_p3
def test_incoming_clock_is_cleared_when_we_later_take_the_link_back():
    s = _Session(entry_pending=True)
    onair._scan_frame(s.rx, audio("ws8eoc-0909-p3-rms.wav"), 100000)
    s.rx.new_cycle()
    assert s.rx._p3_row0 is not None
    s.host.arq.role = arq.ISS
    s.rx.new_cycle()
    assert s.rx._p3_row0 is None
    assert s.rx.sync.packet_at is None


@requires_ws8eoc_p3
def test_short_heads_without_a_crc_body_do_not_seed_a_data_clock():
    s = _Session(entry_pending=True)
    rows = json.loads((P3_FIXTURES / "ws8eoc-0908-p3.json").read_text())
    for row, due in zip(rows[:2], [653749, 713690]):
        samples = audio(row["file"])
        s.rx.new_cycle()
        s.rx.control_signal(samples, row["end_stream_sample"] - len(samples), due)
    assert s.host.arq.role == arq.IRS
    assert s.rx._p3_row0 is None


@pytest.mark.parametrize("reader", ["scan", "anchored"])
@requires_ws8eoc_p3
def test_changeover_preserves_measured_long_clock_and_is_delivered_once(reader):
    s = _Session(entry_pending=True)
    samples = audio("ws8eoc-0909-p3-rms.wav")
    s.rx._p3_cycle_n = 180000
    s.rx._p3_span = 160000
    if reader == "scan":
        onair._scan_frame(s.rx, samples, 100000)
    else:
        s.rx.control_signal(samples, 100000, 103840)
    assert s.rx._p3_cycle_n == 180000
    assert s.rx._p3_span == 160000
    count = s.rx.count
    s.rx.new_cycle()
    onair._scan_frame(s.rx, samples, 100000)
    assert s.rx.count == count
    assert bytes(s.host.channel(s.host.ptchn).rx) == b"RMS"
