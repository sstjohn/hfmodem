# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Bounded tracked windows retain the header of the last complete frame."""
import numpy as np
import pytest

from hfmodem.shrike import onair, p3frame, rxfront, spec
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
