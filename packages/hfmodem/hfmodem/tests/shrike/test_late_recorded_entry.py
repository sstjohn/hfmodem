# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Composite replays: late recorded P3 answers after a repeated grant train.

These join recordings from different sessions to exercise the host/receiver
transition. They are not a claim that VE3KPG transmitted a P3 answer.
"""
import json

import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.tests.shrike.archive import P3_FIXTURES, requires_ws8eoc_p3
from hfmodem.tests.shrike.test_granted_entry_retry import granted, repeat
from hfmodem.tests.shrike.test_p3_gateway_acquisition import audio


@requires_ws8eoc_p3
@pytest.mark.parametrize("requests,silent", [(4, 0), (7, 0), (7, 3)])
def test_recorded_changeover_at_late_entry_boundaries(requests, silent):
    host, keyed = granted()
    queued = bytes(host.arq._outbuf)
    # A first-try announcement is acknowledged by the grant itself
    # (`arq.PactorArq.on_rx_grant`), so only the host's bytes remain queued.
    assert queued == b"pending application bytes"
    for _ in range(requests):
        repeat(host)
    for _ in range(silent):
        host.tick()
    assert host.arq.entry_pending
    count = len(keyed.p3)
    receiver = onair._SessionRx(host, tag="LATE ENTRY")
    receiver.new_cycle()
    receiver.deep_scan(audio("ws8eoc-0909-p3-rms.wav"))
    assert host.protocol is spec.Protocol.PACTOR3
    assert host.arq.role == arq.IRS
    assert not host.arq.entry_pending
    assert not host.arq.upgrade_unanswered
    assert bytes(host.channel(host.ptchn).rx) == b"RMS"
    assert bytes(host.arq._outbuf) == queued
    host.tick()
    assert len(keyed.p3) == count  # The cycle must not send another entry.
    receiver.new_cycle()
    receiver.deep_scan(audio("ws8eoc-0909-p3-rms.wav"))
    assert bytes(host.channel(host.ptchn).rx) == b"RMS"
    assert bytes(host.arq._outbuf) == queued


@requires_ws8eoc_p3
def test_two_recorded_short_heads_confirm_after_seven_repeat_requests():
    host, keyed = granted()
    for _ in range(7):
        repeat(host)
    queued = bytes(host.arq._outbuf)
    receiver = onair._SessionRx(host, tag="LATE HEAD")
    rows = json.loads((P3_FIXTURES / "ws8eoc-0908-p3.json").read_text())
    for row, due in zip(rows[:2], [653749, 713690]):
        receiver.new_cycle()
        samples = audio(row["file"])
        result = receiver.control_signal(
            samples, row["end_stream_sample"] - len(samples), due)
        host.tick()
    assert result == arq.CS_BREAKIN
    assert host.arq.role == arq.IRS
    assert not host.arq.entry_pending
    assert bytes(host.arq._outbuf) == queued
    count = len(keyed.p3)
    receiver.new_cycle()
    receiver.deep_scan(audio("ws8eoc-0908-p3-rms.wav"))
    host.tick()
    assert bytes(host.channel(host.ptchn).rx) == b"RMS"
    assert len(keyed.p3) == count
    assert bytes(host.arq._outbuf) == queued
