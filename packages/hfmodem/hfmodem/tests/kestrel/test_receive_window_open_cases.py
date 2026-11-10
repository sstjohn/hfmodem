# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Held companions remain unread until their retry completes."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A
from hfmodem.kestrel.vara import vara_mfsk as M
from .receive_window_helpers import station, replay_window, receive, feed, query, _feed, full_body
from .test_idle_poll import clock, _Clock  # noqa: F401

@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_held_double_with_undecodable_tail_resent_lower_is_delivered_once(
        monkeypatch, bw, role):
    h, io = station(bw, role)
    singles = [bytes([i]) * 47 for i in range(1, 7)]
    for p in singles:
        feed(h, replay_window(bw, p, level=3))
    a, b = bytes(range(47)), bytes(range(100, 147))
    real = A._RX2300.frame_at

    def tail_fails(mag, level):
        fr = real(mag, level)
        return replace(fr, crc_ok=False) if fr.crc_ok and fr.payload[:47] == b else fr

    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._KEYING)
        patch.setattr(A._RX2300, 'frame_at', tail_fails)
        feed(h, replay_window(bw, a, b, level=3))
    assert b''.join(io.host) == b''.join(singles)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.T is None
    stream = a + b
    for at in range(0, len(stream), 22):
        receive(h, replay_window(bw, stream[at:at + 22], level=2), 'stream')
    assert b''.join(io.host) == b''.join(singles) + a + b, io.msgs


@pytest.mark.parametrize('bw', ['2300', '2750'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('qdelay', [0.2, 6.35])
def test_quiet_misread_companion_still_decodes(monkeypatch, bw, role, qdelay):
    h, io = station(bw, role)
    singles = [bytes([i]) * 22 for i in range(1, 7)]
    for p in singles:
        feed(h, replay_window(bw, p, level=2))
    a, b = b'defghij\r\nHello Kestrel', b' from Xara. abcdefghij'
    data = replay_window(bw, a, b, level=2)
    b_reads = []
    real = A._RX2300.frame_at
    tail = full_body(b, h.caller, 23, 0x99)
    def read(mag, level):
        frame = real(mag, level)
        if frame.crc_ok and bytes(frame.payload) == tail:
            b_reads.append(True)
        return frame
    monkeypatch.setattr(A._RX2300, 'frame_at', read)
    before = len(io.sent)
    step = M.FS // 50
    with monkeypatch.context() as patch:
        patch.setattr(A, '_window_state', lambda *args: A._QUIET)
        for at in range(0, len(data), step):
            _feed(h, data[at:at + step])
            if h._held_answer is not None or len(io.sent) > before:
                break
    assert h._held_answer is not None and len(io.sent) == before
    assert b''.join(io.host) == b''.join(singles) and not b_reads
    feed(h, np.zeros(max(0, len(data) - at - step)))
    query(h, delay=qdelay)
    assert not b_reads
    receive(h, replay_window(bw, a, b, level=2), 'stream')
    assert b''.join(io.host) == b''.join(singles) + a + b, io.msgs
    assert b_reads

from .receive_window_helpers import rx_evidence  # noqa: F401
