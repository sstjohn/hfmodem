# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A lost NAK does not invalidate DATA at a previously requested retry rate."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
import pytest
from hfmodem.kestrel.vara import vara_arq as A
from .test_resend_ladders_1020 import receiver
from .test_responder_control_stream import feed
from .test_idle_poll import clock  # noqa: F401


def recording():
    p = Path(__file__).parent / 'fixtures/kq4qwh-retry-1025'
    meta = json.loads((p / 'provenance.json').read_text())
    assert hashlib.sha256((p / 'retry.wav').read_bytes()).hexdigest() == meta['sha256']
    with wave.open(str(p / 'retry.wav')) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(float)/32768


def station():
    h, io = receiver('500', 8)
    h.role, h.caller, h.called = 'initiator', 'W9SSJ', 'KQ4QWH'
    h.turn = A._TURN_PEER
    # The query across our second direct NAK replaces that unheard step.
    h._rx_resend = (8, (False, True, True))
    h._expect_level = A._resend_level('500', *h._rx_resend)
    assert h._expect_level == 1
    h._expect_500 = h._next_500 = 1
    h._rx_failed = True
    h._answer_owed, h._owed_block = A._OWED_OVER, True
    h._peer_delivery_open = True
    h._start_compression('OFF')
    return h, io


def test_recorded_clean_previous_rate_is_not_discarded():
    h, io = station()
    x = recording()
    bodies = h._peer_data_over_500(x, len(x)/48000)
    assert bodies and all(isinstance(b, bytes) for b in bodies), io.msgs
    assert h._rx_resend is None and not h._rx_failed


@pytest.mark.parametrize('block', [960, 4096])
def test_recorded_retry_delivered_once_and_acknowledged(block):
    h, io = station()
    feed(h, recording(), block)
    delivered = b''.join(io.host)
    assert delivered and h._peer_over == 1, io.msgs
    assert not h._rx_failed and h._rx_resend is None
    assert not any('was not read' in m for m in io.msgs)
    feed(h, recording(), block)
    assert b''.join(io.host) == delivered


@pytest.mark.parametrize('case', ['unrequested-rate', 'our-echo', 'our-turn'])
def test_recorded_retry_still_requires_receive_ownership_and_valid_history(case):
    h, io = station()
    if case == 'unrequested-rate': h._rx_resend = (1, (True,))
    if case == 'our-turn': h.turn = A._TURN_OURS
    if case == 'our-echo':
        d = A._RX500.decode_stream(recording())
        h._keyed_bodies.update(bytes(f.payload)+bytes([f.marker]) for f in d.data_frames)
    assert h._peer_data_over_500(recording(), len(recording())/48000) == []
    assert not io.host


def test_query_replacing_unheard_nak_preserves_previously_requested_rate():
    h, io = station()
    h._rx_resend, h._expect_level = None, 8
    for _ in range(3):
        h._irs_nak(A._phy.LevelRead(h._expect_level, (), None, 1))
    assert h._expect_level == 4
    # The peer's query tells us the last direct NAK was unheard. Replacing
    # that ladder step must not erase the fact that L4 was requested.
    h._nak_unheard = True
    assert h._nak_owed_over()
    assert h._expect_level == 3
    assert h._nak_owed_over()
    assert h._expect_level == 1
    x = recording()
    assert h._peer_data_over_500(x, len(x)/48000)
    assert h._rx_resend is None
    # A new recovery transaction must not borrow the previous rates.
    h._expect_level = 1
    h._advance_resend(missing=True)
    assert not h._previous_base_retry_rate(4)
