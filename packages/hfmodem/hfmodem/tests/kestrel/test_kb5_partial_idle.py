# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""KB5's recorded incoming prefix must survive the stale idle timer.

The local keepalive occupied the middle of a DATA-shaped frame on September 10.
This fixture ends before that historical interference: it proves suppression of
the same premature local keying, not recovery of the overwritten middle bytes.
"""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA

PREFIX = Path(__file__).with_name('fixtures') / 'kb5_20260910_pre_keepalive.wav'


class _IO(VA.VaraIO):
    def __init__(self):
        self.sent = []
        self.delivered = []

    def tx(self, samples):
        self.sent.append(samples)

    def key(self, on):
        pass

    def log(self, message):
        pass

    def data(self, payload):
        self.delivered.append(bytes(payload))


@pytest.mark.skipif(not PREFIX.exists(),
                    reason='the recorded KB5LZK prefix requires the source checkout '
                           '(fixtures/kb5_20260910_pre_keepalive.wav)')
def test_recorded_kb5_partial_frame_survives_idle_after_peer_gap_reack():
    with wave.open(str(PREFIX)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 48000)
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(float)/32768
    io = _IO()
    hs = VA.VaraStationHandshake(['W9SSJ'], io, bw='2750')
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'KB5LZK'
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    hs._answer_owed = VA._OWED_OVER
    hs._reack_frame = VA.OVER_CONTINUE_CAPTURED
    hs._reacks = 1
    hs._peer_over = 1
    # The measured preceding second rung is a generated 32-symbol continue.
    # Exercise the real bookkeeping instead of pre-setting its timer flags.
    assert hs._reack()
    assert len(io.sent) == 1
    io.sent.clear()
    for offset in range(0, len(audio), 960):
        hs.on_rx_stream(audio[offset:offset+960])
    assert hs._held_answer is None
    assert len(hs._ov_buf) > 2*48000
    assert not io.delivered
    retained = hs._ov_buf.copy()
    hs.idle_keepalive()
    assert io.sent == [], 'The already-served peer gap cannot key over incoming DATA'
    np.testing.assert_array_equal(hs._ov_buf, retained)
    assert hs._answer_owed == VA._OWED_OVER

