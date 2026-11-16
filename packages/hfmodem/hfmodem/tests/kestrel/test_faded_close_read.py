# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The caller's close, with a symbol of its head faded, still closes the answerer.

Kestrel answering Kestrel, OpenSlot air loop, 3 dB moderate fading, seeds
20261109 and 20261110 (2026-11-10): the caller closed in the turn the
answerer granted, and the answerer never read it. Replayed through the cells'
own channels the 32-symbol close read 31 and 27 of 32 tones exact with none
wrong, but a faded symbol among the four the candidate screen looked at (3 to
6) kept the fit from running, and the link waited out the 60 s silence bound.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_idle_poll import clock as clock, _Clock as _Clock
from .test_responder_control_stream import feed, station


def _released():
    hs, io = station()
    hs.chat = hs._chat_profile = True
    hs._hold_released_turn()
    assert hs._release_turn()
    return hs, io


def _faded(x, symbols):
    """White noise 12 dB under the burst's power, with ``symbols`` faded into it."""
    x = x.copy()
    for k in symbols:
        x[k * M.HOP:(k + 1) * M.HOP] *= .01
    x = np.r_[np.zeros(4800), x, np.zeros(24000)]
    noise = np.random.default_rng(20261110).standard_normal(len(x))
    return x + noise * np.sqrt(np.mean(x[x != 0] ** 2)) * 10 ** (-12 / 20)


@pytest.mark.parametrize('symbols', [(5,), (3, 6, 10, 23, 28)], ids=['seed-09', 'seed-10'])
def test_a_close_faded_in_its_head_is_read(symbols):
    hs, io = _released()
    close = _faded(M.synth_burst(hs.called, F.for_bw(F.SESSION_DISCONNECT_REQ, '500')), symbols)
    feed(hs, close)
    assert hs.state is A.VaraState.DISCONNECTED, io.msgs[-3:]


@pytest.mark.parametrize('symbols', [(), (5,)], ids=['whole', 'admitted-later'])
def test_a_close_with_a_wrong_tone_is_not_read(symbols):
    hs, io = _released()
    kind = F.for_bw(F.SESSION_DISCONNECT_REQ, '500')
    close = M.synth_burst(hs.called, kind)
    other = M.synth_burst('N0ZZZ', kind)
    k = 20
    assert hs._burst_tones(hs.called, kind)[k] != hs._burst_tones('N0ZZZ', kind)[k]
    close[k * M.HOP:(k + 1) * M.HOP] = other[k * M.HOP:(k + 1) * M.HOP]
    sent = len(io.sent)
    feed(hs, _faded(close, symbols))
    assert len(io.sent) == sent
    assert hs.state is A.VaraState.CONNECTED, io.msgs[-3:]
