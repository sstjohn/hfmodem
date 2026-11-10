# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2% trim taken in front of every key reduces the burst; it never walks it.

A walk is one C call boxing every sample with the GIL held. On a long-cycle
packet that outlasted the audio block, and the 2026-11-06 WS8EOC mail arm lost
capture at nearly every long-cycle key.
"""
import numpy as np
import pytest

from hfmodem.shrike import placement


class _Unwalkable(np.ndarray):
    def __iter__(self):
        raise AssertionError('the burst was iterated sample by sample')


def _walked(audio):
    return int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])


@pytest.mark.parametrize('long_cycle, payload', [(True, bytes(range(36))), (False, b'W9SSJ')])
def test_trim_matches_the_walk_without_walking(long_cycle, payload):
    audio = placement.link_packet(1, payload, 0x22, long_cycle=long_cycle)
    burst = audio.view(_Unwalkable)
    assert placement.leading_trim(burst) == _walked(audio)
    assert placement.pulse_lead(burst) == placement.pulse_lead(audio)
