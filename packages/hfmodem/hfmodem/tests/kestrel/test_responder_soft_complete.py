# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Soft responder pairs retain real weak controls, not missing/contrary symbols."""
import pytest

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_noisy_close_1023 import audio, CUT, CUT_START, STOCK, KESTREL
from .test_responder_control_stream import station


@pytest.mark.skipif(not CUT.is_dir(), reason=f'recording not present ({CUT})')
@pytest.mark.parametrize('name,first,expected', [('noise09-answerer', 75.18, 10),
                                              ('noise055-answerer', 75.01, 11)])
def test_every_previously_read_stock_ack_retains_soft_complete_acceptance(name, first, expected):
    x = audio(name)
    hs, _ = station(called=KESTREL)
    hs.caller = STOCK
    pairs = F.control_bursts(STOCK, '500', 1)[0]
    family = A._control_family(STOCK, KESTREL, '500')
    accepted = 0
    for k in range(11):
        t = first + 2.24*k - CUT_START
        clip = x[int((t-.3)*M.FS):int((t+1)*M.FS)]
        power = A._band_power(clip, hs._band)
        if A._soft_pairs(power, pairs, family, hs._band) is None:
            continue
        accepted += 1
        track = hs._cached_control(clip, band=hs._band)
        assert hs._complete_responder_pair(clip, pairs, track, power), (name, k)
    assert accepted == expected


@pytest.mark.parametrize('seed', range(32))
@pytest.mark.parametrize('rms', [.01, .05, .2])
def test_truncated_handover_with_noise_tail_cannot_take_turn(seed, rms):
    import numpy as np
    from .test_responder_control_stream import feed
    hs, io = station()
    hs._answer_owed = A._OWED_RELEASE
    pairs = F.caller_turn_grant(hs.called, '500')
    frame = M.synth_tone_pairs(pairs)
    tail = np.random.default_rng(seed).normal(0, rms, 3*M.HOP)
    feed(hs, np.concatenate([np.zeros(4800), frame[:-3*M.HOP], tail, np.zeros(24000)]))
    assert hs.turn == A._TURN_PEER and hs._answer_owed == A._OWED_RELEASE
    assert not io.sent
