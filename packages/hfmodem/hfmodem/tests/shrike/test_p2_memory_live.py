# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Memory ARQ in the live PACTOR-2 receiver.

`p2rx.BurstMemory` sums the copies of a field the peer repeats because we could
not read it. These pin that `_SessionRx` actually hands it to the decode on a
PACTOR-2 link, clears it where the run of copies ends, and changes nothing on
the HB9AK recordings, where the one-cycle path is already byte-exact.
"""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p2rx, pactor2, session, spec
from hfmodem.shrike.spec import Protocol
from hfmodem.tests.shrike.test_p2session import (CYCLE_S, FIRST_BURST_S, FS, REAL,
                                                  WINDOW_S, _burst, _linked, _window)

GAIN, NOISE = 0.45, 0.3
"""A faded repeat: the burst at 0.45 of its level under noise. Seeds 1 and 3
each arm the marker and fail alone; summed they decode."""


def _faded(seed):
    burst, field = _burst(level=2, seed=7)
    x = GAIN * _window(burst) + np.random.default_rng(seed).normal(0, NOISE, int(WINDOW_S * FS))
    return x, field


def _p2_link():
    host, seam = _linked()
    host.protocol = Protocol.PACTOR2
    return host, onair._SessionRx(host, tag='TEST')


def test_two_faded_repeats_that_fail_alone_deliver_combined():
    (one, field), (two, _) = _faded(1), _faded(3)
    for copy in (one, two):
        assert p2rx.decode_expected_burst(copy, FS) is None, 'premise: each copy fails alone'
    host, rx = _p2_link()
    assert rx._p2_packet(one) is None
    rx.new_cycle()
    got = rx._p2_packet(two)
    assert got is not None, 'the repeat must combine with the copy before it'
    path = pactor2.PATHS[2]
    assert got[0].packet[2] == spec.field_payload(field[:path.crc_bytes - 3])


def test_a_role_change_ends_the_run_of_copies():
    host, rx = _p2_link()
    assert rx._p2_packet(_faded(1)[0]) is None
    rx.new_cycle()
    host.arq.role = arq.ISS
    rx.new_cycle()
    assert rx._p2_packet(_faded(3)[0]) is None


def test_the_speculative_pass_on_a_p1_link_does_not_feed_the_memory():
    host, rx = _p2_link()
    host.protocol = Protocol.PACTOR1
    assert rx._p2_packet(_faded(1)[0]) is None
    host.protocol = Protocol.PACTOR2
    assert rx._p2_packet(_faded(3)[0]) is None


@pytest.mark.parametrize('path', REAL, ids=lambda p: p.name)
def test_hb9ak_replay_with_memory_stays_byte_exact(path):
    if not path.exists():
        pytest.skip(f'{path.name} is not in this checkout')
    audio = session.load_wav(str(path), FS)
    whole = {round(t, 2): f for t, f in p2rx.decode_bursts(audio, FS)}
    alone, combined = {}, {}
    memory = p2rx.BurstMemory()
    first, n = FIRST_BURST_S[path.name], round(WINDOW_S * FS)
    for i in range(int((audio.size / FS - first) / CYCLE_S)):
        at = first + CYCLE_S * i
        seg = audio[round((at - 0.25) * FS):][:n]
        if seg.size < n:
            break
        if (got := p2rx.decode_expected_burst(seg, FS)) is not None:
            alone[round(at, 2)] = got[2]
        if (got := p2rx.decode_expected_burst(seg, FS, memory=memory)) is not None:
            combined[round(at, 2)] = got[2]
    assert combined.keys() >= alone.keys()
    assert all(combined[t] == alone[t] for t in alone)
    assert all(whole[t] == f for t, f in combined.items() if t in whole)
