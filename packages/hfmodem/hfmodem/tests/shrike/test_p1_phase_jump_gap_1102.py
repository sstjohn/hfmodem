# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A PACTOR-1 phase jump judged across cycles nothing was read in.

WS8EOC-80 onair-20261102T015731: CS1 read live at 13.63 N, 14.83 I, 17.37 N,
18.62 I, 23.62 I. The 16.12 reply -- I again, the acceptance, read offline at
0 bit errors -- answered TX[13] on slot 12. The grid read of 14.83 behind that
key keyed TX[14] at once, and the span between the two carriers reached only
the rolling flush (`test_p1_unread_before_key_1102`). onair-20261102T010129:
4.88 I, then 7.37 N and 9.87 N with 6.12 and 8.62 unread."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

N, I = 0, 1


def _jumps(reads):
    host = object.__new__(PtcHost)
    host._p1_last_sense = None
    return [host._p1_phase_repeats(SimpleNamespace(protocol=Protocol.PACTOR1, t=t, sense=s))
            for t, s in reads]


def test_0157_jump_across_two_lost_slots():
    reads = [(13.63, N), (14.83, I), (17.37, N), (18.62, I), (23.62, I)]
    assert _jumps(reads) == [False, False, True, False, False]


def test_0101_jump_across_an_unread_cycle():
    reads = [(4.88, I), (7.37, N), (9.87, N), (11.12, I), (12.38, N), (13.62, I),
             (14.87, N), (16.13, I), (17.38, N), (18.63, I), (19.83, N), (22.37, N)]
    assert _jumps(reads) == [False, True] + [False] * 10


@pytest.mark.parametrize('gaps', [(2,), (3,), (4,), (2, 3, 4), (3, 2, 1, 4)])
def test_alternation_with_gaps_is_never_a_jump(gaps):
    t, k, reads = 2.0, 0, [(2.0, N)]
    for n in gaps:
        k += n
        reads.append((t + 1.25 * k + (.04 if n % 2 else -.04), k % 2))
    assert not any(_jumps(reads))


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_one_jump_inside_any_judged_gap_is_found(n):
    repeated = I if n % 2 else N
    assert _jumps([(2.0, I), (2.0 + 1.25 * n, repeated)]) == [False, True]


@pytest.mark.parametrize('dt', [6.25, 7.5, 1.25 * 1.5, 1.25 * 2.4, .6])
def test_a_gap_too_long_or_off_the_cycle_is_never_judged(dt):
    for second in (N, I):
        assert _jumps([(2.0, I), (2.0 + dt, second)]) == [False, False]


def _bursts(senses):
    import numpy as np
    from hfmodem.shrike import onair, pactor1
    return [np.zeros(0, np.float32) if s is None else onair._trim_silence(
        np.asarray(pactor1.control_signal(pactor1.CS_ACK_A, invert=s), np.float32))
        for s in senses]


@pytest.mark.parametrize('senses, accepted', [
    ((I, N, None, I, N, I, N, I, N, I), True),
    ((I, N, None, N, I, N, I, N, I, N), False),
], ids=['jump-in-the-gap', 'alternation-across-the-gap'])
def test_replayed_gap_keys_the_shorts_only_on_a_jump(tmp_path, monkeypatch, senses, accepted):
    from hfmodem.tests.shrike.test_p1_jump_to_p4_0929 import _arm
    keyed, log, _ = _arm(tmp_path, monkeypatch, _bursts(senses), 'shorts')
    shorts = [what for _, what in keyed if what.startswith('P4 PROBE')]
    assert bool(shorts) is accepted, keyed
    assert ("the peer's acceptance of the announcement" in log) is accepted


def test_capture_loss_forgets_the_last_sense(tmp_path):
    """Three real cycles, one of them lost to the input, read as two: N then I is
    ordinary alternation and must not become an acceptance."""
    from hfmodem.shrike import onair
    host = object.__new__(PtcHost)
    host._p1_last_sense = None
    ev = lambda t, s: SimpleNamespace(protocol=Protocol.PACTOR1, t=t, sense=s)
    assert not host._p1_phase_repeats(ev(2.0, N))
    # A real cycle lost: its 60000 samples of silence past the block.
    live = SimpleNamespace(fs=48000, lost=60000, xruns=0, lost_at=[(120000, 60000, 60128 / 48000)])
    loss = onair._CycleEvidence(live, SimpleNamespace(host=host), None, tmp_path)
    assert loss.input_loss('hold')
    assert not host._p1_phase_repeats(ev(4.5, I))
