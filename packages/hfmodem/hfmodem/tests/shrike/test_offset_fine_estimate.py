# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The peer's carrier offset is measured off the grid, not read off it.

Since 2026-09-13 our own PACTOR-3 is keyed at the session's receive offset,
and that offset was whatever hypothesis the
search happened to accept: 25 Hz apart on the changeover list, 5 Hz apart on the
control list. A hypothesis is not a measurement, and `p3acquire` accepts a
control only within about 8 Hz of true, so the grid alone leaves the transmitter
up to half a step off the raster it is aiming at.

`p3acquire._refined` takes the measurement out of the accept tensors the search
has already built: one symbol apart, a differential product's phase is the
data's 0 or pi plus 2.pi.df.Tsym, and the codeword is what takes the data off.
Held here against our own transmitter, against the grid it corrects, and against
three recorded WS8EOC cycles whose rows record the hypothesis and whose carriers
are somewhere else -- `test_p3_rx_recovery.CYCLE_13_TRUE_HZ`.

The second half of the file is `compensate`'s transform length, which is the
same file's other cost: see its docstring.

Run:  python -m pytest hfmodem/tests/shrike/test_offset_fine_estimate.py
"""
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import arq, onair, p3acquire, placement
from hfmodem.tests.shrike import test_control_offset_follow as follow
from hfmodem.tests.shrike import test_p3_reply_placement as reply

FS = onair.FS
FIXTURES = Path(__file__).with_name("fixtures")

# WHAT THE 2026-09-13 WS8EOC ARMS ACTUALLY SAT AT, measured three ways on their
# own captures and reproduced by `_refined` to 0.05 Hz: the 16:29 arm that lost
# the greeting at -26.0 Hz where the coarse grid said -25, and the two 15:50 and
# 15:56 arms that delivered it whole at +51.5 and +27.9 where the lists said +50
# and +30. Half a hertz of grid error separated the arm that failed from the two
# that worked, so the grid is not what lost it -- but the transmitter was still
# being aimed with a number nobody had measured.
ARM_OFFSETS_HZ = (-26.0, 51.5, 27.9)


def rendered(hz: float, cs: int = placement.BREAKIN_CS) -> np.ndarray:
    """One codeword as a peer `hz` off our nominal puts it on the air."""
    audio = onair._trim_silence(placement.historical_control_signal(cs))
    return np.pad(p3acquire.compensate(audio, -hz) * .5, (4800, 4800))


# -- the measurement ------------------------------------------------------

@pytest.mark.parametrize("hz", [-8.0, -26.0, 27.9, 51.5, -2.4, 6.3, -71.0])
def test_a_head_off_the_grid_is_reported_where_it_is_and_not_where_it_fits(hz):
    """The whole claim: the grid finds it, the refinement says where it is.

    `coarse_hz` is a member of the list that was swept and `offset_hz` is the
    carriers, within a hertz, whether or not the two coincide.
    """
    got = p3acquire.changeover(rendered(hz))
    assert got is not None and got.event.packet is None
    assert got.coarse_hz in p3acquire.OFFSETS_HZ
    assert abs(got.offset_hz - hz) <= 1.0, (got.coarse_hz, got.offset_hz)


@pytest.mark.parametrize("hz", [-8.0, -2.4, 6.3, -71.0])
def test_the_grid_alone_would_have_aimed_us_somewhere_else(hz):
    """...and the negative control: the hypothesis is the wrong answer.

    Every offset here is far enough off a coarse grid point that keying on the
    hypothesis would put our controls where these tests say they are not.
    """
    got = p3acquire.changeover(rendered(hz))
    assert abs(got.coarse_hz - hz) > 1.0
    assert abs(got.offset_hz - hz) < abs(got.coarse_hz - hz)


@pytest.mark.parametrize("hz", [0.0, 25.0, 50.0, -50.0, -75.0])
def test_a_peer_already_on_a_grid_point_is_left_exactly_where_it_is(hz):
    """The v10-shaped case, which must not move: +50 stays +50.

    Both arms that delivered WS8EOC's greeting whole were keyed off a grid
    hypothesis, so a refinement that nudged an on-grid peer would be changing
    the two arms that worked.
    """
    got = p3acquire.changeover(rendered(hz))
    assert got is not None
    assert got.coarse_hz == hz and got.offset_hz == hz


def test_the_control_list_is_refined_the_same_way():
    """`control_signal`'s 5 Hz list has the same gap, a fifth as wide."""
    for hz in (-2.4, 6.3, 27.9):
        got = p3acquire.control_signal(rendered(hz, arq.CS_ACK))
        assert got is not None and got.event.cs == arq.CS_ACK
        assert got.coarse_hz in p3acquire.CONTROL_OFFSETS_HZ
        assert abs(got.offset_hz - hz) <= 1.0, (hz, got.coarse_hz, got.offset_hz)


def test_a_residual_is_a_correction_and_never_a_search():
    """`FINE_LIMIT_HZ` bounds it to half the coarse step, in both directions."""
    diffs = np.zeros((2, 1, 1, 20), complex)
    for turn, expected in ((.49, p3acquire.FINE_LIMIT_HZ),
                           (-.49, -p3acquire.FINE_LIMIT_HZ)):
        diffs[...] = np.exp(2j * np.pi * turn)
        signs = np.ones((1, 1, 20))
        assert p3acquire._refined(diffs, signs, 0, 0, 0.0) == expected


# -- what the transmitter does with it ------------------------------------

@pytest.mark.parametrize("hz", [-8.0, -26.0, 27.9])
def test_the_follow_keys_the_measured_offset_and_not_the_grid_point(
        tmp_path, monkeypatch, hz):
    """End to end through the production emit, the round-13 seam unchanged.

    A session that acquired this peer at a frequency no list holds keys CS1
    there, and our own reader measures the emission back at it. The transmit
    clamp is still the only thing between the session's number and the carrier.

    NOT THAT THE GRID POINT WOULD HAVE MISSED, which is the claim this must not
    make: `p3acquire` reads a control within about 8 Hz either way, so a peer a
    few hertz off a hypothesis is legible from the hypothesis too. What changes
    is that the number the transmitter is aimed with has been measured.
    """
    g, tx, slot, emitted = follow.bench(tmp_path, monkeypatch, hz)
    assert reply.key_ack(tx, g, slot) is None and not tx.refused
    assert follow.read_control(follow.keyed(emitted)) == (arq.CS_ACK, hz)


# -- the transform length -------------------------------------------------

def bare_compensate(audio, hz):
    """`compensate` on the window's own length, which is what it used to be."""
    if not hz:
        return audio
    x = np.asarray(audio, dtype=float)
    return (hilbert(x) * np.exp(-2j*np.pi*hz*np.arange(len(x))/FS)).real


def named(got):
    if got is None:
        return None
    ev = got.event
    return (ev.cs, ev.start, round(ev.t, 9), got.offset_hz, got.coarse_hz,
            ev.text, None if ev.packet is None else repr(ev.packet))


def test_the_padded_transform_finds_the_same_words_over_the_whole_corpus(
        monkeypatch):
    """What the pad may not change is what the readers RETURN.

    Neither length is the truth -- a finite window's Hilbert transform wraps
    either way -- so the bound that matters is behavioural, and it is the one
    `test_slot_deadline` set for the matched filter's own rewrite. Every PCM16
    fixture in the tree, swept in 460 ms brackets through both searches:
    codeword, start, instant, both frequencies, body and text must match, and
    the ranking quality with them.
    """
    recordings = sorted(FIXTURES.rglob("*.wav"))
    if not recordings:
        pytest.skip(f"no shrike recordings to sweep under {FIXTURES}")
    bad, tried, found = [], 0, 0
    for path in recordings:
        try:
            with wave.open(str(path)) as wav:
                if (wav.getframerate(), wav.getsampwidth(),
                        wav.getnchannels()) != (FS, 2, 1):
                    continue
                raw = wav.readframes(wav.getnframes())
        except wave.Error:
            continue  # a float32 fixture; this reader wants PCM16.
        audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        for lo in range(0, max(1, audio.size - round(.46*FS)), round(.31*FS)):
            seg = audio[lo:lo + round(.46*FS)]
            if seg.size < round(.23*FS):
                break
            for search in (p3acquire.control_signal, p3acquire.changeover):
                fast = search(seg)
                with monkeypatch.context() as m:
                    m.setattr(p3acquire, "compensate", bare_compensate)
                    slow = search(seg)
                tried += 1
                found += slow is not None
                if named(fast) != named(slow):
                    bad.append((path.name, lo, search.__name__))
                elif slow is not None:
                    assert fast.quality == slow.quality
    assert tried > 500 and found > 20, (tried, found)
    assert not bad, bad


def test_an_awkward_window_length_is_what_the_pad_is_for():
    """2 x 41 x 1301 samples: `onair-0913-1629`'s own 2.222 s hold windows.

    Fifteen of that arm's slots went to overruns of 4.2 to 10.2 ms, and the
    cycle runs this twice.
    """
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(106646) * .3

    def timed(call):
        call()
        return min(_elapsed(call) for _ in range(5))

    padded, bare = timed(lambda: p3acquire.compensate(audio, -25.0)), \
        timed(lambda: bare_compensate(audio, -25.0))
    assert padded * 1.5 < bare, (padded, bare)


def test_the_refinement_is_not_a_cost_the_cycle_can_feel():
    """It reads forty terms out of an array the accept tensors already built."""
    audio = rendered(-26.0)

    def timed(call):
        call()
        return min(_elapsed(call) for _ in range(5))

    with_it = timed(lambda: p3acquire.changeover(audio))
    bare = p3acquire._refined
    try:
        p3acquire._refined = lambda *a: 0.0
        without = timed(lambda: p3acquire.changeover(audio))
    finally:
        p3acquire._refined = bare
    assert with_it - without < 1e-3, (with_it, without)


def _elapsed(call) -> float:
    t0 = time.perf_counter()
    call()
    return time.perf_counter() - t0
