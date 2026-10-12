# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""STANAG 4285 end to end, through the channels in ``bateleur.sim``.

Self-consistency only, as for 188-110.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur import s4285, sim, spec

MSG = bytes(np.random.default_rng(12).integers(0, 256, 200, dtype=np.uint8))


def _air(mode, **ch):
    a = np.concatenate([np.zeros(3000), s4285.transmit(MSG, mode), np.zeros(3000)])
    return sim.channel(a, **ch)


@pytest.mark.parametrize("mode", [m for m, v in spec.MODES4285.items() if v.coded])
def test_every_coded_mode_round_trips_to_its_frame(mode):
    """Annex E framing: the receiver hands back the message between the SOM
    and the EOM, and nothing of the flush around it."""
    r = s4285.receive(_air(mode, snr_db=22, freq_hz=-33.0, seed=1), mode)
    assert r.eom and r.data == MSG


@pytest.mark.parametrize("mode", ["3600U", "2400U", "1200U"])
def test_uncoded_modes_carry_a_bare_stream(mode):
    r = s4285.receive(_air(mode, snr_db=22, freq_hz=-33.0, seed=1), mode)
    assert r.data[: len(MSG)] == MSG and not r.eom


def test_the_flush_is_annex_e_s_table():
    """Annex E 5(f), all twelve entries."""
    table = {2400: (24678, 2150), 1200: (12390, 1126), 600: (6246, 614),
             300: (3174, 358), 150: (1638, 230), 75: (870, 166)}
    for rate, (long_, short) in table.items():
        assert spec.MODES4285[f"{rate}L"].flush_bits == long_
        assert spec.MODES4285[f"{rate}S"].flush_bits == short


def test_the_frame_patterns_are_annex_e_s():
    assert s4285.SOM4285_BITS == tuple(int(b) for b in f"{0x03873C3C:032b}")
    assert s4285.EOM_BITS == tuple(int(b) for b in f"{0x4B65A5B2:032b}")


@pytest.mark.parametrize("mode,snr", [("1200S", 15), ("600L", 15), ("150S", 8)])
def test_poor_channel(mode, snr):
    for seed in range(2):
        r = s4285.receive(_air(mode, snr_db=snr, preset="poor", freq_hz=18.0,
                               clock_ppm=-80, seed=seed), mode)
        assert r.data == MSG, seed


def test_the_wrong_mode_does_not_decode():
    r = s4285.receive(_air("1200S", snr_db=25, seed=2), "600S")
    assert r.data[: len(MSG)] != MSG


def test_a_first_frame_lost_in_a_fade_costs_nothing():
    """Acquisition a frame late: the convolutional deinterleaver must start
    at the first frame sent, so the frame missed is put back as erasures
    until the SOM appears -- in the batch receiver and the stream's."""
    from hfmodem.bateleur import phy, stream
    from hfmodem.core.resample import for_card
    msg = MSG * 3
    tx = s4285.transmit(msg, "1200L")
    first = phy.leadin_samples()
    tx[: first + s4285.FRAME4285 * spec.SPS] = 0.0          # the first frame, gone
    y = sim.channel(np.concatenate([np.zeros(4000), tx, np.zeros(4000)]),
                    snr_db=25, freq_hz=12.0, seed=1)
    r = s4285.receive(y, "1200L")
    assert r.eom and r.data == msg
    card = for_card(y.astype(np.float32), spec.FS)
    fe, rx = stream.FrontEnd(48000), stream.Receiver4285("1200L")
    out, events = b"", []
    for i in range(0, card.size, 4800):
        for e in rx.push(fe.push(card[i:i + 4800])):
            if isinstance(e, stream.Data):
                out += e.data
            else:
                events.append(e)
    assert out == msg and events[-1].complete
