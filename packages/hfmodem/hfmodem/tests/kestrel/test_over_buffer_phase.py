# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where a peer's over sits in the receive buffer must not move the alignment.

The stream route empties its buffer at our own keying, so an over that answers a
turn-release starts at whatever phase our unkey left it — anywhere inside a
512-sample column. ``_rec3_alignment`` has to find the same frame start from all
of them.

It did not. It kept the first onset whose reference-column hit count beat the
running best, and the hits saturate at 24 of 24 across a third of a column either
side of the truth, so the earliest onset on that plateau won: 15 of these 16 phases
aligned up to 160 samples early. The reference bins' energy share separates the
plateau — 24.00 at the true onset, under 15 one step away — and ranking onsets by
hits and then share costs no extra decode.

A frame read 160 samples early still decodes off a noise-free synthesis (the margin
runs out at 176), which is why the alignment is asserted here and not only its
consequence. On the bench's phase-3 reply the same error was fatal: `24/24
reference columns but the frame will not decode`, and the ``;OK\\r`` was delivered
on one run in two.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_mfsk as MK

_MYCALL = "W9SSJ"
_REPLY = b";OK\r"
_NCOLS = rx._BASE_NCOLS
_COLUMN = rx.RECORDS[rx.BASE_LEVEL].dw50


@pytest.fixture(scope="module")
def over() -> np.ndarray:
    """The responder's reply to our turn-release, as a stock VARA keys it."""
    return tx.synth_burst(_phy.vara_body(_REPLY, _MYCALL))


@pytest.mark.parametrize("phase", range(0, _COLUMN, 32))
def test_the_reply_over_aligns_and_decodes_at_every_buffer_phase(over, phase):
    hs = VA.VaraStationHandshake([_MYCALL], VA.VaraIO(), bw="2300")
    x = np.concatenate([np.zeros(phase), over, np.zeros(MK.FS // 4)])
    start = phase + len(over) - _NCOLS * _COLUMN

    hits, mag = hs._rec3_alignment(x)
    assert hits == 24, f"{hits}/24 reference columns at buffer phase {phase}"
    assert np.allclose(mag[:_NCOLS], rx._band_mag(x[start:], rx.BASE_LEVEL, _NCOLS)), (
        f"buffer phase {phase} took an alignment other than the frame's own start")

    fr = rx.check_frame(rx.onair_to_frame(rx._onair_llr(mag, rx.BASE_LEVEL),
                                          rx.BASE_LEVEL), rx.BASE_LEVEL)
    assert fr.crc_ok, f"24/24 reference columns at phase {phase} and no decode"
    assert bytes(fr.payload).startswith(_REPLY)
