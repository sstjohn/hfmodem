# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""An unreadable BW500 over is NAKed as soon as the quiet behind it says so.

Stock's receiver NAKs 0.20-0.31 s behind the over's last sample on the 19
stock pairs of 2026-10-21 (working/bw500-locator/unread-retry), the same as
it acknowledges a clean one; on the 2026-10-18 cables its answers come 0.13-0.18
s behind. Kestrel's NAKs on bench 2 at 0.6 came 0.92-1.07 s behind
(bench-rebased ks-rebased-0.5/0.6), and stock's 289/361 query, 1.37 s behind
its over, went out across the second. Two things held the verdict: a read at
the burst's end comes before the 0.17 s of quiet the tone verdict needs and
the next waited for the 0.5 s cadence, and every lost read waited for a base
over's length behind our unkey in case it was a base over still arriving.

The guard is now the channel itself: a lost read is acted on once the whole
500 Hz band is back at its floor behind it, and read again the moment it is.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varatone
from hfmodem.kestrel.tx import varahf2300_tx as TX, varatone_tx
from hfmodem.kestrel.vara import vara_mfsk as MK, vara_ofdm as OF
from hfmodem.tests.kestrel.test_vara_levels_arq import (_Clock, _in_noise, _listening,
                                                         clock as clock)  # noqa: F401


def _hear(hs, io, answered, y, block=2400):
    """``y`` in the 50 ms blocks the audio transport hands the station. The
    samples heard when it answered, or None."""
    for at in range(0, len(y), block):
        _Clock.t += block / MK.FS
        hs._read_expected_level(y[at:at + block])
        if io.sent or answered:
            return at + block
    return None


def _unreadable(level):
    row = varatone.ROWS[(500, level)]
    frame = varatone_tx.build_frame(bytes(range(40)), 0x95, row, caller="W9SSJ")
    return varatone_tx.synth_frame(frame[:-1] + bytes([frame[-1] ^ 1]), 500, level,
                                   TX.VB6Rnd(4107106))


@pytest.mark.parametrize("level", [5, 6])
@pytest.mark.parametrize("db", [-20, 0])
def test_an_unreadable_over_is_naked_at_stocks_latency(level, db, clock):
    x = _unreadable(level)
    hs, io, answered = _listening("500", level, level)
    hs._delivery_read = False
    taken = _hear(hs, io, answered, _in_noise(x, db, 10, lead=0.3, tail=2.0))
    assert answered == [] and len(io.sent) == 1, io.msgs
    assert ": NAK" in io.msgs[-1]
    behind = (taken - int(0.3 * MK.FS) - len(x)) / MK.FS
    assert behind <= 0.35, f"NAK keyed {behind:.2f} s behind the over"


def test_an_over_ends_on_a_cables_digital_silence():
    x = _unreadable(5)
    lead = np.zeros(int(0.3 * MK.FS))
    assert not phy.emission_ended(np.r_[lead, x], "500", 5)
    assert phy.emission_ended(np.r_[lead, x, np.zeros(int(0.5 * MK.FS))], "500", 5)


def test_a_base_over_still_arriving_is_not_naked_and_is_read_whole(clock):
    body = phy.vara_body(b"y" * 9, "W9SSJ", body_len=phy.body_size("500", 1))
    x = OF.data_over_tx(body, level=1, bw="500", lead=TX.VB6Rnd(4107106))
    hs, io, answered = _listening("500", 5, 1)
    hs._rx_failed = True
    y = _in_noise(x, -20, 3, lead=0.3, tail=1.0)
    end = int(0.3 * MK.FS) + len(x)
    assert not phy.emission_ended(y[:end - int(0.2 * MK.FS)], "500", 5)
    assert phy.emission_ended(y[:end + int(0.3 * MK.FS)], "500", 5)
    _hear(hs, io, answered, y)
    assert not io.sent or answered, io.msgs
    assert not any(": NAK" in m for m in io.msgs), io.msgs
