# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A wide over the live stream first hears after the burst has started.

The bracket route was taught to reach back over the record's own training columns
[``varahf2300.decode_over``, ``test_bw2300_cut_head``]. The live route does not go
through it: ``_stream_over`` scores ``_rec3_alignment``, which ran its own onset
search over the raw buffer and so reached no further back than the first sample it
was given.

That is the buffer the station always has after it transmits. PTT is held
0.14-0.15 s past the last sample and the rig's audio returns 0.03 s after PTT-off,
and ``_key`` empties the over-search buffer at every keying, so a peer that keys at
or before our PTT-off is first heard 16-18 columns into its burst with nothing
ahead of it. Its frame's column 0 is then outside the buffer and the 24 reference
columns fall to chance — the 2026-09-11 KC9GHZ greetings, whole and CRC-clean off
the tape, score 5 and 6 of 24 from 128 ms and 85 ms in
[``working/vara-evening-en63bc-0910/analysis/wide-lock``].

The window here opens 181 ms into the burst, past both of those and past the
longest deaf window measured on air. Nothing opens a device or keys a radio.

The BW2750 staged capture holds one wideband burst and it is the link-setup, which
a connected station refuses by design  [see ``_peer_data_over``], so the captured
case is BW2300's and the rendered pair covers both combs.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.tx import varahf2300_tx as tx
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.corpora import (BW2300_CAPTURE,
                                           requires_bw2300_capture)
from hfmodem.tests.kestrel.test_data_over_gate import _connected

_MYCALL = "W9SSJ"
#: Columns of burst the window opens past — 181 ms at the base symbol.
_CUT = 17
_SNR_DB = 20.0
#: The transport's own poll, which is what the live route is fed on.
_POLL = MK.FS // 50


def _heard(bw: str, burst: np.ndarray, cut: int) -> tuple[list[bytes], int]:
    """Stream ``burst`` from ``cut`` columns in to a connected station, poll by
    poll, and return what reached the host and how often it keyed.

    A trail, because the over search declines an alignment that ends at the
    buffer's own last column  [see ``_stream_over``].
    """
    hs, io = _connected(bw)
    x = np.concatenate([burst[cut * rx.RECORDS[rx.BASE_LEVELS[bw]].dw50:],
                        np.zeros(int(0.6 * MK.FS))])
    for i in range(0, len(x), _POLL):
        hs.on_rx_stream(x[i:i + _POLL])
    return io.host, io.keys


@pytest.mark.parametrize("bw", ["2300", "2750"])
def test_a_rendered_over_heard_mid_burst_is_delivered_and_answered(bw):
    level = rx.BASE_LEVELS[bw]
    payload = bytes(range(1, rx.payload_bytes(level) - 1))
    burst = tx.synth_burst(_phy.vara_body(payload, _MYCALL), level)
    rng = np.random.default_rng(7)
    burst = burst + rng.normal(0.0, np.sqrt(np.mean(burst ** 2)
                                            / 10 ** (_SNR_DB / 10)), len(burst))
    host, keys = _heard(bw, burst, _CUT)
    assert host == [payload], f"BW{bw} over lost from {_CUT} columns in"
    assert keys == 1


@requires_bw2300_capture
def test_a_captured_over_heard_mid_burst_is_delivered_and_answered():
    """A real BW2300 DATA over, against what the same station reads when it hears
    the whole burst."""
    a = _wav(str(BW2300_CAPTURE))
    for s, e in rx.detect_overs(a):
        whole, _ = _heard("2300", a[s:e], 0)
        if not whole:
            continue                      # the session's link-setup, refused
        host, keys = _heard("2300", a[s:e], _CUT)
        assert host == whole, f"over at {s / rx.FS:.1f} s lost from {_CUT} in"
        assert keys == 1
        return
    pytest.fail("no BW2300 DATA over in the capture")


def _wav(path: str) -> np.ndarray:
    from scipy.io import wavfile
    _, a = wavfile.read(path)
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a[:, 0]
    return a / np.abs(a).max()
