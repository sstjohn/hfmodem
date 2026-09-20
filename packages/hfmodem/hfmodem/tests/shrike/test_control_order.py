# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Actual control rendering against the progressing reference's observed order."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, placement

RECORDS_PATH = Path(__file__).with_name('fixtures') / 'p3-accepted-control-order.json'
RECORDS = (json.loads(RECORDS_PATH.read_text())['pairs']
           if RECORDS_PATH.exists() else [])
pytestmark = pytest.mark.skipif(
    not RECORDS, reason=f"the recorded control order {RECORDS_PATH.name} "
                        "is not installed")


@pytest.mark.parametrize('elapsed', [0, 1, 2])
@pytest.mark.parametrize('row', RECORDS,
                         ids=lambda r: f"{r['packet_phase_s']:.3f}-CS{r['control_cs']}")
def test_recorded_control_order_at_renderer(tmp_path, monkeypatch, row, elapsed):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path)
    # The packet sender is the opposite endpoint. Connection origin persists
    # when ISS/IRS changes; answering=True means we accepted the original call.
    tx.host = SimpleNamespace(arq=SimpleNamespace(answering=row['direction']=='caller'))
    period = round((3.75 if row['long'] else 1.25)*onair.FS)
    boundary = round(row['leading_gap_ms']*onair.FS/1000)+elapsed*period
    tx.slot = 0
    # These rows are the staggered SCS control's own carrier order, which is the
    # experiment now that the defaults are historical/audio-start.
    tx.p3_control_waveform, tx.p3_control_placement = "current", "pulse-center"
    tx.raster = SimpleNamespace(
        _p3_peer=(0, round(.81*onair.FS), period, 0),
        _p3_peer_swap=row['packet_swapped'],
        boundary=lambda slot: boundary,
        p3_control_refusal=lambda slot: None,
        # The recorded boundary is the input here; re-placing it would be
        # measuring the comb rather than the carrier order on it.
        p3_reply_shift=lambda slot: None,
        note_p3_control=lambda phase: None)
    monkeypatch.setattr(tx, '_flip', lambda: False)
    sent = []

    def capture(samples, label, **kwargs):
        sent.append((samples, kwargs))
        tx.n += 1

    monkeypatch.setattr(tx, '_tx', capture)
    index = row['control_cs']-1
    tx._send_p3_control(index)
    assert len(sent) == 1
    samples, timing = sent[0]
    # Only elapsed=0 is directly recorded; later cases pin cycle projection.
    expected = row['control_ch12_leads'] ^ bool(elapsed & 1)
    np.testing.assert_array_equal(samples, placement.control_signal(index, swapped=expected))
    first, second = timing['pulse_offsets']
    assert (first > second) == expected
    assert abs(first-second) == onair.FS//200
    assert min(first, second) == timing['lead_n']
    assert tx.raster.boundary(0) == boundary
