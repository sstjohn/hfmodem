# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Native station I/O can omit duplicate references without changing the PHY."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_station_io import port


def producer(tmp_path, *, record):
    live, station = port()
    rig = SimpleNamespace(ptt=lambda on: pytest.fail('independent key'), key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, drive=.6, settle=.04,
                      record_audio=record)
    tx.live = live
    tx.reply_clock = onair.ReplyClock()
    return tx, live, station


def test_disabling_duplicate_references_keeps_native_p3_audio_and_keying_exact(tmp_path):
    results = []
    for record in (True, False):
        path = tmp_path / str(record)
        path.mkdir()
        tx, live, station = producer(path, record=record)
        try:
            capacity = tx.send_entry_packet(1, b'', 0x19)
            assert capacity > 0 and len(tx.keyed) == station.arbiter.sent == 1
            assert station.rig.edges == [True, False]
            assert live.pos == tx.tx_end
            assert bool(list(path.glob('tx_*.wav'))) == record
            results.append((station.audio.sent[0], tx.tx_end, tx.tx_pulse_offsets))
        finally:
            live.close()
    assert results[0][0][0] == results[1][0][0]
    np.testing.assert_array_equal(results[0][0][1], results[1][0][1])
    assert results[0][1:] == results[1][1:]


@pytest.mark.parametrize('record', [True, False])
def test_input_loss_and_burst_analysis_do_not_depend_on_reference_recording(tmp_path, monkeypatch, record):
    tx, live, _ = producer(tmp_path, record=record)
    losses, captures = [], []
    raster = SimpleNamespace(note_capture_loss=lambda *a, **kw: losses.append((a, kw)))
    evidence = onair._CycleEvidence(live, tx, raster, tmp_path)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: captures.append((a, kw)))
    live.lost = 1
    signal = pactor1.control_signal(pactor1.CS_SPEED)
    try:
        bursts = evidence.record('native-input', signal, 0)
        assert isinstance(bursts, list)
        assert losses == [((1,), {})]
        assert evidence.lost == 1
        assert len(captures) == int(record)
    finally:
        live.close()


@pytest.mark.parametrize('transmit,rig', [(False, object()), (True, None)])
def test_unrecorded_dry_output_is_not_a_valid_producer_profile(tmp_path, transmit, rig):
    with pytest.raises(ValueError, match='live transmit producer'):
        onair.RadioTx(rig, transmit=transmit, outdir=tmp_path, record_audio=False)
