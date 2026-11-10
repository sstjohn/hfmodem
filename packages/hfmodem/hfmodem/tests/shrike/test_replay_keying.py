# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A replay must enforce the same audio onset that it reports after PTT settle."""
import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, spec


@pytest.mark.parametrize('remaining_ms,expected_slot,notice_ms,shift_changes',
                         [(6, 3, 0, False), (39, 3, 0, False),
                          (40, 1, 0, False), (40, 1, 32, False),
                          (60, 1, 50, False), (40, 2, 50, True)])
def test_replay_reaims_control_when_full_settle_would_make_audio_late(
        tmp_path, remaining_ms, expected_slot, notice_ms, shift_changes):
    capture = tmp_path / 'silence.wav'
    onair.session.write_wav(str(capture), np.zeros(5 * onair.FS, np.float32))
    live = onair._ReplayInput(str(capture))
    if notice_ms:
        # A PCM clock can model DAC notice without exposing a transmitter.
        # Notice and PTT settle overlap; they must not be added together.
        live.key_notice = round(notice_ms / 1000 * onair.FS)
        live.clamp_late = lambda at: max(0, live.pos + live.key_notice - at)
    grid = onair._MasterGrid(0, round(spec.CYCLE_SHORT_S * onair.FS), 0,
                             packet_n=round(spec.P1_PACKET_S * onair.FS),
                             cs_n=round(spec.P1_CS_S * onair.FS), d_max_n=0)
    grid.sending = False
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.040)
    tx.live = live
    tx.aim(grid, 1)
    original_shift = grid.shift(1)
    live.pos = tx.boundary - round(remaining_ms / 1000 * onair.FS)
    tx.send_p1_cs(pactor1.CS_ACK_A)
    assert tx.slot == expected_slot
    # If DAC notice already rules out the original slot, _flip reaims before
    # rendering and can use the opposite arrangement in the very next slot.
    # A later settle-only overrun must retain the already rendered arrangement.
    assert (grid.shift(tx.slot) != original_shift) == shift_changes
    assert tx.tx_audio_start == grid.boundary(tx.slot)
    assert tx.tx_audio_start - tx.tx_key_up == round(.040 * onair.FS)
    assert not tx.refused
