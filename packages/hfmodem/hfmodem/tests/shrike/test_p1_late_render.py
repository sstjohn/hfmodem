# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Choose a late ordinary burst's slot before baking its shift into PCM."""
import wave

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1
from hfmodem.tests.shrike.test_ackplace import _irs_grid
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex


class Clock:
    fs = onair.FS
    _lat = 384
    _blk = 128
    tx_latency_n = 0
    holdback = 0
    key_notice = onair._LiveInput.key_notice
    clamp_late = onair._LiveInput.clamp_late

    def __init__(self, samples):
        self.samples = self.pos = samples

    def take_until(self, at):
        self.samples = self.pos = max(self.pos, at)
        return np.zeros(0, np.float32)

    def wait_until(self, at):
        self.samples = self.pos = max(self.pos, at)

    def sample_now(self):
        return self.samples


def setup(tmp_path):
    tx = onair.RadioTx(transmit=False, outdir=tmp_path, settle=.040)
    grid = _irs_grid()
    tx.aim(grid, 28)
    return tx, grid


@pytest.mark.parametrize("before_render", [True, False])
def test_late_ack_rephases_before_render_instead_of_skipping_two_slots(tmp_path, capsys, before_render):
    tx, grid = setup(tmp_path)
    tx.live = Clock(tx.boundary - 768 + 173)  # actual clamp +3.604 ms
    assert tx.live.clamp_late(tx.boundary) == 173
    if not before_render:
        advance = tx._advance_aim
        tx._advance_aim = lambda **kwargs: advance()  # old pre-render behavior
    tx.send_p1_cs(pactor1.CS_ACK_B)
    expected = 29 if before_render else 30
    assert tx.slot == expected and tx.slots_used == [expected]
    log = capsys.readouterr().out
    assert (onair.LATE_KEY in log) is (not before_render)
    assert ("before choosing its shift" in log) is before_render
    with wave.open(str(tmp_path / "tx_01.wav")) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), "<i2").astype(np.float32) / 32768
        audio = audio.reshape(-1, wav.getnchannels()).mean(axis=1)
    # Decode at a fixed origin using the real codeword reader. Its returned
    # sense must follow slot29, not the already-missed slot28.
    got = onair.p1rx.decode_control_signal(audio, 0, .120)
    assert got is not None and got[0] == pactor1.CS_ACK_B and got[1] == 0
    assert got.sense == grid.shift(expected)


@pytest.mark.parametrize("lead_ms", [28, 30, 33])
def test_placeable_audio_keeps_slot_even_if_full_settle_is_no_longer_available(tmp_path, lead_ms):
    tx, grid = setup(tmp_path)
    tx.live = Clock(tx.boundary - round(lead_ms / 1000 * onair.FS))
    assert tx.live.clamp_late(tx.boundary) == 0
    assert tx.live.clamp_late(tx.boundary - round(.040 * onair.FS)) > 0
    assert tx._flip() == grid.shift(28)
    assert tx.slot == 28


def test_placed_breakin_keeps_its_peer_relative_instant_for_refusal(tmp_path):
    tx, grid = setup(tmp_path)
    tx.live = Clock(tx.boundary - 768 + 173)
    tx.breakin_due = True
    assert tx._flip() == grid.shift(28)
    assert tx.slot == 28  # _place_breakin/_tx owns refusal; no blind later BK


# --- and the same question, asked twice, answered once -----------------------
#
# `_regrid` runs a sub-symbol overrun through `_clamp_forgives` and keeps the
# slot; `_flip` then re-tested the same instant a tick later with a bare
# `clamp_late` and stepped past it. `pactor-p3-100-ws8eoc-40-sense-
# 20260918T152433Z` printed both verdicts twelve times, adjacent, on overruns of
# +0.4 to +2.4 ms -- twelve slots the grid had already admitted, and with them
# five of the ten changeovers that went unacknowledged.

def _irs_control_cycle(tmp_path, slot=28):
    """A PACTOR-3 IRS control cycle on a DUPLEX bench, aimed at `slot`.

    Duplex because `_clamp_forgives` forgives nothing on a stream with no
    transmitter to hand a later instant to.
    """
    grid = _irs_grid()
    tx = duplex(grid, tmp_path, seconds=60.0)
    tx.aim(grid, slot)
    return grid, tx


def _stand_late(tx, at_least):
    """Put the clock at least `at_least` samples past the burst's key instant.

    The delivered count is quantised to the callback block, so the overrun a
    scene can ask for is too: this takes the first one the block allows.
    """
    blk = tx.live._blk
    delivered = -(-(tx.boundary + at_least - tx.live.key_notice) // blk) * blk
    tx.live.now = tx.live.pos = delivered + blk - 1
    return tx.live.clamp_late(tx.boundary)


def _regrid(grid, tx):
    return onair._regrid(tx.live, grid, tx, tx.host, None, tx.slot,
                         np.zeros(0, np.float32), 0,
                         round(tx.settle * onair.FS))[0]


def test_a_slot_the_grid_keeps_is_not_stepped_past_before_the_render(tmp_path, capsys):
    grid, tx = _irs_control_cycle(tmp_path)
    # ONE INSTANT. An ordinary IRS control has no pulse lead, so the key instant
    # `_regrid` measures against and the boundary `_advance_aim` measures
    # against are the same sample -- which is what makes the two verdicts
    # comparable at all.
    assert tx.key_instant(grid, tx.slot) == tx.boundary
    late = _stand_late(tx, 100)
    assert 0 < late <= round(onair.KEY_CLAMP_TOL_S * onair.FS) - tx.cycle_cost_n()
    kept = tx.slot
    assert _regrid(grid, tx) == kept
    assert onair.SLOT_KEPT in capsys.readouterr().out
    assert tx._flip() == grid.shift(kept)
    said = capsys.readouterr().out
    assert tx.slot == kept and tx.boundary == grid.boundary(kept)
    assert onair.SLOT_UNRENDERED not in said
    assert "before choosing its shift" not in said


def test_an_overrun_past_the_forgiveness_is_given_up_and_says_so(tmp_path, capsys):
    """...and the render seam names its refusal rather than vanishing a slot."""
    grid, tx = _irs_control_cycle(tmp_path)
    late = _stand_late(tx, 100)
    # The room is the tolerance LESS what the cycle still owes the tick and the
    # render, so raising that cost past the overrun is what turns the verdict
    # over. `_regrid` asks this same function of this same instant.
    tx.prekey_cost_n = round(onair.KEY_CLAMP_TOL_S * onair.FS) - late + 1
    assert onair._clamp_forgives(tx.live, tx, late) == 0
    gave = tx.slot
    tx._flip()
    said = capsys.readouterr().out
    assert tx.slot > gave
    assert f"SLOT {gave} {onair.SLOT_UNRENDERED}" in said
    assert "before choosing its shift" in said
    # Its own words: a scene counting what the GRID handed back, or what the
    # emission path moved, must not read this as either.
    assert onair.SLOT_GONE not in said and onair.LATE_KEY not in said
