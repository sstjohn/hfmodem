"""An 8 ms delivery batch is three callbacks, not a larger audio block."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_tx_latency_correction import _Stream, _key


def callbacks(*, batch=3):
    return [(i * 128, 128, (i // batch) * batch * 128 / onair.FS
             + (i % batch) * .00001, i * 128 / onair.FS)
            for i in range(24)]


class BatchedStream(_Stream):
    """Production notice/transmit with actual blocking delivery and fake DAC.

    The capture converter leads delivered audio by the recorded 660 samples.
    Three adjoining 128-frame callbacks arrive together; requesting the first
    sample in the next batch blocks until all 384 samples are delivered.
    """
    def __init__(self, phase):
        super().__init__(tx_latency_n=960)
        self.phase = phase
        self.now = 480000
        self.pos = 470000
        self._callback_timing = callbacks()
        self._advance(self.now)

    def _advance(self, at):
        self.now = max(self.now, at)
        self.samples = ((self.now - self.holdback - self.phase) // 384) * 384 + self.phase

    def sample_now(self):
        return self.now

    def wait_until(self, at):
        self._advance(at)

    def read(self, count):
        end = self.pos + count
        if count and end > self.samples:
            delivered = ((end - self.phase + 383) // 384) * 384 + self.phase
            self._advance(delivered + self.holdback)
        self.pos = end
        return np.zeros(count, np.float32)


def test_delivery_quantum_needs_contiguous_callbacks_and_wall_clustering():
    assert onair._callback_delivery_n(SimpleNamespace(_blk=128)) == 128
    for batch in (1, 2, 3, 4):
        stream = SimpleNamespace(_blk=128, _callback_timing=callbacks(batch=batch))
        assert onair._callback_delivery_n(stream) == batch * 128
    # A slow callback without the adjoining sample ranges is not a batch.
    stream._callback_timing = [(0,128,0.,0.), (1280,128,.00001,.1)]
    assert onair._callback_delivery_n(stream) == 128
    # Nor is wall-clock latency alone: these callbacks are still spaced out.
    stream._callback_timing = [(0,128,0.,0.), (128,128,.020,.002667)]
    assert onair._callback_delivery_n(stream) == 128


@pytest.mark.parametrize('phase', [0, 96, 192, 288])
def test_final_drain_preserves_exact_dac_and_ptt_placement_in_every_phase(phase):
    stream = BatchedStream(phase)
    at = 510049
    lead = onair._prekey_lead(stream, 1920)
    # This is RadioTx._tx's final drain. Advancing the capture clock alone is
    # insufficient: take_until must wait for delivery, including holdback.
    stream.take_until(at - lead)
    stream.wait_until(at - lead)
    # A further 1.5 ms for bridging/enqueue does not borrow the late tolerance.
    stream._advance(stream.now + 72)
    assert stream.clamp_late(at) == 0
    start, ptt = _key(stream, at)
    assert start == at - 1268 - 960
    assert stream._dac_time(start) - ptt == pytest.approx(.04)


def test_single_callback_budget_reproduces_phase_dependent_late_audio(monkeypatch):
    late = []
    for phase in (0, 96, 192, 288):
        stream = BatchedStream(phase)
        # The prior code negotiated128, ignoring observed384-frame deliveries.
        monkeypatch.setattr(onair, '_callback_delivery_n', lambda live: live._blk)
        at = 510049
        lead = onair._prekey_lead(stream, 1920)
        stream.take_until(at - lead)
        stream.wait_until(at - lead)
        stream._advance(stream.now + 72)
        late.append(stream.clamp_late(at))
    assert any(late) and not all(late), late


def test_recovered_reply_reserves_ordinary_tick_and_render_before_last_drain(tmp_path):
    from hfmodem.tests.shrike.test_p3_breakin_timing import duplex
    from hfmodem.tests.shrike.test_p3_turn_budget_0914 import (
        _Rx, recovered_host, turn_grid)

    grid = turn_grid()
    tx = duplex(grid, tmp_path)
    tx.defer_p3_cs = True
    tx.prekey_cost_n = 96
    tx.breakin_due = False
    tx.aim(grid, 44)
    stream = tx.live
    stream._callback_timing = callbacks()
    receiver = _Rx()
    receiver.host = recovered_host()
    receiver._p3_row0 = tx.key_instant(grid, 44) - receiver._p3_span - 1300
    stream.now = stream.pos = tx.key_instant(grid, 44) + 480 - stream.key_notice
    slot, _, _ = onair._regrid(stream, grid, tx, receiver.host, receiver, 44,
                             np.zeros(0, np.float32), 0, 1920)
    assert slot > 44
    assert stream.taken[-1][1] == tx.key_instant(grid, slot) - onair._prekey_lead(
        stream, 1920, tx.prekey_cost_n)
