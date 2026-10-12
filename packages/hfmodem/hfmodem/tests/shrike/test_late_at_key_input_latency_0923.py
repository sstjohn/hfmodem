"""The batch `_late_at_key` carries is one the driver will deliver, not the input latency.

0923-1405, PACTOR-1 IRS: `sample_now()` leads the delivered count by the
holdback less a block (660 - 128 = 532 samples), so carrying the converter's
position forward always found a 384-sample batch due and the render gate
refused ordinary CS acks at +5.1 and +7.1 ms that `_tx` would have keyed at
the bare clamp.
"""
import time
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair

FS, BLK, LEAD, NOTICE = 48000, 128, 532, 2612


def stream(since_ms):
    """A duplex stream whose last three-block batch landed `since_ms` ago."""
    samples = 1_000_000
    wall = time.monotonic() - since_ms / 1e3
    return SimpleNamespace(
        transmit=object(), _blk=BLK, holdback=BLK + LEAD, samples=samples, key_notice=NOTICE,
        _callback_timing=[(samples - BLK * (3 - i), BLK, wall + i * 1e-5, 0.0)
                          for i in range(3)],
        sample_now=lambda: samples + LEAD + (time.monotonic() - wall) * FS,
        clamp_late=lambda at: max(0, samples + NOTICE - at))


@pytest.mark.parametrize('margin_ms', [-2.9, -0.9])
def test_the_input_latency_is_not_a_batch_due(margin_ms):
    """Carried over the converter's lead these read +5.1 and +7.1 ms."""
    live = stream(since_ms=0.5)
    at = live.samples + NOTICE - round(margin_ms / 1e3 * FS)
    owed = round(.0006 * FS)
    assert live.clamp_late(at) == 0
    assert onair._late_at_key(live, at, owed) == 0


def test_a_batch_due_inside_the_owed_cost_is_still_carried():
    live = stream(since_ms=7.8)
    at = live.samples + NOTICE + round(.0029 * FS)
    assert onair._late_at_key(live, at, round(.0006 * FS)) == 384 - round(.0029 * FS)


def test_a_refused_burst_does_not_hand_its_admission_to_the_next(tmp_path):
    """0923-1434: the QRM guard refused the admitted burst, a CS decoded in the
    same hold keyed the next, and that one was charged the 535 ms between --
    then carried +430..530 ms past every key with the bare clamp at 0."""
    from hfmodem.tests.shrike.test_p3_missed_slots_0922 import station, stand_late
    s, tx, g = station(tmp_path)
    tx.reply_clock = g.reply_clock = None
    tx.slots_used.clear()
    live, slot = tx.live, 45
    stand_late(live, g, tx, slot, 0)
    tx._admitted = (slot, int(live.sample_now()))
    tx.listening = True
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert tx.refused
    tx.listening = False
    live.spend(round(.535 * FS))
    tx.aim(g, slot + 1)
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert not tx.refused and tx.slots_used == [slot + 1]
    assert tx.prekey_cost_n == 0
    arm = stream(since_ms=0.5)
    at = arm.samples + NOTICE + round(.0029 * FS)
    assert onair._late_at_key(arm, at, tx.cycle_cost_n()) == 0
