# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np

from hfmodem.shrike import onair


FS = onair.FS
# `arm-v23-A-40-ws8eoc`'s own clocks line: block 128, ADC->DAC 1268 samples,
# DAC notice 1652, holdback 660 -- and the 40 ms settle the FT-891 is set for.
BLK, LAT, SETTLE_N = 128, 1268, round(0.04 * FS)


class _Done:
    """`threading.Event` for a burst that is over before it is armed."""

    def clear(self) -> None:
        pass

    def wait(self, _timeout: float) -> bool:
        return True


class _Stream(onair._LiveInput):
    """`_LiveInput`'s transmit arithmetic with no card and no clock.

    Every sleep is recorded instead of taken, so the instants are the
    measurement rather than a race: `slept[0]` is where PTT would be asserted.
    """

    def __init__(self, *, tx_latency_n: int = 0) -> None:
        self.fs = FS
        self._duplex = True
        self._blk = BLK
        self._lat = LAT
        self.tx_latency_n = tx_latency_n
        self.geometry_n = 0
        self.samples = 100 * BLK
        self.pos = self.samples
        self.holdback = 660
        # A fixed epoch, so two streams are comparable to the sample.
        self._dac = (0.0, 0)
        self._last = (0.0, self.samples)
        self._tx = None
        self._tx_at = 0
        self._tx_end_time = 0.0
        self._tx_done = _Done()
        self.keyed_s = 0.0
        self.keyed_at = None
        self.slept: list[float] = []
        self.flushes: list[int] = []

    def _sleep_until(self, t: float) -> None:
        self.slept.append(t)

    def sample_now(self) -> float:
        return float(self.samples)

    def take_until(self, index: int) -> np.ndarray:
        self.pos = max(self.pos, index)
        return np.zeros(0, np.float32)

    def wait_until(self, index: int) -> None:
        pass

    def flush_to(self, index: int) -> None:
        self.flushes.append(index)
        self.pos = index
