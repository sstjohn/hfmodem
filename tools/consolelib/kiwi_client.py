# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Sideband audio adapter for the shipped KiwiSDR stream."""
from types import SimpleNamespace
import numpy as np
from hfhost.kiwi import KiwiStream


class Busy(RuntimeError):
    pass


BUSY_ERRORS = (Busy,)


class KiwiSession:
    def __init__(self, host, port, dial_hz, *, low_cut=300, high_cut=2900, agc_gain=50, passband=None, mode='usb'):
        self.host, self.port, self.dial_hz = host, port, dial_hz
        if passband is not None:
            low_cut, high_cut = passband
        if mode not in ('usb', 'lsb'):
            raise ValueError('Kiwi sideband must be usb or lsb')
        if mode == 'lsb':
            low_cut, high_cut = -high_cut, -low_cut
        self.options = dict(mode=mode, lo_hz=low_cut, hi_hz=high_cut, gain=agc_gain)
        self.stream = None

    def start(self):
        try:
            self.stream = KiwiStream.open(self.host, self.port, self.dial_hz / 1000, **self.options)
        except RuntimeError as error:
            if 'channels are taken' in str(error):
                raise Busy(str(error)) from error
            raise
        self.sample_rate = self.stream.sample_rate
        return self

    def blocks(self):
        for data in self.stream.blocks():
            yield SimpleNamespace(samples=np.frombuffer(data, dtype='>i2').astype(np.float32) / 32768)

    def close(self):
        if self.stream is not None:
            self.stream.close()
