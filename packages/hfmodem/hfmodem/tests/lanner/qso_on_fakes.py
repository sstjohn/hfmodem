# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""`python -m hfmodem.lanner.qso`, whole, on a card and a keying line that exist only
here: the card calls back in real time on a thread, and the line writes every key
and every drawn window size to $FAKE_LOG. The pty test runs this behind the
launcher's own `rtty)` arm."""
from __future__ import annotations

import os
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np

LOG = open(os.environ['FAKE_LOG'], 'ab', buffering=0)


def record(line):
    # Stop signals can interrupt another event write. Keep each complete event
    # in one append syscall, without a reentrant Python buffered-writer lock.
    data = line.encode('utf-8')
    if os.write(LOG.fileno(), data) != len(data):
        raise OSError('short fake event write')


class Status:
    input_overflow = output_underflow = priming_output = False

    def __bool__(self):
        return False


class Stream:
    def __init__(self, callback=None, blocksize=128, **_):
        self.callback, self.blocksize, self.running = callback, blocksize, False
        self.latency = (0.01, 0.01)

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        n, t0 = self.blocksize, time.monotonic()
        k = 0
        while self.running:
            t = t0 + k * n / 48000
            self.callback(np.zeros((n, 1), np.float32), np.zeros((n, 1), np.float32), n,
                          SimpleNamespace(inputBufferAdcTime=t, outputBufferDacTime=t + 0.024),
                          Status())
            k += 1
            time.sleep(max(0.0, t0 + k * n / 48000 - time.monotonic()))

    def stop(self):
        self.running = False

    close = stop


class Keyer:
    def __init__(self, device, log, alarm=None):
        self.up = False
        self.must_retire = False

    def arm(self):
        pass

    def key(self, on, note=''):
        self.up = on
        record(f'key {"up" if on else "down"}\n')
        return True

    def hand_back(self):
        record('hand back\n')


def main(argv):
    sounddevice = ModuleType('sounddevice')
    sounddevice.Stream = Stream
    sys.modules['sounddevice'] = sounddevice
    from hfmodem.core import devices, ptt
    from hfmodem.lanner import qso, terminal

    devices.find_device = lambda name, kind, required=False: name
    ptt.LineKeyer = Keyer
    qso.rigctld_dial = lambda admit=None, **_: None
    draw, seen = terminal.draw, []

    def drawn(screen, *args):
        if seen[-1:] != [screen.getmaxyx()]:
            seen.append(screen.getmaxyx())
            record('size {}x{}\n'.format(*seen[-1]))
        return draw(screen, *args)

    terminal.draw = drawn
    return qso.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
