# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The passive follower runs one BLAS thread, and every thread it has runs niced."""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from hfmodem.contact import slots as S

CF = 14_105_000


def _environment(pid):
    if sys.platform == 'linux':
        return dict(entry.split('=', 1) for entry in
                    Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in entry)
    words = subprocess.run(['ps', '-wwE', '-o', 'command=', '-p', str(pid)],
                           capture_output=True, text=True, check=True).stdout.split()
    return dict(word.split('=', 1) for word in words if '=' in word)


def _thread_nices(pid):
    if sys.platform != 'linux':
        return [os.getpriority(os.PRIO_PROCESS, pid)]
    # Field 19 of stat, counted after the parenthesised command name.
    return [int(Path(task, 'stat').read_text().rsplit(')', 1)[1].split()[16])
            for task in Path(f'/proc/{pid}/task').iterdir()]


def test_the_follower_starts_single_threaded_and_niced_and_the_parent_keeps_its_own(monkeypatch):
    for name in S.SINGLE_THREADED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '4')
    watch = S.SlotWatch(CF, CF, mycall='W9SSJ')
    try:
        child = _environment(watch._process.pid)
        assert {name: child.get(name) for name in S.SINGLE_THREADED} == S.SINGLE_THREADED
        assert os.environ['OPENBLAS_NUM_THREADS'] == '4'
        assert not {name for name in S.SINGLE_THREADED if name != 'OPENBLAS_NUM_THREADS'} & set(os.environ)

        watch.put(np.zeros(S.BLOCK, np.float32))
        end = time.monotonic() + 60
        while watch.table()[1] < .5 and time.monotonic() < end:
            time.sleep(.05)
        assert watch.table()[1] == .5
        wanted = min(19, os.getpriority(os.PRIO_PROCESS, 0) + 10)
        assert set(_thread_nices(watch._process.pid)) == {wanted}
    finally:
        watch.close()
