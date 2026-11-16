# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair


@pytest.mark.parametrize('boundary', ['read', 'feed', 'bridge', 'ahead', 'prepare', 'tail'])
def test_window_keeps_only_contiguous_samples_after_capture_advances(monkeypatch, boundary):
    class Capture:
        pos = 0
        reads = 0

        def read(self, count):
            self.reads += 1
            if boundary == 'read' and self.reads == 2:
                self.pos += 8
            start = self.pos
            self.pos += count
            return np.arange(start, self.pos, dtype=np.float32)

    live = Capture()
    advanced = False

    def advance(where):
        nonlocal advanced
        if boundary == where and not advanced:
            live.pos += 8
            advanced = True

    def feed(audio):
        advance('tail' if boundary == 'tail' and live.reads == 2 else 'feed')

    rx = SimpleNamespace(
        feed=feed, bridge=lambda audio: advance('bridge'),
        sl3_prekey_expected=lambda: boundary == 'prepare',
        prepare_p3_sl3=lambda audio, origin: advance('prepare'))
    host = SimpleNamespace(arq=SimpleNamespace(
        state=arq.State.CONNECTING if boundary == 'tail' else arq.State.CONNECTED))
    monkeypatch.setattr(onair.rxfront, 'p1_reply_starting', lambda audio: boundary == 'tail')
    audio = onair._listen_until_answer(
        live, 32, host, rx, 32 if boundary in ('feed', 'tail') else 0,
        slice_s=8 / onair.FS,
        ahead=(lambda audio, origin: advance('ahead')) if boundary == 'ahead' else None)
    expected = np.zeros(0, np.float32) if boundary == 'tail' else np.arange(16, 32)
    np.testing.assert_array_equal(audio, expected)
    assert live.pos == (40 if boundary == 'tail' else 32)
