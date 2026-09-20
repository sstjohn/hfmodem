# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A due retry must not discard pending replies or key into an open region."""
from types import SimpleNamespace

import numpy as np

from hfmodem.kestrel.vara import vara_frames as VF, vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake
from hfmodem.tests.kestrel.test_connect_retry import kc, _FakeIO, _RESPONSE
from hfmodem.tests.kestrel.test_rx_segmenter import _transport, _record


def clock(monkeypatch):
    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    monkeypatch.setattr(kc, "time", SimpleNamespace(time=lambda: now[0], sleep=advance))
    # Monitoring isn't under test and must not consume wall time in a fake clock.
    monkeypatch.setattr(kc, "classify", lambda *a, **kw: None)
    monkeypatch.setattr(kc, "emit", lambda *a: None)
    return now, advance


def test_due_retry_drains_all_pending_brackets_before_keying(monkeypatch):
    _, advance = clock(monkeypatch)
    ack = MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)

    class QueuedIO(_FakeIO):
        def tx(self, samples):
            super().tx(samples)
            # Model AudioVaraIO.tx clearing pending RX after a repeat.
            if self.link_setups > 1:
                self.replies.clear()

        def next_rx_burst(self, timeout, hs=None):
            advance(.3)
            return self.replies.pop(0) if self.replies else None

    io = QueuedIO([_RESPONSE, np.zeros(12000), ack])
    hs = VaraStationHandshake(["W9SSJ"], io)
    assert kc.connect("NS0A", "W9SSJ", "2300", io, hs=hs, listen_first=0,
                      timeout=2, ls_interval=.1)
    assert io.link_setups == 1


def test_open_reception_defers_retries_but_never_extends_timeout(monkeypatch):
    now, advance = clock(monkeypatch)

    class BusyIO(_FakeIO):
        receiving = True

        def next_rx_burst(self, timeout, hs=None):
            advance(.1)
            return self.replies.pop(0) if self.replies else None

    io = BusyIO([_RESPONSE])
    hs = VaraStationHandshake(["W9SSJ"], io, max_link_setups=6)
    assert not kc.connect("NS0A", "W9SSJ", "2300", io, hs=hs, listen_first=0,
                          timeout=1, ls_interval=.1)
    assert io.link_setups == 1
    assert hs._linksetup_tx == 1
    assert now[0] <= 1.5


def test_deferred_retry_resumes_after_reception_ends_with_same_budget(monkeypatch):
    now, advance = clock(monkeypatch)

    class ClearingIO(_FakeIO):
        @property
        def receiving(self):
            return now[0] < 1.0

        def tx(self, samples):
            if self.link_setups:
                assert not self.receiving
            super().tx(samples)

        def next_rx_burst(self, timeout, hs=None):
            advance(.1)
            return self.replies.pop(0) if self.replies else None

    io = ClearingIO([_RESPONSE])
    hs = VaraStationHandshake(["W9SSJ"], io, max_link_setups=2)
    assert not kc.connect("NS0A", "W9SSJ", "2300", io, hs=hs, listen_first=0,
                          timeout=2, ls_interval=.1, max_cr=1)
    assert io.link_setups == hs._linksetup_tx == 2


def test_stream_tx_does_not_feed_its_old_snapshot_back_to_the_gate(monkeypatch):
    io = _transport(monkeypatch)
    monkeypatch.setattr(kc, "play_drained", lambda *a: None)
    _record(io, np.ones(48000) * .1)
    scans = []
    monkeypatch.setattr(io._seg, "push", lambda x: scans.append(x) or [])

    class Reply:
        def on_rx_stream(self, samples):
            io.tx(np.zeros(12000))

    assert io.next_rx_burst(0, hs=Reply()) is None
    assert scans == []
    assert io._bursts == []
