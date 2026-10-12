# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fixtures shared across the creance suites."""

import pytest


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    import hfhost.transcript as tmod
    c = FakeClock()
    monkeypatch.setattr(tmod.time, "monotonic", c)
    return c


@pytest.fixture
def keyed(request):
    """RTTY as the transmitter shapes it (`rtty.modulate`), or hard-keyed: instant
    tone changes and no envelope ramp, as `rtty.modulate` rendered it before
    transmit shaping and the waveform the runner pins were measured on."""
    from hfmodem.lanner import fsk, ita2, rtty

    if request.param == 'shaped':
        return rtty.modulate

    def hard(text, *, mark_hz=rtty.MARK_HZ, space_hz=rtty.SPACE_HZ, fs=fsk.FS):
        lead = [True] * round(2 * rtty.LEAD_TAIL_UNITS)
        halves = list(lead)
        for code in ita2.encode(text):
            halves += [False] * 2 + [bit for k in range(5) for bit in [bool(code >> k & 1)] * 2]
            halves += [True] * 3
        return fsk.modulate(halves + lead, baud=2 * rtty.BAUD, f_hi=mark_hz, f_lo=space_hz, fs=fs)

    return hard
