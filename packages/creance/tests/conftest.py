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
