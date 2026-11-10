# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.tests import fakeclaim


@pytest.fixture(autouse=True)
def admitted(monkeypatch):
    """Shrike's sessions take the launch claim on fd 9 before opening a device;
    in-process tests have none. Admission order is pinned in
    test_radio_startup_flow, the claim itself in tests/core and tests/coexist."""
    return fakeclaim.install(monkeypatch)
