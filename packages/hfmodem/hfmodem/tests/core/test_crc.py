# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest

from hfhost.ptc import crc16 as host_crc16
from hfmodem.core.crc import crc16, crc16_rows
from hfmodem.tests.core.crc_vectors import CRC_VECTORS


@pytest.mark.parametrize('data,variant,want', CRC_VECTORS)
def test_known_crc_values(data, variant, want):
    assert crc16(data, variant) == want
    rows = np.frombuffer(data, np.uint8).reshape(1, -1)
    assert crc16_rows(rows, variant).tolist() == [want]
    if variant == 'x25':
        assert host_crc16(data) == want
