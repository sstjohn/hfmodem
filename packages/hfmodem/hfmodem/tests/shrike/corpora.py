# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Checked, tracked excerpts used by the P4 recording pins."""
from pathlib import Path

import numpy as np

from hfmodem.tests import evidence

FIXTURES = Path(__file__).with_name('fixtures')
N9VC_NAMES = ('n9vc-0802-robust-shorts.wav', 'n9vc-0802-chirps.wav')


def n9vc_0802():
    """48 kHz PCM units, source samples 82416000:83260800 (t=1717.0 s)."""
    shorts, chirps = (
        evidence.checked_samples(FIXTURES / name, (FIXTURES / name).with_suffix('.json'))
        for name in N9VC_NAMES)
    rows = [evidence.manifest_rows((FIXTURES / name).with_suffix('.json'))[0]
            for name in N9VC_NAMES]
    assert [row['source_start'] for row in rows] == [82416000, 82723200]
    assert [row['rate'] for row in rows] == [48000, 48000]
    # Both slices contain 1723.4-1723.5 s; retain those samples only once.
    join = rows[1]['source_start'] - rows[0]['source_start']
    assert np.array_equal(shorts[join:], chirps[:4800]), 'N9VC fixture overlap differs'
    return 48000, np.concatenate((shorts[:join], chirps))
