# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from hfmodem.kestrel.rx import tablegen


def _reference(n, seed):
    out = np.empty(n, dtype="<i4")
    seen = np.zeros(n, dtype=bool)
    s, k = seed, 0
    while k < n:
        s = tablegen.step(s)
        v = tablegen.draw(s, n)
        if not seen[v]:
            seen[v] = True
            out[k] = v
            k += 1
    return out, s


_RECORDS = sorted(set(tablegen.BW500 + tablegen.BW2300 + tablegen.BW2750))


@pytest.mark.parametrize("seed,n1,n2", _RECORDS)
def test_permutation_matches_both_reference_stages(seed, n1, n2):
    for n in (n1, n2):
        expected, state = _reference(n, seed)
        actual, actual_state = tablegen.permutation(n, seed)
        assert actual.tobytes() == expected.tobytes()
        assert actual.dtype == np.dtype("<i4")
        assert actual_state == state
        seed = state


@pytest.mark.parametrize("n,seed", [(0, 42), (1, 0), (2, -1), (17, 1 << 24),
                                   (255, 12345), (8193, (1 << 24) - 1),
                                   (2104, 5613669), (32585, 12345), (26056, 999)])
def test_permutation_other_lengths_and_seeds(n, seed):
    expected, state = _reference(n, seed)
    actual, actual_state = tablegen.permutation(n, seed)
    assert actual.tobytes() == expected.tobytes()
    assert actual_state == state


def test_receiver_imports_defer_and_cache_tables():
    subprocess.run([sys.executable, "-c", textwrap.dedent('''
        from unittest.mock import patch
        from hfmodem.kestrel.rx import tablegen

        with patch.object(tablegen, "permutation", wraps=tablegen.permutation) as perm, \\
             patch.object(tablegen, "whitener", wraps=tablegen.whitener) as pn:
            from hfmodem.kestrel.rx import varahf500, varahf2300, varatone, varaofdm
            from hfmodem.kestrel.arq import phy
            from hfmodem.kestrel.tx import varahf500_tx
            assert perm.call_count == pn.call_count == 0
            assert tablegen._permutation_jumps.cache_info().currsize == 0

            for receiver in (varahf500, varahf2300):
                tables = receiver._load()
                calls = perm.call_count, pn.call_count
                assert calls[0] > 0 and calls[1] > 0
                assert receiver._load() is tables
                assert receiver._PN is receiver._PN
                assert (perm.call_count, pn.call_count) == calls

            assert varahf500._P is varahf500._P
            assert varahf500.frame_cells(0) is varahf500.frame_cells(0)
            assert varahf2300._ROLES is varahf2300._ROLES
            assert varahf500_tx._cell_index(0) is varahf500_tx._cell_index(0)
            for receiver in (varatone, varaofdm):
                bits = receiver._pn()
                calls = pn.call_count
                assert receiver._pn() is bits
                assert pn.call_count == calls
    ''')], check=True)
