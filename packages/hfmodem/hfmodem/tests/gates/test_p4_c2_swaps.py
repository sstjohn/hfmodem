# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Exact direct-order pins, immutable caches and legacy copy compatibility."""
from hashlib import sha256
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

from hfmodem.shrike import p4c2, p4breakin, p4short, p4shortlevels, p4qam16, p4long

# Independent canonical reference pins, not hashes calculated from swap data.
EXPECTED = {
    280: "8b24b531066d9e020b423c3a05b22304bce162470d40dd4b5600480232a53e3a",
    348: "c5e38726a9247ae94b619c0c64762d706e62314f8ff9582aa9440e8a915c8b0c",
    352: "a96675e46a0e83610c759bf188b1a1829f5ad1bea64c58b79eb7e599624ed0cb",
    704: "70aa4076ce903714266abf094d987226d5ffe8496eb22d467e04c567081cf323",
    1056: "17c2726165e352e9e7e9547d99ce85a05520dd9e8805b61482883c3857eba1b3",
    1160: "ab46bf3a863a8831fa5ad56ab8d57ea3b415af38139e8b3dccd1cb2355a8e621",
    1408: "ae62d0c442d7a252418c4a27f1d2c72c4db5901f80e03859155c76b25c078481",
    1656: "8b6afde2f0b64cef83d314b44a927f4b7a1592237d25c3fb96af6c038585f132",
    4140: "c07e40058cc80445f12749c119e2e09fc6a69099a0ed9cda0e4f9d61ee85d504",
    8280: "5ce426b7fcb363ee9fda0e560a7c4b3c8ae958d79374222e7d52d603a7007c91",
    12420: "4c4b450028c6ac66b87647702dfeeafa04f0621583f60fc3161a1ecc31036a17",
    16560: "c2fadc1e3764cc08f66e0286751eb201012277aed8424dedd3a77e620a55cfff",
    20700: "2a3bd42d9d423fc7ad84df654c4e80f9e801dc842d2c57364cf4965b35c4565b",
}


@pytest.mark.parametrize("size", EXPECTED)
def test_all_exact_bytes_direction_and_immutable_cache(size):
    a = p4c2.permutation(size)
    assert a.dtype == np.dtype(np.intp)
    assert np.array_equal(np.sort(a), np.arange(size))
    assert sha256(a.astype('<u2').tobytes()).hexdigest() == EXPECTED[size]
    assert sha256(np.argsort(a).astype('<u2').tobytes()).hexdigest() != EXPECTED[size]
    assert p4c2.permutation(np.int64(size)) is a
    with pytest.raises(ValueError): a[0] = 0
    with pytest.raises(ValueError): a.setflags(write=True)


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), 280.0, '348', None, 1, 2484])
def test_validation_before_cache_aliases(bad):
    p4c2.permutation(280)
    with pytest.raises(ValueError): p4c2.permutation(bad)


@pytest.mark.parametrize("get,size", [(p4breakin.component_permutation,280),
    (p4short.component_permutation,352), (lambda: p4shortlevels.component_permutation(2),704),
    (lambda: p4shortlevels.component_permutation(3),1056),
    (p4qam16.component_permutation,1408), (p4long.component_permutation,1656)])
def test_legacy_arrays_are_fresh_and_writable(get,size):
    a,b = get(),get()
    assert a is not b and a.flags.writeable and b.flags.writeable
    assert np.array_equal(a,b) and np.array_equal(a,p4c2.permutation(size))
    a[0] ^= 1
    assert np.array_equal(b,p4c2.permutation(size))


@pytest.mark.parametrize("bad", [b'', b'invalid swap resource'])
def test_resource_integrity_failure_cannot_enter_cache(monkeypatch,bad):
    p4c2._permutation.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(p4c2,'_swap_bytes',lambda size:bad)
        with pytest.raises(ValueError,match='swap resource'): p4c2.permutation(280)
    assert sha256(p4c2.permutation(280).astype('<u2').tobytes()).hexdigest()==EXPECTED[280]


def test_optional_all_original_reference_entries():
    root = os.environ.get('P4_CONSTANT_REFERENCE_ROOT')
    if not root: pytest.skip('set P4_CONSTANT_REFERENCE_ROOT for independent original arrays')
    path=Path(root)/'c2-runtime-original.npz'
    with np.load(path,allow_pickle=False) as original:
        for n in EXPECTED:
            a=original[f'c2_{n}']
            b=p4c2.permutation(n)
            assert np.array_equal(a,b)
            assert a.astype('<u2').tobytes()==b.astype('<u2').tobytes()


def test_unresolved_proof_is_not_reported_as_recovered_recipe():
    result = subprocess.run([sys.executable,'-m','hfmodem.shrike.c2proof','--cc','compiler-that-does-not-exist',
                             'verify','1160'],capture_output=True,text=True)
    assert result.returncode == 3
    assert '"cml_proof": false' in result.stdout
    assert 'unresolved_recipe' in result.stdout
