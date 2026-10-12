# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Fast, exact P4 C2 direct-gather permutations.

Start with the stride-isqrt(N) walk and replay compact index swaps. These
reconstruction swaps are not the original generator's accepted swap trace.
Twelve tables have reproducible CML recipes in c2proof; the 1160 seed/session
remains unknown. See docs/protocols/pactor/pactor4-c2-derivation.md.
"""
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from math import isqrt
from numbers import Integral
from types import MappingProxyType
import zlib

import numpy as np

# N: (number of index swaps, compressed swap SHA-256, canonical u16-LE SHA-256).
_TABLES = MappingProxyType({
    280: (6, "920cce2c60c0b0c64c0be50332493bd3be1d0eabadb877853d3a1a583fc447e8", "8b24b531066d9e020b423c3a05b22304bce162470d40dd4b5600480232a53e3a"),
    348: (12, "d02f438ec10d3fc71e9c942be66ab25330404ee590e4e148cbcd34cce0def611", "c5e38726a9247ae94b619c0c64762d706e62314f8ff9582aa9440e8a915c8b0c"),
    352: (4, "18fe9affa46cf4eb48cf5dfb8aed88a2da9eb444f672ee2911a41332644ebecb", "a96675e46a0e83610c759bf188b1a1829f5ad1bea64c58b79eb7e599624ed0cb"),
    704: (14, "6cef87f4b9e478fa053409d3608ab7097474dfc2442083a682f203aa5f796980", "70aa4076ce903714266abf094d987226d5ffe8496eb22d467e04c567081cf323"),
    1056: (86, "f567ba2add7b3913372d633aec823a1802fae50be58a8bddf44d84ff0820c472", "17c2726165e352e9e7e9547d99ce85a05520dd9e8805b61482883c3857eba1b3"),
    1160: (22, "09a56cb1eda2e33b2135558fcb771a710fb31c197348c636cc1e90034c254245", "ab46bf3a863a8831fa5ad56ab8d57ea3b415af38139e8b3dccd1cb2355a8e621"),
    1408: (62, "489e88385f4966ffc8c5ab00b6ec077678a2d55bd8f96d6729f8815df0baf2e4", "ae62d0c442d7a252418c4a27f1d2c72c4db5901f80e03859155c76b25c078481"),
    1656: (158, "c0e1297426ce6c383fc0539c7db22f331227f7318ef17d43cc0b239cfcc77d2f", "8b6afde2f0b64cef83d314b44a927f4b7a1592237d25c3fb96af6c038585f132"),
    4140: (4130, "ee1c2c632af4d53aa704d45b65de02aafb074bf615ff3a41bf80477ddc65f86b", "c07e40058cc80445f12749c119e2e09fc6a69099a0ed9cda0e4f9d61ee85d504"),
    8280: (8270, "8596c5037e674f032def2331098dca8bbc91a35c2e524cfab9efe5f9583fbdd2", "5ce426b7fcb363ee9fda0e560a7c4b3c8ae958d79374222e7d52d603a7007c91"),
    12420: (12412, "bb9f9bf855dc77be44e18493d44ecac1656e31202faa1414a02eeded973d3fa7", "4c4b450028c6ac66b87647702dfeeafa04f0621583f60fc3161a1ecc31036a17"),
    16560: (16548, "20a2e74ecb0f806bb18fd1d5982e0ffaa4787e50e46194ddf2327d74de266e55", "c2fadc1e3764cc08f66e0286751eb201012277aed8424dedd3a77e620a55cfff"),
    20700: (20688, "4048f8a5151897b9c1fc1a4f3f72394f00ccac6d76091d1dfd96d85436148adb", "2a3bd42d9d423fc7ad84df654c4e80f9e801dc842d2c57364cf4965b35c4565b"),
})
TABLE_SHA256 = MappingProxyType({n: r[2] for n, r in _TABLES.items()})


def permutation(size):
    """Immutable cached direct-gather order; validation precedes cache lookup."""
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral):
        raise ValueError("C2 size must be a supported integer")
    size = int(size)
    if size not in _TABLES:
        raise ValueError("unsupported C2 size")
    return _permutation(size)


def _swap_bytes(size):
    return files(__package__).joinpath("assets", f"c2-{size}.swaps.z").read_bytes()


@lru_cache(maxsize=None)
def _permutation(size):
    count, packed_hash, _ = _TABLES[size]
    packed = _swap_bytes(size)
    if sha256(packed).hexdigest() != packed_hash:
        raise ValueError("invalid C2 swap resource")
    raw = zlib.decompress(packed)
    if len(raw) != count * 4:
        raise ValueError("invalid C2 swap length")
    pairs = np.frombuffer(raw, dtype='<u2').reshape(count, 2)
    if np.any(pairs >= size):
        raise ValueError("C2 swap index out of range")
    stride = isqrt(size)
    order = np.concatenate([np.arange(i, size, stride, dtype=np.intp)
                            for i in range(stride)])
    for left, right in pairs:
        order[left], order[right] = order[right], order[left]
    validate(order, size)
    # Bytes backing prevents callers from restoring writeability.
    return np.frombuffer(order.tobytes(), dtype=np.intp)


def validate(order, size):
    """Check shape, index dtype, bijection and canonical u16-LE identity."""
    a = np.asarray(order)
    if (a.shape != (size,) or a.dtype != np.dtype(np.intp)
            or not np.array_equal(np.sort(a), np.arange(size))
            or sha256(a.astype('<u2').tobytes()).hexdigest() != TABLE_SHA256[size]):
        raise ValueError("invalid C2 table")
