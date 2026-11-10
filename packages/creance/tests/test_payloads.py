# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Known-answer, determinism, and length tests for the payload generators."""

import pytest

from creance import payloads
from creance.payloads import build, prbs

PRBS_KAT = {
    "prbs7": "020c28f22cea7d0e",
    "prbs9": "07be2e64129da3cf",
    "prbs15": "0002000c002800f0",
    "prbs23": "00003e000ffc03e0",
    "prbs31": "0000000e000000fc",
}


@pytest.mark.parametrize("name,head", sorted(PRBS_KAT.items()))
def test_prbs_known_answer(name, head):
    assert build(name, 8).hex() == head


def test_prbs7_period():
    bits = "".join(f"{b:08b}" for b in prbs(64, 7))
    period = 2**7 - 1
    assert all(bits[i] == bits[i + period] for i in range(len(bits) - period))
    # 127 is prime, so any shorter period would make the sequence constant
    assert "0" in bits and "1" in bits


def test_fixed_generators():
    assert build("zeros", 5) == b"\x00" * 5
    assert build("ones", 5) == b"\xff" * 5
    assert build("counter", 300)[:4] == b"\x00\x01\x02\x03"
    assert build("counter", 300)[255:258] == b"\xff\x00\x01"


@pytest.mark.parametrize("name", sorted(payloads.GENERATORS))
@pytest.mark.parametrize("n", [0, 1, 17, 4096])
def test_length_and_determinism(name, n):
    a, b = build(name, n), build(name, n)
    assert len(a) == n
    assert a == b


@pytest.mark.parametrize("value,want", [
    (4096, 4096), ("4096", 4096), ("1k", 1024), ("10K", 10240),
    ("100k", 102400), ("1M", 1 << 20), ("2kib", 2048), ("1.5k", 1536),
    ("512b", 512), ("10kb", 10240),
])
def test_size_bytes(value, want):
    # one grammar for the CLI flag, a plan file's params and an API caller:
    # a spelling any of them accepts must not die later at the payload builder
    assert payloads.size_bytes(value) == want


@pytest.mark.parametrize("value", ["", "big", "10g", "k10", "-1k"])
def test_size_bytes_rejects(value):
    with pytest.raises(ValueError):
        payloads.size_bytes(value)


def test_bad_inputs():
    with pytest.raises(ValueError):
        build("noise", 8)
    with pytest.raises(ValueError):
        prbs(8, order=11)
    with pytest.raises(ValueError):
        prbs(8, order=9, seed=0)
