# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import math

import pytest

from hfhost.cbor import CborError, decode, encode


@pytest.mark.parametrize("value,hexed", [
    (0, "00"), (1, "01"), (10, "0a"), (23, "17"), (24, "1818"), (25, "1819"),
    (100, "1864"), (1000, "1903e8"), (1000000, "1a000f4240"),
    (1000000000000, "1b000000e8d4a51000"),
    (-1, "20"), (-10, "29"), (-100, "3863"), (-1000, "3903e7"),
    (b"", "40"), (b"\x01\x02\x03\x04", "4401020304"),
    ("", "60"), ("a", "6161"), ("IETF", "6449455446"), ("ü", "62c3bc"),
    ([], "80"), ([1, 2, 3], "83010203"), ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({}, "a0"), ({1: 2, 3: 4}, "a201020304"),
    (False, "f4"), (True, "f5"), (None, "f6"), (1.5, "fb3ff8000000000000"),
])
def test_rfc8949_vectors(value, hexed):
    assert encode(value).hex() == hexed
    assert decode(bytes.fromhex(hexed)) == value


@pytest.mark.parametrize("hexed,value", [
    ("f93c00", 1.0), ("f90001", 5.960464477539063e-08), ("f9c400", -4.0),
    ("fa47c35000", 100000.0), ("fbc010666666666666", -4.1),
    ("5f42010243030405ff", b"\x01\x02\x03\x04\x05"),
    ("7f657374726561646d696e67ff", "streaming"),
    ("9fff", []), ("bf61610161629f0203ffff", {"a": 1, "b": [2, 3]}),
    ("c074323031332d30332d32315432303a30343a30305a", "2013-03-21T20:04:00Z"),
])
def test_decodes_forms_we_never_emit(hexed, value):
    """Half/single floats, indefinite lengths and tags are legal CBOR a peer may
    send; we must read them even though deterministic encoding never makes them."""
    assert decode(bytes.fromhex(hexed)) == value


def test_nan_and_infinities():
    assert math.isnan(decode(bytes.fromhex("f97e00")))
    assert decode(bytes.fromhex("f97c00")) == math.inf
    assert decode(bytes.fromhex("f9fc00")) == -math.inf


@pytest.mark.parametrize("value", [
    0, -1, 2 ** 64 - 1, -(2 ** 64), b"\x00" * 300, "x" * 300,
    [1, [2, [3, [4]]]], {0: {1: {2: [3, 4]}}}, 3.141592653589793,
])
def test_round_trip(value):
    assert decode(encode(value)) == value


def test_encoding_is_deterministic_regardless_of_insertion_order():
    """Two implementations must agree byte-for-byte, or cross-site transcript
    comparison stops meaning anything."""
    assert encode({3: "c", 1: "a", 2: "b"}) == encode({1: "a", 2: "b", 3: "c"})
    assert encode({"b": 1, "a": 2}) == encode({"a": 2, "b": 1})


def test_arguments_use_the_shortest_form():
    assert encode(23) == b"\x17"          # not 1817
    assert encode(24) == b"\x18\x18"      # not 190018
    assert encode(256) == b"\x19\x01\x00"


@pytest.mark.parametrize("hexed", [
    "",            # empty
    "a2",          # map claiming two pairs, none present
    "4402",        # byte string claiming 4 bytes, one present
    "82 01".replace(" ", ""),  # array claiming two items, one present
    "fc",          # reserved simple value
    "ff",          # break outside an indefinite item
    "1c",          # reserved additional information
])
def test_malformed_input_raises(hexed):
    with pytest.raises(CborError):
        decode(bytes.fromhex(hexed))


def test_trailing_bytes_are_an_error():
    """A frame carries exactly one message; tolerating a tail would hide a
    framing bug rather than surface it."""
    with pytest.raises(CborError):
        decode(encode(1) + encode(2))


def test_unencodable_type_raises():
    with pytest.raises(CborError):
        encode({1, 2, 3})
