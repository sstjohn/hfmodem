# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP Packed6 / StationId / Locator codec tests.

Fixed wire vectors are derived independently of the module under test (an explicit
bit-string assembly, cross-checked by hand) and match the two reference sources:
ardopcf's `Packed6.c`/`StationId.c`/`Locator.c` and M0LTE's C# port. The
`caller`/`grid` values are exactly the callsigns the manifest records for the
IDFrame and ConReq fixtures, so the demodulator will later close the loop by
recovering these same bytes off the air.
"""

from __future__ import annotations

import pytest

from hfmodem.besra.frame.callsign import (
    pack_callsign,
    pack_grid,
    packed6_decode,
    packed6_encode,
    unpack_callsign,
    unpack_grid,
)

# CALL[-SSID] -> 6-byte StationId wire form
CALLSIGN_VECTORS = {
    "M7TFF": "B57D26980010",       # no SSID -> packed byte '0' (StationId.c:233)
    "M7TFF-3": "B57D26980013",     # IDFrame fixture caller
    "GB7RDG-15": "9E25F292701F",   # ConReq fixture target; base-36 "15"->41->'?'->15
    "N0CALL": "B908E1B2C010",
    "GB7RDG": "9E25F2927010",      # Ping fixture target
    "M7TFF-A": "B57D26980021",     # alpha SSID
    "M7TFF-10": "B57D2698001A",    # numeric SSID 10 -> ':'
}

# grid -> 6-byte Locator wire form
GRID_VECTORS = {
    "IO81VK": "A6F611DAB000",  # IDFrame fixture grid
    "IO81": "A6F611000000",
    "JO65": "AAF595000000",
}


@pytest.mark.parametrize("call, hexbytes", CALLSIGN_VECTORS.items())
def test_callsign_wire_vector(call, hexbytes):
    assert pack_callsign(call).hex().upper() == hexbytes


@pytest.mark.parametrize("call", CALLSIGN_VECTORS)
def test_callsign_roundtrip(call):
    assert unpack_callsign(pack_callsign(call)) == call


@pytest.mark.parametrize("grid, hexbytes", GRID_VECTORS.items())
def test_grid_wire_vector(grid, hexbytes):
    assert pack_grid(grid).hex().upper() == hexbytes


@pytest.mark.parametrize("grid", GRID_VECTORS)
def test_grid_roundtrip(grid):
    assert unpack_grid(pack_grid(grid)) == grid


def test_manifest_idframe_pair():
    # txframe_IDFrame.wav caller=M7TFF-3,grid=IO81VK
    assert pack_callsign("M7TFF-3").hex().upper() == "B57D26980013"
    assert pack_grid("IO81VK").hex().upper() == "A6F611DAB000"


def test_manifest_conreq_pair():
    # txframe_ConReq*.wav caller=M7TFF,target=GB7RDG-15
    assert pack_callsign("M7TFF").hex().upper() == "B57D26980010"
    assert pack_callsign("GB7RDG-15").hex().upper() == "9E25F292701F"


def test_packed6_primitive_roundtrip():
    for s in ["ABCD1234", "M7TFF  0", "        ", "____////"]:
        assert packed6_decode(packed6_encode(s)) == s


def test_packed6_folds_lowercase():
    assert packed6_encode("io81vk  ") == packed6_encode("IO81VK  ")


def test_packed6_out_of_range_becomes_space():
    # a tab (0x09) is outside space..underscore -> packs as space
    assert packed6_encode("A\tBCDEFG") == packed6_encode("A BCDEFG")


def test_packed6_decode_length_checked():
    with pytest.raises(ValueError):
        packed6_decode(b"\x00\x00\x00")


def test_callsign_case_folding():
    assert pack_callsign("m7tff-3") == pack_callsign("M7TFF-3")


def test_bad_callsign_rejected():
    for bad in ["X", "TOOLONGCALL", "A B"]:
        with pytest.raises(ValueError):
            pack_callsign(bad)


def test_bad_ssid_rejected():
    with pytest.raises(ValueError):
        pack_callsign("M7TFF-16")   # base-36 "16"->42, past '?'
    with pytest.raises(ValueError):
        pack_callsign("M7TFF-!!")   # non-base-36 SSID text


def test_bad_grid_rejected():
    for bad in ["IO8", "TOOLONGGRID"]:
        with pytest.raises(ValueError):
            pack_grid(bad)


def test_reed_solomon_can_reconstruct_a_field_that_is_not_a_callsign():
    """Both survivors of the 81-recording replay corpus. RS corrects the block,
    the field is the right length and holds no space, and neither is a station."""
    from hfmodem.besra.frame.callsign import packed6_encode, unpack_callsign
    for junk in ('FC<C155', '*?V"*JJ'):
        with pytest.raises(ValueError):
            unpack_callsign(packed6_encode(f"{junk}0"))
