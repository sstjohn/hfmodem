# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import pytest
from hfmodem.contact.broadcast import encode_ui, decode_ui


@pytest.mark.parametrize("command", [0, 1])
def test_direct_ui_roundtrip(command):
    raw = encode_ui("W9SSJ-7", "ALL", "Café ☃".encode(), command=command)
    result = decode_ui(raw, command=command)
    assert (result.source, result.destination, result.text) == (
        "W9SSJ-7",
        "ALL",
        "Café ☃",
    )
    normalized = bytearray(raw)
    normalized[6 + command] &= 0x7F
    assert decode_ui(normalized, command=command) == result


def test_stock_command1_fixture():
    raw = bytes.fromhex("8298984040404060ae72a6a69440406103f05549312054455354")
    assert decode_ui(raw).info == b"UI1 TEST"
    assert decode_ui(raw).source == "W9SSJ"


def test_seven_character_host_envelope_preserves_tactical_identity():
    raw = encode_ui("X-W9SSJ", "ALL", b"test")
    assert decode_ui(raw).source == "X-W9SSJ"
    with pytest.raises(ValueError):
        encode_ui("X-W9SSJ", "ALL", b"test", command=0)


def test_binary_is_preserved_and_not_decoded_as_text():
    result = decode_ui(encode_ui("W9SSJ", "ALL", b"\xff<Q>"))
    assert result.info == b"\xff<Q>"
    with pytest.raises(UnicodeDecodeError):
        result.text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x[:-1][:18],
        lambda x: x[:16] + b"\x13\xf0" + x[18:],
        lambda x: x[:7] + b"\xe1" + x[8:],
        lambda x: b"\x83" + x[1:],
    ],
)
def test_rejects_malformed_envelope(mutation):
    with pytest.raises(ValueError):
        decode_ui(mutation(encode_ui("W9SSJ", "ALL", b"test")))
