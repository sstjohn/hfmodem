# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.ax25.frame import Frame, address, check_fcs, parse, ui_frame, with_fcs


def test_frame_and_fcs_boundaries():
    raw = ui_frame("N0CALL-15", "CQ", b'\xc0\xdbhello', ("WIDE1-1",))
    parsed = parse(raw)
    assert parsed.encode() == raw
    assert parsed.pid == 0xF0 and parsed.control == 3
    assert len(parsed.addresses) == 3
    assert check_fcs(with_fcs(raw)) == raw
    with pytest.raises(ValueError, match="FCS"):
        wire = with_fcs(raw)
        check_fcs(wire[:-1] + bytes([wire[-1] ^ 1]))


@pytest.mark.parametrize("call", ["", "ABCDEFG", "lower", "CALL-16", "CALL--1", "CALL-x", "A!", "CALL-"])
def test_bad_callsigns(call):
    with pytest.raises(ValueError):
        address(call)


@pytest.mark.parametrize("control,pid", [(0, 0xF0), (3, 0xF0), (0x13, 0xF0), (0x3F, None), (1, None)])
def test_control_pid(control, pid):
    prefix = address("CQ", flag=True) + address("N0CALL", last=True)
    raw = prefix + bytes([control]) + (b'' if pid is None else bytes([pid]))
    assert parse(raw).pid == pid
    if pid is not None:
        with pytest.raises(ValueError, match="PID"):
            parse(raw[:-1])


@pytest.mark.parametrize("control,pid", [(0, None), (3, None), (0x13, None),
                                       (0x3F, 0xF0), (1, 0xF0)])
def test_encoder_does_not_reinterpret_information_as_pid(control, pid):
    addresses = (address("CQ", flag=True), address("N0CALL", last=True))
    with pytest.raises(ValueError, match="PID"):
        Frame(addresses, control, pid, b'information').encode()


@pytest.mark.parametrize("count", [1, 11])
def test_address_count(count):
    raw = b''.join(address("A", last=i == count - 1) for i in range(count)) + b'\x03\xf0'
    with pytest.raises(ValueError, match="two to ten"):
        parse(raw)


@pytest.mark.parametrize("kind", ["odd", "alphabet", "truncated", "unterminated", "control"])
def test_malformed_frame(kind):
    raw = bytearray(ui_frame("N0CALL", "CQ", b''))
    if kind == "odd":
        raw[0] |= 1
    elif kind == "alphabet":
        raw[0] = ord('!') << 1
    elif kind == "truncated":
        raw = raw[:12]
    elif kind == "unterminated":
        raw[13] &= 0xFE
    else:
        raw = raw[:14]
    with pytest.raises(ValueError):
        parse(bytes(raw))
