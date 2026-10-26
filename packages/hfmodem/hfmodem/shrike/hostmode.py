# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Compatibility imports for shared WA8DED/SCS host framing."""

from ..core.hostmode import (
    HEADER as HEADER,
    REQUEST as REQUEST,
    OK as OK,
    MSG as MSG,
    FAIL as FAIL,
    LINK as LINK,
    MON_HEADER as MON_HEADER,
    MON_HEADER_INFO as MON_HEADER_INFO,
    MON_INFO as MON_INFO,
    DATA as DATA,
    SENT_ECHO as SENT_ECHO,
    MAX_DATA as MAX_DATA,
    Packet as Packet,
    Response as Response,
    wrap as wrap,
    unwrap as unwrap,
    command as command,
    data as data,
    _to_modem as _to_modem,
    reply as reply,
    reply_text as reply_text,
    CRC_ERROR as CRC_ERROR,
    REQUEST_EVENT as REQUEST_EVENT,
    Decoder as Decoder,
    Modem as Modem,
    _selftest as _selftest,
    crc16 as crc16,
)

if __name__ == "__main__":
    _selftest()
