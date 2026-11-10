# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

CRC_VECTORS = (
    (b'', 'x25', 0x0000),
    (b'123456789', 'x25', 0x906E),
    (b'\x04\x01\x01\x47\x47', 'x25', 0x99D5),
    (b'', 'ccitt-false', 0xFFFF),
    (b'123456789', 'ccitt-false', 0x29B1),
    (b'', 'kermit', 0x0000),
    (b'123456789', 'kermit', 0x2189),
)
