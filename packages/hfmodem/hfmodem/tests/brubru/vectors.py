# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from hfmodem.ax25.frame import ui_frame

CAPACITIES = (15, 21, 45, 75, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 312,
              *range(345, 826, 30))


def frames_for(size: int, seed: int = 0) -> list[bytes]:
    def content(n, variant):
        line = f'{seed + variant:02d} THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789. '.encode()
        return (line * (1 + n // len(line)))[:n]

    if size == 15:
        return [ui_frame("N0CALL", "CQ", b'')[:-2] + b'\x3f']
    if size <= 399:
        return [ui_frame("N0CALL", "CQ", content(size - 16, 0))]
    share, remainder = divmod(size - 26, 3)
    return [ui_frame("N0CALL", "CQ", content(share + (remainder if i == 0 else 0), i))
            for i in range(3)]
