# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Wire-level constants for VARA's documented TCP host protocol.

Shared by the server and its tests. Kept tiny and dependency-free so the real
kestrel firmware could reuse it verbatim. See the protocol notes for the
sourced command/response reference (EA5HVK 2022 doc + Pat-Vara).
"""

from __future__ import annotations

CR = b"\r"                     # every command-channel message ends in one CR
DEFAULT_CMD_PORT = 8300
DEFAULT_DATA_PORT = 8301

# Version string this modem reports to VERSION. The "VERSION" reply itself is a
# newer (Pat-Vara) command, not in EA5HVK's 2022 list -> UNVERIFIED.
VERSION_STRING = "VERSION 4.9.0.KestrelOpen"

# Bandwidths selectable on VARA HF (Hz).
BANDWIDTHS = ("500", "2300", "2750")

# Compression modes.
COMPRESSION_MODES = ("OFF", "TEXT", "FILES")

# Responses / async notifications (modem -> host).
OK = "OK"
WRONG = "WRONG"
IAMALIVE = "IAMALIVE"
