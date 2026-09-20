# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""hfmodem — four HF softmodems and the station that runs them.

Each protocol is a subpackage: `shrike` (PACTOR-1/2/3), `kestrel` (VARA),
`besra` (ARDOP) and `sabir`. They share `core` for everything that touches the
radio rather than the waveform, and `host` for the dialects an application
speaks to a modem.
"""

__version__ = "0.1.0"
