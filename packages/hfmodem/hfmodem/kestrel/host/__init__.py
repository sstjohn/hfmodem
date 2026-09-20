# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""kestrel host_api: the host-side VARA-protocol TCP server.

This package implements the *modem* side of VARA's documented dual-socket TCP
host interface (command 8300 + data 8301). An application (Winlink Express,
Pat, VarAC, or any conforming host-API client) attaches to the two sockets and
speaks the ``\\r``-terminated ASCII protocol; the link behaviour is delegated
to a pluggable :class:`~host_api.modem_core.ModemCore` (the bundled
:class:`~host_api.modem_core.LoopbackModem` runs the whole stack with no radio).

Console entry point: ``kestrel-modem`` -> :func:`host_api.run_server.main`.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.1.0"
