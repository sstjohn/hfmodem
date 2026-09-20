# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A radioless echo peer for demonstrating a live `BesraModem` end to end.

`besra_with_echo_peer` returns a threaded `BesraModem` for the host server to
drive, joined over a `ThreadedAir` to a second threaded modem that listens,
accepts a connection, and echoes back whatever data it receives. So a real
client (Pat) can `ARQCALL` the peer, send bytes, and read them back — the whole
stack (host dialect → ARQ → modulate → air → demodulate → ARQ → back) exercised
with no radio.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..arq.modem import BesraModem
from ..host.modem_core import ModemObserver
from .air import ThreadedAir


class _EchoObserver(ModemObserver):
    """Bounces received ARQ data straight back to the sender."""

    def __init__(self) -> None:
        self.modem: BesraModem | None = None

    def modem_newstate(self, state): pass
    def modem_connected(self, remote, bw): pass
    def modem_disconnected(self): pass
    def modem_ptt(self, on): pass
    def modem_buffer(self, n): pass
    def modem_status(self, text): pass

    def modem_data_received(self, kind, blob):
        if self.modem is not None:
            self.modem.transmit(blob)


def besra_with_echo_peer(peer_call: str = "BESRA-1", *, bandwidth: int = 500,
                         channel: Callable[[np.ndarray], np.ndarray] | None = None
                         ) -> BesraModem:
    """A threaded host modem joined to a listening, echoing peer over a virtual
    air. Hand the returned modem to `HostServer`; the peer runs itself."""
    air = ThreadedAir(channel)
    host_modem = BesraModem(bandwidth=bandwidth, threaded=True)

    peer = BesraModem(bandwidth=bandwidth, threaded=True)
    peer.set_mycall(peer_call.upper())
    peer.set_listen(True)
    echo = _EchoObserver()
    echo.modem = peer

    air.join(host_modem)
    air.join(peer)
    peer.start(echo)
    return host_modem
