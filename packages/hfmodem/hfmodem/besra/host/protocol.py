# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ARDOP host-dialect constants and the two-port derivation.

The authoritative command table and notification grammar are compiled from the
open references in `docs/protocols/ardop/10-HOST-API.md`; this module holds the stable
protocol constants the transport needs regardless of that table's final shape.

The parts of this module ``NOTICE`` names as ardopcf's are under that project's
MIT licence, Copyright (c) 2014-2024 Rick Muething, John Wiseman, Peter LaRue;
the copyright and permission notice it requires ship in ``NOTICE``.
"""

from __future__ import annotations

from enum import StrEnum

#: Control port a stock ARDOP TNC listens on by default.
DEFAULT_CONTROL_PORT = 8515

#: Commands and notifications on the control port are CR-terminated ASCII.
CR = b"\r"


def data_port_for(control: str | int) -> str | int:
    """Derive the data port from the control address the way ARDOP clients do:
    by incrementing the last character of the address string. Pat's transport
    does exactly this and comments it ``// Oh no he didn't!``.

    We reproduce the *behaviour* for compatibility (``8515`` → ``8516``) without
    adopting the technique into any of besra's own interfaces. An ``int`` in,
    an ``int`` out; a host:port string keeps its shape.
    """
    if isinstance(control, int):
        return control + 1
    return control[:-1] + chr(ord(control[-1]) + 1)


# -- the async state machine ARDOP exposes (NEWSTATE <value>) ----------------
# ARDOP's genuine improvement over VARA: the modem publishes its own state
# instead of leaving the host to infer it from side effects. The eight tokens
# and their exact upper-case spelling are the reference's ARDOPStates[8]
# (ardopcf ARDOPC.c:183) — reproduced verbatim so a client sees the incumbent's
# bytes. See docs/protocols/ardop/10-HOST-API.md §4.

class ArdopState(StrEnum):
    """ARDOP's own eight-state vocabulary — a StrEnum so it stays wire-identical.

    Deliberately *separate* from the flock's five-value `SessionState`
    (`DISCONNECTED/LISTENING/CONNECTING/CONNECTED/DISCONNECTING`): ARDOP publishes
    a richer, protocol-normative set with the send/receive role in the state
    itself, and folding it into five would lose that. Members compare and format
    as their string, so `NEWSTATE <value>` and every `state == ArdopState.DISC`
    behave exactly as the plain strings did."""

    OFFLINE = "OFFLINE"        # sound card released, not listening
    DISC = "DISC"              # initialised, listening, no session
    ISS = "ISS"                # Information Sending Station (this end is sending)
    IRS = "IRS"                # Information Receiving Station (this end is receiving)
    IDLE = "IDLE"              # connected, neither side sending
    IRStoISS = "IRStoISS"      # mid-turnover receive→send (transient)
    FECSEND = "FECSEND"        # transmitting FEC (broadcast, no ARQ)
    FECRCV = "FECRCV"          # receiving FEC


#: States in which an ARQ session is up (connect edge has fired, DISC has not).
CONNECTED_STATES = (ArdopState.ISS, ArdopState.IRS, ArdopState.IDLE, ArdopState.IRStoISS)
