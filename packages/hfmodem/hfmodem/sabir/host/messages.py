# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host-interface message schema (HOST-API.md §3-§5, §10).

A message is a plain dict with an integer ``m`` (message-type code) plus named
fields. On the wire it is a CBOR map with integer keys: ``0`` is ``m``, and
each field name maps to a stable integer via the append-only registry below.
Decode is must-ignore: unknown field keys and unknown message types are dropped,
never errors (§10.1), so a newer peer never faults an older one.
"""

from __future__ import annotations

from typing import Any, Dict

from . import cbor

PROTO = "1.0"

# -- message-type codes (§4 commands 0-31, §5 events 32-63) -------------------
HELLO = 0                          # symmetric handshake, both directions
SET_IDENTITY = 1
SET_PROFILE = 2
LISTEN = 3
CONNECT = 4
SEND = 5
DISCONNECT = 6
ABORT = 7
SUBSCRIBE = 8
CONFIGURE = 9
BEACON = 10                        # emit a connectionless presence beacon
SEND_OBJECT = 11                   # connectionless, no delivery acknowledgement

STATE_CHANGED = 32
CAPABILITIES = 33
LINK_STATS = 34
DATA_RECEIVED = 35
SEND_PROGRESS = 36
ID_SENT = 37
PEER_OBSERVED = 38
PHYSICAL_STATE = 39
ERROR = 40
OBJECT_RECEIVED = 41

# -- link-state codes for STATE_CHANGED.state --------------------------------
ST_DISCONNECTED = 0
ST_LISTENING = 1
ST_CONNECTING = 2
ST_CONNECTED = 3
ST_DISCONNECTING = 4

# FSM SessionState.name -> wire code
STATE_CODE = {"DISCONNECTED": ST_DISCONNECTED, "LISTENING": ST_LISTENING,
              "CONNECTING": ST_CONNECTING, "CONNECTED": ST_CONNECTED,
              "DISCONNECTING": ST_DISCONNECTING}

# -- disconnect/refusal reason codes (STATE_CHANGED.reason) ------------------
RS_REMOTE = 1                      # peer disconnected
RS_LOCAL = 2                       # local disconnect/abort
RS_LINK_FAILED = 3                 # rebuild limit / no progress
RS_REFUSED = 4                     # negotiation CFAIL (NEGOTIATION.md §4.3)

# -- operating-profile codes (§7) --------------------------------------------
PF_AMATEUR = 1                     # US Part 97
PF_UNRESTRICTED = 2

# -- error codes (§5) --------------------------------------------------------
ERR_INCOMPATIBLE = 1              # major proto mismatch
ERR_BAD_STATE = 2                # command illegal in the current link state
ERR_NO_IDENTITY = 3             # connect/listen before SetIdentity
ERR_MALFORMED = 4               # undecodable frame

# -- field-name registry: append-only, index is the wire key (0 = m) ---------
_FIELDS = (
    "m", "proto", "features", "client", "modem", "profiles",
    "identity_required", "station_id", "aliases", "profile", "on", "peer_id",
    "deadline", "ref", "data", "stream", "priority", "deflate", "id",
    "events", "stats_period", "radio", "ptt", "state", "reason", "peer_capabilities",
    "usable", None, "peer_profiles", "gear", "rung", "snr3k_db",
    "group_snr_db", "control_tier", "throughput_bps", "queue_bytes", "eta_s",
    "harq_rounds", "rebuilds", "compression_ratio", "sent", "total",
    "delivered", "deflated", "t", "capabilities", "busy", "code", "detail",
    "addressee",
    "data_profile", "receive_profiles", "feedback_iters", "impulse_blank",
    "bandwidth_hz", "destination", "service", "message_id", "parity", "repeats",
    "inactivity_timeout_s",
)
KEY = {name: i for i, name in enumerate(_FIELDS) if name is not None}
NAME = {i: name for i, name in enumerate(_FIELDS) if name is not None}


def encode(msg: Dict[str, Any]) -> bytes:
    """A message dict -> a length-prefixed CBOR frame (4-byte BE length)."""
    out = {}
    for name, val in msg.items():
        key = KEY.get(name)
        if key is None:
            raise KeyError(f"unregistered field {name!r}")
        out[key] = val
    body = cbor.encode(out)
    return len(body).to_bytes(4, "big") + body


def decode(body: bytes) -> Dict[str, Any]:
    """A CBOR frame body (no length prefix) -> a message dict. Unknown integer
    keys are dropped (must-ignore, §10.1)."""
    raw = cbor.decode(body)
    if not isinstance(raw, dict):
        raise ValueError("host message is not a CBOR map")
    if any(type(k) is not int or k < 0 for k in raw):
        raise ValueError("host field keys must be unsigned integers")
    if type(raw.get(0)) is not int or raw[0] < 0:
        raise ValueError("host message type must be an unsigned integer")
    return {NAME[k]: v for k, v in raw.items() if k in NAME}
