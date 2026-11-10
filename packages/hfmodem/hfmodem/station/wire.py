# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ipaddress
import socket

DEFAULT_PORT = 8737
SERVICE_NAME = 'hfmodem station'
SCHEMA = 2

STATE = '/state'
EVENTS = '/events'
COMMANDS = '/commands'

EVENT_TYPES = frozenset({
    'rig', 'owner', 'tx', 'link', 'session', 'tape', 'logbook', 'gap',
    'command', 'ptt', 'warning',
})
SERVICE_STATES = frozenset({
    'starting', 'observing', 'owning', 'owning-receive', 'releasing', 'failed',
})

REFUSAL_STARTING = 'starting'
REFUSAL_SHUTTING_DOWN = 'shutting-down'
REFUSAL_TRANSITIONING = 'transitioning'
REFUSAL_QUEUE_FULL = 'queue-full'
REFUSAL_UNKNOWN_COMMAND = 'unknown-command'
REFUSAL_CLAIMED = 'claimed'
REFUSAL_NOT_CONTROLLER = 'not-controller'
REFUSAL_SUPERSEDED = 'superseded'

EXIT_ATTACHED = 0
EXIT_CLIENT_ERROR = 1
EXIT_ALREADY_RUNNING = 3
EXIT_PORT_CONFLICT = 4
EXIT_CLAIMED = 5
EXIT_LAUNCH_FAILED = 6

# Windows 11 takes about 2.03 s to refuse a loopback connect after retrying SYNs.
LOOPBACK_CONNECT_TIMEOUT = 3.0


def create_connection(address, timeout, source_address=None):
    """Allow loopback refusal to surface, then restore the response timeout."""
    host = address[0]
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == 'localhost'
    connect_timeout = (max(timeout, LOOPBACK_CONNECT_TIMEOUT)
                       if loopback and timeout is not None else timeout)
    connection = socket.create_connection(address, timeout=connect_timeout,
                                          source_address=source_address)
    connection.settimeout(timeout)
    return connection
