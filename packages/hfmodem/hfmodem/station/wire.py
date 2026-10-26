# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

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
