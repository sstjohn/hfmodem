# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Versioned transmit evidence shared by modem adapters.

No payload bytes, decoder guesses, file writes or radio operations. Producers
buffer while keyed and serialize only after leaving their timing-critical path.
A session ID plus tx_id identifies an emission; waveform updates use the same ID.
Sample positions belong only to the named recording's capture clock. Missing
positions are null, never inferred from UTC or a different device's clock.
"""
import json
import time
import uuid

PREFIX = 'tx_event '
SCHEMA = 1
OUTCOMES = {'sent', 'refused', 'error', 'dry_run', 'unconfirmed'}


class TxEvents:
    def __init__(self, protocol, *, session_id=None):
        self.protocol = protocol
        self.session_id = session_id or uuid.uuid4().hex
        self.pending = []

    def record(self, event, tx_id, *, outcome, **fields):
        if event not in {'completed', 'waveform'} or outcome not in OUTCOMES:
            raise ValueError('Invalid transmit event or outcome')
        if fields.keys() & {'schema', 'session_id', 'event', 'tx_id', 'direction', 'available_mono_s'}:
            raise ValueError('Transmit envelope fields cannot be overridden')
        record = {'schema': SCHEMA, 'session_id': self.session_id,
                  'event': event, 'tx_id': tx_id, 'protocol': self.protocol,
                  'direction': 'tx', 'available_mono_s': time.monotonic(),
                  **fields, 'outcome': outcome}
        self.pending.append(record)
        return record

    def lines(self):
        """Drain after unkey. A caller chooses where the records are persisted."""
        lines = [PREFIX + json.dumps(r, separators=(',', ':'), allow_nan=False)
                 for r in self.pending]
        self.pending.clear()
        return lines


def read_line(line):
    """Unknown schema versions are not reinterpreted as this version."""
    if not line.strip().startswith(PREFIX):
        return None
    try:
        record = json.loads(line.strip()[len(PREFIX):])
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        return None
    if record.get('event') not in {'completed', 'waveform'} or record.get('outcome') not in OUTCOMES:
        return None
    if not isinstance(record.get('protocol'), str) or not isinstance(record.get('tx_id'), int):
        return None
    return record
