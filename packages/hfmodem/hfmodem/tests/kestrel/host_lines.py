# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Waiting on a host's command lines through a queue subscribed ahead of time."""
from __future__ import annotations

import queue
import time


def seen(q, prefix, timeout):
    """The first line on ``q`` starting with ``prefix``, or None after ``timeout``.

    ``VaraClient.wait_for`` subscribes only when called, so it misses a line that
    arrived first: the responder decodes the confirm while the initiator's own
    CONNECTED is still on its way to the test thread, and a loaded host lets the
    responder's line win that race.
    """
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        try:
            msg = q.get(timeout=left)
        except queue.Empty:
            return None
        if msg.startswith(prefix):
            return msg
    return None
