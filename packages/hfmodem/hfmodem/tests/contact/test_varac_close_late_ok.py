# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""An outstanding DISCONNECT response is not confirmation of a later ABORT."""
import pytest

from hfmodem.tests.contact.test_varac import Desk


def test_late_disconnect_ok_cannot_confirm_an_unanswered_abort():
    desk = Desk(**{'CLEANTXBUFFER': ['CLEANTXBUFFER FAILED'], 'DISCONNECT': [], 'ABORT': []})
    desk.host.replies['VERSION'] = []
    client = desk.host.client
    client.connected = True
    steps = desk.flow.close_steps(timeout=1)
    assert next(steps) == .1
    client.poll(0)
    desk.host.now += 1
    assert next(steps) == .1
    assert desk.sent[-1] == 'ABORT'
    # DISCONNECT's command response arrives after ABORT was sent. The host has
    # not acknowledged ABORT and still reports a connected session. Its ordered
    # VERSION fence follows that old OK, without any ABORT OK behind it.
    desk.host.lines('OK', 'VERSION test-host')
    client.poll(0)
    desk.host.now += 1
    with pytest.raises(TimeoutError, match='ABORT'):
        client.finish(steps)
    assert client.connected
