# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""besra driven through hfhost's ARDOP client — the two halves, meeting.

hfhost's ARDOP client was written from the protocol document. besra's host server
was written from the same document, separately. Neither imports the other, and the
gate in `tests/gates/test_import_direction.py` is what keeps it that way.

That independence is the only reason this test says anything. A client derived from
the server agrees with it by construction and grades nothing; two implementations
that were never allowed to see each other either interoperate or they do not, and
this is where that gets decided.

These are seam tests, not conformance probes. Conformance is dialect-specific and
lives in `conformance.py` and `conformance_hostapi.py`; what is checked here is
that creance can reach an ARDOP modem at all — which is what `creance chat` and the
throughput campaigns need, and what they could not do before the client existed.
"""
from __future__ import annotations

import os
import queue
import socket
import tempfile
import threading
import time

import pytest

from hfhost.config import ModemConfig
from hfhost.link import open_link
from hfhost.transcript import Transcript

pytest.importorskip("hfmodem.besra.host.server")


def _transcript() -> Transcript:
    return Transcript(os.path.join(tempfile.mkdtemp(), "seam.jsonl"), "seam")


def _free_ports(n: int) -> list[int]:
    """Ports nobody is listening on, from the ephemeral range.

    This fixture used to count up from a fixed 18960. Two copies of the suite
    then served each other's clients: the second bind lost with EADDRINUSE, its
    server thread died, and its tests passed anyway by talking to the first
    copy's modem. A stale process holding one of those ports fails the fixture
    outright, which is the same defect wearing its other face.
    """
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


@pytest.fixture
def besra_link():
    from hfmodem.besra.host.server import HostServer
    cmd_port, data_port = _free_ports(2)
    srv = HostServer(control_port=cmd_port, data_port=data_port, quiet=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    deadline = time.monotonic() + 5.0
    cfg = ModemConfig(name="besra", cmd_port=cmd_port, data_port=data_port,
                      dialect="ardop")
    link = None
    while time.monotonic() < deadline:
        try:
            link = open_link(cfg, "127.0.0.1", _transcript())
            link.configure("W9SSJ")
            link.attach()
            break
        except OSError:
            time.sleep(0.05)
    assert link is not None and link.attached, "never attached to besra"
    yield link
    link.close()
    if hasattr(srv, "stop"):
        srv.stop()


def test_creance_can_reach_an_ardop_modem(besra_link):
    """The gap this closes: creance had no ARDOP client, so besra was the one
    modem of the four it could not drive over a host interface at all."""
    assert besra_link.attached
    assert besra_link.attach_failures == 0


def test_the_modem_identifies_itself(besra_link):
    """A version string means the command socket carried a request and the reply
    came back correlated to it — the whole transport, in one round trip."""
    assert besra_link.request_version().startswith("besra")


def test_desired_state_is_recorded_before_the_socket_exists(besra_link):
    """`configure` runs in the fixture *before* `attach`, which is the documented
    order: a modem may be configured before its process is even up, and the state
    is replayed on every attach. A client that wrote immediately would have raised
    there, so arriving here at all is the assertion."""
    assert besra_link.attached


def test_an_unconnected_modem_reports_an_empty_queue(besra_link):
    assert besra_link.queue_bytes == 0
    assert not besra_link.connected


def test_listening_is_accepted_and_the_link_stays_up(besra_link):
    q: queue.Queue = queue.Queue()
    besra_link.subscribe(q)
    besra_link.set_listen(True)
    time.sleep(0.4)
    assert besra_link.attached, "the modem dropped the host link on LISTEN"
    assert besra_link.attach_failures == 0
