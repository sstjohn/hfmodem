# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import io
import threading
import time

from creance import chat as chatmod
from hfhost.testing.hostapi_bridge import HostApiBridge


class _StubLink:
    """A Link that echoes send() straight into its own recv queue via a peer,
    so a chat can be exercised with no modem at all."""

    def __init__(self, name="a"):
        self.name = name
        self.epoch = 0
        self._rx = bytearray()
        self._cv = threading.Condition()
        self.peer = None
        self.connected = True
        self._subs = []

    def wire(self, other):
        self.peer = other

    def subscribe(self, q):
        self._subs.append(q)

    def send(self, data, label=""):
        if self.peer is not None:
            with self.peer._cv:
                self.peer._rx += data
                self.peer._cv.notify_all()

    def recv(self, n=None, timeout=30.0, epoch=None, cancel=None):
        end = time.monotonic() + timeout
        with self._cv:
            while not self._rx:
                left = end - time.monotonic()
                if left <= 0:
                    return b""
                self._cv.wait(left)
            out, self._rx = bytes(self._rx), bytearray()
            return out


def test_a_typed_line_reaches_the_other_terminal():
    a, b = _StubLink("a"), _StubLink("b")
    a.wire(b)
    b.wire(a)

    import tempfile, pathlib
    from hfhost.transcript import Transcript
    d = pathlib.Path(tempfile.mkdtemp())
    ta = Transcript(str(d / "a.jsonl"), "a")
    tb = Transcript(str(d / "b.jsonl"), "b")

    b_out = io.StringIO()
    peer_b = chatmod._Peer(b, "N0BBB", tb, out=b_out, inp=io.StringIO(""))
    rx = threading.Thread(target=peer_b._pump_in, daemon=True)
    rx.start()

    peer_a = chatmod._Peer(a, "N0AAA", ta,
                           out=io.StringIO(), inp=io.StringIO("hello over the air\n"))
    peer_a._pump_out()
    time.sleep(0.3)
    peer_b._stop.set()
    rx.join(timeout=2.0)

    assert "hello over the air" in b_out.getvalue()
    ta.close(); tb.close()


def test_quit_ends_the_out_pump():
    a = _StubLink("a")
    import tempfile, pathlib
    from hfhost.transcript import Transcript
    t = Transcript(str(pathlib.Path(tempfile.mkdtemp()) / "t.jsonl"), "t")
    peer = chatmod._Peer(a, "N0AAA", t, out=io.StringIO(), inp=io.StringIO("hi\n/quit\nnope\n"))
    peer._pump_out()
    assert peer._stop.is_set()
    t.close()


def test_chat_connects_and_talks_over_a_bridged_pair(tmp_path):
    """The real thing: initiator and responder halves over the structured
    bridge, one line each way, dialect-agnostic through the Link seam."""
    from hfhost.config import ModemConfig
    from creance.config import (Config, InitiatorConfig, ResponderConfig,
                                SiteConfig)
    br = HostApiBridge(tmp_path).start()

    def cfg(name, call, port):
        return Config(
            site=SiteConfig(name=name, mycall=call, results_dir=str(tmp_path / name)),
            responder=ResponderConfig(), initiator=InitiatorConfig(),
            modems=(ModemConfig(name="m", dialect="hostapi",
                                cmd_port=port, data_port=0),))

    ci = cfg("init", "N0AAA", br.a.port)
    cr = cfg("resp", "N0BBB", br.b.port)

    # a Connect on either fake links both, so the responder sees CONNECTED
    r_out = io.StringIO()
    r_done = {}

    def responder():
        r_done["outcome"] = chatmod.chat(
            cr, "m", dst=None, connect_timeout=8.0,
            out=r_out, inp=io.StringIO("hi back\n"))

    t = threading.Thread(target=responder, daemon=True)
    t.start()
    time.sleep(0.5)

    i_out = io.StringIO()
    outcome = chatmod.chat(ci, "m", dst="N0BBB", connect_timeout=8.0,
                           out=i_out, inp=io.StringIO("hello there\n"))
    t.join(timeout=10.0)
    br.stop()

    assert outcome == "ok"
    assert "hello there" in r_out.getvalue()      # initiator's line reached B
    assert "hi back" in i_out.getvalue()          # B's line reached the initiator
    assert r_done.get("outcome") == "ok"
