# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ARDOP client against a fake TNC built from the documented dialect:
command echoes and interleaved status, counted data blocks, faults."""

import queue
import socket
import threading

import pytest

from hfhost import link as link_mod
from hfhost.ardop import (ArdopClient, ArdopFault, ArdopLink, ECHO, FAULT,
                          classify, _arqbw)
from hfhost.config import ModemConfig
from hfhost.link import (BUSY, CONNECTED, DISCONNECTED, ERROR, PTT, open_link)
from hfhost.transcript import Transcript, read as read_transcript
from hfhost.wire import NOTIFICATION, UNKNOWN

CR = b"\r"


def _listen(host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 0))
    s.listen(4)
    return s


class FakeTnc:
    """A fake ARDOP TNC, written from the host-interface document.

    It answers commands the way the document says the deployed modem does — the
    bare verb for an action, `<CMD> now <value>` for a setter, the original line
    for ARQCALL, `DISCONNECT NOW TRUE` for a disconnect in session, and the
    misspelled not-recoginized fault for anything else. on_command overrides any
    of that: return a line, a list of lines, or "" for silence.

    Data blocks are counted, two-byte big-endian, tagged only modem-to-host.
    """

    def __init__(self, *, on_command=None, version="fake_1.0.0"):
        self.on_command = on_command
        self.version = version
        self._cmd_srv, self._data_srv = _listen(), _listen()
        self.cmd_port = self._cmd_srv.getsockname()[1]
        self.data_port = self._data_srv.getsockname()[1]

        self.commands: list[str] = []
        self.received = bytearray()          # payload, deframed
        self.blocks: list[bytes] = []        # one entry per counted block
        self.in_session = False

        self._cv = threading.Condition()
        self._cmd_conn: socket.socket | None = None
        self._data_conn: socket.socket | None = None
        self._running = True
        threading.Thread(target=self._serve, name="faketnc", daemon=True).start()

    # -- lifecycle ---------------------------------------------------------

    def _serve(self):
        try:
            cmd_conn, _ = self._cmd_srv.accept()
            data_conn, _ = self._data_srv.accept()
        except OSError:
            return
        with self._cv:
            self._cmd_conn, self._data_conn = cmd_conn, data_conn
            self._cv.notify_all()
        threading.Thread(target=self._data_loop, args=(data_conn,),
                         daemon=True).start()
        self._cmd_loop(cmd_conn)

    def close(self):
        self._running = False
        for s in (self._cmd_srv, self._data_srv, self._cmd_conn, self._data_conn):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    # -- command socket ----------------------------------------------------

    def _cmd_loop(self, conn):
        buf = b""
        try:
            while self._running:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while CR in buf:
                    raw, buf = buf.split(CR, 1)
                    if raw:
                        self._dispatch(raw.decode("ascii", "replace"))
        except OSError:
            pass

    def _dispatch(self, cmd):
        with self._cv:
            self.commands.append(cmd)
            self._cv.notify_all()
        reply = self.on_command(cmd) if self.on_command else None
        if reply is None:
            reply = self._default_reply(cmd)
        for line in ([reply] if isinstance(reply, str) else reply):
            if line:
                self.notify(line)

    def _default_reply(self, cmd):
        verb, _, rest = cmd.partition(" ")
        verb = verb.upper()
        if verb in ("INITIALIZE", "ABORT", "SENDID", "PURGEBUFFER"):
            return verb
        if verb == "VERSION":
            return f"VERSION {self.version}"
        if verb == "STATE":
            return "STATE ISS" if self.in_session else "STATE DISC"
        if verb == "ARQCALL":
            return cmd                       # the original line, not `now`
        if verb == "DISCONNECT":
            if not self.in_session:
                return "DISCONNECT IGNORED"
            self.in_session = False
            return ["DISCONNECT NOW TRUE", "NEWSTATE DISC ", "DISCONNECTED"]
        if verb in ("MYCALL", "LISTEN", "PROTOCOLMODE", "ARQBW", "GRIDSQUARE",
                    "ARQTIMEOUT", "FSKONLY", "CALLBW"):
            return f"{verb} now {rest}"
        return f"FAULT CMD {cmd} not recoginized"

    def notify(self, line):
        with self._cv:
            conn = self._cmd_conn
        if conn is None:
            raise RuntimeError("no attached host")
        try:
            conn.sendall(line.encode("ascii") + CR)
        except OSError:
            pass

    def answer_call(self, peer, bw=2000):
        self.in_session = True
        self.notify("NEWSTATE ISS ")
        self.notify(f"CONNECTED {peer} {bw}")

    # -- data socket -------------------------------------------------------

    def _data_loop(self, conn):
        buf = bytearray()
        try:
            while self._running:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while len(buf) >= 2:
                    size = int.from_bytes(buf[:2], "big")
                    if len(buf) < 2 + size:
                        break
                    block = bytes(buf[2:2 + size])
                    del buf[:2 + size]
                    with self._cv:
                        self.blocks.append(block)
                        self.received += block
                        self._cv.notify_all()
                    self.notify(f"BUFFER {len(self.received)}")
        except OSError:
            pass

    def send_data(self, tag, payload):
        with self._cv:
            conn = self._data_conn
        if conn is None:
            raise RuntimeError("no attached host")
        block = tag + payload
        conn.sendall(len(block).to_bytes(2, "big") + block)

    # -- waiting -----------------------------------------------------------

    def wait_attached(self, timeout=2.0):
        with self._cv:
            return self._cv.wait_for(lambda: self._cmd_conn is not None, timeout)

    def wait_command(self, prefix, timeout=2.0):
        with self._cv:
            return self._cv.wait_for(
                lambda: any(c.startswith(prefix) for c in self.commands), timeout)

    def wait_data(self, n, timeout=2.0):
        with self._cv:
            return self._cv.wait_for(lambda: len(self.received) >= n, timeout)


@pytest.fixture
def tnc():
    t = FakeTnc()
    yield t
    t.close()


@pytest.fixture
def tr(tmp_path):
    t = Transcript(str(tmp_path / "daemon.jsonl"), "sid-ardop")
    yield t
    t.close()


def make_client(tnc, tr, **kw):
    cfg = ModemConfig(name="ardop", dialect="ardop", cmd_port=tnc.cmd_port,
                      data_port=tnc.data_port, bandwidth=kw.pop("bandwidth", "2300"))
    kw.setdefault("attach_timeout_s", 2.0)
    return ArdopClient(cfg, "127.0.0.1", tr, **kw), cfg


@pytest.fixture
def client(tnc, tr):
    c, _ = make_client(tnc, tr)
    yield c
    c.close()


@pytest.fixture
def lk(tnc, tr):
    c, cfg = make_client(tnc, tr)
    link = ArdopLink(c, cfg)
    yield link
    link.close()


def _events(link):
    q: queue.Queue = queue.Queue()
    link.subscribe(q)
    return q


def _wait(q, kind, timeout=2.0):
    while True:
        ev = q.get(timeout=timeout)
        if ev.kind == kind:
            return ev


# -- grammar -----------------------------------------------------------------

def test_newstate_survives_its_trailing_space():
    line = classify("NEWSTATE IRS \r")
    assert line.kind == NOTIFICATION and line.fields == {"state": "IRS"}


def test_setter_echo_yields_the_value_alone():
    assert classify("MYCALL now K7CALL").kind == ECHO
    assert classify("MYCALL now K7CALL").fields["value"] == "K7CALL"
    # DISCONNECT answers with an upper-case NOW, which is why the token is
    # matched without regard to case.
    assert classify("DISCONNECT NOW TRUE").fields["value"] == "TRUE"
    assert classify("DISCONNECT IGNORED").fields["value"] == "IGNORED"


def test_connected_names_the_remote_station():
    assert classify("CONNECTED W1ABC 500").fields == {"peer": "W1ABC", "bw": 500}


def test_fault_is_its_own_kind():
    line = classify("FAULT CMD FROBNICATE not recoginized")
    assert line.kind == FAULT
    assert line.fields["text"] == "CMD FROBNICATE not recoginized"


def test_unrecognized_and_malformed_lines_are_findings():
    assert classify("XYZZY plugh").kind == UNKNOWN
    assert classify("CONNECTED W1ABC").kind == UNKNOWN     # bandwidth missing
    assert classify("").kind == UNKNOWN


def test_ping_notification_splits_the_callsign_pair():
    f = classify("PING N7CAII>K6CALL 10 95").fields
    assert f == {"caller": "N7CAII", "target": "K6CALL", "snr": 10, "quality": 95}


def test_bandwidth_maps_down_never_up():
    """ARDOP has nothing as wide as 2300, and a station may not use more channel
    than it was configured for."""
    assert _arqbw(ModemConfig(name="m", cmd_port=1, data_port=2,
                              bandwidth="2300")) == "2000MAX"
    assert _arqbw(ModemConfig(name="m", cmd_port=1, data_port=2,
                              bandwidth="500")) == "500MAX"


# -- attach ------------------------------------------------------------------

def test_attach_initializes_then_replays_desired_state(tnc, lk):
    """LISTEN comes last on a reattach: a modem answering calls before MYCALL
    has landed answers them as somebody else."""
    lk.configure("N0CAL")
    lk.set_listen(True)
    lk.attach()
    assert tnc.commands == ["INITIALIZE", "PROTOCOLMODE ARQ", "ARQBW 2000MAX",
                            "MYCALL N0CAL", "LISTEN TRUE"]


def test_data_port_follows_the_command_port_when_unset(tr):
    cfg = ModemConfig(name="a", dialect="ardop", cmd_port=8515, data_port=0)
    assert ArdopClient(cfg, "127.0.0.1", tr).data_port == 8516


def test_open_link_picks_the_ardop_dialect(tnc, tr):
    cfg = ModemConfig(name="a", dialect="ardop", cmd_port=tnc.cmd_port,
                      data_port=tnc.data_port)
    lk = open_link(cfg, "127.0.0.1", tr)
    assert isinstance(lk, ArdopLink)
    lk.close()


def test_the_link_surface_is_complete():
    """The seam is a Protocol, so nothing checks this at runtime and a missing
    method would first be noticed by a scenario halfway through a session."""
    for name in ("configure", "attach", "close", "set_listen", "connect",
                 "disconnect", "abort", "send", "recv", "peek", "wait_for",
                 "bump_epoch", "subscribe", "set_transcript", "stale",
                 "request_version", "attach_failures", "attached", "connected",
                 "epoch", "queue_bytes", "version", "transcript"):
        assert hasattr(ArdopLink, name), name


# -- reply correlation -------------------------------------------------------

def test_status_between_a_command_and_its_echo_does_not_desync(tnc, client):
    """The failure this dialect invites: a reader that takes the next line finds
    a NEWSTATE where MYCALL's echo should be, and every later reply is one
    command out for the rest of the session."""
    def script(cmd):
        if cmd.startswith("MYCALL"):
            return ["NEWSTATE IRS ", "BUSY TRUE", "PTT TRUE",
                    "MYCALL now N0CAL", "PTT FALSE"]
        return None

    tnc.on_command = script
    client.attach()
    assert client.command("MYCALL N0CAL").fields["value"] == "N0CAL"
    assert client.command("ARQBW 500MAX").name == "ARQBW"
    assert client.request_version() == tnc.version


def test_a_query_is_answered_by_its_own_asynchronous_verb(tnc, client):
    """BUFFER is both a query reply and an unsolicited report, and they carry
    the same number — correlating on the verb finds either."""
    tnc.on_command = lambda cmd: "BUFFER 512" if cmd == "BUFFER" else None
    client.attach()
    assert client.command("BUFFER").fields["n"] == 512
    assert client.buffer_bytes == 512


def test_no_reply_times_out_without_consuming_a_later_one(tnc, client):
    tnc.on_command = lambda cmd: "" if cmd == "BREAK" else None
    client.attach()
    with pytest.raises(TimeoutError):
        client.command("BREAK", timeout=0.2)
    assert client.command("ABORT").name == "ABORT"


# -- faults ------------------------------------------------------------------

def test_fault_reaches_the_caller(tnc, client):
    client.attach()
    with pytest.raises(ArdopFault) as exc:
        client.command("FROBNICATE")
    assert "not recoginized" in str(exc.value)


def test_a_fault_provoked_by_a_setter_is_raised_not_swallowed(tnc, client):
    tnc.on_command = lambda cmd: ("FAULT Syntax Err: MYCALL 12"
                                  if cmd.startswith("MYCALL") else None)
    client.attach()
    with pytest.raises(ArdopFault):
        client.set_mycall("12")


def test_an_unsolicited_fault_becomes_an_error_event(tnc, lk, tr):
    lk.attach()
    q = _events(lk)
    tnc.notify("FAULT ARQ call rejected")
    ev = _wait(q, ERROR)
    assert ev.fields["detail"] == "ARQ call rejected"
    errors = [r for r in read_transcript(tr.path)[0] if r.kind == "error"]
    assert any("ARQ call rejected" in r.fields.get("text", "") for r in errors)


def test_a_fault_during_replay_is_recorded_and_attach_survives(tnc, tr):
    tnc.on_command = lambda cmd: ("FAULT Syntax Err: ARQBW 2000MAX"
                                  if cmd.startswith("ARQBW") else None)
    c, cfg = make_client(tnc, tr)
    lk = ArdopLink(c, cfg)
    try:
        lk.configure("N0CAL")
        lk.attach()
        assert lk.attached
        assert tnc.wait_command("MYCALL")     # replay carried on past the fault
    finally:
        lk.close()


# -- session lifecycle -------------------------------------------------------

def test_connect_and_disconnect_lifecycle(tnc, lk):
    lk.configure("N0CAL")
    lk.attach()
    q = _events(lk)
    lk.connect("W1ABC")
    assert tnc.wait_command("ARQCALL W1ABC 10")
    tnc.answer_call("W1ABC", 500)
    ev = _wait(q, CONNECTED)
    assert ev.peer == "W1ABC" and ev.fields["bw"] == 500
    assert lk.connected
    lk.disconnect()
    assert _wait(q, DISCONNECTED) is not None
    assert not lk.connected


def test_ptt_and_busy_are_published(tnc, lk):
    lk.attach()
    q = _events(lk)
    tnc.notify("PTT TRUE")
    assert _wait(q, PTT).fields["on"] is True
    tnc.notify("BUSY FALSE")
    assert _wait(q, BUSY).fields["on"] is False


def test_disconnect_while_idle_is_ignored_not_faulted(tnc, client):
    client.attach()
    assert client.disconnect().fields["value"] == "IGNORED"


# -- data --------------------------------------------------------------------

def test_data_round_trip_through_the_counted_blocks(tnc, lk):
    lk.attach()
    lk.send(b"hello from the host")
    assert tnc.wait_data(19)
    assert bytes(tnc.received) == b"hello from the host"
    assert tnc.blocks == [b"hello from the host"]     # no tag host to modem
    tnc.send_data(b"ARQ", b"and back again")
    assert lk.recv(14, timeout=2.0) == b"and back again"


def test_a_block_larger_than_the_count_goes_as_several(tnc, lk):
    lk.attach()
    blob = bytes(range(256)) * 300                   # 76800 bytes
    lk.send(blob)
    assert tnc.wait_data(len(blob))
    assert len(tnc.blocks) == 2 and bytes(tnc.received) == blob


def test_blocks_split_across_reads_are_reassembled(tnc, lk):
    """The count arrives whenever TCP feels like delivering it, so a reader that
    assumed one block per segment would lose framing on a busy link."""
    lk.attach()
    tnc.wait_attached()
    block = b"ARQ" + b"x" * 40
    framed = len(block).to_bytes(2, "big") + block
    tnc._data_conn.sendall(framed[:1])
    tnc._data_conn.sendall(framed[1:20])
    tnc._data_conn.sendall(framed[20:])
    assert lk.recv(40, timeout=2.0) == b"x" * 40


def test_fec_data_is_delivered_and_uncorrected_data_is_not(tnc, lk, tr):
    lk.attach()
    tnc.send_data(b"ERR", b"garbled")
    tnc.send_data(b"FEC", b"broadcast")
    assert lk.recv(9, timeout=2.0) == b"broadcast"
    notes = [r for r in read_transcript(tr.path)[0] if r.kind == "note"]
    assert any(r.fields.get("text") == "uncorrected data" for r in notes)


def test_an_unknown_tag_is_a_conformance_finding(tnc, lk, tr):
    lk.attach()
    tnc.send_data(b"ZZZ", b"whatever")
    tnc.send_data(b"ARQ", b"ok")
    assert lk.recv(2, timeout=2.0) == b"ok"          # the good block still lands
    confs = [r for r in read_transcript(tr.path)[0] if r.kind == "conf"]
    assert any(r.fields.get("check") == "data_tag" for r in confs)


def test_queue_depth_follows_the_buffer_reports(tnc, lk):
    lk.attach()
    q = _events(lk)
    lk.send(b"y" * 64)
    ev = _wait(q, link_mod.BUFFER)
    assert ev.fields["queue_bytes"] == 64
    assert lk.telemetry() == {"queue_bytes": 64}


# -- teardown ----------------------------------------------------------------

def test_a_dead_socket_detaches_the_client(tnc, tr):
    detached = threading.Event()
    c, _ = make_client(tnc, tr)
    c.on_detach = lambda _: detached.set()
    c.attach()
    tnc.wait_attached()
    tnc.close()
    assert detached.wait(2.0)
    assert not c.attached
