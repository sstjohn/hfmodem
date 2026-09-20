# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ModemClient against the scriptable fake modem: atomic attach, pub/sub,
desired-state replay, epoch fencing, cancellation, staleness."""

import queue
import threading
import time

import pytest

from hfhost.client import AttachError, EpochFenced, ModemClient, NotAttached
from hfhost.config import ModemConfig, Quirks
from hfhost.transcript import Transcript, read as read_transcript
from hfhost import wire

from hfhost.testing.fakemodem import FakeModem


@pytest.fixture
def fake():
    fm = FakeModem().start()
    yield fm
    fm.stop()


@pytest.fixture
def tr(tmp_path):
    t = Transcript(str(tmp_path / "daemon.jsonl"), "sid-daemon")
    yield t
    t.close()


def wait_until(pred, timeout=2.0, step=0.005):
    """Poll a condition to a deadline.

    The barrier these tests want after injecting a notification is "the client
    has taken it in", and the observable for that is the client's own state.
    They used to wait on the LINE instead, via client.wait_for -- which
    subscribes when it is called and so cannot see anything that arrived first.
    The modem answers from its own thread, so under load the line beat the test
    to the subscription and a settled client read as a deaf one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


def _await(q, predicate, timeout=2.0):
    """Scan an already-open subscription for a matching line — for a transient
    no state survives, where wait_until has nothing to poll."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            item = q.get(timeout=remaining)
        except queue.Empty:
            return None
        if isinstance(item, wire.Line) and predicate(item):
            return item


def make_client(fake, tr, **kw):
    cfg = ModemConfig(name="fake", cmd_port=fake.cmd_port,
                      data_port=fake.data_port,
                      quirks=kw.pop("quirks", Quirks()))
    kw.setdefault("attach_timeout_s", 2.0)
    return ModemClient(cfg, "127.0.0.1", tr, **kw)


@pytest.fixture
def client(fake, tr):
    c = make_client(fake, tr)
    yield c
    c.close()


# -- attach ------------------------------------------------------------------

def test_attach_replays_desired_state(fake, client):
    client.set_mycall("N0CAL")
    client.set_bandwidth("2300")
    client.set_compression("OFF")
    client.set_listen(True)
    client.attach()
    assert client.attached
    assert fake.wait_attached()
    assert fake.session_log[0] == [
        "MYCALL N0CAL", "BW2300", "COMPRESSION OFF", "LISTEN ON"]


def test_attach_failure_leaves_no_half_attach(tr):
    fm = FakeModem(data_listener=False).start()
    try:
        c = make_client(fm, tr, attach_timeout_s=0.5)
        with pytest.raises(AttachError):
            c.attach()
        assert not c.attached
        assert c._cmd_sock is None and c._data_sock is None
        assert c.attach_failures == 1
        assert fm.attach_count == 0
        c.close()
    finally:
        fm.stop()


def test_send_data_requires_attach(client):
    with pytest.raises(NotAttached):
        client.send_data(b"nope")


# -- command / pub-sub -------------------------------------------------------

def test_command_exactly_one_reply_with_interleaved_notifications(fake, tr):
    def script(cmd):
        if cmd.startswith("CONNECT"):
            return ["PENDING", "BUSY ON", "OK"]
        return None
    fake.on_command = script
    c = make_client(fake, tr)
    c.attach()
    try:
        line = c.command("CONNECT N0CAL K7XYZ")
        assert line.kind == wire.REPLY_OK
        texts = [r.fields.get("text") for r in read_transcript(tr.path)[0]]
        assert "BUSY ON" in texts                # notification still processed
        assert c.command("FROBNICATE").kind == wire.REPLY_WRONG
        with pytest.raises(TimeoutError):
            fake.on_command = lambda cmd: ""     # silence
            c.command("ABORT", timeout=0.2)
    finally:
        c.close()


def test_request_version(fake, client):
    client.attach()
    assert client.request_version() == "1.0-fake"
    assert client.version == "1.0-fake"


def test_notification_state_tracking(fake, client):
    client.attach()
    fake.wait_attached()
    fake.notify("CONNECTED N0CAL K7XYZ 2300")
    assert wait_until(lambda: client.connected)
    assert client.last_connected == {"src": "N0CAL", "dst": "K7XYZ", "bw": "2300"}
    fake.notify("BUFFER 42")
    assert wait_until(lambda: client.buffer_bytes == 42)
    fake.notify("DISCONNECTED")
    assert wait_until(lambda: not client.connected)
    assert client.buffer_bytes == 0


def test_buffer_bytes_tracks_sends_and_buffer_lines(fake, client):
    client.attach()
    fake.wait_attached()
    client.send_data(b"x" * 100)
    assert client.buffer_bytes == 100            # local increment
    client.send_data(b"y" * 50)
    assert client.buffer_bytes == 150
    fake.notify("BUFFER 37")                     # authoritative overrides
    assert wait_until(lambda: client.buffer_bytes == 37)
    assert fake.wait_data(150)
    assert bytes(fake.received_data) == b"x" * 100 + b"y" * 50


def test_unknown_line_is_conformance_event(fake, client, tr):
    client.attach()
    fake.wait_attached()
    fake.notify("XYZZY plugh")
    assert wait_until(lambda: any(r.kind == "conf"
                                 for r in read_transcript(tr.path)[0]))
    confs = [r for r in read_transcript(tr.path)[0] if r.kind == "conf"]
    assert confs and confs[0].fields["verdict"] == "EXTRA"
    assert "XYZZY" in confs[0].fields["line"]


# -- epoch-guarded rx buffer -------------------------------------------------

def test_epoch_fencing(fake, client):
    client.attach()
    fake.wait_attached()
    epoch = client.epoch
    fake.send_data(b"abc")
    assert client.read_data(3, timeout=1.0, epoch=epoch) == b"abc"
    fake.send_data(b"late")
    assert client.peek_accumulate(4, timeout=1.0, epoch=epoch) == b"late"
    leftover = client.bump_epoch()
    assert leftover == b"late"                   # undrained bytes returned
    with pytest.raises(EpochFenced):
        client.read_data(1, timeout=0.2, epoch=epoch)
    with pytest.raises(EpochFenced):
        client.peek_accumulate(1, timeout=0.2, epoch=epoch)


def test_bump_epoch_wakes_blocked_read(fake, client):
    client.attach()
    fake.wait_attached()
    epoch = client.epoch
    result = {}

    def blocked():
        try:
            client.read_data(10, timeout=5.0, epoch=epoch)
        except EpochFenced:
            result["fenced"] = True

    th = threading.Thread(target=blocked, daemon=True)
    th.start()
    time.sleep(0.05)
    client.bump_epoch()
    th.join(timeout=1.0)
    assert result.get("fenced")


@pytest.mark.realtime
def test_cancel_wakes_blocked_read_promptly(fake, client):
    client.attach()
    cancel = threading.Event()
    out = {}

    def blocked():
        out["data"] = client.read_data(10, timeout=5.0, cancel=cancel)

    th = threading.Thread(target=blocked, daemon=True)
    th.start()
    time.sleep(0.05)
    t0 = time.monotonic()
    cancel.set()
    th.join(timeout=1.0)
    assert not th.is_alive()
    assert time.monotonic() - t0 < 0.5
    assert out["data"] == b""


def test_peek_accumulate_short_on_timeout(fake, client):
    client.attach()
    fake.wait_attached()
    fake.send_data(b"CR")
    client.peek_accumulate(2, timeout=1.0)       # wait for arrival
    assert client.peek_accumulate(4, timeout=0.2) == b"CR"   # short, no consume
    fake.send_data(b"N1")
    assert client.peek_accumulate(4, timeout=1.0) == b"CRN1"
    assert client.read_data(4, timeout=1.0) == b"CRN1"       # still consumable


# -- transcript swapping -----------------------------------------------------

def test_transcript_swap(fake, client, tr, tmp_path):
    client.attach()
    fake.wait_attached()
    session_t = Transcript(str(tmp_path / "session.jsonl"), "sid-1")
    client.set_transcript(session_t)
    fake.notify("BUFFER 7")
    assert wait_until(lambda: client.buffer_bytes == 7)
    fake.send_data(b"hello")
    client.read_data(5, timeout=1.0)
    client.set_transcript(None)
    fake.notify("BUFFER 8")
    assert wait_until(lambda: client.buffer_bytes == 8)
    session_t.close()

    session_kinds = {(r.kind, r.fields.get("text")) for r in
                     read_transcript(session_t.path)[0]}
    assert ("cmd_rx", "BUFFER 7") in session_kinds
    assert any(k == "data_rx" for k, _ in session_kinds)
    daemon_texts = [r.fields.get("text") for r in read_transcript(tr.path)[0]]
    assert "BUFFER 8" in daemon_texts
    assert "BUFFER 7" not in daemon_texts


def test_closed_session_transcript_does_not_deafen_the_client(fake, client,
                                                              tmp_path):
    """The responder closes the session transcript before clearing it from the
    client; late cmd lines and data must survive that window."""
    client.attach()
    fake.wait_attached()
    session_t = Transcript(str(tmp_path / "session.jsonl"), "sid-1")
    client.set_transcript(session_t)
    session_t.close()
    fake.notify("CONNECTED N0CAL K7XYZ 2300")
    assert wait_until(lambda: client.connected)
    fake.send_data(b"late echo")
    assert client.read_data(9, timeout=1.0) == b"late echo"
    assert client.attached
    alive = {th.name for th in threading.enumerate()}
    assert {"fake-cmd-rx", "fake-data-rx"} <= alive


def test_reader_exception_always_detaches(fake, tr):
    """A dead reader thread with attached still True is the deaf-modem state:
    nothing would ever notice, so the reader must always detach on its way
    out — that detach is what puts the responder's tick back on the case."""
    detached = threading.Event()
    c = make_client(fake, tr, on_detach=lambda _: detached.set())
    c.attach()
    fake.wait_attached()
    try:
        def boom(text):
            raise RuntimeError("reader blew up")
        c._on_cmd_line = boom
        fake.notify("BUFFER 1")
        assert detached.wait(2.0)
        assert not c.attached
    finally:
        c.close()


def test_read_data_timeout_does_not_consume_a_partial_count(fake, client):
    client.attach()
    fake.wait_attached()
    fake.send_data(b"abc")
    client.peek_accumulate(3, timeout=1.0)
    assert client.read_data(6, timeout=0.2) == b""      # no short read...
    fake.send_data(b"def")
    assert client.read_data(6, timeout=1.0) == b"abcdef"   # ...and none lost


def test_concurrent_attach_waits_for_replay(fake, tr):
    """Readiness must imply replay-complete, or a caller CONNECTs before
    MYCALL/BW have landed."""
    gate = threading.Event()

    def script(cmd):
        if cmd.startswith("MYCALL"):
            gate.wait(3.0)
        return None

    fake.on_command = script
    c = make_client(fake, tr, attach_timeout_s=5.0)
    c.set_mycall("N0CAL")
    c.set_bandwidth("2300")
    c.set_listen(True)
    seen: dict = {}
    first = threading.Thread(target=c.attach, daemon=True)
    first.start()
    assert fake.wait_command("MYCALL")

    def second():
        c.attach()
        seen["log"] = list(fake.session_log[0])

    later = threading.Thread(target=second, daemon=True)
    later.start()
    try:
        time.sleep(0.1)
        assert "log" not in seen             # blocked while replay is in flight
        gate.set()
        first.join(3.0)
        later.join(3.0)
        assert seen["log"] == ["MYCALL N0CAL", "BW2300", "LISTEN ON"]
    finally:
        gate.set()
        c.close()


# -- fidelity to the real servers --------------------------------------------

def test_buffer_notifications_drain_the_local_estimate(tr):
    fm = FakeModem(buffer_notifications=True, echo_delay_s=0.05).start()
    try:
        c = make_client(fm, tr)
        c.attach()
        fm.wait_attached()
        # Subscribed BEFORE the send, because wait_for subscribes when it is
        # called and only ever sees what arrives after that. The modem answers
        # this write from its own thread, so under load the BUFFER 1000 landed
        # while the test was still between the two statements and was simply
        # gone -- a lost notification that reads as a modem that never sent one.
        q = c.subscribe()
        try:
            c.send_data(b"x" * 1000)
            assert _await(q, lambda l: l.name == "BUFFER" and l.fields["n"] == 1000)
            assert _await(q, lambda l: l.name == "BUFFER" and l.fields["n"] == 0)
        finally:
            c.unsubscribe(q)
        assert c.buffer_bytes == 0           # without this, TX pacing stalls
        c.close()
    finally:
        fm.stop()


def test_heartbeat_keeps_the_client_fresh(tr):
    fm = FakeModem(iamalive_s=0.05).start()
    try:
        c = make_client(fm, tr, quirks=Quirks(iamalive_s=0.05))
        c.attach()
        # The one wait_for left in this file, and safe because the beat repeats
        # every 50 ms: missing the first one costs nothing, the window holds
        # forty more. Elsewhere the awaited line arrives once and wait_until is
        # what to reach for.
        assert c.wait_for(lambda l: l.name == "IAMALIVE", 2.0) is not None
        assert not c.stale()
        c.close()
    finally:
        fm.stop()


def test_silent_modem_goes_stale(fake, tr):
    c = make_client(fake, tr, quirks=Quirks(iamalive_s=0.05))
    c.attach()
    try:
        time.sleep(0.25)                     # 3x the interval, no IAMALIVE
        assert c.stale()
    finally:
        c.close()


def test_disconnect_is_followed_by_disconnected(tr):
    fm = FakeModem(disconnected_on_disconnect=True).start()
    try:
        c = make_client(fm, tr)
        c.attach()
        fm.wait_attached()
        fm.notify("CONNECTED N0CAL K7XYZ 500")
        assert wait_until(lambda: c.connected)
        assert c.command("DISCONNECT").kind == wire.REPLY_OK
        deadline = time.monotonic() + 2.0
        while c.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not c.connected               # only DISCONNECTED clears this
        c.close()
    finally:
        fm.stop()


def test_pre_connect_writes_are_discarded(tr):
    fm = FakeModem(discard_before_connect=True, echo=True).start()
    try:
        c = make_client(fm, tr)
        c.attach()
        fm.wait_attached()
        c.send_data(b"too early")
        assert not fm.wait_data(1, timeout=0.3)
        assert bytes(fm.discarded_data) == b"too early"
        assert c.read_data(timeout=0.3) == b""
        fm.notify("CONNECTED N0CAL K7XYZ 500")
        assert wait_until(lambda: c.connected)
        c.send_data(b"in session")
        assert fm.wait_data(10)
        assert c.read_data(10, timeout=2.0) == b"in session"
        c.close()
    finally:
        fm.stop()


def test_late_echo_lands_after_teardown_and_is_fenced(tr):
    fm = FakeModem(echo=True, echo_delay_s=0.3,
                   disconnected_on_disconnect=True).start()
    try:
        c = make_client(fm, tr)
        c.attach()
        fm.wait_attached()
        fm.notify("CONNECTED N0CAL K7XYZ 500")
        assert wait_until(lambda: c.connected)
        epoch = c.epoch
        c.send_data(b"tail")
        c.command("DISCONNECT")
        c.bump_epoch()                       # session over; echo still airborne
        with pytest.raises(EpochFenced):
            c.read_data(4, timeout=1.0, epoch=epoch)
        assert c.read_data(4, timeout=1.0) == b"tail"
        c.close()
    finally:
        fm.stop()


# -- staleness ---------------------------------------------------------------

def test_iamalive_staleness(fake, tr):
    c = make_client(fake, tr, quirks=Quirks(iamalive_s=10.0))
    c.attach()
    try:
        fake.wait_attached()
        now = time.monotonic()
        assert not c.stale(now)
        assert c.stale(now + 31.0)               # 3x interval from attach
        fake.notify("IAMALIVE")
        assert wait_until(lambda: c.last_iamalive is not None)
        assert not c.stale(c.last_iamalive + 29.0)
        assert c.stale(c.last_iamalive + 31.0)
    finally:
        c.close()


def test_stale_disabled_without_interval(fake, tr):
    c = make_client(fake, tr, quirks=Quirks(iamalive_s=0.0))
    c.attach()
    try:
        assert not c.stale(time.monotonic() + 10_000)
    finally:
        c.close()
