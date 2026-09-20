# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import threading
import time

import pytest

from hfhost import cbor, hostapi
from hfhost.client import EpochFenced, NotAttached
from hfhost.config import ModemConfig
from hfhost.hostapi import (CONNECT, ERROR, HELLO, Incompatible,
                            LISTEN, PROTO, SEND, SET_IDENTITY, ST_CONNECTED,
                            ST_DISCONNECTED, STATE_CHANGED, HostApiClient,
                            pack, unpack)
from hfhost.testing.fakehostapi import FakeHostApiModem
from hfhost.transcript import Transcript, read as read_transcript


@pytest.fixture
def modem():
    m = FakeHostApiModem()
    yield m
    m.close()


def _client(modem, **kw):
    cfg = ModemConfig(name="fake", cmd_port=modem.port, data_port=0)
    return HostApiClient(cfg, **kw)


@pytest.fixture
def client(modem):
    c = _client(modem)
    c.attach()
    yield c
    c.close()


# -- framing and the message registry ------------------------------------


def test_frame_is_length_prefixed_cbor():
    frame = pack({"m": CONNECT, "peer_id": "K7ABC"})
    assert int.from_bytes(frame[:4], "big") == len(frame) - 4
    assert unpack(frame[4:]) == {"m": CONNECT, "peer_id": "K7ABC"}


def test_unknown_field_keys_are_ignored_not_fatal():
    body = cbor.encode({0: STATE_CHANGED, hostapi.KEY["state"]: ST_CONNECTED,
                        999: "from a newer modem"})
    assert unpack(body) == {"m": STATE_CHANGED, "state": ST_CONNECTED}


def test_sending_an_unregistered_field_is_a_bug_here():
    """Must-ignore protects us from a newer peer; it must not silently swallow
    our own typo, because we only ever send fields we know."""
    with pytest.raises(KeyError):
        pack({"m": CONNECT, "peeer_id": "K7ABC"})


def test_oversize_frame_refused_before_the_socket():
    with pytest.raises(hostapi.ModemError):
        pack({"m": SEND, "data": b"\x00" * (hostapi.MAX_FRAME + 1)})


# -- handshake -----------------------------------------------------------


def test_attach_completes_the_hello_exchange(modem, client):
    assert client.attached
    assert client.hello["modem"] == "fake/1.0"
    assert client.hello["proto"] == PROTO
    assert client.version == "fake/1.0"
    sent = modem.commands(HELLO)
    assert len(sent) == 1 and sent[0]["client"] == "creance"


def test_major_version_mismatch_refuses_the_attach():
    m = FakeHostApiModem(proto="2.0")
    try:
        c = _client(m)
        with pytest.raises(Incompatible):
            c.attach()
        assert not c.attached
        assert c.attach_failures == 1
    finally:
        m.close()


def test_attach_failure_leaves_nothing_open():
    cfg = ModemConfig(name="nobody", cmd_port=1, data_port=0)
    c = HostApiClient(cfg)
    with pytest.raises(hostapi.AttachError):
        c.attach(timeout=0.5)
    assert not c.attached and c._sock is None


def test_configuring_before_attach_records_intent(modem):
    """Desired state may be set before the modem's process is even up; it is
    replayed on attach. Only a raw send() demands an open socket."""
    cfg = ModemConfig(name="fake", cmd_port=modem.port, data_port=0)
    c = HostApiClient(cfg)
    c.set_identity("K7CRE")
    c.set_listen(True)
    with pytest.raises(NotAttached):
        c.send(hostapi.CONNECT, peer_id="K7ABC")
    c.attach()
    _until(lambda: modem.identity == "K7CRE" and modem.listening)
    c.close()


# -- state tracking ------------------------------------------------------


def test_state_changed_drives_connected(modem, client):
    assert not client.connected
    modem.state(ST_CONNECTED, peer_id="K7ABC")
    _until(lambda: client.connected)
    assert client.peer == "K7ABC"
    modem.state(ST_DISCONNECTED, reason=hostapi.RS_REMOTE)
    _until(lambda: not client.connected)
    assert client.last_reason == hostapi.RS_REMOTE


def test_link_stats_updates_queue_depth(modem, client):
    modem.emit(hostapi.LINK_STATS, gear="floor", rung=2, queue_bytes=4096,
               throughput_bps=310.0, snr3k_db=6.5)
    _until(lambda: client.queue_bytes == 4096)
    assert client.stats["gear"] == "floor"
    assert client.stats["snr3k_db"] == 6.5


def test_capabilities_event_is_retained(modem, client):
    modem.emit(hostapi.CAPABILITIES, peer_id="K7ABC", peer_capabilities=0x1234,
               peer_profiles=[1, 2, 3])
    _until(lambda: client.caps.get("peer_capabilities") == 0x1234)
    assert client.caps["peer_profiles"] == [1, 2, 3]


def test_errors_are_collected_not_raised(modem, client):
    modem.emit(ERROR, code=hostapi.ERR_BAD_STATE, detail="not connected", ref=7)
    _until(lambda: client.errors)
    assert client.errors[0]["detail"] == "not connected"


# -- must-ignore over the wire -------------------------------------------


def test_unknown_message_type_does_not_kill_the_reader(modem, client):
    modem.emit_unknown_type()
    modem.emit_unknown_field()
    modem.state(ST_CONNECTED, peer_id="K7ABC")
    _until(lambda: client.connected)
    assert client.attached


def test_undecodable_frame_is_a_finding_not_a_crash(modem, client):
    modem.emit_raw(b"\xff\xff\xff")
    modem.state(ST_CONNECTED, peer_id="K7ABC")
    _until(lambda: client.connected)
    assert any("undecodable" in (e.get("detail") or "") for e in client.errors)


# -- data plane ----------------------------------------------------------


def test_send_and_receive_round_trip():
    m = FakeHostApiModem(echo=True)
    try:
        c = _client(m)
        c.attach()
        c.send_data(b"hello world")
        assert c.read_data(timeout=2.0) == b"hello world"
        c.close()
    finally:
        m.close()


def test_read_data_refuses_a_short_count(client):
    """A caller that asked for n is framing something; handing back fewer bytes
    would lose that framing silently, so nothing is consumed at all."""
    assert client.read_data(16, timeout=0.05) == b""


def test_partial_buffer_is_left_intact(modem, client):
    modem.send_data(b"1234")
    _until(lambda: client._rx)
    assert client.read_data(8, timeout=0.1) == b""
    assert client.read_data(4, timeout=0.5) == b"1234"


def test_peek_accumulate_does_not_consume(modem, client):
    modem.send_data(b"CRN1rest")
    assert client.peek_accumulate(4, timeout=2.0) == b"CRN1"
    assert client.read_data(8, timeout=1.0) == b"CRN1rest"


def test_peek_accumulate_waits_for_all_of_it(modem, client):
    def later():
        time.sleep(0.05)
        modem.send_data(b"CR")
        time.sleep(0.05)
        modem.send_data(b"N1")
    threading.Thread(target=later, daemon=True).start()
    assert client.peek_accumulate(4, timeout=2.0) == b"CRN1"


def test_epoch_bump_fences_a_stale_read(modem, client):
    modem.send_data(b"from the last session")
    _until(lambda: client._rx)
    stranded = client.bump_epoch()
    assert stranded == b"from the last session"
    assert client.read_data(16, timeout=0.05) == b""


def test_reading_for_a_fenced_epoch_raises(client):
    epoch = client.epoch
    client.bump_epoch()
    with pytest.raises(EpochFenced):
        client.read_data(16, timeout=0.05, epoch=epoch)


def test_data_after_close_is_inert(modem, client):
    client.close()
    modem.send_data(b"too late")
    assert client.read_data(16, timeout=0.05) == b""


# -- desired-state replay ------------------------------------------------


def test_config_is_replayed_on_reattach(modem):
    c = _client(modem)
    c.attach()
    c.set_identity("K7CRE")
    c.set_profile(hostapi.PF_AMATEUR)
    c.subscribe_events([hostapi.LINK_STATS], stats_period=1.0)
    c.set_listen(True)
    # The modem must have *processed* the first session before we mark the
    # boundary, or its own LISTEN lands after the mark and is mistaken for the
    # replay we are trying to observe.
    _until(lambda: modem.listening)
    c.close()

    # Wait for the modem to actually accept the reconnect before reading its
    # log: clearing on a timer races the old session's teardown.
    seen = modem.accepted
    idx = len(modem.log)
    c.attach()
    _until(lambda: modem.accepted > seen, timeout=5.0)
    _until(lambda: LISTEN in [m["m"] for m in modem.log[idx:]], timeout=5.0)
    kinds = [m["m"] for m in modem.log[idx:]]
    assert SET_IDENTITY in kinds and LISTEN in kinds
    assert modem.identity == "K7CRE"
    assert modem.profile == hostapi.PF_AMATEUR
    assert modem.subscribed == [hostapi.LINK_STATS]
    assert modem.listening is True
    c.close()


def test_identity_required_is_surfaced():
    m = FakeHostApiModem(require_identity=True)
    try:
        c = _client(m)
        c.attach()
        assert c.hello["identity_required"] is True
        c.connect("K7ABC")
        _until(lambda: c.errors)
        assert c.errors[0]["code"] == hostapi.ERR_NO_IDENTITY
        c.close()
    finally:
        m.close()


# -- liveness ------------------------------------------------------------


def test_modem_disappearing_detaches_the_client(modem, client):
    modem.drop()
    _until(lambda: not client.attached, timeout=3.0)


def test_a_quiet_modem_does_not_detach_the_client(modem):
    """Silence is not death. A link with no traffic outlives the attach
    deadline, and the client stays usable across the quiet stretch."""
    c = _client(modem)
    c.attach(timeout=0.3)
    try:
        time.sleep(0.9)                       # three attach deadlines of nothing
        assert c.attached, "an idle modem detached the client"
        modem.emit(hostapi.LINK_STATS, gear="floor", rung=1, queue_bytes=7,
                   throughput_bps=1.0)
        _until(lambda: c.queue_bytes == 7)
    finally:
        c.close()


def test_subscribe_filters_by_type(modem, client):
    sub = client.subscribe(STATE_CHANGED)
    modem.emit(hostapi.LINK_STATS, gear="floor", rung=1, queue_bytes=0,
               throughput_bps=1.0)
    modem.state(ST_CONNECTED, peer_id="K7ABC")
    got = sub.get(timeout=2.0)
    assert got["m"] == STATE_CHANGED
    client.unsubscribe(sub)


# -- transcript ----------------------------------------------------------


def test_traffic_is_transcribed_with_payloads_summarised(modem, tmp_path):
    path = tmp_path / "t.jsonl"
    t = Transcript(str(path), "sid")
    c = _client(modem, transcript=t)
    c.attach()
    c.send_data(b"\x00" * 4096)
    modem.state(ST_CONNECTED, peer_id="K7ABC")
    _until(lambda: c.connected)
    c.close()
    t.close()

    records, _ = read_transcript(str(path))
    blob = "\n".join(str(r.fields) for r in records)
    assert "Hello" in blob and "StateChanged" in blob
    assert "<4096 bytes>" in blob          # summarised, not a second copy
    assert "\\x00" not in blob


def _until(pred, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached within the timeout")
