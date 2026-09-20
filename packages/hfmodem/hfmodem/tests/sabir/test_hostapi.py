# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Host-interface conformance (HOST-API.md): the CBOR codec, the framed
message layer, the HostLink command/event protocol, and the telemetry the
structured interface surfaces that the VARA pipe discarded."""

import socket
import threading

import pytest

from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig
from hfmodem.sabir.host import cbor
from hfmodem.sabir.host import messages as M
from hfmodem.sabir.host.hostlink import HostClient, HostLink, HostLinkServer
from hfmodem.sabir.host.modem_core import SabirModem, ModemCore

_FULL = wire.capabilities(range(3, 8), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)


# -- CBOR codec ---------------------------------------------------------------
@pytest.mark.parametrize("obj,enc", [
    (0, b"\x00"), (23, b"\x17"), (24, b"\x18\x18"), (1000, b"\x19\x03\xe8"),
    (-1, b"\x20"), (-1000, b"\x39\x03\xe7"), (b"\x01\x02", b"\x42\x01\x02"),
    ("a", b"\x61\x61"), ([1, 2, 3], b"\x83\x01\x02\x03"),
    (False, b"\xf4"), (True, b"\xf5"), (None, b"\xf6"),
    (1.5, b"\xfb\x3f\xf8\x00\x00\x00\x00\x00\x00"),
    ({2: 0, 1: 0}, b"\xa2\x01\x00\x02\x00"),           # canonical: keys ascend
])
def test_cbor_canonical_vectors(obj, enc):
    assert cbor.encode(obj) == enc
    assert cbor.decode(enc) == obj


def test_cbor_roundtrip_nested():
    x = {0: 34, 30: "workhorse", 32: [-16.0, 3.25, None], 26: 247}
    assert cbor.decode(cbor.encode(x)) == x


def test_cbor_rejects_trailing_and_truncated():
    with pytest.raises(ValueError):
        cbor.decode(b"\x00\x00")                       # trailing byte
    with pytest.raises(ValueError):
        cbor.decode(b"\x18")                           # truncated argument


# -- message layer ------------------------------------------------------------
def test_message_frame_roundtrip():
    frame = M.encode({"m": M.LINK_STATS, "gear": "fast", "rung": 3,
                      "snr3k_db": 12.5, "group_snr_db": [10.0, 11.0]})
    assert len(frame) == int.from_bytes(frame[:4], "big") + 4
    msg = M.decode(frame[4:])
    assert msg["m"] == M.LINK_STATS and msg["gear"] == "fast" and msg["rung"] == 3


def test_message_must_ignore_unknown_key():
    body = cbor.encode({0: M.HELLO, 1: "1.0", 200: "from the future"})
    assert M.decode(body) == {"m": M.HELLO, "proto": "1.0"}


def test_message_unregistered_field_rejected_on_send():
    with pytest.raises(KeyError):
        M.encode({"m": M.HELLO, "nonsense": 1})


# -- HostLink protocol over a socketpair --------------------------------------
class _FakeModem(ModemCore):
    def __init__(self):
        self.calls = []
        self.obs = None

    def start(self, obs):
        self.obs = obs

    def stop(self):
        self.calls.append(("stop",))

    def set_identity(self, station_id):
        self.calls.append(("identity", station_id))

    def set_profile(self, p):
        self.calls.append(("profile", p))

    def set_listen(self, on):
        self.calls.append(("listen", on))

    def set_compression(self, mode):
        self.calls.append(("compress", mode))

    def connect(self, src, dst):
        self.calls.append(("connect", src, dst))

    def transmit(self, blob, msg_id=None, *, deflate=False):
        self.calls.append(("tx", bytes(blob)))
        return False

    def disconnect(self):
        self.calls.append(("disc",))

    def abort(self):
        self.calls.append(("abort",))

    def beacon(self, addressee=0):
        self.calls.append(("beacon", addressee))

    @property
    def connected(self):
        return False


@pytest.fixture
def linked():
    app, srv = socket.socketpair()
    modem = _FakeModem()
    link = HostLink(srv, modem)
    th = threading.Thread(target=link.run, daemon=True)
    th.start()
    cli = HostClient(app)
    yield cli, modem, link
    cli.close()
    th.join(timeout=2.0)


def test_hello_handshake(linked):
    cli, modem, _ = linked
    hello = cli.hello()
    assert hello["m"] == M.HELLO and hello["identity_required"] is True
    assert M.PF_AMATEUR in hello["profiles"]


def test_identity_guard_then_connect(linked):
    cli, modem, _ = linked
    cli.hello()
    cli.send({"m": M.CONNECT, "peer_id": "K6XYZ", "ref": 7})
    err = cli.recv()
    assert err["m"] == M.ERROR and err["code"] == M.ERR_NO_IDENTITY
    assert err["ref"] == 7
    cli.send({"m": M.SET_IDENTITY, "station_id": "w1aw"})
    cli.send({"m": M.SET_PROFILE, "profile": M.PF_AMATEUR})
    cli.send({"m": M.CONNECT, "peer_id": "k6xyz"})
    cli.send({"m": M.SEND, "data": b"hello", "id": 1})
    prog = cli.recv()                                  # barrier: SEND dispatched
    assert prog["m"] == M.SEND_PROGRESS and prog["id"] == 1 and prog["sent"] == 5
    assert ("identity", "W1AW") in modem.calls
    assert ("profile", M.PF_AMATEUR) in modem.calls
    assert ("connect", "W1AW", "K6XYZ") in modem.calls
    assert ("tx", b"hello") in modem.calls


def test_error_omits_ref_when_absent(linked):
    cli, modem, _ = linked
    cli.hello()
    cli.send({"m": M.CONNECT, "peer_id": "K6XYZ"})      # no ref, no identity
    err = cli.recv()
    assert err["m"] == M.ERROR and err["code"] == M.ERR_NO_IDENTITY
    assert "ref" not in err                             # absent, never null


def test_incompatible_major_version_refused(linked):
    cli, modem, _ = linked
    cli.send({"m": M.HELLO, "proto": "2.0", "client": "future"})
    assert cli.recv()["m"] == M.HELLO                  # modem still greets
    assert cli.recv()["m"] == M.ERROR                  # then refuses


def test_link_stats_gated_by_subscription(linked):
    cli, modem, link = linked
    cli.hello()
    cli._conn.settimeout(0.5)
    link.modem_link_stats({"gear": "fast", "rung": 3})
    with pytest.raises(socket.timeout):
        cli.recv()                                     # unsubscribed -> silent
    cli.send({"m": M.SUBSCRIBE, "events": [M.LINK_STATS]})
    cli.send({"m": M.SEND, "data": b"x", "id": 9})     # barrier
    assert cli.recv()["m"] == M.SEND_PROGRESS
    link.modem_link_stats({"gear": "fast", "rung": 3})
    ls = cli.recv()
    assert ls["m"] == M.LINK_STATS and ls["gear"] == "fast"


def test_server_accepts_and_greets():
    srv = HostLinkServer(lambda: _FakeModem(), port=0)
    srv.start_background()
    try:
        conn = socket.create_connection(("127.0.0.1", srv.port), timeout=2.0)
        cli = HostClient(conn)
        assert cli.hello()["m"] == M.HELLO
        cli.close()
    finally:
        srv.stop()


def test_beacon_command_and_peer_observed_gating(linked):
    cli, modem, link = linked
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "W1AW"})
    cli.send({"m": M.BEACON, "addressee": 0})
    cli.send({"m": M.SUBSCRIBE, "events": [M.PEER_OBSERVED]})
    cli.send({"m": M.SEND, "data": b"x", "id": 6})     # barrier
    assert cli.recv()["m"] == M.SEND_PROGRESS
    assert ("beacon", 0) in modem.calls
    link.modem_peer_observed({"peer_id": "K6XYZ", "profile": 1,
                              "capabilities": {"profiles": [3, 4, 5, 6, 7]}})
    po = cli.recv()
    assert po["m"] == M.PEER_OBSERVED and po["peer_id"] == "K6XYZ"
    assert po["capabilities"]["profiles"] == [3, 4, 5, 6, 7]


def test_oversize_frame_refused(linked):
    cli, modem, link = linked
    cli.hello()
    cli._conn.sendall((32 * 1024 * 1024).to_bytes(4, "big"))   # prefix > 16 MiB
    err = cli.recv()
    assert err["m"] == M.ERROR and err["code"] == M.ERR_MALFORMED
    assert cli.recv() is None                          # connection closed


def test_malformed_command_errors_without_crashing_reader(linked):
    cli, modem, link = linked
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "w1aw"})
    cli.send({"m": M.CONNECT, "ref": 3})               # missing peer_id
    err = cli.recv()
    assert err["m"] == M.ERROR and err["ref"] == 3
    cli.send({"m": M.SEND, "data": b"x", "id": 1})     # reader still alive
    assert cli.recv()["m"] == M.SEND_PROGRESS


def test_connect_station_id_overrides_identity(linked):
    cli, modem, link = linked
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "w1aw"})
    cli.send({"m": M.CONNECT, "peer_id": "k6xyz", "station_id": "w1aw-t"})
    cli.send({"m": M.SEND, "data": b"x", "id": 1})     # barrier
    assert cli.recv()["m"] == M.SEND_PROGRESS
    assert ("connect", "W1AW-T", "K6XYZ") in modem.calls


def test_capabilities_always_delivered(linked):
    cli, modem, link = linked
    cli.hello()
    link.modem_capabilities({"peer_id": "K6XYZ", "peer_capabilities": _FULL,
                             "peer_profiles": [3, 4, 5, 6, 7],
                             "usable": {"fastctl": True, "pback": True,
                                        "loading": True, "deflate": False}})
    cap = cli.recv()
    assert cap["m"] == M.CAPABILITIES and cap["peer_id"] == "K6XYZ"
    assert cap["usable"]["fastctl"] is True and cap["usable"]["deflate"] is False


# -- telemetry content from a real connected modem ----------------------------
class _StubAir:
    """Synchronous air: posts run inline, so a SabirModem drives its FSM
    deterministically with no threads or channel."""

    def __init__(self):
        self._t = 0.0

    def clock(self):
        self._t += 0.5
        return self._t

    def register(self, end):
        pass

    def post(self, fn):
        fn()


class _Capture:
    def __init__(self):
        self.caps = None
        self.stats = []
        self.ids = []
        self.state = []
        self.progress = []
        self.peers = []

    def modem_registered(self, text="LINK REGISTERED"):
        pass

    def modem_connected(self, src, dst, bw):
        self.state.append(("connected", dst))

    def modem_disconnected(self, reason=0):
        self.state.append(("disconnected", reason))

    def modem_ptt(self, on):
        pass

    def modem_buffer(self, n):
        pass

    def modem_data_received(self, blob):
        pass

    def modem_capabilities(self, image):
        self.caps = image

    def modem_link_stats(self, snap):
        self.stats.append(snap)

    def modem_id_sent(self, station_id, t):
        self.ids.append((station_id, t))

    def modem_send_progress(self, msg_id, delivered):
        self.progress.append((msg_id, delivered))

    def modem_peer_observed(self, image):
        self.peers.append(image)

    def modem_state_changed(self, state):
        self.state.append(state)


def _connected_modem():
    modem = SabirModem(_StubAir(), ArqConfig(callsign="ALICE"))
    cap = _Capture()
    modem.start(cap)
    modem.connect("ALICE", "BOB")
    modem.fsm.on_control(
        wire.Control.connect(modem.fsm.session, _FULL, "BOB", ack=True, destination="ALICE"))
    return modem, cap


def test_capability_image_reflects_negotiation():
    modem, cap = _connected_modem()
    assert modem.connected
    img = modem.capability_image()
    assert img["peer_id"] == "BOB" and img["peer_capabilities"] == _FULL
    assert img["peer_profiles"] == list(range(3, 8))
    assert img["usable"] == {"fastctl": True, "pback": True,
                             "loading": True, "deflate": True}
    assert cap.caps == img                             # emitted on connect


def test_link_snapshot_fields():
    modem, cap = _connected_modem()
    modem.transmit(b"a message worth compressing " * 4)
    snap = modem.link_snapshot()
    assert snap["gear"] in ("robust", "workhorse", "workhorse34", "fast", "max")
    assert snap["control_tier"] in (0, 1)
    assert len(snap["group_snr_db"]) == 8              # §5 [f?]: float or null
    assert all(x is None or isinstance(x, float) for x in snap["group_snr_db"])
    assert isinstance(snap["throughput_bps"], float)
    assert snap["queue_bytes"] >= 0
    assert 0.0 < snap["compression_ratio"] <= 1.5      # raw_tx > 0 -> present


def test_per_message_delivery_fires_on_full_ack():
    modem, cap = _connected_modem()
    modem.transmit(b"x" * 200, msg_id=42)
    f = modem.fsm._frame
    assert f is not None                               # a frame is in flight
    modem.after_step()
    assert cap.progress == []                          # not yet peer-ACKed
    modem.fsm.on_control(wire.Control(
        wire.ACK, modem.fsm.session, seq=f.seq,
        mask=wire.cw_mask(range(len(f.chunks))), aux=wire.ack_aux(None)))
    modem.after_step()
    assert (42, True) in cap.progress


def test_beacon_tx_to_peer_observed_end_to_end():
    """A SabirModem emits a presence beacon; a second modem decodes it off
    the floor waveform and surfaces PeerObserved with the advertised image."""
    import numpy as np

    a = SabirModem(_StubAir(), ArqConfig(callsign="ALICE"))
    b = SabirModem(_StubAir(), ArqConfig(callsign="BOB"))
    cap = _Capture()
    b.start(cap)
    a.beacon()
    b.link.on_air(np.concatenate(a.link.outbox))
    assert cap.peers and cap.peers[-1]["peer_id"] == "ALICE"
    assert cap.peers[-1]["capabilities"]["profiles"] == \
        list(range(3, 8))                            # default ceiling = max


def test_state_changed_on_listen_and_connect():
    modem = SabirModem(_StubAir(), ArqConfig(callsign="BOB"))
    cap = _Capture()
    modem.start(cap)
    modem.set_listen(True)
    assert "LISTENING" in cap.state                     # the missing lifecycle
    m2, cap2 = _connected_modem()
    assert "CONNECTING" in cap2.state                    # intermediate emitted


def test_disconnect_reason_mapped_from_cause():
    from hfmodem.sabir.arq.fsm import DISC_LOCAL, DISC_REMOTE
    m, cap = _connected_modem()
    m.fsm.on_control(wire.Control.ident(wire.DISC, m.fsm.session, "BOB"))
    assert ("disconnected", DISC_REMOTE) in cap.state    # peer hung up
    m2, cap2 = _connected_modem()
    m2.abort()
    assert ("disconnected", DISC_LOCAL) in cap2.state     # local abort


def test_state_changed_carries_and_omits_reason(linked):
    from hfmodem.sabir.arq.fsm import DISC_LINK_FAILED
    cli, modem, link = linked
    cli.hello()
    link.modem_disconnected(DISC_LINK_FAILED)
    ev = cli.recv()
    assert ev["state"] == M.ST_DISCONNECTED and ev["reason"] == DISC_LINK_FAILED
    link.modem_disconnected(0)                            # unset -> omitted
    ev2 = cli.recv()
    assert ev2["state"] == M.ST_DISCONNECTED and "reason" not in ev2


def test_queue_bytes_is_application_payload():
    modem, cap = _connected_modem()
    modem.transmit(b"x" * 100)                           # 104-byte record
    assert modem.link_snapshot()["queue_bytes"] == 100   # payload, not record


def test_after_step_emits_stats_and_id():
    modem, cap = _connected_modem()
    modem.after_step()
    assert cap.stats and cap.stats[-1]["gear"]         # a LinkStats went out
    modem.fsm.stats["ids"].append(123.0)               # FSM emitted an ID
    modem.after_step()
    assert cap.ids and cap.ids[-1][0] == "ALICE"


@pytest.mark.parametrize("body", [
    b"\xa2\x00\x00\x00\x01",  # duplicate key 0
    b"\xa1\xf4\x00",            # boolean aliases key 0 in Python
    b"\xc0\xa1\x00\x00",       # transparent tag would hide a type mismatch
    cbor.encode({0: True}),
    cbor.encode({0: -1}),
    cbor.encode({"m": 0}),
    cbor.encode({1: "1.0"}),
])
def test_ambiguous_host_envelopes_rejected(body):
    with pytest.raises(ValueError):
        M.decode(body)


def test_non_hello_first_frame_is_refused(linked):
    cli, _, _ = linked
    cli.send({"m": M.CONNECT, "proto": M.PROTO, "peer_id": "BOB"})
    assert cli.recv()["code"] == M.ERR_MALFORMED
    assert cli.recv() is None


def test_oversized_hello_refused(linked):
    cli, _, _ = linked
    cli._conn.sendall((32 * 1024 * 1024).to_bytes(4, "big"))
    assert cli.recv()["code"] == M.ERR_MALFORMED
    assert cli.recv() is None


@pytest.mark.parametrize("fields", [
    {"data": 10}, {"data": b"x", "deflate": "true"},
    {"data": b"x", "stream": 1}, {"data": b"x", "priority": 1},
    {"data": b"x", "deadline": 10}, {"data": b"x", "id": True},
])
def test_invalid_or_unimplemented_send_fields_do_not_transmit(linked, fields):
    cli, modem, _ = linked
    cli.hello()
    cli.send({"m": M.SEND, "ref": 7, **fields})
    assert cli.recv() == {"m": M.ERROR, "ref": 7, "code": M.ERR_MALFORMED}
    assert not any(call[0] == "tx" for call in modem.calls)


def test_invalid_identity_does_not_enable_connect(linked):
    cli, modem, _ = linked
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "X" * 13})
    assert cli.recv()["code"] == M.ERR_MALFORMED
    cli.send({"m": M.CONNECT, "peer_id": "BOB"})
    assert cli.recv()["code"] == M.ERR_NO_IDENTITY


def test_send_reports_actual_compression():
    modem, cap = _connected_modem()
    assert modem.transmit(b"compressible " * 100, deflate=True) is True
    assert modem.transmit(bytes(range(256)), deflate=True) is False
    modem.fsm.use_deflate = False
    assert modem.transmit(b"compressible " * 100, deflate=True) is False


def test_empty_record_is_acknowledged():
    modem, cap = _connected_modem()
    modem.transmit(b"", msg_id=42)
    f = modem.fsm._frame
    assert f is not None
    modem.fsm.on_control(wire.Control(
        wire.ACK, modem.fsm.session, seq=f.seq,
        mask=wire.cw_mask(range(len(f.chunks))), aux=wire.ack_aux(None)))
    modem.after_step()
    assert cap.progress == [(42, True)]



def test_queued_configurations_apply_in_command_order():
    class DeferredAir(_StubAir):
        def __init__(self):
            super().__init__()
            self.commands = []

        def post(self, fn):
            self.commands.append(fn)

    air = DeferredAir()
    modem = SabirModem(air)
    modem.submit(lambda: modem.configure_data(receive_profiles=["narrow"]))
    modem.submit(lambda: modem.configure_data(feedback_iters=2))
    while air.commands:
        air.commands.pop(0)()
    assert modem.fsm.cfg.advertise_gears == ("narrow",)
    assert modem.fsm.cfg.feedback_iters == 2



def test_preconnect_record_retains_delivery_accounting():
    modem = SabirModem(_StubAir(), ArqConfig(callsign="ALICE"))
    cap = _Capture()
    modem.start(cap)
    modem.transmit(b"queued before connect", msg_id=8)
    modem.connect("ALICE", "BOB")
    modem.fsm.on_control(wire.Control.connect(
        modem.fsm.session, _FULL, "BOB", ack=True, destination="ALICE"))
    assert modem.link_snapshot()["queue_bytes"] == len(b"queued before connect")
    f = modem.fsm._frame
    modem.fsm.on_control(wire.Control(
        wire.ACK, modem.fsm.session, seq=f.seq,
        mask=wire.cw_mask(range(len(f.chunks))), aux=wire.ack_aux(None)))
    modem.after_step()
    assert cap.progress == [(8, True)]



@pytest.mark.parametrize("timeout", [0, -1, float('inf'), float('nan'), True])
def test_inactivity_configuration_is_validated_atomically(timeout):
    modem = SabirModem(_StubAir())
    before = modem.fsm.cfg.feedback_iters
    with pytest.raises(ValueError, match='inactivity'):
        modem.configure_data(inactivity_timeout_s=timeout, feedback_iters=2)
    assert modem.fsm.cfg.feedback_iters == before
    modem.configure_data(inactivity_timeout_s=45.0)
    assert modem.fsm.cfg.inactivity_timeout_s == 45.0



def test_invalid_listen_does_not_change_identity(linked):
    cli, modem, _ = linked
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "ALICE"})
    cli.send({"m": M.LISTEN, "station_id": "BOB", "on": "yes"})
    assert cli.recv()["code"] == M.ERR_MALFORMED
    assert modem.calls == [("identity", "ALICE")]


def test_invalid_object_bytes_are_not_synthesized(linked):
    cli, modem, _ = linked
    sent = []
    modem.send_object = lambda data, **kwargs: sent.append(data) or bytes(16)
    cli.hello()
    cli.send({"m": M.SET_IDENTITY, "station_id": "ALICE"})
    cli.send({"m": M.SEND_OBJECT, "data": 500})
    assert cli.recv()["code"] == M.ERR_MALFORMED
    assert sent == []


def test_active_session_rejects_new_connect_and_beacon():
    modem, _ = _connected_modem()
    session = modem.fsm.session
    with pytest.raises(ValueError, match='active'):
        modem.connect('ALICE', 'CHARLIE')
    with pytest.raises(ValueError, match='idle'):
        modem.beacon()
    assert modem.fsm.session == session
