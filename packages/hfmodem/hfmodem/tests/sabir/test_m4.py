# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M4 acceptance gates. Full sweeps with larger volumes: ``python -m hfmodem.sabir.sim.m4``."""

import time

import numpy as np
import pytest

from hfmodem.sabir import compress
from hfmodem.sabir.arq import wire
from hfmodem.sabir.arq.fsm import ArqConfig, ArqFsm, SessionState
from hfmodem.sabir.host import messages as M
from hfmodem.sabir.sim.m3 import run_pair
from hfmodem.sabir.sim.m4 import email_text, make_pair, round_trip

# a peer advertising the full v2.0 feature set at the top gear ceiling
_FULL = wire.capabilities(range(3, 8), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)


# -- compression records -------------------------------------------------------
def test_compress_record_roundtrip():
    text = email_text(3000, seed=5)
    rec = compress.pack(text, compress=True)
    assert rec[0] == (compress.HASHED | compress.DEFLATE) and len(rec) < len(text) // 2
    raw = compress.pack(text, compress=False)
    assert raw[0] == (compress.HASHED | compress.RAW) and len(raw) == len(text) + 36

    noise = np.random.default_rng(0).integers(
        0, 256, 500, dtype=np.uint8).tobytes()
    assert compress.pack(noise, compress=True)[0] == (compress.HASHED | compress.RAW)

    stream = rec + raw + compress.pack(noise, compress=True)
    up = compress.Unpacker()
    got = []
    for i in range(0, len(stream), 7):          # arbitrary chunking
        got += list(up.feed(stream[i : i + 7]))
    assert got == [text, text, noise]


def test_wire_turn_and_id_blocks():
    for t in (wire.TURN, wire.TURN_REQ, wire.ID):
        c = wire.Control(t, 0x21, aux=b"ALICE   ")
        assert wire.Control.unpack(c.pack()) == c


# -- the FSM as pure logic: turn exchange, session ID, station ID --------------
class _StubIO:
    def __init__(self):
        self.controls, self.datas = [], []

    def send_control(self, ctrl):
        self.controls.append(ctrl)
        return 4.5

    def send_data(self, seq, gear, present, n_cw, nibbles, coded, offset=0):
        self.datas.append((seq, gear, tuple(present), n_cw))
        return 8.0

    def connected(self, *a): ...
    def disconnected(self): ...
    def deliver(self, blob): ...
    def log(self, msg): ...
    def state_changed(self, state): ...


def _connected_initiator(now):
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="ALICE"), clock=lambda: now[0])
    fsm.on_host_connect("BOB")
    fsm.on_control(wire.Control.connect(fsm.session, _FULL, "BOB", ack=True, destination="ALICE"))
    assert fsm.state == SessionState.CONNECTED and fsm._iss
    return io, fsm


def test_fsm_turn_handover_and_reclaim():
    now = [0.0]
    io, fsm = _connected_initiator(now)
    fsm.on_host_data(bytes(61))
    seq = io.datas[-1][0]
    # peer ACKs everything and flags queued traffic (gear bit 0)
    fsm.on_control(wire.Control(wire.ACK, fsm.session, seq=seq, gear=1,
                                mask=wire.cw_mask([0]),
                                aux=wire.ack_aux(None)))
    assert io.controls[-1].type == wire.TURN and not fsm._iss
    # peer never speaks: the turn timer takes the role back
    now[0] = fsm.next_deadline() + 1.0
    fsm.on_timer()
    assert fsm._iss
    # a TURN from the peer grants the role again
    fsm.on_control(wire.Control(wire.TURN, fsm.session))
    assert fsm._iss


def test_fsm_irs_requests_turn_and_sends():
    now = [0.0]
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="BOB"), clock=lambda: now[0])
    fsm.on_host_listen(True)
    tag = 0x81
    fsm.on_control(wire.Control.connect(tag, _FULL, "ALICE", destination="BOB"))
    assert fsm.state == SessionState.CONNECTED and not fsm._iss
    assert io.controls[-1].type == wire.CONNECT_ACK
    fsm.on_host_data(bytes(61))
    assert not io.datas                      # no role, no data on air
    now[0] = fsm.next_deadline() + 0.1       # quiescence grace expires
    fsm.on_timer()
    assert io.controls[-1].type == wire.TURN_REQ
    fsm.on_control(wire.Control(wire.TURN, tag))
    assert fsm._iss and len(io.datas) == 1   # role granted -> payload flows


def test_fsm_session_id_screens_and_stamps():
    now = [0.0]
    io, fsm = _connected_initiator(now)
    assert fsm.session != 0
    fsm.on_host_data(bytes(61))
    assert all(c.session == fsm.session for c in io.controls)
    # a DATA header with a foreign SessionID earns no ACK
    n = len(io.controls)
    fsm.on_data(wire.Control(wire.DATA, (fsm.session + 1) & 0xFF, seq=9,
                             gear=3, aux=wire.data_aux(10, 1, (0,) * 8)),
                None, None)
    assert len(io.controls) == n


def test_fsm_station_id_cadence_and_disc_callsign():
    now = [0.0]
    io, fsm = _connected_initiator(now)
    fsm.on_host_data(bytes(3 * 732))         # three 12-codeword frames
    fsm.on_host_disconnect()
    for _ in range(3):                       # ack a frame every 300 s
        seq, _, _, n_cw = io.datas[-1]
        now[0] += 300.0
        fsm.on_control(wire.Control(wire.ACK, fsm.session, seq=seq,
                                    mask=wire.cw_mask(range(n_cw)),
                                    aux=wire.ack_aux(None)))
    ids = [c for c in io.controls if c.type == wire.ID]
    assert ids and ids[0].call == "ALICE"
    assert fsm.stats["ids"] == [600.0]       # 0 -> 600 s: within §97.119
    disc = [c for c in io.controls if c.type == wire.DISC]
    assert disc and disc[0].call == "ALICE"


# -- realtime pacing (opt-in wall-clock for the virtual pair) ------------------
@pytest.mark.realtime
def test_unpaced_air_is_instant():
    from hfmodem.sabir.sim.air import SimulatedAir
    air = SimulatedAir(realtime=False)
    air._running = True
    air._wall0 = time.monotonic()
    t = time.monotonic()
    air._sleep_until(5.0)                    # 5 virtual seconds, no channel
    assert time.monotonic() - t < 0.05       # default: no wall time spent


@pytest.mark.realtime
def test_realtime_pacing_tracks_wall_clock():
    from hfmodem.sabir.sim.air import SimulatedAir
    air = SimulatedAir(realtime=True)
    air._running = True
    air._wall0 = time.monotonic()
    t = time.monotonic()
    air._sleep_until(0.2)                     # hold until wall reaches +0.2 s
    assert 0.15 < time.monotonic() - t < 0.6


def test_make_pair_realtime_and_hostapi_construct():
    pair = make_pair(None, realtime=True)
    try:
        assert pair.air.realtime
        assert all(hasattr(s, "port") for s in pair.servers)   # CBOR one-port
    finally:
        pair.stop()


@pytest.mark.realtime
def test_unpaced_reverse_report_completes_without_blocking():
    """Regression: a forward transfer, then a reverse message (the request ->
    REPORT pattern) over the *unpaced* pair. The reverse turnaround is driven
    by the idle-timer advance; the pacing change must leave the unpaced path
    teleporting to deadlines, not waiting on the wall clock. The reverse
    arriving byte-exact catches a timer stall; the wall-time bound catches a
    wall-clock block."""
    fwd = email_text(4000, seed=1)
    rev = b"REPORT " + email_text(300, seed=2)
    pair = make_pair(None)                       # clean channel, unpaced
    a = b = None
    t0 = time.monotonic()
    try:
        b = pair.client(1, "BOB")
        b.listen(True)
        a = pair.client(0, "ALICE")
        assert a.connect("BOB", timeout=120.0)
        assert b.wait_connected(timeout=120.0)
        a.send(fwd)
        assert b.recv(len(fwd), timeout=120.0) == fwd        # forward path
        b.send(rev)
        assert a.recv(len(rev), timeout=120.0) == rev        # reverse turnaround
        assert a.disconnect(timeout=120.0)
    finally:
        for c in (a, b):
            if c:
                c.close()
        pair.stop()
    assert time.monotonic() - t0 < 45.0          # unpaced must not pace itself


# -- sample-level bidirectional session ----------------------------------------
def test_session_bidirectional_poor_watterson():
    pa = np.random.default_rng(1).integers(
        0, 256, 1200, dtype=np.uint8).tobytes()
    pb = np.random.default_rng(2).integers(
        0, 256, 1200, dtype=np.uint8).tobytes()
    s = run_pair(pa, "poor", 8.0, seed=2, payload_b=pb)
    assert s.ok, "both directions must arrive byte-exact over Poor"
    assert s.delivered == pa and s.delivered_a == pb
    assert s.stats_b["frames"], "the responder actually sent data frames"


# -- the TCP host API ----------------------------------------------------------
def test_hosted_protocol_conformance():
    pair = make_pair(None)
    a = b = None
    try:
        a, b = pair.client(0, "ALICE"), pair.client(1, "BOB")
        b.listen()
        assert a.connect("BOB") and b.wait_connected()
        a.send(b"Z" * 600)
        assert b.recv(600) == b"Z" * 600
        assert a.flush()
        assert a.disconnect() and b.wait_disconnected()
        assert any(m["m"] == M.PHYSICAL_STATE and m.get("ptt") for m in a.messages)
        assert any(m["m"] == M.SEND_PROGRESS and m["delivered"] for m in a.messages)
    finally:
        for client in (a, b):
            if client is not None:
                client.close()
        pair.stop()


def test_hosted_email_round_trip_poor_watterson():
    rng = np.random.default_rng(4)
    pa = rng.integers(0, 256, 2500, dtype=np.uint8).tobytes()
    pb = rng.integers(0, 256, 1800, dtype=np.uint8).tobytes()
    r = round_trip(pa, pb, "poor", 10.0, seed=7, timeout=300.0)
    assert r["ok_ab"] and r["ok_ba"], "multi-KB payloads byte-exact both ways"
    assert any(m["m"] == M.PHYSICAL_STATE and m.get("ptt") for m in r["msgs_a"])
    assert any(m["m"] == M.LINK_STATS for m in r["msgs_a"])
    assert r["frames_a"] and r["frames_b"]


def test_hosted_compression_cuts_on_air_bytes():
    text = email_text(4000, seed=1)
    off = round_trip(text, b"", None, 25.0, compression=False,
                     seed=9, timeout=120.0)
    on = round_trip(text, b"", None, 25.0, compression=True,
                    seed=9, timeout=120.0)
    assert off["ok_ab"] and on["ok_ab"]
    assert on["link_bytes_a"] < 0.6 * off["link_bytes_a"]
    assert on["airtime_a"] < off["airtime_a"]



@pytest.mark.realtime
def test_unpaced_pair_leaves_time_for_host_to_react_to_connected():
    """A passive peer watchdog must not teleport past the client's next command."""
    pair = make_pair(None)
    a = b = None
    try:
        a, b = pair.client(0, "ALICE"), pair.client(1, "BOB")
        b.listen()
        assert a.connect("BOB", timeout=5)
        assert b.wait_connected(timeout=5)
        time.sleep(0.03)
        assert a.state == M.ST_CONNECTED and b.state == M.ST_CONNECTED
        a.send(b"host reaction")
        assert b.recv(13, timeout=5) == b"host reaction"
        assert a.disconnect(timeout=5) and b.wait_disconnected(timeout=5)
    finally:
        for client in (a, b):
            if client:
                client.close()
        pair.stop()
