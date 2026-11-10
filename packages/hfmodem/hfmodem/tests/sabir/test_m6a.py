# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M6a acceptance gates. Full sweeps with numbers: ``python -m hfmodem.sabir.sim.m6a``."""

import numpy as np

from hfmodem.sabir.arq import ArqConfig, FastControl, wire
from hfmodem.sabir.arq.fsm import ArqFsm, SessionState
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.phy.modem import FS, GUARD_HEAD, GUARD_TAIL
from hfmodem.sabir.sim.m2 import add_noise_snr3k
from hfmodem.sabir.sim.m3 import run_pair
from hfmodem.sabir.sim.m6a import fade_snr, rung_table

# a peer advertising the full v2.0 feature set at the top gear ceiling
_FULL = wire.capabilities(range(3, 7 + 1), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)


def _unit_rms(wav):
    body = wav[GUARD_HEAD : wav.size - GUARD_TAIL]
    return wav / np.sqrt(np.mean(np.abs(body) ** 2))


# -- the fast control burst ----------------------------------------------------
def test_fastctl_roundtrip_and_speed():
    fc = FastControl(Phy(GEARS["workhorse"]))
    rng = np.random.default_rng(0)
    one = wire.Control(wire.ACK, 0x42, seq=3, mask=wire.cw_mask([0, 5, 63]),
                       aux=wire.ack_aux([10.0] * 8)).pack()
    two = one + wire.Control(wire.DATA, 0x42, seq=4, gear=4,
                             mask=wire.cw_mask(range(12)),
                             aux=wire.data_aux(308, 12, (0,) * 8)).pack()
    for blocks, n in ((one, 1), (two, 2)):
        assert fc.n_samples(n) / FS < 1.2          # ~9x faster than floor
        tx = _unit_rms(fc.transmit(blocks))
        assert tx.size == fc.n_samples(n)
        y = add_noise_snr3k(tx, 6.0, rng)
        assert fc.receive(y, n) == (blocks, 0)


def test_fastctl_rejects_noise_and_floor_bursts():
    from hfmodem.sabir.floor.mfsk import FloorModem
    fc = FastControl(Phy(GEARS["workhorse"]))
    rng = np.random.default_rng(1)
    floor_burst = add_noise_snr3k(_unit_rms(FloorModem().transmit(
        wire.Control.connect(0x42, _FULL, "ALICE", destination="BOB").pack())), 10.0, rng)
    n2 = fc.n_samples(2) + 1024
    noise = rng.standard_normal(n2) + 1j * rng.standard_normal(n2)
    for n in (1, 2):
        assert fc.receive(floor_burst[: fc.n_samples(n) + 1024], n)[0] is None
        assert fc.receive(noise[: fc.n_samples(n) + 1024], n)[0] is None


# -- tier policy as pure logic --------------------------------------------------
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


def _pair(now, **kw):
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="ALICE", **kw), clock=lambda: now[0])
    fsm.on_host_connect("BOB")
    fsm.on_control(wire.Control.connect(fsm.session, _FULL, "BOB", ack=True, destination="ALICE"))
    assert fsm.state == SessionState.CONNECTED and not fsm.ctrl_fast
    return io, fsm


def _ack(fsm, seq, n_cw, snr_db, gear=0):
    fsm.on_control(wire.Control(
        wire.ACK, fsm.session, seq=seq, gear=gear,
        mask=wire.cw_mask(range(n_cw)), aux=wire.ack_aux([snr_db] * 8)))


def test_tier_hysteresis_and_timeout_fallback():
    now = [0.0]
    io, fsm = _pair(now)
    fsm.on_host_data(bytes(4 * 732))
    # reported group SNR converts ~ -21 dB to the 3 kHz figure at workhorse
    off = fsm._snr_offset_db(1)
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 12, 4.0 + off)                   # 4 dB: below on-threshold
    assert not fsm.ctrl_fast
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 12, 10.0 + off)                  # 10 dB: fast earned
    assert fsm.ctrl_fast
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 12, 4.0 + off)                   # 4 dB again: hysteresis
    assert fsm.ctrl_fast
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 12, 2.0 + off)                   # 2 dB: below off-threshold
    assert not fsm.ctrl_fast
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 12, 10.0 + off)
    assert fsm.ctrl_fast
    now[0] = fsm.next_deadline() + 1.0              # a control round trip dies
    fsm.on_timer()
    assert not fsm.ctrl_fast and fsm.stats["timeouts"] == 1


def test_fast_ctrl_master_switch():
    now = [0.0]
    io, fsm = _pair(now, fast_ctrl=False)
    fsm.on_host_data(bytes(61))
    seq, *_ = io.datas[-1]
    _ack(fsm, seq, 1, 40.0)
    assert not fsm.ctrl_fast                        # never leaves the floor


def test_irs_never_replies_faster_than_the_header():
    now = [0.0]
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="BOB"), clock=lambda: now[0])
    fsm.on_host_listen(True)
    tag = 0x81
    fsm.on_control(wire.Control.connect(tag, _FULL, "ALICE", destination="BOB"))
    hdr = wire.Control(wire.DATA, tag, seq=5, gear=4,
                       aux=wire.data_aux(10, 1, (0,) * 8))
    fsm.on_data(hdr, None, [30.0] * 8, hdr_fast=True)
    assert fsm.ctrl_fast                            # measured healthy: fast
    fsm.on_data(hdr, None, [30.0] * 8, hdr_fast=False)
    assert not fsm.ctrl_fast                        # floor header: match it


def test_handover_piggybacks_ack_on_reverse_data():
    now = [0.0]
    io, fsm = _pair(now)
    fsm.on_host_data(bytes(800))                    # two frames' worth
    seq, gear, *_ = io.datas[-1]
    assert not gear & wire.HANDOVER                 # no peer traffic known yet
    off = fsm._snr_offset_db(1)
    _ack(fsm, seq, 12, 10.0 + off, gear=wire.ACK_TRAFFIC)
    seq, gear, *_ = io.datas[-1]                    # last frame: queue drained
    assert gear & wire.HANDOVER and fsm.ctrl_fast
    # the peer completes it and answers with a piggybacked ACK + its own DATA
    _ack(fsm, seq, 2, 10.0 + off, gear=wire.ACK_TOOK_ROLE)
    assert not fsm._iss and fsm._frame is None      # no TURN over needed
    # ...whose handover we take right back: queued traffic, fast tier
    hdr = wire.Control(wire.DATA, fsm.session, seq=7, gear=4 | wire.HANDOVER,
                       mask=wire.cw_mask([0]), aux=wire.data_aux(10, 1, (0,) * 8))
    fsm.on_host_data(bytes(61))
    n_data = len(io.datas)
    codec = fsm._codec(1)
    llr = 20.0 * (1 - 2 * codec.encode_cw(bytes(61))[None, :].astype(float))
    fsm.on_data(hdr, llr, [30.0] * 8, hdr_fast=True)
    assert fsm._iss and len(io.datas) == n_data + 1
    assert fsm.stats["pb_acks"] == 1 and fsm.pb_ack is None
    assert not any(c.type == wire.ACK and c.seq == 7 for c in io.controls)


# -- rendered timing ------------------------------------------------------------
def test_control_overhead_gap_closes():
    rows = rung_table()
    for r in rows:
        assert r["new_bps"] > 1.5 * r["old_bps"]    # >= +50% at every rung
    top = rows[-1]
    assert top["old_bps"] < 4500 < 6500 < top["new_bps"]  # ceiling ~4.1 -> ~7.2k
    assert all(r["new_eff"] > 0.78 for r in rows)


# -- sessions -------------------------------------------------------------------
def test_session_fast_control_reduces_overhead_on_strong_awgn():
    payload = np.random.default_rng(1).integers(
        0, 256, 3000, dtype=np.uint8).tobytes()
    new = run_pair(payload, None, 30.0, seed=2)
    old = run_pair(payload, None, 30.0, seed=2, cfg_kw={"fast_ctrl": False})
    assert new.ok and old.ok
    assert new.throughput_bps > 1.4 * old.throughput_bps
    assert new.stats_a["ctrl"]                      # the fast tier engaged


def test_deep_fade_falls_back_to_floor_and_recovers():
    payload = np.random.default_rng(11).integers(
        0, 256, 8000, dtype=np.uint8).tobytes()
    s = run_pair(payload, "poor", fade_snr, seed=6, max_exchanges=900)
    assert s.ok, "the fade must not break delivery"
    tiers = [tier for _, tier in s.stats_a["ctrl"]]
    assert tiers[0] == "fast" and "floor" in tiers[1:]
