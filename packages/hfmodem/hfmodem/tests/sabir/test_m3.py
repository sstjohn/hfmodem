# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M3 acceptance gates. Full sweeps with larger volumes: ``python -m hfmodem.sabir.sim.m3``."""

import numpy as np

from hfmodem.sabir.arq import layout, wire
from hfmodem.sabir.arq.fsm import ArqConfig, ArqFsm, SessionState
from hfmodem.sabir.phy import GEARS, Phy
from hfmodem.sabir.sim.m3 import (dd_demo, harq_demo, loading_demo, ramp_demo,
                            run_pair, selective_demo)

# a peer advertising the full v2.0 feature set at the top gear ceiling
_FULL = wire.capabilities(range(3, 7 + 1), wire.FASTCTL | wire.PBACK | wire.LOADING | wire.DEFLATE)


# -- wire format ---------------------------------------------------------------
def test_control_block_roundtrip():
    c = wire.Control(wire.DATA, 0x5A, seq=7, gear=6,
                     mask=wire.cw_mask([0, 3, 17, 63]),
                     aux=wire.data_aux(432, 24, (0, 1, 2, 3, 4, 0, 1, 2)))
    buf = c.pack()
    assert len(buf) == wire.BLOCK_BYTES
    assert wire.Control.unpack(buf) == c
    assert wire.Control.unpack(buf[:-1] + bytes([buf[-1] ^ 1])) is None
    assert wire.mask_indices(c.mask, 64) == [0, 3, 17, 63]
    assert wire.parse_data_aux(c.aux) == (432, 24, (0, 1, 2, 3, 4, 0, 1, 2))


def test_ack_snr_quantisation():
    snrs = [None, -16.0, 0.0, 12.25, 47.0, None, 3.0, -20.0]
    back = wire.parse_ack_aux(wire.ack_aux(snrs))
    assert back[0] is None and back[5] is None
    assert back[1] == -16.0 and back[3] == 12.25 and back[7] == -16.0
    assert wire.parse_ack_aux(wire.ack_aux(None)) == [None] * 8


def test_loading_expansion():
    g = GEARS["fast"]
    ld = wire.expand_loading((0, 1, 2, 3, 4, 0, 0, 0), g)
    assert (ld[:7] == 4).all()          # default = the gear's 16-QAM
    assert (ld[7:14] == 0).all()        # off
    assert (ld[14:21] == 2).all() and (ld[21:28] == 4).all()
    assert (ld[28:35] == 6).all()
    assert wire.expand_loading((0,) * 8, g) is None


# -- codeword layout through the PHY ------------------------------------------
def test_layout_striped_repeat_roundtrip():
    phy = Phy(GEARS["workhorse"])
    rng = np.random.default_rng(0)
    coded = rng.integers(0, 2, (6, 1024))
    bits, n_syms = layout.assemble(phy, coded, grouped=False, repeat=2)
    _, llr, _ = phy.receive(phy.from_audio(phy.to_audio(phy.transmit(bits))),
                            n_symbols=n_syms)
    rows = layout.extract(phy, llr, 6, 1024, grouped=False, repeat=2)
    assert ((rows < 0).astype(int) == coded).all()


def test_layout_grouped_loading_roundtrip():
    phy = Phy(GEARS["fast"])
    rng = np.random.default_rng(1)
    ld = wire.expand_loading((2, 3, 4, 1, 0, 0, 2, 3), phy.gear)
    coded = rng.integers(0, 2, (10, 1024))
    bits, n_syms = layout.assemble(phy, coded, grouped=True, loading=ld)
    assert bits.size == phy.capacity(n_syms, loading=ld)
    tx = phy.transmit(bits, loading=ld)
    _, llr, _ = phy.receive(phy.from_audio(phy.to_audio(tx)),
                            n_symbols=n_syms, loading=ld)
    rows = layout.extract(phy, llr, 10, 1024, grouped=True, loading=ld,
                          n_syms=n_syms)
    # noiseless loopback: essentially exact (a few 64-QAM cells at the mask
    # edge may flip -- far inside what the codeword corrects)
    assert ((rows < 0).astype(int) != coded).mean() < 1e-3


def test_grouped_confines_codewords():
    phy = Phy(GEARS["fast"])
    groups = layout.assign(phy, 24, 1024, None)
    assert sorted(set(groups)) == list(range(8))    # spread over all groups
    assert max(np.bincount(groups)) == 3            # 24 cws balance 3/group


# -- the FSM as pure logic ------------------------------------------------------
class _StubIO:
    def __init__(self):
        self.controls, self.datas = [], []

    def send_control(self, ctrl):
        self.controls.append(ctrl)
        return 4.5

    def send_data(self, seq, gear, present, n_cw, nibbles, coded, offset=0):
        self.datas.append((seq, gear, tuple(present), n_cw, nibbles))
        return 8.0

    def connected(self, *a): ...
    def disconnected(self): ...
    def deliver(self, blob): ...
    def log(self, msg): ...
    def state_changed(self, state): ...


def test_fsm_selective_retransmit_and_timeout():
    now = [0.0]
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="A"), clock=lambda: now[0])
    fsm.on_host_connect("B")
    assert fsm.state == SessionState.CONNECTING and io.controls[-1].type == wire.CONNECT
    fsm.on_control(wire.Control.connect(fsm.session, _FULL, "B", ack=True, destination="A"))
    assert fsm.state == SessionState.CONNECTED
    fsm.on_host_data(bytes(3 * 61))                 # 3 workhorse codewords
    seq, gear, present, n_cw, _ = io.datas[-1]
    assert present == (0, 1, 2) and n_cw == 3
    # partial ACK: cw 1 still missing -> selective retransmit of just cw 1
    fsm.on_control(wire.Control(wire.ACK, fsm.session, seq=seq,
                                mask=wire.cw_mask([0, 2]),
                                aux=wire.ack_aux(None)))
    assert io.datas[-1][2] == (1,)
    # ACK lost: the timer retransmits the same codeword
    now[0] = fsm.next_deadline() + 1.0
    fsm.on_timer()
    assert io.datas[-1][2] == (1,)
    # full ACK completes the frame; nothing queued -> no more data sends
    fsm.on_control(wire.Control(wire.ACK, fsm.session, seq=seq,
                                mask=wire.cw_mask([0, 1, 2]),
                                aux=wire.ack_aux(None)))
    assert fsm._frame is None and len(io.datas) == 3


def test_fsm_stalled_round_rebuilds_one_rung_down():
    io = _StubIO()
    fsm = ArqFsm(io, ArqConfig(callsign="A", start_rung=2),
                 clock=lambda: 0.0)
    fsm.on_host_connect("B")
    fsm.on_control(wire.Control.connect(fsm.session, _FULL, "B", ack=True, destination="A"))
    fsm.on_host_data(bytes(2 * 93))
    seq = io.datas[-1][0]
    fsm.on_control(wire.Control(wire.ACK, fsm.session, seq=seq,
                                mask=wire.cw_mask([]),
                                aux=wire.ack_aux(None)))
    seq2, gear2, present2, n_cw2, _ = io.datas[-1]
    assert seq2 != seq                              # rebuilt under a new seq
    assert fsm.rung == 1                            # one rung down
    # same 186 payload bytes, re-chunked at workhorse's 61-byte codewords
    assert n_cw2 == 4 and present2 == (0, 1, 2, 3)


# -- session gates --------------------------------------------------------------
def test_session_poor_watterson_byte_exact():
    payload = np.random.default_rng(1).integers(
        0, 256, 3000, dtype=np.uint8).tobytes()
    s = run_pair(payload, "poor", 8.0, seed=2)
    assert s.ok, "multi-KB payload must arrive byte-exact over Poor"
    assert s.throughput_bps > 100
    assert s.stats_a["rebuilds"] <= 2   # the odd failed probe is by design
    assert "DISCONNECTED" in s.log_a[-2:] or "DISCONNECTED" in s.log_a


def test_harq_chase_beats_blind_repeat():
    h = harq_demo(snr_db=-2.0, n_trials=8, seed=0)
    assert h["chase_fail"] == 0                     # combining always lands it
    assert h["blind_fail"] >= 6                     # blind repeat almost never
    assert h["chase_mean_tx"] < 0.6 * h["blind_mean_tx"]


def test_selective_ack_retransmits_only_dead_groups():
    sel = selective_demo(selective=True)
    whole = selective_demo(selective=False)
    assert sel["ok"] and whole["ok"]
    r1, r2 = sel["sends"][0], sel["sends"][1]
    assert r1["n_cw"] == 24
    assert r2["n_cw"] <= 8                          # only the notched groups
    w2 = whole["sends"][1]
    assert w2["n_cw"] == 24
    assert r2["dur"] < 0.75 * w2["dur"]             # airtime actually saved
    assert sel["airtime_a"] < whole["airtime_a"]


def test_gearshift_tracks_snr_ramp():
    r = ramp_demo(payload_bytes=20_000, period=220.0, seed=5)
    assert r["ok"], "delivery must stay byte-exact through the ramp"
    rungs = [f[1] for f in r["traj"]]
    # A short bootstrap reaches the low-SNR start of the ramp sooner; the
    # first successful frame may follow an appropriate robust fallback.
    assert rungs[0] in ("robust", "workhorse")
    assert "fast" in rungs or "max" in rungs        # climbed with the SNR
    order = ["robust", "workhorse", "workhorse34", "fast", "max"]
    idx = [order.index(g) for g in rungs]
    assert max(idx) > idx[0]                        # shifted up
    assert len(r["shifts"]) >= 2


def test_loading_beats_uniform_on_notched_channel():
    ld = loading_demo(payload_bytes=16000)
    assert ld["loaded"]["ok"] and ld["uniform"]["ok"]
    gain = ld["loaded"]["throughput_bps"] / ld["uniform"]["throughput_bps"]
    assert gain > 1.15, f"loading gain only x{gain:.2f}"


def test_dd_estimation_not_harmful_and_helps_on_fading():
    a = dd_demo("workhorse", "poor", 4.0, n_frames=40, seed=21)
    b = dd_demo("fast", "moderate", 15.0, n_frames=40, seed=22)
    assert a["fer_dd1"] <= a["fer_dd0"] + 0.05      # never destabilises
    assert b["fer_dd1"] <= b["fer_dd0"] + 0.05
    assert (a["fer_dd1"] + b["fer_dd1"]) <= (a["fer_dd0"] + b["fer_dd0"])
