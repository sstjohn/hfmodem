# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The acknowledgement rides the free-running grid, because that is where the
peer reads.

A PACTOR-1 ISS latches its read instant once and never searches again:
`receive_cs` reads twelve bits at `ps.rxtime` with no timing search
(hfkernel/fsk/pactor.c:745, :756), and `ps.rxtime` moves only by whole cycles
(:660) and by the changeover's 840 ms rotation (:1122). Where that instant sits
against the ISS's OWN packet depends on which end of the call it is:

    ISS as caller (master)     packet_end + d
    ISS as called (responder)  packet_end + (170 ms - d)

with `d` = 10 ms + the answering station's txdelay (:1745). `_reference` below
runs that register arithmetic rather than quoting it. Against a Winlink RMS we
are always the caller, so the gateway is always the responder and always reads at
`170 ms - d` -- which is `slot - data - cs - d`, the free rotation, at every
turnaround, with the peer's own gap cancelling out of the cycle length.

So there is ONE placement and the grid has no second one to choose between:
nothing the peer does moves a transmit boundary, and the whole of the aim is
that the anchor is left alone. What the peer's measured burst is still for is
`key_refusal` -- our carrier must never come up in the peer's own air -- and the
`[ack]` line, where `rot` measures the carrier against the instant the peer
reads at.

AND THE OTHER END OF THAT SENTENCE IS A GUARD. This station can be the called
one -- `arq.on_rx_connect` answers a burst naming us -- and there the peer is
the caller and reads `2d - (slot - data - cs)` from where this grid keys. The
placement a called station needs is the same free-running one anchored at the
caller's connect instead of our own call, and nothing places that anchor, so
`RadioTx.answered_refusal` keys nothing at all on a link we answered.

THE READ HAS NO SLACK, and that is the other half of this file. Twelve bit
periods are read for a twelve-bit word, so a word that starts outside
`p1rx.CS_ANCHOR_S` is refused at any signal level -- measured here at 20, 12
and 6 dB against the reference's own hard-decision reader.

Run:  pytest hfmodem/tests/shrike/test_ackplace.py
"""
from __future__ import annotations

import re

import numpy as np

from hfmodem.shrike import onair, p1rx, pactor1, ptc, rxfront, spec
from hfmodem.shrike.arq import State
from hfmodem.shrike.spec import Protocol

FS = onair.FS
SLOT_N = round(1.25 * FS)
DATA_N = round(spec.P1_PACKET_S * FS)
CS_N = round(spec.P1_CS_S * FS)
#: What a cycle leaves past a data packet: one control signal and the turnaround
#: either side of it. The ack sits `ROTATION_N - d` past the peer's data end.
ROTATION_N = SLOT_N - DATA_N - CS_N


def _irs_grid(onset=None, d_s: float = 0.0875) -> onair._MasterGrid:
    g = onair._MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                          packet_n=DATA_N, cs_n=CS_N, d_max_n=round(0.13 * FS))
    g.reverse(to_iss=False)      # we hold the IRS side: our burst is the CS
    g.d_n = d_s * FS
    g.acquired = g.corroborated = True
    g.peer_onset = onset
    return g


class _Live:
    """The emission path's stream, reduced to a clock that is never late."""
    pos = 0
    holdback = 0

    def clamp_late(self, at):
        return 0

    def take_until(self, at):
        return np.zeros(0, np.float32)

    def wait_until(self, at):
        pass

    def sample_now(self):
        return self.pos


# -- what the reference does, in the reference's own registers ---------------

_US = 1_000_000
_CYCLE, _FEC = 1_250_000, 960_000                 # pactor.c:52, :54
_CS = 120_000                                     # 12 bits at 100 Bd
_ROT = _FEC - _CS                                 # :1122 / :1214 -- 840 ms


class _Station:
    def __init__(self, txdelay: int):
        self.txdelay = txdelay
        self.txtime = self.rxtime = 0
        self.packet_end = 0
        self.read_at: int | None = None

    def send(self, dur: int) -> None:
        self.packet_end = self.txtime + dur

    def read_cs(self) -> None:
        if self.read_at is None:
            self.read_at = self.rxtime - self.packet_end

    def next_cycle(self) -> None:                 # :660
        self.rxtime += _CYCLE
        self.txtime += _CYCLE


def _reference(txd_master: int, txd_responder: int) -> tuple[int, int]:
    """Both stations' first ISS read, as an offset from their own packet end.

    The call (:1884), the responder latching its transmit on what it decoded
    (:1744/:1745), the master latching its read on the search hit (:1897/:1898,
    :1904), and the changeover rotating each station's own clock by 840 ms
    (:1122 for the new IRS, :1214 for the new ISS).
    """
    m, s = _Station(txd_master), _Station(txd_responder)
    m.rxtime = m.txtime + _FEC
    m.send(_FEC)                                  # the CALL
    s.rxtime = m.txtime
    s.txtime = s.rxtime + _FEC + s.txdelay + 10_000
    m.rxtime = m.txtime + _FEC + (s.txdelay + 10_000)
    m.rxtime += _CYCLE
    m.txtime += 2 * _CYCLE
    iss, irs = m, s
    for _ in range(2):
        for _ in range(3):
            iss.send(_FEC)                        # tx_100_csN  :1156
            iss.next_cycle()
            iss.read_cs()                         # receive_cs  :745
            irs.send(_CS)                         # rx_100_csN  :1071
            irs.next_cycle()
        iss.txtime += _ROT                        # tx_rx_100   :1214
        irs.rxtime += _ROT                        # rx_tx_100   :1122
        iss, irs = irs, iss
    return m.read_at, s.read_at


def test_the_called_station_reads_at_the_rotation_and_the_caller_at_d():
    """The asymmetry the peer-anchored placement was built without.

    Both stations run `receive_cs` at one latched instant, and they latch it
    from opposite ends of the connect -- so the same protocol puts the caller's
    read `d` past its own packet and the called station's `170 ms - d` past
    its own. We are always the caller; the gateway is always the other one.
    """
    for txd_m, txd_s in ((0, 0), (0, 50_000), (30_000, 50_000), (0, 100_000)):
        d = txd_s + 10_000
        master, responder = _reference(txd_m, txd_s)
        assert master == d, (txd_m, txd_s, master)
        assert responder == _CYCLE - _FEC - _CS - d, (txd_m, txd_s, responder)
    # ...and it is the caller's txdelay that is nowhere in either of them.
    assert _reference(0, 50_000) == _reference(30_000, 50_000)


def test_the_grid_keys_where_the_called_station_reads():
    """`slot - data - cs - d` past the peer's data end, at every turnaround --
    and the pair's turn is exactly one cycle there, whatever `d` is."""
    for d_n in range(round(onair.D_MIN_S * FS), round(0.131 * FS)):
        g = _irs_grid(d_s=d_n / FS)
        onset = g.rx_due(4)                   # the peer answering on our raster
        for slot in (5, 9, 14):
            phase = (g.boundary(slot) - onset - DATA_N) % SLOT_N
            assert phase == ROTATION_N - d_n, d_n
            assert DATA_N + phase + CS_N + d_n == SLOT_N, d_n


def test_nothing_the_peer_does_moves_a_transmit_boundary():
    """The whole of the placement. A tracked onset, a corroborated one, a `d`
    walked anywhere the tracker can walk it, either role: the boundary is the
    anchor plus whole cycles and there is no branch that leaves it."""
    free = _irs_grid().boundary
    for d_ms in (-30, 0, 39.1, 45, 70, 92.8, 130, 240):
        for onset in (None, 4321, 7 * SLOT_N + 4321):
            for corroborated in (False, True):
                for sending in (False, True):
                    g = _irs_grid(onset=onset, d_s=d_ms / 1e3)
                    g.corroborated, g.sending = corroborated, sending
                    assert [g.boundary(s) for s in (5, 9, 14)] == \
                        [free(s) for s in (5, 9, 14)], (d_ms, onset)


def test_the_placement_holds_the_raster_and_the_fitted_aim_never_did():
    """`ack_start(n+1) - ack_start(n)` is one cycle identically, which is what
    the peer's latched read requires. The retired fit -- `d - lead`, the aim
    every session before 2026-08-20 flew -- solved that at one turnaround
    (d = 110 ms) and was short of it by `2d - (rotation + lead)` everywhere
    below, against a station whose clock follows ours."""
    g = _irs_grid()
    assert all(g.boundary(k + 1) - g.boundary(k) == SLOT_N for k in range(20))
    lead_n = round(0.050 * FS)               # the renderer's, as it stood
    for d_ms, short_ms in ((96, -28), (92.8, -34.4), (76, -68)):
        d_n = round(d_ms / 1e3 * FS)
        fitted = DATA_N + (d_n - lead_n) + CS_N + d_n
        assert abs((fitted - SLOT_N) / FS * 1e3 - short_ms) < 1.0, d_ms


def test_the_guard_refuses_the_peers_own_air():
    """...and it is asked about the CARRIER, which is a settle in front of the
    boundary. The condition is overlap and nothing else: a carrier at the peer's
    data end is clear, one a millisecond inside it is not, and the acknowledgement
    behind it has to be down before the peer's next packet."""
    onset = 3 * SLOT_N + 1000
    g = _irs_grid(onset=onset)
    end = onset + DATA_N
    settle_n = round(0.040 * FS)
    air_n = settle_n + CS_N
    assert g.key_refusal(end, air_n) is None
    assert g.key_refusal(end + round(0.0375 * FS) + 4 * SLOT_N, air_n) is None
    inside = g.key_refusal(end - 1, air_n)
    assert inside is not None and "still on the air" in inside, inside
    deep = g.key_refusal(onset + DATA_N // 2 + 2 * SLOT_N, air_n)
    assert deep is not None and "still on the air" in deep
    # A carrier late enough that the acknowledgement's tail lands on the head of
    # the peer's next packet. The band is `SLOT_N - DATA_N - air_n` wide.
    late = g.key_refusal(end + SLOT_N - DATA_N - air_n + 1, air_n)
    assert late is not None and "before our" in late, late
    assert g.key_refusal(end + SLOT_N - DATA_N - air_n, air_n) is None
    g.sending = True
    assert g.key_refusal(end - 1, air_n) is None


def test_the_tracked_burst_lives_one_cycle():
    g = _irs_grid()
    at = g.anchor + 5 * SLOT_N + g.packet_n + g.d + 3
    g.update([at])
    assert g.peer_onset == at
    g.update([])
    # The onset lives one cycle: a miss releases it at once, while the
    # receive window keeps its three-cycle grace.
    assert g.peer_onset is None and g.locked


def test_the_refusal_reaches_the_key(capsys, tmp_path):
    """A peer whose packet our own boundary lands inside is not answered at all.

    The grid cannot produce that on its own -- the rotation is `170 ms - d`
    clear of the peer's data by construction -- so it takes a `d` the tracker
    has walked past what we can hear out, or an onset that was never the peer's.
    The production `_tx` must drop the burst rather than key it, with the
    numbers in the line.
    """
    g = _irs_grid(d_s=0.240)
    g.peer_onset = g.boundary(3) - DATA_N // 2       # our key, mid-packet
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path)
    tx.live = _Live()
    tx.aim(g, 3)
    tx.send_p1_cs(0)
    out = capsys.readouterr().out
    assert "NOT KEYING" in out and "still on the air" in out
    assert tx.tx_end is None
    assert not list(tmp_path.glob("tx_*.wav"))


def _answering(tmp_path) -> tuple:
    """A station that answered a call, wired to the emission path it would key
    through: the whole live route, `on_rx_event` to the transmitter."""
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path)
    tx.live = _Live()
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.on_rx_event(rxfront.Event(
        1.0, "connect", "", connect=p1rx.Connect("Normal", "W9SSJ", False)))
    return host, tx


def test_a_link_we_answered_puts_nothing_on_the_air(capsys, tmp_path):
    """The station that called us reads at ITS latched instant, and this grid
    is the caller's -- so an acknowledgement keyed off it lands
    `2d - (slot - data - cs)` from the read, seven bit periods at a typical
    turnaround, which the reference decoder refuses at any level.

    So NOTHING GOES OUT. The thing asserted is the air: no carrier, no burst
    written, no transmission counted, whatever the FSM believes about the
    link. The FSM answers a call whenever `LISTENING` is on -- and a station
    that answers a call is precisely what the host emulation is for."""
    host, tx = _answering(tmp_path)
    assert host.arq.state is State.CONNECTED and host.arq.answering
    assert not list(tmp_path.glob("tx_*.wav"))
    assert (tx.n, tx.keyed, tx.p1_seq, tx.tx_end) == (0, [], [], None)
    assert "NOT KEYING" in capsys.readouterr().out


def test_a_call_we_placed_ourselves_still_keys(capsys, tmp_path):
    """...and the guard is not a mute. Same station, same transmitter, the
    other end of the call: the connect burst goes out."""
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path)
    tx.live = _Live()
    host = ptc.PtcHost(peer=tx, mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    assert not host.arq.answering
    assert len(list(tmp_path.glob("tx_*.wav"))) == 1
    assert "NOT KEYING" not in capsys.readouterr().out


def _keyed(g: onair._MasterGrid, tmp_path, slot: int = 3, late_n: int = 0) -> str:
    # The settle this station flies. `RadioTx` defaults to 0.100, which `_budget`
    # refuses before a rig is opened -- and which puts the carrier of a 92.8 ms
    # turnaround inside the peer's own packet, so the ack guard now stops it.
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path, settle=0.040)
    tx.live = _Live()
    # The stream's clock standing where the burst is aimed, so `rot` is the
    # placement and not the distance to a fake clock parked at zero.
    tx.live.pos = tx.aim(g, slot) - round(tx.settle * FS) + late_n
    tx.send_p1_cs(0)
    return tx


#: Turnarounds this station has measured at a gateway, which is the whole of
#: what moves the ack instant: KB5LZK 74.1 and 78.9 ms, WS8EOC 96.2 and 100.2,
#: VE1YZ 99-108 across one 141-cycle session. The `key-dataend` readings those
#: sessions printed -- 95.9 and 73.8 ms, and 62.0-70.6 through the 2026-09-02
#: stall -- are `ROTATION_N - d` at each of them and are not a drift.
MEASURED_D_S = (0.0741, 0.0789, 0.0928, 0.0962, 0.1002, 0.1005, 0.1080)


def test_the_ack_keys_on_the_instant_the_peer_reads_at(capsys, tmp_path):
    """`rot` is the field a session log is read off: our carrier against the
    boundary the peer's read is latched to, which nothing the peer does can
    move. On the placement that flies it is the transmitter's own error and
    nothing else.

    Asked at every turnaround this station has measured, because `key-dataend`
    is the same instant seen from the far end and moves with `d` alone: a band
    of readings that walks with the gateway is the placement working, and one
    that walks without it is not."""
    for d_s in MEASURED_D_S:
        g = _irs_grid(d_s=d_s)
        g.peer_onset = g.rx_due(2)
        _keyed(g, tmp_path)
        out = capsys.readouterr().out
        assert "NOT KEYING" not in out and "[ack] key-dataend" in out
        rot = float(re.search(r"rot\s*([+-][\d.]+) ms", out).group(1))
        assert abs(rot) < 0.1, out
        gap = float(re.search(r"key-dataend\s*([+-][\d.]+) ms", out).group(1))
        assert abs(gap - (ROTATION_N - g.d) / FS * 1e3) < 0.1, out
    assert list(tmp_path.glob("tx_*.wav"))


def test_an_ack_outside_the_read_anchor_says_so_on_the_air(capsys, tmp_path):
    """The margin the record has no slack in. One of the twelve
    acknowledgements ever keyed at this aim sat at +6.2 ms, outside the
    bracket and refused at every signal level in replay -- and the session it
    flew in read like every other. A live burst outside `CS_ANCHOR_S` is
    named where the operator is watching."""
    g = _irs_grid(d_s=0.0928)
    g.peer_onset = g.rx_due(2)
    inside = round(0.004 * FS)
    _keyed(g, tmp_path, late_n=inside)
    assert "READ ANCHOR" not in capsys.readouterr().out
    outside = round(0.0062 * FS)
    _keyed(g, tmp_path, late_n=outside)
    out = capsys.readouterr().out
    assert "OUTSIDE THE 5 ms READ ANCHOR" in out, out
    assert "+6.2 ms" in out, out


# -- the reader the bracket is a property of --------------------------------

_SPS = FS // 100


def _receive_cs(seg: np.ndarray, at: int) -> int:
    """pactor.c:745 -- twelve bits read from one instant, hard decision, no
    search. Written out rather than imported, because `p1rx` is allowed to be
    cleverer than the station we are transmitting to."""
    t = np.arange(_SPS) / FS
    mark = np.exp(-2j * np.pi * pactor1.MARK * t)
    space = np.exp(-2j * np.pi * pactor1.SPACE * t)
    bits = 0
    for i in range(12):
        w = seg[at + i * _SPS:at + (i + 1) * _SPS]
        bits |= int(abs(w @ mark) > abs(w @ space)) << i
    return bits


def _air(rot_ms: float, rng, noise: float) -> tuple[np.ndarray, int]:
    """What the peer hears: noise, with our codeword `rot` from its read
    instant, which is index `pad`."""
    pad = round(0.4 * FS)
    seg = rng.normal(0, noise, 2 * pad + 24 * _SPS).astype(np.float32)
    burst = pactor1.control_signal(pactor1.CS_ACK_A)
    at = pad + round(rot_ms * 1e-3 * FS)
    seg[at:at + burst.size] += burst
    return seg, pad


def _accepted(rot_ms: float, snr_db: float, trials: int, rng) -> float:
    noise = 0.11 / (10 ** (snr_db / 20))
    want = pactor1.CONTROL_SIGNALS[pactor1.CS_ACK_A]
    hits = 0
    for _ in range(trials):
        seg, pad = _air(rot_ms, rng, noise)
        hits += bin(_receive_cs(seg, pad) ^ want).count("1") <= 1
    return hits / trials


def test_the_reference_read_is_a_cliff_at_the_anchor_and_not_an_snr_problem():
    """Twelve bit periods for a twelve-bit word.

    A CLIFF AND NOT A MARGIN: inside 3 ms the word is read at every level the
    band offers, and 14 dB more signal buys back nothing at 6 ms out. The edge
    sits between them -- 4 ms is 100% at 20 and 12 dB and 59% at 6 -- and
    `CS_ANCHOR_S` is where the `[ack]` line draws it.
    """
    rng = np.random.default_rng(20260820)
    for snr_db in (20, 12, 6):
        for rot_ms in (0.0, -3.0, 3.0):
            got = _accepted(rot_ms, snr_db, 120, rng)
            assert got >= 0.95, (snr_db, rot_ms, got)
        for rot_ms in (-6.0, 6.0, -70.0, -121.0):
            got = _accepted(rot_ms, snr_db, 120, rng)
            assert got <= 0.02, (snr_db, rot_ms, got)
    assert 0.003 < p1rx.CS_ANCHOR_S < 0.006


# -- the session verdict, which the placement never entered ------------------

def _cs(t, cs):
    return rxfront.Event(t, "cs", f"P1 CS{cs + 1}", protocol=Protocol.PACTOR1, cs=cs)


def _verdict(capsys, seq_sent, breakin_at):
    """The verdict line for a session that alternated but sent these counters."""
    onair._summary(cs_log=[_cs(1.0, 0), _cs(2.0, 1)], tx_slots=[0, 1], cycles=2,
                   keyed=[], cycle_s=1.25, seq_sent=seq_sent,
                   evidence=onair._ConnectEvidence(), ended="test",
                   breakin_at=breakin_at)
    return capsys.readouterr().out.splitlines()[-1]


def test_the_breakin_reset_out_of_3_is_not_an_advance(capsys):
    """#3 -> #0 is a real advance from a data packet and a reset from a break-in.

    The arithmetic cannot tell them apart -- both are +1 mod 4 -- so the test
    that only looked at the numbers would have called a session ACKNOWLEDGED on
    the strength of the yield that ended it. The counter it is asking about only
    moves on an acknowledgement, and the changeover packet moves it for free.
    """
    # Nine repeats of #3 -- the peer asking for the same packet nine times --
    # and then the changeover. Nothing was ever acknowledged.
    assert "NOT met" in _verdict(capsys, [3] * 9 + [0], breakin_at={9})
    # The same numbers, where #0 rode a data packet: the peer took #3 and we
    # wrapped. That is the advance, and it still reads as one.
    assert "ACKNOWLEDGED" in _verdict(capsys, [3] * 9 + [0], breakin_at=set())


def test_a_breakin_does_not_mask_an_advance_that_happened_before_it(capsys):
    """Only the step INTO the break-in is excluded, not the session around it."""
    assert "ACKNOWLEDGED" in _verdict(capsys, [1, 2, 0], breakin_at={2})
