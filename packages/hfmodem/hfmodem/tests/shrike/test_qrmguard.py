# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The sending side's refusal, against the third-party receiver that saw it.

On 2026-08-26 this station transmitted over gateway WS8EOC's control signal
twenty times in one session. `_MasterGrid.key_refusal` -- the only code in the
keying path that can refuse a transmission -- returned on its first line every
one of those cycles, because we were the ISS for every keyed cycle of that
grant. Our own recording cannot show the overlap at all: the rig mutes the
receiver while we key, so the collided bursts are the bursts missing from the
file, and `[collide]` printed zero times over twenty real collisions.

THE FIGURES BELOW ARE NOT OURS. `captures/onair-0826-1041/witness.wav` is a
KiwiSDR at Empire, Michigan, 103 mi away, with both stations on one clock. Our
SL1 waveform sits at 1083/1916 Hz and WS8EOC's control signal at 1395/1595, so
the two separate with no cross-recording alignment and no assumption about
either station's grid; the reported precision is +/-5 ms. The gateway held
1249.99 +/- 0.44 ms across twenty-five consecutive control signals with none
missing.

So this is not a round-trip. Each row is a keying instant and a gateway onset
that a third party measured, fed to the production guard, and the overlap the
guard reports is checked against the overlap the witness measured. The four
`template` cycles it must pass and the six `burst` cycles it must refuse are the
same rungs of the same session.

Run:  python -m pytest hfmodem/tests/shrike/test_qrmguard.py
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

from hfmodem.shrike.onair import (BREAKIN_LEAD_S, FS, ISS_GUARD_CYCLES,
                                  Protocol, RadioTx, _forecast_next_key,
                                  _MasterGrid)

CYCLE_S = 1.25
SLOT_N = round(CYCLE_S * FS)
PACKET_N = round(0.960 * FS)
CS_N = round(0.120 * FS)
SETTLE_S = 0.040

#: The gateway's own codeword, witness-measured at ~130 ms.
GW_CS_S = 0.130
#: Its raster, from twenty-five consecutive control signals.
GW_RASTER_S = 1.24999

#: The `burst` rung as rendered -- the entry packet behind `placement`'s 200 ms
#: acquisition preamble -- plus this station's settle.
BURST_AIR_S = 0.040 + 1.074

#: (tag, our carrier down, the gateway's next codeword up, the overlap the
#: witness measured). The `burst` rung, six cycles of six -- every one of them
#: landed on the gateway's answer.
COLLIDED = (
    ("TX[20]", 54.621, 54.601, 20),
    ("TX[21]", 57.137, 57.102, 35),
    ("TX[22]", 59.656, 59.602, 54),
    ("TX[23]", 62.170, 62.102, 68),
    ("TX[24]", 65.929, 65.851, 78),
    ("TX[25]", 68.444, 68.351, 93),
)

#: The `template` rung, four of four, clear at both ends: the gateway's previous
#: codeword ended 113-115 ms before our carrier came up and our carrier dropped
#: 182-184 ms before its next one began.
CLEAR = (
    ("TX[16]", 49.419, 49.602, 113),
    ("TX[17]", 50.669, 50.852, 113),
    ("TX[18]", 51.920, 52.102, 114),
    ("TX[19]", 53.167, 53.351, 115),
)

#: After the PACTOR-1 fallback, our 960 ms packet against the same raster. The
#: log read `HOLD RX (quiet)` and `NO CONTROL SIGNAL -- nothing heard` on every
#: one of these cycles while the gateway was answering.
FALLBACK = (
    ("TX[29]", 77.444, 78.404, 78.352, 52),
    ("TX[30]", 78.696, 79.656, 79.602, 54),
    ("TX[31]", 79.948, 80.908, 80.852, 56),
    ("TX[32]", 81.206, 82.166, 82.102, 64),
    ("TX[33]", 82.446, 83.406, 83.352, 54),
    ("TX[34]", 83.704, 84.664, 84.602, 62),
)


def _sending_grid(cs_up_s: float, *, who: str = "WS8EOC") -> _MasterGrid:
    """A grid we are the ISS on, carrying the codeword one cycle before `cs_up_s`.

    The projection is one cycle, so the word recorded is the one the modem would
    actually have decoded in the window it just closed.
    """
    g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                    packet_n=PACKET_N, cs_n=CS_N, d_max_n=round(0.130 * FS))
    g.sending = True
    g.note_peer_codeword(round((cs_up_s - GW_RASTER_S) * FS),
                         round(GW_CS_S * FS), "0x59A", who)
    return g


def test_every_burst_rung_cycle_the_witness_saw_is_refused():
    """Six of six, and the guard's own figure is the witness's own figure."""
    for tag, down_s, cs_up_s, overlap_ms in COLLIDED:
        g = _sending_grid(cs_up_s)
        carrier = round((down_s - BURST_AIR_S) * FS)
        why = g.key_refusal(carrier, round(BURST_AIR_S * FS))
        assert why is not None, f"{tag} keyed over the gateway and was allowed"
        assert "0x59A" in why and "WS8EOC" in why, f"{tag}: {why}"
        assert f"{overlap_ms} ms" in why, f"{tag}: witness {overlap_ms}, {why}"


def test_no_template_rung_cycle_is_refused():
    """Four of four. The rung that never collided must not lose a cycle."""
    for tag, down_s, cs_up_s, front_ms in CLEAR:
        g = _sending_grid(cs_up_s)
        carrier = round((cs_up_s - GW_RASTER_S + GW_CS_S
                         + front_ms / 1e3) * FS)
        air_n = round(down_s * FS) - carrier
        assert g.key_refusal(carrier, air_n) is None, tag


def test_the_fallback_collisions_are_refused_too():
    """Six more the gateway answered and we sat on top of, at PACTOR-1."""
    for tag, key_s, end_s, cs_up_s, overlap_ms in FALLBACK:
        g = _sending_grid(cs_up_s)
        carrier = round(key_s * FS)
        why = g.key_refusal(carrier, round((end_s - key_s) * FS))
        assert why is not None, tag
        assert f"{overlap_ms} ms" in why, f"{tag}: witness {overlap_ms}, {why}"


def test_the_band_is_the_grids_own_acquisition_bound():
    """A 20 ms courtesy margin on top of this would close the workable turnaround
    at 110 ms, and 110 ms is not free: two of the ninety-three turnarounds this
    station has acquired sit outside it -- WS8EOC's 114.0 of 2026-08-14,
    corroborated in its own cycle 12, and WM4RB's 121.7 the same night, both at
    this station's 40 ms settle. So the condition is overlap and nothing else,
    which puts the band at `_d_max_n` and refuses nothing the grid was willing to
    call a turnaround. `test_ackfloor` holds the receiving side to the same
    band, off the same logs."""
    air_n = round((SETTLE_S + 0.960) * FS)
    for d_ms in (92.6, 103.1, 114.0, 121.7, 130.0):
        # Our data ends a packet after the boundary; the peer answers `d` later;
        # our next carrier comes up a settle before the next boundary.
        cs_at = PACKET_N + round(d_ms / 1e3 * FS)
        g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                        packet_n=PACKET_N, cs_n=CS_N, d_max_n=round(0.130 * FS))
        g.sending = True
        g.note_peer_codeword(cs_at, CS_N, "0x59A", "WS8EOC")
        carrier = SLOT_N - round(SETTLE_S * FS)
        assert g.key_refusal(carrier, air_n) is None, d_ms
    # ...and one past it, where the codeword is genuinely still on the air.
    g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                    packet_n=PACKET_N, cs_n=CS_N, d_max_n=round(0.130 * FS))
    g.sending = True
    g.note_peer_codeword(PACKET_N + round(0.140 * FS), CS_N, "0x59A", "WS8EOC")
    why = g.key_refusal(SLOT_N - round(SETTLE_S * FS), air_n)
    assert why is not None and "still on the air" in why, why


def test_nothing_decoded_refuses_nothing():
    """The guard has no opinion without a word, which is every cycle of every
    link where the peer was never read. An energy onset is not enough: only
    `note_peer_codeword` writes here, and only a zero-error decode calls it."""
    g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                    packet_n=PACKET_N, cs_n=CS_N, d_max_n=round(0.130 * FS))
    g.sending = True
    g.peer_onset = PACKET_N
    assert g.key_refusal(SLOT_N, round(BURST_AIR_S * FS)) is None


def test_a_stale_codeword_stops_refusing():
    tag, down_s, cs_up_s, _ = COLLIDED[0]
    g = _sending_grid(cs_up_s)
    carrier = round((down_s - BURST_AIR_S) * FS)
    air_n = round(BURST_AIR_S * FS)
    g.cycles = ISS_GUARD_CYCLES
    assert g.key_refusal(carrier, air_n) is not None, tag
    g.cycles = ISS_GUARD_CYCLES + 1
    assert g.key_refusal(carrier, air_n) is None, tag


#: `captures/onair-0819-2210`, K4MSU's changeover session, read off the replay:
#: the codeword the grid places 124 ms inside our own packet, and the interval
#: that packet actually occupied.
K4MSU_CS_AT = 2393040
K4MSU_KEYING = (2350992, 2398992)


class _Ev:
    def __init__(self, at: int) -> None:
        self.t, self.cs, self.spare = at / FS, 0, 0
        self.protocol = Protocol.PACTOR1


class _Rx:
    def __init__(self, at: int) -> None:
        self.cs_log = [_Ev(at)]


def _forecast(keyings: list[tuple[int, int]]) -> _MasterGrid:
    """One forecast over one decoded codeword, with `keyings` as our record."""
    g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=round(0.185 * FS),
                    packet_n=PACKET_N, cs_n=CS_N, d_max_n=round(0.130 * FS))
    g.sending = True
    tx = RadioTx(None, transmit=False, outdir=Path("."), settle=SETTLE_S)
    tx.keyings = list(keyings)
    tx.boundary = K4MSU_KEYING[0] + SLOT_N + round(SETTLE_S * FS)
    tx.tx_key_up = K4MSU_KEYING[0] + SLOT_N
    with contextlib.redirect_stdout(io.StringIO()):
        _forecast_next_key(_Rx(K4MSU_CS_AT), tx, g, 0)
    return g


def test_a_codeword_we_read_cannot_have_been_under_our_own_carrier():
    """The rig mutes this receiver while we key, so a word decoded at zero errors
    is proof the transmitter was off while it was on the air. Where the grid puts
    one inside a transmission we actually made, the grid is out of step with the
    file and its projection is not evidence -- `captures/onair-0819-2210` is a
    whole session of exactly that, every codeword clean and every one of them
    placed 124 ms inside our own packet."""
    assert _forecast([K4MSU_KEYING]).peer_cs is None
    kept = _forecast([(K4MSU_KEYING[0] - SLOT_N, K4MSU_KEYING[1] - SLOT_N)])
    assert kept.peer_cs is not None and kept.peer_cs.at == K4MSU_CS_AT


def _refusals(guard: bool, n: int) -> tuple[list[bool], str]:
    """`n` bursts into a grid that refuses every one of them."""
    tag, down_s, cs_up_s, _ = COLLIDED[0]
    tx = RadioTx(None, transmit=False, outdir=Path("."), settle=SETTLE_S)
    tx.raster = _sending_grid(cs_up_s)
    tx.qrm_guard = guard
    carrier = round((down_s - BURST_AIR_S) * FS)
    air_n = round(BURST_AIR_S * FS)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        dropped = [tx._refused(carrier, air_n, "SL1 ENTRY 0B") for _ in range(n)]
    return dropped, log.getvalue()


def test_the_guard_cannot_deadlock_the_link():
    """A peer stuck transmitting on our raster keeps the codeword fresh every
    cycle, so freshness alone never lifts the refusal. The bound is consecutive
    drops: three, then one burst goes and the line says why."""
    dropped, log = _refusals(True, 9)
    assert dropped == [True, True, True, False] * 2 + [True], dropped
    assert log.count("STOOD DOWN") == 2, log
    assert "out of step with the peer's raster" in log


def test_a_refusal_says_what_it_declined_and_how_to_stop_it():
    dropped, log = _refusals(True, 1)
    line = log.strip()
    assert dropped == [True]
    assert line.count("\n") == 0, "a refusal is one line"
    for token in ("QRM GUARD", "0x59A", "WS8EOC", "SL1 ENTRY 0B",
                  "1 of 3 in a row", "--no-qrm-guard"):
        assert token in line, f"{token} missing from: {line}"


def test_the_off_switch_is_on_the_command_line():
    """A two-way door is only one if the operator can find it. The refusal line
    names `--no-qrm-guard`; this is the assertion that the name is real."""
    r = subprocess.run([sys.executable, "-m", "hfmodem.shrike.onair", "--help"],
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, "COLUMNS": "200", "NO_COLOR": "1"})
    assert r.returncode == 0, r.stderr
    assert "--no-qrm-guard" in r.stdout


def test_the_off_switch_keys_and_still_says_what_it_would_have_refused():
    """Two-way door. Off is one token, and off is not silent -- an operator who
    turned the guard off can still see every cycle it would have stopped."""
    dropped, log = _refusals(False, 4)
    assert dropped == [False] * 4
    assert log.count("QRM GUARD OFF") == 4
    assert "0x59A" in log and "--no-qrm-guard" in log


# --------------------------------------------------------------------------
# the one burst the fit test cannot be asked of
# --------------------------------------------------------------------------
def _changeover_grid(onset: int) -> _MasterGrid:
    """A grid we are the ISS on, holding the CS3 head of the peer's packet."""
    g = _MasterGrid(anchor=0, slot_n=SLOT_N, offset_n=0, packet_n=PACKET_N,
                    cs_n=CS_N, d_max_n=round(0.130 * FS))
    g.sending = True
    g.note_peer_codeword(onset, CS_N, "CS3/break-in", "KB5LZK")
    return g


def test_the_changeover_is_refused_only_by_the_packet_it_answers():
    """Taking the link IS keying into the slot the peer would have used.

    The reference modems' break-in begins 71-72 ms after our packet ends and
    runs 960 ms, so it lies across 742 ms of the packet we would have sent next
    -- and nothing collides, because a station that reads the CS3 head in its
    own answer window cancels that transmission. The fit against the peer's next
    projection cannot be asked of this burst. What remains is the packet it is
    answering, of which a decoded codeword is the first 120 ms.
    """
    onset = 4 * SLOT_N
    g = _changeover_grid(onset)
    settle_n = round(SETTLE_S * FS)
    air_n = settle_n + PACKET_N
    carrier = onset + PACKET_N + round(BREAKIN_LEAD_S * FS) - settle_n

    assert g.key_refusal(carrier, air_n, changeover=True) is None
    ordinary = g.key_refusal(carrier, air_n)
    assert ordinary is not None and "before our" in ordinary, ordinary

    # ...and the head is the head OF A PACKET: asked as an ordinary burst the
    # guard sees 120 ms where the peer is holding the channel for 960.
    inside = onset + PACKET_N // 2
    why = g.key_refusal(inside, air_n, changeover=True)
    assert why is not None and "still on the air" in why, why
    assert g.key_refusal(inside, round(0.001 * FS)) is None


def test_the_changeover_never_stands_down_into_the_peers_packet():
    """The release is for a grid out of step with the peer's raster. This burst
    is placed on that raster by construction, so a refusal means the placement
    could not be made -- and keying anyway is the one thing it must not do. On
    2026-09-03 the stand-down let seven of them out at 656-667 ms into the
    packet they were answering, and the gateway answered none."""
    tx = RadioTx(None, transmit=False, outdir=Path("."), settle=SETTLE_S)
    tx.raster = _changeover_grid(4 * SLOT_N)
    carrier = 4 * SLOT_N + PACKET_N // 2
    air_n = round(SETTLE_S * FS) + PACKET_N
    with contextlib.redirect_stdout(io.StringIO()) as log:
        dropped = [tx._refused(carrier, air_n, "P1 BREAK-IN #0 200Bd 0B",
                               changeover=True)
                   for _ in range(3 * 3)]
    assert dropped == [True] * 9, dropped
    assert "STOOD DOWN" not in log.getvalue()
    assert log.getvalue().count("CHANGEOVER NOT PLACED") == 9
    assert "still on the air" in log.getvalue()
