# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Timing references through a missed answer, using arm 30's sample positions."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, pactor1, ptc, spec


def _arm30_grid():
    fs = onair.FS
    grid = onair._MasterGrid(83367, round(1.25 * fs), 0,
                            packet_n=round(spec.P1_PACKET_S * fs),
                            cs_n=round(spec.P1_CS_S * fs),
                            d_max_n=round(0.130 * fs))
    grid.d_n = 0.0859 * fs
    grid.d_ref_n = grid.data_n
    grid.sending = True
    grid.note_peer_codeword(5012535, grid.cs_n, "CS2/ack", "KB5LZK")
    grid.peer_onset = 5473296
    return grid


def test_missed_answer_does_not_resurrect_an_older_peer_raster():
    grid = _arm30_grid()
    before = grid.peer_packet_end(94)
    grid.update([], linked=True)
    after = grid.peer_packet_end(94)
    assert before == after
    assert after.at == 5473296
    assert grid.breakin_refusal(after) is None
    # The collision observation still expires on the first miss.
    assert grid.peer_onset is None


def test_a_new_burst_can_refresh_the_raster_after_a_missed_answer():
    grid = _arm30_grid()
    grid.update([], linked=True)
    next_at = 5473296 + grid.cycle_n
    assert grid.note_peer_bursts([(next_at, grid.cs_n)]) is not None
    assert grid.peer_at == next_at


def test_retaining_the_raster_does_not_waive_age_or_turnaround_guards():
    grid = _arm30_grid()
    grid.update([], linked=True)
    old = grid.peer_packet_end(110)
    assert old.cycles > onair.ONSET_MAX_CYCLES
    assert grid.breakin_refusal(old) is not None
    for _ in range(grid.MAX_MISSES - 1):
        grid.update([], linked=True)
    assert not grid.locked
    assert grid.breakin_refusal(grid.peer_packet_end(94)) is not None


def test_saved_raster_does_not_keep_a_receiving_collision_guard_alive():
    grid = _arm30_grid()
    grid.sending = False
    assert grid._peer_air() is not None
    grid.update([], linked=True)
    assert grid.peer_at == 5473296
    assert grid._peer_air() is None


@pytest.mark.parametrize("changeover", (False, True))
@pytest.mark.parametrize("baud", (100, 200))
@pytest.mark.parametrize("tail_s", (0.02, 0.10, 0.16))
@pytest.mark.parametrize("backlog", (1, 3, 8))
def test_backlogged_packet_scan_answers_the_current_cycle(changeover, baud, tail_s,
                                                         backlog):
    host = ptc.PtcHost(ptc.SimPeer(), mycall="W9SSJ")
    host.arq.role, host.arq.state = arq.IRS, arq.State.CONNECTED
    host.protocol = spec.Protocol.PACTOR1
    heard = []
    host.on_rx_event = heard.append
    rx = onair._SessionRx(host)
    old = pactor1.packet_signal(b"old", baud=baud, packet_count=1, tail_s=0.24)
    render = pactor1.breakin_signal if changeover else pactor1.packet_signal
    current = render(b"current", baud=baud, packet_count=0 if changeover else 2,
                     invert=True, tail_s=tail_s)
    audio = np.concatenate([old] * backlog + [current])
    # Modest deterministic white noise exercises real demodulation at either
    # baud, with the frame landing at different offsets in the bounded scan.
    audio += np.random.default_rng(906).normal(0, 0.015, audio.size)
    rx.deep_scan(audio)
    assert len(heard) == 1
    assert heard[0].packet[2] == b"current"


def test_old_packet_is_not_acked_when_the_latest_cycle_is_unreadable():
    host = ptc.PtcHost(ptc.SimPeer(), mycall="W9SSJ")
    host.arq.role, host.arq.state = arq.IRS, arq.State.CONNECTED
    host.protocol = spec.Protocol.PACTOR1
    heard = []
    host.on_rx_event = heard.append
    rx = onair._SessionRx(host)
    old = pactor1.packet_signal(b"old", packet_count=1)
    rx.deep_scan(np.concatenate([old, np.zeros(round(1.25 * onair.FS))]))
    assert not heard
