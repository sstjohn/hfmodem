# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A P1 answer received during a hush must not retime the caller's first DATA."""
import json
from pathlib import Path
import re

import numpy as np
import pytest

from hfmodem.tests.shrike.recorded_pcm import recorded_pcm

from hfmodem.shrike import onair, pactor1, ptc, spec
from hfmodem.tests.shrike.test_grantslot import _run
from hfmodem.tests.shrike.test_p3_offer import Keyed

FS = onair.FS
HUSH_PATH = Path(__file__).with_name("fixtures") / "connected-hush-0909.json"


def grid(anchor):
    return onair._MasterGrid(anchor, 60000, round(.185 * FS),
                            packet_n=round(spec.P1_PACKET_S * FS),
                            cs_n=round(spec.P1_CS_S * FS),
                            d_max_n=onair._d_max_n(1.25, .04))


@pytest.mark.parametrize("anchor,onset,slot", [
    (2737, 472416, 9),    # KB5LZK 19:36: original d=74.979 ms; moved -30 ms.
    (2730, 1554384, 27),  # N5TW 20:03: original d=116.125 ms; moved +11 ms.
])
def test_connected_hush_preserves_the_recorded_call_grid(anchor, onset, slot):
    g = grid(anchor)
    g.hush_left = 4
    due = g.boundary(slot)
    line = g.update([onset], hushed=True, since_tx=60000, linked=True)
    assert g.boundary(slot) == due
    assert g.anchor == anchor
    assert g.hush_left == 0
    assert g.acquired
    assert g.d_n == (onset-anchor-g.p1_data_n) % g.slot_n
    assert "GRID PLACED" not in line


@pytest.mark.parametrize("onsets", [[], [10000]])
def test_a_connected_link_cancels_remaining_hush_even_without_an_in_band_onset(onsets):
    g = grid(0)
    g.hush_left = 5
    g.update(onsets, hushed=True, since_tx=60000, linked=True)
    assert g.anchor == 0
    assert g.hush_left == 0


def test_unanswered_hush_can_still_place_the_grid():
    g = grid(2737)
    g.hush_left = 4
    line = g.update([472416], hushed=True, since_tx=60000, linked=False)
    assert g.anchor == 1296
    assert "GRID PLACED" in line


def test_first_hush_window_does_not_rephase_an_answer_to_the_last_call():
    g = grid(2737)
    g.hush_left = 4
    g.update([472416], hushed=True, since_tx=0, linked=False)
    assert g.anchor == 2737


@pytest.mark.skipif(not HUSH_PATH.exists(),
                    reason=f"the recorded connect answers {HUSH_PATH.name} "
                           "are not installed")
@pytest.mark.parametrize("index", [0, 1])
def test_recorded_connect_answer_ends_hush_on_the_original_grid(index):
    row = json.loads(HUSH_PATH.read_text())[index]
    audio = recorded_pcm(row)
    event = onair._SessionRx._p1_cs(audio, row["onset"] - row["start"])
    assert event is not None and event.cs == row["cs"]
    host = ptc.PtcHost(peer=Keyed(), mycall="W9SSJ")
    host.arq.on_host_connect("W9SSJ", row["call"])
    g = grid(row["anchor"])
    g.hush_left = 4
    host.on_rx_event(event)
    assert host.arq.state == onair.State.CONNECTED
    g.update([row["onset"]], hushed=True, since_tx=60000,
             linked=host.arq.state in onair.LINKED)
    assert g.boundary(row["first_data_slot"]) == (
        row["anchor"] + 60000 * row["first_data_slot"])
    assert g.hush_left == 0


@pytest.mark.parametrize("turnaround", [.075, .116125])
def test_session_loop_keeps_first_data_on_call_grid(tmp_path, monkeypatch, turnaround):
    # A late response starts while the caller is listening in a hush. The peer
    # holds its own phase throughout; its packets do not move with our scheduler.
    audio = np.zeros(45 * 60000, np.float32)
    for k in range(6, 30):
        answer = onair._trim_silence(pactor1.control_signal(
            pactor1.CS_SPEED, invert=k % 2)).astype(np.float32)
        at = k * 60000 + 46080 + round(turnaround * FS)
        audio[at:at + len(answer)] = answer
    wav = tmp_path / "late-peer.wav"
    onair.session.write_wav(str(wav), audio)
    anchors = []
    send = onair.RadioTx.send_p1_packet

    def packet(tx, *args, **kwargs):
        anchors.append(tx.raster.anchor)
        return send(tx, *args, **kwargs)

    monkeypatch.setattr(onair.RadioTx, "send_p1_packet", packet)
    log = _run(wav, tmp_path / "out", "--max-cycles", "16", "--retries", "20",
               "--pactor1-only", hold=4)
    assert "HUSHED, not keying" in log
    assert "** CONNECTED to" in log
    initial = int(re.search(r"master grid: anchor @ sample (\d+)", log)[1])
    assert anchors and set(anchors) == {initial}
