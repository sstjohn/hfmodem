# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A control answering a packet leads on the packet's own carrier: read off the air.

VE3KPG 2026-09-20 09:36, our ISS stint: each peer control starts 938-943 ms
after the audio start of the packet it answers. Paired that way, 127 of 130
readable controls (SL1-SL6) lead on the SAME carrier as the packet; the
"opposite" count in `working/p3-cs-stagger-ab-0922` pairs each control with
the packet keyed after it, and our packets alternate every cycle.
"""
import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

from hfmodem.shrike import arq, placement
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_p3_reply_placement import (
    CYCLE, FS, IDENTITY, PACKET_N, REPLY_N, driver, irs_grid)

ARM = evidence.CAPTURES / "onair-0920-0936"
OUR_PACKET = "tx_36"             # SL2 pkt, audio start 3603871
PEER_ANSWER = 3648924            # the NAK the log decoded there, 939 ms on


def lead_of(x: np.ndarray) -> str:
    """Which of the two tones rises first, by the half-level edge of each tone's envelope."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    rises = []
    for lo, hi in ((950, 1170), (1800, 2070)):
        band = (f >= lo) & (f <= hi)
        tone = f[band][np.argmax(spectrum[band])]
        sos = butter(4, [tone - 80, tone + 80], btype="band", fs=FS, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, x)))
        body = np.median(env[len(env) // 4: 3 * len(env) // 4])
        rises.append(np.flatnonzero(env > body / 2)[0])
    lead_ms = (rises[0] - rises[1]) / FS * 1e3
    assert abs(lead_ms) > 2, lead_ms
    return "ch5" if lead_ms < 0 else "ch12"


def slice_of(path, start, count):
    with wave.open(str(path)) as w:
        w.setpos(start)
        return np.frombuffer(w.readframes(count), dtype="<i2") / 32768.0


@pytest.mark.skipif(not (ARM / "stream.wav").exists(), reason="capture not on this machine")
def test_the_peer_answers_on_the_packets_carrier_and_so_do_we(tmp_path):
    sidecar = json.loads((ARM / f"{OUR_PACKET}.json").read_text())
    ch5, ch12 = sidecar["pulse_offsets"]
    ours = "ch12" if ch5 > ch12 else "ch5"
    pad = np.zeros(FS // 10)
    with wave.open(str(ARM / f"{OUR_PACKET}.wav")) as w:
        packet = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0
    assert lead_of(np.r_[pad, packet, pad]) == ours == "ch5"
    answer = slice_of(ARM / "stream.wav", PEER_ANSWER - FS // 20,
                      round(0.215 * FS) + FS // 10)
    assert lead_of(answer) == ours

    g = irs_grid()
    at = 1_500_000
    g.note_p3_packet(at, PACKET_N, CYCLE, swapped=ours == "ch12", identity=IDENTITY)
    tx = driver(g, tmp_path)
    tx.host = SimpleNamespace(arq=SimpleNamespace(answering=True))
    swapped = tx._p3_reply_carrier_order(at + REPLY_N, True)
    burst = placement.control_burst(arq.CS_NAK, tail=True, swapped=swapped)
    assert lead_of(np.r_[pad, burst, pad]) == ours
