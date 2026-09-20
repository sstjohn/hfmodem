# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Native lower BW2750 recovery: measured tables, exact host bytes, live RX."""
import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.rx import varahf2300 as rx
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as va

FIXTURES = Path(__file__).with_name("fixtures") / "bw2750-low-levels"
NAMES = ("record1-v0", "record1-v1", "record2-v0", "record2-v1",
         "record2-empty", "record2-training-close")


@cache
def _metadata():
    path = FIXTURES / "provenance.json"
    if not path.exists():
        pytest.skip(f"native BW2750 low-level metadata absent: {path}")
    return json.loads(path.read_text())


@cache
def _audio(name):
    path = FIXTURES / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"native BW2750 low-level recording absent: {path}")
    rate, x = wavfile.read(path)
    assert rate == 48000
    return np.asarray(x, float)


@pytest.mark.parametrize("name", NAMES)
def test_native_lower_record_crc_and_exact_host_bytes(name):
    meta, x = _metadata()[name], _audio(name)
    record = meta["record"]
    fr = rx.decode_over(x, 0, len(x), level=record)
    assert fr.crc_ok
    assert fr.frame_bytes.hex() == meta["decoded_frame_hex"]
    assert phy.vara_payload(fr.payload, caller="W9SSJ", body_len=len(fr.payload)).hex() == meta["expected_payload_hex"]
    scores = {lv: (hits, count) for lv, hits, count, _ in
              rx.index_guard(x, (record, record - 100, 103))}
    assert scores[record] == (24, 24)
    assert scores[record - 100][0] <= 9
    assert scores[103][0] <= 9


class _IO(va.VaraIO):
    def __init__(self):
        self.payloads, self.answers, self.logs = [], [], []
        self.received = 0
    def key(self, on):
        if on:
            self.answers.append(self.received)
    def tx(self, samples): pass
    def pending(self): pass
    def data(self, payload): self.payloads.append(bytes(payload))
    def connected(self, *args): pass
    def log(self, message): self.logs.append(message)


@pytest.mark.parametrize("name", ("record1-v0", "record1-v1", "record2-v0", "record2-v1"))
def test_native_low_record_stream_delivers_once_and_answers_after_body(name):
    meta, x = _metadata()[name], _audio(name)
    io = _IO()
    hs = va.VaraStationHandshake(["W9SSJ"], io, bw="2750")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "KC9GHZ"
    hs.state, hs.step = va.VaraState.CONNECTED, va._I_CONNECTED
    hs.turn = va._TURN_PEER
    samples = np.concatenate([x, np.zeros(24000)])
    for at in range(0, len(samples), 4800):
        io.received = min(at + 4800, len(samples))
        hs.on_rx_stream(samples[at:at + 4800])
    assert io.payloads == [bytes.fromhex(meta["expected_payload_hex"])], io.logs
    assert len(io.answers) == 1, io.logs
    geometry = meta["geometry"]
    body_end = geometry["phase"] + (geometry["live_indices"][-1] + 1) * 1024
    # Native waveform end agrees with this grid to within four samples. A
    # callback can discover the answer late, but must never answer before the
    # frame ends. The four stock crops answer 0.306..0.369 s after the end.
    active_end = int(np.flatnonzero(np.abs(x) > 1e-8)[-1] + 1)
    assert abs(active_end - body_end) <= 8
    assert max(body_end, active_end) <= io.answers[0] <= body_end + 24000, io.logs
    assert hs.turn == va._TURN_PEER


def test_new_receive_records_do_not_extend_transmit_selectors():
    assert rx.BASE_LEVELS["2750"] == 103
    assert rx.INDEX_LEVELS_BW["2750"] == (100, 101, 102, 103)
    for level in (101, 102):
        assert level not in rx.LEVELS
        assert level not in rx.KEYABLE_LEVELS
        assert level not in phy.speed_ladder("2750")
    assert rx.INDEX_LEVELS_BW["2300"] == (0, 1, 2, 3)
