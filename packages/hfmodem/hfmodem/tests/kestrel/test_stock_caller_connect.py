# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Replay the recorded stock caller's retry and low-level confirm sequences."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.evidence import require_input

FIXTURES = Path(__file__).with_name("fixtures") / "bw500-stock-caller-1018"


class IO(VA.VaraIO):
    def __init__(self):
        self.lines, self.keyed = [], []

    def tx(self, samples):
        self.keyed.append(np.asarray(samples))

    def log(self, msg):
        self.lines.append(msg)


def station(bw="500", calls=("W9SSJ",)):
    io = IO()
    hs = VA.VaraStationHandshake(calls, io, bw=bw, high_speed=True,
                                 compression="TEXT")
    hs.listen(True)
    return hs, io


def audio(name):
    rate, x = wavfile.read(require_input(FIXTURES / f"{name}.wav"))
    assert rate == MK.FS and x.dtype == np.int16
    return x.astype(float) / 32767


def test_recorded_retry_request_draws_a_second_connect_response():
    hs, io = station()
    hs.on_rx_audio(audio("chat-s2k-request"))
    assert len(io.keyed) == 1, io.lines
    hs.on_rx_audio(audio("chat-s2k-retry"))
    assert len(io.keyed) == 2, io.lines
    want = MK.synth_burst("W9SSJ", VF.connect_response("500"))
    np.testing.assert_allclose(io.keyed[-1], want)


def test_recorded_setup_and_confirm_draw_the_link_control_ack():
    hs, io = station()
    hs.on_rx_audio(audio("chat-s2k-retry-request"))
    hs.on_rx_audio(audio("chat-s2k-retry-link-setup"))
    assert hs.state is VA.VaraState.CONNECTED and hs.caller == "WI9YZA", io.lines
    before = len(io.keyed)
    hs.on_rx_audio(audio("chat-s2k-retry-confirm"))
    assert len(io.keyed) == before + 1, io.lines
    want = MK.synth_tone_pairs(VF.control_bursts("WI9YZA", "500", 0)[1])
    np.testing.assert_allclose(io.keyed[-1], want)
    assert hs._rec500 == VA._phy.level_key("500", 1)


def feed(hs, io, samples, chunk=1024):
    before = len(io.keyed)
    for at in range(0, len(samples), chunk):
        hs.on_rx_stream(samples[at:at + chunk])
        if len(io.keyed) != before:
            return min(at + chunk, len(samples))
    return None


@pytest.mark.parametrize("chunk,deadlines", [(1024, (91136, 72704)), (4096, (94208, 73728))])
def test_recorded_requests_answer_on_the_stream_before_the_retry_deadline(chunk, deadlines):
    hs, io = station()
    for (name, duration), deadline in zip(
            [("chat-s2k-request", 1.755), ("chat-s2k-retry", 1.370)], deadlines):
        recorded = audio(name)
        accepted = feed(hs, io, np.r_[recorded, np.zeros(MK.FS // 4)], chunk)
        assert accepted is not None, io.lines
        assert accepted <= deadline, (name, accepted, deadline, io.lines)
        delay = accepted / MK.FS - (.1 + duration)
        assert 0 <= delay <= chunk / MK.FS + .001, (name, delay, io.lines)
        assert delay + len(io.keyed[-1]) / MK.FS < 1.295 - .12
        before = len(io.keyed)
        hs.on_rx_audio(recorded)
        assert len(io.keyed) == before, io.lines
    assert len(io.keyed) == 2


def test_recorded_retry_can_be_the_first_request_heard():
    hs, io = station()
    hs.on_rx_audio(audio("chat-s2k-retry"))
    assert hs.state == VA.VaraState.CONNECTING and hs.called == "W9SSJ", io.lines
    assert len(io.keyed) == 1


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_retry_brackets_answer_both_a_new_and_a_pending_call(bw):
    hs, io = station(bw)
    burst = MK.synth_burst("W9SSJ", VF.connect_request_retry(bw))
    for _ in range(2):
        hs.on_rx_audio(np.r_[np.zeros(4800), burst, np.zeros(4800)])
    assert len(io.keyed) == 2, io.lines
    assert hs.called == "W9SSJ" and hs.state == VA.VaraState.CONNECTING


def test_keepalive_a_draws_control_one():
    hs, io = station()
    hs.role, hs.called, hs.caller = "responder", "W9SSJ", "WI9YZA"
    hs.state = VA.VaraState.CONNECTED
    burst = MK.synth_burst("W9SSJ", VF.for_bw(VF.SESSION_KEEPALIVE_A, "500"))
    hs.on_rx_audio(np.r_[np.zeros(4800), burst, np.zeros(4800)])
    assert len(io.keyed) == 1, io.lines
    np.testing.assert_allclose(io.keyed[0], MK.synth_tone_pairs(
        VF.control_bursts("WI9YZA", "500", 1)[1]))
    reply = MK.synth_burst("W9SSJ", VF.for_bw(VF.SESSION_KEEPALIVE_B, "500"))
    hs.on_rx_audio(np.r_[np.zeros(4800), reply, np.zeros(4800)])
    assert len(io.keyed) == 1, io.lines


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("frame", range(6))
def test_every_confirm_frame_sets_the_start_level_and_keys_control_zero(bw, frame):
    hs, io = station(bw)
    hs.role, hs.called, hs.caller = "responder", "W9SSJ", "WI9YZA"
    hs.state = VA.VaraState.CONNECTED
    hs._link_high_speed = True
    kind = replace(VF.SESSION_CONFIRM, preadv=1 + 30 * frame)
    burst = MK.synth_burst("W9SSJ", VF.for_bw(kind, bw))
    hs.on_rx_audio(np.r_[np.zeros(4800), burst, np.zeros(4800)])
    assert len(io.keyed) == 1, io.lines
    np.testing.assert_allclose(io.keyed[0], MK.synth_tone_pairs(
        VF.control_bursts("WI9YZA", bw, 0)[1]))
    assert hs._rec500 == VA._phy.level_key(bw, frame + 1)


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("retry", [False, True])
def test_stream_request_waits_for_the_tail_and_does_not_answer_it_twice(bw, retry):
    hs, io = station(bw)
    kind = VF.connect_request_retry(bw) if retry else VF.connect_request(bw)
    request = MK.synth_burst("W9SSJ", kind)
    assert feed(hs, io, request[:-MK.HOP]) is None
    assert not io.keyed
    assert feed(hs, io, np.r_[request[-MK.HOP:], np.zeros(MK.FS // 4)]) is not None
    assert len(io.keyed) == 1, io.lines
    assert feed(hs, io, np.zeros(MK.FS)) is None
    hs.on_rx_audio(request)
    assert len(io.keyed) == 1


@pytest.mark.parametrize("call,kind", [
    ("K7ABC", VF.connect_request_retry("500")),
    ("W9SSJ", VF.SESSION_KEEPALIVE_A),
    ("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER),
])
def test_stream_rejects_other_calls_and_session_frames(call, kind):
    hs, io = station()
    burst = MK.synth_burst(call, VF.for_bw(kind, "500"))
    assert feed(hs, io, np.r_[burst, np.zeros(MK.FS // 4)]) is None
    assert not io.keyed


def test_stream_ignores_stale_requests_and_clears_audio_on_keying():
    hs, io = station()
    request = MK.synth_burst("W9SSJ", VF.connect_request_retry("500"))
    hs.on_rx_stream(np.r_[request, np.zeros(2 * MK.FS)])
    assert not io.keyed
    feed(hs, io, request[:MK.FS])
    hs._key(True)
    hs._key(False)
    assert feed(hs, io, np.r_[request[MK.FS:], np.zeros(MK.FS // 4)]) is None
    assert not io.keyed
    assert feed(hs, io, np.random.default_rng(1018).normal(0, .1, 4 * MK.FS)) is None
    assert len(hs._connect_request_buf) <= VA._span(VF.connect_request("500")) + 6 * MK.HOP


@pytest.mark.parametrize("retry", [False, True])
def test_initiator_does_not_take_a_stream_request_addressed_to_its_own_call(retry):
    hs, io = station()
    hs.originate("K7ABC")
    before = len(io.keyed)
    kind = VF.connect_request_retry("500") if retry else VF.connect_request("500")
    burst = MK.synth_burst("W9SSJ", kind)
    assert feed(hs, io, np.r_[burst, np.zeros(MK.FS // 4)]) is None
    assert len(io.keyed) == before
    assert hs.role == "initiator" and hs.step == VA._I_CR_SENT


@pytest.mark.parametrize("bw", ["500", "2300"])
@pytest.mark.parametrize("calls", [["W9SSJ"], ["W9SSJ", "K7ABC", "W1AW"]])
def test_noise_transforms_each_window_once_and_never_runs_the_payload_fit(monkeypatch, bw, calls):
    hs = VA.VaraStationHandshake(calls, IO(), bw=bw)
    hs.listen(True)
    windows = []
    transform = VA._top3_track

    def counted(x, *args, **kwargs):
        track = transform(x, *args, **kwargs)
        windows.append(len(track[0]))
        return track

    def no_fit(*args, **kwargs):
        pytest.fail("noise reached the expensive request fit")

    monkeypatch.setattr(VA, "_top3_track", counted)
    monkeypatch.setattr(VA, "_payload_fit", no_fit)
    noise = np.random.default_rng(1018).normal(0, .05, 30 * MK.FS)
    assert feed(hs, hs.io, noise) is None
    assert windows and sum(windows) <= len(noise) // VA._ACK_GRID
    assert max(windows[1:]) <= (VA._TURNAROUND_STEP + 1024) // VA._ACK_GRID


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("stall,answered", [(.1, True), (.3, False)])
def test_one_large_chunk_answers_a_recent_request_or_waits_for_the_retry(bw, retry, stall, answered):
    hs, io = station(bw)
    kind = VF.connect_request_retry(bw) if retry else VF.connect_request(bw)
    request = MK.synth_burst("W9SSJ", kind)
    chunk = np.r_[request, np.zeros(round(stall * MK.FS))]
    hs.on_rx_stream(chunk)
    assert len(io.keyed) == int(answered), io.lines
    hs.on_rx_audio(chunk)
    assert len(io.keyed) == int(answered), io.lines
    if not answered:
        retry = MK.synth_burst("W9SSJ", VF.connect_request_retry(bw))
        assert feed(hs, io, np.r_[retry, np.zeros(MK.FS // 4)]) is not None, io.lines
        hs.on_rx_audio(retry)
        assert len(io.keyed) == 1, io.lines


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("shift", VA._RESP_SHIFTS)
@pytest.mark.parametrize("chunk", [1003, 5003])
def test_incremental_request_track_handles_unaligned_blocks_and_frequency_shifts(bw, shift, chunk):
    hs, io = station(bw, ("K7ABC", "W1AW", "W9SSJ"))
    for kind in (VF.connect_request(bw), VF.connect_request_retry(bw)):
        tones = np.asarray(VF.handshake_tones("W9SSJ", kind)) + shift
        request = MK.synth_tones(tones)
        samples = np.r_[np.zeros(137), request, np.zeros(MK.FS // 4)]
        assert feed(hs, io, samples, chunk) is not None, io.lines
        before = len(io.keyed)
        hs.on_rx_audio(samples)
        assert len(io.keyed) == before, io.lines
    assert len(io.keyed) == 2, io.lines
