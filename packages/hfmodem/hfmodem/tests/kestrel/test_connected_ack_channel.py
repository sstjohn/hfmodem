# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Step 5 has to survive the channel as well as reject spoofs.

``test_connected_ack`` covers the other half of this: only a genuine ack completes
the connect. The ack is 0.47 s of audio and the burst segmenter in front of the
recogniser cuts at the first falling edge of a single-threshold envelope, so one
fade inside it arrives as two short pieces and one sample of segmentation slop
clips it. Neither may end the attempt.

What the waveform allows is asymmetric, and the tests below say so. The ack's
four fixed preamble symbols are at its head [spec 04 §4.2C]: everything behind
them is session state that changes per session and per bandwidth, so the tail can
fade, be clipped, or carry values we have never seen and the burst is still
recognised — but a fade over the preamble itself takes the whole ack with it.
That case must leave the attempt alive rather than end it, because the gateway
repeats its connect-response when it hears no answer and that repeat is the way
back.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _ACK_PLATEAU,
    _ACK_PRE,
    _ACK_PRE_OFF,
    _I_LINKSETUP_SENT,
    _LINKSETUP_MAX_TX,
    VaraState,
    VaraStationHandshake,
    _ack_plateau,
    _pair_track,
)

FS = MK.FS
_CALLED = "NS0A"
_NPRE = len(VF.CONNECTED_ACK_PREAMBLE)
_ACK_LEN = (VF.CONNECTED_ACK_NSYM - 1) * MK.HOP + MK.STRIDE


class _IO:
    def __init__(self):
        self.log_lines: list[str] = []
        self.tx_bursts = 0

    def key(self, on): ...
    def tx(self, samples): self.tx_bursts += 1
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): self.log_lines.append(msg)


def _awaiting_ack(io=None, called: str = _CALLED) -> VaraStationHandshake:
    hs = VaraStationHandshake(["W9SSJ"], io or _IO(), bw="2300", mfsk_only=True)
    hs.originate(called, "W9SSJ")
    hs.step = _I_LINKSETUP_SENT
    return hs


def _segments(audio, min_s: float = 0.2) -> list[np.ndarray]:
    """The envelope cut the connect tool performs, over a fixed recording.

    Mirrors ``AudioVaraIO.next_rx_burst``: one global threshold at 0.2 x the peak
    of a 20 ms RMS envelope, the burst taken from the first rising edge to the
    *first* falling edge, extended 50 ms, pieces under ``min_s`` dropped. It is
    reproduced here rather than imported because it is the input this recogniser
    has to cope with, and the package must be testable without the harness.
    """
    seg = np.asarray(audio, float)
    win = FS // 50
    out, consumed = [], 0
    while True:
        s = seg[consumed:]
        if len(s) < win * 4:
            return out
        env = np.sqrt(np.convolve(s * s, np.ones(win) / win, "same"))
        on = env > max(env.max() * 0.2, 3e-3)
        if not on.any():
            return out
        start = int(np.argmax(on))
        ends = np.flatnonzero(np.diff(on[start:].astype(int)) == -1)
        if len(ends) == 0:
            return out
        end = start + int(ends[0]) + 1
        if (end - start) / FS >= min_s:
            out.append(s[start:min(len(s), end + FS // 20)])
        consumed += end


def _ack() -> np.ndarray:
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


def _on_air(x, lead: float = 1.0, tail: float = 1.0, seed: int = 0) -> np.ndarray:
    """One burst as the segmenter meets it: quiet channel either side."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(int(lead * FS)), np.asarray(x, float),
                        np.zeros(int(tail * FS))])
    return y + rng.standard_normal(len(y)) * 1e-4


def _faded(x, at_s: float, dur_s: float, depth_db: float) -> np.ndarray:
    y = np.array(x, float)
    y[int(at_s * FS):int((at_s + dur_s) * FS)] *= 10 ** (depth_db / 20.0)
    return y


def _drive(audio, called: str = _CALLED, min_s: float = 0.2):
    hs = _awaiting_ack(called=called)
    for piece in _segments(audio, min_s):
        hs.on_rx_audio(piece)
    return hs


# --------------------------------------------------------------------------- #
# The channel cases.
_PRE_END = _NPRE * MK.HOP / FS                      # 0.171 s — the preamble's extent
_TABLE = [
    ("clean ack, 1 s of noise either side", _on_air(_ack())),
    ("31 ms deep fade in the state symbols", _on_air(_faded(_ack(), 0.26, 0.031, -60))),
    ("50 ms -16 dB fade in the state symbols", _on_air(_faded(_ack(), 0.26, 0.050, -16))),
    ("90 ms fade over the last two symbols", _on_air(_faded(_ack(), 0.38, 0.090, -60))),
    ("clipped by one sample", _on_air(_ack()[:_ACK_LEN - 1])),
]


@pytest.mark.parametrize("label,audio", _TABLE, ids=[c[0] for c in _TABLE])
def test_a_fade_behind_the_preamble_does_not_end_the_attempt(label, audio):
    hs = _drive(audio)
    assert hs.state is VaraState.CONNECTED, (
        f"{label}: the gateway answered and the connect was abandoned anyway — "
        "which on the air is reported exactly like a gateway that stayed silent")


def test_every_fade_clear_of_the_preamble_is_survived():
    """Not tuned to one fade: sweep where the notch lands, how deep and how long.

    Measured, the split is exactly where the waveform says it should be — all 30
    of the 30 fades that start at or after the preamble ends complete the connect,
    and none of the 12 that land inside it do.
    """
    grid = [(at, dur, db)
            for at in (0.06, 0.12, 0.20, 0.26, 0.32, 0.38, 0.44)
            for dur in (0.025, 0.050, 0.090)
            for db in (-60, -16)]
    done = {c: _drive(_on_air(_faded(_ack(), *c))).state is VaraState.CONNECTED
            for c in grid}
    late = [c for c in grid if c[0] >= _PRE_END]
    assert all(done[c] for c in late), (
        f"{sum(not done[c] for c in late)}/{len(late)} fades behind the preamble "
        "lost the ack, and the preamble is the only part of it that is fixed")
    assert not any(done[c] for c in grid if c[0] < _PRE_END), (
        "a fade over the preamble completed the connect — then the preamble is not "
        "what the recogniser is reading")


def test_a_clipped_ack_is_still_the_ack():
    """Segmentation slop takes samples off the end, and everything the recogniser
    needs is at the start — so an ack is recognised down to its four preamble
    symbols, six symbols shorter than the burst VARA keyed."""
    for missing in (1, 100, MK.HOP, 2 * MK.HOP, 6 * MK.HOP):
        hs = _awaiting_ack()
        hs.on_rx_audio(_ack()[:_ACK_LEN - missing])
        assert hs.state is VaraState.CONNECTED, f"rejected an ack {missing} samples short"


def test_an_ack_whose_head_was_cut_away_leaves_the_attempt_alive():
    """The cost of a head-anchored preamble, stated as a test rather than left to
    be discovered on the air: the ack is lost, and the attempt is not."""
    io = _IO()
    hs = _awaiting_ack(io)
    hs.on_rx_audio(_ack()[_NPRE * MK.HOP:])
    assert hs.state is not VaraState.CONNECTED
    assert hs.step == _I_LINKSETUP_SENT, "the attempt ended instead of waiting"
    hs.on_rx_audio(_ack())
    assert hs.state is VaraState.CONNECTED, "the gateway's repeat was not recognised"


# --------------------------------------------------------------------------- #
# The state machine must not dead-end while the peer is still talking.
def test_a_burst_that_is_not_the_ack_leaves_the_attempt_alive():
    rng = np.random.default_rng(7)
    hs = _awaiting_ack()
    for _ in range(20):
        hs.on_rx_audio(rng.standard_normal(int(0.4 * FS)))
    assert hs.state is not VaraState.CONNECTED
    hs.on_rx_audio(_ack())
    assert hs.state is VaraState.CONNECTED, (
        "a run of junk bursts poisoned the wait, so the gateway's ack could no "
        "longer be recognised")


def test_a_repeated_connect_response_resends_the_link_setup():
    """A gateway that did not hear our link-setup repeats step 2. This branch used
    to swallow every burst arriving after the link-setup, so that repeat could
    never be recognised and the attempt was over on one missed ack."""
    io = _IO()
    hs = _awaiting_ack(io)
    sent = io.tx_bursts
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.CONNECT_RESPONSE))
    assert hs.step == _I_LINKSETUP_SENT
    assert hs._linksetup_tx == 1
    assert any("repeat" in m for m in io.log_lines), io.log_lines
    hs.on_rx_audio(_ack())
    assert hs.state is VaraState.CONNECTED
    assert io.tx_bursts > sent


def test_the_link_setup_is_not_resent_for_ever():
    io = _IO()
    hs = _awaiting_ack(io)
    for _ in range(_LINKSETUP_MAX_TX + 3):
        hs.on_rx_audio(MK.synth_burst(_CALLED, VF.CONNECT_RESPONSE))
    assert hs._linksetup_tx == _LINKSETUP_MAX_TX


def test_a_connect_response_for_another_station_is_ignored():
    hs = _awaiting_ack()
    hs.on_rx_audio(MK.synth_burst("K7ABC", VF.CONNECT_RESPONSE))
    assert hs._linksetup_tx == 0
    assert hs.state is not VaraState.CONNECTED


# --------------------------------------------------------------------------- #
# False accepts.
def _spoofs():
    rng = np.random.default_rng(11)
    t = np.arange(int(0.35 * FS)) / FS
    ack = _ack()
    other = MK.synth_burst(_CALLED, VF.SESSION_KEEPALIVE_A)
    return [
        ("the ack's tail without its head", [ack[_NPRE * MK.HOP:]]),
        ("halves of a keepalive", [other[:7 * MK.HOP], other[7 * MK.HOP:]]),
        ("a carrier, then another carrier",
         [0.5 * np.sin(2 * np.pi * f * t) for f in (1000.0, 1450.0)]),
        ("two carriers at once, held",
         [0.5 * (np.sin(2 * np.pi * 1500 * t) + np.sin(2 * np.pi * 1570.3 * t))]),
        ("noise fragments", [rng.standard_normal(int(0.33 * FS)) for _ in range(8)]),
        ("silence", [np.zeros(int(0.33 * FS)) for _ in range(4)]),
    ]


@pytest.mark.parametrize("label,pieces", _spoofs(), ids=[s[0] for s in _spoofs()])
def test_the_wrong_thing_never_completes_a_connect(label, pieces):
    hs = _awaiting_ack()
    for piece in pieces:
        hs.on_rx_audio(piece)
    assert hs.state is not VaraState.CONNECTED, (
        f"{label} completed the connect — an on-air run would report a connect "
        "the gateway never made")


def test_a_long_run_of_noise_never_completes_a_connect():
    """400 bursts is a busy quarter of an hour, and none of them may creep past."""
    rng = np.random.default_rng(3)
    hs = _awaiting_ack()
    for _ in range(400):
        n = int(rng.uniform(0.15, 0.9) * FS)
        hs.on_rx_audio(rng.standard_normal(n) * rng.uniform(0.01, 1.0))
        assert hs.state is not VaraState.CONNECTED, "noise completed the connect"


@pytest.mark.parametrize("session", ["NS0A_2300", "KC9GHZ_2300"])
def test_real_off_air_audio_holds_the_preamble_only_where_an_ack_is(session):
    """The false-accept measurement, on real HF rather than a noise model.

    108 s of off-air audio each, carrying real gateway bursts, our own
    transmissions and whatever else was on the band. NS0A's session holds exactly
    one connected-ack — at 17.97 s, answering our link-setup — and the preamble is
    found there over 47 consecutive lattice offsets and at no other alignment in
    either recording.
    """
    path = corpora.OFFAIR / session / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air recording for {session} not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    track = _pair_track(x)
    n = len(track) - int(_ACK_PRE_OFF[-1])
    hit = (track[np.arange(n)[:, None] + _ACK_PRE_OFF] == _ACK_PRE).all(axis=(1, 2))
    at = np.flatnonzero(hit) * 32 / FS
    expect = [17.97] if session == "NS0A_2300" else []
    assert len(at) == 0 or (len(expect) and abs(at.mean() - expect[0]) < 0.2), (
        f"{session}: the ack preamble appears at {at[:5]} s, where the session's "
        "only ack (if any) is at " + (f"{expect[0]} s" if expect else "no time at all"))


def test_the_off_air_ack_is_recognised_end_to_end():
    """The one real gateway connected-ack we hold, through the recogniser the
    state machine uses — read off the rig, not synthesised."""
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording for NS0A_2300 not present")
    x = corpora.wav_mono(path)
    bracket = x[int(17.8 * FS):int(18.6 * FS)]
    assert _ack_plateau(bracket) >= _ACK_PLATEAU, (
        "NS0A's connected-ack is no longer recognised — a live connect to a real "
        "gateway would not complete")
    pairs = MK.demod_tone_pairs(x[int(17.973 * FS):], VF.CONNECTED_ACK_NSYM)
    assert tuple(pairs[:_NPRE]) == VF.CONNECTED_ACK_PREAMBLE, (
        f"the pair reader gets {pairs[:_NPRE]} off a real gateway ack")
    hs = _awaiting_ack()
    hs.on_rx_audio(bracket)
    assert hs.state is VaraState.CONNECTED


def test_the_live_gate_hands_over_the_real_gateway_ack_and_nothing_else():
    """The whole receive path for step 5, over a real gateway session.

    The connect tool's own segmenter, fed the NS0A recording in device-sized
    chunks: of the 27 brackets it hands over across 108 s of 40 m — gateway DATA
    overs, our own transmissions through a muted receiver, and whatever else was
    on the frequency — exactly one scores at all, and it is the 1.1 s bracket
    holding the ack that answered VARA's link-setup. A regression here is a live
    connect that does not complete.
    """
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording for NS0A_2300 not present")
    kc = corpora.harness("kestrel_connect")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    seg = kc._BracketSegmenter()
    scored = []
    for i in range(0, len(x), 4800):
        for at, piece in seg.push(x[i:i + 4800]):
            if _ack_plateau(piece) >= _ACK_PLATEAU:
                scored.append(at / FS)
    assert len(scored) == 1 and 17.5 <= scored[0] <= 18.0, (
        f"the gate scored acks at {scored} s; NS0A's session holds exactly one, "
        "in the bracket that opens around 17.7 s")


# --------------------------------------------------------------------------- #
@pytest.mark.realtime
def test_the_search_stays_inside_the_per_burst_budget():
    """On a busy channel the segmenter hands over multi-second blobs, and there is
    a fifteen-minute operating window to spend."""
    import time
    rng = np.random.default_rng(1)
    hs = _awaiting_ack()
    blob = rng.standard_normal(9 * FS) * 0.05
    t0 = time.perf_counter()
    hs.on_rx_audio(blob)
    assert time.perf_counter() - t0 < 1.0, "the ack search is back to a per-symbol scan"
