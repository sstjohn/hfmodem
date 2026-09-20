# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A BW500 session moving a delivery: what the station does with a real over.

The overs are a stock VARA HF 4.9.0's own, off the 479 s BW500 loopback of
2026-07-13 (`AAAA1` → `BBBB2`) — one bracket at a time, cut round the burst the
way the energy segmenter cuts one. They cannot be synthesised: our transmitter
renders the nine lead-in columns silent, so a burst we key is not found where it
starts  [see ``test_bw500_data_over``], and only a real modem's burst exercises
the path a session actually runs.

What is under test is the whole receive-and-answer turn at this bandwidth: the
over is identified, every block it carries reaches the host in order, and exactly
one burst goes back — the 8-symbol continue burst of the link's own bandwidth.

Two routes hand the station an over. The bracket route is a transport's energy
gate closing round the burst; the stream route is every sample the receiver
produces, in the transport's own 20 ms polls, and it is the one a live session
runs on — a bracket force-closed at 6 s cannot hold a two-frame over, and the
stream has to find the over's end for itself  [vara_arq._stream_over_500].
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.rx import varahf500 as rx500
from hfmodem.tests.kestrel import corpora

_CALLER, _CALLED = "AAAA1", "BBBB2"


class _IO(VA.VaraIO):
    """Counts key-ups and keeps what went out and what reached the host. Nothing
    here opens a device or reaches a transmitter."""

    def __init__(self):
        self.keys = 0
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []
        self.host: list[bytes] = []
        self.fed = 0                    # samples the stream has been handed
        self.keyed_at: list[int] = []   # ``fed`` at each key-down

    def key(self, on):
        self.keys += bool(on)
        if on:
            self.keyed_at.append(self.fed)

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)

    def data(self, payload): self.host.append(bytes(payload))


def _station():
    """A connected BW500 initiator with the turn its peer's, which is where a
    station reading a gateway's delivery stands."""
    io = _IO()
    hs = VA.VaraStationHandshake([_CALLER], io, bw="500")
    hs.role, hs.caller, hs.called = "initiator", _CALLER, _CALLED
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    return hs, io


def _stream(hs, io, x, poll: int = 960) -> None:
    """The audio as the transport hands it to the stream route: 20 ms polls."""
    for i in range(0, len(x), poll):
        hs.on_rx_stream(x[i:i + poll])
        io.fed += min(poll, len(x) - i)


@pytest.fixture(scope="module")
def session():
    return corpora.wav_mono(corpora.MULTIFRAME_SESSION)


@pytest.fixture(scope="module")
def spans(session):
    """Every data burst of the session as ``(start, stop)``, by the number of
    frames it carries. The link-setup is a burst and not a data frame."""
    out: dict[int, list[tuple[int, int]]] = {}
    for a, b in rx500.burst_spans(session):
        n = len(rx500.decode_stream(session[max(0, a - 6000):b + 6000]).data_frames)
        out.setdefault(n, []).append((a, b))
    return out


@pytest.fixture(scope="module")
def brackets(session, spans):
    """The same bursts as segmenter-shaped brackets."""
    return {n: [session[max(0, a - 6000):b + 6000] for a, b in v]
            for n, v in spans.items()}


@corpora.requires_multiframe_session
@pytest.mark.parametrize("blocks", (1, 2))
def test_an_over_is_delivered_whole_and_answered_once(brackets, blocks):
    """One block or two, the host is handed every one and the transmitter is
    keyed exactly once. Delivering the first of two would hand the host a message
    with a hole in it that no later over refills."""
    hs, io = _station()
    hs.on_rx_audio(brackets[blocks][0])
    assert len(io.host) == blocks
    assert [len(p) for p in io.host] == [43] * blocks
    assert io.keys == 1


@corpora.requires_multiframe_session
def test_an_unmeasured_link_gets_the_generated_narrow_continue(brackets):
    """The unknown link gets short16 on its own alphabet; the old foreign
    captured tail cost two idle gaps per over at stock K5FIT."""
    hs, io = _station()
    hs.on_rx_audio(brackets[2][0])
    (out,) = io.sent
    assert VF.over_continue_state(hs.caller, hs.called, *hs._peer_over_state) is None
    kind = VF.for_bw(VF.SESSION_OVER_RESPONSE_SHORT, "500")
    assert MK.demod_burst(out, kind, MK.band_for("500")) == VF.handshake_tones(hs.called, kind)
    assert hs._reack_frame == VA.OVER_CONTINUE_SHORT


@corpora.requires_multiframe_session
def test_a_repeat_of_one_over_reaches_the_host_once(brackets):
    """A peer that missed our answer repeats the over; the repeat is answered
    again and must not be delivered again  [vara_arq._deliver]."""
    hs, io = _station()
    hs.on_rx_audio(brackets[2][0])
    hs.on_rx_audio(brackets[2][0])
    assert len(io.host) == 2
    assert io.keys == 2


@corpora.requires_multiframe_session
def test_consecutive_overs_join_into_one_byte_stream(brackets, session):
    """Four bursts in emission order, and what the host holds is what the
    session's own transfer carried — the blocks in the order they were keyed."""
    hs, io = _station()
    order = brackets[2][:2] + brackets[1][:2]
    for br in order:
        hs.on_rx_audio(br)
    want = [bytes(f.payload) for br in order
            for f in rx500.decode_stream(br).data_frames]
    assert io.host == want
    assert io.keys == len(order)


# --------------------------------------------------------------------------- #
# The stream route: the over found on the receive stream, end and all.
@corpora.requires_multiframe_session
@pytest.mark.parametrize("blocks", (1, 2))
def test_an_over_is_read_off_the_stream_and_answered_in_its_turnaround(
        session, spans, blocks):
    """Fed a poll at a time, both blocks reach the host, one burst goes back, and
    it goes back inside the turnaround: a stock station answers 0.083-0.107 s
    after the over's last sample and the sender re-keys 0.52 s after its unkey,
    so the answer has to be keyed within a tenth of a second of the end the
    stream measured for itself."""
    hs, io = _station()
    a, b = spans[blocks][0]
    lead = MK.FS
    _stream(hs, io, session[a - lead:b + MK.FS])
    assert [len(p) for p in io.host] == [43] * blocks
    assert io.keys == 1
    late = (io.keyed_at[0] - lead - (b - a)) / MK.FS
    assert 0.0 < late <= 0.1, f"answer keyed {late:.3f} s after the over's end"


@corpora.requires_multiframe_session
def test_the_bracket_declines_an_over_the_stream_has_answered(session, spans):
    """A transport that feeds the stream brackets the same audio, later. The
    over must reach the host once and draw one burst, not one per route."""
    hs, io = _station()
    a, b = spans[2][0]
    _stream(hs, io, session[a - MK.FS:b + MK.FS])
    hs.on_rx_audio(session[max(0, a - 6000):b + 6000])
    assert len(io.host) == 2
    assert io.keys == 1


@corpora.requires_multiframe_session
def test_every_over_of_the_session_is_read_off_the_stream_once(session):
    """The whole 479 s transfer as one stream: 51 data bursts, 96 blocks, each
    delivered once and answered once, in emission order. The 52nd burst is the
    link-setup, which is not an over and draws nothing."""
    hs, io = _station()
    _stream(hs, io, session, poll=4800)
    frames = rx500.decode_stream(session).data_frames
    assert len(frames) == 96
    assert io.keys == 51
    assert len(io.host) == 96
    assert io.host[:-1] == [bytes(f.payload) for f in frames[:-1]]
    assert bytes(frames[-1].payload).startswith(io.host[-1])   # the close is short


@corpora.requires_multiframe_session
def test_the_stream_route_keys_nothing_on_band_audio(session):
    """The false-key bound the bracket route is held to, on the stream: real
    off-air HF holding the wide bandwidth's overs and a stranger's traffic, a
    verified-clear channel, gaussian noise and a bare carrier — none of it keys
    a BW500 station. A wide over is found as a span and refused at the decode."""
    audio = []
    if corpora.CLEAR_CHANNEL.exists():
        audio.append(("clear channel", corpora.wav_mono(corpora.CLEAR_CHANNEL)))
    for name in ("KC9GHZ_2300", "NS0A_2300"):
        path = corpora.OFFAIR / name / "rig_rx.wav"
        if path.exists():
            audio.append((name, corpora.wav_mono(path)))
    rng = np.random.default_rng(3)
    audio.append(("noise", rng.standard_normal(int(60 * MK.FS)) * 0.3))
    t = np.arange(int(20 * MK.FS)) / MK.FS
    audio.append(("carrier", np.sin(2 * np.pi * 1500.0 * t) * 0.5))
    for name, x in audio:
        hs, io = _station()
        _stream(hs, io, x / (np.abs(x).max() or 1.0), poll=4800)
        assert io.keys == 0, f"{name} keyed the transmitter: {io.msgs}"
        assert not io.host, f"{name} reached the host: {io.msgs}"
