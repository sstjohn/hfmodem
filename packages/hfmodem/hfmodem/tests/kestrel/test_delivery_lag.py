# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A receive backlog ages every frame VARA names.

The peer's listening gap is wall time. Audio that waited in the station's lane
and queue before it reached the handshake was already that much older, and a
frame named fresh from the samples alone can be one the peer has keyed past.
The freshness limits were proven on healthy hosts with their own delivery lag
inside them, so only the backlog past that counts, and only against those
limits. Nothing here opens a device or keys a radio.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.kestrel.host import station_runtime as SR
from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
from hfmodem.kestrel.monitor import SEG_FRAME, SEG_HANG
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

from . import test_reack_over as reack
from .receive_answer_helpers import audio_clock
from .receive_window_helpers import station, window, feed
from .test_compression import IO
from .test_idle_poll import _Clock
from .test_vara_host_modem import Observer


class Capture:
    """A card whose converter clock runs ``ahead`` samples past the newest
    sample the transport has been handed."""
    samples = 0
    timestamp_limitation = None

    def __init__(self, ahead=0):
        self.ahead, self.transport = ahead, None

    def sample_now(self):
        return float(self.transport.input_clock + self.ahead)


def transport(capture):
    t = SR.StationTransport(SimpleNamespace(
        station=SimpleNamespace(arbiter=None, audio=capture)))
    t.cq_sink = None
    capture.transport = t
    return t


def listener():
    """A handshake that takes the indexed stream and does nothing with it."""
    return SimpleNamespace(state=None, io=SimpleNamespace(), bw='500',
                           rx_gap=lambda at: None, on_rx_stream=lambda *a, **k: None)


def noisy(x, seed=7):
    return x + np.random.default_rng(seed).normal(0, 1e-3, len(x))


def drive(t, hs, audio, at, tick=lambda n: None):
    """Both routes, the way the station owner runs them."""
    def drain():
        while (burst := t.next_rx_burst(.01, hs)) is not None:
            hs.on_rx_audio(burst, sample_start=t.burst_start, backlog=t.burst_backlog)
    for i in range(0, len(audio), SR.CHUNK):
        chunk = audio[i:i + SR.CHUNK]
        tick(len(chunk))
        t.feed(at + i, chunk)
        drain()
    drain()


def stale(io):
    return [m for m in io.msgs if "nothing keyed" in m]


@pytest.fixture
def timed_audio(monkeypatch):
    audio_clock(monkeypatch, VA.VaraStationHandshake, MK.FS)


def idle_over_station(capture):
    hs, io = reack._mid_delivery()
    sent_at, start = [], VA.time.monotonic()
    tx = io.tx

    def timed_tx(samples):
        sent_at.append(VA.time.monotonic() - start)
        tx(samples)
    io.tx = timed_tx
    t = transport(capture)
    before = len(io.sent)
    audio = noisy(np.r_[np.zeros(MK.FS // 2), reack.idle(), np.zeros(2 * MK.FS)])
    drive(t, hs, audio, hs._input_clock)
    return hs, io, t, before, sent_at


def test_an_idle_heard_through_a_backlog_is_not_keyed_into(timed_audio):
    hs, io, t, before, _ = idle_over_station(Capture(int(.6 * MK.FS)))
    assert len(io.sent) == before and hs._positive_reacks == 0, io.msgs
    assert stale(io) and "session-responder-over-idle" in stale(io)[0], io.msgs


@pytest.mark.parametrize("lag", [.03, .15])
def test_an_idle_heard_on_a_healthy_host_keys_as_without_a_clock(timed_audio, lag):
    """A healthy host's lag, and a backlog well inside the gap, key the same
    frame at the same instant as a transport whose clock cannot be read."""
    base = idle_over_station(SimpleNamespace(samples=0))
    hs, io, t, before, sent_at = idle_over_station(Capture(int(lag * MK.FS)))
    assert len(io.sent) == before + 1 and reack.keyed(io) == hs._reack_frame, io.msgs
    assert not stale(io), io.msgs
    assert [len(x) for x in io.sent] == [len(x) for x in base[1].sent]
    assert sent_at == base[4]


def test_the_backlog_handed_over_is_the_lag_past_a_healthy_host():
    capture = Capture()
    t = transport(capture)
    seen = []
    hs = listener()
    hs.on_rx_stream = lambda *a, backlog, **k: seen.append(backlog)
    for i, lag in enumerate((.03, .15, .6)):
        capture.ahead = int(lag * MK.FS)
        t.feed(i * SR.CHUNK, np.zeros(SR.CHUNK))
        t.next_rx_burst(.01, hs)
    assert seen == [0, int(.05 * MK.FS), int(.5 * MK.FS)]


def test_a_query_heard_through_a_backlog_is_not_answered(monkeypatch):
    h, io, before = _query_through(monkeypatch, .6)
    # The fit searches only starts still inside the gap, so a query that
    # arrives already past it is never named, and so never logged stale.
    assert len(io.sent) == before and h._rx_window.W is None, io.msgs


def test_a_query_heard_without_a_backlog_is_answered(monkeypatch):
    h, io, before = _query_through(monkeypatch, 0)
    assert len(io.sent) == before + 1 and not stale(io), io.msgs


def _query_through(monkeypatch, lag):
    monkeypatch.setattr(VA, "time", _Clock)
    _Clock.t = 1000.0
    h, io = station('2300', 'responder')
    feed(h, window('2300', b'A' * 89))
    kind = VF.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
    query = MK.synth_burst(h.called, VF.for_bw(kind, '2300'))
    t = transport(Capture(int(lag * MK.FS)))
    before = len(io.sent)

    def tick(n):
        _Clock.t += n / MK.FS
    drive(t, h, noisy(np.r_[query, np.zeros(MK.FS // 2)]), h._input_clock, tick)
    return h, io, before


def test_the_backlog_is_logged_once_each_way():
    capture = Capture()
    t = transport(capture)
    hs = listener()
    for i, lag in enumerate((0, .3, .4, .05, .02)):
        capture.ahead = int(lag * MK.FS)
        t.feed(i * SR.CHUNK, np.zeros(SR.CHUNK))
        t.next_rx_burst(.01, hs)
    assert list(t.messages) == ["receive backlog 0.300 s",
                                "receive backlog cleared, 0.050 s"]


@pytest.mark.parametrize("audio", [
    SimpleNamespace(samples=0, timestamp_limitation="WASAPI", sample_now=lambda: 1e9),
    SimpleNamespace(samples=0, sample_now=lambda: 1e9)], ids=["estimated", "replay"])
def test_a_clock_that_is_not_the_converter_does_not_age_the_stream(audio):
    t = transport(audio)
    t.feed(0, np.zeros(SR.CHUNK))
    t._rx_end = SR.CHUNK
    assert t.delivery_lag(SR.CHUNK) == 0


@pytest.mark.parametrize("lag", [0, .03, .6])
def test_a_bracket_carries_the_streams_backlog_not_its_hangover(lag):
    """The segmenter hands a bracket over SEG_HANG quiet frames past its crop.
    That hangover was in every bracket the freshness limits were proven on, so
    a bracket carries only the stream's own backlog at handover."""
    t = transport(Capture(int(lag * MK.FS)))
    hs = listener()
    audio = noisy(np.r_[np.zeros(MK.FS), reack.idle(), np.zeros(MK.FS)])
    lags = []
    for i in range(0, len(audio), SR.CHUNK):
        t.feed(i, audio[i:i + SR.CHUNK])
        while (burst := t.next_rx_burst(.01, hs)) is not None:
            lags.append((t.delivery_lag(t.burst_start + len(burst)), t.burst_backlog))
    (behind, backlog), = lags
    assert behind >= (SEG_HANG - 1) * SEG_FRAME + int(lag * MK.FS)
    assert backlog == handed(lag)


def test_the_owner_forwards_the_bracket_backlog():
    class Bracketing(IO):
        burst_start, burst_backlog = 1000, 2400

        def next_rx_burst(self, timeout, hs=None):
            return np.zeros(100)
    runtime = AudioHostRuntime(Bracketing())
    modem = runtime.modem_factory()
    modem.start(Observer())
    heard = []
    modem.station.on_rx_audio = lambda burst, **kw: heard.append(kw)
    runtime.step()
    assert heard == [dict(sample_start=1000, backlog=2400)]


@pytest.mark.parametrize("backlog", [0, .15])
def test_an_idle_whose_tail_has_not_arrived_waits_for_it_whatever_the_backlog(
        timed_audio, backlog):
    """The sign of the held audio says whether the idle's last symbol is in
    the buffer yet; a backlog makes the frame older, not more complete."""
    hs, io = reack._mid_delivery()
    before = len(io.sent)
    cut = len(reack.idle()) - int(.03 * MK.FS)
    audio = np.r_[np.zeros(MK.FS // 2), reack.idle(), np.zeros(2 * MK.FS)]
    head = MK.FS // 2 + cut
    hs.on_rx_stream(audio[:head], backlog=int(backlog * MK.FS))
    assert len(io.sent) == before, io.msgs
    for i in range(head, len(audio), MK.FS // 10):
        hs.on_rx_stream(audio[i:i + MK.FS // 10], backlog=int(backlog * MK.FS))
    assert len(io.sent) == before + 1 and reack.keyed(io) == hs._reack_frame, io.msgs


@pytest.mark.parametrize("backlog", [0, 1])
def test_a_bracketed_idle_counts_its_backlog_against_the_gap(timed_audio, monkeypatch,
                                                            backlog):
    """A fit that costs most of the gap leaves the bracket's own trail inside
    it; the backlog is what tips it, and the stale line says so."""
    probe = reack._mid_delivery()[0]
    probe.on_rx_audio(reack.idle())
    trail = probe._idle_held
    hs, io = reack._mid_delivery()
    base = VA.time.monotonic
    cost = [0.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: base() + cost[0])
    fit = hs._complete_query_fit

    def slow_fit(*args, **kwargs):
        cost[0] += .4
        return fit(*args, **kwargs)

    monkeypatch.setattr(hs, "_complete_query_fit", slow_fit)
    before = len(io.sent)
    tip = VA._GRANT_FRESH_S - .4 - trail + .02
    hs.on_rx_audio(reack.idle(), backlog=int(backlog * tip * MK.FS))
    if backlog:
        assert len(io.sent) == before and len(stale(io)) == 1, io.msgs
    else:
        assert reack.keyed(io) == hs._reack_frame and not stale(io), io.msgs


def handed(lag):
    """The backlog the transport hands VARA for a delivery lag in seconds."""
    return max(0, int(lag * MK.FS) - SR.LAG_NOTICE)


@pytest.mark.parametrize("lag", [.03, .6])
def test_a_grant_heard_through_a_backlog_is_not_taken(lag):
    from .test_turn_law import _connected, _CALLED
    sent = []
    for backlog in (0, handed(lag)):
        hs, io = _connected()
        hs.turn = VA._TURN_ASKED
        x = np.r_[MK.synth_burst(_CALLED, VF.SESSION_DRAINED_RESPONDER),
                  np.zeros(int(.3 * MK.FS))]
        for i in range(0, len(x), MK.FS // 10):
            hs.on_rx_stream(x[i:i + MK.FS // 10], backlog=backlog)
        sent.append((hs.turn, [len(x) for x in io.sent]))
    if lag > .5:
        assert sent[1] == (VA._TURN_ASKED, []), io.msgs
        assert any("behind the newest audio" in m for m in io.msgs), io.msgs
    else:
        assert sent[1] == sent[0] and sent[0][0] == VA._TURN_OURS, io.msgs


@pytest.mark.parametrize("lag", [.03, .6])
def test_a_nak_heard_through_a_backlog_draws_no_resend(lag):
    from .test_bw2750_responder_nak import sending, nak_burst
    runs = []
    for backlog in (0, handed(lag)):
        hs, io = sending()
        hs.on_rx_stream(nak_burst(), backlog=backlog)
        runs.append(([len(x) for x in io.sent], hs._tx_retries))
    if lag > .5:
        assert runs[1] == ([], 0), io.msgs
        assert any("listening gap has gone" in m for m in io.msgs), io.msgs
    else:
        assert runs[1] == runs[0] and len(runs[0][0]) == 1, io.msgs


def test_a_bw500_ask_behind_the_turn_idle_keys_at_the_same_sample_under_backlog(monkeypatch):
    """The lead behind the peer's idle is a hold, not a freshness limit: a
    backlog leaves the instant the ask goes out where it was."""
    from . import test_bw500_turn_idle_1021 as T
    original = VA.VaraStationHandshake.on_rx_stream
    heard = []
    for backlog in (0, int(.05 * MK.FS)):
        monkeypatch.setattr(VA, "time", _Clock)
        _Clock.t = 1000.0
        monkeypatch.setattr(VA.VaraStationHandshake, "on_rx_stream",
                            lambda self, x, backlog=backlog, **kw: original(
                                self, x, backlog=backlog, **kw))
        hs, io, _ = T._released_on_the_ask()
        lead, idle = T._band_noise(0.08, 4), T._frame(VF.SESSION_TURN_IDLE_RESPONDER, seed=5)
        heard.append(T._feed(hs, io, np.r_[lead, idle, T._band_noise(1.0, 6)]))
        assert io.msgs[-1] == f"tx session-turn-request keyed-by={T.CALLER}", io.msgs
    assert heard[0] is not None and heard[1] == heard[0], heard


@pytest.mark.parametrize("lag", [.03, .6])
def test_a_bracketed_drained_grant_near_the_limit_is_taken_only_without_a_backlog(lag):
    from .test_drained_empty_turn_1031 import _after_the_close, BW, CALLED
    grant = MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_DRAINED_RESPONDER, BW))
    runs = []
    for backlog in (0, handed(lag)):
        hs, io = _after_the_close()
        hs.on_rx_audio(np.r_[grant, np.zeros(int(.46 * MK.FS))], backlog=backlog)
        runs.append((hs.turn, len(io.sent), round(hs._grant_held, 2)))
    assert runs[0][:2] == (VA._TURN_OURS, 2), io.log_lines
    assert .45 <= runs[0][2] <= VA._GRANT_FRESH_S, runs
    if lag > .5:
        assert runs[1][:2] == (VA._TURN_PEER, 1), io.log_lines
    else:
        assert runs[1] == runs[0], io.log_lines


@pytest.mark.parametrize("lag", [.03, .3])
def test_a_solicited_final_request_the_backlog_puts_past_the_gap_is_not_taken(
        monkeypatch, lag):
    from . import test_wide_final_turn_request as W
    original = VA.VaraStationHandshake.on_rx_stream
    runs = []
    for backlog in (0, handed(lag)):
        monkeypatch.setattr(VA, "time", _Clock)
        monkeypatch.setattr(VA.VaraStationHandshake, "on_rx_stream", original)
        _Clock.t = 1000.0
        h, io = W.sender('2300', 'initiator')
        unkey = W.solicit(h, io)
        monkeypatch.setattr(VA.VaraStationHandshake, "on_rx_stream",
                            lambda self, x, backlog=backlog, **kw: original(
                                self, x, backlog=backlog, **kw))
        x = W.request(h)
        sent = len(io.sent)
        audio = np.r_[np.zeros(int(.06 * MK.FS)), x, np.zeros(int(.35 * MK.FS))]
        W.feed(h, audio, block=len(audio))
        runs.append(([round(t - unkey, 6) for t in io.at[sent:]], h._tx_pending is None,
                     [m for m in io.msgs if "nothing keyed" in m]))
    assert runs[0][0] and runs[0][1] and not runs[0][2], runs
    if lag > .1:
        assert runs[1][:2] == ([], False), io.msgs
        assert len(runs[1][2]) == 1 and "session-turn-request-responder" in runs[1][2][0]
    else:
        assert runs[1] == runs[0], runs


@pytest.mark.parametrize("lag", [0, .25])
def test_a_link_up_request_under_backlog_still_brings_the_link_up(lag):
    """The unsolicited window keeps the request as the evidence the link is up,
    not as an answer window, so a backlog does not age it out."""
    from .test_connect_ask_stream import awaiting
    hs, io = awaiting()
    request = MK.synth_burst("W9SSJ", VF.SESSION_TURN_REQUEST_RESPONDER)
    hs.on_rx_stream(np.r_[request, np.zeros(int(.12 * MK.FS))], backlog=handed(lag))
    assert hs.state == VA.VaraState.CONNECTED and io.keys == 1, io.msgs
    assert io.up == [("W9SSJ", "K0SI", "2300")]
    assert not stale(io), io.msgs
