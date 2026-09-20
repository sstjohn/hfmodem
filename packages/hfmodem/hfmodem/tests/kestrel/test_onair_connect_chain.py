# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The whole receive chain, end to end, over the on-air call of 2026-07-26.

kestrel called KB9MMT eight times through the rig that afternoon and reported no
answer. KB9MMT answered twice — at 56.021 s and 63.979 s into the recording, both
15/15 against KB9MMT and 0/15 against our own call — and every one of those
bursts went through the tool without being seen. Three separate defects had to be
fixed for that, and each was fixed and measured on its own:

  * :func:`vara_mfsk.demod_tones` took the peak of the whole rfft. A received
    signal carries strong odd-harmonic images of its own tones, and on this
    recording the in-band power sits only 9.0 dB above the out-of-band power, so
    the 3f image beat the fundamental and 16 of the 23 symbols read as three
    times the true bin.
  * the same unbounded search sat in the connected-ack recogniser's
    :func:`vara_arq._tone_track`.
  * the segmenter's noise floor was a running minimum, which one dropout pins for
    the rest of the run — and a dropout is just this rig keying, which mutes its
    own receive audio 35 dB down.

Each of those was validated in isolation. None of it says the chain works, because
the chain is what failed: a gate, a burst segmenter, a preamble locator, a tone
demodulator and a state machine, and the answer had to survive all five. This
module runs the real segmenter and the real :class:`VaraStationHandshake` over the
real audio and asserts that a connect attempt against KB9MMT gets as far as
transmitting the link-setup — the furthest that recording can carry it, since
KB9MMT never sent a connected-ack.

**Our own transmissions are in the input and are suppressed the way the live path
suppresses them.** ``AudioVaraIO.tx`` advances the consumed cursor past everything
recorded through the end of the samples it played plus a 0.1 s codec tail, and
restarts the segmenter, so what the gate sees of one of our overs is only the last
of the receiver mute. That is modelled here by finding our transmissions as runs
of frames more than 15 dB below the median lasting over a second — the receiver
mute, 2.1 s each, eight of them, against a band noise that never moves more than
6 dB — and dropping each run bar its final 0.1 s, restarting the segmenter at each
splice. It is faithful because it is the same cut, made from the same quantity:
the live path knows where its overs are because it made them, this knows because
the rig recorded itself going deaf, and
:func:`test_the_suppression_model_finds_our_eight_transmissions` pins the two
together against the eight independently known transmission times. What is left is
133.6 s of genuine receive audio, and the tool has to find one 0.98 s answer in it
without being fooled by the third station that occupies the frequency from 98.8 s
on.

The connect-requests are not faked either: the handshake originates one at each
splice while it is still waiting for an answer, exactly as ``kestrel_connect``'s
loop re-sends on cadence, so the bursts being skipped are ones this state machine
really emitted.

Two things the isolated fixes did not show, both visible only from end to end:

  * KB9MMT's second answer arrived into our own transmission. It begins at
    63.979 s, 1.6 s into our own eighth connect-request, so all that survives our
    receive mute is its last 0.85 s — less than the 0.98 s a 23-symbol burst
    occupies, and with the preamble gone. It is still recovered, because the
    payload lands inside that tail and ``_response_by_payload`` anchors on the
    payload rather than the preamble; the attempt therefore sends the link-setup
    twice, which is the correct answer to a gateway repeating its response.
  * the answer is the only thing in the 150 s that looks remotely like itself.
    Scanned at fixed alignments over every bracket the segmenter hands over, the
    best KB9MMT connect-response match anywhere else in the recording — the third
    station's traffic included — is 3 of 15.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_arq as ARQ
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import _I_CR_SENT, _I_LINKSETUP_SENT, VaraState, VaraStationHandshake

kc = corpora.harness("kestrel_connect")
VM = corpora.harness("vara_monitor")

FS = MK.FS
FRAME = VM.SEG_FRAME
CHUNK = 4800                      # the device read size AudioVaraIO feeds the gate

GATEWAY = "KB9MMT"
MYCALL = "W9SSJ"

# When we keyed, from the operating log of that afternoon; the recording is the
# same rig's receive audio, so each of these shows up as a mute.
OUR_OVERS = (5.32, 12.40, 20.77, 29.13, 37.46, 45.76, 54.10, 62.37)
# When KB9MMT answered. The first is the one a connect has to catch; the second
# arrives inside our eighth over (see the module docstring).
ANSWER_AT = 56.021
ANSWER_REPEAT_AT = 63.979
# The third station on the frequency, which must never be taken for a gateway.
INTERLOPER_FROM = 98.0


class _RecordingIO:
    """A :class:`VaraIO` that writes down what the handshake asked for.

    Nothing here reaches an audio device, a serial port or rigctld: the point of
    replaying a recording is that the transmitter stays cold.
    """

    def __init__(self):
        self.keyed = 0
        self.tx_secs: list[float] = []
        self.connects: list[tuple] = []
        self.logs: list[str] = []

    def key(self, on: bool) -> None:
        self.keyed += bool(on)

    def tx(self, samples) -> None:
        self.tx_secs.append(len(samples) / FS)

    def pending(self) -> None:
        pass

    def connected(self, *info) -> None:
        self.connects.append(info)

    def log(self, msg: str) -> None:
        self.logs.append(msg)


def _our_overs(x: np.ndarray, tail_s: float = 0.1) -> list[tuple[int, int]]:
    """Frame spans the live path never scans, one per transmission of ours.

    A transmitting rig mutes its own receive audio, so our overs are the runs of
    frames far below the median that last longer than a burst does. ``tail_s``
    is left in on purpose: the mute outlasts the audio we played by about 0.22 s,
    so a few frames 35 dB under the band noise reach the gate after every over.
    Those frames are the whole reason a running-minimum floor was fatal here, and a
    model that tidied them away would be testing a channel we never had. The live
    path is now kinder than this — ``AudioVaraIO.tx`` holds the key ``TX_IDLE_HOLD_S``
    past the last sample and takes its cursor after that, so most of what is left in
    here never reaches its gate at all. A model may be wrong in that direction.
    """
    n = len(x) // FRAME
    rms = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1)) + 1e-12
    muted = rms < np.median(rms) * 10 ** (-15 / 20)
    out, i = [], 0
    while i < n:
        j = i
        while j < n and muted[j]:
            j += 1
        if (j - i) * FRAME / FS > 1.0:
            out.append((i, j - int(tail_s * FS / FRAME)))
        i = j + 1
    return out


def _receive_windows(x: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` sample spans of everything but our own overs."""
    spans, cur = [], 0
    for a, b in _our_overs(x):
        if a * FRAME > cur:
            spans.append((cur, a * FRAME))
        cur = b * FRAME
    if cur < len(x):
        spans.append((cur, len(x)))
    return spans


def _replay(x: np.ndarray, gateway: str,
            stream: bool = False) -> tuple[_RecordingIO, VaraStationHandshake, list]:
    """Run one connect attempt against ``gateway`` over the whole recording.

    The loop is ``kestrel_connect.connect`` with the clock taken out: originate at
    the start and again at every splice while no answer has come back, feed each
    receive window to the gate in device-sized chunks, and hand the handshake each
    bracket the moment the gate closes it. ``restart()`` at each splice is what the
    live path does after a transmission — forget the audio in flight, keep the
    noise floor.

    ``stream`` also hands every chunk to :meth:`VaraStationHandshake.on_rx_stream`
    before the gate sees it, which is what ``AudioVaraIO.next_rx_burst`` does at the
    radio. The state the stream search keeps is deliberately *not* restarted at a
    splice: the live path does not restart it either, and the repeat at
    ``ANSWER_REPEAT_AT`` survives only because the track runs across the cut.

    Returns the IO, the handshake, and ``(recording time, bracket seconds, from,
    to)`` for every step the handshake took.
    """
    io = _RecordingIO()
    hs = VaraStationHandshake([MYCALL], io, bw="2300")
    seg = kc._BracketSegmenter()
    steps: list[tuple[float, float, str | None, str | None]] = []
    hs.originate(gateway, MYCALL)
    for base, end in _receive_windows(x):
        if hs.step == _I_CR_SENT and base:
            hs.originate(gateway, MYCALL)          # re-send on cadence, as the tool does
        seg.restart()
        window = x[base:end]
        for i in range(0, len(window), CHUNK):
            chunk = window[i:i + CHUNK]
            if stream:
                was = hs.step
                hs.on_rx_stream(chunk)
                if hs.step != was:
                    steps.append(((base + i + len(chunk)) / FS, 0.0, was, hs.step))
            for at, piece in seg.push(chunk):
                was = hs.step
                hs.on_rx_audio(piece)
                if hs.step != was:
                    steps.append(((base + at) / FS, len(piece) / FS, was, hs.step))
    return io, hs, steps


def _frame_rms(x: np.ndarray) -> np.ndarray:
    """Frame RMS on the gate's own analysis frame — what the floor is read from."""
    fr = x[:len(x) // FRAME * FRAME].reshape(-1, FRAME)
    return np.sqrt((fr * fr).mean(1))


def _brackets(x: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Every bracket the gate hands over, with its time in the recording."""
    seg = kc._BracketSegmenter()
    out = []
    for base, end in _receive_windows(x):
        seg.restart()
        window = x[base:end]
        for i in range(0, len(window), CHUNK):
            out += [((base + at) / FS, piece) for at, piece in seg.push(window[i:i + CHUNK])]
    return out


@pytest.fixture(scope="module")
def recording() -> np.ndarray:
    x = corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT)
    return x / (np.abs(x).max() or 1.0)


@pytest.fixture(scope="module")
def attempt(recording):
    return _replay(recording, GATEWAY)


@corpora.requires_onair_connect_attempt
def test_the_suppression_model_finds_our_eight_transmissions(recording):
    """The model that stands in for ``AudioVaraIO.tx`` has to cut where we keyed.

    If it cut somewhere else the rest of this module would be measuring a channel
    we never had — a hand-chosen window, which is exactly what an end-to-end test
    is for not doing. The mute leads the audio by a fifth of a second or so
    (the PTT closes before the first sample is played), so the tolerance is a
    third of a second either way.
    """
    starts = [a * FRAME / FS for a, _ in _our_overs(recording)]
    assert len(starts) == len(OUR_OVERS), (
        f"found {len(starts)} transmissions of ours at {[round(s, 2) for s in starts]}, "
        f"expected the {len(OUR_OVERS)} of the operating log")
    for found, keyed in zip(starts, OUR_OVERS):
        assert abs(found - keyed) < 0.35, (
            f"a transmission cut at {found:.2f} s against a logged {keyed:.2f} s")
    kept = sum(b - a for a, b in _receive_windows(recording)) / FS
    assert 130 < kept < 140, f"{kept:.1f} s of receive audio left, expected ~134"


@corpora.requires_onair_connect_attempt
def test_the_gateway_answer_carries_the_handshake_to_the_link_setup(attempt):
    """The failure of 2026-07-26, driven through the chain that failed it.

    One state change, on the bracket that holds KB9MMT's answer, into the
    link-setup — and the link-setup is really rendered and handed to the transport,
    so the wideband TX synth is on the path too rather than assumed.

    The first answer must be found by its preamble, not by the payload fallback.
    Both reach the same state, so without this the module would go green on a
    chain whose cheap primary locator had stopped working entirely — which is
    exactly what an unbounded tone search does to it, since ``lock_preamble``
    demodulates to decide.
    """
    io, hs, steps = attempt
    assert steps, (
        "the handshake never moved off the connect-request over 150 s of audio in "
        f"which {GATEWAY} answered twice — {len(io.logs)} log lines, "
        f"{len(io.tx_secs)} transmissions")
    assert len(steps) == 1, f"expected one state change, got {steps}"
    at, secs, was, now = steps[0]
    assert (was, now) == (_I_CR_SENT, _I_LINKSETUP_SENT)
    assert abs(at - ANSWER_AT) < 0.25, (
        f"the handshake advanced on a bracket at {at:.3f} s, not on {GATEWAY}'s "
        f"answer at {ANSWER_AT:.3f} s")
    assert hs.state is VaraState.CONNECTING
    assert 1 <= hs._linksetup_tx <= ARQ._LINKSETUP_MAX_TX
    assert any(t > 4.0 for t in io.tx_secs), (
        f"no wideband over was transmitted; bursts were {[round(t, 2) for t in io.tx_secs]} s")
    assert sum("preamble locked" in m for m in io.logs) == 1, (
        f"{GATEWAY}'s answer was not located by its preamble; the connect got there "
        "on the payload fallback alone")

    # The answer has to arrive as a burst, not inside a blind slice. The gate
    # force-closes at SEG_MAX_S when it has lost track of the channel, and a
    # 6 s bracket that happens to contain the answer is the pinned-noise-floor
    # failure with the preamble locator papering over it: on air that is the tool
    # answering six seconds late, with someone else's burst in the same piece.
    assert secs < 2.5, (
        f"the answer reached the handshake inside a {secs:.2f} s bracket — the gate "
        f"is not tracking the channel, it is force-closing at {kc.SEG_MAX_S} s")


@corpora.requires_onair_connect_attempt
def test_the_repeated_answer_is_recovered_from_its_tail(attempt):
    """KB9MMT answered a second time, into our own eighth connect-request.

    The repeat starts at 63.979 s and we were transmitting until 64.14, so the
    receive mute takes its first 0.28 s: the preamble, and enough of the burst
    that what reaches the gate — 0.85 s — is shorter than the 0.98 s a 23-symbol
    burst occupies. It is recognised anyway, from the payload, which is the only
    part of it still on the air, and the attempt re-sends the link-setup because
    a gateway repeating its response is a gateway that did not hear the first one.

    None of that is visible from a fixture cut around a burst. It is what the
    chain does with a gateway that answers while we are still calling it.
    """
    io, hs, _steps = attempt
    assert hs._linksetup_tx == 2, (
        f"the link-setup went out {hs._linksetup_tx}x; {GATEWAY} answered twice and "
        f"the repeat at {ANSWER_REPEAT_AT:.3f} s survives only as a headless tail, so "
        "it is recognisable by its payload and by nothing else")
    assert sum("located by payload" in m for m in io.logs) == 1
    assert sum(t > 4.0 for t in io.tx_secs) == 2


@corpora.requires_onair_connect_attempt
def test_the_stream_search_answers_each_answer_once(recording):
    """Both routes live, as they are at the radio, over the same 150 s.

    The stream search finds the connect-response without an energy gate, which is
    what the live path needs: in the SSB passband this answer sits *below* the band
    noise, and what opens the gate on it is out-of-band splatter. But the gate finds
    it too, a second or several later, and a second detection of one burst is
    indistinguishable downstream from a gateway repeating itself — so with both
    routes accepting, the two answers on this recording produce three link-setups,
    a wasted 4.4 s transmission out of a budget of three. The stream takes the
    connect-response and the brackets keep everything else; two answers, two
    link-setups, and each of them found by the route that saw it first.
    """
    io, hs, _steps = _replay(recording, GATEWAY, stream=True)
    assert hs._linksetup_tx == 2, (
        f"the link-setup went out {hs._linksetup_tx}x for the two answers "
        f"{GATEWAY} sent; a third is one route answering the other's burst")
    assert sum(t > 4.0 for t in io.tx_secs) == 2
    assert sum("found on the receive stream" in m for m in io.logs) == 2, (
        f"the stream search did not carry both answers: "
        f"{[m for m in io.logs if 'connect-response' in m]}")
    assert not any("preamble locked" in m and "CONNECT_RESPONSE" in m
                   for m in io.logs), "a bracket answered a connect-response as well"


@corpora.requires_onair_connect_attempt
def test_the_gate_never_loses_track_of_the_channel(recording):
    """Nothing runs to the force-close except a signal that earned it.

    Against the running-minimum floor this was 20 of 22 brackets: a force-close on
    band noise is the gate admitting it cannot tell signal from silence any more,
    and every burst it hands over after that has a boundary somewhere inside it.
    One bracket reaches the cap now, at 139.14 s, and it is 6.8 s of unbroken
    transmission from the third station — nine tenths of it above 0.385 against a
    0.255 median for the receive audio. That one is the cap doing its job, and it
    only appears at all because the floor stopped reading the traffic into itself
    and the gate could hear the station.
    """
    pieces = _brackets(recording)
    band = float(np.median(_frame_rms(np.concatenate(
        [recording[a:b] for a, b in _receive_windows(recording)]))))
    for t, p in pieces:
        if len(p) / FS < kc.SEG_MAX_S - 0.05:
            continue
        occupied = float(np.quantile(_frame_rms(p), 0.10))
        assert occupied > band, (
            f"the bracket at {t:.2f} s ran to the {kc.SEG_MAX_S} s cap with nine "
            f"tenths of it above only {occupied:.3f}, under the {band:.3f} median "
            f"of the receive audio — the gate lost the channel, it did not cut a "
            f"signal")
    assert sum(t >= INTERLOPER_FROM for t, _ in pieces) >= 5, (
        "the third station's traffic did not reach the handshake at all, so the "
        "false-accept tests below are not testing anything")


@corpora.requires_onair_connect_attempt
def test_nothing_else_in_the_recording_completes_a_connect(attempt):
    """The negative, over the same 150 s and the same brackets.

    The recording holds no connected-ack — KB9MMT repeated its connect-response
    instead — so a chain that claims CONNECTED here is claiming it from band
    noise, from our own transmissions leaking back, or from the third station that
    takes the frequency at 98.8 s.
    """
    io, hs, _steps = attempt
    assert hs.state is not VaraState.CONNECTED, (
        f"reported CONNECTED to {GATEWAY} from a recording with no connected-ack in it")
    assert not io.connects, f"announced a connection: {io.connects}"
    assert hs.step == _I_LINKSETUP_SENT
    assert hs._linksetup_tx <= ARQ._LINKSETUP_MAX_TX, (
        f"the link-setup went out {hs._linksetup_tx} times; the recording holds two "
        "connect-responses and nothing else that is one")


@corpora.requires_onair_connect_attempt
@pytest.mark.parametrize("wrong", ["KO2F", "W1AW", "N0CALL", MYCALL])
def test_a_station_that_is_not_on_the_air_is_never_answered(recording, wrong):
    """The same audio, dialled to someone who never transmitted on it.

    ``MYCALL`` is in the list on purpose: our own eight connect-requests are keyed
    to KB9MMT and our own callsign appears nowhere in them, so a chain that
    confused our transmissions with an answer would show up here.
    """
    io, hs, steps = _replay(recording, wrong)
    assert hs.step == _I_CR_SENT, f"a connect to {wrong} advanced: {steps}"
    assert not io.connects
    assert all(t < 4.0 for t in io.tx_secs), (
        f"a link-setup went out for {wrong}, who never answered: "
        f"{[round(t, 2) for t in io.tx_secs]} s")


@corpora.requires_onair_connect_attempt
def test_the_answer_is_unlike_everything_else_on_the_frequency(recording):
    """How much room the recogniser has, measured rather than asserted.

    Every bracket, scanned at fixed alignments, against KB9MMT's connect-response:
    the answer is 15/15 and the best of the other fourteen brackets — 150 s of
    band noise, our own receive-mute tails, and a third station's narrowband
    traffic — is 3/15, against an acceptance cut of 0.8. The headless tail of the
    repeat scores 0 here and is recognised anyway; a fixed-alignment scan cannot
    see a burst whose head is missing, which is what payload anchoring is for.
    """
    kind = VF.CONNECT_RESPONSE
    npre = len(kind.preamble)
    n_sym = npre + kind.n_payload
    scores = []
    for t, piece in _brackets(recording):
        best = 0
        span = len(piece) - MK.STRIDE - (n_sym - 1) * MK.HOP
        for off in range(0, max(1, span), FRAME):
            m, _n = VF.payload_match(MK.demod_tones(piece[off:], n_sym)[npre:],
                                     GATEWAY, kind)
            best = max(best, m)
        scores.append((t, best))
    answer = [m for t, m in scores if abs(t - ANSWER_AT) < 0.25]
    others = [m for t, m in scores if abs(t - ANSWER_AT) >= 0.25]
    assert answer == [kind.n_payload], f"the answer scored {answer}/{kind.n_payload}"
    assert max(others) <= 5, (
        f"something else on the frequency reached {max(others)}/{kind.n_payload}: "
        f"{[(round(t, 1), m) for t, m in scores if m > 5]}")


@corpora.requires_onair_connect_attempt
def test_the_ack_tone_track_reads_a_real_off_air_burst(recording):
    """The connected-ack recogniser's own tone reader, on real off-air MFSK.

    :func:`vara_arq._tone_track` is the front end of the response search, and it
    duplicates :func:`vara_mfsk.demod_tones` for speed — which is how it came to
    carry the same unbounded-argmax defect and keep it after ``demod_tones`` was
    fixed. No connected-ack was ever sent on this recording, so the end-to-end
    replay above cannot exercise it; the only real off-air MFSK we hold from this
    session is KB9MMT's connect-response, and the two readers must read it the
    same way. Band-limited it is 15/15; taking the peak of the whole rfft it is
    5/15, because most symbols read as their own 3f image.
    """
    kind = VF.CONNECT_RESPONSE
    npre = len(kind.preamble)
    n_sym = npre + kind.n_payload
    piece = next(p for t, p in _brackets(recording) if abs(t - ANSWER_AT) < 0.25)
    at = MK.lock_preamble(piece, kind)
    assert at is not None, "the preamble locator lost a burst it locks onto elsewhere"

    grid = np.rint((at + np.arange(n_sym) * MK.HOP + MK._WOFF) / ARQ._ACK_GRID).astype(int)
    track = ARQ._tone_track(np.asarray(piece, float))
    assert grid.max() < len(track)
    tones = track[grid]
    m, n = VF.payload_match(tones[npre:], GATEWAY, kind)
    assert (m, n) == (kind.n_payload, kind.n_payload), (
        f"_tone_track reads {m}/{n} of a burst demod_tones reads in full; the ack "
        f"recogniser is searching outside the tone alphabet — tones {list(tones)}")
    assert tones.min() >= MK.BIN_LO and tones.max() <= MK.BIN_HI
