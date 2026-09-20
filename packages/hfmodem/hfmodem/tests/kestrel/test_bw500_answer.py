# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The BW500 receive half, over the 2026-08-28 call K5FIT answered.

The runbook had asked since 2026-08-26 whether BW500's receive half is deaf
against a real gateway, having only ever been exercised transmitting. It is not.
Calling K5FIT on 7105.4 kHz that afternoon this station keyed eight
connect-requests and reported ``NOT connected``, and the recording holds four
answers — 35.977 s at 15 of 15 tones, then 48.127, 60.196 and 98.732 s at 13 of
13. The operator heard the first of them before reading the log.

Every branch of the handshake that accepts an answer named BW2300's burst by
constant, so none of them could fire:

  * ``_expected_kind`` returned :data:`vara_frames.CONNECT_RESPONSE` whatever the
    session's bandwidth, so the preamble lock and the payload fallback in
    ``on_rx_audio`` both hunted the wide burst;
  * ``on_rx_stream`` regenerated the wide alphabet's fifteen payload tones, which
    is why three of the four answers left no trace anywhere — they never opened
    the energy gate, and the search that reads ungated audio was looking for
    tones K5FIT did not key;
  * the accept branch itself compared ``kind is VF.CONNECT_RESPONSE``, so the one
    answer the gate did bracket arrived correctly named ``connect-response-500``
    and fell through every branch to ``unexpected in state=CONNECTING
    role=initiator step=I_CR_SENT``.

The two bandwidths' handshake bursts carry the same preamble and the same symbol
counts and differ only in the payload alphabet, so a station that names the
request by bandwidth and the response by constant is waiting for a burst its peer
never keys. :func:`vara_frames.connect_response` is the missing half of
:func:`vara_frames.connect_request`, and the pair is now what the state machine
asks.

**Nothing here widens who may bring the link up.** The kind is chosen by our own
session bandwidth and then still has to satisfy ``VF.recognize`` against the
callsign we dialled, so an answer is accepted from the station we called, in the
step where we are waiting for it, and from nobody else: dialled at the four other
callsigns of that slot the same recording carries the handshake nowhere, and
neither does a call nobody answered.

The replay feeds the whole recording, this station's own eight transmissions
included. The live path skips them — ``AudioVaraIO.tx`` takes its cursor past
everything it played — so this is strictly the harder input, and the negative
arms are scored over it too.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (_I_CR_SENT, _I_LINKSETUP_SENT,
                                           VaraStationHandshake)

kc = corpora.harness("kestrel_connect")

FS = MK.FS
CHUNK = 4800                      # the device read size AudioVaraIO feeds the gate

MYCALL = "W9SSJ"
GATEWAY = "K5FIT"
#: When K5FIT answered, in the recording's own clock.
ANSWERS_AT = (35.977, 48.127, 60.196, 98.732)
#: The other stations this station dialled in the same slot. None of them was on
#: this channel, and none of them may take this recording anywhere.
STRANGERS = ("KE8LVA", "N5MDT", "WB4ULT", "W6IDS")


class _RecordingIO:
    """A :class:`VaraIO` that writes down what the handshake asked for.

    Nothing reaches an audio device, a serial port or rigctld: the point of
    replaying a recording is that the transmitter stays cold.
    """

    def __init__(self):
        self.tx_secs: list[float] = []
        self.logs: list[str] = []

    def key(self, on: bool) -> None:
        pass

    def tx(self, samples) -> None:
        self.tx_secs.append(len(samples) / FS)

    def pending(self) -> None:
        pass

    def connected(self, *info) -> None:
        pass

    def log(self, msg: str) -> None:
        self.logs.append(msg)


def _replay(x: np.ndarray, gateway: str, bw: str = "500"):
    """One connect attempt against ``gateway`` over the whole recording.

    Both routes an answer can arrive by are live: every chunk goes to
    :meth:`on_rx_stream` before the gate sees it, exactly as
    ``AudioVaraIO.next_rx_burst`` does at the radio, and every bracket the gate
    closes goes to :meth:`on_rx_audio`. Returns the IO, the handshake, and the
    recording time of each step the handshake took.
    """
    io = _RecordingIO()
    hs = VaraStationHandshake([MYCALL], io, bw=bw)
    seg = kc._BracketSegmenter()
    hs.originate(gateway, MYCALL)
    steps: list[tuple[float, str | None, str | None]] = []
    for i in range(0, len(x), CHUNK):
        chunk = x[i:i + CHUNK]
        was = hs.step
        hs.on_rx_stream(chunk)
        if hs.step != was:
            steps.append(((i + len(chunk)) / FS, was, hs.step))
        for at, piece in seg.push(chunk):
            was = hs.step
            hs.on_rx_audio(piece)
            if hs.step != was:
                steps.append((at / FS, was, hs.step))
    return io, hs, steps


@pytest.fixture(scope="module")
def recording() -> np.ndarray:
    x = corpora.wav_mono(corpora.ONAIR_BW500_ANSWER)
    return x / (np.abs(x).max() or 1.0)


def test_the_handshake_waits_for_the_bandwidth_it_called_on():
    """The transition table, without any audio in it.

    This is the whole defect in two lines: an initiator that has just keyed
    ``connect-request-500`` was waiting for the burst a BW2300 responder keys.
    """
    for bw, want in (("500", VF.CONNECT_RESPONSE_500), ("2300", VF.CONNECT_RESPONSE)):
        hs = VaraStationHandshake([MYCALL], _RecordingIO(), bw=bw)
        hs.originate(GATEWAY, MYCALL)
        assert hs.step == _I_CR_SENT
        assert hs._expected_kind() is want, (
            f"at BW{bw} the initiator waits for {hs._expected_kind().name}")


@corpora.requires_onair_bw500_answer
def test_the_answer_carries_the_handshake_to_the_link_setup(recording):
    """K5FIT's answer has to leave ``I_CR_SENT`` and put step 4 on the air.

    That is as far as this recording can carry an attempt: no connected-ack ever
    came back, and the three repeats after the first answer are a gateway saying
    it never heard the link-setup.
    """
    io, hs, steps = _replay(recording, GATEWAY)
    left = [t for t, was, now in steps if was == _I_CR_SENT and now == _I_LINKSETUP_SENT]
    assert left, ("the handshake never left I_CR_SENT; the log read "
                  + " / ".join(io.logs[:8]))
    assert abs(left[0] - ANSWERS_AT[0]) < 2.0, (
        f"left I_CR_SENT at {left[0]:.2f} s, expected K5FIT's first answer at "
        f"{ANSWERS_AT[0]:.3f} s")
    assert hs.step == _I_LINKSETUP_SENT
    confirmed = [m for m in io.logs if m.startswith(f"rx connect-response for {GATEWAY}")]
    assert len(confirmed) == len(ANSWERS_AT), (
        f"{len(confirmed)} of K5FIT's {len(ANSWERS_AT)} answers were read: {confirmed}")
    assert sum(m.startswith("tx link-setup") for m in io.logs) >= 1


@corpora.requires_onair_bw500_answer
@pytest.mark.parametrize("stranger", STRANGERS)
def test_the_same_recording_answers_nobody_else(recording, stranger):
    """Dialled at a station that is not on this channel, nothing is accepted.

    The recording holds four well-formed BW500 connect-responses, so this is the
    case a widened accept is most easily wrong about: the burst is real, it is
    the right kind, and it names somebody else.
    """
    io, hs, _ = _replay(recording, stranger)
    assert hs.step == _I_CR_SENT, f"{stranger} took the handshake to {hs.step}"
    assert not [m for m in io.logs if m.startswith("rx connect-response for")]


@corpora.requires_onair_bw500_answer
def test_the_wide_alphabet_finds_nothing_in_it(recording):
    """The same audio, the same callsign, read as BW2300 — and no answer in it.

    Which is why the constant could not stand in for the bandwidth: these are not
    two names for one burst.
    """
    io, hs, _ = _replay(recording, GATEWAY, bw="2300")
    assert hs.step == _I_CR_SENT
    assert not [m for m in io.logs if m.startswith("rx connect-response for")]


@corpora.requires_onair_bw500_answer
def test_a_call_nobody_answered_stays_unanswered(recording):
    """The control the positive needs: 70 s of the same rig calling and no reply.

    Scored at BW500 for the station that was dialled and for the one that
    answered a different call on a different day.
    """
    x = corpora.wav_mono(corpora.ONAIR_UNANSWERED_CALL)
    x = x / (np.abs(x).max() or 1.0)
    for call in ("KC9GHZ", GATEWAY):
        io, hs, _ = _replay(x, call)
        assert hs.step == _I_CR_SENT, f"an answer for {call} was found in a call nobody answered"
        assert not [m for m in io.logs if m.startswith("rx connect-response for")]
