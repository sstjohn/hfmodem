# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A branch-B train must not read as quiet.

`p1rx.decode_call_b_all` reads the Robust Call and the two Free Signals, and
`test_callb.py` holds it to the air. This is the other half: the receive front
end has to RUN it, or a monitor transcript beside a channel somebody is calling
on says "nothing heard" -- which is what the transcripts beside the two July
Robust Call trains say (FREE-SIGNAL-ROBUST-CONNECT-PLAN-0915 §2.3).

The line an operator reads is `###CONNECT: [<kind>: <ident>]` -- an independent
monitor's own wording, which is what the corpus notes are written in.

The tone pair is swept unless a link is up, so a train on somebody else's pair --
which is every train on file -- is found as readily as one on ours.

Run:  pytest packages/hfmodem/hfmodem/tests/shrike/test_free_signal_event.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import monitor, pactor1, rxfront, session
from hfmodem.tests.kestrel import corpora

FS = rxfront.FS
RASTER_S = 1.25
"""The connect raster the corpus trains key on: bursts 0.88 s long, 1.25 s apart."""

LABEL = {"fs_normal": "Free Signal Normal",
         "fs_encrypted": "Free Signal Encrypted",
         "robust": "Robust Call"}

OFFAIR_TRAIN = corpora.RF_CORPUS / "10145k_055704.wav"
"""K6SDR calling on 10.145 MHz on 2026-07-24, tones 607/807 -- nowhere near this
station's own pair, and an independent monitor reads ten calls off it. It is the
tape whose transcript said "no PACTOR signal detected", and the reason the front
end cannot put an energy gate at 1400/1600 in front of this decode."""

NORMAL_CONNECT = corpora.RF_CORPUS / "7101k_054347.wav"


def _train(call: str, kind: str, bursts: int = 4, shift_hz: float = 0.0,
           seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 0.01, int((1.5 + bursts * RASTER_S) * FS))
    for i in range(bursts):
        burst = pactor1.build_call_b(call, kind, invert=bool(i % 2), amp=0.3)
        if shift_hz:
            n = np.arange(burst.size)
            burst = np.real(hilbert(burst) * np.exp(2j * np.pi * shift_hz * n / FS))
        at = int((0.5 + i * RASTER_S) * FS)
        audio[at:at + burst.size] += burst
    return audio


def _events(audio: np.ndarray, kind: str) -> list[rxfront.Event]:
    return [ev for ev in rxfront.decode_events(audio) if ev.kind == kind]


@pytest.mark.parametrize("kind", list(LABEL))
@pytest.mark.parametrize("shift_hz", [0.0, 40.0])
def test_a_train_is_reported_burst_by_burst(kind: str, shift_hz: float) -> None:
    call = "MAILHOST" if kind.startswith("fs") else "KY4RY"
    events = _events(_train(call, kind, shift_hz=shift_hz), "callb")
    assert [ev.text for ev in events] == \
        [f"###CONNECT: [{LABEL[kind]}: {call}]"] * 4
    assert [ev.connect.callsign for ev in events] == [call] * 4
    # On the raster it was keyed on, not at the hop that happened to find it: the
    # hops overlap four deep and each of them reads the same burst.
    assert np.allclose([ev.t for ev in events],
                       [0.5 + i * RASTER_S for i in range(4)], atol=0.05)


def test_a_normal_connect_is_not_one() -> None:
    """The two frames share the tones and nothing else, and the front end must not
    answer for one with the other: branch A's address is the CALLER's callsign and
    branch B's is the station being called."""
    audio = np.concatenate([
        pactor1.connect_signal("KB5LZK", amp=0.3) for _ in range(3)])
    audio = audio + np.random.default_rng(3).normal(0, 0.01, audio.size)
    assert _events(audio, "callb") == []
    assert [ev.text for ev in _events(audio, "connect")] == \
        ["###CONNECT: [Normal Call: KB5LZK]"] * 3


def _corpus(path) -> np.ndarray:
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    return session.load_wav(str(path), FS)


def test_the_train_whose_transcript_said_quiet() -> None:
    texts = [ev.text for ev in _events(_corpus(OFFAIR_TRAIN), "callb")]
    assert set(texts) == {"###CONNECT: [Robust Call: K6SDR]"}
    assert len(texts) >= 10, "the independent monitor reads ten calls off this tape"


def test_a_real_connect_on_tape_is_still_a_connect() -> None:
    audio = _corpus(NORMAL_CONNECT)
    assert _events(audio, "callb") == []
    assert [ev.text for ev in _events(audio, "connect")] == \
        ["###CONNECT: [Normal Call: K0NTS]"]


def test_the_monitor_has_a_tag_for_it() -> None:
    assert monitor._TAG["callb"] == "CALL-B"
