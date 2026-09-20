# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Whether the channel `core.busy` refuses can say who is on it.

The gate returns a margin and no identity, and the operator's tools turn that into
a refusal. A refusal is right when the occupant is a third party and wrong when the
occupant is the station we are answering — so a waiver would have to rest on
POSITIVE identification of the occupant as our own correspondent. Nothing weaker
will do: keyed on energy it waives for a plain carrier, and keyed on the absence of
evidence it waives for every channel we cannot hear. The asymmetry is the whole of
the safety argument, and it survives only if the identification is real.

This station does read callsigns off the air, and `gwsurvey.who` is where each
protocol's answer is written down. What none of them reads is a station that is
merely *transmitting*: an ARDOP ConAck or data frame and a PACTOR-1 control signal
carry no callsign at all, and PACTOR's one addressed frame carries the callsign of
the station being CALLED (`docs/protocols/pactor/pactor1-link-request.md` §1 — "das
SLAVE-Rufzeichen"), so a connect burst overheard on a channel names the party being
called and never the station keying it. An occupant mid-session is exactly the case
with no name on it.

Measured here against the incident that raises the question — 2026-08-18, 7101500
kHz, refused at +12.0 dB with WS8EOC the only station known to be transmitting.

    the production front end over the occupant's own bursts
        153 recorded bursts, 101.1 s        201 detections, 0 name a station
    the callsign decoder over every alignment of the same audio and of two
    continuous recordings of Winlink gateways calling with our transmitter off
        240.1 s, 699 windows              41,108 sync alignments
                                          2 accepts, `C` and `XD`, 0 stations

Both halves are needed. The captures are per-cycle bursts, some of them shorter
than the 0.960 s a connect frame occupies, so on their own they could not say
whether a connect was decodable; the continuous recordings can, and the alignment
sweep searches them harder than the live path does.

What the two accepts are worth is the second finding. `p1rx.decode_connect` is a
structural lock — sync byte, character run, terminator — with no CRC anywhere in
the frame, so an accept is not an identification: swept ungated over the shared
off-air corpus, 902 recordings and 37,309.5 s, it returns 1,080 accepts over
144,203 windows and 9,372,437 sync alignments. A callsign-shape check does not
recover it either: `besra.frame.callsign` rejects `C` for length and passes `XD`,
which is the strictest structural test this tree has for a callsign.

So the identification a waiver would key on is not on the air in the state the gate
fires in, and the one that could fire is worth less than the refusal it would
overturn. Nothing here waives anything.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike import archive

FS = 48000

#: The gateway's own transmissions either side of the +12.0 dB refusal: 22:32 is
#: the session that had just been abandoned to it, 22:35 the forced retry two
#: minutes later, and 22:07 and 22:30 the same channel earlier in the same hour.
_CAPTURE_DIRS = ("onair-0818-2232", "onair-0818-2235",
                 "onair-0818-2230", "onair-0818-2207")

#: Callsigns the sweep must never mint. WS8EOC is the occupant of the incident,
#: W6IDS the gateway the second `silent.wav` is named for.
CORRESPONDENTS = ("WS8EOC", "W6IDS")

#: Sweep of the callsign decoder: a window wider than the 0.960 s connect frame,
#: stepped fine enough that no frame falls between two of them.
_WINDOW_S, _STEP_S = 1.5, 0.25

#: What the material has to offer before an absence of callsigns means anything.
#: Measured at 201 detections, 155 windows over the bursts and 544 over the two
#: streams. Absent material skips and says which; material that has SHRUNK fails,
#: because a floor measured over less than it names is a floor measuring nothing.
_MIN_DETECTIONS = 150
_MIN_WINDOWS = 600


def _bursts() -> list[Path]:
    return sorted(p for d in _CAPTURE_DIRS
                  for p in (evidence.CAPTURES / d).glob("*.wav"))


def _streams() -> list[Path]:
    return list(archive.SILENT_GATEWAYS)


_missing = [p for p in (*(evidence.CAPTURES / d for d in _CAPTURE_DIRS),
                        *archive.SILENT_GATEWAYS) if not p.exists()]
requires_occupant = pytest.mark.skipif(
    bool(_missing),
    reason=f"the occupant recordings are not present ({(_missing or [''])[0]}) — "
           "written by a run, in no clone, and named by "
           "tests/gates/test_corpus_present.py")


def _mono(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == FS, f"{path.name} is {w.getframerate()} Hz"
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768.0
    return x.reshape(-1, ch)[:, 0] if ch > 1 else x


@requires_occupant
def test_the_gateway_still_calling_us_names_nobody():
    """The live receive path over the occupant, which is the capability a waiver
    would rest on. Every detection is a control signal or a shape, and a shape
    names no station by construction (`rxfront.EVENT_KINDS`)."""
    heard, named = [], []
    for path in _bursts():
        for ev in rxfront.decode_events(_mono(path), hop_s=0.5):
            heard.append(ev.kind)
            if ev.connect is not None:
                named.append((path.name, ev.connect.callsign))
    assert len(heard) >= _MIN_DETECTIONS, (
        f"only {len(heard)} detections against {_MIN_DETECTIONS} measured — this "
        f"receiver has gone quieter on the occupant than the finding assumes, and "
        f"a silence proves nothing about identification")
    assert not named, (
        f"the occupant is named after all ({named}), so the measurement this "
        f"module rests on has moved and the waiver is worth re-deriving")


@requires_occupant
def test_no_alignment_of_the_occupants_audio_yields_a_correspondent():
    """The callsign decoder swept over every alignment of the same audio, plus the
    continuous recordings the per-cycle bursts are too short to answer for.

    An accept here would not be an identification even if it named the right
    station — the frame carries no CRC and the corpus rate is in this module's
    docstring — but a waiver would have keyed on one, so what it mints is measured
    rather than assumed."""
    width, step = int(_WINDOW_S * FS), int(_STEP_S * FS)
    windows, accepts = 0, []
    for path in [*_bursts(), *_streams()]:
        x = _mono(path)
        for i in range(0, max(1, x.size - width), step):
            windows += 1
            got = p1rx.decode_connect(x[i:i + width])
            if got is not None:
                accepts.append((path.name, round(i / FS, 2), got.callsign))
    assert windows >= _MIN_WINDOWS, (
        f"only {windows} windows swept against {_MIN_WINDOWS} measured — the "
        f"occupant recordings have shrunk and this floor is measuring less than "
        f"it says")
    minted = {c for _, _, c in accepts}
    assert not minted & set(CORRESPONDENTS), (
        f"the decoder minted a correspondent's callsign out of {windows} windows "
        f"of audio that carries no connect frame ({accepts}) — which is a false "
        f"accept naming the one station a waiver would have believed")
