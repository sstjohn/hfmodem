# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""VARA: how many gateway answers were in the audio, and how many we acted on.

The recorder here sits UPSTREAM of every drop the live path makes.
``AudioVaraIO._on_rx`` writes each callback frame to the WAV unconditionally and
only then appends it to the buffer the decoder reads from a cursor, so the file
holds the samples the decoder was denied. That is what makes this a measurement
rather than a restatement of the log.

Both arms run the *same* search — ``VaraStationHandshake`` on the receive stream,
with the same acceptance rule, re-armed after each accept so neither arm stops
counting — and differ in one thing only: which samples they are shown. The live
arm is shown what ``AudioVaraIO``'s cursor and bracket gate would have shown it;
the ungated arm is shown the recording. The difference is therefore the gating
and nothing else.

Alongside that, ``vara_monitor.HandshakeScanner`` reads the whole recording as an
independent census. It is a different and more permissive recogniser, and its
false-accept rate is the one measured in ``tests/kestrel/test_negative_corpus``:
nothing at all over 3.9 h of independently-labelled empty audio scored against
the whole published gateway panel. What it names and the handshake does not is
reported too, marked with which instrument found it.

Where our own transmissions are is decided by what is on the tape rather than by
how loud it is. The energy model in ``test_onair_connect_chain`` — runs of frames
15 dB under the median lasting over a second — was calibrated on 2026-07-26, when
the key was held 0.25 s past the audio and rigctld took a further 0.158 s to
unkey. Both are gone, and on this station's 2026-08-17 and 2026-08-19 sessions it
finds no transmissions at all where the scanner reads eight connect-requests on a
7 s cadence. So our overs are found from our own requests, whose length is fixed
by the waveform (41 tones, 1.777 s), and unioned with whatever mute the audio
does show — which on 2026-07-26 agrees with the request-derived span to 0.15 s at
both edges.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import kestrel_connect as kc
import vara_monitor as vm
from hfmodem.core import wav
from hfmodem.core import config
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _I_CONNECTED, _I_CR_SENT, _I_LINKSETUP_SENT, VaraIO, VaraStationHandshake,
)

from . import UPSTREAM, Heard, Rehearing, sessionlog

PROTOCOL = "kestrel"

FS = MK.FS
FRAME = vm.SEG_FRAME
CHUNK = vm.CHUNK

#: A burst the scanner reports at its first sample and the handshake reports at the
#: end of the block it fell in. One block is 0.1 s, one stream search pass gathers
#: 0.5 s before it scores, and a connect-response runs 1.0 s — so an accept can
#: land more than a second after the burst opened, and the same answer seen by two
#: instruments has to survive that or the diff invents losses.
SLACK_S = 2.0

#: How long our own connect-request occupies the channel. Fixed by the waveform,
#: not measured: 41 tones on the 2080-sample symbol stride.
CR_S = (len(VF.CR.preamble) + VF.CR.n_payload) * MK.STRIDE / MK.FS

#: How far before the first tone of a request the receiver is already ours. PTT
#: settles before the audio starts, and the scanner reads a burst's start from a
#: 32-sample lattice; 0.15 s covers both and matches the 0.11 s by which the
#: 2026-07-26 mute leads its own request.
PTT_LEAD_S = 0.15

#: What ``AudioVaraIO.tx`` skips past the end of our audio: the keyed idle hold
#: plus the codec-latency guard. Read from the tool rather than restated, because
#: it has changed twice and the recordings straddle the change.
CURSOR_SKIP_S = kc.TX_IDLE_HOLD_S + kc.RX_ECHO_GUARD_S

_ONAIR = re.compile(r"\d{8}T\d{6}Z-(?P<mine>[A-Za-z0-9]+)-(?P<peer>[A-Za-z0-9]+(?:-\d+)?)")
_CALLSIGN = re.compile(r"[A-Z0-9]{0,2}\d[A-Z]{1,3}(-\d+)?")

#: A VARA bandwidth token in a filename, bounded either side by a non-digit so a
#: dial or carrier frequency that happens to contain the same digits — 7100500,
#: 3586500 — cannot be read as one.
_BW_IN_NAME = re.compile(r"(?<![0-9])(2300|2750|500)(?![0-9])")


def _bw(path: Path, log, bw: str | None) -> str:
    """Which bandwidth this arm ran: the log's own ``originating:`` line first,
    then an explicit override, then the recording's and the log's own name, and
    2300 where none of the three say anything.

    ``VaraStationHandshake`` recognises a connect-response only in the alphabet
    its own ``bw`` names [vara_arq, VF.for_bw], so an arm searched at the wrong
    one finds nothing. The NS0A 0923-1845 arm connected at 2750 and a rehear
    fixed at 2300 reported zero answers.
    """
    if log is not None and getattr(log, 'selection', None) is not None:
        selected = sessionlog.select(log.selection, path)
        return str(selected.state['bandwidth_hz']) if selected.complete and selected.state.get('bandwidth_hz') else None
    if log is not None and log.bw:
        return log.bw
    if bw:
        return bw
    for name in (path.name, path.parent.name, log.path.name if log is not None else ""):
        if m := _BW_IN_NAME.search(name):
            return m.group(1)
    return "2300"


def _dialled(path: Path, *, selection=None) -> tuple[str, str] | None:
    """``(caller, called)`` this recording is a connect attempt for, or None.

    A session WAV is named for the pair and an off-air corpus session for the
    station in its directory, but `besra-7102300` and `clear_7101800` are named
    the same way and are neither — so the second half has to look like a callsign
    rather than merely sit where one would. An off-air corpus session names only
    the peer, so its caller is this station's configured callsign; with none
    configured it is not claimed.
    """
    if selection is not None:
        selected = sessionlog.select(selection, path)
        if not selected.complete:
            return None
        local, peer = selected.calls.get('local'), selected.calls.get('peer')
        return (peer, local) if selected.local_role == 'called' else (local, peer)
    m = _ONAIR.fullmatch(path.stem)
    if m and _CALLSIGN.fullmatch(m["peer"].upper()):
        return m["mine"].upper(), m["peer"].upper()
    if path.name == "rig_rx.wav":
        peer = path.parent.name.split("_")[0].upper()
        mine = config.mycall()
        if mine and _CALLSIGN.fullmatch(peer):
            return mine.upper(), peer
    return None


def claims(path: Path) -> bool:
    return _dialled(path) is not None


def _load(path: Path) -> np.ndarray:
    """Absolute codec scale, because ``_DEAF_DBFS`` is an absolute floor.

    Normalising a recording to its own peak moves every level in it, and the live
    path's comparability mask is the one thing here reading dBFS rather than a
    ratio — a normalised replay grants the decoder a receiver it did not have.
    """
    first, last = wav.analysis_bounds(path)
    return wav.read(path, FS, start=first, stop=last)


def _muted(x: np.ndarray) -> list[tuple[int, int]]:
    """Sample spans where the receiver went dead — our own key, when the rig mutes.

    Kept because it is the only witness to how long the mute outlasts the audio,
    and dropped as the sole model because a rig whose transmit monitor arrives at
    band level shows no dip at all.
    """
    n = len(x) // FRAME
    rms = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1)) + 1e-12
    dead = rms < np.median(rms) * 10 ** (-15 / 20)
    out, i = [], 0
    while i < n:
        j = i
        while j < n and dead[j]:
            j += 1
        if (j - i) * FRAME / FS > 0.3:
            out.append((i * FRAME, j * FRAME))
        i = j + 1
    return out


def _our_overs(x: np.ndarray, ours: list[float]) -> list[tuple[int, int]]:
    """Sample spans the live path never scans, one per transmission of ours.

    From key-up to where ``AudioVaraIO.tx`` puts its cursor, widened to any mute
    the audio itself shows lasting longer than that. The cursor is the decision
    the code makes; the mute is what the rig did; a peer answering has to survive
    both, so the blind span is the union.
    """
    spans = [(max(0, int((at - PTT_LEAD_S) * FS)),
              int((at + CR_S + CURSOR_SKIP_S) * FS)) for at in ours]
    for a, b in _muted(x):
        for i, (s, e) in enumerate(spans):
            if a < e and s < b:
                spans[i] = (min(s, a), max(e, b))
    return _merge(spans)


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _receive_windows(x: np.ndarray, overs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``(start, end)`` sample spans of everything but our own overs."""
    spans, cur = [], 0
    for a, b in overs:
        if a > cur:
            spans.append((cur, a))
        cur = b
    if cur < len(x):
        spans.append((cur, len(x)))
    return spans


class _Silent(VaraIO):
    """Everything the handshake drives, with the transmitter taken out."""

    def key(self, on: bool) -> None: ...
    def tx(self, samples) -> None: ...
    def pending(self) -> None: ...
    def connected(self, *info) -> None: ...
    def log(self, msg: str) -> None: ...


def _search(x: np.ndarray, windows: list[tuple[int, int]],
            caller: str, called: str, bw: str = "2300") -> tuple[list[float], float | None]:
    """Every connect-response our own search accepts over ``windows``.

    Both routes run, because both run at the radio: every chunk goes to
    ``on_rx_stream`` ahead of the gate, and every bracket the gate closes goes to
    ``on_rx_audio``. Re-originating at a modeled resend keys the handshake and
    resets its shared receive front, just as the live path does. History lost
    under that transmit mask cannot be spliced onto the next receive window.

    Re-armed after each accept so the search does not stop at the first one. The
    live path stops; the *measurement* must not, or the two arms would differ in
    how long they counted as well as in what they were shown, and the difference
    would no longer be the gating. Where the live path stopped is returned
    separately.
    """
    io = _Silent()
    hs = VaraStationHandshake([caller], io, bw=bw)
    seg = kc._BracketSegmenter()
    at_accept: list[float] = []
    connected: float | None = None
    hs.originate(called, caller)

    def step(was: str, at: float) -> None:
        nonlocal connected
        if hs.step == _I_CONNECTED and connected is None:
            connected = at
        if was == _I_CR_SENT and hs.step == _I_LINKSETUP_SENT:
            at_accept.append(at)
            hs.originate(called, caller)

    for base, end in windows:
        if hs.step == _I_CR_SENT and base:
            hs.originate(called, caller)          # the resend, as the tool does on cadence
        seg.restart()
        window = x[base:end]
        for i in range(0, len(window), CHUNK):
            chunk = window[i:i + CHUNK]
            was = hs.step
            hs.on_rx_stream(chunk)
            step(was, (base + i + len(chunk)) / FS)
            for off, piece in seg.push(chunk):
                was = hs.step
                hs.on_rx_audio(piece)
                step(was, (base + off) / FS)
    return at_accept, connected


def _scanned(x: np.ndarray, calls: list[str]) -> list[tuple[float, vm.Result]]:
    scanner = vm.HandshakeScanner(calls)
    return [(start / FS, r)
            for i in range(0, len(x), CHUNK)
            for start, r in scanner.push(x[i:i + CHUNK])]


def rehear(path: Path, log=None, bw: str | None = None, *, selection=None) -> Rehearing:
    selection = selection or getattr(log, 'selection', None) or sessionlog.recording_for(path)
    if selection is None:
        return _rehear(path, log, bw)
    chosen = sessionlog.select(selection, path)
    result = Rehearing(path, PROTOCOL, 0., "source sample clock", UPSTREAM)
    if not chosen.complete:
        result.note = 'Context incomplete: ' + str(chosen.reason)
        return result
    from copy import deepcopy
    from hfmodem.core import wav
    lo, hi = chosen.bounds
    result.seconds = (hi-lo)/chosen.rate
    for span in chosen.spans:
        rec = deepcopy(selection)
        rec['lanes'][chosen.lane]['selection'] = list(span.bounds)
        part = sessionlog.select(rec, path)
        audio = wav.read(path, FS, start=span.bounds[0], stop=span.bounds[1])
        decoded = _rehear(path, part.log, bw, selection=rec, _audio=audio)
        result.append(decoded, (span.bounds[0]-lo)/chosen.rate)
    return result


def _rehear(path: Path, log=None, bw: str | None = None, *, selection=None, _audio=None) -> Rehearing:
    """The session's own log is offered to every adapter and taken here for the
    signature and the bandwidth: kestrel's arms are a tone search over a whole
    capture, not a session decoder, so nothing else the log holds changes what
    either of them hears."""
    caller, called = _dialled(path, selection=selection or getattr(log, 'selection', None))
    x = _audio if _audio is not None else _load(path)
    station_bw = _bw(path, log, bw)
    whole = [(0, len(x))]
    found = _scanned(x, [called, caller])
    overs = _our_overs(x, [at for at, r in found if r.kind == "CR" and r.gateway == called])
    blind = sum(b - a for a, b in overs) / FS

    live, connected = _search(x, _receive_windows(x, overs), caller, called, bw=station_bw)
    open_ear, _ = _search(x, whole, caller, called, bw=station_bw)
    named = [at for at, r in found
             if r.kind == "connect-response" and r.gateway == called]

    what = f"connect-response {called}"
    ungated = [Heard(at, what, "our own search, ungated") for at in open_ear]
    ungated += [Heard(at, what, f"scanner only: {r.quality}") for at, r in found
                if r.kind == "connect-response" and r.gateway == called
                and not any(abs(at - o) <= SLACK_S for o in open_ear)]
    ungated.sort()

    # Once the live path has an answer it stops needing one, and a gateway
    # repeating on its own cadence past that is not a loss. So both censuses end
    # at the first live accept; what precedes it is an answer that arrived while we
    # still had none. Both, not one — truncating the ungated arm alone would leave
    # the live arm counting past a bar the other had stopped at.
    stop = live[0] + SLACK_S if live else None
    if stop is not None:
        ungated = [h for h in ungated if h.at <= stop]
        live = [at for at in live if at <= stop]

    census: dict[str, int] = {}
    for _at, r in found:
        census[f"{r.kind} {r.info}"] = census.get(f"{r.kind} {r.info}", 0) + 1

    return Rehearing(
        recording=path, protocol=PROTOCOL, seconds=len(x) / FS,
        zero="first sample of the session WAV; levels absolute dBFS on the codec's int16 scale",
        recorder=UPSTREAM,
        live=[Heard(at, what, "accepted live") for at in live],
        ungated=ungated, slack_s=SLACK_S,
        uncovered="connected-ack (the scanner does not model it and the initiator's "
                  "own accept is not independently witnessed), DATA overs, link-setup, "
                  "and any BW500 burst no candidate callsign regenerates",
        note=(f"dialled {called} as {caller} at BW{station_bw}; blind to {blind:.1f} s "
              f"of {len(x) / FS:.1f} s "
              f"({100 * blind * FS / len(x):.0f}%) across {len(overs)} transmissions"
              + (f"; connected at {connected:.1f} s" if connected else "")
              + f"; scanner names {len(named)} answer(s); census "
              + (", ".join(f"{n}x {k}" for k, n in sorted(census.items())) or "nothing")),
    )
