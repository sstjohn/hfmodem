# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every way a peer's codeword can reach the state machine, and the cycles none did.

`test_flushdebt` pinned the flush's INPUT against a buffer that had been emptied
before it ran. This pins its REACH, and the answer is that the flush was never the
right instrument: it keeps `FLUSH_CONTEXT_S` of the buffer, and on any cycle the
listen loop fed nothing -- an IRS, a slot `_regrid` hands back, the front of a
window longer than `FEED_MAX_SLOTS` -- everything in front of the last 0.75 s
reached the WAV on disk and no decoder at all.

Widening it is the wrong answer twice over. A sweep is the dearest thing in the
receiver for the cheapest thing in the protocol -- 24 ms over 0.75 s of audio,
434 over 3.89 -- and on this station those milliseconds are lost samples: measured
2026-08-26, 23 of 23 capture-loss intervals sat inside a
`RollingRx` decode holding the interpreter, and lost samples are what walked the
peer's answer out of the turnaround band to begin with. So the codeword is read
where the burst detector already found a burst (`_read_codeword_at_bursts`), for
0.2-0.6 ms a window.

WS8EOC, 7101.5 kHz, 2026-08-26. The gateway answered every cycle of the grant and
went on answering for forty-five seconds after this station stopped listening for
it: `0x59A` -- the PACTOR-3 upgrade grant -- at ZERO bit errors on thirty-six
consecutive cycles of a 1.24999 s raster, twenty-one of them after the modem had
fallen back to PACTOR-1. Then a CW `WS8EOC` and gone. That reading is this
station's own field record; what is asserted here is the part of it that is a
claim about this package's receiver.

TWO RECORDINGS, AND THE FIRST IS SOMEBODY ELSE'S. `witness.wav` is a KiwiSDR at
Empire, Michigan, 103 miles away, on its own clock and its own dial -- a file this
station's transmitter never touched. All thirty-six read at zero bit errors
through `p1rx.decode_control_signal`, with the shift sense alternating on every
cycle as PACTOR-1's Shiftlage requires and no artefact produces. That is the
control-signal reader validated against a real gateway rather than against our own
render, which is the bar this project sets and has four times paid for missing.

The second is our own. Sixteen of the thirty-six are in the session's own hold
windows at zero bit errors; the other twenty arrived while our carrier was up and
are gone for good. The session presented TEN of the sixteen to the state machine.
Five reached it through the grid-anchored read and five through the flush;
six were in front of the trim and no reader was pointed at them -- among them `hold_23`, whose codeword
sits 1.13 s before its window's end, 17-29 dB over its own guard bands, in a
window the log records as

    HOLD 23 RX (quiet)

Run:  pytest hfmodem/tests/shrike/test_codeword_reach.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import onair, p1rx, rxfront
from hfmodem.shrike.arq import IRS
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.session import load_wav
from hfmodem.tests import evidence

FS = rxfront.FS
CHUNK = int(0.25 * FS)

#: The arm the whole reading comes off.
SESSION = evidence.CAPTURES / "onair-0826-1041"

#: The gateway's raster, in the witness's timebase: `k = 0` is its first control
#: signal of the grant and `k = 5` the first `0x59A`.
RASTER_T0 = 42.102
RASTER_S = 1.24999
GRANT = range(5, 41)

#: `0x59A`, the word the gateway spent forty-five seconds sending. PACTOR-1 gives
#: it no meaning; this station reads it as the peer's PACTOR-3 grant.
GRANT_WORD = 5

#: Every hold window of that session whose audio the flush is the last chance for,
#: against the zero-error `0x59A` codewords the recording holds. The short keyed
#: windows are not here: at 0.24-0.34 s they are under `RollingRx`'s half-second
#: decode floor, no flush runs on them at all, and their five codewords reached
#: the state machine through the grid-anchored read instead.
HELD = {"hold_08": 1, "hold_09": 1, "hold_10": 1, "hold_11": 2, "hold_12": 1,
        "hold_13": 3, "hold_14": 1, "hold_23": 1}

#: What the session actually delivered out of those eleven, from its own log:
#: hold_08, 09, 10, 12 and 14. The other six were never read by anything.
DELIVERED_ON_THE_AIR = 5

#: Windows of the same session that hold NO gateway at all: WS8EOC had stopped
#: and IDed in CW by then, and the burst detector still finds eight onsets across
#: them. Not one may produce a codeword.
EMPTY = ("hold_24", "hold_25", "hold_26", "hold_27", "hold_28", "hold_29",
         "hold_30", "hold_31", "hold_32")

requires_session = pytest.mark.skipif(
    not SESSION.is_dir(), reason=f"{SESSION} is not on this machine")


def _sessrx() -> tuple[onair._SessionRx, list]:
    """A receiver in the state the session was in: linked, receiving, holding."""
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    seen: list = []
    host.on_rx_event = seen.append
    return onair._SessionRx(host, tag="TEST"), seen


class _Seam:
    """A transmit seam that keys nothing. The FSM needs one to reach CONNECTED."""

    def attach(self, host) -> None:
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


def _held_cycle(audio: np.ndarray) -> list:
    """One receiving cycle, as the hold loop runs it: hold the whole window,
    transmit, flush behind the carrier, then read at the cycle's own onsets."""
    rx, seen = _sessrx()
    for i in range(0, audio.size, CHUNK):
        rx.bridge(audio[i:i + CHUNK])
    rx.skip(0.96)
    rx.flush()
    onair._read_codeword_at_bursts(
        rx, audio, 0, [at for at, _ in onair._peer_bursts(audio, 0)])
    return seen


def _window(name: str) -> np.ndarray:
    return load_wav(SESSION / f"{name}.wav").astype(np.float32)


#: How wide a run of accepting alignments a real codeword answers over. `cs_bits`
#: positions its window on the signal rather than on the caller's guess, so a
#: burst that is there reads the same word across tens of milliseconds of aim;
#: the gateway's own answer over 37-90 ms on these recordings. Twelve bits of
#: noise landing on one of six words at one alignment and no neighbour is the
#: other thing this sweep finds, three times in the session, and it is not one.
SWEEP_RUN_S = 0.020


def _sweep(audio: np.ndarray, step: float = 0.0025) -> list[float]:
    """Where a zero-error `0x59A` reads over a run of alignments, seconds in.

    The ground truth the receiver is scored against, and it is deliberately not
    the receiver: a bare codeword comparison at every alignment, with no burst
    detector, no gate and no window in front of it.
    """
    hits, at = [], onair.ACQUIRE_T0
    while at + 0.15 < audio.size / FS:
        got = p1rx.decode_control_signal(audio, at, 0.12)
        if got is not None and got.errors == 0 and got.index == GRANT_WORD:
            hits.append(at)
        at += step
    runs: list[list[float]] = []
    for at in hits:
        if runs and at - runs[-1][-1] <= 2 * step:
            runs[-1].append(at)
        else:
            runs.append([at])
    return [r[0] for r in runs if r[-1] - r[0] >= SWEEP_RUN_S]


# -- the gateway's transmissions, out of a receiver that is not ours -----------

@requires_session
def test_the_witness_holds_thirty_six_grants_at_zero_bit_errors() -> None:
    fs = json.loads((SESSION / "witness.json").read_text())["sample_rate"]
    raw = wavfile.read(SESSION / "witness.wav")[1]
    a = np.asarray(raw)
    if a.ndim > 1:
        a = a[:, 0]
    audio = a.astype(np.float64) / -np.iinfo(a.dtype).min

    read = {}
    for k in GRANT:
        due = RASTER_T0 + k * RASTER_S
        # The raster is the gateway's and the search is around it, not for it.
        # +-40 ms covers the two cycles our own carrier took the head off; the
        # other thirty-four land inside the 2.5 ms step.
        for off in np.arange(-0.040, 0.0401, 0.0025):
            got = p1rx.decode_control_signal(audio, due + off, 0.12, fs=fs)
            if got is not None and got.errors == 0:
                read[k] = got
                break

    assert sorted(read) == list(GRANT), (
        f"the witness holds thirty-six consecutive grants; "
        f"{sorted(set(GRANT) - set(read))} did not read")
    assert {cs.index for cs in read.values()} == {GRANT_WORD}
    # "Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage invertiert":
    # a real station's sense alternates every cycle and nothing else does.
    senses = [read[k].sense for k in GRANT]
    assert senses == [k % 2 for k in GRANT] or senses == [1 - k % 2 for k in GRANT]


# -- and out of ours ----------------------------------------------------------

@requires_session
def test_hold_23_holds_the_codeword_the_session_called_quiet() -> None:
    audio = _window("hold_23")
    at = _sweep(audio)
    assert len(at) == 1, "hold_23 holds exactly one grant"
    # 1.13 s before the window's end, which is what put it outside a 0.75 s trim.
    assert at[0] == pytest.approx(0.925, abs=0.01)
    assert audio.size / FS - at[0] == pytest.approx(1.163, abs=0.01)


@requires_session
def test_the_receiver_presents_hold_23_to_the_decoder() -> None:
    got = _held_cycle(_window("hold_23"))
    assert [ev for ev in got if getattr(ev, "spare", None) == GRANT_WORD], (
        "hold_23's grant reached no decoder: 2.088 s of held channel, a codeword "
        "at zero bit errors 969 ms in with an onset on it to the millisecond, "
        "and a flush that kept the last 0.75 s")


@requires_session
def test_every_grant_a_held_window_holds_reaches_the_decoder() -> None:
    reached = {}
    for name, expected in HELD.items():
        audio = _window(name)
        assert len(_sweep(audio)) == expected, f"{name} ground truth moved"
        reached[name] = len([ev for ev in _held_cycle(audio)
                             if getattr(ev, "spare", None) == GRANT_WORD])
    # Both paths, as the cycle runs them: the flush reads what falls in the last
    # 0.75 s and the onset read reads the rest, each declining what the other has
    # already delivered. `hold_08` is the one the onset read alone would miss --
    # its onset lands 58 ms past the codeword, outside `CS_SEARCH_S` -- and the
    # flush has it, which is why both are kept and neither is widened.
    assert reached == HELD, (
        f"the receiver reached {sum(reached.values())} of {sum(HELD.values())} "
        f"grants these windows hold: {reached}")
    assert sum(reached.values()) > DELIVERED_ON_THE_AIR


@requires_session
def test_the_windows_with_no_gateway_in_them_produce_no_codeword() -> None:
    """The other half of the bar, and the one a widened window fails.

    These nine windows are after 92.102 s, where the gateway had signed off in CW
    and left. Eight onsets across them -- our own T/R tails and whatever else was
    on 7101.5 -- and the reader must decline every one.
    """
    for name in EMPTY:
        assert not _held_cycle(_window(name)), name


@pytest.mark.realtime
@requires_session
def test_reading_at_the_onsets_costs_a_fraction_of_a_sweep() -> None:
    """What the receiver may spend to find a 120 ms codeword.

    Not `test_rxcost`'s "less than the audio it covers", which a sweep of a held
    window passes while still costing 434 ms of a 1.25 s cycle -- and on this
    station a decode that holds the interpreter is where the capture loses the
    samples that walk the peer's answer out of the band. One millisecond a window
    is the bar, and the read measures 0.2-0.6.
    """
    for name in HELD:
        audio = _window(name)
        rx, _ = _sessrx()
        onsets = [at for at, _ in onair._peer_bursts(audio, 0)]
        started = time.perf_counter()
        onair._read_codeword_at_bursts(rx, audio, 0, onsets)
        spent = (time.perf_counter() - started) * 1e3
        assert spent < 5.0, f"{name}: {spent:.2f} ms over {len(onsets)} onsets"


# -- and the guarantee itself, with no recording in front of it ---------------

def test_a_codeword_in_front_of_the_trim_is_still_read() -> None:
    """A burst the cycle detected is read wherever in the window it sits.

    The recordings above are the case that happened; this is the rule, and it is
    what stops the same class recurring behind a different constant.
    """
    from hfmodem.shrike import pactor1

    cs = np.asarray(pactor1.control_signal(pactor1.CS_ACK_A, repeats=3),
                    np.float32)
    rng = np.random.default_rng(11)
    early = np.concatenate([
        rng.normal(0, 0.002, int(0.30 * FS)).astype(np.float32), cs,
        rng.normal(0, 0.002,
                   int(2 * onair.FLUSH_CONTEXT_S * FS)).astype(np.float32)])
    assert _held_cycle(early), (
        "a codeword further back in the window than FLUSH_CONTEXT_S was never "
        "presented to the decoder")
