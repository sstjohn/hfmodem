# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Finding the gateway's answer when its preamble is under our own transmission.

On 2026-07-26 KB9MMT answered kestrel twice, and both answers are perfect: 15 of
15 payload tones at 56.021 s and again at 63.979 s. Only the first was ever found.
The repeat began 138 ms before our own eighth connect-request ended, so its eight
preamble symbols lie under our own signal and the receiver mute that follows it —
``lock_preamble`` has nothing to lock and the burst is discarded as silence.

The repeat is the one that matters. A gateway repeats because it heard no
link-setup, so the second answer is exactly the one an attempt is down to when the
first was lost. NS0A did the same thing on its own recording (responses at 9.773 s
and 12.139 s), so it is the protocol behaving normally rather than one bad night.

Every payload tone of a connect-response is fixed by the callsign we dialled, so
the burst can be located by what it says instead of by what precedes it. That is
strictly extra reach, and extra reach is where false accepts come from — hence the
second half of this file, which measures the floor rather than assuming it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _ACK_GRID,
    _I_CR_SENT,
    _I_LINKSETUP_SENT,
    _RESP_MIN_TONES,
    VaraStationHandshake,
    _payload_alignment,
)

kc = corpora.harness("kestrel_connect")

FS = MK.FS
RESP = VF.CONNECT_RESPONSE
NPRE = len(RESP.preamble)
NSYM = NPRE + RESP.n_payload
GATEWAY = "KB9MMT"
MYCALL = "W9SSJ"

# The two answers in the 2026-07-26 recording, and the bracket the segmenter opens
# around each. The second bracket opens 256 ms after its burst started, i.e. six
# symbols into a preamble of eight.
_FIRST, _REPEAT = 56.021, 63.979


class _IO:
    def __init__(self):
        self.log_lines: list[str] = []

    def key(self, on): ...
    def tx(self, samples): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): self.log_lines.append(msg)


def _awaiting_response(io=None, called: str = GATEWAY) -> VaraStationHandshake:
    hs = VaraStationHandshake([MYCALL], io or _IO(), bw="2300", mfsk_only=True)
    hs.originate(called, MYCALL)
    return hs


@lru_cache(maxsize=1)
def _attempt() -> np.ndarray:
    x = corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT)
    return x / (np.abs(x).max() or 1.0)


@lru_cache(maxsize=1)
def _brackets() -> list[tuple[int, np.ndarray]]:
    """Every burst the connect tool's segmenter hands the handshake, in order."""
    seg = kc._BracketSegmenter()
    x = _attempt()
    out = []
    for i in range(0, len(x), 4800):
        out += seg.push(x[i:i + 4800])
    return out + seg.flush()


def _bracket_holding(at_s: float) -> np.ndarray:
    """The bracket the segmenter opens over the answer that starts at ``at_s``.

    Indexed by the answer's first PAYLOAD symbol, not by where the burst begins:
    for the repeat those are 341 ms apart and the bracket opens between them,
    which is the whole difficulty.
    """
    payload = at_s * FS + NPRE * MK.HOP
    for t0, b in _brackets():
        if t0 <= payload < t0 + len(b):
            return b
    pytest.fail(f"the segmenter opened no bracket over the answer at {at_s} s")


def _worst_alignment(x: np.ndarray, calls) -> tuple[int, int]:
    """``(most payload tones any alignment confirms, alignments tried)``.

    Production's own ranking, not a re-spelling of it: ``_payload_alignment``
    answers with the best-fitting alignment's tones and ranks on the matched
    count, so the payload tones its answer confirms IS the most any alignment
    reaches for that callsign. This used to duplicate the whole hypothesis
    arithmetic -- grid, window offset, comparability -- so the calibration under
    ``_RESP_MIN_TONES`` would have gone quietly stale the day production's
    ranking moved. The alignment count is production's lattice too, quoted for
    the failure message.
    """
    x = np.asarray(x, float)
    x = x / (np.abs(x).max() or 1.0)
    tried = len(range(-(NSYM - 1) * MK.HOP, len(x) - MK.STRIDE + 1, _ACK_GRID))
    worst = 0
    for call in calls:
        tones = np.asarray(VF.handshake_tones(call, RESP), dtype=np.int32)
        heard = _payload_alignment(x, RESP, tones)
        worst = max(worst, int((heard[NPRE:] == tones[NPRE:]).sum()))
    return worst, tried * len(calls)


# --------------------------------------------------------------------------- #
# The answer we could not hear.
@corpora.requires_onair_connect_attempt
def test_the_repeat_buried_under_our_own_transmission_is_recovered():
    io = _IO()
    hs = _awaiting_response(io)
    hs.on_rx_audio(_bracket_holding(_REPEAT))
    assert hs.step == _I_LINKSETUP_SENT, (
        "a 15/15 connect-response from the gateway we dialled went unrecognised; "
        f"log: {io.log_lines}")
    assert any("located by payload" in m for m in io.log_lines), io.log_lines


@corpora.requires_onair_connect_attempt
def test_the_recovered_repeat_is_not_matched_to_our_own_callsign():
    """The response is keyed to the CALLED station. Dialling W9SSJ must find
    nothing in the same audio — otherwise the search matches bursts rather than
    callsigns, and every station on the band answers every call."""
    io = _IO()
    hs = _awaiting_response(io, called=MYCALL)
    hs.on_rx_audio(_bracket_holding(_REPEAT))
    assert hs.step == _I_CR_SENT, io.log_lines
    assert not any("located by payload" in m for m in io.log_lines), io.log_lines


@corpora.requires_onair_connect_attempt
def test_the_clean_answer_still_comes_in_through_the_preamble():
    """A clean preamble is better evidence, and free where the fallback is not, so
    the first answer must not start arriving through the fallback."""
    io = _IO()
    hs = _awaiting_response(io)
    hs.on_rx_audio(_bracket_holding(_FIRST))
    assert hs.step == _I_LINKSETUP_SENT, io.log_lines
    assert any("preamble locked" in m for m in io.log_lines), io.log_lines
    assert not any("located by payload" in m for m in io.log_lines), io.log_lines


@corpora.requires_onair_connect_attempt
def test_both_answers_are_now_reachable_where_one_was():
    """The whole point, stated as a count: the recording holds two answers."""
    found = 0
    for t0, b in _brackets():
        hs = _awaiting_response()
        hs.on_rx_audio(b)
        found += hs.step == _I_LINKSETUP_SENT
    assert found == 2, f"{found} of the two KB9MMT answers were recognised"


# --------------------------------------------------------------------------- #
# The false-accept floor, measured on real HF rather than on a noise model.
_WRONG = ("KB9MMT", "W9SSJ", "K7ABC", "NS0A", "KC9GHZ")

# Every region of the four off-air recordings that does NOT hold a connect-response,
# scored against all five callsigns. The excluded windows are the genuine answers:
# KB9MMT at 56.021 and 63.979 s, NS0A at 9.773 and 12.139 s.
_NON_RESPONSE = [
    ("2026-07-26 call, before the first answer", "attempt", 0.0, 55.0),
    ("2026-07-26 call, between the two answers", "attempt", 58.0, 63.0),
    ("2026-07-26 call, after the answers", "attempt", 66.0, 99.0),
    ("2026-07-26 call, the third station's traffic", "attempt", 99.0, 150.0),
    ("NS0A session, before its answers", "NS0A_2300", 0.0, 8.0),
    ("NS0A session, after its answers", "NS0A_2300", 14.0, None),
    ("KC9GHZ session", "KC9GHZ_2300", 0.0, None),
    ("clear channel", "clear", 0.0, None),
]
# Measured over those regions: 2,937,760 alignments of 288 s of real off-air HF —
# our own transmissions heard back through the receiver, five gateway sessions'
# worth of VARA bursts, a third station's narrowband traffic, and band noise — and
# the best any of them reaches is 5 of 15. The genuine answers reach 15 of 15.
_MEASURED_WORST = 5


def _region(source: str, a: float, b: float | None) -> np.ndarray:
    if source == "attempt":
        x = _attempt()
    else:
        path = (corpora.CLEAR_CHANNEL if source == "clear"
                else corpora.OFFAIR / source / "rig_rx.wav")
        if not path.exists():
            pytest.skip(f"off-air recording {source} not present")
        x = corpora.wav_mono(path)
    return x[int(a * FS):int(b * FS) if b is not None else None]


@corpora.requires_onair_connect_attempt
@pytest.mark.parametrize("label,source,a,b", _NON_RESPONSE,
                         ids=[c[0] for c in _NON_RESPONSE])
def test_real_off_air_audio_never_approaches_the_acceptance_bar(label, source, a, b):
    worst, tried = _worst_alignment(_region(source, a, b), _WRONG)
    assert worst <= _MEASURED_WORST, (
        f"{label}: an alignment confirms {worst}/15 response tones over {tried} "
        f"tried, above the {_MEASURED_WORST} the whole corpus was measured at. The "
        f"cut at {_RESP_MIN_TONES} is placed against that measurement and has to be "
        "re-measured, not nudged.")
    assert worst < _RESP_MIN_TONES


@corpora.requires_onair_connect_attempt
def test_the_third_station_never_answers_a_call_it_did_not_make():
    """From ~99 s the recording carries a third station's narrowband traffic — real
    signals, keyed on the same frequency, addressed to nobody here. Driven through
    the real segmenter, none of it may advance a connect attempt."""
    x = _attempt()[int(99 * FS):]
    seg = kc._BracketSegmenter()
    pieces = []
    for i in range(0, len(x), 4800):
        pieces += seg.push(x[i:i + 4800])
    pieces += seg.flush()
    assert pieces, "the segmenter found no traffic in a region known to carry it"
    for called in (GATEWAY, MYCALL):
        hs = _awaiting_response(called=called)
        for _, b in pieces:
            hs.on_rx_audio(b)
        assert hs.step == _I_CR_SENT, (
            f"a third station's traffic was taken for a connect-response to {called}")


@corpora.requires_clear_channel
def test_band_noise_never_answers():
    hs = _awaiting_response()
    x = corpora.wav_mono(corpora.CLEAR_CHANNEL)
    for i in range(0, len(x) - FS, FS):
        hs.on_rx_audio(x[i:i + 2 * FS])
    assert hs.step == _I_CR_SENT


def test_a_long_run_of_synthetic_junk_never_answers():
    """No accumulator here, but the search is wide, so a long run of unrelated
    audio is the population that finds the tail of any wide search."""
    rng = np.random.default_rng(19)
    t = np.arange(2 * FS) / FS
    hs = _awaiting_response()
    for i in range(300):
        n = int(rng.uniform(0.4, 2.5) * FS)
        hs.on_rx_audio(rng.standard_normal(n) * rng.uniform(0.01, 1.0))
        hs.on_rx_audio(0.5 * np.sin(2 * np.pi * rng.uniform(700, 2300) * t[:n]))
        assert hs.step == _I_CR_SENT, f"junk burst {i} was taken for a response"


def test_a_headless_response_for_another_station_is_rejected():
    """The fallback's own worst case: exactly the burst it is built to find, with
    exactly the wrong callsign. Its preamble is callsign-independent, so with the
    preamble gone the payload is the only thing left to tell them apart."""
    for wrong in ("K7ABC", "NS0A", "KC9GHZ", MYCALL):
        headless = MK.synth_burst(wrong, RESP)[NPRE * MK.HOP:]
        hs = _awaiting_response()
        hs.on_rx_audio(headless)
        assert hs.step == _I_CR_SENT, f"a headless response to {wrong} answered a call to {GATEWAY}"
    hs = _awaiting_response()
    hs.on_rx_audio(MK.synth_burst(GATEWAY, RESP)[NPRE * MK.HOP:])
    assert hs.step == _I_LINKSETUP_SENT, "a headless response to the dialled station was missed"


def test_the_bar_sits_above_the_whole_burst_rule():
    """The fallback searches thousands of alignments where the preamble path
    searches one, so it is held above the module's acceptance fraction (12 of 15),
    not at it — and the cut is exactly where it says it is."""
    assert _RESP_MIN_TONES > VF._ACCEPT_FRAC * RESP.n_payload
    tones = VF.handshake_tones(GATEWAY, RESP)
    for damaged in range(5):
        t = list(tones)
        for k in range(damaged):
            j = NPRE + 2 * k
            t[j] = t[j] - 2 if t[j] >= 32 else t[j] + 2    # same parity, wrong tone
        hs = _awaiting_response()
        hs.on_rx_audio(MK.synth_tones(t)[NPRE * MK.HOP:])
        accepted = hs.step == _I_LINKSETUP_SENT
        assert accepted is (RESP.n_payload - damaged >= _RESP_MIN_TONES), (
            f"a headless response with {damaged} of {RESP.n_payload} payload tones "
            f"wrong was {'accepted' if accepted else 'rejected'}")


# --------------------------------------------------------------------------- #
def test_the_fallback_stays_inside_the_per_burst_budget():
    """It runs inside a live ARQ turnaround, on whatever the segmenter brackets —
    up to its 6 s force-close, and more if the gate is held open."""
    import time
    rng = np.random.default_rng(1)
    blob = rng.standard_normal(9 * FS) * 0.05
    hs = _awaiting_response()
    hs.step = _I_LINKSETUP_SENT          # ack search, preamble lock and fallback, all of it
    t0 = time.perf_counter()
    hs.on_rx_audio(blob)
    dt = time.perf_counter() - t0
    assert dt < 1.0, f"a 9 s burst cost {dt:.2f} s of the turnaround"
