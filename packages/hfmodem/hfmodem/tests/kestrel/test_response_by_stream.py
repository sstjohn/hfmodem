# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Finding the gateway's answer without an energy gate to hand it over.

The segmenter's gate opens on broadband frame RMS, and inside the 400-2700 Hz SSB
passband KB9MMT's reply of 2026-07-26 carries less power than the band noise a
second after it. Three quarters of its energy is out of band, in the odd-harmonic
images of an overdriven receive chain, and that splatter is the whole of the
contrast the gate reads. A receiver that does not splatter hands the gate nothing.

So the answer is looked for directly, over the raw receive stream, by the tones it
is bound to carry for the callsign we dialled. The first half of this file is that
capability; the second half is its false-accept floor, measured over every real
off-air recording the project holds rather than assumed from a noise model.
"""
from __future__ import annotations

import csv
import gc
import re
import time
from functools import lru_cache

import numpy as np
import pytest
from scipy.signal import butter, resample_poly, sosfiltfilt

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _ACK_GRID,
    _CLEAR_DB,
    _DEAF_DBFS,
    _I_CR_SENT,
    _I_LINKSETUP_SENT,
    _RESP_MIN_HEARD,
    _RESP_MIN_TONES,
    _RESP_OFF,
    _RESP_SHIFTS,
    _RESP_SPAN,
    _STREAM_BLOCK,
    VaraState,
    VaraStationHandshake,
    _answers,
    _live_track,
    _tone_track,
    _top3_track,
)

FS = MK.FS
RESP = VF.CONNECT_RESPONSE
NPRE = len(RESP.preamble)
GATEWAY = "KB9MMT"
MYCALL = "W9SSJ"

# KB9MMT answered twice; both are 15/15. The repeat began 138 ms before our own
# eighth connect-request ended.
_FIRST, _REPEAT = 56.021, 63.979

# The SSB passband of the rig that made these recordings. Everything the receiver
# actually delivers is inside it; anything outside is the transmit chain's own
# distortion products arriving through an overdriven front end.
_PASSBAND = (400.0, 2700.0)
_SOS = butter(8, list(_PASSBAND), btype="band", fs=FS, output="sos")

# Callsigns to score real audio against. None of them is on any of these
# recordings except where the recording is named for one, and those are excluded
# per file — a genuine answer is a positive, not a false accept.
_PANEL = ("KB9MMT", "NS0A", "KC9GHZ", "KO2F", "W9SSJ", "K7ABC", "W1AW", "N0CALL")

# Measured over the whole corpus below: 16,122,114 alignments of 1443 s of real
# off-air HF, of which the best non-response reaches 5 of 15.
_MEASURED_WORST = 5


def band_limit(x: np.ndarray) -> np.ndarray:
    """What a receiver that is not splattering delivers."""
    return sosfiltfilt(_SOS, np.asarray(x, dtype=np.float64))


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.abs(x).max() or 1.0)


class _IO:
    def __init__(self):
        self.log_lines: list[str] = []

    def key(self, on): ...
    def tx(self, samples): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): self.log_lines.append(msg)


def _awaiting(io=None, called: str = GATEWAY) -> VaraStationHandshake:
    hs = VaraStationHandshake([MYCALL], io or _IO(), bw="2300", mfsk_only=True)
    hs.originate(called, MYCALL)
    return hs


def _feed(hs: VaraStationHandshake, x: np.ndarray, block: int = 4800) -> list[float]:
    """Push ``x`` through the stream search; the receive times of every accept."""
    at = []
    for i in range(0, len(x), block):
        before = hs._linksetup_tx
        hs.on_rx_stream(x[i:i + block])
        if hs._linksetup_tx != before:
            at.append((i + block) / FS)
    return at


def _handed(hs: VaraStationHandshake) -> list[list[int]]:
    """Every burst the stream search hands the recogniser, collected as it goes.

    What :func:`_feed` reports is ``VF.recognize``'s verdict rather than the search's
    own: a search that accepts band noise and a recogniser that turns it away are two
    facts, and only the second is visible from the handshake's step. That gate
    predates the mute mask, so with the mask read as "assume it matched" this file
    passed as it stood. Measured with that reading planted, over the populations
    THIS file feeds — the accept's own arithmetic fires at

        silent_kd9usw.wav (71 s)                     30 alignments,     0 handed
        silent_w8mw.wav   (71 s)                    303 alignments,     1 handed
        30 s of noise, 0.42 s mutes                 559 alignments,     0 handed
        30 s of noise, 4.4 s mutes                6,158 alignments,    20 handed

    and the step never moves through any of it. The long mute is the one that bites,
    and a single figure for "the synthetic mutes" hid it: 4.4 s covers all fifteen
    payload symbols, so "assume it matched" sweeps a whole payload out of band noise
    (29,620 alignments if the accept's own ``_reset_stream`` is suppressed as well,
    which is the exposure rather than what the search as wired reaches). Asserting
    here puts the recogniser behind the assertion instead of in front of it.
    """
    seen: list[list[int]] = []
    inner = hs.on_rx_tones

    def spy(tones, kind):
        seen.append(list(tones))
        inner(tones, kind)

    hs.on_rx_tones = spy
    return seen


@lru_cache(maxsize=1)
def _attempt() -> np.ndarray:
    return _norm(corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT))


def _in_band_db(x: np.ndarray, a: float, b: float) -> float:
    seg = x[int(a * FS):int(b * FS)]
    return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-30)


# --------------------------------------------------------------------------- #
# Why the search exists: the gate has nothing to see.
@corpora.requires_onair_connect_attempt
def test_the_answer_is_below_the_noise_the_gate_measures_it_against():
    """Band-limited, the reply carries less power than the band noise that follows
    it — so the contrast an energy gate reads is out-of-band splatter, and a
    correctly-configured receiver removes it along with the answer."""
    raw, band = _attempt(), band_limit(_attempt())
    end = _FIRST + (NPRE + RESP.n_payload) * MK.HOP / FS
    reply_raw, reply_band = _in_band_db(raw, _FIRST, end), _in_band_db(band, _FIRST, end)
    noise_raw, noise_band = _in_band_db(raw, 57.1, 59.0), _in_band_db(band, 57.1, 59.0)
    assert reply_raw > noise_raw + 3.0, (
        f"unfiltered, the reply is only {reply_raw - noise_raw:.1f} dB over the band "
        "noise — the recording no longer holds the splatter this is about")
    assert reply_band < noise_band, (
        f"inside {_PASSBAND[0]:.0f}-{_PASSBAND[1]:.0f} Hz the reply is "
        f"{reply_band - noise_band:+.1f} dB against the band noise; if it has come "
        "up above it, the gate can see it after all and this route is redundant")


# --------------------------------------------------------------------------- #
# The capability.
@corpora.requires_onair_connect_attempt
def test_both_answers_are_found_on_the_band_limited_stream():
    """The whole point: with the splatter filtered away — which is what a receiver
    that is not overdriven delivers — both of KB9MMT's answers still advance the
    connect, with no gate and no bracket anywhere in the path."""
    io = _IO()
    hs = _awaiting(io)
    at = _feed(hs, band_limit(_attempt()))
    assert hs.step == _I_LINKSETUP_SENT, io.log_lines
    assert len(at) == 2, f"{len(at)} of the two KB9MMT answers were found: {at}"
    burst = (NPRE + RESP.n_payload) * MK.HOP / FS
    for want, got in zip((_FIRST, _REPEAT), at):
        # recognised as the answer ends, give or take a pass — where a bracket
        # waits for the gate to close, up to six seconds later
        assert burst - 0.1 <= got - want <= burst + 2 * _STREAM_BLOCK / FS, (want, got)
    swept = [m for m in io.log_lines
             if re.search(r"stream \((\d+)/\1 tones", m)]
    assert len(swept) == 2, io.log_lines


@corpora.requires_onair_connect_attempt
def test_removing_the_splatter_changes_nothing():
    """:func:`_tone_track` already confines its argmax to the tone alphabet, which
    lies inside the SSB passband, so the search neither gains from the out-of-band
    energy nor needs a filter of its own."""
    raw_at = _feed(_awaiting(), _attempt())
    band_at = _feed(_awaiting(), band_limit(_attempt()))
    assert raw_at == band_at, (raw_at, band_at)


@corpora.requires_onair_connect_attempt
def test_the_stream_search_does_not_answer_a_call_we_did_not_make():
    """A connect-response is keyed to the CALLED station. Dialling ourselves over
    the identical audio must find nothing, or the search matches bursts rather than
    callsigns and every station on the band answers every call."""
    io = _IO()
    hs = _awaiting(io, called=MYCALL)
    assert _feed(hs, band_limit(_attempt())) == []
    assert hs.step == _I_CR_SENT, io.log_lines


@corpora.requires_onair_connect_attempt
def test_the_third_stations_traffic_answers_nobody():
    """From ~99 s the recording carries a third station's narrowband traffic — real
    signals, keyed on the same frequency, addressed to nobody here. It is the best
    adversarial population in the corpus, and the stream sees all of it."""
    x = band_limit(_attempt()[int(99 * FS):])
    for called in (GATEWAY, MYCALL, "NS0A"):
        hs = _awaiting(called=called)
        assert _feed(hs, x) == [], f"a third station's traffic answered a call to {called}"


@corpora.requires_clear_channel
def test_band_noise_answers_nobody():
    x = band_limit(_norm(corpora.wav_mono(corpora.CLEAR_CHANNEL)))
    for called in (GATEWAY, MYCALL, "NS0A"):
        assert _feed(_awaiting(called=called), x) == []


# --------------------------------------------------------------------------- #
# Scope, arithmetic and state.
def test_the_search_runs_only_while_an_initiator_waits():
    """An unbounded search in every state is a false accept looking for a place to
    happen. It is armed by :meth:`_expected_kind`, so there is one definition of the
    window, and it holds no state outside it."""
    answer = MK.synth_burst(GATEWAY, RESP)
    for setup in (lambda hs: hs.listen(True),                     # LISTENING
                  lambda hs: setattr(hs, "state", VaraState.CONNECTED)):
        hs = VaraStationHandshake([MYCALL], _IO(), bw="2300", mfsk_only=True)
        hs.called = GATEWAY
        setup(hs)
        before = hs.state
        _feed(hs, answer)
        assert hs.state == before and hs.step is None
        assert len(hs._st_buf) == 0 and len(hs._st_track) == 0


def test_one_answer_advances_the_handshake_once():
    """One burst matches at every lattice point across its ~1400-sample plateau.
    Each of those would otherwise re-send the 4.4 s link-setup and exhaust the
    retry budget on a single answer."""
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.standard_normal(3 * FS) * 0.02,
                        MK.synth_burst(GATEWAY, RESP),
                        rng.standard_normal(3 * FS) * 0.02])
    hs = _awaiting()
    assert len(_feed(hs, x)) == 1
    assert hs._linksetup_tx == 1


@pytest.mark.parametrize("shift", [0, 7, 16, 31, 1024, 1039])
def test_an_answer_off_the_lattice_is_still_found(shift):
    """The search hypothesises burst starts every 32 samples. A real burst begins
    wherever it begins."""
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.standard_normal(FS + shift) * 0.02,
                        MK.synth_burst(GATEWAY, RESP),
                        rng.standard_normal(2 * FS) * 0.02])
    assert _awaiting_step(x) == _I_LINKSETUP_SENT, f"missed at offset {shift}"


def _awaiting_step(x, called=GATEWAY):
    hs = _awaiting(called=called)
    _feed(hs, x)
    return hs.step


def test_an_answer_to_another_station_is_rejected():
    rng = np.random.default_rng(13)
    for wrong in ("K7ABC", "NS0A", "KC9GHZ", MYCALL):
        x = np.concatenate([rng.standard_normal(FS) * 0.02,
                            MK.synth_burst(wrong, RESP),
                            rng.standard_normal(FS) * 0.02])
        assert _awaiting_step(x) == _I_CR_SENT, f"a response to {wrong} answered {GATEWAY}"


@pytest.mark.parametrize("block", [1024, 4800, 24000, 96000])
def test_the_answer_is_found_whatever_the_transport_hands_over(block):
    """Block size is the caller's business; the search accumulates to its own."""
    rng = np.random.default_rng(17)
    x = np.concatenate([rng.standard_normal(2 * FS) * 0.02,
                        MK.synth_burst(GATEWAY, RESP),
                        rng.standard_normal(2 * FS) * 0.02])
    hs = _awaiting()
    _feed(hs, x, block=block)
    assert hs.step == _I_LINKSETUP_SENT


def test_the_state_it_carries_stays_bounded():
    """It runs for the whole of a connect attempt, so it keeps one burst of tone
    track and less than one pass of audio, never a growing recording."""
    rng = np.random.default_rng(19)
    hs = _awaiting()
    for _ in range(60):
        hs.on_rx_stream(rng.standard_normal(FS) * 0.02)
        assert len(hs._st_track) <= _RESP_SPAN
        assert len(hs._st_buf) < MK.NFFT + _STREAM_BLOCK + FS


def test_a_long_run_of_synthetic_junk_never_answers():
    rng = np.random.default_rng(23)
    t = np.arange(2 * FS) / FS
    hs = _awaiting()
    for i in range(60):
        hs.on_rx_stream(rng.standard_normal(FS) * rng.uniform(0.01, 1.0))
        hs.on_rx_stream(0.5 * np.sin(2 * np.pi * rng.uniform(700, 2300) * t))
        assert hs.step == _I_CR_SENT, f"junk block {i} was taken for an answer"


# --------------------------------------------------------------------------- #
# Cost. It runs continuously while an attempt waits, not once per burst.
def test_it_costs_about_a_hundredth_of_real_time():
    """Measured at 1.15% of real time on the 150 s off-air recording, in one
    batched transform per pass. The bar here is loose enough to survive a loaded
    machine and tight enough to catch a per-hypothesis transform coming back."""
    rng = np.random.default_rng(29)
    x = rng.standard_normal(30 * FS) * 0.05
    hs = _awaiting()
    t0 = time.perf_counter()
    _feed(hs, x)
    dt = time.perf_counter() - t0
    assert dt < 0.10 * 30, f"searching 30 s of stream cost {dt:.2f} s"


# --------------------------------------------------------------------------- #
# The false-accept floor, over every real recording the project holds.
def _corpus() -> list[tuple[str, object, tuple[str, ...]]]:
    """``(label, path, callsigns legitimately on it)`` for every real recording."""
    out = [("2026-07-26 call to KB9MMT", corpora.ONAIR_CONNECT_ATTEMPT, ("KB9MMT",)),
           ("NS0A session", corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav", ("NS0A",)),
           ("KC9GHZ session", corpora.OFFAIR / "KC9GHZ_2300" / "rig_rx.wav", ("KC9GHZ",)),
           ("clear channel", corpora.CLEAR_CHANNEL, ())]
    if corpora.REGRESS_FIXTURES.is_dir():
        for p in sorted(corpora.REGRESS_FIXTURES.glob("*.wav")):
            own = tuple(c for c in _PANEL if c.lower() in p.name.lower())
            out.append((p.stem, p, own))
    return [(label, p, own) for label, p, own in out if p.exists()]


def _load_48k(path) -> np.ndarray:
    """One recording as band-limited 48 kHz mono. The websdr captures are 11999 Hz;
    the 83 ppm that is not 12000 is far below anything the tone lattice resolves."""
    from scipy.io import wavfile

    sr, a = wavfile.read(str(path))
    a = np.asarray(a, float)
    if a.ndim > 1:
        a = a[:, 0]
    if sr < 40000:
        a = resample_poly(a, 4, 1)
    return band_limit(_norm(a))


def _score_histogram(bins: np.ndarray, clear: np.ndarray,
                     call: str) -> tuple[np.ndarray, int]:
    """How many alignments confirm each number of payload tones, and how many sweep
    a comparable set clean, over the same lattice, the same tone shifts and the same
    arithmetic :meth:`on_rx_stream` uses.

    The histogram is taken with every symbol comparable, which is the ceiling
    ``_MEASURED_WORST`` is placed against and does not move when a symbol is
    dropped. The sweep count is the half that :data:`_CLEAR_DB` could move: dropping
    symbols shrinks the set a clean sweep has to cover, so it is counted rather
    than argued about.
    """
    tones = np.asarray(VF.handshake_tones(call, RESP), dtype=np.int32)[NPRE:]
    off = _RESP_OFF[NPRE:]
    n = len(bins) - _RESP_SPAN
    hist = np.zeros(RESP.n_payload + 1, dtype=np.int64)
    sweeps = 0
    for i in range(0, max(n, 0), 50000):
        idx = np.arange(i, min(i + 50000, n))[:, None] + off
        heard, ok = bins[idx], clear[idx] >= _CLEAR_DB
        c = ok.sum(1)
        for shift in _RESP_SHIFTS:
            hit = heard == tones + shift
            hist += np.bincount(hit.sum(1), minlength=len(hist))
            sweeps += int(_answers((hit & ok).sum(1), c).sum())
    return hist, sweeps


@corpora.requires_onair_connect_attempt
@corpora.requires_regress_fixtures
def test_no_real_off_air_audio_comes_near_the_acceptance_bar():
    """1588 s of real off-air HF — our own call to KB9MMT, the BW2300 gateway
    sessions, a third station's narrowband traffic, PACTOR-1/2/3, ARDOP, FT8, WSPR
    and band noise from four continents — band-limited to the SSB passband and
    scored against every panel callsign not on the recording.

    Every alignment is scored at every shift in ``_RESP_SHIFTS``, because that is
    what the search does since a gateway answered 23 Hz low on 2026-08-15. The
    shift axis multiplies the population and does not move the ceiling: 5 of 15 is
    the best a wrong callsign reaches at each shift taken alone, and 5 of 15 is the
    best over all of them together.

    Measured: 16,122,114 alignments at zero offset and 88,962,090 alignment-shifts
    over the five, best non-response 5 of 15 either way, and a geometric tail
    (survival 1.9e-1, 1.7e-2, 9.6e-4, 3.6e-5, 1.3e-6 at 1..5, taken over this corpus
    and the two 2026-08-14 controls together: 11-27x per extra confirmed tone).
    Extrapolating the eight tones from there to the cut at 13 puts a false accept
    far below 1e-15 per alignment against the ~450,000 alignment-shifts a 60 s
    connect attempt searches. The genuine answers in the same audio — KB9MMT at
    56.01 s, NS0A at 9.78 s — reach 15 of 15.

    The bar is placed against that measurement. If this fails, the population has
    changed and the cut has to be re-measured, not nudged.
    """
    hist = np.zeros(RESP.n_payload + 1, dtype=np.int64)
    worst, swept = {}, 0
    for label, path, own in _corpus():
        bins, clear, _resid = _top3_track(_load_48k(path), _ACK_GRID)
        for call in (c for c in _PANEL if c not in own):
            h, sw = _score_histogram(bins[:, 0], clear[:, 0], call)
            hist += h
            swept += sw
            worst[label] = max(worst.get(label, 0), int(np.flatnonzero(h)[-1]))
        del bins, clear
        gc.collect()
    top = int(np.flatnonzero(hist)[-1])
    assert top <= _MEASURED_WORST, (
        f"an alignment confirms {top}/{RESP.n_payload} response tones over "
        f"{hist.sum():,} tried, above the {_MEASURED_WORST} the corpus was measured "
        f"at; worst per recording: {worst}")
    assert top < _RESP_MIN_TONES
    assert hist.sum() > 50_000_000, f"only {hist.sum():,} alignments — corpus shrank"
    assert swept == 0, (
        f"{swept} alignments answer a callsign that is not on the recording once "
        f"tones under {_CLEAR_DB} dB of clearance stop being scored — the rule that "
        "recovers a real answer has moved the false-accept floor with it")


@corpora.requires_onair_connect_attempt
def test_the_real_recordings_accept_only_the_answers_addressed_to_us():
    """The floor above is arithmetic on a lattice; this is the recogniser itself,
    driven over whole recordings exactly as a live attempt would drive it. Each
    recording is dialled to the station on it and to two that are not — the panel
    is swept exhaustively above, where it costs one tone track rather than eight.
    """
    seen = {}
    for label, path, own in _corpus()[:4]:            # the declared off-air captures
        x = _load_48k(path)
        for call in own + ("K7ABC", MYCALL):
            at = _feed(_awaiting(called=call), x)
            if at:
                seen[(label, call)] = at
        del x
        gc.collect()
    # KB9MMT and NS0A each answered twice; the KC9GHZ session was recorded from
    # inside an established link and holds no connect at all.
    assert set(seen) == {("2026-07-26 call to KB9MMT", "KB9MMT"), ("NS0A session", "NS0A")}, \
        seen
    assert all(len(v) == 2 for v in seen.values()), (
        f"both gateways answered twice; found {seen}")


# --------------------------------------------------------------------------- #
# The answer that arrives while the receiver is still muted.
#
# Measured on 2026-08-06, calling KC9GHZ on 7103.5 kHz with a KiwiSDR witnessing
# the same minutes from another site. The gateway answered three of our eight
# connect-requests: 10/15, 12/15 and 14/15 payload tones at the witness, and its
# payload opens ~0.18 s after our own last transmitted sample. With the response's
# eight-symbol preamble that puts its keying inside the tail of our own request,
# which is why no preamble ever locks on one. This station's input is dead from
# TX_end+0.02 to TX_end+0.44, so six payload tones land in the mute; nine reach the
# demodulator, and all nine are right, in all three answers.
#
# Nine is below _RESP_MIN_TONES, so every one of those answers was discarded and
# the attempt reported no answer. That is the whole of why kestrel has never
# connected to a gateway as the caller while connecting 6/6 as the responder: as
# responder nothing of ours is keyed across the burst it has to read.
_MUTE_START, _MUTE_END = 0.02, 0.44        # relative to our last transmitted sample
_ANSWER_LEAD = 0.16                        # the gateway starts this far before it


def _answer_under_the_mute(called: str = GATEWAY, answer_for: str | None = None,
                           seed: int = 31) -> np.ndarray:
    """Our connect-request, our receiver's mute behind it, and the answer that
    arrives across both — the geometry measured off the KC9GHZ recording."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(8 * FS) * 0.02
    cr = MK.synth_burst(called, VF.CR)
    at = 2 * FS
    x[at:at + len(cr)] += cr                       # our own transmission, heard back
    tx_end = (at + len(cr)) / FS
    ans = MK.synth_burst(answer_for or called, RESP) * 0.05
    a0 = int((tx_end - _ANSWER_LEAD) * FS)
    x[a0:a0 + len(ans)] += ans
    x[int((tx_end + _MUTE_START) * FS):int((tx_end + _MUTE_END) * FS)] = 0.0
    return x


def test_an_answer_whose_head_is_under_our_own_mute_is_still_found():
    hs = _awaiting()
    assert _feed(hs, _answer_under_the_mute()), (
        "the gateway answered and the mute swallowed the front of its payload; "
        "what reached the receiver was nine correct tones out of fifteen")
    assert hs.step == _I_LINKSETUP_SENT


def test_the_mute_does_not_let_another_stations_answer_through():
    """Fewer comparable tones must not mean a lower bar for whose they are."""
    for wrong in ("K7ABC", "NS0A", "KC9GHZ", MYCALL):
        hs = _awaiting()
        handed = _handed(hs)
        assert _feed(hs, _answer_under_the_mute(answer_for=wrong)) == [], (
            f"a connect-response to {wrong} was accepted through the mute")
        assert handed == [], (
            f"the search accepted {wrong}'s answer and only VF.recognize refused it")


@pytest.mark.parametrize("mute_s", [0.42, 4.4])
def test_a_mute_with_no_answer_in_it_answers_nobody(mute_s):
    """The mask says "not comparable", not "assume it matched".

    Both lengths this station's receiver actually goes deaf for: 0.42 s after every
    transmission, and the whole 4.4 s of its own link-setup. The long one is the
    counterexample that bites. A response's fifteen payload tones span 0.64 s, so a
    0.42 s hole can never cover enough of them for "assume matched" to sweep a whole
    payload — read the mask that way and the short mutes stay silent while band noise
    under a link-setup is accepted as the gateway's answer.
    """
    rng = np.random.default_rng(37)
    x = rng.standard_normal(30 * FS) * 0.02
    for a in np.arange(1.0, 29.0 - mute_s, mute_s + 0.5):   # mutes all over it
        x[int(a * FS):int((a + mute_s) * FS)] = 0.0
    hs = _awaiting()
    handed = _handed(hs)
    assert _feed(hs, x) == []
    assert handed == [], (
        f"{len(handed)} stretch(es) of band noise under a {mute_s} s mute were "
        f"accepted as KB9MMT's connect-response")


@pytest.mark.parametrize("live,accept", [
    # Fifteen and nine are what the air delivers: a response heard whole, and one
    # whose first six payload tones landed in this station's post-transmit mute on
    # 2026-08-06. Nine has to be accepted or the mute mask above buys nothing.
    (15, True), (9, True),
    # Three and one are not identification. A payload symbol draws from 35 tones, so
    # a wrong callsign sweeps three of them clean once in 42,875 alignments — twice
    # per connect attempt, which searches around 90,000 — and one of them once in 35.
    (3, False), (1, False),
])
def test_a_burst_the_mute_left_almost_nothing_of_is_not_recognised(live, accept):
    """``VF.recognize`` is the gate behind every accept in this file, and it has its
    own floor on how much of a burst has to have arrived: ``MIN_COMPARABLE``. It had
    no counterexample, so the whole of it could be set to three with this suite
    green — and then the stream search's own floor stops mattering, because whatever
    it lets through is judged on three tones."""
    tones = list(VF.handshake_tones(GATEWAY, RESP))
    heard = tones[:NPRE] + [t if i < live else -1
                            for i, t in enumerate(tones[NPRE:])]
    assert VF.recognize(heard, GATEWAY, RESP) is accept, (
        f"{live} correct payload tones of {RESP.n_payload}, the rest never delivered: "
        f"recognize says {not accept}")


@corpora.requires_onair_gateway_answer
@corpora.requires_onair_silent_calls
def test_no_wrong_callsign_sweeps_a_comparable_tone_set_clean():
    """The floor under the mute accept, over the audio it was measured on.

    The clean-sweep rule is the looser of the two acceptances — it confirms as few
    as ``_RESP_MIN_HEARD`` tones where the other confirms thirteen — so it is the one
    with a floor to prove. This is the whole of this station's own off-air audio of
    2026-08-06: the three calls, the receiver mutes in them, scored on the same
    lattice with the same live mask against every published gateway callsign that is
    not on the recording.

    What it bounds is the confirmed count, not the threshold. Nothing here sweeps any
    comparable set clean at all, so the measurement cannot locate a knee — a
    threshold of 4 would come out of it just as clean. What it does say is where the
    population sits: the best a wrong callsign confirms anywhere is 6 of 15, and
    ``_RESP_MIN_HEARD`` has to stand clear of that, exactly as ``_RESP_MIN_TONES``
    stands clear of the 5-of-15 measured above.

    Every alignment is scored at every shift in ``_RESP_SHIFTS``, so this is the
    population the mute accept actually faces — a wrong callsign gets five ways past
    it now rather than one, and a floor measured at only one of them is not a floor.
    """
    gateways = corpora.VARA_GATEWAY_PANEL
    if not gateways.exists():
        pytest.skip("the published gateway list is not in this tree")
    panel = sorted({r["Callsign"].strip().upper()
                    for r in csv.DictReader(gateways.open(newline=""))})
    assert len(panel) > 300, f"only {len(panel)} callsigns — the list has shrunk"

    off = _RESP_OFF[NPRE:]
    best, swept, total = 0, 0, 0
    for path, own in ((corpora.ONAIR_GATEWAY_ANSWER, "KC9GHZ"),
                      (corpora.ONAIR_SILENT_CALLS[0], "KD9USW"),
                      (corpora.ONAIR_SILENT_CALLS[1], "W8MW")):
        x = corpora.wav_mono(path) / 32768.0
        track = _tone_track(x, _ACK_GRID)
        live = _live_track(x, _ACK_GRID)
        n = len(track) - _RESP_SPAN
        for call in panel:
            if call.split("-")[0] in (own, MYCALL):
                continue
            tones = np.asarray(VF.handshake_tones(call, RESP), dtype=np.int32)[NPRE:]
            for i in range(0, n, 40000):
                idx = np.arange(i, min(i + 40000, n))[:, None] + off
                comparable = live[idx]
                heard, c = track[idx], comparable.sum(1)
                for shift in _RESP_SHIFTS:
                    m = ((heard == tones + shift) & comparable).sum(1)
                    total += len(idx)
                    best = max(best, int(m.max()))
                    swept = max(swept, int(c[m == c].max(initial=0)))
        del track, live, x
        gc.collect()

    assert total > 100_000_000, f"only {total:,} alignments — the audio has changed"
    assert swept == 0, (
        f"a wrong callsign swept {swept} comparable tones clean; the accept this "
        f"measures takes {_RESP_MIN_HEARD}")
    assert best < _RESP_MIN_HEARD, (
        f"a wrong callsign confirms {best} of {RESP.n_payload} comparable tones over "
        f"{total:,} alignments, at or above the {_RESP_MIN_HEARD} an all-comparable "
        f"sweep is accepted on — the two populations no longer have room between them")


@corpora.requires_onair_gateway_answer
def test_the_real_gateway_answer_of_20260806_is_accepted():
    """The recording itself: three answers from KC9GHZ, nine live tones each.

    Two accepts rather than three — the third answer arrives while the stream
    search is inside the burst it has already accepted — and two is two more than
    the attempt got on the night."""
    x = corpora.wav_mono(corpora.ONAIR_GATEWAY_ANSWER) / 32768.0
    hs = _awaiting(called="KC9GHZ")
    at = _feed(hs, x)
    assert at, "KC9GHZ's answers are in this recording and none was found"
    assert hs.step == _I_LINKSETUP_SENT
    delivered = [m for m in hs.io.log_lines if "tones the receiver delivered" in m]
    assert delivered, hs.io.log_lines


@corpora.requires_onair_silent_calls
def test_the_same_night_recordings_with_no_answer_stay_silent():
    """The two calls the same evening that nobody answered — 141 s of 40 m and
    80 m band noise through the same receiver, one of them with a narrowband
    signal sitting on the channel centre for half the run."""
    for p, call in zip(corpora.ONAIR_SILENT_CALLS, ("KD9USW", "W8MW")):
        x = corpora.wav_mono(p) / 32768.0
        hs = _awaiting(called=call)
        handed = _handed(hs)
        assert _feed(hs, x) == [], f"{call} answered nothing"
        assert handed == [], (
            f"{len(handed)} stretch(es) of {p.name} were accepted as {call}'s "
            f"connect-response and only VF.recognize refused them")


# --------------------------------------------------------------------------- #
# The answer that arrives off frequency.
#
# 2026-08-15, calling N0LCR-1 on 7103.5 kHz: seven connect-requests went out and the
# tool printed "no answer — resending CR" after six of them. One answer is in the
# recording, at 18.35 s, and it sits one carrier low — -23 Hz, which the search at
# zero offset scores at chance rather than badly.
#
# The transport's own geometry is in the recording, and the replay below uses it
# rather than a continuous feed: this is a defect of the live path, so what has to
# take the answer is the live path. `AudioVaraIO.tx` skips the receive cursor to
# TX-end + 0.1 s, the connect loop feeds `on_rx_stream` from there and calls
# `originate` again on the resend cadence, so the search was given the audio between
# one cursor and the next key-up and nothing else. The figures are the receiver's own
# mute edges, measured off this file at 10 ms: it goes deaf for 0.07-0.10 s at
# key-up and for 0.17 s after the last transmitted sample.
_N0LCR = "N0LCR-1"
_N0LCR_KEYUP = (8.300, 16.400, 24.520, 32.620, 40.740, 48.840, 56.920)
_N0LCR_TXEND = (10.140, 18.260, 26.380, 34.480, 42.600, 50.700, 58.780)
_TX_CURSOR_SKIP = 0.1                      # AudioVaraIO.tx


def _replay(hs: VaraStationHandshake, io: _IO, x: np.ndarray, keyup, txend,
            call: str, block: int = 2400) -> list[float]:
    """Push exactly the samples the live transport handed the search that night.

    ``originate`` between the windows because the connect loop resends the CR there,
    and what that clears — ``_peer_shift`` among it — is cleared on the air too.
    """
    at = []
    for k, (end, nxt) in enumerate(zip(txend, tuple(keyup[1:]) + (len(x) / FS,))):
        if k:
            hs.originate(call)
        a, b = int((end + _TX_CURSOR_SKIP) * FS), int(nxt * FS)
        for i in range(a, b, block):
            before = len(io.log_lines)
            hs.on_rx_stream(x[i:min(i + block, b)])
            if any("found on the receive stream" in m
                   for m in io.log_lines[before:]):
                at.append(min(i + block, b) / FS)
    return at


@corpora.requires_onair_offset_answer
def test_the_off_frequency_answer_of_20260815_is_taken_by_the_live_path():
    """The answer N0LCR-1 sent, taken by the path that walked past it on the night.

    Replayed through the live transport's own geometry — the post-TX cursor skip and
    the CR resends — the search takes it on the second request, 14 of 15 payload
    tones at -1 carrier, and the attempt reaches the link-setup instead of running
    its resend ladder out.
    """
    x = corpora.wav_mono(corpora.ONAIR_OFFSET_ANSWER) / 32768.0
    io = _IO()
    hs = _awaiting(io, called=_N0LCR)
    at = _replay(hs, io, x, _N0LCR_KEYUP, _N0LCR_TXEND, _N0LCR)
    assert at, ("the one answer this recording holds is at 18.35 s and the live "
                "path found nothing at all")
    assert at[0] < _N0LCR_KEYUP[2], (
        f"the answer to request #2 was taken at {at[0]:.2f} s, after the third "
        f"request went out at {_N0LCR_KEYUP[2]:.2f} s")
    assert hs.step == _I_LINKSETUP_SENT, io.log_lines
    assert any("-23 Hz off frequency" in m for m in io.log_lines), io.log_lines


@corpora.requires_onair_offset_answer
def test_that_recording_holds_no_answer_at_all_at_zero_frequency_offset():
    """Why it was walked past, as arithmetic rather than as a story.

    A tuning error moves every tone of a burst by the same amount, so a matcher
    comparing bin indices at one offset does not score an off-frequency answer
    badly — it scores it at chance. Nothing in these sixty seconds reaches even half
    the acceptance at zero offset; one carrier low, the answer is 14 of 15.
    """
    x = corpora.wav_mono(corpora.ONAIR_OFFSET_ANSWER) / 32768.0
    track = _tone_track(x, _ACK_GRID)
    tones = np.asarray(VF.handshake_tones(_N0LCR, RESP), dtype=np.int32)[NPRE:]
    off, n = _RESP_OFF[NPRE:], len(track) - _RESP_SPAN
    best = dict.fromkeys((-1, 0), 0)
    for i in range(0, n, 50000):
        heard = track[np.arange(i, min(i + 50000, n))[:, None] + off]
        for shift in best:
            best[shift] = max(best[shift], int((heard == tones + shift).sum(1).max()))
    assert best[0] <= _MEASURED_WORST, (
        f"at zero offset the recording reaches {best[0]}/{RESP.n_payload}, above the "
        f"corpus floor — the audio is not the one the miss was diagnosed on")
    assert best[0] < _RESP_MIN_TONES
    assert best[-1] >= 14, (
        f"one carrier low the answer confirmed 14 of {RESP.n_payload}; it now reaches "
        f"{best[-1]}, so the fixture or the tone track has changed")


# The three answers KC9GHZ sent on 2026-08-19, by burst start on the lattice. Each
# follows one of our connect-requests — the sixth, seventh and eighth — by 0.10 to
# 0.18 s, and each carries its first payload symbols through the receiver's mute
# release rather than after it.
_KC9GHZ_ANSWERS = (45.580, 53.205, 59.880)
#: How far under the burst's own settled level the mute release leaves the first
#: payload symbol. `_DEAF_DBFS` is an absolute floor and the ramp stays well over
#: it, so those symbols are not caught by level at all.
_RELEASE_SHORTFALL_DB = 9.0
# The transport's geometry that night, read off the recording at 10 ms, the same
# way `_N0LCR_KEYUP`/`_N0LCR_TXEND` were: eight key-ups, eight last samples.
_KC9GHZ_KEYUP = (8.29, 15.24, 21.89, 29.41, 36.58, 43.97, 51.56, 58.24)
_KC9GHZ_TXEND = (10.13, 17.09, 23.75, 31.26, 38.43, 45.78, 53.41, 60.09)


@corpora.requires_onair_faded_greeting
def test_the_gateway_answered_three_of_the_eight_requests_of_20260819():
    """Eight connect-requests went out that night and the log said "no answer"
    after seven of them. Three of the eight were answered.

    Judged on level alone, each answer confirms 12 of 15 payload tones with all 15
    comparable, and on that reading nothing here is acceptable: 12 is one tone
    under ``_RESP_MIN_TONES``, and a clean sweep of a comparable set cannot be
    reached with three tones wrong. The session happened because the eighth answer
    read one tone better on the air than it does in this recording — the whole of
    the margin the link stood on.

    The three tones each answer loses are the receiver's, not the gateway's, and
    the test below says which and why. Once a tone that never stood ``_CLEAR_DB``
    clear of the band is scored as no tone rather than as a wrong one, all three
    sweep their comparable sets clean and the live path takes all three.

    Nothing else in the 127 s exceeds 7 of 15 against the callsign we dialled, so
    the three stand five tones clear of the rest of the recording.
    """
    x = corpora.wav_mono(corpora.ONAIR_FADED_GREETING) / 32768.0
    track, live = _tone_track(x, _ACK_GRID), _live_track(x, _ACK_GRID)
    tones = np.asarray(VF.handshake_tones("KC9GHZ", RESP), dtype=np.int32)[NPRE:]
    off, n = _RESP_OFF[NPRE:], len(track) - _RESP_SPAN
    idx = np.arange(n)[:, None] + off
    heard, ok = track[idx], live[idx]
    comparable = ok.sum(1)
    confirmed = np.zeros(n, int)
    for shift in _RESP_SHIFTS:
        confirmed = np.maximum(confirmed, ((heard == tones + shift) & ok).sum(1))

    for at in _KC9GHZ_ANSWERS:
        i = int(round(at * FS / _ACK_GRID))
        assert (confirmed[i], comparable[i]) == (_RESP_MIN_TONES - 1, RESP.n_payload), (
            f"the answer at {at} s confirms {confirmed[i]}/{comparable[i]}, not "
            f"{_RESP_MIN_TONES - 1}/{RESP.n_payload}")

    t = np.arange(n) * _ACK_GRID / FS
    elsewhere = np.ones(n, bool)
    for at in _KC9GHZ_ANSWERS:
        elsewhere &= np.abs(t - at) > 1.0
    assert confirmed[elsewhere].max() <= 7, (
        f"the rest of the recording reaches {confirmed[elsewhere].max()}/"
        f"{RESP.n_payload} — the three answers are no longer separated from it")

    assert not _answers(confirmed, comparable).any(), (
        "an alignment clears the bar on level alone; these three did not, and that "
        "is the finding the clearance rule was measured against")

    io = _IO()
    hs = _awaiting(io, called="KC9GHZ")
    at = _replay(hs, io, x, _KC9GHZ_KEYUP, _KC9GHZ_TXEND, "KC9GHZ")
    assert len(at) == len(_KC9GHZ_ANSWERS), (
        f"the live path took {len(at)} of the three answers this recording holds: "
        f"{[round(v, 2) for v in at]}")
    for taken, sent in zip(at, _KC9GHZ_ANSWERS):
        assert 0 < taken - sent < 1.5, (
            f"an accept at {taken:.2f} s is not the answer that began at {sent} s")
    for wrong in ("K7ABC", MYCALL, "NS0A", GATEWAY):
        io = _IO()
        hs = _awaiting(io, called=wrong)
        assert _replay(hs, io, x, _KC9GHZ_KEYUP, _KC9GHZ_TXEND, wrong) == [], (
            f"KC9GHZ's answers were accepted as an answer to {wrong}")


@corpora.requires_onair_faded_greeting
def test_what_those_answers_got_wrong_is_the_receivers_own_mute_release():
    """Which tones missed, and why it is not the gateway getting them wrong.

    Preamble symbols 5-7 of every one of the three land in the mute proper, at -63
    to -84 dBFS, and are scored as not comparable. What follows is a ramp: the first
    payload symbol arrives 9 to 18 dB under where the same burst settles four
    symbols later, reads well above ``_DEAF_DBFS``, and is therefore scored as a
    tone the gateway sent and got wrong. Two or three of them go that way in each
    answer; from payload symbol 3 on, no more than one tone of any of the three is
    wrong.

    So the deafness mask is drawn at an absolute level while the loss is relative to
    the burst — which is why widening the mask does not repair this either. Masking
    the ramp would accept the answer to request seven and reject the one the session
    was actually built on.

    What separates the lost tones from the kept ones is not level at all: it is
    whether the bin the argmax returned stood clear of the rest of the band. Eight
    of the nine tones the three answers get wrong read under ``_CLEAR_DB`` — 0.1 to
    1.2 dB — against 2.3 dB and up for all but four of the thirty-six they get
    right, and the ninth reaches 2.6. So the rule does not sweep the burst clean at
    the nominal alignment, and it does not have to: what it does is leave a
    comparable set the search can sweep clean somewhere on the burst's own plateau,
    which is what the drive above asserts.
    """
    x = corpora.wav_mono(corpora.ONAIR_FADED_GREETING) / 32768.0
    bins, clear, _resid = _top3_track(x, _ACK_GRID)
    tones = np.asarray(VF.handshake_tones("KC9GHZ", RESP), dtype=np.int32)
    for at in _KC9GHZ_ANSWERS:
        i = int(round(at * FS / _ACK_GRID)) + _RESP_OFF[NPRE:]
        hit = bins[i, 0] == tones[NPRE:]
        stood = int((clear[i, 0][~hit] >= _CLEAR_DB).sum())
        assert stood <= 1, (
            f"{at} s: {stood} tones the gateway is scored wrong on stand clear of "
            f"the band — they are wrong tones, not unread ones, and the repair "
            "does not cover them")
        kept = int((clear[i, 0][hit] >= _CLEAR_DB).sum())
        assert kept >= _RESP_MIN_HEARD, (
            f"{at} s: {kept} of the tones the gateway got right stand clear, under "
            f"the {_RESP_MIN_HEARD} a clean sweep has to reach")

    for at in _KC9GHZ_ANSWERS:
        s = int(round(at * FS))
        rms = np.array([
            float(np.sqrt(np.mean(x[i:i + MK.NFFT] ** 2)))
            for i in (s + np.arange(len(RESP.preamble) + RESP.n_payload) * MK.HOP
                      + MK._WOFF)])
        db = 20 * np.log10(rms + 1e-15)
        assert (db[5:NPRE] < _DEAF_DBFS).all(), (
            f"{at} s: the mute no longer covers preamble symbols 5-7 ({db[5:NPRE]})")
        settled = np.median(db[NPRE + 3:])
        assert settled - db[NPRE] >= _RELEASE_SHORTFALL_DB, (
            f"{at} s: the first payload symbol reads {db[NPRE]:.1f} dBFS against a "
            f"settled {settled:.1f} — the release ramp is gone from this recording")
        assert db[NPRE] > _DEAF_DBFS, (
            f"{at} s: the ramp is now under the deafness mask, so the tones lost in "
            f"it are no longer scored against the gateway")


@corpora.requires_onair_unanswered_call
def test_the_same_gateway_ninety_minutes_later_answered_nobody():
    """The control for the three answers above, and the one that matters most.

    Same peer, same channel, same receiver, same eight-request cadence, 92 minutes
    later, and `NOT connected` at the end of it. If the clearance rule were reading
    the changeover rather than the gateway it would find answers here too, because
    the changeover is the same: this recording's mute measures 171-189 ms against
    that one's 167-180.

    It finds none. The best any alignment in the 70 s reaches for KC9GHZ is 5 of
    15 — ``_MEASURED_WORST``, what a callsign that is not on a recording reaches —
    so there is no answer in this audio to recover, and neither rule invents one.
    """
    x = corpora.wav_mono(corpora.ONAIR_UNANSWERED_CALL) / 32768.0
    bins, clear, _resid = _top3_track(x, _ACK_GRID)
    for call in ("KC9GHZ", "K7ABC", MYCALL, "NS0A"):
        h, sw = _score_histogram(bins[:, 0], clear[:, 0], call)
        top = int(np.flatnonzero(h)[-1])
        assert top <= _MEASURED_WORST, (
            f"an alignment confirms {top}/{RESP.n_payload} tones for {call} on a "
            "session nobody answered")
        assert sw == 0, f"{sw} alignments answer {call} on a clean sweep"
        hs = _awaiting(called=call)
        handed = _handed(hs)
        assert _feed(hs, x) == [], f"the unanswered call answered {call}"
        assert handed == [], (
            f"{len(handed)} stretch(es) were handed to the recogniser as {call}'s "
            "connect-response and only VF.recognize refused them")


@corpora.requires_onair_faded_greeting
def test_the_mute_this_station_answers_into_is_170_ms_and_not_420():
    """How long the receiver is actually deaf after a transmission, read off the
    recording rather than off a comment.

    ``vara_arq`` documented 0.42 s in the present tense long after it stopped being
    true, and ``_RESP_MIN_HEARD``'s ceiling of nine payload tones was arithmetic on
    that number. 250 ms of the 420 was a deliberate hold past the last transmitted
    sample and came out on 2026-08-06; measured the same way — the interval from the
    last transmitted sample to the first 10 ms frame back over ``_DEAF_DBFS`` — this
    station reads 430 ms on 2026-08-02, 420 and then 224 on 2026-08-06 either side
    of that change, 120-130 on 2026-08-09/10, and 167-180 from 2026-08-15 on.

    Pinned on the 2026-08-19 recording because that is the one the answers are in,
    and because the number decides whether an answer 0.18 s behind us arrives
    whole: at 167 ms all fifteen payload tones reach the demodulator, which is why
    the three answers there score 15 comparable and not nine.
    """
    x = corpora.wav_mono(corpora.ONAIR_FADED_GREETING) / 32768.0
    f = FS // 100
    db = 20 * np.log10(np.sqrt((x[:len(x) // f * f].reshape(-1, f) ** 2).mean(1))
                       + 1e-15)
    mutes = []
    for end in _KC9GHZ_TXEND:
        i = int(round(end * 100))
        live = np.flatnonzero(db[i:i + 100] > _DEAF_DBFS)
        assert len(live), f"the receiver never comes back after {end} s"
        mutes.append(int(live[0]) * 10)
    assert max(mutes) <= 200, (
        f"the mute after a transmission reads {mutes} ms; over 200 and an answer "
        "0.18 s behind us starts losing payload symbols again")
    assert min(mutes) >= 100, (
        f"the mute reads {mutes} ms — shorter than anything measured on this "
        "station, so the fixture is not the recording this was drawn on")
    assert (NPRE * MK.HOP / FS) * 1000 > max(mutes), (
        f"a {max(mutes)} ms mute now reaches past the preamble into the payload of "
        "an answer that starts under our own transmission")


@corpora.requires_carrier_only_channel
@corpora.requires_monitored_quiet
def test_the_2026_08_14_controls_answer_nobody_at_any_offset():
    """The negative controls, against a search that now looks five ways.

    Two receive-only windows from the 2026-08-14 slot, both of which every
    classifier in it read as empty: 211.6 s of 6800 kHz, which has no amateur
    allocation and no Winlink channel, and 542.6 s of 7103.5 kHz across ten minutes
    the RMS feed and the deep pass agree carried no session. Widening a matcher
    until a missed answer is accepted is only worth having if what else becomes
    acceptable is measured, and this is that measurement: the recogniser is silent,
    the search behind it hands it nothing, and the arithmetic underneath stays at
    the corpus floor at every shift.

    Measured over the nine callsigns below at the five shifts: 50,774,490
    alignment-shifts, best confirmed 5 of 15 against the 13 of ``_RESP_MIN_TONES``.
    The clean-sweep acceptance is covered by the drive rather than by arithmetic —
    an alignment that swept a comparable set of ``_RESP_MIN_HEARD`` or wider would
    reach the recogniser, and nothing does.
    """
    panel = (_N0LCR, *_PANEL)
    worst = {}
    for label, path in (("6800 kHz, no allocation", corpora.CARRIER_ONLY_CHANNEL),
                        ("7103.5 kHz, channel empty", corpora.MONITORED_QUIET)):
        x = _load_48k(path)
        for call in panel:
            hs = _awaiting(called=call)
            handed = _handed(hs)
            assert _feed(hs, x) == [], f"{label} answered a call to {call}"
            assert handed == [], (
                f"{len(handed)} stretch(es) of {label} were accepted as {call}'s "
                f"connect-response and only VF.recognize refused them")
        bins, clear, _resid = _top3_track(x, _ACK_GRID)
        for call in panel:
            h, sw = _score_histogram(bins[:, 0], clear[:, 0], call)
            worst[label] = max(worst.get(label, 0), int(np.flatnonzero(h)[-1]))
            assert sw == 0, f"{label} answers {call} on a clean sweep"
        del x, bins, clear
        gc.collect()
    assert max(worst.values()) <= _MEASURED_WORST, (
        f"a wrong callsign confirms {max(worst.values())}/{RESP.n_payload} response "
        f"tones on a channel with nothing on it: {worst}")
