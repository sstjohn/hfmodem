# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The third-party witness: what it records, and how it is registered against us.

A KiwiSDR hears both stations on one clock with no windowing by us, so it is the
only instrument that can say where our own capture lost air. Every figure that
comes out of that comparison is a difference of two timebases, and the tools have
to hold both of them exactly or the difference is manufactured.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
sys.path.insert(0, str(_TOOLS))

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "witness_align.py").exists(),
    reason=f"{_TOOLS} is not in this tree -- the witness tools ship from the "
           "repository, not from the wheel")

witness_align = pytest.importorskip("witness_align")
kiwi_witness = pytest.importorskip("kiwi_witness")

TONES = (1400.0, 1600.0)
RATE = 100.0


def keyed(fs: float, edges, secs: float = 40.0, seed: int = 1) -> np.ndarray:
    """Noise with a tone up across each (start, stop) in SECONDS, not samples."""
    n = int(secs * fs)
    t = np.arange(n) / fs
    x = np.random.default_rng(seed).normal(0.0, 0.01, n)
    for i, (a, b) in enumerate(edges):
        s = slice(int(a * fs), int(b * fs))
        x[s] += 0.5 * np.sin(2 * np.pi * TONES[i % 2] * t[s])
    return x


def first_keyed_s(k: np.ndarray) -> float:
    return float(np.flatnonzero(k)[0]) / RATE


# -- the frame series carries the seconds it claims to -------------------------
#
# `keydown` returns one frame per 1/rate second and every lag this tool prints is
# a count of those frames divided by `rate`. The hop was `int(fs / rate)`, which
# is exact at 48000 Hz and is not at a KiwiSDR's 11999 Hz: 119 samples where
# 119.99 were meant, so the witness frame series ran 0.83% fast -- 8300 ppm -- and
# the tool read that as the session recording falling behind the air.

def test_a_frame_series_advances_at_the_rate_it_is_asked_for():
    fs = 11999.0
    k = witness_align.keydown(keyed(fs, [(30.0, 31.0)]), fs, TONES, RATE)
    assert first_keyed_s(k) == pytest.approx(30.0, abs=0.05)


def test_two_receivers_on_different_clocks_agree_on_when_it_was_keyed():
    """The one measurement the pair exists to make, at its two real rates."""
    edges = [(5.0, 6.0), (20.0, 21.0), (35.0, 36.0)]
    ours = witness_align.keydown(keyed(48000.0, edges), 48000.0, TONES, RATE)
    theirs = witness_align.keydown(keyed(11999.0, edges), 11999.0, TONES, RATE)
    assert first_keyed_s(ours) == pytest.approx(first_keyed_s(theirs), abs=0.05)
    ours_last = float(np.flatnonzero(ours)[-1]) / RATE
    theirs_last = float(np.flatnonzero(theirs)[-1]) / RATE
    assert ours_last == pytest.approx(theirs_last, abs=0.05)


# -- the rate the receiver measured, not the one a WAV header can hold ---------

def _witness_wav(tmp_path: Path, fs: float, header: int) -> Path:
    p = tmp_path / "witness.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(header)
        w.writeframes(np.zeros(header, dtype="<i2").tobytes())
    p.with_suffix(".json").write_text(json.dumps({"sample_rate": fs}))
    return p


def test_the_witness_rate_comes_from_the_sidecar_not_the_rounded_header(tmp_path):
    p = _witness_wav(tmp_path, 11998.874629, 11999)
    assert witness_align.witness_rate(p, 11999.0) == pytest.approx(11998.874629)


def test_a_witness_with_no_sidecar_still_reads(tmp_path):
    p = _witness_wav(tmp_path, 11998.874629, 11999)
    p.with_suffix(".json").unlink()
    assert witness_align.witness_rate(p, 11999.0) == 11999.0


def test_our_own_recording_is_read_at_its_nominal_rate(tmp_path):
    """shrike's sidecar says `samplerate`, and it is NOT a measured clock.

    `stream.wav` counts DELIVERED samples at a nominal 48000, and the whole
    point of this tool is to test that nominal against the air. Correcting it
    from the recorder's own numbers would answer the question with itself.
    """
    p = tmp_path / "stream.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(48000, dtype="<i2").tobytes())
    p.with_suffix(".json").write_text(json.dumps({"samplerate": 48000,
                                                  "capture_ppm": -4200.0}))
    _, fs = witness_align.read(p)
    assert fs == 48000.0


# -- the lag it can find without being told where to look ---------------------

def test_the_whole_file_search_places_a_window():
    """`--guess` is not something an operator has at 3 a.m.

    The auto-guess searched with a window as long as the whole of ours, and
    `range(0, len(a) - w, h)` is empty when the two are equal -- so every run
    without `--guess` printed NO_OVERLAP, which reads as two recordings of
    different air.
    """
    ours = np.zeros(600)
    ours[100:200] = 1.0
    theirs = np.zeros(1400)
    theirs[400:500] = 1.0                      # ours starts 3.00 s into theirs
    c = witness_align.lag_curve(ours, theirs, RATE, 0.0, len(ours) / RATE,
                                len(ours) / RATE, (len(theirs) - len(ours)) / RATE / 2)
    assert len(c) == 1
    assert c[0][1] == pytest.approx(3.0, abs=0.02)


# -- which receiver, and the two ways of getting it wrong ----------------------
#
# The station's own grid at 40 m. Nearest-first put the witness wherever the list
# happened to be
# densest; a receiver inside our ground wave hears us and not the gateway, and
# one past a hop hears neither well enough to time. The arm on record that
# produced 959.40 +/- 2.09 ms against a 960.00 ms render was heard at 103 mi.

def _csv(tmp_path: Path, rows) -> Path:
    p = tmp_path / "cands.csv"
    lines = ["hostname,port,lat_lon,free_channels,name_location"]
    for host, lat, free in rows:
        lines.append(
            f"{host},8073,\"{lat},{kiwi_witness.STATION[1]}\",{free},{host}")
    p.write_text("\n".join(lines) + "\n")
    return p


def _at(miles_north: float) -> float:
    return kiwi_witness.STATION[0] + miles_north / 69.0   # due north of the station


def _hosts(*a, **kw) -> list[str]:
    return [c.host for c in kiwi_witness.candidates(*a, **kw)]


def test_a_receiver_inside_our_ground_wave_is_not_the_first_choice(tmp_path):
    p = _csv(tmp_path, [("close", _at(20), 8), ("good", _at(120), 8)])
    assert _hosts(p, 6)[0] == "good"


def test_a_receiver_past_one_hop_is_not_the_first_choice(tmp_path):
    p = _csv(tmp_path, [("far", _at(1900), 8), ("good", _at(120), 8)])
    assert _hosts(p, 6)[0] == "good"


def test_inside_the_window_it_is_still_nearest_first(tmp_path):
    p = _csv(tmp_path, [("mid", _at(300), 8), ("near", _at(90), 8)])
    assert _hosts(p, 6) == ["near", "mid"]


def test_the_ones_outside_the_window_are_a_fallback_and_not_a_refusal(tmp_path):
    """A witness from the wrong distance still beats no witness at all."""
    p = _csv(tmp_path, [("far", _at(1900), 8), ("close", _at(20), 8)])
    assert len(kiwi_witness.candidates(p, 6)) == 2


def test_a_receiver_with_no_free_slot_is_never_offered(tmp_path):
    p = _csv(tmp_path, [("full", _at(120), 0), ("good", _at(130), 8)])
    assert _hosts(p, 6) == ["good"]


# -- the witness the peer needs, which is not the witness we need --------------
#
# An arm that keys over the peer mutes our own receiver across two of every three
# of the peer's answer slots, so a third party is the only thing that can tell
# "the peer said nothing" from "the peer answered while we transmitted". A
# distance from US cannot say that: on 2026-08-30 `--near 500 --far 900`, meant
# to reach a gateway in Arkansas, chose a receiver 516 mi away in Virginia and
# the arm was scored without the cover the spec requires.

PEER = (kiwi_witness.STATION[0] - 600 / 69.0,   # 600 mi due south of the station
        kiwi_witness.STATION[1])


def _toward_peer(miles_from_peer: float) -> float:
    return PEER[0] + miles_from_peer / 69.0     # due north of the peer


def test_the_receiver_is_placed_around_the_peer_when_one_is_named(tmp_path):
    p = _csv(tmp_path, [("ours", _at(120), 8), ("theirs", _toward_peer(120), 8)])
    assert _hosts(p, 6, peer=PEER)[0] == "theirs"
    assert _hosts(p, 6)[0] == "ours"


def test_a_receiver_in_the_peers_ground_wave_is_not_the_first_choice(tmp_path):
    """The window is the same window; it is measured from the other end."""
    p = _csv(tmp_path, [("onto_it", _toward_peer(15), 8),
                        ("good", _toward_peer(200), 8)])
    assert _hosts(p, 6, peer=PEER)[0] == "good"


def test_a_full_receiver_near_the_peer_does_not_end_the_arm(tmp_path):
    """`all 2 client slots taken` was the whole of one attempt. The walk goes on:
    the ranked list is offered entire, best first, and our own end of it is the
    tail rather than the refusal.

    Both peer-side receivers sit on the path, so what separates them is the
    distance the recording has to be registered over: 340 mi from us is inside
    the 103-329 mi our key-down has ever been found at, and 480 is not.
    """
    p = _csv(tmp_path, [("theirs", _toward_peer(120), 8),
                        ("also_theirs", _toward_peer(260), 8),
                        ("ours", _at(120), 8)])
    assert _hosts(p, kiwi_witness.TRIES, peer=PEER) == ["also_theirs", "theirs", "ours"]


# -- where the peer is, from the only thing an arm holds: its callsign ---------

def _roster(tmp_path: Path, name: str, header: str, rows) -> Path:
    p = tmp_path / name
    p.write_text("\n".join([header] + rows) + "\n")
    return p


def test_a_gateway_callsign_resolves_through_the_published_roster(tmp_path):
    r = _roster(tmp_path, "gw.csv", "Callsign,BaseCallsign,GridSquare,Frequency",
                ["KB5LZK,KB5LZK,EM34UT,\"7,101.600 KHz\""])
    where, label = kiwi_witness.peer_latlon("KB5LZK", [r])
    assert label == "KB5LZK EM34UT"
    assert kiwi_witness.great_circle(kiwi_witness.STATION, where) == pytest.approx(663, abs=5)


def test_the_ssid_an_arm_calls_is_not_how_the_station_is_listed(tmp_path):
    r = _roster(tmp_path, "pat.csv", "miles,freq_khz,callsign,modes,grid",
                ["664,7101.6,K5DAT,Pactor 3,EM12KV"])
    assert kiwi_witness.peer_latlon("K5DAT-13", [r])[1] == "K5DAT EM12KV"


def test_a_grid_or_a_lat_lon_needs_no_roster_at_all():
    assert kiwi_witness.peer_latlon("EM34UT")[0] == pytest.approx((34.81, -92.29), abs=0.02)
    assert kiwi_witness.peer_latlon("34.8,-92.3")[0] == (34.8, -92.3)


def test_a_callsign_nobody_publishes_is_refused_rather_than_guessed(tmp_path):
    r = _roster(tmp_path, "gw.csv", "Callsign,GridSquare", ["KB5LZK,EM34UT"])
    with pytest.raises(LookupError):
        kiwi_witness.peer_latlon("W9SSJ", [r])


# -- and it says so, at selection time, when it cannot do the job -------------
#
# Four witness recordings of a 662 mi gateway were made 103 mi from our own
# transmitter, and read as cover for the peer's answer slots until someone opened
# the sidecars. Three distances answer that -- from us, from the peer, and off
# the path between them -- and the answer is worth nothing after the report is
# written. A placement that fails one of them is taken anyway and says which.

def _cand(mi_us, mi_peer=None):
    return kiwi_witness.Cand("rx", 8073, "rx", mi_us, mi_peer)


APART = 663.0                                   # EN63 to KB5LZK EM34UT


def test_a_receiver_nearer_us_than_the_peer_says_our_carrier_is_over_it():
    said = kiwi_witness.placement(_cand(103, 662), "KB5LZK EM34UT", APART)
    assert said.startswith("WITNESS PLACED SHORT")
    assert "our carrier is over the peer's answer" in said
    assert "103" in said and "662" in said


def test_a_receiver_too_far_from_us_to_register_is_short_however_near_the_peer():
    """A witness is read by registering it against our own keying, so the
    distance that decides is the one from us: the 2026-08-31 arm's receiver was
    717 mi from the gateway and 830 from us, held our key-down in 0.04 of its
    frames, and could not be registered against our own recording at all."""
    said = kiwi_witness.placement(_cand(790, 301), "KB5LZK EM34UT", APART)
    assert said.startswith("WITNESS PLACED SHORT")
    assert "our key-down has ever registered from" in said
    assert "alignment verdict" in said


def test_a_receiver_that_passes_all_three_says_the_pair_is_one_clock():
    said = kiwi_witness.placement(_cand(329, 124), "VE3KPG FN04VE", 440.0)
    assert said.startswith("WITNESS PLACED:")
    assert "both stations are on one clock" in said
    assert "alignment verdict" in said


def test_with_no_peer_named_it_claims_nothing_about_the_peer():
    said = kiwi_witness.placement(_cand(103), None, 0.0)
    assert said.startswith("WITNESS PLACED FOR US")
    assert "no peer named" in said and "SHORT" not in said


# -- the lag is a track, not a fresh search around the caller's guess ----------
#
# `captures/onair-0826-1041`: a PACTOR arm where both stations keyed on the same
# 1.25 s raster for the whole session. A key-down mask of that correlates with
# itself one whole cycle away nearly as well as at zero, and every window was
# searched +-2 s around one fixed guess -- three peaks on the menu. The run
# printed 13.040 s and 16.130 s beside neighbours at 14.89, the 16.130 at
# r=0.814, which cleared the `r > HELD` filter the drift figure is taken over.

def _raster(lags, rate=RATE, cycle=1.25, up=0.12, secs=48.0):
    """A keyed 1.25 s raster in `theirs`, and `ours` at lag(t) inside it."""
    n = int(secs * rate)
    theirs = np.zeros(n)
    for k in range(int(secs / cycle)):
        s = int(k * cycle * rate)
        theirs[s:s + int(up * rate)] = 1.0
    ours = np.zeros(n)
    for i in range(n):
        j = i + int(round(lags(i / rate) * rate))
        if 0 <= j < n:
            ours[i] = theirs[j]
    return ours, theirs


def test_the_lag_walks_a_staircase_past_its_own_search_radius():
    """Four 100 ms drops in 40 s, searched +-150 ms. A capture loses air a block
    at a time, so the last step is 400 ms from where the caller pointed and no
    single step is far from the one before it."""
    ours, theirs = _raster(lambda t: 0.1 * int(t // 10))
    lc = witness_align.lag_curve(ours, theirs, RATE, 0.0, 8.0, 2.0, 0.15)
    assert lc[-1][1] == pytest.approx(0.4, abs=0.02)
    assert lc[:, 1].max() == pytest.approx(0.4, abs=0.02)


def test_a_poor_window_holds_the_track_instead_of_steering_it():
    """The alias wins where the true match has nothing to work with, and a
    tracker that took it would search the next window a cycle out."""
    ours, theirs = _raster(lambda t: 0.2)
    ours[int(20 * RATE):int(28 * RATE)] = 0.0
    ours[int(21 * RATE):int(22 * RATE)] = 1.0        # a shape at the wrong phase
    lc = witness_align.lag_curve(ours, theirs, RATE, 0.2, 8.0, 2.0, 1.0)
    good = lc[lc[:, 2] > witness_align.HELD]
    assert np.allclose(good[:, 1], 0.2, atol=0.02), good


# -- and whether the pair is one measurement at all ---------------------------
#
# Two gates stand in front of a witness -- `coverage` asks which station it is
# nearer, and a 4 s probe drops one that hears nothing -- and NEITHER ASKS
# WHETHER IT HEARS US. Our own keying is the only thing both recordings carry, so
# it is the time base they are registered on. On 2026-08-31 a receiver that passed
# both gates held our key-down in 0.04 of its frames against our own 0.37, drew a
# whole-file lag of 189.69 s at r=0.177 and jumped +-400 ms window to window; the
# arm was written up before anyone read that line.


def test_a_pair_that_tracks_is_declared_one_measurement():
    ours, theirs = _raster(lambda t: 0.2)
    lc = witness_align.lag_curve(ours, theirs, RATE, 0.2, 8.0, 2.0, 0.25)
    assert witness_align.verdict(ours, theirs, lc).startswith("WITNESS ALIGNED")


def test_a_witness_that_does_not_carry_our_keying_refuses_the_licence():
    """The failure neither gate can see: a recording of the same band at the same
    hour whose key-down has nothing to do with ours. What it does not hold in an
    answer slot is not what the peer did not send."""
    ours, _ = _raster(lambda t: 0.2)
    other = (np.random.default_rng(3).random(len(ours)) < 0.04).astype(float)
    lc = witness_align.lag_curve(ours, other, RATE, 0.0, 8.0, 2.0, 0.25)
    said = witness_align.verdict(ours, other, lc)
    assert said.startswith("WITNESS UNALIGNED")
    assert "not the peer's silence" in said


# -- the first lag is found whichever recording is the shorter ----------------
#
# A witness opens on a handshake of its own, stops with the arm and loses blocks
# of its own, so on 24 of the 29 peer-placed pairs in this station's captures it
# is the SHORTER file. Sliding a window of ours around inside it needs the opposite, and
# every one of those runs printed `no overlap to measure`, whose text reads as two
# recordings of different air.


# -- and a witness too short to search is said to be that --------------------
#
# The launcher places the witness once the gate has passed the channel, so the
# receiver can come up seconds into the arm rather than well ahead of it. Past
# `OVERLAP` there is no lag to find, and NO_OVERLAP's text sends the reader off
# to check whether the two files are even of the same air. They are; there is
# just not enough of one of them, which is a number this can print.


def test_a_witness_too_short_to_search_says_so_rather_than_different_air():
    ours, theirs = _raster(lambda t: 0.0, secs=48.0)
    said = witness_align.too_short(ours, theirs[:int(20 * RATE)], RATE)
    assert said is not None
    assert "came up late or dropped early" in said
    assert "not a recording of different air" in said
    assert "20 s" in said and "48 s" in said


def test_a_witness_long_enough_is_not_explained_away():
    ours, theirs = _raster(lambda t: 0.0, secs=48.0)
    assert witness_align.too_short(ours, theirs[:int(30 * RATE)], RATE) is None


def test_our_own_silence_is_not_the_witness_being_short():
    """A recording that never keys fails for its own reason, which NO_OVERLAP
    already names -- claiming the witness was short would be the wrong answer."""
    _, theirs = _raster(lambda t: 0.0, secs=48.0)
    assert witness_align.too_short(np.zeros(int(48 * RATE)),
                                   theirs[:int(10 * RATE)], RATE) is None


def test_the_first_lag_is_found_with_the_witness_the_shorter_file():
    ours, theirs = _raster(lambda t: 0.0, secs=60.0)
    lag, r = witness_align.whole_file_lag(
        ours, theirs[int(9 * RATE):int(45 * RATE)], RATE)
    # Under 1 because 24 s of our 60 has no witness to be found in; the lag it
    # settles on is the whole reason the windowed track has somewhere to start.
    assert lag == pytest.approx(-9.0, abs=0.05) and r > witness_align.HELD


def test_the_first_lag_is_found_with_the_witness_the_longer_file():
    ours, theirs = _raster(lambda t: 0.0, secs=60.0)
    lag, r = witness_align.whole_file_lag(
        ours, np.concatenate([np.zeros(int(7 * RATE)), theirs]), RATE)
    assert lag == pytest.approx(7.0, abs=0.05) and r > 0.9


def test_a_key_down_lands_on_its_second_across_a_chunk_boundary():
    """The frame series is taken a chunk at a time, because the arm's teardown now
    asks for it on whole sessions rather than an operator asking once on a 60 s
    pair. A boundary that dropped or shifted a frame would move every lag read
    after it, and the sessions long enough to cross one are exactly the held
    ones."""
    fs = 11999.0
    on = np.flatnonzero(witness_align.keydown(
        keyed(fs, [(40.9, 41.2), (80.0, 80.5)], secs=90.0),
        fs, TONES, RATE)) / RATE
    assert on.min() == pytest.approx(40.9, abs=0.06)
    assert on.max() == pytest.approx(80.5, abs=0.06)
