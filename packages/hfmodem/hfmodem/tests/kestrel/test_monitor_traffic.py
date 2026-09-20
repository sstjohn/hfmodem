# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the channel monitor makes of a channel whose traffic is known.

The 2026-07-26 recording is the one place the project holds a fully labelled
150 s of real HF: our own eight connect-requests to KB9MMT, KB9MMT's two answers,
and a third station's narrowband session from 100 s on. Run against it, the
monitor used to report eight ``wideband undecoded`` lines about 8.3 s apart and
nothing else — no station named, and not one of the eleven real events among them.

The eight lines were not signals. Each landed within 10 ms of the instant our own
transmission ended: the rig mutes its receiver while it keys, that 2.1 s hole
pulled the gate's noise floor 10 dB below the band, and the band noise *resuming*
on unmute then cleared the enter threshold. The real bursts were inside those
holes, 20 dB below the band and invisible to any energy gate — which is why the
handshake bursts are now found by what they say rather than by how loud they are.
"""
from __future__ import annotations

import subprocess
import sys
import time
from functools import lru_cache

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_control as VC
from hfmodem.tests.kestrel import corpora

VM = corpora.harness("vara_monitor")

# Ground truth for offair/kb9mmt_reply_20260726. Our transmissions are the times
# kestrel keyed; the answers are KB9MMT's, both 15/15 against KB9MMT and 0/15
# against our own call. The third station is unaddressed: its connect-request
# preamble locks 10/10 two tone-bins low, and no candidate callsign fits its
# payload, because the narrowband payload tone law is not reversed.
GATEWAY = "KB9MMT"
OUR_CALLS = (5.32, 12.40, 20.77, 29.13, 37.46, 45.76, 54.10, 62.37)
THEIR_ANSWERS = (56.02, 63.98)
THIRD_STATION = 100.81
UNMUTES = (7.27, 14.34, 22.72, 31.08, 39.40, 47.70, 56.04, 64.32)


@lru_cache(maxsize=4)
def _log(path: str, calls: tuple[str, ...], wideband: bool = True):
    """Everything the monitor reports on a recording, end to end through the same
    generator the command line prints from."""
    session = VM.Session()
    return [(t, r) for t, r in VM.monitor(VM.wav_source(path), list(calls),
                                          session, wideband) if r is not None]


def _onair():
    return _log(str(corpora.ONAIR_CONNECT_ATTEMPT), (GATEWAY,))


def _of(log, kind):
    return [(t, r) for t, r in log if r.kind == kind]


def _near(times, want, tol=0.06):
    return all(any(abs(t - w) <= tol for t in times) for w in want)


# --------------------------------------------------------------------------- #
# The recording with known ground truth.
@corpora.requires_onair_connect_attempt
def test_our_own_connect_requests_are_named():
    """Eight transmissions to KB9MMT, every one heard back through our own muted
    receiver at -33 dBFS against a -13 dBFS band — and every one an exact match."""
    got = _of(_onair(), "CR")
    named = [(t, r) for t, r in got if r.gateway == GATEWAY]
    assert len(named) == 8, f"{len(named)} of {len(got)} CRs named {GATEWAY}"
    assert _near([t for t, _ in named], OUR_CALLS), \
        f"CRs at {[round(t, 2) for t, _ in named]}, expected {list(OUR_CALLS)}"
    for _t, r in named:
        assert r.quality.startswith("31/31"), f"a CR matched {r.quality}"


@corpora.requires_onair_connect_attempt
def test_both_of_the_gateways_answers_are_reported():
    """Both answers, including the one whose preamble never reached us.

    KB9MMT began replying 0.34 s before our own transmission ended, so the eight
    fixed preamble tones of the second answer are entirely under our receiver's
    mute — 0 of 8. It is found by its payload, which is the same fifteen tones
    whichever end of the burst survives."""
    got = _of(_onair(), "connect-response")
    assert [r.gateway for _t, r in got] == [GATEWAY, GATEWAY], \
        f"answers reported as {[(round(t, 2), r.gateway) for t, r in got]}"
    assert _near([t for t, _ in got], THEIR_ANSWERS), \
        f"answers at {[round(t, 2) for t, _ in got]}, expected {list(THEIR_ANSWERS)}"
    for _t, r in got:
        assert r.quality.startswith("15/15"), f"an answer matched {r.quality}"
    assert got[1][1].quality.endswith("0/8 preamble"), (
        "the second answer's preamble is under our own mute; if it now reads, the "
        f"burst found is not the one at {THEIR_ANSWERS[1]} s ({got[1][1].quality})")


@corpora.requires_onair_connect_attempt
def test_the_third_station_is_calling_at_bw500():
    """A station calling somebody at 100.81 s, two tone-bins low. Its preamble is
    the same fixed one, so the burst is located and reported; its payload matched
    no candidate, and this test used to assert only that the monitor said so.

    Measured 2026-08-14: every one of its 28 payload tones lands on the 14-carrier
    46.9 Hz lattice inside 1172-1781 Hz that a BW500 station transmits on, against
    the 70-carrier 23.4 Hz alphabet the payload was being scored against. So the
    payload law was not unreversed — it was the wrong law.

    The right law is :data:`VF.CR500`, solved 2026-08-15, and the scanner
    regenerates against it whenever the lattice says to. This burst stays
    unaddressed anyway: it is a station calling somebody who is not on the
    candidate list, which is a gap in the list rather than in the alphabet."""
    got = [(t, r) for t, r in _of(_onair(), "CR") if not r.gateway]
    assert len(got) == 1, f"{len(got)} unaddressed CRs, expected 1"
    t, r = got[0]
    assert abs(t - THIRD_STATION) <= 0.06, f"unaddressed CR at {t:.2f} s"
    assert "off frequency" in r.info and "-47 Hz" in r.info, r.info
    assert "BW500" in r.info, r.info
    assert r.quality.startswith("28/28 payload on the BW500 tone lattice"), r.quality
    assert "10/10 preamble" in r.quality, r.quality


@corpora.requires_onair_connect_attempt
def test_the_muted_receiver_is_never_reported_as_a_burst():
    """The eight wrong lines, gone. Each sat at the instant the band came back.

    The defect had a shape, and the shape is what is pinned: a bracket opening
    within 10 ms of EVERY unmute, each reported as a 12 s wideband over — eight
    lines, one per transmission, none of them a signal.

    It is not pinned as "no event near an unmute", which was how this read until
    2026-08-14 and is a rule the receiver cannot afford. A gateway answers inside
    the ARQ turnaround, so its first over begins where our own transmission stops:
    on `kc9ghz_connect_20260806` the mute ends at 58.05 s and the CRC-clean BW2300
    greeting starts at 58.08 s, 30 ms later. A rule that forbids a burst 0.3 s
    after an unmute forbids hearing a reply. What tells the two apart is what is in
    the audio, not when it arrives.

    Measured here with the gate as it stands: one of the eight unmutes carries an
    event, a 1.12 s `unknown` at +5.1 dB over the floor, and none of the eight
    carries a wideband over.
    """
    log = _onair()
    at_unmute = [(t, r) for t, r in log
                 for u in UNMUTES if 0 <= t - u < 0.3]
    assert not [r for _, r in at_unmute if r.kind in ("wideband", "long burst")], (
        "a wideband over was reported at an unmute, which is the defect itself: "
        + "; ".join(f"{t:.2f} {r.kind}" for t, r in at_unmute))
    assert len(at_unmute) <= 1, (
        f"{len(at_unmute)} of the eight unmutes reported an event, and the defect "
        "was that all eight did: "
        + "; ".join(f"{t:.2f} {r.kind} {r.info}" for t, r in at_unmute))


@corpora.requires_onair_connect_attempt
def test_the_gate_never_brackets_one_of_the_connect_requests():
    """Why the scanner exists, on the recording that motivated it.

    Our own eight requests arrive through our own muted receiver at -33 dBFS
    against a band of -12, so no threshold over a floor read from the band will
    ever bracket one — and the gate must not be what the handshake search depends
    on. This is the claim `packages/creance/tests/test_monitor_handshake_scan.py`
    makes end to end; here it is against real audio.

    Until 2026-08-14 this asserted the gate opened NOWHERE on the recording. That
    stopped being true when the gate learned to open on a lift that is held rather
    than a peak, and it stopped being true for a reason: from 100 s the recording
    carries one-hot short frames at +4.9 to +5.4 dB alternating with DBPSK control
    bursts and two wideband overs — the shape of a live link, beginning right after
    the gateway's answer at 63.98 s that the scanner reads. The claim worth keeping
    is the narrower one.
    """
    x = corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT)
    x = x / (np.abs(x).max() or 1.0)
    gate = VM.MonitorGate()
    got = [(s / VM.FS, (s + len(b)) / VM.FS)
           for i in range(0, len(x), 4800) for s, b in gate.push(x[i:i + 4800])]
    for t, r in _of(_onair(), "CR"):
        if not r.gateway:                      # the third station, not one of ours
            continue
        assert not [w for w in got if w[0] < t + 1.73 and t < w[1]], (
            f"the gate bracketed our own connect request at {t:.2f} s, so this "
            "recording no longer shows why the stream search is needed")


def test_a_muted_receiver_does_not_pull_the_floor_under_the_band():
    """The mechanism, at its smallest: band noise, a receiver muted 20 dB down for
    two seconds, band noise. The floor has to still describe the band afterwards."""
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.standard_normal(20 * VM.FS),
                        rng.standard_normal(2 * VM.FS) * 10 ** (-20 / 20),
                        rng.standard_normal(20 * VM.FS)]) * 0.05
    gate = VM.MonitorGate()
    got = [s for i in range(0, len(x), 4800) for s, _ in gate.push(x[i:i + 4800])]
    assert not got, f"band noise either side of a mute bracketed {len(got)} burst(s)"
    assert 20 * np.log10(gate.nf) > -32, (
        f"the floor finished at {20 * np.log10(gate.nf):.1f} dBFS, below the "
        "-26 dBFS band noise it is supposed to describe")


@corpora.requires_onair_connect_attempt
def test_it_decodes_far_faster_than_real_time():
    """Which is why the wideband decode is no longer skipped on a live device.
    Measured at 8.1 s for the 150 s recording, wideband decode included — 18x
    real time. The bar is loose enough to survive a loaded machine and tight
    enough to catch a live path that can no longer keep up with its own audio."""
    _log.cache_clear()
    t0 = time.perf_counter()
    _onair()
    dt = time.perf_counter() - t0
    assert dt < 60.0, f"150 s of audio took {dt:.1f} s"


# --------------------------------------------------------------------------- #
# Both ends of somebody's QSO.
@pytest.mark.parametrize("call,answers", [("NS0A", 2), ("KC9GHZ", 0)])
def test_both_ends_of_a_gateway_session_are_followed(call, answers):
    """The initiator's connect-requests, the gateway's answers, and the gateway's
    DATA overs decoded to a byte count — both ends of the exchange, off one side
    of the audio. NS0A answers twice, at 9.78 s and 12.14 s, which is what gateways
    do; the second of those is another whose preamble is under our own mute.

    KC9GHZ_2300 holds no answer to find, and the parameter says so rather than the
    test quietly settling for whatever turns up: scored at every alignment on its
    108 s, the best fit to a connect-response for KC9GHZ reaches 4 of 15 tones,
    against the 15 of 15 NS0A's answer reaches. What can be followed there is the
    calling and the data."""
    path = corpora.OFFAIR / f"{call}_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air {call} session not present ({path})")
    log = _log(str(path), (call,))
    assert [r.gateway for _t, r in _of(log, "CR")].count(call) >= 1, \
        "no connect-request to the gateway was named"
    heard = [r.gateway for _t, r in _of(log, "connect-response")].count(call)
    assert heard == answers, f"{heard} answers from {call}, expected {answers}"
    overs = _of(log, "DATA over")
    assert len(overs) >= 2, f"{len(overs)} DATA overs decoded"
    assert all(r.quality == "CRC ok" for _t, r in overs)


def test_a_second_station_on_the_channel_is_named_too():
    """The point of sitting on a frequency: KC9GHZ's session has somebody else
    calling AG7MM through the middle of it, six times between 39.8 s and 56.0 s
    on a 2.7 s cadence, while our own exchange with KC9GHZ carries on around it.

    Not one of those six carries a preamble this decoder recognises — 0 or 1 of 10
    — and the first call of each series on this channel does: 10 of 10 for KC9GHZ
    at 5.21 s, 9 of 10 for NS0A at 5.20 s, 1 of 10 for every repeat after. So all
    six are found by their payload alone, one of them matching 31 tones of 31.
    They are the reason the payload route exists as well as the preamble one."""
    path = corpora.OFFAIR / "KC9GHZ_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip(f"off-air KC9GHZ session not present ({path})")
    got = [(t, r) for t, r in _of(_log(str(path), ("KC9GHZ", "AG7MM")), "CR")
           if r.gateway == "AG7MM"]
    assert len(got) == 6, f"{len(got)} calls to AG7MM found, expected 6"
    assert 39.0 < got[0][0] < 57.0, f"first AG7MM call at {got[0][0]:.2f} s"
    assert any(r.quality.startswith("31/31") for _t, r in got), \
        f"best AG7MM match was {[r.quality for _t, r in got]}"


@corpora.requires_clear_channel
def test_the_command_line_runs_a_recording_end_to_end():
    """``--help`` reaches argparse; this reaches the decoders. The channel line is
    the one thing printed whether or not anything is heard, so it is what tells a
    run apart from a crash that exited 0."""
    cache = corpora.ROOT / "winlink-vara-gateways.csv"
    if not cache.exists():
        pytest.skip("no cached gateway list; --wav would go to the network")
    tool = corpora.TOOLS / "vara_monitor.py"
    r = subprocess.run(
        [sys.executable, str(tool), "--wav", str(corpora.CLEAR_CHANNEL),
         "--cache", str(cache), "--calls", GATEWAY, "--no-wideband"],
        capture_output=True, text=True, timeout=300, cwd=corpora.REPO)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "channel clear" in r.stdout, r.stdout[-1000:]


@corpora.requires_clear_channel
def test_a_verified_clear_channel_reports_nothing_and_reads_clear():
    session = VM.Session()
    got = [r for _t, r in VM.monitor(VM.wav_source(str(corpora.CLEAR_CHANNEL)),
                                     [GATEWAY], session) if r is not None]
    assert not got, f"{len(got)} event(s) on 30 s of verified clear channel"
    assert session.occupancy is not None and not session.occupancy[0], \
        f"verified clear channel read as {session.occupancy}"


# --------------------------------------------------------------------------- #
# The false-accept floor, over every 48 kHz recording the shared corpus holds.
@corpora.requires_regress_fixtures
def test_no_station_is_named_on_a_recording_that_holds_none():
    """848 s of real off-air HF — four more VARA sessions, PACTOR-1/2/3, ARDOP,
    FT8, WSPR, narrowband and band noise from four continents — scanned against
    343 candidate callsigns. A detection may name a station only if that station's
    call is in the recording's own name; measured, none of them names anything.

    The three CR bursts it does report there are corroborated independently:
    ``vara_mfsk.lock_preamble`` finds a 10/10 connect-request preamble at the same
    place in each, and the payload matches no candidate because the destination is
    somebody who is not a Winlink VARA gateway."""
    calls = tuple(VM.load_gateways(corpora.ROOT / "winlink-vara-gateways.csv",
                                   None, False))
    if not calls:
        pytest.skip("no cached Winlink gateway list to score against")
    for path in sorted(corpora.REGRESS_FIXTURES.glob("*.wav")):
        from scipy.io import wavfile

        if wavfile.read(str(path), mmap=True)[0] != VM.FS:
            continue                     # the websdr captures are 11999 Hz
        session = VM.Session()
        for _t, r in VM.monitor(VM.wav_source(str(path)), list(calls), session,
                                False):
            if r is not None and r.gateway:
                assert r.gateway.lower() in path.name.lower(), (
                    f"{path.name} named {r.gateway}, which is not on it")


# --------------------------------------------------------------------------- #
# What a line that decoded nothing is allowed to claim.
@corpora.requires_regress_fixtures
def test_a_burst_that_was_not_decoded_names_no_protocol_and_no_role():
    """This line used to read ``session-setup · initiator rec3 short frame``.

    Three claims deep — VARA, the caller's, before the first data over — off one
    measurement: a single spectral peak per 512-sample column. That is the shape of
    every keyed narrowband carrier, and it put 30 of these lines on five corpus
    recordings that hold no VARA at all, the PACTOR-2 reference among them. Here it is
    the PACTOR-1 fixtures that make the point, because their traffic could not be
    VARA's caller under any reading.
    """
    p1 = corpora.REGRESS_FIXTURES / "pos_p1_kb8ay_float.wav"
    if not p1.exists():
        pytest.skip(f"PACTOR-1 fixture not present ({p1})")
    got = [r for _t, r in _log(str(p1), (), False) if r.kind == "one-hot burst"]
    assert got, "the fixture no longer reaches this branch; the test is measuring nothing"
    for r in got:
        low = (r.kind + " " + r.info).lower()
        for word in ("session", "initiator", "vara"):
            assert word not in low, f"a burst nothing decoded claimed {word!r}: {r}"


@corpora.requires_regress_fixtures
def test_an_undecoded_burst_carries_its_level_over_the_tracked_floor():
    """The number whose absence made the wording believable.

    Five of these were reported in one 71 s recording, two of them at the noise
    floor, and the line gave a reader no way to see it: the identification and the
    strength of the thing identified were the same sentence. Every classification
    that rests on shape rather than on a decode now says how far over the floor
    the segmenter was tracking its burst stood."""
    p1 = corpora.REGRESS_FIXTURES / "pos_p1_kb8ay_float.wav"
    if not p1.exists():
        pytest.skip(f"PACTOR-1 fixture not present ({p1})")
    shape = {"one-hot burst", "unknown", "DBPSK burst"}
    got = [r for _t, r in _log(str(p1), (), False) if r.kind in shape]
    assert got, "no undecoded bursts on the fixture; the test is measuring nothing"
    for r in got:
        assert "dB" in r.quality, f"{r.kind} reported without a level: {r.quality!r}"


@corpora.requires_regress_fixtures
def test_a_dbpsk_burst_names_no_protocol_and_no_role():
    """The catch-all one duration bin along, and the log that made it visible.

    A DBPSK collapse on VARA's control sub-bands with no token pattern within
    Hamming 2 used to print `session · DBPSK control/keepalive (token vocabulary
    unknown)` — a session it could not place the burst in, a keepalive it had not
    read, and a vocabulary miss dressed as a name. Against KC9GHZ on 2026-08-18 it
    named eleven pre-connect bursts from 22 to 278 columns alike.

    The PACTOR-1 captures are what settle that the kind may not claim VARA either:
    the collapse fires 14 times across two of the three, and their traffic could
    not be a VARA session's under any reading.
    """
    p1 = corpora.REGRESS_FIXTURES / "pos_p1_twosided_14110.wav"
    if not p1.exists():
        pytest.skip(f"PACTOR-1 fixture not present ({p1})")
    got = [r for _t, r in _log(str(p1), (), False) if r.kind == "DBPSK burst"]
    assert got, "the fixture no longer reaches this branch; the test is measuring nothing"
    for r in got:
        low = (r.kind + " " + r.info).lower()
        for word in ("session", "keepalive", "vara", "control"):
            assert word not in low, f"a burst nothing decoded claimed {word!r}: {r}"


def _multicarrier_dbpsk(cols: int, seed: int = 0) -> np.ndarray:
    """Differential BPSK on every one of VARA's control sub-bands at once, `cols`
    columns long and carrying no token pattern — what :func:`_dbpsk_collapse`
    measures, over a floor that keeps it off the one-hot branch ahead of it."""
    rng = np.random.default_rng(seed)
    t = np.arange(cols * 512)
    x = rng.standard_normal(cols * 512) * 0.4
    for f0 in VM._CTRL_CARRIERS:
        phase = np.repeat(np.cumsum(rng.integers(0, 2, cols)) * np.pi, 512)
        x += np.cos(2 * np.pi * f0 * t / VM.FS + phase)
    return x / np.abs(x).max()


def test_a_dbpsk_burst_is_sized_against_the_token_vocabulary():
    """The eleven KC9GHZ bursts, restated as the three answers the line may give.

    Column counts from `working/vara-kc9ghz-force.log`; the vocabulary span is
    `vara_control.TOKENS`' own, so the boundaries move with it and not with
    this test.
    """
    assert VM._TOKEN_COL_LO < VM._TOKEN_COL_HI, (
        "the token vocabulary has no span to size a burst against")
    said = {}
    for cols in (22, 24, 36, 38, 44, 46, 54, 152, 278):
        seg = _multicarrier_dbpsk(cols)
        r = VM._unnamed_burst(seg, len(seg) / VM.FS, len(seg), 4.4)
        assert r.kind == "DBPSK burst", f"{cols} columns landed as {r.kind}: {r}"
        assert f"~{cols}col" in r.quality, f"{cols} columns went unreported: {r}"
        said[cols] = r.info
    assert "shorter than" in said[22] and "shorter than" in said[24]
    assert "within" in said[38] and "within" in said[44]
    for cols in (46, 54, 152, 278):
        assert "longer than" in said[cols], f"{cols} columns read as a token: {said[cols]}"


def test_the_channel_line_says_what_ACTIVE_rests_on():
    """``last_burst_t`` is armed by EVERY classification, an unidentified one
    included, so a bare "ACTIVE" restated "the gate opened recently" in the
    vocabulary of a station being heard. A night of logs was read as evidence of
    activity on that basis."""
    session = VM.Session()
    assert "nothing bracketed" in session._state(10.0)
    session.last_burst_t = 8.0
    assert session._state(10.0).startswith("ACTIVE")
    assert "not an occupancy measurement" in session._state(10.0)
    assert session._state(10.0 + VM.IDLE_AFTER).startswith("idle")


#: Burst lengths the label has to hold one floor across, in 512-sample columns.
#: 32-45 is the token vocabulary's own span; the rest is the range the segmenter
#: hands `_unnamed_burst` off air, 0.17 s to 3.4 s.
_NOISE_COLS = (16, 24, 32, 36, 40, 44, 48, 64, 96, 128, 192, 256, 320)


@corpora.requires_clear_channel
@corpora.requires_onair_silent_calls
def test_band_noise_is_not_a_dbpsk_burst_at_any_length():
    """One threshold, thirteen burst lengths, and the same floor at each.

    `DBPSK burst` used to be a length-dependent claim wearing a fixed number. The
    collapse it rests on is the largest of 160 tries at averaging a burst's ~ncol
    differentials, so on band noise it falls as 1/sqrt(ncol) — and a constant 0.35
    compared against that passed 100% of 16-column noise windows and 0.4% of
    320-column ones. At the 32-45 columns the token vocabulary occupies it passed
    84%, which is what put eight `tokens unresolved` lines under a KD0PYG arm on
    2026-08-23 and twenty-six across that night's four connected arms.

    The statistic is now measured in units of its own noise, and the two halves of
    that are asserted together: a floor that is low everywhere is worth nothing if
    it is low because the statistic died, and a statistic that separates is worth
    nothing if the threshold under it is really thirteen thresholds.

    Sampled over 30 s of a verified-clear 40 m channel and the two 2026-08-06 calls
    nobody answered — the same population `test_vara_control` measures the token
    decoder's own false-accept floor over.
    """
    rng = np.random.default_rng(3)
    noise = [wav / (np.abs(wav).max() or 1.0) for wav in
             (corpora.wav_mono(p) for p in
              (corpora.CLEAR_CHANNEL, *corpora.ONAIR_SILENT_CALLS))]
    named, collapse = {}, {}
    for cols in _NOISE_COLS:
        n = cols * VC.H
        segs = [a[i:i + n] for a in noise
                for i in rng.integers(0, len(a) - n, 40)]
        kinds = [VM.classify(s, [], decode_wideband=False).kind for s in segs]
        named[cols] = kinds.count("DBPSK burst") / len(kinds)
        collapse[cols] = float(np.median([VM._dbpsk_collapse(s) for s in segs]))

    worst = max(named, key=named.get)
    assert named[worst] <= 0.08, (
        f"{100 * named[worst]:.0f}% of {worst}-column band-noise windows were "
        f"reported as a DBPSK burst: " +
        ", ".join(f"{c}col {100 * f:.0f}%" for c, f in named.items()))

    lo, hi = min(collapse.values()), max(collapse.values())
    assert hi / lo < 1.5, (
        "the collapse's own noise level still depends on burst length, so one "
        f"threshold cannot serve all of them: {hi / lo:.1f}x across "
        f"{_NOISE_COLS[0]}-{_NOISE_COLS[-1]} columns — " +
        ", ".join(f"{c}col {v:.2f}" for c, v in collapse.items()))


@corpora.requires_clear_channel
@corpora.requires_onair_silent_calls
@corpora.requires_qrn_80m
def test_the_one_hot_label_is_not_a_band_noise_floor():
    """The other shape gate on the same line, and what its 16% actually was.

    Sampled the same way as the DBPSK floor above — random windows at thirteen
    widths over the same three recordings — `one-hot burst` reads 16% and reads it
    flat in length, which looks exactly like the defect `_dbpsk_collapse` had and
    is not one. Two thirds of that population is this station's own transmit mute,
    and a mute is one-hot: a near-dead codec input is a handful of isolated spurs.

    So the two halves are asserted apart. On audio that holds a band and nothing
    keyed in it the label is quiet; on audio that holds no signal at all it fires
    every time, and it is right to, because shape is all it claims and that shape
    is there. What keeps it out of the tool's output is the gate ahead of it,
    which never opens 16 dB under the floor it is tracking.
    """
    rng = np.random.default_rng(3)

    def rate(paths, window=None):
        auds = []
        for p in paths:
            a = corpora.wav_mono(p)
            if window:
                a = a[int(window[0] * VM.FS):int(window[1] * VM.FS)]
            auds.append(a / (np.abs(a).max() or 1.0))
        out = {}
        for cols in _NOISE_COLS:
            n = cols * VC.H
            segs = [a[i:i + n] for a in auds if len(a) > n
                    for i in rng.integers(0, len(a) - n, 16)]
            kinds = [VM.classify(s, [], decode_wideband=False).kind for s in segs]
            out[cols] = kinds.count("one-hot burst") / len(kinds)
        return out

    quiet = {**rate([corpora.CLEAR_CHANNEL]),
             **rate(corpora.QRN_80M, corpora.QRN_80M_WINDOW)}
    worst = max(quiet, key=quiet.get)
    assert quiet[worst] <= 0.08, (
        f"{100 * quiet[worst]:.0f}% of {worst}-column windows of a band with "
        "nothing keyed in it were named a one-hot burst: " +
        ", ".join(f"{c}col {100 * f:.0f}%" for c, f in quiet.items()))

    # And where the 16% comes from: the mute, which the level says is not a burst.
    a = corpora.wav_mono(corpora.ONAIR_SILENT_CALLS[0])
    a = a / (np.abs(a).max() or 1.0)
    n = 44 * VC.H
    env = np.array([np.sqrt((a[i:i + n] ** 2).mean())
                    for i in range(0, len(a) - n, n)])
    floor = np.median(env)
    fired = {"mute": [], "band": []}
    for k, e in enumerate(env):
        seg = a[k * n:k * n + n]
        side = "mute" if e < floor / 4 else "band"
        fired[side].append(VM.classify(seg, [], decode_wideband=False).kind
                           == "one-hot burst")
    assert np.mean(fired["band"]) <= 0.08, (
        f"{100 * np.mean(fired['band']):.0f}% of the live-band windows of the "
        "same recording were named a one-hot burst")
    assert np.mean(fired["mute"]) >= 0.9, (
        "the mute stopped reading as one-hot, so this test is no longer measuring "
        "what the 16% was made of")


@corpora.requires_clear_channel
def test_a_token_buried_in_that_same_noise_still_clears_the_gate():
    """The sensitivity the floor above was bought at.

    Every token in the vocabulary, keyed at 3 dB over the same clear-channel band
    noise the false-accept floor is measured on. `detect_token` resolves all five
    of these down to 0 dB [test_vara_control]; the gate behind it only has to
    still call them structure when the patterns have missed.
    """
    a = corpora.wav_mono(corpora.CLEAR_CHANNEL)
    a = a / (np.abs(a).max() or 1.0)
    rng = np.random.default_rng(7)
    for name in VC.TOKENS:
        token = VC.synth_token(name)
        i = int(rng.integers(0, len(a) - len(token)))
        nz = a[i:i + len(token)]
        amp = (np.sqrt(np.mean(nz * nz)) * 10 ** (3 / 20)
               / np.sqrt(np.mean(token * token)))
        got = VM._dbpsk_collapse(token * amp + nz)
        assert got > VM._COLLAPSE_MIN, (
            f"{name} at 3 dB SNR collapsed to {got:.2f}, under the "
            f"{VM._COLLAPSE_MIN} the gate asks for")
