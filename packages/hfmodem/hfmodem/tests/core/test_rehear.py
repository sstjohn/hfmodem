# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The instrument that measures what the live path threw away, measured itself.

``tools/rehear`` replays a recording through our own receive path twice, once with
the live gating and once ungated, and calls the difference a loss. Two things have
to hold before that difference means anything, and both are checked here.

The ungated pass must not accept noise. It is allowed to be more permissive than
the live path — that is the entire point of running it — but a search that
answers a callsign nobody transmitted turns every quiet minute of the corpus into
a recovered gateway. So it is run against stations that are demonstrably not on
the recording, which is the same discipline `vara_arq`'s own floor rests on:
110,319,450 alignment-shifts of real off-air HF, best non-response 5 of 15, not
one sweep clean.

And the live pass must not claim what the ungated pass cannot reproduce. The two
arms are the same search shown different samples, so an accept in the gated arm
with no counterpart in the ungated one is not a finding — it is a replay that has
stopped modelling the thing it is named for.

And a replay of one of OUR sessions must hear what the station heard. The live
decoder polls `ArqSession.expected_session` every window: a bare control bearing an
id we are party to is admitted at crisp 8.0 where an unsolicited one faces 9.5, and
`_scan_header`'s fallback mints nothing at all below 9.5 without it. A replay
written without that hint is not a stricter reading of the same session, it is a
reading of half of it — 28 of 51 frames on the recording below — and the same goes
for a mute grown off silence brackets in place of the log's own PTT edges. Both are
measured here against the one recording whose live path left a frame-by-frame record
beside it.

The suppression model is checked against the recording it was calibrated on and
against the recording that broke it. The energy model in
``test_onair_connect_chain`` — frames 15 dB under the median for over a second —
was measured on 2026-07-26, when the key was held 0.25 s past the audio and
rigctld took 0.158 s more to unkey. Both are gone. On the 2026-08-19 05:12 call it
now finds no transmission at all, on a recording carrying eight connect-requests
on a 7 s cadence, so a replay resting on it would have been handed the whole file
and called it the live path's view.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from hfmodem.core.resample import from_card
from hfmodem.tests.kestrel import corpora

R = corpora.harness("rehear")
K = corpora.harness("rehear.kestrel")

#: Stations that did not transmit on the recordings below. Two are gateways this
#: station has connected to on other days, which is the harder negative: their
#: tone sets are ones the corpus really carries, somewhere else.
NOT_ON_AIR = ("KB9MMT", "NS0A", "W1AW")


@corpora.requires_onair_connect_attempt
def test_the_answers_this_project_walked_past_are_still_reachable():
    """2026-07-26: KB9MMT answered twice and kestrel reported no answer.

    That recording is why any of this exists, and it is the one case where the
    ground truth is settled independently — both answers regenerate 15 of 15
    payload tones for a callsign that was dialled and no other. The instrument has
    to find them in the audio and has to show the live geometry reaching one.
    """
    r = K.rehear(corpora.ONAIR_CONNECT_ATTEMPT)
    assert r.recorder == R.UPSTREAM
    assert "2x connect-response to KB9MMT" in r.note, r.note
    assert r.live, f"the live geometry no longer reaches KB9MMT's answer: {r.note}"
    assert not r.only_live(), r.only_live()


@corpora.requires_onair_gateway_answer
@pytest.mark.parametrize("wrong", NOT_ON_AIR)
def test_the_ungated_search_answers_nobody_who_was_not_calling(wrong):
    """The bar. An ungated replay that accepts noise recovers gateways for free."""
    x = K._load(corpora.ONAIR_GATEWAY_ANSWER)
    accepts, _ = K._search(x, [(0, len(x))], "W9SSJ", wrong)
    assert accepts == [], (
        f"the ungated arm answers {wrong}, who is not on this recording, at "
        f"{accepts} — every loss this instrument reports is worth what this is")


@corpora.requires_clear_channel
def test_the_ungated_search_hears_nothing_on_a_clear_channel():
    x = K._load(corpora.CLEAR_CHANNEL)
    for call in ("KC9GHZ", *NOT_ON_AIR):
        accepts, _ = K._search(x, [(0, len(x))], "W9SSJ", call)
        assert accepts == [], f"{call} answered on an empty channel at {accepts}"


@corpora.requires_onair_connect_attempt
def test_our_transmissions_are_found_by_what_they_say_and_agree_with_the_mute():
    """Where the live path is blind, decided two ways, on the one recording that
    can answer both. The request-derived span runs from PTT settle to where
    ``AudioVaraIO.tx`` puts its cursor; the mute is what the rig actually did.
    They have to agree, or the model that survives the rig's audio path changing
    is not measuring the same thing the calibrated one did.
    """
    x = K._load(corpora.ONAIR_CONNECT_ATTEMPT)
    ours = [at for at, res in K._scanned(x, ["KB9MMT", "W9SSJ"])
            if res.kind == "CR" and res.gateway == "KB9MMT"]
    assert len(ours) == 8, f"eight requests went out; the scanner reads {len(ours)}"

    spans, mute = K._our_overs(x, ours), K._muted(x)
    assert len(spans) == len(mute) == 8
    for (a, b), (m, n) in zip(spans, mute):
        assert abs(a - m) / K.FS < 0.2, f"key-up disagrees: {a / K.FS} vs {m / K.FS}"
        assert abs(b - n) / K.FS < 0.25, f"resume disagrees: {b / K.FS} vs {n / K.FS}"


@corpora.requires_onair_unanswered_call
def test_the_suppression_model_still_finds_eight_transmissions_without_a_mute():
    """The recording that retired the energy model: the rig's transmit monitor
    reaches the codec at band level, so nothing on it is 15 dB under the median
    for a second, and the eight requests are found by their tones instead."""
    x = K._load(corpora.ONAIR_UNANSWERED_CALL)
    ours = [at for at, res in K._scanned(x, ["KC9GHZ", "W9SSJ"])
            if res.kind == "CR" and res.gateway == "KC9GHZ"]
    assert len(ours) == 8
    assert len(K._our_overs(x, ours)) == 8


@pytest.mark.parametrize("name", ("ONAIR_GATEWAY_ANSWER", "ONAIR_FADED_GREETING",
                                  "ONAIR_OFFSET_ANSWER"))
def test_the_live_arm_claims_no_answer_the_ungated_arm_cannot_reproduce(name):
    path = getattr(corpora, name)
    if not path.exists():
        pytest.skip(f"{name} is not in this tree")
    r = K.rehear(path)
    assert not r.only_live(), (
        f"{path.name}: the gated arm accepts {r.only_live()} and the ungated arm, "
        f"shown strictly more audio by the same search, does not — the replay has "
        f"stopped modelling the live path rather than measuring it")


def test_every_adapter_says_where_its_zero_is_and_where_the_recorder_sits():
    """A time without a stated zero is how a 6x error survived for months, and a
    difference measured against a recording written downstream of the drops is a
    zero that means nothing. Both go in every report or the report is not one."""
    covered = 0
    for protocol, claims, run in R.adapters():
        assert protocol
        sample = next((p for p in R.recordings([corpora.evidence.LOGS / "onair",
                                                corpora.evidence.CAPTURES])
                       if claims(p)), None)
        if sample is None:
            continue
        covered += 1
        r = run(sample)
        assert r.protocol == protocol
        assert r.zero.strip(), f"{protocol} reports times against an unstated zero"
        assert r.recorder in (R.UPSTREAM, R.DOWNSTREAM), r.recorder
        assert r.uncovered.strip(), (
            f"{protocol} declares nothing uncovered — an adapter that reports no "
            f"difference while declining to look has reported nothing")
    assert covered, "no protocol had a recording to replay"


B = corpora.harness("rehear.besra")
S = corpora.harness("rehear.shrike")

_BESRA = corpora.evidence.LOGS / "onair" / "20260803T151721Z-besra-7102000.wav"


@pytest.mark.skipif(not _BESRA.exists(), reason="the 2026-08-03 KE8LVA session is not here")
def test_the_besra_replay_puts_its_frames_where_the_recording_does():
    """The timebase trap, which is the whole of whether besra's arms can be compared.

    ``RollingDecoder`` counts only the samples it was pushed, so a live frame's
    position indexes audio-delivered-to-the-decoder and runs earlier than the
    recording by every withheld block — eleven of them here, 25.0 s. Unadjusted,
    every frame after the first transmission lands seconds early and reads as one
    the mute destroyed.
    """
    r = B.rehear(_BESRA)
    assert r.recorder == R.UPSTREAM
    assert "11 intervals, 25.0 s" in r.note, r.note
    assert not r.only_live(), (
        f"a besra frame the live arm reads and the ungated arm does not: "
        f"{r.only_live()} — the position has not been put back on the recording")


def test_a_shrike_capture_says_it_cannot_witness_a_dropped_sample():
    """The one adapter whose recorder is behind the drops it would measure.

    ``_save_capture`` writes the segment the session READ, after ``flush_to``
    has walked the input's floor past everything taken while our carrier was up.
    A shrike capture therefore cannot hold a sample the decoder was denied, and
    the honest report of that is a declaration, not a zero.
    """
    sample = next((p for p in R.recordings([corpora.evidence.CAPTURES])
                   if S.claims(p)), None)
    if sample is None:
        pytest.skip("no shrike session capture with its sidecar is in this tree")
    r = S.rehear(sample)
    assert r.recorder == R.DOWNSTREAM
    assert "sample drops of every kind" in r.uncovered


@corpora.requires_onair_offset_answer
def test_the_gateway_opens_its_answer_before_the_cursor_comes_back():
    """Where the cursor lands, against where the answer starts.

    Zero for both: the last sample of our own connect-request as it stands on the
    recording — the scanner's burst start plus the 41 tones the waveform fixes,
    1.777 s. The cursor comes back at that plus ``TX_IDLE_HOLD_S`` and
    ``RX_ECHO_GUARD_S``, 0.100 s, and nothing about the band moves it.

    Across the corpus, 23 answers follow their own request and open 0.078-0.168 s
    after it, median 0.143 — which is `PEER_ANSWERS_FROM_S` measured again from a
    different direction. The margin is therefore a median +0.043 s, and four of
    the 23 are negative: N0LCR-1 twice at 0.078 and 0.091, KC9GHZ at 0.095, W8MW
    at 0.094, all of them 2026-08-15 or later. Those four lose their head to a
    cursor that is set from our own transmit tail and knows nothing about how
    fast the peer is.

    It used to be free. The receiver's own mute outlasted our last sample by
    0.214-0.444 s across sixteen transmissions in the 2026-08-02 to 08-06
    sessions, so the audio the cursor skipped was dead anyway. On the five recent
    transmissions where any mute is measurable it runs -0.040 to +0.164 s, and on
    most recent recordings none is measurable at all. The rig stopped being the
    binding constraint and the constant took over.
    """
    x = K._load(corpora.ONAIR_OFFSET_ANSWER)
    found = K._scanned(x, ["N0LCR-1", "W9SSJ"])
    ours = [at for at, r in found if r.kind == "CR" and r.gateway == "N0LCR-1"]
    answers = [at for at, r in found
               if r.kind == "connect-response" and r.gateway == "N0LCR-1"]
    assert ours and answers

    at = answers[0]
    gap = at - (max(c for c in ours if c < at) + K.CR_S)
    assert 0.05 < gap < 0.15, f"N0LCR-1 opened {gap:.3f} s after our last sample"
    assert gap < K.CURSOR_SKIP_S, (
        f"N0LCR-1 opens {gap:.3f} s after our last sample and the cursor comes "
        f"back at {K.CURSOR_SKIP_S:.3f} s, so this answer no longer loses its "
        f"head — if that is because the cursor moved, this test has done its job "
        f"and the numbers in it are stale")


# -- against the station's own log -------------------------------------------

_W6IDS = corpora.BESRA_LOGGED_SESSION
_W6IDS_LOG = corpora.BESRA_SESSION_LOG
_W4UC = corpora.BESRA_REGULAR_CALL
_W4UC_LOG = corpora.BESRA_REGULAR_CALL_LOG


@corpora.requires_besra_logged_session
def test_the_session_log_holds_what_the_recording_cannot():
    """The three things a replay of one of our own sessions cannot read off audio,
    read off the log instead. Cheap — no decoding — and it pins the figures the
    heavier test below rests on."""
    log = B.sessionlog.read(_W6IDS_LOG)
    assert log.session == 0x0D, "the session id the live decoder was polling"
    assert len(log.frames) == 51 and sum(f.header_only for f in log.frames) == 6
    assert len(log.keyed) == 56
    assert sum(b - a for a, b in log.keyed) == pytest.approx(50.58, abs=0.02), (
        "the mute the station actually set, `TX` line to `PTT OFF` line")
    assert len(log.epochs) == 4, "three turns to IRS and the teardown"
    assert B.sessionlog.find(_W6IDS, [_W6IDS_LOG]) is not None, (
        "the log names its own recording and the search no longer finds it")


@corpora.requires_besra_logged_session
def test_the_besra_replay_hears_what_the_station_heard():
    """The acceptance test for the whole instrument, through the shipped entry point.

    W9SSJ worked W6IDS on 7061.5 kHz on 2026-08-26 and the live path reported 51
    frames. A replay built the way this project had been building them — no session
    hint, the mute grown off the capture's silence brackets — reproduces 28 of them:
    it loses the ConAck that opened the session, the END that closed it, eight
    DATAACKs, three BREAKs and ten data frames. The hint alone brings it to 48; the
    log's own `TX`/`PTT OFF` edges bring the last three.

    Not exact in the other direction, and the numbers here say only what was
    measured: all 45 frames the session decoded come back in order, 0.28-6.46 s
    ahead of where the live path reported them, and each of the six header-only
    sightings has a same-type sighting in the replay placed within 10.3 s.

    50 of the 51, not 51, and the missing one is the point of the difference: the
    `16QAM.500.100.O sess=0x00` this session logged at 08:41:05 is a frame nobody
    sent, and `demodulator._addressed` no longer reports it. A replay of the
    station's log reproduces what the station heard; it does not owe the log a
    frame the receiver has since stopped claiming.
    """
    r = R.rehear(_W6IDS, [_W6IDS_LOG])
    assert r.deaf == "", r.deaf
    assert "the log's own TX/PTT-OFF edges" in r.note, r.note
    assert "session 0x0D" in r.note, r.note
    assert r.against_log == (50, 51), (
        f"{r.against_log} — this replay no longer reproduces the session it is a "
        f"replay of; {r.note}")
    assert [f.name for f in B.sessionlog.read(_W6IDS_LOG).frames
            if f.header_only and f.session != 0x0D] == ["16QAM.500.100.O"], (
        "the one logged frame this replay is allowed to miss")
    assert not r.discarded(), r.discarded()


@corpora.requires_besra_logged_session
def test_the_mute_read_off_the_log_is_the_one_the_station_set():
    """14.1 s of difference, which is not a rounding. The brackets are grown over the
    quiet a key leaves in the capture and overshoot every interval they find; they
    also miss five keyings outright, because the rig did not mute the capture for
    those and no level rule finds them."""
    log = B.sessionlog.read(_W6IDS_LOG)
    card = B._capture(_W6IDS)
    au = from_card(card, B.SAMPLE_RATE)
    edges, quiet, level = B._key_edges(au)
    where = B.sessionlog.align(log, [a for a, _ in edges],
                               B._quieter(level, log.keyed))
    assert where is not None and where.residual < 0.02, str(where)

    # Grown off a hint-less pass, because that is the only run that ever grows them:
    # a capture whose rig muted it this lightly leaves no quiet stretch long enough
    # to stand on its own, so every bracket here is closed around a decoded frame and
    # the model is only as good as the frames the run without a log actually has.
    heard = [(at, f) for at, f in B._rolling(card, (), None, 0.0) if B._decoded(f)]
    brackets = B._keyed_intervals(edges, quiet, heard)
    theirs = B._on_this_recording(log.keyed, where.offset, au.size / B.SAMPLE_RATE)
    assert sum(b - a for a, b in brackets) == pytest.approx(64.5, abs=0.3)
    assert sum(b - a for a, b in theirs) == pytest.approx(50.4, abs=0.3)
    assert len(theirs) - len(brackets) == 5, (
        "the brackets used to miss five of the station's keyings; if they no longer "
        "do, this test has done its job and the numbers in it are stale")


@corpora.requires_besra_regular_call
def test_the_alignment_cannot_be_settled_by_the_key_ups():
    """2026-08-29 15:15z, W4UC: six identical 1.99 s ConReqs on a 3.85 s cadence and
    nothing back.

    A transmission leaves two runs of silence in the capture, the settle at the key
    and the tail at the unkey, and the alignment is a vote over every offset that
    pairs one of them with a `TX` line the log holds. On a regular cadence the
    key-ups pair as often as the key-downs — five each here — and the tie used to go
    to the lower offset. That put this session at -4.058 s against a truth of
    +1.681, and rehear then withheld the *gaps*: it reported `live 6, ungated 5`
    with a `19.82 DATAACK sess 0xEA bare`, and every one of those six frames was one
    of our own ConReqs read back off our own muted receiver.

    The capture settles it without reference to the vote. The rig mutes its receiver
    while we transmit, so the level drops for the length of each keying and comes
    back between them, and the six troughs are the six intervals `+1.681` implies.
    """
    log = B.sessionlog.read(_W4UC_LOG)
    au = from_card(B._capture(_W4UC), B.SAMPLE_RATE)
    edges, quiet, level = B._key_edges(au)
    where = B.sessionlog.align(log, [a for a, _ in edges],
                               B._quieter(level, log.keyed))
    assert where is not None, "the alignment this session's own key-downs support"
    assert where.offset == pytest.approx(corpora.BESRA_REGULAR_CALL_OFFSET, abs=0.01), (
        f"{where.offset:+.3f} s — the key-up comb is one transmission out and its "
        f"offset is negative; {where}")
    assert where.corroborated == 6, (
        f"{where} — the capture's own level agrees with all six keyings here")

    # And the intervals themselves, against the level trace rather than against the
    # log that produced them. `PTT ON` to `PTT OFF`, which trails the `TX` line the
    # alignment anchors on by the rig's settle.
    muted = [(a, b) for a, b in
             ((a - where.offset, b - where.offset) for a, b in _ptt(_W4UC_LOG))]
    for (a, b), (want_a, want_b) in zip(muted, corpora.BESRA_REGULAR_CALL_MUTED):
        assert (a, b) == pytest.approx((want_a, want_b), abs=0.02), muted


@corpora.requires_besra_regular_call
def test_an_alignment_the_capture_cannot_settle_is_refused():
    """The other half: an instrument that inverts its own sign in silence is worse
    than one that declines.

    Told the capture is level everywhere — which is what a rig that does not mute
    its own receiver leaves behind, and this project has such recordings — nothing
    separates the key-down comb from the key-up comb and `align` raises instead of
    returning the lower of the two.
    """
    log = B.sessionlog.read(_W4UC_LOG)
    au = from_card(B._capture(_W4UC), B.SAMPLE_RATE)
    edges, _quiet, _level = B._key_edges(au)
    with pytest.raises(B.sessionlog.Ambiguous) as raised:
        B.sessionlog.align(log, [a for a, _ in edges], lambda _offset: 0)
    assert "Refusing rather than picking" in str(raised.value), str(raised.value)


@corpora.requires_besra_regular_call
def test_the_clocks_settle_what_a_level_capture_cannot():
    """And the third reading, which is the one the corpus is mostly made of.

    Declining costs a session. Eight of the thirty-four ARDOP calls of 2026-08-29
    and 08-30 are six identical ConReqs and nothing back over a rig that mutes
    shallowly, both combs score alike, and the tool refused every one — dropping the
    recording out of its own report rather than replaying it.

    Nothing about that is a tie for the clocks. The recorder stamps its first sample
    in UTC and the log stamps local wall time, and the two are wrong by a truncated
    second where the combs are a whole transmission apart. Told what the sidecar
    says, the same level-blind call that raises above returns the truth.
    """
    log = B.sessionlog.read(_W4UC_LOG)
    au = from_card(B._capture(_W4UC), B.SAMPLE_RATE)
    edges, _quiet, _level = B._key_edges(au)
    predicted = log.predicted(B._started(_W4UC))
    where = B.sessionlog.align(log, [a for a, _ in edges], lambda _offset: 0, predicted)
    assert where is not None, (
        f"the clocks put the capture's first sample at {predicted:+.3f} s and one of "
        f"the tied offsets is there")
    assert where.offset == pytest.approx(corpora.BESRA_REGULAR_CALL_OFFSET, abs=0.01), (
        f"{where.offset:+.3f} s — the key-up comb is one transmission out and the "
        f"clocks do not reach it")

    # The other half of the same information: an offset no clock predicts is not this
    # station's, whatever the key-downs voted. None rather than Ambiguous, because a
    # capture whose silences are not our keyings is not a capture with two readings.
    assert B.sessionlog.align(log, [a for a, _ in edges], lambda _offset: 0,
                              predicted + 60.0) is None


def test_the_zone_divides_out_of_the_two_clocks():
    """A log stamps local wall time, a sidecar stamps UTC, and neither says which
    zone. It does not have to be known: a capture opens seconds after the log's first
    line and every civil offset is a whole number of quarter hours."""
    log = B.sessionlog.SessionLog(path=Path("x.log"), session=None,
                                  zero=datetime(2026, 8, 30, 0, 36, 5, 89000))
    assert log.predicted(datetime(2026, 8, 30, 5, 36, 6)) == pytest.approx(0.911)
    assert log.predicted(datetime(2026, 8, 30, 0, 36, 6)) == pytest.approx(0.911)
    assert B.sessionlog.SessionLog(Path("x.log"), None).predicted(datetime.now()) is None


def _ptt(path: Path) -> list[tuple[float, float]]:
    """`PTT ON` to `PTT OFF` for each keying, on the log's own clock. `SessionLog`
    reads the mute from the `TX` line, which is where `RadioLink._transmit` sets
    `muted`; the rig's own receiver goes quiet at the key itself, one settle later."""
    stamped = [(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"), m.group(2))
               for line in path.read_text(errors="replace").splitlines()
               if (m := B.sessionlog.STAMPED.match(line))]
    base = stamped[0][0]
    out, on = [], None
    for when, text in stamped:
        if text.startswith("PTT ON"):
            on = (when - base).total_seconds()
        elif text.startswith("PTT OFF") and on is not None:
            out.append((on, (when - base).total_seconds()))
            on = None
    return out


def test_a_replay_with_no_log_to_read_announces_that_it_is_deaf():
    """The whole point of `deaf`. A replay of somebody else's QSO has no session log
    and no session id, and is the honest reading of that recording — but it is not a
    reading of what any station heard, and the report has to be the thing that says
    so. Silence here is what let a retracted finding through."""
    r = R.Rehearing(recording=Path("nobody.wav"), protocol="besra",
                    seconds=1.0, zero="the first sample", recorder=R.UPSTREAM,
                    deaf=B._deaf(None, None))
    assert r.deaf and "no session hint" in r.deaf
    assert "DEAF:" in R.report(r), R.report(r)
    assert "28 of the 51" in r.deaf, "the deaf reading no longer says what it costs"
