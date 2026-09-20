# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What an ungated hit IS, on a shrike window — the column that was withdrawn twice.

``tools/rehear``'s shrike adapter used to print `discarded N`: everything a free
sweep of the capture found that a `RollingRx`-only replay did not. That replay is
one of the session's four readers and the one that decodes nothing below
`live.MIN_DECODE_S`, so on a 0.24 s calling window it is silent whatever arrived —
and the adapter's own caveat said so, calling the number "an upper bound on what
was lost and not the loss".

It was read as a receive-gating lead twice in one week and withdrawn twice. On
`onair-0902-2342` it printed `live 6, ungated 20, discarded 14` against a session
log naming 52 answered cycles: four of the fourteen are in that log at the same
second at zero bit errors, and the other ten sit 0.18-0.68 s inside the peer's own
100 Bd changeover packet, in a phase where the peer held the sending role and an
ISS keys no control signals at all. On `onair-0903-1107` it printed `discarded 4
cs CS1` for four CS1 the live log names itself.

Three things had to hold for that to stop, and they are what this file pins. The
arm's log has to be found from a capture it never names and read for the instants
its grid aimed at. The anchored point read — the session's own method, at the
session's own number — has to run on a window and reproduce what the station
reported there. And each ungated hit has to be sorted against those two and
against the geometry of the peer's own transmission, rather than counted.

The first two are rendered, so they hold in a checkout with no recordings in it.
The third is asserted on both arms where they are, and its rules are pinned
directly: the ungated sweep is a point read off the whole window's envelope and
does not fire on a rendered burst in an otherwise empty second, so a rendered
end-to-end that reached it would be a test of the renderer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import pactor1, session, spec
from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora

R = corpora.harness("rehear")
S = corpora.harness("rehear.shrike")

FS = S.FS

#: Where the peer's answer sits after our carrier drops, on both arms below:
#: 92-102 ms of measured turnaround. One number for both windows because the grid
#: has one — `CS due` is a single instant a cycle.
_D_S = 0.100

#: Far enough into the changeover packet that no reading of the answer slot
#: reaches it, and inside `_BREAKIN_S` of the head. The real ones run 0.081-0.579 s
#: past their own head.
_INSIDE_S = 0.45


def _capture(path: Path, audio: np.ndarray, end: int) -> None:
    """One window and the sidecar that puts it on the session's stream clock."""
    session.write_wav(str(path), audio)
    path.with_suffix(".json").write_text(json.dumps({
        "samples": len(audio), "seconds": len(audio) / FS, "peak": 0.5,
        "rms": 0.2, "railed_pct": 0.0, "normalised_on_write": True,
        "lost_samples": 0, "xruns": 0, "grid_loss_seen": False,
        "end_stream_sample": end}))


def _put(audio: np.ndarray, burst: np.ndarray, t: float) -> None:
    at = int(t * FS)
    audio[at:at + len(burst)] = burst[:len(audio) - at]


def _window(seed: int, *bursts) -> np.ndarray:
    audio = np.random.default_rng(seed).normal(
        0, 0.02, round(1.08 * FS)).astype(np.float32)
    for burst, t in bursts:
        _put(audio, burst, t)
    return audio


@pytest.fixture
def arm(tmp_path: Path) -> Path:
    """A two-cycle arm: an acknowledged cycle, then a changeover the peer took.

    `hold_01` is an ordinary held cycle, the peer's CS2 at the answer slot.
    `hold_02` is the shape the whole withdrawal turns on: a CS3 break-in head at
    the same slot with the peer's own 100 Bd packet behind it. The log is written
    the way the runner writes one — it names `stream.wav` and no window, which is
    the case that used to hand this adapter no log at all.
    """
    caps = tmp_path / "onair-fixture"
    caps.mkdir()
    slot = round(_D_S * FS)

    one = _window(1, (pactor1.control_signal(pactor1.CS_ACK_B, repeats=3), _D_S))
    _capture(caps / "hold_01.wav", one, 100_000)
    two = _window(2, (pactor1.breakin_signal(b"RMS Tri", baud=100, lead_s=0.0,
                                             tail_s=0.0), _D_S))
    _capture(caps / "hold_02.wav", two, 200_000)

    aimed = [100_000 - len(one) + slot, 200_000 - len(two) + slot]
    (tmp_path / "arm.log").write_text("\n".join([
        f"session stream: {caps}/stream.wav -- 4.2 s, on the sample clock every "
        f"capture beside it is indexed on",
        f"    [grid] hold 1 slot 2, CS due @ {aimed[0]}; captured 1.080 s, "
        f"bridged 30 ms, off-grid +0.1 ms",
        f"    HOLD RX  {aimed[0] / FS:6.2f}  cs       CS2/at anchor "
        f"(0 bit errors, PACTOR-1, shift normal)",
        f"    [grid] hold 2 slot 3, CS due @ {aimed[1]}; captured 1.080 s, "
        f"bridged 30 ms, off-grid +0.1 ms",
        f"    HOLD RX  {aimed[1] / FS:6.2f}  cs       CS3/break-in head "
        f"(0 bit errors, PACTOR-1, shift normal)",
    ]))
    return tmp_path


def test_an_arms_log_is_found_from_a_window_it_never_names(arm):
    """140-odd windows a session and not one of them written down.

    `sessionlog.find` matched on the recording's own name, so with `--log` given
    it handed the shrike adapter `None` every time — and this is the adapter that
    most needs one. The arm names `stream.wav`, which every window beside it is
    indexed on, and that is the join.
    """
    wav = arm / "onair-fixture" / "hold_01.wav"
    log = R.sessionlog.find(wav, [arm / "arm.log"])
    assert log is not None, "the arm's own log no longer claims its own windows"

    grid = R.sessionlog.shrike_grid(log.path, FS)
    assert len(grid.aimed) == 2, grid.aimed
    assert [w.what for w in grid.words] == ["cs CS2", "cs CS3"], grid.words


def test_the_anchored_read_reproduces_what_the_station_reported(arm):
    """The session's own reader, at the session's own instant, on both windows.

    `_SessionRx._p1_cs` is called rather than restated: it is `p1rx.cs_anchored`
    with `cs_head` behind it and the CS3 refusal in front, and a replay carrying
    its own copy of that would be a second implementation of the one thing this
    tool exists to hold the first to.
    """
    got = {p.name: R.rehear(p, [arm / "arm.log"])
           for p in R.recordings([arm / "onair-fixture"])}

    one = got["hold_01.wav"]
    aimed, = one.aimed
    assert aimed.what == "cs CS2" and abs(aimed.at - _D_S) < spec.P1_CS_S, aimed
    assert aimed in one.live, "an anchored read is a live read"
    assert aimed not in one.only_live(), (
        "an aimed read is not a sweep and is not held to one")
    assert one.verdicts == {} and one.deaf == "", (
        "the window was placed and held nothing ungated to read, which is not the "
        "same statement as not having been placed")

    two = got["hold_02.wav"]
    head, = two.aimed
    assert head.what == "cs CS3" and "break-in head" in head.detail, head
    assert abs(head.at - _D_S) < spec.P1_CS_S, head
    assert not two.discarded()


def test_a_window_with_no_log_is_deaf_rather_than_lossy(arm):
    """The other half, and the one that costs nothing to get wrong quietly.

    The grid's answer instant is a property of when WE keyed. Without the arm's
    log there is nothing to aim the point read at, so the replay is deafer than
    the station by the reader that delivered most of its decodes — and the report
    says so instead of printing a column of losses.
    """
    r = S.rehear(arm / "onair-fixture" / "hold_02.wav")
    assert r.verdicts is None, r.verdicts
    assert r.live == [], "nothing aims the anchored read without a grid"
    assert "no session log names this arm" in r.deaf, r.deaf
    assert "DEAF:" in R.report(r)
    assert "discarded" in R.totals([r]).splitlines()[0], R.totals([r])


# -- the three readings, and the order they are taken in ---------------------

def _hit(at: float, what: str, detail: str = "") -> "R.Heard":
    return R.Heard(at, what, detail)


_HEAD = _hit(0.104, "cs CS3", "CS3/break-in  (0 bit errors, PACTOR-1)")


def test_a_hit_the_arms_log_names_is_seen():
    """The `onair-0903-1107` shape: four CS1 at the answer slot, named by the log
    that printed them, counted as four losses because a rolling replay of a 0.24 s
    window decodes nothing by construction."""
    hit = _hit(0.094, "cs CS1", "CS1/ack  (0 bit errors, PACTOR-1)")
    said = [R.sessionlog.Word(0.09, "cs CS1", "CS1/codeword search")]
    v, = S._verdicts([hit], [], [], said, [0.094]).values()
    assert v.what == R.SEEN, v
    assert "the arm's log names it there too" in v.why, v.why
    assert "+0 ms from the answer slot" in v.why, v.why


def test_a_codeword_inside_the_peers_packet_is_a_false_accept():
    """The `onair-0902-2342` shape, and the ten that were the whole of the lead.

    A station that has taken the link sends data, not control signals. So a
    twelve-bit accept inside a changeover packet the same window holds whole is
    the reader finding a codeword in somebody's bits, and it is named that way
    rather than counted as something our receiver threw away.
    """
    inside = _hit(_INSIDE_S, "cs CS1", "CS1/ack  (acquired, PACTOR-1)")
    got = S._verdicts([_HEAD, inside], [], [_HEAD], [], [0.104])
    assert got[_HEAD].what == R.SEEN, got[_HEAD]
    assert got[inside].what == R.IN_PACKET, got[inside]
    assert "into the peer's own transmission" in got[inside].why
    assert "+346 ms from the answer slot at 0.104 s" in got[inside].why, \
        got[inside].why


def test_the_peers_packet_outranks_a_replay_that_reaches_the_same_hit():
    """`hold_40` of `onair-0902-2342`, which is why the order is what it is.

    The rolling arm here pushes every slice where a held cycle holds them, so it
    decodes strictly more than the session's did — and it takes an accept 147 ms
    inside the peer's changeover packet. Read as "the replay found it too" that is
    a hit our receiver heard; read against the packet it sits in, it is twelve
    bits of the peer's data. The second is the true one.
    """
    inside = _hit(0.255, "cs CS1", "CS1/ack  (0 bit errors, PACTOR-1)")
    got = S._verdicts([_HEAD, inside], [inside], [_HEAD], [], [0.104])
    assert got[inside].what == R.IN_PACKET, got[inside]


def test_a_hit_only_the_rolling_replay_reaches_is_not_seen():
    """And the floor under all of it. The rolling arm is over-permissive on
    purpose, so an accept of its own is not evidence the station heard anything;
    the hit is UNSEEN and the reason says which replay reached it."""
    lone = _hit(0.62, "cs CS1", "CS1/ack  (acquired, PACTOR-1)")
    v, = S._verdicts([lone], [lone], [], [], [0.104]).values()
    assert v.what == R.UNSEEN, v
    assert "not in the arm's log" in v.why and "rolling replay" in v.why, v.why

    quiet, = S._verdicts([lone], [], [], [], []).values()
    assert quiet.what == R.UNSEEN
    assert "against no answer slot this window holds" in quiet.why, quiet.why


def test_the_summary_line_names_the_three_readings():
    """The line the reports quote. Its shape is what a report can be held to, so
    it is asserted rather than described in a docstring somewhere."""
    hits = [_hit(0.10, "cs CS1"), _hit(0.45, "cs CS4"), _hit(0.62, "cs CS1")]
    one = R.Rehearing(recording=Path("hold_01.wav"), protocol="shrike", seconds=1.0,
                      zero="x", recorder=R.DOWNSTREAM, live=hits[:1],
                      ungated=hits, aimed=set(hits[:1]),
                      verdicts={hits[0]: R.Verdict(R.SEEN, "a"),
                                hits[1]: R.Verdict(R.IN_PACKET, "b"),
                                hits[2]: R.Verdict(R.UNSEEN, "c")})
    lines = R.totals([one]).splitlines()
    assert lines[0] == ("shrike: 1 recordings, 0.00 h, live 1, ungated 3, "
                        "seen 1, in a packet 1, unseen 1, live-only 0"), lines[0]
    assert lines[-1] == "    unseen    1  cs CS1", lines[-1]

    # And the other shape, unchanged: an adapter that classified nothing still
    # reports the difference the way besra and kestrel have always reported it.
    plain = R.Rehearing(recording=Path("x.wav"), protocol="besra", seconds=1.0,
                        zero="x", recorder=R.UPSTREAM, ungated=hits[:1])
    assert R.totals([plain]).splitlines()[0].endswith(
        "live 0, ungated 1, discarded 1, live-only 0")


# -- the two arms the column was withdrawn on --------------------------------

#: `working/onair-0902-2342/pactor-night-09-ve1yz-mail.log`, and the fourteen
#: windows it printed `discarded 14` for. The four before the peer's changeover at
#: 63.21 s are in that log at the same second at zero bit errors; the ten after it
#: sit inside the 100 Bd packet the peer sent every cycle from there to teardown.
_VE1YZ = evidence.CAPTURES / "onair-0902-2342"
_VE1YZ_LOG = (evidence.WORKING / "onair-0902-2342"
              / "pactor-night-09-ve1yz-mail.log")
_VE1YZ_SEEN = ("rx_35", "hold_02", "hold_03", "hold_09")
_VE1YZ_IN_PACKET = ("hold_29", "hold_30", "hold_34", "hold_35", "hold_45",
                    "hold_70", "hold_73", "hold_92", "hold_96", "hold_97")

#: `working/onair-0903-1107/pactor-day-11-ws8eoc-clean.log`, and the four windows
#: it printed `discarded 4 cs CS1` for. The log names every one of them.
_WS8EOC = evidence.CAPTURES / "onair-0903-1107"
_WS8EOC_LOG = (evidence.WORKING / "onair-0903-1107"
               / "pactor-day-11-ws8eoc-clean.log")
_WS8EOC_SEEN = ("hold_01", "rx_04", "rx_05", "rx_06")

_ve1yz = pytest.mark.skipif(not (_VE1YZ.is_dir() and _VE1YZ_LOG.exists()),
                            reason="the 2026-09-02 VE1YZ arm is not in this tree")
_ws8eoc = pytest.mark.skipif(not (_WS8EOC.is_dir() and _WS8EOC_LOG.exists()),
                             reason="the 2026-09-03 WS8EOC arm is not in this tree")


def _verdicts(caps: Path, log: Path, stem: str) -> list["R.Verdict"]:
    r = R.rehear(caps / f"{stem}.wav", [log])
    assert r.verdicts, f"{stem}: nothing ungated to read, so nothing was measured"
    return [v for _h, v in sorted(r.verdicts.items())]


@_ve1yz
@pytest.mark.parametrize("stem", _VE1YZ_SEEN)
def test_the_ve1yz_hits_before_the_changeover_are_in_the_arms_own_log(stem):
    v, = _verdicts(_VE1YZ, _VE1YZ_LOG, stem)
    assert v.what == R.SEEN, f"{stem}: {v}"
    assert "the arm's log names it there too" in v.why, v.why


@_ve1yz
@pytest.mark.parametrize("stem", _VE1YZ_IN_PACKET)
def test_the_ve1yz_hits_after_the_changeover_are_inside_the_peers_packet(stem):
    got = _verdicts(_VE1YZ, _VE1YZ_LOG, stem)
    assert R.IN_PACKET in [v.what for v in got], f"{stem}: {got}"
    assert R.UNSEEN not in [v.what for v in got], f"{stem}: {got}"


@_ws8eoc
@pytest.mark.parametrize("stem", _WS8EOC_SEEN)
def test_the_ws8eoc_hits_are_the_ones_its_own_log_printed(stem):
    v, = _verdicts(_WS8EOC, _WS8EOC_LOG, stem)
    assert v.what == R.SEEN, f"{stem}: {v}"
    assert "the arm's log names it there too" in v.why, v.why
