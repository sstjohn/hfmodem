# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2026-08-26 W6IDS stint, and the budget that ended it four frames early.

W9SSJ connected W6IDS on 7061.5 kHz at BW500, took `WWTD6QMC61TV` — the first
message this station has ever received off the air — and had accepted two more by
name when it disconnected itself mid-message. Three consecutive frames failed to
read, the NAK budget expired, and the link went down with the gateway still
transmitting and 120 minutes of our allowance left.

The premise the budget disconnected on was that two NAKs buy a more robust mode,
so a third failure is this receiver's fault. RMS Trimode 1.4.2.2 did not
downshift: it sent 4PSK.200.100 before both NAKs and after both, with two rungs of
the 500 Hz ladder unused beneath it. The receiver it was blamed on had read 46 of
the session's 51 frames.

`ARRIVALS` is that stint verbatim from the live log — every frame this end
reported between conceding W6IDS's BREAK at 08:41:06 and the disconnect at
08:43:49, with the qualities it graded them at. What the log does not record is
the payload bytes, so each frame here carries a distinct stand-in: the arrival
order, types, timings and gradings are the recording's, and the delivery they
drive is what the budget is now graded against.

The stint that has delivered *nothing* kept a count of its own, and 2026-08-29
showed that one was set on the same refuted premise. `ARM01_0829` and
`ARM04_0829` at the foot of this file are the two greetings it closed.
"""

from __future__ import annotations

import re

from hfmodem.besra import crc
from hfmodem.besra.arq.session import (
    BREAK, CONACK, DISC, _UNDELIVERED_BUDGET, ArqSession, _dataack_for,
    _datanak_for, _is_datanak, _type_of,
)
from hfmodem.besra.host import protocol as P

_SID = crc.session_id("W9SSJ", "W6IDS")             # 0x0d on the air that day

ARRIVALS = """\
2026-08-26 08:41:11,862 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=73
2026-08-26 08:41:17,970 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=74
2026-08-26 08:41:24,081 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=69
2026-08-26 08:41:30,480 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=65
2026-08-26 08:41:35,421 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=72
2026-08-26 08:41:40,288 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=74
2026-08-26 08:41:45,206 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=72
2026-08-26 08:41:50,097 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=72
2026-08-26 08:41:55,682 INFO RX 4PSK.200.100.E sess=0x0d ok=False HEADER-ONLY q=63
2026-08-26 08:42:09,821 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=64
2026-08-26 08:42:14,113 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=70
2026-08-26 08:42:19,289 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=80
2026-08-26 08:42:24,178 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=73
2026-08-26 08:42:30,347 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=72
2026-08-26 08:42:35,542 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=80
2026-08-26 08:42:40,646 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=72
2026-08-26 08:42:46,466 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=55
2026-08-26 08:42:54,202 INFO RX 4PSK.200.100.O sess=0x05 ok=True q=66
2026-08-26 08:43:00,233 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=80
2026-08-26 08:43:10,219 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=72
2026-08-26 08:43:15,094 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=74
2026-08-26 08:43:20,227 INFO RX 4PSK.200.100.O sess=0x0d ok=True q=72
2026-08-26 08:43:25,224 INFO RX 4PSK.200.100.E sess=0x0d ok=True q=67
2026-08-26 08:43:30,833 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=60
2026-08-26 08:43:38,639 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=55
2026-08-26 08:43:49,248 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=63
"""

_LINE = re.compile(
    r"\d{4}-\d\d-\d\d (\d\d):(\d\d):(\d\d),(\d{3}) INFO RX (\S+) sess=(\w+) "
    r"ok=(\w+)( HEADER-ONLY)? q=(\d+)")

#: The consecutive header-only reads the transcript ends on — a frame arrived and
#: nothing but its type could be read off it — and the run the budget expired on.
FATAL = 3


def _arrivals(transcript: str = ARRIVALS,
              ) -> list[tuple[float, int, int, bool, int, bool]]:
    out = []
    for line in transcript.strip().splitlines():
        m = _LINE.match(line)
        assert m, line
        h, mi, s, ms, name, sid, ok, header_only, q = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000
        out.append((t, _type_of(name), int(sid, 16), ok == "True", int(q),
                    header_only is not None))
    base = out[0][0]
    return [(t - base, *rest) for t, *rest in out]


def _undelivered(n: int) -> list[tuple[float, int, int, bool, int, bool]]:
    """`n` copies of the frame the 08-26 transcript ends on, at the cadence the
    two before it arrived on. A stint that reads none of them delivers nothing by
    construction, which is the case the undelivered budget bounds."""
    tail = _arrivals()[-1]
    gap = tail[0] - _arrivals()[-2][0]
    return [(gap * i, *tail[1:]) for i in range(n)]


class _Recorder:
    def __init__(self) -> None:
        self.received = bytearray()
        self.statuses: list[str] = []
        self.disconnects = 0

    def newstate(self, s): pass
    def connected(self, r, bw): pass
    def disconnected(self): self.disconnects += 1
    def data_received(self, kind, blob): self.received += blob
    def buffer(self, n): pass
    def pending(self, cancel=False): pass
    def target(self, call): pass
    def status(self, t): self.statuses.append(t)


class _Wire:
    def __init__(self) -> None:
        self.sent: list[int] = []

    def send(self, ft: int, payload: bytes, sid: int) -> float:
        self.sent.append(ft)
        return 0.0


class _Link:
    """One session in W9SSJ's seat, taken through the recorded handshake to the
    point W6IDS held the link and this end was IRS."""

    def __init__(self) -> None:
        self.wire = _Wire()
        self.obs = _Recorder()
        self.sess = ArqSession("W9SSJ", self.wire, self.obs, bandwidth=500)
        self.sess.tick(0.0)
        self.sess.connect("W6IDS")
        self.sess.on_receive(CONACK[500], b"\x18\x18\x18", _SID, True, 86)
        self.sess.on_receive(_dataack_for(90), b"", _SID, True)
        self.sess.on_receive(BREAK, b"", _SID, True)
        assert self.sess.state == P.ArdopState.IRS
        self.now = 0.0
        self.nth = 0
        self.wire.sent.clear()
        self.obs.statuses.clear()

    def replay(self, arrivals) -> None:
        """Advance the clock by the recorded gaps between arrivals, from wherever
        this end has got to — a fragment of the transcript keeps its own timing
        without inheriting the whole session's elapsed time."""
        for i, (t, ft, sid, ok, q, header_only) in enumerate(arrivals):
            self.now += t - (arrivals[i - 1][0] if i else t)
            self.nth += 1
            self.sess.tick(self.now)
            self.sess.on_receive(ft, b"payload %d" % self.nth, sid, ok, q,
                                 header_only=header_only)

    def fade_on(self, arrival, gap: float, until: float) -> int:
        """Keep one arrival coming at its own cadence until `until`, and say how
        many of them the session got.

        Every tick it takes lands strictly before that mark, so the caller can put
        the mark on either side of the session deadline and read the answer off
        `disconnects`. Nothing is fed to a session the tick has just torn down: the
        peer transmits into a link that is gone, and this end is not there to
        answer it."""
        _, ft, sid, ok, q, header_only = arrival
        fed = 0
        while not self.obs.disconnects and self.now + gap < until:
            self.now += gap
            self.sess.tick(self.now)
            if self.obs.disconnects:
                break
            self.nth += 1
            fed += 1
            self.sess.on_receive(ft, b"payload %d" % self.nth, sid, ok, q,
                                 header_only=header_only)
        return fed

    def turnover(self) -> None:
        """Take the link and hand it straight back: the peer's next frame opens a
        stint that has read nothing of its own."""
        self.sess.queue_data(b"a reply")
        self.sess.on_receive(_dataack_for(90), b"", _SID, True)
        self.sess.on_receive(BREAK, b"", _SID, True)


def test_the_recorded_session_id_is_ours():
    """The transcript's provenance in one line: the id both ends derived from the
    callsigns of that connect is the one every frame in it is stamped with."""
    assert _SID == 0x0d


def test_the_stint_survives_the_three_frames_that_ended_it():
    """Fourteen frames had reached the application over this stint when the fade
    arrived. A receiver that has been reading this peer for two and a half minutes
    is not one that cannot read it, so the NAKs go on and the peer keeps its
    transmitter — and the session deadline, which an unreadable frame never
    postpones, is what ends the stint if the fade does not lift."""
    link = _Link()
    link.replay(_arrivals())

    assert DISC not in link.wire.sent, \
        "the link was ended on a fade it had been delivering through"
    assert link.obs.disconnects == 0
    assert link.obs.received, "nothing was delivered, so the stint proves nothing"

    naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
    assert naks[-FATAL:] == [_datanak_for(q) for q in (60, 55, 63)], \
        "the last three frames were not each NAKed at their own reading"


def test_a_fade_that_does_not_lift_ends_on_the_session_deadline():
    """The other half of the stint above, and the whole of what now bounds it. The
    count stopped being a bound the moment delivery started deciding, so the 90 s
    progress clock is the only thing left holding the NAK train — and it holds it
    from both sides: still asking one frame short of the deadline, gone a few
    frames past it, and never on the count in between.

    An unreadable frame is the one arrival that postpones nothing (`_unreadable`
    NAKs without `_progress`), which is what makes the clock reach the end of a
    fade that does not lift. It is the reference's bound too: `ARQ.c` NAKs an
    undecodable data frame without limit and leaves `dttTimeoutTrip` alone doing
    it. Take the clock out and the fade below runs to the cap with the link still
    up, which is this assertion failing rather than a modem NAKing forever.
    """
    link = _Link()
    arrivals = _arrivals()
    link.replay(arrivals)
    assert link.obs.disconnects == 0, "the stint was over before the clock could run"

    ours = [(t, ok) for t, _, sid, ok, *_ in arrivals if sid == _SID]
    deadline = max(t for t, ok in ours if ok) + link.sess._timeout_s
    gap = arrivals[-1][0] - arrivals[-2][0]

    fed = link.fade_on(arrivals[-1], gap, deadline)
    assert link.obs.disconnects == 0, \
        "the link went down early: something other than the clock ended it"
    assert not [s for s in link.obs.statuses if "DISCONNECTING" in s], \
        "the count closed a stint that had delivered"

    fed += link.fade_on(arrivals[-1], gap, deadline + 4 * gap)
    assert link.obs.disconnects == 1, \
        "the fade never lifted and the link was still up past the session deadline"
    assert DISC in link.wire.sent, "the peer was left transmitting to nobody"
    assert any("ARQ Timeout" in s for s in link.obs.statuses), link.obs.statuses

    naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
    unread = sum(1 for _, ok in ours if not ok)
    assert len(naks) == unread + fed, \
        "the NAK train stopped somewhere short of the deadline"


def test_the_status_no_longer_blames_this_receiver():
    """The line that went into the log said the frames were unreadable after two
    NAKs asking for a more robust mode, and disconnected on it. The claim behind
    it — that the fault was here — was measurably false, and a status line that
    misattributes a fade poisons the record. It names the condition it actually
    tested instead, and says it once rather than per frame."""
    link = _Link()
    link.replay(_arrivals())

    held = [s for s in link.obs.statuses if "UNREADABLE" in s]
    assert len(held) == 1, held
    assert "THIS STINT HAS DELIVERED, HOLDING THE LINK" in held[0]
    assert "DISCONNECTING" not in held[0]


def test_a_stint_that_has_delivered_nothing_still_closes():
    """The bound the budget was written for, and it is a real one: an unbounded
    NAK train at a peer this end has never once read spends a gateway's
    transmitter as well as ours. Nothing here has delivered, so the stint closes
    on its own count — at `_UNDELIVERED_BUDGET`, not at the two the delivering
    stint above is bounded by."""
    link = _Link()
    link.replay(_undelivered(_UNDELIVERED_BUDGET + 1))

    assert DISC in link.wire.sent
    naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
    assert len(naks) == _UNDELIVERED_BUDGET, f"{len(naks)} NAKs before the close"
    closed = [s for s in link.obs.statuses if "UNREADABLE" in s]
    assert len(closed) == 1 and "NOTHING DELIVERED THIS STINT" in closed[0]


def test_the_budget_starts_each_stint_at_zero():
    """A turnover is a new stint for the budget as it is for type alternation: the
    count is graded against what this stint has delivered, so carrying it across
    would let a fade from the last one spend the new one's NAKs before it has had
    a frame to read."""
    link = _Link()
    link.replay(_arrivals()[:-1])                      # two unreadable, budget spent
    link.turnover()
    link.wire.sent.clear()

    link.replay(_undelivered(_UNDELIVERED_BUDGET + 1))

    assert DISC in link.wire.sent, "a fresh stint that reads nothing must still close"
    naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
    assert len(naks) == _UNDELIVERED_BUDGET, \
        f"{len(naks)} NAKs — the new stint did not get the full budget"


#: The two greetings the budget of two closed, 2026-08-29, verbatim from
#: `working/onair-0829-1349/` and `working/onair-0829-1400/`. Each is every frame
#: the arm reported between conceding W6IDS's BREAK and this end's own DISC. Both
#: arms connected cleanly, neither reached the banner, and both ended 41 s in.
ARM01_0829 = """\
2026-08-29 13:50:59,459 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=51
2026-08-29 13:51:06,198 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=52
2026-08-29 13:51:10,202 INFO RX 4PSK.200.100.O sess=0x0d ok=False HEADER-ONLY q=54
"""

ARM04_0829 = """\
2026-08-29 14:00:50,524 INFO RX 4PSK.200.100.E sess=0x0d ok=False HEADER-ONLY q=54
2026-08-29 14:01:05,022 INFO RX 4PSK.200.100.E sess=0x0d ok=False HEADER-ONLY q=53
2026-08-29 14:01:17,443 INFO RX 4PSK.200.100.E sess=0x0d ok=False HEADER-ONLY q=52
"""


def test_the_greeting_is_not_cut_at_two():
    """Arms 1 and 4 of the 2026-08-29 daylight slot, and the reason the undelivered
    budget is not the delivered one.

    The greeting is by construction the one stint that has delivered nothing, so
    the bound written for a peer this end has never read was also, every session,
    the gate the link had to pass to start. W6IDS repeated its greeting on one
    rung and never descended, so the per-rung renewal had nothing to fire on, and
    three unlucky frames ended a session before it could have one.

    The peer was there. The operator heard the far end still transmitting on both
    of these, and the channel sense read `+4.8 dB tone at 1512 Hz -> OCCUPIED`
    60 s after arm 1's DISC — our own peer, on our own centre, after we had gone.
    On arm 3, twenty minutes either side of these, the copy after a failure read
    and the stint climbed q 59 -> 84 into a completed B2F login.
    """
    for transcript in (ARM01_0829, ARM04_0829):
        link = _Link()
        link.replay(_arrivals(transcript))

        assert DISC not in link.wire.sent, "the greeting was cut at two again"
        assert link.obs.disconnects == 0
        assert not link.obs.received, "the stint delivered, so it proves nothing"
        naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
        assert naks == [_datanak_for(q) for _, _, _, _, q, _ in _arrivals(transcript)], \
            "each frame is asked for again at its own reading"


def test_a_greeting_the_peer_abandons_still_ends():
    """What the wider budget costs, measured on the cadence that spent it.

    W6IDS's greeting frames arrived 12.4 s apart on arm 4, so a peer that never
    became readable runs the count out around 80 s past the turnover instead of
    41 s. That is the whole cost: a link nobody is filling holds a shared channel
    for the rest of the session deadline.

    And it is bounded either way. An unreadable frame is the one arrival that
    postpones nothing — `_unreadable` NAKs without `_progress` — so a cadence slow
    enough to outlast the count reaches the 90 s deadline instead, and the link
    ends there. Whichever fires first, an abandoned greeting is gone within
    `_timeout_s` of the turnover.
    """
    arrivals = _arrivals(ARM04_0829)
    gap = arrivals[-1][0] - arrivals[-2][0]

    link = _Link()
    link.replay(arrivals + [(arrivals[-1][0] + gap * i, *arrivals[-1][1:])
                            for i in range(1, _UNDELIVERED_BUDGET + 2 - len(arrivals))])

    assert DISC in link.wire.sent
    assert link.now <= link.sess._timeout_s, \
        "the count outlasted the deadline it was supposed to land inside"
    naks = [ft for ft in link.wire.sent if _is_datanak(ft)]
    assert len(naks) == _UNDELIVERED_BUDGET, f"{len(naks)} NAKs before the close"
    assert not [t for t in link.obs.statuses if "ARQ Timeout" in t]

    slow = _Link()
    step = link.sess._timeout_s / 4
    slow.replay([(step * i, *arrivals[-1][1:]) for i in range(_UNDELIVERED_BUDGET)])

    assert slow.obs.disconnects == 1, "a cadence the count cannot reach held the link"
    assert any("ARQ Timeout" in t for t in slow.obs.statuses), slow.obs.statuses
    assert len([ft for ft in slow.wire.sent if _is_datanak(ft)]) < _UNDELIVERED_BUDGET, \
        "the count ran out first, so the clock was never what ended it"
