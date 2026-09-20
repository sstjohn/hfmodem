# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2026-08-26 KN4LQN arms, and the NAK budget that stopped the descent.

W9SSJ connected KN4LQN on 3592.0 kHz three times in eleven minutes — the first
ARDOP connects this station has ever had on 80 m — and every one ended the same
way. The gateway opened at `4PSK.500.100`, whose body carries 5.1 dB less average
power than its own frame-type header (measured on the reference's own transmit
vectors, and reproduced by `phy.modulator` to 0.2 dB), split over two carriers.
On that path the header decoded every time and the body never did.

Two NAKs is what `Gearshift_9` charges for one rung. Arm 01 paid them, the gateway
shifted down exactly as the reference says it should, and its first
`4FSK.500.100` — the best-received body of the whole slot, one rung off the floor
of the 2000 Hz ladder — arrived to a DISC instead of a third NAK.

`ARM01` is that stint verbatim from `~/ardop-night-01-kn4lqn.log`.
"""

from __future__ import annotations

from hfmodem.besra import crc
from hfmodem.besra.arq.session import (
    BREAK, CONACK, DISC, _UNDELIVERED_BUDGET, ArqSession, _dataack_for,
    _is_datanak, _type_of,
)
from hfmodem.besra.host import protocol as P

_SID = crc.session_id("W9SSJ", "KN4LQN")            # 0xac on the air that night

#: (seconds from the BREAK, frame type, quality) — the three data frames arm 01
#: reported, none of which read. The last is the rung the gateway shifted down to.
ARM01 = [
    (17.7, _type_of("4PSK.500.100.O"), 49),
    (36.7, _type_of("4PSK.500.100.O"), 51),
    (41.5, _type_of("4FSK.500.100.O"), 36),
]


class _Recorder:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.disconnects = 0

    def newstate(self, s): pass
    def connected(self, r, bw): pass
    def disconnected(self): self.disconnects += 1
    def data_received(self, kind, blob): pass
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
    point KN4LQN held the link and this end was IRS."""

    def __init__(self, bandwidth: int = 2000) -> None:
        self.wire = _Wire()
        self.obs = _Recorder()
        self.sess = ArqSession("W9SSJ", self.wire, self.obs, bandwidth=bandwidth)
        self.sess.tick(0.0)
        self.sess.connect("KN4LQN")
        self.sess.on_receive(CONACK[bandwidth], b"\x18\x18\x18", _SID, True, 83)
        self.sess.on_receive(_dataack_for(80), b"", _SID, True)
        self.sess.on_receive(BREAK, b"", _SID, True)
        assert self.sess.state == P.ArdopState.IRS
        self.wire.sent.clear()
        self.obs.statuses.clear()

    def replay(self, arrivals) -> None:
        for t, ft, q in arrivals:
            self.sess.tick(t)
            self.sess.on_receive(ft, b"", _SID, False, q, header_only=True)

    def naks(self) -> list[int]:
        return [ft for ft in self.wire.sent if _is_datanak(ft)]


def test_the_rung_the_gateway_shifted_down_to_is_answered():
    """The frame the arms hung up on. Two NAKs bought the shift the NAKs are for,
    and the budget had nothing left to pay for the next one — so the descent
    stopped one rung above the floor with the link ended by this end."""
    link = _Link()
    link.replay(ARM01)

    assert DISC not in link.wire.sent, "the descent was ended on the rung it bought"
    assert link.obs.disconnects == 0
    assert len(link.naks()) == 3, "the shifted-down frame drew no NAK of its own"


def test_a_rung_that_fails_twice_more_still_closes_the_stint():
    """The budget renews per rung, not per frame: the peer has to move for it to
    start again, and a rung that goes on failing spends its own budget and closes
    a stint that has delivered nothing.

    Two frames of ARM01's opening rung, then the rung it shifted down to, held
    there until it runs out. What is being pinned is that the shift bought exactly
    one budget: the frames the first rung spent are not added to the second's
    allowance, and the second rung's own count is what ends the stint."""
    link = _Link()
    held = ARM01[-1][1]
    link.replay(ARM01 + [(41.5 + 4.9 * i, held, 36)
                         for i in range(1, _UNDELIVERED_BUDGET + 1)])

    assert DISC in link.wire.sent
    assert len(link.naks()) == 2 + _UNDELIVERED_BUDGET, \
        "each rung is worth one budget and no more"
    closed = [s for s in link.obs.statuses if "UNREADABLE" in s]
    assert len(closed) == 1 and "4FSK.500.100" in closed[0], closed


def test_a_rung_already_tried_does_not_buy_a_second_budget():
    """What keeps the descent finite. A peer oscillating between two rungs it has
    already failed on is not descending: each buys one restart and no more, so the
    count runs out one frame past the budget however the two are interleaved —
    the first arrival on each of the two rungs restarts it, and every arrival
    after that is spending it."""
    link = _Link()
    a, b = ARM01[0][1], ARM01[-1][1]
    link.replay([(float(t), a if i % 2 == 0 else b, 40)
                 for i, t in enumerate(range(_UNDELIVERED_BUDGET + 2))])

    assert DISC in link.wire.sent
    assert len(link.naks()) == _UNDELIVERED_BUDGET + 1, \
        "an oscillation renewed the budget it had already spent"


def test_a_header_only_control_names_no_rung():
    """Only a data frame is a mode the peer chose. A bodyless control read off ten
    tones is a guess at a type, and letting one renew the budget would hand an
    unbounded NAK train to whatever the acquisition search happened to mint.

    So the budget is spent to its last frame on one rung, and the header-only
    BREAK that arrives next is the frame that closes the stint. Had it named a
    rung it would have restarted the count instead, and the arrival after it would
    have been answered rather than transmitted at a link that had gone."""
    link = _Link()
    rung = ARM01[0][1]
    link.replay([(4.9 * i, rung, 47) for i in range(_UNDELIVERED_BUDGET)])
    assert DISC not in link.wire.sent, "the rung's own budget closed early"

    link.sess.on_receive(BREAK, b"", _SID, True, header_only=True)
    link.replay([(50.0, rung, 47)])

    assert DISC in link.wire.sent
    assert len(link.naks()) == _UNDELIVERED_BUDGET
