# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Two phases of one link, at two RF powers, written over the arm's own CAT.

The FT-891's power control works against the RF output and the soundcard drive
does not stand in for it: on 2026-09-13 a constant-envelope PACTOR-1 packet at
0.52 drive still left above 50 W while the two-tone PACTOR-3 entry, 3 dB of
crest behind it, was the burst the ALC held down. So an arm that wants 15 W of
PACTOR-1 and 60 W of PACTOR-3 has to move RFPOWER itself, at the link's protocol
boundaries, and `--p1-drive` -- which shapes the audio and is unchanged -- cannot
be asked to do it.

WHAT THE CAT WRITE MUST NEVER COST IS A SLOT. Every arm this station flies is a
measurement of where its carriers land, and the one window nothing may be put in
front of is the few milliseconds between a cycle's last read and its key
(`onair.TX_ADMIT_RESERVE_S`, `_LiveInput.key_notice`). So the writes ride the
transitions where they are READ: the grant, which the peer keys a turnaround
ahead of the slot the entry packet is owed, and the two places a link comes back
to PACTOR-1 with nothing keyed behind them. The bench scene below measures that
distance on the session's own clock rather than asserting it from the shape of
the code.

Run:  python -m hfmodem.tests.shrike.test_phase_power_cat
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from hfmodem.shrike import onair, pactor1, rxfront
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

from hfmodem.tests.shrike import test_grid
from hfmodem.tests.shrike.test_grid import FS, _run, check
from hfmodem.tests.shrike.test_granted_entry_slots import (
    ARM, D_S, GRANT_AT, PREKEY_N, QUIET, TAKEOVER_AT, _gateway, _keyed)

P1_WATTS, P3_WATTS = 15.0, 60.0
STARTED_ON = "0.50"
"""What the rig answers `l RFPOWER` with, and what teardown owes it back. 50 W
is the level every arm before this one flew at, launcher-read and never written.
"""


class _FakeCat:
    """A CAT port that records instead of opening one. No device is touched."""

    def __init__(self, *, start: str = STARTED_ON, takes: bool = True):
        self.start, self.takes = start, takes
        self.writes: list[tuple[float, object]] = []
        self.oneshots: list[float] = []
        self.reads = 0
        #: Asked at each write for whatever the scene wants recorded beside it.
        self.note = None

    def get_power(self):
        self.reads += 1
        return self.start

    def set_power(self, level) -> bool:
        self.writes.append((round(float(level), 4),
                            self.note() if self.note is not None else None))
        return self.takes

    def set_power_once(self, level) -> bool:
        self.oneshots.append(round(float(level), 4))
        return True


class _Peer:
    """Everything the station keys, as a list. The link layer's own seam."""

    def __init__(self) -> None:
        self.emissions: list[str] = []
        self.host: PtcHost | None = None

    def attach(self, host: PtcHost) -> None:
        self.host = host

    def pump(self) -> None: ...
    def cycle(self) -> None: ...

    def connect_burst(self, mycall, dxcall) -> None:
        self.emissions.append("connect")

    def send_p1_packet(self, payload, baud, seq, **kw) -> int:
        self.emissions.append("p1 packet")
        return len(payload)

    def send_p1_breakin(self, payload, baud, seq, **kw) -> int:
        self.emissions.append("p1 breakin")
        return len(payload)

    def send_p1_cs(self, index) -> None:
        self.emissions.append("p1 cs")

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        self.emissions.append("p3 packet")
        return len(payload)

    def send_entry_packet(self, sl, payload, status, acquire=False) -> int:
        self.emissions.append("p3 entry")
        return len(payload)

    def send_cs(self, index) -> None:
        self.emissions.append("p3 cs")


def _grant(host: PtcHost) -> None:
    host.on_rx_event(rxfront.Event(0.1, "unassigned", "grant",
                                   protocol=Protocol.PACTOR1,
                                   spare=pactor1.CS_59A, sense=0))


def _cs(index, protocol=Protocol.PACTOR1):
    return rxfront.Event(0.1, "cs", "control", protocol=protocol, cs=index,
                         sense=0)


def _linked(cat, *, p1=P1_WATTS, p3=P3_WATTS, plan=True):
    """A connected PACTOR-1 link, with the arm's own power plan attached.

    The arm's order, not a convenient one: the plan is built while the rig is
    idle -- which is where its read-back belongs, since reading closes the CAT
    port -- the PACTOR-1 level goes on before the call, and the call follows.
    """
    tx = _Peer()
    host = PtcHost(peer=tx, mycall="W9SSJ")
    host.p1_grant_only = True
    power = None
    if plan:
        power = onair._PhasePower(cat, p1=p1, p3=p3)
        host.phase_power = power
        power.select(Protocol.PACTOR1, "before the first call")
    host.arq.on_host_connect("W9SSJ", "WS8EOC")
    host.on_rx_event(_cs(pactor1.CS_SPEED))
    host.tick()
    return host, tx, power


def _levels(cat) -> list[float]:
    return [level for level, _ in cat.writes]


# --------------------------------------------------------------------------- #

def the_boundaries_are_where_the_level_moves() -> None:
    print("\nthe level moves at the link's boundaries and nowhere else")
    cat = _FakeCat()
    host, tx, power = _linked(cat)
    check("the rig was read before anything was written to it",
          cat.reads == 1 and power.baseline == 0.50, f"{cat.reads} reads")
    check("the call goes out at the PACTOR-1 level, written once",
          _levels(cat) == [0.15], str(_levels(cat)))

    # WHAT THE GRANT MUST NOT FIND ALREADY DONE: the entry packet queued. The
    # write is charged to the instant the grant is READ, which is a turnaround
    # in front of the slot `arq.on_rx_grant` then claims for the entry.
    cat.note = lambda: (host.protocol, host.arq.entry_pending)
    _grant(host)
    check("the grant puts the rig on the PACTOR-3 level, written once",
          _levels(cat) == [0.15, 0.60], str(_levels(cat)))
    check("...and it is written BEFORE the entry packet is queued, with the "
          "link still in PACTOR-1",
          cat.writes[-1][1] == (Protocol.PACTOR1, False),
          f"at the write the link was {cat.writes[-1][1]}")
    host.tick()
    check("...and the entry packet is what the next slot then keys",
          tx.emissions[-1] == "p3 entry" and host.arq.entry_pending,
          str(tx.emissions[-3:]))

    cat.note = None
    for _ in range(3):
        _grant(host)
        host.tick()
    check("a grant repeated every cycle is not a write every cycle",
          _levels(cat) == [0.15, 0.60], str(_levels(cat)))

    host.fall_back("the peer never took the entry packet")
    check("a fallback puts the PACTOR-1 level back",
          _levels(cat) == [0.15, 0.60, 0.15], str(_levels(cat)))
    check("...and the link is in PACTOR-1 to go with it",
          host.protocol is Protocol.PACTOR1, str(host.protocol))

    host.disconnected()
    check("the link ending asks for nothing further -- it is already there",
          _levels(cat) == [0.15, 0.60, 0.15], str(_levels(cat)))

    power.restore()
    check("teardown gives the rig back the level it was found on, through a "
          "one-shot, after the key is down",
          cat.oneshots == [0.50], str(cat.oneshots))


def the_link_ending_in_pactor3_comes_back_down() -> None:
    """The other way out of PACTOR-3, and the one `--no-p3-fallback` leaves.

    A link held in PACTOR-3 to its teardown never calls `fall_back`, so if the
    restore lived only there the rig would be left at the PACTOR-3 level for
    whatever ran next -- which on this station is the next call in the same
    process.
    """
    print("\na link that ends in PACTOR-3 still comes back down")
    cat = _FakeCat()
    host, _tx, power = _linked(cat)
    host.no_p3_fallback = True
    _grant(host)
    host.tick()
    check("the link is in PACTOR-3 at the level the grant bought",
          host.protocol is Protocol.PACTOR3 and _levels(cat) == [0.15, 0.60],
          f"{host.protocol}, {_levels(cat)}")
    host.disconnected()
    check("the teardown writes the PACTOR-1 level once",
          _levels(cat) == [0.15, 0.60, 0.15], str(_levels(cat)))
    power.restore()
    check("...and the rig is handed back what it started on",
          cat.oneshots == [0.50], str(cat.oneshots))


def a_level_nobody_named_is_never_written() -> None:
    print("\nthe flags are the whole of the permission to write")
    cat = _FakeCat()
    plan = onair._phase_power(SimpleNamespace(p1_watts=None, p3_watts=None), cat)
    check("no levels named, no plan and not even a read-back",
          plan is None and cat.reads == 0 and not cat.writes,
          f"{plan!r}, {cat.reads} reads")
    check("...and no rig at all is the same answer",
          onair._phase_power(SimpleNamespace(p1_watts=15.0, p3_watts=60.0),
                             None) is None)

    host, tx, _ = _linked(cat, plan=False)
    check("a host built without one holds no plan", host.phase_power is None)
    _grant(host)
    host.tick()
    host.fall_back("the peer never took the entry packet")
    host.disconnected()
    check("...and a whole grant, entry, fallback and teardown touch the rig "
          "not once -- which is every arm flown before tonight",
          not cat.writes and not cat.oneshots and cat.reads == 0,
          f"{cat.writes}, {cat.oneshots}")

    # HALF A PLAN IS HALF THE WRITES. An operator who names one level is asking
    # for that phase and saying nothing about the other, and a level the rig is
    # already on is not this arm's to move.
    only_p3 = _FakeCat()
    host, _tx, _power = _linked(only_p3, p1=None)
    check("with only --p3-watts the call goes out on the rig's own level",
          not only_p3.writes, str(only_p3.writes))
    _grant(host)
    check("...and the grant is still answered at the PACTOR-3 level",
          _levels(only_p3) == [0.60], str(_levels(only_p3)))
    host.fall_back("the peer never took the entry packet")
    check("...with nothing to come back to, so nothing is written",
          _levels(only_p3) == [0.60], str(_levels(only_p3)))


def a_refused_write_does_not_stop_the_session() -> None:
    """The level is the experiment's variable; the session is the experiment."""
    print("\na CAT write that does not go is reported, and the arm goes on")
    cat = _FakeCat(takes=False)
    host, tx, power = _linked(cat)
    check("the refusal leaves the plan believing nothing about the rig",
          power.level is None, str(power.level))
    _grant(host)
    host.tick()
    check("...the grant is still taken and the entry packet still keyed",
          tx.emissions[-1] == "p3 entry" and host.arq.entry_pending,
          str(tx.emissions[-3:]))
    check("...and the write was attempted at each boundary, since nothing "
          "says the rig ever took one",
          _levels(cat) == [0.15, 0.60], str(_levels(cat)))
    power.restore()
    check("...and a level this arm never set is not one it restores",
          not cat.oneshots, str(cat.oneshots))


# --------------------------------------------------------------------------- #
# The whole arm, on the bench clock: where the write lands inside the cycle.

MADE: list["_CatRig"] = []
BENCH: list = []


class _CatRig(test_grid._Rig):
    """The bench's rig, with a CAT that answers RFPOWER and remembers when."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.writes: list[tuple[float, int]] = []   # (level, converter sample)
        self.oneshots: list[float] = []
        MADE.append(self)

    def get_power(self):
        return STARTED_ON

    def set_power(self, level) -> bool:
        self.writes.append((round(float(level), 4),
                            BENCH[0].now if BENCH else 0))
        return True

    def set_power_once(self, level) -> bool:
        self.oneshots.append(round(float(level), 4))
        return True


class _ClockBench(test_grid._Bench):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        BENCH.append(self)


#: One run per command line, because the two scenes below are the same session
#: asked two questions and a bench session is not cheap.
_ARMS: dict[tuple[str, ...], dict] = {}


def _arm(*watts: str) -> dict:
    """`test_granted_entry_slots`'s scene, with the power flags on the line."""
    if watts in _ARMS:
        return _ARMS[watts]
    del MADE[:], BENCH[:]
    answers: list[int] = []
    rig, bench = test_grid._Rig, test_grid._Bench
    test_grid._Rig, test_grid._Bench = _CatRig, _ClockBench
    try:
        got = test_grid._session(
            cycles=6, hold=40, charge=0, decode=True, peer=False,
            seconds=300.0, keep_upgrade=True, prekey_n=PREKEY_N, d=D_S,
            extra_argv=ARM + watts,
            answer=_gateway(grant_at=GRANT_AT, quiet=QUIET,
                            takeover_at=TAKEOVER_AT, answers=answers))
    finally:
        test_grid._Rig, test_grid._Bench = rig, bench
    got["rig"] = MADE[0]
    _ARMS[watts] = got
    return got


def the_write_is_a_turnaround_in_front_of_the_entry() -> None:
    print("\nthe whole arm: where in the cycle the CAT write actually lands")
    got = _arm("--p1-watts", "15", "--p3-watts", "60")
    rig, log = got["rig"], got["log"]
    levels = [level for level, _ in rig.writes]
    # Three, and the third is the scene's own: this gateway grants, goes quiet
    # for the fade that releases the receive window, and takes the channel back,
    # so the link spends its entry budget and falls back to PACTOR-1 -- which is
    # a phase boundary like the other two and is written like one.
    check("the arm wrote the levels the operator named, in the order the link "
          "changed phase: the call, the grant, and the fallback behind it",
          levels == [0.15, 0.60, 0.15], str(levels))
    check("...and said so in the transcript, in watts",
          "RFPOWER 0.15 -> 15 W (PACTOR-1" in log
          and "RFPOWER 0.60 -> 60 W (PACTOR-3" in log,
          "; ".join(ln.strip() for ln in log.splitlines() if "RFPOWER" in ln))

    entries = _keyed(got, "ENTRY")
    check("the granted entry packet reached the air", bool(entries),
          f"{len(entries)} entries")
    at = rig.writes[1][1]
    first = entries[0]["first"]
    bench = got["bench"]
    # THE PRE-KEY WINDOW, in the terms the loop itself uses: the callback notice
    # the placement backstop works in, plus the admission reserve held in front
    # of every key. A write inside that span is a write charged to the slot.
    prekey_n = bench.key_notice + round(onair.TX_ADMIT_RESERVE_S * FS)
    check("...and the PACTOR-3 write landed clear of the pre-key window in "
          "front of it, by the turnaround the grant arrives in",
          first - at >= prekey_n,
          f"{(first - at) / FS * 1e3:.1f} ms in front of the entry carrier, "
          f"against a {prekey_n / FS * 1e3:.1f} ms window")
    check("teardown handed the rig back the level it started on, after the "
          "PTT was down",
          rig.oneshots == [0.50]
          and log.index("PTT off.") < log.index("restored (the level"),
          str(rig.oneshots))


def the_same_arm_without_the_flags_writes_nothing() -> None:
    print("\nNEGATIVE CONTROL: the same arm with no levels named")
    got = _arm()
    rig = got["rig"]
    check("nothing was written to the rig and nothing restored",
          not rig.writes and not rig.oneshots,
          f"{rig.writes}, {rig.oneshots}")
    check("...and the transcript never mentions RFPOWER",
          "RFPOWER" not in got["log"],
          "; ".join(ln.strip() for ln in got["log"].splitlines()
                    if "RFPOWER" in ln))
    check("...while the grant was still read and the entry still keyed, which "
          "is what makes this the same scene",
          "0x59A grant" in got["log"] and bool(_keyed(got, "ENTRY")))

    # THE WHOLE CLAIM, MEASURED RATHER THAN ARGUED: if the writes cost a cycle
    # anything, the two runs place their carriers differently. They are the same
    # peer on the same clock, so every slot has to be the same slot.
    powered = _arm("--p1-watts", "15", "--p3-watts", "60")
    slots = [b["slot"] for b in _keyed(got, "ENTRY")]
    with_cat = [b["slot"] for b in _keyed(powered, "ENTRY")]
    check("...and the arm that wrote the rig keyed its entries on exactly the "
          "slots the arm that did not keyed them on",
          slots == with_cat and bool(slots), f"{slots} against {with_cat}")
    check("...every burst of the session, not only the entries",
          [b["slot"] for b in got["bursts"]]
          == [b["slot"] for b in powered["bursts"]],
          f"{len(got['bursts'])} bursts against {len(powered['bursts'])}")


SCENES = (the_boundaries_are_where_the_level_moves,
          the_link_ending_in_pactor3_comes_back_down,
          a_level_nobody_named_is_never_written,
          a_refused_write_does_not_stop_the_session,
          the_write_is_a_turnaround_in_front_of_the_entry,
          the_same_arm_without_the_flags_writes_nothing)


def test_the_boundaries_are_where_the_level_moves() -> None:
    assert _run(the_boundaries_are_where_the_level_moves)


def test_the_link_ending_in_pactor3_comes_back_down() -> None:
    assert _run(the_link_ending_in_pactor3_comes_back_down)


def test_a_level_nobody_named_is_never_written() -> None:
    assert _run(a_level_nobody_named_is_never_written)


def test_a_refused_write_does_not_stop_the_session() -> None:
    assert _run(a_refused_write_does_not_stop_the_session)


def test_the_write_is_a_turnaround_in_front_of_the_entry() -> None:
    assert _run(the_write_is_a_turnaround_in_front_of_the_entry)


def test_the_same_arm_without_the_flags_writes_nothing() -> None:
    assert _run(the_same_arm_without_the_flags_writes_nothing)


def main() -> int:
    return 0 if _run(*SCENES) else 1


if __name__ == "__main__":
    sys.exit(main())
