# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A link that stops transmitting, which on the air is a station that has gone.

PACTOR is half-duplex on a fixed cycle grid: the sending station puts a packet in
every slot and the receiving station answers every one of them with a control
signal. There is no third thing, and in particular there is no "nothing to say"
on the wire. A slot left empty is a FAILED PACKET at the far end -- charged to a
retry counter, with the peer's timing-recovery estimator spending the cycle on
noise -- and a station that stops filling its slots reads exactly like a station
that has been switched off. Two faults ended in that silence, from opposite ends:

  * an ISS whose transmit buffer had drained transmitted NOTHING, for the rest of
    the session, while its own IRS branch was answering every single cycle. The
    same silence arrived at from the other side: `--message` never reached a
    `--hold` session at all, so a held link had nothing to send and sent nothing.

  * and after the PACTOR-1 -> PACTOR-3 upgrade, a peer that cannot follow stops
    being intelligible and nothing noticed. PACTOR has no capability field and no
    refusal -- a station declines a mode by never transmitting it -- so silence is
    the only signal there will ever be, and a modem that cannot read it cannot
    work with the 83 of 558 Winlink PACTOR channels that are PACTOR-2 only.

Both are asked the same way, and it is the only way worth asking: run a session
against a scripted station and count what actually went on the air. The peer is a
recording placed on the caller's own grid and the session is `shrike.onair`'s own
`--replay`, so every burst below goes through the real renderer, the real cycle
grid and the real receiver -- the arrangement `test_breakin` drives the reversal
with, asked a different question.

Run: python -m hfmodem.tests.shrike.test_silence
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from hfmodem.shrike import arq, onair, pactor1, rxfront, spec
from hfmodem.shrike.arq import GOODBYE_CYCLES, State
from hfmodem.shrike.spec import Protocol

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)
D_MS = 90.0
"""The scripted station's turnaround, `d` -- where in the gap after our packet its
answer lands. Any value the caller can acquire will do; this is the one
`test_breakin`'s peer uses, so the two scripted stations are the same station."""

MESSAGE = "TEST DE W9SSJ"
ANNOUNCE = 7
"""Bytes of the callsign announcement (`1W9SSJ\\r`) the link layer queues itself
when the connect is answered -- so what a session with a `--message` owes the far
end is this plus the message, and a session that never queued the message stops
at exactly this."""

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _peer_wav(path: Path, *, answers: int, slots: int,
              alternate: bool = True, again: Iterable[int] = ()) -> None:
    """A scripted PACTOR-1 station that answers `answers` cycles and then stops.

    On the caller's own grid, which the caller anchors on its own connect and
    free-runs from, so every instant is known in advance: `d` after each of our
    packets would end. CS4 first -- a legal answer to a call, which brings the
    link up at 100 Bd -- and then the acknowledgement alternation, because a
    repeat of one codeword means REQUEST and would stall the exchange rather than
    advance it. The shift inverts every cycle, as a real station's does.

    `alternate=False` is the other station the 2026-08-19 evening recorded: one
    codeword, unbroken, for as many cycles as it is given. It is a repeat request
    every time and it advances nothing, but it is an ANSWER -- so the ARQ's retry
    budget never spends against it.

    Everything past `answers` is silence, and silence is the whole point: it is
    what a station that cannot follow us sounds like, and it is what the two
    faults here produce at OUR end.

    `again` are cycles it comes back on, after the silence has cost us the link.
    Nothing at our end can answer them by then, which is what makes them worth
    recording: they are the peer's account of what it did once we went quiet.
    """
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in [*range(answers), *again]:
        cs = (pactor1.CS_SPEED if k == 0 else
              pactor1.CS_ACK_A if not alternate or k % 2 else pactor1.CS_ACK_B)
        rf = onair._trim_silence(
            np.asarray(pactor1.control_signal(cs, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def _run(wav: Path, outdir: Path, *, hold: int, message: str | None = None,
         ceiling: int | None = None, pactor1_only: bool = False,
         observe: bool = False) -> dict:
    """One whole `shrike.onair` session over `wav`, recording every keyed burst.

    The label each transmission carries is `onair.RadioTx`'s own, so it says which
    renderer built the burst -- `P1 pkt#...` against `SL3 pkt ...` -- which is how
    a fallback is read off the air rather than off the link layer's opinion of
    itself. Both ends agreeing they are in PACTOR-1 would prove nothing.

    The link state is recorded PER BURST, alongside the label. A session that
    completes its hold is then told to disconnect, so whatever the link did in
    the middle it ends up DISCONNECTING -- and a check that reads `arq.state`
    afterwards is asking about the goodbye, not about the thing it names.
    """
    keyed: list[str] = []
    states: list = []
    link: list = []
    made: list = []

    class _Tx(onair.RadioTx):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

        def _tx(self, audio, what, **kwargs):
            state = self.host.arq.state if self.host else None
            # ...and the two things the HOST would have been told, frozen where
            # the burst went out. Asked after `main()` returns they are answers
            # about a torn-down link -- `_ruled_out` is cleared with it -- which
            # is the trap the state field above was already moved out of.
            told = (self.host._status_bytes(),
                    frozenset(self.host._ruled_out)) if self.host else None
            n = self.n
            super()._tx(audio, what, **kwargs)
            if self.n == n:
                # A burst the transmitter refused outright never reached the air,
                # and a record of what went out that carries it cannot be used to
                # ask whether anything did. See `RadioTx.answered_refusal`.
                return
            keyed.append(what)
            states.append(state)
            link.append(told)

    # The per-cycle capture write runs on its own thread and outlives the
    # session; nothing here reads the files, and leaving it on races the
    # temporary directory away from under it.
    # The channel ceiling, brought within reach. It is ten minutes of air, and a
    # session that ran to it here would be 480 cycles of rendered PACTOR; the loop
    # cannot be asked what it says at the ceiling without one.
    budget = onair._HoldBudget
    saved = onair.RadioTx, onair._save_capture_async, budget
    onair.RadioTx = _Tx
    onair._save_capture_async = lambda *a, **kw: None
    if ceiling is not None:
        onair._HoldBudget = lambda cycles, **kwargs: budget(cycles, ceiling, **kwargs)
    argv, log = sys.argv, io.StringIO()
    sys.argv = ["shrike.onair", "--replay", str(wav), "--hold", str(hold),
                "--max-cycles", "3", "--mycall", "W9SSJ", "--dxcall", "K7ABC",
                "--dial", "7100000", "--outdir", str(outdir)]
    if message is not None:
        sys.argv += ["--message", message]
    if pactor1_only:
        sys.argv += ["--pactor1-only"]
    if observe:
        sys.argv += ["--observe-after-link-down"]
    try:
        with contextlib.redirect_stdout(log):
            onair.main()
    finally:
        onair.RadioTx, onair._save_capture_async, onair._HoldBudget = saved
        sys.argv = argv
    return {"host": made[0].host, "keyed": keyed, "states": states,
            "link": link, "log": log.getvalue()}


def _packets(keyed: list[str]) -> list[str]:
    return [w for w in keyed if "pkt" in w or "BREAK-IN" in w]


# --------------------------------------------------------------------------
# 1. a held link with a message, and then with nothing left to say
# --------------------------------------------------------------------------
HOLD = 8


def the_held_link_keeps_talking() -> None:
    """`--message` reaches a held session, and the cycle after it drains is not
    silent.

    Two claims in one session because they are two halves of one requirement: a
    held link transmits in every cycle it holds. The message is the part a host
    gave us; the idle packets are what the protocol requires when the host gave us
    nothing, and the second is the harder half -- it is the one that used to end
    the session without ending the link.
    """
    print("\nA held link, against a station that answers every cycle")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav, answers=HOLD + 3, slots=HOLD + 5)
        # HELD IN PACTOR-1, because the scripted station is: it renders PACTOR-1
        # codewords and nothing else, and a message long enough to outlast the
        # first packet is enough to offer it the upgrade. What that costs is the
        # next arm's subject and this one's confound -- the offer stands for
        # `arq.UPGRADE_SILENCE_CYCLES` cycles the peer cannot read, and the hold
        # budget those come out of is what this arm is counting packets against.
        got = _run(wav, Path(tmp) / "out", hold=HOLD, message=MESSAGE,
                   pactor1_only=True)

    host, keyed = got["host"], got["keyed"]
    pkts = _packets(keyed)
    # DISCONNECTED is the healthy ending now, not a failure to connect: the hold
    # budget's end transmits the QRT and the scripted answers acknowledge it, so
    # a link that came up, carried its traffic and said goodbye reads
    # DISCONNECTED with a QRT on the air. A link that never came up reads
    # DISCONNECTED with no QRT keyed, which is what this still catches.
    check("the scripted station's answer brings the link up, and the hold ends "
          "with a transmitted QRT",
          any("QRT" in w for w in keyed) and host.arq.state in
          (State.CONNECTED, State.DISCONNECTING, State.DISCONNECTED),
          f"{host.arq.state}, keyed {[w for w in keyed if 'QRT' in w]}")
    # WHAT THE HOST ASKED FOR REACHED THE AIR AND WAS ACKNOWLEDGED. `sent_total`
    # counts bytes the far end confirmed, not bytes the host wrote, so a message
    # that was queued and never transmitted does not reach this number. It used to
    # stop dead at the callsign announcement: `--message` was queued only on the
    # connect-send-disconnect path, and a `--hold` session runs a different loop.
    check("--message reaches a --hold session and is acknowledged",
          host.sent_total >= ANNOUNCE + len(MESSAGE),
          f"{host.sent_total}B confirmed, of {ANNOUNCE + len(MESSAGE)}B owed")
    # ONE PACKET PER HELD CYCLE. The message is three packets' worth at 100 Bd, so
    # most of the hold is spent with an empty buffer -- which is exactly the state
    # that used to key nothing at all.
    check("every held cycle puts a packet on the air", len(pkts) >= HOLD,
          f"{len(pkts)} packets keyed over {HOLD} held cycles")
    idle = [w for w in pkts if w.endswith(" 0B")]
    check("...including the cycles after the buffer has drained, which carry an "
          "empty field rather than nothing", len(idle) >= HOLD // 2,
          f"{len(idle)} idle packets of {len(pkts)}: {idle}")
    print(f"    keyed: {keyed}")


# --------------------------------------------------------------------------
# 2. an upgrade the peer cannot follow
# --------------------------------------------------------------------------
FOLLOW = 3
"""Cycles the scripted station answers before it goes quiet: the connect answer,
its acknowledgement of the packet that carries the callsign, and one
acknowledgement of a data packet behind that. The last is what arms the fault --
`arq._on_ack` offers the upgrade on an ACKNOWLEDGED PACKET, so a station that
answers the call and nothing else never provokes one.

The `--message` this arm passes is the other half of arming it. The offer is
taken only with payload still waiting behind the acknowledged packet
(`ptc.PtcHost.upgrade`), and the callsign announcement on its own is gone in the
first packet -- a link with nothing left to say does not climb, and has no reason
to.

Three and not two since 2026-08-28. This peer answers 5 ms inside the nominal
turnaround, which the point read missed and the sweeping decoder delivered a
cycle late, so the connect answer's own acknowledgement was credited to the first
data packet. `p1rx.CS_SEARCH_HALF_S` puts each answer in the cycle it arrived in,
and an acknowledgement of a data packet now has to be one the peer sent after
that packet was on the air."""

QUIET_HOLD = 10
"""Long enough for the silence to be counted out (`arq.UPGRADE_SILENCE_CYCLES`)
and for the link to transmit again afterwards, and short enough that the FSM's own
retry budget has not yet yielded the link."""


def the_peer_that_could_not_follow() -> None:
    print("\nAn upgrade into a station that stops answering")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav, answers=FOLLOW, slots=QUIET_HOLD + 5)
        got = _run(wav, Path(tmp) / "out", hold=QUIET_HOLD, message=MESSAGE)

    host, keyed, log = got["host"], got["keyed"], got["log"]
    at = [i for i, w in enumerate(keyed) if "pkt" in w or "BREAK-IN" in w]
    pkts = _packets(keyed)
    upgraded = [i for i, w in enumerate(pkts) if w.startswith("SL")]
    check("the acknowledged PACTOR-1 packet takes the link to PACTOR-3",
          bool(upgraded), f"packets keyed: {pkts}")
    # THE SILENCE IS THE REFUSAL, and it is all we are ever going to get. Read off
    # the air rather than off `host.protocol`: what matters is that the renderer
    # went back to the waveform the peer was last known to be reading.
    after = pkts[upgraded[-1] + 1:] if upgraded else []
    check("...and after the peer stops answering, the link keys PACTOR-1 again",
          bool(after) and all(w.startswith("P1") for w in after),
          f"after the last PACTOR-3 packet: {after}")
    check("...within the cycles the constant allows",
          bool(upgraded) and len(upgraded) <= arq.UPGRADE_SILENCE_CYCLES + 1,
          f"{len(upgraded)} PACTOR-3 packets keyed, budget "
          f"{arq.UPGRADE_SILENCE_CYCLES}")
    check("the link layer agrees with what it is transmitting",
          host.protocol is Protocol.PACTOR1, str(host.protocol))
    # ...and so must the STATUS BYTES, which are the host's whole view of it. A
    # fallen-back link is the routine outcome of this scenario, and byte 3 kept
    # reporting the PACTOR-3 entry level the upgrade set -- the same defect the
    # level byte had, one byte over.
    st = got["link"][at[upgraded[-1] + 1]][0] if after else b""
    check("...and the status bytes say so: level 1, no PACTOR-3 speed level",
          st[1:3] == b"\x01\x00", st.hex())
    # A FALLBACK THAT DROPPED THE LINK WOULD BE NO BETTER THAN THE SILENCE. The
    # point of going back is to carry on in the protocol that was working.
    #
    # ASKED WHERE THE FALLBACK IS, not at the end of the session. This read
    # `arq.state` after `main()` returned, and a hold that runs to its end is
    # followed by a QRT the session is required to transmit -- so the state
    # there is DISCONNECTING for any run that finishes, whatever the fallback
    # did. It passed because the hush was taking those goodbye cycles off the
    # air: the disconnect was queued on an FSM that got no tick, the QRT never
    # went out, and the session sat in CONNECTED because it had never said
    # anything. The state at the first PACTOR-1 packet after the fallback is the
    # one the check is named for.
    back = got["states"][at[upgraded[-1] + 1]] if after else None
    check("...and the link is still up where it fell back",
          back == State.CONNECTED, str(back))
    # NOTHING THE UPGRADE WAS CARRYING IS LOST BY COMING BACK DOWN. The PACTOR-3
    # field was 59 bytes and the PACTOR-1 one is 8, so the packet in flight is
    # requeued and split (`arq.rechunk_inflight`) rather than truncated by the
    # renderer, and the same information is on the air a cycle later at the rate
    # the peer was last known to read.
    #
    # The QRT is arm 1's to ask for, and it has to be: the goodbye rides a data
    # packet, `on_host_disconnect` requeues whatever is in flight, and the bit
    # goes on the packet that DRAINS the buffer. A peer that stopped answering
    # cannot acknowledge one, so a session with traffic still owed ends on the
    # retry budget instead, saying so in its summary.
    check("...carrying the same information, re-chunked to the PACTOR-1 field",
          bool(after) and all(w.endswith("100Bd 8B") for w in after),
          f"after the fallback: {after}")
    # RULED OUT FOR THE REST OF THE LINK, which is what keeps it from flapping:
    # the FSM offers an upgrade on every acknowledgement, so without this the
    # very next one puts the link back into the protocol that just failed. Read
    # where the fallback is, for the reason above -- a torn-down link has cleared
    # the set, and asking the host afterwards asks about the next link.
    check("PACTOR-3 is not offered again on the next acknowledgement, so the "
          "link cannot flap",
          bool(after) and Protocol.PACTOR3 in got["link"][at[upgraded[-1] + 1]][1]
          and all(w.startswith("P1") for w in after))
    check("the session says why, where an operator will see it",
          "WE HEARD NOTHING AT ALL" in log or "WE READ NO CODEWORD" in log)
    print(f"    keyed: {keyed}")


# --------------------------------------------------------------------------
# 3. a hold that outlived its link
# --------------------------------------------------------------------------
GONE = 3
"""Cycles the scripted station answers: enough to bring the link up and
acknowledge one packet, and then nothing ever again."""

GONE_HOLD = 40
"""Long enough for the ARQ to spend its whole retry budget as the ISS and then to
have most of a hold still left over. That leftover is the fault.

It used to be long enough for a yield and eight more silent cycles as the IRS as
well. A peer that goes quiet no longer buys that handover: `_on_nak` yields to a
data packet from the far end and to nothing weaker, so this budget now runs out
where the ISS is standing."""


def arq_tail() -> int:
    """Cycles a completed hold spends transmitting its goodbye."""
    return GOODBYE_CYCLES + 1


def the_hold_does_not_outlive_the_link() -> None:
    """When the ARQ gives up, the session stops. It used to spin out its budget.

    A `--hold` is a number of cycles to STAY UP for, and the loop tested only
    that number: the ARQ could abort in cycle twelve of ninety and the loop would
    go on raising and lowering the carrier for the other seventy-eight, holding a
    channel that nothing was on. At 1.25 s a cycle that is close to two minutes,
    and it is what the operator heard on 2026-08-06 -- "long periods of silence,
    on the order of minutes" -- while the log went on printing cycle numbers.

    The link ending is the thing the session is for, so it is reported rather
    than absorbed: the summary says which cycle it went down in.
    """
    print("\nA hold whose link is gone")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav, answers=GONE, slots=GONE_HOLD + 5)
        got = _run(wav, Path(tmp) / "out", hold=GONE_HOLD)

    host, log = got["host"], got["log"]
    held = [int(m) for m in re.findall(r"\[grid\] hold (\d+) slot", log)]
    ran = max(held) if held else 0
    # BOTH HALVES, because the end state alone is satisfied by a call that was
    # never answered at all: a session that gives up on a frequency nobody is on
    # also ends DISCONNECTED with "abort" in its log. What this scene is about is
    # a link that came up FIRST, so that is asked of the record of the RF.
    check("the link came up and the ARQ then gave up on the silent station",
          State.CONNECTED in got["states"]
          and host.arq.state in (State.DISCONNECTED, State.LISTENING)
          and "nothing to hand it to" in log,
          f"{host.arq.state}, states seen {sorted({str(s) for s in got['states']})}")
    # THE NUMBER THE OPERATOR HEARS. Everything else here is bookkeeping; this is
    # seconds of dead carrier and dead air on a shared channel.
    down = [int(m) for m in
            re.findall(r"the link went down in hold cycle (\d+)", log)]
    left = GONE_HOLD + arq_tail() - ran
    check("...and the session stopped in the cycle the link went down in, not "
          "at the end of the budget",
          bool(down) and ran == down[0] - 1 and ran < GONE_HOLD,
          f"{ran} hold cycles run, link down in {down}; "
          f"{left} cycles ({left * spec.CYCLE_SHORT_S:.0f} s) of the budget "
          f"never ran")
    check("...and it says so where an operator will see it",
          "LINK DOWN" in log and "the link went down in hold cycle" in log)
    check("...including in the summary, which is what survives a scrolling "
          "terminal", "session ended: the link went down" in log,
          log[log.rfind("session ended:"):][:80])


# --------------------------------------------------------------------------
# 4. two silences that printed one line
# --------------------------------------------------------------------------
TOLD_APART_HOLD = 12
"""`--hold 12`, the budget every PACTOR arm of the 2026-08-19 evening flew."""


def the_ending_says_what_the_peer_did() -> None:
    """A peer answering every cycle and a peer never heard, told apart.

    Both end down the same branch and always will. A repeated codeword is still
    an ANSWER, so `ArqConfig.max_retries` never spends and the hold wins the race
    between the two budgets; the goodbye then goes out and a repeat request is
    not an acknowledgement of it. Six sessions of 2026-08-19 ended on that
    branch, and they were two entirely different evenings:

      * WS8EOC 20:20, KB5LZK 20:41 and K4MSU 20:48 heard a gateway the whole way
        through. `captures/onair-0819-2048` reads CS1 unbroken across `hold_05`
        to `hold_16` -- twelve consecutive cycles of a strong, clean "send that
        again", the richest peer material of the night.
      * KO4HJO 20:53, W9OTR 21:08 and VE3WLR 21:16 heard nothing inside the link
        at all.

    `the peer never acknowledged the goodbye` was printed over both, which is
    true of both and describes neither.
    """
    print("\nTwo sessions that ended on the same branch")
    ends = {}
    for name, alternate, answers in (("stuck", False, TOLD_APART_HOLD + 8),
                                     ("gone", True, 1)):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "peer.wav"
            _peer_wav(wav, answers=answers, slots=TOLD_APART_HOLD + 8,
                      alternate=alternate)
            log = _run(wav, Path(tmp) / "out", hold=TOLD_APART_HOLD)["log"]
        ends[name] = log[log.rfind("session ended: "):].splitlines()[0]
        print(f"    {name}: {ends[name]}")

    check("THE BAR: the two sessions do not print the same line",
          ends["stuck"] != ends["gone"])
    # WHAT THE PEER SAID, counted from the cycle the link came up rather than
    # from the session's first sample: the connect answer is a codeword too, and
    # a summary that counts it has already said "a peer answered" about a session
    # in which nothing answered inside the link.
    held = re.search(r"the peer answered (\d+) held cycles \((CS\d\S*) x(\d+)\)"
                     r" and never alternated -- every one a repeat request",
                     ends["stuck"])
    check("a peer stuck on one codeword is reported as answering, with the "
          "codeword and the count", bool(held)
          and int(held.group(1)) >= TOLD_APART_HOLD, ends["stuck"])
    check("...and it is named a repeat request rather than an acknowledgement, "
          "which is what a codeword repeated IS in PACTOR-1",
          bool(held) and held.group(2).startswith("CS1"),
          held.group(2) if held else "no match")
    check("a link with no decoded controls says so without denying data frames",
          "no control codewords decoded inside the link" in ends["gone"], ends["gone"])
    # The goodbye's own fate is still reported -- it is just no longer the only
    # thing reported, and it is no longer asked of a session that never sent one.
    check("both still say what became of the goodbye",
          all("goodbye" in e for e in ends.values()))


# --------------------------------------------------------------------------
# 5. a reverse channel that is not PACTOR-1
# --------------------------------------------------------------------------
def _cs_events(protocol, count: int, text: str = "") -> list:
    """`count` codewords alternating between the two indices `P1_ACKS` names."""
    return [rxfront.Event(t=float(i), kind="cs", text=text, protocol=protocol,
                          cs=i % 2) for i in range(count)]


def the_alternation_is_pactor_1s_rule_alone() -> None:
    """`P1_ACKS` is a pair of INDICES, and PACTOR-3 has six codewords, not four.

    In PACTOR-1's table those two indices are CS1 and CS2 and their alternation IS
    the acknowledgement -- a repeat means "send that again". In PACTOR-3's they
    are ACK and REQ, so a peer going ACK, REQ, ACK has asked twice for the packet
    it already has, and counting it as five alternations reports the opposite of
    what happened. `_summary`'s own list is filtered on the protocol and on zero
    bit errors; this helper filtered on neither, so one session's output could
    print `alternating 5 times` and `never alternated` about the same codewords.
    """
    print("\nA reverse channel that is not PACTOR-1")
    upgraded = onair._held_answers(_cs_events(Protocol.PACTOR3, 6))
    check("an upgraded link's PACTOR-3 codewords raise no PACTOR-1 alternation",
          "alternating" not in upgraded, upgraded)
    noisy = onair._held_answers(
        _cs_events(Protocol.PACTOR1, 6, "decoded with 3 bit errors"))
    check("...and neither do PACTOR-1 codewords the demodulator was unsure of: "
          "a bit error is not the peer changing its answer",
          "alternating" not in noisy, noisy)
    clean = onair._held_answers(_cs_events(Protocol.PACTOR1, 6))
    check("...while a clean PACTOR-1 alternation is still what it always was",
          "alternating 5 times" in clean, clean)


# --------------------------------------------------------------------------
# 6. the bound that actually closed the hold
# --------------------------------------------------------------------------
CEILING = 5
"""A channel ceiling within reach of a test. The shipped one is
`onair.HOLD_MAX_CYCLES` -- ten minutes of air, and 480 cycles of rendered PACTOR
to reach."""

OVER_HOLD = 40
"""A `--hold` the ceiling is under, which is the arrangement the ceiling is FOR:
the operator asks for a long idle timeout and the channel-fairness bound is what
the session actually ends on."""


def the_ending_names_the_bound_that_closed_it() -> None:
    """A hold stopped by the ceiling said the idle timeout ran out. It had not.

    The two are different budgets with different owners. `--hold N` is the
    operator's, and it expires only when nothing has crossed the link for N
    cycles; the ceiling is the channel's, it is what `_HoldBudget` calls the thing
    that keeps a grant from becoming a lease, and against a peer that keeps
    sending it is the ONLY one that can ever end the session. Printing the idle
    number over that ending names a budget with nothing to do with it and leaves
    the one that closed the channel unmentioned -- the same defect the verdict
    line was rewritten to remove, one sentence over.
    """
    print("\nA hold the channel ceiling ended")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav, answers=CEILING + 8, slots=CEILING + 10)
        log = _run(wav, Path(tmp) / "out", hold=OVER_HOLD, message=MESSAGE,
                   ceiling=CEILING)["log"]

    end = log[log.rfind("session ended: "):].splitlines()[0]
    print(f"    {end}")
    held = [int(m) for m in re.findall(r"\[grid\] hold (\d+) slot", log)]
    check("the hold ends at the ceiling and not at the operator's number",
          bool(held) and max(held) <= CEILING + onair.QRT_CYCLES + 1,
          f"{max(held) if held else 0} hold cycles run, --hold {OVER_HOLD}, "
          f"ceiling {CEILING}")
    check("...and the ending names the ceiling, in the grid slots it is kept "
          "in, as what closed it",
          f"the {CEILING}-slot ceiling" in end and "shared channel" in end, end)
    check("...and does not report an idle budget that never expired",
          f"{OVER_HOLD} idle cycles" not in end, end)


# --------------------------------------------------------------------------
# 7. the rest of a budget, spent listening
# --------------------------------------------------------------------------
BACK_ON = range(28, 36)
"""Cycles the scripted station transmits in again, long after its silence has
cost us the link. Nothing at our end can answer them by then; they exist to be
HEARD, and a run that stops at the give-up hears none of them."""


def the_hold_may_listen_past_its_link() -> None:
    """`--observe-after-link-down`, which spends the rest of the hold receiving.

    Stopping at the give-up is right for the transmitter and wrong for the
    recording. What a peer does after we stop answering is a fact about the peer
    -- it asks again, it signs off, or it was never there -- and the three are
    the difference between our receiver being at fault and the channel being
    empty. A session that ends where the ARQ ended cannot tell them apart, and
    WS8EOC went on transmitting on our own raster for 26 s past our sign-off with
    nothing at this end listening -- a separate tap receiver is what caught it.

    So the flag keeps the loop's receive half running on what is left of the
    budget and takes its transmit half away outright. Both arms below fly the
    same recording and the pair is the whole point: the transcript is bought and
    the channel is charged nothing for it.
    """
    print("\nA hold that listens past its own link")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "peer.wav"
        _peer_wav(wav, answers=GONE, slots=GONE_HOLD + 5, again=BACK_ON)
        watched = _run(wav, Path(tmp) / "watch", hold=GONE_HOLD, observe=True)
        stopped = _run(wav, Path(tmp) / "stop", hold=GONE_HOLD)

    log, plain = watched["log"], stopped["log"]
    down = re.search(r"the link went down in hold cycle (\d+) of (\d+)", log)
    ran = max(int(m) for m in re.findall(r"\[grid\] hold (\d+) slot", log))
    seen = len(re.findall("OBSERVING, not keying", log))
    check("the run kept receiving for the rest of the budget",
          bool(down) and seen > 0 and ran == int(down.group(2)),
          f"link down in cycle {down.group(1) if down else '?'}, "
          f"{ran} hold cycles run, {seen} of them observed")
    # THE CHANNEL IS CHARGED NOTHING FOR IT, and this is the claim that decides
    # whether the flag is safe to fly at all. Compared against the same session
    # without it rather than against a count: what the two runs put on the air is
    # the same list of bursts, in the same order, and one of them then spent
    # thirty seconds listening. `_run` records what the transmitter TOOK, so a
    # burst the FSM tried to key and `observing` refused is absent from both.
    check("...and transmitted nothing at all while it did",
          watched["keyed"] == stopped["keyed"],
          f"{len(watched['keyed'])} bursts observing, "
          f"{len(stopped['keyed'])} stopping")

    def answers_after_link_down(text: str) -> int:
        tail = text[text.index("** LINK DOWN **"):]
        return len(re.findall(r"HOLD RX +[\d.]+ +cs ", tail))

    # The thing the flag is for. A codeword decoded here is one the session that
    # stops never had: same audio, same receiver, and the only difference is
    # whether anything was still listening to it.
    check("...and the peer's codewords after the link went down are in the "
          "transcript, where the run that stops has none",
          answers_after_link_down(log) >= len(BACK_ON) // 2
          and answers_after_link_down(plain) == 0,
          f"{answers_after_link_down(log)} decoded observing, "
          f"{answers_after_link_down(plain)} stopping, of {len(BACK_ON)} sent")
    end = log[log.rfind("session ended: "):].splitlines()[0]
    check("...and the summary says both what ended and what followed it",
          "the link went down in hold cycle" in end
          and f"observed {seen} further cycles without transmitting" in end, end)
    # THE DEFAULT IS THE DEFAULT. Everything above is reachable only through the
    # flag, and the session that does not fly it must be the session that ran
    # before there was one to fly.
    was = max(int(m) for m in re.findall(r"\[grid\] hold (\d+) slot", plain))
    check("without the flag the run still stops in the cycle the link went "
          "down in", bool(down) and was == int(down.group(1)) - 1
          and "OBSERVING" not in plain and "observed" not in plain,
          f"{was} hold cycles run, link down in {down.group(1) if down else '?'}")


def main() -> int:
    print("Three ways a PACTOR link goes quiet, and what its ending says of them")
    the_held_link_keeps_talking()
    the_peer_that_could_not_follow()
    the_hold_does_not_outlive_the_link()
    the_ending_says_what_the_peer_did()
    the_alternation_is_pactor_1s_rule_alone()
    the_ending_names_the_bound_that_closed_it()
    the_hold_may_listen_past_its_link()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
