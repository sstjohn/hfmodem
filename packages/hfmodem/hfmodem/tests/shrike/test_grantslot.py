# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The peer's upgrade grant, taken by the loop that is actually on the air.

`test_p3_upgrade` drives `PtcHost.on_rx_event` with the grant by hand and watches
the entry packet come out of `RadioTx`. It passed, and the first on-air arm to
carry `--p1-act-on-grant` still declined sixteen consecutive `0x59A` at zero bit
errors, because the flag was launched beside `--pactor1-only` and `_take_grant`
refuses under that -- a combination no test could reach, since no test ran the
session's own loop with the session's own flags.

AND A SESSION'S OWN FLAGS ARE USUALLY NONE. Answering the grant was an opt-in
until 2026-08-27, so every caller that did not know to pass the flag -- the mail
command `station.mail` prints among them -- decoded the grant and threw it away.
That is what `the_default_answers_the_grant` is here for: it runs the session the
way a caller with nothing to say about upgrades runs it.

So this runs the whole thing: `shrike.onair --replay` against a scripted station
that answers a call and then commands the upgrade, once per cycle, the way two
gateways and eight of this station's own sessions do. Every claim below is read
off the session's stdout, which is what the operator reads too.

  * the grant is decoded by the HOLD loop, not the setup loop -- the two read
    control signals at different points in their cycles and only one of them was
    ever exercised;
  * the burst after it is PACTOR-3 at speed level 1, in the slot the next
    PACTOR-1 packet would have had, with no cycle in between -- and it is the
    entry packet, an empty field, with the traffic still queued behind it;
  * the cycles after it, which the peer answers every one of, are not counted as
    silence -- and the line the operator reads names the state the peer is in;
  * a session with NO upgrade flags at all keys the entry packet, and keys it
    even where an uninvited attempt was contradicted first;
  * --decline-grant is what it takes to refuse, and no uninvited upgrade goes
    out under --p1-grant-only -- the grant is the only door while it is set;
  * and the parser refuses the pairs of flags that cannot both be honoured.

Run: python -m hfmodem.tests.shrike.test_grantslot
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from hfmodem.shrike import onair, pactor1, rxfront, spec
from hfmodem.shrike.arq import UPGRADE_SILENCE_CYCLES

FS = rxfront.FS
SLOT_N = round(spec.CYCLE_SHORT_S * FS)
PACKET_N = round(spec.P1_PACKET_S * FS)
D_MS = 90.0
"""The scripted station's turnaround -- `test_silence`'s and `test_breakin`'s, so
all three drive the same station."""

MESSAGE = "TEST DE W9SSJ -- " + "THE QUICK BROWN FOX 0123456789 " * 8
"""Long enough to be still draining when the grant arrives, so there is traffic
for the entry packet to be seen HOLDING BACK. A granted upgrade is taken with or
without something to carry, and what the entry packet keys is the same either
way -- `spec.TEMPLATE` and no user data at all."""

GRANT_FROM = 6
"""Cycle the scripted station starts commanding the upgrade in. Late enough that
the link is up and the hold loop is the reader; the acknowledgements before it are
what a granting gateway sends while it waits."""

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _peer_wav(path: Path, *, slots: int, grant: bool) -> None:
    """A PACTOR-1 station that answers the call and then commands PACTOR-3.

    On the caller's own grid, which the caller anchors on its own connect and
    free-runs from. CS4 answers the call at 100 Bd, the alternation acknowledges
    the packets after it, and from `GRANT_FROM` every answer slot carries `0x59A`
    instead -- which is what the gateways do: the word repeats once per cycle for
    as long as the entry packet has not arrived.
    """
    out = np.zeros((slots + 2) * SLOT_N, np.float32)
    for k in range(slots):
        if grant and k >= GRANT_FROM:
            cs = pactor1.CS_59A
        else:
            cs = (pactor1.CS_SPEED if k == 0 else
                  pactor1.CS_ACK_A if k % 2 else pactor1.CS_ACK_B)
        rf = onair._trim_silence(
            np.asarray(pactor1.control_signal(cs, invert=k % 2), np.float32))
        at = k * SLOT_N + PACKET_N + round(D_MS / 1000 * FS)
        out[at:at + rf.size] += rf
    onair.session.write_wav(str(path), out)


def _run(wav: Path, outdir: Path, *flags: str, hold: int = 10) -> str:
    """One whole `shrike.onair` session over `wav`; its stdout is the result."""
    # The per-cycle capture write runs on its own thread and outlives the
    # session; nothing here reads the files, and leaving it on races the
    # temporary directory away from under it.
    save = onair._save_capture_async
    onair._save_capture_async = lambda *a, **kw: None
    argv, log = sys.argv, io.StringIO()
    sys.argv = ["shrike.onair", "--replay", str(wav), "--hold", str(hold),
                "--max-cycles", "3", "--mycall", "W9SSJ", "--dxcall", "K7ABC",
                "--dial", "7100000", "--outdir", str(outdir),
                "--message", MESSAGE, *flags]
    try:
        with contextlib.redirect_stdout(log):
            onair.main()
    finally:
        onair._save_capture_async = save
        sys.argv = argv
    return log.getvalue()


_LINE = re.compile(
    r"^\s*(?:(?P<tag>HOLD RX|RX)\s+[\d.]+\s+(?P<kind>\w+)\s+(?P<text>.*)"
    r"|TX\[\d+\]\s+(?P<tx>.*?)\s\s)", re.M)


def _trace(log: str) -> list[tuple[str, str]]:
    """The session as air: what was heard and what was keyed, in order."""
    out = []
    for m in _LINE.finditer(log):
        if m.group("tx") is not None:
            out.append(("tx", m.group("tx")))
        else:
            out.append((m.group("tag"), f"{m.group('kind')} {m.group('text')}"))
    return out


def _field(what: str) -> int:
    m = re.search(r"(\d+)B", what)
    return int(m.group(1)) if m else 0


def _entry(trace: list[tuple[str, str]]) -> list[str]:
    return [w for tag, w in trace if tag == "tx" and w.startswith("SL1 ENTRY")]


def _first_grant(trace: list[tuple[str, str]]) -> int | None:
    return next((i for i, (_, what) in enumerate(trace)
                 if what.startswith("unassigned") and "0x59A" in what), None)


def _log(tmp: Path, *flags: str, grant: bool = True) -> str:
    wav = tmp / f"peer{'-grant' if grant else ''}.wav"
    if not wav.exists():
        _peer_wav(wav, slots=16, grant=grant)
    return _run(wav, tmp / "out", *flags)


def _session(tmp: Path, *flags: str, grant: bool = True) -> list[tuple[str, str]]:
    return _trace(_log(tmp, *flags, grant=grant))


def the_grant_reaches_the_hold_loop(tmp: Path) -> None:
    print("\nA scripted gateway that commands the upgrade, once per cycle")
    trace = _session(tmp, "--p1-grant-only")
    i = _first_grant(trace)
    check("the session decodes the scripted grant", i is not None,
          f"{sum(1 for _, w in trace if '0x59A' in w)} of them")
    if i is None:
        return
    check("...in the HOLD loop, which is where a granted link stands",
          trace[i][0] == "HOLD RX", f"read by the {trace[i][0]} path")

    after = [what for tag, what in trace[i + 1:] if tag == "tx"]
    keyed = after[0] if after else "nothing keyed after the grant"
    check("the very next burst is the entry packet -- speed level 1, an empty "
          "field, with the traffic the FSM had left to send still behind it",
          keyed.startswith("SL1 ENTRY") and _field(keyed) == 0, keyed)


def the_repeat_is_not_silence(tmp: Path) -> None:
    """The cycles after the entry packet, every one of which the peer answers.

    A granting gateway repeats `0x59A` until the entry packet reaches it, so the
    cycles the upgrade window is counted over each carry a twelve-bit codeword at
    zero bit errors, read at the anchor. `arq.UPGRADE_SILENCE_CYCLES` counts
    SILENCE, and the event reached `PtcHost._take_grant`, which is a no-op after
    the first grant and set nothing -- so the session spent that budget on cycles
    it had decoded in and reported "nothing decoded in 4 cycles". Measured on the
    air three times on 2026-08-22, at WS8EOC and at KB5LZK.

    What the peer is saying is now what the operator reads, and no more than
    that: repeating the previous codeword is how a receiver asks for the packet
    again (pactor3.md §7), so a station that demodulated our entry packet and
    would not take it asks in the same words as one that heard nothing. The line
    reports the request; it does not claim to know which.
    """
    print("\nThe peer answers every cycle after the entry packet")
    wav = tmp / "peer-retry-budget.wav"
    _peer_wav(wav, slots=24, grant=True)
    log = _run(wav, tmp / "out-retry-budget", "--p1-grant-only", hold=18)
    trace = _trace(log)
    i = _first_grant(trace)
    after = log.split("link upgraded to PACTOR-3", 1)[-1]
    ended = after.split("falls back to PACTOR-1", 1)[0]
    heard = ended.count("[host] rx unassigned: 0x59A")
    check("every cycle the upgrade window was counted over carried a codeword",
          heard >= UPGRADE_SILENCE_CYCLES, f"{heard} decoded before the fallback")
    verdict = "; ".join(ln.strip() for ln in log.splitlines()
                        if "since the upgrade" in ln)
    check("...so the session does not report those cycles as silent",
          "nothing decoded in" not in log, verdict)
    check("...it reports that we heard the peer, in the line that falls back",
          "it answered" in verdict and "zero bit errors" in verdict, verdict)
    check("and the entry retry budget still bounds the attempt",
          "entry retry budget exhausted" in log and "falls back to PACTOR-1" in log)
    if i is not None:
        keyed = [w for tag, w in trace[i + 1:] if tag == "tx"]
        check("...having given the peer more than one look at the new waveform, "
              "which is what memory-ARQ combines",
              sum(1 for w in keyed if w.startswith("SL1 ENTRY")) > 1,
              str(keyed[:6]))


def the_default_answers_the_grant(tmp: Path) -> None:
    """A session with nothing said about upgrades at all -- which is most of them.

    THE ONE THE MAIL PATH FLIES. `station.mail` prints a `shrike.onair` line with
    the mail flags and no level flags, so this is that line's upgrade behaviour,
    and until 2026-08-27 it was "decode the grant and discard it" -- at gateways
    like WS8EOC, which serves nothing below PACTOR-2, that was the whole session.

    `GRANT_ENTRY_SL` is 1 and `entry_sl` is 3, so a level-1 packet on the air is
    a grant that was acted on and nothing else can produce one.

    Longer than the other recordings because the default leaves the UNINVITED
    upgrade armed as well, and it fires first: it goes out at SL3 off the first
    acknowledged packet with traffic behind it, is contradicted, and falls back.
    The grant arrives at a link that has already been told PACTOR-3 is ruled out
    -- and it has to be taken anyway, because a peer saying it can read the
    waveform outranks our inference from an answer that never mentioned it.
    """
    print("\nThe same recording with no upgrade flags at all")
    wav = tmp / "peer-grant-long.wav"
    if not wav.exists():
        _peer_wav(wav, slots=24, grant=True)
    log = _run(wav, tmp / "out-default", hold=24)
    trace = _trace(log)
    check("the grant is decoded just the same",
          _first_grant(trace) is not None)
    check("...and the entry packet is keyed on it with no flag asking for it",
          bool(_entry(trace)),
          str([w for tag, w in trace if tag == "tx"][:6]))
    fell_back = log.find("falls back to PACTOR-1")
    granted = log.find("upgraded to PACTOR-3 at SL1 on the peer's grant")
    check("...after an uninvited attempt of its own was contradicted and given "
          "up on, so a grant reopens a door our own inference had shut",
          -1 < fell_back < granted, f"fall-back at {fell_back}, grant at {granted}")

    print("\n...and --decline-grant is what it takes to refuse one")
    trace = _trace(_run(wav, tmp / "out-decline", "--decline-grant", hold=24))
    check("the grant is decoded and nothing is keyed on it", not _entry(trace),
          str([w for tag, w in trace if tag == "tx"][:6]))

    print("\n...and under --pactor1-only, which is what the failed arm carried")
    trace = _session(tmp, "--pactor1-only")
    check("the link stays in PACTOR-1 through every grant -- the promise the "
          "channel sense reads as 1200-1800 Hz is kept",
          not [w for tag, w in trace if tag == "tx" and w.startswith("SL")],
          str([w for tag, w in trace if tag == "tx"][:5]))


def the_grant_is_the_only_door(tmp: Path) -> None:
    """A peer that acknowledges and never grants gets no PACTOR-3 under the flag.

    This is what `--pactor1-only` was reached for and could not do: the arm wants
    the uninvited upgrade suppressed so the grant is its one variable, and the
    flag that suppressed it also declined the grant.
    """
    print("\nA station that acknowledges every packet and never grants")
    trace = _session(tmp, "--p1-grant-only", grant=False)
    keyed = [w for tag, w in trace if tag == "tx"]
    check("no grant, so no PACTOR-3 -- the uninvited upgrade is refused under "
          "the flag", not [w for w in keyed if w.startswith("SL")],
          str(keyed[:6]))
    check("...and the same session without it does offer one",
          bool([w for tag, w in _session(tmp, grant=False)
                if tag == "tx" and w.startswith("SL")]),
          "the arm above is measuring a session that could not upgrade anyway")


def _refused(*flags: str) -> tuple[bool, str]:
    argv, err, out = sys.argv, io.StringIO(), io.StringIO()
    sys.argv = ["shrike.onair", "--replay", "peer.wav", "--mycall", "W9SSJ",
                "--dxcall", "K7ABC", *flags]
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            onair.main()
    except SystemExit as e:
        return e.code == 2 and "not allowed with" in err.getvalue(), err.getvalue()
    else:
        return False, err.getvalue()
    finally:
        sys.argv = argv


def the_parser_refuses_the_pair() -> None:
    print("\nThe flags that cannot both be honoured")
    for pair in (("--pactor1-only", "--p1-act-on-grant"),
                 ("--pactor1-only", "--p1-grant-only"),
                 ("--decline-grant", "--p1-grant-only")):
        refused, err = _refused(*pair)
        check(f"{' with '.join(pair)} is refused at the parser",
              refused, err.strip()[-70:] or "the session was accepted")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        the_grant_reaches_the_hold_loop(tmp)
        the_repeat_is_not_silence(tmp)
        the_default_answers_the_grant(tmp)
        the_grant_is_the_only_door(tmp)
        the_parser_refuses_the_pair()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


def test_mail_with_a_named_door_announces_and_takes_the_grant(tmp_path):
    """Mail defaults to P1; its explicit P3 path must work in the real loop, and
    the door alone has to put the announcement on the air -- nothing draws a
    grant without it."""
    flags = ("--mail-fetch", "--mail-out", str(tmp_path / "inbox"),
             "--p1-grant-only", "--p3-entry", "template")
    log = _log(tmp_path, *flags)
    assert "Announcement: --p1-status-bits45 3" in log, log
    trace = _trace(log)
    grant = _first_grant(trace)
    assert grant is not None, log
    assert not _entry(trace[:grant]), log
    after = [what for tag, what in trace[grant + 1:] if tag == "tx"]
    assert after and after[0].startswith("SL1 ENTRY"), log
    assert _field(after[0]) == 0, log
    assert "mail: stage awaiting greeting" in log
    # A peer that does not grant must not be forced into P3 by queued mail.
    no_grant = _log(tmp_path, *flags, grant=False)
    assert not _entry(_trace(no_grant)), no_grant


def test_an_arm_at_bits_zero_says_no_grant_can_be_drawn(tmp_path):
    """An open door at bits 0 is the shape that keys uninvited PACTOR-3 into a
    peer asking for packet #1, and it reads like a working upgrade arm."""
    for flags in (("--mail-fetch", "--mail-out", str(tmp_path / "inbox"),
                   "--p1-grant-only", "--p1-status-bits45", "0"),
                  ("--pactor3-only", "--p1-status-bits45", "0")):
        log = _log(tmp_path, *flags)
        said = next((ln for ln in log.splitlines()
                     if "Announcement: --p1-status-bits45 0" in ln), "")
        assert said, log
        assert "--p1-status-bits45 3 to request a grant" in said, said
        assert "upgrades enabled" in log, log


if __name__ == "__main__":
    sys.exit(main())
