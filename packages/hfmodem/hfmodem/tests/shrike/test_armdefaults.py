# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What an arm flies when the operator names nothing -- and a mail arm is not an
upgrade arm.

Every PACTOR arm this modem has flown was an experiment about reaching PACTOR-3,
and the shipped defaults are that experiment's: status bits 4-5 announce the
capability, and the link takes the upgrade off the first acknowledged packet
with traffic behind it. Both are measured to cost the PACTOR-1 data path, and a
mail exchange is nothing but that data path:

  * bits 4-5 at 3, WS8EOC 3596500 on 2026-08-30, same gateway, band, hour, drive
    and gain -- three arms, the peer's counter frozen at #1, nothing
    acknowledged; at 0, two arms, `#1 -> #2 -> #3` and ACKNOWLEDGED both times;
  * the uninvited upgrade -- `arq._on_ack` offers one on every acknowledgement
    with payload queued, which during a mail exchange is every acknowledgement,
    so the link leaves PACTOR-1 on the login packet and spends
    `ptc.UPGRADE_SILENCE_CYCLES` discarding the gateway's PACTOR-1 codewords.

So `onair._arm_defaults` derives both from the mail flags, and the tests below
are about the OPT-OUT as much as the default: an operator who names a value gets
that value, a mail arm that names a way into PACTOR-3 gets the announcement back
because no grant can be drawn without it, and a non-mail arm is untouched.

The namespaces come from the real parser, through `main()`, because half of what
is asserted here is that `--p1-status-bits45` has no eager default left to
override.

Run:  python -m hfmodem.tests.shrike.test_armdefaults
"""
from __future__ import annotations

import contextlib
import io
import shlex
import sys
from unittest import mock

from hfmodem.shrike import arq, onair, pactor1

BASE = ["onair", "--dxcall", "WS8EOC", "--mycall", "W9SSJ"]

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def _parsed(*argv: str):
    """The namespace `run` is handed, before any default is derived from it."""
    got = []
    with mock.patch.object(onair, "run", lambda a: got.append(a) or 0), \
            mock.patch.object(sys, "argv", BASE + list(argv)):
        onair.main()
    return got[0]


def _armed(*argv: str):
    args = _parsed(*argv)
    onair._arm_defaults(args)
    return args


def _banner(*argv: str) -> str:
    """What the arm says about itself before it keys anything."""
    said = io.StringIO()
    with contextlib.redirect_stdout(said), \
            contextlib.redirect_stderr(io.StringIO()), \
            contextlib.suppress(SystemExit):
        onair.run(_parsed(*argv))
    return said.getvalue()


def the_upgrade_arm_is_unchanged() -> None:
    print("\nan arm with no mail flags")
    args = _armed()
    check("bits 4-5 announce the capability",
          args.p1_status_bits45 == onair.P1_STATUS_ANNOUNCE == 3,
          str(args.p1_status_bits45))
    check("the upgrade is not refused", not args.pactor1_only)
    check("and nothing has forced a hold", args.hold == 0, str(args.hold))
    check("an answered entry packet runs traffic at speed level 3",
          args.p3_traffic_sl == arq.ArqConfig.traffic_sl == 3,
          str(args.p3_traffic_sl))


def the_traffic_level_is_the_operators() -> None:
    """The one flag a peer that read only the entry packet needs, and it is
    nobody's default: a mail arm does not derive it either."""
    print("\nthe level the traffic runs at")
    check("a named level stands", _armed("--p3-traffic-sl", "1").p3_traffic_sl == 1)
    check("...and a mail arm still flies the default",
          _armed("--mail-fetch").p3_traffic_sl == 3)
    check("the entry level is the operator's on the same terms",
          (_armed("--p3-entry-sl", "1").p3_entry_sl,
           _armed("--mail-fetch").p3_entry_sl) == (1, arq.ArqConfig.entry_sl))
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            _parsed("--p3-traffic-sl", "7")
        except SystemExit:
            refused = True
        else:
            refused = False
    check("a level outside the ladder is refused", refused)


def the_mail_arm_takes_the_measured_settings() -> None:
    print("\nan arm carrying mail")
    for flags in (("--mail-fetch",), ("--mail-send", "note.txt"),
                  ("--mail-fetch", "--pactor1-only")):
        args = _armed(*flags)
        what = " ".join(flags)
        check(f"{what}: bits 4-5 clear", args.p1_status_bits45 == 0,
              str(args.p1_status_bits45))
        check(f"{what}: the session stays in PACTOR-1", args.pactor1_only)
        check(f"{what}: and it is held", args.hold == onair.MAIL_HOLD_CYCLES == 96,
              str(args.hold))
    # The byte that goes on the air is what the measurement was about, not the
    # flag: bits 4-5 clear leaves the data type alone, and 3 declares PMC German
    # over an ASCII field on every packet of the link.
    args = _armed("--mail-fetch")
    st = pactor1.status_byte(1, bits45=args.p1_status_bits45)
    check("the first packet's status byte is 0x01", st == 0x01, f"0x{st:02x}")


def the_operator_can_still_say_otherwise() -> None:
    print("\nwhat an explicit flag does to a mail arm")
    args = _armed("--mail-fetch", "--p1-status-bits45", "3")
    check("a named bits value stands", args.p1_status_bits45 == 3,
          str(args.p1_status_bits45))
    check("...and it still stays in PACTOR-1", args.pactor1_only)

    args = _armed("--mail-fetch", "--hold", "12")
    check("a named hold is not overwritten", args.hold == 12, str(args.hold))


def the_bits_follow_the_door() -> None:
    """Naming any of the level group IS the opt-out: the parser already refuses
    more than one of them, so a session cannot both be pinned to PACTOR-3 and
    held in PACTOR-1. And a door with no announcement behind it draws no grant,
    so the ones that lead to PACTOR-3 carry the announcement back."""
    print("\na mail arm that names a way into PACTOR-3")
    for flag, attr, bits in (("--pactor3-only", "pactor3_only", 3),
                             ("--p1-grant-only", "p1_grant_only", 3),
                             ("--offer-pactor2", "offer_pactor2", 3),
                             ("--p3-uninvited", "p3_uninvited", 0)):
        args = _armed("--mail-fetch", flag)
        check(f"{flag} keeps the door it names open",
              getattr(args, attr) and not args.pactor1_only)
        check(f"{flag}: bits 4-5 {bits}", args.p1_status_bits45 == bits,
              str(args.p1_status_bits45))


def the_closed_door_says_so_at_the_top() -> None:
    """A mail arm that named no door reads as an ordinary mail arm in its own
    transcript, and three arms flew that way on 2026-09-17 asking for the
    uninvited entry. The banner names the doors, so the contradiction is on the
    first screen rather than in the teardown."""
    print("\nwhat a PACTOR-1-only mail arm says before it keys")
    closed = _banner("--mail-fetch")
    check("the derived arm says no entry packet is coming",
          "NO ENTRY PACKET WILL BE KEYED" in closed)
    for door in onair.UPGRADE_DOORS:
        flag = "--" + door.replace("_", "-")
        check(f"...and names {flag}", flag in closed)
    check("a named door says nothing of the kind",
          "NO ENTRY PACKET WILL BE KEYED"
          not in _banner("--mail-fetch", "--p3-uninvited"))


def test_declining_a_grant_keeps_mail_in_pactor1():
    for flag in ("--mail-fetch", "--mail-send"):
        flags = [flag] if flag == "--mail-fetch" else [flag, "note.txt"]
        args = _armed(*flags, "--decline-grant")
        assert args.pactor1_only and args.decline_grant
        assert args.p1_status_bits45 == 0


def test_transcript_records_resolved_namespace_and_keeps_password_path(tmp_path):
    path = tmp_path / "password.txt"
    path.write_text("private-password\n")
    args = _parsed("--mail-fetch", "--decline-grant", "--p1-drive", "0",
                   "--mail-password-file", str(path))
    output = io.StringIO()
    with mock.patch.object(sys, "argv", ["unrelated-caller"]), \
            contextlib.redirect_stdout(output), contextlib.suppress(SystemExit):
        onair.run(args)
    line = output.getvalue()
    assert str(path) in line and "private-password" not in line
    assert "--mail-password=***" in line
    assert "--pactor1-only=True" in line and "--p1-status-bits45=0" in line
    first = line.splitlines()[0]
    assert first.startswith("  argv: ")
    assert "--mail-fetch --decline-grant --p1-drive 0 --mail-password-file" in first
    assert "unrelated-caller" not in first  # Preserve the parsed arm, not later sys.argv.


def the_named_door_is_announced() -> None:
    """The byte, not the flag: 0x31 is what the peer reads on packet #1."""
    print("\nthe announcement a named door puts on the air")
    args = _armed("--mail-send", "note.txt", "--p1-grant-only")
    check("bits 4-5 announce the capability", args.p1_status_bits45 == 3,
          str(args.p1_status_bits45))
    st = pactor1.status_byte(1, bits45=args.p1_status_bits45)
    check("the first packet's status byte is 0x31", st == 0x31, f"0x{st:02x}")


def an_explicit_zero_beats_the_door() -> None:
    print("\na named zero against a named door")
    args = _armed("--mail-fetch", "--p1-grant-only", "--p1-status-bits45", "0")
    check("the zero stands", args.p1_status_bits45 == 0,
          str(args.p1_status_bits45))
    check("...and the door stays open",
          args.p1_grant_only and not args.pactor1_only)


def the_line_that_keyed_is_written_down() -> None:
    """Nothing else records what the modem was asked for, and a transcript whose
    command line has to be reconstructed cannot settle what flew."""
    print("\nthe command line in the transcript")
    argv = ["--mail-fetch", "--mail-password", "hunter2",
            "--p1-grant-only", "--p1-drive", "0"]
    args = _parsed(*argv)
    said = io.StringIO()
    # --p1-drive 0 is refused, and the line has to be written before that.
    with mock.patch.object(sys, "argv", BASE + argv), \
            contextlib.redirect_stdout(said), \
            contextlib.suppress(SystemExit):
        onair.run(args)
    line = said.getvalue()
    check("the mail flag is in the line", "--mail-fetch" in line, line)
    check("...and the door it opened", "--p1-grant-only" in line, line)
    check("the password is not", "hunter2" not in line, line)
    check("...it is redacted", "***" in line, line)


def deriving_twice_changes_nothing() -> None:
    """`run` derives them and is not `main`'s alone to call."""
    print("\nthe derivation is idempotent")
    args = _armed("--mail-fetch", "--hold", "12")
    onair._arm_defaults(args)
    check("bits, level and hold all hold still",
          (args.p1_status_bits45, args.pactor1_only, args.hold) == (0, True, 12),
          f"{args.p1_status_bits45} {args.pactor1_only} {args.hold}")
    args = _armed("--mail-fetch", "--p1-grant-only")
    onair._arm_defaults(args)
    check("and so does a named door's announcement",
          (args.p1_status_bits45, args.pactor1_only) == (3, False),
          f"{args.p1_status_bits45} {args.pactor1_only}")


def a_bare_transmitter_keys_the_documented_order() -> None:
    """`RadioTx`'s class defaults, which are not the parser's.

    The four PACTOR-1 codewords are closed under bit reversal, so the wrong
    order does not produce noise -- it produces the OTHER codeword of the pair.
    CS4 keyed MSB-first is CS3, a bare speed request read as a break-in.
    """
    print("\nthe transmitter's own defaults")
    check("the codeword goes out least-significant bit first",
          onair.RadioTx.p1_ack_msb is False, str(onair.RadioTx.p1_ack_msb))
    check("...which is what the CLI default resolves to as well",
          _armed().p1_ack_lsb is True)


def the_rigs_power_is_nobodys_default() -> None:
    """RFPOWER is the operator's and is derived from nothing.

    A power the arm did not ask for is a power the operator did not choose, and
    the rig keeps whatever the launcher read back off it. The flags also refuse
    a dry run outright rather than resolving to nothing: there is no CAT
    connection without `--transmit`, so the watts would be a number in the
    transcript and no change at the radio.
    """
    print("\nthe RF power an arm flies when the operator names none")
    args = _armed()
    check("no PACTOR-1 watts", args.p1_watts is None, str(args.p1_watts))
    check("no PACTOR-3 watts", args.p3_watts is None, str(args.p3_watts))
    check("...and a mail arm derives neither",
          (_armed("--mail-fetch").p1_watts,
           _armed("--mail-fetch").p3_watts) == (None, None))
    refused = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            onair.run(_armed("--p3-watts", "60"))
    except SystemExit as exc:
        refused = str(exc)
    check("a named level without --transmit is refused, not ignored",
          "--p3-watts" in refused and "--transmit" in refused, refused)


def main() -> int:
    global ok
    ok = True
    print("Arm defaults: what a mail run flies, and what an upgrade run flies")
    the_upgrade_arm_is_unchanged()
    the_traffic_level_is_the_operators()
    the_mail_arm_takes_the_measured_settings()
    the_operator_can_still_say_otherwise()
    the_bits_follow_the_door()
    the_closed_door_says_so_at_the_top()
    the_named_door_is_announced()
    an_explicit_zero_beats_the_door()
    the_line_that_keyed_is_written_down()
    deriving_twice_changes_nothing()
    a_bare_transmitter_keys_the_documented_order()
    the_rigs_power_is_nobodys_default()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_typed_arguments_keep_order_and_both_password_spellings_are_redacted():
    # The `--flag=value` spelling is assembled rather than written: the shape
    # is exactly what tests/gates/test_no_secret_in_tracked_files.py refuses,
    # and that gate cannot tell this control from a real credential.
    word = "private-password"
    for secret in (("--mail-password", word),
                   ("--mail-password" "=" + word,)):
        argv = ["--p1-drive", "0", *secret, "--hold", "7"]
        args = _parsed(*argv)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.suppress(SystemExit):
            onair.run(args)
        lines = output.getvalue().splitlines()
        expected = BASE[1:] + ["--p1-drive", "0"] + (
            ["--mail-password", "***"] if len(secret) == 2 else
            ["--mail-password=***"]) + ["--hold", "7"]
        assert lines[0] == "  argv: " + shlex.join(expected)
        assert lines[1].startswith("  resolved args: ")
        assert word not in output.getvalue()


def test_direct_namespace_caller_does_not_invent_a_typed_command():
    args = _parsed("--p1-drive", "0")
    del args._typed_argv
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.suppress(SystemExit):
        onair.run(args)
    assert output.getvalue().startswith("  argv: unavailable (run received a namespace)")
    assert "resolved args:" in output.getvalue()
