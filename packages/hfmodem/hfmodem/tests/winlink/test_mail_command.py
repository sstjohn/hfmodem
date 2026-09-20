# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The mail verb's handling of the Winlink password.

The secure-login password is a secret with two ways out: the argv of a process,
readable in `ps` by any local user for the life of the run, and the printed
on-air command, which lands in scrollback and redirected session logs. Neither
carries the literal. `WINLINK_PASSWORD` and `--password-file` are the two ways
that keep it off argv, and the printed command names whichever one was used.

A password given as a bare `--password` has neither behind it, and the printed
line then carries an unfilled placeholder. That is a false record, deliberately:
the alternative false record is `"$WINLINK_PASSWORD"` with nothing exported
behind it, which expands to nothing when pasted and spends one of a finite
number of login attempts on an empty password. A placeholder fails where it can
be seen.
"""
from types import SimpleNamespace

from hfmodem.station.mail import _onair_command


def _args(**kw):
    base = dict(send=[], fetch=True, to="", subject="", password="",
                password_file="", sid="", out="logs/mail", protocol="pactor",
                mycall="W9SSJ", gateway="K7ABC", freq=0, bandwidth=500)
    base.update(kw)
    return SimpleNamespace(**base)


def test_an_env_supplied_password_prints_as_the_reference(monkeypatch):
    secret = "correct horse battery"
    monkeypatch.setenv("WINLINK_PASSWORD", secret)
    for protocol in ("pactor", "ardop", "vara"):
        line = _onair_command(_args(protocol=protocol, password=secret))
        assert secret not in line, protocol
        assert line.endswith('--mail-password "$WINLINK_PASSWORD"'), protocol


def test_a_file_supplied_password_prints_the_file(tmp_path):
    """The way in that keeps the secret off argv at both ends: the session gets
    the password, and the printed line names the file rather than its content."""
    pwfile = tmp_path / "winlink-pw"
    pwfile.write_text("correct horse battery\n")
    for protocol in ("pactor", "ardop", "vara"):
        line = _onair_command(_args(protocol=protocol, password="correct horse battery",
                                    password_file=str(pwfile)))
        assert "correct horse battery" not in line, protocol
        assert line.endswith(f"--mail-password-file {pwfile}"), protocol


def test_a_flag_supplied_password_prints_no_literal(monkeypatch):
    """A bare `--password` leaves nothing true to print. The line carries an
    unfilled placeholder, which cannot be pasted unnoticed — where the literal
    it replaces would have put the password into every log holding this run."""
    monkeypatch.delenv("WINLINK_PASSWORD", raising=False)
    secret = "correct horse battery"
    for protocol in ("pactor", "ardop", "vara"):
        line = _onair_command(_args(protocol=protocol, password=secret))
        assert secret not in line, protocol
        assert "$WINLINK_PASSWORD" not in line, protocol
        assert line.endswith("--mail-password-file <YOUR-PASSWORD-FILE>"), protocol


def test_the_pactor_line_names_the_ports_the_arm_gate_demands():
    """`shrike.onair --transmit` dies at "NOT KEYING: --serial is required"
    without its ports, and --serial has no shipped default on purpose: a
    default once put 22.1 s of reported carrier into a radio nobody had
    opened. So the printed line carries the ports the same way it carries the
    password — as shell references, here to CAT_PORT and PTT_PORT, the
    environment contract every launcher in tools/ answers to. Unset, they
    expand to nothing and the arm gate still refuses by name; a literal
    device path would be right on one machine and silently wrong at the next
    rig."""
    line = _onair_command(_args(freq=7101500))
    assert "-m hfmodem.shrike.onair" in line and "--transmit" in line
    assert '--serial "$CAT_PORT"' in line, line
    assert '--ptt-port "$PTT_PORT"' in line, line
    # The launcher verbs own their ports themselves; the references belong to
    # the one line that reaches shrike directly.
    for protocol in ("ardop", "vara"):
        assert "$CAT_PORT" not in _onair_command(_args(protocol=protocol))


def test_the_pactor_line_can_leave_pactor_1():
    """The printed line has to be able to reach the gateways that carry mail.

    Winlink's PACTOR channels mostly want a level above 1 -- WS8EOC answers a
    PACTOR-1 caller with "This RMS channel is set to only accept Pactor levels
    above 2 - Disconnecting" -- and the only invitation into PACTOR-3 this
    station has ever been sent is the peer's `0x59A`. Answering it was an opt-in
    until 2026-08-27 and this line never carried the flag, so 31 PACTOR mail arms
    decoded the grant, logged it and discarded it: the link could not leave
    PACTOR-1 by construction, whatever the band did.

    So the claim is about what the line does NOT say. It names no level policy at
    all, and the defaults it therefore inherits are the ones that go as far as
    the peer allows -- which is where this is checked, because a line that
    carries no flag is only as good as the default behind it.
    """
    from hfmodem.shrike.arq import ArqConfig, ENTRY_RUNGS_GROUNDED
    from hfmodem.shrike.ptc import PtcHost

    line = _onair_command(_args(freq=7101500))
    assert "--decline-grant" not in line, line
    assert "--pactor1-only" not in line, line

    host = PtcHost(mycall="W9SSJ")
    assert host.p1_act_on_grant, "the line carries no flag; the default is the arm"
    assert not set(ArqConfig().entry_ladder) & set(ENTRY_RUNGS_GROUNDED), \
        "and the entry packets it would key are ones this station can key"


def test_no_password_means_no_password_flag():
    for protocol in ("pactor", "ardop", "vara"):
        assert "--mail-password" not in _onair_command(_args(protocol=protocol))


def test_a_rehearsed_client_type_is_carried_to_the_air():
    """A rehearsal that announced something other than the station file's
    client type must print the line that announces the same thing — and,
    unoverridden, must print no flag at all, so the on-air tool keeps reading
    the file and the line stays true after the file is edited."""
    for protocol in ("pactor", "ardop", "vara"):
        line = _onair_command(_args(protocol=protocol, sid="Sparrowhawk-3.2"))
        assert "--mail-sid Sparrowhawk-3.2" in line, protocol
        assert "--mail-sid" not in _onair_command(_args(protocol=protocol))


def test_the_environment_supplies_the_password(tmp_path, capsys, monkeypatch):
    """WINLINK_PASSWORD reaches the session without ever touching argv, and
    the on-air command printed at the end still shows only the reference."""
    from hfmodem import cli
    monkeypatch.setenv("WINLINK_PASSWORD", "swordfish")
    rc = cli.main(["mail", "--gateway", "K7ABC", "--protocol", "pactor",
                   "--mycall", "W9SSJ", "--fetch",
                   "--out", str(tmp_path / "in")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "swordfish" not in out
    assert '--mail-password "$WINLINK_PASSWORD"' in out
