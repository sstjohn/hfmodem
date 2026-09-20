# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A password spelled out beside its flag, in anything git is tracking.

On 2026-08-28 the operator's Winlink password went into a tee'd slot log. Nothing
typed it there: `tools/onair.sh` wrote its own `ps -o command=` into the rig claim
so the next launcher could say what was already running, and the next launcher
printed it back. The mechanism was two hops from anywhere anyone was looking, and
what makes it worth a gate is that neither hop is unusual — every launcher here
echoes something, and the next secret to be carried on a command line will take
the same route.

So this reads the index, not the working tree. An untracked scratch log is the
operator's business; a tracked one is in every clone, in the history for good,
and cannot be taken back by editing it. The index is also what makes the check
cheap enough to run every time: `git ls-files` names about five thousand files
here, and the ones that are not text are skipped by failing to decode.

WHAT IT LOOKS FOR is the SHAPE and never a value. A gate that held the password
itself would be the exposure it is guarding against, and it would go stale the
first time the account was rotated. `--mail-password <word>` is refused; what the
word is does not matter.

WHAT KEEPS IT OFF CORRECT WRITING, because the flag is documented in a dozen
places and a gate that flags documentation is a gate someone switches off:

  * **A shell reference or expansion.** `"$WINLINK_PASSWORD"`, `$(cat …)` and a
    backticked substitution all name a password without holding one.
  * **A bracketed placeholder.** `<YOUR-PASSWORD-FILE>` is what the `mail` verb
    prints when it has nothing safe to print, and `{…}` is the same idea.
  * **An argparse metavar.** `[--mail-password MAIL_PASSWORD]` is the usage line
    argparse writes for the flag, and it is in six committed session logs here
    for the ordinary reason that a run got its arguments wrong. ALL-CAPS with no
    lower-case letter in it is the discriminator, which also spares the flag
    named in prose ahead of a shouted word.
  * **The flag alone.** `--mail-password` at the end of a line, in backticks, or
    followed by another flag, is a mention and not a use.

`--mail-password-file <path>` is not this shape at all: the argument is a path,
the file's mode is the operator's business, and the hyphen after `password` means
the pattern never reaches it.

ONE FILE IS EXEMPT AND IT IS THIS ONE. The negative control below has to hold
lines of exactly the shape being refused, or it proves nothing. That is the whole
exemption — named here rather than left as a path list nobody reads, because an
unnamed exception is indistinguishable from a rule nobody enforces, and because
it is also the one place a real secret could be hidden from this scan.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]

#: A password-ish flag and the argument sitting next to it. Quoted runs are taken
#: whole so that `--mail-password 'two words'` is one argument and not one word.
_ARGUMENT = r"""(?P<arg>"[^"\n]*"|'[^'\n]*'|[^\s\n]+)"""
_SPELLED_OUT = re.compile(
    r"--(?:mail-)?(?:password|passwd|secret|token)(?:=|[ \t]+)" + _ARGUMENT)

#: Trailing syntax the argument picks up from the line it was written on: a usage
#: line's `]`, a sentence's `.`, a code fence's backtick.
_TRAILING = "]).,;:`\"'"

#: Names a password without carrying one.
_REFERENCE = ("$", "`", "<", "{", "*", "\\")


def spelled_out(text: str) -> list[tuple[int, str]]:
    """Every (line number, flag-and-argument) where a secret is written out."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in _SPELLED_OUT.finditer(line):
            arg = m.group("arg").strip(_TRAILING)
            if not arg or arg.startswith(_REFERENCE) or arg.startswith("--"):
                continue
            if arg.isupper():                     # an argparse metavar
                continue
            found.append((n, m.group(0)))
    return found


def _tracked() -> list[str]:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                           capture_output=True, text=True, timeout=60)
    except OSError as exc:                        # no git binary on this machine
        pytest.skip(f"cannot read the index: git would not run ({exc})")
    if r.returncode != 0:
        pytest.skip(f"cannot read the index: git found no repository at {REPO} "
                    f"({r.stderr.strip() or 'no message'})")
    return sorted(set(r.stdout.split("\0")) - {""})


#: See the last paragraph of the module docstring.
_EXEMPT = Path(__file__).resolve().relative_to(REPO).as_posix()


def test_no_tracked_file_spells_out_a_password():
    caught = []
    for rel in _tracked():
        if rel == _EXEMPT:
            continue
        p = REPO / rel
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):     # gone, or not text
            continue
        caught += [f"{rel}:{n}: {hit}" for n, hit in spelled_out(text)]
    assert not caught, (
        f"{len(caught)} tracked line(s) carry a secret beside its flag. Editing "
        "the file does not undo this: it is in the history, in every clone, and "
        "on whatever the branch was pushed to. Rotate the credential first, then "
        "redact here, then decide about the history:\n  " + "\n  ".join(caught)
        + "\nNothing needs to reach a command line at all — "
        "`--mail-password-file FILE` and `WINLINK_PASSWORD` are both honoured by "
        "every tool here, and `tools/onair.sh` strips the flag before any verb "
        "sees it.")


def test_the_scan_catches_a_password_that_is_there():
    """The negative control. Without it this file passes by finding nothing,
    which is also what it does when the pattern has quietly stopped matching."""
    planted = [
        "./tools/onair.sh ardop-mail W6IDS 7061500 500 --mail-password s3kr1t-pw",
        "python -m hfmodem.shrike.onair --mail-password=s3kr1t-pw --transmit",
        "kestrel_connect.py --mail-password 'two words here'",
        "curl -H auth --token abcd1234",
    ]
    for line in planted:
        assert spelled_out(line), line


def test_the_scan_leaves_the_documented_flag_alone():
    """Every one of these is real text from this tree, and a gate that fired on
    any of them would be turned off within the week."""
    for line in [
        "usage: kestrel_connect.py [-h] [--mail-password MAIL_PASSWORD]",
        '  --mail-password "$WINLINK_PASSWORD" --serial "$CAT_PORT"',
        '  --mail-password "$(cat ~/.winlink-pw)" \\',
        "  --mail-password-file ~/.winlink-pw \\",
        "  --mail-password-file <YOUR-PASSWORD-FILE>",
        "4. **`--mail-password` must be spelled out.** The `WINLINK_PASSWORD`",
        "#                        `--mail-password` STILL WORKS and is a mistake",
    ]:
        assert not spelled_out(line), line
