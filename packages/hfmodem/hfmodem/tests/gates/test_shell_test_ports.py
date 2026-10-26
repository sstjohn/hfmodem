# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Every shell test in tools/tests starts inside tools/tests/hermetic.sh.

The launchers fill in what a test leaves out from the station itself: unset
ports fall back to the rig profile's, HAMLIB_BIN puts a real rigctl on PATH, and
a sound card is whatever sounddevice finds. test_onair_gate.sh ran its pounce
under `env -u CAT_PORT`, which rewrote the station's claim in /tmp and could have
gone on into a real VARA pounce. hermetic.sh resets the environment and puts
tripwires in front of everything that reaches the station, so what is left to
check here is that each test is inside it from its first line and never hands a
station value back to anything it runs.
"""
import re
import shlex
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[5] / "tools" / "tests"
SCRIPTS = sorted(TESTS.glob("test_*.sh"))
SOURCE = 'source "$(dirname "${BASH_SOURCE[0]}")/hermetic.sh"'
HELD = r"CAT_PORT|PTT_PORT|HFMODEM_\w+|OPERATOR_DATA|HAMLIB_BIN|PATH|PYTHONPATH|HOME|RIG_CLAIMED|WORK|HERMETIC_\w+"
SETS = re.compile(rf"(?<![\w$])({HELD})=(\S*)")
LOCAL = ('"$WORK', '$WORK', '"$FAKECAT"', '"$FAKEPTT"', '/dev/null')
KEEPS = {"PATH": "$PATH", "PYTHONPATH": "$PYTHONPATH"}
DROPS = re.compile(rf"\benv\s+(-\w*i\b|--ignore-environment)|\benv\b[^\n]*\s-u\s*({HELD})\b"
                   rf"|\bexport\s+-n\b|\bhermetic_exit\s*\(\)|\bfunction\s+hermetic_exit\b"
                   rf"|\bunset\b[^;\n]*\bhermetic_exit\b")
TRAP = re.compile(r"(?:^|[;&|(){}]|\b(?:then|do|else))\s*trap\b(.*)")
ENDS = {";", "&", "|", "&&", "||", ";;"}
UNSETS = re.compile(r"\bunset\s+([^;|&)]*)")
MAY_UNSET = {"RIG_CLAIMED", "HFMODEM_STATION"}

pytestmark = pytest.mark.skipif(not SCRIPTS, reason=f"no shell tests at {TESTS} (installed-wheel run)")


def _code(script):
    lines = script.read_text().splitlines()[1:]
    return [line for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _replaces_the_exit_trap(line):
    """A `trap` in command position whose signals include EXIT, however spelt."""
    for found in TRAP.finditer(line):
        try:
            words = list(shlex.shlex(found.group(1), posix=True, punctuation_chars=";&|"))
        except ValueError:
            return True
        words = words[:next((i for i, w in enumerate(words) if w in ENDS), len(words))]
        if any(re.fullmatch(r"EXIT|exit|0+", w) for w in words[1:]):
            return True
    return False


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_shell_test_is_hermetic_from_its_first_line(script):
    code = _code(script)
    assert code and code[0] == SOURCE, f"{script.name} runs something before hermetic.sh"
    assert SOURCE not in code[1:], f"{script.name} starts hermetic.sh twice"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_shell_test_never_hands_back_a_station_value(script):
    for line in _code(script)[1:]:
        assert not DROPS.search(line), f"{script.name} resets what hermetic.sh set: {line}"
        assert not _replaces_the_exit_trap(line), f"{script.name} replaces hermetic.sh's exit trap: {line}"
        for words in UNSETS.findall(line):
            held = [w for w in words.split() if re.fullmatch(HELD, w) and w not in MAY_UNSET]
            assert not held, f"{script.name} unsets {held}: {line}"
        for var, value in SETS.findall(line):
            assert value.startswith(LOCAL), f"{script.name} sets {var} to {value or 'nothing'}: {line}"
            assert KEEPS.get(var, "") in value, f"{script.name} drops hermetic.sh's {var}: {line}"
