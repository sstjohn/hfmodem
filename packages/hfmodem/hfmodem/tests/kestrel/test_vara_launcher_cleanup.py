# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Exercise the launcher's actual cleanup function with no device access."""
from pathlib import Path
import re
import subprocess

import pytest

_LAUNCHER = Path(__file__).resolve().parents[5] / "tools/onair.sh"

pytestmark = pytest.mark.skipif(not _LAUNCHER.exists(),
                                reason="tools/onair.sh not present (installed-wheel run)")


@pytest.mark.parametrize("rc", (0, 1))
@pytest.mark.parametrize("stop_ok", (False, True))
def test_vara_releases_cat_before_power_and_preserves_unkey_check(rc, stop_ok):
    result = _cleanup("kestrel_connect.py", rc, stop_ok)
    lines = result.stdout.splitlines()
    assert lines[0] == "stop-daemon"
    assert ("power" in lines) == stop_ok
    assert lines[-2:] == ["check-unkey", "stop-witness"]
    assert result.returncode == (rc if stop_ok else 1)


def test_pactor_cleanup_does_not_stop_a_daemon_it_did_not_start():
    result = _cleanup("shrike onair", 0, False)
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["power", "check-unkey", "stop-witness"]


def _cleanup(tool, rc, stop_ok):
    source = _LAUNCHER.read_text()
    function = re.search(r"(?ms)^unkey_and_exit\(\).*?^}", source)
    assert function is not None
    script = """set -e
refuse_usage_error() { :; }
rigctld_stop() { echo stop-daemon; return "$STOP_RC"; }
rig_power() { echo power; }
prove_unkeyed() { echo check-unkey; return 1; }
witness_stop() { echo stop-witness; }
""" + function.group() + '\nunkey_and_exit "$1" 7100500 "$2"\n'
    return subprocess.run(
        ["bash", "-c", script, "cleanup-test", str(rc), tool],
        env={"STOP_RC": "0" if stop_ok else "1"},
        capture_output=True, text=True, timeout=5,
    )
