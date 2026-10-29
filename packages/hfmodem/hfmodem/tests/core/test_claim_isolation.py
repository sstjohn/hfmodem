# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import hfmodem

ROOT = Path(hfmodem.__file__).resolve().parents[3]
PORTS = ROOT / "tools/lib/ports.sh"

HOLD = """
import json, os
from hfmodem.core import claim
descriptor = claim.acquire("test", "isolation",
                           devices=claim.Devices(cat="none", capture="hw:99,0", playback="hw:99,0"))
print(json.dumps(dict(pid=os.getpid(), path=str(claim.claim_path()),
                      held=claim.claim_path().read_text())))
claim.release(descriptor)
"""

# Every Python child resolves its home here instead of the operator's, so a guard
# that fails open still lands in the test's own directory. Anything a child does
# under the real state directory regardless is refused and written to REAL_TOUCHES:
# the operator's own claims there are theirs, and never this test's concern.
FAKE_HOME = """
import os, pwd, sys
_real = pwd.getpwuid
guarded = [os.path.realpath(path) for path in (
    os.path.join(_real(os.getuid()).pw_dir, ".local/state/hfmodem"), os.environ.get("ALSO_GUARDED"))
    if path]
pwd.getpwuid = lambda uid: pwd.struct_passwd((*_real(uid)[:5], os.environ["FAKE_HOME"], _real(uid)[6]))
PATHS = {"open": 1, "os.mkdir": 1, "os.remove": 1, "os.rmdir": 1, "os.chmod": 1,
         "os.truncate": 1, "os.utime": 1, "os.rename": 2, "os.replace": 2, "os.link": 2,
         "os.symlink": 2}

def _guarded(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    path = os.path.realpath(os.fsdecode(path))
    return any(path == root or path.startswith(root + os.sep) for root in guarded)

def _refuse(event, args):
    touched = [path for path in args[:PATHS.get(event, 0)] if _guarded(path)]
    if touched:
        with open(os.environ["REAL_TOUCHES"], "a") as log:
            log.write(f"{os.getpid()} {event} {touched}\\n")
        raise PermissionError(f"a test child reached the operator's state directory: {touched}")

sys.addaudithook(_refuse)
"""


@pytest.fixture
def child_env(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(FAKE_HOME)
    home = tmp_path / "home"
    home.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    ps = binaries / "ps"
    ps.write_text(f"#!{sys.executable}\nimport sys\n"
                  "print('Sat Oct 17 12:00:00 2026' if 'lstart=' in sys.argv else 'isolation-test')\n")
    ps.chmod(0o700)
    touches = tmp_path / "real-touches"
    env = {key: value for key, value in os.environ.items()
           if key not in ("HFMODEM_STATION", "HFMODEM_RIG")}
    env.update(FAKE_HOME=str(home), REAL_TOUCHES=str(touches), HFMODEM_PYTHON=sys.executable,
               PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               CAT_PORT=str(tmp_path / "no-cat"), PTT_PORT=str(tmp_path / "no-ptt"),
               PYTHONPATH=os.pathsep.join([str(site), str(Path(hfmodem.__file__).parents[1])]))
    yield home, env
    assert not touches.exists(), touches.read_text()


def test_a_child_is_refused_what_it_would_touch_there(child_env, tmp_path):
    home, env = child_env
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    run = subprocess.run([sys.executable, "-c", "import sys; open(sys.argv[1], 'w')",
                          str(guarded / "rig.claim")], env=dict(env, ALSO_GUARDED=str(guarded)),
                         capture_output=True, text=True, timeout=60)
    assert "PermissionError" in run.stderr, run
    assert not (guarded / "rig.claim").exists()
    touches = Path(env["REAL_TOUCHES"])
    assert "rig.claim" in touches.read_text()
    touches.unlink()


def test_a_spawned_acquisition_claims_only_the_isolated_state(child_env):
    home, env = child_env
    result = subprocess.run([sys.executable, "-c", HOLD], env=env, capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout.splitlines()[-1])
    isolated = Path(os.environ["HFMODEM_STATE_DIR"]).resolve()
    assert isolated in Path(seen["path"]).resolve().parents
    assert seen["held"].startswith(f"{seen['pid']} ")
    assert not (home / ".local").exists()


# The last leaves a shell test's child nothing but hermetic.sh's own mark.
DROPS = [("HFMODEM_COEXIST_HARNESS",), ("HFMODEM_STATE_DIR",),
         ("HFMODEM_COEXIST_HARNESS", "HFMODEM_STATE_DIR"),
         ("HFMODEM_COEXIST_HARNESS", "HFMODEM_STATE_DIR", "PYTEST_CURRENT_TEST")]


def _refused(run):
    assert run.returncode == 1, run
    assert "refusing the operator's state directory" in run.stderr, run
    assert "CLAIM ACQUIRED" not in run.stdout


@pytest.mark.parametrize("dropped", DROPS)
def test_a_child_that_drops_the_isolation_fails_closed(child_env, tmp_path, dropped):
    home, env = child_env
    for key in dropped:
        env.pop(key)
    held = tmp_path / "held"
    held.touch()
    cli = [sys.executable, "-m", "hfmodem.core.claim"]
    _refused(subprocess.run([*cli, "path"], env=env, capture_output=True, text=True, timeout=60))
    _refused(subprocess.run(["bash", "-c", 'exec 9<>"$0"; exec "$@"', str(held), *cli,
                             "acquire", "9", "--kind", "test", "--operation", "guard", "--cat", "none"],
                            env=env, capture_output=True, text=True, timeout=60))
    assert held.read_bytes() == b""
    assert not (home / ".local").exists()


@pytest.mark.skipif(not PORTS.exists(), reason=f"{PORTS} is not present (installed-wheel run)")
@pytest.mark.parametrize("dropped", DROPS)
def test_the_shell_claim_helper_fails_closed_in_such_a_child(child_env, dropped):
    home, env = child_env
    for key in dropped:
        env.pop(key)
    _refused(subprocess.run(["bash", "-c", 'source "$0" && claim_rig guard', str(PORTS)],
                            env=env, capture_output=True, text=True, timeout=60))
    assert not (home / ".local").exists()


@pytest.mark.skipif(not PORTS.exists(), reason=f"{PORTS} is not present (installed-wheel run)")
def test_every_shell_test_isolates_its_claims():
    """Every shell test starts inside hermetic.sh (gates/test_shell_test_ports.py
    holds them to it), and hermetic.sh exports both halves of the isolation."""
    base = (PORTS.parents[1] / "tests" / "hermetic.sh").read_text()
    assert 'HFMODEM_STATE_DIR="$WORK/claim-state" HFMODEM_COEXIST_HARNESS=shell' in base


@pytest.mark.skipif(not PORTS.exists(), reason=f"{PORTS} is not present (installed-wheel run)")
@pytest.mark.parametrize("failing", ["path", "path --mutex"])
def test_the_shell_claim_opens_nothing_when_its_paths_do_not_resolve(child_env, tmp_path, failing):
    home, env = child_env
    claim = tmp_path / "rig.claim"
    checked = tmp_path / "checked"
    run = subprocess.run(["bash", "-c", '''
source "$0"
claim_twin() { [ "$*" != "$FAILING" ] || return 1; echo "$CLAIM_FILE"; }
claim_checked() { echo "$*" >> "$CHECKED"; }
claim_rig guard
echo "rc=$?"
if (: >&9) 2>/dev/null; then echo "fd 9 open"; fi
''', str(PORTS)], env=dict(env, FAILING=failing, CLAIM_FILE=str(claim), CHECKED=str(checked)),
        capture_output=True, text=True, timeout=60)
    assert run.stdout == "rc=1\n", run
    assert run.stderr == "", run
    assert not claim.exists()
    assert not checked.exists()

