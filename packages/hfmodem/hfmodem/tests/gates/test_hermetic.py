# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The hermetic guards themselves, each pinned by what it must stop.

The pytest side runs a session of its own in a child, under the same fixture,
so a test that trips can be watched tripping. The shell side sources
tools/tests/hermetic.sh in a scratch script.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[5] / "tools" / "tests"
NONE = "/dev/cu.hermetic-pin-nonexistent"

INNER_CONFTEST = '''
import sys
import pytest
from hfmodem.tests.hermetic import station_kept_out  # noqa: F401

def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: real card")
'''


def _session(tmp_path, body, **env):
    """Run `body` as a test module in a pytest session of its own."""
    (tmp_path / "conftest.py").write_text(INNER_CONFTEST)
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(body))
    # This session's import path, less any directory carrying a sitecustomize of
    # its own: an audit hook installed ahead of the fixture's would answer first.
    path = [p for p in sys.path if p and not os.path.isfile(os.path.join(p, "sitecustomize.py"))]
    # Baseline refusal tests run outside coexist; dedicated provisioned-pair
    # tests exercise its explicit software-device exception.
    child = {**os.environ, "PYTHONPATH": os.pathsep.join(path),
             "HFMODEM_QUALIFICATION_PHASE": "parallel", **env}
    for key in ("HERMETIC_WORK", "HERMETIC_TAG"):
        child.pop(key, None)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                        "-p", "no:randomly", str(tmp_path)],
                       cwd=tmp_path, env=child, capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


def test_a_test_opening_a_serial_device_is_refused_and_fails(tmp_path):
    out = _session(tmp_path, f'''
        import pytest
        def test_open():
            with pytest.raises(PermissionError):
                open({NONE!r})
    ''')
    assert f"HERMETIC TRIP: python opened {NONE}" in out, out


def test_every_device_node_but_a_tests_own_is_refused(tmp_path):
    """Outside explicit coexist, a tty0tty pair, a Bluetooth
    serial link or a sound card is a device no test of its own has."""
    refused = ["/dev/tnt0", "/dev/rfcomm0", "/dev/snd/pcmC0D0c"]
    out = _session(tmp_path, f'''
        import pytest
        @pytest.mark.parametrize("path", {refused!r})
        def test_refused(path):
            with pytest.raises(PermissionError):
                open(path)
        @pytest.mark.parametrize("path", ["/dev/null", "/dev/zero", "/dev/urandom", "/dev/stdout"])
        def test_opens(path):
            open(path, "rb" if path in ("/dev/zero", "/dev/urandom") else "w").close()
    ''')
    for path in refused:
        assert f"HERMETIC TRIP: python opened {path}" in out, out
    assert out.count("PASSED test_inner.py::test_opens") == 4, out


def test_no_bash_env_reaches_a_tests_shells(tmp_path):
    hook = tmp_path / "bashenv"
    hook.write_text("echo BASH_ENV-RAN\n")
    out = _session(tmp_path, '''
        import os, subprocess
        def test_shell():
            assert "BASH_ENV" not in os.environ and "ENV" not in os.environ
            said = subprocess.run(["bash", "-c", "echo child"], capture_output=True, text=True).stdout
            assert "BASH_ENV-RAN" not in said, said
    ''', BASH_ENV=str(hook), ENV=str(hook))
    assert "1 passed" in out, out


def test_the_card_stub_does_not_outlive_its_test(tmp_path):
    (tmp_path / "card").mkdir()
    (tmp_path / "card" / "sounddevice.py").write_text("WHO = 'the card'\n")
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "takes_card.py").write_text("import sounddevice as sd\n")
    out = _session(tmp_path, f'''
        import sys
        import pytest
        sys.path.append({str(tmp_path / "mods")!r})

        def test_a_plain_test_imports_it_mid_test():
            import takes_card
            assert takes_card.sd is sys.modules["sounddevice"]

        @pytest.mark.hardware
        def test_b_a_later_hardware_test_gets_the_card():
            sys.path.insert(0, {str(tmp_path / "card")!r})
            import takes_card
            assert takes_card.sd.WHO == "the card"
    ''')
    assert "2 passed" in out, out


def test_a_claim_not_named_for_the_run_trips_and_stays(tmp_path):
    foreign = Path(f"/tmp/hfmodem-rig-hermetic-pin-foreign-{os.getpid()}.claim")
    foreign.touch()
    try:
        out = _session(tmp_path, f'''
            import fcntl, os, subprocess
            def test_claims():
                fd = os.open({str(foreign)!r}, os.O_WRONLY | os.O_APPEND)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.close(fd)
                subprocess.run(["bash", "-c",
                                'exec 9>>"/tmp/hfmodem-rig-shell-$HERMETIC_TAG.claim"; echo "$HERMETIC_TAG" > {tmp_path}/tag'],
                               check=True)
        ''')
        assert "is not this run's -- left in place" in out, out
        assert foreign.exists(), "the sweep removed a claim that was not the run's"
        tag = (tmp_path / "tag").read_text().strip()
        assert not Path(f"/tmp/hfmodem-rig-shell-{tag}.claim").exists(), "a shell-taken claim was left"
    finally:
        foreign.unlink(missing_ok=True)


# -- the shell side -----------------------------------------------------------------

shell = pytest.mark.skipif(not (TESTS / "hermetic.sh").is_file(), reason=f"{TESTS} is not present")


def _sourced(tmp_path, body, **env):
    script = tmp_path / "probe.sh"
    script.write_text('source "$1/hermetic.sh"\n' + textwrap.dedent(body))
    r = subprocess.run(["bash", str(script), str(TESTS)], capture_output=True, text=True, timeout=120,
                       env={**os.environ, **env})
    return r.returncode, r.stdout + r.stderr


@shell
def test_a_shell_tests_python_imports_this_tree(tmp_path):
    """The venv's editable install may be another checkout's."""
    root = TESTS.parents[1].resolve()
    rc, out = _sourced(tmp_path, '''
        "$HERMETIC_PY" -c 'import creance, hfhost, hfmodem
for m in (creance, hfhost, hfmodem): print("from", m.__file__)'
    ''')
    found = [Path(line.split(None, 1)[1]).resolve() for line in out.splitlines() if line.startswith("from ")]
    assert len(found) == 3 and all(root in f.parents for f in found), out


@shell
def test_pkill_on_the_tests_own_children_spares_the_shim(tmp_path):
    rc, out = _sourced(tmp_path, '''
        sleep 30 & hermetic_own $!
        pkill -P $$; echo "pkill rc=$?"
    ''')
    assert "pkill rc=0" in out and rc == 0, out


@shell
def test_pkill_options_are_not_read_as_signals(tmp_path):
    rc, out = _sourced(tmp_path, '''
        pkill -f "$WORK/nothing-matches"; echo "pkill rc=$?"
    ''')
    assert "pkill rc=1" in out and rc == 0, out


@shell
def test_a_word_that_is_no_signal_trips_rather_than_crashing(tmp_path):
    rc, out = _sourced(tmp_path, '''
        pkill -fx "$WORK"
        hermetic_kill -NOPE $$
    ''')
    assert rc == 99 and "Traceback" not in out, out
    assert "not a signal this shim knows: -fx" in out and "not a signal this shim knows: -NOPE" in out, out


@shell
def test_no_bash_env_reaches_a_shell_tests_children(tmp_path):
    hook = tmp_path / "bashenv"
    hook.write_text("echo BASH_ENV-RAN\n")
    rc, out = _sourced(tmp_path, '''
        bash -c 'echo "child [${BASH_ENV:-}]"'
    ''', BASH_ENV=str(hook))
    assert "child []" in out and out.count("BASH_ENV-RAN") <= 1, out


@shell
def test_the_shell_sweep_keeps_what_is_not_the_runs(tmp_path):
    foreign = Path(f"/tmp/hfmodem-rig-hermetic-pin-shell-foreign-{os.getpid()}.claim")
    foreign.touch()
    try:
        rc, out = _sourced(tmp_path, f'''
            "$HERMETIC_PY" -c 'import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_WRONLY); fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)' {foreign}
            exec 9>>"/tmp/hfmodem-rig-shell-$HERMETIC_TAG.claim"; exec 9>&-
            echo "$HERMETIC_TAG" > {tmp_path}/tag
        ''')
        assert rc == 99 and "is not this run's -- left in place" in out, out
        assert foreign.exists(), "the sweep removed a claim that was not the run's"
        tag = (tmp_path / "tag").read_text().strip()
        assert not Path(f"/tmp/hfmodem-rig-shell-{tag}.claim").exists(), "a shell-taken claim was left"
    finally:
        foreign.unlink(missing_ok=True)


@shell
def test_a_child_that_drops_every_mark_of_the_run_still_trips_on_the_real_state(tmp_path):
    rc, out = _sourced(tmp_path, '''
        env -u HFMODEM_STATE_DIR -u HFMODEM_COEXIST_HARNESS -u HERMETIC_WORK "$HERMETIC_PY" -c '
from hfmodem.core import claim
try:
    open(claim.claim_path())
except PermissionError as exc:
    print("refused:", exc)
'
    ''')
    assert "refused: HERMETIC TRIP: reached the operator's state directory" in out, out


@shell
def test_down_in_a_shell_test_sees_only_the_tests_own_processes(tmp_path):
    """The launcher's legacy reap matches modems by name across the machine. In a
    shell test its process listing is the test's own, so a witness the test did
    not start is neither listed nor stopped."""
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                              str(tmp_path / "kiwi_witness.py")])
    try:
        rc, out = _sourced(tmp_path, f'''
            ps -axo pid=,ppid=,uid=,lstart=,command= | awk '{{print "listed", $1}}'
            sleep 30 & hermetic_own $!
            ps -axo pid=,ppid=,uid=,lstart=,command= | grep -c "[s]leep 30" | sed 's/^/own /'
            bash "$1/../onair.sh" down >/dev/null 2>&1; echo "down rc=$?"
            kill -0 {decoy.pid} && echo "decoy alive"
        ''')
        assert f"listed {decoy.pid}\n" not in out and "own 1" in out, out
        assert "decoy alive" in out and decoy.poll() is None, out
    finally:
        decoy.kill()
        decoy.wait()


@shell
def test_only_this_runs_environment_marks_a_process_as_the_tests(tmp_path):
    """A process the test did not start is its own only by the environment both
    marks of this run are in: not by a command line carrying them, and not by an
    earlier run's orphan that has the pid but another WORK."""
    root = subprocess.Popen(["sleep", "60"])            # stands in for the test shell
    work = tmp_path / "work"
    work.mkdir()
    mark = {"HERMETIC_PID": str(root.pid), "HERMETIC_WORK": str(work)}
    plain = {"PATH": "/usr/bin:/bin"}
    cases = {
        "marked": (["sleep", "60"], {**plain, **mark}),
        "command line": (["/bin/sh", "-c", "sleep 60; :", "sh", *(f"{k}={v}" for k, v in mark.items())],
                         plain),
        "stale orphan": (["sleep", "60"], {**plain, **mark, "HERMETIC_WORK": str(tmp_path / "earlier")}),
    }
    procs = {name: subprocess.Popen(argv, env=env, start_new_session=True)
             for name, (argv, env) in cases.items()}
    shim = {**os.environ, **mark, "HERMETIC_PORT": "1", "HERMETIC_STATION_PORT": "2",
            "HERMETIC_TAG": "pin", "HERMETIC_PATH": os.environ["PATH"]}
    try:
        adopted = {name: subprocess.run([sys.executable, "-I", str(TESTS / "hermetic_shims.py"), "ps",
                                         "-p", str(p.pid), "-o", "pid="],
                                        env=shim, capture_output=True, timeout=60).returncode == 0
                   for name, p in procs.items()}
        assert adopted == {"marked": True, "command line": False, "stale orphan": False}
    finally:
        for p in (root, *procs.values()):
            p.kill()
            p.wait()
