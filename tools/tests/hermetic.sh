# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
# Sourced first by every shell test here, before anything else runs.
#
# WHY THE ENVIRONMENT IS RESET RATHER THAN TRIMMED. The launchers fill in what is
# missing from the station: unset ports fall back to the rig profile's, which are
# the station's own, HAMLIB_BIN puts a real rigctl on PATH, and a sound card is
# whatever sounddevice finds. A test that dropped one variable too many claimed
# the station's rig, and one more could have pounced on the air. So a test starts
# from a short allowlist, ports and data under WORK that exist nowhere else, and
# tripwires in front of every tool that reaches the radio, the codec, rigctld or
# another session's processes: a call the test did not scope to itself fails the
# test (hermetic_shims.py says what counts as scoped) and is not passed on.
#
# A test stops its own processes with hermetic_kill, records the ones that may
# outlive their parent with hermetic_own, and adds its own cleanup with
# hermetic_at_exit; the EXIT trap is this file's. If any of this cannot be set
# up, the test stops here, before it has run anything.
#
# WHAT IT DOES NOT STOP. This guards against accidents, not against a test
# written to escape it. Past the shims: a tool named by absolute path
# (/usr/sbin/lsof), reached with `command -p`, or found on a PATH the test
# rebuilds; anything not shimmed at all (fuser, launchctl); the builtin `kill`;
# and ps and pgrep forms the launchers do not use. Past the Python tripwire:
# `python -I` or `-E`, `env -u PYTHONPATH`, a child started with an env= of its
# own, and a file opened relative to a directory descriptor. Past both: the
# shell itself, redirecting to a device (`exec 3<>/dev/cu.*`) or opening
# /dev/tcp/127.0.0.1/4532. And past the exit trap: `exec` replacing the test
# shell, or a hermetic_at_exit command that removes WORK before the trips are
# read.

# Functions first: an inherited one would answer for any command below.
while IFS= read -r _fn; do unset -f "${_fn##* }"; done < <(declare -F)
unset BASH_ENV ENV _fn
HERMETIC_TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMETIC_ROOT="$(dirname "$(dirname "$HERMETIC_TESTS")")"
HERMETIC_GUARD="$HERMETIC_ROOT/packages/hfmodem/hfmodem/tests/hermetic_py"
HERMETIC_PY="$HERMETIC_ROOT/.venv/bin/python"
[ -x "$HERMETIC_PY" ] || HERMETIC_PY="$(command -v python3)"
HERMETIC_PATH="$PATH"

hermetic_abort() {
  printf '\n!! HERMETIC SETUP FAILED -- %s. Nothing in this test has run.\n' "$*" >&2
  [ -n "${HERMETIC_FRESH:-}" ] && rm -rf "$WORK"
  exit 98
}

while IFS= read -r _var; do
  case "$_var" in
    HOME|USER|LOGNAME|SHELL|TERM|TMPDIR|LANG|LC_*|TZ|PATH|HFMODEM_TEST_NO_STATION) ;;
    *) unset "$_var" 2>/dev/null ;;
  esac
done < <(compgen -e)
unset _var
hash -r

WORK="$(mktemp -d)" || hermetic_abort "mktemp gave no work directory"
[ -n "$WORK" ] && [ -d "$WORK" ] && [ ! -L "$WORK" ] && [ -O "$WORK" ] && [ -z "$(ls -A "$WORK")" ] \
  || hermetic_abort "'$WORK' is not a fresh directory of this test's own"
HERMETIC_FRESH=1
HERMETIC_TAG="hermetic$$"
_ports="$("$HERMETIC_PY" -c 'import socket
held = [socket.socket() for _ in range(2)]
for s in held: s.bind(("127.0.0.1", 0))
print(*(s.getsockname()[1] for s in held))')" || hermetic_abort "no free ports for the test's rigctld and station"
read -r HERMETIC_PORT HERMETIC_STATION_PORT _ <<<"$_ports"
[ -n "$HERMETIC_STATION_PORT" ] || hermetic_abort "no free ports for the test's rigctld and station"
unset _ports
mkdir -p "$WORK/shim" "$WORK/home" "$WORK/operator-data" || hermetic_abort "could not lay out $WORK"
HERMETIC_SHIMS="rigctl rigctld nc lsof ps pgrep pkill killall"
for _tool in $HERMETIC_SHIMS; do
  printf '#!/bin/sh\nexec "%s" -I "%s" %s "$@"\n' "$HERMETIC_PY" "$HERMETIC_TESTS/hermetic_shims.py" "$_tool" \
    > "$WORK/shim/$_tool" && chmod +x "$WORK/shim/$_tool" || hermetic_abort "could not write the $_tool shim"
done

# The claim lives in a state directory under WORK: the claim layer refuses the
# operator's own under either variable alone, and so does every child that keeps one.
export HOME="$WORK/home" PATH="$WORK/shim:$HERMETIC_PATH" HAMLIB_BIN="$WORK/shim" \
  PYTHONPATH="$HERMETIC_GUARD:$HERMETIC_ROOT/packages/hfmodem:$HERMETIC_ROOT/packages/hfhost:$HERMETIC_ROOT/packages/creance" \
  CAT_PORT="$WORK/no-cat.$HERMETIC_TAG" PTT_PORT="$WORK/no-ptt.$HERMETIC_TAG" \
  HFMODEM_STATE_DIR="$WORK/claim-state" HFMODEM_COEXIST_HARNESS=shell \
  HFMODEM_DATA_DIR="$WORK/operator-data" OPERATOR_DATA="$WORK/operator-data" \
  HERMETIC_WORK="$WORK" HERMETIC_PID=$$ HERMETIC_TAG HERMETIC_PORT HERMETIC_STATION_PORT HERMETIC_PATH

[ "${PATH%%:*}" = "$WORK/shim" ] || hermetic_abort "the shims are not first on PATH"
for _tool in $HERMETIC_SHIMS; do
  [ "$(command -v "$_tool")" = "$WORK/shim/$_tool" ] || hermetic_abort "$_tool resolves to $(command -v "$_tool"), not its shim"
done
unset _tool
"$HERMETIC_PY" -c 'import sys, hermetic_guard; sys.exit(0 if hermetic_guard.armed else 1)' \
  || hermetic_abort "the Python tripwire in $HERMETIC_GUARD did not arm"

hermetic_own() { "$HERMETIC_PY" -I "$HERMETIC_TESTS/hermetic_shims.py" own "$@"; }
hermetic_kill() { "$HERMETIC_PY" -I "$HERMETIC_TESTS/hermetic_shims.py" kill "$@"; }
HERMETIC_AT_EXIT=
hermetic_at_exit() { HERMETIC_AT_EXIT="$HERMETIC_AT_EXIT$1"$'\n'; }

hermetic_exit() {
  local rc=$?
  eval "$HERMETIC_AT_EXIT"
  "$HERMETIC_PY" -I "$HERMETIC_TESTS/hermetic_shims.py" reap
  "$HERMETIC_PY" -I "$HERMETIC_GUARD/hermetic_guard.py" sweep
  if [ -s "$WORK/trips" ]; then
    printf '\n!! HERMETIC TRIP -- this test reached past its own sandbox:\n' >&2
    sed 's/^/     /' "$WORK/trips" >&2
    rc=99
  fi
  rm -rf "$WORK"
  exit "$rc"
}
trap hermetic_exit EXIT
