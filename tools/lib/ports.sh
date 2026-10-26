# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
# The station's serial ports, sourced by every launcher in tools/.
#
# It lives in one file because the four scripts share one environment contract,
# and a refusal kept in four copies is a refusal with three stale ones: onair.sh
# was taught to name the variables on 2026-08-10 and campaign, daywatch and
# nightwatch went on shipping the placeholder for another day, each of them
# beside a copy nobody was reading at the same time.
#
# THE PORTS ARE NOT IN HERE, AND NOTHING HERE WILL GUESS THEM.
#
# They were: a redaction placeholder, a device path with the adapter's serial
# number Xed out, which exists on no machine. Every real run needed CAT_PORT= and
# PTT_PORT= in front of it and nothing said so, so a station that was on and
# correctly cabled met "CAT port … missing — is the rig on and cabled?" and went
# looking at its cable. The same placeholder had already let a 27-cycle session
# report 22.1 s of carrier into a radio it never touched.
#
# So: no default, and the unset case teaches instead. One export is enough on the
# usual two-interface adapter, since the keying line is derived from the CAT one.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$(dirname "${HFMODEM_PYTHON:-$REPO/.venv/bin/python}")"
_rig_fields=$(PYTHONPATH="$REPO/packages/hfmodem${PYTHONPATH:+:$PYTHONPATH}" "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.rigs) || exit 1
{
  IFS= read -r RIG_NAME
  IFS= read -r MODEL
  IFS= read -r BAUD
  IFS= read -r RIG_MODE
  IFS= read -r AUDIO
  IFS= read -r _rig_cat
  IFS= read -r _rig_ptt
  IFS= read -r AUDIO_OUT
} <<< "$_rig_fields"
CAT_PORT="${CAT_PORT-}"
PTT_PORT="${PTT_PORT-}"
CAT_PORT="${CAT_PORT:-$_rig_cat}"
PTT_PORT="${PTT_PORT:-$_rig_ptt}"
case "$RIG_NAME" in
  ft891) RIG_EXPECT_MODEL=FT-891 ;;
  ftdx10) RIG_EXPECT_MODEL=FTDX-10 ;;
  g90) RIG_EXPECT_MODEL=G90 ;;
  x6100) RIG_EXPECT_MODEL=X6100 ;;
esac
AUDIO_IN="$AUDIO"
export AUDIO_IN AUDIO_OUT
export HFMODEM_RIG="$RIG_NAME"
unset _rig_fields _rig_cat _rig_ptt

# The callsign the launchers hand every modem: the operator's MYCALL when it is
# set, else the station file's. Without either this says so and the launcher goes
# on; a callsign written in here would put someone else's name on the air.
station_mycall() { _station_field mycall; }
station_grid() { _station_field grid; }
_station_field() {
  PYTHONPATH="$REPO/packages/hfmodem${PYTHONPATH:+:$PYTHONPATH}" "${HFMODEM_PYTHON:-$VENV/python}" -c \
      'import sys; from hfmodem.core import config; print(getattr(config, sys.argv[1])())' "$1" || true
}

# The sabir beacon's grid arguments: GRID_ARGS goes ahead of the operator's flags,
# GRID_FALLBACK after them. The wspr beacon cannot be built without a grid, so a
# station that declared none sends the presence beacon, which needs none.
beacon_grid() {
  local grid arg pending= beacon=presence effective_grid
  grid="${MYGRID:-$(station_grid)}"
  GRID_ARGS=(); GRID_FALLBACK=()
  if [ -n "$grid" ]; then GRID_ARGS=(--grid "$grid"); fi
  effective_grid="$grid"
  # The tool receives these flags after GRID_ARGS, so its final explicit grid
  # and beacon choices win over station configuration, as argparse does.
  for arg in "$@"; do
    if [ -n "$pending" ]; then
      case "$pending" in grid) effective_grid="$arg" ;; beacon) beacon="$arg" ;; esac
      pending=
    else
      case "$arg" in
        --grid) pending=grid ;;
        --grid=*) effective_grid="${arg#--grid=}" ;;
        --beacon) pending=beacon ;;
        --beacon=*) beacon="${arg#--beacon=}" ;;
      esac
    fi
  done
  if [ -z "$effective_grid" ] && [ "$beacon" = wspr ]; then
    echo "warning: no grid, so the wspr beacon is sent as the presence beacon" >&2
    GRID_FALLBACK=(--beacon presence)
  fi
}

# HAMLIB, ON PATH, FOR EVERYTHING A LAUNCHER STARTS.
#
# The launchers run `rigctld` themselves, and the modems they start run `rigctl`
# by name -- shrike through `ota.Rig`, besra through `besra.radio.Rig`, sabir
# through `sabir.radio`. None of those may carry a machine-local directory, so
# they all resolve off PATH; and a Hamlib built from source is on no machine's
# PATH by default. One export here answers every one of them at once, and
# answers the next one nobody has written yet, which threading a --rigctl-dir
# flag through each consumer does not.
#
# HAMLIB_BIN names a build that lives somewhere else. An ordinary package
# install (Homebrew, apt) is already on PATH and needs neither.
HAMLIB_BIN="${HAMLIB_BIN:-$HOME/src/radio/hamlib/_install/bin}"
[ -x "$HAMLIB_BIN/rigctl" ] && export PATH="$HAMLIB_BIN:$PATH"

# NOTHING HERE PRINTS A COMMAND LINE WITHOUT PASSING IT THROUGH THIS FIRST.
#
# A command line is not a safe thing to echo. `claim_rig` records the launcher's
# own `ps -o command=` so the next launcher can say what is already running, and
# on 2026-08-28 what it said back was the operator's Winlink password, in a log
# that was being tee'd at the time.
#
# The password itself now leaves argv in `onair.sh` before any verb sees it, so
# this is the second line of defence and not the first: it holds for a hand-typed
# invocation, for a `ps` of some other process, and for whatever flag carries a
# secret next. It matches the SHAPE -- a password-ish flag and the word after it
# -- and never a value, so there is nothing here to keep in step with a secret.
redact_secrets() {
  sed -E 's/(--(mail-)?(password|passwd|secret|token)(=|[[:space:]]+))[^[:space:]]+/\1<redacted>/g'
}

die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

# Before anything keys or tunes. Named here rather than left to fail inside
# whichever tool reaches for it first: `rigctl` missing surfaces deep in a
# transmit path as a FileNotFoundError from a subprocess spawn, with the audio
# device already open, and at the radio that reads as an audio fault.
require_hamlib() {
  command -v rigctl >/dev/null && command -v rigctld >/dev/null || die \
"hamlib is not on PATH, and nothing here can tune or key a radio without it.
   rigctl $(command -v rigctl || echo 'not found'), rigctld $(command -v rigctld || echo 'not found')

     brew install hamlib          # or your package manager

   For a build of your own, name its bin directory and the launchers will put
   it on PATH for everything they start:

     export HAMLIB_BIN=$HAMLIB_BIN

   Check with: command -v rigctl rigctld"
}

# For a watch that only listens. It needs the CAT port to tune and nothing else,
# and demanding a keying line as well is how an operator ends up exporting a
# made-up PTT_PORT to get a receive-only run started.
require_cat() {
  [ -n "$CAT_PORT" ] || die "$(unconfigured_ports)"
}

# For anything that may key.
require_ports() {
  require_cat
  if [ -z "$PTT_PORT" ]; then
    PTT_PORT="$(derive_ptt "$CAT_PORT")" \
      || die "set PTT_PORT to the line that keys this rig, and run again"
  fi
}

# What to set, and what this machine is offering — the devices attached right
# now, rather than an example that would be copied whether or not it is true here.
unconfigured_ports() {
  local how list
  case "$(uname -s)" in
    Darwin) how="ls /dev/cu.*"
            list=$(ls -d /dev/cu.usbserial-* /dev/cu.usbmodem* 2>/dev/null || true) ;;
    *)      how="ls /dev/serial/by-id/*"
            list=$(ls -d /dev/serial/by-id/* 2>/dev/null || true) ;;
  esac
  if [ -n "$list" ]; then list=$(printf '%s\n' "$list" | sed 's/^/     /')
  else list="     (nothing — is the interface plugged in?)"; fi
  cat <<EOF
no serial ports configured — CAT_PORT is not set. That is a station setting,
   not a fault: device names differ on every machine, so these launchers ship
   with none rather than with one that is right here and wrong at the next rig.

     CAT_PORT   the rig's CAT interface — rigctld and rigctl own this one
     PTT_PORT   the line whose RTS keys the rig. Leave it unset on a
                two-interface adapter, where it derives from CAT_PORT

   Find yours with '$how' — attached right now:
$list

   Then, in your shell profile so it outlives this session:
     export CAT_PORT=<the CAT interface above>
EOF
}

RIG_LOCK_FD=9
CLAIM_OWNED=0
CLAIM_ACTIVE=0
CLAIM_ADMITTED=0
CLAIM_FAILED=0
CLAIM_MUTEX=0
CLAIM_CHILDREN=
CLAIM_LAUNCH=0
CLAIM_PARENT=0
CLAIM_ATTEMPTED=0
CLAIM_AFTER_DOWN=
CLAIM_AFTER_DOWN_S=15

claim_twin() {
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.claim "$@"
}

claim_paths() {
  CLAIM=$(claim_twin path) || return
  MUTEX=$(claim_twin path --mutex) || return
}

claim_checked() {
  local expected="$1" result rc=0
  shift
  result=$(mktemp) || return
  claim_twin "$@" >"$result" || rc=$?
  cat "$result"
  if [ "$rc" -eq 0 ] && ! grep -qx "$expected" "$result"; then rc=1; fi
  rm -f "$result"
  return "$rc"
}

claim_rig() {
  [ "$CLAIM_ACTIVE" = 0 ] || return 0
  claim_paths || return
  if [ -n "${RIG_CLAIMED:-}" ]; then
    claim_checked 'CLAIM INHERITED' inherited 9 7>&- 8>&- || return
  else
    exec 9<>"$CLAIM"
    # No --prove-line: I4 holds through P1's LineKeyer.arm reading the line low before its first assert.
    if claim_checked 'CLAIM ACQUIRED' acquire 9 --kind shell --operation "${1:-launch}" \
        --cat-port "$CAT_PORT" --ptt-port "$PTT_PORT" --capture "$AUDIO_IN" --playback "$AUDIO_OUT" 7>&- 8>&-; then
      CLAIM_OWNED=1
    else
      exec 9>&-
      return 1
    fi
  fi
  CLAIM_ACTIVE=1
  CLAIM_FAILED=0
  case "${1:-launch}" in campaign|autopilot|rigtest|pounce) CLAIM_PARENT=1 ;; esac
  RIG_CLAIMED="$CLAIM"; export RIG_CLAIMED
  trap 'claim_exit $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  CLAIM_CHILDREN=$(mktemp) || return
  claim_launch_hold
}

claim_launch_hold() {
  [ "$CLAIM_LAUNCH" = 0 ] || return 0
  exec 7<>"$MUTEX"
  claim_checked 'CLAIM INHERITED' inherited 9 --launch 7 8>&- || { exec 7>&-; return 1; }
  CLAIM_LAUNCH=1
}

claim_admit() {
  exec 7>&-
  CLAIM_LAUNCH=0
  CLAIM_ADMITTED=1
  printf 'CLAIM ADMITTED\n'
}

claim_close() {
  local rc=0
  if [ "$CLAIM_ACTIVE" = 1 ]; then
    if [ "$CLAIM_OWNED" = 1 ] && [ "$CLAIM_FAILED" = 0 ]; then
      claim_checked 'CLAIM RELEASED' release 9 7>&- 8>&- || rc=$?
    else
      claim_checked 'CLAIM CLOSED' close 9 7>&- 8>&- || rc=$?
    fi
    exec 9>&-
    CLAIM_ACTIVE=0
  fi
  return "$rc"
}

claim_spawn() {
  if [ "$CLAIM_MUTEX" = 1 ]; then
    claim_twin spawn --fd 9 --cleanup 8 "$@" 7>&-
  else
    claim_twin spawn --fd 9 "$@" 7>&- 8>&-
  fi
}

claim_native() { claim_spawn --timeout 10 -- "$@"; }

claim_record() {
  claim_processes record "$1" >>"$CLAIM_CHILDREN"
}

claim_start() {
  local fifo line deadline child_rc=0
  claim_launch_hold || return
  fifo=$(mktemp -d) || return
  mkfifo "$fifo/output" || { rm -rf "$fifo"; return 1; }
  "$@" <&0 7>&- 8>&- >"$fifo/output" &
  CLAIM_PID=$!
  CLAIM_ATTEMPTED=1
  claim_record "$CLAIM_PID" || return
  exec 6<"$fifo/output"
  rm -rf "$fifo"
  deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if IFS= read -r -t 1 line <&6; then
      if [ "$line" = 'CLAIM ADMITTED' ]; then
        if [ "${CLAIM_INTERNAL:-0}" = 0 ]; then claim_admit; fi
        cat <&6 6>&- 7>&- 8>&- 9>&- &
        CLAIM_RELAY=$!
        exec 6>&-
        return 0
      fi
      printf '%s\n' "$line"
      case "$line" in 'CLAIM NOT ADMITTED'*) break ;; esac
    elif ! claim_processes alive "$CLAIM_PID"; then
      break
    fi
  done
  claim_processes end "$CLAIM_PID" || { CLAIM_FAILED=1; return 1; }
  wait "$CLAIM_PID" 2>/dev/null || child_rc=$?
  exec 6>&-
  # An unacknowledged inherited launcher must unwind without down; its parent
  # still holds admission. An owning parent can prove after its child is gone.
  exec 7>&-
  CLAIM_LAUNCH=0
  printf 'CLAIM NOT ADMITTED child exited or acknowledgement timed out\n'
  [ "$child_rc" -ne 0 ] || child_rc=1
  return "$child_rc"
}

claim_wait() {
  local rc=0
  wait "$CLAIM_PID" || rc=$?
  wait "$CLAIM_RELAY" || rc=1
  return "$rc"
}

claim_run() { claim_start "$@" && claim_wait; }

# Setup captures acknowledge their own opens, never the launcher's readiness.
claim_worker() { local CLAIM_INTERNAL=1; claim_run "$@"; }

claim_down() { "$REPO/tools/onair.sh" down 7>&- 8>&-; }

# Readouts after the line has read low, while the claim is still held. A subshell,
# so nothing it does reaches the release; bounded, because every signal is
# ignored by now and a hung readout would hold the claim with no way to stop it.
# The session's last lines come first, within the same bound: a descendant still
# holding its output would otherwise hold the claim as long as it lives.
claim_after_down() {
  local hook mark child started deadline=$((SECONDS + CLAIM_AFTER_DOWN_S))
  while [ -n "${CLAIM_RELAY:-}" ] && kill -0 "$CLAIM_RELAY" 2>/dev/null && \
      [ "$SECONDS" -lt "$deadline" ]; do
    sleep 0.1
  done
  deadline=$((SECONDS + CLAIM_AFTER_DOWN_S))
  mark=$(wc -l <"$CLAIM_CHILDREN")
  ( "$CLAIM_AFTER_DOWN" ) &
  hook=$!
  while kill -0 "$hook" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      printf '  readout abandoned after %ss\n' "$CLAIM_AFTER_DOWN_S"
      kill -KILL "$hook" 2>/dev/null || true
      while read -r child started; do
        claim_processes end "$child" || CLAIM_FAILED=1
      done < <(tail -n "+$((mark + 1))" "$CLAIM_CHILDREN")
      break
    fi
    sleep 0.1
  done
  wait "$hook" 2>/dev/null || true
}

claim_exit() {
  local rc="${1:-0}" cleanup_rc=0
  trap - EXIT
  trap '' INT TERM HUP
  if [ "$CLAIM_ACTIVE" = 1 ]; then
    claim_processes owner || cleanup_rc=$?
    exec 7>&-
    CLAIM_LAUNCH=0
    if [ "$cleanup_rc" -eq 0 ] && [ "$CLAIM_FAILED" = 0 ] && \
        { [ "$CLAIM_ADMITTED" = 1 ] || \
          { [ "$CLAIM_OWNED" = 1 ] && [ "$CLAIM_PARENT" = 1 ] && [ "$CLAIM_ATTEMPTED" = 1 ]; }; }; then
      claim_down || cleanup_rc=$?
      if [ "$cleanup_rc" -eq 0 ] && [ -n "$CLAIM_AFTER_DOWN" ]; then claim_after_down; fi
    else
      printf 'CLAIM NOT ADMITTED launcher initialization or termination failed\n'
      CLAIM_FAILED=1
      [ "$rc" -ne 0 ] || rc=1
    fi
    if [ "$cleanup_rc" -ne 0 ]; then CLAIM_FAILED=1; rc=1; fi
    claim_close || rc=1
  fi
  # Reap recorded direct children, leaving unrelated shell jobs alone.
  if [ -n "$CLAIM_CHILDREN" ] && [ "$CLAIM_FAILED" = 0 ]; then
    while read -r child started; do wait "$child" 2>/dev/null || true; done <"$CLAIM_CHILDREN"
    [ -z "${CLAIM_RELAY:-}" ] || wait "$CLAIM_RELAY" 2>/dev/null || true
  fi
  [ -z "$CLAIM_CHILDREN" ] || rm -f "$CLAIM_CHILDREN"
  if declare -F capture_exit >/dev/null; then capture_exit "$rc" || rc=$?; fi
  exit "$rc"
}

claim_processes() {
  "${HFMODEM_PYTHON:-$VENV/python}" - "$@" "$$" "$REPO" "$CAT_PORT" "$PTT_PORT" \
      "$MODEL" "$AUDIO_IN" "$AUDIO_OUT" "$CLAIM_CHILDREN" <<'CLAIM_PROCESSES'
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tomllib
from hfmodem.core import claim
from hfmodem.core.rigs import CAT_BACKENDS, profile_cat
from hfmodem.core.native import NativeChild, REAP_GRACE_S
from hfmodem.station import client

mode, *args = sys.argv[1:]
root, repo, cat_port, ptt_port, model, capture, playback, records = args[-8:]
root = int(root)
args = args[:-8]
# pid -> (identity, parent, parent identity) as first seen; ancestry is lost on reparenting.
lineage = {}
ended = {}


def table():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,uid=,lstart=,command='],
                            capture_output=True, text=True, timeout=5, check=True)
    rows = {}
    for line in result.stdout.splitlines():
        pid, parent, uid, *fields = line.strip().split(None, 8)
        if len(fields) != 6:
            raise claim.ClaimError("partial process metadata")
        rows[int(pid)] = (int(parent), int(uid), fields[-1], "_".join(fields[:5]))
    if root not in rows:
        raise claim.ClaimError('process inventory unavailable')
    for pid, (parent, _, _, started) in rows.items():
        lineage.setdefault(pid, (started, parent, rows[parent][3] if parent in rows else None))
    return rows


def inheritor(pid, matched, identities):
    started, parent, parent_started = lineage.get(pid, (None, None, None))
    return (pid in matched and started == identities[pid] and parent in ended
            and parent_started in (None, ended[parent]))


def within(pid, ancestor, rows):
    seen = set()
    while pid in rows and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = rows[pid][0]
    return False


def station(command):
    return bool(re.search(r'(?:^|[/\s])hfmodem(?:\.cli)?\s+station(?:\s|$)', command))


def legacy_target(pid, rows, matched, delegated):
    parent, uid, command, started = rows[pid]
    if uid != os.getuid() or pid in delegated:
        return False
    words = command.split()
    daemon = any(Path(w).name == 'rigctld' for w in words) and any(
        w == '-m' and i + 1 < len(words) and words[i + 1] == model
        for i, w in enumerate(words))
    # A repository path and PPID 1 prove neither radio ownership nor a legacy
    # modem identity. Detached corpus/analysis workers can have both. Unknown
    # actual device holders remain fail-closed in inventory(), never signalled.
    return (pid in matched or daemon or bool(re.search(
        r'(?:hfmodem\.)?(?:besra\.host\.run_server|shrike\.onair|lanner\.qso|sabir\.onair)|'
        r'(?:^|[/\s])(?:run_server|kestrel_connect|kiwi_witness)(?:\.py)?(?:\s|$)|creance\.cli', command)))


def listeners(port):
    result = subprocess.run(['lsof', '-t', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN'],
                            capture_output=True, text=True, timeout=5)
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise claim.ClaimError('listener inventory unavailable: ' + result.stderr.strip())
    return {int(pid) for pid in result.stdout.split()}


def settings():
    path = os.environ.get('HFMODEM_STATION')
    body = tomllib.loads(Path(path).read_text()) if path else {}
    rig = body.get('rig', {})
    cat = rig['cat'] if 'cat' in rig else profile_cat(legacy_model=rig.get('model') or None)
    if cat not in CAT_BACKENDS:
        raise claim.ClaimError('unknown CAT backend: ' + str(cat))
    backend = CAT_BACKENDS[cat]
    return path, body.get('station', {}).get('port', 8737), backend, rig.get('host', '127.0.0.1'), rig.get('port', 4532)


def recorded():
    if not records:
        return {}
    return {int(pid): started for pid, started in
            (line.split() for line in Path(records).read_text().splitlines())}


def end(children):
    for child in children:
        if child.alive():
            print(f'pid {child.pid} -- SIGTERM', flush=True)
            child.terminate()
    deadline = time.monotonic() + REAP_GRACE_S
    for child in children:
        child.wait_gone(max(0, deadline - time.monotonic()))
    cleanup = {r.pid for r in claim.holders(8, cleanup=True)} if mode == 'down' else set()
    if mode != 'down':
        fd = os.open(claim.mutex_path(), os.O_RDONLY)
        try:
            # Empty is the normal unpublished mutex. A present but unreadable
            # publication or failed holder discovery cannot authorize SIGKILL.
            if os.fstat(fd).st_size:
                cleanup = {r.pid for r in claim.holders(fd, cleanup=True)}
        finally:
            os.close(fd)
    rows = table()
    for child in children:
        if child.alive():
            if any(within(pid, child.pid, rows) for pid in cleanup):
                raise claim.ClaimError(f'cleanup in progress (pid {child.pid}); no escalation')
            print(f'pid {child.pid} -- SIGKILL', flush=True)
            child.kill()
    for child in children:
        if not child.wait_gone(5):
            raise claim.ClaimError(f'termination unresolved for pid {child.pid}')
        ended[child.pid] = child.started


def inventory():
    if mode == 'down':
        claim.verify_cleanup(8)
    else:
        claim.inherited(9)
    path, station_port, backend, host, port = settings()
    devices = claim.Devices(cat_port, ptt_port, capture, playback, backend)
    nodes = []
    for endpoint in (devices.cat_port if backend == 'serial' else None, devices.ptt_port):
        nodes.extend(claim.serial_nodes(endpoint))
    audio = []
    for endpoint, direction in ((capture, 'capture'), (playback, 'playback')):
        try:
            audio.extend(claim.audio_nodes(endpoint, direction))
        except Exception:
            pass
    issued = None
    released = False
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        rows = table()
        identities = {pid: row[3] for pid, row in rows.items()}
        if issued is None:
            issued = identities
        ancestors = {pid for pid in rows if within(root, pid, rows)} | {os.getpid()}
        delegates = listeners(port) if backend == 'rigctld' else set()
        for pid in delegates:
            if pid not in rows or 'rigctld' not in rows[pid][2]:
                raise claim.ClaimError(f'delegated CAT listener unidentified: pid {pid}')
            print(f'delegated rigctld pid {pid} on {host}:{port}', flush=True)
        stations = {pid for pid, row in rows.items() if station(row[2])} | listeners(station_port)
        protected = {pid for pid in rows if any(within(pid, s, rows) for s in stations)}
        device_rows = claim.openers(nodes) if nodes else []
        audio_rows = claim.openers(audio) if audio else []
        device_rows += [r for r in audio_rows if Path(r.command).name not in
                        {'pipewire', 'pulseaudio', 'coreaudiod', 'wireplumber', 'jackd'}]
        device_pids = {r.pid for r in device_rows} - delegates
        descriptor = claim.ClaimDescriptor(9)
        inherited = bool(os.environ.get('RIG_CLAIMED')) or mode != 'down'
        held = True
        if not inherited:
            probe = os.open(claim.claim_path(), os.O_RDWR)
            try:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = False
                except BlockingIOError:
                    pass
            finally:
                os.close(probe)
        published = claim.holder(descriptor) if held else None
        matched = {r.pid for r in claim.holders(descriptor)} if held else set()
        candidates = device_pids | matched | stations | (set(recorded()) & rows.keys()) | {
            pid for pid in rows if legacy_target(pid, rows, set(), delegates)}
        if not candidates <= rows.keys():
            raise claim.ClaimError('process metadata missing; retry cleanup')
        if mode == 'down' and stations:
            if not path:
                raise claim.ClaimError('station found without a station file; release it explicitly')
            service = client.Client(path, port=station_port)
            try:
                snapshot = service.state(timeout=min(2, max(.001, deadline - time.monotonic())))
                if snapshot['service']['state'] != 'observing' or matched & protected and not released:
                    try:
                        service.release(timeout=max(.001, deadline - time.monotonic()))
                        released = True
                    except client.CommandFailed as exc:
                        if exc.result.get('reason') != 'transitioning':
                            raise
                    time.sleep(.05)
                    continue
            except Exception as exc:
                raise claim.ClaimError('station not released, held by pid '
                                       + ', '.join(map(str, sorted(stations))) + f': {exc}') from exc
            if (device_pids | matched) & protected:
                raise claim.ClaimError('observing station still holds a device or claim')
        if mode == 'down' and not inherited and published and published.pid in rows:
            if identities[published.pid] != published.started:
                raise claim.ClaimError('claim publisher pid reused; refusing cleanup')
            print(f'pid {published.pid} ({published.command}) holds the claim; interrupt it and run down again', flush=True)
            raise SystemExit(7)
        targets, unknown = [], []
        own = recorded()
        for pid in sorted(candidates - delegates - protected):
            if pid in ancestors:
                if pid in device_pids:
                    unknown.append(pid)
                continue
            if mode != 'down':
                if not within(pid, root, rows) and own.get(pid) != identities[pid]:
                    continue
                if mode == 'daemon' and 'rigctld' not in rows[pid][2]:
                    continue
            if issued.get(pid) != identities[pid] and not inheritor(pid, matched, identities):
                raise claim.ClaimError(f'pid {pid} started after this down was issued')
            if legacy_target(pid, rows, matched, delegates) or mode != 'down' and own.get(pid) == identities[pid]:
                targets.append(NativeChild(pid, identities[pid], None, None, None))
            else:
                unknown.append(pid)
        if unknown:
            raise claim.ClaimError('unidentified device/claim holder(s): ' + ', '.join(map(str, unknown)))
        if not targets:
            if mode == 'down':
                claim.check_devices(devices)
            return
        end(targets)
    raise claim.ClaimError('cleanup inventory did not settle within 45 s')


try:
    if mode == 'record':
        pid = int(args[0])
        print(pid, claim.process_started(pid), flush=True)
    elif mode in ('alive', 'end'):
        pid = int(args[0])
        started = recorded().get(pid)
        if not started:
            raise claim.ClaimError(f'child identity missing: pid {pid}')
        child = NativeChild(pid, started, None, None, None)
        if mode == 'alive':
            raise SystemExit(0 if child.alive() else 1)
        # Snapshot inheritors before terminating their parent; ancestry is lost
        # on reparenting, so a later owner inventory cannot recover this scope.
        rows = table()
        matched = {r.pid for r in claim.holders(claim.ClaimDescriptor(9))}
        descendants = [NativeChild(p, rows[p][3], None, None, None)
                       for p in matched if p != pid and within(p, pid, rows)]
        end([child, *descendants])
    elif mode == 'settings':
        print('\n'.join(map(str, settings()[2:])))
    elif mode == 'evidence':
        claim.verify_cleanup(8)
        claim.inherited(9)
        target = claim.state_dir() / 'cleanup.json'
        temporary = target.with_name(f'cleanup-{root}.tmp')
        try:
            with temporary.open('w') as stream:
                json.dump(dict(pid=root, at=time.time(), line=args[0], cat=args[1]), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        inventory()
except Exception as exc:
    print(f'NOT CONFIRMED: {exc}', file=sys.stderr, flush=True)
    raise SystemExit(1)
CLAIM_PROCESSES
}

# One rule for the CAT-to-PTT relationship, and it is the station's own:
# `core.ptt.derive_ptt_port` advances the interface index of a two-port adapter
# and refuses every other shape, rather than advancing any trailing digit — which
# on other naming conventions would name a wholly different adapter, quite
# possibly another radio's CAT port. Its refusal is the message printed here.
derive_ptt() {
  "${HFMODEM_PYTHON:-$VENV/python}" - "$1" <<'PY'
import sys
from hfmodem.core.ptt import derive_ptt_port
try:
    print(derive_ptt_port(sys.argv[1]))
except ValueError as exc:
    sys.exit(f"!! {exc}")
PY
}
