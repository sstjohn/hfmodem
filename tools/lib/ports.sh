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

# ONE OWNER OF THE TRANSMITTER AT A TIME, AND THE CLAIM DIES WITH ITS HOLDERS.
#
# On 2026-08-28 an arm was judged not to have started and launched a second time
# while the first was still running. Two modems keyed one rig for eight seconds,
# and the gateway's connect-response and three preamble alignments fell inside
# that window -- the arm cannot be scored in either direction. The surviving run
# did detect it, from a 53 dB collapse in its own receiver, but that reading only
# arrives once both are already transmitting. This is the same refusal, taken
# before either one is capable of keying.
#
# An flock on a descriptor, not a pidfile: the kernel closes descriptors however
# a process ends -- SIGKILL, a pulled cable, a panic -- so the claim cannot
# outlive the processes holding it. It is inherited, which is the second half of
# the property: an arm killed at the shell but still running a modem has not
# released the transmitter, and is still refused. A file left behind by a crash,
# refusing every launch after it, would be a worse fault at a rig than the one
# this is here to stop.
#
# /tmp and not $TMPDIR: on macOS $TMPDIR is per-login-session, so two shells that
# disagree about it would each claim a different file and neither would see the
# other. Named for the CAT port, so a second radio on this machine is a second
# claim rather than a queue behind the first.
RIG_LOCK_FD=9
claim_rig() {
  require_cat
  # `pounce`, rigtest.sh and autopilot's own `onair.sh check` reach the rig by
  # re-invoking a launcher, and this is what tells the child the claim is already
  # its own tree's. The child must not take it again: it inherits the descriptor,
  # but a second OPEN of the same file is a second holder and would refuse.
  [ -n "${RIG_CLAIMED:-}" ] && return 0
  local lock="/tmp/hfmodem-rig-$(basename "$CAT_PORT").claim"
  # Appended, never truncated, on the way in: a launcher that is about to be
  # refused opens this file too, and it must not erase the description of the run
  # it is about to be told about.
  eval "exec $RIG_LOCK_FD>>'$lock'"
  if ! "${HFMODEM_PYTHON:-$VENV/python}" - "$RIG_LOCK_FD" <<'LOCK'
import fcntl, sys
try:
    fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    raise SystemExit(1)
LOCK
  then
    # Ours goes before we ask who else has it open, so the answer is the holders
    # and not this process standing in its own light.
    eval "exec $RIG_LOCK_FD>&-"
    die "THE RIG IS ALREADY CLAIMED, and nothing here has keyed or opened a port.

   Two modems on one transmitter is how 2026-08-28's arm 2 was lost: eight
   seconds of overlap swallowed the gateway's answer, and the arm could not be
   scored in either direction.

   Claimed $(sed -n 1p "$lock" 2>/dev/null) by

     $(sed -n 2p "$lock" 2>/dev/null)

   Still holding the rig now:

$(claim_holders "$lock")

   Let it finish, or stop it and call again:

     tools/onair.sh down"
  fi
  : > "$lock"
  printf 'at %s, pid %s\n%s\n' "$(date -u +%H:%M:%SZ)" "$$" \
      "$(ps -o command= -p $$ 2>/dev/null | redact_secrets)" >> "$lock"
  RIG_CLAIMED="$lock"; export RIG_CLAIMED
}

# WHAT IS STILL ON THE RIG, WHICH IS NOT THE SAME QUESTION AS WHAT STARTED IT.
#
# The claim is a descriptor and a descriptor is inherited, so a launcher's modem,
# its recorder and its witness all carry it and the claim lasts exactly as long
# as one of them could still key. That is the property wanted -- a SIGKILLed
# launcher whose modem is still transmitting has not released the transmitter --
# and it is why the pid in the lock file is not the answer: on that path it names
# the one process that is gone. So the refusal reads the descriptor back and
# names every process still holding it, which is also the list `onair.sh down`
# has to clear.
claim_holders() {
  local pid found=
  for pid in $(lsof -t "$1" 2>/dev/null); do
    found=1
    printf '     %s\n' "$(ps -o command= -p "$pid" 2>/dev/null | redact_secrets)"
  done
  # Between another launcher's `exec 9>>` and its own `flock`, briefly, and after
  # a holder exits before this reads the list. Neither is a state to guess in.
  [ -n "$found" ] || printf '     (nothing has it open now -- ask again in a moment)\n'
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
