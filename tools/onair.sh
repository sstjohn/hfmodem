#!/usr/bin/env bash
# W9SSJ / FT-891 on-air launcher. One command per modem, so a rig window is spent
# transmitting rather than rediscovering how to start things.
#
# Everything here was learned the hard way on 2026-07-30. The comments say which
# part of the evening each line is standing in for.
#
#   ./tools/onair.sh check                      # no TX: every prerequisite, then stop
#   ./tools/onair.sh listen 7101500 [300]       # no TX: tune and decode what is there
#   ./tools/onair.sh power  [7101500]           # no TX: what the transmitter is set to
#   ./tools/onair.sh preflight                  # keys unmodulated; measures the rig
#   ./tools/onair.sh pactor WS8EOC [7101500]    # PACTOR-1 connect  (rigctld OFF)
#   ./tools/onair.sh vara   K9WRA  [7103800]    # VARA connect      (rigctld ON, RTS)
#   ./tools/onair.sh ardop  NS0A   [7103500] [BW]  # ARDOP connect  (rigctld tunes only)
#   ./tools/onair.sh ardop-mail GW CENTRE BW --mail-…   # Winlink exchange, ARDOP
#   ./tools/onair.sh vara-mail GW CENTRE [BW] --mail-…  # Winlink exchange, VARA
#            # BW is 2300 (the default), 500, or 2750 — 2750 is bench-proven only
#   ./tools/onair.sh sabir 14100000             # ONE-WAY beacon for a WebSDR
#   ./tools/onair.sh rtty 14080000              # RTTY terminal; you key every over
#   ./tools/onair.sh pounce vara K9WRA 7103800  # wait out the occupant, then go
#   ./tools/onair.sh down                       # stop everything, prove PTT is low
#
# `down` goes BEFORE an arm on a pasted line, never after it: a trailing one is
# still running when the next arm begins, and it may only stop what was there
# when it was issued -- anything started later is named and left alone. Every
# verb that opens the rig first waits, a few seconds at most, for the previous
# arm's daemon to let go of the CAT port, and says how long that took.
#
# --witness adds a third-party KiwiSDR recording of the arm, alongside our own
# capture and on a clock that is not ours. It is placed once the gate has passed
# the channel, so an arm refused an occupied channel takes nobody's receiver and
# says nothing about a witness. It goes anywhere on the line and every keying
# verb takes it:
#
#   ./tools/onair.sh pactor WS8EOC 7101500 --witness
#
# The recording lands beside the session's own as captures/onair-MMDD-HHMM/
# witness.wav, and tools/witness_align.py registers the pair. It cannot delay or
# refuse an arm: a Kiwi with no free slot leaves the transmission untouched and
# says NO WITNESS at the end of it.
#
# The receiver is chosen near the GATEWAY, which is the station our own receiver
# is muted against while we key; the arm's teardown says which one it got, how far
# that one is from the gateway, and what it actually heard, so an arm that had no
# cover of the peer says so where it is read rather than in the write-up.
#
# --log NAME keeps the arm's transcript itself, so no `| tee` is needed and none
# should be used. It goes anywhere on the line and every verb takes it:
#
#   ./tools/onair.sh vara KE8LVA 7103800 --log vara-day-01-ke8lva
#
# The transcript lands as working/onair-MMDD-HHMM/NAME.log, under the same minute
# the session's audio is filed by and inside the index, so a report can cite it
# and a clone can open it. It holds every line the arm printed; what the terminal
# shows is the contact, and `ONAIR_VIEW=full ./tools/onair.sh ...` shows the rest
# of it there too.
#
# --force does NOT wait. It is the operator saying the channel is clear now, so
# the sense runs once, its reading is recorded, and the refusal is waived --
# POUNCE_WAIT is not spent reaching a verdict that has already been overruled.
# It waives the receive level too, which decides whether the arm can be read back
# rather than whether it may go out. It waives nothing about a codec that is
# delivering no audio at all: there is no station there to overrule.
#
# POUNCE_WAIT is how long an occupied channel is waited out rather than refused,
# in seconds. Every verb in this file honours it — pactor, vara, ardop, both
# -mail forms and sabir — at 900 under `pounce` and 0, refuse at once,
# everywhere else. So does tools/autopilot.sh, whose --wait is the same budget
# said at the command line. tools/campaign.sh alone does not: it is unattended by
# definition, and its gate says out loud that it is dropping the budget rather
# than discarding it in silence.
#
# The serial ports come from the environment and this file has no default for
# them, because a device name is per-machine:
#   export CAT_PORT=…    the rig's CAT interface — `ls /dev/cu.*` to find it
#   export PTT_PORT=…    the line whose RTS keys the rig; on a two-interface
#                        adapter it is derived from CAT_PORT and can be left unset
#   export HAMLIB_BIN=…  only if your hamlib is a build of your own; a
#                        package-manager one is on PATH already
#   export HFMODEM_STATION=…  your station .toml. Every modem started from here
#                        then transmits at its [audio] tx_drive; without it they
#                        fall back to core.levels.TX_DRIVE
#   export WINLINK_PASSWORD=…  the secure-login answer for the -mail verbs. It
#                        is read from here, or from --mail-password-file FILE,
#                        and is never passed on to a modem as an argument.
#                        `--mail-password` STILL WORKS and is still a mistake:
#                        it is readable in `ps` by anything on this machine.
#
# Frequencies are the WINLINK PUBLISHED CENTRE. Every tool here derives
# dial = centre - 1500. Getting that backwards is silent and kills both
# directions at once.
set -euo pipefail

# REPO, VENV, the two serial ports and the refusal when they are unset, and
# hamlib on PATH for everything started from here -- shrike, besra and sabir all
# run `rigctl` by name, and a build from source is on no machine's PATH. Shared
# with the three unattended launchers, which answer to the same environment.
source "$(dirname "${BASH_SOURCE[0]}")/lib/ports.sh"
# The rig and the codec this station drives, `tune`, and the channel sense --
# shared with the unattended launchers, which need the verified dial and the
# same instrument held to a stricter rule.
source "$(dirname "${BASH_SOURCE[0]}")/lib/rig.sh"
MYGRID="${MYGRID:-}"
identity_verb="${1:-}"
[ "$identity_verb" != "--capture-prepared" ] || identity_verb="${2:-}"
case "$identity_verb" in
  pactor*|vara*|ardop*|sabir)
    MYCALL="$(station_mycall)"
    [ -n "$MYCALL" ] || die "set MYCALL or [station] mycall before calling"
    ;;
  rtty) MYCALL="$(station_mycall)" ;;
esac
case "$identity_verb" in
  pactor|pactor-answer|vara|vara-mail|ardop-mail)
    # Resolve final identity flags before CAT/audio setup. Freeze the fallback
    # so edits to the station file during a pounce cannot change this attempt.
    # Module launchers must validate in their actual snapshot environment.
    # VARA's script inserts its own package tree; mirror only that launcher.
    identity_pythonpath="${PYTHONPATH:-}"
    case "$identity_verb" in
      vara|vara-mail) identity_pythonpath="$REPO/packages/hfmodem${PYTHONPATH:+:$PYTHONPATH}" ;;
    esac
    call_identity=$(PYTHONPATH="$identity_pythonpath" \
      "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.identification \
      --mycall "$MYCALL" "${@:2}") || exit "$?"
    { IFS= read -r MYCALL; IFS= read -r IDCALL; } <<< "$call_identity"
    export MYCALL IDCALL
    ;;
esac
OPERATOR_DATA="${HFMODEM_DATA_DIR:-$REPO/operator-data}"

# PACTOR evidence is claimed before transcript/witness/modem/device work. The
# pure helper re-execs this same script with its exclusively opened transcript
# descriptor retained; other modem launchers keep their existing flow.
CAPTURE_PREPARED=0
if [ "${1:-}" = "--capture-prepared" ]; then
  CAPTURE_PREPARED=1
  shift
else
  if "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/capture_identity.py" classify --repo "$OPERATOR_DATA" -- "$@"; then
      exec "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/capture_identity.py" launch \
        --repo "$OPERATOR_DATA" --script "${BASH_SOURCE[0]}" -- "$@"
  fi
fi
if [ "$CAPTURE_PREPARED" = 1 ]; then
  # The helper emits only fixed, quoted metadata assignments after checking the
  # explicit registered identity, final --outdir and actual retained descriptor.
  capture_metadata="$("${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/capture_identity.py" prepared --repo "$OPERATOR_DATA" -- "$@")"
  eval "$capture_metadata"
  capture_exit() {
    capture_rc=${1:-$?}
    if declare -F witness_stop >/dev/null; then witness_stop || true; fi
    "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/capture_identity.py" ended --repo "$OPERATOR_DATA" \
      --returncode "$capture_rc" -- --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" || true
    return "$capture_rc"
  }
  capture_transcript_finish() {
    capture_outcome=finalized
    [ "$1" = 0 ] || capture_outcome=partial
    "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/capture_identity.py" transcript-complete --repo "$OPERATOR_DATA" \
      --outcome "$capture_outcome" -- --capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT"
  }
  trap 'capture_exit' EXIT
fi

# ONE NAME FOR THIS RUN, DECIDED ONCE. pactor and sabir each formed exactly this
# string for their captures directory after their gate; it is settled here so the
# witness recording can be written into the same directory as `stream.wav` while
# the arm is still running, and the pair is then one `ls` rather than a search by
# clock. Under a pounce the name is the minute the verb started rather than the
# minute it keyed, which is what the witness log inside it is stamped with too.
if [ "$CAPTURE_PREPARED" = 0 ]; then
  SESSION="$OPERATOR_DATA/captures/onair-$(date +%m%d-%H%M)"
fi

# THE ARM'S TRANSCRIPT IS OPENED HERE, UNDER THE MINUTE THIS RUN OWNS.
#
# The runbooks used to end every arm with `| tee ~/<arm>.log`. The shell opens
# that file with O_TRUNC before this script runs a line, so relaunching an arm
# under the same name destroyed the first run's transcript -- and destroyed it
# whether or not the second launch was allowed to proceed. That is what happened
# to ~/vara-day-02-ke8lva.log on 2026-08-28: the relaunch was refused by the rig
# claim, nothing keyed, and all that was left of the arm was the refusal.
#
# The minute is the run's identity and it is already decided, so the transcript
# is filed by it and a relaunch cannot reach the earlier one. Within one minute
# the open appends, because two runs the clock cannot tell apart are still two
# runs and neither is worth less than the other.
#
# NAME is the arm's own, never a fixed word: hfmodem.tests.evidence keys logs by
# bare basename and the last path wins, so a tree full of arm.log would answer a
# report's quotation with some other slot's run and the gate would pass pointing
# at the wrong evidence.
#
# `working/` rather than `captures/` because it is the only one of the three
# evidence trees in the index: a transcript under a gitignored tree satisfies the
# quotation gate on this disk and leaves a clone with nothing. It costs the
# distribution nothing either -- publish/manifest.toml denies working/**.
#
# `pounce` re-invokes this file and hands the child --witness, but not this one:
# the child writes down the pipe this exec installed, so its output is in the
# transcript already, and passing it on would open a second tee onto one file.
LOG_NAME=
log_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --capture-id|--capture-root|--capture-parent-id|--capture-parent-kind) shift 2 ;;
    --capture-id=*|--capture-root=*|--capture-parent-id=*|--capture-parent-kind=*) shift ;;
    --log)   LOG_NAME="${2:?a name must follow --log}"; shift 2 ;;
    --log=*) LOG_NAME="${1#*=}"; shift ;;
    *) log_args+=("$1"); shift ;;
  esac
done
set -- ${log_args[@]+"${log_args[@]}"}
#
# WHAT A PERSON SEES IS THE FILTERED END OF THAT TEE, AND ONLY WHEN IT IS A PERSON.
#
# `tee` is upstream and untouched, so the transcript is the whole stream exactly
# as it always was -- the sweeps, tools/rehear and the console's milestone reader
# all read sample indices and per-cycle boundaries out of it. consolelib/view.py
# drops only the copy going to a screen, where five or six lines of grid
# arithmetic a cycle bury the one that says what came back.
#
# `-t 1` is the whole of the condition, and it is not a convenience. The console
# runs this launcher down a pipe and writes ITS stdout to the arm.log every
# analysis reads (consolelib/arm.py), so a filter here would empty that file.
# A pipe is another program's input and keeps everything; a terminal is a person
# watching a contact.
#
# ONAIR_VIEW=full shows every line, without editing anything mid-slot.
if [ "$CAPTURE_PREPARED" = 1 ] || [ -n "$LOG_NAME" ]; then
  if [ "$CAPTURE_PREPARED" = 0 ]; then
    ARM_LOG="$OPERATOR_DATA/working/${SESSION##*/}/$LOG_NAME.log"
    mkdir -p "${ARM_LOG%/*}"
  fi
  transcript_target="$ARM_LOG"
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    transcript_target="/dev/fd/$HFMODEM_CAPTURE_FD"
  fi
  if [ "$CAPTURE_PREPARED" = 1 ]; then
  if [ -t 1 ] && [ -x "${HFMODEM_PYTHON:-$VENV/python}" ]; then
    exec > >(capture_tee_rc=0; tee -a "$transcript_target" | "${HFMODEM_PYTHON:-$VENV/python}" -u "$REPO/tools/consolelib/view.py" || capture_tee_rc=$?; capture_transcript_finish "$capture_tee_rc") 2>&1
  else
    exec > >(capture_tee_rc=0; tee -a "$transcript_target" || capture_tee_rc=$?; capture_transcript_finish "$capture_tee_rc") 2>&1
  fi
  else
  if [ -t 1 ] && [ -x "${HFMODEM_PYTHON:-$VENV/python}" ]; then
    exec > >(tee -a "$transcript_target" | "${HFMODEM_PYTHON:-$VENV/python}" -u "$REPO/tools/consolelib/view.py") 2>&1
  else
    exec > >(tee -a "$transcript_target") 2>&1
  fi
  fi
  printf '  transcript: %s\n' "$ARM_LOG"
fi

capture_modem_args=()
capture_witness_args=()
if [ "$CAPTURE_PREPARED" = 1 ] && [ "$CAPTURE_KIND" = arm ]; then
  capture_modem_args=(--capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" --capture-role modem)
  capture_witness_args=(--capture-id "$CAPTURE_ID" --capture-root "$CAPTURE_ROOT" --capture-role launcher-witness)
fi

# WHICH TREE KEYED THE RADIO, WRITTEN DOWN WHERE THE ARM IS READ.
#
# Every transcript names the gateway, the centre, the bandwidth and the codec
# gain, and none of them named the code. So a review reconstructing 2026-08-31's
# arms had to date each one against the commit log, and 05:14z resisted it: that
# arm ran on edits nobody had committed, and it cannot be reproduced.
#
# It is taken here, once, before any verb reads its positionals, so every arm in
# this file carries it and no protocol has to know. `pounce` re-invokes the
# launcher and the child inherits the answer rather than asking again.
#
# git is bounded and its failure is a word, never a wait: nothing in a rig window
# may queue behind a version-control question, so an unanswering repository is
# `tree: unknown` and the arm goes on to transmit.
TREE_TIMEOUT="$(command -v timeout || command -v gtimeout || true)"
tree_git() { ${TREE_TIMEOUT:+$TREE_TIMEOUT 2} git -C "$REPO" "$@" 2>/dev/null; }
tree_line() {
  local head branch dirty
  head="$(tree_git rev-parse --short HEAD)" || true
  [ -n "$head" ] || { printf '  tree: unknown\n'; return 0; }
  branch="$(tree_git rev-parse --abbrev-ref HEAD)" || true
  # Tracked paths only: an arm fills working/ with its own output while it runs,
  # and that is not a change to the code that ran.
  dirty="$(tree_git status --porcelain --untracked-files=no | cut -c4- | paste -sd, - | sed 's/,/, /g')" || true
  if [ -n "$dirty" ]; then
    printf '  tree: %s (%s) dirty — %s\n' "$head" "${branch:-?}" "$dirty"
  else
    printf '  tree: %s (%s) clean\n' "$head" "${branch:-?}"
  fi
}
if [ -z "${ONAIR_TREE:-}" ]; then
  ONAIR_TREE="$(tree_line)"; export ONAIR_TREE
  printf '%s\n' "$ONAIR_TREE"
fi
# --witness is the whole command line's and not any one verb's, so it is taken
# out here, before a verb reads its positionals -- no arm parses it, and no modem
# can be handed it by a shift that lands one place to the left.
WITNESS=
witness_args=()
for a in "$@"; do
  case "$a" in --witness) WITNESS=1 ;; *) witness_args+=("$a") ;; esac
done
set -- ${witness_args[@]+"${witness_args[@]}"}

# THE WINLINK PASSWORD LEAVES ARGV HERE AND DOES NOT GO BACK.
#
# A secret spelled out as a flag argument is readable in `ps` by every process on
# this machine for the life of the run, and it is copied by anything that echoes
# a command line. Both happened: claim_rig records its own `ps -o command=` and
# the next launcher's "THE RIG IS ALREADY CLAIMED" prints it back, which put the
# password into a tee'd slot log on 2026-08-28.
#
# So `--mail-password` and `--mail-password-file` are taken off the line before
# any verb reads it, and what reaches the modem is WINLINK_PASSWORD in the
# environment, which no `ps` shows and no echo of a command line can carry. Every
# tool below honours it: shrike, kestrel and besra all resolve through
# `core.config.mail_password`. `pounce` re-invokes this file, and an exported
# variable survives that where a stripped flag would not.
pw_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mail-password)      WINLINK_PASSWORD="${2:?the password must follow --mail-password}"; shift 2 ;;
    --mail-password=*)    WINLINK_PASSWORD="${1#*=}"; shift ;;
    --mail-password-file) WINLINK_PASSWORD="$(cat "${2:?--mail-password-file needs a path}")"; shift 2 ;;
    --mail-password-file=*) WINLINK_PASSWORD="$(cat "${1#*=}")"; shift ;;
    *) pw_args+=("$1"); shift ;;
  esac
done
set -- ${pw_args[@]+"${pw_args[@]}"}
export WINLINK_PASSWORD="${WINLINK_PASSWORD:-}"

# A flag that silently did not apply is the trap this file spends its length
# avoiding, so a verb with nothing to witness says so rather than ignoring it.
if [ -n "$WITNESS" ]; then
  case "${1:-}" in
    pactor|vara|ardop|ardop-mail|vara-mail|sabir|pounce) ;;
    *) die "--witness records a third-party receiver alongside a transmission, \
and '${1:-}' does not key" ;;
  esac
fi

# The wait budget is judged here, once, before anything opens. It reaches the
# sense as a `float()` in a heredoc, so `POUNCE_WAIT=15m` used to die inside that
# python -- which the caller reads as exit 1, which is BUSY, which is a refusal
# naming an occupant that was never measured. A misspelt budget must not come
# back as somebody on the channel.
case "${POUNCE_WAIT:-0}" in
  *[!0-9]*) die "POUNCE_WAIT is whole seconds, and '$POUNCE_WAIT' is not" ;;
esac

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# WHERE THE SECONDS BEFORE THE GATE WENT. The stretch between a campaign's dwell
# and key-down was accounted for once by reading the code, and the reading named
# the wrong item -- the witness, which is backgrounded and costs the arm nothing
# at all. Nothing in this file measured it, so the answer could only ever be a
# guess at which step was the expensive one.
#
# Each step ahead of the gate closes the one before it here, and the gate prints
# the line. Whole seconds off the shell's own clock: what is wanted is the step
# worth seconds, not the four that are worth milliseconds.
SETUP_AT=0
SETUP_LAST=
SETUP_MARKS=
took() {             # <the step that has just ended>
  # rigctld_start calls rigctld_stop, and the nested one must not close the
  # segment its caller already closed -- a repeat of the same name is that.
  if [ "$1" != "$SETUP_LAST" ]; then
    SETUP_MARKS="${SETUP_MARKS:+$SETUP_MARKS, }$1 $((SECONDS - SETUP_AT))s"
    SETUP_LAST="$1"; SETUP_AT=$SECONDS
  fi
}
before_the_gate() {
  took power
  printf '  before the gate: %s (%ss since launch)\n' \
      "${SETUP_MARKS:-nothing timed}" "$SECONDS"
}

# A flag standing where a number should be is the quiet way to lose a session.
# Every keying verb takes its channel, and some a bandwidth, as bare positionals
# before the flags, so `ardop-mail GW 7103600 --mail-fetch` reads the bandwidth as
# `--mail-fetch` and shifts everything after it by one: the modem is then handed a
# bandwidth it cannot parse, refuses the whole call, and reports a usage error that
# says nothing about the frequency the operator actually meant. Numbers here never
# begin with a dash, so this cannot refuse a real one.
positional() {   # <what it is> <value>
  case "$2" in -*) die "<$1> was given the flag '$2'. Put $1 in before the flags -- \
otherwise it is read as $1 and every flag after it lands one place to the left." ;; esac
}

# WHAT ACTUALLY HAD TO BE STOPPED, appended to by the three functions that do the
# stopping. `down` closes on it, and the two sentences it chooses between --
# "nothing was running" and "stopped: ..." -- are the ones the operator reads
# under a clock to decide whether the station is theirs again.
STOPPED=
stopped_a() { STOPPED="$STOPPED $1,"; }

# `down` MAY ONLY REACH WHAT EXISTED WHEN IT WAS ISSUED. Arm 11 of 2026-09-03 was
# keyed over by the previous arm's trailing `down`, pasted after it and still
# running when the next arm began: it found that arm's modem on the keying line
# and sent it the same SIGTERM it sends a stale one. The instant of issue is a
# pid snapshot taken as the verb starts; a pid absent from it started later and
# belongs to whoever started it. Outside `down` there is no snapshot and every
# pid passes, so a launch clearing a stale daemon is unchanged.
DOWN_PIDS=
FENCED=
KEPT=
issued_before() {       # <pids…> -> KEPT holds the ones this verb may signal
  local pid
  KEPT=
  for pid in "$@"; do
    if [ -z "$DOWN_PIDS" ] || [[ "$DOWN_PIDS" == *" $pid "* ]]; then
      KEPT="$KEPT$pid "
    elif [[ "$FENCED" != *" $pid "* ]]; then
      FENCED="$FENCED $pid "
      printf "  pid %s started after this down was issued -- not this down's to stop:\n     %s\n" \
          "$pid" "$(ps -o command= -p "$pid" 2>/dev/null | redact_secrets)"
    fi
  done
  return 0
}
# Between stages: once a later arm is on the station, the line, the CAT port and
# the receiver reading are its, and nothing below may reach for them.
fenced_exit() {
  [ -n "$FENCED" ] || return 0
  say "an arm started after this down was issued (pid$(printf '%s' "$FENCED" | tr -s ' ' | sed 's/ $//')) — the station is its, and nothing of it was stopped"
  exit 0
}

# THE PORT COMES FREE AFTER `down` RETURNS, NOT BEFORE. Arm 04 of 2026-09-03 was
# launched on the same line as the `down` before it, met a CAT port the previous
# daemon still held, and died at the dial readback; every arm after it waited on
# lsof and pgrep by hand and settled inside a second. Bounded, so it can hold a
# transmission the operator asked for by at most this long, and then goes on --
# a holder still there is named by the refusal that meets it.
SETTLE_MAX_S=5
port_unsettled() {
  lsof -t "$CAT_PORT" >/dev/null 2>&1 || pgrep -f "rigctld -m $MODEL" >/dev/null 2>&1
}
settle_port() {
  local waited=0
  while [ "$waited" -lt "$SETTLE_MAX_S" ] && port_unsettled; do
    sleep 1; waited=$((waited + 1))
  done
  if port_unsettled; then
    printf '  !! %s not settled after %ss -- going on\n' "$CAT_PORT" "$SETTLE_MAX_S"
  else
    printf '  port settled after %ss\n' "$waited"
  fi
  return 0
}

# SIGTERM is not enough and waiting a second for it is not either. A rigctld that
# has stopped answering is usually blocked in a read on the CAT port, where it
# ignores TERM and keeps the port — so the next daemon cannot open the radio, and
# the launcher blames the rig. Ask nicely, then insist, then wait for the port
# itself to come free rather than for a guess at how long that takes.
rigctld_stop() {
  took hardware
  local had= rc=0 pids
  issued_before $(pgrep -f "rigctld -m $MODEL" 2>/dev/null || true); pids="$KEPT"
  [ -z "$pids" ] || { had=1; kill -TERM $pids 2>/dev/null || true; }
  for _ in $(seq 10); do
    [ -n "$(alive_of $pids)" ] || break
    sleep 0.5
  done
  if [ -n "$(alive_of $pids)" ]; then
    echo "  a rigctld ignored SIGTERM — it is wedged on the CAT port; killing it"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  # AND THEN THE RESOURCE, BECAUSE THE NAME IS NOT THE RESOURCE. Everything above
  # matches one spelling of one command line. A daemon started by hand, one left
  # by a build under another model number, one written `-m1036`: none of them is
  # found there, and every one of them goes on listening on 4532 after this says
  # the station is stopped. `free_port` learned the same lesson about the serial
  # devices on 2026-08-02 -- ask the resource who holds it -- and this is the
  # port that was still being cleared by hand before arms 4, 5 and 6.
  if lsof -t -nP -iTCP:4532 -sTCP:LISTEN >/dev/null 2>&1; then
    had=1
    rigctld_4532_free || rc=1
  fi
  # THE TWO ARE NOT THE SAME SENTENCE. `down` is read mid-slot against a clock,
  # and "already stopped" was reaching the operator as silence -- indistinguishable
  # from a verb that had done nothing because it had gone wrong.
  if [ -n "$had" ]; then
    echo "  rigctld stopped"; stopped_a rigctld
  else
    echo "  no rigctld was running, and 4532 is free"
  fi
  for _ in $(seq 10); do
    issued_before $(lsof -t "$CAT_PORT" 2>/dev/null || true)
    [ -n "$KEPT" ] || return "$rc"
    sleep 0.5
  done
  # A port still held after all of that is a real inability to stop something,
  # and those are the only conditions here worth a non-zero status.
  echo "  !! something still holds $CAT_PORT"
  return 1
}

# The daemon's socket, asked directly, and only when the caller has already found
# something on it.
rigctld_4532_free() {
  local sig pid pids who
  for sig in TERM KILL; do
    issued_before $(lsof -t -nP -iTCP:4532 -sTCP:LISTEN 2>/dev/null || true); pids="$KEPT"
    [ -n "$pids" ] || return 0
    RIGCTLD_4532_HELD=1
    for pid in $pids; do
      who="$(ps -o command= -p "$pid" 2>/dev/null | redact_secrets)"
      case "$who" in
        # A rigctld here runs ptt_type=None and holds no key, so there is nothing
        # for it to put down and the escalation costs a transmitter nothing --
        # which is exactly why `reap_orphans` and `free_port`, which do reach
        # things that key, may not escalate the same way.
        *rigctld*) printf '  4532 listened on by pid %s -- SIG%s\n' "$pid" "$sig"
                   kill "-$sig" "$pid" 2>/dev/null || true ;;
        # Not ours, and this is a shared machine: named, and left alone. The next
        # launcher still cannot have the port, and it will now be told why by
        # something other than a daemon that mysteriously will not start.
        *) printf '  !! 4532 IS HELD AND NOT BY A RIGCTLD -- not ours to signal:\n     %s\n' \
               "${who:-pid $pid, already gone}"
           return 1 ;;
      esac
    done
    sleep 1
  done
  issued_before $(lsof -t -nP -iTCP:4532 -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$KEPT" ] || return 0
  printf '  !! 4532 IS STILL LISTENED ON -- the next daemon would not bind it\n'
  return 1
}

# RELEASE THE HARDWARE BY ASKING THE DEVICES, NOT BY GUESSING PROCESS NAMES.
#
# `down` used to pkill three command patterns. Three separate holders survived
# that on 2026-08-02, each of them silently spoiling every session that followed:
# a kestrel that ignored SIGTERM and sat on the codec for 82 minutes; a rigctld
# that answered nothing while still owning the CAT port, so `rigctl f` timed out
# and every launcher read it as a dead radio; and an orphaned analysis script
# whose multiprocessing children crash-looped for two hours because a heredoc
# parent cannot be re-imported by a spawned worker.
#
# None of them matched the patterns. All three were visible in `lsof` on the
# port, or as our own code reparented to init. So this asks the serial devices
# who holds them and escalates, rather than trusting a list of names to be
# complete -- a list is only ever as good as the last failure someone remembered.
free_port() {
  local dev="$1" sig pid held asked=
  for sig in TERM KILL; do
    issued_before $(lsof -t "$dev" 2>/dev/null || true); held="$KEPT"
    if [ -z "$held" ]; then
      if [ -n "$asked" ]; then printf '  %s is free\n' "$dev"; stopped_a "$(basename "$dev")"
      elif [ -n "$FENCED" ]; then printf '  %s is left to the arm above\n' "$dev"
      else printf '  %s was already free\n' "$dev"; fi
      return 0
    fi
    asked=1
    for pid in $held; do
      printf '  %s held by pid %s -- SIG%s\n' "$dev" "$pid" "$sig"
      kill "-$sig" "$pid" 2>/dev/null || true
    done
    sleep 2
  done
  issued_before $(lsof -t "$dev" 2>/dev/null || true)
  if [ -n "$KEPT" ]; then
    printf '  !! %s STILL HELD\n' "$dev"
    return 1
  fi
  printf '  %s is free\n' "$dev"
  stopped_a "$(basename "$dev")"
  return 0
}

# How long an orphan gets to put its own key down before it is insisted on. Every
# modem here unkeys on the way out of a SIGTERM -- the handler raises, the
# `finally` chain drops the line, the codec drains -- and onair_session.py budgets
# 12 s for a child doing exactly that work. Spent by polling, so a modem that goes
# quietly costs nothing at all.
REAP_GRACE_S=12

# Which of these pids are still standing. A function rather than an inline loop
# because a `kill -0` that finds nothing exits 1, and `left=$(...)` inheriting
# that status ends the run under `set -e` -- in the case where every orphan died
# on the first ask, which is the good one.
alive_of() {
  local pid
  for pid in "$@"; do kill -0 "$pid" 2>/dev/null && printf '%s ' "$pid"; done
  return 0
}

# Our own processes that outlived the shell that started them. Matched on this
# repository's path so a reparented worker of ours is caught and the operator's
# unrelated work is not: this runs on a shared machine and must never reach for
# anything it did not start.
#
# ASK, WAIT, THEN INSIST. This opened with SIGKILL, which is the one signal a
# modem cannot unkey through: the handler, the `finally` chain and the keying
# watchdog all die with it, and whatever the line was doing it goes on doing --
# process death is not known to lower it on this hardware (docs/STATION.md). An
# orphan is exactly the process most likely to be holding a transmitter up, so it
# gets the same escalation `hfhost.supervisor._terminate` gives a modem it
# recycles, and for the same reason. The `down` verb reads the line back once the
# ports are free, because a signal is not evidence.
reap_orphans() {
  local pid pids left
  issued_before $(ps -Ao pid,ppid,command | awk -v r="$REPO" \
           '$2==1 && index($0,r) && !/onair\.sh/ {print $1}'); pids="$KEPT"
  # `cond && printf` as the last statement makes the CONDITION this function's
  # exit status, so reaping anything returned 1 -- and under `set -e` that ended
  # `down` before it freed PTT_PORT or reached prove_unkeyed. The cleanup verb
  # aborted in exactly the case that needed it.
  if [ -z "$pids" ]; then printf '  no orphaned workers\n'; return 0; fi
  stopped_a "orphaned worker(s)"
  for pid in $pids; do
    printf '  orphan pid %s (parent gone) -- SIGTERM\n' "$pid"
    # The children too, and by name: an orphan's own workers key nothing, but a
    # reparented modem under it does, and it is not reachable through a dead
    # parent's handler.
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq $((REAP_GRACE_S * 2))); do
    left=$(alive_of $pids)
    [ -n "$left" ] || break
    sleep 0.5
  done
  left=$(alive_of $pids)
  for pid in $left; do
    printf '  orphan pid %s ignored SIGTERM for %ss -- SIGKILL, so it never unkeyed\n' \
        "$pid" "$REAP_GRACE_S"
    pkill -9 -P "$pid" 2>/dev/null || true
    kill -9 "$pid" 2>/dev/null || true
  done
  [ -z "$left" ] || sleep 1
  return 0
}

# The daemon never holds the key. Every armed verb drives RTS on $PTT_PORT
# itself and REFUSES to arm if the daemon also owns PTT — two owners of one
# transmitter is how it keys unasked — so rigctld runs ptt_type=None and tunes,
# which it does before there is any RF about. The last verb keying through the
# daemon was `vara`, and on 2026-08-13 its wire records ended the argument:
# rigctld answered every command until the first key-down, then answered
# nothing — connect in 1 ms, `T 0` swallowed, 751 ms of silence, four times
# over, ~4.7 s of carrier past the end of audio on every attempt — while the
# line itself read LOW in microseconds, eight of eight. The socket was healthy
# the whole time; it is the daemon's rig side that dies with PTT asserted. A
# path that must work while the antenna radiates does not travel that.
# shrike is the one exception to the daemon itself: it opens the CAT port
# directly and cannot share it, so it runs with no daemon at all.
RIGCTLD_PID=
rigctld_cleanup() {
  # Only this shell's child: an EXIT trap must never sweep up another arm.
  # The daemon inherits the rig claim, so even a pre-key refusal must reap it.
  local pid="${RIGCTLD_PID:-}" i
  [ -n "$pid" ] || return 0
  if jobs -pr | grep -qx "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    echo "  rigctld pid $pid stopped (launcher cleanup)"
  fi
  wait "$pid" 2>/dev/null || true
  RIGCTLD_PID=
  return 0
}

rigctld_start() {
  took hardware
  local ptt="$1"
  [ "$ptt" = none ] || die "rigctld never holds the key (asked for ptt=$ptt)"
  rigctld_stop || die "CAT cleanup failed -- cannot start another rigctld while the previous owner remains"
  # A DAEMON THAT ANSWERS FOR A RADIO IT CANNOT REACH is the failure the sweep
  # above exists to prevent, and it is silent from here: the daemon started below
  # would fail to bind, exit, and every `nc` after it would be answered by the
  # stale one. That reads as a healthy start on a rig nothing is holding.
  if lsof -t -nP -iTCP:4532 -sTCP:LISTEN >/dev/null 2>&1; then
    die "4532 is still listened on by something the stop above could not clear --
   see the line naming it. A daemon started now would bind nothing and be
   ANSWERED BY THE STALE ONE, on a CAT port it may no longer reach."
  fi
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    trap 'capture_trap_rc=$?; rigctld_cleanup; witness_stop; capture_exit "$capture_trap_rc"' EXIT
  else
    trap 'rigctld_cleanup; witness_stop' EXIT
  fi
  # Bash does not run EXIT for an unhandled termination signal on every platform.
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  rigctld -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" --set-conf=ptt_type=None \
      >"${TMPDIR:-/tmp}/rigctld.log" 2>&1 &
  local spawned=$!
  RIGCTLD_PID=$spawned
  # Wait for the daemon to ANSWER rather than sleeping a guess at how long it takes.
  # Three seconds is usually enough and sometimes is not: the adapter this rig is on
  # can take longer to open when the previous holder has only just let go, and the
  # launcher then aborted on an empty pipeline before printing a word about why —
  # `set -o pipefail` turning a slow start into a session that never began.
  local reply=""
  for _ in $(seq 12); do
    kill -0 "$spawned" 2>/dev/null || break
    reply="$(printf '+\\get_conf ptt_type\nq\n' | nc -w 2 127.0.0.1 4532 2>/dev/null || true)"
    if ! kill -0 "$spawned" 2>/dev/null; then reply=""; break; fi
    case "$reply" in *ptt_type=*) break ;; esac
    reply=""
    sleep 1
  done
  # THE DAEMON GOES WITH THE REFUSAL. This died with the process it had just
  # spawned still running, and on 2026-08-28 that happened inside `down` twice:
  # the cleanup verb exited before its own final stop and left a rigctld holding
  # the CAT port -- the verb whose whole job is clearing that trap creating it,
  # and returning 1 while it did, which the operator read as the launch having
  # short-circuited.
  if [ -z "$reply" ]; then
    rigctld_cleanup
    die "rigctld did not answer on 4532 within 12 s — see ${TMPDIR:-/tmp}/rigctld.log"
  fi
  # None is hamlib's name for a value of the ptt_type enum, not English for
  # missing. Printed bare, four lines under the `PTT … present` line, the two read
  # as a contradiction, and this line is the only one that shows the operator the
  # daemon is not a second owner of the key. So it says which of the two it is.
  case "$reply" in
    *ptt_type=None*) echo "  daemon keys nothing (ptt_type=None) — the line is ours alone" ;;
    *) printf '%s\n' "$reply" | sed -n 's/^ptt_type=/  !! THE DAEMON WOULD KEY THE RIG: ptt_type=/p' ;;
  esac
}

# Everything that has to be present before a modem is started. Named here rather
# than left to fail inside whichever tool reaches for it first: at the radio, a
# Python traceback out of a launcher reads like a code fault, and the operator
# goes looking for the wrong thing.
require_hardware() {
  took launch
  # THE CLAIM COMES FIRST, before a port is opened or a device is even looked
  # for: a second launcher has to be turned away while it is still incapable of
  # keying, not after it has taken the codec and the CAT line off the first.
  claim_rig
  require_ports
  [ -e "$CAT_PORT" ] || die "CAT port $CAT_PORT missing — is the rig on and cabled?"
  [ -e "$PTT_PORT" ] || die "PTT port $PTT_PORT missing — set PTT_PORT if the key \
is not on the second interface of the CAT adapter"
  echo "  CAT $CAT_PORT / PTT $PTT_PORT present"
  settle_port
  require_hamlib
  echo "  rigctl $(command -v rigctl)"
  # RESOLVED THE WAY THE MODEMS RESOLVE IT, and named. This asked only whether
  # the string appeared anywhere in the device table, so "codec present" was true
  # of a name that `find_device` would go on to refuse as ambiguous, and it said
  # nothing about which card a modem would end up on. What is printed now is the
  # index and full name each one will bind, from the one function that decides it.
  AUDIO_IN="$AUDIO_IN" AUDIO_OUT="$AUDIO_OUT" "${HFMODEM_PYTHON:-$VENV/python}" - <<'PY' || die "the modems cannot bind '$AUDIO' — the line above says why"
import sounddevice as sd
from hfmodem.core.devices import find_device
import os
for kind, name in (("in", os.environ["AUDIO_IN"]), ("out", os.environ["AUDIO_OUT"])):
    i = find_device(name, kind, required=True)
    print(f"  codec {kind:<3} {i}: {sd.query_devices(i)['name']}")
PY
}

# A usage error is not a failed connect. A modem's own non-zero exit means it ran
# and did not connect, which is an ordinary band result and must still reach
# prove_unkeyed. Exit 2 is argparse refusing the arguments, so nothing was ever
# transmitted -- and that printed a usage line followed by a clean PTT VERDICT:
# down, which reads exactly like an attempt that happened. `onair.sh vara GW C 500`
# was the case, back when --bw offered 2300 alone and the run looked complete.
refuse_usage_error() {
  if [ "${1:-0}" -eq 2 ]; then
    die "$2 refused its arguments (exit 2) -- NOTHING WAS TRANSMITTED. That is a
   launcher bug, not a band result; do not record it as an attempt."
  fi
}

# A THIRD-PARTY RECEIVER, RUNNING BESIDE THE ARM, UNABLE TO COST IT ANYTHING.
#
# Everything this station knows about its own emission comes from its own capture
# stream, which is windowed, blanked across the keyed interval and now known to
# lose air in lumps -- and a splice cannot be found inside one of our recordings
# after the fact, because the grid, the window indices and stream.wav all count
# DELIVERED samples and stay consistent with each other while the whole
# arithmetic slides against the air. A KiwiSDR a few hundred miles off hears both
# stations on one clock with no windowing by us, which is the only vantage from
# which "did our packet land on the raster the gateway set" and "what did the
# gateway send, and when" are the same measurement.
#
# IT IS NEVER ALLOWED TO DELAY OR FAIL AN ARM. A Kiwi with no free slot, a
# receiver that drops mid-session, a network that stalls: all ordinary, and under
# `set -euo pipefail` any of them would end the run. So the recorder is a
# background job nothing waits on and nothing tests the status of, its output
# goes to a log rather than into the arm's own, and both functions here end at
# `return 0`. A witness failure that costs a transmit arm is worse than no
# witness at all -- and an arm that went out unwitnessed says so, because an
# absent file read later as "we did not witness this one" is a different claim
# from "the receiver dropped at 03:14".
#
# STARTED AFTER THE GATE, so a channel the sense refuses costs a volunteer's
# receiver nothing. It opened ahead of the sense until 2026-09-16, and the
# refusals threw the placement away: 02:28:36 that morning walked eight
# candidates, connected, recorded 9.6 s and was told CHANNEL_BUSY -- and a
# `pounce` re-invokes this file once per attempt, so five refusals placed five.
#
# Below every other refusal a verb has left, too -- an unreadable argument, a CAT
# port somebody else is holding -- because each of those ends the arm as surely
# as the gate does and none of them costs anything to reach.
#
# NOTHING IS SPENT IN THE GAP EITHER WAY. The recorder is backgrounded and no
# line here waits on it, so what moved is when the receiver is READY, not when
# the verb gets on; the walk down the candidate list now overlaps the modem's
# own startup instead of the sense.
#
# WHAT IT COSTS IS THE HEAD OF THE ARM. Over the 30 witnessed arms on file the
# recording led `stream.wav` by about 21 s; the handshake takes 6 s to first
# audio (median, 27 s at the 90th and 165 s at the worst in the record), so the
# witness now opens a few seconds INTO our own recording rather than well ahead
# of it. witness_align searches either way round and asks for half of ours to
# lie inside the witness, so what this can lose is a short connect arm whose
# handshake spent the tail of that spread walking the list.
WITNESS_PID=
WITNESS_WAV=
# HOW LONG IT RECORDS. An arm's length is not known when it starts, so this is a
# ceiling and not a plan: overrunning costs a Kiwi slot somebody volunteered, and
# stopping early loses the tail, which is where the changeover and the
# acknowledgement live. What actually ends the recording is the arm ending. The
# ceiling is what bounds it when the launcher is killed outright and the
# recorder never hears the ask.
WITNESS_MAX_S="${WITNESS_MAX_S:-3600}"
witness_start() {       # <dial Hz> [gateway callsign]
  [ -n "$WITNESS" ] || return 0
  mkdir -p "$SESSION" 2>/dev/null || true
  WITNESS_WAV="$SESSION/witness.wav"
  witness_log_target="$SESSION/witness.log"
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    witness_log_target="/dev/fd/$HFMODEM_CAPTURE_WITNESS_LOG_FD"
  fi
  # THE GATEWAY IS THE POINT OF THE WITNESS ON ANY ARM THAT KEYS OVER IT. Our own
  # receiver is muted while we transmit, so a Kiwi placed near US records our
  # carrier and nothing of what the peer sent underneath it. Passing the callsign
  # puts the receiver near the station that has to be heard; without one the
  # recorder says so in its own log rather than leaving the coverage to be
  # assumed. The beacon verb has no gateway and passes none.
  "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/kiwi_witness.py" \
      --dial "$(printf '%d.%03d' "$(($1 / 1000))" "$(($1 % 1000))")" \
      --seconds "$WITNESS_MAX_S" --out "$WITNESS_WAV" \
      --candidates "${HFMODEM_WITNESS_CANDIDATES:-$OPERATOR_DATA/kiwi-candidates.csv}" ${capture_witness_args[@]+"${capture_witness_args[@]}"} \
      ${2:+--peer "$2"} \
      >"$witness_log_target" 2>&1 &
  WITNESS_PID=$!
  # Every path out of this shell, including the gate's own refusal and `die`.
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    trap 'capture_trap_rc=$?; rigctld_cleanup; witness_stop; capture_exit "$capture_trap_rc"' EXIT
  else
    trap 'rigctld_cleanup; witness_stop' EXIT
  fi
  printf '  witness: kiwi_witness pid %s -> %s (log %s/witness.log)\n' \
      "$WITNESS_PID" "$WITNESS_WAV" "$SESSION"
  return 0
}

# ASKED TO STOP, NOT KILLED. kiwi_witness accumulates the audio and writes the
# file once at the end, so SIGKILL is the whole recording gone; it takes SIGTERM
# as "write what you heard and go", and it is given the time to do it. Called
# after the PTT verdict rather than before it, so nothing here can stand between
# a finished transmission and the reading that says the line is down.
witness_stop() {
  [ -n "$WITNESS_PID" ] || return 0
  local pid="$WITNESS_PID"
  WITNESS_PID=
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    printf '  the witness ignored SIGTERM for 10s -- killing it, so it wrote nothing\n'
    kill -9 "$pid" 2>/dev/null || true
  fi
  # WHAT THE RECEIVER COULD AND COULD NOT DO, BESIDE THE ARM IT WITNESSED. The
  # recorder marks those lines; they carry which receiver it got, how far that
  # one is from the PEER, and the true level, which the .wav cannot show because
  # every witness is normalised to 0.9 on disk. Read at teardown they reach the
  # report; read out of the JSON afterwards they did not.
  sed -n '/^WITNESS/s/^/  /p' "$SESSION/witness.log" 2>/dev/null || true
  if [ -s "$WITNESS_WAV" ]; then
    # AND WHETHER THE PAIR IS ONE MEASUREMENT, which neither of the recorder's
    # own gates can ask. The geometry says where the receiver stands and the
    # probe says it hears something; a recording that cannot be registered
    # against our own keying still holds no answer slot to read a silence out
    # of, and on 2026-08-31 one that passed both gates was exactly that. Nothing
    # here reaches the transmitter -- it is two files and a correlation, run
    # after the PTT verdict, timed out rather than trusted, and its failure
    # costs the arm nothing.
    if [ -s "$SESSION/stream.wav" ] || [ -s "$SESSION/stream.flac" ]; then
      timeout 120 "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/witness_align.py" \
          "$SESSION/stream.wav" "$WITNESS_WAV" 2>/dev/null \
          | sed -n '/^WITNESS/s/^/  /p' || true
    fi
    printf '  witness: %s\n  the whole lag table is:\n    %s/python %s/tools/witness_align.py %s %s\n' \
        "$WITNESS_WAV" "$VENV" "$REPO" "$SESSION/stream.wav" "$WITNESS_WAV"
  else
    printf '  !! NO WITNESS for this arm -- %s/witness.log says why. The arm above
   stands on its own, and nothing in it was delayed or refused by this.\n' "$SESSION"
  fi
  return 0
}

# The end of every keying verb. A modem's own non-zero exit is an ordinary band
# result, and under `set -e` it takes the power reading and the PTT verdict with
# it; prove_unkeyed's own 1 is a quiet band, which is routine and not the verb's
# answer to report.
#
# The tool named is the one whose exit 2 is argparse. sabir names none: it spends
# 2 on its own refusals ahead of the key, and those must reach the verdict.
unkey_and_exit() {          # <the modem's status> <dial> [<tool argparsed>]
  local rc="$1" cat_ready=1
  [ -z "${3:-}" ] || refuse_usage_error "$rc" "$3"
  if [ "${3:-}" = kestrel_connect.py ]; then
    # VARA has exited and released PTT. Release its CAT daemon before the
    # power query can fall back to opening the serial port directly.
    if ! rigctld_stop; then
      echo "  VARA CAT cleanup failed -- skipping the contended power query"
      cat_ready=0
      [ "$rc" -ne 0 ] || rc=1
    fi
  fi
  [ "$cat_ready" -eq 0 ] || rig_power "$2"
  prove_unkeyed || true
  witness_stop
  exit "$rc"
}

# What a verb spends when the gate refused an occupied channel, and the only
# status `pounce` will key again on.
CHANNEL_BUSY=4
refuse_busy() {
  printf '\n!! %s\n' "$*" >&2
  exit "$CHANNEL_BUSY"
}

# ONE GATE, AND EVERY VERB THAT KEYS GOES THROUGH IT. `pactor` grew a sense on
# 2026-08-09 and `vara` learned to spell `--force` on 2026-08-14; `ardop` and
# `ardop-mail` went from require_hardware to key-down with no check of any kind,
# and when the operator asked whether occupancy sensing had been switched off on
# that path the answer was that it never had any. Roughly 120 keyings went out
# blind on 7101.5, 7103.2, 7103.5, 7103.8, 7106.5 and three 80 m channels, with
# nothing for `--force` to override because there was nothing to override.
#
# It takes the dial and the verb's trailing arguments. What is left after
# `--force` is taken out lands in KEY_ARGS for the tool; KEY_FORCE carries the
# flag on to tools with their own gate. VARA judges occupancy after its codec
# is ready; the other verbs keep their shell occupancy decision here.
#
# THIS IS THE LAST THING A VERB DOES BEFORE IT STARTS A MODEM, and everything a
# verb can do beforehand it does beforehand — the hardware check, the daemon, the
# power reading. Under POUNCE_WAIT this is where the run stands and waits, so
# anything left ahead of it would be paid out of the gap instead.
gate_channel() {
  local dial="$1" modem="$2" bw="$3" a rc=0 free_ident free_kind free_at
  shift 3
  # A BARE WORD AHEAD OF THE FLAGS IS A POSITIONAL THIS VERB HAS NO PLACE FOR.
  # Every positional a verb takes it reads before it gets here, so anything left
  # over is the operator meaning something the launcher cannot do -- and it went
  # to the modem unremarked, which put a call announced at BW200 on the air at
  # 2000. station.toml refuses an unknown key for exactly this reason: a setting
  # that silently did not apply is, on a transmitter, the difference between a
  # hot input and a working one. After a flag the rule stops, because a flag's
  # value is a bare word too and it is the modem's business.
  KEY_ARGS=(); KEY_FORCE=; local flagged=
  for a in "$@"; do
    case "$a" in
      --force) KEY_FORCE=--force; flagged=1 ;;
      force)   die "say --force -- a bare 'force' here used to be swallowed silently" ;;
      -*)      KEY_ARGS+=("$a"); flagged=1 ;;
      *)       [ -n "$flagged" ] || die "'$a' is one positional more than this verb \
takes, and nothing here knows what to do with it -- the usage at the top of \
${BASH_SOURCE[0]##*/} says what each verb reads, and its flags come after"
               KEY_ARGS+=("$a") ;;
    esac
  done
  before_the_gate
  # A wait that cannot change the answer is the slot spent for nothing: --force
  # waives the refusal below, and the operator gave the word because the channel
  # is clear NOW. The sense still runs, prints and records; deaf is still fatal.
  local wait_s="${POUNCE_WAIT:-0}"; [ -n "$KEY_FORCE" ] && wait_s=0
  if [ "$modem" = vara ]; then
    # Kestrel opens and warms the codec before judging fresh RX windows. A
    # shell clear decision followed by startup and another eight-second sense
    # cost the September 10 pounces their gap. Keep receiver readiness here,
    # and put the one occupancy decision next to the ready caller instead.
    tune "$dial"
    gate_rx_level
    # Append these after user flags: --listen-first 0 must not bypass the
    # launcher's guard now that its evidence is collected by the modem.
    KEY_ARGS+=(--pounce-wait "$wait_s" --listen-first 8)
    printf '  occupancy check delegated to ready VARA caller (wait budget %ss)\n' "$wait_s"
    return 0
  fi
  sense_channel "$dial" "$modem" "$bw" "$wait_s" || rc=$?
  # Deaf is answered ahead of --force and is fatal: the flag is the operator's
  # ears overruling the detector, and a station with no receiver gives them
  # nothing to weigh. The two shared an exit code once, and --force then keyed
  # over an occupant nobody here could have heard.
  case "$rc" in
    0) ;;
    2) die "no '$AUDIO' input — this station cannot hear the channel, so it may not call on it" ;;
    3) exit 3 ;;
    # A HOST SAYING IT IS FREE IS A GO-AHEAD, NOT A WAIVER. Nothing was
    # overruled here: the sense read an ident out of the channel and the station
    # that owns it has no link in progress, so the wait budget has nothing left
    # to buy and is not spent. `--force` never came into it.
    5) read -r free_ident free_kind free_at <<<"$(host_free)"
       printf '  host %s free on %s — %s, calling now\n' \
              "$free_ident" "$dial" "$free_kind"
       export HOST_FREE_IDENT="$free_ident" HOST_FREE_AT="$free_at" ;;
    *) [ -n "$KEY_FORCE" ] || refuse_busy "channel in use, and the ${wait_s}s wait budget is \
spent — listen first, or pass --force to override"
       # The operator listening to the channel is a better instrument than a
       # window, and is the only thing that should ever override it — never a
       # threshold edited to let a run through.
       echo "  --force: calling anyway over an occupant" ;;
  esac
  gate_rx_level
}

# WHAT THE RECEIVER IS DELIVERING, ASKED WHILE NOTHING IS KEYED -- and the policy
# an operator at the console gets. `rx_level` (lib/rig.sh) is the instrument, and
# carries what its three answers mean and what moved 16 dB the night they were
# measured; lib/attempts.sh holds the other policy over the same reading.
#
# ASKED AFTER TUNING (by the shell sense, or explicitly for VARA), so the level
# is the channel this arm gets rather than the last one's. A refusal is the
# operator's to waive, because what the level decides is whether the arm can be
# read afterwards and not whether it may go out; NO AUDIO is not, for the reason
# deafness is not -- there is nothing there for their own ears to overrule.
#
# IT REFUSES WITH 1 RATHER THAN CHANNEL_BUSY, so `pounce` does not go round
# again: another window is what a taken channel needs, and no number of them
# turns a gain up.
gate_rx_level() {
  local rc=0
  rx_level || rc=$?
  case "$rc" in
    0) ;;
    2) die "no audio from '$AUDIO' -- this station cannot hear the channel it is
   about to key on, and --force is not an answer to that" ;;
    *) [ -n "$KEY_FORCE" ] || die "the receive level cannot carry this arm -- set the
   gain the line above names and call again, or pass --force to key anyway"
       echo "  --force: calling anyway on a receiver at that level" ;;
  esac
}

# WHAT THE TRANSMITTER IS SET TO. No session record in this tree has ever carried
# it: three ARDOP attempts on 2026-08-14 went out at RFPOWER 0.05 — 5 W — and
# nothing in the software could say so, the operator read it off the meter. Every
# silence recorded before this has to be read with that qualifier, because no
# record can now say which of them were called into a full-power transmitter.
#
# It has to come over CAT. The witness tap cannot supply it: measured the same
# night, the monitor reads rms 0.3157 at 5 W and 0.3163 at 50 W, because it
# watches the modulation stage and not the output.
#
# Over the daemon when one is up, over the port when none is — rigctld owns the
# CAT port for as long as it runs. Hamlib converts to watts, because 0.05 means
# 5 W on this rig and something else on the next one, and a launcher does not get
# to assume a scale it can ask for.
#
# Asked before the gate rather than after it: RFPOWER does not change while a
# verb waits for the channel, and over the port it is two `rigctl` opens of a
# 38400-baud adapter — 0.5 s that used to fall between the channel clearing and
# key-down.
rig_power() {
  took daemon
  local dial="$1" level watts
  level="$(printf '\\get_level RFPOWER\nq\n' | nc -w 3 127.0.0.1 4532 2>/dev/null | sed -n 1p || true)"
  case "$level" in ''|*[!0-9.]*)
    level="$(rigctl -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" l RFPOWER 2>/dev/null | sed -n 1p || true)" ;;
  esac
  case "$level" in ''|*[!0-9.]*)
    echo "  TX power: CAT did not answer; power setting unavailable"
    return 0 ;;
  esac
  watts="$(printf '\\power2mW %s %s PKTUSB\nq\n' "$level" "$dial" \
             | nc -w 3 127.0.0.1 4532 2>/dev/null | sed -n 1p || true)"
  case "$watts" in ''|*[!0-9]*)
    watts="$(rigctl -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" 2 "$level" "$dial" PKTUSB \
               2>/dev/null | sed -n 1p || true)" ;;
  esac
  case "$watts" in
    ''|*[!0-9]*) printf '  TX power: RFPOWER %s (setting; RF output unmeasured)\n' "$level" ;;
    *)           printf '  TX power: RFPOWER %s -> %d.%01d W setting (not measured RF output)\n' "$level" \
                        "$((watts / 1000))" "$((watts % 1000 / 100))" ;;
  esac
}

# THE CAT PORT CHANGES HANDS ONCE, AND THE LAUNCHER PROVES IT RATHER THAN
# ASSUMING IT. shrike is the one modem that opens the port itself: a long-lived
# `rigctl` sets the dial and a one-shot reads it back, and a `rigctl` that cannot
# open the device writes its refusal to stderr and prints nothing on stdout. So a
# contended open reaches shrike as an EMPTY readback -- and since 2026-08-15 an
# empty readback refuses the run outright (`QSY UNVERIFIED`, shrike/onair.py),
# where it used to warn and key. That turned a wrong-frequency risk into a
# refusal, which costs the whole slot: the gate has already waited the channel
# out by then and the gap closes while shrike exits.
#
# The pactor arm opens CAT twice before this -- `rig_power`, then `tune` inside
# the gate -- and both are short-lived `rigctl`s that this shell waits on, so the
# usual answer is yes on the first ask. It is asked anyway, because "the last
# opener has probably exited" is not a handover, and this runs on a shared
# machine where another launcher's rigctld can appear between the gate and here.
# One `lsof` on this station's adapter: 0.15 s, measured 2026-08-15.
#
# ASKED, NOT WAITED OUT. lsof is the only instrument that answers *who* -- a
# failed open only answers "not you" -- and it is the same one `down`,
# `rigctld_stop` and `rigtest.sh preflight` ask. A holder still there after
# `rigctld_stop` is not a `rigctl` mid-exit; it is a process that will not leave
# on its own, and sleeping on it spends the gap before saying so.
cat_free_or_die() {
  local held
  held="$(lsof "$CAT_PORT" 2>/dev/null)" || return 0
  [ -n "$held" ] || return 0
  printf '%s\n' "$held" | sed 's/^/  /'
  die "$CAT_PORT is held, and shrike opens it directly -- it would read no
   frequency back and refuse the run (QSY UNVERIFIED). Stop the holder above,
   or run tools/onair.sh down, and call again."
}

# A receiver that hears the band is the only proof PTT is actually down. A dead
# input and a keyed transmitter look identical from every other angle.
prove_unkeyed() {
  "${HFMODEM_PYTHON:-$VENV/python}" - <<PY
import numpy as np, sounddevice as sd
from hfmodem.core.levels import NO_AUDIO_CAUSE, RX_SILENT_RMS
ins = [i for i, x in enumerate(sd.query_devices())
       if "$AUDIO" in x["name"] and x["max_input_channels"] > 0]
if not ins:
    # Not a check to skip when the codec is absent: with no receiver there is no
    # way to tell a quiet band from a transmitter that never let go of the line.
    print("  no '$AUDIO' input — CANNOT PROVE PTT IS DOWN")
    raise SystemExit(1)
a = sd.rec(int(0.7 * 48000), samplerate=48000, channels=1, device=ins[0], blocking=True)
r = float(np.sqrt(np.mean(a.astype(float) ** 2)))
# THREE OUTCOMES, NOT TWO. This named one cause -- a stuck transmitter -- for
# every reading under 0.01, and on 2026-08-14 printed it beside the exit trap's
# own "PTT VERDICT: down" from the same run: two statements, one output, flatly
# contradicting each other. A dead codec reads about 0.00007, and a live
# receiver 0.05-0.20; the decade between is a codec that is present and
# delivering quiet audio, which a keyed transmitter does not produce. Naming a
# cause the reading cannot support sends the operator to the wrong end of the
# station -- to the keying line, when the radio's own AF gain is where the
# answer was.
if r > 0.01:
    print(f"  receiver rms {r:.5f} -> LIVE, not transmitting")
elif r > RX_SILENT_RMS:
    print(f"  receiver rms {r:.5f} -> QUIET: the codec is delivering audio and "
          "the band is not. Check the radio's AF gain, its VFO and its mode "
          "before suspecting the keying line. This reading is of the room; the "
          "one that measures the line itself is the PTT verdict 'down' prints.")
else:
    print(f"  receiver rms {r:.5f} -> NO AUDIO: {NO_AUDIO_CAUSE}")
raise SystemExit(0 if r > 0.01 else 1)
PY
}

case "${1:-}" in

check)   # Everything that can be wrong before key-down, and nothing that keys.
  say "hardware"
  require_hardware
  # THE MARKERS IN THIS TABLE ARE THE HOST'S, NOT OURS. sounddevice writes `>`
  # and `<` against the system default input and output — the laptop's own
  # microphone and speakers on this machine — and nothing here binds them: every
  # modem is handed "$AUDIO" by name, which is the codec line above. An operator
  # read those arrows as this run's selection, saw the wrong card marked, and
  # distrusted a hardware block that was correct.
  say "the host's sound cards — '>' and '<' mark the system defaults, not ours"
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem devices | sed -n 's/^/  /p'
  say "receiver"
  prove_unkeyed
  say "ready — nothing above has transmitted"
  ;;

preflight)  # Keys UNMODULATED, so a sideband transmitter emits nothing.
  require_hardware
  rigctld_start none
  say "preflight (keys unmodulated; no RF)"
  # ITS OWN FILE, deliberately. Preflight measures the audio path and the keying
  # line; it is not a transmission and must not inherit a rule set that can
  # refuse one. Pointing it at the session's station.toml made it ask the
  # operator for a licence class to measure a sound card, which is the wrong
  # question at the wrong moment. The device name is what has to agree between
  # the two files, and that is checked below.
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem rig "${HFMODEM_STATION:-$REPO/station.toml}" --preflight || true
  prove_unkeyed
  ;;

pactor)  # shrike opens the CAT port ITSELF, so the daemon must let go of it, and
         # stay gone: this is the one verb that runs with no daemon at all.
  gw="${2:?gateway callsign}"; centre="${3:-7101500}"
  positional centre "$centre"
  require_hardware
  say "PACTOR-1 $MYCALL -> $gw, centre $centre (dial $((centre - 1500)))"
  # A port still held after this is `cat_free_or_die`'s to refuse, and it names
  # the holder; ending the arm here would be the same refusal without the name.
  rigctld_stop || true
  rig_power "$((centre - 1500))"
  # ONE FLAG SETS THE SENSE AND THE SESSION. The P1/P2 ceiling is read here to
  # pick the band judged, and passed through untouched to shrike, which is what
  # holds the link there -- a call sensed at 1200-1800 Hz and then allowed to
  # upgrade would key 380-2620 over a channel nobody measured. P2's two tones
  # also fit inside this band; --p2-mail enforces the ceiling in the modem.
  pbw=
  for a in "${@:4}"; do case "$a" in --pactor1-only|--pactor2-only|--p2-mail) pbw=1 ;; esac; done
  # The sense reads the destination out of any Robust Call in the window, and a
  # call addressed to the station we are about to call is worth saying so.
  export DXCALL="$gw"
  unset HOST_FREE_IDENT HOST_FREE_AT
  # `--force` is consumed here, not passed on: shrike has no such flag, and the
  # decision it overrides is this one.
  gate_channel "$((centre - 1500))" pactor "$pbw" "${@:4}"
  out="$SESSION"
  # The handover itself, and the last thing before the port stops being ours.
  cat_free_or_die
  witness_start "$((centre - 1500))" "$gw"
  # Let a mail arm select its own idle hold (96 cycles); passing the probe's
  # 12 here suppresses that default. Explicit --hold below still wins.
  pactor_hold=12
  for a in "${KEY_ARGS[@]+"${KEY_ARGS[@]}"}"; do
    case "$a" in --mail-fetch|--mail-send|--mail-send=*) pactor_hold=0 ;; esac
  done
  # --serial/--ptt-port are the arm gate's demand: without them shrike refuses
  # at "NOT KEYING" before the rig object exists.
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.shrike.onair --transmit \
      --serial "$CAT_PORT" --ptt-port "$PTT_PORT" \
      --audio-out "$AUDIO_OUT" --audio-in "$AUDIO_IN" \
      --mycall "$MYCALL" --dxcall "$gw" --dial "$((centre - 1500))" \
      --retries 20 --max-cycles 48 --hold "$pactor_hold" --outdir "$out" ${capture_modem_args[@]+"${capture_modem_args[@]}"} ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} || rc=$?
  echo "  captures: $out"
  unkey_and_exit "$rc" "$((centre - 1500))" "shrike onair"
  ;;

pactor-answer)  # Keys only when a station calls $MYCALL, so no channel gate.
  centre="${2:?centre frequency}"; secs="${3:-900}"
  positional centre "$centre"
  positional seconds "$secs"
  require_hardware
  say "PACTOR-1 answer as $MYCALL, centre $centre (dial $((centre - 1500))), ${secs}s"
  rigctld_stop || true
  rig_power "$((centre - 1500))"
  cat_free_or_die
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.shrike.onair --transmit \
      --serial "$CAT_PORT" --ptt-port "$PTT_PORT" \
      --audio-out "$AUDIO_OUT" --audio-in "$AUDIO_IN" \
      --mycall "$MYCALL" --dial "$((centre - 1500))" \
      --answer "$secs" --outdir "$SESSION" ${capture_modem_args[@]+"${capture_modem_args[@]}"} "${@:4}" || rc=$?
  echo "  captures: $SESSION"
  unkey_and_exit "$rc" "$((centre - 1500))" "shrike onair"
  ;;

vara)    # kestrel drives the keying line itself, the shape vara-mail proved.
         # This verb was the last to key THROUGH the daemon, and on 2026-08-13
         # that bought ~4.7 s of unintended carrier on every attempt: rigctld
         # answered everything until the first key-down, then answered nothing
         # (see rigctld_start). rigctld is left to tune, before there is RF.
  gw="${2:?gateway callsign}"; centre="${3:-7103800}"; bw="${4:-2300}"
  positional centre "$centre"; positional bandwidth "$bw"
  require_hardware
  say "VARA $MYCALL -> $gw, centre $centre (dial $((centre - 1500))), BW$bw"
  # THE DAEMON IS UP BEFORE THE GATE, not after it. sense_channel tunes through a
  # live daemon (see `tune`), so it no longer needs the CAT port to itself and the
  # daemon no longer has to be stopped and restarted around it — ~2 s that fell
  # between the channel clearing and key-down on every pounce down this path.
  rigctld_start none
  rig_power "$((centre - 1500))"
  # This slot took a bare positional `force` until 2026-08-14, so the flag every
  # other verb takes was swallowed here without a word -- two refused attempts
  # and one wrong claim before anyone saw why. The bare word is a loud stop in
  # gate_channel now. `--force` and the wait budget go on to kestrel, which
  # senses after its codec is ready, and everything else reaches it unchanged:
  # --timeout, say, which the connect-request cadence is now derived from, so
  # widening the window widens the ladder instead of leaving it spent in the
  # first third  [see kestrel_connect.cr_cadence].
  gate_channel "$((centre - 1500))" vara "$bw" "${@:5}"
  witness_start "$((centre - 1500))" "$gw"
  # --expect-model is not optional with --arm: it refuses to key unless the rig
  # identifies as the one we think we are pointed at. Neither is --ptt-device:
  # on 2026-08-10 this launch omitted it, rigctld took the unkey and never
  # answered, and the transmitter stayed keyed with no modulation until the
  # operator unkeyed it by hand. kestrel refuses --arm without it.
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/kestrel_connect.py" \
      --gateway "$gw" --mycall "$MYCALL" --bw "$bw" \
      --tx-device "$AUDIO" --rx-device "$AUDIO" \
      --rigctld 127.0.0.1:4532 --freq "$(printf '%d.%03d' "$(((centre - 1500) / 1000))" "$(((centre - 1500) % 1000))")" \
      --arm --expect-model "$RIG_EXPECT_MODEL" --ptt-device "$PTT_PORT" --line-ptt \
      $KEY_FORCE ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} \
      --gateways "${HFMODEM_DATA_DIR:-$REPO/operator-data}/cache/winlink-vara-gateways.csv" || rc=$?
  unkey_and_exit "$rc" "$((centre - 1500))" kestrel_connect.py
  ;;

ardop)   # besra keys the line itself (see vara-mail), and records by default now.
  gw="${2:?gateway callsign}"; centre="${3:-7103500}"; bw="${4:-500}"
  positional centre "$centre"; positional bandwidth "$bw"
  require_hardware
  say "ARDOP $MYCALL -> $gw, centre $centre (dial $((centre - 1500))), BW$bw"
  rigctld_start none
  rig_power "$((centre - 1500))"
  # `--force` is consumed here, as it is on the pactor path: besra has no such
  # flag, and the decision it overrides is gate_channel's.
  gate_channel "$((centre - 1500))" ardop "$bw" "${@:5}"
  # Nothing else after the bandwidth goes anywhere, so refuse it rather than
  # swallow it -- a flag dropped in silence here is the same trap `vara` set with
  # its bare positional `force`, and it costs attempts before anyone sees why.
  [ ${#KEY_ARGS[@]} -eq 0 ] || die "ardop takes <gw> [centre] [bw] [--force] and \
nothing after -- '${KEY_ARGS[0]}' would go nowhere"
  witness_start "$((centre - 1500))" "$gw"
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.besra.host.run_server \
      --radio "$RIG_NAME" --rigctld 127.0.0.1:4532 \
      --audio-in "$AUDIO" --audio-out "$AUDIO" \
      --channel "$centre" --bandwidth "$bw" \
      --ptt-device "$PTT_PORT" --line-ptt &
  modem=$!
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    trap 'capture_trap_rc=$?; kill $modem 2>/dev/null || true; rigctld_cleanup; witness_stop; capture_exit "$capture_trap_rc"' EXIT
  else
    trap 'kill $modem 2>/dev/null || true; rigctld_cleanup; witness_stop' EXIT
  fi
  # besra is ready when it answers, and a fixed sleep here was ten seconds of a
  # pounce's gap. `run_server` binds the host port only after the radio link is
  # up -- audio and rig both open -- so the port is a stricter precondition than
  # the sleep as well as a sooner one.
  serving=
  for _ in $(seq 40); do
    if nc -z 127.0.0.1 8515 2>/dev/null; then serving=1; break; fi
    sleep 0.25
  done
  [ -n "$serving" ] || die "besra did not serve on 8515 within 10 s"
  # THE BANDWIDTH GOES TO BOTH ENDS. `ARQBW` from the host governs the next
  # ConReq, so besra started at 200 and called through a client holding its own
  # default went out at 2000: the line above announced BW200 and the session came
  # up `ARQBW now 2000MAX`, four times the occupancy nobody asked for.
  rc=0; "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/ardop_call.py" "$gw" 90 --bandwidth "$bw" || rc=$?
  kill $modem 2>/dev/null || true
  if [ "$CAPTURE_PREPARED" = 1 ]; then
    trap 'capture_trap_rc=$?; rigctld_cleanup; witness_stop; capture_exit "$capture_trap_rc"' EXIT
  else
    trap 'rigctld_cleanup; witness_stop' EXIT
  fi
  sleep 2
  unkey_and_exit "$rc" "$((centre - 1500))" ardop_call.py
  ;;

ardop-mail)  # A whole Winlink exchange over one held ARDOP link: run_server owns
             # the radio AND the B2F session, no host client. PTT does NOT go
             # through the daemon here, for the reason spelled out under vara-mail:
             # besra drives $PTT_PORT itself and rigctld is left to tune, before
             # there is any RF about. `hfmodem mail --protocol ardop` prints this
             # line with the mail flags filled in.
  gw="${2:?gateway callsign}"; centre="${3:?channel centre Hz}"; bw="${4:-500}"
  positional centre "$centre"; positional bandwidth "$bw"
  require_hardware
  say "ARDOP mail $MYCALL <-> $gw, centre $centre (dial $((centre - 1500))), BW$bw"
  rigctld_start none
  rig_power "$((centre - 1500))"
  # `--force` is consumed here; besra has no such flag. Everything else reaches
  # it unchanged -- --mail-fetch, --mail-timeout and the rest.
  gate_channel "$((centre - 1500))" ardop "$bw" "${@:5}"
  witness_start "$((centre - 1500))" "$gw"
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.besra.host.run_server \
      --radio "$RIG_NAME" --rigctld 127.0.0.1:4532 \
      --audio-in "$AUDIO" --audio-out "$AUDIO" \
      --channel "$centre" --bandwidth "$bw" \
      --ptt-device "$PTT_PORT" --line-ptt \
      --mycall "$MYCALL" --call "$gw" --transmit ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} || rc=$?
  unkey_and_exit "$rc" "$((centre - 1500))" "besra run_server"
  ;;

vara-mail)   # The same exchange over a held VARA link. PTT does NOT go through the
             # daemon here: the key on this station is RTS on $PTT_PORT, and kestrel
             # drives that line itself, so a transmit path that has to work while the
             # antenna is radiating does not depend on a socket that RF can wedge.
             # rigctld is left to tune, which it does before there is any RF about —
             # hence ptt_type=None, so nothing else opens the line we are holding.
             # `hfmodem mail --protocol vara` prints this line.
  gw="${2:?gateway callsign}"; centre="${3:?channel centre Hz}"
  positional centre "$centre"
  # THE BANDWIDTH IS OPTIONAL AND STANDS WHERE IT DOES ON `vara`, so every line
  # already pasted -- `vara-mail GW CENTRE --mail-fetch` -- still reads its flags
  # as flags. A bare word here is only ever a bandwidth, and one kestrel does not
  # call is refused rather than carried into the gate as a band nobody named.
  bw=2300; rest=4
  case "${4:-}" in
    ''|-*)         ;;
    500|2300|2750) bw="$4"; rest=5 ;;
    *) die "BW '$4' is not one this station calls -- 500, 2300 or 2750" ;;
  esac
  require_hardware
  say "VARA mail $MYCALL <-> $gw, centre $centre (dial $((centre - 1500))), BW$bw"
  rigctld_start none
  rig_power "$((centre - 1500))"
  gate_channel "$((centre - 1500))" vara "$bw" "${@:$rest}"
  witness_start "$((centre - 1500))" "$gw"
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/kestrel_connect.py" \
      --gateway "$gw" --mycall "$MYCALL" --bw "$bw" \
      --tx-device "$AUDIO" --rx-device "$AUDIO" \
      --rigctld 127.0.0.1:4532 --freq "$(printf '%d.%03d' "$(((centre - 1500) / 1000))" "$(((centre - 1500) % 1000))")" \
      --arm --expect-model "$RIG_EXPECT_MODEL" --ptt-device "$PTT_PORT" --line-ptt \
      $KEY_FORCE ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} \
      --gateways "${HFMODEM_DATA_DIR:-$REPO/operator-data}/cache/winlink-vara-gateways.csv" || rc=$?
  unkey_and_exit "$rc" "$((centre - 1500))" kestrel_connect.py
  ;;

sabir)   # One-way beacon or prepared DATAGRAM proof; no peer reply is expected.
         # Independent reception is verified from a remote recording. This
         # launcher does not establish an ARQ session (see sabir/ONAIR.md).
  centre="${2:?authorized clear channel centre Hz required}"
  require_hardware
  say "sabir one-way transmission on centre $centre (dial $((centre - 1500))) — no reply expected"
  rigctld_start none
  rig_power "$((centre - 1500))"
  # `--force` goes on rather than being consumed here: sabir senses the channel
  # itself in the last moment before key-down, the way kestrel does.
  #
  # SO DOES THE EMISSION PROFILE, AND THIS LAUNCHER DOES NOT CHOOSE IT.
  # sabir checks its emission against the limits the operator selects, or
  # `--regulatory unregulated --because "…"` to apply no emission limits.
  # Everything after the centre reaches sabir unchanged, so the operator says it:
  #
  #   ./tools/onair.sh sabir 14100000 --regulatory part97 --licence general
  #
  # On a band that carries its own power limit -- 30 m, 60 m -- `--power <watts
  # PEP>` goes the same way; sabir's gate demands it there rather than let an
  # unknown amount into a capped band. If sabir refuses for want of a
  # declaration, that refusal is its own and says what it wants.
  gate_channel "$((centre - 1500))" sabir "" "${@:3}"
  witness_start "$((centre - 1500))"
  out="$SESSION"
  mkdir -p "$out"
  # --grid is the station's own and only the wspr beacon carries it; without one
  # the beacon_grid fallback sends the presence beacon instead of refusing.
  # --wav keeps exactly what went out beside the run: the far end's decode
  # arrives days later, and a one-way test is that recording against this file.
  beacon_grid ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
  rc=0
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.sabir.onair --transmit \
      --radio "$RIG_NAME" --rigctld 127.0.0.1:4532 \
      --ptt-device "$PTT_PORT" --line-ptt \
      --channel "$centre" \
      --audio-in "$AUDIO" --audio-out "$AUDIO" \
      --mycall "$MYCALL" ${GRID_ARGS[@]+"${GRID_ARGS[@]}"} \
      --wav "$out/beacon.wav" \
      $KEY_FORCE ${KEY_ARGS[@]+"${KEY_ARGS[@]}"} ${GRID_FALLBACK[@]+"${GRID_FALLBACK[@]}"} || rc=$?
  # No tool named: sabir spends exit 2 on every refusal ahead of the key,
  # including "something else is already transmitting", and that is the one
  # answer that must reach prove_unkeyed rather than die in front of it.
  echo "  captures: $out"
  # A beacon that never went out is not a quiet band, and this status is the only
  # place that difference survives into whatever ran the launcher.
  unkey_and_exit "$rc" "$((centre - 1500))"
  ;;

rtty)    # Keyboard-to-keyboard RTTY, and the operator keys every over by hand. No
         # gate_channel: they listen before they key, and a sense per over would
         # delay every turnover. The rig goes to its data USB mode on either
         # path, where the terminal's mark (audio 2295 Hz) is the high tone on
         # RF, at dial + 2295. `-` for the dial keeps the dial and reads it back.
  dial="${2:?a dial in Hz, or - to leave the rig where it is}"
  require_hardware
  say "RTTY terminal as $MYCALL -- nothing transmits until you key"
  rigctld_start none
  printf '+\\set_mode %s 3000\nq\n' "$RIG_MODE" | nc -w 5 127.0.0.1 4532 >/dev/null || true
  mode="$(printf '+\\get_mode\nq\n' | nc -w 5 127.0.0.1 4532 2>/dev/null \
          | sed -n 's/^Mode: //p' | head -1)"
  # The terminal owns the screen and this stderr goes only to arm.log, so the
  # warning also rides into the session, which holds it on its status line.
  warn=()
  if [ "$mode" != "$RIG_MODE" ]; then
    printf '\n!! the rig reads mode %s, not %s: mark RF and the log assume %s\n' \
      "${mode:-unread}" "$RIG_MODE" "$RIG_MODE" >&2
    warn=(--warn "rig mode ${mode:-unread}, not $RIG_MODE")
  fi
  if [ "$dial" = - ]; then
    dial="$(printf '+\\get_freq\nq\n' | nc -w 5 127.0.0.1 4532 2>/dev/null \
            | sed -n 's/^Frequency: //p' | head -1)"
    case "$dial" in
      ''|*[!0-9]*) die "DIAL UNREAD -- the rig reported no frequency. Name the dial." ;;
    esac
    printf '  dial %s, as the rig sits\n' "$dial"
  else
    positional dial "$dial"
    tune "$dial"
  fi
  rig_power "$dial"
  # In the background and waited for, so a signal to this shell runs its trap at
  # once rather than after the terminal exits: TERM or HUP halts the terminal (its
  # TERM unkeys), and the verdict below still runs once it has gone. INT forwards
  # the operator's stop to the terminal too, including a signal to this shell only.
  "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.lanner.qso --dial "$dial" --rig "$RIG_NAME" \
      --ptt-device "$PTT_PORT" --audio-in "$AUDIO_IN" --audio-out "$AUDIO_OUT" \
      --mycall "$MYCALL" --launcher "$$" ${warn[@]+"${warn[@]}"} "${@:3}" &
  modem=$!
  trap 'kill -TERM "$modem" 2>/dev/null || true' TERM HUP
  trap 'kill -INT "$modem" 2>/dev/null || true' INT
  while kill -0 "$modem" 2>/dev/null; do wait "$modem" || true; done
  rc=0; wait "$modem" || rc=$?
  trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
  unkey_and_exit "$rc" "$dial" "lanner qso"
  ;;

pounce)  # ./tools/onair.sh pounce <verb> <gw> <centre> [more…]
         # Run the verb with a bounded wait for occupancy to clear. VARA waits
         # inside its initialized caller; other verbs wait in the shell gate.
         # --force records one measurement and waives occupancy, not a deaf
         # receiver. An expired wait spends 3 and is terminal. Only an immediate
         # occupancy refusal spends CHANNEL_BUSY and can retry here; protocol,
         # rig, and receiver failures do not invite another attempt.
  verb="${2:?vara-mail|ardop-mail|pactor}"; gw="${3:?gateway callsign}"
  centre="${4:?channel centre Hz}"
  # Claimed out here rather than left to the verb: two pounces on one channel
  # would otherwise both wait it out and both go when it cleared, which is the
  # collision this file exists to prevent arriving on a timer. The verb below
  # inherits this one.
  claim_rig
  say "waiting for $((centre / 1000)).$((centre % 1000 / 100)) kHz to clear, then $verb -> $gw"
  for _ in 1 2 3 4 5; do
    rc=0
    parent_capture_args=()
    if [ "${CAPTURE_KIND:-}" = launch-group ]; then
      parent_capture_args=(--capture-parent-id "$CAPTURE_ID" --capture-parent-kind launch-group)
    fi
    POUNCE_WAIT="${POUNCE_WAIT:-900}" "$0" ${WITNESS:+--witness} "${@:2}" \
      ${parent_capture_args[@]+"${parent_capture_args[@]}"} || rc=$?
    [ "$rc" -eq 3 ] && die "channel never cleared"
    [ "$rc" -eq "$CHANNEL_BUSY" ] || exit "$rc"
    echo "  the channel was taken again before we keyed — back to waiting"
  done
  die "gave up after five attempts"
  ;;

listen)  # Receive only. Nothing here can key: the daemon is told ptt_type=None
         # and the monitor has no transmit path at all.
  centre="${2:?centre frequency in Hz, e.g. 7101500}"; secs="${3:-300}"
  require_hardware
  rigctld_start none
  say "listening on centre $centre (dial $((centre - 1500))) for ${secs}s — NO TX"
  printf '+\\set_mode PKTUSB 3000\nq\n' | nc -w 5 127.0.0.1 4532 >/dev/null
  # Through the shared tune, which reads the dial back and refuses a mismatch.
  # This verb had its own copy of that, printing the frequency without checking
  # it -- and it was the only verb doing the readback at all, while the four that
  # can key were not. One path now, and the receive side gets the tolerance too:
  # a frequency that did not take is silent, and looks exactly like a dead band.
  tune "$((centre - 1500))"
  timeout "$secs" "${HFMODEM_PYTHON:-$VENV/python}" -m creance.cli monitor --device "$AUDIO" || true
  prove_unkeyed
  ;;

power)   # No TX, and no key: one CAT question, asked of a daemon if one is up.
         # The centre only picks the frequency the watts are converted at, so it
         # defaults to where the station usually calls rather than demanding one.
  centre="${2:-7101500}"
  require_cat
  require_hamlib
  settle_port
  rig_power "$((centre - 1500))"
  ;;

down)
  # The one verb that frees the ports without first proving them present: it runs
  # when things are already wrong. It still has to know which ports those are.
  #
  # ALREADY STOPPED IS SUCCESS, AND IT SAYS SO OUT LOUD. This came back 1 when
  # there was nothing left to stop, and on 2026-08-28 the operator read that as
  # the launch having short-circuited and started arm 2 a second time on top of
  # itself. Nothing below reports failure for a thing that was not running; the
  # status is non-zero only for a port this could not free, which is the one
  # condition that will actually spoil the next arm.
  DOWN_PIDS=" $(ps -Ao pid= | tr -s '[:space:]' ' ') "
  require_ports
  say "stopping everything"
  stuck=0
  # Ask nicely first: a modem that shuts down on its own unkeys deliberately,
  # where a kill leaves the last PTT state to the watchdog.
  issued_before $(pgrep -f "run_server|kestrel_connect|shrike.onair|lanner.qso|kiwi_witness|creance.cli" 2>/dev/null || true)
  modems="$KEPT"
  if [ -z "$modems" ]; then
    echo "  no modem was running"
  else
    printf '  asking %s modem(s) to stop: %s\n' "$(echo $modems | wc -w | tr -d ' ')" "$(echo $modems)"
    kill -TERM $modems 2>/dev/null || true
    stopped_a "modem(s)"
    # Only when there was something to wait for. A cleanup that costs two seconds
    # of a slot for nothing is two seconds of the gap the next arm launches into.
    sleep 2
  fi
  fenced_exit
  reap_orphans
  fenced_exit
  rigctld_stop      || stuck=$((stuck + 1))
  fenced_exit
  free_port "$CAT_PORT" || stuck=$((stuck + 1))
  fenced_exit
  free_port "$PTT_PORT" || stuck=$((stuck + 1))
  fenced_exit
  # THE LINE ITSELF, once nothing holds the port. Everything above this is a
  # signal, and a signal is not evidence: `free_port` and `reap_orphans` both end
  # in SIGKILL, which is the one way out a modem cannot unkey through, and this
  # station has never measured whether the line falls on its own when a process
  # dies. `drop_rts` takes both output lines down on the hardware and reads them
  # back, so it answers whether the key is down as well as putting it there.
  # Before the receiver check rather than after: that one exits 1 on a quiet
  # band, and under `set -e` would take the line verdict with it.
  prove_down
  # ONE SHOT, NOT A DAEMON. This asked the question by starting a rigctld and
  # stopping it again, and it left one orphaned twice on 2026-08-28: the start's
  # own `die` -- the daemon spawned, then not answering inside 12 s -- ended
  # `down` before its own stop, so the cleanup verb walked away holding the CAT
  # port it exists to free, and rigctld was killed by hand before three arms.
  # `rigctl` opens the port, answers, and is gone; there is nothing left to
  # orphan and nothing left holding the port when this verb returns.
  dial="$(rigctl -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" f 2>/dev/null | sed -n 1p || true)"
  case "$dial" in
    ''|*[!0-9]*) printf '  !! CAT DID NOT ANSWER -- check the radio, not the software\n' ;;
    *)           printf '  CAT answers, dial %s\n' "$dial" ;;
  esac
  # Both readings below are the station's condition, not this verb's result. A
  # radio that is off and a band that is quiet are things `down` reports; they
  # are not things it failed to stop, and reporting them as failure is what cost
  # arm 2.
  prove_unkeyed || true
  if [ "$stuck" -gt 0 ]; then
    die "$stuck port(s) above are STILL HELD. The next arm would open the wrong
   thing or nothing at all -- clear the holder named above before calling again."
  fi
  if [ -z "$STOPPED" ]; then
    say "nothing was running — the station was already down, and that is success"
  else
    say "stopped:${STOPPED%,} — the station is down"
  fi
  exit 0
  ;;

# The header, however long it grows. This was `sed -n '2,36p'`, and adding a
# paragraph to the usage above silently cut the last one off the bottom of it.
*) awk 'NR > 1 && !/^#/ {exit} NR > 1' "${BASH_SOURCE[0]}" ;;
esac
