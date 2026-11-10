# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
# The radio the launchers drive, and the three things every one of them does
# through it: set the dial and read it back, ask whether anyone else is already
# on the channel, and ask what the receiver is delivering on it. Sourced after
# lib/ports.sh, whose `die` and `$VENV` these use.

# The codec's WHOLE name. `core.devices.find_device` refuses an ambiguous one,
# and `KT USB Audio` -- the rig-monitor witness tap -- contains the substring
# this station named the receiver by until 2026-08-14.


# THE DIAL, THROUGH WHICHEVER DOOR IS OPEN, AND READ BACK. rigctld owns the CAT
# port for as long as it runs, and a second opener of a serial device gets a
# garbled conversation rather than a refusal; `RPRT 0` says the command was
tune() {
  local dial="$1" got tol reply rc=0 daemon=0 package commands width roof shift
  if [ "$RIG_NAME" = ftdx10 ]; then
    package="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/packages/hfmodem"
    commands=$(PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 commands)
    read -r roof width shift <<< "$commands"
    # The dial goes first: on another band the band stack brings back that band's own
    # mode and filters. Then the mode, then the filters, which belong to it.
    if nc -z -w 3 127.0.0.1 4532 2>/dev/null; then
      daemon=1
      reply=$(PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 batch "$dial" \
              | nc -w 5 127.0.0.1 4532 2>&1 | tr '\000' '\n'; exit "${PIPESTATUS[1]}") || rc=$?
    else
      reply=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" -o \
              V VFOA F VFOA "$dial" f VFOA w 'MD0C;' \
              L VFOA AGC 0 l VFOA AGC L VFOA PREAMP 0 l VFOA PREAMP \
              w "$roof" w "$width" w "$shift" v 2>&1 \
              | tr '\000' '\n'; exit "${PIPESTATUS[0]}") || rc=$?
    fi
    got=$(printf '%s\n' "$reply" | PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" \
          "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 dial "$rc" "$daemon")
  else
    if ! printf '+\\set_freq %d\nq\n' "$dial" \
         | nc -w 3 127.0.0.1 4532 2>/dev/null | grep -q '^RPRT 0'; then
      claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" F "$dial" >/dev/null 2>&1 || true
    fi
    got=$(printf '+\\get_freq\nq\n' | nc -w 5 127.0.0.1 4532 2>/dev/null \
          | sed -n 's/^Frequency: //p' | head -1)
    [ -n "$got" ] || got=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" f 2>/dev/null | head -1)
  fi
  case "$got" in
    ''|*[!0-9]*) die "DIAL UNVERIFIED -- the rig reported no frequency, so the
   channel this would key on is unknown. Ask the operator what the display reads." ;;
  esac
  # The one definition, read from the package rather than typed here again.
  tol=$("${HFMODEM_PYTHON:-$VENV/python}" -c 'from hfmodem.core.band import QSY_TOLERANCE_HZ as t; print(t)')
  if [ "$(( got > dial ? got - dial : dial - got ))" -gt "$tol" ]; then
    die "DIAL MISMATCH -- asked $dial, rig reads $got (over $tol Hz). The rig is
   not where this would transmit. Ask the operator what the display reads."
  fi
  printf '  dial %s verified\n' "$got"
  if [ "$RIG_NAME" = ftdx10 ]; then
    printf '%s\n' "$reply" | PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" \
      "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 session-report "$rc" "$daemon"
    sleep 1
    rig_readback
  elif [ "$RIG_NAME" = ft891 ]; then
    ipo_off
  fi
  return 0
}

# MODE, WIDTH, ROOFING AND IF SHIFT AS THE FTDX10 ANSWERS THEM, asked in a connection
# of their own once `tune` has let them settle. A reply inside the batch that set them
# answers before the rig has finished changing.
rig_readback() {
  local reply rc=0 daemon=0 package
  package="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/packages/hfmodem"
  if nc -z -w 3 127.0.0.1 4532 2>/dev/null; then
    daemon=1
    reply=$(PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 readback-batch \
            | nc -w 5 127.0.0.1 4532 2>&1 | tr '\000' '\n'; exit "${PIPESTATUS[1]}") || rc=$?
  else
    reply=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" \
            w 'MD0;' w 'SH0;' w 'RF0;' w 'IS0;' 2>&1 | tr '\000' '\n'; exit "${PIPESTATUS[0]}") || rc=$?
  fi
  printf '%s\n' "$reply" | PYTHONPATH="$package${PYTHONPATH:+:$PYTHONPATH}" \
    "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.core.ftdx10 readback "$rc" "$daemon"
}

# The passband a hamlib set_mode asks for: no change where the profile owns the width.
mode_passband() {
  PYTHONPATH="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/packages/hfmodem${PYTHONPATH:+:$PYTHONPATH}" \
    "${HFMODEM_PYTHON:-$VENV/python}" -c 'import sys
from hfmodem.core.rigs import RIGS, mode_passband
print(mode_passband(RIGS[sys.argv[1]], int(sys.argv[2])))' "${RIG_NAME:-ft891}" "$1"
}

ipo_off() {
  local reply reading error rc=0 daemon=0
  if nc -z -w 3 127.0.0.1 4532 2>/dev/null; then
    daemon=1
    reply=$(printf '+\\set_level PREAMP 0\n+\\get_level PREAMP\nq\n' \
            | nc -w 3 127.0.0.1 4532 2>&1) || rc=$?
    # Up to the get_level's own status: the closing q answers RPRT 0 as well.
    reply=$(printf '%s\n' "$reply" | awk '{ print } /^RPRT / && ++n == 2 { exit }')
  else
    reply=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" \
            L PREAMP 0 l PREAMP 2>&1) || rc=$?
  fi
  error=$(printf '%s\n' "$reply" | grep -E 'error =|^RPRT -[0-9]+' | head -1) || true
  reading=$(printf '%s\n' "$reply" | grep -E '^-?[0-9]+$' | head -1) || true
  if [ -z "$error" ] && [ "$daemon" -eq 1 ] \
     && [ "$(printf '%s\n' "$reply" | grep -c '^RPRT 0$')" -ne 2 ]; then
    error='incomplete rigctld reply'
  fi
  if [ -n "$error" ] || [ "$rc" -ne 0 ] || [ -z "$reading" ]; then
    if [ -z "$error" ]; then
      error=$(printf '%s\n' "$reply" | sed '/^$/d' | head -1)
      error="${error:-no PREAMP reading (exit $rc)}"
    fi
    printf '  !! IPO unverified (rigctl: %s)\n' "$error"
  elif [ "$reading" = 0 ]; then
    printf '  IPO verified\n'
  else
    printf '  !! PREAMP reads %s (not IPO)\n' "$reading"
  fi
  return 0
}

# The FTDX10 recalls AGC per band and mode, so a retune can quietly turn it back
# on. The station runs AGC off by design; say so when it would not stay off.
agc_off() {
  [ "$RIG_NAME" = ftdx10 ] || return 0
  local reading reply part level rc=0 error daemon=0
  if [ "${1:-}" = --rigctld ] \
     || { [ "$#" -eq 0 ] && nc -z -w 3 127.0.0.1 4532 2>/dev/null; }; then
    daemon=1
    reply=$(printf '+\\set_level AGC 0\n+\\get_level AGC\n+\\set_level PREAMP 0\n+\\get_level PREAMP\nq\n' \
            | nc -w 3 127.0.0.1 4532 2>&1) || rc=$?
  elif [ "$#" -gt 0 ]; then
    reply="$1" rc="${2:-0}"
  else
    reply=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" \
            L AGC 0 l AGC L PREAMP 0 l PREAMP 2>&1) || rc=$?
  fi
  for level in AGC PREAMP; do
    if [ "$daemon" -eq 1 ]; then
      part=$(printf '%s\n' "$reply" | awk -v level="$level" '
        /^set_level: / { active = ($2 == level); n = 0 }
        active { print; if (/^RPRT / && ++n == 2) active = 0 }
      ')
      if ! printf '%s\n' "$reply" | grep -q '^set_level: '; then
        part="$reply"
      fi
    else
      part=$(printf '%s\n' "$reply" | awk -v level="$level" '
        (level == "AGC" && !seen) || (level == "PREAMP" && seen) { print }
        /^-?[0-9]+$/ { seen = 1 }
      ')
    fi
    error=$(printf '%s\n' "$part" | grep -E 'error =|^RPRT -[0-9]+' | head -1) || true
    reading=$(printf '%s\n' "$part" | grep -E '^-?[0-9]+$' | head -1) || true
    if [ -z "$error" ] && [ "$daemon" -eq 1 ] \
       && [ "$(printf '%s\n' "$part" | grep -c '^RPRT 0$')" -ne 2 ]; then
      error='incomplete rigctld reply'
    fi
    if [ -n "$error" ] || [ "$rc" -ne 0 ] || [ -z "$reading" ]; then
      if [ -z "$error" ]; then
        error=$(printf '%s\n' "$part" | sed '/^$/d' | head -1)
        error="${error:-no $level reading (exit $rc)}"
      fi
      if [ "$level" = AGC ]; then
        printf '  !! AGC unverified (rigctl: %s)\n' "$error"
      else
        printf '  !! IPO unverified (rigctl: %s)\n' "$error"
      fi
    elif [ "$reading" = 0 ]; then
      if [ "$level" = AGC ]; then
        printf '  AGC off verified\n'
      else
        printf '  IPO verified\n'
      fi
    elif [ "$level" = AGC ]; then
      printf '  !! AGC reads %s (not off)\n' "$reading"
    else
      printf '  !! PREAMP reads %s (not IPO)\n' "$reading"
    fi
  done
}

# Is anyone already using this frequency? shrike and besra have no occupancy
# check of their own, so the launchers make it out here -- tuning first, because
# sensing whatever the rig was left on measures the last attempt's channel.
#
# THE MODEM AND ITS BANDWIDTH ARE ARGUMENTS because the question is not "is this
# 2300 Hz of spectrum in use" but "is the part of it we are about to fill in use".
# `core.occupied` turns the pair into the band, and an unlisted one answers wide.
#
# Exit 0 clear, 1 busy, 2 deaf, 3 a wait budget that ran out, 5 a host on the
# channel saying it is free. Busy and deaf are different answers because the
# caller does different things with them: --force is the operator's ears
# overruling the detector, and a station that cannot hear the channel has
# nothing for them to weigh.
#
# 5 IS NOT A KIND OF CLEAR. A PACTOR Free Signal is a host with no link in
# progress inviting a client to call it [Professional 3.2 §1.4], so the channel
# is occupied -- by the station this arm wants -- and the answer is to call now
# rather than to wait for the energy to stop. Nothing about it waives an
# occupancy refusal: it is read before the scores are consulted and decides on
# its own, and the Robust Call -- the same frame family, a client calling
# somebody else -- is an ordinary busy with the destination named.
#
# TWO POLICIES OVER ONE INSTRUMENT, AND THE DIFFERENCE IS DELIBERATE.
#
# `sense_channel` is the operator's gate. With no budget it is one window of
# `core.busy.WINDOW_S` and that window decides. Given a budget it is a pounce: it
# waits the occupant out on this same open stream in `core.busy.POUNCE_WINDOW_S`
# windows and keys on three clear ones in a row, the third being the one that
# decides. Three of them are 9 s, which a live session's turnarounds never give
# (tests/core/test_busy_evidence.py holds the measurement) and an empty channel
# always does.
#
# `sense_channel_unattended` has nobody listening: no ear to catch a turnaround
# the detector misses, and no --force to overrule it. It refuses at the first
# busy window and wants a second clear one before it agrees, and with no budget
# it does not wait -- a busy channel is an attempt skipped rather than a run
# stalled. That is the default and it is what an unattended pass gets.
#
# A caller whose operator IS at the monitor can hand it a budget, and then it
# waits the occupant out on the same open stream exactly as the gate above does.
# The refusal is not waived by it -- nothing waives it here -- and the last
# window still decides. A budget it cannot spend is said out loud rather than
# dropped: POUNCE_WAIT reaches every verb in onair.sh and tools/autopilot.sh's
# --wait, and campaign.sh is the one caller left that cannot spend it. An
# operator who exports it there deserves to hear that it goes nowhere.
#
# NEITHER RULE CONTAINS THE OTHER, which is why there are two rather than one:
# the pounce's run of short windows catches the gaps in an over that a single
# window sits inside, and the long window catches the occupant a short one reads
# as noise. Unattended is the stricter where it counts -- nothing can waive it,
# and one busy window is the end of the attempt.
sense_channel()            { _sense_channel "$1" "$2" "${3:-}" "${4:-0}"; }
sense_channel_unattended() {
  case "${4:-0}" in
    ''|0|0.0)
      case "${POUNCE_WAIT:-0}" in
        ''|0|0.0) ;;
        *) printf '  POUNCE_WAIT=%s is not spent here -- nobody is at this pass to
   hear a turnaround, so a busy channel is this attempt skipped. Every verb in
   tools/onair.sh spends it, and tools/autopilot.sh --wait %s does.\n' \
             "$POUNCE_WAIT" "$POUNCE_WAIT" ;;
      esac
      _sense_channel "$1" "$2" "${3:-}" ;;
    *)
      printf '  someone is listening: a busy channel is waited out, up to %ss\n' "$4"
      _sense_channel "$1" "$2" "${3:-}" "$4" ;;
  esac
}

# WHICH HOST SAID IT WAS FREE, for the caller that has to hand it on: the ident,
# the kind of Free Signal, and the epoch of its last burst. Written by the sense
# on a 5 and cleared before every run, so a stale one can never be read as this
# window's.
host_free() { cat "$REPO/logs/sense/host-free" 2>/dev/null; }

_sense_channel() {
  local dial="$1" rc=0
  shift
  tune "$dial"
  rm -f "$REPO/logs/sense/host-free"
  PYTHONPATH="$REPO/tools${PYTHONPATH:+:$PYTHONPATH}" AUDIO_IN="${AUDIO_IN:-$AUDIO}" \
    CLAIM_PENDING_LINE="$(claim_pending sense)" \
    claim_worker "${HFMODEM_PYTHON:-$VENV/python}" -m consolelib.sense "$dial" "$@" || rc=$?
  return "$rc"
}

# WHAT THE RECEIVER IS DELIVERING, MEASURED WHILE NOTHING IS KEYED. The sense
# above says whether anyone else is on the channel and refuses a receiver too
# dead to judge one; past that an arm keyed at whatever level the chain happened
# to be at. On 2026-08-22 that level moved 16 dB between calls minutes apart on
# one rig, and only the receive side had moved: two sessions read back none of
# their own thirty-odd keyings out of their own recordings while the five around
# them read all of theirs.
#
# Exit 0 live, 1 a level no arm can be read back at, 2 a codec delivering
# nothing. core/levels.py holds the bounds, what each was measured off and the
# wording, so a refusal names the knob that reaches THIS converter -- which is
# the part that has been got wrong here before.
#
# ONE INSTRUMENT, TWO POLICIES, exactly as the sense above has. Attended,
# onair.sh's `gate_rx_level` refuses and the operator turns something. Unattended,
# attempts.sh skips the arm and writes what it read into the pass's own summary.
# Each policy lives with its caller because a refusal is worth no more than the
# record it leaves, and those two records are different files.
#
# ASK IT ON THE CHANNEL THE ARM WILL KEY, after `tune`: the in-band level is the
# band's as much as the chain's, and read before the dial is set it is the last
# attempt's channel.
rx_level() {
  PYTHONPATH="$REPO/tools${PYTHONPATH:+:$PYTHONPATH}" AUDIO_IN="${AUDIO_IN:-$AUDIO}" \
    claim_worker "${HFMODEM_PYTHON:-$VENV/python}" -m consolelib.sense rx-level
}

claim_daemon_start() {
  local log="$1" result pid started reply deadline
  if [ -n "${RIGCTLD_PID:-}" ] && claim_processes alive "$RIGCTLD_PID"; then return 0; fi
  claim_spawn --long-lived --device "$CAT_PORT" -- \
      rigctld -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" --set-conf=ptt_type=None \
      >"$log" 2>&1 || return
  result=$(sed -n 's/^CLAIM NATIVE /CLAIM NATIVE /p' "$log" | tail -1)
  read -r _ _ pid started <<< "$result"
  case "$result" in 'CLAIM NATIVE '*) ;; *) return 1 ;; esac
  printf '%s %s\n' "$pid" "$started" >>"$CLAIM_CHILDREN"
  RIGCTLD_PID="$pid"
  deadline=$((SECONDS + 12))
  while [ "$SECONDS" -lt "$deadline" ]; do
    claim_processes alive "$pid" || break
    reply=$(printf '\\get_conf ptt_type\nq\n' | nc -w 3 127.0.0.1 4532 2>/dev/null) || reply=
    case "$reply" in *ptt_type=None*|None) return 0 ;; esac
    sleep 0.1
  done
  claim_processes end "$pid" || return
  printf 'rigctld did not confirm ptt_type=None within 12 s\n' >&2
  return 1
}
