#!/usr/bin/env bash
# The ack-placement experiment, start to finish, with the transmitter proved
# down at the end whatever happens.
#
# WHY THIS EXISTS. The experiment is one command inside `onair.sh pactor`, and
# everything around it — checking the station is really there, proving the dial
# took, recording what happened, proving the key came back down — was being done
# from memory each time. Memory lost a slot to a placeholder port, three
# measurement runs to a frequency that silently did not take, and one evening to
# a transmitter that stayed up after the audio stopped. So the sequence is here,
# it fails closed, and its exit trap drops the line whatever went wrong above.
#
# It does not invent a keying path. Keying is `onair.sh pactor`, which is the
# one tested route to the transmitter; this wraps it.
#
#   ./tools/rigtest.sh preflight     # touches nothing, keys nothing
#   ./tools/rigtest.sh run           # the experiment
#   ./tools/rigtest.sh down          # prove the key is down, any time
#
# CAT_PORT and PTT_PORT come from the environment; there are no defaults
# anywhere in this tree and there will not be. See tools/lib/ports.sh.

set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/ports.sh"
# `prove_down` -- the one thing that must happen on every exit path, and it
# lives beside the rig it lowers, shared with the unattended launchers.
source "$(dirname "${BASH_SOURCE[0]}")/lib/rig.sh"

GW="${GW:-}"
CENTRE="${CENTRE:-0}"
DIAL=$((CENTRE - 1500))
STAMP="$(date +%Y%m%d-%H%M%S)"
SESSION="${HFMODEM_DATA_DIR:-$REPO/operator-data}/working/rig-session-$STAMP"

say()  { printf '\n== %s ==\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\n!! %s\n' "$*" >&2; exit 1; }

# ONE ROUTE TO THE AIR for all three modems, so POUNCE means the same thing
# whichever one is calling. `onair.sh pounce` has been generic in its verb since
# it was written -- it takes <verb> <gw> <centre> and re-invokes itself, and
# every verb carries the centre in the same position so the wait knows where to
# listen. Only this wrapper was not generic: PACTOR read POUNCE and the other two
# silently ignored it, which is worse than refusing, because an operator who set
# it on a busy channel had no way to see it had not taken.
air() {
  [ -n "$GW" ] && [ "$CENTRE" -gt 0 ] || die "Set GW and CENTRE explicitly before a test transmission."
  local argv=("$@")
  [ -n "${POUNCE:-}" ] && argv=(pounce "${argv[@]}")
  "$REPO/tools/onair.sh" "${argv[@]}"
}

# TRANSMIT POWER, AT BOTH ENDS OF EVERY SESSION -- OUT OF THE RUN'S OWN LOG.
#
# No rig-session directory in this tree carried it until 2026-08-15, so no
# silence in the archive can be read as "the gateway was absent" without that
# qualifier: three ARDOP attempts on 2026-08-14 went out at RFPOWER 0.05 -- 5 W
# -- and only the operator's meter knew. It is recorded at both ends because it
# moves mid-slot, which is what happened that night, and it comes over CAT
# because the witness tap cannot answer it: the monitor read rms 0.3157 at 5 W
# and 0.3163 at 50 W the same evening, watching the modulation stage and not the
# output.
#
# THIS WRAPPER NO LONGER ASKS THE RIG. `onair.sh` already reads RFPOWER inside
# every keying verb -- once before the gate and once after the modem exits -- and
# tees both into session.log, so asking again from here was a third and fourth
# open of the CAT port for a number already in hand. On the PACTOR path that is
# not merely redundant: shrike opens the CAT port ITSELF, and since 2026-08-15 a
# readback that comes back empty -- which is what a contended open looks like --
# refuses the run rather than warning. The wrapper does not invent a keying path;
# it does not invent a CAT path either.
record_power() {
  awk -F'TX power: ' '/TX power: / { if (!first) first = $2; last = $2 }
    END { if (first) { printf "    TX power  %s   at session start\n", first
                       printf "    TX power  %s   at session end\n",   last }
          else         print  "    TX power  UNRECORDED -- the run logged none" }' \
      "$SESSION/session.log"
}

trap 'echo; echo "-- proving the transmitter is down --"; prove_down' EXIT

# -- checks that key nothing --------------------------------------------------

preflight() {
  require_ports
  require_hamlib
  say "station"
  note "CAT  $CAT_PORT"
  note "PTT  $PTT_PORT"
  note "rigctl $(command -v rigctl)"
  for p in "$CAT_PORT" "$PTT_PORT"; do
    [ -c "$p" ] || die "$p is not a character device — is the adapter plugged in?"
  done

  say "nothing else is holding the ports"
  if lsof "$CAT_PORT" "$PTT_PORT" 2>/dev/null | grep -q .; then
    lsof "$CAT_PORT" "$PTT_PORT" 2>/dev/null
    die "something already owns a port; stop it before keying (tools/onair.sh down)"
  fi
  note "both free"

  say "the receiver is alive"
  # `onair.sh check` opens no transmitter. A reading near 0.00007 is a radio
  # that is off or unplugged, and reads exactly like a quiet band.
  "$REPO/tools/onair.sh" check || die "the check failed — do not go on"

  say "preflight passed"
  note "nothing above keyed anything"
}

# -- the experiment -----------------------------------------------------------

run() {
  # THE GATEWAY AND CENTRE READ POSITIONALLY, as `ardop` and `vara` have always
  # read them. They arrived here only through the environment, so
  # `rigtest.sh run KY4RY 14103000` -- the form the other two verbs teach --
  # passed both words through to shrike as flags and called the 40 m DEFAULT
  # instead: the wrong station on the wrong band, while the operator read the
  # 20 m dial they had just set by hand off the rig. Flags still pass through
  # untouched, which is what the leading `-` distinguishes.
  case "${1:-}" in -*|"") ;; *) GW="$1"; shift ;; esac
  case "${1:-}" in -*|"") ;; *) CENTRE="$1"; shift ;; esac
  DIAL=$((CENTRE - 1500))
  preflight
  mkdir -p "$SESSION"
  say "recording to $SESSION"

  {
    echo "# Rig session $STAMP"
    echo
    echo "    gateway   $GW"
    echo "    centre    $CENTRE Hz   (dial $DIAL)"
    echo "    CAT       $CAT_PORT"
    echo "    PTT       $PTT_PORT"
    echo "    commit    $(git -C "$REPO" rev-parse --short HEAD)"
    echo "    TX power  in session.log, at both ends of the run"
    echo
    echo 'PASS is the peer status-byte counter advancing three cycles running.'
    echo 'There is no three-cycle abort. Call until the operator says stop, the'
    echo 'retry budget runs out, or something is heard worth acting on. A cycle'
    echo 'where the peer is heard but not decoded does not spend the budget.'
    echo
  } > "$SESSION/README.md"

  say "calling $GW on $CENTRE"
  note "everything below is also written to $SESSION/session.log"
  note "Ctrl-C is safe: the key comes down on the way out"

  # The tested path. --max-cycles bounds the slot; --hold lets a gateway that
  # answers keep the link long enough for the counter to move.
  #
  # CALL PROPERLY OR DO NOT CALL. `--retries 2` sat here from 2026-08-13, when a
  # brief said "abort after three cycles" and a slot report complained the tool
  # did not obey it. Enforcing it was the wrong lesson from a real problem: the
  # problem was 61 cycles at a wall with nothing heard, and the answer to that is
  # the FSM's own exemption -- a cycle where the peer is HEARD but not decoded
  # already does not count -- not a hard cap of three at every gateway alike.
  # Three cycles is under four seconds. A gateway serving somebody else, or
  # fading, or simply slow to notice, is gone before it could answer, and the one
  # connect candidate ever recorded at KY4RY came from an operator overriding
  # this by hand on 2026-08-14 and asking for more. An operator who wants to stop
  # early can still say --retries 2; the wrapper is what guarantees the key comes
  # down on every exit.
  # Trailing args reach the tool unchanged and land last, so an operator who has
  # listened and judged the channel clear can say --force through the wrapper
  # rather than around it.
  #
  # POUNCE=1 takes the channel the moment it clears, via onair.sh's own pounce
  # verb — which waits on two clear windows, re-asks the verb's own sense, and
  # never forces. Running it from here keeps the exit trap and the recording.
  air pactor "$GW" "$CENTRE" \
      --mail-fetch --hold 150 --max-cycles 400 --retries 20 "$@" \
      2>&1 | tee "$SESSION/session.log"

  record_power | tee -a "$SESSION/README.md"

  say "what the session decided"
  # The lines worth reading afterwards, pulled out so they are not buried -- and
  # the power the slot ran at is one of them: a silence at 5 W and a silence at
  # 100 W are different results, and until 2026-08-15 the report read the same.
  grep -E 'TX power|\[ack\]|counter|ACKNOWLEDG|verdict:|keyed [0-9]+ of|receiver rms|PTT lead:' \
       "$SESSION/session.log" | tee "$SESSION/verdict.txt" || true

  say "captures"
  ls -1 "$REPO/captures" 2>/dev/null | tail -3 | sed 's/^/   /'
  note "session record: $SESSION"
}

# -- the second and third experiments, if the operator asks for them ----------
#
# Each is one attempt with its own bar, not an open-ended session. They are here
# so that "yes, go on" does not mean improvising a command at the rig; the same
# preflight, the same recording, the same exit trap.

ardop() {
  local gw="${1:?gateway callsign}" centre="${2:?channel centre Hz}" bw="${3:-500}"
  shift 3 2>/dev/null || shift $#
  preflight
  mkdir -p "$SESSION"; say "recording to $SESSION"
  {
    echo "# ARDOP session $STAMP"
    echo; echo "    gateway $gw   centre $centre Hz   BW$bw"
    echo "    commit  $(git -C "$REPO" rev-parse --short HEAD)"
    echo "    TX power  in session.log, at both ends of the run"
    echo
    echo 'The old two-halves PASS bar is SETTLED -- both halves were met against'
    echo 'W6IDS and WM4RB on 2026-08-14 and confirmed off the captures. Do not'
    echo 'spend a slot re-asking it. (0x51 and 0x40 in that old wording are FRAME'
    echo 'TYPES, not session ids; 0x51 equals the KE8LVA session id by accident.)'
    echo
    echo 'PASS, and it is readable live: THE GATEWAY CHANGES MODE after a DATANAK.'
    echo 'On 2026-08-14 every NAK went out while the gateway was already'
    echo 'retransmitting -- WM4RB ran a 4.41 s frame on a 6.42 s metronome and our'
    echo 'answers keyed at frame-end +2.25 to +2.70 s, four of four, into a window'
    echo 'that had closed at +2.01. It heard none of them, its mode and metronome'
    echo 'never moved across ~10 transmissions, and every data frame failed.'
    echo
    echo 'The hold-back now waits the frame own length instead of a flat 6 s, which'
    echo 'off the same captures moves the answer to end +0.23 to +0.55 s -- inside'
    echo 'both gateways windows. THIS RUN IS THE FIRST TIME THAT HAS BEEN ON AIR.'
    echo 'So: watch for a mode name that changes after a NAK, and record the mode'
    echo 'on every RX line either way. No mode change and no decode is a real'
    echo 'result; say so plainly rather than explaining it.'
    echo
    echo 'A silent gateway is a silent gateway, not a regression -- and a thin'
    echo 'sample: six connect requests in ~21 s against a minute of PACTOR cycles.'
    echo 'Two of WM4RB four frames were late for a SECOND reason -- acquisition did'
    echo 'not find them until ~1.8 s after they ended -- which this run does not fix.'
  } > "$SESSION/README.md"
  say "calling $gw on $centre, BW$bw"
  note "besra reads its dial back before it keys: a rig that does not report"
  note "$((centre - 1500)) refuses the run outright (QSY FAILED ... not keying)"
  air ardop-mail "$gw" "$centre" "$bw" \
      --mycall "$(station_mycall)" --mail-fetch --mail-timeout 280 "$@" \
      2>&1 | tee "$SESSION/session.log"
  record_power | tee -a "$SESSION/README.md"
  say "what the session decided"
  # Every mode name, because PASS is a mode that CHANGES and a grep for one mode
  # cannot show that. `ok=` carries the CRC verdict beside it. The power the slot
  # ran at rides along: three attempts here went out at 5 W on 2026-08-14 and the
  # report of that silence could not say so.
  grep -E 'TX power|4FSK|4PSK|8PSK|16QAM|DATANAK|DATAACK|IDFrame|ok=|verdict:|keyed [0-9]+ of' \
       "$SESSION/session.log" | tee "$SESSION/verdict.txt" || true
  say "the modes the gateway used, in order — a change here is the PASS"
  grep -oE '(4FSK|4PSK|8PSK|16QAM)\.[0-9]+\.[0-9]+(\.[EO])?' "$SESSION/session.log" \
       | uniq -c | sed 's/^/   /' || true
  note "session record: $SESSION"
}

vara() {
  local gw="${1:?gateway callsign}" centre="${2:-7103800}" bw="${3:-2300}"
  shift 3 2>/dev/null || shift $#
  # kestrel synthesises exactly one connect-request waveform, BW2300, whatever
  # --bw is told (SPEAKABLE, tools/kestrel_connect.py). A 2750-only gateway
  # answers that request and the session comes up as BW2300, so 2750 stations
  # are called AT 2300 -- four of the nearest gateways are 2750-only, and this
  # wrapper advertising a bandwidth the tool refuses cost two full preflights
  # on 2026-08-13. BW500 is a genuinely different alphabet -- 14 carriers on a
  # 46.9 Hz lattice against kestrel's 70 on 23.4 Hz. kestrel both reads and keys
  # it as of 2026-08-15, and a real VARA answers the request -- but the connect
  # cannot COMPLETE: step 4 rides a wideband DATA frame and the 500 Hz byte
  # layout is unestablished, so the gateway answers and then waits out its
  # pending timer for a caller-ID that never arrives. Still refused, now for the
  # right reason: a station answered and left hanging is a channel spent twice.
  # Refused here,
  # before the preflight is spent.
  case "$bw" in
    2300) ;;
    -*|force) die "expected a bandwidth in the third slot, got '$bw' -- say: rigtest.sh vara <gw> <centre> 2300 [flags]" ;;
    2750) die "BW2750 is not a waveform kestrel can send. It emits the BW2300 connect-request whatever --bw says, and a 2750-only gateway answers it -- the session comes up as BW2300. Call $gw with bw 2300." ;;
    *)    die "BW$bw is not a waveform kestrel can send -- it emits BW2300 only. A 2750-only gateway answers that at bw 2300; BW500 is a different alphabet kestrel can now read AND key, but the session cannot complete -- no BW500 link-setup -- so calling one answers a gateway and leaves it hanging." ;;
  esac
  preflight
  mkdir -p "$SESSION"; say "recording to $SESSION"
  {
    echo "# VARA session $STAMP"
    echo; echo "    gateway $gw   centre $centre Hz   BW$bw"
    echo "    commit  $(git -C "$REPO" rev-parse --short HEAD)"
    echo "    TX power  in session.log, at both ends of the run"
    echo
    echo 'ONE ATTEMPT, on a channel the gate calls clear. Nothing has changed in'
    echo 'VARA since it went 0-for-3, and what remains -- the frame that answers a'
    echo 'turn-request -- is an analysis problem and not a rig problem. There is'
    echo 'no PASS here to chase: record what the gateway sends and stop.'
  } > "$SESSION/README.md"
  say "calling $gw on $centre, BW$bw"
  # The 8.4 s path is gone by construction: the daemon no longer keys at all
  # (`rigctld_start none` + --line-ptt), and a quiet transport now forfeits its
  # budget after one ask. What is not yet done is watching it key a real rig.
  # The bench check this used to ask for was paid on 2026-08-14 at 00:20, on air,
  # and the notice went on printing for three more runs that day. A warning that
  # outlives its danger is one an operator learns to key past, which is the only
  # thing that makes the next real warning cheap to ignore.
  note "kestrel drives the keying line itself; the daemon is out of the path."
  air vara "$gw" "$centre" "$bw" "$@" \
      2>&1 | tee "$SESSION/session.log"
  record_power | tee -a "$SESSION/README.md"
  say "what the session decided"
  grep -E 'TX power|turn|TURN|verdict:|keyed [0-9]+ of|connected|CONNECT' \
       "$SESSION/session.log" | tee "$SESSION/verdict.txt" || true
  note "session record: $SESSION"
}

case "${1:-}" in
  preflight) preflight ;;
  run)       shift; run "$@" ;;
  ardop)     shift; ardop "$@" ;;
  vara)      shift; vara "$@" ;;
  down)      require_ports; prove_down; trap - EXIT ;;
  *)         die "usage: $0 preflight                                  keys nothing
       $0 run   [gw] [centre] [extra shrike flags]     PACTOR, the ack experiment
       $0 ardop <gw> <centre> [bw] [extra flags]       ARDOP
       $0 vara  <gw> <centre> [bw=2300] [extra flags]  VARA
       $0 down                                         prove the key is down

   Centre in Hz; the dial is 1500 Hz below it and the tools work that out.
   POUNCE=1 before any of the three waits for a busy channel to clear, then keys.
   CAT_PORT and PTT_PORT come from the environment; there are no defaults." ;;
esac
