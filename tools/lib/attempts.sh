# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
# One bounded, sensed, unattended call attempt per mode, and the readback between
# them. Sourced by tools/campaign.sh and tools/autopilot.sh, which differ in how
# they choose targets and in nothing else that touches the transmitter.
#
# NO --force UNLESS THE CALLER IS ATTENDED AND SAYS SO. The occupancy refusal
# stands on every attempt made from here: kestrel
# senses inside the modem before each VARA call, and the PACTOR and ARDOP
# attempts are sensed out here (`sense_channel_unattended`, lib/rig.sh) because
# shrike and besra have no check of their own -- the reason onair.sh's verbs
# sense outside the modem too, and it is the same instrument held to the
# stricter of two rules. The operator's ears outrank the sense and --force is
# theirs to give; the operator who could give it is by definition not here --
# that is what unattended means. An attempt skipped for a busy channel is the
# script working.
#
# The caller sets OUT (a capture directory that exists), SUM (its summary.tsv)
# and MYCALL; lib/ports.sh and lib/rig.sh are already sourced.
#
# TRY_WAIT is the caller saying an operator is at the monitor after all: seconds
# an occupied channel is waited out before the attempt is given up. Unset is 0,
# which is the rule above word for word, and no budget waives the refusal -- the
# last window before key-down decides either way.
#
# TRY_FORCE is the same caller saying the operator has heard the channel and
# given the word. It waives the refusal and NOTHING ELSE: the sense still runs,
# the reading is still printed and still recorded, and a station that cannot hear
# the channel at all still ends the pass -- forcing a transmission you are blind
# to is worse than being refused, not better. It waives the receive level on the
# same terms, and the arm it lets through says FORCED_LEVEL in the summary rather
# than passing for one the gate was happy with. campaign.sh and autopilot.sh
# never set it and must not: the operator who could give it is by definition not
# there. tools/autopilot.sh spends `--wait`, and campaign.sh spends nothing
# because nobody is there.

# A child wedged in a CAT transaction holds the port with the line as it stands,
# and whether the line falls when the owning process dies is unmeasured on this
# hardware (core/ptt.py). So read it back rather than assume.
settle() {
  claim_processes owner || return
  exec 7>&-
  CLAIM_LAUNCH=0
  if [ "$CLAIM_OWNED" = 0 ] && [ "$CLAIM_ADMITTED" = 0 ]; then
    die "CLAIM NOT ADMITTED attempt failed before readiness"
  fi
  "$REPO/tools/onair.sh" down 7>&- 8>&-
}

# Clear to call? Exit 0 yes, 1 the attempt is over -- because somebody else is on
# the channel, or because the receiver could not read this arm back off it. A
# station that cannot hear the channel at all ends the whole pass: every attempt
# after it would be exactly as blind.
gate_unattended() {
  local mode="$1" call="$2" centre="$3" modem="$4" bw="$5" clear=0 why
  local free_ident free_kind free_at
  # A Robust Call in the window names the station it is calling; say so when that
  # station is this attempt's. The Free Signal of the LAST attempt is cleared
  # here, or an arm on a quiet channel inherits the raster of another host.
  export DXCALL="$call"
  unset HOST_FREE_IDENT HOST_FREE_AT
  # THE WAIT IS SPENT ONLY WHERE IT CAN CHANGE THE ANSWER. Under TRY_FORCE the
  # refusal is waived below whatever the sense says, so standing through the
  # budget first spends the operator's slot to reach a verdict already overruled
  # -- and they gave the word because the channel is clear NOW, not in 900 s.
  # The sense still RUNS: the reading is printed, recorded, and the deafness
  # check below still ends the pass.
  local wait_s="${TRY_WAIT:-0}"; [ -n "${TRY_FORCE:-}" ] && wait_s=0
  sense_channel_unattended "$((centre - 1500))" "$modem" "$bw" "$wait_s" || clear=$?
  if [ "$clear" -eq 2 ]; then
    die "this station cannot hear the channel -- ending the pass; every attempt \
after this one would be exactly as blind"
  elif [ "$clear" -eq 5 ]; then
    # NOT A WAIVED REFUSAL AND NOT A SKIP. The sense read an ident out of the
    # channel and the station that owns it has no link in progress, so this
    # attempt goes ahead and no wait budget is spent reaching that.
    read -r free_ident free_kind free_at <<<"$(host_free)"
    printf '  host %s free on %s -- %s, calling now\n' \
        "$free_ident" "$((centre - 1500))" "$free_kind"
    export HOST_FREE_IDENT="$free_ident" HOST_FREE_AT="$free_at"
  elif [ "$clear" -ne 0 ] && [ -n "${TRY_FORCE:-}" ]; then
    printf '%s\t%s\t%s\t%s\tFORCED\tover a channel the sense refused\n' \
        "$(date +%H:%M:%S)" "$mode" "$call" "$centre" >> "$SUM"
    printf '  %-7s %-9s %-9s  FORCED         over a channel the sense refused -- operator authorised\n' \
        "$mode" "$call" "$centre"
  elif [ "$clear" -ne 0 ]; then
    # WHICH REFUSAL IT WAS, because the operator's next move differs: a channel
    # that never cleared inside the budget is a frequency to leave alone for a
    # while, one that cleared and was taken again in the deciding window is a
    # turnaround worth another attempt in a minute, and one refused with no
    # budget at all is a wait nobody asked for. All three read `SKIPPED_BUSY
    # channel in use`, so an operator who had just spent fifteen minutes could
    # not tell which of them they had bought.
    case "$clear:${TRY_WAIT:-0}" in
      3:*)     why="in use for the whole ${TRY_WAIT:-0}s wait" ;;
      *:0|*:)  why="in use, and no wait budget was set" ;;
      *)       why="cleared inside the ${TRY_WAIT:-0}s wait, in use again on the window that decides" ;;
    esac
    printf '%s\t%s\t%s\t%s\tSKIPPED_BUSY\t%s\n' \
        "$(date +%H:%M:%S)" "$mode" "$call" "$centre" "$why" >> "$SUM"
    printf '  %-7s %-9s %-9s  SKIPPED_BUSY   %s\n' "$mode" "$call" "$centre" "$why"
    return 1
  fi
  gate_rx_unattended "$mode" "$call" "$centre"
}

# THE SAME READING, WITH NOBODY AT THE KNOB. `rx_level` (lib/rig.sh) is the
# instrument and onair.sh holds the other policy over it: there, a level no arm
# can be read back at is a refusal the operator answers by turning something.
# Here there is no one, and both of the obvious answers waste the night -- ending
# the pass on the first quiet arm throws away every hour after it, and keying on
# regardless fills a night of recordings with arms whose answer cannot be read
# out of them.
#
# SO THE ARM IS SKIPPED AND THE NEXT ONE IS MEASURED AGAIN. A level is not a
# station fault the way a missing codec is: it moves with the noise under the
# receiver and with whatever is on the channel, and the next arm is on another
# frequency minutes later. Skipping costs the two seconds of one reading and
# leaves the pass free to recover on its own, where dying costs the slot; a run
# that never recovers leaves a column of SKIPPED_LEVEL rows all reading the same
# level, which is the diagnosis rather than the loss of one.
#
# A CODEC DELIVERING NOTHING STILL ENDS THE PASS, for the reason deafness does:
# there is no gain to recover to, and every arm after it would be exactly as
# unreadable. TRY_FORCE does not waive that one either.
#
# AND WHICH ARMS WERE KEYED ANYWAY IS THE RECORD'S TO SAY. A pass read in the
# morning off summary.tsv has to tell an arm that never keyed from one that keyed
# under a level the gate refused, or the whole night's silences are unreadable
# together: SKIPPED_LEVEL never keyed, FORCED_LEVEL keyed with the operator's
# word on it and its own result on the line below, and both carry the reading
# they were decided on. Neither is SKIPPED_BUSY, because no part of TRY_WAIT is
# spent here -- waiting is what a taken channel needs, and no number of windows
# turns a gain up.
gate_rx_unattended() {
  local mode="$1" call="$2" centre="$3" rc=0 out reading res ret
  out="$(rx_level)" || rc=$?
  printf '%s\n' "$out"
  case "$rc" in
    0) return 0 ;;
    2) die "this station's codec is delivering no audio -- ending the pass; every \
attempt after this one would be exactly as unreadable" ;;
  esac
  reading="$(printf '%s\n' "$out" | tail -1 | sed 's/^ *//')"
  res=SKIPPED_LEVEL; ret=1
  if [ -n "${TRY_FORCE:-}" ]; then res=FORCED_LEVEL; ret=0; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%H:%M:%S)" "$mode" "$call" "$centre" "$res" "$reading" >> "$SUM"
  printf '  %-7s %-9s %-9s  %-14s %s\n' "$mode" "$call" "$centre" "$res" "$reading"
  return "$ret"
}

# A receiver that cannot hear the channel ends the pass wherever the judgement is
# made — out here by sense_channel_unattended, or inside kestrel, which reaches
# the same verdict in `core.busy` and writes it to its own log. Continuing past
# it would make every remaining attempt exactly as blind.
deaf_or_die() {
  if grep -q "RECEIVER DEAF\|RECEIVER INTERMITTENT" "$1" 2>/dev/null; then
    grep -m1 "RECEIVER DEAF\|RECEIVER INTERMITTENT" "$1" | sed 's/^ */  /'
    die "this station cannot hear the channel -- ending the pass"
  fi
}

# What the log says happened, on the attempt's own line, followed by every mail:
# line verbatim -- the SID the peer answered with is the whole question tonight
# and it is not a thing to summarise.
#
# THE RESULT IS WHAT WE DECODED, NOT WHO WAS ON THE AIR. The word here was
# `nothing` until a summary of it became "WS8EOC never answers on 40 m", and the
# operator contradicted that from having heard WS8EOC answer on 40 m the same
# evening. Our receiver decoding nothing across the cycles we called in is the
# measurement; whether the far station transmitted is not ours to say. NO_DECODE
# says the first and stops, and the detail beside it carries the count.
report() {
  local mode="$1" call="$2" centre="$3" res="$4" detail="$5" log="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date +%H:%M:%S)" "$mode" "$call" "$centre" "$res" "$detail" >> "$SUM"
  printf '  %-7s %-9s %-9s  %-14s %s\n' "$mode" "$call" "$centre" "$res" "$detail"
  grep '^mail:' "$log" 2>/dev/null | sed 's/^/      /' || true
}

# Trailing arguments reach shrike, and TRY_TAG names the attempt where one
# gateway is called more than once in a pass -- a comparison keeps two logs and
# two capture directories or it keeps neither. Both are for the caller that
# needs them; a survey sets neither and is unchanged.
pactor_try() {
  local call="$1" centre="$2" tag
  # On its own line because bash 3.2 -- the /bin/bash every launcher here runs
  # under -- expands every word of a `local` before it assigns any of them, so
  # `tag="…$call…"` in the declaration above reads whatever GLOBAL `call` the
  # caller happens to have. campaign.sh has one, by coincidence of its read
  # loop; onair.sh does not, and under `set -u` that is the end of the run.
  tag="$OUT/pactor-$call-$centre${TRY_TAG:+-$TRY_TAG}"
  shift 2
  # --serial/--ptt-port are the arm gate's demand: without them shrike refuses
  # at "NOT KEYING" and an unattended pass logs 'nothing' for every target.
  local cmd=(claim_run timeout -k 5 200 "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.shrike.onair --transmit
      --serial "$CAT_PORT" --ptt-port "$PTT_PORT"
      --audio-out "$AUDIO" --audio-in "$AUDIO"
      --mycall "$MYCALL" --dxcall "$call" --dial "$((centre - 1500))"
      --message "DE $MYCALL QSL" --over
      --retries 6 --max-cycles 26 --hold 18 --outdir "$tag" "$@")
  # The plan, printed out of the array that would have run it. A dry run that
  # spells the invocation out a second time is a dry run of the spelling.
  if [ -n "${TRY_DRY:-}" ]; then
    # Quoted per word, because a device called "USB Audio Device" flattens into
    # three arguments the moment the operator copies the line back out.
    printf '  WOULD SENSE %s (pactor%s), then RUN:\n   ' \
        "$centre" "${TRY_FORCE:+, and key over a refusal -- FORCED}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  # The daemon lets go of the CAT port first: shrike opens it itself, and the
  # sense's oneshot rigctl needs it free for the tune.
  claim_processes daemon || return
  gate_unattended PACTOR "$call" "$centre" pactor "" || return 0
  "${cmd[@]}" >"$tag.log" 2>&1 || true
  local cs res cy
  # The unqualified total includes non-zero-error decodes; onair.py breaks out
  # the trustworthy subset precisely so a survey does not read the wrong number.
  cs=$(grep -oE "\\(([0-9]+) PACTOR-1 at zero errors" "$tag.log" 2>/dev/null | grep -oE "[0-9]+" | head -1 || true)
  cs=${cs:-0}
  # How many cycles that count is out of, from the session's own summary. Zero
  # control signals over three cycles and zero over fifty-one are not the same
  # observation, and the line carried only the zero.
  cy=$(sed -nE 's/^keyed [0-9]+ of ([0-9]+) cycles.*/\1/p' "$tag.log" 2>/dev/null | head -1 || true)
  if   grep -q "ACKNOWLEDGED"   "$tag.log"; then res=DATA_PHASE
  elif grep -q "GRID REVERSED"  "$tag.log"; then res=CHANGEOVER
  elif grep -q "CONNECTED to"   "$tag.log"; then res=CONNECTED
  elif [ "$cs" -gt 0 ];                     then res=HEARD
  else res=NO_DECODE; fi
  report "PACTOR${TRY_TAG:+/$TRY_TAG}" "$call" "$centre" "$res" \
      "$cs CS decoded in ${cy:-?} cycles" "$tag.log"
  settle
}

vara_try() {
  local call="$1" centre="$2" tag
  tag="$OUT/vara-$call-$centre"          # see pactor_try: bash 3.2 and `local`
  shift 2
  # THE DAEMON NEVER HOLDS THE KEY, so it is started with ptt_type=None and
  # kestrel drives $PTT_PORT itself — the shape onair.sh's vara and vara-mail
  # verbs hold. The daemon held RTS on $PTT_PORT here until 2026-08-13, when
  # that night's wire records said why it may not: rigctld answered every
  # command until the first key-down and then answered nothing, twelve unkeys
  # out of twelve, while the line itself read LOW instantly, eight of eight.
  # Unattended, that is a pass of ~8.4 s carriers nobody is there to hear.
  # Reuse a daemon that is already up rather than paying the CAT port's reopen.
  claim_daemon_start "$tag.rigctld" || return
  # kestrel senses in the last moment before key-down and that sense stands, but
  # it has no way to wait: a budget is spent out here first, and with none there
  # is nothing to add and the occupancy gate would only be a second opinion. What
  # kestrel does not do at either end is read a level, so that is asked on both
  # paths -- and on the channel the arm will key, which is why the dial is set
  # here where the sense would otherwise have set it.
  case "${TRY_WAIT:-0}" in
    ''|0|0.0) tune "$((centre - 1500))"
              gate_rx_unattended VARA "$call" "$centre" || return 0 ;;
    *) gate_unattended VARA "$call" "$centre" vara 2300 || return 0 ;;
  esac
  mkdir -p "$tag"
  claim_run timeout -k 5 300 "${HFMODEM_PYTHON:-$VENV/python}" "$REPO/tools/kestrel_connect.py" \
      --gateway "$call" --mycall "$MYCALL" --bw 2300 \
      --tx-device "$AUDIO" --rx-device "$AUDIO" \
      --rigctld 127.0.0.1:4532 --freq "$(printf '%d.%03d' "$(((centre - 1500) / 1000))" "$(((centre - 1500) % 1000))")" \
      --arm --expect-model "$RIG_EXPECT_MODEL" --ptt-device "$PTT_PORT" --line-ptt \
      --tx-tail 0.05 --cr-interval 8.0 \
      --timeout 95 --record "$tag" "$@" \
      --gateways "${HFMODEM_DATA_DIR:-$REPO/operator-data}/cache/winlink-vara-gateways.csv" >"$tag.log" 2>&1 || true
  local br res detail
  br=$(grep -c "bracketed burst is not the connected-ack" "$tag.log" 2>/dev/null | head -1 || true)
  br=${br:-0}
  deaf_or_die "$tag.log"
  detail="$br bursts"
  if   grep -q "^CONNECTED to"        "$tag.log"; then res=CONNECTED
  elif grep -q "tx link-setup"        "$tag.log"; then res=LINKSETUP
  elif [ "$br" -gt 0 ];                           then res=BURSTS
  # kestrel does its own sensing in the last moment before key-down, so a busy
  # channel comes back as its refusal rather than as this pass's own gate -- and
  # the line carries the reading it refused on. A burst count belongs to an
  # attempt that keyed; this one never did, and `SKIPPED_BUSY  0 bursts` read as
  # a gateway that answered nothing rather than as a channel we stayed off.
  elif grep -q "channel in use"       "$tag.log"; then
    res=SKIPPED_BUSY
    detail="$(sed -n 's/.*channel sense: //p' "$tag.log" | tail -1)"
    detail="${detail:-channel in use}"
  # A BW500-only channel answers a different waveform than this tool speaks, so
  # kestrel refuses it. Nothing was keyed and it is not a silent gateway.
  elif grep -q "cannot answer"        "$tag.log"; then
    res=CANNOT_ANSWER
    detail="$(grep -m1 "cannot answer" "$tag.log" | sed 's/^ *//')"
  else
    res=NO_DECODE
    detail="no burst decoded on this channel"
  fi
  report VARA "$call" "$centre" "$res" "$detail" "$tag.log"
  settle
}

ardop_try() {
  local call="$1" centre="$2" bw="${3:-500}" tag
  tag="$OUT/ardop-$call-$centre"
  shift 3
  # besra drives $PTT_PORT itself and rigctld is left to tune, which it does
  # before there is any RF about — the shape onair.sh's ardop-mail verb holds.
  claim_daemon_start "$tag.rigctld" || return
  gate_unattended ARDOP "$call" "$centre" ardop "$bw" || return 0
  mkdir -p "$tag"
  # One process owns the radio AND the B2F session: no host client, and the
  # whole exchange is bounded by --mail-timeout inside and `timeout` outside.
  claim_run timeout -k 5 420 "${HFMODEM_PYTHON:-$VENV/python}" -m hfmodem.besra.host.run_server \
      --radio ft891 --rigctld 127.0.0.1:4532 \
      --audio-in "$AUDIO" --audio-out "$AUDIO" \
      --channel "$centre" --bandwidth "$bw" \
      --ptt-device "$PTT_PORT" --line-ptt \
      --mycall "$MYCALL" --call "$call" --transmit \
      --mail-fetch --mail-timeout 300 --record "$tag" "$@" >"$tag.log" 2>&1 || true
  local rx res
  rx=$(grep -c " RX " "$tag.log" 2>/dev/null | head -1 || true)
  rx=${rx:-0}
  if   grep -q "^mail: received"     "$tag.log"; then res=MAIL
  elif grep -q "Unknown client type" "$tag.log"; then res=SID_REFUSED
  elif grep -q "ARQ CONNECTION ESTABLISHED" "$tag.log"; then res=CONNECTED
  elif [ "$rx" -gt 0 ];                              then res=HEARD
  else res=NO_DECODE; fi
  report ARDOP "$call" "$centre" "$res" "$rx frames decoded" "$tag.log"
  settle
}
