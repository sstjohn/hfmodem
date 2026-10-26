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
  claim_worker "${HFMODEM_PYTHON:-$VENV/python}" - "$@" <<PY || rc=$?
import json, os, sys, time, wave
from hfmodem.core import claim
import atexit
admission = claim.inherited(9, launch=True)
def admission_failed():
    if admission.fd is not None:
        admission.failed()
        print("CLAIM NOT ADMITTED capture open failed", flush=True)
atexit.register(admission_failed)
from pathlib import Path

import numpy as np, sounddevice as sd
from hfmodem.core.busy import THRESHOLD_DB, POUNCE_WINDOW_S, WINDOW_S, dominant, scores
from hfmodem.core.devices import find_device
from hfmodem.core.occupied import occupied_hz
from hfmodem.shrike.rxfront import call_b_frames

FS = 48000
FREE_SIGNAL = ("FreeSignalNormal", "FreeSignalEncrypted")
# The station this arm is about to call, where the verb knows one. Only used to
# say so when a Robust Call in the window is addressed to it.
DXCALL = os.environ.get("DXCALL", "").upper()
emission = " ".join(a for a in sys.argv[1:3] if a)
band = occupied_hz(sys.argv[1], sys.argv[2])
# No budget at all is the unattended rule; 0 is an operator who will not wait.
budget = float(sys.argv[3]) if len(sys.argv) > 3 else None
try:
    dev = find_device(os.environ.get("AUDIO_IN") or "$AUDIO", "in", required=True)
except SystemExit as exc:
    # find_device refuses an absent or an ambiguous name, and spends 1 on both.
    # 1 is BUSY to the caller, which would skip this attempt and make the next
    # one just as blind. Deaf is 2.
    print(f"  {exc}")
    raise SystemExit(2) from None


# WHO IS ON THE CHANNEL, WHERE THE CHANNEL SAYS SO IN WORDS. The busy scores
# below read energy and shape and can name nobody; the branch-B frame carries an
# eight-character ident behind a CRC-16, so a window holding one answers a
# question no margin can.
#
# NEVER ON ONE BURST. The Free Signal reference tape carries a garbage ident that
# closes the CRC among five good ones, so a single valid frame is not an
# identification -- two agreeing on the ident are. Each train in the window is the
# last burst's reading, how many there were, and the epoch it arrived at.
def call_b_trains(a, ended):
    seen = {}
    for t, cn in call_b_frames(a):
        seen.setdefault(cn.callsign, []).append((ended - len(a) / FS + t, cn))
    out = {}
    # Longest train last, so the ident with the most bursts is the one each kind
    # answers for.
    for bursts in sorted(seen.values(), key=len):
        if len(bursts) >= 2:
            at, cn = bursts[-1]
            out["free" if cn.variant in FREE_SIGNAL else "robust"] = (cn, len(bursts), at)
    return out


# WHAT WAS HEARD, AND NOT ONLY HOW LOUD. The margin alone cannot be re-read: on
# 2026-08-18 four calls were refused at +0.1, +1.4, +9.1 and +12.0 dB and the
# record cannot say whether any of them was an occupant lying across our emission
# or a carrier sitting beside it -- which is the difference between a channel to
# wait for and one to work, and the only refusal a narrower judged band could
# ever waive.
def window(stream, secs):
    a, over = stream.read(int(secs * FS))
    ended = time.time()
    if over:
        # A dropped fragment is an occupant this could have missed, so it is said
        # out loud rather than folded into the margin.
        print("  !! input overflowed -- this window is missing samples", flush=True)
    a = np.asarray(a, dtype=float).ravel()
    s = scores(a, FS, band)
    # DECIDED BEFORE THE SCORES ARE CONSULTED, because a Free Signal is not an
    # occupancy question: it is a host with no link in progress inviting a client
    # to call. Read as energy it is an occupant, and waiting it out waits for the
    # one thing on the channel that wants to be called.
    trains = call_b_trains(a, ended)
    if "free" in trains:
        host_free(a, s, *trains["free"])
    robust = trains.get("robust")
    if s is None:
        return True, 0.0, "no verdict, the input was not live enough to judge", a, s
    name, margin = dominant(s)
    # A carrier's frequency, against the judged band printed above it: a feature
    # at 1500 Hz sits on our own tones and one at 2600 is at the far edge of what
    # only an upgraded link would ever reach.
    if name == "tone":
        name = f"tone at {s['tone_hz']:.0f} Hz"
    if robust is not None:
        cn, n, _at = robust
        # NAMING THE OCCUPANT, WHICH WAIVES NOTHING (core/busy.py:88-100): a
        # client calling somebody else is still a client on this channel, and a
        # refusal is not argued out of by knowing whose call it was. It is said
        # because the operator's next move differs when the station being called
        # is the one this arm was about to call -- that caller is about to link,
        # and the wait is worth spending.
        mine = " -- the station this arm is calling" if cn.callsign == DXCALL else ""
        name = f"occupied: {n} robust calls to {cn.callsign} on $dial{mine}"
        return True, margin, name, a, s
    return margin >= 0.0, margin, name, a, s


# A REFUSAL THAT KEEPS NO AUDIO CANNOT BE SETTLED AFTERWARDS, and until now none
# of them did. This gate reads a live stream, prints one margin and drops the
# samples; every recorder the other verbs open starts after it has passed. So on
# 2026-08-19 eight refusals from +0.7 to +16.3 dB stood against an operator who
# heard nothing on all eight, and not one of them could be replayed -- including
# the +16.3 that landed on the deciding window between two clear ones, which is
# the whole of what would say whether it was an occupant or the estimator.
#
# The window that ENDS the sense is the one the verdict rests on, so it is the
# one kept: one file per refusal rather than one per window, or a 900 s wait
# would leave 86 MB behind it. A verdict of clear keeps nothing, because the
# session recording that follows it already holds those minutes.
def keep(audio, s, verdict, extra=None):
    d = Path("$REPO/logs/sense")
    d.mkdir(parents=True, exist_ok=True)
    stem = d / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-sense-$dial")
    with wave.open(str(stem) + ".wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    stem.with_suffix(".json").write_text(json.dumps(dict(
        dial=$dial, emission=emission, band=list(band), verdict=verdict,
        thresholds=THRESHOLD_DB, scores=s, samplerate=FS,
        **(extra or {})), indent=2) + "\n")
    print(f"  the window this rests on is kept: {stem}.wav")


def host_free(audio, s, cn, n, at):
    """A host announced itself free: say which, keep the window, and spend 5.

    KEPT LIKE A REFUSAL, for the same reason: a "host free" claim that leaves no
    audio behind cannot be settled afterwards, and this one ends with the station
    transmitting. The ident and the last burst's epoch also go to the marker file,
    which is where the launcher picks them up to hand to the modem.
    """
    print(f"  {cn.report()}  ({n} bursts)", flush=True)
    print(f"  HOST FREE -- {cn.callsign} has no link in progress on $dial", flush=True)
    keep(audio, s, f"host {cn.callsign} free", dict(
        ident=cn.callsign, kind=cn.variant, bursts=n, last_burst_epoch=round(at, 3)))
    free = Path("$REPO/logs/sense/host-free")
    free.parent.mkdir(parents=True, exist_ok=True)
    free.write_text(f"{cn.callsign} {cn.variant} {at:.3f}\n")
    raise SystemExit(5)


def refuse(margin, what, s, audio, line, code=1):
    """Say what was heard on all three scores, keep it, and spend 'code'."""
    print(line, flush=True)
    if s is not None:
        print(f"  burst {s['burst']:.2f}/{THRESHOLD_DB['burst']}, "
              f"shape {s['shape']:.2f}/{THRESHOLD_DB['shape']}, "
              f"tone {s['tone']:.2f}/{THRESHOLD_DB['tone']} at "
              f"{s['tone_hz']:.0f} Hz over {s['live_s']:.1f} s of live audio")
    keep(audio, s, f"{margin:+.1f} dB {what}")
    raise SystemExit(code)


print(f"  judging {band[0]:.0f}-{band[1]:.0f} Hz -- what {emission} will occupy")
with sd.InputStream(samplerate=FS, channels=1, device=dev) as stream:
    admission.opened()
    if budget is None:
        for n in (1, 2):
            busy, margin, what, a, s = window(stream, WINDOW_S)
            line = (f"  channel sense {n}/2: {margin:+.1f} dB {what} -> "
                    f"{'OCCUPIED' if busy else 'clear'}")
            if busy:
                refuse(margin, what, s, a, line)
            print(line)
        raise SystemExit(0)
    # A pounce waits in short windows, three of them clear back to back; with no
    # budget there is nothing to wait for and the one window is the long one.
    secs = POUNCE_WINDOW_S if budget else WINDOW_S
    clear = 0
    start = time.monotonic()
    # A DISAGREEING CONFIRMATION IS THE OCCUPANT COMING BACK, NOT A VERDICT. Two
    # clear windows and one occupied is what an intermittently busy channel looks
    # like, and ending the run there spends a 300 s budget in 24: three arms of
    # POUNCE_WAIT=300 came back at 24, 72 and 40 s that way, two of them left
    # with 228 s and 276 s of authorised waiting on channels that had been asked
    # for. So the wait resumes, and only the budget ends it.
    while True:
        while clear < 2 and time.monotonic() - start < budget:
            busy, margin, what, a, s = window(stream, secs)
            clear = 0 if busy else clear + 1
            print(f"  {time.monotonic() - start:6.0f}s  {margin:+6.1f} dB  "
                  f"{'busy' if busy else 'CLEAR ' + str(clear)}  {what}", flush=True)
        if budget and clear < 2:
            refuse(margin, what, s, a, f"  still busy after "
                   f"{time.monotonic() - start:.0f}s of the {budget:g}s budget", code=3)
        if budget:
            print("  channel is free -- one more window, and it is the one that decides")
        busy, margin, what, a, s = window(stream, secs)
        line = (f"  channel sense: {margin:+.1f} dB {what} -> "
                f"{'OCCUPIED' if busy else 'clear'}")
        if not busy:
            print(line)
            raise SystemExit(0)
        spent = time.monotonic() - start
        # HOW MUCH OF THE BUDGET WENT, in the refusal itself: a refusal at 24 s
        # of 300 and one at 300 are different answers, and without the share
        # spent they read the same.
        tail = "" if not budget else (
            f"\n  the channel was taken again on the window that decides, "
            f"{spent:.0f}s of the {budget:g}s budget spent")
        if spent >= budget:
            refuse(margin, what, s, a, line + tail)
        print(line + tail + " -- back to waiting", flush=True)
        clear = 0
PY
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
  claim_worker "${HFMODEM_PYTHON:-$VENV/python}" - <<PY
import os
from hfmodem.core import claim
import atexit
admission = claim.inherited(9, launch=True)
def admission_failed():
    if admission.fd is not None:
        admission.failed()
        print("CLAIM NOT ADMITTED capture open failed", flush=True)
atexit.register(admission_failed)
import numpy as np, sounddevice as sd
from hfmodem.core.devices import find_device
from hfmodem.core.levels import rx_verdict

try:
    dev = find_device(os.environ.get("AUDIO_IN") or "$AUDIO", "in", required=True)
except SystemExit as exc:
    # An absent or an ambiguous name spends 1, which here is the refusal a caller
    # may waive. A station with no capture device at all is a 2.
    print(f"  {exc}")
    raise SystemExit(2) from None
with sd.InputStream(samplerate=48000, channels=1, device=dev) as stream:
    admission.opened()
    a, overflow = stream.read(int(2.0 * 48000))
state, line = rx_verdict(a, 48000)
print("  " + line, flush=True)
raise SystemExit({"live": 0, "silent": 2}.get(state, 1))
PY
}

prove_down() {
  claim_native "${HFMODEM_PYTHON:-$VENV/python}" - "$PTT_PORT" <<'PROVE_DOWN'
import sys
from hfmodem.core import claim
claim.verify_cleanup(8)
claim.inherited(9)
try:
    claim.line_proof(claim.line_factory(sys.argv[1]))
except Exception as exc:
    print(f'NOT CONFIRMED: {exc}', flush=True)
    raise SystemExit(1)
PROVE_DOWN
}

down_exit() {
  local rc="$1"
  trap - EXIT
  trap '' INT TERM HUP
  if [ "$rc" -ne 0 ]; then printf 'NOT CONFIRMED\n'; fi
  claim_close || rc=1
  if [ "$CLAIM_MUTEX" = 1 ]; then
    claim_checked 'CLAIM MUTEX RELEASED' mutex_release 8 7>&- 9>&- || rc=1
    exec 8>&-
  fi
  exit "$rc"
}

station_down() {
  local rc=0 configuration backend host port dial line cat_result proof
  claim_paths || return
  if [ -n "${RIG_CLAIMED:-}" ]; then
    claim_checked 'CLAIM INHERITED' inherited 9 7>&- 8>&- || return
    CLAIM_ACTIVE=1
  else
    exec 9<>"$CLAIM"
  fi
  trap 'down_exit $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  exec 8<>"$MUTEX"
  if claim_checked 'CLAIM MUTEX' mutex 8 7>&- 9>&-; then
    CLAIM_MUTEX=1
  else
    exec 8>&- 9>&-
    CLAIM_ACTIVE=0
    return 2
  fi
  claim_processes down || return $?
  configuration=$(claim_processes settings) || return
  { IFS= read -r backend; IFS= read -r host; IFS= read -r port; } <<< "$configuration"
  if [ "$CLAIM_ACTIVE" = 0 ]; then
    claim_checked 'CLAIM ACQUIRED' acquire 9 --kind cleanup --operation down --cleanup 8 \
        --cat "$backend" --cat-port "$CAT_PORT" --ptt-port "$PTT_PORT" \
        --capture "$AUDIO_IN" --playback "$AUDIO_OUT" 7>&- || return
    CLAIM_ACTIVE=1
    CLAIM_OWNED=1
  fi
  proof=$(mktemp) || return
  prove_down >"$proof" || rc=$?
  cat "$proof"
  line=$(sed -n 's/^PTT LINE //p' "$proof")
  rm -f "$proof"
  [ "$rc" -eq 0 ] || return "$rc"
  case "$line" in low|none) ;; *) return 1 ;; esac
  if [ "$backend" = none ]; then
    cat_result=none
  else
    rc=0
    if [ "$backend" = rigctld ]; then
      dial=$(claim_native rigctl -P NONE -m 2 -r "$host:$port" f) || rc=$?
    else
      dial=$(claim_native rigctl -P NONE -m "$MODEL" -r "$CAT_PORT" -s "$BAUD" f) || rc=$?
    fi
    case "$dial" in ''|*[!0-9]*) rc=1 ;; esac
    if [ "$rc" -ne 0 ]; then printf 'CAT failed\n'; return 1; fi
    cat_result=ok
  fi
  printf 'CAT %s\n' "$cat_result"
  claim_processes evidence "$line" "$cat_result" || return
  printf 'CLEANUP COMPLETE\n'
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
