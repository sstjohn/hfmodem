# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""THE LIVE VIEW: a contact on the terminal, the telemetry in the transcript.

An arm writes one stream and two things read it. The transcripts -- the `tee`
inside `tools/onair.sh --log`, and the `arm.log` `arm.fly` writes beside the tap
-- are what every sweep, `tools/rehear` and the milestone reader are written
against, and they keep every line. The terminal is a person watching a contact
at 1.25 s a cycle, and six lines of arithmetic per depression of the PTT is what
buries it.

So this filters a terminal and never a file. Both writers keep the whole stream
and run `View` only over the copy going to a screen, which is why the rule lives
here rather than at the several hundred places that print. The launcher's own
stage asks `-t 1` first, because the console runs it down a pipe and writes that
pipe to the `arm.log` every analysis reads.

THE CUT IS MEANING AGAINST TELEMETRY, not exception against routine. Four things
belong on a screen:

    operational   what we keyed and what came back, as it happens
    explanatory   why the arm did that -- why it stayed on the air, why it
                  would not place a changeover
    failures      a deadline missed, a frame skipped, a refusal, a teardown
    unexplained   a burst nobody can attribute, an occupied answer slot: the
                  observations that turn into findings later

What is left is telemetry: callback tables, sample indices, the per-burst
arithmetic that sits inside a tolerance the arm already judged it against, and
the measurements that only mean anything in aggregate. Those keep their place in
the transcript, where the sweeps read them.

A REPEATED EXPLANATION IS STATED ONCE AND THEN COUNTED. The reasoning earns its
line; the sixtieth copy of it does not. `PTT LEAD SHORT` fired on 60 of 60
bursts of one session and is a property of this rig, not of that contact. So a
tagged line prints the first time its wording appears, again on a widening
ladder as it recurs, and once more in a tally at the end. Change the wording --
go off the air, come back on -- and it is a new explanation and prints again.

A STATE TRANSITION IS NOT A REASON THAT REPEATS. The grid reverses ten times in a
session and the operator watches every one of them: which way the turn went, and
when, is the contact itself and not a count of it. Those never collapse.

WHAT IS NOT DONE HERE. The arm prints an `off-grid` figure every cycle and it is
large as often as it is small, because a mode change and a hushed cycle both
move the instant it is measured against. A threshold invented out here would
fire on a third of all cycles or on none. The arm already turns that same
measurement into a verdict against the budget it alone knows -- SLOT n IS GONE
when the overrun cost the carrier, SLOT n IS KEPT when the reader forgave it --
and those are what reach the screen. A tolerance belongs beside the measurement,
not in a filter downstream of it.

A P4 LINK IS READ BACK AS OVERS. Its driver prints a decision, a read, a cycle
cost and a detail row for every 1.25 s, and each is a timing the transcript
needs and the operator does not. `P4Link` takes those lines and says, per over,
what the peer sent and what we keyed: two lines with a stream time on each,
the decoded bytes in full, the counters that stand between the link and its
next state, and a line of its own for each state that changed.

`ONAIR_VIEW=full` shows every line, for a session spent watching the timing
itself rather than the contact.
"""
from __future__ import annotations

import ast
from collections import Counter
import json
import os
import re
import sys

#: Measurement with no meaning in one cycle. The transcript keeps all of it.
TELEMETRY = re.compile(
    r"\[p\d timing\]"                                   # sixteen callback tuples
    r"|\[ack\] P\d (?:pulse centers|key-dataend)"
    r"|\[p\d\] follow RX "
    r"|\[grid\] (?:hold \d+ )?slot \d+"                 # boundary, capture, off-grid
    r"|\[grid\] RF started "                            # per-burst key arithmetic
    r"|\[grid\] d [\d.]+ ms @ sample "
    r"|\[grid\] the peer's raster re-read "
    r"|\[grid\] P\d receive position from CRC-valid "
    r"|\[predict\] .* -- clear$"                        # the prediction that held
    r"|^resolved args: "
    # A VARA over costs four lines before the over itself: the keying line up and
    # down, the callback records behind them, and the post-transmit cursor where
    # it found its interval. The two `post-TX level check:` wordings are not here
    # -- a check that found no interval at all means the receiver never muted and
    # our own emission is in the capture, which is the anomaly, not the reading.
    r"|^PTT (?:ON|OFF) -> line"
    r"|^timing event="
    r"|^tx_event "
    r"|post-TX level threshold from \+"
    # Two facts about the machine, said where the loss figure is so that a
    # regression has both on one row. Neither is ever a verdict  [core.rates].
    r"|^\s*host: \d+ s since boot"
    # The clock line carries its own: a shortfall past one blocksize says THE
    # STREAM IS SPLICED here and nowhere else, and xruns count on this line too.
    # Only the all-zero, unspliced reading is a measurement.
    r"|capture clock: \d+ samples \([\d.]+ s\), 0 xruns, 0 TX underruns,"
    r".* -- and no loss we can see"
)

#: A failure the arm judged against a budget only it knows, printed inside a line
#: the rest of which is telemetry. It comes back out.
FAILURES = re.compile(r"PTT LEAD SHORT|SLOT \d+ IS (?:GONE|KEPT)")

#: A state transition prints every time, because the count of them is not the
#: point: which way the turn went, and when, is the contact. A session reverses
#: the grid ten times and the operator watches every one.
TRANSITIONS = re.compile(
    r"GRID REVERSED"
    r"|TURNAROUND ACQUIRED"
    r"|receive window released"
    r"|keying PACTOR-\d at"
)

#: A reason, or a condition the arm marked with `!!`. Both repeat, and neither is
#: worth sixty lines: stated once per wording, then counted. `!! RX 4% SATURATED`
#: fired 102 times in one mail arm, and the operator lowers the gain once.
#: `[host]` and `[status]` are not here -- they are the contact itself, and every
#: one of them prints.
#:
#: VARA has no tags, so its recurring explanations are named by their wording. A
#: responder-idle train ran fifteen deep in one fetch and every copy said the same
#: thing -- the peer is still in its window -- with a different two-decimal offset
#: on it; the continue-class per-over response says which burst it chose, and why,
#: on every intermediate over of a delivery. The reason is worth a line and the
#: train is worth a count.
#:
#: Two neighbours of theirs are deliberately absent. `tx per-over response --
#: 11-symbol control burst` ends a delivery and `answered the last over` moves the
#: turn: those are transitions, and so is every keying of ours. They print.
RECURRING = re.compile(
    r"^\s*(?:\[(?:grid|ack|predict|clear|entry|timing|p\d)\]|!!)"
    r"|^\s*rx session-responder(?:-over)?-idle — "
    r"|^\s*session-responder-idle named inside "
    r"|^\s*tx per-over response — (?:generated|captured) "
    r"|^\s*-- channel \S+ .*; no stations identified yet --"
    r"|^\s*\S+ (?:is still keying|has been keying for) ")

#: A GAUGE, WHOSE VALUE IS THE CONTACT. The reference columns a wideband over
#: decoded on are how fast this link is running: 24 of 24 falling to 21 is
#: interference arriving and 21 back to 24 is it leaving, and an operator watching
#: a link wants both. A run at one value is not a third event, so the change
#: prints -- the arm's own line, unaltered -- and the run under it is counted like
#: any repeated reason. The value stays in the count's wording, because which
#: value it sat at is the whole of what the count says.
#:
#: Only the plain base-record reading is a gauge. `at record N`, `(N in this
#: window)`, a body we keyed and a burst that will not decode are each the decoder
#: with something else to say, and each prints every time.
TRAJECTORY = re.compile(
    r"rx wideband frame: (\d+)/\d+ reference columns, CRC clean, not a connect"
    r" — a DATA over of this session$")

#: Where a recurring explanation says so again. Often enough that a fault still
#: running is visible within a few cycles of starting, sparse enough that the
#: hundredth copy costs two lines rather than a screen.
LADDER = frozenset((5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000))

_NUMBER = re.compile(r"\d+")


def diagnostic(line: str) -> str:
    fault = FAILURES.search(line)
    return (line[fault.start():] if fault else line).strip()


def wording(line: str) -> str:
    """Group repeated diagnostics without their changing numbers."""
    return _NUMBER.sub("#", diagnostic(line))


def counted(line: str) -> str:
    """The key a repeat is counted under, which for a gauge keeps the value that
    moved: a run at 21 of 24 columns and a run at 24 say opposite things, and a
    count that had stripped the number would say neither.
    """
    moved = TRAJECTORY.search(line)
    return wording(line).replace("#", moved.group(1), 1) if moved else wording(line)


P4_BEGINS = re.compile(r"\[p4 (?:probe|link|experiment)\]|^\s*TX\[\d+\] P4 ")
P4_OBS = re.compile(
    r"\[p4 link\] (\w+) (\S+) (new|enriched|duplicate|confirmed|contradicted|expired|extended) "
    r"at ([\d.]+) (provisional|full) .*cfo=([-+]?[\d.]+)Hz(?: status=0x([0-9a-f]{2}) CRC=valid)?")
P4_FRAME = re.compile(r"\[p4 frame\] (\w+) at ([\d.]+) status=0x([0-9a-f]{2}) data=([0-9a-f]*)")
P4_DECISION = re.compile(
    r"\[p4 decision\] (\w+) peer=([\d.]+) expect=(\w+) key=(\S+) reason=(.+?); margin=([-\d.]+)ms")
P4_OVER = re.compile(r"\[p4 over\] at [\d.]+ (.*)")
P4_SILENT = re.compile(r"\[p4 link\] no live packet at ([\d.]+); silent (\d+/\d+)")
P4_UNREAD = re.compile(r"\[p4 link\] BREAKIN_UNREAD (\d+) ")
P4_DETAIL = re.compile(r"\[p4 detail\] (\S+) at ([\d.]+)(.*)")
P4_TX = re.compile(r"^\s*TX\[\d+\] P4 (?:LINK )?(.+?)\s+\([\d.]+s\)\s+--")
P4_KEYED = re.compile(r"\[p4 link\] Keyed (\w+) status=0x([0-9a-f]{2}) payload=(b(['\"]).*\4) at ([\d.]+)")
P4_REFUSED = re.compile(r"\[p4 link\] Refused (.+?): (\w+) at ([\d.]+)")
P4_DOWN = re.compile(r"\[p4 link\] LINK DOWN: (.*)")
P4_LOSS = re.compile(r"CAPTURE LOSS: the input lost (\d+) samples \((\d+) ms\) by ([\d.]+) in the P4 link")
P4_LISTENED = re.compile(r"\[p4 probe\] listened to ([\d.]+); (entry ceiling .*)")
#: A cycle's lines, flushed together after its key. Its `[p4 over]` closes it.
P4_CYCLE = re.compile(r"\[p4 (?:link\] (?:\w+ \S+ \w+ at |no live packet|BREAKIN_UNREAD)"
                      r"|frame\]|decision\]|read\]|detail\]|cycle\])")
#: Timing and accounting the transcript keeps and the over already says.
P4_QUIET = re.compile(r"\[p4 (?:query|probe\] query schedule|probe\] (?:robust|chirp|reply) \d)"
                      r"|\[p4 link\] .* responds to key |^P4 link: ")
MAIL = re.compile(r"^(?:\S+ \S+Z )?mail: |^\s*\[mail\] ")
COMPOSED = re.compile(r"^mail: we (?:said|queued): (.*)")
#: A carrier that moves this far off the one the link was tracking gets a line.
CARRIER_JUMP_HZ = 6.0


def readable(data: bytes) -> str:
    return ''.join('␍' if b == 13 else '␊' if b == 10 else chr(b) if 32 <= b < 127
                   else f'\\x{b:02x}' for b in data)


def _fields(text: str) -> dict:
    return dict(re.findall(r"(\w+)=(\S+)", text))


class P4Link:
    """A P4 link, over by over. `feed` returns the screen's lines for a line, and
    whether the line itself is still left to the ordinary rules."""

    def __init__(self):
        self.active = self.ended = False
        self.cycle = self._fresh()
        self.ours = None
        self.held = []
        self.state = self.carrier = None
        self.run = (None, 0, '')
        self.our_run = (None, 0)
        self.answered = False
        self.decoded = self.keys = self.greeting = 0
        self.keyed_text = ''
        self.frames = {}

    @staticmethod
    def _fresh():
        return dict(heard=[], decision=None, over=None, silent=None, unread=None, details=[])

    def feed(self, line: str):
        if not self.active:
            if not P4_BEGINS.search(line):
                return [], True
            self.active = True
        if self.ended:
            return self._after(line)
        if P4_CYCLE.search(line):
            shown = self._render() if self._next_cycle(line) else []
            self._collect(line)
            return shown, False
        over = P4_OVER.search(line)
        if over:
            self.cycle['over'] = _fields(over.group(1))
            return self._render(), False
        if P4_QUIET.search(line):
            return [], False
        if line.startswith('tx_event '):
            if self.ours is not None:
                event = json.loads(line[len('tx_event '):])
                self.ours['at'] = event['audio_start_sample'] / event['sample_rate']
            return [], False
        loss = P4_LOSS.search(line)
        if loss:
            return [f"{float(loss.group(3)):8.2f} !! stream lost {loss.group(1)} samples "
                    f"({loss.group(2)} ms)\n"], False
        listened = P4_LISTENED.search(line)
        if listened:
            return [f"{float(listened.group(1)):8.2f} ← nothing "
                    f"(listening; {listened.group(2)})\n"], False
        shown = self._render() if self._open() else []
        tx = P4_TX.search(line)
        if tx:
            if self.ours is not None:
                shown += self._render(force=True)
            self.ours = dict(label=tx.group(1).strip())
            self.keys += 1
            return shown, False
        keyed = P4_KEYED.search(line)
        if keyed and self.ours is not None:
            payload = ast.literal_eval(keyed.group(3))
            self.ours.update(status=int(keyed.group(2), 16), payload=payload)
            self.ours.setdefault('at', float(keyed.group(5)))
            self.keyed_text += payload.decode('latin-1')
            return shown, False
        refused = P4_REFUSED.search(line)
        if refused:
            self.ours = dict(refused=refused.group(1), label=refused.group(2),
                             at=float(refused.group(3)))
            return shown, False
        if MAIL.search(line):
            if 'banner:' in line or 'remote SID' in line:
                self.greeting += 1
            self.held.append(line)
            return shown, False
        down = P4_DOWN.search(line)
        if down:
            shown += self._render(force=True)
            self.ended = True
            stuck = (f"; stuck at {self.run[2]} ×{self.run[1]}" if self.run[1] > 1 else
                     f"; stuck at our {self.our_run[0]} ×{self.our_run[1]}"
                     if self.our_run[1] > 1 and not re.match(r"[A-Z]\d\d$", self.our_run[0]) else '')
            shown.append(f"         == LINK DOWN: {down.group(1)} -- {self.decoded} frames decoded, "
                         f"{self.keys} keys sent, greeting lines read: {self.greeting}{stuck}\n")
            return shown, False
        return shown, True

    def close(self):
        """What an arm stopped mid-link still owes the screen."""
        if not self.active or self.ended:
            return []
        shown = self._render(force=True) if self._open() or self.ours else []
        shown, self.held = shown + self.held, []
        return shown

    def _after(self, line):
        if P4_QUIET.search(line) or P4_CYCLE.search(line):
            return [], False
        composed = COMPOSED.match(line)
        if composed:
            text = composed.group(1)
            sent = re.compile(r'\S+'.join(map(re.escape, text.split('########'))))
            if not sent.search(self.keyed_text):
                return [f"mail: composed, not sent: {text}\n"], False
        return [], True

    def _next_cycle(self, line):
        """A transcript without `[p4 over]` runs one cycle into the next."""
        decision = self.cycle['decision']
        if decision is None:
            return False
        if P4_DECISION.search(line):
            return True
        at = re.search(r"\[p4 (?:link|frame)\] .*?\bat ([\d.]+)", line)
        return bool(at) and float(at.group(1)) > decision['peer'] + 1.

    def _open(self):
        return any(self.cycle.values())

    def _collect(self, line):
        cycle = self.cycle
        obs = P4_OBS.search(line)
        if obs:
            kind, name, outcome, at, extent, cfo, status = obs.groups()
            heard = self._heard_at(float(at))
            heard['kinds'].add(kind)
            if extent == 'provisional' and outcome not in ('expired', 'contradicted'):
                heard['maybe'] = name if kind == 'word' else kind
            if outcome in ('expired', 'contradicted'):
                heard['expired'] = True
            elif extent == 'full':
                heard['cfo'] = float(cfo)
                if kind == 'word':
                    heard['word'] = name
                elif kind == 'short':
                    heard['short'] = name
            if status:
                heard['frame'] = kind
                heard['status'] = int(status, 16)
            return
        frame = P4_FRAME.search(line)
        if frame:
            heard = self._heard_at(float(frame.group(2)))
            heard['data'] = bytes.fromhex(frame.group(4))
            return
        decision = P4_DECISION.search(line)
        if decision:
            state, peer, expect, key, reason, margin = decision.groups()
            cycle['decision'] = dict(state=state, peer=float(peer), expect=expect, key=key,
                                     reason=reason, margin=float(margin))
            return
        silent = P4_SILENT.search(line)
        if silent:
            cycle['silent'] = (float(silent.group(1)), silent.group(2))
            return
        unread = P4_UNREAD.search(line)
        if unread:
            cycle['unread'] = unread.group(1)
            return
        detail = P4_DETAIL.search(line)
        if detail:
            event, at, rest = detail.groups()
            row = dict(_fields(rest), event=event, at=float(at))
            cycle['details'].append(row)
            if event == 'normal-header':
                self._heard_at(row['at'])['hdr'] = row['header']
            elif event == 'normal-decode' and row.get('crc_valid') == 'False':
                self._heard_at(row['at'])['crc_bad'] = True
            elif event == 'peer-control-no-progress':
                limit = f"/{row['limit']}" if 'limit' in row else ''
                self._heard_at(row['at'])['progress'] = f"no progress {row['copies']}{limit}"

    def _heard_at(self, at):
        for heard in self.cycle['heard']:
            if abs(heard['at'] - at) < .1:
                return heard
        heard = dict(at=at, kinds=set())
        self.cycle['heard'].append(heard)
        return heard

    def _render(self, force=False):
        cycle, ours = self.cycle, self.ours
        if not (force or self._open()):
            return []
        silent = cycle['silent']
        beside = next((h for h in cycle['heard'] if silent and abs(h['at'] - silent[0]) < .1
                       and h.get('status') is None), None)
        if beside:
            beside['silent'], silent = silent[1], None
        heard = [(h['at'], self._peer(h)) for h in sorted(cycle['heard'], key=lambda h: h['at'])]
        decision, over = cycle['decision'], cycle['over'] or {}
        onset = decision['peer'] if decision else None
        at_onset = next((h for h in cycle['heard'] if onset is not None and abs(h['at'] - onset) < .1
                         and (h.get('word') or h.get('short') or h.get('frame')
                              or h['kinds'] - {'word', 'short'})), None)
        tail = self._counters(over, keyed=ours is not None and 'refused' not in ours)
        if silent and not (at_onset and at_onset.get('status') is not None):
            at, silent = silent
            expect = f"expected {decision['expect']}, " if decision else ''
            heard.append((at, f"{at:8.2f} ← nothing ({expect}silent {silent})"))
        answered = ours is not None and not (decision and ours.get('at') is not None
                                             and ours['at'] < decision['peer'])
        if ours is not None:
            heard.append((ours.get('at') or float('inf'),
                          self._ours(ours, at_onset if answered else None, decision, cycle['details'],
                                     over.get('purpose'), over)))
        if not answered and decision and decision['key'] == 'none' and at_onset and decision['state'] != 'DOWN':
            heard.append((at_onset['at'] + .001, f"{'':8} → (no answer: {self._why(decision, cycle)})"))
        lines = [said for _, said in sorted(heard, key=lambda h: h[0]) if said]
        if lines and tail:
            lines[-1] = (lines[-1][:-1] + f", {tail})" if lines[-1].endswith(')')
                         else lines[-1] + f" ({tail})")
        state = over.get('state') or (decision and decision['state'])
        if state and state != self.state and state != 'DOWN':
            was = f" (was {self.state})" if self.state else ''
            lines.append(f"         == {state}{was}")
            self.state = state
        out = [line + '\n' for line in lines]
        if cycle['heard'] or decision:
            out, self.held = out + self.held, []
        self.cycle, self.ours = self._fresh(), None
        return out

    def _peer(self, heard):
        at = f"{heard['at']:8.2f} ← "
        lines = []
        cfo = heard.get('cfo')
        if cfo is not None and (self.carrier is None or abs(cfo - self.carrier) > CARRIER_JUMP_HZ):
            was = f"{self.carrier:+.1f} → " if self.carrier is not None else ''
            lines.append(f"{'':8} ~ peer carrier {was}{cfo:+.1f} Hz\n")
            self.carrier = cfo
        status = heard.get('status')
        if status is not None:
            word = f"{heard['word']} + " if heard.get('word') else ''
            hdr = f" hdr {heard['hdr']}" if heard.get('hdr') else ''
            head = f"{word}{heard['frame']} c{status & 3} 0x{status:02x}{hdr} CRC ok"
            data = heard.get('data')
            identity = (heard['frame'], status, data) if data is not None else None
            text = f' "{readable(data)}"' if data is not None else ''
            said = self._repeat(identity, f"{heard['frame']} c{status & 3}")
            if said:
                prefix = readable(data)
                text = f' "{prefix[:24]}…"' if len(prefix) > 24 else text
                said = f" ({said}, identical)"
            line = at + head + text + said
            self.frames[heard['at']] = status
        elif heard.get('word') or heard.get('short'):
            name = heard.get('word') or f"short {heard['short']}"
            said = self._repeat(('word', name.split('/')[0]), name.split('/')[0])
            extra = [s for s in (said, heard.get('progress')) if s]
            line = at + name + (f" ({', '.join(extra)})" if extra else '')
        elif heard['kinds'] - {'word', 'short'}:
            kind = sorted(heard['kinds'] - {'word', 'short'})[0]
            hdr = f" hdr {heard['hdr']}" if heard.get('hdr') else ''
            line = at + f"{kind}{hdr} ({'CRC bad' if heard.get('crc_bad') else 'unread'})"
        elif heard.get('maybe'):
            line = at + f"{heard['maybe']}? (unconfirmed)"
        else:
            return None
        if heard.get('silent'):
            line = (line[:-1] + f", silent {heard['silent']})" if line.endswith(')')
                    else line + f" (silent {heard['silent']})")
        if status is not None or heard.get('word') or heard.get('short'):
            self.decoded += 1
            if not self.answered:
                self.answered = True
                lines.append("         == probe answered\n")
        return ''.join(lines) + line

    def _repeat(self, identity, label):
        """Counted only while nothing else came from the peer between copies."""
        if identity is not None and identity == self.run[0]:
            self.run = (identity, self.run[1] + 1, label)
            return f"repeat ×{self.run[1]}"
        self.run = (identity, 1, label)
        return ''

    def _ours(self, ours, at_onset, decision, details, purpose=None, over=None):
        at = f"{ours['at']:8.2f} → " if ours.get('at') is not None else (
            f"{float(decision['key']):8.2f} → " if decision and decision['key'] != 'none'
            else f"{'':8} → ")
        if 'refused' in ours:
            return at.replace('→', '→ (no answer: refused,') + f" {ours['refused']})"
        label = ours['label']
        if 'status' in ours:
            status, payload = ours['status'], ours['payload']
            text = f' "{readable(payload)}"' if payload else ' (empty)'
            return at + f"{label.split()[0]} c{status & 3} 0x{status:02x}{text}"
        word = label.startswith('word ')
        label = (label[5:] if word else
                 re.sub(r"^short (\d+)$", r"short v\1",
                        re.sub(r"v=(\d+) hdr=(\w)", r"v\1/\2",
                               label.replace('PROBE ROBUST', 'probe robust')
                               .replace('PROBE CHIRP', 'probe chirp'))))
        self.our_run = (label, self.our_run[1] + 1 if self.our_run[0] == label else 1)
        acked = ' (repeat request)' if word and purpose == 'repeat-request' else ''
        if word and over and over.get('slot_accepted') == 'false':
            if over.get('held_status') is not None:
                target = f"c{int(over['held_status']) & 3}"
            elif over.get('held_control') is not None:
                target = f"v{over['held_control']}"
            else:
                target = None
            acked = (f" (ack held {target} of {float(over['held_at']):.2f}; current slot unread)"
                     if target else ' (bootstrap seed; current slot unread)')
        if word and not acked and at_onset:
            acked = (f" (ack c{at_onset['status'] & 3})" if at_onset.get('status') is not None
                     else f" (ack {at_onset.get('word') or at_onset.get('short')})"
                     if at_onset.get('word') or at_onset.get('short') else '')
        elif word and not acked:
            earlier = next((d['at'] for d in details if d['event'] in ('acked-repeat', 'normal-answer')), None)
            status = next((s for t, s in self.frames.items()
                           if earlier is not None and abs(t - earlier) < .1), None)
            if status is not None:
                acked = f" (ack c{status & 3} of {earlier:.2f})"
        return at + label + acked

    @staticmethod
    def _counters(over, keyed):
        tail = []
        if keyed and 'try' in over:
            tail.append(f"try {over['try']}")
        for name in ('grants', 'breakins', 'hold'):
            if name in over:
                tail.append(f"{name} {over[name]}")
        if over.get('unread', '0') != '0':
            tail.append(f"unread {over['unread']}")
        if 'link' in over:
            spent, ceiling = map(float, over['link'].rstrip('s').split('/'))
            if spent >= ceiling / 2:
                tail.append(f"link {spent:.0f}/{ceiling:g} s")
        return ', '.join(tail)

    @staticmethod
    def _why(decision, cycle):
        reason = decision['reason']
        late = f"decoded {-decision['margin']:.0f} ms late, " if decision['margin'] < 0 else ''
        if reason == 'answer deferred':
            due = next((float(d['due']) / 48000 for d in cycle['details']
                        if d['event'] == 'answer-deferred'), None)
            return late + (f"deferred to {due:.2f}" if due is not None else 'deferred')
        if reason == 'body unread at decision':
            return late + 'body unread' + (f", unread {cycle['unread']}" if cycle['unread'] else '')
        if reason == 'packet observed; no response selected':
            return 'no response selected'
        return reason


class View:
    """One arm's terminal. `feed` a line, take the lines to print; `close` at EOF."""

    def __init__(self):
        self.said = Counter()
        self.at = {}
        self.latest = {}
        self.p4 = P4Link()

    def feed(self, line: str):
        if os.environ.get("ONAIR_VIEW") == "full":
            return (line,)
        shown, leave = self.p4.feed(line)
        return (*shown, *(self._line(line) if leave else ()))

    def _line(self, line: str):
        if TRANSITIONS.search(line):
            return (line,)
        if TELEMETRY.search(line) and not FAILURES.search(line):
            return ()
        moved = TRAJECTORY.search(line)
        if not (moved or RECURRING.match(line)):
            return (line,)
        gauge, reason = wording(line), counted(line)
        self.latest[reason] = diagnostic(line)
        if moved and self.at.get(gauge) != reason:
            self.at[gauge] = reason
            self.said[reason] += 1
            return (line,)
        said = self.said[reason] = self.said[reason] + 1
        if said == 1:
            return (line,)
        return (f"    ... x{said}: {self.latest[reason]}\n",) if said in LADDER else ()

    def close(self):
        pending = self.p4.close()
        again = sorted(((n, w) for w, n in self.said.items() if n > 1), reverse=True)
        if not again:
            return tuple(pending)
        return (*pending, "  said once above; latest values below; the transcript holds every copy:\n",
                *(f"    x{n:<5} {self.latest[w]}\n" for n, w in again))


def main() -> int:
    """Filter a stream for the terminal. `tools/onair.sh` runs this after its tee.

    surrogateescape both ways, because the arm colours its banner and a decode
    may carry anything: the bytes that go out are the bytes that came in.
    """
    sys.stdin.reconfigure(errors="surrogateescape")
    sys.stdout.reconfigure(errors="surrogateescape", line_buffering=True)
    view, out = View(), sys.stdout
    # A closed terminal stops the writing and not the reading: `tee` is upstream,
    # and it loses the arm the moment this end stops draining it.
    for line in sys.stdin:
        for shown in view.feed(line):
            try:
                out.write(shown)
            except OSError:
                return 0
    for shown in view.close():
        try:
            out.write(shown)
        except OSError:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
