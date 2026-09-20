# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What a session is doing, in a dozen lines instead of three hundred.

A mail fetch prints a frame line every second or two, a PTT pair around every
one of them, and the peer's own words only at teardown. The operator — usually
an agent with a session to run as well as a log to watch — has to notice, live,
that the link came up, that the peer refused the login, that the same packet
number has gone out thirty times, or that we have been answering an idle peer
for seventy seconds. On 2026-08-16 a VARA link kept keying into an empty channel
for minutes after the peer left, and on 2026-08-18 an ARDOP link deadlocked at
the gateway's login prompt and transmitted for seventy seconds before SIGTERM
stopped it. Both were found the next day by re-reading the log.

Fed a session's output line by line, this says only what changed: state
transitions and alarms. Silence means nothing has happened. It parses text and
nothing else — no port, no card, no socket — so it cannot key.

IT MAY NOT CLAIM MORE THAN THE STREAM IT WAS HANDED ESTABLISHES. One stream is
one process's output, and a launcher that runs the modem as a child redirects
that child somewhere else entirely; a channel we decode nothing from is not a
channel nobody is on. So where this cannot see, it says it cannot see, and where
it measured something it reports the measurement and stops there. The alternative
is an instrument the operator has to spend the transmitter to check.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The `Z` and the missing level are the mail lines: `winlink.client` stamps
# those UTC so a session can be aligned against the tape, and everything else
# on this stream is the local clock with a level in front of it.
STAMPED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3}(Z?) "
                     r"(?:(\w+) )?(.*)$")
RX_FRAME = re.compile(r"^RX (\S+) sess=(0x[0-9a-f]+) ok=(True|False)"
                      r"( HEADER-ONLY)?(?: q=(\d+))?")
TX_FRAME = re.compile(r"^TX (\S+) [\d.]+s$")
DATA_FRAME = re.compile(r"^\d*[A-Z]+\.\d+\.")
CONNECTED = re.compile(r"^CONNECTED (\S+) @ (\d+) Hz")
PACTOR_TX = re.compile(r"^TX\[\d+\] (?:.*?pkt#(\d+))?")
PACTOR_RX = re.compile(r"^(?:HOLD )?RX (?!\(nothing decoded\))")
PACTOR_LINK = re.compile(r"^\*\* CONNECTED to (\S+)")
HOST_STATE = re.compile(r"^\[host\] state \w+ -> (\w+)")
BANNER = re.compile(r"^== (.+?)(?: ==)?$")
STAGE = re.compile(r"^mail: stage (.+?)(?: — (.*))?$")
MOVED = re.compile(r"^mail: (?:received|sent|wrote) |^mail: \d+ messages? ")
SENSE = re.compile(r"^channel sense(?: \d/\d)?: (.*)$")
PEER_ADDRESS = re.compile(r"(?:<->|->|originating: \S+ ->)\s*([A-Z0-9]+(?:-\d+)?)")
# lib/attempts.sh prints one column-formatted line per attempt it makes, and the
# verb that called it names the directory those attempts' own logs are in.
ATTEMPT = re.compile(r"^(PACTOR|VARA|ARDOP)(?:/(\S+))? +(\S+) +(\d+) +"
                     r"([A-Z_]+)(?: +(.*))?$")
CAPTURES = re.compile(r"^captures: (\S+)$")
COLUMNS = re.compile(r" {2,}")

# Verdicts of the gate rather than of an attempt: neither one started a child.
GATE = ("FORCED", "SKIPPED_BUSY")

# A stalled ARQ link goes on exchanging control frames, so a run of them carrying
# no data either way is what tells a deadlock from a healthy turnaround. Six is
# past any turnaround these protocols make legitimately.
IDLE_RUN = 6
TX_RUN = 6
PACKET_REPEAT = 10
REALARM = 12
KEYED_LIMIT_S = 30.0
BAD_FRAME_RUN = 3
BAD_FRAME_DETAIL = 5
BUSY_REFUSALS = 8

# Three `!!` lines are routine notices rather than anything the operator has to
# act on. Only `channel in use` is the launcher's; `shrike.onair` prints the
# other two and the launcher relays them.
ROUTINE_BANGS = ("TRANSMIT ARMED", "LATE TO THE KEY", "channel in use")


@dataclass(frozen=True)
class Event:
    at: str
    tag: str
    text: str
    alarm: bool = False

    def __str__(self) -> str:
        return f"{self.at:>8}  {'ALARM' if self.alarm else self.tag:<5}  {self.text}"


def _dur(a: datetime | None, b: datetime | None) -> str:
    if a is None or b is None:
        return ""
    s = int((b - a).total_seconds())
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class Narrator:
    """Fed one line at a time, returns the events that line caused."""

    def __init__(self) -> None:
        self.now: datetime | None = None
        self.lineno = 0
        self.target: str | None = None
        self.run: str | None = None
        self.session: str | None = None
        self.stage: str | None = None
        self.outcome: str | None = None
        self.last_peer: str | None = None
        self.link_up = False
        self.link_since: datetime | None = None
        self.keyings = 0
        self.keyed_since: datetime | None = None
        self.keyed_alarmed = False
        self.last_key: datetime | None = None
        self.rx_ok = 0
        self.rx_bad_ours = 0
        self.rx_bad_foreign = 0
        self.tx_frames = 0
        self.alarms: list[Event] = []
        self._sessions: Counter[str] = Counter()
        self._tx_run = 0
        self._tx_since: datetime | None = None
        self._idle_run = 0
        self._idle_since: datetime | None = None
        self._bad_run = 0
        self._packet: str | None = None
        self._packet_run = 0
        self._busy = 0
        self._said_rx_data = False
        self._said_tx_data = False
        self._unkeyed = False
        self._last_rx: datetime | None = None
        self.attempts: list[str] = []
        self.captures: str | None = None

    @property
    def at(self) -> str:
        return self.now.strftime("%H:%M:%S") if self.now else f"L{self.lineno}"

    @property
    def to_peer(self) -> str:
        """` to CALL`, or nothing at all where the stream has not named one."""
        return f" to {self.target}" if self.target else ""

    def _ev(self, out: list[Event], tag: str, text: str, alarm: bool = False) -> None:
        e = Event(self.at, tag, text, alarm)
        out.append(e)
        if alarm:
            self.alarms.append(e)

    def feed(self, raw: str) -> list[Event]:
        self.lineno += 1
        line = ANSI.sub("", raw).strip()
        level = ""
        if m := STAMPED.match(line):
            if not m.group(2):          # a UTC stamp is not this stream's clock
                self.now = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            level, line = m.group(3) or "", m.group(4)
        out: list[Event] = []
        if line:
            self._keyed_watchdog(out)
            self._dispatch(line, level, out)
        return out

    def close(self) -> list[Event]:
        out: list[Event] = []
        if not self.keyings and not self.tx_frames:
            self._ev(out, "tx", "no keying visible on this stream" + self._blind())
        elif self.keyed_since:
            self._ev(out, "tx", "stream ended with the transmitter keyed", True)
        if self.link_up:
            self._ev(out, "link", f"stream ended with the link{self.to_peer} up")
        return out

    def _blind(self) -> str:
        """What ran where this stream could not follow it.

        `never keyed -- this run transmitted nothing` was printed over the
        ack-placement A/B of 2026-08-19, whose two arms keyed seventy times
        between them: `ackab` transmits through lib/attempts.sh, which runs the
        modem as a child with its output redirected to a log of its own, so a
        launcher stream carries the attempt's verdict and not one line of what
        it did. An agent reading that would have retried a launcher that worked,
        and spent the transmitter disproving its own instrument. What this
        stream did not carry is not a thing that did not happen.
        """
        if not self.attempts:
            return ""
        n = len(self.attempts)
        return (f" — {n} attempt{'s' if n > 1 else ''} ran as a child process, "
                "whose own log this stream does not carry"
                + (f" (under {self.captures})" if self.captures else ""))

    def _keyed_watchdog(self, out: list[Event]) -> None:
        if self.keyed_since is None or self.keyed_alarmed or self.now is None:
            return
        if (self.now - self.keyed_since).total_seconds() > KEYED_LIMIT_S:
            self.keyed_alarmed = True
            self._ev(out, "tx", f"keyed for {_dur(self.keyed_since, self.now)} "
                                "with no unkey", True)

    def _dispatch(self, line: str, level: str, out: list[Event]) -> None:
        if m := RX_FRAME.match(line):
            return self._rx_frame(m, out)
        if m := TX_FRAME.match(line):
            self._outbound(out)
            if DATA_FRAME.match(m.group(1)):
                self._data_moved()
                if not self._said_tx_data:
                    self._said_tx_data = True
                    self._ev(out, "tx", f"sending data ({m.group(1).rsplit('.', 1)[0]})")
            return
        if m := PACTOR_TX.match(line):
            return self._packet_out(m.group(1), line.endswith("-- keying"), out)
        if line.startswith("PTT ON"):
            return self._key_down(out)
        if line.startswith("PTT OFF") or "handed back" in line:
            self.keyed_since = None
            if "NOT CONFIRMED" in line:
                self._ev(out, "tx", line, True)
            return
        if line.startswith("*** PTT"):
            return self._ev(out, "tx", line, True)
        if line.startswith("emergency unkey:"):
            if not self._unkeyed:
                self._unkeyed = True
                self._ev(out, "tx", line)
            return
        if line.startswith("mail:"):
            return self._mail(line, out)
        if PACTOR_RX.match(line) or line.startswith("rx "):
            self._inbound()
            if "DATA over" in line or "rx data" in line:
                self._data_moved()
            if "CONNECTED" in line:
                self._connected(self.target or "peer", out)
            return
        if line.startswith("tx "):
            return self._outbound(out)
        if "[host] rx data" in line:
            self._inbound()
            self._data_moved()
            return self._ev(out, "rx", line.split("[host] ", 1)[1])
        return self._plain(line, level, out)

    def _plain(self, line: str, level: str, out: list[Event]) -> None:
        if m := BANNER.match(line):
            if t := PEER_ADDRESS.search(m.group(1)):
                self.run, self.target = m.group(1), t.group(1)
                self._ev(out, "run", self.run)
            return
        if m := CONNECTED.match(line):
            return self._connected(m.group(1), out, f"@ {m.group(2)} Hz")
        if m := PACTOR_LINK.match(line):
            return self._connected(m.group(1), out)
        if m := HOST_STATE.match(line):
            if m.group(1) == "CONNECTED":
                return self._connected(self.target or "peer", out)
            if m.group(1) == "DISCONNECTED":
                return self._link_down(out)
        if m := ATTEMPT.match(line):
            return self._attempt(m, out)
        if m := CAPTURES.match(line):
            self.captures = m.group(1)
            return
        if line.startswith("verdict:") or line.startswith("PTT VERDICT:"):
            return self._ev(out, "run", line,
                            line.startswith("PTT VERDICT:") and "down" not in line)
        if line.startswith("originating: "):
            if t := PEER_ADDRESS.search(line):
                self.target = t.group(1)
            return
        if line == "link down" or line.startswith("END NOT RECEIVED"):
            self._link_down(out)
            if line.startswith("END NOT RECEIVED"):
                self._ev(out, "link", "ARQ session closed without END", True)
            return
        if line.startswith("!!"):
            if "channel in use" in line:
                return self._busy_channel(out)
            routine = any(w in line for w in ROUTINE_BANGS)
            return self._ev(out, "tx" if routine else "link", line, not routine)
        if "FAILED!" in line or line.startswith("FAULT "):
            return self._ev(out, "link", line, True)
        if line.startswith("--force"):
            return self._ev(out, "chan", line, True)
        if m := SENSE.match(line):
            return self._ev(out, "chan", m.group(1))
        if "unanswered" in line and "turn-request" in line:
            return self._ev(out, "tx", line)
        if level in ("ERROR", "CRITICAL"):
            self._ev(out, "log", line, True)

    def _attempt(self, m: re.Match[str], out: list[Event]) -> None:
        if m.group(5) not in GATE:
            self.attempts.append(m.group(0))
        self._ev(out, "run", COLUMNS.sub("  ", m.group(0)))

    def _busy_channel(self, out: list[Event]) -> None:
        # Whether anything is waiting is the caller's, not ours. `waiting` was
        # printed on refusals that ended the run, where nothing waited for
        # anything: what a refusal establishes is that we did not key.
        self._busy += 1
        if self._busy == BUSY_REFUSALS:
            self._ev(out, "chan", f"channel refused {self._busy} times, still not "
                                  "keyed — ask the operator whether it sounds clear",
                     True)
        elif self._busy == 1 or self._busy % 4 == 0:
            self._ev(out, "chan", f"channel in use, not keyed "
                                  f"(refusal {self._busy})")

    def _connected(self, peer: str, out: list[Event], detail: str = "") -> None:
        if self.link_up:
            return
        self.link_up = True
        self.link_since = self.now
        self.target = peer if peer != "peer" else self.target
        self._said_rx_data = self._said_tx_data = False
        self._ev(out, "link", " ".join(
            w for w in ("CONNECTED", self.target or "", detail) if w))

    def _link_down(self, out: list[Event]) -> None:
        if not self.link_up:
            return
        self.link_up = False
        held = _dur(self.link_since, self.now)
        self._ev(out, "link", f"link{self.to_peer} down"
                              + (f" after {held}" if held else ""))

    def _rx_frame(self, m: re.Match[str], out: list[Event]) -> None:
        name, sess, ok = m.group(1), m.group(2), m.group(3) == "True"
        header_only, q = bool(m.group(4)), m.group(5)
        if ok:
            self._sessions[sess] += 1
        if self.session is None and (name.startswith("ConAck")
                                     or self._sessions[sess] >= 4):
            self.session = sess
        if self.session is not None and sess != self.session:
            if not ok:
                self.rx_bad_foreign += 1
            return
        if not ok:
            self.rx_bad_ours += 1
            self._bad_run += 1
            why = "HEADER-ONLY" if header_only else "bad"
            if self._bad_run >= BAD_FRAME_RUN and self._bad_run % BAD_FRAME_RUN == 0:
                self._ev(out, "frame", f"{self._bad_run} {why} frames in a row on our "
                                       f"session ({self.rx_bad_ours} this run)", True)
            elif self.rx_bad_ours <= BAD_FRAME_DETAIL:
                self._ev(out, "frame", f"{why} {name} on our session"
                                       + (f" q={q}" if q else ""))
            elif self.rx_bad_ours % 10 == 0:
                self._ev(out, "frame", f"{self.rx_bad_ours} bad frames on our session "
                                       "so far")
            return
        self._bad_run = 0
        self._inbound()
        if DATA_FRAME.match(name):
            self._data_moved()
            if not self._said_rx_data:
                self._said_rx_data = True
                self._ev(out, "rx", f"peer sending data ({name.rsplit('.', 1)[0]})")
            return
        self._idle_run += 1
        self._idle_since = self._idle_since or self.now
        if self._idle_run >= IDLE_RUN and (
                self._idle_run - IDLE_RUN) % REALARM == 0:
            held = _dur(self._idle_since, self.now)
            with_peer = f" with {self.target}" if self.target else ""
            self._ev(out, "link", f"stalled: {self._idle_run} control exchanges"
                                  f"{with_peer}, no data either way"
                                  + (f" in {held}" if held else ""), True)

    def _packet_out(self, pkt: str | None, keying: bool, out: list[Event]) -> None:
        if keying:
            self._key_down(out)
        self._outbound(out)
        if pkt is None:
            self._packet, self._packet_run = None, 0
            return
        if pkt == self._packet:
            self._packet_run += 1
        else:
            self._packet, self._packet_run = pkt, 1
        if self._packet_run >= PACKET_REPEAT and (
                self._packet_run - PACKET_REPEAT) % PACKET_REPEAT == 0:
            self._ev(out, "tx", f"packet #{pkt} sent {self._packet_run} times with no "
                                "advance — the data phase is not moving", True)

    def _key_down(self, out: list[Event]) -> None:
        self.keyings += 1
        self.keyed_since = self.last_key = self.now
        self.keyed_alarmed = False
        if self.keyings == 1:
            self._ev(out, "tx", "first key-down")

    def _outbound(self, out: list[Event]) -> None:
        self.tx_frames += 1
        self._tx_run += 1
        self._tx_since = self._tx_since or self.now
        if self._tx_run >= TX_RUN and (self._tx_run - TX_RUN) % REALARM == 0:
            held = _dur(self._tx_since, self.now)
            self._ev(out, "tx", f"{self._tx_run} transmissions with nothing decoded "
                                "back" + (f" over {held}" if held else "")
                                + f" — {self._decoded_before()}", True)

    def _decoded_before(self) -> str:
        """What our receiver last got, said instead of who was on the air.

        `transmitting into an empty channel` fired over a PACTOR run that had
        just decoded seventeen inbound frames, thirteen of them control signals
        at zero bit errors, and over a VARA call three lines above the gateway's
        answer arriving at 15 of 15 tones. The count of unanswered transmissions
        is a measurement and was right both times; the channel being empty is a
        guess about who else is on the air, and it is the half an operator
        reads. This is the same rule lib/attempts.sh's NO_DECODE stops at.
        """
        if not self.rx_ok:
            return "nothing decoded inbound on this stream yet"
        if self._last_rx:
            return f"last inbound decode at {self._last_rx:%H:%M:%S}"
        return (f"{self.rx_ok} inbound decode{'s' if self.rx_ok > 1 else ''} "
                "earlier on this stream")

    def _inbound(self) -> None:
        self.rx_ok += 1
        self._last_rx = self.now or self._last_rx
        self._tx_run = 0
        self._tx_since = None

    def _data_moved(self) -> None:
        self._idle_run = 0
        self._idle_since = None

    def _mail(self, line: str, out: list[Event]) -> None:
        if line.startswith("mail: peer said: "):
            said = line.split("mail: peer said: ", 1)[1]
            self.last_peer = said
            self._ev(out, "peer", said)
            if "attempts remaining" in said or "Login failed" in said:
                self._ev(out, "peer", f"the gateway refused us: {said}", True)
            elif said.startswith("***"):
                self._ev(out, "peer", said, True)
            elif said.startswith(";PM:"):
                self._ev(out, "MAIL", f"message offered — {said}")
            return
        if m := STAGE.match(line):
            self.stage = m.group(1)
            if m.group(2) or m.group(1) == "failed":
                return self._ev(out, "stage", line, True)
            return self._ev(out, "stage", f"stage {self.stage}")
        if MOVED.match(line):
            self.outcome = line[len("mail: "):]
            return self._ev(out, "MAIL", self.outcome)
        if line == "mail: nothing moved":
            self.outcome = "nothing moved"
            return self._ev(out, "mail", "nothing moved")
        if line == "mail: NO LINK" or line.startswith("mail: exchange failed"):
            return self._ev(out, "link", line, True)
        if line.startswith("mail: remote SID"):
            return self._ev(out, "peer", line[len("mail: "):])
        if " -> " in line and "connecting as" in line:
            return self._ev(out, "run", line[len("mail: "):])

    def state(self) -> list[str]:
        rows: list[tuple[str, str]] = []
        if self.run:
            rows.append(("run", self.run))
        if self.link_up:
            held = _dur(self.link_since, self.now)
            rows.append(("link", f"UP{self.to_peer}"
                                 + (f" for {held}" if held else "")))
        elif self.link_since:
            held = _dur(self.link_since, self.now)
            rows.append(("link", f"down (was {self.target or 'up'}"
                                 + (f", held {held}" if held else "") + ")"))
        elif self.target:
            rows.append(("link", f"no link to {self.target}"))
        keyed = "KEYED" if self.keyed_since else "down"
        rows.append(("keying", f"{keyed}, {self.keyings} keyings on this stream"
                     + (f", last {self.last_key:%H:%M:%S}" if self.last_key else "")))
        if self.attempts:
            rows.append(("attempt", f"{len(self.attempts)} ran as a child process, "
                                     "each with a log of its own"))
        if self.rx_ok or self.tx_frames:
            bad = f"{self.rx_bad_ours} bad ours / {self.rx_bad_foreign} foreign"
            rows.append(("frames", f"tx {self.tx_frames}, rx {self.rx_ok} ok, {bad}"))
        if self.stage:
            rows.append(("stage", self.stage))
        if self.last_peer:
            rows.append(("peer", self.last_peer))
        if self.outcome:
            rows.append(("mail", self.outcome))
        rows.append(("alarms", str(len(self.alarms)) if self.alarms else "none"))
        return [f"-- state at {self.at} after {self.lineno} lines --"] + [
            f"  {k:<7} {v}" for k, v in rows]


def narrate(lines: Iterable[str]) -> tuple[list[Event], Narrator]:
    n = Narrator()
    events = [e for line in lines for e in n.feed(line)]
    return events + n.close(), n
