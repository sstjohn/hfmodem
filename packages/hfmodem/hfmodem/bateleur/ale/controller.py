# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE controller: scanning, calling, answering, linking (MIL-STD-188-141B
A.5.3 to A.5.5, A.5.7.2).

It keeps no clock of its own. Every method takes ``now`` in seconds, and the
three things it does to the outside world are callbacks: ``tune(channel)``
returns success, ``send(words)`` returns the frame's duration or a refusal
reason, and ``busy()`` reports whether the channel is occupied. The same
controller runs on the air and in a simulation through ``sim``.

**States** (A.5.5.2.2). *Available*: scanning the channel list, pausing on
any ALE signal and stopping on a word addressed to one of its own addresses.
*Linking*: a three-way handshake in progress, as caller or as called.
*Linked*: a handshake completed; a wait-for-activity timer (Twa) runs, and the
link ends on it, on a TWAS from the other end, or on ``hang_up``.

**Timing** is annex B's, with one allowance of this implementation's own:
the receiver reports a word a redundant word after it began (it has to hear
all three copies) and a frame a word after it ended (the next word slot has
to come up empty, which is how A.5.5.2.4 finds a frame's end). The waits for
a reply are stretched by that latency, so a reply that starts within Twr of
the call's end is not missed for being reported later.
"""
from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import NamedTuple

from . import words as W
from .fsk import Heard, Received, air_seconds

#: What this controller reports itself as (A.5.6.6.1: manufacturer-specific,
#: printable), sent as AMD, the one report format it has.
VERSION_TEXT = "HFMODEM BATELEUR ALE 1"

Tw = 0.392 / 3
Trw = 0.392
Tlww = Trw
#: The receiver's own reporting delay, beyond the frame: a word is known
#: two redundant words after it starts (acquisition waits one out), and a
#: frame's end one after its last word.
_LATENCY = 2 * Trw
_DTM_TRIES = 7                                  # A.5.7.3


@dataclass
class Config:
    addresses: list[str]                        # self addresses; the first is used to call
    channels: list[str]                         # names; the controller only sees indices
    scan_rate: float = 2.0                      # channels a second while available (A.4.2.1)
    scanning: bool = True                       # False: sit on channels[0]
    tune_s: float = 8 * Tw                      # Tt: first-try default (appendix A, annex B)
    listen_s: float = 2.0                       # Twt before calling (appendix A, annex B, general use)
    activity_s: float = 30.0                    # Twa; 0 disables
    respond: bool = True                        # answer individual calls
    answer_allcall: bool = True
    #: Tps: each channel is sounded this often while available (A.5.3.1;
    #: annex B's typical value is 30 minutes). 0: only when asked.
    sound_s: float = 0.0
    #: The channels sounded (indices); None: all of them.
    sound_channels: list | None = None
    #: How a sound ends (A.5.3.2): TIS, which invites a call back, or TWAS,
    #: which does not.
    sound_terminator: int = W.TIS
    #: The scanning call's length in seconds, overriding the channel count
    #: (a network that agrees on one, whatever each station scans).
    call_s: float | None = None
    #: Nets this station belongs to (A.4.3.2): net address -> (the member
    #: address it answers with, its slot number).
    nets: dict = field(default_factory=dict)
    answer_anycall: bool = True
    #: DTM (A.5.7.3): the largest EXTENDED block sent, in words (each a Trw;
    #: 60 is some 24 s on the air).
    max_key_s: float = 30.0
    dtm_words: int = 60
    #: DBM (A.5.7.4): the deepest EXTENDED block's interleaver, in rows (a
    #: multiple of 49; each 49 is 3.136 s on the air).
    dbm_depth: int = 49 * 4
    #: Linking protection (appendix B): 0 none, 1 AL-1 (60 s PI), 2 AL-2
    #: (2 s PI); the 56-bit key in hex; each channel's frequency, which the
    #: seed carries. A protected station reads only protected frames.
    lp_level: int = 0
    lp_key: str = ""
    channel_hz: list = field(default_factory=list)
    #: Links from stations without protection (AL-0) are taken too; off, a
    #: protected station hears them only for time exchange (B.4.1.1).
    lp_accept_clear: bool = False
    #: The time of day's quality at start (table A-XXII; a computer's clock
    #: kept by NTP is well within 3, half a second), and whether this
    #: station answers time requests (B.5.5).
    time_quality: int = 3
    time_server: bool = True


@dataclass
class Event:
    """What the controller reports."""

    kind: str                                   # LINKED, CALL_FAILED, REJECTED, CALLED,
    peer: str = ""                              # LINK_ENDED, AMD, HEARD, SOUNDED, STATE
    channel: int = -1
    text: str = ""
    lqa: dict = field(default_factory=dict)
    t: float = 0.0
    data: bytes | str = b""                     # a DTM or DBM message


@dataclass
class _Lqa:
    ber: float = 48.0
    sinad: float = 0.0
    when: float = -1e9


class _Message(NamedTuple):
    peer: str
    data: bytes | str
    blocks: list
    arq: bool
    mode: str


class Controller:
    AVAILABLE, CALLING, AWAIT_RESPONSE, AWAIT_ACK, LINKED, SOUNDING = (
        "available", "calling", "await-response", "await-ack", "linked", "sounding")
    #: One-to-many (A.5.5.4): a called station waiting for its slot, and a
    #: caller gathering the slotted responses.
    AWAIT_SLOT, COLLECT = "await-slot", "collect"
    #: Linked, a DTM or DBM block sent, waiting for its ARQ (A.5.7.3, A.5.7.4).
    DTM_WAIT = "dtm-wait"
    #: A broadcast (TWAS, answered by no one) on the air.
    SENDING = "sending"

    def __init__(self, config: Config, *, tune, send, busy=lambda: False,
                 expect=lambda ws: None, utc=None) -> None:
        if not math.isfinite(config.sound_s) or (config.sound_s != 0 and config.sound_s < 60):
            raise ValueError("sound_s must be zero or at least 60 seconds")
        if not math.isfinite(config.activity_s) or config.activity_s < 0:
            raise ValueError("activity_s must be finite and nonnegative")
        if not math.isfinite(config.max_key_s) or config.max_key_s <= 0:
            raise ValueError("max_key_s must be positive and finite")
        if any("?" in a or "@" in a for a in config.addresses):
            raise ValueError("own addresses cannot contain wildcards or stuffing")
        self.cfg = config
        self.refusal = ""
        self._responded = {}
        self._signal_since = None
        self._word_since = None
        self._exchange_until = 0.0
        #: The time of day, UTC seconds, at controller time ``now``: linking
        #: protection's seeds are made of it.
        self._utc = utc or (lambda now: now)
        #: The adjustment time exchanges have made to that clock, and its
        #: uncertainty window (seconds) as of ``_uncertain_at``.
        self._offset = 0.0
        q = config.time_quality
        # Configured, a quality is taken as the middle of its window.
        self._uncertain = W.TIME_QUALITY_S[q] if q in (0, 7) \
            else (W.TIME_QUALITY_S[q - 1] + W.TIME_QUALITY_S[q]) / 2
        self._uncertain_at = 0.0
        self._time_req: dict | None = None
        self._time_served: set = set()
        self.lp = None
        if config.lp_level:
            from .lp import Protection
            if len(config.channel_hz) != len(config.channels):
                raise ValueError("linking protection needs each channel's frequency")
            self.lp = Protection(config.lp_level, bytes.fromhex(config.lp_key),
                                 config.channel_hz)
        self.me = [W.check_address(a) for a in config.addresses]
        self._tune, self._send, self._busy = tune, send, busy
        #: Tells the receiver what the protocol expects next (A.5.2.6.3).
        self._expect = expect
        self.state = self.AVAILABLE
        self.channel = 0
        self.peer = ""
        self.linked: set[str] = set()
        self.events: list[Event] = []
        self.lqa: dict[tuple[str, int], _Lqa] = {}
        self.amd_log: list[tuple[str, str]] = []
        self._deadline = 0.0
        self._dwell_until = 0.0
        self._pause_until = 0.0
        self._tx_end = 0.0
        self._last_activity = 0.0
        self._call_plan: list[int] = []
        self._call_message: list[int] = []
        self._pending_amd: str = ""
        self._reply_words = 0
        self._reply_after = 0.0
        self._now = 0.0
        # One-to-many.
        self._group: list[str] | None = None        # a group or net call being made
        self._slots: list[str] = []                 # who answers in slots 1.. (for timing)
        self._slot_width: float | None = None       # AnyCall: uniform slots
        self._responders: list[str] = []
        self._respond_at = 0.0
        self._response: list[int] = []
        self._acked_by: list[str] = []              # addresses an ack may name us by
        self._rng = random.Random()
        # DTM and DBM: messages to send, [peer, data, blocks, arq, mode], the
        # one in hand first; the present block's tries; the KD1/KD2 (KB1/KB2)
        # sequences, each mode its own; what each station is sending us.
        self._dtm_out: list[_Message] = []
        self._dtm_tries = 0
        self._seq = {"DTM": [0, 0], "DBM": [0, 0]}
        self._dtm_in: dict[tuple[str, str], dict] = {}
        self._dtm_replies: list[int] = []
        # Periodic sounding: when each channel is next due, staggered so the
        # channels are not all due at once.
        n = len(config.channels)
        sounded = range(n) if config.sound_channels is None else set(config.sound_channels)
        self._sound_due = [config.sound_s * (i + 1) / n if i in sounded else float("inf")
                           for i in range(n)] if config.sound_s and n else []
        self._broadcast = False                     # the call being made wants no answer
        self._tuned = bool(self._tune(self.channel))

    # -- helpers ---------------------------------------------------------------

    def _emit(self, kind: str, **kw) -> None:
        self.events.append(Event(kind, t=self._now, **kw))

    def _set_state(self, s: str) -> None:
        if s in (self.AVAILABLE, self.LINKED, self.CALLING):
            self._expect([])
        if s != self.state:
            self.state = s
            self._emit("STATE", text=s, channel=self.channel, peer=self.peer)

    def _go(self, channel: int) -> bool:
        if channel != self.channel or not self._tuned:
            self._tuned = bool(self._tune(channel))
            if not self._tuned:
                self.refusal = f"could not tune to {self.cfg.channels[channel]}"
                self._emit("TUNE_FAILED", channel=channel, text=self.refusal)
                return False
            self.channel = channel
            self._signal_since = self._word_since = None
        return True

    def lp_view(self):
        """What a receiver needs to read this station's protected frames,
        or None if it is not protected."""
        if self.lp is None:
            return None
        from .lp import View
        def coarse():
            if self._time_req is not None and not self._time_req["clear"]:
                return (0, 1)
            return (0,) if self.cfg.time_server else ()
        return View(self.lp, lambda: self.utc(self._now), lambda: self.channel,
                    coarse=coarse, clear=True)

    # -- time of day (A.5.6.4, B.5.5) --------------------------------------------

    def utc(self, now: float) -> float:
        """This station's time of day at controller time ``now``."""
        return self._utc(now) + self._offset

    def time_quality(self, now: float) -> int:
        """Table A-XXII's code for the time's uncertainty now: it grows by
        72 ms an hour (a 10 ppm time base, A.5.6.4.6)."""
        grown = self._uncertain + max(0.0, now - self._uncertain_at) * 20e-6
        return W.time_quality(grown)

    def _time_words(self, t: float, quality: int) -> tuple[int, int]:
        """Time Is and coarse time for the time ``t`` at the start of the
        coarse time word (A.5.6.4.3.2, A.5.6.4.4)."""
        from datetime import datetime, timezone
        d = datetime.fromtimestamp(t, tz=timezone.utc)
        ticks = min(24, int((t % 1.0) / 0.040))
        return (W.time_is(quality, d.second, ticks),
                W.coarse_time(d.month, d.day, d.hour * 60 + d.minute))

    def time_request(self, peer: str, now: float) -> None:
        """Ask ``peer`` for the time (B.5.5.2, B.5.5.3): protected, under the
        coarse seed if this station's time is only good to the minute, if it
        knows the time that well; in the clear, authenticated, if not."""
        self._now = now
        peer = W.check_address(peer)
        if self.lp is None:
            raise RuntimeError("time exchange is linking protection's; this station has none")
        q = self.time_quality(now)
        clear = q >= 7
        coarse = not clear and W.TIME_QUALITY_S[q] > self.lp.pi_s
        start = max(now, self._tx_end)
        lead = len(W.address_words(peer, W.TO)) * 2 + 1          # words before the coarse word
        t = self.utc(start + lead * Trw)
        cmd, crs = self._time_words(t, q)
        if clear:
            cmd = W.TIME_REQUEST
            auth = W.word(W.REP, bits=self._rng.getrandbits(21))
        else:
            auth = W.authenticator([cmd, crs])
        ws = W.call(peer, self.me[0], message=[cmd, crs, auth])
        if self._transmit(ws, now, clear=clear, coarse=coarse) is None:
            self._emit("TIME_FAILED", peer=peer, text=f"transmission refused: {self.refusal}")
            return
        self._time_req = {"peer": peer, "auth": auth, "clear": clear,
                          "deadline": self._tx_end + 5 * Tw + 12 * Trw + _LATENCY}
        self._pause_until = max(self._pause_until, self._time_req["deadline"])
        self._emit("TIME_REQUESTED", peer=peer, channel=self.channel,
                   text="clear" if clear else "coarse" if coarse else "fine")

    def time_broadcast(self, now: float, to: str = W.ALLCALL) -> bool:
        """An unsolicited Time Is to ``to`` (B.5.5.5), TWAS: protected, the
        authenticator the exclusive-or of the command and coarse words."""
        self._now = now
        if self.lp is None or self.state not in (self.AVAILABLE, self.LINKED):
            return False
        q = self.time_quality(now)
        start = max(now, self._tx_end)
        lead = len(W.address_words(to, W.TO)) * 2 + 1
        cmd, crs = self._time_words(self.utc(start + lead * Trw), q)
        ws = W.call(to, self.me[0], message=[cmd, crs, W.authenticator([cmd, crs])],
                    terminator=W.TWAS)
        return self._transmit(ws, now) is not None

    def _time_exchange(self, r: Received, f: W.Frame, caller: str, end: float, now: float) -> None:
        """A time exchange addressed to us: a request to serve, or the
        response to ours."""
        t = f.time
        clear = r.seed == "clear"
        if f.terminator == W.TIS and (t["request"] or not clear):
            # A Time Request call: serve it (B.5.5.2.2, B.5.5.3.2), once per
            # seed (a minute in the clear or coarse, a PI protected).
            if not self.cfg.time_server or t["auth"] is None or self.time_quality(now) >= 7:
                return
            per = 60.0 if (clear or r.seed == "coarse") else self.lp.pi_s
            key = (clear, r.seed, int(self.utc(now) // per))
            if key in self._time_served:
                return
            self._time_served.add(key)
            start = max(now, self._tx_end)
            lead = len(W.address_words(caller, W.TO)) * 2 + 1
            tsend = self.utc(start + lead * Trw)
            cmd, crs = self._time_words(tsend, self.time_quality(now))
            x = (cmd ^ crs ^ t["auth"]) & 0xFFFFFF
            if clear:
                auth = W.word(W.REP, bits=self.lp.authenticate_clear(
                    x, (crs >> 16) & 15, (crs >> 11) & 31, crs & 0x7FF, self.channel))
            else:
                auth = W.word(W.REP, bits=x & 0x1FFFFF)
            ws = W.call(caller, self._pertinent(f.to)[1] or self.me[0],
                        message=[cmd, crs, auth], terminator=W.TWAS)
            if self._transmit(ws, now, clear=clear, coarse=r.seed == "coarse") is not None:
                self._emit("TIME_SERVED", peer=caller, channel=self.channel,
                           text="clear" if clear else r.seed)
            return
        req = self._time_req
        if f.terminator != W.TWAS or req is None or req["peer"] != caller \
                or t["coarse"] is None or t["null"] or clear != req["clear"]:
            return
        x = (t["cmd"] ^ t["coarse"]["word"] ^ req["auth"]) & 0xFFFFFF
        c = t["coarse"]
        want = self.lp.authenticate_clear(x, c["month"], c["day"], c["minute"], self.channel) \
            if clear else x & 0x1FFFFF
        if want != t["auth"] & 0x1FFFFF:
            self._emit("TIME_FAILED", peer=caller, channel=self.channel,
                       text="the authenticator did not check")
            return
        self._time_req = None
        # The coarse word began this many redundant words before the end.
        after = f.words - (t["index"] + 1)
        heard_at = end - after * Trw
        from datetime import datetime, timezone
        here = self.utc(heard_at)
        year = datetime.fromtimestamp(here, tz=timezone.utc).year
        best = None
        for y in (year - 1, year, year + 1):
            try:
                d = datetime(y, c["month"], c["day"], tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
            there = d + c["minute"] * 60 + t["seconds"] + 0.040 * t["ticks"]
            if best is None or abs(there - here) < abs(best - here):
                best = there
        if best is None:
            return
        delta = best - here
        self._offset += delta
        # Its uncertainty, and ours in reading it: processing and a
        # propagation bound (A.5.6.4.6's example: 100 and 35 ms either way).
        self._uncertain = W.TIME_QUALITY_S[t["quality"]] + 2 * (0.100 + 0.035)
        self._uncertain_at = now
        self._emit("TIME", peer=caller, channel=self.channel,
                   text=f"adjusted {delta:+.3f} s, quality {self.time_quality(now)}")

    def _transmit(self, ws: list, now: float, *, scan: int = 0, sound: bool = False,
                  clear: bool = False, coarse: bool = False) -> float | None:
        self.refusal = ""
        if not self._tuned:
            self.refusal = "channel tune has not succeeded"
            return None
        seconds = air_seconds(ws)
        if seconds > self.cfg.max_key_s:
            self.refusal = f"frame {seconds:.3f} s exceeds max_key_s {self.cfg.max_key_s:.3f} s"
            self._emit("TX_REFUSED", channel=self.channel, text=self.refusal)
            return None
        if self.lp is not None and not clear:
            start = max(now, self._tx_end)
            ws = self.lp.protect(ws, self.utc(start), self.channel, scan=scan, sound=sound,
                                 coarse=coarse)
        secs = self._send(ws)
        if secs is None or isinstance(secs, str):
            self.refusal = secs or "transmission refused"
            return None
        self._tx_end = max(now, self._tx_end) + secs
        self._last_activity = self._tx_end
        return self._tx_end

    def _is_me(self, addr: str) -> bool:
        return any(_matches(addr, a) for a in self.me)

    def _scan_words(self) -> int:
        if self.cfg.call_s is not None:
            return max(0, int(round(self.cfg.call_s / Trw)))
        c = len(self.cfg.channels) if self.cfg.scanning else 0
        return 2 * c if c > 1 else 0              # 2 Trw a channel; each word is a Trw

    # -- operator commands -----------------------------------------------------

    def call(self, peer: str | list[str], now: float, *, amd: str = "",
             channel: int | None = None, lqa: bool = False, version: bool = False,
             capabilities: bool = False, members: list[str] | None = None) -> None:
        """Link with ``peer``: on ``channel``, or on each channel in turn, best
        first (A.5.4). ``lqa``, ``version`` and ``capabilities`` ask the far
        end to report them in its response (A.5.4.2, A.5.6.6).

        A list of addresses is a star group call (A.5.5.4.3): each answers in
        the slot its place in the list gives it. A net address with its
        ``members`` in slot order is a star net call (A.5.5.4.2). An address
        with "?" in it is a wildcard call, answered in random slots as an
        AnyCall is (A.5.5.4.6). Either way the link is made with whoever
        answered."""
        self._now = now
        if self.state not in (self.AVAILABLE, self.LINKED):
            raise RuntimeError(f"busy: {self.state}")
        self._slot_width = None
        if isinstance(peer, (list, tuple)):
            group = [W.check_address(a) for a in peer]
            self._group, self._slots = group, list(group)
            peer = ",".join(group)
        elif members:
            peer = W.check_address(peer)
            self._group, self._slots = [peer], [W.check_address(m) for m in members]
        elif "?" in peer:
            peer = W.check_address(peer)
            self._group, self._slots = [peer], []
            self._slot_width = (23 if lqa else 20) * Tw
        else:
            peer = W.check_address(peer)
            self._group, self._slots = None, []
        self.peer = peer
        self._call_message = W.amd_words(amd) if amd else []
        self._reply_words = 0
        self._broadcast = False
        if lqa:
            self._call_message.append(W.lqa_word(request=True))
            self._reply_words += 1
        if version:
            self._call_message.append(W.version_query())
            self._reply_words += len(W.amd_words(VERSION_TEXT)) + 2
        if capabilities:
            self._call_message.append(W.capabilities_query())
            self._reply_words += 5
        if channel is not None:
            self._call_plan = [channel]
        elif self.state == self.LINKED and peer in self.linked:
            self._call_plan = [self.channel]
        elif not self.cfg.scanning:
            self._call_plan = [self.channel]
        else:
            n = len(self.cfg.channels)
            self._call_plan = sorted(range(n), key=lambda c: self._score(peer, c), reverse=True)
        self._next_channel(now)

    def anycall(self, now: float, *, selective: str = "", lqa: bool = False,
                channel: int | None = None) -> None:
        """An AnyCall (A.5.5.4.5): every station hearing it that is not told
        otherwise answers in a slot it picks at random; the link is made
        with whoever was read. ``selective``: one or two characters the
        answering stations' addresses must end with."""
        if selective and not (1 <= len(selective) <= 2 and selective.isalnum()):
            raise ValueError("a selective AnyCall names one or two final characters")
        addr = ("@@" + selective.upper()) if len(selective) == 1 else \
            ("@" + selective.upper()) if selective else W.ANYCALL
        self._now = now
        if self.state not in (self.AVAILABLE, self.LINKED):
            raise RuntimeError(f"busy: {self.state}")
        self.peer = addr
        self._group, self._slots = [addr], []
        self._slot_width = (23 if lqa else 20) * Tw
        self._broadcast = False
        self._call_message = [W.lqa_word(request=True)] if lqa else []
        self._reply_words = 1 if lqa else 0
        self._call_plan = [self.channel if channel is None else channel]
        self._next_channel(now)

    def allcall(self, now: float, *, selective: str = "", amd: str = "",
                terminator: int = W.TIS, channel: int | None = None) -> bool:
        """An AllCall (A.5.5.4.4): every station hearing it stops and listens;
        none answers. Concluded with TIS, those that heard it are linked
        with this station. ``selective``: the final character the listening
        stations' addresses must end with."""
        addr = ("@" + selective.upper() + "@") if selective else W.ALLCALL
        if selective and not (len(selective) == 1 and selective.isalnum()):
            raise ValueError("a selective AllCall names one final character")
        self._now = now
        if self.state != self.AVAILABLE or self._busy():
            return False
        if channel is not None and not self._go(channel):
            return False
        ws = W.call(addr, self.me[0], scan_words=self._scan_words(),
                    message=W.amd_words(amd) if amd else [], terminator=terminator)
        if self._transmit(ws, now, scan=self._scan_words()) is None:
            return False
        self._emit("SENT", peer=addr, channel=self.channel,
                   text="allcall" + (f": {amd.upper()}" if amd else ""))
        if terminator == W.TIS:
            self.linked.add(addr)
            self.peer = addr
            self._set_state(self.LINKED)
        return True

    def send_dtm(self, peer: str, data: bytes | str, now: float, *, arq: bool = True) -> None:
        """A DTM message (A.5.7.3): text as 7-bit ASCII, bytes as binary bits,
        in data blocks each acknowledged before the next (``arq``), or sent
        one after another without. Over the link with ``peer``, which is
        called first if there is none."""
        self._send_blocks("DTM", peer, data, now, arq)

    def send_dbm(self, peer: str, data: bytes | str, now: float, *, arq: bool = True) -> None:
        """A DBM message (A.5.7.4): as ``send_dtm``, in deeply interleaved
        blocks, some three times as fast and without the words' redundancy."""
        self._send_blocks("DBM", peer, data, now, arq)

    def _send_blocks(self, mode: str, peer: str, data, now: float, arq: bool) -> None:
        self._now = now
        if len(data) > 65536 or len(self._dtm_out) >= 16 or sum(len(m.data) for m in self._dtm_out) + len(data) > 65536:
            raise ValueError("DTM/DBM queue exceeds 16 messages or 65536 bytes")
        peer = W.check_address(peer)
        overhead = len(W.call(peer, self.me[0])) * Trw
        room = self.cfg.max_key_s - overhead
        words = min(self.cfg.dtm_words, math.floor(room / Trw) - 2)
        depth = min(self.cfg.dbm_depth, math.floor((room - Trw) * 375 / (24 * 49)) * 49)
        if (mode == "DTM" and words < 1) or (mode == "DBM" and depth < 49):
            raise ValueError(f"max_key_s {self.cfg.max_key_s:g} cannot hold a {mode} block with these addresses")
        seq = self._seq[mode]
        seq[1] ^= 1                                  # a new message
        if mode == "DTM":
            blocks = W.dtm_blocks(data, arq=arq, kd2=seq[1], kd1=seq[0],
                                  max_words=words)
        else:
            blocks = W.dbm_blocks(data, arq=arq, kb2=seq[1], kb1=seq[0],
                                  max_depth=depth)
        seq[0] ^= len(blocks) & 1
        self._dtm_out.append(_Message(peer, data, blocks, arq, mode))
        if peer not in self.linked and self.state == self.AVAILABLE:
            self.call(peer, now)

    def broadcast(self, to: str, text: str, now: float, *,
                  channels: list[int] | None = None) -> None:
        """An AMD to ``to`` — an individual, a net, an AllCall — ending TWAS,
        so that no one answers and no link is made (A.5.3.2's terminator on a
        call): on each of ``channels`` in turn (default the present one),
        each after listening first. ``SENT`` when it is done."""
        self._now = now
        if self.state != self.AVAILABLE:
            raise RuntimeError(f"busy: {self.state}")
        self._slot_width = None
        self._group, self._slots = None, []
        self.peer = W.check_address(to)
        self._call_message = W.amd_words(text)
        self._reply_words = 0
        self._broadcast = True
        self._call_plan = list(channels) if channels is not None else [self.channel]
        self._next_channel(now)


    def sound(self, now: float, *, terminator: int | None = None,
              seconds: float | None = None) -> bool:
        """A sound on the present channel (A.5.3): the whole address, repeated
        for a scan period of the stations listening, then twice more."""
        if self.state != self.AVAILABLE or self._busy():
            return False
        return self._sound(now, terminator=terminator, seconds=seconds)

    def _sound(self, now: float, *, terminator: int | None = None,
               seconds: float | None = None) -> bool:
        if terminator is None:
            terminator = self.cfg.sound_terminator
        if seconds is None and self.cfg.call_s is not None:
            seconds = self.cfg.call_s
        span = seconds if seconds is not None else len(self.cfg.channels) / self.cfg.scan_rate
        addr = W.address_words(self.me[0], terminator)
        each = len(addr) * Trw
        repeats = max(2, int(span / each + 0.999) + 2)
        if self._transmit(W.sound(self.me[0], repeats=repeats, terminator=terminator), now,
                          sound=True) is None:
            return False
        self._emit("SOUNDED", channel=self.channel)
        return True

    def hang_up(self, now: float) -> None:
        """End every link: TO each peer, TWAS me (A.5.5.3.5)."""
        if self.state not in (self.LINKED, self.DTM_WAIT):
            return
        for peer in sorted(self.linked):
            self._transmit(W.call(peer, self.me[0], terminator=W.TWAS), now)
            self._emit("LINK_ENDED", peer=peer, channel=self.channel, text="hung up")
        self._drop_links()

    def stay(self, channel: int, now: float) -> None:
        """Stop scanning and sit on ``channel`` (the operator's choice)."""
        self._now = now
        if not 0 <= channel < len(self.cfg.channels):
            raise ValueError(f"no channel {channel}")
        if self.state != self.AVAILABLE:
            raise RuntimeError(f"cannot change channel while {self.state}; hang up a link or wait for available")
        if not self._go(channel):
            raise RuntimeError(self.refusal)
        self.cfg.scanning = False
        self._emit("STATE", text=self.state, channel=channel)

    def activity(self, now: float) -> None:
        """Traffic on the link that was not ALE: holds off Twa."""
        self._last_activity = now

    # -- calling ---------------------------------------------------------------

    def _next_channel(self, now: float) -> None:
        if not self._call_plan and self._broadcast:
            self._broadcast = False
            self._emit("SENT", peer=self.peer, channel=self.channel)
            self.peer = ""
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        if not self._call_plan:
            self._emit("CALL_FAILED", peer=self.peer, text="no channel answered")
            for m in [m for m in self._dtm_out if m.peer not in self.linked]:
                self._emit(m[4] + "_FAILED", peer=m[0], text="no link")
                self._dtm_out.remove(m)
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        while self._call_plan:
            if self._go(self._call_plan.pop(0)):
                break
        else:
            self._emit("CALL_FAILED", peer=self.peer, text=self.refusal)
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        self._set_state(self.CALLING)
        linked_here = self.peer in self.linked
        # Listen before transmit (A.5.4.7), unless the link is already here.
        self._deadline = now + (0.0 if linked_here else self.cfg.listen_s)

    def _send_call(self, now: float) -> None:
        linked_here = self.peer in self.linked
        called = self._group if self._group is not None else self.peer
        scan = 0 if linked_here else self._scan_words()
        ws = W.call(called, self.me[0], scan_words=scan, message=self._call_message,
                    terminator=W.TWAS if self._broadcast else W.TIS)
        end = self._transmit(ws, now, scan=scan)
        if end is not None and self._broadcast:
            self._set_state(self.SENDING)       # the next channel once this is sent
            self._deadline = end
            return
        if end is None:
            self._broadcast = False
            self._emit("CALL_FAILED", peer=self.peer, text=f"transmission refused: {self.refusal}")
            self._call_plan = []
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        if self._group is not None:
            # Slotted responses (A.5.5.4.1): gather them until the last slot
            # has passed (Twrn), then acknowledge whoever answered.
            self._responders = []
            self._set_state(self.COLLECT)
            self._expect(W.call(self.me[0], self.me[0]))    # TO us, at least
            self._deadline = end + self._twrn() + _LATENCY + Trw
            return
        self._set_state(self.AWAIT_RESPONSE)
        # The response: TO us (whole address, twice), TIS them.
        self._expect(W.call(self.me[0], self.peer))
        tune = 0.0 if linked_here else self.cfg.tune_s
        # A.5.5.3.1: the reply starts within Twrt of the call's end; the
        # response frame runs three redundant words a word of address.
        resp = (2 * len(W.address_words(self.me[0], W.TO))
                + len(W.address_words(self.peer, W.TIS)) + self._reply_words) * Trw
        self._deadline = end + 5 * Tw + tune + resp + _LATENCY + Trw

    # -- slots (A.5.5.4.1.3) -----------------------------------------------------

    @staticmethod
    def _tswt(sn: int, caller: str, before: list[str], message_words: int = 0) -> float:
        """Tswt(SN) = SN [5 Tw + 2 Ta(caller) + Tm] + Ta(caller) + the addresses
        of the stations in slots 1 .. SN-1, in seconds."""
        ta = lambda a: len(W.address_words(a, W.TO)) * Trw
        return sn * (5 * Tw + 2 * ta(caller) + message_words * Trw) + ta(caller) \
            + sum(ta(a) for a in before)

    def _twrn(self) -> float:
        """How long after its call the caller waits for the last slot."""
        if self._slot_width is not None:                  # AnyCall: 16 slots and slot 0
            return 17 * self._slot_width
        return self._tswt(len(self._slots) + 1, self.me[0], self._slots, self._reply_words)

    def _acknowledge(self, now: float) -> None:
        """The caller's acknowledgment to those it heard (A.5.5.4.2.3,
        A.5.5.4.3.6): to the net address for a net, else to each of them."""
        heard = list(dict.fromkeys(self._responders))
        # As many as one acknowledgment can name (A.5.5.4.3): the rest were
        # heard too late to be linked.
        while len(heard) > 1:
            try:
                W.call(heard, self.me[0])
                break
            except ValueError:
                heard.pop()
        if not heard:
            self._next_channel(now)
            return
        net = self._group is not None and len(self._group) == 1 and self._slots \
            and not self._group[0].startswith("@")
        to = self._group[0] if net else heard
        if self._transmit(W.call(to, self.me[0]), now) is None:
            self._emit("CALL_FAILED", peer=self.peer, text="acknowledgment refused")
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        self.linked.update(heard)
        self._set_state(self.LINKED)
        for a in heard:
            self._emit("LINKED", peer=a, channel=self.channel, lqa=self._lqa_dict(a))
        self._group = None

    # -- DTM and DBM (A.5.7.3, A.5.7.4) ------------------------------------------

    @staticmethod
    def _fields(w: int) -> dict:
        return W.dbm_fields(w) if W.is_dbm(w) else W.dtm_fields(w)

    def _dtm_send(self, now: float) -> None:
        """The present message's next block, or the same one again."""
        peer, data, blocks, arq, mode = self._dtm_out[0]
        if self._dtm_tries >= _DTM_TRIES or (self._dtm_tries and now >= self._exchange_until):
            self._emit(mode + "_FAILED", peer=peer, channel=self.channel,
                       text=f"no acknowledgment after {self._dtm_tries} tries")
            self._dtm_out.pop(0)
            self._set_state(self.LINKED)
            return
        if self._dtm_tries == 0:
            self._exchange_until = now + 300.0
        self._dtm_tries += 1
        ws = W.call(peer, self.me[0], message=blocks[0])
        end = self._transmit(ws, now)
        if end is None:
            self._emit(mode + "_FAILED", peer=peer, channel=self.channel,
                       text=f"transmission refused: {self.refusal}")
            self._dtm_out.pop(0)
            self._set_state(self.LINKED)
            return
        if not arq:
            # Sent once each, one frame after another.
            blocks.pop(0)
            if not blocks:
                self._emit(mode + "_SENT", peer=peer, channel=self.channel, data=data)
                self._dtm_out.pop(0)
            self._set_state(self.LINKED)
            return
        self._set_state(self.DTM_WAIT)
        f = self._fields(blocks[0][0])
        arq_word = W.dtm_arq(kd2=f["kd2"], kd1=f["kd1"]) if mode == "DTM" \
            else W.dbm_arq(kb2=f["kd2"], kb1=f["kd1"])
        ack = W.call(self.me[0], peer, message=[arq_word])
        self._expect(ack)
        # The reply comes as a call's response does (Twr), a DTM frame
        # being read a word late for the two windows to agree on it.
        self._deadline = end + 5 * Tw + len(ack) * Trw + _LATENCY + 2 * Trw

    def _dtm_arq(self, caller: str, d, now: float) -> None:
        """A DTM or DBM ARQ from the station we are sending to."""
        if self.state != self.DTM_WAIT or not self._dtm_out or self._dtm_out[0].peer != caller:
            return
        peer, data, blocks, arq, mode = self._dtm_out[0]
        f = self._fields(blocks[0][0])
        if (d.kd1, d.kd2) != (f["kd1"], f["kd2"]) or isinstance(d, W.Dbm) != (mode == "DBM"):
            return                                 # about another block
        if d.kd3:                                  # pause: wait for "continue"
            self._deadline = min(self._exchange_until, now + max(self.cfg.activity_s, 60.0))
            return
        if d.kd4:                                  # NAK: the same block again
            self._dtm_send(now)
            return
        blocks.pop(0)
        self._dtm_tries = 0
        if not blocks:
            self._emit(mode + "_SENT", peer=peer, channel=self.channel, data=data)
            self._dtm_out.pop(0)
        self._set_state(self.LINKED)

    def _dtm_receive(self, caller: str, f: W.Frame, now: float) -> list[int]:
        """The DTM and DBM words in a frame to us: blocks checked, combined
        with an earlier failed copy (figure A-48; a DBM's soft values added),
        reassembled and delivered; ARQs passed to the sending side. Returns
        the ARQ words to reply."""
        replies = []
        for d in list(f.dtm) + list(f.dbm):
            mode = "DBM" if isinstance(d, W.Dbm) else "DTM"
            st = self._dtm_in.setdefault((caller, mode), {
                "last": None, "bad": None, "bits": [], "kd2": None, "ascii": True})
            if d.kind == "arq":
                self._dtm_arq(caller, d, now)
                continue
            if d.kind == "null":
                if d.kd4:                          # an interrogation: we take both types
                    replies.append(d.arq_word(nak=True))
                self._dtm_deliver(caller, st, mode)    # a break ends any message
                continue
            if d.kind not in ("basic", "extended"):
                continue
            ident = (d.kd1, d.kd2, d.size)
            if st["last"] == ident and (st["bad"] is None or st["bad"][0] != ident):
                # A duplicate of a block already taken: our ACK was lost.
                if d.kd4:
                    replies.append(d.arq_word())
                continue
            if not d.ok and st["bad"] is not None and st["bad"][0] == ident:
                d = st["bad"][1].combined(d)
            if not d.ok:
                st["bad"] = (ident, d)
                self._emit(mode + "_DAMAGED", peer=caller, channel=self.channel,
                           text=f"{len(d.damaged)} of {d.size} "
                                f"{'words' if mode == 'DTM' else 'rows'} unread")
                if d.kd4:
                    replies.append(d.arq_word(nak=True))
                continue
            st["bad"] = None
            st["last"] = ident
            if st["kd2"] is not None and st["kd2"] != d.kd2:
                self._dtm_deliver(caller, st, mode)
            st["kd2"], st["ascii"] = d.kd2, d.ascii
            st["bits"] += d.data_bits
            if d.kind == "basic":
                self._dtm_deliver(caller, st, mode)
            if d.kd4:
                replies.append(d.arq_word())
        return replies

    def _dtm_deliver(self, caller: str, st: dict, mode: str = "DTM") -> None:
        b = st["bits"]
        if not b:
            return
        if st["ascii"]:
            data = "".join(chr(W.bits_int(b[i:i + 7]))
                           for i in range(0, len(b) - 6, 7))
        else:
            data = bytes(W.bits_int(b[i:i + 8]) for i in range(0, len(b) - 7, 8))
        st["bits"], st["kd2"] = [], None
        self._emit(mode, peer=caller, channel=self.channel, data=data,
                   text=data if isinstance(data, str) else data.hex())

    # -- the clock -------------------------------------------------------------

    def tick(self, now: float) -> None:
        self._now = now
        if now < self._tx_end:
            return
        if self._time_req is not None and now >= self._time_req["deadline"]:
            self._emit("TIME_FAILED", peer=self._time_req["peer"], channel=self.channel,
                       text="no time service response")
            self._time_req = None
        if self.state == self.AVAILABLE:
            if not self._sound_due_now(now):
                self._scan(now)
        elif self.state == self.SOUNDING and now >= self._deadline:
            ch = self.channel
            if not self._busy() and self._sound(now):
                self._sound_due[ch] = now + self.cfg.sound_s
            else:
                self._sound_due[ch] = now + 60.0           # busy: try again soon
            self._set_state(self.AVAILABLE)
        elif self.state == self.CALLING and now >= self._deadline:
            # The listen period is over: judge what it heard (A.5.4.7).
            if self.peer not in self.linked and self._busy():
                self._emit("STATE", text="channel busy", channel=self.channel)
                self._next_channel(now)
            else:
                self._send_call(now)
        elif self.state in (self.AWAIT_RESPONSE, self.SENDING) and now >= self._deadline:
            self._next_channel(now)
        elif self.state == self.COLLECT and now >= self._deadline:
            self._acknowledge(now)
        elif self.state == self.AWAIT_SLOT and now >= self._respond_at:
            end = self._transmit(self._response, now)
            if end is None:
                self.peer = ""
                self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            else:
                self._set_state(self.AWAIT_ACK)
        elif self.state == self.AWAIT_ACK and now >= self._deadline:
            self._emit("CALL_FAILED", peer=self.peer, channel=self.channel,
                       text="no acknowledgment")
            self.peer = ""
            self._set_state(self.LINKED if self.linked else self.AVAILABLE)
        elif self.state == self.DTM_WAIT and now >= self._deadline:
            self._dtm_send(now)                        # no ARQ read: send it again
        elif self.state == self.LINKED and self._dtm_out and self._dtm_out[0].peer in self.linked:
            self._dtm_tries = 0
            self._dtm_send(now)
        elif self.state == self.LINKED and self.cfg.activity_s \
                and now - self._last_activity > self.cfg.activity_s:
            for peer in sorted(self.linked):
                self._emit("LINK_ENDED", peer=peer, channel=self.channel, text="no activity")
            self._drop_links()

    def _sound_due_now(self, now: float) -> bool:
        """A channel whose periodic sound is due: go there and listen first
        (A.5.3.1: on clear channels only). Not while a signal holds the scan."""
        if not self._sound_due or now < self._pause_until:
            return False
        eligible = range(len(self._sound_due)) if self.cfg.scanning else [self.channel]
        ch = min(eligible, key=lambda c: self._sound_due[c])
        if now < self._sound_due[ch]:
            return False
        if not self._go(ch):
            self._sound_due[ch] = now + max(self.cfg.sound_s, 60)
            return False
        self._set_state(self.SOUNDING)
        self._deadline = now + self.cfg.listen_s
        return True

    def _scan(self, now: float) -> None:
        if not self.cfg.scanning or len(self.cfg.channels) < 2:
            return
        if now < max(self._dwell_until, self._pause_until):
            return
        self._go((self.channel + 1) % len(self.cfg.channels))
        self._signal_since = self._word_since = None
        self._dwell_until = now + 1.0 / self.cfg.scan_rate

    def _drop_links(self) -> None:
        for peer, _, _, _, mode in self._dtm_out:
            self._emit(mode + "_FAILED", peer=peer, channel=self.channel, text="link ended")
        self._dtm_out = []
        self.linked.clear()
        self.peer = ""
        self._set_state(self.AVAILABLE)

    # -- what the receiver heard ------------------------------------------------

    def heard(self, ev, now: float, age: float = 0.0) -> None:
        """A ``fsk.Heard`` word or ``fsk.Received`` frame, as the receiver
        reported it, on the present channel, ``age`` seconds after it ended
        (slotted responses are timed from the call's end, not its report)."""
        self._now = now
        if isinstance(ev, Heard):
            self._word(ev, now)
        elif isinstance(ev, Received):
            self._frame(ev, now, now - age)

    def signal(self, now: float) -> None:
        """ALE signalling on the present channel: stay to read a word (Tdrw,
        two redundant words, table A-XV) before scanning on."""
        if self.state in (self.AVAILABLE, self.SOUNDING):
            if self._signal_since is None:
                self._signal_since = now
            self._pause_until = max(self._pause_until,
                                    min(now + 2 * Trw, self._signal_since + 4 * Trw))

    def _word(self, h: Heard, now: float) -> None:
        # Any ALE signal pauses the scan long enough to read a frame; a call
        # that may be to us holds it for the calling cycle (Twce, A.5.5.3.2):
        # the first word of one of our addresses or nets, in TO, or in THRU
        # or REP as a group's scanning call rotates them.
        if self.state != self.AVAILABLE:
            return
        hold = 3 * Trw
        if W.is_dbm(h.word):
            depth = W.dbm_fields(h.word)["depth"]
            if 0 < depth <= self.cfg.dbm_depth:
                hold = depth * 24 / 375 + 8 * Trw
        elif W.is_dtm(h.word):
            words = W.dtm_fields(h.word)["words"]
            if 0 < words <= self.cfg.dtm_words:
                hold = (words + 9) * Trw
        first = W.chars_of(h.word)
        mine = list(self.me) + list(self.cfg.nets)
        if h.preamble in (W.TO, W.THRU, W.REP) and (
                any(_matches(first, _first3(a)) for a in mine) or first.startswith("@")):
            ts = len(self.cfg.channels) / self.cfg.scan_rate if self.cfg.scanning else 0
            hold = max(hold, 2 * ts + 4 * Trw)
        if self._word_since is None:
            self._word_since = now
        self._pause_until = max(self._pause_until, min(now + hold, self._word_since + 30.0))

    def _frame(self, r: Received, now: float, end: float | None = None) -> None:
        f = r.frame
        if not f.complete:
            return
        end = now if end is None else end
        caller = f.caller
        self._note(caller, r, now)
        if f.sound:
            self._emit("HEARD", peer=caller, channel=self.channel,
                       text="sound" if f.terminator == W.TIS else "sound (not accepting calls)",
                       lqa=self._lqa_dict(caller))
            return
        kind, mine = self._pertinent(f.to)
        if kind is None:
            self._emit("HEARD", peer=caller, channel=self.channel, text="call to " + ",".join(f.to),
                       lqa=self._lqa_dict(caller))
            return
        if f.time is not None and self.lp is not None and kind == "individual":
            self._time_exchange(r, f, caller, end, now)
            return
        if r.seed == "clear" and not self.cfg.lp_accept_clear:
            # B.4.1.1: not linked automatically from a lower level.
            self._emit("HEARD", peer=caller, channel=self.channel,
                       text="unprotected call to " + ",".join(f.to), lqa=self._lqa_dict(caller))
            return
        for text in f.amd:
            self.amd_log.append((caller, text))
            del self.amd_log[:-1024]
            self._emit("AMD", peer=caller, channel=self.channel, text=text)
        if caller in self.linked and kind != "allcall":
            self._last_activity = now
        if f.terminator == W.TWAS:
            if caller == self.peer and self.state in (self.AWAIT_RESPONSE, self.AWAIT_ACK):
                self._emit("REJECTED", peer=caller, channel=self.channel)
                self._call_plan = []
                self.peer = ""
                self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            elif caller in self.linked:
                self.linked.discard(caller)
                self._emit("LINK_ENDED", peer=caller, channel=self.channel, text="by the far end")
                if not self.linked:
                    self._drop_links()
            return
        # TIS: the handshake continues.
        self._reports(f, caller)
        blocks = bool(f.dtm or f.dbm)
        self._dtm_replies = self._dtm_receive(caller, f, now) if blocks else []
        if blocks and caller in self.linked and self.state in (self.LINKED, self.DTM_WAIT) \
                and kind == "individual":
            # DTM traffic within the link: answered with the DTM ARQs asked
            # for, and the link goes on (A.5.7.3).
            if self._dtm_replies:
                self._transmit(W.call(caller, mine, message=self._dtm_replies), now)
            self._last_activity = now
            return
        if self.state == self.COLLECT:
            # A slotted response to our one-to-many call (A.5.5.4.1).
            if kind == "individual" and caller not in self._responders:
                self._responders.append(caller)
                self._emit("RESPONDED", peer=caller, channel=self.channel,
                           lqa=self._lqa_dict(caller))
            return
        if self.state == self.AWAIT_RESPONSE and caller == self.peer:
            # The response. Acknowledge, and the link is made (A.5.5.3.4).
            if self._transmit(W.call(caller, self.me[0]), now) is None:
                self._emit("CALL_FAILED", peer=caller, text="acknowledgment refused")
                self._set_state(self.LINKED if self.linked else self.AVAILABLE)
                return
            self.linked.add(caller)
            self._set_state(self.LINKED)
            self._emit("LINKED", peer=caller, channel=self.channel, lqa=self._lqa_dict(caller))
            return
        if self.state in (self.AWAIT_ACK, self.AWAIT_SLOT) and caller == self.peer:
            # The acknowledgment: to our address, our net, or a group we
            # are in. If it leaves us out we were not heard (A.5.5.4.3.6).
            if self.state == self.AWAIT_SLOT:
                return                                 # our slot is yet to come
            if any(a in self._acked_by or self._is_me(a) for a in f.to):
                self.linked.add(caller)
                self._set_state(self.LINKED)
                self._emit("LINKED", peer=caller, channel=self.channel,
                           lqa=self._lqa_dict(caller))
            else:
                self._emit("CALL_FAILED", peer=caller, channel=self.channel,
                           text="not acknowledged")
                self.peer = ""
                self._set_state(self.LINKED if self.linked else self.AVAILABLE)
            return
        if self.state not in (self.AVAILABLE, self.LINKED, self.SOUNDING):
            return
        if kind == "allcall":
            # A.5.5.4.4: nobody answers; concluded with TIS, all who heard
            # it are linked with the caller.
            if self.cfg.answer_allcall and not self.linked:
                self.linked.add(caller)
                self.peer = caller
                self._last_activity = now
                self._set_state(self.LINKED)
                self._emit("LINKED", peer=caller, channel=self.channel, text="allcall",
                           lqa=self._lqa_dict(caller))
            return
        if not self.cfg.respond:
            return
        if f.do_not_respond and not any(q["request"] for q in f.lqa):
            return                                     # A.5.6.7
        replies = self._replies(f, r)
        msg = len(replies)
        if kind == "individual":
            if now - self._responded.get(caller, -1e9) < 60.0:
                return
            self._responded = {p: t for p, t in self._responded.items() if now - t < 60.0}
            if len(self._responded) >= 64:
                return
            self._responded[caller] = now
            self._respond(f, caller, mine, replies, now)
            return
        if kind == "net":
            member, sn = self.cfg.nets[mine]
            # The slots before ours hold members we do not know the
            # addresses of; the text times them as one word each.
            tswt = self._tswt(sn, caller, ["@@@"] * (sn - 1), msg)
            ack = [mine]
            answer = member
        elif kind == "group":
            sn = f.to.index(mine) + 1
            tswt = self._tswt(sn, caller, f.to[:sn - 1], msg)
            ack = [mine]
            answer = mine
        else:                                          # anycall, wildcard
            if not self.cfg.answer_anycall:
                return
            # A.5.5.4.5: a slot picked at random from 1 to 16, each wide
            # enough for a one-word address and an LQA report.
            width = (23 if any(q["request"] for q in f.lqa) else 20) * Tw
            sn = self._rng.randint(1, 16)
            tswt = sn * width
            if kind == "anycall":
                # An address no longer than five words less twice the
                # caller's, so the response fits its slot.
                room = 5 - 2 * len(W.address_words(caller, W.TO))
                fits = [m for m in self.me if len(W.address_words(m, W.TO)) <= room]
                if not fits:
                    return
                mine = min(fits, key=len)
            ack = [mine]
            answer = mine
        self._emit("CALLED", peer=caller, channel=self.channel, text=kind)
        self.peer = caller
        self._response = W.call(caller, answer, message=replies)
        self._acked_by = ack + [answer]
        self._respond_at = end + tswt
        self._set_state(self.AWAIT_SLOT)
        self._expect(W.call(answer, caller))
        # The acknowledgment follows the last slot; how long that is we
        # know only for a group, so wait for the most a call can hold.
        last = self._tswt(len(f.to) + 1, caller, f.to, msg) if kind == "group" \
            else max(tswt, 17 * 23 * Tw)
        self._deadline = end + max(last, tswt + 5 * Tw) + 2 * Trw \
            + (24 + len(W.address_words(caller, W.TIS))) * Trw + _LATENCY + Trw

    def _respond(self, f: W.Frame, caller: str, mine: str, replies: list[int],
                 now: float) -> None:
        """A call to us alone: respond at once (A.5.5.3.3), and wait Twr for
        the acknowledgment. The call itself is what the channel was just
        carrying, so no listening first: Twr leaves no time for it."""
        self._emit("CALLED", peer=caller, channel=self.channel)
        self.peer = caller
        end = self._transmit(W.call(caller, mine, message=replies), now)
        if end is None:
            self.peer = ""
            return
        self._acked_by = [mine]
        self._set_state(self.AWAIT_ACK)
        self._expect(W.call(mine, caller) + W.call(mine, caller, terminator=W.TWAS))
        ack = (2 * len(W.address_words(mine, W.TO))
               + len(W.address_words(caller, W.TIS))) * Trw
        self._deadline = end + 5 * Tw + ack + _LATENCY + Trw

    def _pertinent(self, called: list[str]) -> tuple[str | None, str]:
        """How a call's addresses concern us, and the address they reach us
        by: an individual call, one to a net or group we are in, an AllCall
        or AnyCall we answer to, or a wildcard matching us (A.5.2.4)."""
        for a in called:
            if W.is_allcall(a) and (a == W.ALLCALL or any(m[-1] == a[1] for m in self.me)):
                return "allcall", a
            if W.is_anycall(a):
                tail = a.lstrip("@")
                if a == W.ANYCALL or any(m.endswith(tail) for m in self.me):
                    return "anycall", a
            if a in self.cfg.nets:
                return "net", a
        for a in called:
            for m in self.me:
                if a == m:
                    return ("group" if len(called) > 1 else "individual"), m
        for a in called:
            if "?" in a and not a.startswith("@"):
                for m in self.me:
                    if _matches(a, m):
                        return "wildcard", m
        return None, ""

    def _replies(self, f: W.Frame, r: Received) -> list[int]:
        """What a call asked us to report, as message words for the response."""
        out = list(self._dtm_replies)
        if any(q["request"] for q in f.lqa):
            # Our measurement of the call just heard (A.5.4.1).
            out.append(W.lqa_word(W.ber_code(r.ber), W.sinad_code(r.sinad_db)))
        if f.version_query and f.version_query[1] & 1:
            out += W.amd_words(VERSION_TEXT)            # AMD is the format we have
        if f.capabilities_query:
            out += W.capabilities_report(
                scan_rate=round(self.cfg.scan_rate) if self.cfg.scanning else 0,
                channels=len(self.cfg.channels) if self.cfg.scanning else 1,
                tune_100ms=round(self.cfg.tune_s * 10),
                protocols=(W.VAP_ALLCALL if self.cfg.answer_allcall else 0) | W.VAP_AMD)
        return out

    def _reports(self, f: W.Frame, station: str) -> None:
        """Reports another station sent us: LQA, capabilities."""
        for q in f.lqa:
            if not q["request"] and (q["ber"] != 31 or q["sinad"] != 31):
                self._emit("LQA", peer=station, channel=self.channel,
                           lqa={"ber": q["ber"], "sinad": q["sinad"]})
        if f.capabilities is not None:
            c = f.capabilities
            self._emit("CAPABILITIES", peer=station, channel=self.channel,
                       text=" ".join(f"{k}={v}" for k, v in c.items()))

    # -- LQA -------------------------------------------------------------------

    def _note(self, station: str, r: Received, now: float) -> None:
        self.lqa[(station, self.channel)] = _Lqa(r.ber, r.sinad_db, now)

    def _score(self, station: str, channel: int) -> float:
        q = self.lqa.get((station, channel))
        return -1e9 if q is None else q.sinad - q.ber

    def _lqa_dict(self, station: str) -> dict:
        q = self.lqa.get((station, self.channel))
        return {} if q is None else {"ber": round(q.ber, 1), "sinad": round(q.sinad, 1)}


def _first3(a: str) -> str:
    return (a + "@@")[:3]


def _matches(called: str, mine: str) -> bool:
    """A called address against one of ours, with "?" as a wildcard for one
    character (A.5.2.4.9)."""
    if len(called) != len(mine):
        return False
    return all(c == m or c == "?" for c, m in zip(called, mine))
