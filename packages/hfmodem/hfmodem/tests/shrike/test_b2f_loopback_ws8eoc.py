# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A whole Winlink exchange over PACTOR-3, against WS8EOC as it actually flew.

The seams are the station's own -- `PtcHost`, `PactorArq`, `MailClient` and
`B2FSession` -- and the far end is scripted rather than simulated, for the
reason `tests/winlink/corpora.py` gives about its own arbiter: a second
`B2FSession` at the gateway end can cancel a shared error and stay green. This
gateway answers with bytes somebody else's software would have sent, cut into
the packets `working/pactor3-header-0913` watched WS8EOC key.

What it flew, holds 30-56 of `arm-v10-A-40-ws8eoc-c` (17:14) and the same seven
steps an hour later:

  * the entry packet is answered by a CS3-headed changeover packet carrying
    `RMS` -- the first three bytes of the greeting, and the link changing hands;
  * the greeting arrives in 5-byte speed-level-1 fields under the mod-4 counter,
    speeds up to 23-byte level 2 partway through, and once re-cut DOWNWARD under
    a counter it had already spent (`test_replay_recut`);
  * `CMS via WS8EOC >` closes it, the mail client writes its login, this end
    breaks in, and the gateway cedes with bit 6 standing on a 0-byte packet;
  * B2F from there: the login, `FF`, the gateway's `FC`/`F>` proposal, our
    `FS +`, the SOH/STX/EOT body, `FF`, `FQ`.

Six scenes, one harness. Run:

    python -m pytest hfmodem/tests/shrike/test_b2f_loopback_ws8eoc.py
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import numpy as np

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.shrike.compress import SB, Supervisor, bit_stream, transparent
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import IDLE, Protocol
from hfmodem.tests.shrike.test_iss_turn_changeover import iss_grid
from hfmodem.tests.shrike.test_p3_breakin_timing import tx_at
from hfmodem.winlink import (B2FSession, MailClient, compose, compress,
                             write_inbox)
from hfmodem.winlink.session import EOT, SOH, STX

#: `arm-v10-A-40-ws8eoc-b.launch.log`, the 15:56 arm's own `mail: peer said:`
#: lines rejoined. Four banner lines and a prompt.
GREETING = (b"RMS Trimode 1.4.3.0\r"
            b"W9SSJ has 94 daily minutes remaining with WS8EOC (EN72QQ)\r"
            b"[WL2K-5.0-B2FWIHJM$]\r"
            b";PQ: 14402315\r"
            b"CMS via WS8EOC >\r")

#: `ptc._answer_link_setup` under `--announce-lower`, requeued by the grant path
#: and sitting at the head of the buffer when the greeting completes.
ANNOUNCE = b"1w9ssj\r"

#: K0NTS over PACTOR-1, `working/onair-0909-2353/midnight-04-k0nts-p1`: a CMS
#: that names the host it dialled and reports the connection with `***` before
#: it says who it is.
K0NTS_GREETING = (b"Trying ec2-34-196-189-192.compute-1.amazonaws.com\r"
                  b"*** W9SSJ Connected to CMS\r"
                  b"[WL2K-5.0-B2FWIHJM$]\r"
                  b";PQ: 23296760\r"
                  b"CMS via K0NTS >\r")

#: K5FIT's prompt, off the same corpus: two spaces before the `>`.
K5FIT_PROMPT = b"CMS via K5FIT  >\r"

CHANGEOVER_FIELD = 3            # placement.CHANGEOVER.crc_bytes - 3
SL1_FIELD, SL2_FIELD = 5, 23    # the two cuts the arms watched the peer use
CEDE = 0x43                     # seq 3, changeover requested, 0 bytes, SL1

CYCLES = 400                    # the bench's budget, not the gateway's law


def offered():
    """The message the gateway holds for us, and its compressed block."""
    msg = compose("WS8EOC", "W9SSJ", "EN54 mail check",
                  b"one message, over PACTOR-3.\r\n", mid="B2FLOOPBCK1")
    return msg, compress(msg.render())


def proposal_block(*props: str) -> bytes:
    checksum = (-sum(sum(p.encode()) + 0x0D for p in props)) & 0xFF
    return ("".join(p + "\r" for p in props) + f"F> {checksum:02X}\r").encode()


def body_stream(blob: bytes, title: bytes = b"EN54 mail check") -> bytes:
    head = title + b"\0" + b"0" + b"\0"
    out = bytes((SOH, len(head))) + head
    for i in range(0, len(blob), 250):
        chunk = blob[i:i + 250]
        out += bytes((STX, len(chunk) & 0xFF)) + chunk
    return out + bytes((EOT, (-sum(blob)) & 0xFF))


def cms_script(blob: bytes, mid: str, usize: int):
    """What the gateway says, and what of ours makes it say it.

    Three triggers in order, each a complete line of ours: the `FF` that closes
    our first turn, the `FS` that answers the proposal, and the `FF` that closes
    the turn the delivered message gave back to us.
    """
    return [(b"FF\r", proposal_block(f"FC EM {mid} {usize} {len(blob)} 0")),
            (b"FS +\r", body_stream(blob)),
            (b"FF\r", b"FQ\r")]


class Ws8eoc:
    """The gateway: a PACTOR-3 far end with a scripted CMS above it.

    It holds the channel first, as the recorded arms' gateway did, and the whole
    of its link behaviour is the four things it can be doing -- keying its own
    stint, ceding to a changeover it has read, acknowledging ours, or taking the
    channel back off the turn we offered it.

    The knobs are the scenes, and each is one thing a recorded arm did:
    `repeat` keys every packet more than once (an acknowledgement a cycle late),
    `sl2_after` is the speed-up partway through the greeting, `recut` is the
    17:14 re-cut downward, `cede_delay` holds the cede back a cycle, and `deaf`
    names the emissions this end never decodes.
    """

    def __init__(self, script, *, out: bytes = GREETING, repeat: int = 0,
                 sl2_after: int | None = None, recut: tuple[int, int] | None = None,
                 cede_delay: int = 0, deaf: tuple[int, ...] = ()):
        self.script = list(script)
        # A gateway writes 0x1C and 0x1E as the supervisor sequence the
        # character stream reserves for them -- the escape `on_host_data` puts
        # on ours. A compressed body carries both, so a peer without it would be
        # scripting an impossibility.
        self.out = bytearray(transparent(out))
        self.repeat, self.sl2_after, self.recut = repeat, sl2_after, recut
        self.cede_delay, self.deaf = cede_delay, set(deaf)
        self.host = None
        self.mode = "taking"        # it breaks in on the entry packet
        self.tape: list[str] = []
        self.bursts: list[tuple[int, bytes, int, bool]] = []
        self.words: list[int] = []
        self.rx = bytearray()
        self.refuse = False
        self.cut, self.sl, self.seq = SL1_FIELD, 1, 0
        self.keyed = 0              # emissions in this sending stint
        self._held = 0              # repeats still owed on the packet in hand
        self._last = None           # (sl, status, field) for a repeat
        self._read_seq = None
        self._step = self._cursor = 0
        self._waited = self._recut_left = 0
        # The inverse of that escape, and stateful across fields on purpose: a
        # supervisor block spans a packet boundary as readily as it sits inside
        # one field.
        self._si = Supervisor()

    # -- the link seam PtcHost keys into ---------------------------------- #
    def attach(self, host: PtcHost) -> None:
        self.host = host

    def send_packet(self, sl, payload, status, breakin=False):
        self.bursts.append((sl, bytes(payload), status, breakin))
        if self.refuse:
            self.tape.append("us  SEAM REFUSED")
            return arq.REFUSED
        self.tape.append(f"us  SL{sl} {status:#04x}"
                         f"{' BK' if breakin else ''} {bytes(payload)!r}")
        if breakin and self.mode != "ceding":
            # The wait runs from the FIRST changeover: a re-key is this end
            # asking again, not the gateway starting over.
            self.mode, self._waited, self._read_seq = "ceding", 0, None
        self._read(payload, status)
        if status & spec.STATUS_CHANGEOVER and not breakin:
            self.mode = "taking"
        return len(payload)

    def send_cs(self, index):
        self.words.append(index)
        self.tape.append(f"us  CS{index + 1}")

    def send_p1_cs(self, index):
        self.send_cs(index)

    def pump(self):
        pass

    def cycle(self):
        pass

    # -- the gateway's own slot -------------------------------------------- #
    def key(self) -> None:
        if self.mode == "taking":
            self._take()
        elif self.mode == "sending":
            self._send()
        elif self.mode == "ceding":
            self._cede()
        elif self.mode == "receiving":
            self._acknowledge()

    def _take(self) -> None:
        """The CS3-headed changeover packet: three bytes, and the link with it."""
        field, self.out = bytes(self.out[:CHANGEOVER_FIELD]), \
            self.out[CHANGEOVER_FIELD:]
        self.seq, self.cut, self.sl, self.keyed = 0, SL1_FIELD, 1, 0
        self._arrives(1, 0x00, field, breakin=True)
        self.seq, self.mode = 1, "sending"

    def _send(self) -> None:
        if self._held:
            self._held -= 1
            sl, status, field = self._last
            self._arrives(sl, status, field)
            return
        if self.recut and self.keyed == self.recut[0] + 1:
            # 17:14: the gateway never read the acknowledgement of the field it
            # had just keyed at level 2, and came back at level 1 under the same
            # counter with the first five of those bytes.
            sl, status, field = self._last
            self.out[:0] = field
            self.cut, self.sl = SL1_FIELD, 1
            self.seq = status & spec.STATUS_SEQ
            self._recut_left, self.recut = self.recut[1], None
        if self.sl2_after is not None and self.keyed == self.sl2_after:
            self.cut, self.sl = SL2_FIELD, 2
        if not self.out:
            # Nothing left to say, and bit 6 held up on every packet until this
            # end takes the channel -- a standing invitation, not a report.
            self._arrives(self.sl, spec.STATUS_CHANGEOVER | self.seq, b"")
            return
        # Stop-and-wait: a field this end did not decode stays in hand, under
        # the counter it was keyed with, and goes out again next cycle.
        field = bytes(self.out[:self.cut])
        if self._arrives(self.sl, self.seq, field):
            if self._recut_left:
                self._recut_left -= 1       # the same five bytes, same counter
                return
            del self.out[:self.cut]
            self.seq = (self.seq + 1) % 4
            self._held = self.repeat

    def _cede(self) -> None:
        if self._waited < self.cede_delay:
            self._waited += 1
            self.tape.append("gw  (silent)")
            return
        if self._arrives(1, CEDE, b"", tag="cede"):
            self.mode = "receiving"
            # A later standing bit-6 packet does not confirm our CS3. This
            # scripted peer has read it in send_packet, so acknowledge that
            # packet explicitly before expecting the remaining login bytes.
            if self.host.arq.unconfirmed_breakin:
                self._acknowledge()

    def _acknowledge(self) -> None:
        """The codeword an IRS owes the packet it has just read: CS1 against an
        even counter, CS2 against an odd one."""
        if not self.bursts:
            return
        counter = self.bursts[-1][2] & spec.STATUS_SEQ
        self.tape.append(f"gw  CS{(counter & 1) + 1}")
        self.host.on_rx_event(rxfront.Event(
            0.0, "cs", "ws8eoc", protocol=Protocol.PACTOR3, cs=counter & 1))

    # -- plumbing ---------------------------------------------------------- #
    def _arrives(self, sl, status, field, *, breakin=False, tag="") -> bool:
        """One of the gateway's packets at this end's receiver -- or not."""
        self._last, self.keyed = (sl, status, field), self.keyed + 1
        heard = self.keyed - 1 not in self.deaf
        self.tape.append(f"gw  SL{sl} {status:#04x}"
                         f"{' BK' if breakin else ''} {field!r}"
                         f"{f'  [{tag}]' if tag else ''}"
                         f"{'' if heard else '  [not decoded]'}")
        if heard:
            self.host.on_rx_event(rxfront.Event(
                0.1, "packet", "ws8eoc", protocol=Protocol.PACTOR3,
                breakin=breakin, packet=(sl, status, field, True),
                cycle_long=False))
        return heard

    def _read(self, payload: bytes, status: int) -> None:
        """Our stream, as an IRS assembles it: a counter it has taken is a
        repeat of a packet it already holds."""
        seq = status & spec.STATUS_SEQ
        if seq == self._read_seq:
            return
        self._read_seq = seq
        self.rx += self._si.feed(payload)
        while self._step < len(self.script):
            trigger, reply = self.script[self._step]
            at = self.rx.find(trigger, self._cursor)
            if at < 0:
                return
            self._cursor, self._step = at + len(trigger), self._step + 1
            self.out += transparent(reply)
            self.tape.append(f"gw  queues {reply[:40]!r}")


# --------------------------------------------------------------------------- #

def linked(gw: Ws8eoc, *, announce: bytes = b"", **kw) -> tuple[PtcHost, B2FSession]:
    """A PACTOR-3 link the moment before the gateway's break-in.

    `speed_up_after` is parked out of reach and `long_cycle` is off, as every
    arm in `working/pactor3-header-0913` flew them. `announce` is the PACTOR-1
    callsign announcement the grant path puts back at the head of the buffer.
    """
    host = PtcHost(gw, mycall="W9SSJ")
    host.arq.cfg.speed_up_after = 1000
    host.arq.cfg.long_cycle = False
    host.arq.role, host.arq.dxcall = arq.ISS, "WS8EOC"
    host.arq._enter_connected()
    host.protocol = Protocol.PACTOR3
    host.arq._sl = 3
    gw.attach(host)
    if announce:
        host._setup_bytes = announce
        host.arq.on_host_data(announce)
    session = B2FSession("W9SSJ", role="calling", target="WS8EOC", **kw)
    host.app = MailClient(session, host.arq.on_host_data)
    host.app.link_up()
    return host, session


def run(host: PtcHost, gw: Ws8eoc, cycles: int = CYCLES) -> int:
    """Cycles until the exchange closes, or 0 if it did not.

    One cycle is the gateway's slot, the turn discipline `onair` runs between
    them, and then ours. A link that leaves CONNECTED has ended the run
    whatever the session thinks, and the tape says so.
    """
    for n in range(1, cycles + 1):
        if host.app.done:
            return n
        if host.arq.state is not arq.State.CONNECTED:
            gw.tape.append(f"--  the link left CONNECTED ({host.arq.state})")
            return 0
        gw.key()
        onair._mail_app_turns(host, host.app, wait_greeting=True)
        host.tick()
    return 0


def exchange(**kw) -> tuple[PtcHost, B2FSession, Ws8eoc, int]:
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())), **kw)
    host, session = linked(gw)
    return host, session, gw, run(host, gw)


# --------------------------------------------------------------------------- #
# Scene 1 -- the happy path, to a message on disk.

def test_scene1_a_message_reaches_disk_over_the_whole_pactor3_exchange(tmp_path):
    msg, _ = offered()
    host, session, gw, cycles = exchange()

    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert session.done and session.stage == "closed"
    assert [m.render() for m in session.inbox] == [msg.render()]
    assert [p.name for p in write_inbox(session, tmp_path)] == \
        [f"{msg.mid}.b2f"]
    assert (tmp_path / f"{msg.mid}.b2f").read_bytes() == msg.render()


def test_scene1_the_greeting_reaches_the_mail_client_byte_exact():
    host, session, gw, cycles = exchange()
    assert session.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert session.challenge == "14402315"
    assert session.remote_text[:2] == [
        "RMS Trimode 1.4.3.0",
        "W9SSJ has 94 daily minutes remaining with WS8EOC (EN72QQ)"]


def test_scene1_the_login_goes_out_once_and_in_order_behind_the_changeover():
    host, session, gw, cycles = exchange()
    login = b";FW: W9SSJ\r[HFM-0.1-B2FHM$]\r; WS8EOC DE W9SSJ\rFF\r"
    assert gw.rx.startswith(login), bytes(gw.rx)
    assert bytes(gw.rx).count(b";FW: W9SSJ") == 1
    first = next(b for b in gw.bursts if b[3])
    assert first[1] == b";FW", "the changeover's field is the login's head"


def test_scene1_the_transcript_holds_every_packet_both_ways():
    """The record the scene is for: each side's packets and each side's words."""
    host, session, gw, cycles = exchange()
    tape = gw.tape
    assert tape[0] == "gw  SL1 0x00 BK b'RMS'"
    assert any(line.startswith("us  CS") for line in tape)
    assert any(line == "gw  SL1 0x43 b''  [cede]" for line in tape)
    assert sum(line.startswith("us  SL") for line in tape) == len(gw.bursts)
    assert sum(line.startswith("us  CS") for line in tape) == len(gw.words)


def test_scene1_the_channel_changes_hands_by_changeover_packet_only():
    host, session, gw, cycles = exchange()
    ours = [b for b in gw.bursts if b[3]]
    assert len(ours) == 3, "the login, the FS and the FF -- one changeover each"
    assert all(b[2] & spec.STATUS_SEQ == 0 for b in ours)
    assert [b[1] for b in ours] == [b";FW", b"FS ", b"FF\r"]
    assert sum(1 for line in gw.tape if "[cede]" in line) == 3


# --------------------------------------------------------------------------- #
# Scene 2 -- the peer repeating packets, our acknowledgement a cycle late.

def test_scene2_a_repeating_gateway_still_completes_the_exchange():
    msg, _ = offered()
    host, session, gw, cycles = exchange(repeat=1)
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert [m.render() for m in session.inbox] == [msg.render()]


def test_scene2_a_repeat_is_answered_every_time_and_delivered_once():
    plain = exchange()[1]
    host, session, gw, cycles = exchange(repeat=1)
    assert len(gw.words) > len(gw.bursts), \
        "every copy of a packet draws a codeword"
    assert session.remote_text == plain.remote_text
    assert session.sent_text == plain.sent_text


# --------------------------------------------------------------------------- #
# Scene 3 -- the slot that was gone at the turn.

def test_scene3_a_refused_changeover_slot_costs_a_cycle_and_nothing_else():
    """The seam refuses the cycle the changeover was due in.

    `test_iss_login_advance` settles what the ARQ does with one; this is the
    same refusal with a mail exchange standing behind it.
    """
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    refused = []

    original = gw.send_packet

    def once(sl, payload, status, breakin=False):
        if breakin and not refused:
            refused.append(True)
            gw.refuse = True
            try:
                return original(sl, payload, status, breakin=breakin)
            finally:
                gw.refuse = False
        return original(sl, payload, status, breakin=breakin)

    gw.send_packet = once
    cycles = run(host, gw)
    assert refused, "the scene never reached the turn"
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert [m.render() for m in session.inbox] == [msg.render()]
    assert bytes(gw.rx).count(b";FW: W9SSJ") == 1


# --------------------------------------------------------------------------- #
# Scene 4 -- the speed-up, and the frames this end did not read.

def test_scene4_a_speed_up_partway_through_the_greeting_is_read_whole():
    host, session, gw, cycles = exchange(sl2_after=4)
    assert cycles, "\n".join(gw.tape)
    assert any("SL2" in line for line in gw.tape), "the gateway did speed up"
    assert not session.failure, session.failure
    assert session.remote_sid == "[WL2K-5.0-B2FWIHJM$]"


def test_scene4_frames_this_end_never_decoded_are_repeated_and_read():
    """Round 25: 35 of 35 level-2 frames failed to decode at the 21:35 arm."""
    host, session, gw, cycles = exchange(sl2_after=4, deaf=(5, 6, 7), repeat=1)
    assert cycles, "\n".join(gw.tape)
    assert any("[not decoded]" in line for line in gw.tape)
    assert not session.failure, session.failure
    assert session.remote_sid == "[WL2K-5.0-B2FWIHJM$]"


def speeds_up_mid_packet(gw: Ws8eoc, at: int) -> None:
    """The mirror of `recut`: the field just keyed goes back and is re-keyed at
    the LARGER cut, under the same counter, with eighteen more bytes behind it."""
    send, done = gw._send, []

    def recut_upward():
        if not done and gw.keyed == at:
            sl, status, field = gw._last
            gw.out[:0] = field
            gw.cut, gw.sl, gw.seq = SL2_FIELD, 2, status & spec.STATUS_SEQ
            done.append(True)
        send()

    gw._send = recut_upward


def downward_only(self, payload, data_type):
    """`_note_recut` as it stood before the upward arm: the shorter repeat read,
    the longer one left to be suppressed with the packet that carried it."""
    if not payload or self._last_rx_field is None or bit_stream(data_type):
        return
    last_type, last = self._last_rx_field
    if data_type == last_type and len(payload) < len(last) \
            and last.startswith(payload):
        self._rx_ahead = len(last) - len(payload)


def speeds_up_mid_greeting():
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    speeds_up_mid_packet(gw, 4)
    return msg, host, session, gw, run(host, gw)


def test_scene4_a_speed_up_recut_delivers_the_bytes_past_the_old_boundary():
    """The other half of `_note_recut`.

    The gateway keys five bytes at level 1, never reads the acknowledgement, and
    comes back at level 2 under the same counter with the same five bytes and
    eighteen more. The counter law suppresses the copy -- correctly, as far as
    it goes -- and the eighteen past the old boundary are characters the host
    has never had, so the re-cut reading hands that tail to the stream.

    Nothing on file shows a gateway doing this; here the loss would fall in
    banner text and the exchange would still close. Eighteen bytes out of a SID
    line, a `;PQ:` challenge, an `FC` proposal or a compressed body is a
    session.
    """
    msg, host, session, gw, cycles = speeds_up_mid_greeting()

    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert session.remote_text[:2] == [
        "RMS Trimode 1.4.3.0",
        "W9SSJ has 94 daily minutes remaining with WS8EOC (EN72QQ)"]
    assert session.remote_sid == "[WL2K-5.0-B2FWIHJM$]"
    assert session.challenge == "14402315"
    assert [m.render() for m in session.inbox] == [msg.render()]
    assert [line for line in host.log_lines if "re-cut counter" in line] == \
        ["the peer re-cut counter 3 from 5 bytes to 23 -> 18 characters the "
         "host had not had"]


def test_scene4_without_the_upward_arm_the_speed_up_recut_loses_the_tail(
        monkeypatch):
    """The negative control: the same air with the upward reading taken out.

    Eighteen bytes gone out of the middle of the banner, no line on the log,
    and the session closes clean around the hole -- which is why the scene is
    here and not left to a checksum to notice."""
    monkeypatch.setattr(arq.PactorArq, "_note_recut", downward_only)
    msg, host, session, gw, cycles = speeds_up_mid_greeting()

    assert cycles, "\n".join(gw.tape)
    assert session.remote_text[0] == \
        "RMS Trimode 1.4.3.ly minutes remaining with WS8EOC (EN72QQ)"
    assert not any("re-cut counter" in line for line in host.log_lines)
    assert not session.failure, "the hole fell in free text and nothing caught it"


def test_scene4_the_gateways_own_recut_does_not_duplicate_the_greeting():
    """17:14's stint: level 2 under counter n, then the first five of those same
    bytes at level 1 under the same counter, four times."""
    host, session, gw, cycles = exchange(sl2_after=3, recut=(3, 3))
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert session.remote_text[1] == \
        "W9SSJ has 94 daily minutes remaining with WS8EOC (EN72QQ)"
    assert session.remote_sid == "[WL2K-5.0-B2FWIHJM$]"


# --------------------------------------------------------------------------- #
# Scene 5 -- the cede a cycle late.

@pytest.mark.parametrize("delay", [1, 2])
def test_scene5_a_cede_on_a_later_cycle_still_carries_the_login(delay):
    msg, _ = offered()
    host, session, gw, cycles = exchange(cede_delay=delay)
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert bytes(gw.rx).count(b";FW: W9SSJ") == 1, bytes(gw.rx)
    assert [m.render() for m in session.inbox] == [msg.render()]


# --------------------------------------------------------------------------- #
# Scene 6 -- the CMS's own words.

def test_scene6_a_cms_that_reports_its_connection_with_stars_is_not_a_refusal():
    """K0NTS, 2026-09-10: `*** W9SSJ Connected to CMS` ahead of the SID."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())),
                out=K0NTS_GREETING)
    host, session = linked(gw)
    cycles = run(host, gw)
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert "*** W9SSJ Connected to CMS" in session.remote_text
    assert [m.render() for m in session.inbox] == [msg.render()]


def test_scene6_a_prompt_with_two_spaces_before_the_bracket_releases_the_gate():
    """K5FIT's own prompt line."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())),
                out=GREETING.replace(b"CMS via WS8EOC >\r", K5FIT_PROMPT))
    host, session = linked(gw)
    cycles = run(host, gw)
    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert [m.render() for m in session.inbox] == [msg.render()]


def test_scene6_a_half_delivered_star_line_is_not_yet_anything():
    """`arm-v10-A-40-ws8eoc-b`: the gateway answered our `FF` with `**` and the
    link went down before the rest of the line arrived. Two bytes with no
    terminator are not a verdict, and the record has to show them anyway."""
    msg, blob = offered()
    gw = Ws8eoc([(b"FF\r", b"**")], out=GREETING)
    host, session = linked(gw)
    run(host, gw, cycles=120)
    assert not session.failure, session.failure
    assert session.stage == "their turn"
    assert session.remote_text[-1] == "**"


def test_scene6_the_line_that_star_line_became_is_recorded_as_the_refusal():
    msg, blob = offered()
    refusal = b"*** Unknown client types are not allowed on production servers\r"
    gw = Ws8eoc([(b"FF\r", refusal)], out=GREETING)
    host, session = linked(gw)
    run(host, gw, cycles=200)
    assert session.done and session.failure
    assert session.remote_refusal == refusal.decode().strip()
    assert not session.inbox


# --------------------------------------------------------------------------- #
# The announcement the grant path requeues, and the body a compressor produces.

def test_the_queued_announcement_never_reaches_the_gateway_as_a_login():
    """0913-2143 keyed `1w9` eight times with the whole login behind it.

    `PtcHost._drop_stale_setup` is the fix and this is it end to end: the link
    is holding the announcement when the prompt arrives, and what the gateway
    reads is a login.
    """
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw, announce=ANNOUNCE)
    assert bytes(host.arq._outbuf) == ANNOUNCE
    cycles = run(host, gw)

    assert cycles, "\n".join(gw.tape)
    assert not session.failure, session.failure
    assert ANNOUNCE not in bytes(gw.rx)
    assert bytes(gw.rx).startswith(b";FW: W9SSJ")
    assert [m.render() for m in session.inbox] == [msg.render()]
    assert any("unsent link-setup bytes dropped" in line
               for line in host.log_lines)


def binary_mail():
    """A message whose compressed body carries both bytes the character stream
    keeps for itself, so the escape has to survive a five-byte field."""
    body = np.random.default_rng(7).integers(0, 256, 1200,
                                             dtype=np.uint8).tobytes()
    return compose("WS8EOC", "W9SSJ", "binary body", body, mid="BINARYBODY1")


def test_a_compressed_body_crosses_the_link_byte_exact_in_five_byte_fields():
    msg = binary_mail()
    blob = compress(msg.render())
    wire = transparent(blob)
    assert blob.count(SB) and blob.count(IDLE), "the specimen carries both"
    assert any(wire[i] == SB and (i + 1) % SL1_FIELD == 0
               for i in range(len(wire) - 1)), \
        "...and at least one escape is split across a field boundary"

    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    cycles = run(host, gw, cycles=1000)
    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert [m.render() for m in session.inbox] == [msg.render()]


# --------------------------------------------------------------------------- #
# Scene 3, the other half: the slot the changeover had nowhere to go in.

def at_the_turn() -> tuple[PtcHost, B2FSession, Ws8eoc]:
    """The exchange stopped on the cycle the break-in is armed for."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    for _ in range(CYCLES):
        if host.arq._breakin_pending or host.arq._breakin_armed:
            return host, session, gw
        gw.key()
        onair._mail_app_turns(host, host.app, wait_greeting=True)
        host.tick()
    raise AssertionError("the exchange never reached its turn")


def test_scene3_the_turn_a_mail_client_asks_for_is_placeable(tmp_path):
    """The login is queued, the break-in is armed, and the arm's own geometry
    has somewhere to put the changeover packet."""
    host, session, gw = at_the_turn()
    assert session.sent_text, "the client has an answer"
    assert bytes(host.arq._outbuf).startswith(b";FW")

    g = iss_grid()
    tx = tx_at(g, (g.anchor + 2 * g.cycle_n - g.anchor) // g.cycle_n + 1, tmp_path)
    assert tx._place_breakin() and tx.placed and tx.unplaceable is None


def test_scene3_a_slot_that_is_gone_costs_the_cycle_and_not_the_login(tmp_path):
    """NEGATIVE CONTROL. Nothing re-anchors a pair nothing corroborates, and a
    changeover with nowhere to be placed leaves the login where it was."""
    host, session, gw = at_the_turn()
    queued = bytes(host.arq._outbuf)

    g = iss_grid()
    g.cycles += onair.ONSET_MAX_CYCLES + 1
    assert g._p3_reply_timing() is None
    tx = tx_at(g, (g.anchor + 9 * g.cycle_n - g.anchor) // g.cycle_n + 1, tmp_path)
    assert tx._place_breakin() == ""
    assert "no fresh corroborated" in tx.unplaceable
    assert bytes(host.arq._outbuf) == queued


# --------------------------------------------------------------------------- #
# Where the budgets end: the arms that did not complete.

def test_a_gateway_this_end_cannot_read_ends_the_link_rather_than_hanging():
    """Round 25's 21:35 arm: 35 level-2 frames on the air, none decoded. The
    link is not left holding a channel nothing is coming back on."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())),
                sl2_after=4, deaf=tuple(range(5, 40)))
    host, session = linked(gw)
    assert run(host, gw) == 0
    assert not session.inbox and session.stage == "awaiting greeting"
    assert host.arq._qrt_pending or host.arq.state is not arq.State.CONNECTED
    assert any("[not decoded]" in line for line in gw.tape)


def test_a_cede_that_never_arrives_spends_the_retry_budget_and_says_goodbye():
    """0913-2143's own ending, with the mail exchange standing behind it."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    lost = []
    cede = gw._cede

    def deaf_cede():
        if len(lost) < host.arq.cfg.max_retries + 4:
            lost.append(True)
            gw.tape.append("gw  SL1 0x43 b''  [cede]  [not decoded]")
            return
        cede()

    gw._cede = deaf_cede
    assert run(host, gw) == 0
    assert len(lost) > host.arq.cfg.max_retries
    assert not session.inbox
    assert any("max retries" in line for line in host.log_lines)


def test_a_message_of_ours_crosses_the_changeover_as_binary_and_byte_exact():
    """The other direction, and the seam that makes it hard: the changeover
    packet's three-byte field is the head of an SOH block, and 0x1C and 0x1E
    inside the compressed body go out as the supervisor sequence.

    The gateway answers the proposal with `FS` alone and reports its own
    mailbox empty in the turn after -- WW2MI's shape, not both in one over.
    """
    body = np.random.default_rng(15).integers(0, 256, 700,
                                              dtype=np.uint8).tobytes()
    # THE DATE IS PART OF THE SPECIMEN, like the seed and the MID beside it.
    # `compose` stamps the current minute, so this body changed every sixty
    # seconds and the line below -- which asks for both control bytes to survive
    # the compressor -- was a coin toss against the clock: 76 minutes in 600
    # carry no 0x1E at all, and the scene fails in them before any modem runs.
    mine = compose("W9SSJ", "SMTP:op@example.net", "outbound binary", body,
                   mid="OUTBOUND001",
                   date=datetime(2026, 9, 10, 18, 32, tzinfo=timezone.utc))
    blob = compress(mine.render())
    assert blob.count(SB) and blob.count(IDLE)

    gw = Ws8eoc([(b"F> ", b"FS +\r"), (bytes((SOH,)), b"FF\r")])
    host, session = linked(gw, outbox=[mine])
    cycles = run(host, gw, cycles=2000)

    assert cycles, "\n".join(gw.tape[-20:])
    assert session.done and not session.failure, session.failure
    assert session.sent_mids == [mine.mid]
    rx = bytes(gw.rx)
    assert rx.startswith(b";FW: W9SSJ")
    assert f"FC EM {mine.mid} {len(mine.render())} {len(blob)} 0\r".encode() in rx
    assert rx[rx.index(bytes((SOH,))):] == body_stream(blob, b"outbound binary")


def test_the_close_out_is_still_in_the_buffer_when_the_session_calls_itself_done():
    """`FQ` is written in the same `feed` that ends the exchange, so it has not
    been transmitted yet -- and it is the teardown, not the session, that gets
    it onto the air. A driver that drops the link on `done` alone sends the
    gateway no close-out at all.
    """
    body = np.random.default_rng(15).integers(0, 256, 700,
                                              dtype=np.uint8).tobytes()
    mine = compose("W9SSJ", "SMTP:op@example.net", "outbound binary", body,
                   mid="OUTBOUND001")
    gw = Ws8eoc([(b"F> ", b"FS +\r"), (bytes((SOH,)), b"FF\r")])
    host, session = linked(gw, outbox=[mine])
    assert run(host, gw, cycles=2000)
    assert bytes(host.arq._outbuf) == b"FQ\r" and host._txbuf == 3
    assert b"FQ\r" not in bytes(gw.rx)

    host.arq.on_host_disconnect()
    for _ in range(30):
        gw.key()
        host.tick()
        if host.arq.state is not arq.State.CONNECTED:
            break
    assert bytes(gw.rx).endswith(b"FQ\r"), "the teardown carries the close-out"


# --------------------------------------------------------------------------- #
# What a gateway may say twice, and what it may not go on saying.

def test_a_gateway_that_repeats_its_banner_into_our_turn_is_carried():
    """KY4RY's own habit, and `_free_text`'s reason for existing: the gateway
    says its greeting again where a proposal was owed, and the exchange goes on.
    """
    msg, blob = offered()
    prop = proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0")
    gw = Ws8eoc([(b"FF\r", GREETING * 5 + prop),
                 (b"FS +\r", body_stream(blob)), (b"FF\r", b"FQ\r")])
    host, session = linked(gw)
    cycles = run(host, gw, cycles=1500)
    assert cycles, "\n".join(gw.tape[-20:])
    assert not session.failure, session.failure
    assert [m.render() for m in session.inbox] == [msg.render()]


def test_a_stream_that_never_stops_being_banner_is_named_and_stopped():
    """NEGATIVE CONTROL. Past the budget the far end is not being chatty; it is
    the line framing lost, and a session that says so beats one that absorbs it.
    """
    msg, blob = offered()
    prop = proposal_block(f"FC EM {msg.mid} {len(msg.render())} {len(blob)} 0")
    gw = Ws8eoc([(b"FF\r", GREETING * 6 + prop)])
    host, session = linked(gw)
    run(host, gw, cycles=1500)
    assert "lines of free text where the protocol was owed" in session.failure
    assert not session.inbox


def test_a_repeated_changeover_packet_delivers_its_field_once():
    """18:49: the gateway repeated its CS3-headed changeover 35 times."""
    msg, blob = offered()
    gw = Ws8eoc(cms_script(blob, msg.mid, len(msg.render())))
    host, session = linked(gw)
    take = gw._take

    def re_keyed():
        take()
        sl, status, field = gw._last
        for _ in range(11):
            gw._arrives(sl, status, field, breakin=True, tag="re-key")

    gw._take = re_keyed
    cycles = run(host, gw)
    assert cycles, "\n".join(gw.tape[:20])
    assert not session.failure, session.failure
    assert session.remote_text[0] == "RMS Trimode 1.4.3.0"
    assert [m.render() for m in session.inbox] == [msg.render()]
