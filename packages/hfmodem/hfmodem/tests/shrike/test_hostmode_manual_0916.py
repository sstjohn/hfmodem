# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The hostmode chapter's own rules, where the emulation had been guessing.

Written from the PTC-IIIusb 4.1 manual chapter 10 (SCS_Manual_PTC-IIIusb_4.1,
line numbers of its text extraction; §10.4.2 D at
:6847, §10.4.8 L at :6907, §10.4.31 %L at :7184, §10.7 TRX channel at :7405,
§10.9.6 hostmode start at :7595, §6.49 Listen at :3492, §6.90 STatus at :4324)
and from the three client-side sources the 0916 audit could not reach:

  WA8DED host mode user's guide -- https://www.ir3ip.net/iw3fqg/doc/wa8ded.htm
      The reply-code table the SCS manual never prints: 0 success/nothing,
      1 success with a NUL-terminated message, 2 failure with one, 3 link
      status, 4/5 monitor header, 6/7 counted bodies. "If the command is
      successful and returns information (like any command without an
      argument), a code 1 transmission will be the reply." Its four failure
      texts are INVALID COMMAND, TNC BUSY - LINE IGNORED, CHANNEL ALREADY
      CONNECTED and STATION ALREADY CONNECTED -- CHANNEL NOT CONNECTED is not
      among them and is not in the SCS manual either, so it is not emitted here
      (unknowns.HOST_REFUSAL_TEXTS). Link-status texts include
      "({n}) LINK FAILURE with {call}", which §10.4.10 also names. L is sent as
      a bare command to the channel of interest, and channel 0 answers with two
      fields, not six.

  harenber/ptc-go (Pat's PACTOR driver) -- pactor/modem.go
      JHOST4 in, JHOST0 out, CRC hostmode, PACTOR channel 31. Terminal init
      before hostmode: MYcall, PTCH 31, MAXE 35, REM 0, CHOB 0, TONES 4,
      MARK 1600, SPACE 1400, CWID 0, CONType 3, MODE 0. In hostmode it sends
      bare `L` on channel 31 (never with an argument, and it refuses to send it
      to channel 0 at all), `G` on 255 and on 31, `C <call>`, `D`, and `DD` for
      its forced disconnect. It never sends %L.

  BPQ32 SCSPactor.c -- https://github.com/g8bpq/linbpq/blob/master/SCSPactor.c
      JHOST4 as well. Terminal init ends "LISTEN 0", "STATUS 2", "PTCHN 31",
      "PDTIMER 5", "PDUPLEX 1", "MYCALL <call>", with the comment "Automatic
      Status must be enabled for BPQ32" -- and it serves inbound connects with
      LISTEN 0 set, which is what proves listen mode is monitoring and not the
      gate on answering a call. In hostmode: `I<call>` per stream with no
      space, bare `L` on the stream's channel, `G` on 255 and 254, `%T`, `@B`,
      `%W0`/`%W1` for scan control, and the undocumented `#` prefix ("a hidden
      feature where you can send any normal mode command in host mode by
      preceeding with a #") for `#DD` and `#MYL <n>`.

    python3 tests/shrike/test_hostmode_manual_0916.py
"""
from __future__ import annotations

import select
import sys
import time
from types import SimpleNamespace

from hfmodem.shrike import hostmode
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost, SimPeer, serve

from hfmodem.tests.shrike.test_ptc import PACTOR_CH, Master, check


def open_hostmode(mode: str = "JHOST4", **kwargs) -> tuple[PtcHost, Master]:
    host = PtcHost(mycall="N0CALL", **kwargs)
    m = Master(host)
    host.open()
    m.terminal(f"PTCH {PACTOR_CH}")
    m.terminal(mode)
    return host, m


def connected(host: PtcHost, m: Master) -> None:
    m.cmd(PACTOR_CH, "C N0DX")
    for _ in range(40):
        host.tick()
        if m.link_status()[5] == 4:
            return
    raise AssertionError("link never came up")


def test_reentry_runs_the_replayed_command() -> None:
    """§10.9.6: at a hostmode start the modem's REQUEST bit is undefined again.

    The reproduction from the audit: leave with JHOST0, come back with JHOST4,
    and send the byte-identical frame that left. Held state answered it from the
    previous session's buffer and never ran it, so the client's first command
    after reconnecting was silently dropped.
    """
    host, m = open_hostmode()
    m.cmd(PACTOR_CH, "L")
    leave = hostmode.command(0, "JHOST0", counter=1)
    host.feed(leave)
    check("JHOST0 left hostmode", not host.hostmode)

    m.terminal("JHOST4")
    host.feed(leave)
    check("a byte-identical frame after re-entry is executed, not replayed",
          not host.hostmode)


def test_a_served_modem_answers_a_call() -> None:
    """%L/Listen default is 1 [§10.4.31, §6.49], and it is not the answering gate.

    `serve()` is the modem on a wire, so it applies the default; the audit found
    a served PtcHost taking no call at all until a client sent %L1, which
    neither ptc-go nor BPQ32 ever sends.
    """
    host = PtcHost(SimPeer(), mycall="N0CALL")
    check("a bare PtcHost is not listening", host.arq.state == State.DISCONNECTED)

    real_select = select.select
    select.select = _interrupt
    try:
        serve(host, verbose=False)
    finally:
        select.select = real_select
    check("serve() arms the receiver before its first read",
          host.arq.state == State.LISTENING, str(host.arq.state))

    host.arq.on_rx_connect("N0CALL", "N0DX")
    check("...so an inbound call is answered",
          host.arq.state in (State.CONNECTING, State.CONNECTED), str(host.arq.state))

    # BPQ32 sends LISTEN 0 in terminal mode and serves inbound connects anyway.
    host = PtcHost(mycall="N0CALL")
    m = Master(host)
    host.open()
    host.set_listen(True)
    m.terminal("LISTEN 0")
    check("terminal LISTEN 0 reaches the modem", host.settings["LISTEN"] == "0")
    check("...and does not stop it answering calls",
          host.arq.state == State.LISTENING, str(host.arq.state))
    m.terminal(f"PTCH {PACTOR_CH}")
    m.terminal("JHOST4")
    m.cmd(PACTOR_CH, "%L0")
    check("%L 0 does not either", host.arq.state == State.LISTENING,
          str(host.arq.state))
    check("%L reads back", m.cmd(PACTOR_CH, "%L").data == b"0")


def _interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


def test_esc_prefixed_terminal_command() -> None:
    """"<ESC>JHOST4<CR>" is the manual's own form [§10.9.6]; ptc-go sends it bare."""
    host = PtcHost(mycall="N0CALL")
    m = Master(host)
    host.open()
    check("an ESC-prefixed command is read",
          "4.1" in m.terminal("\x1bVERsion"))
    m.terminal("\x1bJHOST4")
    check("...including the one that enters hostmode", host.hostmode)


def test_plain_hostmode() -> None:
    """JHOST1 is plain WA8DED hostmode [§10.4.6]: the same packets, no envelope.

    No client we can read uses it -- ptc-go and BPQ32 both send JHOST4 -- but
    the parameter is documented, and ignoring it left the modem echoing WA8DED
    packets back at a line editor.
    """
    host, _ = open_hostmode("JHOST1")
    check("JHOST1 enters hostmode", host.hostmode)

    frame = hostmode.unwrap(hostmode.command(PACTOR_CH, "L", counter=0))
    out = host.feed(frame)
    check("no CRC envelope on the wire", not out.startswith(hostmode.HEADER), out.hex())
    (resp,) = list(hostmode.Decoder("master", crc=False).feed(out))
    check("plain hostmode answers the same L vector",
          (resp.channel, resp.code) == (PACTOR_CH, hostmode.MSG), str(resp))

    # The whole surface is the same one, so a link runs over it unchanged.
    host, _ = open_hostmode("JHOST1", peer=SimPeer())
    decoder = hostmode.Decoder("master", crc=False)

    def plain(frame: bytes) -> list:
        return list(decoder.feed(host.feed(hostmode.unwrap(frame))))

    plain(hostmode.command(PACTOR_CH, "C N0DX", counter=0))
    for _ in range(40):
        host.tick()
        if host.arq.state == State.CONNECTED:
            break
    plain(hostmode.data(PACTOR_CH, b"HELLO\r", counter=1))
    for _ in range(60):
        host.tick()
        if host.peer.received:
            break
    check("a link runs over plain hostmode too",
          bytes(host.peer.received) == b"HELLO\r", str(bytes(host.peer.received)))

    host, m = open_hostmode()
    check("JHOST4 still frames its replies", host.feed(
        hostmode.command(PACTOR_CH, "L", counter=0)).startswith(hostmode.HEADER))


def test_parameter_read_back() -> None:
    """A command with no argument answers code 1 with the value [WA8DED guide]."""
    host, m = open_hostmode()
    for verb, value in (("Y", "5"), ("N", "12"), ("K", "1")):
        check(f"{verb} {value} accepted",
              m.cmd(PACTOR_CH, f"{verb} {value}").code == hostmode.OK)
        resp = m.cmd(PACTOR_CH, verb)
        check(f"{verb} reads back", (resp.code, resp.data) == (hostmode.MSG,
                                                              value.encode()), str(resp))
    check("an unset parameter reads back 0",
          m.cmd(PACTOR_CH, "U").data == b"0")
    check("%M reads back", m.cmd(PACTOR_CH, "%M").data == b"0")
    resp = m.cmd(PACTOR_CH, "%M1")
    check("%M above the supported level is refused with the maximum [§10.4.32]",
          (resp.code, resp.data) == (hostmode.FAIL, b"0"), str(resp))


def test_l_takes_a_channel_argument() -> None:
    """§10.4.8 gives L a channel parameter; the WA8DED guide sends it bare."""
    host, m = open_hostmode()
    bare = m.cmd(PACTOR_CH, "L").data
    check("bare L on the PACTOR channel answers six fields",
          len(bare.split()) == 6, bare.decode())
    elsewhere = m.cmd(0, f"L {PACTOR_CH}").data
    check("L <ch> answers for the channel in the argument, wherever it arrives",
          elsewhere == bare, elsewhere.decode())
    own = m.cmd(0, "L").data
    check("channel 0's own status is two fields [WA8DED guide]",
          len(own.split()) == 2, own.decode())


def test_disconnect_twice_breaks_the_link() -> None:
    """"If the Disconnect command is given twice ... the link is broken
    immediately" [§10.4.2]."""
    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    live = (State.CONNECTED, State.DISCONNECTING)
    check("first D is graceful", m.cmd(PACTOR_CH, "D").code == hostmode.OK)
    check("...and the link is still up", host.arq.state in live, str(host.arq.state))
    check("second D is accepted", m.cmd(PACTOR_CH, "D").code == hostmode.OK)
    check("...and breaks the link immediately", host.arq.state not in live,
          str(host.arq.state))

    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    m.cmd(PACTOR_CH, "D")
    m.link_status()
    m.cmd(PACTOR_CH, "G")
    m.cmd(PACTOR_CH, "D")
    check("the polls every client sends in between do not spend the first D",
          host.arq.state not in live, str(host.arq.state))

    # BPQ32's dirty disconnect goes through the '#' escape, so the terminal
    # commands behind it -- DD [§6.34] and Disconnect [§6.37] -- have to exist.
    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    check("#DD is accepted", m.cmd(PACTOR_CH, "#DD").code == hostmode.OK)
    check("...and breaks the link [BPQ32]", host.arq.state not in live,
          str(host.arq.state))


def test_refusals_and_link_failure() -> None:
    """Code 2 where the sources give one, and §10.4.10's LINK FAILURE."""
    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    resp = m.cmd(PACTOR_CH, "C N0DX")
    check("a second connect is refused [WA8DED guide]",
          (resp.code, resp.data) == (hostmode.FAIL, b"CHANNEL ALREADY CONNECTED"),
          str(resp))

    # No peer: the call is never answered, so it fails rather than disconnects.
    host, m = open_hostmode()
    m.cmd(PACTOR_CH, "C N0DX")
    m.cmd(PACTOR_CH, "D")
    events, _ = m.drain()
    check("a call that never came up is a LINK FAILURE [§10.4.10]",
          events == [f"({PACTOR_CH}) LINK FAILURE with N0DX"], str(events))

    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    m.cmd(PACTOR_CH, "DD")
    events, _ = m.drain()
    check("a link that came up disconnects",
          events == [f"({PACTOR_CH}) CONNECTED to N0DX",
                     f"({PACTOR_CH}) DISCONNECTED fm N0DX"], str(events))


def test_wait_state() -> None:
    """%W0's "1" is a promise to take no call for ten seconds [§10.4.37]."""
    host, m = open_hostmode(peer=SimPeer())
    host.set_listen(True)
    check("%W0 says it is safe to retune", m.cmd(PACTOR_CH, "%W0").data == b"1")
    check("...and the modem has entered the WAIT state",
          host.arq.state != State.LISTENING, str(host.arq.state))
    host.arq.on_rx_connect("N0CALL", "N0DX")
    check("...which takes no call", host.arq.state == State.DISCONNECTED,
          str(host.arq.state))
    check("%W1 releases it", m.cmd(PACTOR_CH, "%W1").code == hostmode.MSG)
    check("...and the receiver is armed again", host.arq.state == State.LISTENING,
          str(host.arq.state))

    m.cmd(PACTOR_CH, "%W0")
    m.cmd(PACTOR_CH, "%W0")           # a scanner that asks twice still gets it back
    host.wait_until = time.monotonic() - 0.001
    host.tick()
    check("the WAIT state always times out on its own",
          host.wait_until is None and host.arq.state == State.LISTENING,
          str(host.arq.state))


def test_status_byte_and_auto_status() -> None:
    """Byte 4 [§10.6] and the automatic status BPQ32's driver requires [§10.6.1]."""
    host = PtcHost(SimPeer(), mycall="N0CALL")
    m = Master(host)
    host.open()
    m.terminal(f"PTCH {PACTOR_CH}")
    check("auto status is off until STatus 2", not host.autostatus)
    m.terminal("STATUS 2")            # BPQ32 sends this in its terminal init
    check("STatus 2 switches it on [§6.90]", host.autostatus)
    m.terminal("JHOST4")
    check("byte 4 reports 'no frequency estimate', not a confident zero",
          m.status(3)[3] == 128, str(m.status(3)[3]))
    m.status(3)
    m.drain()
    check("no status change, no channel 254", 254 not in m.poll_channels())

    m.cmd(PACTOR_CH, "C N0DX")
    check("a status change lists channel 254 in the 255 poll", 254 in m.poll_channels())
    m.status(3)
    check("...and polling 254 clears it [§10.6.1]", 254 not in m.poll_channels())

    cur = host._status_bytes()
    host._laststatus = bytes((cur[0], cur[1] ^ 1, cur[2], cur[3]))
    check("a change in byte 2 alone is not a status change [§10.6]",
          254 not in m.poll_channels())


def test_channel_busy() -> None:
    """"An occupied HF channel is indicated by a status value of 247" [§6.90].

    A signal decoded off the air while the modem is in standby is exactly the
    condition, and an automatic station holds its transmission off it.
    """
    host, m = open_hostmode()
    check("a clear channel in standby is 0x87", m.status(0) == b"\x87",
          m.status(0).hex())
    host.on_rx_event(SimpleNamespace(kind="detect", protocol=None, packet=None,
                                     text="carrier"))
    check("a signal on the channel is reported as 247", m.status(0) == b"\xf7",
          str(m.status(0)[0]))
    host.set_listen(True)
    check("...but never in listen mode [§6.90]", m.status(0) == b"\xe7",
          m.status(0).hex())
    host.set_listen(False)
    host.busy_until = time.monotonic() - 0.001
    check("...and it lapses", m.status(0) == b"\x87", m.status(0).hex())


def test_percent_t_and_per_channel_callsign() -> None:
    """%T is reset at the end of a connection [§10.4.35]; I is per channel [§10.4.5]."""
    host, m = open_hostmode(peer=SimPeer())
    connected(host, m)
    m.write(PACTOR_CH, b"HELLO WORLD\r")
    for _ in range(60):
        host.tick()
        if m.link_status()[2:4] == [0, 0]:
            break
    check("%T counts confirmed bytes", int(m.cmd(PACTOR_CH, "%T").data) > 0,
          m.cmd(PACTOR_CH, "%T").data.decode())
    m.cmd(PACTOR_CH, "DD")
    check("...and is reset when the connection ends",
          m.cmd(PACTOR_CH, "%T").data == b"0", m.cmd(PACTOR_CH, "%T").data.decode())

    check("I<call> with no space is the callsign, not a parameter [BPQ32]",
          m.cmd(PACTOR_CH, "IW1ABC").code == hostmode.OK)
    check("...readable back on its own channel",
          m.cmd(PACTOR_CH, "I").data == b"W1ABC")
    check("...and another channel keeps channel 0's", m.cmd(5, "I").data == b"N0CALL")
    m.cmd(PACTOR_CH, "C N0DX")
    check("the connect uses the channel's callsign", host.arq.mycall == "W1ABC")
    m.cmd(PACTOR_CH, "DD")
    check("after a disconnect the channel is back on channel 0's callsign [§10.4.5]",
          m.cmd(PACTOR_CH, "I").data == b"N0CALL")


def test_trx_and_nmea_channels() -> None:
    """Channel 253 [§10.7] and channel 249 [§10.8], neither of which has hardware."""
    host, m = open_hostmode()
    check("CAT bytes written to 253 are refused, not silently dropped",
          m.write(253, b"FA00014070000;").code == hostmode.FAIL)
    check("a poll of 253 has nothing to give", m.cmd(253, "G").code == hostmode.OK)
    check("the NMEA channel is simply empty", m.cmd(249, "G").code == hostmode.OK)


def test_reply_cannot_be_truncated() -> None:
    """A string body carrying a NUL would end the frame early [§10.9.3]."""
    frame = hostmode.reply_text(0, hostmode.FAIL, "bad\x00call")
    (resp,) = list(hostmode.Decoder("master").feed(frame))
    check("a NUL in a string body cannot truncate the frame",
          resp.data == b"badcall", str(resp))

    frame = hostmode.reply_text(0, hostmode.MSG, "x" * 400)
    (resp,) = list(hostmode.Decoder("master").feed(frame))
    check("an over-long string body is capped at the packet maximum",
          len(resp.data) == hostmode.MAX_DATA, str(len(resp.data)))


def test_request_condition() -> None:
    """§10.9.4's request-condition, and the one deviation this tree takes.

    A byte-identical packet under an unchanged counter means the master never
    saw the reply: repeat the buffered one and do not run the command again.
    The deviation is the other half of the rule -- the same counter carrying
    *different* bytes is treated as new work rather than discarded, because
    ptc-go writes data packets without advancing its counter. Pinned so that a
    later move to the strict reading is a decision and not a silence.
    """
    ran: list[str] = []

    def handler(pkt: hostmode.Packet) -> bytes:
        ran.append(pkt.text)
        return hostmode.reply_text(pkt.channel, hostmode.MSG, str(len(ran)))

    modem = hostmode.Modem(handler)
    first = modem.feed(hostmode.command(PACTOR_CH, "L", counter=0))
    check("the repeat is the buffered reply, verbatim",
          modem.feed(hostmode.command(PACTOR_CH, "L", counter=0)) == first)
    check("...and the command did not run twice", ran == ["L"], str(ran))

    again = modem.feed(hostmode.command(PACTOR_CH, "V", counter=0))
    check("a different body under the same counter is new work (deviation)",
          ran == ["L", "V"] and again != first, str(ran))

    modem.feed(hostmode.command(PACTOR_CH, "L", counter=0, ignore_counter=True))
    modem.feed(hostmode.command(PACTOR_CH, "L", counter=0, ignore_counter=True))
    check("bit 6 is always an ACK-condition [§10.9.4]", ran[-2:] == ["L", "L"], str(ran))

    before = len(ran)
    modem.reset()
    modem.feed(hostmode.command(PACTOR_CH, "L", counter=0))
    check("a hostmode start forgets the buffered reply [§10.9.6]",
          len(ran) == before + 1, str(ran))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    print("PTC hostmode: manual chapter 10 conformance\n")
    for fn in TESTS:
        print(f"{fn.__name__}:")
        fn()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
