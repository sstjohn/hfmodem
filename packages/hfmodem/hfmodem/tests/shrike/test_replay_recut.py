# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The repeat that re-cuts a field the host has already read.

WS8EOC, 40 m, 2026-09-13 (`working/pactor3-header-0913`). Two arms an hour
apart delivered the gateway's greeting with a segment of it twice:

    W9SSJ has 82 daily mSSJ has 82 daily minutes remaining   (17:14)
    W9SSJ has 77 daily mSSJ has 77 daily minutes remainin    (17:48)

Both transcripts read the same. The gateway keyed its greeting as one 23-byte
speed-level-2 field under counter 0, this station acknowledged it CS1 and put
all 23 bytes on the host port -- and the gateway, which never read that CS1,
came back at speed level 1 with the same counter and the first FIVE of those
bytes. The counter law caught that copy and threw it away, exactly as it
should; what it cannot see is that the peer has re-cut its field, so the three
counters behind the copy carried bytes 5-22 a second time under numbers that
were genuinely new.

`_note_recut` is what reads the re-cut, and the distance it measures is spent
in `_accept_field`. `test_without_the_recut_arithmetic_the_same_air_duplicates`
is the negative control: the same six cycles with that reading taken out
produce the 17:14 line character for character.

The last scenes are the other half of the same gateway's stall. On 2026-09-13
18:49 it repeated its CS3-headed changeover packet 35 times and drew 46 CS1s,
because the repeat gear declined to count a break-in; the 18:37 arm, stalled on
an ordinary packet, drew CS4 on its fourth repeat and read the whole greeting.

Run:  python -m pytest hfmodem/tests/shrike/test_replay_recut.py
"""
from __future__ import annotations

from hfmodem.shrike import arq, rxfront, spec
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol

#: The 17:14 stint, cycle by cycle: (speed level, status byte, field). The
#: four copies of the re-cut packet are the four the gateway keyed.
RECUT_STINT = (
    (2, 0x20, b"0\r\nW9SSJ has 82 daily m"),
    (1, 0x00, b"0\r\nW9"),
    (1, 0x00, b"0\r\nW9"),
    (1, 0x00, b"0\r\nW9"),
    (1, 0x00, b"0\r\nW9"),
    (1, 0x21, b"SSJ h"),
    (1, 0x22, b"as 82"),
    (2, 0x23, b" daily minutes remainin"),
    (2, 0x00, b"g with WS8EOC (EN72QQ)\r"),
)

#: What the gateway meant to send: its distinct fields, in order.
GREETING = b"0\r\nW9SSJ has 82 daily minutes remaining with WS8EOC (EN72QQ)\r"

#: And what the host read on the air.
DUPLICATED = (b"0\r\nW9SSJ has 82 daily m" b"SSJ h" b"as 82"
              b" daily minutes remainin" b"g with WS8EOC (EN72QQ)\r")

CHANGEOVER_SL = 1          # the two-carrier comb reports PACTOR-3's level 1
CHANGEOVER_FIELD = b"RMS"


class Seam:
    """A transmit seam that keys nothing and remembers every codeword.

    `send_p1_cs` is what routes `PtcHost.send_cs` through `_counter_cs_for`, so
    the words collected here are the physical ones the alternation produces.
    """

    # Do not let __getattr__'s truthy no-op impersonate a deferred TX queue.
    defer_p3_cs = False

    def __init__(self) -> None:
        self.words: list[int] = []

    def attach(self, host) -> None: ...

    def send_cs(self, index: int) -> None:
        self.words.append(index)

    def send_p1_cs(self, index: int) -> None:
        self.words.append(index)

    def send_packet(self, sl, payload, status, breakin=False) -> int:
        return len(payload)

    def __getattr__(self, name):
        return lambda *a, **k: None


def linked(*, role: int = arq.IRS, repeat_gear: int = 0) -> tuple[PtcHost, Seam]:
    """A PACTOR-3 link in the state the gateway's break-in left.

    `speed_up_after` is parked out of reach and `long_cycle` is off, as both
    arms flew them, so the only codewords these scenes can produce are the
    alternation's own and whatever the repeat gear substitutes for one.
    """
    seam = Seam()
    host = PtcHost(peer=seam, mycall="W9SSJ")
    host.arq.cfg.repeat_gear = repeat_gear
    host.arq.cfg.speed_up_after = 1000
    host.arq.cfg.long_cycle = False
    host.arq.role, host.arq.dxcall = role, "WS8EOC"
    host.arq._enter_connected()
    host.protocol = Protocol.PACTOR3
    return host, seam


def arrives(host: PtcHost, sl: int, status: int, field: bytes, *,
            breakin: bool = False) -> None:
    """One cycle of the peer's, through the receiver's own event."""
    host.on_rx_event(rxfront.Event(0.1, "packet", "p3",
                                   protocol=Protocol.PACTOR3, breakin=breakin,
                                   packet=(sl, status, field, True),
                                   cycle_long=False))


def delivered(host: PtcHost) -> bytes:
    return bytes(host.channel(host.ptchn).rx)


def stint(host: PtcHost, cycles=RECUT_STINT) -> None:
    for sl, status, field in cycles:
        arrives(host, sl, status, field)


def status_at(seq: int, *, data_type: int = spec.DataType.ASCII_8BIT) -> int:
    return spec.status_byte(seq, data_type=data_type)


def test_the_recut_greeting_reaches_the_host_exactly_once():
    """The 17:14 stint, and the text the gateway keyed."""
    host, _ = linked()
    stint(host)
    assert delivered(host) == GREETING


def test_without_the_recut_arithmetic_the_same_air_duplicates(monkeypatch):
    """The negative control: the transcript's own line, character for character."""
    monkeypatch.setattr(arq.PactorArq, "_note_recut",
                        lambda self, payload, data_type: None)
    host, _ = linked()
    stint(host)
    assert delivered(host) == DUPLICATED
    assert b"82 daily mSSJ has 82 daily" in delivered(host)
    assert DUPLICATED != GREETING


def test_every_copy_of_the_recut_packet_draws_the_word_that_acknowledged_it():
    """A repeat is answered, and answered the same way: counter 0 is CS1."""
    host, seam = linked()
    stint(host)
    assert seam.words[:5] == [arq.CS_ACK] * 5
    assert seam.words == [arq.CS_ACK] * 5 + [arq.CS_REQUEST, arq.CS_ACK,
                                             arq.CS_REQUEST, arq.CS_ACK]


def test_the_counter_still_advances_through_the_recut():
    host, _ = linked()
    stint(host)
    assert host.arq.rx_seq == 0
    assert host.arq.rx_progress == 5


def test_a_repeat_of_the_same_field_holds_nothing_back():
    """Memory-ARQ's own case: the same counter, the same bytes, no re-cut."""
    host, _ = linked()
    arrives(host, 2, status_at(0), b"the whole field")
    arrives(host, 2, status_at(0), b"the whole field")
    arrives(host, 2, status_at(1), b" and the next")
    assert delivered(host) == b"the whole field and the next"


def test_a_repeat_that_is_not_a_prefix_holds_nothing_back():
    """Shorter is not enough: the bytes have to be the ones already delivered."""
    host, _ = linked()
    arrives(host, 2, status_at(0), b"the whole field")
    arrives(host, 1, status_at(0), b"other")
    arrives(host, 1, status_at(1), b" and the next")
    assert delivered(host) == b"the whole field and the next"


def test_a_speed_up_recut_delivers_only_the_tail():
    """The mirror: the repeat EXTENDS the field, and the extension is bytes the
    host has never had."""
    host, _ = linked()
    arrives(host, 1, status_at(0), b"the w")
    arrives(host, 2, status_at(0), b"the whole field")
    arrives(host, 2, status_at(1), b" and the next")
    assert delivered(host) == b"the whole field and the next"


def test_a_longer_repeat_that_is_not_a_prefix_delivers_nothing():
    """Longer is not enough either: the bytes already delivered have to be the
    ones it opens with, or there is no boundary to measure a tail from."""
    host, _ = linked()
    arrives(host, 1, status_at(0), b"the w")
    arrives(host, 2, status_at(0), b"something else entirely")
    arrives(host, 2, status_at(1), b" and the next")
    assert delivered(host) == b"the w and the next"


def test_an_idle_repeat_holds_nothing_back():
    """An empty field is a prefix of everything and evidence of nothing."""
    host, _ = linked()
    arrives(host, 2, status_at(0), b"the whole field")
    arrives(host, 2, status_at(0), b"")
    arrives(host, 2, status_at(1), b" and the next")
    assert delivered(host) == b"the whole field and the next"


def test_a_coded_field_is_left_where_it_is():
    """Under Huffman a prefix of the wire says nothing about a count of
    characters, so the re-cut reading declines to guess."""
    coded = status_at(0, data_type=spec.DataType.HUFFMAN)
    host, _ = linked()
    arrives(host, 2, coded, bytes([0x40, 0x21, 0x08, 0x63]))
    arrives(host, 1, coded, bytes([0x40, 0x21]))
    arrives(host, 1, status_at(1), b"plain")
    assert delivered(host).endswith(b"plain")


def test_the_first_field_after_a_reversal_is_still_delivered():
    """The reversal resets the numbering; the field behind it is new."""
    host, _ = linked()
    arrives(host, 2, status_at(3), b"the peer's last field")
    host.arq._reset_rx_seq()
    arrives(host, 2, status_at(0), b"and its first after the turn")
    assert delivered(host) == b"the peer's last field" \
                              b"and its first after the turn"


def test_the_replay_after_a_reversal_is_still_suppressed():
    """The other half of the same seam: same bytes, same coding, not delivered."""
    host, _ = linked()
    arrives(host, 2, status_at(3), b"the peer's last field")
    host.arq._reset_rx_seq()
    arrives(host, 2, status_at(0), b"the peer's last field")
    arrives(host, 2, status_at(1), b" and then the new one")
    assert delivered(host) == b"the peer's last field and then the new one"


def test_the_counter_wraps_from_three_to_zero_without_a_repeat():
    host, _ = linked()
    for seq, field in enumerate((b"zero ", b"one ", b"two ", b"three ")):
        arrives(host, 2, status_at(seq), field)
    arrives(host, 2, status_at(0), b"and around again")
    assert delivered(host) == b"zero one two three and around again"
    assert host.arq.rx_progress == 5


def _changeover(host: PtcHost, times: int) -> None:
    for _ in range(times):
        arrives(host, CHANGEOVER_SL, 0x00, CHANGEOVER_FIELD, breakin=True)


def test_a_repeated_changeover_packet_draws_a_gear_on_the_fourth():
    """18:49's stall, with the flag the 18:37 arm flew."""
    host, seam = linked(role=arq.ISS, repeat_gear=3)
    _changeover(host, 4)
    assert seam.words == [arq.CS_ACK] * 3 + [arq.CS_SPEED_UP]
    assert host.arq.role == arq.IRS


def test_the_changeover_field_is_delivered_once_however_often_it_repeats():
    host, _ = linked(role=arq.ISS, repeat_gear=3)
    _changeover(host, 8)
    assert delivered(host) == CHANGEOVER_FIELD
    assert host.arq.rx_progress == 1


def test_the_flag_at_zero_answers_every_changeover_repeat_with_CS1():
    """35 repeats and 46 CS1s: what the arm that died actually keyed."""
    host, seam = linked(role=arq.ISS, repeat_gear=0)
    _changeover(host, 12)
    assert seam.words == [arq.CS_ACK] * 12
    assert arq.CS_SPEED_UP not in seam.words


def test_the_gear_run_restarts_behind_its_own_request():
    """A gateway that goes on repeating is asked again, not asked every cycle."""
    host, seam = linked(role=arq.ISS, repeat_gear=3)
    _changeover(host, 8)
    assert seam.words.count(arq.CS_SPEED_UP) == 2
    assert seam.words[3] == seam.words[7] == arq.CS_SPEED_UP
