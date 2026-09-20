# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The receiver reads what a real peer compresses, and the operator can read it.

Every compressed vector here is real off-air material, and every expectation is
plaintext that is known independently: the PACTOR-3 fields are DL6MAA
transmitting the SCS PTC-II manual (the corpus fixture), and the PACTOR-2 fields
are the HB9AK ground-truth bytes beside it. The counterexamples are the point of
the file as much as the positives -- a decoder handed the WRONG mode, or one
that carries tree state across fields, must fail these assertions, so a
regression in either direction goes red instead of quietly delivering soup.

Run:  pytest -q packages/hfmodem/hfmodem/tests/shrike/test_compress.py
"""
from __future__ import annotations

import pytest

from hfmodem.shrike import compress, spec
from hfmodem.shrike.arq import P1_SPEED_LEVEL, ArqIO, PactorArq
from hfmodem.shrike.spec import DataType, Protocol
from hfmodem.tests.kestrel import corpora

FIXTURES = corpora.REGRESS_FIXTURES

# DL6MAA, 2026: two consecutive PACTOR-3 SL3 long-cycle fields, PMC German
# (status type 4), payloads exactly as the CRC validated them. The first ends
# mid-run -- the eleven-dash rule under "Allgemeines" puts its run prefix in
# this field and its code and count in the next -- and the second ends mid-word.
P3_A = bytes.fromhex(
    "0cc344e28bd44cf811d94792bf1f4512766b4c9560450a0eb1e40570846101358b09"
    "bac24ac144e28bd48cfcb07968ca3094908c80e7d708f50f1b")
P3_B = bytes.fromhex(
    "a84719c36037ff0e305c1613748595a2679ffc3e16d96e66011c6158405d9767ad65"
    "d94d14933a56bdeb46fe1e83c1fc888cba3df608e04e01d95a02ca9750f38785de2f"
    "e9b78c29d910415ce17e559935e2c1467c4698094308303481cbb2cc1966d9b32b01"
    "7a0f29fc8bdde893981c8325ffec53712771931884690a2eca0042d7df3328833165"
    "232892b05be35924805edba5e0de12af5e500caa3284b16f42bac788572f2806a17e"
    "de822240e8fa3b92b05b63661609a0d7762938a81818678c1dbe9f49c0c80c915601"
    "a346161ce8c71e0104fb2a05d0048619a0172ea1ce96058853ac30da45a2a45807de"
    "a87f25514739f8a0de82948f2f2806a1fee9794f4fdd0f5007f13da4976118ac8827"
    "dda80474")

# HB9AK, 7.051 MHz: the first 31 data bytes of two PACTOR-2 SL3 ground-truth
# fields (regress .fields sidecar rows, less the last data byte, status, CRC).
# English-tree traffic; the tails are pure idle fill and must vanish.
P2_ENGLISH = bytes.fromhex(
    "340104564b4488f579783c3c1e1e0f8f87c7c3e3e1f1f078783c3c1e1e0f8f")
P2_HUFFMAN = bytes.fromhex(
    "123c80b887c7c3e3e1f1f078783c3c1e1e0f8f87c7c3e3e1f1f078783c3c1e")
IDLE_FILL = bytes.fromhex("0f8f87c7c3e3e1f1f078783c3c1e")

# DL6MAA's 200 Bd link-setup announcement (`test_p1data`, and an independent
# decoder reads the same characters off the same audio). Bit 4 of its status
# byte is the capability declaration bits 4-5 carry, not part of the data mode.
P1_ANNOUNCEMENT = bytes.fromhex("1438842e2d04f3f078783c3c1e1e0f8f87c7c3e3")
P1_ANNOUNCEMENT_STATUS = 0x35


def test_real_fields_decode_to_the_known_plaintext():
    assert compress.decompress(P3_A, DataType.PMC_GERMAN) == (
        b"\r\r\r             Kurzinfo zur Hayes-kompatiblen Firmware 2.6aH\r"
        b"             =============================================\r\r"
        b"Allgemeines\r-")
    assert compress.decompress(P2_ENGLISH, DataType.PMC_ENGLISH) == \
        b"moment with a "
    assert compress.decompress(P2_HUFFMAN, DataType.HUFFMAN) == b"of the "


def test_the_escape_restores_codepage_437():
    """Symbol 0x7F + seven raw bits is a character above 0x7F. Read as a
    character instead, every umlaut in this field opened a desynchronised
    stretch -- 'verf' was followed by 'Dduf D  Eexlst', not by gbar."""
    text = compress.decompress(P3_B, DataType.PMC_GERMAN)
    assert b"verf\x81gbar" in text            # u-umlaut, CP437
    assert b"sch\x84dliche" in text           # a-umlaut
    assert b"m\x94glich" in text              # o-umlaut
    assert b"keinerlei Vorkehrungen getroffen" in text


def test_each_field_is_coded_fresh():
    """The second field opens in clean plaintext on a fresh tree walk. A
    decoder that carries tree or context state across the boundary starts it
    with the first field's leftovers and fails here."""
    assert compress.decompress(P3_B, DataType.PMC_GERMAN).endswith(
        b"Datentransparen")
    stream = compress.Decoder()
    stream.feed(P3_B, DataType.PMC_GERMAN)
    assert stream.feed(P3_B, DataType.PMC_GERMAN).endswith(b"Datentransparen")


def test_the_run_length_seam_crosses_fields():
    """...but an unfinished run does carry: prefix at the end of one field,
    code and count at the start of the next, eleven dashes under the
    eleven-letter heading they underline."""
    stream = compress.Decoder()
    text = stream.feed(P3_A, DataType.PMC_GERMAN)
    text += stream.feed(P3_B, DataType.PMC_GERMAN)
    assert b"Allgemeines\r-----------\r\rDie Version 2.6aH" in text


def test_the_wrong_mode_reads_as_soup():
    """The mode is the packet's own declaration, never a guess -- and this is
    the counterexample that keeps the dispatch honest: the same bytes under
    the wrong trees carry none of the known plaintext."""
    for wrong in (DataType.HUFFMAN, DataType.PMC_ENGLISH,
                  DataType.PMC_GERMAN_SWAPPED):
        assert b"Kurzinfo" not in compress.decompress(P3_A, wrong)
    assert b"moment" not in compress.decompress(P2_ENGLISH, DataType.PMC_GERMAN)


def test_swapped_modes_invert_ascii_case_only():
    assert compress.decompress(P2_ENGLISH, DataType.PMC_ENGLISH_SWAPPED) == \
        b"MOMENT WITH A "


def test_idle_fill_decodes_to_nothing():
    assert compress.decompress(IDLE_FILL, DataType.PMC_ENGLISH) == b""


def test_reserved_mode_passes_through():
    blob = bytes(range(256))
    assert compress.decompress(blob, DataType.RESERVED) == blob


def test_supervisor_reservations_after_run_expansion():
    """The supervisor reserves two values; the preceding run layer reserves 1D.

    The old all-byte reference vector hid 1D inside a 1C supervisor block. It did
    not establish that an isolated 1D is transparent in uncompressed packets.
    """
    plain = bytes(b for b in range(256) if b not in (compress.SB, spec.IDLE))
    assert compress.Supervisor().feed(plain) == plain
    assert compress.Supervisor().feed(b"A\x1eB") == b"AB"
    assert compress.Supervisor().feed(b"A\x1c\x411 B") == b"AB"


def test_the_supervisor_sequence_carries_all_three_reserved_values():
    """`transparent` is what a B2F body goes out as, and it round-trips.

    The SIC alone carries the character -- MEASURED, `1c 7e 20` reads back as
    0x1E followed by an ordinary space -- so the encoder emits two bytes and
    the decoder must not eat a third."""
    blob = bytes(range(256))
    wire = compress.transparent(blob)
    assert len(wire) == len(blob) + 3
    assert spec.IDLE not in wire
    assert compress.Supervisor().feed(wire) == blob
    assert compress.transparent(b"\x1e") == b"\x1c\x7e"
    assert compress.transparent(b"\x1c") == b"\x1c\x7c"
    assert compress.transparent(b"\x1d") == b"\x1c\x7d"
    assert compress.Supervisor().feed(b"\x1c\x7e ") == b"\x1e "


@pytest.mark.parametrize('wire,expected', [
    (bytes.fromhex('1dff5cebc3'), b'\xff' * 92 + b'\xeb\xc3'),
    (bytes.fromhex('f31d4b9272'), b'\xf3\x72'),
    (bytes.fromhex('1dcc22b70d'), b'\xcc' * 34 + b'\xb7\r'),
])
def test_stock_p3_uncompressed_run_witnesses(wire, expected):
    """The SCS reference decoder, actual 2026-09-20 VE3KPG/WS8EOC DAC packets.

    The reference decoder renders CR as CRLF even when supervisor-quoted; that
    presentation newline is not part of this character decoder's output.
    """
    for split in range(len(wire) + 1):
        decoder = compress.Decoder()
        assert (decoder.feed(wire[:split], DataType.ASCII_8BIT)
                + decoder.feed(wire[split:], DataType.ASCII_8BIT)) == expected


@pytest.mark.parametrize('width', [1, 3, 5, 8, 20, 59])
def test_binary_transparency_through_run_and_supervisor_layers(width):
    """All octets, including adjacent reserved values and every split escape."""
    body = bytes(range(256)) * 2 + bytes.fromhex('1dff5cebc3f31d4b9272')
    wire = compress.transparent(body)
    decoder, supervisor = compress.Decoder(), compress.Supervisor()
    result = b''.join(supervisor.feed(decoder.feed(wire[i:i+width], DataType.ASCII_8BIT))
                      for i in range(0, len(wire), width))
    assert result == body


def test_stock_quoted_run_prefix_spans_fields_without_expansion():
    """SCS paired probe: ...1C / 7D FF 5C XY delivers literal 1D FF 5C XY."""
    decoder, supervisor = compress.Decoder(), compress.Supervisor()
    assert supervisor.feed(decoder.feed(b'ABCD\x1c', 0)) == b'ABCD'
    assert supervisor.feed(decoder.feed(b'\x7d\xff\x5cXY', 0)) == b'\x1d\xff\x5cXY'


def test_a_supervisor_block_spans_the_packet_it_started_in():
    """SB at the end of one field, its function code at the start of the next.

    MEASURED both ways: `41*19 1c` then `7e 42*19` delivered one 0x1E and
    nineteen `B`, and `43*18 1c 41` then `31 20 44*18` delivered only the
    eighteen `D` -- the parameter and its <SPACE> terminator were swallowed
    across the boundary too."""
    si = compress.Supervisor()
    assert si.feed(b"A" * 19 + b"\x1c") == b"A" * 19
    assert si.feed(b"\x7e" + b"B" * 19) == b"\x1e" + b"B" * 19
    assert si.feed(b"C" * 18 + b"\x1c\x41") == b"C" * 18
    assert si.feed(b"1 " + b"D" * 18) == b"D" * 18


def _receiving_link() -> tuple[PactorArq, ArqIO]:
    """A `PactorArq` that was called and is now the IRS, and its host port."""

    class IO(ArqIO):
        def __init__(self):
            self.delivered = bytearray()
        def connect_burst(self, mycall, dxcall): pass
        def send_packet(self, sl, payload, status, breakin=False): pass
        def send_cs(self, cs_index): pass
        def connected(self, mycall, dxcall): pass
        def disconnected(self): pass
        def deliver(self, blob): self.delivered += blob
        def buffer(self, nbytes): pass
        def log(self, msg): pass

    io = IO()
    arq = PactorArq(io)
    arq.on_host_listen(True)
    arq.on_rx_connect("DL6MAA", "W9SSJ")           # we are called: IRS, seq 1
    return arq, io


def test_the_link_layer_delivers_characters_not_wire_coding():
    """`PactorArq` hands the host what the peer WROTE: compressed fields are
    decoded by their own status byte, the run seam survives the packet
    boundary, idle fill is not delivered at all, and 8-bit payload arrives
    byte-exact."""
    arq, io = _receiving_link()
    arq.on_rx_packet(3, P3_A, spec.status_byte(1, DataType.PMC_GERMAN), True)
    arq.on_rx_packet(3, P3_B, spec.status_byte(2, DataType.PMC_GERMAN), True)
    assert b"Allgemeines\r-----------\r\rDie Version" in io.delivered
    assert b"verf\x81gbar" in io.delivered

    before = bytes(io.delivered)
    arq.on_rx_packet(3, IDLE_FILL, spec.status_byte(3, DataType.PMC_ENGLISH),
                     True)
    assert bytes(io.delivered) == before           # idle delivered nothing

    arq.on_rx_packet(3, b"de W9SSJ\r", spec.status_byte(0), True)
    assert io.delivered.endswith(b"de W9SSJ\r")    # 8-bit mode is transparent


@pytest.mark.parametrize('first,second,expected', [
    (b'ABCD\x1d', b'\xff\x5cXYZ', b'ABCD' + b'\xff' * 92 + b'XYZ'),
    (b'ABCD\x1c', b'\x7d\xff\x5cXY', b'ABCD\x1d\xff\x5cXY'),
])
def test_repeated_packet_does_not_refeed_partial_run_or_escape(first, second, expected):
    arq, io = _receiving_link()
    for payload, seq in ((first, 1), (first, 1), (second, 2), (second, 2)):
        arq.on_rx_packet(1, payload, spec.status_byte(seq), True,
                         protocol=Protocol.PACTOR3)
    assert bytes(io.delivered) == expected


def test_the_data_type_width_follows_the_protocol_not_the_speed_level():
    """Two bits of Datenmodus in PACTOR-1, three from PACTOR-2 on -- decided by
    the protocol the decoder read, because the speed level cannot answer it: the
    session's PACTOR-2 seam reports the PACTOR-1 level on purpose, so that the
    IRS asks a PACTOR-2 peer for no gear change on a ladder this station cannot
    climb (`onair._SessionRx._p2_packet`).

    Both vectors have bit 4 set and each is soup under the other's width, which
    is what makes them a test: DL6MAA's 200 Bd announcement is `1dl6maa` at two
    bits and `1DIT)   WS DUNZ0 AN` at three, and HB9AK's PMC English field is
    `moment with a ` at three and unreadable at two."""
    arq, io = _receiving_link()
    arq.on_rx_packet(P1_SPEED_LEVEL, P1_ANNOUNCEMENT, P1_ANNOUNCEMENT_STATUS,
                     True, protocol=Protocol.PACTOR1)
    assert bytes(io.delivered) == b"1dl6maa\r"

    status = spec.status_byte(1, DataType.PMC_ENGLISH)
    assert status & 0b10000
    arq, io = _receiving_link()
    arq.on_rx_packet(P1_SPEED_LEVEL, P2_ENGLISH, status, True,
                     protocol=Protocol.PACTOR2)
    assert bytes(io.delivered) == b"moment with a "
    assert b"moment" not in compress.decompress(P2_ENGLISH, (status >> 2) & 0b11)


def test_the_fixture_message_reads_across_its_packet_boundary():
    """End to end off the air: the receiver's own packets from the DL6MAA
    fixture, decoded by their declared modes in sequence order, join into the
    manual's continuous prose -- 'Datentransparenz' arrives split across two
    CRC-validated fields."""
    wav = FIXTURES / "oracle_pactor3_dl6maa.wav"
    if not wav.exists():
        pytest.skip(f"{wav} not present")
    from hfmodem.shrike import p3rx, rxfront
    audio = rxfront.load_wav(str(wav))
    lo, hi = int(15 * rxfront.FS), int(26 * rxfront.FS)
    pkts = sorted(p3rx.decode_p3_packets(audio[lo:hi], fs=rxfront.FS).packets,
                  key=lambda p: p.start)
    assert len(pkts) >= 3, "fixture slice no longer decodes"
    stream = compress.Decoder()
    text = b"".join(stream.feed(p.payload, p.data_type) for p in pkts)
    assert b"Datentransparenz\r----------------" in text
    assert b"wunsch verf\x81gbar. Es sind keinerlei Vorkehrungen" in text
    # ...and the same packets under a forced wrong mode carry none of it: the
    # per-packet declared mode is load-bearing, not decoration.
    soup = b"".join(compress.decompress(p.payload, DataType.PMC_ENGLISH)
                    for p in pkts)
    assert b"Datentransparenz" not in soup


def test_the_p2_qso_reads_from_the_air():
    """The monitor's whole PACTOR-2 pass on the HB9AK fixture: acquire, decode
    every burst the grid covers, take each field's status byte from the END --
    payload, status, CRC -- and the operator's own sentence comes out, in the
    order it was typed, per-field coding switches and all (the QSO flips between
    PMC English and plain Huffman mid-sentence).

    The counterexample is the layout a first attempt assumed: a status byte at
    the HEAD of the field. Byte 0 is the head of the compressed bit stream, so
    that parse must deliver soup -- and if it ever delivers these words, the
    field layout has changed under this test and it deserves to go red."""
    wav = FIXTURES / "pos_p2_sl3_hb9ak.wav"
    if not wav.exists():
        pytest.skip(f"{wav} not present")
    from hfmodem.shrike import monitor, rxfront
    audio = rxfront.load_wav(str(wav))
    events = sorted(monitor._p2_data_events(audio), key=lambda ev: ev.t)
    assert len(events) >= 25, "the fixture no longer decodes"

    text = monitor._Text()
    body = "\n".join(line for ev in events for line in text.lines(ev))
    at = 0
    for phrase in ("is a ", "pointless ", "exercise", "as ", "they ",
                   "collect ", "every week ", "of the ", "year! ", "the ",
                   "moment with a "):
        found = body.find(phrase, at)
        assert found >= 0, f"{phrase!r} missing or out of order in {body!r}"
        at = found + len(phrase)

    soup = b"".join(compress.decompress(pl[1:], (pl[0] >> 2) & 7)
                    for ev in events for pl in [ev.packet[2]] if pl)
    for word in (b"pointless", b"exercise", b"collect", b"moment"):
        assert word not in soup


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
