# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""2G ALE's coding, against the text of MIL-STD-188-141B appendix A."""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.bateleur.ale import golay, words as W


def test_the_golay_code_is_the_one_figure_a6_defines():
    """Minimum distance 8 is what makes it the extended Golay code; the
    matrix is transcribed from a scan, so this is the check on the scan."""
    weights = {bin(golay.encode(d)).count("1") for d in range(1, 4096)}
    assert weights == {8, 12, 16, 24}


def test_figure_a8_encoding_example():
    """Data 110100010101 -> check bits 010101100110 (figure A-8)."""
    assert golay.encode(0b110100010101) == (0b110100010101 << 12) | 0b010101100110


def test_three_errors_are_corrected_and_four_detected():
    rng = np.random.default_rng(0)
    for _ in range(300):
        d = int(rng.integers(4096))
        pos = rng.choice(24, 4, replace=False)
        e3 = sum(1 << int(p) for p in pos[:3])
        assert golay.decode(golay.encode(d) ^ e3) == (d, 3)
        e4 = e3 | (1 << int(pos[3]))
        got, n = golay.decode(golay.encode(d) ^ e4)
        assert n == -1 or got != d          # never silently "corrected" to d


def test_soft_decoding_corrects_what_hard_cannot():
    """Five wrong bits, each only weakly wrong: the hard decoder is beyond
    its power, the soft one is not."""
    d = 0xA5C
    cw = golay.encode(d)
    llr = np.array([4.0 if not (cw >> (23 - i)) & 1 else -4.0 for i in range(24)])
    for i in (1, 5, 9, 14, 20):
        llr[i] = -0.3 * np.sign(llr[i])
    hard = sum(1 << (23 - i) for i in range(24) if llr[i] < 0)
    assert golay.decode(hard)[0] != d or golay.decode(hard)[1] == -1
    assert golay.decode_soft(llr)[0] == d


def test_a_word_survives_its_trip_through_the_air_format():
    for pre, txt in ((W.TO, "JOE"), (W.TIS, "SAM"), (W.DATA, "UEL"), (W.CMD, "HI!")):
        w = W.word(pre, txt)
        bits = W.to_air(w)
        assert bits.size == 49 and bits[48] == 0              # S49
        got, ea, eb, score = W.from_air_soft(1.0 - 2.0 * bits)
        assert (got, ea, eb) == (w, 0, 0)


def test_b_check_bits_go_out_inverted():
    """Figure A-10: G13-G24 inverted after encoding. The all-zero word then
    does not send twelve zeros in a row from B."""
    bits = W.to_air(W.word(W.DATA, bits=0))
    b = bits[1:48:2]
    assert list(b[:12]) == [0] * 12 and list(b[12:]) == [1] * 12


def test_addresses_are_stuffed_and_extended_as_the_text_shows():
    """A.5.2.4.3: "MIAMI" is TO MIA, DATA MI@. A.5.2.4.4.2: "MISSISSIPPI" is
    TO MIS, DATA SIS, REP SIP, DATA PI@."""
    assert [W.describe(w) for w in W.address_words("MIAMI", W.TO)] == \
        ["TO 'MIA'", "DATA 'MI@'"]
    assert [W.describe(w) for w in W.address_words("MISSISSIPPI", W.TO)] == \
        ["TO 'MIS'", "DATA 'SIS'", "REP 'SIP'", "DATA 'PI@'"]


@pytest.mark.parametrize("bad", ["", "SIXTEENCHARACTER", "JO E", "jo@", "A.B"])
def test_an_address_the_standard_does_not_allow_is_refused(bad):
    with pytest.raises(ValueError):
        W.check_address(bad)


def test_an_amd_is_cmd_then_data_and_rep_padded_with_spaces():
    """A.5.7.2.2."""
    ws = W.amd_words("hello")
    assert [W.describe(w) for w in ws] == ["CMD 'HEL'", "DATA 'LO '"]
    with pytest.raises(ValueError):
        W.amd_words("tabs\tare not in the set")
    with pytest.raises(ValueError):
        W.amd_words("X" * 91)


def test_a_call_frame_reads_back():
    f = W.parse(W.call("JOE", "SAMUEL", scan_words=6,
                       message=W.amd_words("meet on 7") + [W.lqa_word(3, 21, request=True)]))
    assert f.to == ["JOE"] and f.caller == "SAMUEL" and f.terminator == W.TIS
    assert f.scanning and f.amd == ["MEET ON 7"]
    assert f.lqa == [{"request": True, "mp": 7, "sinad": 21, "ber": 3}]


def test_a_sound_reads_back_once_however_often_it_repeats():
    f = W.parse(W.sound("MISSISSIPPI", repeats=3, terminator=W.TWAS))
    assert f.sound and f.caller == "MISSISSIPPI" and f.terminator == W.TWAS


def test_the_capabilities_report_is_figure_a45s_four_words():
    ws = W.capabilities_report(scan_rate=5, channels=10, tune_100ms=10,
                               protocols=W.VAP_ALLCALL | W.VAP_AMD)
    assert [W.preamble_of(w) for w in ws] == [W.CMD, W.DATA, W.REP, W.DATA, W.REP]
    assert W.chars_of(ws[0]) == "vcr"
    f = W.parse(W.call("SAM", "JOE", message=ws))
    assert f.capabilities["scan_rate"] == 5 and f.capabilities["channels"] == 10
    assert f.capabilities["protocols"] == 0b1010000


def test_ber_and_sinad_codes_follow_table_a13():
    assert W.ber_code(0.4) == 0 and W.ber_code(12.6) == 13 and W.ber_code(44) == 30
    assert W.ber_code(None) == 31 and W.sinad_code(None) == 31
    assert W.sinad_code(-3) == 0 and W.sinad_code(33) == 30


def test_a_dtm_basic_block_counts_its_bits_exactly():
    """Table A-XXXII: DC = 32m + p, p data words and m the bits in the last
    word plus 10. Twenty characters are 140 bits: 7 words, 14 in the last,
    stuffed with a DEL (figure A-47's example)."""
    (blk,) = W.dtm_blocks("THE QUICK BROWN FOX!")
    d = W.dtm_fields(blk[0])
    assert d["kind"] == "basic" and d["dc"] == 32 * 24 + 7 and d["bits"] == 140
    assert [W.preamble_of(w) for w in blk] == [W.CMD] + [W.DATA, W.REP] * 3 + [W.DATA, W.CMD]
    assert W.chars_of(blk[7])[2] == "\x7f" and W.is_crc(blk[8])


def test_a_long_message_goes_in_extended_blocks_and_ends_in_a_basic_one():
    data = bytes(range(256)) * 3
    blocks = W.dtm_blocks(data, max_words=100)
    kinds = [W.dtm_fields(b[0])["kind"] for b in blocks]
    assert kinds[-1] == "basic" and set(kinds[:-1]) == {"extended"}
    assert [W.dtm_fields(b[0])["kd1"] for b in blocks] == [k % 2 for k in range(len(blocks))]
    assert len({W.dtm_fields(b[0])["kd2"] for b in blocks}) == 1
    got = sum((W.parse(W.call("JOE", "SAM", message=b)).dtm[0].data_bits for b in blocks), [])
    assert bytes(int("".join(map(str, got[i:i + 8])), 2) for i in range(0, len(got), 8)) == data


def test_the_crc_word_carries_the_fcs_as_table_a17_draws_it():
    w = W.crc_word(0xC001)
    assert W.chars_of(w)[0] == "{" and W.is_crc(w) and w & 0xFFFF == 0xC001
    assert W.chars_of(W.crc_word(0x0000))[0] == "x"


def test_two_damaged_copies_overlay_into_a_good_block():
    """Figure A-48: each copy has words the other read well."""
    (blk,) = W.dtm_blocks("THE QUICK BROWN FOX!", arq=True)
    ws = W.call("JOE", "SAM", message=blk)
    at = ws.index(blk[0])
    one = [w if k not in (at + 2, at + 5) else w ^ 0x5555 for k, w in enumerate(ws)]
    two = [w if k != at + 3 else w ^ 0x1234 for k, w in enumerate(ws)]
    a = W.parse(one, {at + 2, at + 5}).dtm[0]
    b = W.parse(two, {at + 3}).dtm[0]
    assert not a.ok and not b.ok
    c = a.combined(b)
    assert c.ok and c.data == "THE QUICK BROWN FOX!"


def test_the_lattice_algorithm_reproduces_b564s_examples():
    """Appendix B, B.5.6.4: key c2284a1ce7be2f, seeds for 8 May 15:57:34 at
    1755 kHz, words 0 to 2; also the seed built from those fields."""
    from hfmodem.bateleur.ale.lp import Lattice, seed
    lat = Lattice(bytes.fromhex("c2284a1ce7be2f"))
    for sd, plain, cipher in (("543bd88000017550", 0x54E0CD, 0xC0D705),
                              ("543bd88040017550", 0x54E0CD, 0x708434),
                              ("543bd88080017550", 0xB2A7C5, 0x28ED4A)):
        s = bytes.fromhex(sd)
        assert lat.encrypt(plain, s) == cipher and lat.decrypt(cipher, s) == plain
    assert W.word(W.TO, "SAM") == 0x54E0CD and W.word(W.TIS, "JOE") == 0xB2A7C5
    assert seed(month=5, day=8, minute=15 * 60 + 57, fine=34, word=0,
                freq_hz=1_755_000).hex() == "543bd88000017550"


def test_protected_word_numbers_follow_b531():
    """A scanning call alternates w = 0 and 1 ending on 1, the PI moving only
    at a 0; the leading call starts at 0; wmax rolls the PI over."""
    from hfmodem.bateleur.ale import lp
    p = lp.Protection(2, bytes(7), [7_102_000.0])
    seen = []
    real = p.encrypt
    p.encrypt = lambda w, st, ch: (seen.append(st), real(w, st, ch))[1]
    t0 = 1_778_255_854.0
    p.protect(W.call("JOE", "SAM", scan_words=5, message=W.amd_words("abcdefghijk")), t0, 0, scan=5)
    ws = [st.w for st in seen]
    assert ws[:5] == [1, 0, 1, 0, 1] and ws[5:11] == [0, 1, 2, 3, 4, 5] and ws[11] == 0
    assert seen[11].pi == seen[10].pi + 1
    assert all(seen[i].pi == seen[i - 1].pi for i in range(1, 5) if seen[i].w == 1)


def test_dtm_crc_records_the_unresolved_figure_a48_reading():
    from hfmodem.bateleur.ale import words as W
    message = "THE QUICK BROWN FOX!"
    bits = [(ord(c) >> shift) & 1 for c in message for shift in range(6, -1, -1)]
    assert W.crc16(bits + [1] * 7) == 0xDB2A
    assert W.crc16(bits + [0] * 7) == 0x5452
    assert W.dtm_blocks(message)[0][-1] == W.crc_word(0xDB2A)
    assert W.crc16(bits + [1] * 7) != 0x2C5B
