# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""TEXT grammar completeness and generated cross-table continuation cases."""
import pytest

from hfmodem.kestrel.vara import compression as C
from hfmodem.tests.kestrel.test_compression import packed


@pytest.mark.parametrize('symbols,escapes', [
    (C.LINE_SYMBOLS, []), (C.SYMBOLS, [C.ESC]),
    (C.SFI_SYMBOLS, [C.SFI_ESC]),
    (C.SID_SYMBOLS, [C.SID_ESC, C.SID_COMMENT]),
    (C.WORD_SYMBOLS, [C.WORD_ESC]), (C.NUMBER_SYMBOLS, [C.NUMBER_ESC]),
    (C.FW_SYMBOLS, [C.FW_ESC, C.FW_SPACE]),
])
def test_every_nonraw_dictionary_is_a_complete_prefix_code(symbols, escapes):
    codes = list(symbols)+escapes
    assert len(codes) == len(set(codes))
    assert not any(a != b and b.startswith(a) for a in codes for b in codes)
    assert sum(2**-len(code) for code in codes) == 1


def test_sfi_date_continuations_and_prose_escape_at_each_symbol_boundary():
    # Each pair names the state after its atom, independently of decoder state.
    units = [('0011', 3), ('0110', 3), ('0101', 3), ('11110', 3),
             ('1001', 3), ('0101', 3), ('11100', 3),
             ('101', 3), ('0110', 3), ('00', 3), ('101', 3), ('0101', 3),
             ('00', 3), ('1100', 3), ('0110', 3), ('0101', 3), ('1000', 3),
             ('11110', 3), ('11111', 3), ('1000', 3), ('01001', 3),
             ('11100', 3), ('1100', 3), ('0111', 0),
             (C.PROSE, 1), (C.BYTE_CODES[97], 1)]
    expected = b'{SFI = 123 On 2026-10-20 12:34:56 UTC}\r\na'
    for split in range(1, len(units)+1):
        first = packed('000'+''.join(code for code, _ in units[:split]))
        first_data = C.decode_text(first)
        assert first_data.error is None
        second = packed(f'{units[split-1][1]:03b}'+''.join(
            code for code, _ in units[split:]))
        second_data = C.decode_text(second)
        assert second_data.error is None
        assert first_data.data+second_data.data == expected
    assert C.decode_text(packed('011'+'01000'+C.BYTE_CODES[97])).data == b'a'


@pytest.mark.parametrize('header,literal', [
    ('010', b'[RMS Express-'), ('0010', b'RMS Trimode '),
    ('100010', b'[RMS Relay-'), ('1010111', b'[BPQ-'),
])
def test_sid_headers_suffix_variants_and_numeric_transitions(header, literal):
    bits = ('000'+header+'011'+'11'+'011'+'1011011'
            +'0010'+C.NUMBER_CODES[ord('2')]+C.NUMBER_CODES[13]
            +'1000111'+'10000'+'10101100'+C.FW_CODES[ord('N')]+C.FW_CODES[13])
    result = C.decode_text(packed(bits))
    assert result.error is None
    assert result.data == literal+b'1.1-B2FWIHM$]\r;PQ: 2\r;FR:\r\nde N\r'


def test_new_prose_phrases_change_to_callsign_and_line_tables():
    bits = ('001'+'10000110'+C.BYTE_CODES[97]+'01010010'+'101101111'
            +'111010011'+'110111000'+'1011110010'+'01010001'
            +C.FW_CODES[ord('N')]+C.FW_CODES[13]
            +'101010'+'10000'+'1000110'+'01100100'+'01010'
            +C.PROSE+'101101101'+C.PROSE+'010100000')
    result = C.decode_text(packed(bits))
    assert result.error is None
    assert result.data == (
        b'Welcome to the a RMS Trimode RMS Relay'
        b'forwarded to the recipient via HF radio through the Internet  //WL2K '
        b' minutes remaining with N\r;WL2K-Radio/Internet_Network\r\n'
        b'*** IS0GRB\r Connected to CMS\r >\r')


def test_sid_space_returns_to_prose_and_standalone_cr_starts_a_new_line():
    result = C.decode_text(packed('100'+'1001110'+C.BYTE_CODES[97]
                                 +C.BYTE_CODES[13]+'1110'+'1001000'+'1111'))
    assert result.error is None and result.data == b' a\r[WL2K-\rFF\r'


@pytest.mark.parametrize('table,tail', [
    ('011', '0100'), ('000', '10001'), ('100', '101100'),
])
def test_new_table_truncations_preserve_the_damage_latch(table, tail):
    receiver = C.ReceiveCompression('TEXT')
    data, error = receiver.feed(packed(table+tail), C.TEXT_FLAG, True)
    assert error and receiver.damaged and data == b''
    assert receiver.feed(b'later plain bytes', 0x82, True) == (b'', None)


def test_previously_unknown_prose_leaf_is_now_a_verified_literal():
    result = C.decode_text(bytes.fromhex('29b8ff'))
    assert result.error is None and result.data == b'a through the Internet '
