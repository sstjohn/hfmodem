# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The M.625 seven-unit code: constant ratio, the ITA2 mapping, transmission order."""
from __future__ import annotations

from itertools import combinations

from hfmodem.lanner import ccir476

SPARES = {"idle_alpha", "idle_beta", "signal_repetition"}


def _all_constant_ratio_words():
    return sorted(sum((1 << p - 1 for p in combo), 0) for combo in combinations(range(1, 8), 3))


def test_traffic_plus_spares_cover_all_35_constant_ratio_words_exactly_once():
    traffic = ccir476.TRAFFIC_WORDS
    spares = [ccir476._word_from_binary(ccir476.SERVICE_SIGNALS[n].translate(str.maketrans("BY", "01")))
             for n in SPARES]
    assert len(set(traffic)) == 32
    assert not set(traffic) & set(spares)
    assert sorted(set(traffic) | set(spares)) == _all_constant_ratio_words()


def test_every_traffic_word_is_valid():
    for word in ccir476.TRAFFIC_WORDS:
        assert ccir476.valid(word)


def test_to_ita2_and_from_ita2_are_inverse():
    for _, _, _, bits, _, b in ccir476.CCIR476_TABLE:
        word = ccir476._word_from_binary(b)
        code = ccir476._code_from_bits(bits)
        assert ccir476.to_ita2(word) == code
        assert ccir476.from_ita2(code) == word


def test_units_puts_position_1_first_and_b_true():
    # Row 1 (A) from the doc: BBBYYYB.
    assert ccir476.units(56) == [True, True, True, False, False, False, True]
    assert ccir476.word([True, True, True, False, False, False, True]) == 56
    for word in _all_constant_ratio_words():
        assert sum(ccir476.units(word)) == 4
        assert ccir476.word(ccir476.units(word)) == word


def test_specific_rows_against_doc_literals():
    # Row 1 (A): binary 0001110 -> word 56. Row 20 (T): binary 1101000 -> word 11.
    assert ccir476._word_from_binary("0001110") == 56
    assert ccir476._word_from_binary("1101000") == 11
    # CS1 reuses combination 12 (L); CS2 reuses 32 (NUL); CS4 and CS5 are the
    # M.625 additions onto the unassigned figure rows 7 (G) and 8 (H).
    assert ccir476.SERVICE_SIGNALS["CS1"] == "BYBYYBB"
    assert ccir476.to_ita2(ccir476._word_from_binary(ccir476.SERVICE_SIGNALS["CS1"].translate(str.maketrans("BY", "01")))) == 18
    assert ccir476.DUAL_MEANING["CS4"] == 7
    assert ccir476.DUAL_MEANING["CS5"] == 8
    assert ccir476.SERVICE_SIGNALS["signal_repetition"] == "YBBYYBB"
    assert ccir476.SERVICE_SIGNALS["idle_alpha"] == "BBBBYYY"


def test_rows_map_to_the_ita2_codes_s1_gives_them():
    # (M.625 pattern, the same character's S.1 pattern), both transcribed by hand.
    for pattern, s1 in [("BBBYYYB", "ZZAAA"), ("YBBYBYB", "ZAAAA"), ("YYBBBYB", "AAZAA"),
                        ("YBYBBYB", "ZZZZZ"), ("YBBYBBY", "ZZAZZ"), ("YYYBBBB", "AAAZA")]:
        word = ccir476.word([u == "B" for u in pattern])
        assert ccir476.valid(word)
        assert ccir476.to_ita2(word) == int(s1.translate(str.maketrans("AZ", "01"))[::-1], 2)


def test_phasing_constants_are_spare_word_ints():
    assert ccir476.PHASING_1 == 112
    assert ccir476.PHASING_2 == 25
    for word in (ccir476.PHASING_1, ccir476.PHASING_2):
        assert ccir476.valid(word)
        assert word not in ccir476.TRAFFIC_WORDS
        assert ccir476.to_ita2(word) is None
