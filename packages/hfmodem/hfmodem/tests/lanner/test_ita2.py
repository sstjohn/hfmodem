# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The S.1 table both ways: encode, decode, shift state, and the US variant.

Codes here are written as the module's ints (bit i-1 = unit i, Z = 1), which
are not the doc's combination numbers.
"""
from __future__ import annotations

from hfmodem.lanner import ita2

LTRS, FIGS, CR, LF, SPACE = 31, 27, 8, 2, 4


def _round_trip(text, variant):
    d = ita2.Decoder(variant)
    return "".join(d.feed(c) for c in ita2.encode(text, variant))


def test_full_table_round_trip():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    figures_itu = "-?:38().,9014'57=2/6+"
    assert _round_trip(letters + " " + figures_itu, "itu") == letters + " " + figures_itu
    figures_us = '-?:$!&#38\'().,901457;/26"'
    assert _round_trip(letters + " " + figures_us, "us") == letters + " " + figures_us


def test_encode_opens_with_ltrs_and_shifts_between_letters_and_figures():
    assert ita2.encode("AB") == [LTRS, 3, 25]
    assert ita2.encode("A:B") == [LTRS, 3, FIGS, 14, LTRS, 25]
    assert ita2.encode("\n") == [LTRS, CR, LF]
    assert ita2.encode("A@B") == [LTRS, 3, 25]


def test_decoder_tracks_shift_state():
    d = ita2.Decoder("itu")
    assert d.feed(3) == "A"
    assert d.feed(FIGS) == ""
    assert d.feed(6) == "8"
    assert d.feed(LTRS) == ""
    assert d.feed(1) == "E"
    assert d.feed(CR) == ""
    assert d.feed(LF) == "\n"


def test_unshift_on_space():
    d = ita2.Decoder("itu")
    d.feed(FIGS), d.feed(6)
    assert d.feed(SPACE) == " "
    assert d.feed(1) == "E"

    d2 = ita2.Decoder("itu", unshift_on_space=False)
    d2.feed(FIGS), d2.feed(6)
    assert d2.feed(SPACE) == " "
    assert d2.feed(1) == "3"


def test_us_variant_differs_from_itu_where_the_doc_says():
    assert ita2.US_FIGURE_OVERRIDES == {
        4: ("WRU", "$"), 6: (None, "!"), 7: (None, "&"), 8: (None, "#"),
        10: ("BEL", "'"), 19: ("'", "BEL"), 22: ("=", ";"), 26: ("+", '"'),
    }
    # (code, itu_decodes, us_decodes) for the eight figure-case rows; BEL emits nothing.
    for code, itu_out, us_out in [
        (9, "", "$"), (13, "", "!"), (26, "", "&"), (20, "", "#"),
        (11, "", "'"), (5, "'", ""), (30, "=", ";"), (17, "+", '"'),
    ]:
        d_itu, d_us = ita2.Decoder("itu"), ita2.Decoder("us")
        d_itu.feed(FIGS), d_us.feed(FIGS)
        assert d_itu.feed(code) == itu_out
        assert d_us.feed(code) == us_out


def test_figures_are_reshifted_after_a_space():
    assert ita2.encode("5 9") == [LTRS, FIGS, 16, SPACE, FIGS, 24]
    assert _round_trip("RST 599 73", "us") == "RST 599 73"


def test_cr_and_lf_decode_in_figures_case():
    d = ita2.Decoder("itu")
    d.feed(FIGS)
    assert d.feed(CR) == "" and d.feed(LF) == "\n"
    assert _round_trip("12\n34", "us") == "12\n34"



def test_encoder_works_with_either_space_shift_policy():
    for variant in ("itu", "us"):
        for text in ("RST 599 73 12 ABC 34", "12 ABC", "12  \n@34 ABC"):
            for unshift in (True, False):
                decoder = ita2.Decoder(variant, unshift_on_space=unshift)
                assert "".join(decoder.feed(c) for c in ita2.encode(text, variant)) == text.replace("@", "")
    assert ita2.encode("12 A")[-2:] == [LTRS, 3]


def test_encoder_carries_the_case_across_calls():
    enc = ita2.Encoder()
    assert enc.feed("A1") == ([LTRS, 3, FIGS, 23], "")
    assert enc.feed("2") == ([19], "")
    assert enc.feed("B") == ([LTRS, 25], "")
    assert enc.feed("") == ([], "")
    enc.reset()
    assert enc.feed("C") == ([LTRS, 14], "")


def test_encoder_reshifts_after_a_figures_space_across_calls():
    enc = ita2.Encoder()
    enc.feed("5")
    enc.feed(" ")
    assert enc.feed("9") == ([FIGS, 24], "")


def test_encoder_in_pieces_equals_encode_whole():
    text = "CQ CQ DE W9SSJ 599 73 12 ABC\nRST 5NN @ 4/5 K"
    for variant in ("itu", "us"):
        enc = ita2.Encoder(variant)
        pieces = [enc.feed(text[i:i + 3])[0] for i in range(0, len(text), 3)]
        assert [c for p in pieces for c in p] == ita2.encode(text, variant)


def test_encoder_returns_what_it_cannot_encode():
    enc = ita2.Encoder()
    assert enc.feed("A@B%") == ([LTRS, 3, 25], "@%")
    assert ita2.Encoder().feed("@") == ([], "@")
    assert ita2.encode("") == [LTRS]
