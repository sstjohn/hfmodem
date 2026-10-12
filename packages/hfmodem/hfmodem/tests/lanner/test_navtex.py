# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""M.540 framing, parsing with garbles, and a full air path."""
from __future__ import annotations

import pytest

from hfmodem.lanner import navtex

MSG = navtex.Message("K", "A", 42,
                     "NAVAREA IV 1234/26\nGULF OF MEXICO. RIG MOVED TO 27-40N 090-12W.")


def test_frame_parse_round_trip():
    assert navtex.parse(navtex.frame(MSG)) == [MSG]


def test_frame_shape():
    assert navtex.frame(MSG).startswith("ZCZC KA42\n")
    assert navtex.frame(MSG).endswith("NNNN\n\n")


def test_two_messages_in_one_stream():
    second = navtex.Message("M", "D", 7, "SAR ON FOOT")
    assert navtex.parse(navtex.frame(MSG) + navtex.frame(second)) == [MSG, second]


def test_garbled_header_skipped_body_kept():
    text = "ZCZC _KA42\nJUNK\nNNNN\n\n" + navtex.frame(MSG)
    assert navtex.parse(text) == [MSG]
    text = "ZCZC KA42\nBODY _ARBLE\nNNNN\n\n"
    assert navtex.parse(text) == [navtex.Message("K", "A", 42, "BODY _ARBLE")]


def test_modulate_demodulate_round_trip():
    assert navtex.demodulate(navtex.modulate(MSG)) == [MSG]


def test_header_after_the_previous_tail_on_the_same_line():
    text = "NNNN\nBRKUWZCZC QA20\nSEC SD BNM 107-26\nNNNN\n"
    assert navtex.parse(text) == [navtex.Message("Q", "A", 20, "SEC SD BNM 107-26")]


@pytest.mark.parametrize("noise", ["SPQYU__N", "BPQY____", " 123!?\r", ""])
def test_header_line_noise_is_discarded_and_port_san_luis_body_kept(noise):
    body = "1. THE PORT SAN LUIS LWB 3 (LLNR 3815) HAS BEEN REP EXTING.\n2. REP BY PSLHP."
    message = navtex.Message("Q", "A", 15, body)
    text = f"ZCZC QA15{noise}\n{body}\nNNNN\n" + navtex.frame(MSG)
    assert navtex.parse(text) == [message, MSG]


@pytest.mark.parametrize("header", ["_A15", "Q_15", "QA_5", "QA1_", "QA1", "1A15", "Q115"])
def test_garbled_header_fields_are_rejected_even_with_trailing_noise(header):
    text = f"ZCZC {header}SPQYU__N\nJUNK\nNNNN\n" + navtex.frame(MSG)
    assert navtex.parse(text) == [MSG]


@pytest.mark.parametrize("noise", ["", "BPQY____"])
def test_missing_terminator_cannot_swallow_the_next_header(noise):
    second = navtex.Message("Q", "A", 20, "SECOND BULLETIN")
    text = "ZCZC QA17\nUNTERMINATED\n" + f"ZCZC QA20{noise}\n{second.text}\nNNNN"
    assert navtex.parse(text) == [second]
    messages, remainder = navtex.scan(text)
    assert messages == [(text.index("ZCZC QA20"), len(text), second)]
    assert remainder == ""


def test_scan_recovers_every_split_in_header_body_and_terminator():
    text = "NOISE" + navtex.frame(MSG).replace("KA42\n", "KA42BPQY____\n")
    for split in range(len(text) + 1):
        first, remainder = navtex.scan(text[:split])
        second, remainder = navtex.scan(remainder + text[split:])
        assert [message for _, _, message in first + second] == [MSG]
        assert remainder == ""


def test_scan_retains_partial_identity_across_locks_without_a_newline():
    messages, remainder = navtex.scan("NOISEZCZC Q")
    assert messages == [] and remainder == "ZCZC Q"
    messages, remainder = navtex.scan(remainder + "A17\nBODY\nNNNN")
    assert messages == [(0, len("ZCZC QA17\nBODY\nNNNN"), navtex.Message("Q", "A", 17, "BODY"))]
    assert remainder == ""


def test_scan_spans_and_pending_message():
    first = navtex.frame(MSG)
    pending = "ZCZC QA17\nUNFINISHED"
    messages, remainder = navtex.scan("NOISE" + first + pending)
    assert messages == [(5, 5 + first.index("NNNN") + 4, MSG)]
    assert remainder == pending


def test_garbled_next_header_still_bounds_a_message_body():
    text = "ZCZC QA17\nUNTERMINATED\nZCZC Q_20\nOTHER\nNNNN"
    assert navtex.parse(text) == []


def test_demodulate_recovers_a_header_split_across_locks(monkeypatch):
    monkeypatch.setattr(navtex.sitor, "transmissions", lambda *a, **kw: [
        (0, "ZCZC Q"), (100, "A17\nBODY\nNNNN")])
    assert navtex.demodulate([]) == [navtex.Message("Q", "A", 17, "BODY")]


def test_another_header_on_the_header_line_supersedes_the_first():
    text = "ZCZC QA17ZCZC QA20BPQY____\nBODY\nNNNN"
    assert navtex.parse(text) == [navtex.Message("Q", "A", 20, "BODY")]
