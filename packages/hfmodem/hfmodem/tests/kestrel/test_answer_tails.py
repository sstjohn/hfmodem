# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The answer-burst generator against every tail this repo ever tabled by hand."""

import pytest

from hfmodem.kestrel.vara import vara_frames as VF

# (caller's, responder's) 8-symbol continue, W9SSJ -> W1AW; caller's side only.
# 2026-08-30 / 09-04 bench pairs, one cable per direction.
OVER_CONTINUE_CALLER = {
    "2300": ((64, 67), (30, 74), (39, 51), (32, 62), (29, 89), (70, 82), (45, 83), (52, 70)),
    "2750": ((64, 67), (37, 81), (36, 58), (29, 81), (36, 96), (31, 77), (76, 80), (43, 59)),
    "500": ((64, 67), (50, 66), (60, 64), (50, 68), (58, 76), (52, 76), (58, 76), (60, 68)),
}

# Responder's continue: K0SI 09-20 (k0si-data-ack-0920), KC9GHZ 09-11..09-19
# (head-cut-continue, data-replies-0919).
OVER_CONTINUE_RESPONDER = {
    ("K0SI", "2300"): ((64, 67), (58, 92), (77, 91), (48, 64),
                       (65, 71), (38, 92), (81, 85), (58, 98)),
    ("KC9GHZ", "2300"): ((64, 67), (32, 82), (37, 81), (54, 66),
                         (49, 81), (64, 84), (63, 79), (66, 96)),
    ("KC9GHZ", "2750"): ((64, 67), (25, 89), (30, 88), (47, 73),
                         (42, 88), (71, 103), (56, 76), (73, 79)),
    ("KC9GHZ", "500"): ((64, 67), (52, 74), (58, 74), (60, 72),
                        (56, 74), (56, 72), (62, 66), (50, 74)),
}

# Responder's continue to a 0x81 descent, KC9GHZ, all four low records (kc9ghz-downshift-0920).
OVER_CONTINUE_RESPONDER_DOWN = {
    "2300": ((64, 67), (44, 96), (33, 53), (56, 68), (45, 93), (68, 94), (81, 93), (50, 98)),
    "2750": ((64, 67), (77, 91), (36, 82), (27, 83), (74, 94), (43, 101), (88, 100), (83, 101)),
    "500": ((64, 67), (58, 68), (62, 64), (52, 76), (52, 74), (50, 72), (56, 76), (58, 64)),
}

# NAK, W9SSJ -> W1AW BW2300, noise-injected bench 2026-09-07 (both roles).
NAK_W1AW_2300 = (
    ((64, 67), (40, 62), (43, 83), (80, 98), (67, 95), (74, 88), (67, 83), (66, 74)),
    ((64, 67), (66, 78), (61, 95), (70, 98), (41, 71), (34, 98), (35, 51), (46, 92)),
)

# Responder's CRC-rejection NAK, KC9GHZ, 2026-09-19 all-bandwidth campaign (data-naks-0919).
DATA_NAK_RESPONDER = {
    "2300": ((64, 67), (42, 46), (43, 67), (74, 96), (53, 93), (70, 80), (35, 43), (42, 76)),
    "2750": ((64, 67), (39, 97), (34, 74), (67, 91), (46, 74), (77, 87), (28, 36), (35, 69)),
    "500": ((64, 67), (54, 76), (56, 68), (62, 68), (58, 74), (60, 76), (58, 72), (62, 64)),
}

# BW500 caller's continue by (level, 0x81), W9SSJ -> KC9GHZ, 90 keyings (bw500-continue).
CALLER_500_STATE = {
    (4, 0x95): ((62, 68), (58, 66), (50, 70), (52, 66), (52, 64), (60, 66), (56, 74)),
    (4, 0x81): ((60, 68), (58, 70), (54, 64), (56, 76), (62, 72), (58, 74), (50, 70)),
    (2, 0x95): ((54, 74), (60, 64), (52, 76), (54, 76), (62, 70), (62, 76), (60, 74)),
    (1, 0x99): ((58, 76), (58, 72), (54, 74), (50, 72), (54, 68), (60, 66), (62, 74)),
}

# 11-symbol control bursts' seven state symbols, W9SSJ-called; BW2750 09-04 bench,
# BW500 08-30 bench.
CONTROL_TAILS = {
    "2750": (((53, 89), (32, 42), (49, 95), (22, 68), (61, 71), (68, 102), (63, 71)),
             ((27, 87), (90, 104), (43, 73), (66, 102), (71, 103), (66, 84), (63, 65))),
    "500": (((52, 74), (56, 76), (52, 64), (56, 68), (56, 74), (58, 68), (60, 68)),
            ((54, 72), (56, 70), (56, 76), (54, 66), (58, 72), (54, 76), (50, 72))),
}

# The BW2300 tables were misread at four symbols; these are the ref128 tape's
# (20260814-065053-W9SSJ-W1AW-bw2300-ref128, 41.982 s caller, 8.570 s responder),
# where the old tables had (61,63) / (34,79) (80,82) (57,72).
CONTROL_TAILS["2300"] = (
    ((60, 96), (39, 49), (56, 88), (29, 61), (64, 80), (49, 95), (66, 90)),
    ((34, 80), (83, 97), (50, 66), (59, 95), (64, 96), (59, 77), (56, 72)))


@pytest.mark.parametrize("bw", ["2300", "2750", "500"])
def test_caller_continue(bw):
    assert VF.over_continue("W1AW", bw)[0] == OVER_CONTINUE_CALLER[bw]


@pytest.mark.parametrize("called,bw", list(OVER_CONTINUE_RESPONDER))
def test_responder_continue(called, bw):
    assert VF.over_continue(called, bw)[1] == OVER_CONTINUE_RESPONDER[called, bw]


@pytest.mark.parametrize("bw", ["2300", "2750", "500"])
def test_responder_continue_down(bw):
    assert VF.over_continue_down("KC9GHZ", bw)[1] == OVER_CONTINUE_RESPONDER_DOWN[bw]


def test_nak_both_roles():
    assert VF.nak("W1AW", "2300") == NAK_W1AW_2300


@pytest.mark.parametrize("bw", ["2300", "2750", "500"])
def test_data_nak(bw):
    assert VF.nak("KC9GHZ", bw)[1] == DATA_NAK_RESPONDER[bw]


@pytest.mark.parametrize("state", list(CALLER_500_STATE))
def test_bw500_caller_state(state):
    assert VF.over_continue_state("KC9GHZ", *state) == (
        VF.OVER_CONTINUE_LEAD,) + CALLER_500_STATE[state]


def test_bw500_caller_state_unseen():
    assert VF.over_continue_state("KC9GHZ", 3, 0x95) is None


@pytest.mark.parametrize("bw", ["2300", "2750", "500"])
def test_control_bursts(bw):
    assert VF.control_bursts("W9SSJ", bw) == tuple(
        VF.CONNECTED_ACK_PREAMBLE + tail for tail in CONTROL_TAILS[bw])


def test_keyed_by_callsign_as_dialled():
    assert VF.nak("kc9ghz", "2300") == VF.nak("KC9GHZ", "2300")
    assert VF.over_continue("KC9GHZ", "2300") != VF.over_continue("W1AW", "2300")
    assert VF.control_bursts("N0XYZ", "2300") != VF.control_bursts("W9SSJ", "2300")


def test_the_aaaa1_links_control_tails_are_frames_of_its_streams():
    assert VF.answer_tail("AAAA1", 1080, "500", 0) == VF.CONTROL_TAIL_ACK
    assert VF.answer_tail("AAAA1", 1080, "500", 1) == VF.CONTROL_TAIL_POLL
    assert VF.answer_tail("BBBB2", 290, "500", 13) == VF.CONTROL_TAIL_TURN
