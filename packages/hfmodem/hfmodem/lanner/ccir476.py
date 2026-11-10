# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CCIR 476 — the seven-unit constant-ratio code of ITU-R M.625 (SITOR/AMTOR).

Words are 7-bit ints with bit i-1 = position i; position 1 is transmitted
first, B = 0 and Y = 1, and every valid word has exactly four B and three Y.
On air B is the higher emitted frequency. ITA2 codes follow the ita2.py
convention: int with bit i-1 = unit i. Section marks refer to
`docs/protocols/sitor/code-tables.md`.
"""
from __future__ import annotations

# (combo_no, letters_case, figure_case, ITA2_bits12345, seven_unit_BY, seven_unit_binary) — §2.2.
CCIR476_TABLE = [
    (1,  "A",     "-",    "ZZAAA", "BBBYYYB", "0001110"),
    (2,  "B",     "?",    "ZAAZZ", "YBYYBBB", "1011000"),
    (3,  "C",     ":",    "AZZZA", "BYBBBYY", "0100011"),
    (4,  "D",     "WRU",  "ZAAZA", "BBYYBYB", "0011010"),
    (5,  "E",     "3",    "ZAAAA", "YBBYBYB", "1001010"),
    (6,  "F",     None,   "ZAZZA", "BBYBBYY", "0010011"),
    (7,  "G",     None,   "AZAZZ", "BYBYBBY", "0101001"),
    (8,  "H",     None,   "AAZAZ", "BYYBYBB", "0110100"),
    (9,  "I",     "8",    "AZZAA", "BYBBYYB", "0100110"),
    (10, "J",     "BEL",  "ZZAZA", "BBBYBYY", "0001011"),
    (11, "K",     "(",    "ZZZZA", "YBBBBYY", "1000011"),
    (12, "L",     ")",    "AZAAZ", "BYBYYBB", "0101100"),
    (13, "M",     ".",    "AAZZZ", "BYYBBBY", "0110001"),
    (14, "N",     ",",    "AAZZA", "BYYBBYB", "0110010"),
    (15, "O",     "9",    "AAAZZ", "BYYYBBB", "0111000"),
    (16, "P",     "0",    "AZZAZ", "BYBBYBY", "0100101"),
    (17, "Q",     "1",    "ZZZAZ", "YBBBYBY", "1000101"),
    (18, "R",     "4",    "AZAZA", "BYBYBYB", "0101010"),
    (19, "S",     "'",    "ZAZAA", "BBYBYYB", "0010110"),
    (20, "T",     "5",    "AAAAZ", "YYBYBBB", "1101000"),
    (21, "U",     "7",    "ZZZAA", "YBBBYYB", "1000110"),
    (22, "V",     "=",    "AZZZZ", "YYBBBBY", "1100001"),
    (23, "W",     "2",    "ZZAAZ", "BBBYYBY", "0001101"),
    (24, "X",     "/",    "ZAZZZ", "YBYBBBY", "1010001"),
    (25, "Y",     "6",    "ZAZAZ", "BBYBYBY", "0010101"),
    (26, "Z",     "+",    "ZAAAZ", "BBYYYBB", "0011100"),
    (27, "CR",    None,   "AAAZA", "YYYBBBB", "1110000"),
    (28, "LF",    None,   "AZAAA", "YYBBYBB", "1100100"),
    (29, "LTRS",  None,   "ZZZZZ", "YBYBBYB", "1010010"),
    (30, "FIGS",  None,   "ZZAZZ", "YBBYBBY", "1001001"),
    (31, "SPACE", None,   "AAZAA", "YYBBBYB", "1100010"),
    (32, "NUL",   None,   "AAAAA", "YBYBYBB", "1010100"),
]

# Table 2 (mode A controls; idle_alpha doubles as mode-B phasing signal 1 and
# signal_repetition as phasing signal 2) — §2.3.
SERVICE_SIGNALS = {
    "CS1": "BYBYYBB",
    "CS2": "YBYBYBB",
    "CS3": "BYYBBYB",
    "CS4": "BYBYBBY",
    "CS5": "BYYBYBB",
    "idle_alpha": "BBBBYYY",
    "idle_beta": "BBYYBBY",
    "signal_repetition": "YBBYYBB",
}

# The traffic combination each service signal reuses; None for the three spare
# codewords — §2.4.
DUAL_MEANING = {
    "CS1": 12,
    "CS2": 32,
    "CS3": 14,
    "CS4": 7,
    "CS5": 8,
    "idle_alpha": None,
    "idle_beta": None,
    "signal_repetition": None,
}


def _word_from_binary(binary: str) -> int:
    return int(binary[::-1], 2)


def _code_from_bits(bits: str) -> int:
    return int(bits.translate(str.maketrans("AZ", "01"))[::-1], 2)


_WORD_TO_CODE = {_word_from_binary(b): _code_from_bits(bits) for _, _, _, bits, _, b in CCIR476_TABLE}
_CODE_TO_WORD = {v: k for k, v in _WORD_TO_CODE.items()}

TRAFFIC_WORDS = frozenset(_WORD_TO_CODE)


def valid(word: int) -> bool:
    return 0 <= word < 128 and bin(word).count("1") == 3


def to_ita2(word: int) -> int | None:
    return _WORD_TO_CODE.get(word)


def from_ita2(code: int) -> int:
    return _CODE_TO_WORD[code]


def units(word: int) -> list[bool]:
    """Transmission order, position 1 first; True is the higher tone (B)."""
    return [not word >> i & 1 for i in range(7)]


def word(units: list[bool]) -> int:
    return sum((0 if b else 1) << i for i, b in enumerate(units))


# Mode-B phasing codewords (§2.3): idle alpha in RX, repetition in DX.
PHASING_1 = word([c == "B" for c in SERVICE_SIGNALS["idle_alpha"]])
PHASING_2 = word([c == "B" for c in SERVICE_SIGNALS["signal_repetition"]])
