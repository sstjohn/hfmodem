# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""ITA2 — the International Telegraph Alphabet No. 2, ITU-T Rec. S.1.

Codes are ints 0..31 with bit i-1 = code unit i, so unit 1 is the least
significant bit and is transmitted first (Table 1/S.1 note). Z (perforation,
stop polarity) is a set bit. `variant` selects the international figure case
("itu") or the US-TTY figure case ("us"), which differ in eight combinations.
Control names that are not printable (WRU, BEL, LTRS, FIGS, NUL) emit nothing
from the decoder; CR emits "" and LF emits "\\n".
"""
from __future__ import annotations

# (combination_no, letters_case, figure_case_intl, bits "12345" A/Z) — §1.2.
TABLE = [
    (1,  "A", "-",    "ZZAAA"),
    (2,  "B", "?",    "ZAAZZ"),
    (3,  "C", ":",    "AZZZA"),
    (4,  "D", "WRU",  "ZAAZA"),
    (5,  "E", "3",    "ZAAAA"),
    (6,  "F", None,   "ZAZZA"),
    (7,  "G", None,   "AZAZZ"),
    (8,  "H", None,   "AAZAZ"),
    (9,  "I", "8",    "AZZAA"),
    (10, "J", "BEL",  "ZZAZA"),
    (11, "K", "(",    "ZZZZA"),
    (12, "L", ")",    "AZAAZ"),
    (13, "M", ".",    "AAZZZ"),
    (14, "N", ",",    "AAZZA"),
    (15, "O", "9",    "AAAZZ"),
    (16, "P", "0",    "AZZAZ"),
    (17, "Q", "1",    "ZZZAZ"),
    (18, "R", "4",    "AZAZA"),
    (19, "S", "'",    "ZAZAA"),
    (20, "T", "5",    "AAAAZ"),
    (21, "U", "7",    "ZZZAA"),
    (22, "V", "=",    "AZZZZ"),
    (23, "W", "2",    "ZZAAZ"),
    (24, "X", "/",    "ZAZZZ"),
    (25, "Y", "6",    "ZAZAZ"),
    (26, "Z", "+",    "ZAAAZ"),
    (27, "CR", None,  "AAAZA"),
    (28, "LF", None,  "AZAAA"),
    (29, "LTRS", None, "ZZZZZ"),
    (30, "FIGS", None, "ZZAZZ"),
    (31, "SPACE", None, "AAZAA"),
    (32, "NUL", None, "AAAAA"),
]

# combo_no: (ITA2_intl_figure, USTTY_figure) — §1.3.
US_FIGURE_OVERRIDES = {
    4:  ("WRU", "$"),
    6:  (None, "!"),
    7:  (None, "&"),
    8:  (None, "#"),
    10: ("BEL", "'"),
    19: ("'",  "BEL"),
    22: ("=",  ";"),
    26: ("+",  '"'),
}

_TRANS = str.maketrans("AZ", "01")


def _code(bits: str) -> int:
    return int(bits.translate(_TRANS)[::-1], 2)


def _emit(name):
    if name is None or name in ("WRU", "BEL", "LTRS", "FIGS", "NUL"):
        return ""
    if name == "CR":
        return ""
    if name == "LF":
        return "\n"
    if name == "SPACE":
        return " "
    return name


_CODE = {combo: _code(bits) for combo, _, _, bits in TABLE}
_LTRS, _FIGS, _CR, _LF, _SPACE = _CODE[29], _CODE[30], _CODE[27], _CODE[28], _CODE[31]

_DECODE: dict = {"itu": {}, "us": {}}
_ENCODE: dict = {"itu": {}, "us": {}}
for _combo, _letters, _figures, _bits in TABLE:
    _c = _CODE[_combo]
    for _variant in ("itu", "us"):
        _fig = US_FIGURE_OVERRIDES.get(_combo, (None, _figures))[1] if _variant == "us" else _figures
        if _fig is None and len(_letters) > 1:
            _fig = _letters     # CR, LF and the shifts mean the same in both cases
        _DECODE[_variant][_c] = (_emit(_letters), _emit(_fig))
        if len(_letters) == 1:
            _ENCODE[_variant][_letters] = (_c, _LTRS)
        if _fig is not None and len(_fig) == 1:
            _ENCODE[_variant][_fig] = (_c, _FIGS)


class Encoder:
    """ITA2 with the case carried across calls, for text that arrives as it is typed.

    The first code after construction or `reset()` is LTRS (S.1 §4.4); after that
    a shift goes out only when the case changes. After a figures space the next
    printable character is shifted again, so either receiver space policy reads it.
    "\\n" becomes CR LF. `feed` returns the codes and the characters it could not
    encode, which a caller can show rather than lose.
    """

    def __init__(self, variant: str = "us"):
        self._table = _ENCODE[variant]
        self.reset()

    def reset(self) -> None:
        self._opening = True
        self._case = _LTRS

    def feed(self, text: str) -> tuple[list[int], str]:
        codes, uncodable = [], []
        for ch in text.upper():
            if ch == "\n":
                codes += [_CR, _LF]
                continue
            if ch == "\r":
                codes.append(_CR)
                continue
            if ch == " ":
                codes.append(_SPACE)
                if self._case == _FIGS:
                    self._case = None     # receivers may either retain or reset their shift
                continue
            entry = self._table.get(ch)
            if entry is None:
                uncodable.append(ch)
                continue
            code, ch_case = entry
            if ch_case != self._case:
                codes.append(ch_case)
                self._case = ch_case
            codes.append(code)
        if codes and self._opening:
            codes.insert(0, _LTRS)
            self._opening = False
        return codes, "".join(uncodable)


def encode(text: str, variant: str = "us") -> list[int]:
    """One whole text from a fresh `Encoder`; characters with no code are skipped."""
    return Encoder(variant).feed(text)[0] or [_LTRS]


class Decoder:
    """Feed one code int at a time; returns the text that decodes, if any."""

    def __init__(self, variant: str = "us", unshift_on_space: bool = True):
        self._map = _DECODE[variant]
        self.unshift_on_space = unshift_on_space
        self._case = _LTRS

    def feed(self, code: int) -> str:
        if code == _LTRS:
            self._case = _LTRS
            return ""
        if code == _FIGS:
            self._case = _FIGS
            return ""
        if code == _SPACE:
            if self.unshift_on_space:
                self._case = _LTRS
            return " "
        letters, figures = self._map[code]
        return figures if self._case == _FIGS else letters
