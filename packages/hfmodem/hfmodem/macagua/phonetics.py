# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Spoken-form transcript -> amateur callsigns."""
import re
from hfmodem.contact.calls import CALL

# Old-alphabet words that are also names or everyday speech ("Roger", "how", "David")
# are left out: on air they are far more often words than letters.
_LETTERS = {}
for letters, words in {
    "A": "alpha alfa", "B": "bravo baker boston", "C": "charlie",
    "D": "delta", "E": "echo", "F": "foxtrot fox",
    "G": "golf gulf", "H": "hotel", "I": "india",
    "J": "juliet juliett juliette japan", "K": "kilo king", "L": "lima",
    "M": "mike mexico", "N": "november", "O": "oscar",
    "P": "papa", "Q": "quebec", "R": "romeo",
    "S": "sierra sugar", "T": "tango", "U": "uniform",
    "V": "victor", "W": "whiskey whisky", "X": "xray", "Y": "yankee",
    "Z": "zulu",
}.items():
    for w in words.split():
        _LETTERS[w] = letters

_DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "tree": "3", "four": "4",
           "fower": "4", "five": "5", "fife": "5", "six": "6", "seven": "7", "eight": "8",
           "nine": "9", "niner": "9"}

# Single-letter prefixes exist only for these ITU blocks; anything else needs two characters.
_RAW = re.compile(r"[a-z]|[a-z]{1,2}[0-9]{1,2}|[0-9]+|(?=[a-z0-9]*[0-9])[a-z0-9]{3,7}")
_TOKEN = re.compile(r"[a-z0-9]+|[.!?;]")


def _pieces(text):
    out = []
    for tok in _TOKEN.findall(text.lower().replace("x-ray", "xray")):
        if tok in _LETTERS:
            out.append(_LETTERS[tok])
        elif tok in _DIGITS:
            out.append(_DIGITS[tok])
        elif tok in ("a", "i") or not _RAW.fullmatch(tok):
            out.append(None)
        else:
            out.append(tok.upper())
    return out


def _ends(p, i):
    """(end, call) for every valid call starting at piece i, shortest first.
    A call already written as one token ("K7L, Whiskey four...") is never extended."""
    if len(p[i]) > 1 and CALL.fullmatch(p[i]):
        return [(i + 1, p[i])]
    out, s = [], ""
    for j in range(i, len(p)):
        if p[j] is None:
            break
        s += p[j]
        if len(s) > 7:
            break
        if CALL.fullmatch(s):
            out.append((j + 1, s))
    return out


def _pick(p, i):
    ends = _ends(p, i)
    for j, s in ends:
        if "".join(x or " " for x in p[j:j + len(s)]).startswith(s):
            return j, s
    return ends[-1] if ends else None


def callsigns(text):
    """Leftmost-longest calls, preferring the shorter call when it is spoken twice in a row
    and a letter-led call over a digit-led one with the same end ("five nine W9SSJ")."""
    p, found, i = _pieces(text), [], 0
    while i < len(p):
        best = p[i] and _pick(p, i)
        if not best:
            i += 1
            continue
        nxt = p[i][0].isdigit() and _pick(p, i + 1)
        if nxt and nxt[0] == best[0]:
            i += 1
            continue
        found.append(best[1])
        i = best[0]
    return found
