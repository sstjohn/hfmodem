# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""NAVTEX — the message layer of ITU-R M.540, carried by SITOR mode B.

B1 identifies the transmitter, B2 the subject, B3B4 the two-digit serial.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from hfmodem.lanner import sitor


@dataclass
class Message:
    station: str
    subject: str
    serial: int
    text: str


def frame(message: Message) -> str:
    return f"ZCZC {message.station}{message.subject}{message.serial:02d}\n{message.text}\nNNNN\n\n"


# M.540-2 Annex II §7 lets garbles stand in the body; a garbled B1, B2 or
# two-digit serial is skipped rather than guessed at. Off air, ZCZC usually
# follows the previous message's tail on the same line, so it is not anchored.
# Noise after the serial is accepted as part of the header line, not the body.
_HEADER = re.compile(r"ZCZC ([A-Z])([A-Z])([0-9]{2})(?:(?!ZCZC )[^\n])*\n")
_START = re.compile(r"ZCZC ")
_END = re.compile(r"\nNNNN")
_PARTIAL_HEADER = re.compile(
    r"(?:Z|ZC|ZCZ|ZCZC|ZCZC |ZCZC [A-Z]|ZCZC [A-Z]{2}|"
    r"ZCZC [A-Z]{2}[0-9]|ZCZC [A-Z]{2}[0-9]{2}[^\n]*)\Z")


def scan(text: str) -> tuple[list[tuple[int, int, Message]], str]:
    """Return complete (start, end, Message) spans and a pending header/message.

    Unterminated messages superseded by another header are omitted.
    Spans are half-open indices in ``text``, ending just after NNNN. Feed the
    next lock with ``scan(remainder + lock_text)``; insert no newline, since
    lock boundaries are not transmitted characters. Noise without a pending
    header is discarded; even a trailing partial ZCZC or identity is retained.
    """
    messages = []
    remainder = ""
    for header in _HEADER.finditer(text):
        following = _START.search(text, header.end())
        limit = following.start() if following else len(text)
        end = _END.search(text, header.end(), limit)
        if end is not None:
            b1, b2, serial = header.groups()
            message = Message(b1, b2, int(serial), text[header.end():end.start()])
            messages.append((header.start(), end.end(), message))
        elif following is None:
            remainder = text[header.start():]
    if not remainder:
        partial = _PARTIAL_HEADER.search(text)
        if partial is not None:
            remainder = partial.group()
    return messages, remainder


def parse(text: str) -> list[Message]:
    """Complete messages only; omit unterminated messages before another header."""
    messages, _ = scan(text)
    return [message for _, _, message in messages]


def modulate(message: Message, **kw) -> np.ndarray:
    return sitor.modulate(frame(message), **kw)


def demodulate(audio, **kw) -> list[Message]:
    messages = []
    remainder = ""
    for _, text in sitor.transmissions(audio, **kw):
        found, remainder = scan(remainder + text)
        messages.extend(message for _, _, message in found)
    return messages
