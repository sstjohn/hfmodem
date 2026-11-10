# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Test-time access to the winlink layer's external material.

One fixture matters here: a real Winlink message (headers, ISO-8859-1 body,
jpeg attachment) together with its compressed form, produced outside this
project. It is the arbiter that keeps the codec honest — a round trip through
our own encoder and decoder can cancel a shared error and stay green, and this
pair cannot. It lives outside the package like every other corpus, and the
tests that read it skip when it is absent -- which is the right answer from a
wheel and a hole on a source tree, so which of the two this is gets decided once
in ``tests/gates/test_corpus_present.py`` off the two declarations below.
"""
from __future__ import annotations

import pytest

from hfmodem.tests import evidence

FIXTURES = evidence.WORKING / "winlink"

PLAIN = FIXTURES / "LPE5NXDVLVSQ.b2f"
COMPRESSED = FIXTURES / "LPE5NXDVLVSQ.b2f.lzh"

WW2MI_STREAM = FIXTURES / "LJ2AJE2IHO9B.stream"
WW2MI_PLAIN = FIXTURES / "LJ2AJE2IHO9B.b2f"

#: The specimen pair is in the index, so an absence of it is a damaged checkout.
COMMITTED = (PLAIN, COMPRESSED)
#: WW2MI's turn is not: it was cut out of a receive recording on the machine that
#: made it. A clone never has it and only this station can produce another.
WW2MI = (WW2MI_STREAM, WW2MI_PLAIN)


def real_pair() -> tuple[bytes, bytes]:
    """(message bytes, compressed bytes), or skip."""
    if not (PLAIN.exists() and COMPRESSED.exists()):
        pytest.skip("external winlink fixture pair not present")
    return PLAIN.read_bytes(), COMPRESSED.read_bytes()


def ww2mi() -> tuple[bytes, bytes]:
    """(WW2MI's whole turn as it flew, the message inside it), or skip.

    The other arbiter, and the one this station paid for: a production CMS
    talking to us on 2026-08-16, recovered from the receive recording after
    the session that heard it kept nothing. See working/winlink/SOURCE.txt.
    """
    if not (WW2MI_STREAM.exists() and WW2MI_PLAIN.exists()):
        pytest.skip("WW2MI capture not present")
    return WW2MI_STREAM.read_bytes(), WW2MI_PLAIN.read_bytes()
